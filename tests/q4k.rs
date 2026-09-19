//! ggml's Q4_K against the port's kernel.
//!
//! Reference: tools/dump_q4k.py -- the only case in this suite whose oracle is
//! not MLX. mlx has no k-quant mode and is not getting one (`Q4K.md` quotes
//! the maintainer closing the request), so the contract that has caught every
//! bug in this port is not available in its usual form. `ggml` implements
//! exactly this format, ships it to everyone, and did not learn it from here,
//! so it takes the oracle's place.
//!
//! **What is pinned exactly, and what cannot be.** `dequantize_row_q4_K` is a
//! fixed sequence of operations on fixed inputs, so `q4k_dequantize` is held
//! to `max_abs == 0.0` against ggml's own output, and that single check covers
//! all three of the format's traps: the 12-byte scale packing, the nibble
//! interleave, and `w = d*sc*q - dmin*m` subtracting where affine adds.
//!
//! A matvec is different: its summation order belongs to the kernel, and no
//! reference computed any other way can be bit-identical to it. So the product
//! is checked two ways instead -- against a float64 reference with a stated
//! bound, and against a one-hot activation, where the sum has exactly one
//! nonzero term per output and an addressing error cannot hide inside a
//! tolerance.
//!
//! The tiled gemm (`kquant::qmm_t`) carries those two and a third, because it
//! *replaces* a kernel that is already checked: it must agree with the mat-vec
//! to within the difference in summation order. The cases for it are at the
//! bottom of this file.

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::kquant::{
    Hoist, KMatrix, KQuant, dequantize, gather_qmm_t, gather_qmv, qmm_t, qmv,
    set_mm_rows, set_qmv_hoist,
};
use ornith_rs::refs::{RefCase, compare, refs_root};
use ornith_rs::ops::linear::Linear;
use ornith_rs::weights::WeightStore;

/// The kernel accumulates in float32 over `K` terms and the reference is
/// float64, so the two differ by rounding alone. `K = 2048` terms of
/// comparable magnitude put the expected relative error near `sqrt(K) * 2^-24`
/// ~ 2.7e-6; this is that with room, and the test prints what it measured so a
/// regression shows as a number rather than as a pass.
const MATVEC_MAX_REL: f32 = 1e-5;

/// One term per output, so only the association differs: ~16 float32 ULPs.
const ONE_HOT_MAX_REL: f32 = 1e-6;

fn matrix(ctx: &Context, case: &RefCase, name: &str) -> KMatrix {
    KMatrix::new(
        KQuant::Q4K,
        case.array(ctx, &format!("{name}_qs")),
        case.array(ctx, &format!("{name}_scales")),
        case.array(ctx, &format!("{name}_dm")),
    )
}

#[test]
fn dequantize_matches_ggml() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    let ctx = Context::new().unwrap();
    for name in ["head", "qkv"] {
        let w = matrix(&ctx, &case, name);
        let rows = case.meta_usize(&format!("{name}_rows"));
        let cols = case.meta_usize(&format!("{name}_cols"));
        assert_eq!((w.out_dims(), w.in_dims()), (rows, cols));

        let got = dequantize(&ctx, &w, Dtype::F32);
        let want = case.f32(&format!("{name}_deq"));
        let d = compare(&got.to_vec_f32(&ctx), &want);
        assert_eq!(d.max_abs, 0.0, "{name} dequantize: {d}");
    }
}

/// The same weights read back through the matvec instead of the dequantizer.
///
/// `x = e_j` leaves one nonzero term in every output's sum, so the answer is
/// column `j` of the dequantized matrix. The two kernels associate the
/// arithmetic differently -- the dequantizer computes `(d*sc)*q - dmin*m`
/// while the matvec computes `d*(q*sc) - dmin*(1*m)` -- so this is a float32
/// rounding apart rather than bit-identical, and it is still tight enough that
/// a row or block addressed wrongly cannot pass. It measures **exactly zero**
/// on both cases today, over 768 outputs; the bound is a few float32 ULPs
/// rather than `0.0` because nothing guarantees the two associations round the
/// same way on weights this reference does not contain.
#[test]
fn matvec_on_a_one_hot_returns_a_column() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    let ctx = Context::new().unwrap();
    for name in ["head", "qkv"] {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let j = case.meta_usize(&format!("{name}_onehot_j"));

        let mut x = vec![0.0f32; cols];
        x[j] = 1.0;
        let x = Array::from_f32(&ctx, &x, &[1, cols]);
        let got = qmv(&ctx, &x, &w).to_vec_f32(&ctx);
        let want = case.f32(&format!("{name}_y_onehot"));
        let d = compare(&got, &want);
        eprintln!("{name} one-hot column {j}: {d}");
        assert!(
            d.max_rel <= ONE_HOT_MAX_REL,
            "{name} one-hot: max_rel {:.3e} over {ONE_HOT_MAX_REL:.0e}\n  {d}",
            d.max_rel
        );
    }
}

/// A window of rows is the one-row answer M times over, bit for bit.
///
/// `q4k_qmv_m*` reads a weight block once and accumulates it into M row
/// accumulators, where `q4k_qmv` reads the whole matrix once per row. What
/// makes that a kernel the window mode can keep rather than a tile it has to
/// refuse is that nothing else moves: the same blocks reach each row's
/// accumulator in the same order, so the two agree exactly and not to a
/// tolerance. `max_abs == 0.0` is the whole contract.
///
/// The widths past 4 are the tail: a group covering rows nobody asked for
/// repeats the last one and must not store it.
#[test]
fn the_multi_row_matvec_is_the_one_row_matvec() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    let ctx = Context::new().unwrap();
    for name in ["head", "qkv"] {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let base = case.get(&format!("{name}_x")).to_f32();

        // Rows that differ from one another, so a wrong stride or a row
        // reading its neighbour's accumulator shows up as a wrong answer
        // instead of a lucky match.
        let row_of = |m: usize| -> Vec<f32> {
            base.iter()
                .enumerate()
                .map(|(i, v)| v * (1.0 + m as f32 * 0.25) + ((m * 13 + i) % 7) as f32 * 1e-3)
                .collect()
        };

        for width in [2usize, 3, 4, 5, 8] {
            let mut all = Vec::with_capacity(width * cols);
            let mut want = Vec::new();
            for m in 0..width {
                let r = row_of(m);
                // `qmv` at one row is the one-row kernel whatever the knobs
                // say, so this baseline is the thing being replaced.
                let one = Array::from_f32_as_bf16(&ctx, &r, &[1, cols]);
                want.extend(qmv(&ctx, &one, &w).to_vec_f32(&ctx));
                all.extend(r);
            }
            let x = Array::from_f32_as_bf16(&ctx, &all, &[width, cols]);
            // Both multi-row kernels: the one that shares the weight read and
            // the one that also unpacks the codes once. Only one of them is
            // dispatched for this format by default, and both owe the same
            // answer.
            for hoist in [Hoist::Off, Hoist::All] {
                set_qmv_hoist(hoist);
                let d = compare(&qmv(&ctx, &x, &w).to_vec_f32(&ctx), &want);
                assert_eq!(d.max_abs, 0.0, "{name} at M = {width}, {hoist:?}: {d}");
            }
            set_qmv_hoist(Hoist::Q6K);
        }
    }
}

#[test]
fn matvec_matches_a_float64_reference() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    let ctx = Context::new().unwrap();
    for name in ["head", "qkv"] {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));

        // float32 activations first: the kernel's own accumulation is float,
        // so this isolates the product from the bfloat16 conversion.
        let xb = case.get(&format!("{name}_x"));
        let x32 = Array::from_f32(&ctx, &xb.to_f32(), &[1, cols]);
        let want = case.f32(&format!("{name}_y"));
        let d = compare(&qmv(&ctx, &x32, &w).to_vec_f32(&ctx), &want);
        eprintln!("{name} matvec, float32 activations: {d}");
        assert!(
            d.max_rel <= MATVEC_MAX_REL,
            "{name} float32 matvec: max_rel {:.3e}\n  {d}",
            d.max_rel
        );

        // And in bfloat16, which is what the model will hand it. The reference
        // was built from the same bfloat16 values, so the extra error here is
        // the output rounding -- one bfloat16 ULP, ~2^-9 relative.
        let x = case.array(&ctx, &format!("{name}_x")).reshape(&[1, cols]);
        let d = compare(&qmv(&ctx, &x, &w).to_vec_f32(&ctx), &want);
        eprintln!("{name} matvec, bfloat16 activations: {d}");
        assert!(
            d.max_rel <= 1.0 / 256.0,
            "{name} bfloat16 matvec: max_rel {:.3e}\n  {d}",
            d.max_rel
        );
    }
}

/// The three-tensor layout, read back through the loader that will read the
/// artifact.
///
/// `dump_q4k.py` writes a one-module artifact beside the reference -- the same
/// three tensors, in safetensors, with a `config.json` whose quantization
/// block says `q4k`. Storage is a decision until something writes it down and
/// something else reads it back; this is the something else, and it fails if
/// the names, the dtypes or the per-row shapes drift.
#[test]
fn the_loader_reads_the_three_tensors() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    let dir = refs_root().join("q4k/artifact");
    if !dir.exists() {
        eprintln!("skipping: no {}; rerun tools/dump_q4k.py", dir.display());
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let module = "language_model.model.layers.0.linear_attn.out_proj";
    let w = ws.kquant_matrix(&ctx, module);
    assert_eq!(
        (w.out_dims(), w.in_dims()),
        (case.meta_usize("head_rows"), case.meta_usize("head_cols"))
    );

    // The loaded matrix has to be the same matrix, not merely a well-shaped
    // one: run it through the dequantizer and compare with ggml's own output.
    let got = dequantize(&ctx, &w, Dtype::F32);
    let d = compare(&got.to_vec_f32(&ctx), &case.f32("head_deq"));
    assert_eq!(d.max_abs, 0.0, "loaded q4k module: {d}");
}

/// The token axis: M rows of `x` must each give the M = 1 answer.
///
/// ggml's kernel walks the columns of `src1` on the grid's y, and the port
/// keeps that -- it is what gives a q4k artifact a prefill at all. Every row is
/// independent by construction, so stacking a vector M times and finding M
/// different answers means the axis is addressed wrongly.
#[test]
fn the_token_axis_repeats_the_same_answer() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    let ctx = Context::new().unwrap();
    for name in ["head", "qkv"] {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let rows = case.meta_usize(&format!("{name}_rows"));
        let one = case.get(&format!("{name}_x")).to_f32();

        let y1 = qmv(&ctx, &Array::from_f32(&ctx, &one, &[1, cols]), &w).to_vec_f32(&ctx);

        // Three tokens: the same row, a scaled copy, and zeros -- so a wrong
        // stride shows up as the wrong row rather than as a lucky match.
        let mut many = one.clone();
        many.extend(one.iter().map(|v| v * 0.5));
        many.extend(std::iter::repeat_n(0.0f32, cols));
        let y3 = qmv(&ctx, &Array::from_f32(&ctx, &many, &[3, cols]), &w).to_vec_f32(&ctx);

        let d = compare(&y3[..rows], &y1);
        assert_eq!(d.max_abs, 0.0, "{name} token 0: {d}");
        let half: Vec<f32> = y3[rows..2 * rows].to_vec();
        let d = compare(&half, &y1.iter().map(|v| v * 0.5).collect::<Vec<_>>());
        assert!(d.max_rel <= MATVEC_MAX_REL, "{name} token 1 (halved): {d}");
        // A zero row still carries the minima: `- dmin * m * sum(y)` is zero
        // only because `sum(y)` is, so this is not a trivially empty check.
        assert!(
            y3[2 * rows..].iter().all(|v| *v == 0.0),
            "{name} token 2 should be all zeros"
        );
    }
}

/// The forged artifact, module by module, against ggml's own dequantizer.
///
/// The cases above pin the kernel on two matrices that never went through a
/// safetensors file. This asks the question a verdict is actually read off:
/// **are the weights the port holds the weights ggml wrote?** It covers the
/// forge's split, the index, the loader and the kernel's read path at once,
/// over all 91 converted modules.
///
/// `tools/q4k_audit.py` writes the digests; an order-sensitive 64-bit sum over
/// the float32 bits, because the alternative is 0.85 GB of reference floats
/// and a tolerance would defeat `max_abs == 0.0`.
///
/// ```
/// ORNITH_GGML_BUILD=../.ggml-build python3 tools/q4k_audit.py <artifact>
/// ORNITH_MODEL=<artifact> cargo test --release --test q4k -- --ignored
/// ```
#[test]
#[ignore = "loads a whole forged artifact; run with --ignored and ORNITH_MODEL"]
fn the_forged_artifact_reads_back_as_ggml_wrote_it() {
    const MIX: u64 = 0x9E37_79B9_7F4A_7C15;
    const ADD: u64 = 0x1656_67B1_9E37_79F9;
    fn digest(v: &[f32]) -> u64 {
        v.iter().enumerate().fold(0u64, |h, (i, x)| {
            let m = (i as u64).wrapping_mul(MIX).wrapping_add(ADD);
            h.wrapping_add((x.to_bits() as u64).wrapping_mul(m))
        })
    }

    let dir = ornith_rs::refs::model_dir();
    let audit = dir.join("q4k_audit.json");
    if !audit.exists() {
        eprintln!("skipping: no {}; run tools/q4k_audit.py", audit.display());
        return;
    }
    let raw = std::fs::read(&audit).unwrap();
    let doc = ornith_rs::json::parse(&raw).unwrap();
    let mods = doc.as_obj().expect("q4k_audit.json is an object");

    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let mut n = 0;
    for (module, want) in mods {
        let w = ws.kquant_matrix(&ctx, module);
        let got = dequantize(&ctx, &w, Dtype::F32).to_vec_f32(&ctx);
        let want = want.get("digest").and_then(|v| v.as_str()).expect("digest");
        let want = u64::from_str_radix(want.trim_start_matches("0x"), 16).unwrap();
        assert_eq!(
            digest(&got),
            want,
            "{module}: the port's dequantize disagrees with ggml's"
        );
        n += 1;
    }
    assert!(n > 0, "q4k_audit.json named no modules");
    eprintln!("{n} q4k modules agree with ggml, digest for digest");
}

/// The routed gather: `kernel_mul_mv_id` at M = 1.
///
/// One output row per (token, expert) pair, reading `x[lhs[r]]` against expert
/// `rhs[r]` -- `mx.gather_qmm`'s contract and what `SwitchGLU` produces. The
/// expert stride is the one thing the two-dimensional cases cannot exercise,
/// so it is checked twice: against a float64 reference over eleven rows whose
/// experts repeat and whose tokens do not, and again with a one-hot
/// activation, where the answer is one column of one expert and a stride that
/// lands on a neighbour cannot pass.
#[test]
fn the_routed_gather_picks_the_right_expert() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    if !case.has("moe_qs") {
        eprintln!("skipping: the reference predates the routed case; rerun dump_q4k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    let w = KMatrix::new(
        KQuant::Q4K,
        case.array(&ctx, "moe_qs"),
        case.array(&ctx, "moe_scales"),
        case.array(&ctx, "moe_dm"),
    );
    let experts = case.meta_usize("moe_experts");
    let rows = case.meta_usize("moe_rows");
    let cols = case.meta_usize("moe_cols");
    let n = case.meta_usize("moe_n");
    assert_eq!((w.num_experts(), w.out_dims(), w.in_dims()), (experts, rows, cols));

    // The stacked weights dequantize to the same values one at a time.
    let deq = dequantize(&ctx, &w, Dtype::F32);
    let d = compare(&deq.to_vec_f32(&ctx), &case.f32("moe_deq"));
    assert_eq!(d.max_abs, 0.0, "routed dequantize: {d}");

    let lhs = case.array(&ctx, "moe_lhs");
    let rhs = case.array(&ctx, "moe_rhs");
    let x = case.array(&ctx, "moe_x");
    let got = gather_qmv(&ctx, &x, &w, &lhs, &rhs).to_vec_f32(&ctx);
    let d = compare(&got, &case.f32("moe_y"));
    eprintln!("routed gather, {n} rows over {experts} experts: {d}");
    assert!(d.max_rel <= MATVEC_MAX_REL, "routed gather: {d}");

    // One-hot per row: output row `r` must be column `r` of expert `rhs[r]`.
    let j = case.meta_usize("head_onehot_j") % cols;
    let mut xs = vec![0.0f32; n * cols];
    for r in 0..n {
        xs[r * cols + j] = 1.0;
    }
    let xs = Array::from_f32(&ctx, &xs, &[n, cols]);
    let got = gather_qmv(&ctx, &xs, &w, &lhs, &rhs).to_vec_f32(&ctx);
    let deq = case.f32("moe_deq");
    let rhs_ids = case.get("moe_rhs").to_u32();
    let mut want = Vec::with_capacity(n * rows);
    for r in 0..n {
        let e = rhs_ids[r] as usize;
        want.extend((0..rows).map(|o| deq[(e * rows + o) * cols + j]));
    }
    let d = compare(&got, &want);
    eprintln!("routed one-hot, column {j}: {d}");
    assert!(d.max_rel <= ONE_HOT_MAX_REL, "routed one-hot: {d}");
}

// ---------------------------------------------------------------------------
// `qmm_t`: the tile, and the three things it has to be
// ---------------------------------------------------------------------------
//
// The mat-vec's contract cannot be reused wholesale here. `q4k_dequantize` is
// pinned bit-exactly because ggml performs the same operations in the same
// order; a product cannot be, because the order is the kernel's. For the gemm
// there is one more: it replaces a kernel that is already checked, so it must
// **agree with that kernel** -- and only to within the difference in summation
// order, which is why its accumulator tile is float and not ggml's half.
//
//   1. a float64 reference, as for the mat-vec, with a stated bound
//   2. a one-hot activation per row, where each output is one term and an
//      addressing error in either axis cannot hide inside a tolerance
//   3. the mat-vec itself, on the same rows

/// Two float32 sums over the same 2048 terms in different orders. Both walk K
/// in 32-wide steps; the gemm accumulates through `simdgroup_float8x8` and the
/// mat-vec through `simd_sum`, so this is reassociation and nothing else.
const REASSOCIATION_MAX_REL: f32 = 1e-5;

fn mrows(case: &RefCase, name: &str) -> usize {
    case.meta_usize(&format!("{name}_m"))
}

/// The prefill shape against a float64 reference, in both activation dtypes.
///
/// `MROWS` is 37 -- not a multiple of the tile's 32 -- and `tail` is 100 rows,
/// not a multiple of its 64, so between them the two partial-tile bounds are
/// both exercised.
#[test]
fn qmm_t_matches_a_float64_reference() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    if !case.has("head_xm") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q4k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    for name in ["head", "qkv", "tail"] {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let m = mrows(&case, name);
        let want = case.f32(&format!("{name}_ym"));

        let x32 = Array::from_f32(&ctx, &case.get(&format!("{name}_xm")).to_f32(), &[m, cols]);
        let d = compare(&qmm_t(&ctx, &x32, &w).to_vec_f32(&ctx), &want);
        eprintln!("{name} gemm, {m} rows, float32 activations: {d}");
        assert!(
            d.max_rel <= MATVEC_MAX_REL,
            "{name} float32 gemm: max_rel {:.3e}\n  {d}",
            d.max_rel
        );

        let xb = case.array(&ctx, &format!("{name}_xm")).reshape(&[m, cols]);
        let d = compare(&qmm_t(&ctx, &xb, &w).to_vec_f32(&ctx), &want);
        eprintln!("{name} gemm, {m} rows, bfloat16 activations: {d}");
        assert!(
            d.max_rel <= 1.0 / 256.0,
            "{name} bfloat16 gemm: max_rel {:.3e}\n  {d}",
            d.max_rel
        );
    }
}

/// A different one-hot per row, so the answer is a different column of the
/// dequantized matrix in every row of the output.
///
/// The mat-vec's one-hot case moves one column; this one moves 37 of them
/// through a kernel that addresses rows and columns on different axes of the
/// grid, which is where a tiled gemm goes wrong. It measures exactly zero.
#[test]
fn qmm_t_on_one_hot_rows_returns_columns() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    if !case.has("head_xm") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q4k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    for name in ["head", "qkv", "tail"] {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let rows = case.meta_usize(&format!("{name}_rows"));
        let m = mrows(&case, name);
        let deq = case.f32(&format!("{name}_deq"));

        // Columns spread over the whole input, one per output row.
        let js: Vec<usize> = (0..m).map(|r| (r * 61 + 7) % cols).collect();
        let mut xs = vec![0.0f32; m * cols];
        for (r, j) in js.iter().enumerate() {
            xs[r * cols + j] = 1.0;
        }
        let xs = Array::from_f32(&ctx, &xs, &[m, cols]);
        let got = qmm_t(&ctx, &xs, &w).to_vec_f32(&ctx);

        let mut want = Vec::with_capacity(m * rows);
        for j in &js {
            want.extend((0..rows).map(|o| deq[o * cols + j]));
        }
        let d = compare(&got, &want);
        eprintln!("{name} gemm one-hot, {m} rows: {d}");
        assert!(
            d.max_rel <= ONE_HOT_MAX_REL,
            "{name} gemm one-hot: max_rel {:.3e}\n  {d}",
            d.max_rel
        );
    }
}

/// The contract that makes the gemm a replacement rather than a second model:
/// on the same rows it must answer what the mat-vec answers.
///
/// Float32 activations, so the comparison is not swallowed by the bfloat16
/// output rounding both kernels would otherwise share. What is left is the
/// association -- the mat-vec factors `d` out of a block's sum where the gemm
/// dequantizes each weight first -- and it measures a few float32 ULPs.
#[test]
fn qmm_t_agrees_with_the_matvec() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    if !case.has("head_xm") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q4k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    for name in ["head", "qkv"] {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let m = mrows(&case, name);
        let x = Array::from_f32(&ctx, &case.get(&format!("{name}_xm")).to_f32(), &[m, cols]);

        let vec = qmv(&ctx, &x, &w).to_vec_f32(&ctx);
        let tile = qmm_t(&ctx, &x, &w).to_vec_f32(&ctx);
        let d = compare(&tile, &vec);
        eprintln!("{name} gemm against the mat-vec it replaces, {m} rows: {d}");
        assert!(
            d.max_rel <= REASSOCIATION_MAX_REL,
            "{name} gemm vs mat-vec: max_rel {:.3e} -- that is more than a \
             summation order apart\n  {d}",
            d.max_rel
        );
    }
}

/// The routed gemm: `kernel_mul_mm_id`, on rows grouped by expert.
///
/// The reference's run lengths are 5, 40, 3 and 22 over 70 rows, so the three
/// 32-row tiles straddle expert boundaries in three different ways -- the
/// first holds two experts, the second three, the third one. That is the whole
/// difference between this and the mat-vec, which reads one expert per row and
/// cannot get a run wrong.
#[test]
fn the_routed_gemm_picks_the_right_experts() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    if !case.has("moes_x") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q4k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    let w = KMatrix::new(
        KQuant::Q4K,
        case.array(&ctx, "moe_qs"),
        case.array(&ctx, "moe_scales"),
        case.array(&ctx, "moe_dm"),
    );
    let rows = case.meta_usize("moe_rows");
    let cols = case.meta_usize("moe_cols");
    let n = case.meta_usize("moes_n");

    let lhs = case.array(&ctx, "moes_lhs");
    let rhs = case.array(&ctx, "moes_rhs");
    let x = case.array(&ctx, "moes_x");

    let got = gather_qmm_t(&ctx, &x, &w, &lhs, &rhs).to_vec_f32(&ctx);
    let d = compare(&got, &case.f32("moes_y"));
    eprintln!("routed gemm, {n} sorted rows: {d}");
    assert!(d.max_rel <= MATVEC_MAX_REL, "routed gemm: {d}");

    // And against the mat-vec on the same rows, which is the same contract the
    // plain gemm carries.
    let vec = gather_qmv(&ctx, &x, &w, &lhs, &rhs).to_vec_f32(&ctx);
    let d = compare(&got, &vec);
    eprintln!("routed gemm against the routed mat-vec: {d}");
    assert!(
        d.max_rel <= REASSOCIATION_MAX_REL,
        "routed gemm vs mat-vec: max_rel {:.3e}\n  {d}",
        d.max_rel
    );

    // One-hot per row: output row `r` is column `j_r` of expert `rhs[r]`, so a
    // run boundary read one row early lands on the wrong expert's column.
    let deq = case.f32("moe_deq");
    let rhs_ids = case.get("moes_rhs").to_u32();
    let js: Vec<usize> = (0..n).map(|r| (r * 61 + 7) % cols).collect();
    let lhs_ids = case.get("moes_lhs").to_u32();
    let mut xs = vec![0.0f32; n * cols];
    for r in 0..n {
        // Row `r` of the output reads row `lhs[r]` of `x`, so the one-hot has
        // to be planted there rather than at `r`.
        xs[lhs_ids[r] as usize * cols + js[r]] = 1.0;
    }
    let xs = Array::from_f32(&ctx, &xs, &[n, cols]);
    let got = gather_qmm_t(&ctx, &xs, &w, &lhs, &rhs).to_vec_f32(&ctx);
    let mut want = Vec::with_capacity(n * rows);
    for r in 0..n {
        let e = rhs_ids[r] as usize;
        want.extend((0..rows).map(|o| deq[(e * rows + o) * cols + js[r]]));
    }
    let d = compare(&got, &want);
    eprintln!("routed gemm one-hot, {n} rows: {d}");
    assert!(d.max_rel <= ONE_HOT_MAX_REL, "routed gemm one-hot: {d}");
}

/// The run loop is an optimization, not a precondition.
///
/// mlx's `gather_qmm_rhs` requires the rows to be sorted by expert; this
/// kernel finds its own runs, so an unsorted batch is slower and not wrong.
/// That claim is worth a test rather than a comment: shuffle the same 70 rows
/// and the answers must follow the shuffle.
#[test]
fn the_routed_gemm_does_not_need_sorted_rows() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    if !case.has("moes_x") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q4k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    let w = KMatrix::new(
        KQuant::Q4K,
        case.array(&ctx, "moe_qs"),
        case.array(&ctx, "moe_scales"),
        case.array(&ctx, "moe_dm"),
    );
    let rows = case.meta_usize("moe_rows");
    let n = case.meta_usize("moes_n");
    let x = case.array(&ctx, "moes_x");
    let lhs = case.get("moes_lhs").to_u32();
    let rhs = case.get("moes_rhs").to_u32();

    // A fixed shuffle, so every tile holds a different expert in nearly every
    // row and the run loop degenerates to one pass per row.
    let perm: Vec<usize> = (0..n).map(|r| (r * 37 + 5) % n).collect();
    let plhs: Vec<u32> = perm.iter().map(|&r| lhs[r]).collect();
    let prhs: Vec<u32> = perm.iter().map(|&r| rhs[r]).collect();
    let plhs = Array::from_slice(&ctx, &plhs, &[n], Dtype::U32);
    let prhs = Array::from_slice(&ctx, &prhs, &[n], Dtype::U32);
    let got = gather_qmm_t(&ctx, &x, &w, &plhs, &prhs).to_vec_f32(&ctx);

    let want = case.f32("moes_y");
    let mut permuted = Vec::with_capacity(n * rows);
    for &r in &perm {
        permuted.extend_from_slice(&want[r * rows..(r + 1) * rows]);
    }
    let d = compare(&got, &permuted);
    eprintln!("routed gemm, {n} shuffled rows: {d}");
    assert!(d.max_rel <= MATVEC_MAX_REL, "routed gemm, unsorted: {d}");
}

/// The three tile widths are one kernel, and they owe each other an answer.
///
/// `NR1` is coupled to four other things -- the accumulator count, `lsmb`'s
/// offset and stride, and the output staging offset -- so a narrower tile is
/// not a constant edit. The one-hot cases above catch an addressing mistake on
/// their own; this catches the subtler one, where both widths are
/// self-consistent and disagree with each other. They may differ only by
/// summation order, and at 32 against 16 rows over the same K there is not
/// even any of that: each output element sums the same K terms in the same
/// eight-wide fragments either way.
#[test]
fn the_tile_widths_agree() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    let ctx = Context::new().unwrap();
    let w = matrix(&ctx, &case, "qkv");
    let k = w.in_dims();
    for m in [1usize, 5, 16, 17, 32, 37, 64, 65, 100] {
        let host: Vec<f32> = (0..m * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 97.0).collect();
        let x = Array::from_f32_as_bf16(&ctx, &host, &[m, k]);
        set_mm_rows(32);
        let a = qmm_t(&ctx, &x, &w).to_vec_f32(&ctx);
        for rows in [16usize, 64] {
            set_mm_rows(rows);
            let b = qmm_t(&ctx, &x, &w).to_vec_f32(&ctx);
            let d = compare(&b, &a);
            assert_eq!(
                d.max_abs, 0.0,
                "M={m}: the {rows}-row tile differs from the 32-row one: {d}"
            );
        }
        set_mm_rows(32);
    }
}

/// A pruned row is the row it came from, bit for bit.
///
/// FR-Spec drafts through `lm_head[ids]`, so the contract is not "close": the
/// gathered row holds the same whole super-blocks the original did, and the
/// same kernel reduces them in the same order, so the logit for a kept token
/// has to be the *same number* it would have been over the full vocabulary.
/// Anything else is a wrong token wearing a fast one's clothes.
#[test]
fn a_pruned_k_quant_row_is_the_row_it_came_from() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    let ctx = Context::new().unwrap();
    let w = matrix(&ctx, &case, "head");
    let (o, k) = (w.out_dims(), w.in_dims());
    assert!(o >= 64, "need rows to prune from");

    // A scattered subset, including the first and last rows.
    let ids: Vec<u32> = (0..o).step_by(7).map(|i| i as u32).chain([o as u32 - 1]).collect();
    let n = ids.len();
    let idx = Array::from_slice(&ctx, &ids, &[n], Dtype::U32);

    let host: Vec<f32> = (0..k).map(|i| ((i * 31 % 71) as f32 - 35.0) / 67.0).collect();
    let x = Array::from_f32_as_bf16(&ctx, &host, &[1, k]);

    let full = qmv(&ctx, &x, &w).to_vec_f32(&ctx);
    let pruned_head = ornith_rs::frspec::prune(&ctx, &Linear::K(w), &idx, n);
    let pruned = pruned_head.matmul(&ctx, &x).to_vec_f32(&ctx);

    assert_eq!(pruned.len(), n);
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(
            pruned[i], full[id as usize],
            "row {i} (vocabulary id {id}) came back {} where the full head has {}",
            pruned[i], full[id as usize]
        );
    }
}

/// `RMSNormGated` folded into the projection that reads it, against the two
/// dispatches it replaces.
///
/// Port against port, and that is the whole chain: `rms_norm_gated` is pinned
/// to mlx in `tests/gdn` and `qmv` to a float64 reference above, so an exact
/// match here pins the fused kernel to both. The prologue's reduction has to
/// be mlx's `rms_single_row` layout -- 32 lanes, four contiguous elements,
/// one `simd_sum` -- and a head of any other width would show up here as a
/// last-bit difference rather than as a wrong answer.
#[test]
fn the_fused_norm_projection_is_the_two_dispatches() {
    let Some(case) = RefCase::open_or_skip("q4k") else { return };
    let ctx = Context::new().unwrap();
    const HD: usize = 128;
    for name in ["head", "qkv"] {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        assert_eq!(cols % HD, 0);
        let heads = cols / HD;

        // Arbitrary but deterministic: the reference's own activations, and a
        // norm weight sliced out of them so it is neither ones nor tiny.
        let x = case.array(&ctx, &format!("{name}_x")).reshape(&[1, heads, HD]);
        let gate = case.array(&ctx, &format!("{name}_x")).reshape(&[1, heads, HD]);
        let nw = case.array(&ctx, &format!("{name}_x")).reshape(&[heads, HD]).index(0);

        let flat = [1usize, cols];
        let want = qmv(
            &ctx,
            &ornith_rs::ops::norm::rms_norm_gated(&ctx, &x, Some(&nw), &gate, 1e-6)
                .reshape(&flat),
            &w,
        );
        // Every width of the kernel, since the only thing `nr0` changes is
        // how many threadgroups repeat the prologue -- if it changed a value
        // the sweep that chooses it would be choosing between two models.
        for nr0 in [2usize, 4, 8, 16] {
            let was = ornith_rs::ops::kquant::set_rmsgate_nr0(nr0);
            let got = ornith_rs::ops::kquant::rmsgate_qmv(
                &ctx,
                &x.reshape(&flat),
                &w,
                &nw,
                &gate.reshape(&flat),
                HD,
                1e-6,
            );
            ornith_rs::ops::kquant::set_rmsgate_nr0(was);
            let Some(got) = got else { continue };
            let d = compare(&got.to_vec_f32(&ctx), &want.to_vec_f32(&ctx));
            assert_eq!(d.max_abs, 0.0, "{name} fused norm+projection, nr0={nr0}: {d}");
        }
    }
}
