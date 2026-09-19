//! ggml's Q6_K against the port's kernel. The sibling of `tests/q4k.rs`.
//!
//! Reference: tools/dump_q6k.py. Same contract and same reason -- mlx has no
//! k-quant, so `ggml` takes the oracle's place: `q6k_dequantize` is held to
//! `max_abs == 0.0` against `dequantize_row_q6_K` on bytes `quantize_q6_K`
//! wrote, and the matvec, whose summation order belongs to the kernel, is
//! checked against a float64 reference and against a one-hot activation. The
//! tiled gemm adds a third check -- that it answers what the mat-vec answers.
//!
//! **What is different from Q4_K, and it is what this file is for.** Q6_K's
//! sub-block scale is **signed** and its code is **biased by 32**, so
//! `w = d * sc * (q - 32)` and there is no minimum at all. A reader that keeps
//! Q4_K's unsigned, minimum-subtracting conventions is wrong by an offset and
//! a sign, and still produces fluent text -- which is exactly why the
//! dequantize check is bit-exact rather than a tolerance.

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::kquant::{
    Hoist, KMatrix, KQuant, dequantize, gather_qmm_t, gather_qmv, qmm_t, qmv,
    set_qmv_hoist,
};
use ornith_rs::refs::{RefCase, compare};

/// `sqrt(K) * 2^-24` with room; the cases print what they measured.
const MATVEC_MAX_REL: f32 = 1e-5;
/// One term per output, so only the association differs.
const ONE_HOT_MAX_REL: f32 = 1e-6;

const CASES: [&str; 2] = ["attn", "shared"];

fn matrix(ctx: &Context, case: &RefCase, name: &str) -> KMatrix {
    KMatrix::new(
        KQuant::Q6K,
        case.array(ctx, &format!("{name}_qs")),
        case.array(ctx, &format!("{name}_scales")),
        case.array(ctx, &format!("{name}_dm")),
    )
}

#[test]
fn dequantize_matches_ggml() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    let ctx = Context::new().unwrap();
    for name in CASES {
        let w = matrix(&ctx, &case, name);
        let rows = case.meta_usize(&format!("{name}_rows"));
        let cols = case.meta_usize(&format!("{name}_cols"));
        assert_eq!((w.out_dims(), w.in_dims()), (rows, cols));

        let got = dequantize(&ctx, &w, Dtype::F32);
        let d = compare(&got.to_vec_f32(&ctx), &case.f32(&format!("{name}_deq")));
        assert_eq!(d.max_abs, 0.0, "{name} dequantize: {d}");
    }
}

/// A window of rows is the one-row answer M times over, bit for bit.
///
/// `q6k_qmv_m*` reads a weight block once and accumulates it into M row
/// accumulators, where `q6k_qmv` reads the whole matrix once per row. What
/// makes that a kernel the window mode can keep rather than a tile it has to
/// refuse is that nothing else moves: the same blocks reach each row's
/// accumulator in the same order, so the two agree exactly and not to a
/// tolerance. `max_abs == 0.0` is the whole contract.
///
/// The widths past 4 are the tail: a group covering rows nobody asked for
/// repeats the last one and must not store it.
#[test]
fn the_multi_row_matvec_is_the_one_row_matvec() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    let ctx = Context::new().unwrap();
    for name in CASES {
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
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    let ctx = Context::new().unwrap();
    for name in CASES {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let want = case.f32(&format!("{name}_y"));

        let xb = case.get(&format!("{name}_x"));
        let x32 = Array::from_f32(&ctx, &xb.to_f32(), &[1, cols]);
        let d = compare(&qmv(&ctx, &x32, &w).to_vec_f32(&ctx), &want);
        eprintln!("{name} matvec, float32 activations: {d}");
        assert!(d.max_rel <= MATVEC_MAX_REL, "{name} float32 matvec: {d}");

        let x = case.array(&ctx, &format!("{name}_x")).reshape(&[1, cols]);
        let d = compare(&qmv(&ctx, &x, &w).to_vec_f32(&ctx), &want);
        eprintln!("{name} matvec, bfloat16 activations: {d}");
        assert!(d.max_rel <= 1.0 / 256.0, "{name} bfloat16 matvec: {d}");
    }
}

/// `x = e_j` leaves one term per output, so the answer is a column of the
/// dequantized matrix -- and a sign or a bias read wrongly cannot hide in it.
#[test]
fn matvec_on_a_one_hot_returns_a_column() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    let ctx = Context::new().unwrap();
    for name in CASES {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let j = case.meta_usize(&format!("{name}_onehot_j"));

        let mut x = vec![0.0f32; cols];
        x[j] = 1.0;
        let x = Array::from_f32(&ctx, &x, &[1, cols]);
        let d = compare(
            &qmv(&ctx, &x, &w).to_vec_f32(&ctx),
            &case.f32(&format!("{name}_y_onehot")),
        );
        eprintln!("{name} one-hot column {j}: {d}");
        assert!(d.max_rel <= ONE_HOT_MAX_REL, "{name} one-hot: {d}");
    }
}

/// The routed gather over experts held at 8 bits today, which are the ones
/// Q6_K would take.
#[test]
fn the_routed_gather_picks_the_right_expert() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    let ctx = Context::new().unwrap();
    let w = KMatrix::new(
        KQuant::Q6K,
        case.array(&ctx, "moe_qs"),
        case.array(&ctx, "moe_scales"),
        case.array(&ctx, "moe_dm"),
    );
    let experts = case.meta_usize("moe_experts");
    let rows = case.meta_usize("moe_rows");
    let cols = case.meta_usize("moe_cols");
    assert_eq!((w.num_experts(), w.out_dims(), w.in_dims()), (experts, rows, cols));

    let d = compare(&dequantize(&ctx, &w, Dtype::F32).to_vec_f32(&ctx), &case.f32("moe_deq"));
    assert_eq!(d.max_abs, 0.0, "routed dequantize: {d}");

    let lhs = case.array(&ctx, "moe_lhs");
    let rhs = case.array(&ctx, "moe_rhs");
    let x = case.array(&ctx, "moe_x");
    let d = compare(
        &gather_qmv(&ctx, &x, &w, &lhs, &rhs).to_vec_f32(&ctx),
        &case.f32("moe_y"),
    );
    eprintln!("routed gather over {experts} experts: {d}");
    assert!(d.max_rel <= MATVEC_MAX_REL, "routed gather: {d}");
}

// ---------------------------------------------------------------------------
// `qmm_t`: the tile, on the format whose scale is signed
// ---------------------------------------------------------------------------
//
// `tests/q4k.rs` states the three checks and why the mat-vec's bit-exact
// contract is not available to a product. The one thing this file adds is the
// format: `Q6KFmt::dequantize` assembles four codes inside a uint32 and
// recovers them by dividing the super-scale by powers of 256, with the bias of
// 32 folded into a single subtracted `ml`. Get the sign of `scales` wrong and
// the one-hot case reads a negated column; get the bias wrong and it reads a
// shifted one.
//
// `shared` is [1024, 512] -- two super-blocks a row -- so the K loop runs 16
// times rather than 64, which is the shortest K any converted module has.

/// Two float32 sums over the same terms in different orders.
const REASSOCIATION_MAX_REL: f32 = 1e-5;

fn mrows(case: &RefCase, name: &str) -> usize {
    case.meta_usize(&format!("{name}_m"))
}

#[test]
fn qmm_t_matches_a_float64_reference() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    if !case.has("attn_xm") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q6k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    for name in CASES {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let m = mrows(&case, name);
        let want = case.f32(&format!("{name}_ym"));

        let x32 = Array::from_f32(&ctx, &case.get(&format!("{name}_xm")).to_f32(), &[m, cols]);
        let d = compare(&qmm_t(&ctx, &x32, &w).to_vec_f32(&ctx), &want);
        eprintln!("{name} gemm, {m} rows, float32 activations: {d}");
        assert!(d.max_rel <= MATVEC_MAX_REL, "{name} float32 gemm: {d}");

        let xb = case.array(&ctx, &format!("{name}_xm")).reshape(&[m, cols]);
        let d = compare(&qmm_t(&ctx, &xb, &w).to_vec_f32(&ctx), &want);
        eprintln!("{name} gemm, {m} rows, bfloat16 activations: {d}");
        assert!(d.max_rel <= 1.0 / 256.0, "{name} bfloat16 gemm: {d}");
    }
}

#[test]
fn qmm_t_on_one_hot_rows_returns_columns() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    if !case.has("attn_xm") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q6k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    for name in CASES {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let rows = case.meta_usize(&format!("{name}_rows"));
        let m = mrows(&case, name);
        let deq = case.f32(&format!("{name}_deq"));

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
        assert!(d.max_rel <= ONE_HOT_MAX_REL, "{name} gemm one-hot: {d}");
    }
}

#[test]
fn qmm_t_agrees_with_the_matvec() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    if !case.has("attn_xm") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q6k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    for name in CASES {
        let w = matrix(&ctx, &case, name);
        let cols = case.meta_usize(&format!("{name}_cols"));
        let m = mrows(&case, name);
        let x = Array::from_f32(&ctx, &case.get(&format!("{name}_xm")).to_f32(), &[m, cols]);

        let d = compare(
            &qmm_t(&ctx, &x, &w).to_vec_f32(&ctx),
            &qmv(&ctx, &x, &w).to_vec_f32(&ctx),
        );
        eprintln!("{name} gemm against the mat-vec it replaces, {m} rows: {d}");
        assert!(
            d.max_rel <= REASSOCIATION_MAX_REL,
            "{name} gemm vs mat-vec: max_rel {:.3e} -- more than a summation \
             order apart\n  {d}",
            d.max_rel
        );
    }
}

/// The routed gemm, with the run lengths 5, 40, 3, 22 that make the three
/// 32-row tiles straddle expert boundaries three different ways.
#[test]
fn the_routed_gemm_picks_the_right_experts() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    if !case.has("moes_x") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q6k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    let w = KMatrix::new(
        KQuant::Q6K,
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

    let vec = gather_qmv(&ctx, &x, &w, &lhs, &rhs).to_vec_f32(&ctx);
    let d = compare(&got, &vec);
    eprintln!("routed gemm against the routed mat-vec: {d}");
    assert!(d.max_rel <= REASSOCIATION_MAX_REL, "routed gemm vs mat-vec: {d}");

    let deq = case.f32("moe_deq");
    let rhs_ids = case.get("moes_rhs").to_u32();
    let lhs_ids = case.get("moes_lhs").to_u32();
    let js: Vec<usize> = (0..n).map(|r| (r * 61 + 7) % cols).collect();
    let mut xs = vec![0.0f32; n * cols];
    for r in 0..n {
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

/// The activation gather the MoE block used to do is the gemm's own indexing.
///
/// `switch_glu_rows` hands this kernel `x` and the permutation where it used
/// to hand it `x[order]` and the identity -- which removes a 67 MB copy a
/// layer at a prefill chunk (`REBUILD.md` Phase 36). The two are the same
/// tile loading the same row of the same numbers against the same weights,
/// so the test is `max_abs == 0.0` and not a tolerance: nothing here may
/// reassociate, and the one failure this shape can have -- a row read from
/// the wrong place -- is not small.
#[test]
fn indexing_the_rows_is_gathering_them() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    if !case.has("moes_x") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q6k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    let w = KMatrix::new(
        KQuant::Q6K,
        case.array(&ctx, "moe_qs"),
        case.array(&ctx, "moe_scales"),
        case.array(&ctx, "moe_dm"),
    );
    let cols = case.meta_usize("moe_cols");
    let n = case.meta_usize("moes_n");
    let x = case.array(&ctx, "moes_x");
    let lhs = case.array(&ctx, "moes_lhs");
    let rhs = case.array(&ctx, "moes_rhs");

    // `xs[r] = x[lhs[r]]`, which is what `gather_rows_div` wrote. bf16 to f32
    // and back is exact, so the copy is the copy.
    let lhs_ids = case.get("moes_lhs").to_u32();
    let xh = x.to_vec_f32(&ctx);
    let mut gathered = Vec::with_capacity(n * cols);
    for r in 0..n {
        let src = lhs_ids[r] as usize * cols;
        gathered.extend_from_slice(&xh[src..src + cols]);
    }
    let xs = match x.dtype {
        Dtype::BF16 => Array::from_f32_as_bf16(&ctx, &gathered, &[n, cols]),
        _ => Array::from_f32(&ctx, &gathered, &[n, cols]),
    };
    let ident: Vec<u32> = (0..n as u32).collect();
    let ident = Array::from_slice(&ctx, &ident, &[n], Dtype::U32);

    let copied = gather_qmm_t(&ctx, &xs, &w, &ident, &rhs).to_vec_f32(&ctx);
    let indexed = gather_qmm_t(&ctx, &x, &w, &lhs, &rhs).to_vec_f32(&ctx);
    let d = compare(&indexed, &copied);
    eprintln!("routed gemm, {n} rows indexed against gathered: {d}");
    assert_eq!(d.max_abs, 0.0, "indexing the rows is not gathering them: {d}");
}

/// Unsorted rows are slower, not wrong -- see `tests/q4k.rs`.
#[test]
fn the_routed_gemm_does_not_need_sorted_rows() {
    let Some(case) = RefCase::open_or_skip("q6k") else { return };
    if !case.has("moes_x") {
        eprintln!("skipping: the reference predates the gemm; rerun dump_q6k.py");
        return;
    }
    let ctx = Context::new().unwrap();
    let w = KMatrix::new(
        KQuant::Q6K,
        case.array(&ctx, "moe_qs"),
        case.array(&ctx, "moe_scales"),
        case.array(&ctx, "moe_dm"),
    );
    let rows = case.meta_usize("moe_rows");
    let n = case.meta_usize("moes_n");
    let x = case.array(&ctx, "moes_x");
    let lhs = case.get("moes_lhs").to_u32();
    let rhs = case.get("moes_rhs").to_u32();

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
