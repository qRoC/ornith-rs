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

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::q4k::{Q4kMatrix, dequantize, qmv};
use ornith_rs::refs::{RefCase, compare, refs_root};
use ornith_rs::weights::WeightStore;

/// The kernel accumulates in float32 over `K` terms and the reference is
/// float64, so the two differ by rounding alone. `K = 2048` terms of
/// comparable magnitude put the expected relative error near `sqrt(K) * 2^-24`
/// ~ 2.7e-6; this is that with room, and the test prints what it measured so a
/// regression shows as a number rather than as a pass.
const MATVEC_MAX_REL: f32 = 1e-5;

/// One term per output, so only the association differs: ~16 float32 ULPs.
const ONE_HOT_MAX_REL: f32 = 1e-6;

fn matrix(ctx: &Context, case: &RefCase, name: &str) -> Q4kMatrix {
    Q4kMatrix::new(
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
    let w = ws.q4k_matrix(&ctx, module);
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
