//! ggml's Q6_K against the port's kernel. The sibling of `tests/q4k.rs`.
//!
//! Reference: tools/dump_q6k.py. Same contract and same reason -- mlx has no
//! k-quant, so `ggml` takes the oracle's place: `q6k_dequantize` is held to
//! `max_abs == 0.0` against `dequantize_row_q6_K` on bytes `quantize_q6_K`
//! wrote, and the matvec, whose summation order belongs to the kernel, is
//! checked against a float64 reference and against a one-hot activation.
//!
//! **What is different from Q4_K, and it is what this file is for.** Q6_K's
//! sub-block scale is **signed** and its code is **biased by 32**, so
//! `w = d * sc * (q - 32)` and there is no minimum at all. A reader that keeps
//! Q4_K's unsigned, minimum-subtracting conventions is wrong by an offset and
//! a sign, and still produces fluent text -- which is exactly why the
//! dequantize check is bit-exact rather than a tolerance.

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::kquant::{KMatrix, KQuant, dequantize, gather_qmv, qmv};
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
