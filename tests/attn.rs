//! matmul, softmax and attention against mlx.
//!
//! Reference: tools/dump_attn.py

use ornith_rs::metal::Context;
use ornith_rs::ops::matmul::matmul;
use ornith_rs::ops::sdpa::scaled_dot_product_attention;
use ornith_rs::ops::softmax::softmax;
use ornith_rs::refs::{RefCase, compare};

fn exact(name: &str, got: &[f32], want: &[f32]) {
    let d = compare(got, want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}

#[test]
fn matmul_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("attn") else { return };
    let ctx = Context::new().unwrap();

    // A broadcast batch axis: b is [2, 1, K, N] against a [2, 3, M, K].
    let a = case.array(&ctx, "mm_a");
    let b = case.array(&ctx, "mm_b");
    exact("matmul nn", &matmul(&ctx, &a, &b, false).to_vec_f32(&ctx), &case.f32("mm_out"));

    let at = case.array(&ctx, "mmt_a");
    let bt = case.array(&ctx, "mmt_b");
    exact("matmul nt", &matmul(&ctx, &at, &bt, true).to_vec_f32(&ctx), &case.f32("mmt_out"));
}

#[test]
fn softmax_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("attn") else { return };
    let ctx = Context::new().unwrap();
    let x = case.array(&ctx, "sm_x");
    exact("softmax precise", &softmax(&ctx, &x, true).to_vec_f32(&ctx), &case.f32("sm_precise"));
    exact("softmax plain", &softmax(&ctx, &x, false).to_vec_f32(&ctx), &case.f32("sm_plain"));
}

#[test]
fn attention_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("attn") else { return };
    let ctx = Context::new().unwrap();
    let scale = case.meta().unwrap().get("scale").unwrap().as_f32().unwrap();

    for (lq, lk, causal) in [
        (1, 512, false),   // fused vector, one pass
        (1, 2048, false),  // fused vector, two pass
        (4, 4, true),      // fused vector with a causal mask (MTP verify)
        (7, 7, true),      // still fused: the cutoff is 8
        (64, 64, true),    // unfused: head_dim 256 sends >8 to the ops path
        (512, 512, true),  // unfused, a real prefill
    ] {
        let tag = format!("q{lq}_k{lk}_{}", if causal { "c" } else { "nc" });
        let q = case.array(&ctx, &format!("attn_q_{tag}"));
        let k = case.array(&ctx, &format!("attn_k_{tag}"));
        let v = case.array(&ctx, &format!("attn_v_{tag}"));
        let o = scaled_dot_product_attention(&ctx, &q, &k, &v, scale, causal);
        exact(&format!("attn {tag}"), &o.to_vec_f32(&ctx), &case.f32(&format!("attn_o_{tag}")));
    }
}
