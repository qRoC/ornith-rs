//! rms_norm, rope and the fused elementwise ops against mlx.
//!
//! Reference: tools/dump_ops.py

use ornith_rs::metal::Context;
use ornith_rs::ops::elementwise::{add, precise_swiglu, sigmoid_gate, swiglu};
use ornith_rs::ops::norm::rms_norm;
use ornith_rs::ops::rope::rope;
use ornith_rs::refs::{RefCase, compare};

fn exact(name: &str, got: &[f32], want: &[f32]) {
    let d = compare(got, want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}

#[test]
fn rms_norm_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("ops") else { return };
    let ctx = Context::new().unwrap();
    let eps = case.meta().unwrap().get("eps").unwrap().as_f32().unwrap();

    let w = case.array(&ctx, "ln_weight");
    for t in [1, 7, 512] {
        let x = case.array(&ctx, &format!("rms_x_t{t}"));
        let y = rms_norm(&ctx, &x, Some(&w), eps);
        exact(&format!("rms T={t}"), &y.to_vec_f32(&ctx), &case.f32(&format!("rms_y_t{t}")));
    }

    // GatedDeltaNet normalises q and k with no weight: mlx passes a scalar 1
    // and a zero stride rather than taking a separate code path.
    for t in [1, 7] {
        let x = case.array(&ctx, &format!("rms_nw_x_t{t}"));
        let y = rms_norm(&ctx, &x, None, 1e-6);
        exact(&format!("rms weightless T={t}"), &y.to_vec_f32(&ctx), &case.f32(&format!("rms_nw_y_t{t}")));
    }

    // q_norm / k_norm run over head_dim = 256, which is under the looped
    // limit but wide enough to need 2 simdgroups.
    let qn = case.array(&ctx, "q_norm_weight");
    for t in [1, 7] {
        let x = case.array(&ctx, &format!("rms_hd_x_t{t}"));
        let y = rms_norm(&ctx, &x, Some(&qn), eps);
        exact(&format!("rms head_dim T={t}"), &y.to_vec_f32(&ctx), &case.f32(&format!("rms_hd_y_t{t}")));
    }
}

#[test]
fn rope_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("ops") else { return };
    let ctx = Context::new().unwrap();
    let meta = case.meta().unwrap();
    let dims = meta.get("rope_dims").unwrap().as_usize().unwrap();
    let theta = meta.get("rope_theta").unwrap().as_f32().unwrap();

    for (t, off) in [(1, 0), (1, 37), (7, 0), (64, 128)] {
        let x = case.array(&ctx, &format!("rope_x_t{t}_o{off}"));
        let y = rope(&ctx, &x, dims, theta, 1.0, off);
        let want = case.f32(&format!("rope_y_t{t}_o{off}"));
        exact(&format!("rope T={t} offset={off}"), &y.to_vec_f32(&ctx), &want);
    }
}

#[test]
fn elementwise_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("ops") else { return };
    let ctx = Context::new().unwrap();
    let a = case.array(&ctx, "ew_a");
    let b = case.array(&ctx, "ew_b");

    exact("swiglu", &swiglu(&ctx, &a, &b).to_vec_f32(&ctx), &case.f32("ew_swiglu"));
    exact(
        "precise_swiglu",
        &precise_swiglu(&ctx, &b, &a).to_vec_f32(&ctx),
        &case.f32("ew_precise_swiglu"),
    );
    exact("add", &add(&ctx, &a, &b).to_vec_f32(&ctx), &case.f32("ew_add"));

    let g = case.array(&ctx, "ew_gate");
    exact(
        "sigmoid_gate elementwise",
        &sigmoid_gate(&ctx, &g, &b).to_vec_f32(&ctx),
        &case.f32("ew_sigmoid_gate"),
    );
    let g1 = case.array(&ctx, "ew_gate1");
    exact(
        "sigmoid_gate broadcast",
        &sigmoid_gate(&ctx, &g1, &b).to_vec_f32(&ctx),
        &case.f32("ew_sigmoid_gate1"),
    );
}
