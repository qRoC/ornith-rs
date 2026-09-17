//! The ops the vision tower adds, against mlx.
//!
//! Reference: tools/dump_vision_ops.py

use ornith_rs::array::Array;
use ornith_rs::metal::Context;
use ornith_rs::ops::conv::{pad_conv_weight_channels, patch_embed_conv3d};
use ornith_rs::ops::elementwise::{add_bias_last, gelu_exact, gelu_tanh};
use ornith_rs::ops::matmul::addmm;
use ornith_rs::ops::norm::layer_norm;
use ornith_rs::ops::sdpa::steel_attention;
use ornith_rs::refs::{RefCase, compare};

fn exact(name: &str, got: &[f32], want: &[f32]) {
    let d = compare(got, want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}

/// mlx compares NaN payloads bit for bit here: `compare` would see NaN != NaN
/// as a difference of NaN, which is not `> 0.0`, so the activation sweeps get
/// their own bitwise comparison.
fn exact_bits(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{name}: element {i} differs: got {g} ({:#010x}) want {w} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

#[test]
fn layer_norm_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("vision_ops") else { return };
    let ctx = Context::new().unwrap();
    let eps = case.meta().unwrap().get("eps").unwrap().as_f32().unwrap();
    let w = case.array(&ctx, "ln_weight");
    let b = case.array(&ctx, "ln_bias");
    for n in [1, 256, 280] {
        let x = case.array(&ctx, &format!("ln_x_{n}"));
        let y = layer_norm(&ctx, &x, &w, &b, eps);
        exact(&format!("layer_norm n={n}"), &y.to_vec_f32(&ctx), &case.f32(&format!("ln_y_{n}")));
    }
}

/// Both gelu forms over every bfloat16 input, the way `tests/activations.rs`
/// pins silu and sigmoid: these are the only two places the tower's arithmetic
/// leaves the elementary operations, so an ULP here would be invisible
/// anywhere else.
#[test]
fn gelu_matches_mlx_on_every_bfloat16() {
    let Some(case) = RefCase::open_or_skip("vision_ops") else { return };
    let ctx = Context::new().unwrap();
    let x = case.array(&ctx, "gelu_x");
    exact_bits("gelu (erf)", &gelu_exact(&ctx, &x).to_vec_f32(&ctx), &case.f32("gelu_exact"));
    exact_bits("gelu (tanh)", &gelu_tanh(&ctx, &x).to_vec_f32(&ctx), &case.f32("gelu_tanh"));
}

#[test]
fn addmm_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("vision_ops") else { return };
    let ctx = Context::new().unwrap();
    for tag in [
        "qkv",
        "qkv_odd",
        "proj_splitk",
        "proj_regular",
        "fc1",
        "fc2_splitk",
        "fc2_regular",
        "merge1_splitk",
        "merge1_regular",
        "merge2",
    ] {
        let x = case.array(&ctx, &format!("addmm_{tag}_x"));
        let w = case.array(&ctx, &format!("addmm_{tag}_w"));
        let b = case.array(&ctx, &format!("addmm_{tag}_b"));
        let y = addmm(&ctx, &x, &w, &b);
        exact(
            &format!("addmm {tag}"),
            &y.to_vec_f32(&ctx),
            &case.f32(&format!("addmm_{tag}_y")),
        );
    }
}

#[test]
fn patch_embed_conv3d_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("vision_ops") else { return };
    let ctx = Context::new().unwrap();
    let w = case.array(&ctx, "conv_w");
    let bias = case.array(&ctx, "conv_b");
    let wp = pad_conv_weight_channels(&ctx, &w, 16);
    for n in [256, 280, 1024] {
        let flat = case.array(&ctx, &format!("conv_x_{n}"));
        let x: Array = flat.reshape(&[n, 3, 2, 16, 16]);
        let y = patch_embed_conv3d(&ctx, &x, &wp, 3);
        exact(&format!("conv3d n={n}"), &y.to_vec_f32(&ctx), &case.f32(&format!("conv_y_{n}")));
        let yb = add_bias_last(&ctx, &y, &bias);
        exact(
            &format!("conv3d + bias n={n}"),
            &yb.to_vec_f32(&ctx),
            &case.f32(&format!("conv_yb_{n}")),
        );
    }
}

/// MTPLX's `_fused_sdpa`: pad head dim 72 up to 80, run mlx's fused *full*
/// attention, slice back. Nothing in the trunk reaches that kernel.
#[test]
fn vision_attention_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("vision_ops") else { return };
    let ctx = Context::new().unwrap();
    let scale = case.meta().unwrap().get("sdpa_scale").unwrap().as_f32().unwrap();
    for n in [256, 280, 1024] {
        let q = case.array(&ctx, &format!("sdpa_q_{n}"));
        let k = case.array(&ctx, &format!("sdpa_k_{n}"));
        let v = case.array(&ctx, &format!("sdpa_v_{n}"));
        let qp = ornith_rs::ops::elementwise::pad_last(&ctx, &q, 80);
        let kp = ornith_rs::ops::elementwise::pad_last(&ctx, &k, 80);
        let vp = ornith_rs::ops::elementwise::pad_last(&ctx, &v, 80);
        let y = steel_attention(&ctx, &qp, &kp, &vp, scale);
        let y = ornith_rs::ops::layout::slice_inner(&ctx, &y, 0, 72);
        exact(&format!("vision sdpa n={n}"), &y.to_vec_f32(&ctx), &case.f32(&format!("sdpa_y_{n}")));
    }
}
