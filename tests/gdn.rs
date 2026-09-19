//! The gated delta rule against mlx-lm.
//!
//! Reference: tools/dump_gdn.py

use ornith_rs::metal::Context;
use ornith_rs::ops::elementwise::silu;
use ornith_rs::ops::gdn::{
    beta, compute_g, concat_time, depthwise_conv1d, gated_delta_kernel, gated_delta_update,
};
use ornith_rs::refs::{RefCase, compare};

const TS: [usize; 3] = [1, 7, 512];

fn exact(name: &str, got: &[f32], want: &[f32]) {
    let d = compare(got, want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}

#[test]
fn causal_conv_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("gdn") else { return };
    let ctx = Context::new().unwrap();
    let w = case.array(&ctx, "conv1d_weight");
    for t in TS {
        let state = case.array(&ctx, &format!("conv_state_t{t}"));
        let qkv = case.array(&ctx, &format!("conv_qkv_t{t}"));

        // The concatenation is part of the layer, so check it separately:
        // an off-by-one here would look like a subtly wrong convolution.
        let conv_in = concat_time(&ctx, &state, &qkv);
        exact(
            &format!("conv concat T={t}"),
            &conv_in.to_vec_f32(&ctx),
            &case.f32(&format!("conv_in_t{t}")),
        );

        let out = silu(&ctx, &depthwise_conv1d(&ctx, &conv_in, &w));
        exact(
            &format!("conv+silu T={t}"),
            &out.to_vec_f32(&ctx),
            &case.f32(&format!("conv_out_t{t}")),
        );
    }
}

#[test]
fn gate_and_beta_match_mlx() {
    let Some(case) = RefCase::open_or_skip("gdn") else { return };
    let ctx = Context::new().unwrap();
    let a_log = case.array(&ctx, "A_log");
    let dt_bias = case.array(&ctx, "dt_bias");
    for t in TS {
        let a = case.array(&ctx, &format!("a_t{t}"));
        let b = case.array(&ctx, &format!("b_t{t}"));
        exact(
            &format!("compute_g T={t}"),
            &compute_g(&ctx, &a_log, &a, &dt_bias).to_vec_f32(&ctx),
            &case.f32(&format!("g_t{t}")),
        );
        exact(
            &format!("beta T={t}"),
            &beta(&ctx, &b).to_vec_f32(&ctx),
            &case.f32(&format!("beta_t{t}")),
        );
    }
}

#[test]
fn recurrence_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("gdn") else { return };
    let ctx = Context::new().unwrap();
    let a_log = case.array(&ctx, "A_log");
    let dt_bias = case.array(&ctx, "dt_bias");

    for t in TS {
        let q = case.array(&ctx, &format!("q_t{t}"));
        let k = case.array(&ctx, &format!("k_t{t}"));
        let v = case.array(&ctx, &format!("v_t{t}"));
        let a = case.array(&ctx, &format!("a_t{t}"));
        let b = case.array(&ctx, &format!("b_t{t}"));
        let s0 = case.array(&ctx, &format!("state_in_t{t}"));

        // The packed kernel: what this model always runs (Dk = 128, scalar
        // gate, no padding mask).
        let (y, s1) = gated_delta_update(&ctx, &q, &k, &v, &a, &b, &a_log, &dt_bias, &s0, None);
        exact(&format!("gdn y T={t}"), &y.to_vec_f32(&ctx), &case.f32(&format!("y_t{t}")));
        exact(
            &format!("gdn state T={t}"),
            &s1.to_vec_f32(&ctx),
            &case.f32(&format!("state_out_t{t}")),
        );

        // The unpacked fallback must agree with mlx's unpacked kernel too --
        // mlx-lm claims the two are bit-identical, and this is where that
        // claim gets checked on our side.
        let g = compute_g(&ctx, &a_log, &a, &dt_bias);
        let bt = beta(&ctx, &b);
        let (yu, su) = {
            // Force the generic kernel by asking for it directly.
            ornith_rs::ops::gdn::gated_delta_kernel_unpacked(&ctx, &q, &k, &v, &g, &bt, &s0)
        };
        exact(
            &format!("gdn y unpacked T={t}"),
            &yu.to_vec_f32(&ctx),
            &case.f32(&format!("y_unpacked_t{t}")),
        );
        exact(
            &format!("gdn state unpacked T={t}"),
            &su.to_vec_f32(&ctx),
            &case.f32(&format!("state_out_unpacked_t{t}")),
        );

        // And the packed kernel must match the unpacked one bit for bit, the
        // property mlx-lm's comment asserts by construction.
        let (yp, _) = gated_delta_kernel(&ctx, &q, &k, &v, &g, &bt, &s0, None);
        exact(
            &format!("packed == unpacked T={t}"),
            &yp.to_vec_f32(&ctx),
            &yu.to_vec_f32(&ctx),
        );
    }
}

/// The layer's whole front end, against mlx's `GatedDeltaNet.__call__`.
///
/// `causal_conv_matches_mlx` and `gate_and_beta_match_mlx` pin the pieces;
/// this pins the one kernel that replaced them, against the same reference
/// and at the same `max_abs == 0.0`. Reference: `tools/dump_gdn.py`.
#[test]
fn front_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("gdn") else { return };
    let ctx = Context::new().unwrap();
    let (hk, hv) = (case.meta_usize("hk"), case.meta_usize("hv"));
    let (dk, dv) = (case.meta_usize("dk"), case.meta_usize("dv"));
    let cw = case.array(&ctx, "conv1d_weight");
    let a_log = case.array(&ctx, "A_log");
    let dt_bias = case.array(&ctx, "dt_bias");
    let inv = (dk as f32).powf(-0.5);

    for t in TS {
        let state = case.array(&ctx, &format!("conv_state_t{t}"));
        let qkv = case.array(&ctx, &format!("conv_qkv_t{t}"));
        let a = case.array(&ctx, &format!("a_t{t}"));
        let b = case.array(&ctx, &format!("b_t{t}"));
        let f = ornith_rs::ops::gdn::gdn_front(
            &ctx, &state, &qkv, &cw, &a, &b, &a_log, &dt_bias, hk, dk, hv, dv,
            inv * inv, inv, 1e-6, true,
        )
        .expect("this model's shapes are the ones the kernel is written for");

        exact(&format!("front q T={t}"), &f.q.to_vec_f32(&ctx), &case.f32(&format!("front_q_t{t}")));
        exact(&format!("front k T={t}"), &f.k.to_vec_f32(&ctx), &case.f32(&format!("front_k_t{t}")));
        exact(&format!("front v T={t}"), &f.v.to_vec_f32(&ctx), &case.f32(&format!("front_v_t{t}")));
        exact(
            &format!("front conv window T={t}"),
            &f.conv_next.to_vec_f32(&ctx),
            &case.f32(&format!("front_conv_next_t{t}")),
        );
        exact(
            &format!("front conv input T={t}"),
            &f.conv_input.unwrap().to_vec_f32(&ctx),
            &case.f32(&format!("conv_in_t{t}")),
        );
        exact(
            &format!("front beta T={t}"),
            &f.beta.to_vec_f32(&ctx),
            &case.f32(&format!("beta_t{t}")),
        );
        exact(&format!("front g T={t}"), &f.g.to_vec_f32(&ctx), &case.f32(&format!("g_t{t}")));
    }
}

/// And against the ten dispatches it replaces, which is the control the
/// timing runs alternate against -- if the two ever disagreed, every A/B
/// would be comparing two different models.
#[test]
fn front_matches_the_ten_it_replaces() {
    let Some(case) = RefCase::open_or_skip("gdn") else { return };
    let ctx = Context::new().unwrap();
    let (hk, hv) = (case.meta_usize("hk"), case.meta_usize("hv"));
    let (dk, dv) = (case.meta_usize("dk"), case.meta_usize("dv"));
    let cw = case.array(&ctx, "conv1d_weight");
    let a_log = case.array(&ctx, "A_log");
    let dt_bias = case.array(&ctx, "dt_bias");
    let inv = (dk as f32).powf(-0.5);
    let key_dim = hk * dk;

    for t in TS {
        let state = case.array(&ctx, &format!("conv_state_t{t}"));
        let qkv = case.array(&ctx, &format!("conv_qkv_t{t}"));
        let a = case.array(&ctx, &format!("a_t{t}"));
        let b = case.array(&ctx, &format!("b_t{t}"));

        let conv_in = concat_time(&ctx, &state, &qkv);
        let conv_out = silu(&ctx, &depthwise_conv1d(&ctx, &conv_in, &cw));
        let sl = |start, len| ornith_rs::ops::layout::slice_inner(&ctx, &conv_out, start, len);
        let q = sl(0, key_dim).reshape(&[1, t, hk, dk]);
        let k = sl(key_dim, key_dim).reshape(&[1, t, hk, dk]);
        let v = sl(2 * key_dim, hv * dv).reshape(&[1, t, hv, dv]);
        let q = ornith_rs::ops::elementwise::scalar_mul(
            &ctx,
            &ornith_rs::ops::norm::rms_norm(&ctx, &q, None, 1e-6),
            inv * inv,
        );
        let k = ornith_rs::ops::elementwise::scalar_mul(
            &ctx,
            &ornith_rs::ops::norm::rms_norm(&ctx, &k, None, 1e-6),
            inv,
        );

        let f = ornith_rs::ops::gdn::gdn_front(
            &ctx, &state, &qkv, &cw, &a, &b, &a_log, &dt_bias, hk, dk, hv, dv,
            inv * inv, inv, 1e-6, false,
        )
        .expect("fused front");
        exact(&format!("fused q T={t}"), &f.q.to_vec_f32(&ctx), &q.to_vec_f32(&ctx));
        exact(&format!("fused k T={t}"), &f.k.to_vec_f32(&ctx), &k.to_vec_f32(&ctx));
        exact(&format!("fused v T={t}"), &f.v.to_vec_f32(&ctx), &v.to_vec_f32(&ctx));
        exact(
            &format!("fused beta T={t}"),
            &f.beta.to_vec_f32(&ctx),
            &beta(&ctx, &b).to_vec_f32(&ctx),
        );
        exact(
            &format!("fused g T={t}"),
            &f.g.to_vec_f32(&ctx),
            &compute_g(&ctx, &a_log, &a, &dt_bias).to_vec_f32(&ctx),
        );
    }
}

/// `Qwen3NextRMSNormGated`, one dispatch against mlx's two.
#[test]
fn rms_norm_gated_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("gdn") else { return };
    if !case.has("gated_out_t1") {
        return;
    }
    let ctx = Context::new().unwrap();
    let w = case.array(&ctx, "gated_weight");
    for t in TS {
        let y = case.array(&ctx, &format!("gated_y_t{t}"));
        let z = case.array(&ctx, &format!("gated_z_t{t}"));
        let got = ornith_rs::ops::norm::rms_norm_gated(&ctx, &y, Some(&w), &z, 1e-6);
        exact(
            &format!("RMSNormGated T={t}"),
            &got.to_vec_f32(&ctx),
            &case.f32(&format!("gated_out_t{t}")),
        );

        // And the two dispatches it replaces.
        let was = ornith_rs::ops::norm::set_fuse_gdn_norm(false);
        let apart = ornith_rs::ops::norm::rms_norm_gated(&ctx, &y, Some(&w), &z, 1e-6);
        ornith_rs::ops::norm::set_fuse_gdn_norm(was);
        exact(
            &format!("RMSNormGated fused == apart T={t}"),
            &got.to_vec_f32(&ctx),
            &apart.to_vec_f32(&ctx),
        );
    }
}
