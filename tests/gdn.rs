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
