//! Layer 0's gated-delta path, step by step, on the real weights.
//!
//! Reference: tools/dump_layer0.py

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::model::config::TextConfig;
use ornith_rs::ops::elementwise::{precise_swiglu, scalar_mul, silu};
use ornith_rs::ops::gdn::{concat_time, depthwise_conv1d, gated_delta_update};
use ornith_rs::ops::layout::slice_inner;
use ornith_rs::ops::matmul::matmul;
use ornith_rs::ops::norm::rms_norm;
use ornith_rs::ops::quantized::quantized_matmul;
use ornith_rs::refs::{RefCase, compare, model_dir};
use ornith_rs::weights::WeightStore;

const P: &str = "language_model.model.layers.0";

#[test]
fn gated_delta_layer_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("layer0") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();

    let check = |name: &str, got: &Array| {
        let d = compare(&got.to_vec_f32(&ctx), &case.f32(name));
        assert_eq!(d.max_abs, 0.0, "{name}: {d}");
    };

    let x = case.array(&ctx, "x");
    let (b, s) = (x.shape[0], x.shape[1]);

    let ln = ws.tensor(&ctx, &format!("{P}.input_layernorm.weight"));
    let normed = rms_norm(&ctx, &x, Some(&ln), cfg.rms_norm_eps);
    check("normed", &normed);

    let qkv_w = ws.quant_matrix(&ctx, &format!("{P}.linear_attn.in_proj_qkv"));
    let z_w = ws.quant_matrix(&ctx, &format!("{P}.linear_attn.in_proj_z"));
    let b_w = ws.tensor(&ctx, &format!("{P}.linear_attn.in_proj_b.weight"));
    let a_w = ws.tensor(&ctx, &format!("{P}.linear_attn.in_proj_a.weight"));

    let qkv = quantized_matmul(&ctx, &normed, &qkv_w);
    check("qkv", &qkv);
    let z = quantized_matmul(&ctx, &normed, &z_w)
        .reshape(&[b, s, cfg.linear_num_value_heads, cfg.linear_value_head_dim]);
    check("z", &z);

    let flat = normed.reshape(&[b * s, cfg.hidden_size]);
    let bb = matmul(&ctx, &flat, &b_w, true).reshape(&[b, s, cfg.linear_num_value_heads]);
    check("b", &bb);
    let aa = matmul(&ctx, &flat, &a_w, true).reshape(&[b, s, cfg.linear_num_value_heads]);
    check("a", &aa);

    let convw = ws.tensor(&ctx, &format!("{P}.linear_attn.conv1d.weight"));
    let state0 = Array::empty(&ctx, &[b, cfg.linear_conv_kernel_dim - 1, cfg.conv_dim()], qkv.dtype);
    ornith_rs::model::cache::zero(&ctx, &state0);
    let conv_in = concat_time(&ctx, &state0, &qkv);
    check("conv_in", &conv_in);
    let conv_raw = depthwise_conv1d(&ctx, &conv_in, &convw);
    check("conv_raw", &conv_raw);
    let conv_out = silu(&ctx, &conv_raw);
    check("conv_out", &conv_out);

    let key_dim = cfg.key_dim();
    let (hk, dk) = (cfg.linear_num_key_heads, cfg.linear_key_head_dim);
    let (hv, dv) = (cfg.linear_num_value_heads, cfg.linear_value_head_dim);
    let q = slice_inner(&ctx, &conv_out, 0, key_dim).reshape(&[b, s, hk, dk]);
    let k = slice_inner(&ctx, &conv_out, key_dim, key_dim).reshape(&[b, s, hk, dk]);
    let v = slice_inner(&ctx, &conv_out, 2 * key_dim, cfg.value_dim()).reshape(&[b, s, hv, dv]);
    let inv = (dk as f32).powf(-0.5);
    let q = scalar_mul(&ctx, &rms_norm(&ctx, &q, None, 1e-6), inv * inv);
    let k = scalar_mul(&ctx, &rms_norm(&ctx, &k, None, 1e-6), inv);
    check("q", &q);
    check("k", &k);
    check("v", &v);

    let a_log = ws.tensor(&ctx, &format!("{P}.linear_attn.A_log"));
    let dt_bias = ws.tensor(&ctx, &format!("{P}.linear_attn.dt_bias"));
    let st = Array::empty(&ctx, &[b, hv, dv, dk], Dtype::F32);
    ornith_rs::model::cache::zero(&ctx, &st);
    let (out, state) = gated_delta_update(&ctx, &q, &k, &v, &aa, &bb, &a_log, &dt_bias, &st, None);
    check("gdn_out", &out);
    check("gdn_state", &state);

    let gnorm = ws.tensor(&ctx, &format!("{P}.linear_attn.norm.weight"));
    let normed_out = rms_norm(&ctx, &out, Some(&gnorm), cfg.rms_norm_eps);
    let gated = precise_swiglu(&ctx, &z, &normed_out);
    check("normed_gated", &gated);

    let out_proj = ws.quant_matrix(&ctx, &format!("{P}.linear_attn.out_proj"));
    let r = quantized_matmul(&ctx, &gated.reshape(&[b, s, hv * dv]), &out_proj);
    check("r", &r);
}
