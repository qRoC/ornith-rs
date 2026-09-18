//! The whole language model against mlx-lm, layer by layer.
//!
//! Reference: tools/dump_model.py
//!
//! This is the test that matters: every op is already pinned individually, so
//! a failure here is a wiring error -- a weight read from the wrong name, a
//! cache written at the wrong offset, a residual added in the wrong place.

use ornith_rs::array::Array;
use ornith_rs::metal::Context;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::ops::norm::rms_norm;
use ornith_rs::ops::quantized::quantized_matmul;
use ornith_rs::refs::{RefCase, compare, model_dir};
use ornith_rs::weights::WeightStore;

#[test]
#[ignore = "loads the whole 28 GB model; run with --ignored"]
fn full_forward_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("model") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    ws.prefetch();

    let t0 = std::time::Instant::now();
    let model = TextModel::load(&ctx, &mut ws, cfg);
    ctx.sync();
    eprintln!(
        "loaded {:.1} GB in {:.1}s",
        ws.loaded_bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    let mut caches = model.make_cache();
    let ids_ref = case.get("ids");
    let seq = case.meta_usize("seq");
    let ids = Array::from_slice(&ctx, &ids_ref.to_u32(), &[seq], ornith_rs::array::Dtype::U32);

    check_pass(&ctx, &model, &mut caches, &case, "prefill", &ids);

    for step in 0..case.meta_usize("decode_steps") {
        let tok = case.get(&format!("decode{step}_id")).to_u32();
        let tok = Array::from_slice(&ctx, &tok, &[1], ornith_rs::array::Dtype::U32);
        check_pass(&ctx, &model, &mut caches, &case, &format!("decode{step}"), &tok);
    }
}

fn check_pass(
    ctx: &Context,
    model: &TextModel,
    caches: &mut [ornith_rs::model::cache::LayerCache],
    case: &RefCase,
    tag: &str,
    ids: &Array,
) {
    let n = ids.numel();
    let mut h = ornith_rs::ops::embedding::embedding(ctx, &model.embed_tokens, ids)
        .reshape(&[1, n, model.cfg.hidden_size]);
    check(case, ctx, &format!("{tag}_embed"), &h);

    for (i, layer) in model.layers.iter().enumerate() {
        h = layer.forward(ctx, &model.cfg, &h, &mut caches[i]);
        // Whichever layers the reference holds: eight in the frozen case,
        // all forty when `dump_model.py` is asked to bisect a divergence.
        if case.has(&format!("{tag}_layer{i}")) {
            check(case, ctx, &format!("{tag}_layer{i}"), &h);
        }
    }
    check(case, ctx, &format!("{tag}_prenorm"), &h);
    let normed = rms_norm(ctx, &h, Some(&model.norm), model.cfg.rms_norm_eps);
    check(case, ctx, &format!("{tag}_hidden"), &normed);
    let logits = quantized_matmul(ctx, &normed, &model.lm_head);
    check(case, ctx, &format!("{tag}_logits"), &logits);
}

fn check(case: &RefCase, ctx: &Context, name: &str, got: &Array) {
    let want = case.f32(name);
    let d = compare(&got.to_vec_f32(ctx), &want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}
