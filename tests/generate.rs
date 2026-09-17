//! The whole pipeline against `mlx_lm.generate`, greedy.
//!
//! Reference: tools/dump_generate.py
//!
//! Text in, text out: the tokenizer, the chat template, the prefill chunking,
//! the argmax and the streaming detokenizer all at once. Token ids are
//! compared as well as the text, so a detokenizer slip and a sampling slip are
//! told apart.

use ornith_rs::generate::{DEFAULT_PREFILL_STEP_SIZE, Greedy, Sampler, generate_step};
use ornith_rs::metal::Context;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::refs::{RefCase, model_dir};
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::tokenizer::split::Dialect;
use ornith_rs::weights::WeightStore;

#[test]
#[ignore = "loads the whole 25 GB model; run with --ignored"]
fn greedy_matches_mlx_lm() {
    let Some(case) = RefCase::open_or_skip("generate") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    // mlx-lm reaches the tokenizer through `AutoTokenizer`, so the reference
    // is transformers' dialect.
    let tk = Tokenizer::load_as(&dir, Dialect::Transformers).unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg);
    ctx.sync();
    drop(ws);

    let prompts = case.strings("prompts");
    let texts = case.strings("texts");
    let kinds = case.strings("kinds");
    let ids = case.json("ids");
    let ids = ids.as_arr().unwrap();
    let max_tokens = case.meta_usize("max_tokens");

    for (i, prompt) in prompts.iter().enumerate() {
        let want_ids: Vec<u32> =
            ids[i].as_arr().unwrap().iter().map(|v| v.as_usize().unwrap() as u32).collect();
        let prompt_ids = tk.encode(prompt);
        let mut caches = model.make_cache();
        let mut detok = tk.detokenizer();
        let mut got_ids: Vec<u32> = Vec::new();
        let mut text = String::new();
        let mut n = 0usize;

        let t0 = std::time::Instant::now();
        generate_step(
            &ctx,
            &model,
            &mut caches,
            &prompt_ids,
            &mut Greedy,
            max_tokens,
            DEFAULT_PREFILL_STEP_SIZE,
            None,
            |_, _| {},
            |token| {
                got_ids.push(token);
                if tk.eos_ids.contains(&token) {
                    return false;
                }
                detok.add_token(token);
                n += 1;
                text.push_str(&detok.last_segment());
                n < max_tokens
            },
        );
        detok.finalize();
        text.push_str(&detok.last_segment());

        eprintln!(
            "[{i}] {} prompt {} tokens, {} generated in {:.1}s",
            kinds[i],
            prompt_ids.len(),
            got_ids.len(),
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(got_ids, want_ids, "case {i} ({}): token ids", kinds[i]);
        assert_eq!(text, texts[i], "case {i} ({}): text", kinds[i]);
    }
}

/// The same loop with the real sampler, seeded the way the reference seeded
/// mlx's global key sequence.
///
/// Reference: tools/dump_sampler.py
#[test]
#[ignore = "loads the whole 25 GB model; run with --ignored"]
fn sampled_generation_matches_mlx_lm() {
    let Some(case) = RefCase::open_or_skip("sampler") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let tk = Tokenizer::load_as(&dir, Dialect::Transformers).unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg);
    ctx.sync();
    drop(ws);

    let prompt = case.json("run_prompt");
    let prompt = prompt.as_str().unwrap();
    let want_ids: Vec<u32> = case
        .json("run_ids")
        .as_arr()
        .unwrap()
        .iter()
        .map(|v| v.as_usize().unwrap() as u32)
        .collect();
    let want_text = case.json("run_text");
    let want_text = want_text.as_str().unwrap();
    let max_tokens = case.meta_usize("max_tokens");
    let seed = case.meta_usize("seed") as u64;

    let mut sampler = ornith_rs::sample::TopSampler::new(0.6, 0.95, 20, seed);
    let prompt_ids = tk.encode(prompt);
    let mut caches = model.make_cache();
    let mut detok = tk.detokenizer();
    let mut got_ids: Vec<u32> = Vec::new();
    let mut text = String::new();
    let mut n = 0usize;

    generate_step(
        &ctx,
        &model,
        &mut caches,
        &prompt_ids,
        &mut sampler as &mut dyn Sampler,
        max_tokens,
        DEFAULT_PREFILL_STEP_SIZE,
        None,
        |_, _| {},
        |token| {
            got_ids.push(token);
            if tk.eos_ids.contains(&token) {
                return false;
            }
            detok.add_token(token);
            n += 1;
            text.push_str(&detok.last_segment());
            n < max_tokens
        },
    );
    detok.finalize();
    text.push_str(&detok.last_segment());

    assert_eq!(got_ids, want_ids, "sampled token ids");
    assert_eq!(text, want_text, "sampled text");
}
