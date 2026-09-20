//! The whole vision pipeline against MTPLX plus `mlx_lm.generate`, greedy.
//!
//! Reference: tools/dump_vision_generate.py
//!
//! Pixels in, text out: reading the file, the preprocessing, the tower, the
//! chat template's list-shaped content, the pad expansion, the splice over the
//! embedded prefill chunks, and the decode loop. Token ids are compared as
//! well as the text, so a detokenizer slip and a model slip are told apart.

use ornith_rs::array::Array;
use ornith_rs::chat::{Content, Message, Part, Role, apply_chat_template};
use ornith_rs::generate::{DEFAULT_PREFILL_STEP_SIZE, Greedy, generate_step};
use ornith_rs::metal::Context;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::ops::layout::concat_leading;
use ornith_rs::refs::{RefCase, model_dir};
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::tokenizer::split::Dialect;
use ornith_rs::vision::preprocess::{Image, PreprocessorConfig, preprocess};
use ornith_rs::vision::splice::{VisionSplice, expand_image_pads};
use ornith_rs::vision::tower::VisionTower;
use ornith_rs::vision::VisionConfig;
use ornith_rs::weights::WeightStore;

#[test]
#[ignore = "loads the 25 GB trunk plus the 0.9 GB tower; run with --ignored --test-threads=1"]
fn vision_greedy_matches_mtplx() {
    let Some(case) = RefCase::open_or_skip("vision_generate") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let tk = Tokenizer::load_as(&dir, Dialect::Transformers).unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let (vcfg, spec) = VisionConfig::load(&dir).unwrap();
    let pre = PreprocessorConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg);
    let tower = VisionTower::load(&ctx, &mut ws, vcfg);
    ctx.sync();
    eprintln!("loaded {:.1} GB", ws.loaded_bytes() as f64 / 1e9);
    drop(ws);

    let files = case.json("files");
    let files = files.as_arr().unwrap();
    let prompts = case.strings("prompts");
    let texts = case.strings("texts");
    let ids = case.json("ids");
    let ids = ids.as_arr().unwrap();
    let max_tokens = case.meta_usize("max_tokens");
    assert_eq!(
        case.meta_usize("image_token_id") as u32,
        spec.image_token_id,
        "the reference used a different <|image_pad|>"
    );

    // One tower call per image file, kept across cases exactly as MTPLX's
    // digest cache keeps them.
    let mut tower_rows: std::collections::HashMap<String, (Array, usize)> =
        std::collections::HashMap::new();

    for (i, want_prompt) in prompts.iter().enumerate() {
        let names: Vec<String> = files[i]
            .as_arr()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        let mut blocks: Vec<Array> = Vec::new();
        let mut pad_counts: Vec<usize> = Vec::new();
        for name in &names {
            let entry = tower_rows.entry(name.clone()).or_insert_with(|| {
                let image = Image::read(&case.dir.join(name), None).unwrap();
                let p = preprocess(&image, &pre).unwrap();
                let pixels =
                    Array::from_f32(&ctx, &p.values, &[p.rows, p.row_len]);
                let t0 = std::time::Instant::now();
                let rows = tower.forward(&ctx, &pixels, &[p.grid]);
                ctx.sync();
                eprintln!(
                    "{name}: {}x{} -> {} rows in {:.2}s",
                    image.width,
                    image.height,
                    rows.shape[0],
                    t0.elapsed().as_secs_f64()
                );
                (rows, tower.cfg.pad_token_count(p.grid))
            });
            blocks.push(entry.0.clone());
            pad_counts.push(entry.1);
        }

        // The template renders one `<|image_pad|>` per `Part::Image`.
        let mut parts: Vec<Part> = names.iter().map(|_| Part::Image).collect();
        // The reference's question is the tail of the rendered prompt; rebuild
        // the same message from it rather than duplicating the list here.
        let question = question_of(want_prompt);
        parts.push(Part::Text(question.clone()));
        let msg = Message {
            role: Role::User,
            content: Content::Parts(parts),
            reasoning_content: None,
            tool_calls: Vec::new(),
        };
        let prompt = apply_chat_template(&[msg], &[], true, false).unwrap();
        assert_eq!(&prompt, want_prompt, "case {i}: rendered prompt");

        let prompt_ids = tk.encode(&prompt);
        let expanded =
            expand_image_pads(&prompt_ids, spec.image_token_id, &pad_counts).unwrap();
        let refs: Vec<&Array> = blocks.iter().collect();
        let mut splice = VisionSplice {
            image_pad_token_id: spec.image_token_id,
            embeddings: concat_leading(&ctx, &refs),
            pad_counts: pad_counts.clone(),
            // No cache to reuse in a test, so nothing keys the images.
        keys: Vec::new(),
        cursor: 0,
        };
        assert_eq!(
            splice.total_rows(),
            pad_counts.iter().sum::<usize>(),
            "case {i}: tower rows vs expanded pads"
        );

        let want_ids: Vec<u32> =
            ids[i].as_arr().unwrap().iter().map(|v| v.as_usize().unwrap() as u32).collect();

        let mut detok = tk.detokenizer();
        let mut got_ids: Vec<u32> = Vec::new();
        let mut text = String::new();
        let mut n = 0usize;
        let mut caches = model.make_cache();
        let t0 = std::time::Instant::now();
        generate_step(
            &ctx,
            &model,
            &mut caches,
            &expanded,
            &mut Greedy,
            max_tokens,
            DEFAULT_PREFILL_STEP_SIZE,
            Some(&mut splice),
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

        assert_eq!(splice.remaining(), 0, "case {i}: not every vision row was consumed");
        eprintln!(
            "case {i} ({}): {} tokens in {:.1}s -> {:?}",
            names.join(", "),
            got_ids.len(),
            t0.elapsed().as_secs_f64(),
            &text.chars().take(72).collect::<String>()
        );
        assert_eq!(got_ids, want_ids, "case {i}: token ids");
        assert_eq!(text, texts[i], "case {i}: text");
    }
}

/// The user's question, recovered from the rendered prompt: everything between
/// the last `<|vision_end|>` and the closing `<|im_end|>`.
fn question_of(prompt: &str) -> String {
    let after = prompt.rsplit("<|vision_end|>").next().unwrap();
    after.split("<|im_end|>").next().unwrap().to_string()
}
