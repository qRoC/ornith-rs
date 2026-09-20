//! Perplexity over a fixed corpus, from the port rather than from mlx.
//!
//! `tools/dump_artifact_baseline.py` has computed this since Phase 0 and it
//! goes through `mlx_lm.load`, so it stops working on exactly the artifacts
//! this project now ships: **mlx cannot load a k-quant**. That was tolerable
//! while perplexity was a rebuild check on the affine parent. It stopped
//! being tolerable in Phase 44, which left `q6k-full` against
//! `q6k-perexpert` at a sign test of 36 of 57, p = 0.063 -- a lead and not a
//! win -- with both artifacts k-quant and the only instrument that could
//! break the tie unable to load either of them.
//!
//! So the port computes it, the same way `bin/probes` came to exist.
//!
//! **The definition is copied rather than invented**, from
//! `dump_artifact_baseline.py`: chunks of `CHUNK` tokens through one cache,
//! logits cast to float32 *before* the reduction, `logsumexp` over the last
//! axis, and the next token's log-probability scored for every position that
//! has one. The corpus's final token is not scored because nothing follows
//! it. `ops::sample::logsumexp_last_axis` is a port of mlx's own
//! `logsumexp.cpp`, so the reduction order is mlx's and not a second one
//! invented here -- the same care `bin/probes` takes by refusing to
//! reimplement it at all.
//!
//! **The oracle, and it exists because one artifact can be read by both.**
//! On an affine artifact mlx and the port both run, so `tools/dump_ppl.py`
//! writes mlx's *per-token* NLL in this file's layout and the two are
//! compared element by element rather than as two scalars that happen to
//! agree to a few digits. A mean is a summary; 6620 numbers are a test.
//!
//! One place the two deliberately differ: mlx sums each chunk in float32 and
//! accumulates the chunk sums in a Python float, while this sums the whole
//! corpus in `f64` in sequence order. Both are defensible and neither is the
//! measurement -- `nll.bin` is. The means are reported by each side from the
//! same per-token numbers, and `dump_ppl.py` prints both.
//!
//! ```text
//! ORNITH_MODEL=~/.mtplx/models/<artifact> ORNITH_INVARIANT=all \
//!   ORNITH_PPL_OUT=refs/artifact/<capture>-pinned \
//!   cargo run --release --bin ppl
//! ```
//!
//! `ORNITH_PPL_CORPUS` overrides the text and `ORNITH_PPL_CHUNK` the chunk
//! width. Both are recorded in `ppl.json`, because the chunk is a free
//! parameter of this instrument in exactly the way the prefill kernel was a
//! free parameter of a probe capture (`REBUILD.md` Phase 44) -- and the
//! remedy is the same one: the capture says what made it.

use std::path::PathBuf;

use ornith_rs::array::Dtype;
use ornith_rs::generate::model_call;
use ornith_rs::metal::Context;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::ops::elementwise::copy_cast;
use ornith_rs::ops::sample::logsumexp_last_axis;
use ornith_rs::refs::{instrument_json, json_esc, model_dir, refs_root};
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::tokenizer::split::Dialect;
use ornith_rs::weights::WeightStore;

/// `dump_artifact_baseline.py`'s `CHUNK`. Changing it changes the number:
/// every position still attends to the whole prefix, but the *kernel* a
/// 256-row call picks is not the one a 64-row call picks.
const CHUNK: usize = 256;

fn main() {
    let dir = model_dir();
    assert!(dir.exists(), "no model at {}", dir.display());
    let corpus = PathBuf::from(std::env::var("ORNITH_PPL_CORPUS").unwrap_or_else(|_| {
        refs_root().join("artifact/ppl_corpus.txt").to_string_lossy().into_owned()
    }));
    let out = PathBuf::from(std::env::var("ORNITH_PPL_OUT").unwrap_or_else(|_| {
        refs_root().join("artifact/port").to_string_lossy().into_owned()
    }));
    let chunk: usize = std::env::var("ORNITH_PPL_CHUNK")
        .map(|v| v.parse().expect("ORNITH_PPL_CHUNK is a token count"))
        .unwrap_or(CHUNK);
    std::fs::create_dir_all(&out).unwrap();

    let text = std::fs::read_to_string(&corpus)
        .unwrap_or_else(|e| panic!("{}: {e}", corpus.display()));

    let ctx = Context::new().expect("metal");
    let t0 = std::time::Instant::now();
    // `dump_artifact_baseline.py` tokenizes through `mlx_lm.load`, i.e.
    // transformers' own pre-tokenizer, with `add_special_tokens=False`.
    let tokenizer = Tokenizer::load_as(&dir, Dialect::Transformers).unwrap();
    let ids = tokenizer.encode(&text);
    assert!(ids.len() > 1, "a corpus of one token scores nothing");

    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg);
    ctx.sync();
    eprintln!(
        "{}: {:.1} GB in {:.1}s, {} tokens, chunk {}",
        ctx.device_name(),
        ws.loaded_bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64(),
        ids.len(),
        chunk
    );
    drop(ws);

    let mut caches = model.make_cache();
    let mut nll: Vec<f32> = Vec::with_capacity(ids.len() - 1);
    let mut i = 0usize;
    while i < ids.len() - 1 {
        let end = (i + chunk).min(ids.len());
        let piece = &ids[i..end];
        // Scored positions: row `r` of this chunk predicts `ids[i + r + 1]`,
        // so the corpus's last token is a row with no target and the chunk
        // is one shorter than the piece exactly once, at the end.
        let targets = &ids[i + 1..(i + 1 + piece.len()).min(ids.len())];
        let m = targets.len();
        assert!(m > 0, "a chunk with no targets should have ended the loop");

        let (h, _) = model_call(&ctx, &model, &mut caches, piece, None);
        let logits = model.lm_head.matmul(&ctx, &h);
        let l32 = copy_cast(&ctx, &logits, Dtype::F32);
        let lse = logsumexp_last_axis(&ctx, &l32);

        let vocab = l32.dim(-1);
        let rows = l32.to_vec_f32(&ctx);
        let lse = lse.to_vec_f32(&ctx);
        for (r, &t) in targets.iter().enumerate() {
            // -(logit[t] - logsumexp) in float32, which is the subtraction
            // mlx does before it gathers, in the other order and exactly.
            nll.push(lse[r] - rows[r * vocab + t as usize]);
        }
        eprint!("\r  {} / {} tokens scored", nll.len(), ids.len() - 1);
        i = end;
    }
    eprintln!();

    // f64 in sequence order: the per-token file is the measurement and this
    // is one summary of it. `dump_ppl.py` prints mlx's summary of the same
    // numbers beside it.
    let total: f64 = nll.iter().map(|&v| v as f64).sum();
    let n = nll.len();
    let mean = total / n as f64;

    let bytes: Vec<u8> = nll.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out.join("nll.bin"), &bytes).unwrap();
    std::fs::write(
        out.join("ppl.json"),
        format!(
            "{{\n \"model_dir\": \"{}\",\n \"source\": \"ornith-rs src/bin/ppl.rs\",\n \
             \"instrument\": {},\n \
             \"corpus\": {{\"file\": \"{}\", \"bytes\": {}, \"tokens\": {}}},\n \
             \"chunk\": {},\n \"tokens_scored\": {},\n \"mean_nll\": {},\n \"ppl\": {},\n \
             \"nll\": {{\"file\": \"nll.bin\", \"dtype\": \"float32\", \"shape\": [{}]}}\n}}\n",
            json_esc(&dir.to_string_lossy()),
            instrument_json(),
            json_esc(&corpus.file_name().unwrap().to_string_lossy()),
            text.len(),
            ids.len(),
            chunk,
            n,
            mean,
            mean.exp(),
            n,
        ),
    )
    .unwrap();
    println!("perplexity {:.4} over {} tokens (mean nll {:.6})", mean.exp(), n, mean);
    eprintln!("wrote {}/ppl.json and nll.bin ({} bytes)", out.display(), bytes.len());
}
