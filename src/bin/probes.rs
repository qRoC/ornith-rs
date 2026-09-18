//! Capture the wide probe set from the port, the way `dump_probes.py`
//! captures it from mlx.
//!
//! `Q4K.md` names this as the hidden cost of a k-quant: an artifact in a mode
//! mlx has no quantization mode for cannot be loaded by `mlx_lm.load`, so the
//! instrument every verdict since Phase 5 rests on stops working exactly when
//! the format under test is the interesting one. The port can run that
//! artifact -- that is what the port is -- so the port captures the probes.
//!
//! **What makes the two instruments interchangeable is not an argument, it is
//! a test.** Run this against an artifact mlx *can* load and compare the rows
//! to `dump_probes.py`'s: the port is bit-exact against mlx everywhere else,
//! and if it is bit-exact here then a port-side capture can be compared with
//! an mlx-side one without a correction term. `tools/probes_from_logits.py`
//! does that comparison.
//!
//! Only the logits are written here. The normalisation
//! (`lp = logits - mx.logsumexp(logits)`), the entropy and the top-1 are left
//! to `probes_from_logits.py`, which runs mlx's own reductions on them -- so
//! the one arithmetic step this file would have had to reimplement is instead
//! the same code on both sides.
//!
//! ```
//! cd tools && python3 export_probes.py
//! ORNITH_MODEL=~/.mtplx/models/<artifact> \
//!   ORNITH_PROBES_OUT=refs/artifact/<capture> cargo run --release --bin probes
//! ```

use std::path::PathBuf;

use ornith_rs::generate::{model_call, prefill};
use ornith_rs::json;
use ornith_rs::metal::Context;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::ops::quantized::quantized_matmul;
use ornith_rs::refs::{model_dir, refs_root};
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::tokenizer::split::Dialect;
use ornith_rs::weights::WeightStore;

/// `dump_probes.py`'s `CHUNK`. Chunking changes reduction shapes, so it has to
/// be the same number on both sides or the captures are not comparable.
const CHUNK: usize = 128;

struct Probe {
    group: String,
    id: String,
    text: String,
}

fn load_probes(path: &PathBuf) -> Vec<Probe> {
    let raw = std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}\n  run `cd tools && python3 export_probes.py`", path.display()));
    let v = json::parse(&raw).expect("probe_set.json");
    v.as_arr()
        .expect("probe_set.json is a list")
        .iter()
        .map(|p| Probe {
            group: p.get("group").and_then(|v| v.as_str()).unwrap().to_string(),
            id: p.get("id").and_then(|v| v.as_str()).unwrap().to_string(),
            text: p.get("text").and_then(|v| v.as_str()).unwrap().to_string(),
        })
        .collect()
}

/// JSON string escaping, enough for probe text: quotes, backslashes and the
/// control characters. Everything else goes out as UTF-8.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let dir = model_dir();
    assert!(dir.exists(), "no model at {}", dir.display());
    let out = PathBuf::from(
        std::env::var("ORNITH_PROBES_OUT").unwrap_or_else(|_| {
            refs_root().join("artifact/port").to_string_lossy().into_owned()
        }),
    );
    let set_path = PathBuf::from(
        std::env::var("ORNITH_PROBE_SET")
            .unwrap_or_else(|_| refs_root().join("probe_set.json").to_string_lossy().into_owned()),
    );
    let probes = load_probes(&set_path);
    std::fs::create_dir_all(&out).unwrap();
    // The tokenization is half of what makes two captures comparable, and it
    // costs nothing to check on its own: `ORNITH_PROBES_TOKENS_ONLY=1` writes
    // the ids and stops, so a tokenizer disagreement is a one-second run
    // rather than a model load and 64 forwards.
    let tokens_only = std::env::var("ORNITH_PROBES_TOKENS_ONLY").is_ok();

    let ctx = Context::new().expect("metal");
    let t0 = std::time::Instant::now();
    // `dump_probes.py` goes through `mlx_lm.load`, i.e. transformers' own
    // pre-tokenizer rather than the one `tokenizer.json` declares.
    let tokenizer = Tokenizer::load_as(&dir, Dialect::Transformers).unwrap();
    if tokens_only {
        let mut ids_json = String::new();
        for (n, probe) in probes.iter().enumerate() {
            let ids = tokenizer.encode(&probe.text);
            ids_json.push_str(&format!(
                "{}  {{\"id\": \"{}\", \"tokens\": {}, \"ids\": [{}]}}",
                if n == 0 { "" } else { ",\n" },
                esc(&probe.id),
                ids.len(),
                ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
            ));
        }
        std::fs::write(out.join("probe_tokens.json"), format!("[\n{ids_json}\n]\n")).unwrap();
        eprintln!("wrote {}/probe_tokens.json", out.display());
        return;
    }

    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg);
    ctx.sync();
    eprintln!(
        "{}: {:.1} GB in {:.1}s",
        ctx.device_name(),
        ws.loaded_bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    drop(ws);

    let mut rows: Vec<f32> = Vec::new();
    let mut vocab = 0usize;
    let mut meta = String::new();
    for (n, probe) in probes.iter().enumerate() {
        let ids = tokenizer.encode(&probe.text);
        assert!(!ids.is_empty(), "probe {} tokenized to nothing", probe.id);
        let mut caches = model.make_cache();
        let processed = prefill(&ctx, &model, &mut caches, &ids, CHUNK, None, |_, _| {});
        let (h, _) = model_call(&ctx, &model, &mut caches, &ids[processed..], None);
        let logits = quantized_matmul(&ctx, &h, &model.lm_head);
        let last = logits.numel() / logits.dim(-1);
        assert_eq!(last, 1, "the logit-producing step is a single token in mlx-lm");
        let lg = logits.to_vec_f32(&ctx);
        vocab = lg.len();
        let top = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        rows.extend_from_slice(&lg);
        meta.push_str(&format!(
            "{}  {{\"group\": \"{}\", \"id\": \"{}\", \"text\": \"{}\", \"tokens\": {}, \
             \"argmax_id\": {}}}",
            if n == 0 { "" } else { ",\n" },
            esc(&probe.group),
            esc(&probe.id),
            esc(&probe.text),
            ids.len(),
            top
        ));
        eprintln!(
            "{n:>3} {:<10} {:<18} {:>4} tok  -> {:?}",
            probe.group,
            probe.id,
            ids.len(),
            tokenizer.decode(&[top as u32])
        );
    }

    let bin = out.join("logits.bin");
    let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&bin, &bytes).unwrap();
    std::fs::write(
        out.join("logits.json"),
        format!(
            "{{\n \"model_dir\": \"{}\",\n \"source\": \"ornith-rs src/bin/probes.rs\",\n \
             \"logits\": {{\"file\": \"logits.bin\", \"dtype\": \"float32\", \
             \"shape\": [{}, {}], \"chunk\": {}}},\n \"set\": [\n{}\n ]\n}}\n",
            esc(&dir.to_string_lossy()),
            probes.len(),
            vocab,
            CHUNK,
            meta
        ),
    )
    .unwrap();
    eprintln!(
        "\nwrote {}/logits.json and logits.bin ({}, {})",
        out.display(),
        probes.len(),
        vocab
    );
}
