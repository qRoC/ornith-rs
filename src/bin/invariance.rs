//! What a batched row costs on a whole forward, not on one matmul.
//!
//! `tests/batch_invariance.rs` shows that a quantized linear's rows move when
//! they are batched, and that `ops::invariant` stops them. This runs the same
//! question through the model, two ways, because they are two different
//! questions:
//!
//!   - **pinned.** At each window the caches are snapshotted, the window is
//!     run, the caches are put back, and the same tokens are run one at a
//!     time. Both arms therefore start from *identical* state, so what is
//!     measured is the batching of this window alone. This is the number a
//!     rejected draft is charged against.
//!   - **drift.** A windowed run and a one-at-a-time run from the same prompt,
//!     each keeping its own caches. Every window writes its own rows into the
//!     KV cache and the recurrent state, so the two trajectories separate and
//!     stay separated. This is what a speculative run actually is.
//!
//! A window of `depth + 1` rows is exactly the speculative verify shape, so
//! the `top-1 moved` column is the number that matters: **a row whose argmax
//! moved is a draft rejected because the target changed its mind, not because
//! the draft was wrong.**
//!
//! ```
//! ORNITH_MODEL=<artifact> cargo run --release --bin invariance -- --steps 64
//! ```
//!
//! Both arms run inside one process and the flag moves between them, because
//! this machine's wall clock swings 3x between sittings and two runs an hour
//! apart are not a comparison.

use ornith_rs::array::{Array, Dtype};
use ornith_rs::generate::{model_call, prefill, prefill_step_size};
use ornith_rs::metal::Context;
use ornith_rs::model::cache::LayerCache;
use ornith_rs::model::qwen3_5::Mixer;
use ornith_rs::ops::embedding::embedding;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::ops::invariant;
use ornith_rs::refs::model_dir;
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::weights::WeightStore;

const DEFAULT_PROMPT: &str = "The history of computing is usually told as a \
    story about machines, but it is at least as much a story about the people \
    who decided what the machines were for. In 1843 Ada Lovelace wrote that";

struct Args {
    prompt: String,
    steps: usize,
    windows: Vec<usize>,
}

fn args() -> Args {
    let mut a = Args { prompt: DEFAULT_PROMPT.to_string(), steps: 64, windows: vec![2, 4] };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--prompt" | "-p" => {
                a.prompt = argv[i + 1].clone();
                i += 2;
            }
            "--steps" => {
                a.steps = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--window" | "-w" => {
                a.windows = argv[i + 1].split(',').map(|s| s.parse().unwrap()).collect();
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    a
}

/// Top-1 index and the margin to the runner-up, which is what decides whether
/// a perturbation can move an argmax at all.
fn top2(row: &[f32]) -> (usize, f32) {
    let (mut i0, mut v0, mut v1) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > v0 {
            v1 = v0;
            v0 = v;
            i0 = i;
        } else if v > v1 {
            v1 = v;
        }
    }
    (i0, v0 - v1)
}

/// Decode `steps` tokens greedily, keeping every step's logits.
fn autoregressive(
    ctx: &Context,
    model: &TextModel,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let _ = tokenizer;
    let mut caches = model.make_cache();
    let processed = prefill(ctx, model, &mut caches, prompt, prefill_step_size(), None, |_, _| {});
    let mut fed = vec![prompt[processed..].to_vec()].concat();
    let (mut inputs, mut logits, mut hidden) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..steps {
        inputs.extend_from_slice(&fed);
        let (h, _) = model_call(ctx, model, &mut caches, &fed, None);
        let lg = model.lm_head.matmul(ctx, &h);
        let row = lg.to_vec_f32(ctx);
        let (tok, _) = top2(&row);
        hidden.push(h.to_vec_f32(ctx));
        logits.push(row);
        fed = vec![tok as u32];
    }
    (inputs, logits, hidden)
}

/// Everything a cache holds that a forward changes.
///
/// A KV cache is written in place, so its content is restored by re-running
/// the tokens that own those positions rather than by copying anything back;
/// the offset is all this has to keep. A gated-delta cache replaces its two
/// arrays each step, so holding a reference to the old pair *is* the snapshot
/// -- the same reason `TrunkCapture` is affordable.
enum Snap {
    Kv(usize),
    Gdn(Option<Array>, Option<Array>, usize),
}

fn snapshot(caches: &[LayerCache]) -> Vec<Snap> {
    caches
        .iter()
        .map(|c| match c {
            LayerCache::Kv(k) => Snap::Kv(k.offset),
            LayerCache::Gdn(g) => Snap::Gdn(g.conv.clone(), g.state.clone(), g.offset),
        })
        .collect()
}

fn restore(caches: &mut [LayerCache], snap: &[Snap]) {
    for (c, s) in caches.iter_mut().zip(snap) {
        match (c, s) {
            (LayerCache::Kv(k), Snap::Kv(off)) => k.offset = *off,
            (LayerCache::Gdn(g), Snap::Gdn(conv, state, off)) => {
                g.conv = conv.clone();
                g.state = state.clone();
                g.offset = *off;
            }
            _ => panic!("snapshot does not match the caches it came from"),
        }
    }
}

/// One forward over `tokens`, returning the logit rows and the hidden rows.
///
/// `capture` picks the forward the speculative loop actually runs:
/// `forward_capture` keeps a gated-delta snapshot per token so a partially
/// accepted window can be rewound, and it gets them by splitting the
/// recurrence into one call per token. That split is claimed to be exact, and
/// this is where the claim gets tested against the packed call.
fn forward_rows(
    ctx: &Context,
    model: &TextModel,
    caches: &mut [LayerCache],
    tokens: &[u32],
    capture: bool,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let hsize = model.cfg.hidden_size;
    let ids = Array::from_slice(ctx, tokens, &[tokens.len()], Dtype::U32);
    let h = if capture {
        model.forward_capture(ctx, &ids, caches).0
    } else {
        model.forward_hidden(ctx, &ids, caches).0
    };
    let lg = model.lm_head.matmul(ctx, &h).to_vec_f32(ctx);
    let hv = h.to_vec_f32(ctx);
    let vocab = lg.len() / tokens.len();
    let logits = (0..tokens.len())
        .map(|r| lg[r * vocab..(r + 1) * vocab].to_vec())
        .collect();
    let hidden = (0..tokens.len())
        .map(|r| hv[r * hsize..(r + 1) * hsize].to_vec())
        .collect();
    (logits, hidden)
}

/// The pinned arm: both trajectories are the autoregressive one, and each
/// window is measured against the state it actually started from.
fn pinned(
    ctx: &Context,
    model: &TextModel,
    prompt: &[u32],
    steps: usize,
    w: usize,
    capture: bool,
) -> Report {
    let mut caches = model.make_cache();
    let processed = prefill(ctx, model, &mut caches, prompt, prefill_step_size(), None, |_, _| {});
    let mut next = prompt[processed..].to_vec();
    let mut rep = Report::new(w);
    let mut done = 0usize;

    while done + w <= steps {
        let snap = snapshot(&caches);

        // The autoregressive arm first, because its tokens are the window's
        // input and they are not known until it has run.
        let mut chunk = Vec::with_capacity(w);
        let (mut ar_l, mut ar_h) = (Vec::new(), Vec::new());
        let mut fed = next.clone();
        for _ in 0..w {
            chunk.extend_from_slice(&fed);
            let (l, h) = forward_rows(ctx, model, &mut caches, &fed, false);
            let (tok, _) = top2(&l[0]);
            ar_l.push(l[0].clone());
            ar_h.push(h[0].clone());
            fed = vec![tok as u32];
        }
        let after = fed.clone();

        // The same tokens as one window, from the same state.
        let base = caches[0].offset();
        restore(&mut caches, &snap);
        let (win_l, win_h) = forward_rows(ctx, model, &mut caches, &chunk, capture);
        let before = rep.exact.iter().sum::<usize>();
        for r in 0..w {
            rep.add(r, &win_l[r], &ar_l[r], &win_h[r], &ar_h[r]);
        }
        if rep.exact.iter().sum::<usize>() != before + w {
            rep.bad_offsets.push(base - w);
            if let Some((i, kind, d)) = bisect_window(ctx, model, &mut caches, &snap, &chunk) {
                println!(
                    "    offset {}: first layer to disagree is {i} ({kind}), by {d:.3e}",
                    base - w
                );
            }
        }

        // Put the caches back on the autoregressive trajectory. The window
        // overwrote this window's KV positions, so they are rewritten rather
        // than restored -- and that the rewrite reproduces the first pass
        // exactly is worth asserting, since everything here assumes it.
        restore(&mut caches, &snap);
        let mut fed = next.clone();
        for r in 0..w {
            let (l, _) = forward_rows(ctx, model, &mut caches, &fed, false);
            let d: f32 = l[0]
                .iter()
                .zip(&ar_l[r])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert_eq!(d, 0.0, "the same tokens from the same state gave a different answer");
            fed = vec![chunk.get(r + 1).copied().unwrap_or(after[0])];
        }
        next = after;
        done += w;
    }
    rep
}

/// Which layer first disagrees between a window and the same tokens one at a
/// time, when something still does under the flag.
///
/// The flag covers the quantized linears. Everything else a layer holds --
/// the attention, the gated-delta recurrence, the norms, the router's sort --
/// also sees a different row count, and this says which of them noticed.
fn bisect_window(
    ctx: &Context,
    model: &TextModel,
    caches: &mut [LayerCache],
    snap: &[Snap],
    chunk: &[u32],
) -> Option<(usize, &'static str, f32)> {
    let hsize = model.cfg.hidden_size;
    let w = chunk.len();

    // One token at a time, keeping every layer's output for every row.
    restore(caches, snap);
    let mut alone: Vec<Vec<Vec<f32>>> = vec![Vec::new(); model.layers.len()];
    for &tok in chunk {
        let ids = Array::from_slice(ctx, &[tok], &[1], Dtype::U32);
        let mut h = embedding(ctx, &model.embed_tokens, &ids).reshape(&[1, 1, hsize]);
        for (i, layer) in model.layers.iter().enumerate() {
            h = layer.forward(ctx, &model.cfg, &h, &mut caches[i]);
            alone[i].push(h.to_vec_f32(ctx));
        }
    }

    // The same tokens as one window.
    restore(caches, snap);
    let ids = Array::from_slice(ctx, chunk, &[w], Dtype::U32);
    let mut h = embedding(ctx, &model.embed_tokens, &ids).reshape(&[1, w, hsize]);
    for (i, layer) in model.layers.iter().enumerate() {
        h = layer.forward(ctx, &model.cfg, &h, &mut caches[i]);
        let got = h.to_vec_f32(ctx);
        let mut worst = 0.0f32;
        for r in 0..w {
            for (a, b) in got[r * hsize..(r + 1) * hsize].iter().zip(&alone[i][r]) {
                worst = worst.max((a - b).abs());
            }
        }
        if worst > 0.0 {
            let kind = match &layer.mixer {
                Mixer::Linear(_) => "linear_attention",
                Mixer::Full(_) => "full_attention",
            };
            return Some((i, kind, worst));
        }
    }
    None
}

/// Does a rewound cache hold what the autoregressive path would have left?
///
/// The pinned arm above measures a window's rows. The speculative loop does
/// one more thing with that window: it *keeps* the first `k + 1` rows, rewinds
/// the caches to them, and carries on. A KV cache rewinds by moving an offset,
/// but a gated-delta layer has no offset -- `TrunkCapture` restores the
/// snapshot taken after that token, from a recurrence that was split one call
/// per token. So the check is: rewind to `keep`, feed the next token, and see
/// whether it answers what the one-at-a-time path answered.
fn rewound(
    ctx: &Context,
    model: &TextModel,
    prompt: &[u32],
    steps: usize,
    w: usize,
) -> Report {
    let hsize = model.cfg.hidden_size;
    let mut caches = model.make_cache();
    let processed = prefill(ctx, model, &mut caches, prompt, prefill_step_size(), None, |_, _| {});
    let mut next = prompt[processed..].to_vec();
    let mut rep = Report::new(w);
    let mut done = 0usize;
    let mut cycle = 0usize;

    while done + w + 1 <= steps {
        let snap = snapshot(&caches);
        // Truth: w + 1 tokens one at a time.
        let mut chunk = Vec::with_capacity(w + 1);
        let (mut ar_l, mut ar_h) = (Vec::new(), Vec::new());
        let mut fed = next.clone();
        for _ in 0..=w {
            chunk.extend_from_slice(&fed);
            let (l, h) = forward_rows(ctx, model, &mut caches, &fed, false);
            let (tok, _) = top2(&l[0]);
            ar_l.push(l[0].clone());
            ar_h.push(h[0].clone());
            fed = vec![tok as u32];
        }
        let after = fed.clone();

        // A window of w rows, of which `keep` are committed -- every
        // acceptance count the loop can produce, in turn.
        let keep = 1 + cycle % w;
        restore(&mut caches, &snap);
        let ids = Array::from_slice(ctx, &chunk[..w], &[w], Dtype::U32);
        let (_, _, cap) = model.forward_capture(ctx, &ids, &mut caches);
        cap.rewind(&ctx, &mut caches, keep);

        // The next token, from a rewound cache against an untouched one.
        let (l, h) = forward_rows(ctx, model, &mut caches, &[chunk[keep]], false);
        rep.add(keep - 1, &l[0], &ar_l[keep], &h[0], &ar_h[keep]);

        restore(&mut caches, &snap);
        let mut fed = next.clone();
        for r in 0..=w {
            forward_rows(ctx, model, &mut caches, &fed, false);
            fed = vec![chunk.get(r + 1).copied().unwrap_or(after[0])];
        }
        let _ = hsize;
        next = after;
        done += w + 1;
        cycle += 1;
    }
    rep
}

/// Replay the same inputs in windows of `w`, comparing each row against the
/// autoregressive step that computed it alone.
fn windowed(
    ctx: &Context,
    model: &TextModel,
    prompt: &[u32],
    inputs: &[u32],
    ar_logits: &[Vec<f32>],
    ar_hidden: &[Vec<f32>],
    w: usize,
) -> Report {
    let hsize = model.cfg.hidden_size;
    let mut caches = model.make_cache();
    let processed = prefill(ctx, model, &mut caches, prompt, prefill_step_size(), None, |_, _| {});
    assert_eq!(processed, prompt.len() - 1);

    let mut rep = Report::new(w);
    let mut at = 0usize;
    while at + w <= inputs.len() {
        let chunk = &inputs[at..at + w];
        let ids = Array::from_slice(ctx, chunk, &[chunk.len()], Dtype::U32);
        let (h, _) = model.forward_hidden(ctx, &ids, &mut caches);
        let lg = model.lm_head.matmul(ctx, &h).to_vec_f32(ctx);
        let hv = h.to_vec_f32(ctx);
        let vocab = lg.len() / w;
        for r in 0..w {
            let got = &lg[r * vocab..(r + 1) * vocab];
            let want = &ar_logits[at + r];
            rep.add(
                r,
                got,
                want,
                &hv[r * hsize..(r + 1) * hsize],
                &ar_hidden[at + r],
            );
        }
        at += w;
    }
    rep
}

struct Report {
    w: usize,
    rows: usize,
    logit_max: Vec<f32>,
    hidden_max: Vec<f32>,
    moved: Vec<usize>,
    exact: Vec<usize>,
    counted: Vec<usize>,
    /// The top-2 margin of the rows whose argmax moved, and of all rows.
    margins_moved: Vec<f32>,
    margins_all: Vec<f32>,
    /// Cache offsets whose window had a row that was not bit-identical.
    pub bad_offsets: Vec<usize>,
}

impl Report {
    fn new(w: usize) -> Report {
        Report {
            w,
            rows: 0,
            logit_max: vec![0.0; w],
            hidden_max: vec![0.0; w],
            moved: vec![0; w],
            exact: vec![0; w],
            counted: vec![0; w],
            margins_moved: Vec::new(),
            margins_all: Vec::new(),
            bad_offsets: Vec::new(),
        }
    }

    fn add(&mut self, r: usize, got: &[f32], want: &[f32], got_h: &[f32], want_h: &[f32]) {
        self.rows += 1;
        self.counted[r] += 1;
        let mut m = 0.0f32;
        for (a, b) in got.iter().zip(want) {
            m = m.max((a - b).abs());
        }
        self.logit_max[r] = self.logit_max[r].max(m);
        if m == 0.0 {
            self.exact[r] += 1;
        }
        let mut mh = 0.0f32;
        for (a, b) in got_h.iter().zip(want_h) {
            mh = mh.max((a - b).abs());
        }
        self.hidden_max[r] = self.hidden_max[r].max(mh);
        let (gi, _) = top2(got);
        let (wi, margin) = top2(want);
        self.margins_all.push(margin);
        if gi != wi {
            self.moved[r] += 1;
            self.margins_moved.push(margin);
        }
    }

    fn print(&self, arm: &str, label: &str) {
        let total_moved: usize = self.moved.iter().sum();
        let total_exact: usize = self.exact.iter().sum();
        println!(
            "\n  {arm}, window {} rows, invariant={label}: {}/{} rows bit-identical \
             to the same row alone, top-1 moved on {}",
            self.w, total_exact, self.rows, total_moved
        );
        println!(
            "    {:>3}  {:>10} {:>10} {:>10} {:>12}",
            "row", "logit max", "hidden max", "exact", "top-1 moved"
        );
        for r in 0..self.w {
            println!(
                "    {r:>3}  {:>10.3e} {:>10.3e} {:>6}/{:<4} {:>8}/{:<4}",
                self.logit_max[r],
                self.hidden_max[r],
                self.exact[r],
                self.counted[r],
                self.moved[r],
                self.counted[r]
            );
        }
        if !self.bad_offsets.is_empty() {
            let mut o = self.bad_offsets.clone();
            o.dedup();
            println!("    windows that moved, by cache offset: {o:?}");
        }
        if !self.margins_moved.is_empty() {
            let mut ms = self.margins_moved.clone();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "    the moved rows' top-2 margins: {:.4} to {:.4} (median of all rows {:.4})",
                ms[0],
                ms[ms.len() - 1],
                median(&self.margins_all)
            );
        }
    }
}

fn median(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn main() {
    let a = args();
    let dir = model_dir();
    assert!(dir.exists(), "no model at {}", dir.display());
    let ctx = Context::new().expect("metal");
    let t0 = std::time::Instant::now();
    let tokenizer = Tokenizer::load(&dir).unwrap();
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

    let prompt = tokenizer.encode(&a.prompt);
    println!(
        "\nartifact {}\nprompt {} tokens, {} decode steps\n",
        dir.file_name().unwrap().to_string_lossy(),
        prompt.len(),
        a.steps
    );

    for on in [false, true] {
        // `all` rather than the window mode, so that a window past
        // `EXACT_WINDOW` shows what is left rather than what the mode declined
        // to cover: at w = 5 the quantized linears stay invariant and the first
        // layer to disagree is the first full-attention one, which is mlx's
        // fused-SDPA limit and not this flag's.
        invariant::set_rows(if on { usize::MAX - 1 } else { 0 });
        let label = if on { "on" } else { "off" };
        let t = std::time::Instant::now();
        let (inputs, ar_logits, ar_hidden) =
            autoregressive(&ctx, &model, &tokenizer, &prompt, a.steps);
        eprintln!("  autoregressive arm ({label}): {:.1}s", t.elapsed().as_secs_f64());
        for &w in &a.windows {
            pinned(&ctx, &model, &prompt, a.steps, w, false).print("pinned", label);
            pinned(&ctx, &model, &prompt, a.steps, w, true).print("pinned-capture", label);
            rewound(&ctx, &model, &prompt, a.steps, w).print("rewound", label);
            windowed(&ctx, &model, &prompt, &inputs, &ar_logits, &ar_hidden, w)
                .print("drift", label);
        }
    }
    invariant::set(false);
}
