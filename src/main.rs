//! The console chat.
//!
//! ```text
//!   ornith                          chat, sampling as mtplx_runtime.json asks
//!   ornith --greedy                 chat with temperature 0
//!   ornith --mtp                    speculative decoding, MTP depth 3
//!   ornith --prompt "..."           one shot, for comparing against mlx_lm
//!   ornith --raw --prompt "..."     one shot with no chat template
//!   ornith --image a.bmp -p "..."   one shot with an image
//! ```
//!
//! The loop is mlx-lm's: `apply_chat_template`, `generate_step`, and the
//! `BPEStreamingDetokenizer` in between, so the text that lands on the
//! terminal is the text `mlx_lm.generate` would print.
//!
//! `--image` takes already-decoded pixels -- a binary PPM, an uncompressed
//! BMP, or raw RGB with `--image-size WxH`. There is no image *decoder* here;
//! macOS ships one:
//!
//! ```text
//!     sips -s format bmp photo.jpg --out photo.bmp
//! ```
//!
//! Any size goes: the processor's `smart_resize` and Pillow's bicubic
//! resampler are both ported, so ornith resizes the picture exactly as MTPLX
//! would have.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use ornith_rs::chat::{Content, Message, Part, Role, apply_chat_template};
use ornith_rs::generate::{DEFAULT_PREFILL_STEP_SIZE, Greedy, Sampler, generate_step};
use ornith_rs::metal::{Context, Kind, Why};
use ornith_rs::model::cache::LayerCache;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::mtp::{MtpHead, sidecar_path};
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::mtp_generate::{HistoryPolicy, MtpStats, generate_step_mtp};
use ornith_rs::refs::model_dir;
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::tokenizer::split::Dialect;
use ornith_rs::vision::preprocess::{Image, PreprocessorConfig, preprocess};
use ornith_rs::vision::splice::{VisionSplice, expand_image_pads};
use ornith_rs::vision::tower::VisionTower;
use ornith_rs::vision::{VisionConfig, VisionSpec};
use ornith_rs::weights::WeightStore;

/// Everything a vision request needs, loaded only when `--image` is given --
/// the tower is another 0.9 GB on top of the trunk's 25.5.
struct Vision {
    tower: VisionTower,
    spec: VisionSpec,
    pre: PreprocessorConfig,
    /// Tower output per image, in the order the images were given. MTPLX
    /// caches the same thing by content digest (`_VISION_EMBED_CACHE`), for
    /// the same reason: an agent resends the identical image every turn.
    rows: Vec<ornith_rs::array::Array>,
    pad_counts: Vec<usize>,
}

impl Vision {
    fn load(
        ctx: &Context,
        ws: &mut WeightStore,
        dir: &std::path::Path,
        paths: &[PathBuf],
        raw_size: Option<(usize, usize)>,
    ) -> Vision {
        let (cfg, spec) = VisionConfig::load(dir).expect("vision_config");
        let pre = PreprocessorConfig::load(dir).expect("preprocessor_config.json");
        let tower = VisionTower::load(ctx, ws, cfg);
        let mut v = Vision {
            tower,
            spec,
            pre,
            rows: Vec::new(),
            pad_counts: Vec::new(),
        };
        for path in paths {
            let image = Image::read(path, raw_size).unwrap_or_else(|e| die(&e));
            let patches = preprocess(&image, &v.pre).unwrap_or_else(|e| die(&e));
            let pixels = ornith_rs::array::Array::from_f32(
                ctx,
                &patches.values,
                &[patches.rows, patches.row_len],
            );
            let t0 = std::time::Instant::now();
            let rows = v.tower.forward(ctx, &pixels, &[patches.grid]);
            ctx.sync();
            let (_, gh, gw) = patches.grid;
            eprintln!(
                "{}: {}x{} -> {}x{} patches -> {} tokens in {:.2}s",
                path.display(),
                image.width,
                image.height,
                gw,
                gh,
                rows.shape[0],
                t0.elapsed().as_secs_f64()
            );
            v.pad_counts.push(v.tower.cfg.pad_token_count(patches.grid));
            v.rows.push(rows);
        }
        v
    }

    /// One `VisionSplice` over every image, in prompt order. The tower's rows
    /// are already contiguous per image, so a single image needs no copy.
    fn splice(&self, ctx: &Context) -> VisionSplice {
        let parts: Vec<&ornith_rs::array::Array> = self.rows.iter().collect();
        let embeddings = ornith_rs::ops::layout::concat_leading(ctx, &parts);
        VisionSplice {
            image_pad_token_id: self.spec.image_token_id,
            embeddings,
            pad_counts: self.pad_counts.clone(),
            cursor: 0,
        }
    }
}

/// `mtplx_runtime.json`'s `recommended_settings`.
const TEMP: f64 = 0.6;
const TOP_P: f64 = 0.95;
const TOP_K: usize = 20;
const ENABLE_THINKING: bool = false;

struct Args {
    prompt: Option<String>,
    raw: bool,
    greedy: bool,
    max_tokens: usize,
    system: Option<String>,
    seed: u64,
    dialect: Dialect,
    /// `mtplx_runtime.json` recommends depth 3; 0 is plain autoregression.
    mtp_depth: usize,
    mtp_policy: HistoryPolicy,
    images: Vec<PathBuf>,
    /// Dimensions for a headerless raw RGB file.
    image_size: Option<(usize, usize)>,
}

fn parse_args() -> Args {
    let mut a = Args {
        prompt: None,
        raw: false,
        greedy: false,
        max_tokens: 512,
        system: None,
        seed: 0,
        dialect: Dialect::Transformers,
        mtp_depth: 0,
        mtp_policy: HistoryPolicy::Committed,
        images: Vec::new(),
        image_size: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--prompt" | "-p" => a.prompt = Some(it.next().expect("--prompt needs a value")),
            "--system" | "-s" => a.system = Some(it.next().expect("--system needs a value")),
            "--max-tokens" | "-m" => {
                a.max_tokens = it.next().expect("--max-tokens needs a value").parse().unwrap()
            }
            "--seed" => a.seed = it.next().expect("--seed needs a value").parse().unwrap(),
            "--mtp" => a.mtp_depth = 3,
            "--mtp-depth" => {
                a.mtp_depth = it.next().expect("--mtp-depth needs a value").parse().unwrap()
            }
            "--mtp-cycle" => a.mtp_policy = HistoryPolicy::Cycle,
            "--image" | "-i" => {
                a.images.push(PathBuf::from(it.next().expect("--image needs a path")))
            }
            "--image-size" => {
                let v = it.next().expect("--image-size needs WxH");
                let (w, h) = v.split_once('x').expect("--image-size wants WxH");
                a.image_size = Some((w.parse().unwrap(), h.parse().unwrap()));
            }
            "--raw" => a.raw = true,
            "--greedy" => a.greedy = true,
            "--tokenizer-json" => a.dialect = Dialect::TokenizerJson,
            "--help" | "-h" => {
                eprintln!(
                    "ornith [--prompt TEXT] [--system TEXT] [--max-tokens N] [--seed N]\n\
                     \x20      [--raw] [--greedy] [--tokenizer-json]\n\
                     \x20      [--image PATH]... [--image-size WxH]\n\
                     \x20      [--mtp] [--mtp-depth N] [--mtp-cycle]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other:?}"),
        }
    }
    a
}

/// `ORNITH_PROF=1`: where a decode step's wall time goes.
///
/// `gpu` is summed `GPUEndTime - GPUStartTime` over the step's command
/// buffers, so `wall - gpu` is what the GPU spent idle -- waiting on the CPU
/// to encode, or on the round trip that hands the sampled token back.
fn report_prof(ctx: &Context, steps: usize, gen_time: f64) {
    if steps == 0 {
        return;
    }
    let (commits, dispatches, gpu_s, wait_s) = ctx.prof_totals();
    let n = steps as f64;
    let ms = |s: f64| s * 1e3 / n;
    eprintln!(
        "[prof: {:.2} ms/token wall, {:.2} gpu ({:.0}%), {:.2} blocked in wait; \
         {:.0} dispatches, {:.0} barriers, {:.1} commits, {:.2} syncs per token]",
        ms(gen_time),
        ms(gpu_s),
        100.0 * gpu_s / gen_time.max(1e-9),
        ms(wait_s),
        dispatches as f64 / n,
        ctx.prof_barriers() as f64 / n,
        commits as f64 / n,
        ctx.prof_syncs() as f64 / n,
    );
    report_gaps(ctx, n, gen_time);
    let (live, peak) = ctx.pool_stats();
    eprintln!(
        "[prof: pooled intermediates {:.1} MB live, {:.1} MB peak]",
        live as f64 / 1e6,
        peak as f64 / 1e6,
    );
    report_barriers(ctx, n);
    let (by, mixed) = ctx.prof_kernel_times();
    if mixed > 0.0 {
        // Every command buffer held a whole batch of dispatches, so its
        // timestamps cannot be split between them.
        let shown: Vec<String> = by
            .iter()
            .take(12)
            .map(|(k, c, _)| format!("{} {:.0}", k, *c as f64 / n))
            .collect();
        eprintln!(
            "[prof: {} kernels, most dispatched per token: {}]",
            by.len(),
            shown.join(", ")
        );
        eprintln!("[prof: re-run with ORNITH_COMMIT_EVERY=1 to attribute GPU time per kernel]");
        return;
    }
    eprintln!("[prof: GPU time per kernel, per token]");
    let total: f64 = by.iter().map(|(_, _, g)| *g).sum();
    for (k, c, g) in by.iter().take(64) {
        eprintln!(
            "  {:>7.3} ms  {:>5.1}%  x{:<5.0}  {}",
            g * 1e3 / n,
            100.0 * g / total.max(1e-12),
            *c as f64 / n,
            k
        );
    }
}

/// Where the GPU had nothing to do.
///
/// `gpu` above is summed `GPUEndTime - GPUStartTime`, so it measures only
/// where the GPU works. This is the other half: the queue is serial, so the
/// stretch between one command buffer ending and the next starting is idle,
/// and it is charged to the first kernel of the buffer that ended it -- what
/// the GPU was waiting to be handed. A gap that follows a host round trip is
/// counted separately, since there the CPU was reading a result back and
/// deciding what to encode next rather than merely encoding slowly.
fn report_gaps(ctx: &Context, n: f64, gen_time: f64) {
    let (gap, after_sync) = ctx.prof_gaps();
    if gap <= 0.0 {
        return;
    }
    eprintln!(
        "[prof: {:.2} ms/token idle between command buffers ({:.0}% of wall), \
         {:.2} of it after a round trip]",
        gap * 1e3 / n,
        100.0 * gap / gen_time.max(1e-9),
        after_sync * 1e3 / n,
    );
    let (wake, wakes) = ctx.prof_wake();
    if wakes > 0 {
        eprintln!(
            "  {:>7.3} ms  x{:<6.1} of it waiting for a core, not for the GPU \
             ({:.0} us a round trip)",
            wake * 1e3 / n,
            wakes as f64 / n,
            wake * 1e6 / wakes as f64,
        );
    }
    for (k, s, times) in ctx.prof_gap_by_kernel().iter().take(8) {
        eprintln!(
            "  {:>7.3} ms  x{:<6.1} waiting to be handed {}",
            s * 1e3 / n,
            *times as f64 / n,
            k
        );
    }
}

/// Which pairs of dispatches the barriers are between, and how many of them
/// are there because two dispatches share bytes rather than merely an address.
///
/// `real` is a dependency the data has. `disjoint` is one allocation bound at
/// two ranges that do not overlap; `recycled` is two allocations that share an
/// address because the pool handed the freed one straight back out. Neither of
/// the last two moves any data, and both cost a barrier all the same, because
/// the tracking keys on the buffer -- which is what mlx does too.
/// The pool does not recycle inside an epoch, so `recycled` should be zero;
/// `ORNITH_RECYCLE=1` puts mlx's behaviour back and it becomes 30 a token.
fn report_barriers(ctx: &Context, n: f64) {
    let causes = ctx.prof_barrier_causes();
    if causes.is_empty() {
        return;
    }
    let total = ctx.prof_barriers();
    let spurious = ctx.prof_barriers_spurious();
    eprintln!(
        "[prof: {:.0} barriers per token, {:.0} of them ({:.0}%) with no data \
         behind them]",
        total as f64 / n,
        spurious as f64 / n,
        100.0 * spurious as f64 / (total as f64).max(1.0),
    );
    let line = |(kernel, culprit, kind, why, _sp, count): &(String, String, Kind, Why, bool, usize)| {
        eprintln!(
            "  x{:<6.1} {:<4} {:<8} {} after {}",
            *count as f64 / n,
            kind.as_str(),
            why.as_str(),
            kernel,
            culprit,
        );
    };
    eprintln!("[prof: the pairs that barrier most]");
    for c in causes.iter().take(15) {
        line(c);
    }
    let idle: Vec<_> = causes.iter().filter(|c| c.4).collect();
    if !idle.is_empty() {
        eprintln!("[prof: and the ones with nothing flowing through them]");
        for c in &idle {
            line(c);
        }
    }
    if let Ok(path) = std::env::var("ORNITH_TRACE") {
        let trace = ctx.prof_trace();
        if !trace.is_empty() && std::fs::write(&path, trace.join("\n") + "\n").is_ok() {
            eprintln!("[prof: {} lines of one step's dispatches in {path}]", trace.len());
        }
    }
    let sizes = ctx.prof_epoch_sizes();
    let epochs: usize = sizes.iter().map(|(_, c)| *c).sum();
    let disp: usize = sizes.iter().map(|(k, c)| k * c).sum();
    let alone: usize = sizes.iter().filter(|(k, _)| *k <= 1).map(|(_, c)| *c).sum();
    eprintln!(
        "[prof: {:.0} epochs per token, {:.2} dispatches each, {:.0}% of them \
         holding one dispatch or none -- an epoch is what can overlap]",
        epochs as f64 / n,
        disp as f64 / (epochs as f64).max(1.0),
        100.0 * alone as f64 / (epochs as f64).max(1.0),
    );
    let hist: Vec<String> = sizes
        .iter()
        .take(10)
        .map(|(k, c)| format!("{k}: {:.0}", *c as f64 / n))
        .collect();
    eprintln!("  dispatches per epoch, epochs per token -- {}", hist.join(", "));
}

/// A user error -- an unreadable image, or one the processor would have to
/// resample -- reported as a message rather than a panic with a backtrace.
fn die(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// The list-shaped `content` the template wants for a vision turn:
/// `render_content` emits `<|vision_start|><|image_pad|><|vision_end|>` for
/// each `Part::Image`, then the text.
fn user_message(text: &str, n_images: usize) -> Message {
    if n_images == 0 {
        return Message::text(Role::User, text);
    }
    let mut parts: Vec<Part> = (0..n_images).map(|_| Part::Image).collect();
    parts.push(Part::Text(text.to_string()));
    Message { role: Role::User, content: Content::Parts(parts), reasoning_content: None }
}

fn main() {
    let args = parse_args();
    let dir = model_dir();
    assert!(dir.exists(), "no model at {}", dir.display());

    let ctx = Context::new().expect("metal");
    let t0 = std::time::Instant::now();
    let tokenizer = Tokenizer::load_as(&dir, args.dialect).unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let extra: Vec<std::path::PathBuf> =
        if args.mtp_depth > 0 { vec![sidecar_path(&dir)] } else { Vec::new() };
    let mut ws = WeightStore::open(&dir, &extra).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg.clone());
    let head = (args.mtp_depth > 0).then(|| MtpHead::load(&ctx, &mut ws, &cfg));
    let vision = (!args.images.is_empty())
        .then(|| Vision::load(&ctx, &mut ws, &dir, &args.images, args.image_size));
    ctx.sync();
    eprintln!(
        "{}: {:.1} GB in {:.1}s",
        ctx.device_name(),
        ws.loaded_bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    drop(ws);

    let mut sampler: Box<dyn Sampler> = if args.greedy {
        Box::new(Greedy)
    } else {
        Box::new(ornith_rs::sample::TopSampler::new(TEMP, TOP_P, TOP_K, args.seed))
    };

    if args.mtp_depth > 0 && !args.greedy {
        eprintln!(
            "note: acceptance is by token equality, which only preserves the \
             sampled distribution for greedy decoding"
        );
    }

    let mut session = Session {
        ctx: &ctx,
        model: &model,
        head: head.as_ref(),
        depth: args.mtp_depth,
        policy: args.mtp_policy,
        tokenizer: &tokenizer,
        caches: model.make_cache(),
        cached: Vec::new(),
        max_tokens: args.max_tokens,
    };

    if let Some(p) = &args.prompt {
        let ids = if args.raw {
            tokenizer.encode(p)
        } else {
            let mut msgs = Vec::new();
            if let Some(s) = &args.system {
                msgs.push(Message::text(Role::System, s));
            }
            msgs.push(user_message(p, args.images.len()));
            tokenizer.encode(&apply_chat_template(&msgs, true, ENABLE_THINKING).unwrap())
        };
        // `_expand_image_pads`: the template renders one `<|image_pad|>` per
        // image, and prefill needs one per tower row.
        let (ids, mut splice) = match &vision {
            Some(v) => (
                expand_image_pads(&ids, v.spec.image_token_id, &v.pad_counts).unwrap(),
                Some(v.splice(&ctx)),
            ),
            None => (ids, None),
        };
        session.run(&ids, sampler.as_mut(), splice.as_mut());
        println!();
        return;
    }

    let mut messages: Vec<Message> = Vec::new();
    if let Some(s) = &args.system {
        messages.push(Message::text(Role::System, s));
    }
    eprintln!("type a message, or /exit to quit, /reset to start over");
    let stdin = std::io::stdin();
    loop {
        print!("\n> ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break;
        }
        let line = line.trim_end_matches('\n');
        match line.trim() {
            "/exit" | "/quit" => break,
            "/reset" => {
                messages.retain(|m| m.role == Role::System);
                session.caches = session.model.make_cache();
                session.cached.clear();
                eprintln!("(cleared)");
                continue;
            }
            "" => continue,
            _ => {}
        }
        // The images are attached to the first user turn, which is where
        // `--image` puts them; later turns are text.
        let first_turn = !messages.iter().any(|m| m.role == Role::User);
        let n_images = if first_turn { args.images.len() } else { 0 };
        messages.push(user_message(line, n_images));
        let prompt = apply_chat_template(&messages, true, ENABLE_THINKING).unwrap();
        let ids = tokenizer.encode(&prompt);
        let (ids, mut splice) = match &vision {
            Some(v) => (
                expand_image_pads(&ids, v.spec.image_token_id, &v.pad_counts).unwrap(),
                Some(v.splice(&ctx)),
            ),
            None => (ids, None),
        };
        let reply = session.run(&ids, sampler.as_mut(), splice.as_mut());
        println!();
        messages.push(Message::text(Role::Assistant, &reply));
    }
}

struct Session<'a> {
    ctx: &'a Context,
    model: &'a TextModel,
    head: Option<&'a MtpHead>,
    depth: usize,
    policy: HistoryPolicy,
    tokenizer: &'a Tokenizer,
    caches: Vec<LayerCache>,
    /// Every token the caches have been fed, so a follow-up turn that extends
    /// this prefix can reuse them. A gated-delta state cannot be trimmed, so
    /// anything that is not an extension starts over.
    cached: Vec<u32>,
    max_tokens: usize,
}

impl Session<'_> {
    fn run(
        &mut self,
        ids: &[u32],
        sampler: &mut dyn Sampler,
        splice: Option<&mut VisionSplice>,
    ) -> String {
        // A prompt carrying images always starts over. Every image expands to
        // the same `<|image_pad|>` id, so a token-id prefix match cannot tell
        // two different pictures apart -- MTPLX guards the same hazard from
        // the other side, with `vision_image_spans` and a content digest.
        let has_images = splice.is_some();
        let reuse =
            !has_images && self.cached.len() < ids.len() && ids.starts_with(&self.cached);
        if !reuse {
            self.caches = self.model.make_cache();
            self.cached.clear();
        }
        let fresh = &ids[self.cached.len()..];
        let mut splice = splice;

        let mut detok = self.tokenizer.detokenizer();
        let eos = &self.tokenizer.eos_ids;
        let mut out = std::io::stdout();
        let t0 = std::time::Instant::now();
        let mut n_generated = 0usize;
        let mut prompt_time = 0.0;

        let mut fed: Vec<u32> = Vec::new();
        let mut stats: Option<MtpStats> = None;
        let ctx = self.ctx;
        {
            let progress = |done: usize, total: usize| {
                if done == total {
                    prompt_time = t0.elapsed().as_secs_f64();
                    // Prefill has its own shapes and dispatch counts; the
                    // decode steps are what the counters are meant to describe.
                    ctx.prof_reset();
                } else if total > DEFAULT_PREFILL_STEP_SIZE {
                    eprint!("\rprompt {done}/{total}");
                }
            };
            let mut emit = |token: u32| {
                fed.push(token);
                if eos.contains(&token) {
                    return false;
                }
                detok.add_token(token);
                n_generated += 1;
                let seg = detok.last_segment();
                if !seg.is_empty() {
                    print!("{seg}");
                    let _ = out.flush();
                }
                n_generated < self.max_tokens
            };
            match self.head {
                Some(head) => {
                    stats = Some(generate_step_mtp(
                        self.ctx,
                        self.model,
                        head,
                        &mut self.caches,
                        fresh,
                        sampler,
                        self.depth,
                        self.max_tokens,
                        DEFAULT_PREFILL_STEP_SIZE,
                        self.policy,
                        splice.as_deref_mut(),
                        progress,
                        &mut emit,
                    ));
                }
                None => generate_step(
                    self.ctx,
                    self.model,
                    &mut self.caches,
                    fresh,
                    sampler,
                    self.max_tokens,
                    DEFAULT_PREFILL_STEP_SIZE,
                    splice.as_deref_mut(),
                    progress,
                    &mut emit,
                ),
            }
        }
        detok.finalize();
        let seg = detok.last_segment();
        if !seg.is_empty() {
            print!("{seg}");
            let _ = out.flush();
        }

        self.cached.extend_from_slice(fresh);
        self.cached.extend_from_slice(&fed);

        let gen_time = t0.elapsed().as_secs_f64() - prompt_time;
        if let Some(st) = &stats {
            let by: Vec<String> = st
                .acceptance_by_depth()
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect();
            eprintln!(
                "\n[mtp: {} cycles, {}/{} drafts accepted, by depth {}]",
                st.cycles,
                st.accepted,
                st.drafted,
                by.join(" / ")
            );
        }
        eprintln!(
            "\n[{} prompt tokens {:.1} tok/s, {} generated {:.1} tok/s]",
            fresh.len(),
            fresh.len() as f64 / prompt_time.max(1e-9),
            n_generated,
            n_generated as f64 / gen_time.max(1e-9),
        );
        if ctx.prof_on() {
            report_prof(ctx, n_generated, gen_time);
        }
        detok.text.clone()
    }
}
