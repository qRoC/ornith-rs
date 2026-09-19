//! What a decode step's dispatches are made of, one layer at a time.
//!
//! `HANDOFF.md`'s per-kernel table sums GPU time by kernel *name* over a whole
//! run. That says which kernels exist and what they cost in isolation; it
//! cannot say what one layer issues, in what order, over how many bytes, or
//! which of its dispatches are paying for memory and which are paying for
//! being a dispatch at all. This bin is that inventory.
//!
//! Three things are measured, in one process, on one set of weights, from one
//! restored cache state:
//!
//! - **the step**, at the real commit size, which is the level everything
//!   else is a fraction of;
//! - **the same step with every dispatch clamped to one threadgroup**
//!   ([`Context::set_grid1`]) -- the same 1669 dispatches, the same barriers,
//!   the same command buffers, and none of the work. That is the issue floor
//!   as a *measured term* rather than a dispatch count times a coefficient
//!   from somewhere else. Its control is width-1-shaped: a kernel already
//!   dispatched over one threadgroup must come out unchanged, and the table
//!   prints those separately so it can be read;
//! - **per dispatch**, at `commit_every(1)` so a command buffer's timestamps
//!   belong to a single dispatch, with and without the clamp -- which gives
//!   each dispatch its launch cost and its work separately, against the bytes
//!   its bindings say it moved.
//!
//! The per-dispatch times are isolated costs: one dispatch per command buffer
//! is one dispatch at a time, where the real step runs a concurrent encoder
//! and hides small kernels under big ones. Read the table for what a dispatch
//! *is* -- its bytes, its grid, its floor against its work -- and never as a
//! decomposition of the step's milliseconds. The step's own two numbers, real
//! and clamped, are the decomposition.
//!
//! Run:
//!
//! ```text
//!   ORNITH_MODEL=~/.mtplx/models/ornith-1.5-35b-a3b-q6k-full \
//!     ORNITH_PROF=1 cargo run --release --bin step_inventory
//! ```

use ornith_rs::generate::{model_call, prefill, prefill_step_size};
use ornith_rs::metal::Context;
use ornith_rs::model::cache::LayerCache;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::mtp::sidecar_path;
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::array::Array;
use ornith_rs::refs::model_dir;
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::weights::WeightStore;

const PROMPT: &str = "Explain in one paragraph why a mixture-of-experts model \
                      reads fewer bytes per token than a dense one of the same \
                      parameter count, and what that costs.";

const WARM_STEPS: usize = 8;
/// Whole-step repetitions. A step is ~8 ms, so this is ~0.2 s a cell.
const REPS: usize = 24;
/// Draws of each whole-step cell, interleaved; the median is the number.
const DRAWS: usize = 3;

/// One dispatch, as the trace and the two timing runs describe it.
#[derive(Clone)]
struct Row {
    section: String,
    kernel: String,
    grid: (usize, usize, usize),
    tg: (usize, usize, usize),
    /// Bytes the bindings say it reads and writes. An op that marks no output
    /// binds everything as a write, so a `w` figure can be a read.
    read: usize,
    write: usize,
    /// Of `read`, what a gather actually touches: `k` experts of `E`.
    read_live: usize,
    t_real: f64,
    t_floor: f64,
}

impl Row {
    fn work(&self) -> f64 {
        (self.t_real - self.t_floor).max(0.0)
    }
    /// Achieved over the work, not over the whole dispatch: the launch is not
    /// reading anything. Times are microseconds, so bytes per us is MB/s.
    fn gbs(&self) -> f64 {
        if self.work() <= 0.0 { 0.0 } else { self.read_live as f64 / self.work() / 1e3 }
    }
    fn one_group(&self) -> bool {
        self.grid.0 <= self.tg.0 && self.grid.1 <= self.tg.1 && self.grid.2 <= self.tg.2
    }
}

/// `kernel grid:AxBxC tg:AxBxC r:id:serial:start:end ...` -> the parts of it
/// this bin needs. Lines starting `--` or `==` are not dispatches.
fn parse(line: &str, section: &str, k: usize, e: usize) -> Option<Row> {
    if line.starts_with("--") || line.starts_with("==") {
        return None;
    }
    let mut it = line.split_whitespace();
    let kernel = it.next()?.to_string();
    let gather = kernel.contains("gather");
    let (mut read, mut write) = (0usize, 0usize);
    let (mut grid, mut tg) = ((0, 0, 0), (0, 0, 0));
    let mut reads: Vec<usize> = Vec::new();
    for f in it {
        if let Some(v) = f.strip_prefix("grid:") {
            grid = triple(v);
        } else if let Some(v) = f.strip_prefix("tg:") {
            tg = triple(v);
        } else {
            let mut p = f.split(':');
            let dir = p.next()?;
            let _id = p.next()?;
            let _serial = p.next()?;
            let start: usize = p.next()?.parse().ok()?;
            let end: usize = p.next()?.parse().ok()?;
            let n = end.saturating_sub(start);
            match dir {
                "r" => {
                    read += n;
                    reads.push(n);
                }
                _ => write += n,
            }
        }
    }
    // A gather binds the whole stack of `E` experts and reads `k` of them.
    // Only the per-expert bindings scale: the activation and the index vector
    // are read whole, and both are small.
    let read_live = if gather {
        reads.iter().map(|n| if *n >= 1 << 20 { n * k / e } else { *n }).sum()
    } else {
        read
    };
    Some(Row {
        section: section.to_string(),
        kernel,
        grid,
        tg,
        read,
        write,
        read_live,
        t_real: 0.0,
        t_floor: 0.0,
    })
}

fn triple(s: &str) -> (usize, usize, usize) {
    let mut p = s.split('x').map(|v| v.parse().unwrap_or(0));
    (p.next().unwrap_or(0), p.next().unwrap_or(0), p.next().unwrap_or(0))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

type Snap = Vec<(Option<Array>, Option<Array>, usize)>;

fn snapshot(cs: &[LayerCache]) -> Snap {
    cs.iter()
        .map(|c| match c {
            LayerCache::Kv(k) => (k.keys.clone(), k.values.clone(), k.offset),
            LayerCache::Gdn(g) => (g.conv.clone(), g.state.clone(), g.offset),
        })
        .collect()
}

fn restore(cs: &mut [LayerCache], s: &Snap) {
    for (c, (a, b, o)) in cs.iter_mut().zip(s) {
        match c {
            LayerCache::Kv(k) => {
                k.keys = a.clone();
                k.values = b.clone();
                k.offset = *o;
            }
            LayerCache::Gdn(g) => {
                g.conv = a.clone();
                g.state = b.clone();
                g.offset = *o;
            }
        }
    }
}

fn main() {
    let dir = model_dir();
    let ctx = Context::new().expect("metal");
    assert!(ctx.prof_on(), "run with ORNITH_PROF=1: GPU seconds are the only measurement here");

    let cfg = TextConfig::load(&dir).unwrap();
    let tok = Tokenizer::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[sidecar_path(&dir)]).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg.clone());
    drop(ws);
    println!("{}\n{}", dir.display(), ctx.device_name());

    let hsize = cfg.hidden_size;
    let ids = tok.encode(PROMPT);
    let mut caches = model.make_cache();
    let processed =
        prefill(&ctx, &model, &mut caches, &ids, prefill_step_size(), None, &mut |_, _| {});
    let (mut hidden, _) = model_call(&ctx, &model, &mut caches, &ids[processed..], None);
    let rows = hidden.numel() / hsize;
    let h = hidden.reshape(&[rows, hsize]).slice_leading(rows - 1, 1);
    let mut primary = ornith_rs::ops::sample::greedy_token(&ctx, &model.lm_head.matmul(&ctx, &h));
    for _ in 0..WARM_STEPS {
        let (nh, _) = model_call(&ctx, &model, &mut caches, &[primary], None);
        hidden = nh;
        let rows = hidden.numel() / hsize;
        let h = hidden.reshape(&[rows, hsize]).slice_leading(rows - 1, 1);
        primary = ornith_rs::ops::sample::greedy_token(&ctx, &model.lm_head.matmul(&ctx, &h));
    }
    ctx.sync();
    let base = snapshot(&caches);
    // The trace and the times describe whichever code paths are in force, so
    // put back what the environment asked for: with both controls set this
    // bin dumps the *old* stream, which is what a change is ranked against.
    let want_fuse = std::env::var("ORNITH_FUSE_ADD_RMS").as_deref() != Ok("0");
    let want_late = std::env::var("ORNITH_MOE_EARLY_GATE").as_deref() != Ok("1");
    ornith_rs::ops::norm::set_fuse_add_rms(want_fuse);
    ornith_rs::ops::moe::set_late_shared_gate(want_late);


    // --- the step, and the same step with the work taken out ---------------
    //
    // Two objects, because they are two different numbers and have been
    // confused before: `model_call` is the trunk alone, which is what
    // `draft_cost` prints and what Phase 28 recorded at 7.56 ms; a decode
    // step is that plus the head and the sampler, which is what a 200-token
    // run's `ms/token` counts.
    let mut step = |ctx: &Context, caches: &mut [LayerCache], head: bool| -> (f64, f64) {
        let run = |caches: &mut [LayerCache]| {
            if head {
                let y = ornith_rs::generate::step(
                    ctx,
                    &model,
                    caches,
                    &[primary],
                    &mut ornith_rs::generate::Greedy,
                    None,
                );
                std::hint::black_box(y);
            } else {
                let (hh, _) = model_call(ctx, &model, caches, &[primary], None);
                std::hint::black_box(&hh);
            }
        };
        run(caches);
        restore(caches, &base);
        ctx.sync();
        ctx.prof_reset();
        for _ in 0..REPS {
            run(caches);
            restore(caches, &base);
        }
        ctx.sync();
        let (_, disp, gpu, _) = ctx.prof_totals();
        (gpu * 1e3 / REPS as f64, disp as f64 / REPS as f64)
    };

    println!("\nthe step, and the same dispatches with their work removed\n");
    println!(
        "  {:<28} {:>9} {:>9} {:>9} {:>7} {:>9}",
        "", "GPU ms", "clamped", "work ms", "issue", "dispatches"
    );
    let mut level = [(0.0f64, 0.0f64, 0.0f64); 2];
    for (j, (name, head)) in [("trunk (model_call)", false), ("a decode step, head and all", true)]
        .into_iter()
        .enumerate()
    {
        let (mut real, mut floor) = (Vec::new(), Vec::new());
        let (mut dr, mut df) = (0.0, 0.0);
        for _ in 0..DRAWS {
            let (ms, d) = step(&ctx, &mut caches, head);
            real.push(ms);
            dr = d;
            ctx.set_grid1(true);
            let (ms, d) = step(&ctx, &mut caches, head);
            floor.push(ms);
            df = d;
            ctx.set_grid1(false);
        }
        assert_eq!(dr, df, "the clamp changed the dispatch stream");
        let (rms, fms) = (median(&mut real), median(&mut floor));
        level[j] = (rms, fms, dr);
        println!(
            "  {name:<28} {rms:>9.3} {fms:>9.3} {:>9.3} {:>6.1}% {dr:>9.0}",
            rms - fms,
            100.0 * fms / rms,
        );
        println!(
            "  {:<28} {:>9} {:>9.3} {:>9} {:>7} {:>9}",
            "  per dispatch, us",
            "",
            fms * 1e3 / dr,
            "",
            "",
            ""
        );
        println!("  {:<28} real {real:.3?}  clamped {floor:.3?}", "  draws");
    }
    let (rms, fms, dr) = level[0];

    let ab_only = std::env::args().any(|a| a == "ab");
    let ab_draws = if ab_only { 8 } else { DRAWS + 1 };
    println!("\nthe add and the norm that reads it, fused and not\n");
    println!("  {:<24} {:>9} {:>11} {:>10}", "", "GPU ms", "dispatches", "draws");
    let (mut on, mut off) = (Vec::new(), Vec::new());
    let (mut don, mut doff) = (0.0, 0.0);
    for _ in 0..ab_draws {
        ornith_rs::ops::norm::set_fuse_add_rms(false);
        let (ms, d) = step(&ctx, &mut caches, false);
        off.push(ms);
        doff = d;
        ornith_rs::ops::norm::set_fuse_add_rms(true);
        let (ms, d) = step(&ctx, &mut caches, false);
        on.push(ms);
        don = d;
    }
    ornith_rs::ops::norm::set_fuse_add_rms(want_fuse);
    let (a, b) = (median(&mut off.clone()), median(&mut on.clone()));
    println!("  {:<24} {a:>9.3} {doff:>11.0}  {off:.3?}", "separate");
    println!("  {:<24} {b:>9.3} {don:>11.0}  {on:.3?}", "fused");
    println!(
        "  {:<24} {:>9.3} {:>11.0}  {:+.2}%",
        "so",
        b - a,
        don - doff,
        100.0 * (b - a) / a
    );

    // --- and the reorder: where the shared expert's gate is issued ----------
    println!("\nthe shared expert's gate, beside the router and beside the gathers\n");
    println!("  {:<24} {:>9} {:>11} {:>10}", "", "GPU ms", "dispatches", "draws");
    let (mut late, mut early) = (Vec::new(), Vec::new());
    for _ in 0..ab_draws {
        ornith_rs::ops::moe::set_late_shared_gate(false);
        early.push(step(&ctx, &mut caches, false).0);
        ornith_rs::ops::moe::set_late_shared_gate(true);
        late.push(step(&ctx, &mut caches, false).0);
    }
    ornith_rs::ops::moe::set_late_shared_gate(want_late);
    let pairs: Vec<f64> = late.iter().zip(&early).map(|(l, e)| l - e).collect();
    let (a, b) = (median(&mut early.clone()), median(&mut late.clone()));
    println!("  {:<24} {a:>9.3} {:>11}  {early:.3?}", "beside the router", "");
    println!("  {:<24} {b:>9.3} {:>11}  {late:.3?}", "beside the gathers", "");
    println!(
        "  {:<24} {:>9.3} {:>11}  mean of pairs {:+.3} ms, {} of {} down",
        "so",
        b - a,
        "",
        pairs.iter().sum::<f64>() / pairs.len() as f64,
        pairs.iter().filter(|d| **d < 0.0).count(),
        pairs.len(),
    );

    // --- and what the router's sort costs, which removal cannot say --------
    //
    // `carg_block_sort` cannot be priced by dropping it: its output is an
    // address, and without it the expert gather reads eight experts at
    // random, which is slower. `ORNITH_FAKE_SORT` swaps it for a kernel that
    // writes a valid order cheaply -- wrong experts, right shape, right
    // count of distinct weights -- so the difference is the sort's own cost
    // less the stand-in's.
    println!("\nthe router's 256-wide argsort, against a stand-in that is not one\n");
    println!("  {:<24} {:>9} {:>11} {:>10}", "", "GPU ms", "dispatches", "draws");
    let (mut sorted, mut faked) = (Vec::new(), Vec::new());
    let (mut ds, mut df2) = (0.0, 0.0);
    for _ in 0..ab_draws {
        ornith_rs::ops::moe::set_fake_sort(false);
        let (ms, d) = step(&ctx, &mut caches, false);
        sorted.push(ms);
        ds = d;
        ornith_rs::ops::moe::set_fake_sort(true);
        let (ms, d) = step(&ctx, &mut caches, false);
        faked.push(ms);
        df2 = d;
    }
    ornith_rs::ops::moe::set_fake_sort(false);
    let pairs: Vec<f64> = faked.iter().zip(&sorted).map(|(f, s)| f - s).collect();
    let (a, b) = (median(&mut sorted.clone()), median(&mut faked.clone()));
    println!("  {:<24} {a:>9.3} {ds:>11.0}  {sorted:.3?}", "the sort");
    println!("  {:<24} {b:>9.3} {df2:>11.0}  {faked:.3?}", "a stand-in");
    println!(
        "  {:<24} {:>9.3} {:>11.0}  mean of pairs {:+.3} ms, {} of {} down",
        "so",
        b - a,
        df2 - ds,
        pairs.iter().sum::<f64>() / pairs.len() as f64,
        pairs.iter().filter(|d| **d < 0.0).count(),
        pairs.len(),
    );

    // --- and what the selection kernel took of it ---------------------------
    println!("\nmlx's sort of 256 experts, against k rounds of selection\n");
    println!("  {:<24} {:>9} {:>11} {:>10}", "", "GPU ms", "dispatches", "draws");
    let (mut srt, mut sel) = (Vec::new(), Vec::new());
    let (mut d1, mut d2) = (0.0, 0.0);
    for _ in 0..ab_draws {
        ornith_rs::ops::moe::set_router_select(false);
        let (ms, d) = step(&ctx, &mut caches, false);
        srt.push(ms);
        d1 = d;
        ornith_rs::ops::moe::set_router_select(true);
        let (ms, d) = step(&ctx, &mut caches, false);
        sel.push(ms);
        d2 = d;
    }
    let pairs: Vec<f64> = sel.iter().zip(&srt).map(|(a, b)| a - b).collect();
    let (a, b) = (median(&mut srt.clone()), median(&mut sel.clone()));
    println!("  {:<24} {a:>9.3} {d1:>11.0}  {srt:.3?}", "sort, slice, gather");
    println!("  {:<24} {b:>9.3} {d2:>11.0}  {sel:.3?}", "one selection");
    println!(
        "  {:<24} {:>9.3} {:>11.0}  mean of pairs {:+.3} ms, {} of {} down",
        "so",
        b - a,
        d2 - d1,
        pairs.iter().sum::<f64>() / pairs.len() as f64,
        pairs.iter().filter(|d| **d < 0.0).count(),
        pairs.len(),
    );

    if ab_only {
        return;
    }

    // --- and what the command buffer costs ----------------------------------
    // (skipped under `ab`, which wants the machine's attention on one pair)
    //
    // `max_dispatches_per_commit` was set at 128 on the wall clock, against a
    // comment that says a command buffer costs ~60 us of GPU time. If that is
    // still true it is 0.8 ms of this step, and it costs nothing to ask: the
    // clamped column is where a fixed per-buffer cost would show up cleanly,
    // since nothing else in it is doing any work.
    println!("\nthe commit size, in GPU time rather than on the wall\n");
    println!(
        "  {:>8} {:>9} {:>9} {:>9} {:>9}",
        "commit", "buffers", "GPU ms", "clamped", "wall ms"
    );
    for n in [32usize, 64, 128, 256, 512, 1024, 4096] {
        ctx.set_commit_every(n);
        let mut r = Vec::new();
        let mut f = Vec::new();
        let mut w = Vec::new();
        for _ in 0..DRAWS {
            let t0 = std::time::Instant::now();
            r.push(step(&ctx, &mut caches, false).0);
            w.push(t0.elapsed().as_secs_f64() * 1e3 / REPS as f64);
            ctx.set_grid1(true);
            f.push(step(&ctx, &mut caches, false).0);
            ctx.set_grid1(false);
        }
        let (_, commits) = {
            let (c, _, _, _) = ctx.prof_totals();
            (0, c as f64 / REPS as f64)
        };
        println!(
            "  {n:>8} {commits:>9.1} {:>9.3} {:>9.3} {:>9.3}",
            median(&mut r),
            median(&mut f),
            median(&mut w),
        );
    }
    ctx.set_commit_every(128);

    // --- the fusion this phase built, against its own control ---------------
    //
    // Two code paths in one process, alternating, from one cache state: the
    // residual add and the norm that reads it as one dispatch, against the
    // two they were. Same arithmetic either way -- the suite and a
    // byte-identical `--greedy` are what say so -- and the only difference
    // that should show up here is 80 dispatches and 77 barriers a token.
    // --- the same step, one dispatch at a time ------------------------------
    //
    // The trace is taken at the real commit size, where it fits in the budget;
    // the times come from a second and a third run of the identical stream at
    // one dispatch a buffer. Encode order does not depend on commit size, so
    // the n'th dispatch line is the n'th time.
    restore(&mut caches, &base);
    ctx.sync();
    ctx.prof_trace_start();
    let (hh, _) = model_call(&ctx, &model, &mut caches, &[primary], None);
    std::hint::black_box(&hh);
    ctx.sync();
    let trace = ctx.prof_trace();
    restore(&mut caches, &base);

    ctx.set_commit_every(1);
    let time_one = |ctx: &Context, caches: &mut [LayerCache]| -> Vec<(String, f64)> {
        ctx.sync();
        ctx.prof_trace_start();
        let (hh, _) = model_call(ctx, &model, caches, &[primary], None);
        std::hint::black_box(&hh);
        ctx.sync();
        ctx.prof_per_dispatch()
    };
    // Interleaved, and the median of the draws dispatch by dispatch: a single
    // 2 us command buffer is not a measurement, and whichever pass runs late
    // carries whatever the machine is doing by then.
    let (mut dr_real, mut dr_floor) = (Vec::new(), Vec::new());
    for _ in 0..DRAWS {
        dr_real.push(time_one(&ctx, &mut caches));
        restore(&mut caches, &base);
        ctx.set_grid1(true);
        dr_floor.push(time_one(&ctx, &mut caches));
        ctx.set_grid1(false);
        restore(&mut caches, &base);
    }
    let pick = |d: &Vec<Vec<(String, f64)>>| -> Vec<(String, f64)> {
        (0..d[0].len())
            .map(|i| {
                let mut v: Vec<f64> = d.iter().map(|p| p[i].1).collect();
                (d[0][i].0.clone(), median(&mut v))
            })
            .collect()
    };
    let t_real = pick(&dr_real);
    let t_floor = pick(&dr_floor);
    ctx.set_commit_every(128);
    restore(&mut caches, &base);

    let (k, e) = (cfg.num_experts_per_tok, cfg.num_experts);
    let mut section = String::from("embed");
    let mut rows: Vec<Row> = Vec::new();
    for line in &trace {
        if let Some(rest) = line.strip_prefix("== ") {
            section = rest.to_string();
            continue;
        }
        if let Some(r) = parse(line, &section, k, e) {
            rows.push(r);
        }
    }
    println!(
        "\ntrace: {} lines, {} dispatches; timed {} and {}",
        trace.len(),
        rows.len(),
        t_real.len(),
        t_floor.len()
    );
    let n = rows.len().min(t_real.len()).min(t_floor.len());
    for i in 0..n {
        assert_eq!(rows[i].kernel, t_real[i].0, "trace and timing disagree at {i}");
        assert_eq!(rows[i].kernel, t_floor[i].0, "the clamp moved dispatch {i}");
        rows[i].t_real = t_real[i].1 * 1e6;
        rows[i].t_floor = t_floor[i].1 * 1e6;
    }
    rows.truncate(n);

    // The control: a kernel that was one threadgroup already is not clamped,
    // so its two times are the same measurement twice.
    let ctl: Vec<&Row> = rows.iter().filter(|r| r.one_group()).collect();
    let mut dv: Vec<f64> = ctl.iter().map(|r| r.t_real - r.t_floor).collect();
    let dmed = median(&mut dv);
    let d: f64 = ctl.iter().map(|r| r.t_real - r.t_floor).sum::<f64>() / ctl.len().max(1) as f64;
    let m: f64 = ctl.iter().map(|r| r.t_real).sum::<f64>() / ctl.len().max(1) as f64;
    println!(
        "control: {} of {} dispatches were one threadgroup already, so the clamp \
         did not touch them\n         mean {m:.3} us, difference mean {d:+.3} us, \
         median {dmed:+.3} us -- anything but zero is this pair of runs' bias",
        ctl.len(),
        rows.len(),
    );

    // --- the joined table, for `tools/epoch_slack.py` -----------------------
    //
    // The trace carries the bindings, which is what says which dispatches
    // could share an epoch; the times say what an epoch would then cost. The
    // ranking wants both, and the join is by position.
    if let Ok(dir) = std::env::var("ORNITH_INV_DUMP") {
        use std::io::Write;
        let mut t = std::fs::File::create(format!("{dir}.trace")).expect("trace file");
        for line in &trace {
            writeln!(t, "{line}").unwrap();
        }
        let mut f = std::fs::File::create(format!("{dir}.tsv")).expect("times file");
        writeln!(f, "#i\tsection\tkernel\tgrid\ttg\tread\twrite\tus\tfloor_us").unwrap();
        for (i, r) in rows.iter().enumerate() {
            writeln!(
                f,
                "{i}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}",
                r.section,
                r.kernel,
                r.grid.0.max(1) * r.grid.1.max(1) * r.grid.2.max(1),
                r.tg.0.max(1) * r.tg.1.max(1) * r.tg.2.max(1),
                r.read_live,
                r.write,
                r.t_real,
                r.t_floor
            )
            .unwrap();
        }
        println!("\ndumped {} dispatches to {dir}.trace and {dir}.tsv", rows.len());
    }

    // --- a layer, dispatch by dispatch --------------------------------------
    for want in ["layer.linear 5", "layer.full 7"] {
        let layer: Vec<&Row> = rows.iter().filter(|r| r.section == want).collect();
        if layer.is_empty() {
            continue;
        }
        println!("\n{want}: {} dispatches\n", layer.len());
        println!(
            "  {:>3} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7}  {}",
            "#", "grid", "tg", "read KB", "wrote KB", "us", "floor", "GB/s", "kernel"
        );
        for (i, r) in layer.iter().enumerate() {
            println!(
                "  {:>3} {:>7} {:>7} {:>8.1} {:>8.1} {:>8.3} {:>8.3} {:>7}  {}",
                i,
                r.grid.0.max(1) * r.grid.1.max(1) * r.grid.2.max(1),
                r.tg.0.max(1) * r.tg.1.max(1) * r.tg.2.max(1),
                r.read_live as f64 / 1024.0,
                r.write as f64 / 1024.0,
                r.t_real,
                r.t_floor,
                if r.work() > 1.0 { format!("{:.0}", r.gbs()) } else { "--".to_string() },
                short(&r.kernel),
            );
        }
        let sum_t: f64 = layer.iter().map(|r| r.t_real).sum();
        let sum_f: f64 = layer.iter().map(|r| r.t_floor).sum();
        let sum_b: usize = layer.iter().map(|r| r.read_live).sum();
        println!(
            "  {:>3} {:>15} {:>8.1} {:>8} {:>8.3} {:>8.3} {:>7.0}",
            "",
            "",
            sum_b as f64 / 1024.0,
            "",
            sum_t,
            sum_f,
            sum_b as f64 / (sum_t - sum_f).max(1e-9) / 1e3,
        );
    }

    // --- the step, by what a dispatch moves ---------------------------------
    println!("\nthe step's dispatches by what they move, one at a time\n");
    println!(
        "  {:<22} {:>6} {:>9} {:>10} {:>10} {:>9} {:>7}",
        "bytes read", "count", "MB", "us", "of it floor", "work us", "GB/s"
    );
    let bands: [(&str, usize, usize); 4] = [
        (">= 4 MB", 4 << 20, usize::MAX),
        ("1 .. 4 MB", 1 << 20, 4 << 20),
        ("64 KB .. 1 MB", 1 << 16, 1 << 20),
        ("< 64 KB", 0, 1 << 16),
    ];
    let (mut tt, mut tf, mut tb, mut tn) = (0.0, 0.0, 0usize, 0usize);
    for (name, lo, hi) in bands {
        let sel: Vec<&Row> = rows.iter().filter(|r| r.read_live >= lo && r.read_live < hi).collect();
        let t: f64 = sel.iter().map(|r| r.t_real).sum();
        let f: f64 = sel.iter().map(|r| r.t_floor).sum();
        let b: usize = sel.iter().map(|r| r.read_live).sum();
        tt += t;
        tf += f;
        tb += b;
        tn += sel.len();
        println!(
            "  {name:<22} {:>6} {:>9.1} {t:>10.1} {f:>10.1} {:>9.1} {:>7.0}",
            sel.len(),
            b as f64 / 1048576.0,
            t - f,
            b as f64 / (t - f).max(1e-9) / 1e3,
        );
    }
    println!(
        "  {:<22} {tn:>6} {:>9.1} {tt:>10.1} {tf:>10.1} {:>9.1} {:>7.0}",
        "the step",
        tb as f64 / 1048576.0,
        tt - tf,
        tb as f64 / (tt - tf).max(1e-9) / 1e3,
    );
    println!(
        "\n  one dispatch a command buffer inflates that to {:.3} ms against the \
         {rms:.3} the step\n  really takes, and the clamp leaves {:.3} ms of it. \
         The step's own two numbers:\n  {:.3} GB read against {:.3} ms with the \
         floor taken off is {:.0} GB/s.",
        tt / 1e3,
        tf / 1e3,
        tb as f64 / 1e9,
        rms - fms,
        tb as f64 / (rms - fms) / 1e6,
    );

    // --- and by kernel, worst first ------------------------------------------
    println!("\nthe same, by kernel\n");
    let mut by: std::collections::HashMap<String, (usize, f64, f64, usize)> = Default::default();
    for r in &rows {
        let e = by.entry(short(&r.kernel).to_string()).or_default();
        e.0 += 1;
        e.1 += r.t_real;
        e.2 += r.t_floor;
        e.3 += r.read_live;
    }
    let mut by: Vec<(String, (usize, f64, f64, usize))> = by.into_iter().collect();
    by.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    println!(
        "  {:<34} {:>6} {:>9} {:>9} {:>9} {:>8}",
        "kernel", "count", "us", "floor", "MB", "GB/s"
    );
    for (name, (n, t, f, b)) in by.iter().take(20) {
        println!(
            "  {name:<34} {n:>6} {t:>9.1} {f:>9.1} {:>9.1} {:>8.0}",
            *b as f64 / 1048576.0,
            *b as f64 / (t - f).max(1e-9) / 1e3,
        );
    }

    // A dump run is for `epoch_slack.py`, which wants the trace and the times
    // and not the hour the class sweep takes.
    if std::env::var_os("ORNITH_INV_DUMP").is_some() {
        return;
    }

    // --- and what one class costs where it actually runs --------------------
    //
    // The table above is isolated cost: one dispatch a command buffer, one at
    // a time. This one asks the other question -- take one class of dispatch,
    // leave everything else alone, and run the real step at the real commit
    // size. Two removals, because they price different things: clamping the
    // grid leaves the launch and takes the work, dropping the dispatch takes
    // both, and the difference between the two differences is what the class
    // pays for being dispatched at all.
    //
    // The first row is the control: a pattern nothing matches must come out
    // at zero, and it is the only thing standing between this table and the
    // machine's drift.
    let mut full: std::collections::HashMap<String, (usize, f64)> = Default::default();
    for r in &rows {
        let e = full.entry(r.kernel.clone()).or_default();
        e.0 += 1;
        e.1 += r.t_real;
    }
    let mut full: Vec<(String, (usize, f64))> = full.into_iter().collect();
    full.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    let mut want: Vec<(String, usize)> =
        vec![("nothing-matches-this".to_string(), 0)];
    want.extend(full.iter().map(|(k, (n, _))| (k.clone(), *n)));

    println!("\nwhat a class of dispatch costs in the step it runs in\n");
    println!(
        "  {:<30} {:>5} {:>9} {:>9} {:>9} {:>9}",
        "kernel", "count", "work ms", "drop ms", "launch us", "isolated"
    );
    let (mut drops, mut works) = (0.0, 0.0);
    for (pat, count) in &want {
        let mut base = Vec::new();
        let mut work = Vec::new();
        let mut gone = Vec::new();
        for _ in 0..DRAWS {
            ctx.set_neuter(None);
            base.push(step(&ctx, &mut caches, false).0);
            ctx.set_neuter(Some((pat, false)));
            work.push(step(&ctx, &mut caches, false).0);
            ctx.set_neuter(Some((pat, true)));
            gone.push(step(&ctx, &mut caches, false).0);
        }
        ctx.set_neuter(None);
        let (b, w, g) = (median(&mut base), median(&mut work), median(&mut gone));
        let iso: f64 = rows.iter().filter(|r| &r.kernel == pat).map(|r| r.t_real).sum();
        println!(
            "  {:<30} {count:>5} {:>9.3} {:>9.3} {:>9.2} {:>9.3}",
            short(pat),
            b - w,
            b - g,
            if *count > 0 { (w - g) * 1e3 / *count as f64 } else { 0.0 },
            iso / 1e3,
        );
        drops += (b - g).max(0.0);
        works += (b - w).max(0.0);
    }
    println!(
        "  {:<30} {:>5} {works:>9.3} {drops:>9.3}",
        "summed, positive terms only",
        rows.len()
    );

    // --- the floor, decomposed the same way ---------------------------------
    //
    // With every grid clamped there is no work left to hide anything, so what
    // dropping a class saves there is its launch and nothing else. If the
    // launches were additive the column would sum to the clamped step.
    println!("\nand inside the clamped step, where nothing is doing any work\n");
    println!("  {:<30} {:>5} {:>9} {:>9}", "kernel dropped", "count", "saves ms", "us/each");
    let mut floor_sum = 0.0;
    for (pat, count) in want.iter().take(9) {
        let mut base = Vec::new();
        let mut gone = Vec::new();
        for _ in 0..DRAWS {
            ctx.set_grid1(true);
            ctx.set_neuter(None);
            base.push(step(&ctx, &mut caches, false).0);
            ctx.set_neuter(Some((pat, true)));
            gone.push(step(&ctx, &mut caches, false).0);
            ctx.set_neuter(None);
            ctx.set_grid1(false);
        }
        let (b, g) = (median(&mut base), median(&mut gone));
        floor_sum += (b - g).max(0.0);
        println!(
            "  {:<30} {count:>5} {:>9.3} {:>9.2}",
            short(pat),
            b - g,
            if *count > 0 { (b - g) * 1e3 / *count as f64 } else { 0.0 },
        );
    }
    println!("  {:<30} {:>5} {floor_sum:>9.3}", "those nine, summed", "");
}

/// The instantiation suffixes carry the dtype and the group size, which are
/// the same everywhere here and cost a column each.
fn short(k: &str) -> &str {
    for pat in ["_bfloat16", "bfloat16", "_uint32", "_float32"] {
        if let Some(i) = k.find(pat) {
            return &k[..i];
        }
    }
    k
}
