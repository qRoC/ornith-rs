//! Where a speculative cycle's time goes, with the draft and the verify
//! separated by construction rather than by arithmetic.
//!
//! `REBUILD.md` Phase 26 decomposed a cycle by differencing two depths:
//!
//! ```text
//!   cycle at depth 3   21.57 ms = 3 drafts + one verify
//!   cycle at depth 1   12.95 ms = 1 draft  + one verify
//!   => one draft        4.31 ms
//! ```
//!
//! That subtraction charges the whole difference to the draft levels, which
//! is only right if a verify costs the same over a 4-token window as over a
//! 2-token one. Nothing had measured that. `ORNITH_PROF` attributes GPU time
//! per kernel over a whole run, and a speculative run is a draft *and* a
//! verify sharing every kernel name they use, so the run could not answer it
//! either.
//!
//! This bin runs each half on its own, in one process, on one set of weights:
//! a draft level with nothing else in flight, and a verify forward at every
//! window width the loop can produce. Every number is GPU seconds (summed
//! `GPUEndTime - GPUStartTime`), never wall.
//!
//! Run:
//!
//! ```text
//!   ORNITH_MODEL=~/.mtplx/models/ornith-1.5-35b-a3b-q6k-full \
//!     ORNITH_PROF=1 cargo run --release --bin draft_cost           # blocks
//!   ... ORNITH_PROF=1 ORNITH_COMMIT_EVERY=1 cargo run --release --bin draft_cost
//!                                                                  # kernels
//! ```

use ornith_rs::array::{Array, Dtype};
use ornith_rs::frspec::{self, FrSpec};
use ornith_rs::generate::{model_call, prefill, prefill_step_size};
use ornith_rs::metal::Context;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::mtp::{MtpHead, make_mtp_cache, sidecar_path};
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::ops::elementwise::swiglu;
use ornith_rs::ops::embedding::embedding;
use ornith_rs::ops::layout::concat_inner;
use ornith_rs::ops::linear::Linear;
use ornith_rs::ops::matmul::{gather_mv, matmul};
use ornith_rs::ops::moe::{
    mix_experts, normalize_scores, sparse_moe_block_dense, switch_glu_dense, top_k,
};
use ornith_rs::ops::norm::rms_norm;
use ornith_rs::ops::sample::greedy_argmax_into;
use ornith_rs::ops::softmax::softmax;
use ornith_rs::refs::model_dir;
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::weights::WeightStore;

const PROMPT: &str = "Explain in one paragraph why a mixture-of-experts model \
                      reads fewer bytes per token than a dense one of the same \
                      parameter count, and what that costs.";

/// Decode steps to run before measuring, so the caches hold a real generation
/// rather than a prompt's last chunk.
const WARM_STEPS: usize = 8;

const REPS: usize = 24;
/// A verify forward is ~8 ms, so fewer of those.
const VERIFY_REPS: usize = 8;

/// GPU milliseconds a closure costs, per repetition.
///
/// The queue is drained on both sides, so every command buffer the closure
/// commits is harvested inside the window and none from outside it is.
fn gpu_ms(ctx: &Context, reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    ctx.sync();
    ctx.prof_reset();
    for _ in 0..reps {
        f();
    }
    ctx.sync();
    let (_, _, gpu_s, _) = ctx.prof_totals();
    gpu_s * 1e3 / reps as f64
}

/// The same, also reporting dispatches per repetition.
fn gpu_ms_n(ctx: &Context, reps: usize, mut f: impl FnMut()) -> (f64, f64) {
    f();
    ctx.sync();
    ctx.prof_reset();
    for _ in 0..reps {
        f();
    }
    ctx.sync();
    let (_, disp, gpu_s, _) = ctx.prof_totals();
    (gpu_s * 1e3 / reps as f64, disp as f64 / reps as f64)
}

/// The least of three passes. A block measured on its own is a few hundred
/// microseconds with a sync on each side, which is exactly the size this
/// machine's interference comes in; the whole level reproduces to 0.5% and
/// its parts do not unless they are taken this way.
fn best_ms(ctx: &Context, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        best = best.min(gpu_ms(ctx, reps, &mut f));
    }
    best
}

fn row(name: &str, ms: f64, total: f64) {
    println!("  {name:<38} {ms:>8.3} ms {:>6.1}%", 100.0 * ms / total);
}

/// The per-kernel table for whatever was just measured. Only the GPU column
/// needs `ORNITH_COMMIT_EVERY=1`; the counts are right either way.
fn kernels(ctx: &Context, reps: usize, top: usize, label: &str) {
    let (by, mixed) = ctx.prof_kernel_times();
    let total: f64 = by.iter().map(|(_, _, s)| *s).sum::<f64>();
    println!("\n{label}, per kernel\n");
    println!("  {:>9} {:>7} {:>6}  {}", "ms/call", "share", "count", "kernel");
    for (name, n, s) in by.iter().take(top) {
        if *n == 0 {
            continue;
        }
        println!(
            "  {:>9.3} {:>6.1}% {:>6.1}  {name}",
            s * 1e3 / reps as f64,
            100.0 * s / total.max(1e-12),
            *n as f64 / reps as f64,
        );
    }
    println!(
        "  attributed {:.3} ms/call; in multi-dispatch buffers {:.3} ms \
         (set ORNITH_COMMIT_EVERY=1 to attribute those)",
        total * 1e3 / reps as f64,
        mixed * 1e3 / reps as f64
    );
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
    let head = MtpHead::load(&ctx, &mut ws, &cfg);
    drop(ws);
    let fr = FrSpec::load(&ctx, &model.lm_head, &frspec::default_path()).ok();
    let lm: &Linear = fr.as_ref().map(|f| &f.head).unwrap_or(&model.lm_head);
    println!("{}\n{}", dir.display(), ctx.device_name());
    match &fr {
        Some(f) => println!("frspec: {} of {} rows", f.n, f.vocab),
        None => println!("frspec: absent, drafting through the full head"),
    }
    println!(
        "invariant window: {}\n",
        if ornith_rs::ops::invariant::for_rows(2) { "on" } else { "off" }
    );

    // A real state: prompt, then a few committed tokens.
    let hsize = cfg.hidden_size;
    let ids = tok.encode(PROMPT);
    let mut caches = model.make_cache();
    let processed = prefill(
        &ctx,
        &model,
        &mut caches,
        &ids,
        prefill_step_size(),
        None,
        &mut |_, _| {},
    );
    let (mut hidden, _) = model_call(&ctx, &model, &mut caches, &ids[processed..], None);
    let mut rows = hidden.numel() / hsize;
    let mut h = hidden.reshape(&[rows, hsize]).slice_leading(rows - 1, 1);
    let mut primary = ornith_rs::ops::sample::greedy_token(&ctx, &model.lm_head.matmul(&ctx, &h));
    for _ in 0..WARM_STEPS {
        let (nh, _) = model_call(&ctx, &model, &mut caches, &[primary], None);
        hidden = nh;
        rows = hidden.numel() / hsize;
        h = hidden.reshape(&[rows, hsize]).slice_leading(rows - 1, 1);
        primary = ornith_rs::ops::sample::greedy_token(&ctx, &model.lm_head.matmul(&ctx, &h));
    }
    ctx.sync();

    // --- one draft level, whole -------------------------------------------
    //
    // Exactly what `mtp_generate::draft` does at depth 1: a fresh cache (the
    // `Cycle` policy's), one `head.forward`, and the argmax whose result the
    // next level would read.
    let one_level = || {
        let mut c = make_mtp_cache();
        let window = Array::from_slice(&ctx, &[primary, 0u32], &[2], Dtype::U32);
        let t0 = window.slice_leading(0, 1);
        let (logits, _hid) =
            head.forward(&ctx, &model.cfg, &model.embed_tokens, lm, &h, &t0, &mut c[0]);
        let slot = window.slice_leading(1, 1);
        match &fr {
            Some(f) => {
                let local = Array::empty(&ctx, &[1], Dtype::U32);
                greedy_argmax_into(&ctx, &logits, &local);
                f.map_id(&ctx, &local, &slot);
            }
            None => greedy_argmax_into(&ctx, &logits, &slot),
        }
    };
    let (mut level_ms, level_disp) = gpu_ms_n(&ctx, REPS, one_level);
    level_ms = level_ms.min(best_ms(&ctx, REPS, one_level));
    kernels(&ctx, REPS, 12, &format!("a draft level, {level_disp:.0} dispatches"));

    // --- the draft's blocks -------------------------------------------------
    //
    // `head.hidden` minus the pieces that can be run on their own. The MoE
    // block and the head's prefix are public ops called with the head's own
    // weights, so what is left over is the attention plus the two residual
    // adds and their norms.
    let ids1 = Array::from_slice(&ctx, &[primary], &[1], Dtype::U32);
    let hidden_ms = best_ms(&ctx, REPS, || {
        let mut c = make_mtp_cache();
        let _ = head.hidden(&ctx, &model.cfg, &model.embed_tokens, &h, &ids1, &mut c[0]);
    });
    let pre_ms = best_ms(&ctx, REPS, || {
        let emb = embedding(&ctx, &model.embed_tokens, &ids1).reshape(&[1, 1, hsize]);
        let e = rms_norm(&ctx, &emb, Some(&head.pre_fc_norm_embedding), cfg.rms_norm_eps);
        let hh = rms_norm(
            &ctx,
            &h.reshape(&[1, 1, hsize]),
            Some(&head.pre_fc_norm_hidden),
            cfg.rms_norm_eps,
        );
        let joined = concat_inner(&ctx, &e, &hh);
        let _ = matmul(&ctx, &joined.reshape(&[1, 2 * hsize]), &head.fc, true);
    });
    let x1 = h.reshape(&[1, 1, hsize]);
    let moe_ms = best_ms(&ctx, REPS, || {
        let _ = sparse_moe_block_dense(
            &ctx,
            &x1,
            &head.mlp.gate,
            &head.mlp.switch_gate,
            &head.mlp.switch_up,
            &head.mlp.switch_down,
            &head.mlp.shared_gate,
            &head.mlp.shared_up,
            &head.mlp.shared_down,
            &head.mlp.shared_expert_gate,
            cfg.num_experts_per_tok,
            cfg.norm_topk_prob,
        );
    });
    let hrow = h.reshape(&[1, hsize]);
    let lm_ms = best_ms(&ctx, REPS, || {
        let _ = lm.matmul(&ctx, &hrow);
    });
    let lm_full_ms = best_ms(&ctx, REPS, || {
        let _ = model.lm_head.matmul(&ctx, &hrow);
    });

    println!("\na draft level, by block\n");
    row("whole level", level_ms, level_ms);
    row("  head.hidden", hidden_ms, level_ms);
    row("    prefix: embed, norms, fc", pre_ms, level_ms);
    row("    MoE block, 256 experts", moe_ms, level_ms);
    row("    attention + residuals (rest)", hidden_ms - pre_ms - moe_ms, level_ms);
    row("  lm_head, pruned", lm_ms, level_ms);
    println!("  {:<38} {lm_full_ms:>8.3} ms  (not on the draft path)", "lm_head, full");

    // --- inside the MoE block ----------------------------------------------
    let x2 = h.reshape(&[1, hsize]);
    let gates = softmax(&ctx, &matmul(&ctx, &x2, &head.mlp.gate, true), true);
    let (inds, _) = top_k(&ctx, &gates, cfg.num_experts_per_tok);
    ctx.sync();

    let gate_ms = best_ms(&ctx, REPS, || {
        let _ = matmul(&ctx, &x2, &head.mlp.gate, true);
    });
    let softmax_ms = best_ms(&ctx, REPS, || {
        let g = matmul(&ctx, &x2, &head.mlp.gate, true);
        let _ = softmax(&ctx, &g, true);
    });
    let topk_ms = best_ms(&ctx, REPS, || {
        let (i, s) = top_k(&ctx, &gates, cfg.num_experts_per_tok);
        let _ = normalize_scores(&ctx, &s);
        let _ = i;
    });
    let glu_ms = best_ms(&ctx, REPS, || {
        let _ = switch_glu_dense(
            &ctx,
            &x2,
            &inds,
            &head.mlp.switch_gate,
            &head.mlp.switch_up,
            &head.mlp.switch_down,
        );
    });
    let shared_ms = best_ms(&ctx, REPS, || {
        let sg = matmul(&ctx, &x2, &head.mlp.shared_gate, true);
        let su = matmul(&ctx, &x2, &head.mlp.shared_up, true);
        let sact = swiglu(&ctx, &sg, &su);
        let _ = matmul(&ctx, &sact, &head.mlp.shared_down, true);
        let _ = matmul(&ctx, &x2, &head.mlp.shared_expert_gate, true);
    });
    let y = switch_glu_dense(
        &ctx,
        &x2,
        &inds,
        &head.mlp.switch_gate,
        &head.mlp.switch_up,
        &head.mlp.switch_down,
    );
    let scores = Array::empty(&ctx, &[1, cfg.num_experts_per_tok], Dtype::BF16);
    ornith_rs::model::cache::zero(&ctx, &scores);
    ctx.sync();
    let mix_ms = best_ms(&ctx, REPS, || {
        let _ = mix_experts(&ctx, &y, &scores);
    });

    println!("\ninside the head's MoE block ({moe_ms:.3} ms)\n");
    row("router gate matmul", gate_ms, moe_ms);
    row("  + softmax", softmax_ms - gate_ms, moe_ms);
    row("top-k: argsort 256, slice, gather, norm", topk_ms, moe_ms);
    row("switch_glu: 3 gather_mv, 8 experts", glu_ms, moe_ms);
    row("shared expert: 4 matmuls + swiglu", shared_ms, moe_ms);
    row("mix: scale rows, sum over k", mix_ms, moe_ms);

    // --- the bf16 expert gather, by row count -------------------------------
    println!("\nthe head's expert gather alone, by rows\n");
    println!("  {:>5} {:>12} {:>10} {:>10}", "rows", "gather ms", "MB", "GB/s");
    for k in [1usize, 2, 4, 8, 16, 32] {
        let idx: Vec<u32> = (0..k as u32).collect();
        let rhs = Array::from_slice(&ctx, &idx, &[k], Dtype::U32);
        let lhs_i = Array::from_slice(&ctx, &vec![0u32; k], &[k], Dtype::U32);
        let ms = best_ms(&ctx, REPS, || {
            let _ = gather_mv(&ctx, &x2, &head.mlp.switch_gate, &lhs_i, &rhs);
        });
        let mb =
            (k * head.mlp.switch_gate.shape[1] * head.mlp.switch_gate.shape[2] * 2) as f64 / 1e6;
        println!("  {k:>5} {ms:>12.3} {mb:>10.1} {:>10.1}", mb / ms);
    }

    // --- the verify forward, by window width --------------------------------
    //
    // The half Phase 26's subtraction assumed was flat. A window of `W` tokens
    // puts `8W` rows through every layer's expert gather, so this is where the
    // depth difference could be hiding. Each repetition commits one token --
    // `rewind(.., 1)` is what the loop does when no draft is accepted -- so
    // the caches advance by one per rep, exactly as a real cycle would.
    println!("\nthe verify forward, by window width\n");
    println!(
        "  {:>5} {:>11} {:>11} {:>11} {:>9}",
        "width", "verify ms", "vs width 1", "per token", "dispatch"
    );
    let mut verify_ms = [0.0f64; 5];
    let mut tables: Vec<(usize, Vec<(String, usize, f64)>)> = Vec::new();
    for w in 1..=4usize {
        let win: Vec<u32> = (0..w).map(|_| primary).collect();
        let window = Array::from_slice(&ctx, &win, &[w], Dtype::U32);
        let (ms, disp) = gpu_ms_n(&ctx, VERIFY_REPS, || {
            let (vh, _, capture) = model.forward_capture(&ctx, &window, &mut caches);
            let _ = model.lm_head.matmul(&ctx, &vh);
            capture.rewind(&mut caches, 1);
        });
        verify_ms[w] = ms;
        println!(
            "  {w:>5} {ms:>11.3} {:>11.3} {:>11.3} {disp:>9.0}",
            ms - verify_ms[1],
            ms / w as f64,
        );
        let (by, _) = ctx.prof_kernel_times();
        tables.push((w, by));
    }
    // What grew, per kernel, over the widths. A kernel whose dispatch count
    // does not move but whose time does is re-reading the same weights once
    // per row; one whose count moves with the width is doing new work.
    let names: Vec<String> = {
        let mut v: Vec<(String, f64)> = tables[3]
            .1
            .iter()
            .map(|(n, _, s)| {
                let at1 = tables[0].1.iter().find(|(m, _, _)| m == n).map_or(0.0, |(_, _, s)| *s);
                (n.clone(), s - at1)
            })
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.into_iter().take(12).map(|(n, _)| n).collect()
    };
    println!("\nwhat grows with the window, per kernel (ms/verify, count)\n");
    println!("  {:>16} {:>16} {:>16}  kernel", "width 1", "width 2", "width 4");
    for n in &names {
        let cell = |t: &(usize, Vec<(String, usize, f64)>)| {
            t.1.iter()
                .find(|(m, _, _)| m == n)
                .map(|(_, c, s)| format!("{:>9.3} x{:<5}", s * 1e3 / VERIFY_REPS as f64, c))
                .unwrap_or_else(|| "        --      ".to_string())
        };
        println!("  {} {} {}  {n}", cell(&tables[0]), cell(&tables[1]), cell(&tables[3]));
    }

    // --- what the cycle is made of ------------------------------------------
    let step_ms = gpu_ms(&ctx, VERIFY_REPS, || {
        let (hh, _) = model_call(&ctx, &model, &mut caches, &[primary], None);
        let _ = model.lm_head.matmul(&ctx, &hh);
    });
    println!("\nthe cycle, assembled from its halves\n");
    println!("  {:<38} {step_ms:>8.3} ms", "an autoregressive decode step");
    println!("  {:<38} {level_ms:>8.3} ms", "one draft level");
    for d in [1usize, 3] {
        let cycle = d as f64 * level_ms + verify_ms[d + 1];
        println!(
            "  depth {d}: {d} draft + verify(w={})        {cycle:>8.3} ms",
            d + 1
        );
    }
    println!(
        "\n  Phase 26 differenced two depths and charged all of it to the draft:\n  \
         (cycle(3) - cycle(1)) / 2 = {:.3} ms here, against {:.3} ms measured directly.",
        ((3.0 * level_ms + verify_ms[4]) - (level_ms + verify_ms[2])) / 2.0,
        level_ms
    );
}
