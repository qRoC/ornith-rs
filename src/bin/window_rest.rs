//! What the *other* two thirds of a verify window row are made of.
//!
//! `REBUILD.md` Phase 27 split the cost of an extra window row three ways --
//! 41% dense k-quant mat-vecs, 30% the expert gather, 29% `forward_capture`
//! stepping the gated-delta recurrence one token at a time -- and Phase 28
//! then took the first third apart and found its 41% was not what it was
//! called: the weight re-read that was going to be taken back cost almost
//! nothing, because past the first row the mat-vec is arithmetic-bound.
//!
//! **That is the lesson this bin exists to apply to the other two.** A
//! quantity that grows with the window is not evidence about *which* term
//! grows with it. In the recurrence, two things grow together: the *work*
//! (one token's recurrence, W times) and the *dispatch count* (W dispatches
//! instead of one). The fix Phase 27 named -- "one recurrence dispatch
//! writing all `T` snapshots instead of one per token" -- is only worth
//! anything if the second is the larger, and nothing has measured that.
//!
//! It is measurable **without writing a kernel**, which is the point.
//! `gated_delta_kernel` already carries a `T` axis: its `for (t = 0; t < T;
//! ++t)` runs inside the kernel with the state in float32 registers, and
//! `GatedDeltaNet::forward_capture` splits it into `T` calls only to get a
//! state snapshot per token. So one call at `T = W` against `W` calls at
//! `T = 1`, on the same inputs, separates the work from the dispatches
//! exactly:
//!
//!   - stepped ~ W x fused  =>  it is the work, and fusing buys nothing.
//!   - stepped ~ W x fused with fused ~ one dispatch  =>  it is the
//!     dispatches, and the whole difference is available.
//!
//! The same for the three other per-token dispatches in that loop: `beta`
//! and `compute_g` (which `gated_delta_update` calls per token), the snapshot
//! slice, and the `concat` that reassembles `y` -- which a fused kernel would
//! not need at all, since it writes `[B, W, Hv, Dv]` in one piece.
//!
//! The expert gather's own version of this question is already answered and
//! is not repeated here: `src/bin/moe_reuse.rs` holds the row count fixed and
//! varies the distinct experts, i.e. holds the arithmetic and varies the
//! traffic, and measures 3% between 32 distinct experts and 8.
//!
//! Run (GPU seconds, so `ORNITH_PROF=1`):
//!
//! ```text
//!   ORNITH_MODEL=~/.mtplx/models/ornith-1.5-35b-a3b-q6k-full \
//!     ORNITH_PROF=1 cargo run --release --bin window_rest
//! ```

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::model::cache::zero;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::qwen3_5::slice_time;
use ornith_rs::ops::gdn::{beta, compute_g, concat_time, gated_delta_kernel};
use ornith_rs::refs::model_dir;
use ornith_rs::weights::WeightStore;

const REPS: usize = 64;
/// Three passes, least of each cell: these are microseconds apiece and this
/// machine's interference comes in exactly that size.
const PASSES: usize = 7;
const WIDTHS: [usize; 4] = [1, 2, 3, 4];

/// GPU milliseconds a repetition, with the queue drained between them.
///
/// Serial on purpose for everything that is not already a dependency chain:
/// letting repetitions overlap would hide the per-dispatch cost this bin is
/// trying to weigh. The price is that each repetition then carries one
/// command buffer's own overhead, which is why the recurrence -- the one
/// measurement that has to be right -- uses [`chained_ms`] instead.
fn gpu_ms(ctx: &Context, mut f: impl FnMut()) -> (f64, f64) {
    f();
    ctx.sync();
    ctx.prof_reset();
    for _ in 0..REPS {
        f();
        ctx.sync();
    }
    let (_, disp, gpu_s, _) = ctx.prof_totals();
    (gpu_s * 1e3 / REPS as f64, disp as f64 / REPS as f64)
}

/// The same, for a closure that threads one array from each repetition into
/// the next.
///
/// The recurrence is a chain by nature -- token `t`'s state is token
/// `t + 1`'s input -- so feeding each repetition the previous one's state
/// makes the whole run one chain: nothing can overlap, no repetition needs a
/// sync to stop it, and one command buffer's overhead is divided by `REPS`
/// instead of charged to every repetition. Without this the fixed cost of a
/// buffer lands on whichever form issues fewer dispatches, which is the form
/// under test.
fn chained_ms(ctx: &Context, state: &Array, mut f: impl FnMut(&Array) -> Array) -> (f64, f64) {
    let mut s = f(state);
    ctx.sync();
    ctx.prof_reset();
    for _ in 0..REPS {
        s = f(&s);
    }
    ctx.sync();
    std::hint::black_box(&s);
    let (_, disp, gpu_s, _) = ctx.prof_totals();
    (gpu_s * 1e3 / REPS as f64, disp as f64 / REPS as f64)
}

/// The least of three passes, and the dispatch count that went with it --
/// which is the check that the thing being timed is the thing intended. A
/// stepped form that is not `W` dispatches a repetition is not stepped.
fn best_chained(ctx: &Context, state: &Array, mut f: impl FnMut(&Array) -> Array) -> (f64, f64) {
    let mut v: Vec<(f64, f64)> = (0..PASSES).map(|_| chained_ms(ctx, state, &mut f)).collect();
    pick(&mut v)
}

/// The **median** pass, not the least.
///
/// `window_mv` takes the least of three, which is right when interference can
/// only add time. Here it is not: these are microsecond dispatches in a chain,
/// and a pass whose command buffers land outside the profiling window reads
/// *low*, not high. A 3x-low outlier next to a correct dispatch count is what
/// that looks like, and taking the minimum would report it. The median over
/// seven survives both failures; the spread is printed so it cannot hide.
fn pick(v: &mut [(f64, f64)]) -> (f64, f64) {
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    v[v.len() / 2]
}

fn best(ctx: &Context, mut f: impl FnMut()) -> (f64, f64) {
    let mut v: Vec<(f64, f64)> = (0..PASSES).map(|_| gpu_ms(ctx, &mut f)).collect();
    pick(&mut v)
}

fn main() {
    let dir = model_dir();
    let ctx = Context::new().expect("metal");
    assert!(ctx.prof_on(), "run with ORNITH_PROF=1");
    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    println!("{}\n{}\n", dir.display(), ctx.device_name());

    let (hk, hv) = (cfg.linear_num_key_heads, cfg.linear_num_value_heads);
    let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
    let lp = "language_model.model.layers.0.linear_attn";
    let a_log = ws.tensor(&ctx, &format!("{lp}.A_log"));
    let dt_bias = ws.tensor(&ctx, &format!("{lp}.dt_bias"));
    let conv_w = cfg.linear_conv_kernel_dim - 1;
    let qkv_dim = 2 * cfg.key_dim() + cfg.value_dim();
    println!(
        "  Hk={hk} Dk={dk}, Hv={hv} Dv={dv}, state {} floats, conv window {conv_w}\n",
        hv * dv * dk
    );

    // The recurrence's state is [1, Hv, Dv, Dk] float32 -- 2 MB a layer -- and
    // every step reads it and writes it whole. That is the quantity the table
    // below is really about, so it is printed with the rate each row works
    // out to.
    //
    // (An earlier version measured a bare `zero` of the same shape here as a
    // reference. It was removed: keeping a second state-sized array alive for
    // the whole run made the W = 4 cell read 733 GB/s -- above this machine's
    // ceiling, so not a measurement -- while W = 1..3 stayed on their line.
    // An instrument that perturbs the thing it measures is not a reference.)
    let state_mb = (hv * dv * dk * 4) as f64 / 1e6;
    println!("  the recurrence state is {state_mb:.1} MB a layer\n");

    println!("one dispatch over W tokens, against W dispatches of one token\n");
    println!(
        "  {:>5} {:>14} {:>14} {:>8} {:>9} {:>8}",
        "W", "fused", "stepped", "stepped/", "per step", "stepped"
    );
    println!(
        "  {:>5} {:>14} {:>14} {:>8} {:>9} {:>8}\n",
        "", "ms/dispatches", "ms/dispatches", "fused", "ms", "vs W=1"
    );

    for part in ["recurrence", "gate (beta + compute_g)", "snapshot slice", "output concat"] {
        println!("  {part}");
        let mut base = 0.0;
        for w in WIDTHS {
            // Inputs for a window of `w`, in the shapes `forward_capture`
            // hands the recurrence.
            let q = Array::empty(&ctx, &[1, w, hk, dk], Dtype::BF16);
            let k = Array::empty(&ctx, &[1, w, hk, dk], Dtype::BF16);
            let v = Array::empty(&ctx, &[1, w, hv, dv], Dtype::BF16);
            let a = Array::empty(&ctx, &[1, w, hv], Dtype::BF16);
            let b = Array::empty(&ctx, &[1, w, hv], Dtype::BF16);
            let state = Array::empty(&ctx, &[1, hv, dv, dk], Dtype::F32);
            let conv_input = Array::empty(&ctx, &[1, w + conv_w, qkv_dim], Dtype::BF16);
            for arr in [&q, &k, &v, &a, &b, &state, &conv_input] {
                zero(&ctx, arr);
            }
            let g = compute_g(&ctx, &a_log, &a, &dt_bias);
            let bt = beta(&ctx, &b);

            let slice_t = |t: usize, x: &Array, n: usize| -> Array {
                let (_, _, d) = (x.shape[0], x.shape[1], x.shape[2]);
                let _ = d;
                slice_time(&ctx, x, t, n)
            };

            let ((fused, fd), (stepped, sd)) = match part {
                "recurrence" => (
                    best_chained(&ctx, &state, |s| {
                        gated_delta_kernel(&ctx, &q, &k, &v, &g, &bt, s, None).1
                    }),
                    best_chained(&ctx, &state, |s0| {
                        let mut s = s0.clone();
                        for t in 0..w {
                            let qt = q.reshape(&[w, hk * dk]).slice_leading(t, 1).reshape(&[1, 1, hk, dk]);
                            let kt = k.reshape(&[w, hk * dk]).slice_leading(t, 1).reshape(&[1, 1, hk, dk]);
                            let vt = v.reshape(&[w, hv * dv]).slice_leading(t, 1).reshape(&[1, 1, hv, dv]);
                            let gt = g.reshape(&[w, hv]).slice_leading(t, 1).reshape(&[1, 1, hv]);
                            let bp = bt.reshape(&[w, hv]).slice_leading(t, 1).reshape(&[1, 1, hv]);
                            s = gated_delta_kernel(&ctx, &qt, &kt, &vt, &gt, &bp, &s, None).1;
                        }
                        s
                    }),
                ),
                "gate (beta + compute_g)" => (
                    best(&ctx, || {
                        let _ = beta(&ctx, &b);
                        let _ = compute_g(&ctx, &a_log, &a, &dt_bias);
                    }),
                    best(&ctx, || {
                        for t in 0..w {
                            let at = a.reshape(&[w, hv]).slice_leading(t, 1).reshape(&[1, 1, hv]);
                            let bp = b.reshape(&[w, hv]).slice_leading(t, 1).reshape(&[1, 1, hv]);
                            let _ = beta(&ctx, &bp);
                            let _ = compute_g(&ctx, &a_log, &at, &dt_bias);
                        }
                    }),
                ),
                // The snapshot has no fused form today -- that is the kernel
                // Phase 27 asked for -- so `fused` is the one slice a
                // non-capturing forward makes and `stepped` is the W the
                // capture makes.
                "snapshot slice" => (
                    best(&ctx, || {
                        let _ = slice_t(w, &conv_input, conv_w);
                    }),
                    best(&ctx, || {
                        for t in 0..w {
                            let _ = slice_t(t + 1, &conv_input, conv_w);
                        }
                    }),
                ),
                // A fused recurrence writes `[B, W, Hv, Dv]` in one piece, so
                // its `fused` column is zero by construction: there is nothing
                // to concatenate. What `stepped` costs is what it would save.
                _ => {
                    // The parts are made and zeroed once, outside the timer:
                    // `zero` is a dispatch too, and counting it here is how
                    // this row first read 7 dispatches for 3 concatenations.
                    let parts: Vec<Array> = (0..w)
                        .map(|_| {
                            let a = Array::empty(&ctx, &[1, 1, hv * dv], Dtype::BF16);
                            zero(&ctx, &a);
                            a
                        })
                        .collect();
                    (
                        (0.0, 0.0),
                        best(&ctx, || {
                            let mut acc = parts[0].clone();
                            for p in &parts[1..] {
                                acc = concat_time(&ctx, &acc, p);
                            }
                            std::hint::black_box(&acc);
                        }),
                    )
                }
            };
            if w == 1 {
                base = stepped.max(1e-9);
            }
            let ratio = if fused > 0.0 {
                format!("{:.2}", stepped / fused)
            } else {
                "--".to_string()
            };
            // For the recurrence, what the stepped form's time works out to
            // as a rate over the state it moves: `W` dispatches, each reading
            // and writing the whole state.
            let rate = if part == "recurrence" {
                format!("{:>7.0}", 2.0 * state_mb * w as f64 / stepped)
            } else {
                "      -".to_string()
            };
            println!(
                "  {w:>5} {fused:>9.4}/{fd:<4.1} {stepped:>9.4}/{sd:<4.1} {ratio:>8} {:>9.4} {:>8.2} {rate}",
                stepped / w as f64,
                stepped / base,
            );
        }
        println!();
    }
}
