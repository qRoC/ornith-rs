//! Does the expert gather pay for rows, or for distinct experts?
//!
//! `HANDOFF.md` wrote the MoE side of a speculative window off with a number:
//! "a 4-token window reads ~30.5 distinct experts for its 32 (token, expert)
//! pairs -- 1.05 rows per expert, which is nothing to group". That is
//! `256 * (1 - (1 - 8/256)^4)`, the expectation under *independent* routing,
//! and routing is not independent. `tools/router_stats.py` measures **20.85**
//! distinct for those 32 pairs, 1.53 rows per expert; shuffling the token
//! order lands back at 28.3, which is the control that says the structure is
//! temporal and real. Adjacent tokens share 2.84 of their 8 experts against
//! 0.25 under independence.
//!
//! So there is something to group. What that is worth depends on a question
//! no statistic can answer: whether the second read of a repeated expert
//! already comes from cache. At d = 4 the gather touches ~47 MB of distinct
//! weights across 32 reads, which is close enough to this machine's
//! last-level cache that it could go either way.
//!
//! This holds the row count fixed and varies only how many distinct experts
//! those rows name. Flat means the cache is already doing it and there is
//! nothing to win. Sloped means the sort threshold in `switch_glu` -- which
//! today groups nothing below 64 rows, i.e. depth 8 -- is leaving that slope
//! on the table.
//!
//! Run: `cargo run --release --bin moe_reuse`

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::model::cache::zero;
use ornith_rs::ops::linear::Linear;
use ornith_rs::ops::moe::switch_glu_sorted;
use ornith_rs::refs::model_dir;
use ornith_rs::weights::WeightStore;

const P: &str = "language_model.model.layers.0.mlp";
const K: usize = 8;

/// GPU seconds when `ORNITH_PROF=1`, else the drained wall clock.
///
/// The wall was what this bin measured when it was written, and on this
/// machine the wall moves 3x between sittings while GPU seconds hold to 1%
/// inside one.
fn best_of<F: FnMut()>(ctx: &Context, reps: usize, mut run: F) -> f64 {
    if ctx.prof_on() {
        let mut best = f64::MAX;
        for _ in 0..reps {
            run();
            ctx.sync();
            ctx.prof_reset();
            run();
            ctx.sync();
            let (_, _, gpu_s, _) = ctx.prof_totals();
            best = best.min(gpu_s);
        }
        return best;
    }
    let mut best = f64::MAX;
    for _ in 0..reps {
        ctx.sync();
        let t0 = std::time::Instant::now();
        run();
        ctx.sync();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

/// `d` tokens of `K` experts each, where each token reuses `share` of the
/// previous token's experts and takes `K - share` fresh ones. Distinct total
/// is `K + (d - 1) * (K - share)`, so `share` sweeps the whole range with the
/// row count held at `d * K`.
fn indices(d: usize, share: usize) -> (Vec<u32>, usize) {
    let mut out: Vec<u32> = (0..K as u32).collect();
    let mut next = K as u32;
    for t in 1..d {
        let prev = out[(t - 1) * K..t * K].to_vec();
        for i in 0..K {
            if i < share {
                out.push(prev[i]);
            } else {
                out.push(next % 256);
                next += 1;
            }
        }
    }
    let mut uniq = out.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let n = uniq.len();
    (out, n)
}

fn main() {
    let dir = model_dir();
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    // `ws.linear` and not `ws.quant_matrix`: this bin was written against the
    // affine artifact and panicked on the one the project ships, whose expert
    // modules are Q4_K. The question it asks -- does the gather pay for rows
    // or for distinct experts -- is one that has to be asked on the shipping
    // artifact, since Phase 28 showed the two families answer the
    // traffic-against-arithmetic question differently.
    let g = ws.linear(&ctx, &format!("{P}.switch_mlp.gate_proj"));
    let u = ws.linear(&ctx, &format!("{P}.switch_mlp.up_proj"));
    let dn = ws.linear(&ctx, &format!("{P}.switch_mlp.down_proj"));
    let dim = g.in_dims();
    println!("{}\n{} experts, dim {dim}\n", dir.display(), g.num_experts());

    for d in [2usize, 3, 4, 6, 8] {
        let x = Array::empty(&ctx, &[d, dim], Dtype::BF16);
        zero(&ctx, &x);
        println!("d = {d}, {} rows", d * K);
        println!(
            "  {:>5} {:>8} {:>11} {:>11} {:>9}",
            "share", "distinct", "unsorted ms", "sorted ms", "sorted/uns"
        );
        let mut base = None;
        for share in [0usize, 2, 4, 6, 8] {
            if share >= K && d > 1 && K - share == 0 && share != 8 {
                continue;
            }
            let (ind, n) = indices(d, share);
            let inds = Array::from_slice(&ctx, &ind, &[d, K], Dtype::U32);
            let uns = best_of(&ctx, 20, || {
                let _ = switch_glu_sorted(&ctx, &x, &inds, &g, &u, &dn, Some(false));
            });
            let srt = best_of(&ctx, 20, || {
                let _ = switch_glu_sorted(&ctx, &x, &inds, &g, &u, &dn, Some(true));
            });
            if base.is_none() {
                base = Some(uns);
            }
            println!(
                "  {share:>5} {n:>8} {:>11.3} {:>11.3} {:>9.2}",
                uns * 1e3,
                srt * 1e3,
                srt / uns
            );
        }
        let (_, n_real) = indices(d, if d == 2 { 3 } else { 4 });
        println!("  (measured routing sits near distinct = {n_real} for this d)\n");
    }
}
