//! What a window of rows costs a mat-vec, module by module.
//!
//! A speculative verify runs the trunk over `depth + 1` rows, and
//! `src/bin/draft_cost.rs` measures what that costs end to end: on `q6k-full`
//! a verify forward is 8.22 ms at one row and 18.15 at four. This is the same
//! question one module at a time -- how a quantized mat-vec's GPU time grows
//! with the row count, and what the alternatives on the same weights cost.
//!
//! Three things are compared, and which two depends on the family:
//!
//!   - **as dispatched**, which is what a verify window runs today: ggml's
//!     mat-vec for a k-quant at every width, and (with the window mode on)
//!     mlx's `qmv` for an affine module.
//!   - **the tile**, `kquant::qmm_t` -- the prefill kernel, which reads the
//!     matrix once whatever M is. A different association, so it is a bound
//!     on what a multi-row mat-vec could buy rather than a drop-in.
//!   - **mlx's own choice**, for an affine module with the window mode off:
//!     `qmv_wide` at 2..13 rows, which exists precisely to share one weight
//!     read between five rows.
//!
//! The `vs M=1` column is the one to read. A kernel that re-reads the whole
//! matrix per row is `M`; one that shares the read is flat.
//!
//! One thing this bin cannot show, and it is why the end-to-end sweep in
//! `draft_cost` is the measurement of record: sixteen repetitions of one
//! module leave its weights in cache, where a real forward reads each module
//! once and evicts it with the next. The ratios here are therefore the
//! optimistic end of the real ones.
//!
//! Run (GPU seconds, so `ORNITH_PROF=1`):
//!
//! ```text
//!   ORNITH_MODEL=~/.mtplx/models/ornith-1.5-35b-a3b-q6k-full \
//!     ORNITH_PROF=1 cargo run --release --bin window_mv
//! ```

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::model::cache::zero;
use ornith_rs::ops::kquant;
use ornith_rs::ops::linear::Linear;
use ornith_rs::refs::model_dir;
use ornith_rs::weights::WeightStore;

const REPS: usize = 16;
/// Three passes over the sweep, keeping the least of each cell. A single pass
/// on this machine throws 2-5x outliers that are the machine, not the kernel.
const PASSES: usize = 3;
const WIDTHS: [usize; 6] = [1, 2, 3, 4, 8, 16];

/// The modules a verify window reads, one of each shape that matters.
const DENSE: [&str; 4] = [
    "language_model.model.layers.0.linear_attn.in_proj_qkv",
    "language_model.model.layers.0.linear_attn.out_proj",
    "language_model.model.layers.3.self_attn.q_proj",
    "language_model.lm_head",
];

const ROUTED: [&str; 2] = [
    "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
    "language_model.model.layers.0.mlp.switch_mlp.down_proj",
];

fn gpu_ms(ctx: &Context, mut f: impl FnMut()) -> f64 {
    f();
    ctx.sync();
    ctx.prof_reset();
    for _ in 0..REPS {
        f();
        if serial() {
            ctx.sync();
        }
    }
    ctx.sync();
    let (_, _, gpu_s, _) = ctx.prof_totals();
    gpu_s * 1e3 / REPS as f64
}

/// `ORNITH_SERIAL=1`: drain the queue between repetitions.
///
/// The bin's own caveat, made measurable. Sixteen repetitions back to back
/// are sixteen *independent* dispatches, and the encoder runs them
/// concurrently -- so several copies of the weight read are in flight at once
/// and serve each other out of cache. A verify runs its mat-vecs in a
/// dependency chain, one at a time, each evicting the last. This mode is that
/// one, and the difference between the two columns is how much of a module's
/// re-read the cache was absorbing.
fn serial() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ORNITH_SERIAL").as_deref() == Ok("1"))
}

fn best(ctx: &Context, mut f: impl FnMut()) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..PASSES {
        b = b.min(gpu_ms(ctx, &mut f));
    }
    b
}

fn bytes_of(l: &Linear) -> f64 {
    match l {
        Linear::K(w) => (w.out_dims() * w.in_dims()) as f64 * w.kind.bits() / 8.0,
        _ => {
            let q = l.as_mlx();
            (q.w.numel() * 4
                + q.scales.numel() * 2
                + q.biases.as_ref().map_or(0, |b| b.numel()) * 2) as f64
        }
    }
}

fn family(l: &Linear) -> String {
    match l {
        Linear::K(k) => k.kind.name().to_string(),
        _ => "affine".to_string(),
    }
}

/// The other kernel on the same weights: the tile for a k-quant, mlx's own
/// dispatch (`qmv_wide` at these widths) for an affine module.
fn alternative(ctx: &Context, w: &Linear, x: &Array) -> (f64, &'static str) {
    match w {
        Linear::K(k) => (
            best(ctx, || {
                let _ = kquant::qmm_t(ctx, x, k);
            }),
            "tile",
        ),
        _ => (
            ornith_rs::ops::invariant::with(false, || {
                best(ctx, || {
                    let _ = w.matmul(ctx, x);
                })
            }),
            "mlx",
        ),
    }
}

fn main() {
    let dir = model_dir();
    let ctx = Context::new().expect("metal");
    assert!(ctx.prof_on(), "run with ORNITH_PROF=1");
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    println!("{}\n{}\n", dir.display(), ctx.device_name());

    println!("dense modules: x @ w.T at M rows, ms of GPU\n");
    for path in DENSE {
        if !ws.has(&format!("{path}.weight")) {
            println!("  {path}: absent");
            continue;
        }
        let w = ws.linear(&ctx, path);
        let mb = bytes_of(&w) / 1e6;
        println!("  {path} [{}], {mb:.1} MB", family(&w));
        let mut base = (0.0, 0.0);
        for (i, m) in WIDTHS.iter().enumerate() {
            let x = Array::empty(&ctx, &[*m, w.in_dims()], Dtype::BF16);
            zero(&ctx, &x);
            // The A/B this bin was reopened for, both halves in one process:
            // the token axis on the grid (a matrix read a row) against the
            // token axis inside the kernel (a matrix read once).
            kquant::set_qmv_rows(1);
            let t = best(&ctx, || {
                let _ = w.matmul(&ctx, &x);
            });
            kquant::set_qmv_rows(4);
            let inside = match &w {
                Linear::K(_) => best(&ctx, || {
                    let _ = w.matmul(&ctx, &x);
                }),
                _ => f64::NAN,
            };
            let (alt, alt_name) = alternative(&ctx, &w, &x);
            if i == 0 {
                base = (t, alt);
                println!(
                    "  {:>6} {:>13} {:>8} {:>13} {:>8} {:>13} {:>8}",
                    "M", "a row each", "vs M=1", "M inside", "vs M=1", alt_name, "vs M=1"
                );
            }
            println!(
                "  {m:>6} {t:>13.3} {:>8.2} {inside:>13.3} {:>8.2} {alt:>13.3} {:>8.2}",
                t / base.0,
                inside / base.0,
                alt / base.1
            );
        }
        println!();
    }

    println!("routed modules: 8 experts a token, so a window of M is 8M rows\n");
    for path in ROUTED {
        if !ws.has(&format!("{path}.weight")) {
            println!("  {path}: absent");
            continue;
        }
        let w = ws.linear(&ctx, path);
        let experts = w.num_experts();
        // `bytes_of` reads the per-expert shape, which is what one row costs.
        let per = bytes_of(&w) / 1e6;
        println!("  {path} [{}], {per:.2} MB an expert", family(&w));
        println!(
            "  {:>6} {:>9} {:>13} {:>8} {:>13} {:>8}",
            "rows", "distinct", "as dispatched", "vs 8", "tile", "vs 8"
        );
        let mut base = (0.0, 0.0);
        for (i, m) in WIDTHS.iter().enumerate() {
            // What a window of `m` tokens routes: 8 experts each, with an
            // overlap between adjacent tokens rather than the independence
            // the old note assumed (`tools/router_stats.py`).
            let rows = 8 * m;
            let mut idx: Vec<u32> = (0..8u32).collect();
            let mut next = 8u32;
            for t in 1..*m {
                // Three of the eight repeat the previous token's, which is
                // the overlap `tools/router_stats.py` measures (2.84 of 8).
                let prev: Vec<u32> = idx[(t - 1) * 8..t * 8].to_vec();
                for i in 0..8 {
                    if i < 3 {
                        idx.push(prev[i]);
                    } else {
                        idx.push(next % experts as u32);
                        next += 1;
                    }
                }
            }
            let mut uniq = idx.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let rhs = Array::from_slice(&ctx, &idx, &[rows], Dtype::U32);
            let lhs: Vec<u32> = (0..rows).map(|r| (r / 8) as u32).collect();
            let lhs_i = Array::from_slice(&ctx, &lhs, &[rows], Dtype::U32);
            let x = Array::empty(&ctx, &[*m, w.in_dims()], Dtype::BF16);
            zero(&ctx, &x);
            let t = best(&ctx, || {
                let _ = w.gather_matmul(&ctx, &x, &lhs_i, &rhs, false);
            });
            let alt = match &w {
                Linear::K(k) => best(&ctx, || {
                    let _ = kquant::gather_qmm_t(&ctx, &x, k, &lhs_i, &rhs);
                }),
                _ => f64::NAN,
            };
            if i == 0 {
                base = (t, alt);
            }
            println!(
                "  {rows:>6} {:>9} {t:>13.3} {:>8.2} {alt:>13.3} {:>8.2}",
                uniq.len(),
                t / base.0,
                alt / base.1
            );
        }
        println!();
    }
}
