//! What one dispatch costs, with no useful work in it.
//!
//! A decode step here is ~1759 dispatches long, and the model only has to read
//! ~1.9 GB of weights to produce a token -- which at this machine's 546 GB/s is
//! ~3.4 ms, against a measured 12.2 ms. This measures the other term: the fixed
//! cost of encoding and running a dispatch, and of the barrier Metal puts
//! between two of them in a serial encoder.
//!
//! Run: `ORNITH_PROF=1 cargo run --release --bin dispatch_cost`

use ornith_rs::array::Array;
use ornith_rs::metal::Context;

/// Time `n` back-to-back dispatches of a kernel that touches `elems`
/// elements, in command buffers of `per_commit` dispatches.
fn run(ctx: &Context, n: usize, elems: usize, per_commit: usize) -> (f64, f64) {
    let x = Array::from_f32(ctx, &vec![1.0f32; elems], &[elems]);
    let y = Array::empty(ctx, &[elems], x.dtype);
    let tg = elems.min(1024);
    // Warm the pipeline cache, then let the queue drain.
    ctx.kernel("sampler_exp_float32").arr(&x).arr(&y).threads((elems, 1, 1), (tg, 1, 1));
    ctx.sync();
    ctx.prof_reset();

    let t0 = std::time::Instant::now();
    for i in 0..n {
        ctx.kernel("sampler_exp_float32").arr(&x).arr(&y).threads((elems, 1, 1), (tg, 1, 1));
        if (i + 1) % per_commit == 0 {
            ctx.flush();
        }
    }
    ctx.sync();
    let wall = t0.elapsed().as_secs_f64();
    let (_, _, gpu, _) = ctx.prof_totals();
    (wall / n as f64 * 1e6, gpu / n as f64 * 1e6)
}

fn main() {
    let ctx = Context::new().expect("metal");
    assert!(ctx.prof_on(), "run with ORNITH_PROF=1 to get the GPU-side number");
    println!("{}, {}", ctx.device_name(), ctx.arch);
    println!("\n  elems  per-commit   wall us/dispatch   gpu us/dispatch");
    for &elems in &[256usize, 2048, 16384] {
        for &per_commit in &[128usize, 1024] {
            let (wall, gpu) = run(&ctx, 20_000, elems, per_commit);
            println!("{elems:7}  {per_commit:10}   {wall:16.3}   {gpu:15.3}");
        }
    }
}
