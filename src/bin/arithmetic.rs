//! What this machine's arithmetic sustains, against `bin/bandwidth`'s 465 GB/s.
//!
//! `bandwidth` exists because "546 GB/s" is a number off a box and the
//! question was always "of what ceiling". After Phase 36 a prefill chunk is
//! **81% quantized gemm** -- 1264 ms of 1564 -- so every remaining question
//! about prefill is an arithmetic question, and the arithmetic side had no
//! such ceiling at all. This is that sibling.
//!
//! Three things, in the order that makes each one interpretable:
//!
//!   1. **the scalar ALUs**, swept over how many independent FMA chains a
//!      thread carries. A dependent chain measures FMA *latency*; the sweep
//!      is where that stops mattering, and its shape is the pipeline depth;
//!   2. **the matrix units**, swept over accumulators and over the fragment
//!      type. This is the ceiling that matters: `kquant_mm_tile` and mlx's
//!      steel gemm both spend their whole K loop in
//!      `simdgroup_multiply_accumulate`. The port's tile instantiates at
//!      `S = float` with `S = half` behind a flag that measured *free* on
//!      the real kernel -- if the two differ here and not there, the real
//!      kernel is not matrix-unit-bound, and that is a finding;
//!   3. **the real kernels at prefill shapes** against that ceiling, with a
//!      dense bf16 gemm at the identical shape beside them. The gap between
//!      the dense gemm and the quantized one at the same M, N, K is the
//!      price of the format, paid in arithmetic rather than in bytes.
//!
//! No model load: the weights are random bytes, because these kernels' time
//! depends on their shapes and not on their values (`bin/kquant_mm` makes the
//! same trade and says why). That also makes this the one performance bin
//! that can be run while the machine is busy with something else -- though
//! the numbers still should not be, see `HANDOFF.md`.
//!
//! Run: `cargo run --release --bin arithmetic`

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::kquant::{KMatrix, KQuant, QK_K, gather_qmm_t, qmm_t, set_mm_half, set_mm_rows};
use ornith_rs::ops::matmul::matmul;

/// Dispatches inside one sync window. `bin/kquant_mm`'s lesson: a single
/// dispatch between two syncs measures the ~120 us submit-and-wait round
/// trip, not the kernel.
const CHAIN: usize = 10;

fn best_of<F: FnMut()>(ctx: &Context, reps: usize, mut run: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        ctx.sync();
        let t0 = std::time::Instant::now();
        for _ in 0..CHAIN {
            run();
        }
        ctx.sync();
        best = best.min(t0.elapsed().as_secs_f64() / CHAIN as f64);
    }
    best
}

fn tflops(flop: f64, secs: f64) -> f64 {
    flop / secs / 1e12
}

// --- 1. the scalar ALUs ------------------------------------------------------

fn fma_sweep(ctx: &Context) -> f64 {
    const THREADS: usize = 1 << 18;
    const ITERS: u32 = 4000;
    println!("\nindependent FMA chains a thread, {THREADS} threads, {ITERS} iterations\n");
    println!("  {:>6} {:>10} {:>10}  {}", "chains", "ms", "TFLOP/s", "");
    let out = Array::empty(ctx, &[THREADS], Dtype::F32);
    let mut best = 0.0f64;
    for acc in [1usize, 2, 4, 8, 16, 32] {
        let secs = best_of(ctx, 5, || {
            ctx.kernel(&format!("fma_peak_{acc}"))
                .out(&out)
                .bytes(&ITERS)
                .bytes(&1.0001f32)
                .threads((THREADS, 1, 1), (256, 1, 1));
        });
        let flop = THREADS as f64 * ITERS as f64 * acc as f64 * 2.0;
        let t = tflops(flop, secs);
        best = best.max(t);
        println!(
            "  {acc:>6} {:>10.3} {t:>10.2}  {}",
            secs * 1e3,
            if acc == 1 { "<- latency, not throughput" } else { "" }
        );
    }
    println!("  the ALUs sustain {best:.2} TFLOP/s of scalar fp32 FMA");
    best
}

// --- 2. the matrix units -----------------------------------------------------

/// One `simdgroup_multiply_accumulate` on 8x8x8 is 512 MACs.
const MMA_FLOP: f64 = 8.0 * 8.0 * 8.0 * 2.0;

fn mma_sweep(ctx: &Context) -> f64 {
    const GROUPS: usize = 1024;
    const TG: usize = 128;
    const SIMDGROUPS: usize = GROUPS * (TG / 32);
    const ITERS: u32 = 1000;
    println!(
        "\nsimdgroup 8x8x8 matrix multiply-accumulate, {SIMDGROUPS} simdgroups, \
         {ITERS} iterations\n"
    );
    println!("  {:>10} {:>6} {:>10} {:>10}", "fragment", "acc", "ms", "TFLOP/s");
    let mut best = 0.0f64;
    let mut by_type: Vec<(&str, f64)> = Vec::new();
    for (name, tag) in [("float32", "float32"), ("float16", "float16")] {
        let mut ty_best = 0.0f64;
        for tiles in [1usize, 2, 4, 8, 16] {
            let out = Array::empty(ctx, &[SIMDGROUPS * tiles * 64], Dtype::F32);
            let secs = best_of(ctx, 5, || {
                ctx.kernel(&format!("mma_peak_{tag}_{tiles}"))
                    .out(&out)
                    .bytes(&ITERS)
                    .bytes(&1.0001f32)
                    .groups((GROUPS, 1, 1), (TG, 1, 1));
            });
            let flop = SIMDGROUPS as f64 * ITERS as f64 * tiles as f64 * MMA_FLOP;
            let t = tflops(flop, secs);
            ty_best = ty_best.max(t);
            println!("  {name:>10} {tiles:>6} {:>10.3} {t:>10.2}", secs * 1e3);
        }
        by_type.push((name, ty_best));
        best = best.max(ty_best);
    }
    // bfloat16 at one width, because that is what the activations are.
    {
        let tiles = 4usize;
        let out = Array::empty(ctx, &[SIMDGROUPS * tiles * 64], Dtype::F32);
        let secs = best_of(ctx, 5, || {
            ctx.kernel(&format!("mma_peak_bfloat16_{tiles}"))
                .out(&out)
                .bytes(&ITERS)
                .bytes(&1.0001f32)
                .groups((GROUPS, 1, 1), (TG, 1, 1));
        });
        let flop = SIMDGROUPS as f64 * ITERS as f64 * tiles as f64 * MMA_FLOP;
        println!("  {:>10} {tiles:>6} {:>10.3} {:>10.2}", "bfloat16", secs * 1e3, tflops(flop, secs));
    }
    for (name, t) in &by_type {
        println!("  {name} fragments sustain {t:.2} TFLOP/s");
    }
    println!(
        "  the port's tile instantiates at float32; half is {:.2}x here",
        by_type[1].1 / by_type[0].1.max(1e-9)
    );
    best
}

// --- 3. the real kernels -----------------------------------------------------

fn fake(ctx: &Context, kind: KQuant, o: usize, k: usize) -> KMatrix {
    let nb = k / QK_K;
    let (codes, scales, dm) = match kind {
        KQuant::Q4K => (128, 12, 2),
        KQuant::Q6K => (192, 16, 1),
    };
    let mut s: u32 = 0x9E37_79B9;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let qs: Vec<u32> = (0..o * nb * codes / 4).map(|_| next()).collect();
    let sc: Vec<u8> = (0..o * nb * scales).map(|_| next() as u8).collect();
    // Finite and small, or the timing measures denormal handling.
    let d: Vec<u16> = (0..o * nb * dm).map(|_| 0x1C00 | (next() as u16 & 0x03FF)).collect();
    KMatrix::new(
        kind,
        Array::from_slice(ctx, &qs, &[o, nb * codes / 4], Dtype::U32),
        Array::from_slice(ctx, &sc, &[o, nb, scales], Dtype::U8),
        Array::from_slice(ctx, &d, &[o, nb, dm], Dtype::F16),
    )
}

fn routed(ctx: &Context, kind: KQuant, e: usize, o: usize, k: usize) -> KMatrix {
    let w = fake(ctx, kind, e * o, k);
    KMatrix::new(
        kind,
        w.qs.reshape(&[e, o, w.qs.shape[1]]),
        w.scales.reshape(&[e, o, w.scales.shape[1], w.scales.shape[2]]),
        w.dm.reshape(&[e, o, w.dm.shape[1], w.dm.shape[2]]),
    )
}

fn real(ctx: &Context, ceiling: f64) {
    println!("\nthe kernels a prefill chunk is 81% made of, at a chunk's own shapes\n");
    println!(
        "  {:<28} {:>6} {:>6} {:>6} {:>9} {:>9} {:>7}",
        "kernel / shape", "M", "N", "K", "ms", "TFLOP/s", "of mma"
    );
    let m = 2048usize;
    // The four dense shapes the trunk actually has, and the MoE's two.
    for (kind, tag) in [(KQuant::Q4K, "q4k"), (KQuant::Q6K, "q6k")] {
        for (o, k, what) in [(8192usize, 2048usize, "in_proj_qkv"), (2048, 4096, "out_proj")] {
            let w = fake(ctx, kind, o, k);
            let x = Array::from_f32_as_bf16(ctx, &vec![0.01f32; m * k], &[m, k]);
            let secs = best_of(ctx, 5, || {
                let _ = qmm_t(ctx, &x, &w);
            });
            let flop = 2.0 * m as f64 * o as f64 * k as f64;
            let t = tflops(flop, secs);
            println!(
                "  {:<28} {m:>6} {o:>6} {k:>6} {:>9.3} {t:>9.2} {:>6.0}%",
                format!("{tag}_qmm_t {what}"),
                secs * 1e3,
                100.0 * t / ceiling
            );
        }
    }
    // The routed pair, at the row count a 2048-token chunk produces.
    {
        let rows = 16384usize;
        let (e, o, k) = (256usize, 512usize, 2048usize);
        let w = routed(ctx, KQuant::Q4K, e, o, k);
        let x = Array::from_f32_as_bf16(ctx, &vec![0.01f32; rows * k], &[rows, k]);
        // 64 rows an expert, which is what this model routes at prefill.
        let lhs: Vec<u32> = (0..rows as u32).collect();
        let rhs: Vec<u32> = (0..rows as u32).map(|r| (r / 64) % e as u32).collect();
        let lhs = Array::from_slice(ctx, &lhs, &[rows], Dtype::U32);
        let rhs = Array::from_slice(ctx, &rhs, &[rows], Dtype::U32);
        let secs = best_of(ctx, 5, || {
            let _ = gather_qmm_t(ctx, &x, &w, &lhs, &rhs);
        });
        let flop = 2.0 * rows as f64 * o as f64 * k as f64;
        let t = tflops(flop, secs);
        println!(
            "  {:<28} {rows:>6} {o:>6} {k:>6} {:>9.3} {t:>9.2} {:>6.0}%",
            "q4k_gather_qmm_t experts",
            secs * 1e3,
            100.0 * t / ceiling
        );
    }
    // And the same arithmetic with nothing to unpack, which is the format's
    // price stated as a ratio rather than as a guess.
    for (o, k, what) in [(8192usize, 2048usize, "in_proj_qkv"), (2048, 4096, "out_proj")] {
        let a = Array::from_f32_as_bf16(ctx, &vec![0.01f32; m * k], &[m, k]);
        let b = Array::from_f32_as_bf16(ctx, &vec![0.01f32; o * k], &[o, k]);
        let secs = best_of(ctx, 5, || {
            let _ = matmul(ctx, &a, &b, true);
        });
        let flop = 2.0 * m as f64 * o as f64 * k as f64;
        let t = tflops(flop, secs);
        println!(
            "  {:<28} {m:>6} {o:>6} {k:>6} {:>9.3} {t:>9.2} {:>6.0}%",
            format!("dense bf16 {what}"),
            secs * 1e3,
            100.0 * t / ceiling
        );
    }
}

fn main() {
    let ctx = Context::new().unwrap();
    println!("{}", ctx.device_name());
    let alu = fma_sweep(&ctx);
    let mma = mma_sweep(&ctx);
    println!("\n  matrix units are {:.1}x the scalar ALUs here", mma / alu.max(1e-9));
    real(&ctx, mma);

    // The flag the tile carries, on the real kernel, beside the synthetic
    // ratio above. If the matrix unit prefers half and the kernel does not,
    // the kernel is bound by something else.
    // Pinned to the 32-row tile in *both* arms: `half` is instantiated at
    // that width only, and leaving the default 64 would compare two widths
    // while claiming to compare two scalars -- or, here, look up a kernel
    // that does not exist.
    println!("\nthe tile's own fragment type, on the real kernel (32-row tile, both arms)");
    set_mm_rows(32);
    let m = 2048usize;
    let (o, k) = (8192usize, 2048usize);
    let w = fake(&ctx, KQuant::Q4K, o, k);
    let x = Array::from_f32_as_bf16(&ctx, &vec![0.01f32; m * k], &[m, k]);
    for half in [false, true] {
        set_mm_half(half);
        let secs = best_of(&ctx, 5, || {
            let _ = qmm_t(&ctx, &x, &w);
        });
        let flop = 2.0 * m as f64 * o as f64 * k as f64;
        println!(
            "  S = {:<8} {:>9.3} ms  {:>7.2} TFLOP/s",
            if half { "half" } else { "float" },
            secs * 1e3,
            tflops(flop, secs)
        );
    }
    set_mm_half(false);
    set_mm_rows(32);
}
