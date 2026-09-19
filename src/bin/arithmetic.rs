//! What this machine's arithmetic sustains, against `bin/bandwidth`'s 465 GB/s.
//!
//! `bandwidth` exists because "546 GB/s" is a number off a box and the
//! question was always "of what ceiling". After Phase 36 a prefill chunk is
//! **81% quantized gemm** -- 1264 ms of 1564 -- so every remaining question
//! about prefill is an arithmetic question, and the arithmetic side had no
//! such ceiling at all. This is that sibling.
//!
//! Six things, in the order that makes each one interpretable -- the first
//! three are the ceiling, the last three are what it was built to ask:
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
//!      dense bf16 gemm at the identical shape beside them;
//!   4. **the routed gemm's eight points**, one term at a time -- `N`,
//!      routing, the weight footprint, the tile width -- because a ratio
//!      between two cells that differ in four ways is evidence about none
//!      of them;
//!   5. **what the unpack itself costs**, against the same tile with only
//!      its arithmetic removed (`Q4KBareFmt`) and against mlx's own
//!      quantized gemm. The gap in (3) is *not* the price of the format,
//!      and this is what says so;
//!   6. **the elementwise kernels' access width**, measured here rather
//!      than at a chunk because the chunk cell's own noise is larger than
//!      the effect.
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
use ornith_rs::ops::kquant::{
    KMatrix, KQuant, QK_K, gather_qmm_t, qmm_t, qmm_t_bare, set_mm_half, set_mm_rows,
    set_mm_rows_id,
};
use ornith_rs::ops::elementwise::{set_wide_elem, sigmoid_gate, swiglu};
use ornith_rs::ops::layout::{slice_inner, transpose_hl, transpose_lh, transpose_would_widen};
use ornith_rs::ops::matmul::matmul;
use ornith_rs::ops::quantized::{QuantMatrix, QuantMode, quantized_matmul};

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

// --- 4. why the routed gemm is eight points down -----------------------------

/// The routed expert gemm reads 72% of the matrix ceiling where its own
/// family reads 80%, and it is 632 ms of a 1264 ms chunk -- the largest
/// single object left in prefill. **But the two cells differ in four things
/// at once**, which is the eighth axis of `measure-the-same-object` exactly:
///
///   * routing -- one kernel walks `rhs_indices` and re-runs its K loop per
///     run of one expert, the other does not;
///   * `N` -- 512 for the experts against 2048-8192 for the dense shapes, so
///     a quarter as many column tiles reuse each loaded A fragment;
///   * the **weight footprint** -- at 256 experts the routed call touches
///     134 MB of weights for the same flops a dense call does with 0.5 MB,
///     because each expert's rows are 64 of 16384 and nothing is re-read
///     from cache;
///   * the **tile width** -- `mm_rows` is 64 and `mm_rows_id` is 32.
///
/// A ratio between two cells that differ in four ways is not evidence about
/// any one of them. So: `M` and `K` fixed, the tile pinned to 32 in every
/// arm, and one term moved at a time. The `experts = 1` row is the isolate
/// that matters -- all the routing machinery, a dense weight footprint.
fn decompose_routed(ctx: &Context, ceiling: f64) {
    const M: usize = 16384;
    const K: usize = 2048;
    let was_rows = 64usize;
    set_mm_rows(32);
    set_mm_rows_id(32);
    println!(
        "\nthe routed gemm's eight points, one term at a time \
         (M={M}, K={K}, 32-row tile in every arm)\n"
    );
    let x = Array::from_f32_as_bf16(ctx, &vec![0.01f32; M * K], &[M, K]);

    println!("  {:<34} {:>6} {:>9} {:>9} {:>7}", "arm", "N", "wt MB", "TFLOP/s", "of mma");
    // (a) N alone, with no routing at all.
    for n in [512usize, 1024, 2048] {
        let w = fake(ctx, KQuant::Q4K, n, K);
        let secs = best_of(ctx, 5, || {
            let _ = qmm_t(ctx, &x, &w);
        });
        let flop = 2.0 * M as f64 * n as f64 * K as f64;
        let t = tflops(flop, secs);
        let mb = (n * (K / QK_K) * 128) as f64 / 1e6;
        println!(
            "  {:<34} {n:>6} {mb:>9.1} {t:>9.2} {:>6.0}%",
            "dense, no routing",
            100.0 * t / ceiling
        );
    }
    // (b) routing and the weight footprint, at one N. Same flops, same tile,
    //     same shape in every row; only how many experts the rows spread over
    //     -- and therefore how much of the weight set is touched -- moves.
    for e in [1usize, 16, 64, 256] {
        let n = 512usize;
        let run = M / e;
        let w = routed(ctx, KQuant::Q4K, e, n, K);
        let lhs: Vec<u32> = (0..M as u32).collect();
        let rhs: Vec<u32> = (0..M as u32).map(|r| (r / run as u32) % e as u32).collect();
        let lhs = Array::from_slice(ctx, &lhs, &[M], Dtype::U32);
        let rhs = Array::from_slice(ctx, &rhs, &[M], Dtype::U32);
        let secs = best_of(ctx, 5, || {
            let _ = gather_qmm_t(ctx, &x, &w, &lhs, &rhs);
        });
        let flop = 2.0 * M as f64 * n as f64 * K as f64;
        let t = tflops(flop, secs);
        let mb = (e * n * (K / QK_K) * 128) as f64 / 1e6;
        println!(
            "  {:<34} {n:>6} {mb:>9.1} {t:>9.2} {:>6.0}%",
            format!("routed, {e} expert{} of {run}-row runs", if e == 1 { "" } else { "s" }),
            100.0 * t / ceiling
        );
    }
    // And the one term every row above held fixed, which is therefore the
    // only one left: the tile width. The dense gemms that read 80% earlier
    // ran at `mm_rows` = 64; every arm above ran at 32 because that is what
    // `mm_rows_id` gives the routed path.
    println!("\n  and the term all seven rows above held constant\n");
    println!("  {:<34} {:>6} {:>9} {:>9} {:>7}", "arm", "tile", "wt MB", "TFLOP/s", "of mma");
    for rows in [16usize, 32, 64] {
        set_mm_rows(rows);
        let n = 2048usize;
        let w = fake(ctx, KQuant::Q4K, n, K);
        let secs = best_of(ctx, 5, || {
            let _ = qmm_t(ctx, &x, &w);
        });
        let flop = 2.0 * M as f64 * n as f64 * K as f64;
        let t = tflops(flop, secs);
        println!(
            "  {:<34} {rows:>6} {:>9.1} {t:>9.2} {:>6.0}%",
            "dense, N=2048",
            (n * (K / QK_K) * 128) as f64 / 1e6,
            100.0 * t / ceiling
        );
    }
    set_mm_rows(32);
    {
        let (e, n, run) = (256usize, 512usize, 64usize);
        let w = routed(ctx, KQuant::Q4K, e, n, K);
        let lhs: Vec<u32> = (0..M as u32).collect();
        let rhs: Vec<u32> = (0..M as u32).map(|r| (r / run as u32) % e as u32).collect();
        let lhs = Array::from_slice(ctx, &lhs, &[M], Dtype::U32);
        let rhs = Array::from_slice(ctx, &rhs, &[M], Dtype::U32);
        for rows in [16usize, 32, 64] {
            set_mm_rows_id(rows);
            let secs = best_of(ctx, 5, || {
                let _ = gather_qmm_t(ctx, &x, &w, &lhs, &rhs);
            });
            let flop = 2.0 * M as f64 * n as f64 * K as f64;
            let t = tflops(flop, secs);
            println!(
                "  {:<34} {rows:>6} {:>9.1} {t:>9.2} {:>6.0}%",
                "routed, 256 experts, 64-row runs",
                (e * n * (K / QK_K) * 128) as f64 / 1e6,
                100.0 * t / ceiling
            );
        }
    }
    set_mm_rows(was_rows);
    set_mm_rows_id(32);
}

// --- 5. what the unpack itself costs, which is not the gap to mlx's gemm -----

/// The gap between the k-quant tile (79%) and mlx's dense bf16 gemm (96%) is
/// **not** the price of unpacking 4-bit weights, however tempting it is to
/// read it that way. Those are two kernels that differ in their tiling,
/// their loaders and their threadgroup shape as well as in whether they
/// unpack anything -- the eighth axis, and this bin walked into it once
/// already with the routed gemm's eight points.
///
/// The control is the same tile with only the arithmetic gone
/// (`Q4KBareFmt`): same loads, same registers, same everything else. What
/// the difference prices is `scale_min`, the `d / 16`, and the mask,
/// multiply and subtract a weight.
///
/// The static count says ~3 ALU ops a weight against `NR1` MACs a weight, so
/// with matrix and scalar throughput equal on this machine (14.5 against
/// 14.1 TFLOP/s) the arithmetic predicts **3/64 = 4.7%** at the 64-row tile
/// and 9.4% at 32. A prediction from an op count is a hypothesis; this is
/// the A/B.
fn unpack_cost(ctx: &Context, ceiling: f64) {
    const M: usize = 16384;
    const K: usize = 2048;
    const N: usize = 2048;
    println!("\nwhat unpacking a 4-bit weight costs, against the same tile without it\n");
    println!(
        "  {:<30} {:>6} {:>9} {:>9} {:>7} {:>9}",
        "arm", "tile", "ms", "TFLOP/s", "of mma", "vs bare"
    );
    let x = Array::from_f32_as_bf16(ctx, &vec![0.01f32; M * K], &[M, K]);
    let w = fake(ctx, KQuant::Q4K, N, K);
    let flop = 2.0 * M as f64 * N as f64 * K as f64;
    for rows in [32usize, 64] {
        set_mm_rows(rows);
        let bare = best_of(ctx, 5, || {
            let _ = qmm_t_bare(ctx, &x, &w);
        });
        let real_t = best_of(ctx, 5, || {
            let _ = qmm_t(ctx, &x, &w);
        });
        for (name, secs) in [("no unpack (wrong answers)", bare), ("Q4KFmt, as it ships", real_t)] {
            let t = tflops(flop, secs);
            println!(
                "  {name:<30} {rows:>6} {:>9.3} {t:>9.2} {:>6.0}% {:>8.1}%",
                secs * 1e3,
                100.0 * t / ceiling,
                100.0 * (secs / bare - 1.0)
            );
        }
    }
    set_mm_rows(64);

    // And mlx's dense bf16 gemm at **exactly** this M, N and K, because the
    // earlier "96% against 79%" row was taken at another shape entirely
    // (M=2048, N=8192) and comparing across it is the mistake this whole
    // section exists to stop making.
    {
        let a = Array::from_f32_as_bf16(ctx, &vec![0.01f32; M * K], &[M, K]);
        let b = Array::from_f32_as_bf16(ctx, &vec![0.01f32; N * K], &[N, K]);
        let secs = best_of(ctx, 5, || {
            let _ = matmul(ctx, &a, &b, true);
        });
        let t = tflops(flop, secs);
        println!(
            "  {:<30} {:>6} {:>9.3} {t:>9.2} {:>6.0}%",
            "mlx steel gemm, dense bf16", "-", secs * 1e3, 100.0 * t / ceiling
        );
    }

    // And the arm that decides whether the 16% between the two is "what a
    // quantized gemm costs" or "what *this* tile costs": mlx's own affine
    // quantized gemm, same M, N and K, same 4 bits and group of 32 the
    // experts use. It unpacks too -- but on steel's `BlockMMA` tiling
    // rather than ggml's.
    {
        let mut st: u32 = 0x1234_5678;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st
        };
        let groups = K / 32;
        let packed: Vec<u32> = (0..N * K / 8).map(|_| next()).collect();
        let sc: Vec<f32> = (0..N * groups).map(|_| 0.01).collect();
        let bi: Vec<f32> = (0..N * groups).map(|_| -0.08).collect();
        let qw = QuantMatrix {
            w: Array::from_slice(ctx, &packed, &[N, K / 8], Dtype::U32),
            scales: Array::from_f32_as_bf16(ctx, &sc, &[N, groups]),
            biases: Some(Array::from_f32_as_bf16(ctx, &bi, &[N, groups])),
            mode: QuantMode::Affine,
            group_size: 32,
            bits: 4,
        };
        let secs = best_of(ctx, 5, || {
            let _ = quantized_matmul(ctx, &x, &qw);
        });
        let t = tflops(flop, secs);
        println!(
            "  {:<30} {:>6} {:>9.3} {t:>9.2} {:>6.0}%",
            "mlx affine qmm, 4-bit g32", "-", secs * 1e3, 100.0 * t / ceiling
        );

        // **And the same kernel with float operands, which is the term a
        // port of steel's tiling onto the k-quants could not take.**
        //
        // mlx's quantized gemm dequantizes into a threadgroup tile of `T`,
        // and `T` is the activation type -- so on this model every weight
        // reaches the multiplier rounded to **bfloat16**, eight significand
        // bits. `kquant_mm.h` declined `half`'s eleven for exactly this
        // reason (a 5e-4 perturbation where reassociation is 2e-6), and
        // eight is 8x worse again. So "carry the k-quants onto steel's
        // tiling" means either giving that contract up or instantiating the
        // tile at `float`, and this row is what the second one costs on
        // steel's own tiling rather than on ggml's.
        //
        // **It is an upper bound, not the term itself**: `T` widens the
        // activation tile and the output as well as the weight tile, where
        // a k-quant port would widen only the weights. The lower bound is
        // the other end of the same question, measured on the tile we have
        // -- `ORNITH_KQUANT_TILE_HALF`, at the bottom of this bin.
        let xf = Array::from_f32(ctx, &vec![0.01f32; M * K], &[M, K]);
        let qwf = QuantMatrix {
            w: Array::from_slice(ctx, &packed, &[N, K / 8], Dtype::U32),
            scales: Array::from_f32(ctx, &sc, &[N, groups]),
            biases: Some(Array::from_f32(ctx, &bi, &[N, groups])),
            mode: QuantMode::Affine,
            group_size: 32,
            bits: 4,
        };
        let secs_f = best_of(ctx, 5, || {
            let _ = quantized_matmul(ctx, &xf, &qwf);
        });
        let tf = tflops(flop, secs_f);
        println!(
            "  {:<30} {:>6} {:>9.3} {tf:>9.2} {:>6.0}%  {:+.1}% vs bf16",
            "  ... the same, float operands",
            "-",
            secs_f * 1e3,
            100.0 * tf / ceiling,
            100.0 * (secs_f / secs - 1.0)
        );
    }
}

// --- 6. the elementwise kernels' access width -------------------------------

/// The two elementwise kernels Phase 37's inventory still reads far from the
/// memory ceiling -- `swiglu` at 106 GB/s over 21.4 ms a chunk,
/// `sigmoid_gate` at 105 over 4.8 -- against `add_rms` at 425 and
/// `block_softmax` at 404, which are done.
///
/// **Measured here and not in the chunk cell, and that is the finding about
/// the instrument.** The effect is ~1% of a chunk; `step_inventory prefill
/// elem`'s own control-against-itself row reads +12 ms of noise on a
/// ~1200 ms chunk and the base drifted from 1.1 s to 2.0 s inside the cell,
/// so it cannot resolve this and said so. A kernel timed on its own with a
/// best-of can: nothing here competes for DRAM with anything else, and the
/// minimum of five draws is robust to a busy machine in a way a paired
/// chunk is not.
fn elementwise_width(ctx: &Context) {
    println!("\nthe elementwise kernels, sixteen bytes a thread and one element\n");
    println!("  {:<28} {:>10} {:>10} {:>9} {:>8}", "kernel / shape", "narrow ms", "wide ms", "GB/s", "saved");
    // The shapes a prefill chunk actually dispatches them at: the experts'
    // activation, the shared expert's, and the attention output gate.
    let cases: [(&str, usize, usize, bool); 3] = [
        ("swiglu, experts", 16384, 512, false),
        ("swiglu, shared expert", 2048, 512, false),
        ("sigmoid_gate, output gate", 2048, 4096, true),
    ];
    let mut total_saved = 0.0f64;
    for (name, rows, dim, gated) in cases {
        let n = rows * dim;
        let a = Array::from_f32_as_bf16(ctx, &vec![0.3f32; n], &[rows, dim]);
        let b = Array::from_f32_as_bf16(ctx, &vec![0.7f32; n], &[rows, dim]);
        let g = Array::from_f32_as_bf16(ctx, &vec![0.2f32; rows], &[rows]);
        let mut t = [0.0f64; 2];
        for (i, on) in [false, true].into_iter().enumerate() {
            set_wide_elem(on);
            t[i] = best_of(ctx, 5, || {
                let _ = if gated { sigmoid_gate(ctx, &g, &b) } else { swiglu(ctx, &a, &b) };
            });
        }
        set_wide_elem(true);
        // Three arrays of `n` bf16 for swiglu, two and a row for the gate.
        let bytes = if gated { 2.0 * n as f64 * 2.0 } else { 3.0 * n as f64 * 2.0 };
        let count = if gated { 10.0 } else if rows == 16384 { 40.0 } else { 40.0 };
        let saved = (t[0] - t[1]) * 1e3 * count;
        total_saved += saved;
        println!(
            "  {name:<28} {:>10.3} {:>10.3} {:>9.0} {:>7.1} ms",
            t[0] * 1e3,
            t[1] * 1e3,
            bytes / t[1] / 1e9,
            saved
        );
    }
    // And the slice, which is the same copy with a stride. Its shape is the
    // gated-delta layers' q/gate split: 4096 of a 8192-wide row.
    {
        let (rows, src_row, len) = (2048usize, 8192usize, 4096usize);
        let x = Array::from_f32_as_bf16(ctx, &vec![0.3f32; rows * src_row], &[rows, src_row]);
        let mut t = [0.0f64; 2];
        for (i, on) in [false, true].into_iter().enumerate() {
            set_wide_elem(on);
            t[i] = best_of(ctx, 5, || {
                let _ = slice_inner(ctx, &x, 0, len);
            });
        }
        set_wide_elem(true);
        let bytes = 2.0 * (rows * len) as f64 * 2.0;
        let saved = (t[0] - t[1]) * 1e3 * 20.0;
        total_saved += saved;
        println!(
            "  {:<28} {:>10.3} {:>10.3} {:>9.0} {:>7.1} ms",
            "slice_inner, q/gate split",
            t[0] * 1e3,
            t[1] * 1e3,
            bytes / t[1] / 1e9,
            saved
        );
    }
    println!(
        "  {:<28} {:>32} {:>7.1} ms a chunk, isolated",
        "so, at a chunk's dispatch counts", "", total_saved
    );
}

/// The last two kernels the inventory had under the memory ceiling: the
/// head/sequence transposes, at 67.5 and 54.2 GB/s over 12.4 ms a chunk.
///
/// **They were priced as a tiled kernel and they are not one.** A transpose
/// whose innermost axis is contiguous on both sides is a permutation of
/// runs, not a strided copy -- 512 bytes a run at this model's head shape --
/// so the 32x32 threadgroup tile a transpose usually wants would stage a
/// coalesced read through shared memory for nothing. What these were losing
/// is what `swiglu`, `sigmoid_gate` and `slice_inner` were losing one phase
/// earlier, and it comes back the same way: one vector a thread.
///
/// Same instrument as `elementwise_width`, and for the same reason -- the
/// effect is under 1% of a chunk and the chunk cell's noise floor is larger
/// than that.
fn transpose_width(ctx: &Context) {
    println!("\nthe transposes, sixteen bytes a thread and one element\n");
    println!(
        "  {:<28} {:>10} {:>10} {:>9} {:>8}",
        "kernel / shape", "narrow ms", "wide ms", "GB/s", "saved"
    );
    // A full-attention layer's own four: q, k, v in, the output out. Ten
    // layers of them in a 2048-token chunk, which is the inventory's 30 and
    // 10 dispatches.
    let seq = 2048usize;
    let hd = 256usize;
    let cases: [(&str, bool, usize); 4] = [
        ("transpose_lh, 16 q heads", true, 16),
        ("transpose_lh, 2 k heads", true, 2),
        ("transpose_lh, 2 v heads", true, 2),
        ("transpose_hl, 16 heads out", false, 16),
    ];
    let mut total_saved = 0.0f64;
    for (name, lh, heads) in cases {
        let n = seq * heads * hd;
        let shape = if lh { [1, seq, heads, hd] } else { [1, heads, seq, hd] };
        let x = Array::from_f32_as_bf16(ctx, &vec![0.3f32; n], &shape);
        assert!(transpose_would_widen(&x), "{name}: the wide arm is not live");
        let mut t = [0.0f64; 2];
        for (i, on) in [false, true].into_iter().enumerate() {
            set_wide_elem(on);
            t[i] = best_of(ctx, 5, || {
                let _ = if lh { transpose_lh(ctx, &x) } else { transpose_hl(ctx, &x) };
            });
        }
        set_wide_elem(true);
        // Read and write, once each.
        let bytes = 2.0 * n as f64 * 2.0;
        let saved = (t[0] - t[1]) * 1e3 * 10.0;
        total_saved += saved;
        println!(
            "  {name:<28} {:>10.3} {:>10.3} {:>9.0} {:>7.1} ms",
            t[0] * 1e3,
            t[1] * 1e3,
            bytes / t[1] / 1e9,
            saved
        );
    }
    println!(
        "  {:<28} {:>32} {:>7.1} ms a chunk, isolated",
        "so, over ten full-attention layers", "", total_saved
    );
}

fn main() {
    let ctx = Context::new().unwrap();
    println!("{}", ctx.device_name());
    let alu = fma_sweep(&ctx);
    let mma = mma_sweep(&ctx);
    println!("\n  matrix units are {:.1}x the scalar ALUs here", mma / alu.max(1e-9));
    real(&ctx, mma);
    decompose_routed(&ctx, mma);
    unpack_cost(&ctx, mma);
    elementwise_width(&ctx);
    transpose_width(&ctx);

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
