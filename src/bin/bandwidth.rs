//! What this machine actually sustains, against the 546 GB/s on the box.
//!
//! A decode step reads ~2.83 GB of weights and takes ~11.3 ms, which is
//! ~250 GB/s -- 46% of the peak. That number alone cannot say whether the
//! other 54% is headroom or fiction, so this measures two things:
//!
//!   1. a kernel that only reads, at several sizes: the practical ceiling;
//!   2. the real `quantized_matmul` on real weights at M = 1: what the kernel
//!      that spends 38% of a decode step actually achieves;
//!   3. the same matmul across M = 1..8, which is what a speculative verify
//!      window costs. mlx-dspark's `small_m_qmm.py` says `quantized_matmul`
//!      grows ~linearly in M there -- the weight read re-paid per row until
//!      the GEMM tiling takes over around M >= 13 -- and a window of
//!      `depth + 1` tokens lives exactly in that range. This measures whether
//!      that holds here, because it prices every speculative depth.
//!
//! The gap between the two is ours to close. The gap between the first and
//! 546 belongs to the machine.
//!
//! Run: `cargo run --release --bin bandwidth`

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::model::cache::zero;
use ornith_rs::ops::quantized::{QuantMatrix, quantized_matmul};
use ornith_rs::refs::model_dir;
use ornith_rs::weights::WeightStore;

const PEAK: f64 = 546.0;

fn best_of<F: FnMut()>(ctx: &Context, reps: usize, mut run: F) -> f64 {
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

/// Read `bytes` and do nothing with them.
fn stream(ctx: &Context, bytes: usize) -> f64 {
    let n4 = bytes / 16;
    let x = Array::empty(ctx, &[n4 * 4], Dtype::U32);
    zero(ctx, &x);
    let threads = 1usize << 18;
    let out = Array::empty(ctx, &[threads], Dtype::U32);
    ctx.sync();

    let secs = best_of(ctx, 5, || {
        ctx.kernel("stream_read")
            .arr(&x)
            .out(&out)
            .bytes(&(n4 as u32))
            .threads((threads, 1, 1), (256, 1, 1));
    });
    bytes as f64 / secs / 1e9
}

/// Bytes a `quantized_matmul` has to read: the packed weights plus the scales
/// and the biases, which live in their own arrays.
fn matrix_bytes(qw: &QuantMatrix) -> usize {
    qw.w.numel() * qw.w.dtype.size()
        + qw.scales.numel() * qw.scales.dtype.size()
        + qw.biases.numel() * qw.biases.dtype.size()
}

fn main() {
    let ctx = Context::new().expect("metal");
    println!("{}, {} -- peak {PEAK} GB/s\n", ctx.device_name(), ctx.arch);

    println!("a kernel that only reads:");
    println!("  {:>8}  {:>10}  {:>8}", "MB", "GB/s", "of peak");
    for mb in [64usize, 256, 1024, 2048] {
        let gbs = stream(&ctx, mb << 20);
        println!("  {mb:>8}  {gbs:>10.1}  {:>7.0}%", 100.0 * gbs / PEAK);
    }

    let dir = model_dir();
    if !dir.exists() {
        eprintln!("\nmodel not present, skipping the quantized_matmul half");
        return;
    }
    let mut ws = WeightStore::open(&dir, &[]).expect("weights");
    println!("\nquantized_matmul at M = 1, on real weights:");
    println!("  {:>28}  {:>8}  {:>10}  {:>8}", "matrix", "MB", "GB/s", "of peak");
    for (name, path) in [
        ("lm_head 8-bit g64", "language_model.lm_head"),
        (
            "in_proj_qkv 8-bit g64",
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
        ),
        (
            "o_proj 8-bit g64",
            "language_model.model.layers.3.self_attn.o_proj",
        ),
    ] {
        let qw = ws.quant_matrix(&ctx, path);
        let bytes = matrix_bytes(&qw);
        let x = Array::empty(&ctx, &[1, qw.in_dims()], Dtype::BF16);
        zero(&ctx, &x);
        ctx.sync();
        let secs = best_of(&ctx, 20, || {
            let _ = quantized_matmul(&ctx, &x, &qw);
        });
        let gbs = bytes as f64 / secs / 1e9;
        println!(
            "  {name:>28}  {:>8.1}  {gbs:>10.1}  {:>7.0}%",
            bytes as f64 / 1e6,
            100.0 * gbs / PEAK
        );
    }

    // The verify window's price. `x1` is what one row costs; a kernel that
    // amortized the weight read across the window would hold that, and one
    // that re-pays it per row would climb with M.
    const MS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    println!("\nthe same matmuls across the verify window's M, as a multiple of M = 1:");
    print!("  {:>28}", "matrix");
    for m in MS {
        print!("  {:>6}", format!("M={m}"));
    }
    println!("  {:>9}", "us at M=1");
    for (name, path) in [
        ("lm_head 8-bit g64", "language_model.lm_head"),
        (
            "in_proj_qkv 8-bit g64",
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
        ),
        (
            "gate_up 4-bit g32 (shared)",
            "language_model.model.layers.0.mlp.shared_expert.gate_proj",
        ),
    ] {
        let qw = ws.quant_matrix(&ctx, path);
        print!("  {name:>28}");
        let mut base = 0.0;
        for m in MS {
            let x = Array::empty(&ctx, &[m, qw.in_dims()], Dtype::BF16);
            zero(&ctx, &x);
            ctx.sync();
            let secs = best_of(&ctx, 20, || {
                let _ = quantized_matmul(&ctx, &x, &qw);
            });
            if m == 1 {
                base = secs;
            }
            print!("  {:>6.2}", secs / base);
        }
        println!("  {:>9.0}", base * 1e6);
    }
}
