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

/// What the CPU can pull from the same memory, one pass over `bytes`.
///
/// Plain summed reads across `threads` P-cores. Not a matmul and not meant to
/// be one: the question is only whether the memory system has capacity left
/// when the GPU is already reading, which is what decides whether handing the
/// CPU a share of the work could ever add throughput rather than take it.
fn cpu_stream(bytes: usize, threads: usize) -> f64 {
    let n = bytes / 8;
    let data: Vec<u64> = (0..n as u64).collect();
    let chunk = n.div_ceil(threads);
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        for part in data.chunks(chunk) {
            s.spawn(move || {
                let mut acc = 0u64;
                for v in part {
                    acc = acc.wrapping_add(*v);
                }
                std::hint::black_box(acc);
            });
        }
    });
    bytes as f64 / t0.elapsed().as_secs_f64() / 1e9
}

/// The GPU read again, but as a prepared closure so it can be timed while the
/// CPU is reading too.
fn gpu_reader(ctx: &Context, bytes: usize) -> impl Fn() + '_ {
    let n4 = bytes / 16;
    let x = Array::empty(ctx, &[n4 * 4], Dtype::U32);
    zero(ctx, &x);
    let threads = 1usize << 18;
    let out = Array::empty(ctx, &[threads], Dtype::U32);
    ctx.sync();
    move || {
        ctx.kernel("stream_read")
            .arr(&x)
            .out(&out)
            .bytes(&(n4 as u32))
            .threads((threads, 1, 1), (256, 1, 1));
        ctx.sync();
    }
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

    // Can a second engine add anything, or is the memory system the wall?
    // A decode step gets ~250 GB/s of a possible 465, so the bus is not full
    // and the question is fair. If CPU and GPU together beat the GPU alone,
    // handing the CPU a share of the work has somewhere to put it; if the two
    // only divide one budget, the idea is dead before any kernel is written.
    const MB: usize = 1024;
    let cores = 12; // this part's P-cores
    println!("\nthe CPU on the same memory, and both at once ({MB} MB each):");
    let gpu_alone = {
        let run = gpu_reader(&ctx, MB << 20);
        let t0 = std::time::Instant::now();
        run();
        (MB << 20) as f64 / t0.elapsed().as_secs_f64() / 1e9
    };
    let cpu_alone = cpu_stream(MB << 20, cores);
    let (gpu_both, cpu_both) = {
        let run = gpu_reader(&ctx, MB << 20);
        std::thread::scope(|s| {
            let h = s.spawn(|| cpu_stream(MB << 20, cores));
            let t0 = std::time::Instant::now();
            run();
            let g = (MB << 20) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            (g, h.join().unwrap())
        })
    };
    println!("  {:>28}  {:>10.1}", "GPU alone", gpu_alone);
    println!("  {:>28}  {:>10.1}", "CPU alone (12 threads)", cpu_alone);
    println!("  {:>28}  {:>10.1}", "GPU while the CPU reads", gpu_both);
    println!("  {:>28}  {:>10.1}", "CPU while the GPU reads", cpu_both);
    println!(
        "  {:>28}  {:>10.1}  against {:.1} for the GPU alone",
        "together", gpu_both + cpu_both, gpu_alone
    );

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
