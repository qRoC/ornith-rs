//! Can the CPU reproduce one `affine_qmv_fast` output row bit for bit?
//!
//! This is the question that gates handing the CPU a share of a decode step's
//! output rows (see HANDOFF, "The CPU on the same memory"). The split itself is
//! sound -- each output row's reduction is self-contained, so giving the CPU
//! the last rows and the GPU the rest computes the same quantities. What is
//! not obvious is whether the CPU can compute them to the same *bits*, because
//! `max_abs == 0.0` against MLX is the rule this port does not bend.
//!
//! The kernel's order is fully determined (`quantized.h`, `qmv_fast_impl` and
//! `qdot`), for 8-bit / group 64: 32 lanes, each taking 8 consecutive values
//! per 256-wide block, accumulating in float, then one `simd_sum` across the
//! lanes and a round to bfloat16. Two things in it are *not* written down
//! anywhere:
//!
//!   1. whether the Metal compiler contracts `accum += x[i] * w[i]` and
//!      `scale * accum + sum * bias` into fused multiply-adds;
//!   2. what order `simd_sum` folds its 32 lanes in.
//!
//! So this tries every combination and reports which, if any, reproduces the
//! GPU exactly. It is deliberately a straight transcription -- no NEON, no
//! threads -- because the question is arithmetic, not throughput.
//!
//! Run: `cargo run --release --bin cpu_row`

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::quantized::{QuantMatrix, quantized_matmul};
use ornith_rs::refs::model_dir;
use ornith_rs::weights::WeightStore;

const ROWS_CHECKED: usize = 8192;

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `static_cast<bfloat16_t>(float)`: round to nearest, ties to even.
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// How `simd_sum` might fold 32 lanes.
#[derive(Clone, Copy, PartialEq)]
enum Tree {
    /// `shuffle_xor` butterfly, masks 1, 2, 4, 8, 16 -- the usual all-reduce.
    Butterfly,
    /// `shuffle_down` halving: 16, 8, 4, 2, 1.
    Halving,
    /// Lane order, which is what a naive reading would do.
    Sequential,
}

fn fold(v: &[f32; 32], tree: Tree) -> f32 {
    match tree {
        Tree::Butterfly => {
            let mut a = *v;
            for d in [1usize, 2, 4, 8, 16] {
                let mut n = a;
                for i in 0..32 {
                    n[i] = a[i] + a[i ^ d];
                }
                a = n;
            }
            a[0]
        }
        Tree::Halving => {
            let mut a = *v;
            let mut d = 16;
            while d >= 1 {
                for i in 0..d {
                    a[i] += a[i + d];
                }
                d /= 2;
            }
            a[0]
        }
        Tree::Sequential => v.iter().fold(0.0f32, |s, x| s + x),
    }
}

/// One output row, in `qmv_fast`'s order. 8-bit, group 64.
#[allow(clippy::too_many_arguments)]
fn row(
    w: &[u8],
    scales: &[u16],
    biases: &[u16],
    x: &[u16],
    k: usize,
    r: usize,
    fma: bool,
    tree: Tree,
) -> u16 {
    const VPT: usize = 8; // values_per_thread
    const BLOCK: usize = VPT * 32; // block_size
    const GROUP: usize = 64;
    let groups = k / GROUP;
    let mut acc = [0.0f32; 32];
    for lane in 0..32 {
        let mut a = 0.0f32;
        for blk in (0..k).step_by(BLOCK) {
            let base = blk + lane * VPT;
            let mut xs = [0.0f32; VPT];
            let mut sum = 0.0f32;
            for i in 0..VPT {
                xs[i] = bf16_to_f32(x[base + i]);
                sum += xs[i];
            }
            let g = blk / GROUP + lane / (GROUP / VPT);
            let s = bf16_to_f32(scales[r * groups + g]);
            let b = bf16_to_f32(biases[r * groups + g]);
            let mut accum = 0.0f32;
            for i in 0..VPT {
                let wv = w[r * k + base + i] as f32;
                accum = if fma { xs[i].mul_add(wv, accum) } else { accum + xs[i] * wv };
            }
            a += if fma { s.mul_add(accum, sum * b) } else { s * accum + sum * b };
        }
        acc[lane] = a;
    }
    f32_to_bf16(fold(&acc, tree))
}

/// The same row, stopping before the round to bfloat16, so the spread between
/// orderings can be compared with the room the rounding leaves.
#[allow(clippy::too_many_arguments)]
fn row_f32(
    w: &[u8],
    scales: &[u16],
    biases: &[u16],
    x: &[u16],
    k: usize,
    r: usize,
    fma: bool,
    tree: Tree,
) -> f32 {
    bf16_to_f32(0) + {
        // duplicated deliberately: `row` is the transcription under test and
        // is left exactly as the kernel reads.
        const VPT: usize = 8;
        const BLOCK: usize = VPT * 32;
        const GROUP: usize = 64;
        let groups = k / GROUP;
        let mut acc = [0.0f32; 32];
        for lane in 0..32 {
            let mut a = 0.0f32;
            for blk in (0..k).step_by(BLOCK) {
                let base = blk + lane * VPT;
                let mut xs = [0.0f32; VPT];
                let mut sum = 0.0f32;
                for i in 0..VPT {
                    xs[i] = bf16_to_f32(x[base + i]);
                    sum += xs[i];
                }
                let g = blk / GROUP + lane / (GROUP / VPT);
                let s = bf16_to_f32(scales[r * groups + g]);
                let b = bf16_to_f32(biases[r * groups + g]);
                let mut accum = 0.0f32;
                for i in 0..VPT {
                    let wv = w[r * k + base + i] as f32;
                    accum = if fma { xs[i].mul_add(wv, accum) } else { accum + xs[i] * wv };
                }
                a += if fma { s.mul_add(accum, sum * b) } else { s * accum + sum * b };
            }
            acc[lane] = a;
        }
        fold(&acc, tree)
    }
}

/// How far `v` sits from the nearest value that would round to a different
/// bfloat16 -- the room an ordering difference has before it changes the
/// answer.
fn margin_to_tie(v: f32) -> f32 {
    let b = f32_to_bf16(v);
    let centre = bf16_to_f32(b);
    // bfloat16 steps by one in the top 16 bits; half a step each way.
    let step = (bf16_to_f32(b.wrapping_add(1)) - centre).abs();
    (step * 0.5 - (v - centre).abs()).abs()
}

fn main() {
    let ctx = Context::new().expect("metal");
    let dir = model_dir();
    if !dir.exists() {
        eprintln!("model not present");
        return;
    }
    let mut ws = WeightStore::open(&dir, &[]).expect("weights");
    let qw: QuantMatrix = ws.quant_matrix(&ctx, "language_model.model.layers.0.linear_attn.in_proj_qkv");
    assert_eq!(qw.bits, 8, "this probe is the 8-bit path");
    assert_eq!(qw.group_size, 64);
    let k = qw.in_dims();
    let n = qw.out_dims();
    assert_eq!(k % 256, 0, "the block loop wants a multiple of 256");
    println!("in_proj_qkv: {n} rows of {k}, 8-bit group 64");

    // A real-looking activation: the kernel's arithmetic is what is on trial,
    // so the values have to exercise the rounding rather than be tidy.
    let xs: Vec<u16> = (0..k)
        .map(|i| {
            let v = ((i as f32 * 0.7391).sin() * 1.37 + 0.11) as f32;
            f32_to_bf16(v)
        })
        .collect();
    let x = Array::from_slice(&ctx, &xs, &[1, k], Dtype::BF16);
    let y = quantized_matmul(&ctx, &x, &qw);
    ctx.sync();

    let gpu = unsafe { y.buf.as_slice::<u16>(y.offset, n) }.to_vec();
    let wb = unsafe { qw.w.buf.as_slice::<u8>(qw.w.offset, n * k) };
    let sc = unsafe { qw.scales.buf.as_slice::<u16>(qw.scales.offset, n * k / 64) };
    let bi = unsafe { qw.biases.buf.as_slice::<u16>(qw.biases.offset, n * k / 64) };

    let checked = ROWS_CHECKED.min(n);
    println!("\nchecking {checked} rows against the GPU:");
    println!("  {:>5}  {:>11}  {:>8}  {}", "fma", "simd_sum", "exact", "worst row");
    for fma in [true, false] {
        for tree in [Tree::Butterfly, Tree::Halving, Tree::Sequential] {
            let mut exact = 0usize;
            let mut worst = 0.0f32;
            for r in 0..checked {
                let got = row(wb, sc, bi, &xs, k, r, fma, tree);
                if got == gpu[r] {
                    exact += 1;
                } else {
                    let d = (bf16_to_f32(got) - bf16_to_f32(gpu[r])).abs();
                    worst = worst.max(d);
                }
            }
            let name = match tree {
                Tree::Butterfly => "butterfly",
                Tree::Halving => "halving",
                Tree::Sequential => "sequential",
            };
            println!(
                "  {:>5}  {name:>11}  {exact:>4}/{checked:<3}  {worst:.3e}",
                if fma { "yes" } else { "no" }
            );
        }
    }

    // How much room there actually is. If the spread between orderings is far
    // below the distance to a rounding tie, the bfloat16 output absorbs the
    // difference and neither unknown has to be pinned; if they are the same
    // size, some rows would flip on other inputs and it does.
    let mut worst_spread = 0.0f32;
    let mut worst_ratio = 0.0f32;
    let mut at_risk = 0usize;
    for r in 0..checked {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for fma in [true, false] {
            for tree in [Tree::Butterfly, Tree::Halving, Tree::Sequential] {
                let v = row_f32(wb, sc, bi, &xs, k, r, fma, tree);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        let spread = hi - lo;
        let margin = margin_to_tie((lo + hi) * 0.5);
        worst_spread = worst_spread.max(spread);
        if margin > 0.0 {
            worst_ratio = worst_ratio.max(spread / margin);
        }
        if spread >= margin {
            at_risk += 1;
        }
    }
    println!(
        "\nspread between the six orderings against the room bfloat16 leaves:\n  \
         worst spread {worst_spread:.3e}, worst spread/margin {worst_ratio:.4}, \
         rows where the spread could change the answer: {at_risk}/{checked}"
    );
}
