//! Where the crossover is: ggml's k-quant mat-vec against the tile, per M.
//!
//! `kquant::matmul` has to choose one at every call, and the choice is not
//! obvious in either direction. The mat-vec re-reads the whole weight matrix
//! once per row, so its cost is linear in M from the second row on; the tile
//! reads the matrix once per 32 rows but pays for the tiling whether or not
//! the rows are there -- a batch of 2 through a 32-wide tile leaves 30 columns
//! of the accumulator empty. Somewhere between 1 and 32 they cross, and
//! `MM_ROW_LIMIT` is that number.
//!
//! This measures it rather than guessing it, on the shapes this model actually
//! has, and on both formats. The weights are random bytes: the kernels' time
//! does not depend on the values, only on the shapes, and a real matrix would
//! mean loading 25 GB to time a kernel.
//!
//! It also prices the thing the crossover exists for -- the speculative verify
//! window, which is `depth + 1` rows and lands in exactly this range.
//!
//! Run: `cargo run --release --bin kquant_mm`

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::kquant::{KMatrix, KQuant, QK_K, qmm_t, qmv};

/// `bandwidth.rs`'s timer, with one change that the first run of this file
/// forced: **`CHAIN` dispatches inside each sync window**.
///
/// Timing one dispatch between two `sync`s measures the submit-and-wait round
/// trip, which on this machine is ~120 us -- more than thirty times a small
/// matrix's own kernel time, so every shape came out at the same number and
/// the crossover was invisible. Chaining amortises the round trip and leaves
/// the marginal cost of the kernel, which is what the dispatcher is choosing
/// between. `HANDOFF.md`'s rule about not timing by the gap between calls is
/// the same mistake one level up.
const CHAIN: usize = 30;

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

/// A weight matrix of the right shape filled with arbitrary bytes.
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
    // Super-scales have to be finite and small, or the accumulators saturate
    // and the timing measures denormal handling instead of the kernel.
    let d: Vec<u16> = (0..o * nb * dm).map(|_| 0x1C00 | (next() as u16 & 0x03FF)).collect();
    KMatrix::new(
        kind,
        Array::from_slice(ctx, &qs, &[o, nb * codes / 4], Dtype::U32),
        Array::from_slice(ctx, &sc, &[o, nb, scales], Dtype::U8),
        Array::from_slice(ctx, &d, &[o, nb, dm], Dtype::F16),
    )
}

fn main() {
    let ctx = Context::new().unwrap();
    // The shapes the two formats are actually used at: `in_proj_qkv` and
    // `out_proj` of the gated-delta path for Q4_K, `q_proj` and the shared
    // expert's `down_proj` for Q6_K.
    let shapes: [(KQuant, &str, usize, usize); 4] = [
        (KQuant::Q4K, "in_proj_qkv [8192, 2048]", 8192, 2048),
        (KQuant::Q4K, "out_proj    [2048, 4096]", 2048, 4096),
        (KQuant::Q6K, "q_proj      [2048, 2048]", 2048, 2048),
        (KQuant::Q6K, "down_proj   [2048,  512]", 2048, 512),
    ];
    const MS: [usize; 9] = [1, 2, 4, 6, 8, 12, 16, 32, 64];
    const REPS: usize = 40;

    println!("us per call, best of {REPS} windows of {CHAIN}; ratio < 1 means the tile wins\n");
    println!("{:<26} {:>4}  {:>9} {:>9} {:>7}", "shape", "M", "mat-vec", "tile", "ratio");
    for (kind, name, o, k) in shapes {
        let w = fake(&ctx, kind, o, k);
        for m in MS {
            let x = Array::from_f32_as_bf16(&ctx, &vec![0.01f32; m * k], &[m, k]);
            let v = best_of(&ctx, REPS, || {
                std::hint::black_box(qmv(&ctx, &x, &w));
            });
            let t = best_of(&ctx, REPS, || {
                std::hint::black_box(qmm_t(&ctx, &x, &w));
            });
            println!(
                "{name:<26} {m:>4}  {:>9.1} {:>9.1} {:>7.2}",
                v * 1e6,
                t * 1e6,
                t / v
            );
        }
        println!();
    }
}
