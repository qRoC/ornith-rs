//! Is a row of a batch the same computation as that row on its own?
//!
//! Every other test in this suite checks the port against mlx. This one checks
//! a dispatcher against itself, because a verdict depends on it: a quantized
//! linear picks its kernel by row count -- `qmv` at M = 1, `qmv_wide` at
//! 2..`qmv_batch_limit`, the split-K gemm above that, and ggml's mat-vec or
//! its tile on the k-quant side -- and the speculative path runs the target
//! over a window of rows where the autoregressive path would have run it one
//! row at a time. If row `r` of an M-row call is not bit-identical to that row
//! computed alone, then some of the draft rejections MTP counts are the
//! *target* answering differently, not the draft being wrong.
//!
//! Nothing here needs a reference file: both arms are this machine's own
//! kernels on the same inputs, so the comparison is exact by construction or
//! it is not. `ops::invariant` is the flag that makes it so, and the second
//! half of this file is its contract.
//!
//! It runs against whatever artifact `ORNITH_MODEL` points at, so the same
//! test covers the affine families and both k-quants.

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::invariant::{self, EXACT_WINDOW};
use ornith_rs::ops::linear::Linear;
use ornith_rs::refs::{compare, model_dir};
use ornith_rs::weights::WeightStore;

const L0: &str = "language_model.model.layers.0";

/// The widest row count `qmv_wide` can see here: `qmv_batch_limit` is 13 for
/// every shape in this model, so 12 is the last row count below the gemm.
const M_MAX: usize = 12;

/// Standard normals from a splitmix64 stream, so the draw is the same on every
/// run and on every machine without carrying a reference file for it.
fn normals(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let u1 = next().max(1e-12);
            let u2 = next();
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        })
        .collect()
}

/// Row `r` of an `m`-row call against row `r` computed alone: the worst
/// difference over the rows, and how many of them came back identical.
fn rows_against_singletons(
    ctx: &Context,
    lin: &Linear,
    x: &Array,
    alone: &[Vec<f32>],
    m: usize,
) -> (f32, f32, usize) {
    let n = lin.out_dims();
    let y = lin.matmul(ctx, &x.slice_leading(0, m));
    let got = y.to_vec_f32(ctx);
    let (mut max_abs, mut max_rel, mut exact) = (0.0f32, 0.0f32, 0usize);
    for r in 0..m {
        let d = compare(&got[r * n..(r + 1) * n], &alone[r]);
        if d.max_abs == 0.0 {
            exact += 1;
        }
        max_abs = max_abs.max(d.max_abs);
        max_rel = max_rel.max(d.max_rel);
    }
    (max_abs, max_rel, exact)
}

/// Every row on its own, which is what the autoregressive path computes.
fn singletons(ctx: &Context, lin: &Linear, x: &Array, m: usize) -> Vec<Vec<f32>> {
    (0..m)
        .map(|r| lin.matmul(ctx, &x.slice_leading(r, 1)).to_vec_f32(ctx))
        .collect()
}

#[test]
fn a_batched_row_is_not_the_row_alone_until_the_flag_is_on() {
    let dir = model_dir();
    if !dir.exists() {
        eprintln!("model not present, skipping");
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();

    let modules = [
        ("in_proj_qkv", format!("{L0}.linear_attn.in_proj_qkv")),
        ("out_proj", format!("{L0}.linear_attn.out_proj")),
        ("gate", format!("{L0}.mlp.gate")),
        ("shared.gate_proj", format!("{L0}.mlp.shared_expert.gate_proj")),
        ("lm_head", "language_model.lm_head".to_string()),
    ];

    // Both arms in one test on purpose: the flag is process-wide, so two tests
    // setting it would race.
    let mut any_moved = false;
    println!(
        "\n{:<17} {:>14} {:>3}  {:>10} {:>10} {:>8}   {:>10} {:>8}",
        "module", "K->N", "M", "max_abs", "max_rel", "exact", "invariant", "exact"
    );
    for (tag, path) in &modules {
        let lin = ws.linear(&ctx, path);
        let k = lin.in_dims();
        let n = lin.out_dims();
        let x_host = normals(M_MAX * k, 0xa1b2_c3d4);
        let x = Array::from_f32_as_bf16(&ctx, &x_host, &[M_MAX, k]);

        let alone = invariant::with(false, || singletons(&ctx, &lin, &x, M_MAX));
        let alone_on = invariant::with(true, || singletons(&ctx, &lin, &x, M_MAX));
        let _ = EXACT_WINDOW;
        // A single row is a mat-vec either way, so the flag cannot move it.
        for r in 0..M_MAX {
            let d = compare(&alone_on[r], &alone[r]);
            assert_eq!(d.max_abs, 0.0, "{tag}: the flag moved a single row: {d}");
        }

        for m in 2..=M_MAX {
            let (off_abs, off_rel, off_exact) =
                invariant::with(false, || rows_against_singletons(&ctx, &lin, &x, &alone, m));
            // `all`, not the window mode: this sweep runs past `EXACT_WINDOW`
            // on purpose, to show that what the flag fixes is the kernel
            // choice and not one particular width.
            let (on_abs, _, on_exact) = invariant::with_rows(usize::MAX - 1, || {
                rows_against_singletons(&ctx, &lin, &x, &alone, m)
            });
            any_moved |= off_exact != m;
            println!(
                "{tag:<17} {k:>6}->{n:<6} {m:>3}  {off_abs:>10.3e} {off_rel:>10.3e} \
                 {off_exact:>4}/{m}   {on_abs:>10.3e} {on_exact:>4}/{m}"
            );
            assert_eq!(
                on_abs, 0.0,
                "{tag} M={m}: invariance is the whole point of the flag"
            );
        }
    }
    assert!(
        any_moved,
        "no batched row moved at all -- if the dispatcher stopped changing \
         kernels by row count, `ops::invariant` has nothing left to fix"
    );
}

/// The prefill widths too, where the flag is trading a tiled gemm for M
/// mat-vecs rather than `qmv_wide` for one.
#[test]
#[ignore = "M = 256 mat-vecs re-read the weights 256 times"]
fn the_gemm_widths_too() {
    let dir = model_dir();
    if !dir.exists() {
        eprintln!("model not present, skipping");
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let lin = ws.linear(&ctx, &format!("{L0}.linear_attn.in_proj_qkv"));
    let k = lin.in_dims();
    let x = Array::from_f32_as_bf16(&ctx, &normals(256 * k, 7), &[256, k]);
    let alone = invariant::with_rows(usize::MAX - 1, || singletons(&ctx, &lin, &x, 256));
    for m in [16usize, 64, 256] {
        let (abs, _, exact) = invariant::with_rows(usize::MAX - 1, || {
            rows_against_singletons(&ctx, &lin, &x, &alone, m)
        });
        println!("in_proj_qkv M={m}: max_abs {abs:.3e}, {exact}/{m} exact");
        assert_eq!(abs, 0.0, "M={m}");
    }
}

/// The routed half: one output row per (token, expert) pair, which is what a
/// decode step's eight rows and a verify window's thirty-two both take.
#[test]
fn a_routed_row_against_the_same_row_alone() {
    let dir = model_dir();
    if !dir.exists() {
        eprintln!("model not present, skipping");
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let lin = ws.linear(&ctx, &format!("{L0}.mlp.switch_mlp.gate_proj"));
    let experts = lin.num_experts();
    let k = lin.in_dims();
    let n = lin.out_dims();

    // 2048 rows over 256 experts, sorted: mlx's threshold for the streaming
    // gemm is rows/experts >= 4, so this is the shape that takes it.
    let rows = 2048usize;
    let x = Array::from_f32_as_bf16(&ctx, &normals(rows * k, 11), &[rows, k]);
    let lhs: Vec<u32> = (0..rows as u32).collect();
    let rhs: Vec<u32> = (0..rows).map(|r| (r * experts / rows) as u32).collect();
    let lhs_i = Array::from_slice(&ctx, &lhs, &[rows], Dtype::U32);
    let rhs_i = Array::from_slice(&ctx, &rhs, &[rows], Dtype::U32);

    // The reference: each pair on its own, one row at a time.
    let alone: Vec<Vec<f32>> = invariant::with_rows(usize::MAX - 1, || {
        (0..rows)
            .map(|r| {
                let one_l = Array::from_slice(&ctx, &[0u32], &[1], Dtype::U32);
                let one_r = Array::from_slice(&ctx, &[rhs[r]], &[1], Dtype::U32);
                lin.gather_matmul(&ctx, &x.slice_leading(r, 1), &one_l, &one_r, false)
                    .to_vec_f32(&ctx)
            })
            .collect()
    });

    for on in [false, true] {
        let y = invariant::with_rows(if on { usize::MAX - 1 } else { 0 }, || {
            lin.gather_matmul(&ctx, &x, &lhs_i, &rhs_i, true).to_vec_f32(&ctx)
        });
        let (mut max_abs, mut exact) = (0.0f32, 0usize);
        for r in 0..rows {
            let d = compare(&y[r * n..(r + 1) * n], &alone[r]);
            if d.max_abs == 0.0 {
                exact += 1;
            }
            max_abs = max_abs.max(d.max_abs);
        }
        println!("routed, invariant={on}: max_abs {max_abs:.3e}, {exact}/{rows} exact");
        if on {
            assert_eq!(max_abs, 0.0, "the routed rows are the flag's other half");
        }
    }
}

/// What the window mode covers, and where mlx's own routing ends it.
///
/// `ORNITH_INVARIANT=1` is not "invariant at every M" -- a prefill chunk is
/// not a batch of rows that would otherwise have run alone, and leaving it on
/// the tile is what keeps the mode free at a long prompt. So the contract is a
/// width: every batch of at most [`EXACT_WINDOW`] rows computes row by row,
/// and wider ones keep their kernel.
#[test]
fn the_window_mode_covers_a_verify_window_and_stops() {
    let dir = model_dir();
    if !dir.exists() {
        eprintln!("model not present, skipping");
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let lin = ws.linear(&ctx, "language_model.lm_head");
    let k = lin.in_dims();
    let x = Array::from_f32_as_bf16(&ctx, &normals(16 * k, 3), &[16, k]);
    let alone = invariant::with(true, || singletons(&ctx, &lin, &x, 16));

    for m in 2..=8 {
        let (abs, _, exact) =
            invariant::with(true, || rows_against_singletons(&ctx, &lin, &x, &alone, m));
        println!("window mode, lm_head M={m}: max_abs {abs:.3e}, {exact}/{m} exact");
        if m <= EXACT_WINDOW {
            assert_eq!(abs, 0.0, "M={m} is inside the window the mode promises");
        }
    }
}
