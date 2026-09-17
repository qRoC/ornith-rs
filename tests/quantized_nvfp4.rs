//! Our nvfp4 dispatch against mlx's, on this model's weights recut as nvfp4.
//!
//! The twin of `tests/quantized.rs`. Both sides run the same kernels (mlx's,
//! compiled into our metallib), so a correct dispatch must agree bit for bit
//! -- any difference means we picked a different kernel, grid or argument
//! order than `quantized.cpp` does. That is the whole point of the check for
//! nvfp4 in particular: its kernels take the same slot affine gives the zero
//! point for an optional global scale, drop it entirely in two places, and
//! use `k_lanes = 16` where affine uses 8, so there are four ways to bind the
//! arguments plausibly and only one that is right.
//!
//! The reference carries the quantized weights itself, so this runs against
//! any artifact on disk -- it is the pre-flight for the nvfp4 rebuild rather
//! than a check of it.
//!
//! Reference: tools/dump_quantized_nvfp4.py

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::embedding::dequantize;
use ornith_rs::ops::quantized::{QuantMatrix, QuantMode, gather_qmm_rows, quantized_matmul};
use ornith_rs::refs::{RefCase, compare};

const ROWS: [usize; 7] = [1, 2, 4, 8, 16, 64, 256];
const GROUP_SIZE: usize = 16;
const BITS: usize = 4;

fn matrix(case: &RefCase, ctx: &Context, tag: &str) -> QuantMatrix {
    QuantMatrix {
        w: case.array(ctx, &format!("{tag}_w")),
        scales: case.array(ctx, &format!("{tag}_scales")),
        biases: None,
        mode: QuantMode::Nvfp4,
        group_size: GROUP_SIZE,
        bits: BITS,
    }
}

#[test]
fn matmul_matches_mlx_bit_for_bit() {
    let Some(case) = RefCase::open_or_skip("quantized_matmul_nvfp4") else { return };
    let ctx = Context::new().unwrap();

    let mut checked = 0;
    for tag in ["in_proj_qkv", "out_proj", "gate"] {
        let qw = matrix(&case, &ctx, tag);
        assert_eq!(qw.scales.dtype, Dtype::U8, "{tag}: nvfp4 scales are E4M3 bytes");
        assert_eq!(qw.in_dims(), case.meta_usize(&format!("{tag}.in_dims")));
        assert_eq!(qw.out_dims(), case.meta_usize(&format!("{tag}.out_dims")));

        for m in ROWS {
            let x = case.array(&ctx, &format!("{tag}_x_m{m}"));
            let y = quantized_matmul(&ctx, &x, &qw);
            let got = y.to_vec_f32(&ctx);
            let want = case.f32(&format!("{tag}_y_m{m}"));
            let d = compare(&got, &want);
            assert_eq!(
                d.max_abs, 0.0,
                "{tag} M={m}: {d}\n(same kernels on both sides, so any \
                 difference is a dispatch mismatch)"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 3 * ROWS.len());
}

/// `nvfp4_dequantize`, which is the format on its own: no matmul, no
/// accumulation order, just the E2M1 codes and their E4M3 group scales.
/// If this disagrees the layout is being read wrong; if only the matmuls
/// disagree the layout is right and the dispatch is not.
#[test]
fn dequantize_matches_mlx_bit_for_bit() {
    let Some(case) = RefCase::open_or_skip("quantized_matmul_nvfp4") else { return };
    let ctx = Context::new().unwrap();

    for tag in ["in_proj_qkv", "out_proj", "gate"] {
        let qw = matrix(&case, &ctx, tag);
        let rows = 64;
        let in_dims = qw.in_dims();
        // The reference dequantized the first 64 rows, which are the first
        // 64 slabs of a row-contiguous array.
        let w = head_rows(&qw.w, rows);
        let scales = head_rows(&qw.scales, rows);
        let out = Array::empty(&ctx, &[rows, in_dims], Dtype::BF16);
        dequantize(&ctx, &w, &scales, None, &out, QuantMode::Nvfp4, GROUP_SIZE, BITS);

        let d = compare(&out.to_vec_f32(&ctx), &case.f32(&format!("{tag}_deq")));
        assert_eq!(d.max_abs, 0.0, "{tag} dequantize: {d}");
    }
}

/// The two gathered branches `GatherQMM::eval_gpu` picks between, which is
/// where the MoE body's 4-bit experts go.
#[test]
fn gather_matches_mlx_bit_for_bit() {
    let Some(case) = RefCase::open_or_skip("quantized_matmul_nvfp4") else { return };
    let ctx = Context::new().unwrap();

    let qw = matrix(&case, &ctx, "switch");
    assert_eq!(qw.num_experts(), case.meta_usize("switch.experts"));

    for (tag, sorted) in [("unsorted", false), ("sorted", true)] {
        let x = case.array(&ctx, &format!("switch_{tag}_x"));
        let lhs = case.array(&ctx, &format!("switch_{tag}_lhs"));
        let rhs = case.array(&ctx, &format!("switch_{tag}_rhs"));
        let y = gather_qmm_rows(&ctx, &x, &qw, &lhs, &rhs, sorted);

        let d = compare(&y.to_vec_f32(&ctx), &case.f32(&format!("switch_{tag}_y")));
        assert_eq!(
            d.max_abs, 0.0,
            "switch {tag}: {d}\n(the {} branch)",
            if sorted { "gather_qmm_rhs" } else { "gather_qmv" }
        );
    }
}

/// The first `n` rows of a row-contiguous 2-D array: the same buffer at the
/// same offset with a shorter leading axis. `Array::index` is the same view
/// for a single row.
fn head_rows(a: &Array, n: usize) -> Array {
    assert_eq!(a.ndim(), 2);
    assert!(n <= a.shape[0]);
    Array { buf: a.buf.clone(), offset: a.offset, shape: vec![n, a.shape[1]], dtype: a.dtype }
}
