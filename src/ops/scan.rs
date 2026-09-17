//! `scan_gpu_inplace` from `vendor/mlx/mlx/backend/metal/scan.cpp`, for the
//! contiguous last-axis case the sampler needs.
//!
//! The threadgroup size is not just "as wide as fits": mlx sizes it to the row
//! for short rows, halves it in the middle band, and only then goes to the
//! kernel's maximum. That choice changes how the partial sums are combined, so
//! it is copied exactly.

use crate::array::{Array, Dtype};
use crate::metal::Context;

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no scan kernel for {}", d.name()),
    }
}

/// `mx.cumsum(x, axis=-1)` over a row-contiguous array.
///
/// The output dtype is the input dtype -- mlx instantiates
/// `sum_bfloat16_bfloat16`, so a bfloat16 scan rounds to bfloat16 at every
/// step and saturates far sooner than a float32 one would.
pub fn cumsum_last_axis(ctx: &Context, x: &Array, inclusive: bool) -> Array {
    let axis_size = x.dim(-1);
    let n_rows = x.numel() / axis_size;
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let t = tname(x.dtype);
    let kind = if inclusive { "inclusive" } else { "exclusive" };
    let name = format!("contig_scan_{kind}_sum_{t}_{t}");

    let d = ctx.kernel(&name);
    let n_reads: usize = if x.dtype.size() <= 4 { 4 } else { 2 };
    const SIMD_SIZE: usize = 32;
    let elements_per_simd = n_reads * SIMD_SIZE;
    let max_tg = d.max_threads_per_threadgroup();
    let mut tg = max_tg;
    if axis_size <= n_reads * 1024 {
        tg = axis_size.div_ceil(elements_per_simd) * SIMD_SIZE;
    } else if axis_size <= n_reads * 2048 {
        tg = (axis_size / 2).div_ceil(elements_per_simd) * SIMD_SIZE;
    }
    tg = tg.min(max_tg);

    // `get_2d_grid_dims(in.shape(), in.strides(), divisor=size)` collapses to
    // the row count for a row-contiguous input.
    d.arr(x)
        .arr(&out)
        .bytes(&(axis_size as u64))
        .threads((tg, n_rows, 1), (tg, 1, 1));
    out
}
