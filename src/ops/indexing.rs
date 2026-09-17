//! `take_along_axis` / `put_along_axis`, i.e. `GatherAxis::eval_gpu` and
//! `ScatterAxis::eval_gpu` from `vendor/mlx/mlx/backend/metal/indexing.cpp`.
//!
//! Narrowed to the last axis of a `[1, n]` row, which is every use the sampler
//! makes of them. The grid mlx launches is `[size post, index size, size pre]`
//! -- with the gathered axis last, `size_post` is 1 and `size_pre` is the row
//! count.

use crate::array::{Array, Dtype};
use crate::metal::Context;

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        Dtype::Bool => "bool_",
        d => panic!("no indexing kernel for {}", d.name()),
    }
}

/// `get_block_dims(dim0, dim1, dim2)` from `mlx/backend/metal/utils.h`: the
/// largest power-of-two block under 1024 that fits the grid.
pub fn block_dims(dim0: usize, dim1: usize, dim2: usize) -> (usize, usize, usize) {
    let pows = |v: usize| -> u32 {
        let mut p = 0;
        while (1usize << (p + 1)) <= v && p < 10 {
            p += 1;
        }
        p
    };
    let (mut p0, mut p1, mut p2) = (pows(dim0), pows(dim1), pows(dim2));
    let mut total = p0 + p1 + p2;
    while total > 10 {
        if p0 >= p1 && p0 >= p2 {
            p0 -= 1;
        } else if p1 >= p2 {
            p1 -= 1;
        } else {
            p2 -= 1;
        }
        total -= 1;
    }
    (1 << p0, 1 << p1, 1 << p2)
}

/// `mx.take_along_axis(src, idx, axis=-1)` for row-contiguous `[rows, n]`.
pub fn take_along_last_axis(ctx: &Context, src: &Array, idx: &Array) -> Array {
    let n = src.dim(-1);
    let rows = src.numel() / n;
    let idx_n = idx.dim(-1);
    let out = Array::empty(ctx, &idx.shape, src.dtype);

    // shape/strides with the gathered axis removed; here that leaves the rows.
    let shape = [rows as i32];
    let src_strides = [n as i64];
    let idx_strides = [idx_n as i64];
    let axis = src.ndim() - 1;

    let grid = (1usize, idx_n, rows);
    ctx.kernel(&format!("gather_axis{}uint32_intcc", tname(src.dtype)))
        .arr(src)
        .arr(idx)
        .out(&out)
        .bytes_slice(&shape)
        .bytes_slice(&src_strides)
        .bytes_slice(&idx_strides)
        .bytes(&(src.ndim() as u64 - 1))
        .bytes(&(axis as i32))
        .bytes(&(n as i32))
        .bytes(&1u64) // src_ax_stride
        .bytes(&1u64) // idx_ax_stride
        .threads(grid, block_dims(grid.0, grid.1, grid.2));
    out
}

/// `mx.put_along_axis(src, idx, values, axis=-1)`.
///
/// `values` is either a full `[rows, idx_n]` array (`contiguous_updates`) or a
/// single element broadcast across it, which is the shape mlx sees when the
/// caller writes a scalar and is why the `nc` kernel variant exists.
pub fn put_along_last_axis(
    ctx: &Context,
    src: &Array,
    idx: &Array,
    values: &Array,
    contiguous_updates: bool,
) -> Array {
    let n = src.dim(-1);
    let rows = src.numel() / n;
    let idx_n = idx.dim(-1);

    // `ScatterAxis::eval_gpu` copies `src` into `out` before scattering.
    let out = Array::empty(ctx, &src.shape, src.dtype);
    let total = out.numel();
    ctx.kernel(&format!("sampler_copy_{}", tname(src.dtype)))
        .arr(src)
        .out(&out)
        .threads((total, 1, 1), (1024.min(total), 1, 1));

    let shape = [rows as i32];
    // A broadcast scalar update has stride 0 on every axis.
    let upd_strides = [if contiguous_updates { idx_n as i64 } else { 0 }];
    let idx_strides = [idx_n as i64];
    let axis = src.ndim() - 1;
    let upd_ax_stride: u64 = if contiguous_updates { 1 } else { 0 };
    let variant = if contiguous_updates { "cc" } else { "ncc" };

    let grid = (1usize, idx_n, rows);
    ctx.kernel(&format!("scatter_axis{}uint32_none_int{variant}", tname(src.dtype)))
        .arr(values)
        .arr(idx)
        .out(&out)
        .bytes_slice(&shape)
        .bytes_slice(&upd_strides)
        .bytes_slice(&idx_strides)
        .bytes(&(src.ndim() as u64 - 1))
        .bytes(&(axis as i32))
        .bytes(&(n as i32))
        .bytes(&upd_ax_stride)
        .bytes(&1u64) // idx_ax_stride
        .threads(grid, block_dims(grid.0, grid.1, grid.2));
    out
}
