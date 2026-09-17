//! Reductions.
//!
//! Port of `strided_reduce_general_dispatch` and friends from
//! `vendor/mlx/mlx/backend/metal/reduce.cpp`, narrowed to the one shape this
//! model reduces over: the leading axis of a row-contiguous array. That is
//! mlx's own `ColReduceArgs(const array& intermediate)` case -- "reducing the
//! 1st axis of the row contiguous intermediate array" -- so nothing about the
//! argument encoding is invented here.
//!
//! Getting the *branch* right matters for more than speed: each of the four
//! kernels accumulates in a different order, and since mlx sums bfloat16 in
//! bfloat16, a different order is a different answer. Split-K with split_k=8
//! lands in `col_reduce_small` (strictly sequential) and split_k=32 in
//! `col_reduce_looped` (a simd_sum butterfly); they differ by an ULP.

use crate::array::{Array, Dtype};
use crate::metal::Context;

/// Arguments mlx's column-reduce kernels take, for a row-contiguous input
/// reduced along axis 0.
pub(crate) struct ColReduceArgs {
    reduction_size: usize,
    reduction_stride: usize,
    non_col_reductions: usize,
    // `shape`/`strides` describe the *non-reduced* dimensions and collapse to
    // nothing here, and `reduce_shape`/`reduce_strides` get the reduction axis
    // pushed onto them by each dispatch helper.
    shape: Vec<i32>,
    strides: Vec<i64>,
    ndim: i32,
    reduce_shape: Vec<i32>,
    reduce_strides: Vec<i64>,
    reduce_ndim: i32,
}

impl ColReduceArgs {
    fn for_leading_axis(rows: usize, cols: usize) -> ColReduceArgs {
        ColReduceArgs {
            reduction_size: rows,
            reduction_stride: cols,
            non_col_reductions: 1,
            shape: Vec::new(),
            strides: Vec::new(),
            ndim: 0,
            reduce_shape: Vec::new(),
            reduce_strides: Vec::new(),
            reduce_ndim: 0,
        }
    }

    /// `args.reduce_shape.push_back(reduction_size)` etc., which every
    /// dispatch helper does before launching.
    fn push_reduction_axis(&mut self) {
        self.reduce_shape.push(self.reduction_size as i32);
        self.reduce_strides.push(self.reduction_stride as i64);
        self.reduce_ndim += 1;
    }

    /// `ColReduceArgs::encode`: buffers 2..10, with the empty-vector padding
    /// mlx applies so Metal never sees a zero-length `setBytes`.
    fn encode<'a>(&self, d: crate::metal::Dispatch<'a>) -> crate::metal::Dispatch<'a> {
        let shape = if self.ndim == 0 { vec![0i32] } else { self.shape.clone() };
        let strides = if self.ndim == 0 { vec![0i64] } else { self.strides.clone() };
        let rshape = if self.reduce_ndim == 0 { vec![0i32] } else { self.reduce_shape.clone() };
        let rstrides = if self.reduce_ndim == 0 { vec![0i64] } else { self.reduce_strides.clone() };
        d.bytes(&(self.reduction_size as u64))
            .bytes(&(self.reduction_stride as i64))
            .bytes_slice(&shape)
            .bytes_slice(&strides)
            .bytes(&self.ndim)
            .bytes_slice(&rshape)
            .bytes_slice(&rstrides)
            .bytes(&self.reduce_ndim)
            .bytes(&(self.non_col_reductions as u64))
    }
}

fn type_name(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no reduction instantiated for {}", d.name()),
    }
}

/// `out = input.sum(axis=0)` for a row-contiguous `input`, reproducing mlx's
/// kernel choice exactly.
pub fn sum_axis0(ctx: &Context, input: &Array, out: &Array) {
    let rows = input.shape[0];
    let cols = input.numel() / rows;
    assert_eq!(out.numel(), cols, "output must hold one element per column");
    assert_eq!(out.dtype, input.dtype, "mlx sums bfloat16 into bfloat16");
    let mut args = ColReduceArgs::for_leading_axis(rows, cols);

    // strided_reduce_general_dispatch, in order.
    if args.reduction_size * args.non_col_reductions < 32 {
        return strided_reduce_small(ctx, input, out, &mut args);
    }
    if args.reduction_stride < 32 && args.reduction_size * args.non_col_reductions >= 1024 {
        return strided_reduce_longcolumn(ctx, input, out, &mut args);
    }
    if args.reduction_size * args.non_col_reductions > 256 && out.numel() / 32 < 1024 {
        return strided_reduce_2pass(ctx, input, out, &mut args);
    }
    strided_reduce_looped(ctx, input, out, &mut args)
}

fn strided_reduce_small(ctx: &Context, input: &Array, out: &Array, args: &mut ColReduceArgs) {
    args.push_reduction_axis();
    let name = format!("col_reduce_small_1_reduce_sum{}", type_name(input.dtype));

    let n_reads = 4;
    let reduction_stride_blocks = args.reduction_stride.div_ceil(n_reads);
    let total = args.reduction_size * args.non_col_reductions;
    let threadgroup_x = reduction_stride_blocks.min(32);
    let d = ctx.kernel(&name);
    let threadgroup_y = 8.min((d.max_threads_per_threadgroup() / threadgroup_x).min(total));

    // `output_grid_for_col_reduce` collapses every output axis whose stride is
    // below the reduction stride, which for a flat output is all of them.
    let grid = (reduction_stride_blocks.div_ceil(threadgroup_x), 1, 1);
    args.encode(d.arr(input).out(out)).groups(grid, (threadgroup_x, threadgroup_y, 1));
}

fn strided_reduce_looped(ctx: &Context, input: &Array, out: &Array, args: &mut ColReduceArgs) {
    args.push_reduction_axis();
    let bn = 32;
    let bm = 1024 / bn;
    let threadgroup_size = 8 * 32;
    let name = format!("col_reduce_looped_1_{bm}_{bn}_reduce_sum{}", type_name(input.dtype));
    let grid = (threadgroup_size * args.reduction_stride.div_ceil(bn), 1, 1);
    let d = ctx.kernel(&name);
    // Note: threads, not threadgroups -- mlx uses dispatch_threads here.
    args.encode(d.arr(input).out(out)).threads(grid, (threadgroup_size, 1, 1));
}

fn strided_reduce_2pass(ctx: &Context, input: &Array, out: &Array, args: &mut ColReduceArgs) {
    // Pass 1 writes 32 partial rows, pass 2 collapses them with the same
    // col_reduce_small the first branch uses.
    let intermediate = Array::empty(ctx, &[32, out.numel()], out.dtype);
    args.push_reduction_axis();

    let bn = 32;
    let bm = 1024 / bn;
    let threadgroup_size = 8 * 32;
    let name = format!("col_reduce_2pass_1_{bm}_{bn}_reduce_sum{}", type_name(input.dtype));
    let grid = (threadgroup_size * args.reduction_stride.div_ceil(bn), 32, 1);
    let d = ctx.kernel(&name);
    args.encode(d.arr(input).out(&intermediate)).threads(grid, (threadgroup_size, 1, 1));

    let mut second = ColReduceArgs::for_leading_axis(32, out.numel());
    strided_reduce_small(ctx, &intermediate, out, &mut second);
}

fn strided_reduce_longcolumn(ctx: &Context, input: &Array, out: &Array, args: &mut ColReduceArgs) {
    let total_reduction_size = args.reduction_size * args.non_col_reductions;
    let outer_blocks = if total_reduction_size >= 32768 { 128 } else { 32 };
    let intermediate = Array::empty(ctx, &[outer_blocks, out.numel()], out.dtype);
    args.push_reduction_axis();

    let name = format!("col_reduce_longcolumn_1_reduce_sum{}", type_name(input.dtype));
    let d = ctx.kernel(&name);
    let threadgroup_x = args.reduction_stride.min(32);
    let threadgroup_y = (d.max_threads_per_threadgroup() / threadgroup_x).min(total_reduction_size);
    let grid = (args.reduction_stride.div_ceil(threadgroup_x), outer_blocks, 1);
    args.encode(d.arr(input).out(&intermediate))
        .bytes(&(out.numel() as u64))
        .groups(grid, (threadgroup_x, threadgroup_y, 1));

    let mut second = ColReduceArgs::for_leading_axis(outer_blocks, out.numel());
    strided_reduce_small(ctx, &intermediate, out, &mut second);
}

// ---------------------------------------------------------------------------
// Row reductions
// ---------------------------------------------------------------------------

/// `x.sum(axis=-1, keepdims=True)` for a row-contiguous `x`.
///
/// `row_reduce_general_dispatch` sends rows of 64 or fewer elements to
/// `row_reduce_small`, which is what the MoE router's 8-wide
/// `scores.sum(axis=-1)` hits.
pub fn sum_last_axis(ctx: &Context, input: &Array, out: &Array) {
    let row_size = input.dim(-1);
    let n_rows = input.numel() / row_size;
    assert_eq!(out.numel(), n_rows);
    assert_eq!(out.dtype, input.dtype);
    assert!(row_size <= 64, "wider rows would take row_reduce_simple or _looped");

    let name = format!("row_reduce_small_1_reduce_sum{}", type_name(input.dtype));
    // RowReduceArgs for a contiguous reduce over the last axis: the reduction
    // axis is the row itself, so reduce_shape is empty and the remaining
    // (non-reduced) axes collapse to a single [n_rows] with stride row_size.
    let shape = [n_rows as i32];
    let strides = [row_size as i64];
    let d = ctx.kernel(&name);
    // `(non_row_reductions < 32 && row_size <= 8) || non_row_reductions <= 8`
    // holds here (non_row_reductions == 1), so one thread per output row.
    let tg = n_rows.min(1024);
    d.arr(input)
        .out(out)
        .bytes(&(row_size as u64))
        .bytes(&1u64) // non_row_reductions
        .bytes_slice(&shape)
        .bytes_slice(&strides)
        .bytes(&1i32) // ndim
        .bytes_slice(&[0i32]) // reduce_shape, padded as mlx does
        .bytes_slice(&[0i64]) // reduce_strides
        .bytes(&0i32) // reduce_ndim
        .threads((n_rows, 1, 1), (tg, 1, 1));
}

/// `x.sum(axis=-2)` for a row-contiguous `[outer, n, dim]`, giving
/// `[outer, dim]`. This is the MoE mixture `(y * scores).sum(axis=-2)`.
pub fn sum_second_last_axis(ctx: &Context, input: &Array, out: &Array) {
    assert_eq!(input.ndim(), 3);
    let (outer, n, dim) = (input.shape[0], input.shape[1], input.shape[2]);
    assert_eq!(out.shape, vec![outer, dim]);
    let mut args = ColReduceArgs::for_leading_axis(n, dim);
    // Unlike the split-K case the output is not flat: one row per `outer`.
    args.shape = vec![outer as i32];
    args.strides = vec![(n * dim) as i64];
    args.ndim = 1;

    if args.reduction_size * args.non_col_reductions < 32 {
        return strided_reduce_small_rows(ctx, input, out, &mut args, outer);
    }
    panic!("top_k > 32 would need the other column-reduce branches");
}

fn strided_reduce_small_rows(
    ctx: &Context,
    input: &Array,
    out: &Array,
    args: &mut ColReduceArgs,
    out_rows: usize,
) {
    args.push_reduction_axis();
    let name = format!("col_reduce_small_1_reduce_sum{}", type_name(input.dtype));
    let n_reads = 4;
    let reduction_stride_blocks = args.reduction_stride.div_ceil(n_reads);
    let total = args.reduction_size * args.non_col_reductions;
    let threadgroup_x = reduction_stride_blocks.min(32);
    let d = ctx.kernel(&name);
    let threadgroup_y = 8.min((d.max_threads_per_threadgroup() / threadgroup_x).min(total));
    let grid = (reduction_stride_blocks.div_ceil(threadgroup_x), out_rows, 1);
    args.encode(d.arr(input).out(out)).groups(grid, (threadgroup_x, threadgroup_y, 1));
}

// --- whole-array reductions --------------------------------------------------

/// `all_reduce_dispatch` from `mlx/backend/metal/reduce.cpp`: `max(x)` and
/// `sum(x)` over every element.
///
/// Anything past 4096 elements takes the two-pass path, and the row count of
/// the first pass depends on the *byte* size of the input, not its length --
/// which decides how many partial sums get combined and therefore, for float,
/// the answer's last bit.
pub fn all_reduce(ctx: &Context, input: &Array, op: &str) -> Array {
    const REDUCE_N_READS: usize = 4;
    let t = type_name(input.dtype);
    let out = Array::empty(ctx, &[1], input.dtype);
    let in_size = input.numel();
    let name = format!("all_reduce_{op}{t}");

    if in_size <= REDUCE_N_READS * 1024 {
        let mut tg = in_size.div_ceil(REDUCE_N_READS);
        tg = tg.div_ceil(32) * 32;
        ctx.kernel(&name)
            .arr(input)
            .out(&out)
            .bytes(&(in_size as u64))
            .bytes(&(in_size as u64))
            .threads((tg, 1, 1), (tg, 1, 1));
        return out;
    }

    let (n_rows, tg_2nd) =
        if input.nbytes() <= (1 << 26) { (32 * REDUCE_N_READS, 32) } else { (1024 * REDUCE_N_READS, 1024) };
    let intermediate = Array::empty(ctx, &[n_rows], input.dtype);

    let row_size = in_size.div_ceil(n_rows);
    let mut tg = row_size.div_ceil(REDUCE_N_READS).min(1024);
    tg = tg.div_ceil(32) * 32;
    ctx.kernel(&name)
        .arr(input)
        .out(&intermediate)
        .bytes(&(in_size as u64))
        .bytes(&(row_size as u64))
        .threads((tg, n_rows, 1), (tg, 1, 1));

    ctx.kernel(&name)
        .arr(&intermediate)
        .out(&out)
        .bytes(&(n_rows as u64))
        .bytes(&(n_rows as u64))
        .threads((tg_2nd, 1, 1), (tg_2nd, 1, 1));
    out
}
