//! What `generate_step` does to the logits before the model is asked again.
//!
//! mlx-lm's `_step` is three lines:
//!
//! ```python
//! logits = logits[:, -1, :]
//! logprobs = logits - mx.logsumexp(logits, keepdims=True)
//! sampled = sampler(logprobs)
//! ```
//!
//! and `greedy_sampler` is `mx.argmax(logprobs, axis=-1)`. The subtraction
//! cannot change which entry is largest in exact arithmetic, but it is done in
//! bfloat16, so two logits an ulp apart can land on the same bfloat16 and hand
//! the argmax's "keep the lower index" tie-break a different answer. It is
//! therefore not an optimization to skip, and it is not skipped.
//!
//! Ports of `mlx/backend/metal/logsumexp.cpp`, `ArgReduce::eval_gpu` from
//! `mlx/backend/metal/primitives.cpp`, and `binary_op_gpu_inplace` from
//! `mlx/backend/metal/binary.cpp`.

use crate::array::{Array, Dtype};
use crate::metal::Context;

fn type_name(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        Dtype::F16 => "float16",
        d => panic!("no kernel for {}", d.name()),
    }
}

/// `mx.logsumexp(x, keepdims=True)` over the last axis.
///
/// `LOGSUMEXP_LOOPED_LIMIT` is 4096: a shorter row gets one threadgroup sized
/// to the row, a longer one the looped kernel at the full threadgroup width.
pub fn logsumexp_last_axis(ctx: &Context, x: &Array) -> Array {
    let axis_size = x.dim(-1);
    let n_rows = x.numel() / axis_size;
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = 1;
    let out = Array::empty(ctx, &shape, x.dtype);

    const SIMD_SIZE: usize = 32;
    const N_READS: usize = 4;
    const LOOPED_LIMIT: usize = 4096;

    let looped = axis_size > LOOPED_LIMIT;
    let prefix = if looped { "looped" } else { "block" };
    let name = format!("{prefix}_logsumexp_{}", type_name(x.dtype));

    let d = ctx.kernel(&name);
    let tg = if looped {
        d.max_threads_per_threadgroup()
    } else {
        let needed = axis_size.div_ceil(N_READS);
        SIMD_SIZE * needed.div_ceil(SIMD_SIZE)
    };
    assert!(tg <= d.max_threads_per_threadgroup());
    d.arr(x).arr(&out).bytes(&(axis_size as i32)).threads((n_rows * tg, 1, 1), (tg, 1, 1));
    out
}

/// `mx.argmax(x, axis=-1)`, dropping the reduced axis.
pub fn argmax_last_axis(ctx: &Context, x: &Array) -> Array {
    let axis = x.ndim() - 1;
    let axis_size = x.shape[axis];
    let axis_stride = x.strides()[axis];
    // `ArgReduce::eval_gpu` erases the reduced axis from the shape and the
    // input strides; the output already has it erased, so its strides are
    // left alone.
    let shape: Vec<i32> = x.shape[..axis].iter().map(|&v| v as i32).collect();
    let in_strides: Vec<i64> = x.strides()[..axis].to_vec();
    let out_shape: Vec<usize> =
        if shape.is_empty() { vec![1] } else { x.shape[..axis].to_vec() };
    let out = Array::empty(ctx, &out_shape, Dtype::U32);
    let out_strides: Vec<i64> = out.strides();
    let ndim = shape.len();

    const SIMD_SIZE: usize = 32;
    const N_READS: usize = 4;
    let d = ctx.kernel(&format!("argmax_{}", type_name(x.dtype)));
    let mut tg = axis_size.div_ceil(N_READS).min(d.max_threads_per_threadgroup());
    tg = tg.div_ceil(SIMD_SIZE) * SIMD_SIZE;
    assert!(tg <= d.max_threads_per_threadgroup());

    // `get_2d_grid_dims(out.shape(), out.strides())`, which for the contiguous
    // outputs here is just the element count laid out along x.
    let rows: usize = out.numel();
    let d = d.arr(x).arr(&out);
    let d = if ndim == 0 {
        d.bytes(&0i32).bytes(&0i64).bytes(&0i64)
    } else {
        d.bytes_slice(&shape).bytes_slice(&in_strides).bytes_slice(&out_strides)
    };
    d.bytes(&(ndim as u64))
        .bytes(&axis_stride)
        .bytes(&(axis_size as u64))
        .threads((tg, rows, 1), (tg, 1, 1));
    out
}

/// `a - b` where `b` holds a single element: mlx's `VectorScalar` binary.
pub fn subtract_scalar(ctx: &Context, a: &Array, b: &Array) -> Array {
    assert_eq!(b.numel(), 1, "subtract_scalar wants a one-element right side");
    assert_eq!(a.dtype, b.dtype);
    let out = Array::empty(ctx, &a.shape, a.dtype);
    let n = a.numel();
    // `get_work_per_thread(dtype, size)`: 1 below 1 << 16 elements, else
    // 8 / sizeof(dtype).
    let wpt = if n < (1 << 16) { 1 } else { (8 / a.dtype.size()).max(1) };
    let name = format!(
        "vs{}_Subtract{}",
        if wpt > 1 { "n" } else { "" },
        type_name(a.dtype)
    );
    let nthreads = n.div_ceil(wpt);
    let d = ctx.kernel(&name);
    let tg = d.max_threads_per_threadgroup().min(nthreads);
    d.arr(a).arr(b).arr(&out).bytes(&(n as u32)).threads((nthreads, 1, 1), (tg, 1, 1));
    out
}

/// The whole of mlx-lm's greedy `_step` after the forward pass: take the last
/// row of the logits, normalize, and take the argmax.
pub fn greedy_token(ctx: &Context, logits: &Array) -> u32 {
    let logprobs = log_softmax_last_row(ctx, logits);
    let idx = argmax_last_axis(ctx, &logprobs);
    idx.to_vec_u32(ctx)[0]
}

/// `logits[:, -1, :] - mx.logsumexp(logits[:, -1, :], keepdims=True)`.
pub fn log_softmax_last_row(ctx: &Context, logits: &Array) -> Array {
    let v = logits.dim(-1);
    let last = logits.reshape(&[logits.numel() / v, v]);
    let last = last.slice_leading(last.shape[0] - 1, 1);
    let lse = logsumexp_last_axis(ctx, &last);
    subtract_scalar(ctx, &last, &lse)
}

// --- test hooks --------------------------------------------------------------

/// `mx.exp(x)`, the first step of `apply_top_p`, exposed so the sampler test
/// can pin it on its own.
pub fn exp_for_test(ctx: &Context, x: &Array) -> Array {
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let n = x.numel();
    ctx.kernel(&format!("sampler_exp_{}", type_name(x.dtype)))
        .arr(x)
        .arr(&out)
        .threads((n, 1, 1), (1024.min(n), 1, 1));
    out
}

/// `-x`, the input `apply_top_k` argpartitions.
pub fn negate_for_test(ctx: &Context, x: &Array) -> Array {
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let n = x.numel();
    ctx.kernel(&format!("sampler_negative_{}", type_name(x.dtype)))
        .arr(x)
        .arr(&out)
        .threads((n, 1, 1), (1024.min(n), 1, 1));
    out
}

/// `mx.argmax(logits[:, -1, :], axis=-1)` without the logsumexp -- the draft
/// chain's `next_tok = mx.argmax(row, axis=-1)`, which reads the raw logits.
pub fn greedy_argmax(ctx: &Context, logits: &Array) -> u32 {
    let v = logits.dim(-1);
    let rows = logits.numel() / v;
    let last = logits.reshape(&[rows, v]).slice_leading(rows - 1, 1);
    argmax_last_axis(ctx, &last).to_vec_u32(ctx)[0]
}
