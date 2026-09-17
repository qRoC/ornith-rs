//! `mx.fast.rms_norm` and `mx.fast.layer_norm`.
//!
//! Port of `RMSNorm::eval_gpu` and `LayerNorm::eval_gpu`
//! (vendor/mlx/mlx/backend/metal/normalization.cpp) over mlx's own
//! `rms_norm.metal` and `layer_norm.metal`.

use crate::array::{Array, Dtype, f32_to_bf16};
use crate::metal::Context;

const RMS_N_READS: usize = 4;
const RMS_LOOPED_LIMIT: usize = 4096;

/// `rms_norm(x, w, eps)`, normalising over the last axis.
///
/// `w = None` is how GatedDeltaNet normalises q and k. mlx implements that by
/// passing a 0-dim array of 1 and setting `w_stride = 0`, so the kernel reads
/// the same scalar for every column -- reproduced here with a one-element
/// buffer rather than a special case in the kernel.
pub fn rms_norm(ctx: &Context, x: &Array, w: Option<&Array>, eps: f32) -> Array {
    let axis_size = x.dim(-1);
    let n_rows = x.numel() / axis_size;
    let out = Array::empty(ctx, &x.shape, x.dtype);

    let looped = axis_size > RMS_LOOPED_LIMIT;
    let name = format!(
        "rms{}{}",
        if looped { "_looped" } else { "" },
        match x.dtype {
            Dtype::BF16 => "bfloat16",
            Dtype::F32 => "float32",
            d => panic!("no rms_norm kernel for {}", d.name()),
        }
    );

    let ones;
    let (w_arr, w_stride) = match w {
        Some(w) => {
            assert_eq!(w.numel(), axis_size, "rms_norm weight must match the last axis");
            (w, w.strides()[0] as u32)
        }
        None => {
            ones = scalar_one(ctx, x.dtype);
            (&ones, 0u32)
        }
    };

    let d = ctx.kernel(&name);
    let threadgroup_size = if looped {
        d.max_threads_per_threadgroup()
    } else {
        let needed = axis_size.div_ceil(RMS_N_READS);
        let simds = needed.div_ceil(32);
        32 * simds
    };
    assert!(threadgroup_size <= d.max_threads_per_threadgroup());
    d.arr(x)
        .arr(w_arr)
        .out(&out)
        .bytes(&eps)
        .bytes(&(axis_size as u32))
        .bytes(&w_stride)
        .threads((n_rows * threadgroup_size, 1, 1), (threadgroup_size, 1, 1));
    out
}

const LN_LOOPED_LIMIT: usize = 6656;

/// `layer_norm(x, w, b, eps)`, normalising over the last axis.
///
/// The vision tower is the only thing in this model with a LayerNorm -- the
/// decoder is RMSNorm throughout -- and every one of them is affine, with both
/// a weight and a bias, so mlx's `w.ndim() == 1` stride is always 1 here. The
/// kernel keeps the mean, the variance and the normalizer in float32 whatever
/// the array dtype is, and casts back per element when it writes.
pub fn layer_norm(ctx: &Context, x: &Array, w: &Array, b: &Array, eps: f32) -> Array {
    let axis_size = x.dim(-1);
    let n_rows = x.numel() / axis_size;
    let out = Array::empty(ctx, &x.shape, x.dtype);
    assert_eq!(w.numel(), axis_size, "layer_norm weight must match the last axis");
    assert_eq!(b.numel(), axis_size, "layer_norm bias must match the last axis");

    let looped = axis_size > LN_LOOPED_LIMIT;
    let n_reads = if looped { 4 } else { 8 };
    let name = format!(
        "layer_norm{}{}",
        if looped { "_looped" } else { "" },
        match x.dtype {
            Dtype::BF16 => "bfloat16",
            Dtype::F32 => "float32",
            d => panic!("no layer_norm kernel for {}", d.name()),
        }
    );

    let d = ctx.kernel(&name);
    let threadgroup_size = if looped {
        d.max_threads_per_threadgroup()
    } else {
        32 * axis_size.div_ceil(n_reads).div_ceil(32)
    };
    assert!(threadgroup_size <= d.max_threads_per_threadgroup());
    d.arr(x)
        .arr(w)
        .arr(b)
        .out(&out)
        .bytes(&eps)
        .bytes(&(axis_size as u32))
        .bytes(&(w.strides()[0] as u32))
        .bytes(&(b.strides()[0] as u32))
        .threads((n_rows * threadgroup_size, 1, 1), (threadgroup_size, 1, 1));
    out
}

fn scalar_one(ctx: &Context, dtype: Dtype) -> Array {
    match dtype {
        Dtype::BF16 => Array::from_slice(ctx, &[f32_to_bf16(1.0)], &[1], Dtype::BF16),
        Dtype::F32 => Array::from_slice(ctx, &[1.0f32], &[1], Dtype::F32),
        d => panic!("no rms_norm scalar for {}", d.name()),
    }
}
