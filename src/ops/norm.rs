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

/// `h = a + b`, then `rms_norm(h, w, eps)` -- one dispatch for both, and both
/// arrays out, since the sum is the residual the layer needs again.
///
/// The pair is the most frequent barrier in a decode step (78 a token: every
/// layer adds its residual and then normalises it) and neither half does any
/// work -- `REBUILD.md` Phase 31 measured the two at 0.33 and 0.22 ms a token
/// with their grids clamped to a single threadgroup, which is to say all of
/// it is the launch. This removes one of the two launches and the barrier
/// between them.
///
/// Bit-exact by construction rather than by tolerance: the kernel evaluates
/// `a[i] + b[i]` in the array's own dtype, exactly as `add_elem` does, and
/// the sum of squares accumulates the **rounded** result, exactly as
/// `rms_norm` would having read it back from memory. `ORNITH_FUSE_ADD_RMS=0`
/// runs the two separately, as the control.
pub fn add_rms_norm(
    ctx: &Context,
    a: &Array,
    b: &Array,
    w: Option<&Array>,
    eps: f32,
) -> (Array, Array) {
    let axis_size = a.dim(-1);
    if !fuse_add_rms() || axis_size > RMS_LOOPED_LIMIT {
        let h = crate::ops::elementwise::add(ctx, a, b);
        let normed = rms_norm(ctx, &h, w, eps);
        return (h, normed);
    }
    assert_eq!(a.shape, b.shape, "the residual and the branch must have one shape");
    let n_rows = a.numel() / axis_size;
    let h = Array::empty(ctx, &a.shape, a.dtype);
    let out = Array::empty(ctx, &a.shape, a.dtype);

    let name = format!(
        "add_rms_{}",
        match a.dtype {
            Dtype::BF16 => "bfloat16",
            Dtype::F32 => "float32",
            d => panic!("no add_rms kernel for {}", d.name()),
        }
    );

    let ones;
    let (w_arr, w_stride) = match w {
        Some(w) => {
            assert_eq!(w.numel(), axis_size, "rms_norm weight must match the last axis");
            (w, w.strides()[0] as u32)
        }
        None => {
            ones = scalar_one(ctx, a.dtype);
            (&ones, 0u32)
        }
    };

    let d = ctx.kernel(&name);
    let threadgroup_size = 32 * axis_size.div_ceil(RMS_N_READS).div_ceil(32);
    assert!(threadgroup_size <= d.max_threads_per_threadgroup());
    d.arr(a)
        .arr(b)
        .arr(w_arr)
        .out(&h)
        .out(&out)
        .bytes(&eps)
        .bytes(&(axis_size as u32))
        .bytes(&w_stride)
        .threads((n_rows * threadgroup_size, 1, 1), (threadgroup_size, 1, 1));
    (h, out)
}

/// An atomic rather than a `OnceLock` so that one process can run both code
/// paths: a change worth ~0.15 ms a token is below what a 200-token run
/// separates on this machine, and the only measurement that resolves it is
/// the two paths alternating inside one process.
static FUSE_ADD_RMS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static FUSE_READ_ENV: std::sync::Once = std::sync::Once::new();

fn fuse_add_rms() -> bool {
    FUSE_READ_ENV.call_once(|| {
        if std::env::var("ORNITH_FUSE_ADD_RMS").as_deref() == Ok("0") {
            FUSE_ADD_RMS.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    FUSE_ADD_RMS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turn the fusion on or off at runtime; returns what it was.
pub fn set_fuse_add_rms(on: bool) -> bool {
    let was = fuse_add_rms();
    FUSE_ADD_RMS.store(on, std::sync::atomic::Ordering::Relaxed);
    was
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
