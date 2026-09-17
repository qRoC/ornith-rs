//! `SearchSorted::eval_gpu` from `vendor/mlx/mlx/backend/metal/sort.cpp`.
//!
//! One thread per query, a binary search over a 1-D sorted sequence.
//! `categorical_inverse_cdf` asks for the `right` side and subtracts one,
//! which turns the exclusive cumulative distribution into a sampled index.

use crate::array::{Array, Dtype};
use crate::metal::Context;

pub fn searchsorted(ctx: &Context, sorted: &Array, values: &Array, right: bool) -> Array {
    assert_eq!(sorted.ndim(), 1, "the sorted sequence must be 1-D");
    let t = match sorted.dtype {
        Dtype::F32 => "float32",
        Dtype::BF16 => "bfloat16",
        d => panic!("no searchsorted kernel for {}", d.name()),
    };
    let out = Array::empty(ctx, &values.shape, Dtype::U32);
    let n = sorted.numel() as u32;
    let a_stride = 1i64;
    let side = if right { "right" } else { "left" };
    let d = ctx.kernel(&format!("searchsorted_{t}_{side}"));
    let tg = d.max_threads_per_threadgroup().min(out.numel());
    d.arr(sorted)
        .arr(values)
        .out(&out)
        .bytes(&n)
        .bytes(&a_stride)
        .threads((out.numel(), 1, 1), (tg, 1, 1));
    out
}
