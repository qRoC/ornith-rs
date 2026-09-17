//! `mx.softmax(x, axis=-1, precise=True)`.
//!
//! Port of `Softmax::eval_gpu` (vendor/mlx/mlx/backend/metal/softmax.cpp) over
//! mlx's `softmax.metal`.
//!
//! `precise` matters: it accumulates the max and the sum in float32 even for a
//! bfloat16 input. Both the MoE router (`mx.softmax(gates, axis=-1,
//! precise=True)`) and attention (`softmax(scores, [-1], true)` inside mlx's
//! own SDPA fallback) ask for it.

use crate::array::{Array, Dtype};
use crate::metal::Context;

const SOFTMAX_N_READS: usize = 4;
const SOFTMAX_LOOPED_LIMIT: usize = 4096;

pub fn softmax(ctx: &Context, x: &Array, precise: bool) -> Array {
    let axis_size = x.dim(-1);
    let n_rows = x.numel() / axis_size;
    let out = Array::empty(ctx, &x.shape, x.dtype);

    let tname = match x.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no softmax kernel for {}", d.name()),
    };
    let looped = axis_size > SOFTMAX_LOOPED_LIMIT;
    // The precise variants only exist for the half types: for float32 input
    // the accumulator is already float32.
    let precise = precise && x.dtype != Dtype::F32;
    let name = format!(
        "{}softmax_{}{}",
        if looped { "looped_" } else { "block_" },
        if precise { "precise_" } else { "" },
        tname
    );

    let d = ctx.kernel(&name);
    let threadgroup_size = if looped {
        d.max_threads_per_threadgroup()
    } else {
        let needed = axis_size.div_ceil(SOFTMAX_N_READS);
        32 * needed.div_ceil(32)
    };
    assert!(threadgroup_size <= d.max_threads_per_threadgroup());
    d.arr(x)
        .arr(&out)
        .bytes(&(axis_size as i32))
        .threads((n_rows * threadgroup_size, 1, 1), (threadgroup_size, 1, 1));
    out
}
