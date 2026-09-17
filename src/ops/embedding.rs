//! `nn.QuantizedEmbedding`.
//!
//! mlx implements the lookup as three row gathers followed by a dequantize:
//!
//! ```text
//!     mx.dequantize(self.weight[x], scales=self.scales[x],
//!                   biases=self.biases[x], group_size=..., bits=...)
//! ```
//!
//! so this does the same rather than fusing, and the dequantize kernel is
//! mlx's own `affine_dequantize`.

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::ops::quantized::QuantMatrix;

/// `embed_tokens(ids)` -> `[n_ids, dims]` in the scales' dtype.
pub fn embedding(ctx: &Context, table: &QuantMatrix, ids: &Array) -> Array {
    assert_eq!(ids.dtype, Dtype::U32);
    let n = ids.numel();
    let dims = table.in_dims();
    let words = table.w.shape[1];
    let groups = table.scales.shape[1];

    let w = gather_rows(ctx, &table.w, ids, words);
    let scales = gather_rows(ctx, &table.scales, ids, groups);
    let biases = gather_rows(ctx, &table.biases, ids, groups);

    let out = Array::empty(ctx, &[n, dims], table.scales.dtype);
    dequantize(ctx, &w, &scales, &biases, &out, table.group_size, table.bits);
    out
}

/// `table[ids]` for a 2-D table of `row`-wide rows -- mlx's `Gather` with a
/// single index array over the leading axis.
pub fn gather_rows(ctx: &Context, table: &Array, ids: &Array, row: usize) -> Array {
    let n = ids.numel();
    let out = Array::empty(ctx, &[n, row], table.dtype);
    let total = n * row;
    let tg = 256.min(total.next_power_of_two()).max(1);
    let name = match table.dtype {
        Dtype::U32 => "gather_rows_by_id_uint32",
        Dtype::BF16 => "gather_rows_by_id_bfloat16",
        Dtype::F32 => "gather_rows_by_id_float32",
        d => panic!("no gather for {}", d.name()),
    };
    ctx.kernel(name)
        .arr(table)
        .arr(ids)
        .out(&out)
        .bytes(&(row as u32))
        .threads((total, 1, 1), (tg, 1, 1));
    out
}

/// `mx.dequantize`, affine mode. Port of the dequantize half of
/// `quantize_impl` / `get_quantize_kernel_dims` in quantized.cpp.
pub fn dequantize(
    ctx: &Context,
    w: &Array,
    scales: &Array,
    biases: &Array,
    out: &Array,
    group_size: usize,
    bits: usize,
) {
    let tname = match out.dtype {
        Dtype::BF16 => "bfloat16_t",
        Dtype::F32 => "float",
        d => panic!("no dequantize kernel for {}", d.name()),
    };
    let name = format!("affine_dequantize_{tname}_gs_{group_size}_b_{bits}");
    // `packs_per_int` values come out of each byte-pack; one thread per pack.
    let packs_per_int = match bits {
        3 | 5 => 8,
        6 => 4,
        b => 8 / b,
    };
    let nthreads = out.numel() / packs_per_int;
    let d = ctx.kernel(&name);
    let tg = d.max_threads_per_threadgroup().min(nthreads);
    d.arr(w)
        .arr(scales)
        .arr(biases)
        .out(out)
        .threads((nthreads, 1, 1), (tg, 1, 1));
}
