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
use crate::ops::quantized::{QuantMatrix, QuantMode};

/// `embed_tokens(ids)` -> `[n_ids, dims]` in the scales' dtype.
pub fn embedding(ctx: &Context, table: &QuantMatrix, ids: &Array) -> Array {
    assert_eq!(ids.dtype, Dtype::U32);
    // The output dtype is the scales' in affine mode, where the scales are
    // bf16. nvfp4's scales are uint8 E4M3, so mlx takes the dtype from the
    // dequantize's own output instead; every recipe built here keeps
    // `embed_tokens` affine (it is 0.0% of a decode step's traffic, so there
    // is nothing to win by moving it), and this asserts that rather than
    // guessing a dtype no reference has been taken against.
    assert_eq!(
        table.mode,
        QuantMode::Affine,
        "embed_tokens is {:?}; the lookup's output dtype is the scales' \
         dtype, which only holds for affine",
        table.mode
    );
    let n = ids.numel();
    let dims = table.in_dims();
    let words = table.w.shape[1];
    let groups = table.scales.shape[1];
    let biases = table.biases.as_ref().expect("affine embedding has a zero point");

    let w = gather_rows(ctx, &table.w, ids, words);
    let scales = gather_rows(ctx, &table.scales, ids, groups);
    let biases = gather_rows(ctx, biases, ids, groups);

    let out = Array::empty(ctx, &[n, dims], table.scales.dtype);
    dequantize(ctx, &w, &scales, Some(&biases), &out, table.mode, table.group_size, table.bits);
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
        // nvfp4's scales are uint8 E4M3, which FR-Spec's pruning of `lm_head`
        // has to gather alongside the weight.
        Dtype::U8 => "gather_rows_by_id_uint8",
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

/// `mx.dequantize`. Port of the dequantize half of `quantize_impl` /
/// `get_quantize_kernel_dims` in quantized.cpp.
///
/// The grid is mode-independent -- it keys off `bits` alone -- and so is the
/// argument order, except that the fp kernels take the optional global scale
/// in the slot the affine ones give the zero point. Neither mode gets both,
/// and nvfp4 as forged here gets neither, so that slot is left unbound.
pub fn dequantize(
    ctx: &Context,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    out: &Array,
    mode: QuantMode,
    group_size: usize,
    bits: usize,
) {
    let tname = match out.dtype {
        Dtype::BF16 => "bfloat16_t",
        Dtype::F32 => "float",
        d => panic!("no dequantize kernel for {}", d.name()),
    };
    // mlx appends the `has_global_scale` flag to the name for the fp modes
    // only; affine's name ends at the bit width.
    let name = match mode {
        QuantMode::Affine => format!("affine_dequantize_{tname}_gs_{group_size}_b_{bits}"),
        QuantMode::Nvfp4 => format!("nvfp4_dequantize_{tname}_gs_{group_size}_b_{bits}_hgs_false"),
        // `ops::kquant::dequantize` is the k-quants', and it takes three
        // tensors rather than a `QuantMatrix`.
        QuantMode::K(_) => panic!("a k-quant's dequantize goes through ops::kquant"),
    };
    // `packs_per_int` values come out of each byte-pack; one thread per pack.
    let packs_per_int = match bits {
        3 | 5 => 8,
        6 => 4,
        b => 8 / b,
    };
    let nthreads = out.numel() / packs_per_int;
    let d = ctx.kernel(&name);
    let tg = d.max_threads_per_threadgroup().min(nthreads);
    let d = d.arr(w).arr(scales);
    let d = match biases {
        Some(b) => d.arr(b),
        None => d.skip(),
    };
    d.out(out).threads((nthreads, 1, 1), (tg, 1, 1));
}
