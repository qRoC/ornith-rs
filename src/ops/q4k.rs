//! ggml's Q4_K: the dispatch side of `kernels/q4k.metal`.
//!
//! This is the one mode in the port with no mlx behind it. mlx has four
//! quantization modes and none is a k-quant, and the lead maintainer closed
//! the request for one -- `Q4K.md` quotes it -- so there is no
//! `quantized_matmul` to port and no `max_abs == 0.0` against MLX to be had.
//! What replaces it is `ggml`, which implements exactly this format and did
//! not learn it from us: [`dequantize`] reproduces `dequantize_row_q4_K` bit
//! for bit on bytes `quantize_q4_K` wrote, and `tests/q4k.rs` asserts it.
//!
//! **Three tensors, not one packed blob.** ggml keeps a 144-byte
//! `block_q4_K`; the artifact keeps the codes, the packed 6-bit fields and the
//! two super-scales as separate tensors, so the safetensors shapes stay the
//! ones the loader and `check_rebuild.py` already understand -- the same way
//! nvfp4's two-tensor modules did. The bytes inside each are ggml's own, so
//! only the addressing differs, and `tools/q4k.py` splits a ggml block into
//! exactly these three.
//!
//!   qs      uint32  [O, I / 8]        the 4-bit codes, 128 bytes a block
//!   scales  uint8   [O, I / 256, 12]  8 x 6-bit scale + 8 x 6-bit min
//!   dm      float16 [O, I / 256, 2]   d and dmin
//!
//! Only the M = 1 path is here. `Q4K.md` scopes the first cut that way on
//! purpose: `affine_qmv_fast` is 38% of a decode step, so one kernel is enough
//! to measure GB/token and GPU busy for real, and prefill can keep running the
//! affine artifact until the format is judged.

use crate::array::{Array, Dtype};
use crate::metal::Context;

/// ggml's `QK_K`: weights per super-block.
pub const QK_K: usize = 256;
/// `N_SG_Q4_K` and `N_R0_Q4_K` from `ggml-metal-impl.h`: simdgroups per
/// threadgroup, and output rows per simdgroup.
const NSG: usize = 2;
const NR0: usize = 2;

/// A Q4_K weight matrix, as three tensors.
#[derive(Clone)]
pub struct Q4kMatrix {
    /// `[O, I / 8]` uint32 -- ggml's `qs`, 128 bytes per super-block.
    pub qs: Array,
    /// `[O, I / 256, 12]` uint8 -- ggml's `scales`.
    pub scales: Array,
    /// `[O, I / 256, 2]` float16 -- ggml's `d` and `dmin`.
    pub dm: Array,
}

impl Q4kMatrix {
    pub fn new(qs: Array, scales: Array, dm: Array) -> Q4kMatrix {
        let m = Q4kMatrix { qs, scales, dm };
        m.check();
        m
    }

    /// Output features: rows of the stored `[O, I]` matrix.
    pub fn out_dims(&self) -> usize {
        self.qs.shape[self.qs.ndim() - 2]
    }

    /// Input features. A super-block is 256 weights and `dm` carries one pair
    /// per super-block, so the block count is the tensor's own.
    pub fn in_dims(&self) -> usize {
        self.blocks_per_row() * QK_K
    }

    fn blocks_per_row(&self) -> usize {
        self.dm.shape[self.dm.ndim() - 2]
    }

    /// Everything the kernel assumes about the three tensors, checked once at
    /// load rather than read as garbage per dispatch.
    fn check(&self) {
        let (o, nb) = (self.out_dims(), self.blocks_per_row());
        assert_eq!(self.qs.dtype, Dtype::U32, "q4k qs must be uint32");
        assert_eq!(self.scales.dtype, Dtype::U8, "q4k scales must be uint8");
        assert_eq!(self.dm.dtype, Dtype::F16, "q4k dm must be float16");
        assert_eq!(
            self.qs.shape[self.qs.ndim() - 1],
            nb * QK_K / 8,
            "q4k qs is {} words a row; {nb} blocks want {}",
            self.qs.shape[self.qs.ndim() - 1],
            nb * QK_K / 8
        );
        assert_eq!(self.scales.numel(), o * nb * 12, "q4k scales size");
        assert_eq!(self.dm.numel(), o * nb * 2, "q4k dm size");
    }
}

/// `dequantize_row_q4_K` over the whole matrix: `[O, I]` in `dtype`.
///
/// One thread per 256-weight super-block, which is the shape the reference
/// loop has. Used by the tests and by anything that wants the weights back;
/// the model path never calls it.
pub fn dequantize(ctx: &Context, w: &Q4kMatrix, dtype: Dtype) -> Array {
    let (o, k) = (w.out_dims(), w.in_dims());
    let out = Array::empty(ctx, &[o, k], dtype);
    let blocks = o * k / QK_K;
    let tg = 256.min(blocks.next_power_of_two()).max(1);
    ctx.kernel(&format!("q4k_dequantize_{}", type_name(dtype)))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .out(&out)
        .threads((blocks, 1, 1), (tg, 1, 1));
    out
}

/// `x @ w.T` for a single row of `x`: ggml's `kernel_mul_mv_q4_K_f32_impl`.
///
/// `x` is `[.., 1, I]` or `[I]`; the result is `[.., 1, O]`, keeping the
/// caller's leading shape the way `quantized_matmul` does.
pub fn qmv(ctx: &Context, x: &Array, w: &Q4kMatrix) -> Array {
    let (o, k) = (w.out_dims(), w.in_dims());
    assert_eq!(x.shape[x.ndim() - 1], k, "q4k qmv: x is {k}-wide?");
    assert_eq!(x.numel(), k, "q4k qmv is the M = 1 path only");
    assert_eq!(k % QK_K, 0, "q4k qmv: {k} is not a multiple of {QK_K}");
    // The kernel's tail guard covers a short last row, but a threadgroup that
    // starts past the end would still walk the weight tensors, so refuse the
    // shape instead of reading off the end of the buffer.
    assert_eq!(
        o % (NSG * NR0),
        0,
        "q4k qmv: {o} rows is not a multiple of {}",
        NSG * NR0
    );

    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    let out = Array::empty(ctx, &shape, x.dtype);
    ctx.kernel(&format!("q4k_qmv_{}", type_name(x.dtype)))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .arr(x)
        .out(&out)
        .bytes(&(k as i32))
        .bytes(&(o as i32))
        .groups((o / (NSG * NR0), 1, 1), (32, NSG, 1));
    out
}

fn type_name(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no q4k kernel for {}", d.name()),
    }
}
