//! ggml's k-quants: the dispatch side of `kernels/q4k.metal` and
//! `kernels/q6k.metal`.
//!
//! These are the only modes in the port with no mlx behind them. mlx has four
//! quantization modes and none is a k-quant, and the lead maintainer closed
//! the request for one -- `Q4K.md` quotes it -- so there is no
//! `quantized_matmul` to port and no `max_abs == 0.0` against MLX to be had.
//! What replaces it is `ggml`, which implements exactly these formats and did
//! not learn them from us: [`dequantize`] reproduces `dequantize_row_q4_K` and
//! `dequantize_row_q6_K` bit for bit on bytes ggml's own quantizers wrote, and
//! `tests/q4k.rs` asserts it.
//!
//! **Three tensors, not one packed blob.** ggml keeps a packed block -- 144
//! bytes for Q4_K, 210 for Q6_K -- and the artifact keeps the codes, the
//! sub-block fields and the super-scales as separate tensors, so the
//! safetensors shapes stay the ones the loader and `check_rebuild.py` already
//! understand, the way nvfp4's two-tensor modules did. The bytes inside each
//! are ggml's own, so only the addressing differs.
//!
//!   Q4_K   qs      uint32  [O, I / 8]        the 4-bit codes, 128 B a block
//!          scales  uint8   [O, I / 256, 12]  8 x 6-bit scale + 8 x 6-bit min
//!          dm      float16 [O, I / 256, 2]   d and dmin
//!
//!   Q6_K   qs      uint32  [O, I * 3 / 16]   `ql` then `qh`, 192 B a block
//!          scales  uint8   [O, I / 256, 16]  one *signed* scale per 16
//!          dm      float16 [O, I / 256, 1]   d alone -- Q6_K has no minimum
//!
//! A routed expert module carries the same three with an `[E, ...]` axis in
//! front, exactly as the affine ones do, and [`gather_qmv`] indexes it.
//!
//! **One kernel each, and it is a mat-vec.** `Q4K.md` scopes the first cut to
//! M = 1, because `affine_qmv_fast` is 38% of a decode step and that alone is
//! enough to measure GB/token and GPU busy for real. ggml's kernels also carry
//! a token axis, and [`qmv`] keeps it: at M > 1 it is a mat-vec per token,
//! re-reading the weights once each rather than tiling them. That is the wrong
//! shape for prefill and the reason a k-quant artifact can be *run* at all
//! before `qmm_t` exists -- without it there is no prefill, and without a
//! prefill there is no probe capture and no verdict.

use crate::array::{Array, Dtype};
use crate::metal::Context;

/// ggml's `QK_K`: weights per super-block, the same for both formats.
pub const QK_K: usize = 256;
/// `N_SG_*` and `N_R0_*` from `ggml-metal-impl.h`: simdgroups per threadgroup,
/// and output rows per simdgroup. Both formats use 2 and 2.
const NSG: usize = 2;
const NR0: usize = 2;

/// Which k-quant a weight is stored in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KQuant {
    /// 4.50 bits: 6-bit integer scale *and minimum* per 32, two fp16
    /// super-scales per 256. `w = d*sc*q - dmin*m`, both fields unsigned.
    Q4K,
    /// 6.5625 bits: a *signed* 8-bit scale per 16 and one fp16 super-scale per
    /// 256, with the code biased by 32. `w = d*sc*(q - 32)`; no minimum.
    Q6K,
}

impl KQuant {
    /// The `mode` string `config.json` carries and the kernel-name prefix.
    pub fn name(self) -> &'static str {
        match self {
            KQuant::Q4K => "q4k",
            KQuant::Q6K => "q6k",
        }
    }

    /// Bytes of codes per super-block: Q4_K's `qs`, Q6_K's `ql` plus `qh`.
    fn code_bytes(self) -> usize {
        match self {
            KQuant::Q4K => 128,
            KQuant::Q6K => 192,
        }
    }

    /// Sub-block fields per super-block, in bytes.
    fn scale_bytes(self) -> usize {
        match self {
            KQuant::Q4K => 12,
            KQuant::Q6K => 16,
        }
    }

    /// Super-scales per super-block: `d` and `dmin`, or `d` alone.
    fn super_scales(self) -> usize {
        match self {
            KQuant::Q4K => 2,
            KQuant::Q6K => 1,
        }
    }

    /// Bits per weight, for the byte arithmetic.
    pub fn bits(self) -> f64 {
        (self.code_bytes() + self.scale_bytes() + 2 * self.super_scales()) as f64 * 8.0
            / QK_K as f64
    }
}

/// A k-quantized weight matrix, as three tensors.
#[derive(Clone)]
pub struct KMatrix {
    pub kind: KQuant,
    /// uint32 -- ggml's codes, `code_bytes` per super-block.
    pub qs: Array,
    /// uint8 -- ggml's sub-block fields, `scale_bytes` per super-block.
    pub scales: Array,
    /// float16 -- the super-scales, `super_scales` per super-block.
    pub dm: Array,
}

impl KMatrix {
    pub fn new(kind: KQuant, qs: Array, scales: Array, dm: Array) -> KMatrix {
        let m = KMatrix { kind, qs, scales, dm };
        m.check();
        m
    }

    /// Output features: rows of the stored `[.., O, I]` matrix.
    pub fn out_dims(&self) -> usize {
        self.qs.shape[self.qs.ndim() - 2]
    }

    /// Input features. A super-block is 256 weights and `scales` carries one
    /// group of fields per super-block, so the block count is the tensor's own.
    pub fn in_dims(&self) -> usize {
        self.blocks_per_row() * QK_K
    }

    fn blocks_per_row(&self) -> usize {
        self.scales.shape[self.scales.ndim() - 2]
    }

    /// Experts for a 3-D (routed) weight, else 1 -- `QuantMatrix` counts the
    /// same way.
    pub fn num_experts(&self) -> usize {
        if self.qs.ndim() >= 3 { self.qs.shape[0] } else { 1 }
    }

    /// Everything the kernel assumes about the three tensors, checked once at
    /// load rather than read as garbage per dispatch.
    fn check(&self) {
        let (o, nb, e) = (self.out_dims(), self.blocks_per_row(), self.num_experts());
        let k = self.kind;
        assert_eq!(self.qs.dtype, Dtype::U32, "{} qs must be uint32", k.name());
        assert_eq!(self.scales.dtype, Dtype::U8, "{} scales must be uint8", k.name());
        assert_eq!(self.dm.dtype, Dtype::F16, "{} dm must be float16", k.name());
        assert_eq!(
            self.qs.shape[self.qs.ndim() - 1] * 4,
            nb * k.code_bytes(),
            "{} qs is {} bytes a row; {nb} blocks want {}",
            k.name(),
            self.qs.shape[self.qs.ndim() - 1] * 4,
            nb * k.code_bytes()
        );
        assert_eq!(self.qs.numel() * 4, e * o * nb * k.code_bytes(), "{} qs size", k.name());
        assert_eq!(self.scales.numel(), e * o * nb * k.scale_bytes(), "{} scales size", k.name());
        assert_eq!(self.dm.numel(), e * o * nb * k.super_scales(), "{} dm size", k.name());
    }

    fn kernel(&self, name: &str, dtype: Dtype) -> String {
        format!("{}_{name}_{}", self.kind.name(), type_name(dtype))
    }
}

/// `dequantize_row_*` over the whole matrix: `[.., O, I]` in `dtype`.
///
/// One thread per 256-weight super-block, which is the shape the reference
/// loop has. Used by the tests and by anything that wants the weights back;
/// the model path never calls it.
pub fn dequantize(ctx: &Context, w: &KMatrix, dtype: Dtype) -> Array {
    let (o, k, e) = (w.out_dims(), w.in_dims(), w.num_experts());
    // A routed weight keeps its `[E, O, I]` shape; the kernel is per block and
    // does not care, but the output has to have room for all of them.
    let shape = if e > 1 { vec![e, o, k] } else { vec![o, k] };
    let out = Array::empty(ctx, &shape, dtype);
    let blocks = e * o * k / QK_K;
    let tg = 256.min(blocks.next_power_of_two()).max(1);
    ctx.kernel(&w.kernel("dequantize", dtype))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .out(&out)
        .threads((blocks, 1, 1), (tg, 1, 1));
    out
}

/// `x @ w.T`: ggml's `kernel_mul_mv_*_f32_impl`.
///
/// `x` is `[.., M, I]`; the result is `[.., M, O]`, keeping the caller's
/// leading shape the way `quantized_matmul` does. M is the grid's y axis, as
/// in ggml -- a mat-vec per token rather than a tiled gemm, so the weights are
/// re-read once per token. At M = 1, the decode step, that costs nothing; at
/// prefill it is the price of not having written `qmm_t` yet, and it is paid
/// in time rather than in correctness.
pub fn qmv(ctx: &Context, x: &Array, w: &KMatrix) -> Array {
    let (o, k) = (w.out_dims(), w.in_dims());
    assert_eq!(x.shape[x.ndim() - 1], k, "{} qmv: x is {k}-wide?", w.kind.name());
    assert_eq!(k % QK_K, 0, "{} qmv: {k} is not a multiple of {QK_K}", w.kind.name());
    let m = x.numel() / k;
    // The kernel's tail guard covers a short last row, but a threadgroup that
    // starts past the end would still walk the weight tensors, so refuse the
    // shape instead of reading off the end of the buffer.
    assert_eq!(
        o % (NSG * NR0),
        0,
        "{} qmv: {o} rows is not a multiple of {}",
        w.kind.name(),
        NSG * NR0
    );

    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    let out = Array::empty(ctx, &shape, x.dtype);
    ctx.kernel(&w.kernel("qmv", x.dtype))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .arr(x)
        .out(&out)
        .bytes(&(k as i32))
        .bytes(&(o as i32))
        .groups((o / (NSG * NR0), m, 1), (32, NSG, 1));
    out
}

/// `mx.gather_qmm(x, w, rhs_indices=..., transpose=True)` at M = 1, which is
/// the only shape `SwitchGLU` produces -- ggml's `kernel_mul_mv_id`.
///
/// One output row per (token, expert) pair: row `r` reads `x[lhs[r]]` against
/// expert `rhs[r]`. `x` is `[R, I]` or `[T, I]` depending on whether the
/// caller sorted, `lhs`/`rhs` are `[R]`, and the result is `[R, O]`.
///
/// There is no rhs-sorted streaming variant here. mlx has one and takes it
/// when the rows outnumber the experts four to one, because it reads each
/// expert's weights once for a run of rows instead of once per row. At the
/// batch this model decodes -- 8 rows a token -- that path is unreachable
/// anyway, and at prefill it would be the difference between a mat-vec and a
/// gemm, which is the same debt `qmv` already owes at M > 1.
pub fn gather_qmv(
    ctx: &Context,
    x: &Array,
    w: &KMatrix,
    lhs_indices: &Array,
    rhs_indices: &Array,
) -> Array {
    let (o, k) = (w.out_dims(), w.in_dims());
    assert_eq!(x.dim(-1), k, "{} gather: x is {k}-wide?", w.kind.name());
    assert_eq!(k % QK_K, 0, "{} gather: {k} is not a multiple of {QK_K}", w.kind.name());
    assert_eq!(o % (NSG * NR0), 0, "{} gather: {o} rows is not a multiple of 4", w.kind.name());
    let rows = rhs_indices.numel();
    assert_eq!(lhs_indices.numel(), rows, "{} gather: index arrays disagree", w.kind.name());

    let out = Array::empty(ctx, &[rows, o], x.dtype);
    ctx.kernel(&w.kernel("gather_qmv", x.dtype))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .arr(x)
        .out(&out)
        .bytes(&(k as i32))
        .bytes(&(o as i32))
        .arr(lhs_indices)
        .arr(rhs_indices)
        .groups((o / (NSG * NR0), rows, 1), (32, NSG, 1));
    out
}

fn type_name(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no k-quant kernel for {}", d.name()),
    }
}
