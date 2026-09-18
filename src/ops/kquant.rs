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
//! **Two kernels each: a mat-vec and a tiled gemm.** [`qmv`] is ggml's
//! `kernel_mul_mv_*_f32_impl` and it is the decode shape -- one output row,
//! the weights read once. It also carries ggml's token axis, so it *can* run
//! at M > 1, as a mat-vec per token that re-reads the whole matrix for each
//! one; that is what let a k-quant artifact be measured at all before a gemm
//! existed, and `HANDOFF.md` put a number on what it cost at prefill.
//! [`qmm_t`] is ggml's `kernel_mul_mm`, the 64x32 tile, and [`matmul`] picks
//! between them. The two are held to each other: the gemm's contract is that
//! it agrees with the mat-vec to within the difference in summation order,
//! which is why its accumulator tile is float rather than ggml's half.

use crate::array::{Array, Dtype};
use crate::metal::Context;

/// ggml's `QK_K`: weights per super-block, the same for both formats.
pub const QK_K: usize = 256;
/// `N_SG_*` and `N_R0_*` from `ggml-metal-impl.h`: simdgroups per threadgroup,
/// and output rows per simdgroup. Both formats use 2 and 2.
const NSG: usize = 2;
const NR0: usize = 2;

/// The gemm tile, from the legacy branch of ggml's `kernel_mul_mm`: 64 output
/// features by 32 rows, four simdgroups, 128 threads. `kernels/kquant_mm.h`
/// holds the same numbers and the two must agree.
const MM_NR0: usize = 64;
const MM_NR1: usize = 32;
const MM_NK: usize = 32;
const MM_THREADS: usize = 128;
/// A tile of A plus a tile of B, both in the float the kernel multiplies in.
/// The output staging reuses the same allocation and is smaller (64 x 32
/// floats), so this is the maximum of the two.
const MM_SHMEM: usize = 4 * (MM_NR0 * MM_NK + MM_NR1 * MM_NK);

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

/// `x @ w.T` at M > 1: ggml's `kernel_mul_mm`, the 64x32 tile.
///
/// This is the kernel [`qmv`]'s doc comment owes an apology to. A mat-vec per
/// token re-reads the whole weight matrix once per token; a tile reads it once
/// per 32 of them, which is invisible while the matrix fits in L2 and is not
/// once it does not -- `HANDOFF.md` measured level at 600 prompt tokens and
/// -8% / -21% at 3060.
///
/// `x` is `[.., M, I]` and the result `[.., M, O]`, the same contract [`qmv`]
/// has. Every shape is bounds-checked inside the tile, so unlike [`qmv`] this
/// wants no divisibility of `O`.
pub fn qmm_t(ctx: &Context, x: &Array, w: &KMatrix) -> Array {
    let (o, k) = (w.out_dims(), w.in_dims());
    assert_eq!(x.shape[x.ndim() - 1], k, "{} qmm_t: x is {k}-wide?", w.kind.name());
    assert_eq!(k % QK_K, 0, "{} qmm_t: {k} is not a multiple of {QK_K}", w.kind.name());
    assert_eq!(
        w.num_experts(),
        1,
        "{} qmm_t: a routed weight goes through the gather",
        w.kind.name()
    );
    let m = x.numel() / k;

    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    let out = Array::empty(ctx, &shape, x.dtype);
    ctx.kernel(&w.kernel("qmm_t", x.dtype))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .arr(x)
        .out(&out)
        .bytes(&(k as i32))
        .bytes(&(o as i32))
        .bytes(&(m as i32))
        .threadgroup_memory(MM_SHMEM, 0)
        .groups((m.div_ceil(MM_NR1), o.div_ceil(MM_NR0), 1), (MM_THREADS, 1, 1));
    out
}

/// `mx.gather_qmm(..., transpose=True)` at M > 1: ggml's `kernel_mul_mm_id`,
/// tiled the way mlx's `gather_qmm_rhs` is.
///
/// Output row `r` is `x[lhs[r]] @ w[rhs[r]].T`, which is [`gather_qmv`]'s
/// contract unchanged. The tile walks its own 32 rows of `rhs`, finds each run
/// of one expert and runs the whole K loop once per run, so the weight traffic
/// is one read per *run* where the mat-vec pays one per row. It is therefore
/// never worse than the mat-vec in weight traffic and is much better when the
/// rows are grouped -- which `switch_glu_sorted` makes them, at every batch of
/// 64 rows or more.
pub fn gather_qmm_t(
    ctx: &Context,
    x: &Array,
    w: &KMatrix,
    lhs_indices: &Array,
    rhs_indices: &Array,
) -> Array {
    let (o, k) = (w.out_dims(), w.in_dims());
    assert_eq!(x.dim(-1), k, "{} gather gemm: x is {k}-wide?", w.kind.name());
    assert_eq!(k % QK_K, 0, "{} gather gemm: {k} is not a multiple of {QK_K}", w.kind.name());
    let rows = rhs_indices.numel();
    assert_eq!(lhs_indices.numel(), rows, "{} gather gemm: index arrays disagree", w.kind.name());

    let out = Array::empty(ctx, &[rows, o], x.dtype);
    ctx.kernel(&w.kernel("gather_qmm_t", x.dtype))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .arr(x)
        .out(&out)
        .bytes(&(k as i32))
        .bytes(&(o as i32))
        .bytes(&(rows as i32))
        .arr(lhs_indices)
        .arr(rhs_indices)
        .threadgroup_memory(MM_SHMEM, 0)
        .groups((rows.div_ceil(MM_NR1), o.div_ceil(MM_NR0), 1), (MM_THREADS, 1, 1));
    out
}

/// How many rows it takes before the tile beats the mat-vec.
///
/// mlx decides this per shape (`qmv_batch_limit` weighs the weight read
/// against the activation read and lands at 13-15 for this model's linears).
/// The k-quants have no such calibration behind them, so this is one number,
/// and it is measured rather than guessed -- `src/bin/kquant_mm.rs` times both
/// kernels at M = 1..64 on the four shapes the two formats are used at, best
/// of forty windows of thirty dispatches:
///
///   shape                     M=1    M=4    M=6    M=8   M=16   crossover
///   in_proj_qkv [8192, 2048]  3.91   1.65   1.13   0.86   0.44    6 -> 8
///   out_proj    [2048, 4096]  4.62   1.74   1.22   0.94   0.50    6 -> 8
///   q_proj      [2048, 2048]  2.89   1.11   0.80   0.63   0.33    4 -> 6
///   down_proj   [2048,  512]  1.81   0.93   0.84   0.67   0.40    2 -> 4
///
/// (tile time over mat-vec time; below 1 the tile wins.) The tile's own time
/// is *flat* from M = 1 to M = 32 -- it does a 32-row tile's work whether or
/// not the rows are there -- so the ratio is the mat-vec's linear growth
/// divided by a constant, and the crossover is where that line passes it.
///
/// **8, which is deliberately the conservative end of 4 to 8.** It is right
/// for the two widest shapes and leaves the two narrow ones a little on the
/// mat-vec, and it keeps the speculative verify window (`depth + 1` = 4 rows)
/// exactly where it was, on a path this change has no business moving.
const MM_ROW_LIMIT: usize = 8;

/// `ORNITH_KQUANT_GEMM=0` sends every shape back through the mat-vec.
///
/// The gemm replaces a kernel that works, so the question it has to answer is
/// not "is it fast" but "is it faster than what it replaced, on this machine,
/// in this thermal state". A flag makes that an A/B inside one process instead
/// of two runs an hour apart -- which matters here, where `HANDOFF.md` records
/// identical traffic measuring 110.4 and 124.2 tok/s on different days.
fn gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ORNITH_KQUANT_GEMM").as_deref() != Ok("0"))
}

/// `x @ w.T` for whatever M the caller has: [`qmv`] at a decode step's one
/// row, [`qmm_t`] at a prefill chunk's thousands.
pub fn matmul(ctx: &Context, x: &Array, w: &KMatrix) -> Array {
    let m = x.numel() / w.in_dims();
    // `qmv` fixes four output rows per threadgroup, so a matrix whose O is not
    // a multiple of four has to take the tile, which bounds-checks. No module
    // in this artifact is such a shape, but the fallback is free to state.
    if (m >= MM_ROW_LIMIT && gemm_enabled()) || w.out_dims() % (NSG * NR0) != 0 {
        qmm_t(ctx, x, w)
    } else {
        qmv(ctx, x, w)
    }
}

/// The routed pair of [`matmul`].
///
/// `sorted` is the caller's claim that the rows are grouped by expert. Unlike
/// mlx's it is not a correctness precondition -- [`gather_qmm_t`] finds its own
/// runs and is right either way -- it only says whether the runs will be long
/// enough to be worth the tile. The thresholds are mlx's own, from
/// `GatherQMM::eval_gpu`.
pub fn gather_matmul(
    ctx: &Context,
    x: &Array,
    w: &KMatrix,
    lhs_indices: &Array,
    rhs_indices: &Array,
    sorted: bool,
) -> Array {
    let rows = rhs_indices.numel();
    let experts = w.num_experts();
    if (rows >= 16 && sorted && rows / experts >= 4 && gemm_enabled())
        || w.out_dims() % (NSG * NR0) != 0
    {
        gather_qmm_t(ctx, x, w, lhs_indices, rhs_indices)
    } else {
        gather_qmv(ctx, x, w, lhs_indices, rhs_indices)
    }
}

fn type_name(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no k-quant kernel for {}", d.name()),
    }
}
