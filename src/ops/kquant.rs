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
use crate::ops::invariant;

/// ggml's `QK_K`: weights per super-block, the same for both formats.
pub const QK_K: usize = 256;
/// `N_SG_*` and `N_R0_*` from `ggml-metal-impl.h`: simdgroups per threadgroup,
/// and output rows per simdgroup. Both formats use 2 and 2.
const NSG: usize = 2;
const NR0: usize = 2;

/// The gemm tile, from the legacy branch of ggml's `kernel_mul_mm`: 64 output
/// features by [`mm_rows`] rows, four simdgroups, 128 threads.
/// `kernels/kquant_mm.h` holds the same numbers and the two must agree.
const MM_NR0: usize = 64;
const MM_NK: usize = 32;
const MM_THREADS: usize = 128;

/// Rows per tile. ggml's is 32; `ORNITH_KQUANT_TILE_ROWS=16` is the narrow
/// instantiation, and both are compiled.
///
/// The tile does a full tile's work whether the rows are there or not --
/// `src/bin/kquant_mm.rs` measures its time as flat from M = 1 to 32 -- so the
/// row count is not a free parameter, it is the tile's cost below a full
/// batch. It is also what decides how often a routed run of ~64 rows an expert
/// straddles a tile boundary, and every straddle is one more pass over K.
static MM_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static MM_ROWS_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn rows_from(var: &str, cell: &std::sync::atomic::AtomicUsize, default: usize) -> usize {
    use std::sync::atomic::Ordering;
    match cell.load(Ordering::Relaxed) {
        0 => {
            let n = match std::env::var(var).as_deref() {
                Ok("16") => 16,
                Ok("32") => 32,
                Ok("64") => 64,
                Err(_) => default,
                Ok(other) => {
                    panic!("{var}={other}: only 16, 32 and 64 are instantiated")
                }
            };
            cell.store(n, Ordering::Relaxed);
            n
        }
        n => n,
    }
}

/// Rows per tile on the **dense** path, `ORNITH_KQUANT_TILE_ROWS`.
///
/// The tile's cost is a fixed part plus a part per row -- `src/bin/kquant_mm.rs`
/// separates them by timing two widths, and the fixed part is 32-60% of a full
/// 32-row block, because the A side is dequantized once per block whatever the
/// rows are. So a *wider* tile amortises it, and 64 rows measures 5-7% faster
/// than 32 at every prefill width. (It is also why the tile's time is flat
/// from M = 1 to its width: below a full block the fixed part is all there is.)
pub fn mm_rows() -> usize {
    rows_from("ORNITH_KQUANT_TILE_ROWS", &MM_ROWS, 64)
}

/// Rows per tile on the **routed** path.
///
/// A different number from the dense one, and not a constant, because the
/// routed tile pays a second cost: it runs the whole K loop once per maximal
/// run of one expert inside its own rows, so a tile wider than the runs
/// straddles boundaries and pays for it. `src/bin/kquant_mm.rs` times that
/// directly -- at runs of 16 rows a 32-row tile takes 2.1x its aligned time,
/// and a 64-row tile 1.9x a 32-row one.
///
/// So the width follows the average run, which is `rows / experts` -- the same
/// quantity mlx's own `GatherQMM::eval_gpu` thresholds on. Measured end to
/// end, GPU seconds on `q6k-full`, two draws each:
///
///   prompt   rows an expert   tile 16    tile 32
///     748          23        **0.77**     0.80
///    1887          59          1.72     **1.70**
///    3402          64          3.16     **3.13**
///
/// 64 is never the answer here even when the mean run is 64, because the
/// distribution has short runs in it: `routed = 64` measures 3.43 s against
/// 3.13 at the same prompt. `ORNITH_KQUANT_TILE_ROWS_ID` pins a width.
pub fn mm_rows_id(rows: usize, experts: usize) -> usize {
    match MM_ROWS_ID.load(std::sync::atomic::Ordering::Relaxed) {
        0 => {
            if let Ok(v) = std::env::var("ORNITH_KQUANT_TILE_ROWS_ID") {
                let n = rows_from("ORNITH_KQUANT_TILE_ROWS_ID", &MM_ROWS_ID, 32);
                let _ = v;
                return n;
            }
            if experts > 0 && rows / experts < 32 { 16 } else { 32 }
        }
        n => n,
    }
}

/// Rows of `x` one dispatch of the mat-vec carries, `ORNITH_QMV_ROWS`.
///
/// 1 turns the multi-row kernel off and sends every width back through
/// `q*_qmv` one row at a time, which is what the port did before Phase 28 and
/// what the A/B is against; 2..4 pin a width. The default is 4, which is a
/// verify window at the deepest depth `sdpa` will fuse.
///
/// Worth, end to end on `q6k-full`, 200 tokens, greedy, two draws each and
/// the acceptance byte-identical between them (GPU ms a token):
///
///   config                  a row each   this kernel
///   --mtp-depth 1 --frspec   6.78/6.80   **6.60/6.60**
///   --mtp-depth 3 --frspec   8.72/8.72   **8.28/8.30**
///
/// -2.8% and -5.0%, against the -12% to -21% Phase 27 priced it at.
///
/// Instantiated widths only -- a kernel per width is how the row loop unrolls,
/// and `MW` past four runs out of registers for the activations it holds.
static QMV_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn qmv_rows() -> usize {
    use std::sync::atomic::Ordering;
    match QMV_ROWS.load(Ordering::Relaxed) {
        0 => {
            let n = match std::env::var("ORNITH_QMV_ROWS").as_deref() {
                Ok("1") => 1,
                Ok("2") => 2,
                Ok("3") => 3,
                Err(_) | Ok("4") => 4,
                Ok(other) => panic!("ORNITH_QMV_ROWS={other}: 1 (off) to 4"),
            };
            QMV_ROWS.store(n, Ordering::Relaxed);
            n
        }
        n => n,
    }
}

/// `ORNITH_QMV_HOIST=0`: go back to unpacking the codes once a row.
///
/// The second half of the multi-row kernel, and the one the measurement asked
/// for: at MW rows the kernel is arithmetic-bound, and what repeats per row
/// is the dequantize rather than the read.
///
/// **Q6_K only, and that is measured rather than assumed.** Both kernels
/// exist and both are bit-exact; inside a width-4 verify, per kernel, with
/// `ORNITH_COMMIT_EVERY=1`:
///
///   ms a verify        a row each   one read   + unpacked
///   q6k dense mat-vec      2.118      2.505      **1.677**
///   q4k dense mat-vec      5.297    **5.220**      5.789
///
/// Q6_K unpacks five operations a weight (two masks, a shift, an or and the
/// bias) against the one multiply-add that spends it, so hoisting it out of
/// the row loop is most of the inner loop. Q4_K unpacks one mask and one
/// conversion, and the hoist costs it what it saves: holding 32 unpacked
/// weights forces the output row outside the row-of-`x` loop, which doubles
/// the activation reads to one a multiply-add. So `ORNITH_QMV_HOIST` applies
/// to Q6_K and Q4_K stays on the shared read alone.
static QMV_HOIST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Which formats unpack once a block: none, Q6_K, or both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hoist {
    Off,
    /// The default, and the only one that measures a win.
    Q6K,
    /// Q4_K too, which is 9% slower and is kept so the next session can
    /// re-measure it instead of re-deriving it.
    All,
}

pub fn qmv_hoist() -> Hoist {
    use std::sync::atomic::Ordering;
    match QMV_HOIST.load(Ordering::Relaxed) {
        0 => {
            let h = match std::env::var("ORNITH_QMV_HOIST").as_deref() {
                Ok("0") => Hoist::Off,
                Ok("all") => Hoist::All,
                Err(_) | Ok("1") => Hoist::Q6K,
                Ok(other) => panic!("ORNITH_QMV_HOIST={other}: 0, 1 or all"),
            };
            QMV_HOIST.store(h as u8 + 1, Ordering::Relaxed);
            h
        }
        1 => Hoist::Off,
        2 => Hoist::Q6K,
        _ => Hoist::All,
    }
}

pub fn set_qmv_hoist(h: Hoist) {
    QMV_HOIST.store(h as u8 + 1, std::sync::atomic::Ordering::Relaxed);
}

/// The same for everything dispatched after this call, so one process can
/// measure both kernels -- the only honest A/B on a machine whose wall clock
/// swings between sittings.
pub fn set_qmv_rows(rows: usize) {
    assert!((1..=4).contains(&rows), "only 1 (off) to 4 are instantiated");
    QMV_ROWS.store(rows, std::sync::atomic::Ordering::Relaxed);
}

/// `ORNITH_KQUANT_TILE_HALF=1`: ggml's own tile scalar instead of this port's.
///
/// ggml dequantizes into `half` and multiplies half by half; this port
/// instantiates the same kernel at `float` because a gemm that replaces a
/// checked kernel owes agreement with it, and eleven significand bits is a
/// 5e-4 perturbation where reassociation is 2e-6 (`kernels/kquant_mm.h`).
/// That was a decision taken on a contract and never priced. **Priced now, and
/// it was free**: 0-4% on every shape, which is inside the measurement. The
/// flag stays so the next person does not have to ask again.
pub fn mm_half() -> bool {
    use std::sync::atomic::Ordering;
    match MM_HALF.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("ORNITH_KQUANT_TILE_HALF").as_deref() == Ok("1");
            MM_HALF.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        n => n == 2,
    }
}

static MM_HALF: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_mm_half(on: bool) {
    MM_HALF.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

/// The kernel-name suffix a width and a scalar produce.
fn mm_variant(rows: usize) -> String {
    format!("r{}{}", rows, if mm_half() { "h" } else { "" })
}

/// Pick the tile width for everything dispatched after this call.
///
/// All three widths are compiled, so the A/B runs inside one process -- the
/// only way to compare two kernels on a machine whose wall clock swings 3x
/// between sittings while its GPU seconds hold to 1% within one.
pub fn set_mm_rows(rows: usize) {
    assert!(matches!(rows, 16 | 32 | 64), "only 16, 32 and 64 are instantiated");
    MM_ROWS.store(rows, std::sync::atomic::Ordering::Relaxed);
}

/// The same, for the routed tile.
pub fn set_mm_rows_id(rows: usize) {
    assert!(matches!(rows, 16 | 32 | 64), "only 16, 32 and 64 are instantiated");
    MM_ROWS_ID.store(rows, std::sync::atomic::Ordering::Relaxed);
}

/// A tile of A plus a tile of B, both in the float the kernel multiplies in.
/// The output staging reuses the same allocation and is smaller (64 x rows
/// floats), so this is the maximum of the two.
fn mm_shmem(rows: usize) -> usize {
    4 * (MM_NR0 * MM_NK + rows * MM_NK)
}

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
///
/// **At a speculative verify window it is neither free nor prefill, and
/// Phase 28 is what that turned out to be worth.** Phase 27 measured a
/// verify row at 3.4 ms on `q6k-full`, 41% of it this kernel, with the
/// dispatch count fixed and the time 2.8x at four rows -- and read that as
/// the weights being read once per row. The kernel that reads them once for
/// all M rows (`q*_qmv_m{2,3,4}`, [`qmv_rows`]) is bit-exact against this one
/// and measures **~0 on the verify**: 3.14 ms a row against 3.14.
///
/// The re-read was never the binding cost. On `lm_head`, dispatched one at a
/// time, a single row reads 286 MB in 0.550 ms -- 520 GB/s, this machine's
/// ceiling -- and four rows through the shared-read kernel read the same
/// 286 MB in 1.488, which is 192 GB/s. Past the first row the kernel is
/// **arithmetic**-bound, and the three reads it saves were already hidden
/// under the arithmetic that consumes them. What repeats per row and is not
/// hidden is the *dequantize*, which is why the kernel that pays is the one
/// that hoists that ([`qmv_hoist`]) and why it pays only on Q6_K.
///
/// The other three candidates are still refuted: [`qmm_t`] reads the matrix
/// once but its floor is 7.5x a mat-vec ([`MM_ROW_LIMIT`]), mlx's `qmv_wide`
/// shares a read between five rows at 0.57x and a different association, and
/// the grid axis order measured 1.00.
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

    // The M axis on the grid re-reads the matrix per row; the M axis inside
    // the kernel reads it once. Same arithmetic per row either way, so the
    // choice is free of the invariance question -- see `qmv_rows`.
    let mw = qmv_rows().min(m);
    let hoisted = match qmv_hoist() {
        Hoist::Off => false,
        Hoist::Q6K => w.kind == KQuant::Q6K,
        Hoist::All => true,
    };
    let name = match (mw > 1, hoisted) {
        (false, _) => "qmv".to_string(),
        (true, false) => format!("qmv_m{mw}"),
        (true, true) => format!("qmv_mh{mw}"),
    };
    let call = ctx
        .kernel(&w.kernel(&name, x.dtype))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .arr(x)
        .out(&out)
        .bytes(&(k as i32))
        .bytes(&(o as i32));
    if mw > 1 {
        // The last group covers a short tail by repeating the last row, which
        // the kernel computes and does not store.
        call.bytes(&(m as i32)).groups((o / (NSG * NR0), m.div_ceil(mw), 1), (32, NSG, 1));
    } else {
        call.groups((o / (NSG * NR0), m, 1), (32, NSG, 1));
    }
    out
}

/// `Qwen3NextRMSNormGated` and the projection that reads it, in one dispatch.
///
/// Returns `None` when this is not the shape the fused kernel is written for,
/// and the caller then issues the norm and the matvec separately. The
/// conditions are Q4_K, one row (a decode step or nothing), a head of exactly
/// `32 * RMS_N_READS` so the norm's reduction is one simdgroup, and an
/// activation row that fits the prologue's threadgroup buffer.
///
/// `REBUILD.md` Phase 33: the pair it replaces is 4.8 us a dispatch, three
/// times the elementwise chain's average, and gluing the two elementwise ops
/// together instead measures zero -- this is the only other way to that time,
/// and it is the deferred op layer's mechanism tried on the one norm in the
/// model that has a single consumer.
pub fn rmsgate_qmv(
    ctx: &Context,
    x: &Array,
    w: &KMatrix,
    norm_w: &Array,
    gate: &Array,
    head_dim: usize,
    eps: f32,
) -> Option<Array> {
    const MAX_ROW: usize = 4096; // ORNITH_RMSGATE_MAX in kernels/ornith_rmsgate.h
    if !fuse_norm_proj() {
        return None;
    }
    let (o, k) = (w.out_dims(), w.in_dims());
    let m = x.numel() / k;
    if w.kind != KQuant::Q4K
        || m != 1
        || k > MAX_ROW
        || head_dim != 32 * 4
        || k % head_dim != 0
        || k % QK_K != 0
        || o % (rmsgate_nsg() * rmsgate_nr0()) != 0
        || x.dtype != Dtype::BF16
        || norm_w.numel() != head_dim
        || gate.numel() != x.numel()
    {
        return None;
    }
    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    let out = Array::empty(ctx, &shape, x.dtype);
    let (nr0, nsg) = (rmsgate_nr0(), rmsgate_nsg());
    let name = match nr0 {
        2 => "rmsgate_qmv".to_string(),
        n => format!("rmsgate_qmv_r{n}"),
    };
    ctx.kernel(&w.kernel(&name, x.dtype))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .arr(x)
        .out(&out)
        .bytes(&(k as i32))
        .bytes(&(o as i32))
        .arr(norm_w)
        .arr(gate)
        .bytes(&(head_dim as i32))
        .bytes(&eps)
        .groups((o / (nsg * nr0), 1, 1), (32, nsg, 1));
    Some(out)
}

/// Output rows a threadgroup of the fused norm+projection takes, which is
/// the only knob on how many times its prologue normalises the same row:
/// `O / (NSG * nr0)` threadgroups each do it once. 2 is ggml's shape for the
/// plain matvec and 512 normalisations for `out_proj`; 16 is 32 of them.
/// 4 is the measured minimum of the four: 512 threadgroups at ggml's 2 cost
/// +0.304 ms, 128 cost +0.079, 64 cost +0.188 and 32 cost +0.468 -- the
/// prologue gets cheaper as the count falls and the mat-vec gets worse.
static RMSGATE_NR0: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4);

pub fn rmsgate_nr0() -> usize {
    if let Ok(v) = std::env::var("ORNITH_RMSGATE_NR0") {
        if let Ok(n) = v.parse() {
            RMSGATE_NR0.store(n, std::sync::atomic::Ordering::Relaxed);
        }
    }
    RMSGATE_NR0.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn rmsgate_nsg() -> usize {
    if rmsgate_nr0() == 2 { 2 } else { 4 }
}

/// Set the fused kernel's rows per simdgroup; returns what it was.
pub fn set_rmsgate_nr0(n: usize) -> usize {
    let was = rmsgate_nr0();
    RMSGATE_NR0.store(n, std::sync::atomic::Ordering::Relaxed);
    was
}

/// Whether `RMSNormGated` rides into the projection that reads it, or is its
/// own dispatch.
///
/// **Off, and it is the control that says why** -- the other way round from
/// every other switch here, because this one lost. `REBUILD.md` Phase 34:
/// at its best of four widths it costs **+0.079 ms** against the two
/// dispatches it replaces, whose entire removal is worth 0.145. A mat-vec
/// parallelises over *output rows* and every threadgroup reads the whole
/// input, so folding a producer into it is not fusing it once, it is
/// repeating it once per threadgroup. `ORNITH_FUSE_NORM_PROJ=1` turns it on.
static FUSE_NORM_PROJ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FUSE_NORM_PROJ_ENV: std::sync::Once = std::sync::Once::new();

pub fn fuse_norm_proj() -> bool {
    FUSE_NORM_PROJ_ENV.call_once(|| {
        if std::env::var("ORNITH_FUSE_NORM_PROJ").as_deref() == Ok("1") {
            FUSE_NORM_PROJ.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    FUSE_NORM_PROJ.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turn the fusion on or off at runtime; returns what it was.
pub fn set_fuse_norm_proj(on: bool) -> bool {
    let was = fuse_norm_proj();
    FUSE_NORM_PROJ.store(on, std::sync::atomic::Ordering::Relaxed);
    was
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
///
/// A verify window is where that costs: four tokens send 32 rows through
/// here, of which `tools/router_stats.py` measures ~21 distinct, and the
/// kernel pays for all 32 -- 30% of what a window row costs (`REBUILD.md`
/// Phase 27). The routed tile is worse rather than better at that shape,
/// since it runs its K loop once per run of one expert and a window's runs
/// are one row long.
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
    let rows = mm_rows();
    ctx.kernel(&w.kernel(&format!("qmm_t_{}", mm_variant(rows)), x.dtype))
        .arr(&w.qs)
        .arr(&w.scales)
        .arr(&w.dm)
        .arr(x)
        .out(&out)
        .bytes(&(k as i32))
        .bytes(&(o as i32))
        .bytes(&(m as i32))
        .threadgroup_memory(mm_shmem(rows), 0)
        .groups((m.div_ceil(rows), o.div_ceil(MM_NR0), 1), (MM_THREADS, 1, 1));
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
    let tile_rows = mm_rows_id(rows, w.num_experts());
    assert!(
        !(mm_half() && tile_rows != 32),
        "the half scalar is instantiated at 32 rows only"
    );
    ctx.kernel(&w.kernel(&format!("gather_qmm_t_{}", mm_variant(tile_rows)), x.dtype))
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
        .threadgroup_memory(mm_shmem(tile_rows), 0)
        .groups((rows.div_ceil(tile_rows), o.div_ceil(MM_NR0), 1), (MM_THREADS, 1, 1));
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
/// **16, and it moved with the tile.** It was 8 when the dense tile was 32
/// rows wide; a 64-row tile costs 1.7x a 32-row one below a full block, so the
/// crossover moved with it -- 13 and 14 rows on the two wide shapes, 8 and 9
/// on the narrow ones. 16 is where the tile wins on all four, which is the
/// same conservative rule the old 8 was picked by, and it still leaves the
/// speculative verify window (`depth + 1` = 4 rows) on the mat-vec.
///
/// That last clause reads like a happy accident and is the right call for a
/// reason Phase 28 had to measure: the tile is 2x the mat-vec at four rows,
/// and the mat-vec's four reads of the matrix -- which looked like the thing
/// to fix -- cost almost nothing, because the kernel is arithmetic-bound
/// there. See [`qmv`].
const MM_ROW_LIMIT: usize = 16;

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
    if (m >= MM_ROW_LIMIT && gemm_enabled() && !invariant::for_rows(m) && !invariant::all())
        || w.out_dims() % (NSG * NR0) != 0
    {
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
    if (rows >= 16 && sorted && rows / experts >= 4 && gemm_enabled() && !invariant::all())
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
