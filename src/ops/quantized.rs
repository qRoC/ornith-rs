//! `mx.quantized_matmul` for affine quantization.
//!
//! Port of `QuantizedMatmul::eval_gpu` and the `qmv*` / `qmm*` dispatch helpers
//! in `vendor/mlx/mlx/backend/metal/quantized.cpp`. The kernels themselves are
//! mlx's, compiled from `kernels/mlx/.../quantized.h` via
//! `kernels/quantized_inst.metal`; what lives here is only the decision of
//! *which* kernel runs and with which grid.
//!
//! Affine layout, as produced by `mx.quantize(w, group_size, bits, "affine")`:
//!
//!   w:      [..., O, I * bits / 32]  uint32, `32 / bits` values per word,
//!                                    low bits first
//!   scales: [..., O, I / group_size] same dtype as the activations
//!   biases: [..., O, I / group_size]
//!
//! and a weight element is `scale * q + bias`. `transpose = true` means the
//! stored matrix is [O, I] and the product is `x @ w.T`, which is the layout
//! every linear layer in this model uses.

use crate::array::Array;
use crate::metal::Context;

/// A quantized weight matrix plus its quantization parameters.
#[derive(Clone)]
pub struct QuantMatrix {
    pub w: Array,
    pub scales: Array,
    pub biases: Array,
    pub group_size: usize,
    pub bits: usize,
}

impl QuantMatrix {
    /// Output features (rows of the stored [O, I] matrix).
    pub fn out_dims(&self) -> usize {
        self.w.shape[self.w.ndim() - 2]
    }

    /// Input features, recovered from the scales: mlx's
    /// `QuantizedLinear.input_dims` does the same.
    pub fn in_dims(&self) -> usize {
        self.scales.shape[self.scales.ndim() - 1] * self.group_size
    }

    /// Number of experts for a 3-D (gathered) weight, else 1.
    pub fn num_experts(&self) -> usize {
        if self.w.ndim() >= 3 { self.w.shape[0] } else { 1 }
    }

    fn suffix(&self, type_string: &str) -> String {
        format!("{}_gs_{}_b_{}", type_string, self.group_size, self.bits)
    }
}

/// `pack_factor<bits, 32>() * (bits == 2 ? 1 : 2) * SIMD_SIZE`, the K step of
/// `qmv_fast_impl`. Copied from `qmv_fast_k_alignment` in quantized.cpp.
fn qmv_fast_k_alignment(bits: usize) -> usize {
    let pack_factor = match bits {
        3 | 5 => 8,
        6 => 4,
        b => 32 / b,
    };
    pack_factor * if bits == 2 { 1 } else { 2 } * 32
}

/// `get_qmv_batch_limit` from quantized.cpp: above this many rows the matrix
/// kernels beat the vector kernels.
fn qmv_batch_limit(ctx: &Context, d: usize, o: usize) -> usize {
    let arch_size = ctx.arch.as_bytes().last().copied().unwrap_or(b'g');
    let arch_gen = ctx.arch_gen;
    if arch_gen >= 17 && arch_size != b'd' {
        if d <= 2048 && o <= 2048 {
            33
        } else if d <= 4096 && o <= 4096 {
            25
        } else {
            13
        }
    } else if arch_gen >= 15 && arch_size != b'd' {
        if d <= 2048 && o <= 2048 {
            13
        } else if d <= 4096 && o <= 4096 {
            15
        } else {
            13
        }
    } else if arch_gen >= 13 {
        match arch_size {
            b'd' => {
                if d <= 2048 && o <= 2048 {
                    32
                } else if d <= 4096 && o <= 4096 {
                    18
                } else {
                    12
                }
            }
            _ => {
                if d <= 2048 && o <= 2048 {
                    14
                } else if d <= 4096 && o <= 4096 {
                    10
                } else {
                    6
                }
            }
        }
    } else {
        11
    }
}

/// affine `qmv_wide` only beats `qmv` on gen-15 and up.
fn use_qmv_wide(ctx: &Context) -> bool {
    ctx.arch_gen >= 15
}

/// `x @ w.T` with `w` affine-quantized. `x` is [..., K], the result [..., N].
pub fn quantized_matmul(ctx: &Context, x: &Array, qw: &QuantMatrix) -> Array {
    assert_eq!(qw.w.ndim(), 2, "batched weights go through gather_qmm");
    let k = x.dim(-1);
    let n = qw.out_dims();
    assert_eq!(k, qw.in_dims(), "x has {k} columns, weight expects {}", qw.in_dims());
    let m = x.numel() / k;

    let mut out_shape = x.shape[..x.ndim() - 1].to_vec();
    out_shape.push(n);
    let out = Array::empty(ctx, &out_shape, x.dtype);

    let limit = qmv_batch_limit(ctx, k, n);
    if m >= limit {
        // B == 1 here (2-D weights), and every linear in this model is
        // transposed, so mlx takes the split-K path.
        qmm_splitk(ctx, x, qw, &out, m, n, k);
    } else {
        dispatch_qmv(ctx, x, qw, &out, m, n, k);
    }
    out
}

/// `dispatch_qmv` from quantized.cpp.
fn dispatch_qmv(ctx: &Context, x: &Array, qw: &QuantMatrix, out: &Array, m: usize, n: usize, k: usize) {
    assert!(
        k != 128 && k != 64,
        "K={k} would route to affine_qmv_quad in mlx; no layer of this model \
         has a 64- or 128-wide input, so that kernel is not instantiated"
    );
    if m >= 2 && use_qmv_wide(ctx) {
        qmv_wide(ctx, x, qw, out, m, n, k);
    } else {
        qmv(ctx, x, qw, out, m, n, k);
    }
}

/// `qmv` / `qmv_fast`: one output vector per threadgroup row.
fn qmv(ctx: &Context, x: &Array, qw: &QuantMatrix, out: &Array, m: usize, n: usize, k: usize) {
    let bn = 8; // 2 * results_per_simdgroup, with results_per_simdgroup = 4
    let bk = 32;
    let fast = n % bn == 0 && k % qmv_fast_k_alignment(qw.bits) == 0;
    let name = format!(
        "affine_{}_{}_batch_0",
        if fast { "qmv_fast" } else { "qmv" },
        qw.suffix(x.type_string())
    );
    ctx.kernel(&name)
        .arr(&qw.w)
        .arr(&qw.scales)
        .arr(&qw.biases)
        .arr(x)
        .out(out)
        .bytes(&(k as i32))
        .bytes(&(n as i32))
        .groups((m, n.div_ceil(bn), 1), (bk, 2, 1));
}

/// `qmv_wide`: several input vectors share one weight read. This is the path a
/// 2..limit row batch takes -- MTP verification and speculative batches.
fn qmv_wide(ctx: &Context, x: &Array, qw: &QuantMatrix, out: &Array, m: usize, n: usize, k: usize) {
    let n_tiles = m.div_ceil(5);
    let vecs_per_tg = m.div_ceil(n_tiles);
    let k_lanes = 8; // affine
    let num_simdgroups = 2;
    let rows_per_tg = (32 / k_lanes) * num_simdgroups;
    let name = format!(
        "affine_qmv_wide_{}_nv_{vecs_per_tg}_kl_{k_lanes}_batch_0",
        qw.suffix(x.type_string())
    );
    ctx.kernel(&name)
        .arr(&qw.w)
        .arr(&qw.scales)
        .arr(&qw.biases)
        .arr(x)
        .out(out)
        .bytes(&(k as i32))
        .bytes(&(n as i32))
        .bytes(&(m as i32))
        .groups((m.div_ceil(vecs_per_tg), n.div_ceil(rows_per_tg), 1), (32, num_simdgroups, 1));
}

/// `qmm` with `transpose = true`: the 32x32 tiled gemm.
pub fn qmm_t(ctx: &Context, x: &Array, qw: &QuantMatrix, out: &Array, m: usize, n: usize, k: usize) {
    assert!(!ctx.nax_available(), "this GPU would take mlx's nax gemm path");
    let (wm, wn) = (2, 2);
    let (bm, bn) = (32, 32);
    let aligned = n % 32 == 0;
    let name = format!(
        "affine_qmm_t_{}_alN_{aligned}_batch_0",
        qw.suffix(x.type_string())
    );
    ctx.kernel(&name)
        .arr(&qw.w)
        .arr(&qw.scales)
        .arr(&qw.biases)
        .arr(x)
        .out(out)
        .bytes(&(k as i32))
        .bytes(&(n as i32))
        .bytes(&(m as i32))
        .groups((n.div_ceil(bn), m.div_ceil(bm), 1), (32, wn, wm));
}

/// `qmm_splitk`: partition K so there is enough parallelism for a short batch
/// of rows, then sum the partial products.
fn qmm_splitk(ctx: &Context, x: &Array, qw: &QuantMatrix, out: &Array, m: usize, n: usize, k: usize) {
    let (bm, bn) = (32usize, 32usize);
    let n_tiles = n.div_ceil(bn);
    let m_tiles = m.div_ceil(bm);
    let mut split_k = (512 / (n_tiles * m_tiles)).max(1);
    let k_align = if qw.group_size > 32 { qw.group_size } else { 32 };
    split_k = split_k.min(k / k_align);
    while split_k > 1 && k % (split_k * k_align) != 0 {
        split_k -= 1;
    }
    if split_k <= 1 {
        qmm_t(ctx, x, qw, out, m, n, k);
        return;
    }

    let k_partition_size = k / split_k;
    let split_k_partition_stride = m * n;
    let partials = Array::empty(ctx, &[split_k, m, n], x.dtype);

    let aligned = n % 32 == 0;
    let name = format!(
        "affine_qmm_t_splitk_{}_alN_{aligned}",
        qw.suffix(x.type_string())
    );
    ctx.kernel(&name)
        .arr(&qw.w)
        .arr(&qw.scales)
        .arr(&qw.biases)
        .arr(x)
        .out(&partials)
        .bytes(&(k as i32))
        .bytes(&(n as i32))
        .bytes(&(m as i32))
        .bytes(&(k_partition_size as i32))
        .bytes(&(split_k_partition_stride as i32))
        .groups((n_tiles, m_tiles, split_k), (32, 2, 2));

    // mlx finishes with its general strided reduction over axis 0. The shape
    // here is always [split_k, M*N] with split_k <= 16, so this port uses a
    // dedicated kernel instead of porting the reduction planner.
    super::reduce::sum_axis0(ctx, &partials, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_alignment_matches_mlx() {
        assert_eq!(qmv_fast_k_alignment(4), 8 * 2 * 32);
        assert_eq!(qmv_fast_k_alignment(8), 4 * 2 * 32);
        assert_eq!(qmv_fast_k_alignment(2), 16 * 1 * 32);
    }

    /// The shapes this model actually asks for, and the branch each takes.
    #[test]
    fn dispatch_choices_for_this_model() {
        let ctx = Context::new().unwrap();
        if ctx.arch_gen != 16 {
            eprintln!("limits are architecture dependent; this asserts the M4 numbers");
            return;
        }
        // in_proj_qkv: K=2048 N=8192, 8-bit
        assert_eq!(qmv_batch_limit(&ctx, 2048, 8192), 13);
        // o_proj: K=4096 N=2048
        assert_eq!(qmv_batch_limit(&ctx, 4096, 2048), 15);
        // lm_head: K=2048 N=248320
        assert_eq!(qmv_batch_limit(&ctx, 2048, 248320), 13);
        assert!(use_qmv_wide(&ctx));
        assert!(!ctx.nax_available(), "M4 must not reach the nax kernels");

        // 8-bit weights with K=2048 are qmv_fast eligible; the router gate
        // (N=256) too, since 256 % 8 == 0.
        assert_eq!(2048 % qmv_fast_k_alignment(8), 0);
        assert_eq!(256 % 8, 0);
        // shared_expert.down_proj: K=512, 8-bit -> 512 % 256 == 0, fast.
        assert_eq!(512 % qmv_fast_k_alignment(8), 0);
    }
}

// ---------------------------------------------------------------------------
// Gathered (mixture-of-experts) variants
// ---------------------------------------------------------------------------

/// `mx.gather_qmm(x, w, ..., rhs_indices=indices, transpose=True)`.
///
/// Port of `GatherQMM::eval_gpu` narrowed to `M == 1`, which is the only shape
/// `SwitchGLU` produces: it expands the activations to `[..., 1, 1, D]` before
/// calling, so every gathered matmul is one row against one expert.
///
/// `x` is `[R, K]` (one row per (token, expert) pair, already gathered when
/// sorted), `lhs`/`rhs` are `[R]` index arrays, and the output is `[R, N]`.
pub fn gather_qmm_rows(
    ctx: &Context,
    x: &Array,
    qw: &QuantMatrix,
    lhs_indices: &Array,
    rhs_indices: &Array,
    sorted: bool,
) -> Array {
    let k = x.dim(-1);
    let n = qw.out_dims();
    let rows = rhs_indices.numel();
    let experts = qw.num_experts();
    assert_eq!(k, qw.in_dims());
    let out = Array::empty(ctx, &[rows, n], x.dtype);

    // `GatherQMM::eval_gpu`, in order: the rhs-sorted batched path first, then
    // the matrix path, then the vector path. M is 1 here, so the matrix path
    // (M >= vector_limit) is unreachable.
    if rows >= 16 && sorted && rows / experts >= 4 {
        gather_qmm_rhs(ctx, x, qw, rhs_indices, &out, rows, n, k);
    } else {
        gather_qmv(ctx, x, qw, lhs_indices, rhs_indices, &out, n, k);
    }
    out
}

/// `gather_qmv` / `gather_qmv_fast`: one output row per (token, expert) pair.
fn gather_qmv(
    ctx: &Context,
    x: &Array,
    qw: &QuantMatrix,
    lhs_indices: &Array,
    rhs_indices: &Array,
    out: &Array,
    n: usize,
    k: usize,
) {
    let bn = 8;
    let bk = 32;
    let b = rhs_indices.numel();
    let fast = n % bn == 0 && k % qmv_fast_k_alignment(qw.bits) == 0;
    let name = format!(
        "affine_{}_{}",
        if fast { "gather_qmv_fast" } else { "gather_qmv" },
        qw.suffix(x.type_string())
    );

    // x is [R, 1, K] as far as the kernel is concerned: batch dims [R], then
    // the 1xK matrix. mlx passes the batch shapes and strides of x and w so
    // the kernel can apply the two index arrays itself.
    let x_shape = [b as i32, 1, k as i32];
    let x_strides = [k as i64, k as i64, 1];
    let w_shape = [qw.num_experts() as i32];
    let w_strides = [(qw.w.shape[1] * qw.w.shape[2]) as i64];
    let s_strides = [(qw.scales.shape[1] * qw.scales.shape[2]) as i64];
    let b_strides = [(qw.biases.shape[1] * qw.biases.shape[2]) as i64];
    // Both index arrays are materialised contiguous, so their strides are
    // trivially row-major and no `collapse_contiguous_dims` is needed.
    let idx_shape = [b as i32];
    let idx_strides = [1i64];

    ctx.kernel(&name)
        .arr(&qw.w)
        .arr(&qw.scales)
        .arr(&qw.biases)
        .arr(x)
        .arr(lhs_indices)
        .arr(rhs_indices)
        .out(out)
        .bytes(&(k as i32))
        .bytes(&(n as i32))
        .bytes(&1i32) // x_batch_ndims
        .bytes_slice(&x_shape)
        .bytes_slice(&x_strides)
        .bytes(&1i32) // w_batch_ndims
        .bytes_slice(&w_shape)
        .bytes_slice(&w_strides)
        .bytes_slice(&s_strides)
        .bytes_slice(&b_strides)
        .bytes(&1i32) // index ndims
        .bytes_slice(&idx_shape)
        .bytes_slice(&idx_strides)
        .bytes_slice(&idx_strides)
        .groups((1, n.div_ceil(bn), b), (bk, 2, 1));
}

/// `gather_qmm_rhs`: rows are already grouped by expert, so the kernel streams
/// each expert's weights once for a whole tile of rows. This is the prefill
/// path.
fn gather_qmm_rhs(
    ctx: &Context,
    x: &Array,
    qw: &QuantMatrix,
    rhs_indices: &Array,
    out: &Array,
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(!ctx.nax_available(), "this GPU would take mlx's nax gather gemm");
    let (bm, bn, bk) = (16usize, 32usize, 32usize);
    let (wm, wn) = (1usize, 2usize);
    let name = format!(
        "affine_gather_qmm_rhs_nt_{}_bm_{bm}_bn_{bn}_bk_{bk}_wm_{wm}_wn_{wn}",
        qw.suffix(x.type_string())
    );
    let consts = [
        (200usize, m % bm == 0),
        (201, n % bn == 0),
        (202, k % bk == 0),
    ];
    ctx.kernel_with(&name, &consts)
        .arr(x)
        .arr(&qw.w)
        .arr(&qw.scales)
        .arr(&qw.biases)
        .arr(rhs_indices)
        .out(out)
        .bytes(&(m as i32))
        .bytes(&(n as i32))
        .bytes(&(k as i32))
        .groups((n.div_ceil(bn), m.div_ceil(bm), 1), (32, wn, wm));
}
