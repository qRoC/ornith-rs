//! The sparse mixture-of-experts block.
//!
//! Port of `Qwen3NextSparseMoeBlock` and `SwitchGLU`
//! (vendor/mlx-lm/mlx_lm/models/qwen3_next.py and switch_layers.py) with the
//! kernel dispatch from `GatherQMM::eval_gpu`.
//!
//! The Python is short, and every line of it is a decision that shows up in the
//! numbers:
//!
//! ```text
//!     gates = self.gate(x)
//!     gates = mx.softmax(gates, axis=-1, precise=True)
//!     inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
//!     scores = mx.take_along_axis(gates, inds, axis=-1)
//!     scores = scores / scores.sum(axis=-1, keepdims=True)      # norm_topk_prob
//!     y = self.switch_mlp(x, inds)
//!     y = (y * scores[..., None]).sum(axis=-2)
//!     shared_y = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
//!     y = y + shared_y
//! ```

use crate::array::{Array, Dtype};
use crate::metal::Context;

use super::quantized::{QuantMatrix, gather_qmm_rows, quantized_matmul};
use super::reduce::{sum_last_axis, sum_second_last_axis};

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        Dtype::U32 => "uint32",
        d => panic!("no moe kernel for {}", d.name()),
    }
}

/// Launch geometry for a flat one-thread-per-element kernel.
///
/// The grid is the exact element count, never rounded up to a multiple of the
/// threadgroup: `dispatchThreads:` handles a non-uniform last threadgroup, and
/// rounding up instead would run threads past the end of the buffer. None of
/// these kernels bounds-check, because with an exact grid they do not have to
/// -- and a rounded grid corrupted memory in a way that only showed up at
/// sequence lengths whose element count was not a multiple of 256.
fn flat(n: usize) -> ((usize, usize, usize), (usize, usize, usize)) {
    let tg = 256.min(n.next_power_of_two()).max(1);
    ((n, 1, 1), (tg, 1, 1))
}

/// `mx.argsort(x, axis=-1)` for a row-contiguous 2-D `x`.
///
/// Port of `single_block_sort`. mlx routes `argpartition` here too, so the
/// router's top-k comes out of a full ascending sort.
pub fn argsort_last_axis(ctx: &Context, x: &Array) -> Array {
    let n = x.dim(-1);
    let n_rows = x.numel() / n;
    let out = Array::empty(ctx, &x.shape, Dtype::U32);

    // `gpu_merge_sort`: tn is always 4, and bn is the first bucket that fits.
    let tn = 4usize;
    let potential_bn = n.div_ceil(tn);
    let mut bn = if potential_bn > 256 {
        512
    } else if potential_bn > 128 {
        256
    } else if potential_bn > 64 {
        128
    } else if potential_bn > 32 {
        64
    } else {
        32
    };
    if bn == 512 && x.dtype.size() > 4 {
        bn = 256;
    }
    let n_blocks = n.div_ceil(bn * tn);
    assert_eq!(
        n_blocks, 1,
        "a {n}-wide sort needs mlx's multi_block_sort, which this port does not carry"
    );

    let name = format!("carg_block_sort_{}_uint32_bn{bn}_tn{tn}", tname(x.dtype));
    ctx.kernel(&name)
        .arr(x)
        .out(&out)
        .bytes(&(n as i32)) // size_sorted_axis
        .bytes(&1i32) // in_stride_sorted_axis
        .bytes(&1i32) // out_stride_sorted_axis
        .bytes(&(n as i32)) // in_stride_segment_axis
        .bytes(&(n as i32)) // out_stride_segment_axis
        .groups((1, n_rows, 1), (bn, 1, 1));
    out
}

/// `argpartition(x, kth=-k)[..., -k:]` plus the gathered scores, i.e. the
/// router's expert selection.
pub fn top_k(ctx: &Context, gates: &Array, k: usize) -> (Array, Array) {
    let n = gates.dim(-1);
    let rows = gates.numel() / n;
    let order = argsort_last_axis(ctx, gates);

    let inds = Array::empty(ctx, &[rows, k], Dtype::U32);
    let (grid, tg) = flat(rows * k);
    ctx.kernel("slice_tail_uint32")
        .arr(&order)
        .out(&inds)
        .bytes(&(n as u32))
        .bytes(&(k as u32))
        .threads(grid, tg);

    let scores = Array::empty(ctx, &[rows, k], gates.dtype);
    ctx.kernel(&format!("take_along_last_{}", tname(gates.dtype)))
        .arr(gates)
        .arr(&inds)
        .out(&scores)
        .bytes(&(n as u32))
        .bytes(&(k as u32))
        .threads(grid, tg);

    (inds, scores)
}

/// `scores / scores.sum(axis=-1, keepdims=True)`.
pub fn normalize_scores(ctx: &Context, scores: &Array) -> Array {
    let k = scores.dim(-1);
    let rows = scores.numel() / k;
    let denom = Array::empty(ctx, &[rows, 1], scores.dtype);
    sum_last_axis(ctx, scores, &denom);

    let out = Array::empty(ctx, &scores.shape, scores.dtype);
    let (grid, tg) = flat(scores.numel());
    ctx.kernel(&format!("div_by_row_{}", tname(scores.dtype)))
        .arr(scores)
        .arr(&denom)
        .out(&out)
        .bytes(&(k as u32))
        .threads(grid, tg);
    out
}

/// `SwitchGLU.__call__`: route each token to its `k` experts.
///
/// `x` is `[tokens, dim]`, `inds` is `[tokens, k]`. Returns `[tokens, k, dim]`.
pub fn switch_glu(
    ctx: &Context,
    x: &Array,
    inds: &Array,
    gate_proj: &QuantMatrix,
    up_proj: &QuantMatrix,
    down_proj: &QuantMatrix,
) -> Array {
    switch_glu_sorted(ctx, x, inds, gate_proj, up_proj, down_proj, None)
}

/// `force_sort` overrides mlx-lm's threshold; sorting is a permutation, so it
/// must not change the result, which makes this a useful cross-check.
pub fn switch_glu_sorted(
    ctx: &Context,
    x: &Array,
    inds: &Array,
    gate_proj: &QuantMatrix,
    up_proj: &QuantMatrix,
    down_proj: &QuantMatrix,
    force_sort: Option<bool>,
) -> Array {
    let tokens = x.shape[0];
    let dim = x.shape[1];
    let k = inds.shape[1];
    let rows = tokens * k;

    // mlx-lm sorts once the batch is big enough for the weight reads to
    // dominate: `do_sort = indices.size >= 64`.
    let do_sort = force_sort.unwrap_or(rows >= 64);

    let (xs, idx, inv_order) = if do_sort {
        let (order, inv_order) = counting_sort(ctx, inds, gate_proj.num_experts());
        let xs = Array::empty(ctx, &[rows, dim], x.dtype);
        let (grid, tg) = flat(rows * dim);
        ctx.kernel(&format!("gather_rows_div_{}", tname(x.dtype)))
            .arr(x)
            .arr(&order)
            .out(&xs)
            .bytes(&(dim as u32))
            .bytes(&(k as u32))
            .threads(grid, tg);

        let idx = Array::empty(ctx, &[rows], Dtype::U32);
        let (g2, t2) = flat(rows);
        ctx.kernel("gather_u32").arr(inds).arr(&order).out(&idx).threads(g2, t2);
        (xs, idx, Some(inv_order))
    } else {
        // Unsorted: each token's row is reused for all k experts, which the
        // kernel expresses through `lhs_indices`.
        (x.clone(), inds.reshape(&[rows]), None)
    };

    // lhs_indices: which row of `xs` each output row reads. Sorted rows are
    // already one-to-one; unsorted ones repeat each token k times.
    let lhs: Vec<u32> = if do_sort {
        (0..rows as u32).collect()
    } else {
        (0..rows as u32).map(|i| i / k as u32).collect()
    };
    let lhs_indices = Array::from_slice(ctx, &lhs, &[rows], Dtype::U32);

    let x_up = gather_qmm_rows(ctx, &xs, up_proj, &lhs_indices, &idx, do_sort);
    let x_gate = gather_qmm_rows(ctx, &xs, gate_proj, &lhs_indices, &idx, do_sort);
    let act = super::elementwise::swiglu(ctx, &x_gate, &x_up);
    // down_proj consumes the activation row-for-row, so its lhs indices are
    // the identity regardless of sorting.
    let identity: Vec<u32> = (0..rows as u32).collect();
    let lhs_down = Array::from_slice(ctx, &identity, &[rows], Dtype::U32);
    let y = gather_qmm_rows(ctx, &act, down_proj, &lhs_down, &idx, do_sort);

    match inv_order {
        Some(inv) => {
            let out = Array::empty(ctx, &[rows, dim], y.dtype);
            let (grid, tg) = flat(rows * dim);
            ctx.kernel(&format!("unsort_rows_{}", tname(y.dtype)))
                .arr(&y)
                .arr(&inv)
                .out(&out)
                .bytes(&(dim as u32))
                .threads(grid, tg);
            out.reshape(&[tokens, k, dim])
        }
        None => y.reshape(&[tokens, k, dim]),
    }
}

/// Exposed for the unsort debug test.
pub fn unsort_for_test(ctx: &Context, y: &Array, inv: &Array) -> Array {
    let (rows, dim) = (y.shape[0], y.shape[1]);
    let out = Array::empty(ctx, &y.shape, y.dtype);
    let (grid, tg) = flat(rows * dim);
    ctx.kernel(&format!("unsort_rows_{}", tname(y.dtype)))
        .arr(y)
        .arr(inv)
        .out(&out)
        .bytes(&(dim as u32))
        .threads(grid, tg);
    out
}

/// Exposed for the gather debug test.
pub fn gather_rows_for_test(ctx: &Context, x: &Array, order: &Array, k: usize) -> Array {
    let rows = order.numel();
    let dim = x.shape[1];
    let xs = Array::empty(ctx, &[rows, dim], x.dtype);
    let (grid, tg) = flat(rows * dim);
    ctx.kernel(&format!("gather_rows_div_{}", tname(x.dtype)))
        .arr(x)
        .arr(order)
        .out(&xs)
        .bytes(&(dim as u32))
        .bytes(&(k as u32))
        .threads(grid, tg);
    xs
}

/// Exposed so tests can compare the permutation with mlx's `mx.argsort`.
pub fn counting_sort_for_test(ctx: &Context, inds: &Array, n_experts: usize) -> (Array, Array) {
    counting_sort(ctx, inds, n_experts)
}

/// `(order, inv_order)` for mlx-lm's `_gather_sort`. See the kernel comment in
/// kernels/moe.metal for why a counting sort reproduces mlx's argsort here.
fn counting_sort(ctx: &Context, inds: &Array, n_experts: usize) -> (Array, Array) {
    let n = inds.numel();
    let order = Array::empty(ctx, &[n], Dtype::U32);
    let inv_order = Array::empty(ctx, &[n], Dtype::U32);
    let counts = Array::empty(ctx, &[n_experts], Dtype::U32);
    ctx.kernel("counting_sort_experts")
        .arr(inds)
        .out(&order)
        .out(&inv_order)
        .out(&counts)
        .bytes(&(n as u32))
        .bytes(&(n_experts as u32))
        .threads((1, 1, 1), (1, 1, 1));
    (order, inv_order)
}

/// The whole `Qwen3NextSparseMoeBlock`, given the layer's weights.
#[allow(clippy::too_many_arguments)]
pub fn sparse_moe_block(
    ctx: &Context,
    x: &Array,
    gate: &QuantMatrix,
    switch_gate: &QuantMatrix,
    switch_up: &QuantMatrix,
    switch_down: &QuantMatrix,
    shared_gate: &QuantMatrix,
    shared_up: &QuantMatrix,
    shared_down: &QuantMatrix,
    shared_expert_gate: &QuantMatrix,
    top_k_n: usize,
    norm_topk_prob: bool,
) -> Array {
    let dim = x.dim(-1);
    let tokens = x.numel() / dim;
    let x2 = x.reshape(&[tokens, dim]);

    let gates = quantized_matmul(ctx, &x2, gate);
    let gates = super::softmax::softmax(ctx, &gates, true);

    let (inds, scores) = top_k(ctx, &gates, top_k_n);
    let scores = if norm_topk_prob { normalize_scores(ctx, &scores) } else { scores };

    let y = switch_glu(ctx, &x2, &inds, switch_gate, switch_up, switch_down);

    let mixed = mix_experts(ctx, &y, &scores);

    // The shared expert runs on every token, gated by a scalar per token.
    let sg = quantized_matmul(ctx, &x2, shared_gate);
    let su = quantized_matmul(ctx, &x2, shared_up);
    let sact = super::elementwise::swiglu(ctx, &sg, &su);
    let shared_y = quantized_matmul(ctx, &sact, shared_down);
    let gate_scalar = quantized_matmul(ctx, &x2, shared_expert_gate);
    let shared_y = super::elementwise::sigmoid_gate(ctx, &gate_scalar, &shared_y);

    let out = super::elementwise::add(ctx, &mixed, &shared_y);
    out.reshape(&x.shape)
}

/// `(y * scores[..., None]).sum(axis=-2)`: weight each expert's output by its
/// router score and mix them.
pub fn mix_experts(ctx: &Context, y: &Array, scores: &Array) -> Array {
    let (tokens, dim) = (y.shape[0], y.shape[2]);
    let weighted = scale_rows(ctx, y, scores);
    let mixed = Array::empty(ctx, &[tokens, dim], y.dtype);
    sum_second_last_axis(ctx, &weighted, &mixed);
    mixed
}

/// `y * scores[..., None]` for `y` `[tokens, k, dim]` and `scores`
/// `[tokens, k]`.
fn scale_rows(ctx: &Context, y: &Array, scores: &Array) -> Array {
    let dim = y.shape[2];
    let out = Array::empty(ctx, &y.shape, y.dtype);
    let (grid, tg) = flat(y.numel());
    // `sigmoid_gate` is the wrong op here; this is a plain broadcast multiply,
    // so reuse the row-broadcast shape with a dedicated kernel.
    ctx.kernel(&format!("mul_by_row_{}", tname(y.dtype)))
        .arr(scores)
        .arr(y)
        .out(&out)
        .bytes(&(dim as u32))
        .threads(grid, tg);
    out
}

// --- the unquantized block ---------------------------------------------------
//
// The MTP head carries the same 256-expert block as a trunk layer, but stored
// bfloat16 rather than quantized, so every `quantized_matmul` becomes a plain
// `matmul` and `gather_qmm_rows` becomes `gather_mv`. The routing, the sort and
// the mixing are dtype-independent and are shared with the quantized path
// above.

/// `SwitchGLU.__call__` over stacked bfloat16 experts.
///
/// Each weight is `[experts, out, in]`, the layout `stack_numbered_experts`
/// leaves behind.
pub fn switch_glu_dense(
    ctx: &Context,
    x: &Array,
    inds: &Array,
    gate_proj: &Array,
    up_proj: &Array,
    down_proj: &Array,
) -> Array {
    let tokens = x.shape[0];
    let dim = x.shape[1];
    let k = inds.shape[1];
    let rows = tokens * k;
    let experts = gate_proj.shape[0];

    let do_sort = rows >= 64;
    assert!(
        !do_sort,
        "an unquantized SwitchGLU with {rows} rows sorts, which routes to \
         mlx's steel gather gemm; only the M == 1 gather_mv path is ported"
    );
    let _ = experts;

    // Unsorted: each token's row feeds all k experts, which the kernel reads
    // through `lhs_indices`.
    let idx = inds.reshape(&[rows]);
    let lhs: Vec<u32> = (0..rows as u32).map(|i| i / k as u32).collect();
    let lhs_indices = Array::from_slice(ctx, &lhs, &[rows], Dtype::U32);

    let x_up = crate::ops::matmul::gather_mv(ctx, x, up_proj, &lhs_indices, &idx);
    let x_gate = crate::ops::matmul::gather_mv(ctx, x, gate_proj, &lhs_indices, &idx);
    let act = super::elementwise::swiglu(ctx, &x_gate, &x_up);
    let identity: Vec<u32> = (0..rows as u32).collect();
    let lhs_down = Array::from_slice(ctx, &identity, &[rows], Dtype::U32);
    let y = crate::ops::matmul::gather_mv(ctx, &act, down_proj, &lhs_down, &idx);
    y.reshape(&[tokens, k, dim])
}

/// `qwen3_next.SparseMoeBlock` with bfloat16 weights throughout.
#[allow(clippy::too_many_arguments)]
pub fn sparse_moe_block_dense(
    ctx: &Context,
    x: &Array,
    gate: &Array,
    switch_gate: &Array,
    switch_up: &Array,
    switch_down: &Array,
    shared_gate: &Array,
    shared_up: &Array,
    shared_down: &Array,
    shared_expert_gate: &Array,
    top_k_n: usize,
    norm_topk_prob: bool,
) -> Array {
    use crate::ops::matmul::matmul;

    let dim = x.dim(-1);
    let tokens = x.numel() / dim;
    let x2 = x.reshape(&[tokens, dim]);

    let gates = matmul(ctx, &x2, gate, true);
    let gates = super::softmax::softmax(ctx, &gates, true);

    let (inds, scores) = top_k(ctx, &gates, top_k_n);
    let scores = if norm_topk_prob { normalize_scores(ctx, &scores) } else { scores };

    let y = switch_glu_dense(ctx, &x2, &inds, switch_gate, switch_up, switch_down);
    let mixed = mix_experts(ctx, &y, &scores);

    let sg = matmul(ctx, &x2, shared_gate, true);
    let su = matmul(ctx, &x2, shared_up, true);
    let sact = super::elementwise::swiglu(ctx, &sg, &su);
    let shared_y = matmul(ctx, &sact, shared_down, true);
    let gate_scalar = matmul(ctx, &x2, shared_expert_gate, true);
    let shared_y = super::elementwise::sigmoid_gate(ctx, &gate_scalar, &shared_y);

    let out = super::elementwise::add(ctx, &mixed, &shared_y);
    out.reshape(&x.shape)
}
