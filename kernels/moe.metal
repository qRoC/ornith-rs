// Indexing helpers for the sparse MoE block.
//
// These stand in for mlx's general gather/scatter/slice primitives, each of
// which carries arbitrary shapes, strides and broadcasting. The MoE block only
// ever indexes along one axis with a known layout, and all of these kernels
// move data without arithmetic, so they are bit-exact by construction --
// unlike the reductions and the elementwise ops, where the *order* and the
// intermediate dtype had to be matched exactly.

#include "mlx/backend/metal/kernels/utils.h"

// `a[..., n - k:]`: the top-k slice of an ascending argsort.
template <typename T>
[[kernel]] void slice_tail(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant uint& n [[buffer(2)]],
    const constant uint& k [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  uint row = i / k;
  uint col = i % k;
  out[i] = in[row * n + (n - k) + col];
}

// `mx.take_along_axis(values, idx, axis=-1)` for a 2-D last axis.
template <typename T>
[[kernel]] void take_along_last(
    const device T* values [[buffer(0)]],
    const device uint32_t* idx [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& n [[buffer(3)]],
    const constant uint& k [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  uint row = i / k;
  out[i] = values[row * n + idx[i]];
}

// `scores / scores.sum(axis=-1, keepdims=True)`, with the sum supplied.
template <typename T>
[[kernel]] void div_by_row(
    const device T* x [[buffer(0)]],
    const device T* denom [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& k [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = x[i] / denom[i / k];
}

// `x.flatten(0, -3)[order // k]`: gather whole rows of the token activations
// into expert-sorted order (mlx_lm's `_gather_sort`).
template <typename T>
[[kernel]] void gather_rows_div(
    const device T* x [[buffer(0)]],
    const device uint32_t* order [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& dim [[buffer(3)]],
    const constant uint& k [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  uint row = i / dim;
  uint col = i % dim;
  out[i] = x[(order[row] / k) * dim + col];
}

// `indices.flatten()[order]`.
[[kernel]] void gather_u32(
    const device uint32_t* x [[buffer(0)]],
    const device uint32_t* order [[buffer(1)]],
    device uint32_t* out [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = x[order[i]];
}

// `x[inv_order]`: mlx_lm's `_scatter_unsort`.
template <typename T>
[[kernel]] void unsort_rows(
    const device T* x [[buffer(0)]],
    const device uint32_t* inv_order [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& dim [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  uint row = i / dim;
  uint col = i % dim;
  out[i] = x[inv_order[row] * dim + col];
}

// `y * scores[..., None]`: one scalar per row of `dim` values.
template <typename T>
[[kernel]] void mul_by_row(
    const device T* scale [[buffer(0)]],
    const device T* x [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& dim [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = x[i] * scale[i / dim];
}

// A stable counting sort of the expert indices, producing the same permutation
// mlx's `mx.argsort` does.
//
// mlx sorts with a block merge sort whose comparator looks only at the value,
// and a merge sort is stable, so equal expert ids keep their input order --
// which is exactly what a stable counting sort produces. The permutation is
// also the only thing the sort is used for: `gather_qmm_rhs` needs rows
// grouped by expert, and every output row's arithmetic depends on its (token,
// expert) pair alone, so it would be invariant to tie-breaking anyway. A
// single thread per row keeps the counting exact and the code obvious; the
// arrays here are at most tokens x 8 entries.
[[kernel]] void counting_sort_experts(
    const device uint32_t* keys [[buffer(0)]],
    device uint32_t* order [[buffer(1)]],
    device uint32_t* inv_order [[buffer(2)]],
    device uint32_t* counts [[buffer(3)]],
    const constant uint& n [[buffer(4)]],
    const constant uint& n_experts [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }
  for (uint e = 0; e < n_experts; ++e) {
    counts[e] = 0;
  }
  for (uint i = 0; i < n; ++i) {
    counts[keys[i]] += 1;
  }
  uint running = 0;
  for (uint e = 0; e < n_experts; ++e) {
    uint c = counts[e];
    counts[e] = running;
    running += c;
  }
  for (uint i = 0; i < n; ++i) {
    uint slot = counts[keys[i]]++;
    order[slot] = i;
    inv_order[i] = slot;
  }
}

#define instantiate_moe(tname, type)                                   \
  instantiate_kernel("slice_tail_" #tname, slice_tail, type)           \
  instantiate_kernel("take_along_last_" #tname, take_along_last, type) \
  instantiate_kernel("div_by_row_" #tname, div_by_row, type)           \
  instantiate_kernel("gather_rows_div_" #tname, gather_rows_div, type) \
  instantiate_kernel("unsort_rows_" #tname, unsort_rows, type)         \
  instantiate_kernel("mul_by_row_" #tname, mul_by_row, type)

instantiate_moe(bfloat16, bfloat16_t)
instantiate_moe(float32, float)
instantiate_kernel("slice_tail_uint32", slice_tail, uint32_t)

// Row gather for the quantized embedding table: `w[ids]`, `scales[ids]`,
// `biases[ids]`. mlx expresses `QuantizedEmbedding.__call__` as
// `mx.dequantize(self.weight[x], scales=self.scales[x], biases=self.biases[x],
// ...)`, i.e. three gathers followed by a dequantize.
template <typename T>
[[kernel]] void gather_rows_by_id(
    const device T* table [[buffer(0)]],
    const device uint32_t* ids [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& row [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  uint r = i / row;
  uint c = i % row;
  out[i] = table[ids[r] * row + c];
}

instantiate_kernel("gather_rows_by_id_uint32", gather_rows_by_id, uint32_t)
instantiate_kernel("gather_rows_by_id_bfloat16", gather_rows_by_id, bfloat16_t)
instantiate_kernel("gather_rows_by_id_float32", gather_rows_by_id, float)
// nvfp4 stores its scales as uint8 E4M3 rather than in the activation dtype,
// so pruning `lm_head` for FR-Spec gathers a byte row rather than a bf16 one.
instantiate_kernel("gather_rows_by_id_uint8", gather_rows_by_id, uint8_t)
