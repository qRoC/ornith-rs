// Indexing helpers for the sparse MoE block.
//
// These stand in for mlx's general gather/scatter/slice primitives, each of
// which carries arbitrary shapes, strides and broadcasting. The MoE block only
// ever indexes along one axis with a known layout, and all of these kernels
// move data without arithmetic, so they are bit-exact by construction --
// unlike the reductions and the elementwise ops, where the *order* and the
// intermediate dtype had to be matched exactly.

#include "mlx/backend/metal/kernels/utils.h"
#include "ornith_sigmoid.h"

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

// The whole tail of the MoE block, in one dispatch.
//
// Six kernels used to do this, one after another, none of them doing any work
// worth its launch: `row_reduce_small` over the k router scores,
// `div_by_row`, `mul_by_row` over the k expert outputs, `col_reduce_small`
// over the same, `sigmoid_gate` on the shared expert and `add_elem` to join
// the two. That is 240 dispatches a token in a 40-layer model, and
// `REBUILD.md` Phase 32 measured them at 0.40 ms of a 6.70 ms trunk.
//
// Bit-exact by construction, which for a reduction means reproducing its
// accumulator and its order as well as its arithmetic. mlx instantiates `Sum`
// over bfloat16 with `U = bfloat16_t`, so both sums accumulate **in
// bfloat16**, rounding at every step, and both run in ascending index order:
//
//   * `row_reduce_small` folds the score row as `total = vals[j] + total`,
//     reading it in blocks of `N_READS = 4` -- for k = 8 that is v0 .. v7 in
//     sequence;
//   * `col_reduce_small` hands each of the k rows to one thread of `lid.y`,
//     each folding `total = val + init`, and thread 0 then folds the k
//     partials out of threadgroup memory as `total = shared[j] + total` with
//     j ascending -- the same sequence again.
//
// Every value the six kernels wrote to memory is a `T` local here, so it
// rounds to bfloat16 at the same point it used to: the score sum, the
// normalised score, each weighted expert row, the mixture, and the gated
// shared expert.
template <typename T>
[[kernel]] void moe_combine(
    const device T* scores [[buffer(0)]],  // [tokens, k]
    const device T* y [[buffer(1)]],       // [tokens, k, dim]
    const device T* gate [[buffer(2)]],    // [tokens, 1]
    const device T* shared [[buffer(3)]],  // [tokens, dim]
    device T* out [[buffer(4)]],           // [tokens, dim]
    const constant uint& k [[buffer(5)]],
    const constant uint& dim [[buffer(6)]],
    const constant uint& normalize [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
  uint token = i / dim;
  uint col = i - token * dim;

  const device T* s = scores + token * k;
  T denom = 0;
  if (normalize) {
    for (uint j = 0; j < k; ++j) {
      denom = s[j] + denom;
    }
  }

  const device T* row = y + (token * k) * dim + col;
  T acc = 0;
  for (uint j = 0; j < k; ++j) {
    T w = normalize ? T(s[j] / denom) : s[j];
    T weighted = row[j * dim] * w;
    acc = weighted + acc;
  }

  T g = mlx_sigmoid(gate[token]);
  T gated = g * shared[i];
  out[i] = acc + gated;
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
// The same stable counting sort, parallel over the *input* rather than over
// nothing.
//
// The kernel below is one thread, and its comment says why that was fine:
// "the arrays here are at most tokens x 8 entries". At a decode step tokens
// is 1 and it is eight entries. At a prefill chunk of 2048 it is **16384**,
// and `REBUILD.md` Phase 35 measured the one thread at 3.5 ms a dispatch,
// 40 dispatches a chunk, **11% of a prefill chunk** -- two thirds of the
// whole gap to mlx on a prompt. The sorted path starts at 64 rows, so a
// decode step never reaches it and no instrument here had ever looked.
//
// Parallelising over experts does not help: each expert's thread would still
// walk all n keys, so the serial depth is unchanged. Parallelising over the
// input does, and stability survives because the slot an element gets is
// decided by (its expert, its block, its position in the block) in that
// order, which is the order the single thread visits them in.
//
//   1. `sort_block_hist`   one threadgroup a block, a histogram in
//                          threadgroup memory -> counts[expert][block]
//   2. `sort_block_offsets` one threadgroup: each expert sums its blocks,
//                          one thread scans the expert totals, each expert
//                          turns its blocks into starting slots
//   3. `sort_block_scatter` one thread a block, walking its own S elements
//                          in order -- serial depth S, not n
//
// Bit-identical to the one-thread kernel by construction, and
// `tests/moe::the_parallel_sort_is_the_serial_one` holds it there.

#define ORNITH_SORT_BLOCK 256

[[kernel]] void sort_block_hist(
    const device uint32_t* keys [[buffer(0)]],
    device uint32_t* blk_counts [[buffer(1)]],  // [n_experts, n_blocks]
    const constant uint& n [[buffer(2)]],
    const constant uint& n_experts [[buffer(3)]],
    const constant uint& n_blocks [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
  threadgroup atomic_uint hist[1024];
  for (uint e = lid; e < n_experts; e += tgsize) {
    atomic_store_explicit(&hist[e], 0, metal::memory_order_relaxed);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  const uint b = tgpig;
  const uint start = b * ORNITH_SORT_BLOCK;
  for (uint j = lid; j < ORNITH_SORT_BLOCK; j += tgsize) {
    const uint i = start + j;
    if (i < n) {
      atomic_fetch_add_explicit(&hist[keys[i]], 1, metal::memory_order_relaxed);
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  for (uint e = lid; e < n_experts; e += tgsize) {
    blk_counts[e * n_blocks + b] =
        atomic_load_explicit(&hist[e], metal::memory_order_relaxed);
  }
}

[[kernel]] void sort_block_offsets(
    device uint32_t* blk_counts [[buffer(0)]],  // in: counts, out: start slots
    const constant uint& n_experts [[buffer(1)]],
    const constant uint& n_blocks [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
  threadgroup uint total[1024];
  for (uint e = lid; e < n_experts; e += tgsize) {
    uint t = 0;
    for (uint b = 0; b < n_blocks; ++b) {
      t += blk_counts[e * n_blocks + b];
    }
    total[e] = t;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // The expert bases, in ascending expert order -- the outer key of the sort.
  if (lid == 0) {
    uint running = 0;
    for (uint e = 0; e < n_experts; ++e) {
      uint c = total[e];
      total[e] = running;
      running += c;
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // And within an expert, ascending block order -- the middle key.
  for (uint e = lid; e < n_experts; e += tgsize) {
    uint off = total[e];
    for (uint b = 0; b < n_blocks; ++b) {
      uint c = blk_counts[e * n_blocks + b];
      blk_counts[e * n_blocks + b] = off;
      off += c;
    }
  }
}

[[kernel]] void sort_block_scatter(
    const device uint32_t* keys [[buffer(0)]],
    device uint32_t* blk_off [[buffer(1)]],  // start slots, bumped in place
    device uint32_t* order [[buffer(2)]],
    device uint32_t* inv_order [[buffer(3)]],
    const constant uint& n [[buffer(4)]],
    const constant uint& n_blocks [[buffer(5)]],
    uint b [[thread_position_in_grid]]) {
  // One thread a block: within a block the elements must be taken in input
  // order, which is the inner key of the sort. Serial depth is the block,
  // not the array, and no two blocks touch the same (expert, block) slot.
  const uint start = b * ORNITH_SORT_BLOCK;
  const uint end = metal::min(start + ORNITH_SORT_BLOCK, n);
  for (uint i = start; i < end; ++i) {
    const uint e = keys[i];
    const uint slot = blk_off[e * n_blocks + b]++;
    order[slot] = i;
    inv_order[i] = slot;
  }
}

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
  instantiate_kernel("mul_by_row_" #tname, mul_by_row, type)           \
  instantiate_kernel("moe_combine_" #tname, moe_combine, type)

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
// And a k-quant's super-scales are float16: FR-Spec prunes `lm_head` by rows,
// and a Q4_K row is three tensors rather than a slice.
instantiate_kernel("gather_rows_by_id_float16", gather_rows_by_id, half)

// A stand-in for the router's argsort, for **pricing** it and nothing else.
//
// `carg_block_sort` is 10.5 us a layer, 40 a token, one threadgroup ordering
// 256 numbers of which only the last eight are ever read -- the biggest
// single-dispatch cost left in a decode step after `REBUILD.md` Phase 31.
// What it is worth cannot be measured by dropping it: its output is an
// *address*, and without it the expert gather reads eight experts at random,
// which is slower than reading the eight the router chose. So this writes a
// valid order instead of no order: `i * 37 + seed (mod 256)` is a
// permutation, since 37 is coprime with 256, so the last k of it are k
// distinct experts and everything downstream indexes in range and reads the
// same number of distinct weights.
//
// The tokens that come out are wrong, and that is the point -- this is an
// instrument, not a kernel. `ORNITH_FAKE_SORT=1`, inert by default.
template <typename T>
[[kernel]] void router_fake_order(
    const device T* gates [[buffer(0)]],
    device uint32_t* out [[buffer(1)]],
    const constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  uint row = i / n;
  uint col = i % n;
  // Seeded from the row's own scores, so the eight it picks vary from layer
  // to layer the way the real selection does.
  uint seed = as_type<uint>(static_cast<float>(gates[row * n])) >> 3;
  out[i] = (col * 37u + seed) % n;
}

instantiate_kernel("router_fake_order_bfloat16", router_fake_order, bfloat16_t)
instantiate_kernel("router_fake_order_float32", router_fake_order, float)

// The most experts a token may take: `num_experts_per_tok` is 8 in this
// model, and the dispatcher falls back to mlx's sort above it.
#define ROUTER_MAX_K 8
// And the widest row it carries: one simdgroup holds 32 * 8 = 256 experts,
// which is this model's `num_experts` exactly.
#define ROUTER_MAX_PER_LANE 8

// The router's top-k, without sorting the 255 experts it will not use.
//
// mlx-lm routes `mx.argpartition(gates, kth=-k)[..., -k:]` through a full
// ascending sort of all 256 and then slices the last k. That sort is the most
// expensive single dispatch left in a decode step -- 10.5 us a layer, 40 a
// token, one threadgroup, and 0.385 ms of a 7.0 ms step measured against a
// stand-in (`REBUILD.md` Phase 31). This does k rounds of selection instead,
// and the whole of its correctness is the tie rule:
//
// mlx's block sort is **stable ascending** -- `LessThan` compares values
// alone, `ThreadSort` swaps only on a strict `<`, and `merge_step` takes from
// the right run only when `b < a` -- so equal values keep their input order
// and the last k of the result are exactly the k largest under "by value,
// then by index". Selecting the maximum under that same total order, k
// times, and writing the r'th into slot `k - 1 - r` reproduces the slice of
// the sort element for element. `tests/moe.rs` checks the indices against
// mlx's own at T = 1, 7, 24 and 512.
//
// One simdgroup a row: each lane strides over its share of the experts,
// `simd_max` picks the winning value and then the largest index holding it,
// and the r already-taken indices are held in registers rather than marked
// in memory. That is k * (n/32) comparisons, all of them out of a 512-byte
// row that stays in cache.
template <typename T>
[[kernel]] void router_top_k(
    const device T* gates [[buffer(0)]],
    device uint32_t* inds [[buffer(1)]],
    device T* scores [[buffer(2)]],
    const constant uint& n [[buffer(3)]],
    const constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const device T* g = gates + row * size_t(n);

  // The row, once, into registers: lane `l` holds experts l, l+32, l+64 ...
  // so the winner's owner is `mi % 32` and its slot `mi / 32`, and a taken
  // expert is removed by writing -INF over it rather than by remembering it.
  float v[ROUTER_MAX_PER_LANE];
  uint held = 0;
  for (uint i = lane, s = 0; i < n; i += 32, ++s) {
    v[s] = static_cast<float>(g[i]);
    held = s + 1;
  }

  for (uint r = 0; r < k; ++r) {
    float best = -INFINITY;
    uint besti = 0;
    for (uint s = 0; s < held; ++s) {
      uint i = lane + 32 * s;
      // "By value, then by index", which is what a stable ascending sort
      // leaves behind.
      if (v[s] > best || (v[s] == best && i > besti)) {
        best = v[s];
        besti = i;
      }
    }
    float m = simd_max(best);
    // Among the lanes holding the winning value, the largest index wins.
    // The +1 keeps "this lane has no candidate" distinct from index 0.
    uint mi = simd_max(best == m ? besti + 1 : 0u) - 1;
    if (mi % 32 == lane) {
      v[mi / 32] = -INFINITY;
    }
    if (lane == 0) {
      uint slot = row * k + (k - 1 - r);
      inds[slot] = mi;
      scores[slot] = g[mi];
    }
  }
}

instantiate_kernel("router_top_k_bfloat16", router_top_k, bfloat16_t)
instantiate_kernel("router_top_k_float32", router_top_k, float)
