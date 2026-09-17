// Instantiations of mlx's affine quantized kernels, restricted to what this
// artifact actually contains.
//
// The kernel bodies are mlx's, verbatim, in
// kernels/mlx/mlx/backend/metal/kernels/quantized.h. This file only decides
// which template specialisations get compiled, mirroring
// vendor/mlx/mlx/backend/metal/kernels/quantized.metal.
//
// The model's quantization config (config.json) has exactly two settings:
//
//   group_size=32, bits=4   -- the 256 MoE experts (switch_mlp.*)
//   group_size=64, bits=8   -- attention, linear_attn projections, the router
//                              gates, shared experts, embed_tokens, lm_head
//
// and activations are bfloat16 throughout. float is instantiated alongside so
// the golden tests can drive the same kernels in f32 and separate a kernel bug
// from a bf16 rounding difference.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define instantiate_quantized(name, type, group_size, bits)     \
  instantiate_kernel(                                           \
      #name "_" #type "_gs_" #group_size "_b_" #bits,           \
      name,                                                     \
      type,                                                     \
      group_size,                                               \
      bits)

#define instantiate_quantized_batched(name, type, group_size, bits, batched) \
  instantiate_kernel(                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched,      \
      name,                                                                   \
      type,                                                                   \
      group_size,                                                             \
      bits,                                                                   \
      batched)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,        \
      name,                                                                   \
      type,                                                                   \
      group_size,                                                             \
      bits,                                                                   \
      aligned)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, aligned, batched) \
  instantiate_kernel(                                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched,       \
      name,                                                                                    \
      type,                                                                                    \
      group_size,                                                                              \
      bits,                                                                                    \
      aligned,                                                                                 \
      batched)

#define instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, batched) \
  instantiate_kernel(                                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched, \
      name,                                                                                      \
      type,                                                                                      \
      group_size,                                                                                \
      bits,                                                                                      \
      vecs_per_tg,                                                                               \
      k_lanes,                                                                                   \
      batched)

#define instantiate_quantized_splitk_qmm(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                           \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,           \
      name,                                                                      \
      type,                                                                      \
      group_size,                                                                \
      bits,                                                                      \
      aligned)

#define instantiate_gather_qmm_rhs(func, name, type, group_size, bits, bm, bn, bk, wm, wn, transpose) \
  instantiate_kernel(                                                                                  \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,                                                                                            \
      type,                                                                                            \
      group_size,                                                                                      \
      bits,                                                                                            \
      bm,                                                                                              \
      bn,                                                                                              \
      bk,                                                                                              \
      wm,                                                                                              \
      wn,                                                                                              \
      transpose)

// --- what we dispatch -------------------------------------------------------
//
// Decode (M == 1):            affine_qmv_fast / affine_qmv
// Small batch (2 <= M < lim): affine_qmv_wide           (MTP verify, batches)
// Prefill (M >= lim):         affine_qmm_t / affine_qmm_t_splitk
// MoE decode:                 affine_gather_qmv_fast / affine_gather_qmv
// MoE prefill:                affine_gather_qmm_rhs (sorted) / affine_gather_qmm_t
// Embedding lookup:           affine_dequantize
//
// Every branch above is the one mlx's quantized.cpp would take for this model's
// shapes; see src/ops/quantized.rs, which reimplements that dispatch.

#define instantiate_quantized_vector(type, group_size, bits)                        \
  instantiate_quantized_batched(affine_qmv_fast, type, group_size, bits, 0)          \
  instantiate_quantized_batched(affine_qmv_fast, type, group_size, bits, 1)          \
  instantiate_quantized_batched(affine_qmv, type, group_size, bits, 0)               \
  instantiate_quantized_batched(affine_qmv, type, group_size, bits, 1)               \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, 2, 8, 0)       \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, 3, 8, 0)       \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, 4, 8, 0)       \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, 5, 8, 0)       \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, 2, 8, 1)       \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, 3, 8, 1)       \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, 4, 8, 1)       \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, 5, 8, 1)

#define instantiate_quantized_matrix(type, group_size, bits)                                \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 0)       \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 1)       \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 0)      \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 1)      \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, true)        \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, false)

#define instantiate_quantized_gather(type, group_size, bits)                                 \
  instantiate_quantized(affine_gather_qmv_fast, type, group_size, bits)                       \
  instantiate_quantized(affine_gather_qmv, type, group_size, bits)                            \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, true)            \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, false)           \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nt, type, group_size, bits, 16, 32, 32, 1, 2, true)

#define instantiate_quantized_funcs(type, group_size, bits)   \
  instantiate_quantized(affine_dequantize, type, group_size, bits) \
  instantiate_quantized_vector(type, group_size, bits)        \
  instantiate_quantized_matrix(type, group_size, bits)        \
  instantiate_quantized_gather(type, group_size, bits)

#define instantiate_quantized_types(group_size, bits)    \
  instantiate_quantized_funcs(bfloat16_t, group_size, bits) \
  instantiate_quantized_funcs(float, group_size, bits)

instantiate_quantized_types(64, 8)
instantiate_quantized_types(32, 4) // clang-format on
