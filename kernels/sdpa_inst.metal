// Instantiations of mlx's vector SDPA kernels.
//
// Kernel bodies are mlx's (kernels/mlx/.../sdpa_vector.h); this is the
// trimmed equivalent of
// vendor/mlx/mlx/backend/metal/kernels/scaled_dot_product_attention.metal.
//
// Only head_dim 256 is built, because that is this model's `head_dim`, and
// only for the decode path: `ScaledDotProductAttention::use_fallback` ends
// with
//
//     if (query_sequence_length > 8) {
//       return query_head_dim == 192 || query_head_dim == 256;
//     } else {
//       return query_head_dim == value_head_dim && query_head_dim == 192;
//     }
//
// so at head_dim 256 mlx uses the fused kernel for query lengths up to 8 and
// the unfused softmax(QK^T)V path above that. src/ops/sdpa.rs implements both
// halves of that split.

// clang-format off
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector, type, qk_dim, value_dim)                    \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1, type, qk_dim, value_dim)

#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2, type, value_dim)

instantiate_sdpa_vector(bfloat16_t, 256, 256)
instantiate_sdpa_vector_aggregation(bfloat16_t, 256)
instantiate_sdpa_vector(float, 256, 256)
instantiate_sdpa_vector_aggregation(float, 256) // clang-format on
