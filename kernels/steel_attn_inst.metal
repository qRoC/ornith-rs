// Instantiation of mlx's *full* steel attention kernel, for the vision tower.
//
// The kernel body is mlx's
// (kernels/mlx/.../steel/attn/kernels/steel_attention.h); this is the trimmed
// equivalent of
// vendor/mlx/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal.
//
// Only one shape is built. The trunk never reaches this kernel -- at head_dim
// 256 `use_fallback` sends anything longer than 8 query rows down the unfused
// softmax(QK^T)V path -- but the vision tower does:
//
//   * `_fused_sdpa` in MTPLX's qwen3_vl_tower.py pads head_dim 72 up to 80
//     (`_FUSED_SDPA_DIMS = (64, 80, 128)`) before calling
//     `mx.fast.scaled_dot_product_attention`, and slices the result back;
//   * an image is always more than 8 patches, so `has_fused_kernel` takes the
//     `query_sequence_length > 8` branch, where head dim 80 is supported, and
//     `use_fallback`'s tail only rejects 192 and 256. So the fused full
//     kernel runs.
//
// `sdpa_full_self_attention_metal` then picks bq = 32, bk = (bd < 128 ? 32 :
// 16) = 32, wm = 4, wn = 1, and the mask type name is the query's own dtype
// because there is no mask. `is_nax_available()` is false on this machine, so
// neither the nax kernel nor its 72/80 -> 96 padding path can run.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

instantiate_attn(bfloat16, bfloat16_t, 32, 32, 80, 4, 1, bfloat16, bfloat16_t)
// clang-format on
