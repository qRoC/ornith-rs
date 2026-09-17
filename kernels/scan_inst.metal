// Instantiations of mlx's scan kernels, restricted to what the sampler
// dispatches.
//
// mlx's own scan.metal instantiates four variants (inclusive/exclusive x
// forward/reverse) of four ops over every type; the sampler needs two:
//
//   apply_top_p           `mx.cumsum(sorted_probs, axis=-1)` over bfloat16,
//                         inclusive and forward.  Note the accumulator type:
//                         `instantiate_scan_helper(sum_bfloat16_bfloat16,
//                         ...)` means bfloat16 in, bfloat16 out, so the
//                         running sum rounds to bfloat16 at every step.
//   categorical_inverse_cdf
//                         `cumsum(w, 0, reverse=false, inclusive=false)`
//                         over float32.
//
// Both rows are contiguous, so only `contiguous_scan` is instantiated.

// clang-format off
#include <metal_math>
#include <metal_simdgroup>

using namespace metal;

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/scan.h"

#define instantiate_contiguous_scan(                                    \
    name, itype, otype, op, inclusive, reverse, nreads)                 \
  template [[host_name("contig_scan_" #name)]] [[kernel]] void          \
  contiguous_scan<itype, otype, op<otype>, nreads, inclusive, reverse>( \
      const device itype* in [[buffer(0)]],                             \
      device otype* out [[buffer(1)]],                                  \
      const constant size_t& axis_size [[buffer(2)]],                   \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint3 lsize [[threads_per_threadgroup]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

instantiate_contiguous_scan(
    inclusive_sum_bfloat16_bfloat16, bfloat16_t, bfloat16_t, CumSum, true, false, 4)
instantiate_contiguous_scan(
    exclusive_sum_bfloat16_bfloat16, bfloat16_t, bfloat16_t, CumSum, false, false, 4)
instantiate_contiguous_scan(
    inclusive_sum_float32_float32, float, float, CumSum, true, false, 4)
instantiate_contiguous_scan(
    exclusive_sum_float32_float32, float, float, CumSum, false, false, 4)
// clang-format on
