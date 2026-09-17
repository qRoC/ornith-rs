// Instantiations of mlx's `take_along_axis` / `put_along_axis` kernels.
//
// mlx JITs these per (out type, index type, index width, contiguity), from
// `GatherAxis::eval_gpu` and `ScatterAxis::eval_gpu` in
// `mlx/backend/metal/indexing.cpp`.  The sampler makes three calls, all with
// uint32 indices over a [1, vocab] row, so `int` locations are enough
// (248320 is well under INT32_MAX) and only these combinations exist:
//
//   take_along_axis(probs, sorted_indices)          src contiguous, idx contiguous
//   put_along_axis(zeros, sorted_indices, keep)     upd contiguous, idx contiguous
//   put_along_axis(x, mask_idx, -inf)               upd broadcast (so nc),
//                                                   idx contiguous
//
// `scatter_axis` writes through `mlx_atomic<T>`; with `Op = None` that is a
// plain non-atomic store, which is why a bool output is fine.

// clang-format off
#include <metal_atomic>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/indexing/gather_axis.h"
#include "mlx/backend/metal/kernels/indexing/scatter_axis.h"

#define instantiate_gather_axis(tname, type)                                  \
  instantiate_kernel("gather_axis" #tname "uint32_int" "cc",                  \
                     gather_axis, type, uint32_t, int, true, true)            \
  instantiate_kernel("gather_axis" #tname "uint32_int" "nc" "c",              \
                     gather_axis, type, uint32_t, int, false, true)

#define instantiate_scatter_axis(tname, type)                                 \
  instantiate_kernel("scatter_axis" #tname "uint32_none_int" "cc",            \
                     scatter_axis, type, uint32_t, int, None, true, true)     \
  instantiate_kernel("scatter_axis" #tname "uint32_none_int" "nc" "c",        \
                     scatter_axis, type, uint32_t, int, None, false, true)

instantiate_gather_axis(bfloat16, bfloat16_t)
instantiate_gather_axis(float32, float)
instantiate_scatter_axis(bfloat16, bfloat16_t)
instantiate_scatter_axis(float32, float)
instantiate_scatter_axis(bool_, bool) // clang-format on
