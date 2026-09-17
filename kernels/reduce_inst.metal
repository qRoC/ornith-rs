// Instantiations of mlx's reduction kernels, restricted to what this port
// dispatches.
//
// The kernel bodies are mlx's (kernels/mlx/.../reduction/*.h); this file is the
// trimmed equivalent of vendor/mlx/mlx/backend/metal/kernels/reduce.metal.
//
// Only `sum` is needed, over bfloat16 and float, and only with reduce_ndim 1
// (`get_kernel_reduce_ndim(1)`), because every reduction in this model is over
// the leading axis of a row-contiguous array:
//
//   * the split-K quantized gemm's partial products, [split_k, M*N] -> [M, N]
//   * the MoE expert mixture, [tokens, top_k, dim] -> [tokens, dim]
//
// Note the accumulator type: mlx instantiates sum with otype == itype
// (`instantiate_sum_prod`), so a bfloat16 sum rounds to bfloat16 after every
// addition. Accumulating in float would be more accurate and would not match.

// clang-format off
#include <metal_atomic>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"

#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduction/reduce_all.h"
#include "mlx/backend/metal/kernels/reduction/reduce_init.h"
#include "mlx/backend/metal/kernels/reduction/reduce_col.h"
#include "mlx/backend/metal/kernels/reduction/reduce_row.h"

#define instantiate_col_reduce_small(name, itype, otype, op, dim)          \
  instantiate_kernel("col_reduce_small_" #dim "_reduce_" #name,            \
                     col_reduce_small,                                     \
                     itype, otype, op, int, dim)                           \
  instantiate_kernel("col_reduce_longcolumn_" #dim "_reduce_" #name,       \
                     col_reduce_longcolumn,                                \
                     itype, otype, op, int, dim)

#define instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, bm, bn)        \
  instantiate_kernel("col_reduce_looped_" #dim "_" #bm "_" #bn "_reduce_" #name,       \
                     col_reduce_looped,                                                \
                     itype, otype, op, int, dim, bm, bn)

#define instantiate_col_reduce_2pass_tile(name, itype, otype, op, dim, bm, bn)        \
  instantiate_kernel("col_reduce_2pass_" #dim "_" #bm "_" #bn "_reduce_" #name,       \
                     col_reduce_2pass,                                                \
                     itype, otype, op, int, dim, bm, bn)

#define instantiate_row_reduce_small(name, itype, otype, op, dim)     \
  instantiate_kernel("row_reduce_small_" #dim "_reduce_" #name,       \
                     row_reduce_small,                                \
                     itype, otype, op, int, dim)

#define instantiate_row_reduce_looped(name, itype, otype, op, dim)       \
  instantiate_kernel("row_reduce_looped_" #dim "_reduce_" #name,         \
                     row_reduce_looped,                                  \
                     itype, otype, op, int, dim)

#define instantiate_reduce_sum(tname, itype)                                  \
  instantiate_col_reduce_small(sum##tname, itype, itype, Sum<itype>, 1)       \
  instantiate_col_reduce_looped_tile(sum##tname, itype, itype, Sum<itype>, 1, 32, 32) \
  instantiate_col_reduce_2pass_tile(sum##tname, itype, itype, Sum<itype>, 1, 32, 32)  \
  instantiate_row_reduce_small(sum##tname, itype, itype, Sum<itype>, 1)       \
  instantiate_row_reduce_looped(sum##tname, itype, itype, Sum<itype>, 1)      \
  instantiate_kernel("row_reduce_simple_sum" #tname, row_reduce_simple, itype, itype, Sum<itype>) \
  instantiate_kernel("init_reduce_sum" #tname, init_reduce, itype, Sum<itype>)

instantiate_reduce_sum(bfloat16, bfloat16_t)
instantiate_reduce_sum(float32, float) // clang-format on

// --- whole-array reductions --------------------------------------------------
//
// `categorical_inverse_cdf` takes `max(x)` and `sum(w)` over a flat float32
// row of the whole vocabulary, which is `all_reduce_dispatch`'s two-pass path.

#define instantiate_all_reduce(name, itype, otype, op) \
  instantiate_kernel("all_reduce_" #name, all_reduce, itype, otype, op)

instantiate_all_reduce(sumfloat32, float, float, Sum<float>)
instantiate_all_reduce(maxfloat32, float, float, Max<float>)
