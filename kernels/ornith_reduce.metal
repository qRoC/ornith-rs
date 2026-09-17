// Reductions this port needs that are not worth pulling mlx's reduction
// planner in for.
//
// mlx finishes a split-K quantized gemm with
// `strided_reduce_general_dispatch(intermediate, out, "sum", ...)`. For the
// shape that produces -- reduction_size = split_k (<= 16),
// non_col_reductions = 1, reduction_stride = M*N -- that dispatcher takes the
// first branch, `strided_reduce_small`, and runs `col_reduce_small`
// (vendor/mlx/mlx/backend/metal/kernels/reduction/reduce_col.h).
//
// With threadgroup_x = 32 and threadgroup_y = 8 >= reduction_size, every
// thread of that kernel executes the `r` loop exactly once, so the whole
// reduction degenerates to: start from Op::init (zero), then add row 0, row 1,
// ... row n-1 in order. That is what this kernel does.
//
// The accumulator type matters and is easy to get wrong: mlx instantiates sum
// as `instantiate_reduce_functions(name, bfloat16, bfloat16_t, bfloat16_t, op)`
// (kernels/reduce.metal), i.e. U == T, so a bfloat16 sum rounds to bfloat16
// after *every* addition. Accumulating in float instead is more accurate and
// wrong: it differs from mlx by one ULP on the split-K path.

#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void sum_axis0(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant int& rows [[buffer(2)]],
    const constant int& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (static_cast<int>(gid) >= cols) {
    return;
  }
  T acc = T(0);
  for (int r = 0; r < rows; ++r) {
    acc = in[r * cols + gid] + acc;
  }
  out[gid] = acc;
}

#define instantiate_sum_axis0(name, type) \
  instantiate_kernel("sum_axis0_" #name, sum_axis0, type)

instantiate_sum_axis0(float32, float)
instantiate_sum_axis0(bfloat16, bfloat16_t)
