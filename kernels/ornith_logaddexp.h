// mlx's LogAddExp, verbatim from
// kernels/mlx/mlx/backend/metal/kernels/binary_ops.h.
//
// It lives in a header because two files need it: `gdn_ops.metal`, which is
// `compute_g` on its own, and `gdn_front.metal`, which carries `compute_g`
// inside the fused front end of the gated-delta layer.

#pragma once

#include <metal_stdlib>

template <typename T>
inline T mlx_logaddexp(T x, T y) {
  if (metal::isnan(x) || metal::isnan(y)) {
    return metal::numeric_limits<T>::quiet_NaN();
  }
  constexpr T inf = metal::numeric_limits<T>::infinity();
  T maxval = metal::max(x, y);
  T minval = metal::min(x, y);
  return (minval == -inf || maxval == inf)
      ? maxval
      : (maxval + log1p(metal::exp(minval - maxval)));
}
