// The scalar side of the gated delta rule: the decay gate and the conv-state
// concatenation.
//
// mlx-lm computes the gate with
//
//   @partial(mx.compile, shapeless=True)
//   def compute_g(A_log, a, dt_bias):
//       return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))
//
// The dtypes are load-bearing and are reproduced exactly below:
//   * `a` and `dt_bias` are bfloat16, so `a + dt_bias` rounds to bfloat16;
//   * `nn.softplus(x)` is `mx.logaddexp(x, 0)`, and 0 is a weakly-typed
//     Python int, so logaddexp also runs at bfloat16 -- including mlx's
//     stable form `max + log1p(exp(min - max))` (binary_ops.h::LogAddExp);
//   * `A_log` is cast to float32 first, so the product and the outer exp are
//     float32 and the gate comes out float32, which is what the recurrence
//     kernel's `const device float* g` expects.

#include "mlx/backend/metal/kernels/utils.h"

// mlx's LogAddExp, verbatim from
// kernels/mlx/mlx/backend/metal/kernels/binary_ops.h.
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

// a, out: [B, T, Hv]; A_log, dt_bias: [Hv]
template <typename T>
[[kernel]] void compute_g(
    const device T* A_log [[buffer(0)]],
    const device T* a [[buffer(1)]],
    const device T* dt_bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    const constant uint& n_heads [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  uint h = i % n_heads;
  T sp = mlx_logaddexp<T>(a[i] + dt_bias[h], T(0));
  float decay = metal::precise::exp(static_cast<float>(A_log[h]));
  out[i] = metal::precise::exp(-decay * static_cast<float>(sp));
}

// beta = mx.sigmoid(b), evaluated in b's own dtype (mlx's stable sigmoid).
template <typename T>
[[kernel]] void gdn_beta(
    const device T* b [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  T x = b[i];
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  out[i] = (x < 0) ? y : 1 - y;
}

// `mx.concatenate([conv_state, qkv], axis=1)` for [B, *, D] arrays, plus the
// slice back out that updates the cache. Both are pure data movement.
template <typename T>
[[kernel]] void concat_time(
    const device T* head [[buffer(0)]],
    const device T* tail [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& head_t [[buffer(3)]],
    const constant uint& tail_t [[buffer(4)]],
    const constant uint& dim [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  uint d = i % dim;
  uint t = (i / dim) % (head_t + tail_t);
  uint b = i / (dim * (head_t + tail_t));
  out[i] = (t < head_t) ? head[(b * head_t + t) * dim + d]
                        : tail[(b * tail_t + (t - head_t)) * dim + d];
}

#define instantiate_gdn_ops(tname, type)                              \
  instantiate_kernel("compute_g_" #tname, compute_g, type)            \
  instantiate_kernel("gdn_beta_" #tname, gdn_beta, type)              \
  instantiate_kernel("concat_time_" #tname, concat_time, type)

instantiate_gdn_ops(bfloat16, bfloat16_t)
instantiate_gdn_ops(float32, float)
