// The elementwise steps of mlx-lm's sampler.
//
// `apply_top_p`, `apply_top_k` and `categorical_sampling` are all decorated
// with `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)`,
// so in mlx the elementwise work between the sorts, scans and reductions runs
// as JIT-fused kernels rather than as `unary.metal` / `binary.metal`
// instantiations.  This port does not have mlx's fusion engine (see
// HANDOFF.md), so the fused forms are written out here by hand.
//
// That is only safe because fusion does not change the arithmetic: mlx's
// `build_kernel` declares every intermediate with its own
// `get_type_string(x.dtype())` and applies the same op functors from
// `unary_ops.h` / `binary_ops.h` / `ternary_ops.h`.  Those headers are the
// vendored ones and the functors below are used unchanged, so a bfloat16
// intermediate still rounds to bfloat16 exactly where mlx rounds it.
//
// Scalars arrive as bits rather than as float: mlx treats a Python scalar in a
// binary op as a weak type and casts it to the array's dtype (`to_arrays` in
// python/src/utils.h), so `logits * (1.0 / 0.6)` multiplies by a *bfloat16*
// 1.666..., not by a float32 one.  The host rounds them and passes the bits.

#include <metal_math>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/unary_ops.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/ternary_ops.h"

using namespace metal;

// --- apply_top_p -------------------------------------------------------------

// probs = mx.exp(logprobs)
template <typename T>
[[kernel]] void sampler_exp(
    device const T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = Exp{}(in[i]);
}

// sorted_keep = cumulative_probs > (1 - top_p)
template <typename T>
[[kernel]] void sampler_greater_scalar(
    device const T* in [[buffer(0)]],
    device bool* out [[buffer(1)]],
    constant const T& rhs [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = Greater{}(in[i], rhs);
}

// mx.where(keep, logprobs, -inf)
template <typename T>
[[kernel]] void sampler_select_scalar(
    device const bool* cond [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const T& b [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = Select{}(cond[i], a[i], b);
}

// --- apply_top_k -------------------------------------------------------------

// -logprobs, the input to argpartition
template <typename T>
[[kernel]] void sampler_negative(
    device const T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = Negative{}(in[i]);
}

// --- categorical_sampling ----------------------------------------------------

// logits * (1.0 / temp)
template <typename T>
[[kernel]] void sampler_mul_scalar(
    device const T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const T& rhs [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = Multiply{}(in[i], rhs);
}

// astype(logits, float32) -- promote_types(bfloat16, float32) in
// categorical_inverse_cdf
template <typename T>
[[kernel]] void sampler_to_float(
    device const T* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = static_cast<float>(in[i]);
}

// w = where(isinf(m), astype(equal(x, m), f32), exp(x - m))
//
// `m` is a one-element array broadcast over the row, so both branches are
// evaluated for every element exactly as the fused kernel does.
[[kernel]] void sampler_cdf_weights(
    device const float* x [[buffer(0)]],
    device const float* m [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  float maxval = m[0];
  // mlx's `isinf` is `logical_or(isposinf(x), isneginf(x))`, and each of
  // those is an `Equal` against the infinity of the dtype.
  bool inf = LogicalOr{}(
      Equal{}(maxval, INFINITY), Equal{}(maxval, -INFINITY));
  float eq = static_cast<float>(Equal{}(x[i], maxval));
  float ex = Exp{}(Subtract{}(x[i], maxval));
  out[i] = Select{}(inf, eq, ex);
}

// u = uniform(shape, float32, key) * sum(w)
//
//   bits -> divide by float(UINT32_MAX) -> minimum with nextafter(1, 0)
//        -> range * u + lo, with range 1 and lo 0
//
// then the multiply `categorical_inverse_cdf` applies.
[[kernel]] void sampler_uniform_scaled(
    device const uint32_t* bits [[buffer(0)]],
    device const float* total [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant const float& maxval [[buffer(3)]],
    constant const float& upper [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  float u = Divide{}(static_cast<float>(bits[i]), maxval);
  u = Minimum{}(u, upper);
  u = Add{}(Multiply{}(1.0f, u), 0.0f);
  out[i] = Multiply{}(u, total[0]);
}

// searchsorted(cdf, u, "right") - 1
[[kernel]] void sampler_minus_one(
    device const uint32_t* in [[buffer(0)]],
    device uint32_t* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = Subtract{}(in[i], 1u);
}

// --- plumbing ----------------------------------------------------------------

// `mx.zeros_like`, and the `copy_gpu(src, out, CopyType::Vector)` that
// `ScatterAxis::eval_gpu` runs before scattering, and the final copy out of
// `multi_block_sort`.
template <typename T>
[[kernel]] void sampler_fill(
    device T* out [[buffer(0)]],
    constant const T& v [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = v;
}

template <typename T>
[[kernel]] void sampler_copy(
    device const T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = in[i];
}

// clang-format off
#define instantiate_sampler(tname, type)                                      \
  instantiate_kernel("sampler_exp_" #tname, sampler_exp, type)                \
  instantiate_kernel("sampler_greater_scalar_" #tname,                        \
                     sampler_greater_scalar, type)                            \
  instantiate_kernel("sampler_select_scalar_" #tname,                         \
                     sampler_select_scalar, type)                             \
  instantiate_kernel("sampler_negative_" #tname, sampler_negative, type)      \
  instantiate_kernel("sampler_mul_scalar_" #tname, sampler_mul_scalar, type)  \
  instantiate_kernel("sampler_to_float_" #tname, sampler_to_float, type)      \
  instantiate_kernel("sampler_fill_" #tname, sampler_fill, type)              \
  instantiate_kernel("sampler_copy_" #tname, sampler_copy, type)

instantiate_sampler(bfloat16, bfloat16_t)
instantiate_sampler(float32, float)
instantiate_kernel("sampler_fill_bool_", sampler_fill, bool)
instantiate_kernel("sampler_copy_bool_", sampler_copy, bool)
instantiate_kernel("sampler_copy_uint32", sampler_copy, uint32_t) // clang-format on
