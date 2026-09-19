// mlx's sigmoid, in both of the forms mlx itself evaluates it in.
//
// It lives in a header because two files need it: the elementwise kernels
// that are mlx's compiled unary ops, and the MoE block's fused tail, which
// carries the shared expert's gate inside it.

#pragma once

#include <metal_stdlib>

// mlx's Sigmoid, verbatim from
// kernels/mlx/mlx/backend/metal/kernels/unary_ops.h. The stable form
// `1/(1+exp(|x|))` with a sign flip is not interchangeable with
// `1/(1+exp(-x))`: in bfloat16 they differ by an ULP, and mlx's compiled
// kernels instantiate this at the array's own dtype, so every intermediate
// rounds to bfloat16 too.
template <typename T>
inline T mlx_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

// The same operator, with the exponential taken from `metal::fast` instead of
// `metal::precise`.
//
// mlx has two code paths for the identical C++ source and they do not always
// agree in the last bit:
//
//   * its *precompiled* metallib is built with `-fno-fast-math`, so
//     `Sigmoid` there evaluates `metal::precise::exp`;
//   * its *JIT* kernels -- the ones `mx.compile` builds, whose source is
//     assembled in mlx/backend/metal/compiled.cpp -- go through
//     `MTLCompileOptions`, and there the default math functions are the fast
//     ones.
//
// The model needs both. `nn.silu`, and mlx-lm's `swiglu` / `precise_swiglu`,
// are `@partial(mx.compile, shapeless=True)`, so they take the JIT path; the
// bare `mx.sigmoid(gate)` in the attention output gate, in the shared-expert
// gate and in gated_delta's beta does not.
//
// The disagreement is rare enough to hide: across every bfloat16 value the two
// differ on exactly one, x = -6.84375, where exp(|x|) = 938.00006 sits a
// hair above the midpoint between the bfloat16 values 936 and 940. The precise
// exp rounds it up, the fast one lands just below and rounds down, and the
// resulting sigmoid differs by one ULP. tests/activations.rs checks every
// bfloat16 input against mlx for exactly this reason.
template <typename T>
inline T mlx_sigmoid_f32(T x) {
  auto y = 1 / (1 + metal::fast::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
