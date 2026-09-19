// Elementwise kernels.
//
// mlx builds these at runtime: `mx.compile` fuses an expression like
// `nn.silu(gate) * x` into one generated kernel, and plain ops go through
// mlx's templated unary/binary machinery with full stride and broadcast
// handling. This port writes the fused forms the model actually evaluates,
// which is what mlx ends up executing, without carrying the fusion engine.
//
// Every kernel here works on contiguous data and computes in the same
// precision mlx does -- that part is not a detail: `precise_swiglu` casts to
// float before the multiply (mlx_lm/models/activations.py) while `swiglu`
// does not, and the two do not agree in bfloat16.

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "ornith_sigmoid.h"

using namespace metal;

// out[i] = in[i], with a dtype change. Data movement only, so bit-exact by
// construction; float -> bfloat16 rounds through Metal's bfloat16_t, the same
// conversion mlx's copy kernels use.
template <typename T, typename U>
[[kernel]] void copy_cast(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = static_cast<U>(in[i]);
}

// mlx_lm/models/activations.py: swiglu(gate, x) = nn.silu(gate) * x.
// nn.silu is `x * sigmoid(x)` evaluated in the array's own dtype.
template <typename T>
[[kernel]] void swiglu(
    const device T* gate [[buffer(0)]],
    const device T* x [[buffer(1)]],
    device T* out [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  T g = gate[i];
  out[i] = (g * mlx_sigmoid_f32(g)) * x[i];
}

// precise_swiglu(h, gate, x): silu and the product are evaluated in float and
// only the result is cast back to h's dtype. Used by Qwen3NextRMSNormGated,
// where `x` is the rms-normed hidden state and `gate` the z projection.
template <typename T>
[[kernel]] void precise_swiglu(
    const device T* gate [[buffer(0)]],
    const device T* x [[buffer(1)]],
    device T* out [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  float g = static_cast<float>(gate[i]);
  g = g * mlx_sigmoid(g);  // already float, so both forms agree
  out[i] = static_cast<T>(g * static_cast<float>(x[i]));
}

// out = sigmoid(gate) * x, with `gate` broadcast over the last `inner`
// elements of x. Covers both `o_proj(output * mx.sigmoid(gate))` (inner = 1,
// elementwise) and `mx.sigmoid(shared_expert_gate(x)) * shared_y`
// (inner = hidden size).
template <typename T>
[[kernel]] void sigmoid_gate(
    const device T* gate [[buffer(0)]],
    const device T* x [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& inner [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  T g = gate[i / inner];
  out[i] = mlx_sigmoid(g) * x[i];
}

// nn.silu on its own: the GatedDeltaNet conv output.
template <typename T>
[[kernel]] void silu(
    const device T* x [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  T v = x[i];
  out[i] = v * mlx_sigmoid_f32(v);
}

// out = scalar * x, evaluated in x's dtype. mlx's SDPA fallback starts with
// `multiply(array(scale, q.dtype()), q)`, so the scale rounds to bfloat16
// before the matmul rather than being folded into it.
template <typename T>
[[kernel]] void scalar_mul(
    const device T* x [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant float& scale [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = static_cast<T>(scale) * x[i];
}

// mlx's SDPA fallback builds a causal mask as `q_idx >= k_idx` with
// `q_idx = arange(offset, qL + offset)` and applies
// `where(mask, scores, finfo(scores.dtype).min)`. Fused here because the
// where is a select -- it changes no arithmetic -- while materialising a
// [Lq, Lk] bool array per layer would not.
template <typename T>
[[kernel]] void causal_mask_(
    device T* scores [[buffer(0)]],
    const constant uint& q_len [[buffer(1)]],
    const constant uint& k_len [[buffer(2)]],
    const constant uint& rows [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
  uint j = pos.x;             // key position
  uint row = pos.y;           // flattened (batch, head, query position)
  if (j >= k_len || row >= rows) {
    return;
  }
  uint i = row % q_len;       // query position
  uint offset = k_len - q_len;
  if (i + offset < j) {
    // mlx's fallback substitutes `finfo(scores.dtype).min`, which for a
    // float type is -max, not the denormal `numeric_limits::min()`. That is
    // what `Limits<T>::finite_min` is (utils.h).
    scores[row * k_len + j] = Limits<T>::finite_min;
  }
}

// out[i] = a[i] + b[i]; the residual connections.
template <typename T>
[[kernel]] void add_elem(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = a[i] + b[i];
}

#define instantiate_elementwise(tname, type)                                  \
  instantiate_kernel("swiglu_" #tname, swiglu, type)                          \
  instantiate_kernel("precise_swiglu_" #tname, precise_swiglu, type)          \
  instantiate_kernel("sigmoid_gate_" #tname, sigmoid_gate, type)              \
  instantiate_kernel("silu_" #tname, silu, type)                               \
  instantiate_kernel("scalar_mul_" #tname, scalar_mul, type)                   \
  instantiate_kernel("causal_mask_" #tname, causal_mask_, type)                               \
  instantiate_kernel("add_elem_" #tname, add_elem, type)

instantiate_elementwise(bfloat16, bfloat16_t)
instantiate_elementwise(float32, float)

#define instantiate_copy_cast(name, itype, otype) \
  instantiate_kernel("copy_cast_" #name, copy_cast, itype, otype)

instantiate_copy_cast(bfloat16_bfloat16, bfloat16_t, bfloat16_t)
instantiate_copy_cast(float32_float32, float, float)
instantiate_copy_cast(bfloat16_float32, bfloat16_t, float)
instantiate_copy_cast(float32_bfloat16, float, bfloat16_t)
instantiate_copy_cast(uint32_uint32, uint32_t, uint32_t)
instantiate_copy_cast(int32_int32, int32_t, int32_t)
