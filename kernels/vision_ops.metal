// Kernels the vision tower needs that the text path never reached.
//
// Same rule as kernels/ornith_elementwise.metal: mlx builds these at runtime
// (either through `mx.compile`, or as a chain of its templated unary/binary
// kernels with full stride handling), and this writes out the form mlx ends
// up executing. Every intermediate is kept in the dtype mlx would keep it in
// -- for a chain of plain ops that is the array's own dtype, so a bfloat16
// expression rounds at *every* step, not just at the end.
//
// Which means *every* intermediate needs its own named variable of that type.
// Writing `a * b + c` in one expression lets the compiler keep `a * b` at
// float precision, and that is not what mlx computes: mlx's `a * b` is a
// bfloat16 array. Nine of the 65536 bfloat16 inputs to `gelu_tanh` disagree
// between the two forms, which is exactly what
// `tests/vision_ops.rs::gelu_matches_mlx_on_every_bfloat16` is for.

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

// mlx's `erf.h` (and the `expm1f.h` it pulls in) define their helpers with
// external linkage, and `#pragma once` only guards one translation unit. This
// metallib already holds a copy from kernels/sampler_ops.metal, by way of
// `unary_ops.h`, so including it again at file scope makes the metallib linker
// fail with `multiple symbols`. Including it in an anonymous namespace gives
// this copy internal linkage; unqualified lookup still finds `erf` from the
// enclosing scope, exactly as `unary_ops.h`'s `Erf` does.
#include <metal_math>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmodules-import-nested-redundant"
namespace {
#include "mlx/backend/metal/kernels/erf.h"
} // namespace
#pragma clang diagnostic pop

using namespace metal;

// --- activations ------------------------------------------------------------

// `nn.gelu`, the exact one: `x * (1 + mx.erf(x / sqrt(2))) / 2`.
//
// It is *not* `mx.compile`d (mlx/nn/layers/activations.py), so mlx evaluates
// it as five separate kernels -- divide, erf, add, multiply, divide -- each
// writing a bfloat16 array. The Python scalars become weak-typed arrays that
// take the input's dtype, so `sqrt(2)` is rounded to bfloat16 before the
// division. mlx's `Erf` is `static_cast<T>(erf(static_cast<float>(x)))` over
// its own `erf(float)` from erf.h, so that step rounds twice.
//
// `PatchMerger.act_fn` is `nn.GELU()` with no approximation, which is this.
template <typename T>
[[kernel]] void gelu_exact(
    const device T* x [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  T v = x[i];
  T t = v / static_cast<T>(1.4142135623730951f);
  T e = static_cast<T>(erf(static_cast<float>(t)));
  T s = static_cast<T>(1.0f) + e;
  T p = v * s;
  out[i] = p / static_cast<T>(2.0f);
}

// `nn.gelu_approx`, i.e. `approx="tanh"` / `"precise"`:
//
//     0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))
//
// This one *is* `@partial(mx.compile, shapeless=True)`, so mlx fuses it into
// one generated kernel -- but the generated kernel still declares each
// intermediate at the graph's dtype, so the rounding points are the same as
// the unfused chain. `x**3` is mlx's `Power`, which is `metal::pow` at the
// array dtype, and `tanh` is `metal::precise::tanh`, whose bfloat16 overload
// computes in float and rounds the result back.
//
// `MLP.act_fn` is `nn.GELU(approx="tanh")`, which the config calls
// `gelu_pytorch_tanh`.
template <typename T>
[[kernel]] void gelu_tanh(
    const device T* x [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  T v = x[i];
  T c = metal::pow(v, static_cast<T>(3.0f));
  T scaled = static_cast<T>(0.044715f) * c;
  T inner = v + scaled;
  T arg = static_cast<T>(0.7978845608028654f) * inner;
  T t = metal::precise::tanh(arg);
  T one_plus = static_cast<T>(1.0f) + t;
  T half_x = static_cast<T>(0.5f) * v;
  out[i] = half_x * one_plus;
}

// --- padding and layout -----------------------------------------------------

// Zero-pad the last axis: `out[..., j] = j < in_row ? in[..., j] : 0`.
//
// mlx's `pad_gpu` with a zero pad value, which both
// `pad_and_slice_conv_3D_gpu` (3 -> 16 channels) and MTPLX's `_fused_sdpa`
// (head dim 72 -> 80) reach.
template <typename T>
[[kernel]] void pad_last(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant uint& in_row [[buffer(2)]],
    const constant uint& out_row [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  uint j = i % out_row;
  uint row = i / out_row;
  out[i] = j < in_row ? in[row * in_row + j] : static_cast<T>(0.0f);
}

// `reshape(-1, C, T, P, P).moveaxis(1, 4)` followed by the channel pad, in one
// pass: `out[n, t, ph, pw, c] = c < C ? in[n, c, t, ph, pw] : 0`.
//
// mlx does these as two steps -- the moveaxis is a strided view and the pad is
// `pad_gpu` over it -- but both are pure data movement over exact values, so
// fusing them cannot change a bit.
template <typename T>
[[kernel]] void patch_moveaxis_pad(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant uint& channels [[buffer(2)]],
    const constant uint& padded_channels [[buffer(3)]],
    const constant uint& spatial [[buffer(4)]], // T * P * P
    uint i [[thread_position_in_grid]]) {
  uint c = i % padded_channels;
  uint rest = i / padded_channels;
  uint s = rest % spatial;
  uint n = rest / spatial;
  out[i] = c < channels
      ? in[(n * channels + c) * spatial + s]
      : static_cast<T>(0.0f);
}

// `y + bias`, with `bias` broadcast over the last axis. `nn.Conv3d.__call__`
// adds its bias as a separate op after `mx.conv3d`.
template <typename T>
[[kernel]] void add_bias_last(
    const device T* x [[buffer(0)]],
    const device T* bias [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& row [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  out[i] = x[i] + bias[i % row];
}

// --- 2D rotary embedding ----------------------------------------------------

// `VisionRotaryEmbedding.__call__`'s inverse frequencies:
//
//     1.0 / (theta ** (mx.arange(0, dim, 2, float32) / dim))
//
// three float32 ops -- arange, divide, power -- so one float32 kernel with the
// same three steps is bit-identical. `mx.power` is `metal::pow`.
[[kernel]] void vision_inv_freq(
    device float* out [[buffer(0)]],
    const constant float& theta [[buffer(1)]],
    const constant uint& dim [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  float e = static_cast<float>(2 * i) / static_cast<float>(dim);
  out[i] = 1.0f / metal::pow(theta, e);
}

// `mx.outer(mx.arange(seqlen, float32), inv_freq)`.
[[kernel]] void vision_freq_table(
    const device float* inv_freq [[buffer(0)]],
    device float* out [[buffer(1)]],
    const constant uint& half_dim [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  uint j = i % half_dim;
  uint t = i / half_dim;
  out[i] = static_cast<float>(t) * inv_freq[j];
}

// `mx.cos(freqs)` / `mx.sin(freqs)`, float32. mlx's `Cos` and `Sin` are
// `metal::precise::cos` / `metal::precise::sin`; they run from the
// precompiled metallib, not the JIT.
[[kernel]] void vision_cos_sin(
    const device float* freqs [[buffer(0)]],
    device float* cosv [[buffer(1)]],
    device float* sinv [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  float f = freqs[i];
  cosv[i] = metal::precise::cos(f);
  sinv[i] = metal::precise::sin(f);
}

// `_apply_rotary_pos_emb_vision(tensor, freqs)`:
//
//     output = (tensor * cos) + (rotate_half(tensor) * sin)
//     return output.astype(tensor.dtype)
//
// `tensor` is bfloat16 and `cos`/`sin` are float32, so mlx promotes: both
// products and the sum are evaluated in float32 and only the result is cast
// back. `rotate_half` itself is `concatenate([-x2, x1])` on the bfloat16
// tensor, so the negation happens before the promotion (exact either way).
//
// `cos` and `sin` are `[seq, half]` tables which mlx tiles to `[seq, 1, 2 *
// half]` and broadcasts over the head axis; the tile is just `d % half`.
template <typename T>
[[kernel]] void vision_rope(
    const device T* x [[buffer(0)]],
    const device float* cosv [[buffer(1)]],
    const device float* sinv [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant uint& heads [[buffer(4)]],
    const constant uint& head_dim [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  uint hd = head_dim / 2;
  uint d = i % head_dim;
  uint rest = i / head_dim;
  uint t = rest / heads;

  T rot = d < hd ? -x[i + hd] : x[i - hd];
  float c = cosv[t * hd + (d % hd)];
  float s = sinv[t * hd + (d % hd)];
  float a = static_cast<float>(x[i]) * c;
  float b = static_cast<float>(rot) * s;
  out[i] = static_cast<T>(a + b);
}

// --- interpolated position embedding ----------------------------------------

// `fast_pos_embed_interpolate`'s blend:
//
//     pos_embeds = pos_embed(idx) * weight[:, :, None]
//     patch = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
//
// one multiply and three adds, all in the embedding's dtype and in that
// order. `rows` here is the gathered `[4 * total, dims]` block.
template <typename T>
[[kernel]] void pos_embed_blend(
    const device T* rows [[buffer(0)]],
    const device T* weights [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant uint& total [[buffer(3)]],
    const constant uint& dims [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  uint d = i % dims;
  uint r = i / dims;
  T acc = rows[r * dims + d] * weights[r];
  for (uint k = 1; k < 4; k++) {
    uint rk = k * total + r;
    T term = rows[rk * dims + d] * weights[rk];
    acc = acc + term;
  }
  out[i] = acc;
}

// The spatial-merge permutation the same function ends with:
//
//     pos_embed = mx.tile(pos_embed, (t, 1))
//     pos_embed.reshape(t, h, w, D)
//              .reshape(t, h/m, m, w/m, m, D)
//              .transpose(0, 1, 3, 2, 4, 5)
//              .reshape(-1, D)
//
// i.e. the rows come out ordered by (frame, block row, block col, row in
// block, col in block), which is the order the patches themselves are in. The
// tile repeats the *same* `h * w` table for every frame, so `in` stays the
// untiled table and the frame index only selects the output row.
template <typename T>
[[kernel]] void pos_embed_permute(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant uint& h [[buffer(2)]],
    const constant uint& w [[buffer(3)]],
    const constant uint& merge [[buffer(4)]],
    const constant uint& dims [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  uint d = i % dims;
  uint r = i / dims;

  uint m = merge;
  uint bh = h / m;
  uint bw = w / m;

  uint cin = r % m;
  r /= m;
  uint rin = r % m;
  r /= m;
  uint bc = r % bw;
  r /= bw;
  uint br = r % bh;

  uint src = (br * m + rin) * w + (bc * m + cin);
  out[i] = in[src * dims + d];
}

// --- splice -----------------------------------------------------------------

// `flat[positions] = rows`, the scatter behind MTPLX's
// `_splice_rows_into_embedded`: the image-pad rows of an embedded prompt
// chunk are replaced by the tower's output rows, in prompt order.
template <typename T>
[[kernel]] void splice_rows(
    device T* flat [[buffer(0)]],
    const device T* rows [[buffer(1)]],
    const device uint* positions [[buffer(2)]],
    const constant uint& dims [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  uint d = i % dims;
  uint r = i / dims;
  flat[positions[r] * dims + d] = rows[i];
}

// --- instantiations ---------------------------------------------------------

#define instantiate_vision(tname, type)                                 \
  instantiate_kernel("gelu_exact_" #tname, gelu_exact, type)            \
  instantiate_kernel("gelu_tanh_" #tname, gelu_tanh, type)              \
  instantiate_kernel("pad_last_" #tname, pad_last, type)                \
  instantiate_kernel("patch_moveaxis_pad_" #tname, patch_moveaxis_pad, type) \
  instantiate_kernel("add_bias_last_" #tname, add_bias_last, type)      \
  instantiate_kernel("vision_rope_" #tname, vision_rope, type)          \
  instantiate_kernel("pos_embed_blend_" #tname, pos_embed_blend, type)  \
  instantiate_kernel("pos_embed_permute_" #tname, pos_embed_permute, type) \
  instantiate_kernel("splice_rows_" #tname, splice_rows, type)

instantiate_vision(bfloat16, bfloat16_t)
instantiate_vision(float32, float)

// `vision_inv_freq`, `vision_freq_table` and `vision_cos_sin` are not
// templates, so they need no instantiation: their C++ names are already the
// names the pipeline cache looks up.
