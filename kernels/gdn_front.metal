// The front end of a gated-delta layer, in one dispatch.
//
// Ten kernels ran in sequence between the input projections and the
// recurrence, and none of them did any work worth its launch (`REBUILD.md`
// Phase 32 priced the set at 1.6 us a dispatch against 0.3 us of memory):
//
//   concat_time         the cached conv window ahead of this step's tokens
//   slice_inner         the tail of that, which is the next cached window
//   depthwise_conv_1d   the causal conv
//   silu
//   rms x2, scalar_mul x2   q and k normalised and scaled by Dk**-0.5
//   gdn_beta            sigmoid over [B, T, Hv]
//   compute_g           the decay gate over the same
//
// Every one of them is a function of the same 128-channel block, and the two
// norms reduce over exactly that block -- so one simdgroup can hold a head's
// channels in registers from the convolution through the normalisation, and
// the concatenation and the slice become the addresses it reads and writes.
//
// **Bit-exact by construction, and for a norm that means its reduction.**
// mlx's `rms_single_row` gives each of 32 threads `RMS_N_READS = 4`
// contiguous elements, sums the squares in float in ascending order within a
// thread and folds the threads with `simd_sum`; at Dk = 128 that is one
// simdgroup and exactly the layout below, so the reduction here is the same
// arithmetic in the same order. The convolution accumulates in float over
// ascending taps and rounds once, as `depthwise_conv_1d` does; the silu is
// `ornith_elementwise.metal`'s, on the rounded conv output; `beta` and
// `compute_g` are `gdn_ops.metal`'s, unchanged.
//
// The one line that is not a copy: `rms_norm(q, None, eps)` multiplies by a
// weight of bfloat16 1, which the fused form leaves out. `1 * x` is exact in
// IEEE arithmetic and `x` is already a bfloat16 value, so the round-trip
// through the product cannot move it.
//
// `ORNITH_FUSE_GDN_FRONT=0` issues the ten separately, as the control.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"
#include "ornith_logaddexp.h"
#include "ornith_sigmoid.h"

using namespace metal;

// The capture path keeps the concatenated window, because `rewind` slices a
// cached conv state out of it; plain decode does not, and writing it would
// be the one copy this kernel exists to remove.
constant bool gdn_keep_conv_input [[function_constant(10)]];

struct GdnFrontParams {
  uint t_len;     // T, this call's tokens
  uint n_state;   // K - 1, the cached window
  uint ksize;     // K, the conv kernel
  uint dim;       // D = 2 * key_dim + value_dim, the conv channels
  uint head_dim;  // Dk == Dv
  uint qk_slots;  // 2 * Hk: the head slots q and k occupy, q first
  uint n_heads;   // Hv
  uint simds;     // simdgroups per threadgroup
  float q_scale;
  float k_scale;
  float eps;
};

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void gdn_front(
    const device T* conv_state [[buffer(0)]],   // [B, K-1, D]
    const device T* qkv [[buffer(1)]],          // [B, T, D]
    const device T* cw [[buffer(2)]],           // [D, K]
    const device T* a_in [[buffer(3)]],         // [B, T, Hv]
    const device T* b_in [[buffer(4)]],         // [B, T, Hv]
    const device T* a_log [[buffer(5)]],        // [Hv]
    const device T* dt_bias [[buffer(6)]],      // [Hv]
    device T* conv_next [[buffer(7)]],          // [B, K-1, D]
    device T* q_out [[buffer(8)]],              // [B, T, Hk, Dk]
    device T* k_out [[buffer(9)]],              // [B, T, Hk, Dk]
    device T* v_out [[buffer(10)]],             // [B, T, Hv, Dv]
    device T* beta_out [[buffer(11)]],          // [B, T, Hv]
    device float* g_out [[buffer(12)]],         // [B, T, Hv]
    const constant GdnFrontParams& p [[buffer(13)]],
    device T* conv_input [[buffer(14), function_constant(gdn_keep_conv_input)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  const uint D = p.dim;
  const uint K = p.ksize;
  const uint S = p.n_state;
  const uint TOK = p.t_len;
  const uint HD = p.head_dim;

  // One thread column per (batch, token); one simdgroup per 128-channel head.
  const uint bi = tgid.z / TOK;
  const uint t = tgid.z - bi * TOK;
  const uint slot = tgid.y * p.simds + simd_group_id;
  const uint c0 = slot * HD + simd_lane_id * N_READS;

  const device T* cs = conv_state + bi * S * D;
  const device T* qk = qkv + bi * TOK * D;

  // `concat_time` into `depthwise_conv_1d`: row j of the concatenation is
  // the cached window below `S` and this call's tokens above it, so the
  // concatenation is this select and never a buffer.
  float acc[N_READS];
  for (int r = 0; r < N_READS; ++r) {
    acc[r] = 0.0f;
  }
  for (uint i = 0; i < K; ++i) {
    const uint j = t + i;
    const device T* src = (j < S) ? (cs + j * D) : (qk + (j - S) * D);
    for (int r = 0; r < N_READS; ++r) {
      acc[r] += static_cast<float>(src[c0 + r]) * cw[(c0 + r) * K + i];
    }
  }

  // `silu`, on the rounded convolution output.
  T y[N_READS];
  for (int r = 0; r < N_READS; ++r) {
    T cv = static_cast<T>(acc[r]);
    y[r] = cv * mlx_sigmoid_f32(cv);
  }

  // `keep_tail`: the last K-1 rows of the concatenation are the next cached
  // window, and the thread holding the last token is the one that has read
  // them -- rows T .. T+K-2, whatever T is.
  if (t + 1 == TOK) {
    device T* dst = conv_next + bi * S * D;
    for (uint j = 0; j < S; ++j) {
      const uint row = TOK + j;
      const device T* src = (row < S) ? (cs + row * D) : (qk + (row - S) * D);
      for (int r = 0; r < N_READS; ++r) {
        dst[j * D + c0 + r] = src[c0 + r];
      }
    }
  }

  if (gdn_keep_conv_input) {
    device T* ci = conv_input + bi * (S + TOK) * D;
    for (int r = 0; r < N_READS; ++r) {
      ci[(S + t) * D + c0 + r] = qk[t * D + c0 + r];
    }
    if (t == 0) {
      for (uint j = 0; j < S; ++j) {
        for (int r = 0; r < N_READS; ++r) {
          ci[j * D + c0 + r] = cs[j * D + c0 + r];
        }
      }
    }
  }

  // `slice_inner` into `rms_norm` into `scalar_mul` -- or, for the value
  // half, `slice_inner` alone.
  const uint key_dim = (p.qk_slots / 2) * HD;
  if (slot < p.qk_slots) {
    float thread_x[N_READS];
    float sum = 0;
    for (int r = 0; r < N_READS; ++r) {
      thread_x[r] = static_cast<float>(y[r]);
      sum += thread_x[r] * thread_x[r];
    }
    sum = simd_sum(sum);
    const float inv_mean = metal::precise::rsqrt(sum / HD + p.eps);
    const bool is_q = c0 < key_dim;
    device T* dst = (is_q ? q_out : k_out) + (bi * TOK + t) * key_dim
        + (is_q ? c0 : c0 - key_dim);
    const T scale = static_cast<T>(is_q ? p.q_scale : p.k_scale);
    for (int r = 0; r < N_READS; ++r) {
      dst[r] = scale * static_cast<T>(thread_x[r] * inv_mean);
    }
  } else {
    device T* dst = v_out + (bi * TOK + t) * (D - 2 * key_dim) + (c0 - 2 * key_dim);
    for (int r = 0; r < N_READS; ++r) {
      dst[r] = y[r];
    }
  }

  // `gdn_beta` and `compute_g`, over [B, T, Hv]. One simdgroup of the grid
  // does both: they read the two one-row projections and nothing this kernel
  // computes, so they are free wherever they are put.
  if (tgid.y == 0 && simd_group_id == 0) {
    for (uint h = simd_lane_id; h < p.n_heads; h += 32) {
      const uint i = (bi * TOK + t) * p.n_heads + h;
      T sp = mlx_logaddexp<T>(a_in[i] + dt_bias[h], T(0));
      float decay = metal::precise::exp(static_cast<float>(a_log[h]));
      g_out[i] = metal::precise::exp(-decay * static_cast<float>(sp));
      T x = b_in[i];
      auto yb = 1 / (1 + metal::precise::exp(metal::abs(x)));
      beta_out[i] = (x < 0) ? yb : 1 - yb;
    }
  }
}

#define instantiate_gdn_front(tname, type) \
  instantiate_kernel("gdn_front_" #tname, gdn_front, type)

instantiate_gdn_front(bfloat16, bfloat16_t)
instantiate_gdn_front(float32, float)
