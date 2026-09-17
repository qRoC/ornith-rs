// The gated delta rule recurrence -- the core of the 30 linear-attention
// layers.
//
// mlx-lm builds these at runtime with `mx.fast.metal_kernel`
// (vendor/mlx-lm/mlx_lm/models/gated_delta.py). The bodies below are that
// file's kernel sources verbatim; only the signature is written out by hand
// instead of being generated, in exactly the form
// `vendor/mlx/mlx/backend/common/metal_kernel.cpp::write_signature` produces
// (confirmed against mlx's own `verbose=True` dump):
//
//   template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv>
//   [[kernel]] void ...(
//     const device InT* q [[buffer(0)]], ... const constant int32_t& T [[buffer(6)]],
//     device InT* y [[buffer(7)]], device StT* state_out [[buffer(8)]],
//     uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
//     uint3 thread_position_in_grid [[thread_position_in_grid]]) { ... }
//
// Two kernels are carried:
//
//   gated_delta_step_packed_btree -- the Dk=128 specialisation. Eight value
//     rows share a SIMD-group, four lanes per row, 32 state elements per lane,
//     and the cross-lane reduction is an explicitly written ascending
//     butterfly. This is what runs for this model: Dk = linear_key_head_dim =
//     128, Dv = 128, the gate is scalar per head and there is no padding mask.
//
//   gated_delta_step -- mlx-lm's original one-row-per-SIMD-group kernel, with
//     `simd_sum`. It is the fallback for a padding mask or Dk != 128, and is
//     kept so the masked path exists rather than silently not existing.
//
// The vectorized-gate variants (`g` shaped [B, T, Hv, Dk]) are not carried:
// Qwen3.5's `compute_g` produces a scalar gate per value head, so mlx-lm's
// `g.ndim == 4` branch is unreachable for this model.

#include "mlx/backend/metal/kernels/utils.h"

template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_step_packed_btree(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device InT* beta [[buffer(4)]],
    const device StT* state_in [[buffer(5)]],
    const constant int32_t& T [[buffer(6)]],
    device InT* y [[buffer(7)]],
    device StT* state_out [[buffer(8)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  constexpr int lanes_per_row = 4;
  constexpr int rows_per_simdgroup = 32 / lanes_per_row;
  constexpr int values_per_lane = Dk / lanes_per_row;
  constexpr int partials_per_lane = values_per_lane / 4;

  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto lane = thread_index_in_simdgroup;
  auto row_in_simdgroup = lane / lanes_per_row;
  auto lane_in_row = lane & (lanes_per_row - 1);
  auto row_group = thread_position_in_grid.y;
  auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

  // q, k: [B, T, Hk, Dk]
  auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
  auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

  // v, y: [B, T, Hv, Dv]
  auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
  y += (b_idx * T * Hv + hv_idx) * Dv;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

  float state[values_per_lane];
  for (int i = 0; i < values_per_lane; ++i) {
    state[i] = static_cast<float>(i_state[i]);
  }

  // g, beta: [B, T, Hv]
  auto g_ = g + b_idx * T * Hv;
  auto beta_ = beta + b_idx * T * Hv;

  for (int t = 0; t < T; ++t) {
    float gt = static_cast<float>(g_[hv_idx]);

    // Partials mirror the generic kernel: each 4-element chain is one
    // original lane's sequential accumulation.
    float part[partials_per_lane];
    for (int pb = 0; pb < partials_per_lane; ++pb) {
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        int e = pb * 4 + i;
        state[e] = state[e] * gt;
        acc += state[e] * static_cast<float>(k_[e]);
      }
      part[pb] = acc;
    }
    // Butterfly levels xor 1,2,4 stay inside this lane (commutative
    // pairwise tree); levels xor 8,16 become the row-group shuffles.
    float kv_mem =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    kv_mem += simd_shuffle_xor(kv_mem, 1);
    kv_mem += simd_shuffle_xor(kv_mem, 2);

    auto delta =
        (static_cast<float>(v_[dv_idx]) - kv_mem) *
        static_cast<float>(beta_[hv_idx]);

    for (int pb = 0; pb < partials_per_lane; ++pb) {
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        int e = pb * 4 + i;
        state[e] = state[e] + static_cast<float>(k_[e]) * delta;
        acc += state[e] * static_cast<float>(q_[e]);
      }
      part[pb] = acc;
    }
    float out =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    out += simd_shuffle_xor(out, 1);
    out += simd_shuffle_xor(out, 2);
    if (lane_in_row == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }

    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
  }

  for (int i = 0; i < values_per_lane; ++i) {
    o_state[i] = static_cast<StT>(state[i]);
  }
}

// mlx-lm's `_make_gated_delta_kernel(has_mask=False, vectorized=False)`.
template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_step(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device InT* beta [[buffer(4)]],
    const device StT* state_in [[buffer(5)]],
    const constant int32_t& T [[buffer(6)]],
    device InT* y [[buffer(7)]],
    device StT* state_out [[buffer(8)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);
  constexpr int n_per_t = Dk / 32;

  // q, k: [B, T, Hk, Dk]
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

  // v, y: [B, T, Hv, Dv]
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  y += b_idx * T * Hv * Dv + hv_idx * Dv;

  auto dk_idx = thread_position_in_threadgroup.x;
  auto dv_idx = thread_position_in_grid.y;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = static_cast<float>(i_state[s_idx]);
  }

  // g: [B, T, Hv]
  auto g_ = g + b_idx * T * Hv;
  auto beta_ = beta + b_idx * T * Hv;

  for (int t = 0; t < T; ++t) {
    if (true) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
    } else {
      y[dv_idx] = static_cast<InT>(0);
    }
    // Increment data pointers to next time step
    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
  }
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    o_state[s_idx] = static_cast<StT>(state[i]);
  }
}

// mlx-lm's `_make_gated_delta_kernel(has_mask=True, vectorized=False)`: the
// padding-mask fallback, dispatched when a batch has ragged sequences.
template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_step_mask(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device InT* beta [[buffer(4)]],
    const device StT* state_in [[buffer(5)]],
    const constant int32_t& T [[buffer(6)]],
    const device bool* mask [[buffer(7)]],
    device InT* y [[buffer(8)]],
    device StT* state_out [[buffer(9)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);
  constexpr int n_per_t = Dk / 32;

  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  y += b_idx * T * Hv * Dv + hv_idx * Dv;

  auto dk_idx = thread_position_in_threadgroup.x;
  auto dv_idx = thread_position_in_grid.y;

  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = static_cast<float>(i_state[s_idx]);
  }

  auto g_ = g + b_idx * T * Hv;
  auto beta_ = beta + b_idx * T * Hv;

  for (int t = 0; t < T; ++t) {
    if (mask[b_idx * T + t]) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
    } else {
      y[dv_idx] = static_cast<InT>(0);
    }
    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
  }
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    o_state[s_idx] = static_cast<StT>(state[i]);
  }
}

// The model's linear attention: Dk = Dv = 128, Hk = 16, Hv = 32, activations
// bfloat16, state float32 (`mamba_ssm_dtype`).
#define instantiate_gdn(iname, itype, sname, stype, dk, dv, hk, hv)         \
  instantiate_kernel(                                                        \
      "gated_delta_packed_" #iname "_" #sname "_" #dk "_" #dv "_" #hk "_" #hv, \
      gated_delta_step_packed_btree, itype, stype, dk, dv, hk, hv)            \
  instantiate_kernel(                                                        \
      "gated_delta_" #iname "_" #sname "_" #dk "_" #dv "_" #hk "_" #hv,       \
      gated_delta_step, itype, stype, dk, dv, hk, hv)                         \
  instantiate_kernel(                                                        \
      "gated_delta_mask_" #iname "_" #sname "_" #dk "_" #dv "_" #hk "_" #hv,  \
      gated_delta_step_mask, itype, stype, dk, dv, hk, hv)

instantiate_gdn(bfloat16, bfloat16_t, float32, float, 128, 128, 16, 32)
instantiate_gdn(float32, float, float32, float, 128, 128, 16, 32)
