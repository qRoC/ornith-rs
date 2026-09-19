// The residual add and the norm that reads it, in one dispatch.
//
// `REBUILD.md` Phase 31 measured what a decode step's dispatches are made of:
// of 1665 a token, 191 are `rms` and 120 are `add_elem`, and neither does any
// work worth the name -- clamping their grids to one threadgroup, so that
// they launch and compute nothing, does not move the step. What they cost is
// being dispatched: 0.33 ms and 0.22 ms a token, all of it launch.
//
// `rms after add_elem` is also the most frequent barrier pair in the step, 78
// of them a token: every layer's residual is added, and the sum is then
// normalised. So the two are one kernel here.
//
// Exactness is the whole point, and it is by construction rather than by
// tolerance: `h = a + b` is evaluated in T exactly as `add_elem` does it, the
// **rounded** h is what the sum of squares accumulates and what the output is
// scaled from, and the write-back is `rms_single_row`'s line unchanged. The
// only thing that moves is that one kernel does both and h is kept in
// registers between them instead of making a round trip to memory.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"
#include "ornith_sigmoid.h"

using namespace metal;

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void add_rms_single_row(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    const device T* w [[buffer(2)]],
    device T* h [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& axis_size [[buffer(6)]],
    constant uint& w_stride [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  float thread_x[N_READS];
  size_t offset = gid * size_t(axis_size) + lid * N_READS;
  a += offset;
  b += offset;
  h += offset;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      T sum = a[i] + b[i];
      h[i] = sum;
      thread_x[i] = sum;
      acc += thread_x[i] * thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (lid * N_READS + i < axis_size) {
        T sum = a[i] + b[i];
        h[i] = sum;
        thread_x[i] = sum;
      } else {
        thread_x[i] = 0;
      }
      acc += thread_x[i] * thread_x[i];
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  out += offset;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] =
          w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] =
            w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      }
    }
  }
}

// `Qwen3NextRMSNormGated`: the norm and the gate that reads it, in one
// dispatch.
//
// The gated-delta layer ends with `precise_swiglu(z, rms_norm(y, w, eps))`,
// two dispatches over 128 elements a head that between them do no work at
// all. This is `rms_single_row` with `precise_swiglu`'s three lines in place
// of its write-back, so the normalised value is a `T` local where it used to
// be a round trip to memory -- and it rounds to bfloat16 at the same point,
// since the store it replaces was the rounding.
//
// `ORNITH_FUSE_GDN_NORM=0` issues the two separately, as the control.
template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_gated_single_row(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    const device T* gate [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& axis_size [[buffer(5)]],
    constant uint& w_stride [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  float thread_x[N_READS];
  size_t offset = gid * size_t(axis_size) + lid * N_READS;
  x += offset;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      acc += thread_x[i] * thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = (lid * N_READS + i < axis_size) ? (float)x[i] : 0;
      acc += thread_x[i] * thread_x[i];
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  gate += offset;
  out += offset;
  for (int i = 0; i < N_READS; i++) {
    if (lid * N_READS + i < axis_size) {
      T normed =
          w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      float g = static_cast<float>(gate[i]);
      g = g * mlx_sigmoid(g);
      out[i] = static_cast<T>(g * static_cast<float>(normed));
    }
  }
}

#define instantiate_add_rms(tname, type) \
  instantiate_kernel("add_rms_" #tname, add_rms_single_row, type)

instantiate_add_rms(bfloat16, bfloat16_t)
instantiate_add_rms(float32, float)

#define instantiate_rms_gated(tname, type) \
  instantiate_kernel("rms_gated_" #tname, rms_gated_single_row, type)

instantiate_rms_gated(bfloat16, bfloat16_t)
instantiate_rms_gated(float32, float)
