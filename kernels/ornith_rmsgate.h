// `Qwen3NextRMSNormGated` as a prologue, so the matvec that reads it can be
// the dispatch that computes it.
//
// `REBUILD.md` Phase 33 measured the two dispatches that close every
// gated-delta layer at **4.8 us each, three times the elementwise chain's
// average**, because they are the serial spine between the recurrence and
// `out_proj` with nothing to overlap into -- and measured a pairwise fusion
// of the two at exactly zero. The only remaining way to that time is to
// fold the norm into the matvec that consumes it, which is what a deferred
// op layer would do in general and what this does for the one place in the
// model where a norm has exactly one consumer.
//
// The cost of doing it here rather than in a layer is redundancy: a mat-vec
// puts one threadgroup on each group of output rows and every one of them
// reads the whole activation row, so every one of them computes the whole
// norm. That is the thing the measurement is for.
//
// **Bit-exact by construction.** `rms_gated_single_row` gives each of 32
// threads `RMS_N_READS = 4` contiguous elements, sums the squares in float in
// ascending order within a thread, folds the threads with `simd_sum`, and
// then writes `precise_swiglu(gate, w * T(x * inv_mean))`. At `head_dim =
// 128` that is one simdgroup and exactly the layout below, so this is the
// same arithmetic in the same order, and the value that lands in threadgroup
// memory is the value that would have landed in device memory.

#pragma once

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "ornith_sigmoid.h"

// The longest activation row this prologue will hold, in elements. The one
// call site is `out_proj`, whose input is `Hv * Dv = 4096`.
#define ORNITH_RMSGATE_MAX 4096

// Fill `row[0..K)` with `precise_swiglu(gate, rms_norm(x, w, eps))`, the
// threadgroup cooperating over `K / head_dim` rows of `head_dim` each.
// Leaves a threadgroup barrier behind it, so every thread may read all of it.
template <typename T, int N_READS = 4>
static inline void ornith_rmsgate_row(
    const device T* x,
    const device T* w,
    const device T* gate,
    threadgroup T* row,
    int K,
    int head_dim,
    float eps,
    ushort tiisg,
    ushort sgitg,
    ushort nsg) {
  const int n_heads = K / head_dim;
  for (int h = sgitg; h < n_heads; h += nsg) {
    const int base = h * head_dim + tiisg * N_READS;
    float thread_x[N_READS];
    float acc = 0;
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[base + i];
      acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    const float inv_mean = metal::precise::rsqrt(acc / head_dim + eps);
    for (int i = 0; i < N_READS; i++) {
      T normed = w[tiisg * N_READS + i] * static_cast<T>(thread_x[i] * inv_mean);
      float g = static_cast<float>(gate[base + i]);
      g = g * mlx_sigmoid(g);
      row[base + i] = static_cast<T>(g * static_cast<float>(normed));
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
