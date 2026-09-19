// Measurement kernels. Nothing in the model dispatches these; they exist so
// `bin/bandwidth` can ask what this machine actually sustains, as opposed to
// what its datasheet says.

#include "mlx/backend/metal/kernels/utils.h"

// A read and nothing else: grid-stride over the buffer in 16-byte chunks, one
// xor per chunk so the loads cannot be optimised away. The xor is a single ALU
// op against 16 bytes of traffic, so what this measures is the memory system.
[[kernel]] void stream_read(
    const device uint4* x [[buffer(0)]],
    device uint* out [[buffer(1)]],
    const constant uint& n4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]]) {
  uint4 acc = uint4(0);
  for (uint i = gid; i < n4; i += gsize) {
    acc ^= x[i];
  }
  out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

// --- what the ALUs sustain, as opposed to what the datasheet says -----------
//
// `stream_read` above answers the memory side and `bin/bandwidth` reads it.
// After Phase 36 a prefill chunk is **81% quantized gemm**, so every
// remaining question about it is an arithmetic question, and there was no
// measured ceiling to ask it against -- only a number off a box.
//
// Two kernels, and the first one exists mostly as a trap to avoid.
//
// **A dependent FMA chain measures latency, not throughput.** `a = fma(a, b,
// c)` cannot start until the previous one retires, so a loop of them reads
// the pipeline depth however many ALUs are idle behind it. `ACC` independent
// chains fix that, and the right `ACC` is not something to guess: it is swept
// (`bin/arithmetic`), and the shape of that sweep is itself the answer to
// "how deep is the pipe". Same rule as Phase 34 -- sweep the one parameter
// that trades a mechanism's cost against its benefit, and report the floor.
//
// `-fno-fast-math` is on for every kernel in this port, which matters here
// more than anywhere: without it `a * b + c` is *not* licensed to fuse, so
// counting it as two flops in one instruction would overstate the machine by
// 2x. These call `metal::fma` explicitly and count 2 flops each.
template <int ACC>
[[kernel]] void fma_peak(
    device float* out [[buffer(0)]],
    const constant uint& iters [[buffer(1)]],
    const constant float& seed [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  // From a runtime constant, so nothing here folds at compile time.
  float b = seed + float(gid & 7);
  float c = seed * 0.5f;
  float a[ACC];
#pragma clang loop unroll(full)
  for (int i = 0; i < ACC; ++i) {
    a[i] = seed + float(i);
  }
  for (uint t = 0; t < iters; ++t) {
#pragma clang loop unroll(full)
    for (int i = 0; i < ACC; ++i) {
      a[i] = metal::fma(a[i], b, c);
    }
  }
  float s = 0;
#pragma clang loop unroll(full)
  for (int i = 0; i < ACC; ++i) {
    s += a[i];
  }
  out[gid] = s;
}

// The same question of the matrix units, which is the one that matters: both
// the k-quant tile and mlx's steel gemm spend their whole K loop in
// `simdgroup_multiply_accumulate`.
//
// `S` is the A/B fragment type and the accumulator stays float32, which is
// exactly `kquant_mm_tile`'s arrangement -- it instantiates at `S = float`
// and carries `S = half` behind `ORNITH_KQUANT_TILE_HALF`. That flag measured
// *free* on the real kernel (0-4%, `src/ops/kquant.rs`), and this is the
// kernel that says what that means: if the matrix unit runs half faster than
// float and the real kernel does not care, the real kernel is not bound by
// the matrix unit.
//
// The fragments are loaded from threadgroup memory once, before the loop, the
// way the real tile loads them -- not filled from a literal, which would let
// the compiler hoist the whole product.
template <typename S, int TILES>
[[kernel]] void mma_peak(
    device float* out [[buffer(0)]],
    const constant uint& iters [[buffer(1)]],
    const constant float& seed [[buffer(2)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tptg [[threads_per_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tgsg [[simdgroups_per_threadgroup]]) {
  threadgroup S tile[128];
  for (ushort i = tiitg; i < 128; i += tptg) {
    tile[i] = S(seed + float(i & 3));
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  metal::simdgroup_matrix<S, 8, 8> ma;
  metal::simdgroup_matrix<S, 8, 8> mb;
  metal::simdgroup_load(ma, tile, 8, 0, false);
  metal::simdgroup_load(mb, tile + 64, 8, 0, false);

  metal::simdgroup_float8x8 mc[TILES];
#pragma clang loop unroll(full)
  for (int i = 0; i < TILES; ++i) {
    mc[i] = metal::make_filled_simdgroup_matrix<float, 8>(0.f);
  }

  for (uint t = 0; t < iters; ++t) {
#pragma clang loop unroll(full)
    for (int i = 0; i < TILES; ++i) {
      metal::simdgroup_multiply_accumulate(mc[i], ma, mb, mc[i]);
    }
  }

  device float* dst = out + (tgpig * tgsg + sgitg) * (TILES * 64);
#pragma clang loop unroll(full)
  for (int i = 0; i < TILES; ++i) {
    metal::simdgroup_store(mc[i], dst + i * 64, 8);
  }
}

#define instantiate_fma(acc) \
  instantiate_kernel("fma_peak_" #acc, fma_peak, acc)

instantiate_fma(1)
instantiate_fma(2)
instantiate_fma(4)
instantiate_fma(8)
instantiate_fma(16)
instantiate_fma(32)

#define instantiate_mma(tname, type, tiles) \
  instantiate_kernel("mma_peak_" #tname "_" #tiles, mma_peak, type, tiles)

instantiate_mma(float32, float, 1)
instantiate_mma(float32, float, 2)
instantiate_mma(float32, float, 4)
instantiate_mma(float32, float, 8)
instantiate_mma(float32, float, 16)
instantiate_mma(float16, half, 1)
instantiate_mma(float16, half, 2)
instantiate_mma(float16, half, 4)
instantiate_mma(float16, half, 8)
instantiate_mma(float16, half, 16)
instantiate_mma(bfloat16, bfloat16_t, 4)
