// The spike for Phase 38's one open object: Q4_K on mlx's steel tiling.
//
// **What this is testing, and it is a premise rather than a speed.** Phase 37
// measured mlx's quantized gemm 13% faster than ggml's tile on one shape while
// doing more work, and Phase 38 showed that part of that margin is bought with
// a rounding this port refuses: `QuantizedBlockLoader` dequantizes into a
// `threadgroup T*`, so every weight mlx multiplies has been through bfloat16.
// This tile therefore keeps the threadgroup arrays at **`float`**, which is
// what `kquant_mm.h` promises and what `max_abs == 0.0` against it requires.
//
// So there are two questions and the first one is not about time:
//
//   1. is a steel-tiled gemm at `float` tiles **bit-identical** to
//      `kquant_qmm_t`? Both walk K in ascending 8-element fragments into a
//      float accumulator, so it should be -- but "should be" is what the last
//      phase was about, and an 8-term dot product's internal order is the
//      hardware's, not ours;
//   2. if it is, how much of the eight points does it reach?
//
// A "no" to the first makes this a different project. A "yes" to the first and
// a "nothing" to the second means the eight points are not where they were
// counted. Either is worth more than a finished loader built on the guess.
//
// Dense Q4_K only, and no partial tiles: the shape it is measured at divides
// by 32 every way, exactly as mlx's own `aligned_N` arm does, so the two are
// compared running the same branches. Q6_K and the routed variant follow only
// if this answers yes.

#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/utils.h"

// The activations, bfloat16 in device memory and **float** in the tile.
//
// mlx's `BlockLoader` cannot do this: it copies `ReadVector`s of `sizeof(T)`
// bytes, one type at both ends. Widening is a conversion, not a copy, so this
// reads elementwise -- 8 contiguous values a thread, which is a 16-byte read
// at bfloat16 and the width Phase 36/37 cared about.
template <typename T, typename S, short BROWS, short BCOLS, short dst_ld, short tgp_size>
struct WideningBlockLoader {
  STEEL_CONST short n_reads = (BROWS * BCOLS) / tgp_size;
  STEEL_CONST short threads_per_row = BCOLS / n_reads;

  const int src_ld;
  const device T* src;
  threadgroup S* dst;
  const short bi;
  const short bj;

  METAL_FUNC WideningBlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup S* dst_,
      ushort simd_group_id,
      ushort simd_lane_id)
      : src_ld(src_ld_),
        src(src_),
        dst(dst_),
        bi((simd_group_id * 32 + simd_lane_id) / threads_per_row),
        bj(((simd_group_id * 32 + simd_lane_id) % threads_per_row) * n_reads) {}

  METAL_FUNC void load_unsafe() const {
    const device T* s = src + bi * src_ld + bj;
    threadgroup S* d = dst + bi * dst_ld + bj;
#pragma clang loop unroll(full)
    for (short i = 0; i < n_reads; i++) {
      d[i] = S(s[i]);
    }
  }

  METAL_FUNC void next() {
    src += BCOLS;
  }
};

// The weights: the k-quant analogue of `QuantizedBlockLoader`.
//
// **The arithmetic is `Q4KFmt::dequantize`'s, expression for expression**, and
// that is load-bearing rather than tidy: ggml compensates for not shifting the
// high nibble down by dividing `d` by 16, so `(d / 16) * sc * (q & 0xF0)` and
// `d * sc * (q >> 4)` are the same number in algebra and not in float. The
// oracle is bit-exactness against the kernel that ships, so the expression it
// ships is the one that goes here.
//
// **A 32-wide K tile is exactly one Q4_K sub-block**, which is the one piece
// of luck in this format: the 256-weight super-block carries eight sub-blocks
// of 32, each with its own 6-bit scale and 6-bit minimum, so a thread's eight
// weights share one `scale_min` and it is hoisted out of the loop. (Q6_K's
// sub-block is 16, so the same tile spans two of them and needs two.)
template <typename Q, typename S, short BROWS, short BCOLS, short dst_ld, short tgp_size>
struct KQuantBlockLoader {
  static_assert(BCOLS == 32, "a 32-wide K tile is one Q4_K sub-block");
  STEEL_CONST short n_reads = (BROWS * BCOLS) / tgp_size;
  STEEL_CONST short threads_per_row = BCOLS / n_reads;

  const int nb; // super-blocks a row
  const device uint8_t* qs;
  const device uint8_t* scales;
  const device half* dm;
  threadgroup S* dst;
  const short bi;
  const short bj;
  int k0;

  METAL_FUNC KQuantBlockLoader(
      const device uint8_t* qs_,
      const device uint8_t* scales_,
      const device half* dm_,
      const int K,
      threadgroup S* dst_,
      ushort simd_group_id,
      ushort simd_lane_id)
      : nb(K / Q::QK),
        qs(qs_),
        scales(scales_),
        dm(dm_),
        dst(dst_),
        bi((simd_group_id * 32 + simd_lane_id) / threads_per_row),
        bj(((simd_group_id * 32 + simd_lane_id) % threads_per_row) * n_reads),
        k0(0) {}

  METAL_FUNC void load_unsafe() const {
    // Where this thread's eight weights are: super-block, sub-block, and the
    // offset inside it. The tile is 32-aligned in K, so all eight share the
    // sub-block and therefore the scale, the minimum, the mask and the `d`.
    const int k = k0 + bj;
    const int sb = k >> 8;
    const short s = short((k >> 5) & 7);
    const short w = short(k & 31);

    const device uint8_t* q =
        qs + (int64_t)bi * nb * Q::CODES + (int64_t)sb * Q::CODES + (s >> 1) * 32 + w;
    const device uint8_t* sp = scales + (int64_t)bi * nb * Q::SCALES + (int64_t)sb * Q::SCALES;
    const device half* dp = dm + (int64_t)bi * nb * Q::DM + (int64_t)sb * Q::DM;

    const uchar2 sc = Q::scale_min(s & ~1, s & 1, sp);
    const float d = (s & 1) ? float(dp[0]) / 16.f : float(dp[0]);
    const float dl = d * sc[0];
    const float ml = float(dp[1]) * sc[1];
    const ushort mask = (s & 1) ? 0xF0 : 0x0F;

    threadgroup S* o = dst + bi * dst_ld + bj;
#pragma clang loop unroll(full)
    for (short i = 0; i < n_reads; i++) {
      o[i] = S(dl * (q[i] & mask) - ml);
    }
  }

  METAL_FUNC void next() {
    k0 += BCOLS;
  }
};

// **And the arm that separates the tile type from this loader.** Instantiated
// at `S = T` as well as `S = float`, the first is mlx's own choice: the
// dequantized weight rounded to bfloat16 on the way into shared memory, half
// the threadgroup traffic and half the threadgroup footprint. It **produces
// different numbers**, like `Q4KBareFmt` and `ORNITH_FAKE_SORT`, and exists
// for one reason: if the bfloat16 arm reaches mlx's rate and the float arm
// does not, the gap is the tile type and this loader is sound. If neither
// does, the loader is what is slow and the eight points were never the
// tiling. A control has to be able to fail.
//
// `x @ w.T` on steel's `BlockMMA`, tile scalar `S`.
//
// The shape is mlx's own for this call (`ops::quantized::qmm_t`): BM = BN =
// BK = 32, WM = WN = 2, 128 threads, one threadgroup a 32x32 output tile. The
// padding is `16 / sizeof(float)` = 4 rather than bfloat16's 8, which is what
// makes the float tile cost 9 KB of threadgroup memory instead of 4.5.
template <typename T, typename Q, typename S, short BM, short BN, short BK>
[[kernel]] void kquant_steel_qmm_t(
    const device uint8_t* qs [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& O [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort simd_lid [[thread_index_in_simdgroup]]) {
  constexpr short WM = 2;
  constexpr short WN = 2;
  constexpr short BK_padded = BK + 16 / sizeof(S);

  threadgroup S Xs[BM * BK_padded];
  threadgroup S Ws[BN * BK_padded];

  using mma_t = mlx::steel::
      BlockMMA<S, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = WideningBlockLoader<T, S, BM, BK, BK_padded, WM * WN * 32>;
  using loader_w_t = KQuantBlockLoader<Q, S, BN, BK, BK_padded, WM * WN * 32>;

  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;
  const int nb = K / Q::QK;

  loader_x_t loader_x(x + (int64_t)y_row * K, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(
      qs + (int64_t)y_col * nb * Q::CODES,
      scales + (int64_t)y_col * nb * Q::SCALES,
      dm + (int64_t)y_col * nb * Q::DM,
      K,
      Ws,
      simd_gid,
      simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_unsafe();
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.store_result(out + (int64_t)y_row * O + y_col, O);
}
