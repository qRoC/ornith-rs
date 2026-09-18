// ggml's Q6_K, for the other end of the artifact.
//
// Ported from `vendor/ggml/src/ggml-metal/kernels/mul_mv.metal`
// (`kernel_mul_mv_q6_K_f32_impl`) and `vendor/ggml/src/ggml-quants.c`
// (`dequantize_row_q6_K`), the same way `q4k.metal` was, and for the same
// reason: mlx has no k-quant, so ggml is the oracle.
//
// **Why this format and not another 8-bit one.** The 169 modules the artifact
// leaves at affine int8 g64 are 8.50 bits; Q6_K is 6.5625, so moving them cuts
// that side by 22.8%. `Q4K.md` priced it: 2.5x the weight error, which reads
// like the `mxfp8` Phase 4 rejected at 5.4x until it is put per bit -- 0.68
// error-doublings a bit here against the 0.96 the 4-bit side already pays.
// Those are the artifact's cheapest bits to give up.
//
// The block, and note it has **no minimum**: unlike Q4_K, reconstruction is
// `w = d * sc_i * (q - 32)` with a *signed* 8-bit sub-block scale and a
// 6-bit code biased by 32. There is nothing to subtract and nothing unsigned.
//
//   uint8_t ql[128]      the low 4 bits of 256 codes
//   uint8_t qh[64]       the high 2 bits of the same 256
//   int8_t  scales[16]   one per 16 weights, signed
//   half    d            one super-scale
//                        210 bytes / 256 weights = 6.5625 bits
//
// Storage keeps the three-tensor shape `Q4K.md` chose, with `ql` and `qh`
// packed into `.weight` back to back (192 bytes a block) so the loader sees
// the same three names it already knows. `.dm` is one value wide here, since
// Q6_K has no `dmin`.

#include "mlx/backend/metal/kernels/utils.h"

#define QK_K 256
#define Q6K_QL_BYTES 128
#define Q6K_QH_BYTES 64
#define Q6K_W_BYTES 192 // ql then qh, per block
#define Q6K_SC 16
#define Q6K_D 1

// `dequantize_row_q6_K`, one thread per 256-weight block.
//
// The check the test pins at `max_abs == 0.0` against ggml's own output. Two
// traps live here: the code is **biased by 32** and the sub-block scale is
// **signed**, so a reader that keeps Q4_K's unsigned conventions is wrong by
// an offset and a sign and still produces fluent text.
template <typename T>
[[kernel]] void q6k_dequantize(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint ib [[thread_position_in_grid]]) {
  const device uint8_t* ql = w + ib * Q6K_W_BYTES;
  const device uint8_t* qh = ql + Q6K_QL_BYTES;
  const device int8_t* sc = (const device int8_t*)(scales + ib * Q6K_SC);
  const float d = float(dm[ib * Q6K_D]);
  device T* y = out + ib * QK_K;

  for (int n = 0; n < QK_K; n += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int8_t q1 = int8_t((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int8_t q2 = int8_t((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int8_t q3 = int8_t((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int8_t q4 = int8_t((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l + 0] = T(d * sc[is + 0] * q1);
      y[l + 32] = T(d * sc[is + 2] * q2);
      y[l + 64] = T(d * sc[is + 4] * q3);
      y[l + 96] = T(d * sc[is + 6] * q4);
    }
    y += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

// `kernel_mul_mv_q6_K_f32_impl`, with the pointers resolved by the caller --
// the same split `q4k.metal` uses, so the routed wrapper is a wrapper.
//
// Two lanes share a block (`ix = tiisg % 2`) and the other sixteen ways split
// the block's 256 weights, so each lane holds 16 of them and the row loop
// re-reads only the weights.
template <typename T, short nr0, short NSG>
static inline void q6k_qmv_impl(
    const device uint8_t* w,
    const device uint8_t* scales,
    const device half* dm,
    const device T* yy,
    device T* dst,
    int K,
    int O,
    int tgx,
    ushort tiisg,
    ushort sgitg) {
  constexpr uint8_t kmask1 = 0x03;
  constexpr uint8_t kmask2 = 0x0C;
  constexpr uint8_t kmask3 = 0x30;
  constexpr uint8_t kmask4 = 0xC0;

  const int nb = K / QK_K;
  const int first_row = (tgx * NSG + sgitg) * nr0;

  // Row strides, in each tensor's own unit.
  const int w_row = nb * Q6K_W_BYTES;
  const int sc_row = nb * Q6K_SC;
  const int d_row = nb * Q6K_D;

  const short tid = tiisg / 2;
  const short ix = tiisg % 2;
  const short ip = tid / 8; // 0 or 1, which half of the block
  const short il = tid % 8;
  const short l0 = 4 * il;
  const short is = 8 * ip + l0 / 16;

  const short y_offset = 128 * ip + l0;
  const short q_offset_l = 64 * ip + l0;
  const short q_offset_h = 32 * ip + l0;

  float sumf[nr0] = {0.f};
  float yl[16];

  for (int i = ix; i < nb; i += 2) {
    const device uint8_t* q1 = w + first_row * w_row + i * Q6K_W_BYTES + q_offset_l;
    const device uint8_t* qh =
        w + first_row * w_row + i * Q6K_W_BYTES + Q6K_QL_BYTES + q_offset_h;
    const device int8_t* sc =
        (const device int8_t*)(scales + first_row * sc_row + i * Q6K_SC) + is;
    const device half* dh = dm + first_row * d_row + i * Q6K_D;

    const device T* y = yy + i * QK_K + y_offset;
    for (short l = 0; l < 4; ++l) {
      yl[4 * l + 0] = float(y[l + 0]);
      yl[4 * l + 1] = float(y[l + 32]);
      yl[4 * l + 2] = float(y[l + 64]);
      yl[4 * l + 3] = float(y[l + 96]);
    }

    for (short row = 0; row < nr0; ++row) {
      const device uint8_t* q2 = q1 + 32;
      float4 sums = {0.f, 0.f, 0.f, 0.f};

      for (short l = 0; l < 4; ++l) {
        sums[0] += yl[4 * l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
        sums[1] += yl[4 * l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
        sums[2] += yl[4 * l + 2] * ((int8_t)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
        sums[3] += yl[4 * l + 3] * ((int8_t)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
      }

      sumf[row] +=
          dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

      q1 += w_row;
      qh += w_row;
      sc += sc_row;
      dh += d_row;
    }
  }

  for (short row = 0; row < nr0 && first_row + row < O; ++row) {
    const float sum_all = simd_sum(sumf[row]);
    if (tiisg == 0) {
      dst[first_row + row] = T(sum_all);
    }
  }
}

template <typename T, short nr0, short NSG>
[[kernel]] void q6k_qmv(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& O [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  const int r1 = int(tgpig.y);
  q6k_qmv_impl<T, nr0, NSG>(
      w,
      scales,
      dm,
      x + (int64_t)r1 * K,
      out + (int64_t)r1 * O,
      K,
      O,
      int(tgpig.x),
      tiisg,
      sgitg);
}

// The MoE gather, as in `q4k.metal`: `lhs[r]` picks the token, `rhs[r]` the
// expert.
template <typename T, short nr0, short NSG>
[[kernel]] void q6k_gather_qmv(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& O [[buffer(6)]],
    const device uint32_t* lhs [[buffer(7)]],
    const device uint32_t* rhs [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  const int r = int(tgpig.y);
  const int64_t nb = K / QK_K;
  const int64_t e = int64_t(rhs[r]);
  q6k_qmv_impl<T, nr0, NSG>(
      w + e * O * nb * Q6K_W_BYTES,
      scales + e * O * nb * Q6K_SC,
      dm + e * O * nb * Q6K_D,
      x + int64_t(lhs[r]) * K,
      out + (int64_t)r * O,
      K,
      O,
      int(tgpig.x),
      tiisg,
      sgitg);
}

// ggml's `N_SG_Q6_K` / `N_R0_Q6_K`.
#define instantiate_q6k(tname, type)                                  \
  instantiate_kernel("q6k_dequantize_" #tname, q6k_dequantize, type)  \
  instantiate_kernel("q6k_qmv_" #tname, q6k_qmv, type, 2, 2)          \
  instantiate_kernel("q6k_gather_qmv_" #tname, q6k_gather_qmv, type, 2, 2)

instantiate_q6k(bfloat16, bfloat16_t) instantiate_q6k(float32, float)
