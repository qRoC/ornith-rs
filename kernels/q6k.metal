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

#include "kquant_mm.h"

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
// the same split `q4k.metal` uses, so the routed wrapper is a wrapper. It is a
// mat-vec at every M, and `kquant_mm.h` is the tile that takes over past
// `kquant::MM_ROW_LIMIT` rows.
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

// The format for `kquant_mm.h`'s tile, as `Q4KFmt` is for Q4_K.
//
// ggml's `dequantize_q6_K`, with `xb->ql` / `xb->qh` resolved out of the one
// `.weight` tensor the artifact stores -- 128 bytes of low nibbles followed by
// 64 of high pairs -- and `xb->scales` and `xb->d` out of the other two. Its
// packing trick is kept verbatim: four codes are assembled inside one uint32
// and unpacked by dividing the super-scale by successive powers of 256, which
// is exact, and the bias of 32 is folded into `ml = d * sc * 32` and
// subtracted rather than applied to each code.
//
// `il` means what it means in `Q4KFmt`: the il-th run of 16 weights. Q6_K has
// one signed scale per 16, so that run is exactly `scales[il]`.
struct Q6KFmt {
  static constant constexpr const short QK = QK_K;
  static constant constexpr const short NL = 16;
  static constant constexpr const int CODES = Q6K_W_BYTES; // ql then qh
  static constant constexpr const int SCALES = Q6K_SC; // one int8 per 16
  static constant constexpr const int DM = Q6K_D; // d alone; no minimum

  static inline void dequantize(
      const device uint8_t* w,
      const device uint8_t* scales,
      const device half* dm,
      short il,
      thread float4x4& reg) {
    const half d_all = dm[0];
    const device uint16_t* ql = (const device uint16_t*)w;
    const device uint16_t* qh = (const device uint16_t*)(w + Q6K_QL_BYTES);
    const device int8_t* sc = (const device int8_t*)scales;

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    const float scale = sc[(il % 2) + 2 * ((il / 2))];
    il = (il / 2) & 3;

    const uint32_t kmask1 =
        il > 1 ? (il > 2 ? 0xC0C0C0C0 : 0x30303030) : (il > 0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml = float(d_all) * scale * 32.f;
    const float dl0 = float(d_all) * scale;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il > 2 ? 2 : 0;
    const uint8_t shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
    const uint8_t shr_l = il > 1 ? 4 : 0;
    for (short i = 0; i < 4; ++i) {
      const uint32_t low = (ql[2 * i] | (uint32_t)(ql[2 * i + 1] << 16)) & kmask2;
      const uint32_t high = (qh[2 * i] | (uint32_t)(qh[2 * i + 1] << 16)) & kmask1;
      const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
      reg[i][0] = dl0 * ((float)(q & 0xFF)) - ml;
      reg[i][1] = dl1 * ((float)(q & 0xFF00)) - ml;
      reg[i][2] = dl2 * ((float)(q & 0xFF0000)) - ml;
      reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
  }
};

// ggml's `N_SG_Q6_K` / `N_R0_Q6_K`.
#define instantiate_q6k(tname, type)                                           \
  instantiate_kernel("q6k_dequantize_" #tname, q6k_dequantize, type)           \
  instantiate_kernel("q6k_qmv_" #tname, q6k_qmv, type, 2, 2)                   \
  instantiate_kernel("q6k_gather_qmv_" #tname, q6k_gather_qmv, type, 2, 2)     \
  instantiate_kernel(                                                          \
      "q6k_qmm_t_r32_" #tname, kquant_qmm_t, type, Q6KFmt, float, 32)         \
  instantiate_kernel(                                                          \
      "q6k_qmm_t_r32h_" #tname, kquant_qmm_t, type, Q6KFmt, half, 32)         \
  instantiate_kernel(                                                          \
      "q6k_gather_qmm_t_r32h_" #tname, kquant_gather_qmm_t, type, Q6KFmt,     \
      half, 32)                                                                \
  instantiate_kernel(                                                          \
      "q6k_qmm_t_r16_" #tname, kquant_qmm_t, type, Q6KFmt, float, 16)         \
  instantiate_kernel(                                                          \
      "q6k_qmm_t_r64_" #tname, kquant_qmm_t, type, Q6KFmt, float, 64)         \
  instantiate_kernel(                                                          \
      "q6k_gather_qmm_t_r64_" #tname, kquant_gather_qmm_t, type, Q6KFmt,      \
      float, 64)                                                               \
  instantiate_kernel(                                                          \
      "q6k_gather_qmm_t_r32_" #tname, kquant_gather_qmm_t, type, Q6KFmt,      \
      float, 32)                                                               \
  instantiate_kernel(                                                          \
      "q6k_gather_qmm_t_r16_" #tname, kquant_gather_qmm_t, type, Q6KFmt,      \
      float, 16)

instantiate_q6k(bfloat16, bfloat16_t) instantiate_q6k(float32, float)
