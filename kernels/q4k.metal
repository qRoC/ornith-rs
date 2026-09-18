// ggml's Q4_K, for the one shape a decode step needs.
//
// Ported from `vendor/ggml/src/ggml-metal/kernels/mul_mv.metal`
// (`kernel_mul_mv_q4_K_f32_impl`) and `vendor/ggml/src/ggml-quants.c`
// (`dequantize_row_q4_K`), the same relationship this port already has with
// mlx's kernels: the arithmetic is read off a working implementation rather
// than re-derived from the format description. `Q4K.md` says why -- mlx has no
// k-quant mode and is not going to get one, so ggml is the oracle, and an
// oracle is only an oracle if the kernel is *its* kernel.
//
// Two differences from ggml's, both deliberate:
//
//   the layout    ggml keeps one packed 144-byte `block_q4_K`; the artifact
//                 keeps three tensors per module, so the loader and
//                 `check_rebuild.py` see the shapes they already know. The
//                 bytes within each are ggml's own, so only the addressing
//                 moves: a row's blocks are `nb * 64` uint16 of codes,
//                 `nb * 6` uint16 of packed 6-bit fields and `nb * 2` halves.
//
//   the activation  ggml multiplies against float32; this model's activations
//                 are bfloat16, so the type is a template parameter and the
//                 accumulation stays in float exactly as ggml has it.
//
// Reconstruction, per sub-block of 32:  w = d * sc - dmin * m,  both fields
// unsigned 6-bit and the second **subtracted**. An affine dequantizer's sign
// convention is wrong here in a way that still produces fluent text.

#include "mlx/backend/metal/kernels/utils.h"

#define QK_K 256
#define Q4K_QS_U16 64 // 128 bytes of 4-bit codes, as uint16
#define Q4K_SC_U16 6  // 12 bytes of packed 6-bit fields, as uint16
#define Q4K_DM_F16 2  // d and dmin

// `get_scale_min_k4`: the sub-block's 6-bit scale and minimum, out of the
// twelve bytes that hold all sixteen of them. Six of those bytes carry the low
// bits of the last eight fields, which is why this is not the obvious layout
// and why it is ported rather than re-derived.
static inline void get_scale_min_k4(
    int j,
    const device uint8_t* q,
    thread uint8_t& d,
    thread uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// `dequantize_row_q4_K`, one thread per 256-weight block.
//
// This is the kernel the test pins at `max_abs == 0.0` against ggml's own
// output on ggml's own bytes. A matvec cannot be pinned that way -- its
// summation order is the kernel's, not the reference's -- so the bit-exact
// contract lives here, and it covers all three traps at once: the 12-byte
// packing above, the nibble interleave below, and the subtraction.
template <typename T>
[[kernel]] void q4k_dequantize(
    const device uint8_t* qs [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint ib [[thread_position_in_grid]]) {
  const device uint8_t* q = qs + ib * 128;
  const device uint8_t* sc = scales + ib * 12;
  const float d = float(dm[ib * 2 + 0]);
  const float dmin = float(dm[ib * 2 + 1]);
  device T* y = out + ib * QK_K;

  int is = 0;
  uint8_t s, m;
  for (int j = 0; j < QK_K; j += 64) {
    get_scale_min_k4(is + 0, sc, s, m);
    const float d1 = d * s;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, sc, s, m);
    const float d2 = d * s;
    const float m2 = dmin * m;
    for (int l = 0; l < 32; ++l) {
      y[l] = T(d1 * (q[l] & 0xF) - m1);
    }
    for (int l = 0; l < 32; ++l) {
      y[32 + l] = T(d2 * (q[l] >> 4) - m2);
    }
    q += 32;
    y += 64;
    is += 2;
  }
}

// `kernel_mul_mv_q4_K_f32_impl`.
//
// One simdgroup carries `nr0` output rows; `NSG` of them share a threadgroup.
// The 32 lanes split four ways over blocks (`ix`) and eight ways within one
// (`it`), so each lane holds 32 of the 256 activations for its block and the
// row loop re-reads only the weights. ggml's own `N_SG_Q4_K` / `N_R0_Q4_K` are
// 2 and 2.
//
// **This is a mat-VEC, including when M > 1.** ggml's kernel carries an `r1`
// axis -- the grid's y -- that walks the columns of `src1`, and a decode step
// never uses it. It is here for prefill, and the cost is explicit: the weights
// are re-read once per token instead of being tiled, so a chunk of 128 reads
// the matrix 128 times. That is the wrong shape for prefill and the right
// shape for *having* a prefill at all -- `Q4K.md` leaves `q4k_qmm_t` for last,
// and until it exists this is what lets a q4k artifact be measured rather than
// only benchmarked at M = 1.
// The body, with every pointer already resolved: the caller decides which
// expert's weights and which token's activations these are. ggml splits it the
// same way -- `kernel_mul_mv_id` is a wrapper that offsets `src0` by the
// expert and `src1` by the token and then calls this.
template <typename T, short nr0, short NSG>
static inline void q4k_qmv_impl(
    const device uint16_t* qs,
    const device uint16_t* scales,
    const device half* dm,
    const device T* y,
    device T* dst,
    int K,
    int O,
    int tgx,
    ushort tiisg,
    ushort sgitg) {
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;

  const short ix = tiisg / 8; // 0...3, which block of the four in flight
  const short it = tiisg % 8; // 0...7, where in the block
  const short iq = it / 4; // 0 or 1, which half of the super-block
  const short ir = it % 4; // 0...3

  const int nb = K / QK_K;
  const int first_row = (tgx * NSG + sgitg) * nr0;

  // Row strides, in each tensor's own unit.
  const int qs_row = nb * Q4K_QS_U16;
  const int sc_row = nb * Q4K_SC_U16;
  const int dm_row = nb * Q4K_DM_F16;

  float yl[16];
  float yh[16];
  float sumf[nr0] = {0.f};

  const device T* y4 = y + ix * QK_K + 64 * iq + 8 * ir;

  uint16_t sc16[4];
  thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

  for (int ib = ix; ib < nb; ib += 4) {
    float4 sumy = {0.f, 0.f, 0.f, 0.f};

    for (short i = 0; i < 8; ++i) {
      yl[i + 0] = float(y4[i + 0]);
      sumy[0] += yl[i + 0];
      yl[i + 8] = float(y4[i + 32]);
      sumy[1] += yl[i + 8];
      yh[i + 0] = float(y4[i + 128]);
      sumy[2] += yh[i + 0];
      yh[i + 8] = float(y4[i + 160]);
      sumy[3] += yh[i + 8];
    }

    const device uint16_t* sc = scales + first_row * sc_row + ib * Q4K_SC_U16 + iq;
    const device uint16_t* q1 = qs + first_row * qs_row + ib * Q4K_QS_U16 + 16 * iq + 4 * ir;
    const device half* dh = dm + first_row * dm_row + ib * Q4K_DM_F16;

    for (short row = 0; row < nr0; row++) {
      sc16[0] = sc[0] & kmask1;
      sc16[1] = sc[2] & kmask1;
      sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
      sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

      const device uint16_t* q2 = q1 + 32;

      float4 acc1 = {0.f, 0.f, 0.f, 0.f};
      float4 acc2 = {0.f, 0.f, 0.f, 0.f};

      for (short i = 0; i < 4; ++i) {
        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
      }

      sumf[row] +=
          dh[0] *
              ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
               (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
               (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
               (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
          dh[1] *
              (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] +
               sumy[3] * sc8[7]);

      q1 += qs_row;
      sc += sc_row;
      dh += dm_row;
    }

    y4 += 4 * QK_K;
  }

  for (short row = 0; row < nr0 && first_row + row < O; ++row) {
    const float sum_all = simd_sum(sumf[row]);
    if (tiisg == 0) {
      dst[first_row + row] = T(sum_all);
    }
  }
}

// The plain matvec: row `r1` of `x` against the one weight matrix.
template <typename T, short nr0, short NSG>
[[kernel]] void q4k_qmv(
    const device uint16_t* qs [[buffer(0)]],
    const device uint16_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& O [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  const int r1 = int(tgpig.y);
  q4k_qmv_impl<T, nr0, NSG>(
      qs,
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

// `kernel_mul_mv_id`: the MoE gather. One output row per (token, expert) pair
// -- `lhs[r]` says which row of `x` it reads and `rhs[r]` which expert's
// weights it reads, which is `mx.gather_qmm`'s contract and `SwitchGLU`'s
// shape. The weights are `[E, O, I]`, so an expert is one stride away in each
// of the three tensors.
template <typename T, short nr0, short NSG>
[[kernel]] void q4k_gather_qmv(
    const device uint16_t* qs [[buffer(0)]],
    const device uint16_t* scales [[buffer(1)]],
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
  q4k_qmv_impl<T, nr0, NSG>(
      qs + e * O * nb * Q4K_QS_U16,
      scales + e * O * nb * Q4K_SC_U16,
      dm + e * O * nb * Q4K_DM_F16,
      x + int64_t(lhs[r]) * K,
      out + (int64_t)r * O,
      K,
      O,
      int(tgpig.x),
      tiisg,
      sgitg);
}

#define instantiate_q4k(tname, type)                                  \
  instantiate_kernel("q4k_dequantize_" #tname, q4k_dequantize, type)  \
  instantiate_kernel("q4k_qmv_" #tname, q4k_qmv, type, 2, 2)          \
  instantiate_kernel("q4k_gather_qmv_" #tname, q4k_gather_qmv, type, 2, 2)

instantiate_q4k(bfloat16, bfloat16_t) instantiate_q4k(float32, float)
