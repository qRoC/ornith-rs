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

#include "kquant_mm.h"
#include "kquant_steel.h"
#include "ornith_rmsgate.h"

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
// never uses it. The weights are then re-read once per token instead of being
// tiled, so a chunk of 128 reads the matrix 128 times: the right shape up to a
// handful of rows and the wrong one past that. `kquant_mm.h` is the tile that
// takes over, and `kquant::MM_ROW_LIMIT` is the measured row count where it
// does -- 8, with this kernel keeping everything below.
// The body, with every pointer already resolved: the caller decides which
// expert's weights and which token's activations these are. ggml splits it the
// same way -- `kernel_mul_mv_id` is a wrapper that offsets `src0` by the
// expert and `src1` by the token and then calls this.
// `YP` is the activation pointer's address space: `const device T*` for every
// caller here but the fused `rmsgate` one, which has already normalised the
// row into threadgroup memory. Nothing in the body depends on which it is.
template <typename T, short nr0, short NSG, typename YP>
static inline void q4k_qmv_impl(
    const device uint16_t* qs,
    const device uint16_t* scales,
    const device half* dm,
    YP y,
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

  YP y4 = y + ix * QK_K + 64 * iq + 8 * ir;

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

// `Qwen3NextRMSNormGated` and the projection that reads it, in one dispatch.
//
// The one place in this model where a norm has exactly one consumer, which is
// what makes it a fusion rather than an op layer: every other norm here feeds
// four or six matvecs. See `ornith_rmsgate.h` for the exactness argument and
// `REBUILD.md` Phase 33 for why this time is worth reaching for.
//
// M = 1 only. Each threadgroup normalises the whole activation row into
// threadgroup memory and then runs the mat-vec above over it unchanged --
// so the activation reads the body makes get cheaper, and the price is that
// all `O / (NSG * nr0)` threadgroups do the same normalisation. That count is
// what `nr0` and `NSG` are swept for: at ggml's 2 and 2 an `O = 2048`
// projection runs 512 threadgroups and normalises its input 512 times.
//
// `ORNITH_FUSE_NORM_PROJ=0` issues the norm and the matvec separately.
template <typename T, short nr0, short NSG>
[[kernel]] void q4k_rmsgate_qmv(
    const device uint16_t* qs [[buffer(0)]],
    const device uint16_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& O [[buffer(6)]],
    const device T* nw [[buffer(7)]],
    const device T* gate [[buffer(8)]],
    const constant int& head_dim [[buffer(9)]],
    const constant float& eps [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  threadgroup T row[ORNITH_RMSGATE_MAX];
  ornith_rmsgate_row<T>(x, nw, gate, row, K, head_dim, eps, tiisg, sgitg, NSG);
  q4k_qmv_impl<T, nr0, NSG>(
      qs,
      scales,
      dm,
      (threadgroup const T*)row,
      out,
      K,
      O,
      int(tgpig.x),
      tiisg,
      sgitg);
}

// The same mat-vec with the token axis moved **inside**.
//
// `q4k_qmv` carries M on the grid, so an M-row call reads the weight matrix M
// times; at a speculative verify window that re-read is 41% of what an extra
// window row costs (`REBUILD.md` Phase 27). Here one threadgroup takes `MW`
// rows of `x` at once: the block loop reads a block's weights once into
// registers and feeds all `MW` of them.
//
// **Every row still accumulates exactly what it would alone.** The `ib` loop
// order, the four-block lane split, the per-block expression and the final
// `simd_sum` are the ones above, copied; only the surrounding loops move, and
// `sumf[m]` sees the same blocks in the same order as `sumf` does in a one-row
// call. That is what makes this a batch-invariant kernel rather than a tile --
// `ops::invariant`'s window mode can keep it on.
//
// The weights hoisted per block are small (eight codes, four packed fields and
// two halves a row), so the registers go to the `MW` activation loads instead;
// activations are the cheap side, they stay in cache, and the ratio of loads
// to FMAs is the one-row kernel's.
template <typename T, short nr0, short NSG, short MW>
[[kernel]] void q4k_qmv_m(
    const device uint16_t* qs [[buffer(0)]],
    const device uint16_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& O [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;

  const short ix = tiisg / 8; // 0...3, which block of the four in flight
  const short it = tiisg % 8; // 0...7, where in the block
  const short iq = it / 4; // 0 or 1, which half of the super-block
  const short ir = it % 4; // 0...3

  const int nb = K / QK_K;
  const int first_row = (int(tgpig.x) * NSG + sgitg) * nr0;
  const int r1 = int(tgpig.y) * MW;

  const int qs_row = nb * Q4K_QS_U16;
  const int sc_row = nb * Q4K_SC_U16;
  const int dm_row = nb * Q4K_DM_F16;

  // A tail threadgroup clamps its spare rows onto the last real one. They
  // compute an answer nobody stores, which keeps a bound test out of the
  // inner loops; only the last group of a window does it, and only when M is
  // not a multiple of MW.
  const device T* y4[MW];
  for (short m = 0; m < MW; ++m) {
    y4[m] = x + (int64_t)min(r1 + m, M - 1) * K + ix * QK_K + 64 * iq + 8 * ir;
  }

  float sumf[MW][nr0];
  for (short m = 0; m < MW; ++m) {
    for (short row = 0; row < nr0; ++row) {
      sumf[m][row] = 0.f;
    }
  }

  for (int ib = ix; ib < nb; ib += 4) {
    // This block's weights, once, for every row of `x` below.
    const device uint16_t* sc = scales + first_row * sc_row + ib * Q4K_SC_U16 + iq;
    const device uint16_t* q1 = qs + first_row * qs_row + ib * Q4K_QS_U16 + 16 * iq + 4 * ir;
    const device half* dh = dm + first_row * dm_row + ib * Q4K_DM_F16;

    uint16_t wl[nr0][4];
    uint16_t wh[nr0][4];
    uint8_t sc8[nr0][8];
    float2 dv[nr0];

    for (short row = 0; row < nr0; ++row) {
      // `sc16` as the one-row kernel builds it, unpacked into its bytes here
      // rather than read through a pointer alias -- the same eight values.
      const uint16_t s0 = sc[0] & kmask1;
      const uint16_t s1 = sc[2] & kmask1;
      const uint16_t s2 = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
      const uint16_t s3 = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
      sc8[row][0] = uint8_t(s0 & 0xff);
      sc8[row][1] = uint8_t(s0 >> 8);
      sc8[row][2] = uint8_t(s1 & 0xff);
      sc8[row][3] = uint8_t(s1 >> 8);
      sc8[row][4] = uint8_t(s2 & 0xff);
      sc8[row][5] = uint8_t(s2 >> 8);
      sc8[row][6] = uint8_t(s3 & 0xff);
      sc8[row][7] = uint8_t(s3 >> 8);

      for (short i = 0; i < 4; ++i) {
        wl[row][i] = q1[i];
        wh[row][i] = q1[32 + i];
      }
      dv[row] = float2(float(dh[0]), float(dh[1]));

      q1 += qs_row;
      sc += sc_row;
      dh += dm_row;
    }

    for (short m = 0; m < MW; ++m) {
      const device T* y4m = y4[m];

      float yl[16];
      float yh[16];
      float4 sumy = {0.f, 0.f, 0.f, 0.f};

      for (short i = 0; i < 8; ++i) {
        yl[i + 0] = float(y4m[i + 0]);
        sumy[0] += yl[i + 0];
        yl[i + 8] = float(y4m[i + 32]);
        sumy[1] += yl[i + 8];
        yh[i + 0] = float(y4m[i + 128]);
        sumy[2] += yh[i + 0];
        yh[i + 8] = float(y4m[i + 160]);
        sumy[3] += yh[i + 8];
      }

      for (short row = 0; row < nr0; ++row) {
        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 4; ++i) {
          acc1[0] += yl[2 * i + 0] * (wl[row][i] & 0x000F);
          acc1[1] += yl[2 * i + 1] * (wl[row][i] & 0x0F00);
          acc1[2] += yl[2 * i + 8] * (wl[row][i] & 0x00F0);
          acc1[3] += yl[2 * i + 9] * (wl[row][i] & 0xF000);
          acc2[0] += yh[2 * i + 0] * (wh[row][i] & 0x000F);
          acc2[1] += yh[2 * i + 1] * (wh[row][i] & 0x0F00);
          acc2[2] += yh[2 * i + 8] * (wh[row][i] & 0x00F0);
          acc2[3] += yh[2 * i + 9] * (wh[row][i] & 0xF000);
        }

        sumf[m][row] +=
            dv[row][0] *
                ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[row][0] +
                 (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[row][1] * 1.f / 16.f +
                 (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[row][4] +
                 (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[row][5] * 1.f / 16.f) -
            dv[row][1] *
                (sumy[0] * sc8[row][2] + sumy[1] * sc8[row][3] +
                 sumy[2] * sc8[row][6] + sumy[3] * sc8[row][7]);
      }

      y4[m] += 4 * QK_K;
    }
  }

  for (short m = 0; m < MW; ++m) {
    if (r1 + m >= M) {
      break;
    }
    for (short row = 0; row < nr0 && first_row + row < O; ++row) {
      const float sum_all = simd_sum(sumf[m][row]);
      if (tiisg == 0) {
        out[(int64_t)(r1 + m) * O + first_row + row] = T(sum_all);
      }
    }
  }
}

// The multi-row mat-vec again, with the **unpack** hoisted as well as the
// read -- which is the half the measurement says is binding.
//
// `q4k_qmv_m` above shares one read of a weight block between MW rows and
// measures ~10% against the one-row kernel, because at MW rows the kernel is
// not reading-bound any more: on `lm_head` a single row sustains 520 GB/s,
// the machine's ceiling, and four rows through `q4k_qmv_m` sustain 192 -- the
// weights arrive long before the arithmetic that consumes them is done.
// What repeats per row is the dequantize: a mask and an integer-to-float
// conversion for every code, against the one fused multiply-add that uses it.
//
// So this one unpacks a block's codes into registers once and spends them MW
// times. The loop order is the other way round from `q4k_qmv_m` -- output row
// outside, row of `x` inside -- because 32 unpacked weights and 32
// activations do not both fit for two output rows at once; the activations
// are re-read per output row instead, which is the cheap side (they are a few
// kilobytes and they stay in cache).
//
// Every product is the product the one-row kernel forms, in the same order
// into the same accumulator: `float(q & mask)` is the same float the implicit
// conversion made. Bit-exactness is `tests/q4k.rs`, not this comment.
template <typename T, short nr0, short NSG, short MW>
[[kernel]] void q4k_qmv_mh(
    const device uint16_t* qs [[buffer(0)]],
    const device uint16_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& O [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;

  const short ix = tiisg / 8;
  const short it = tiisg % 8;
  const short iq = it / 4;
  const short ir = it % 4;

  const int nb = K / QK_K;
  const int first_row = (int(tgpig.x) * NSG + sgitg) * nr0;
  const int r1 = int(tgpig.y) * MW;

  const int qs_row = nb * Q4K_QS_U16;
  const int sc_row = nb * Q4K_SC_U16;
  const int dm_row = nb * Q4K_DM_F16;

  // A tail group clamps its spare rows onto the last real one and does not
  // store them.
  int64_t ybase[MW];
  for (short m = 0; m < MW; ++m) {
    ybase[m] = (int64_t)min(r1 + m, M - 1) * K + 64 * iq + 8 * ir;
  }

  float sumf[MW][nr0];
  for (short m = 0; m < MW; ++m) {
    for (short row = 0; row < nr0; ++row) {
      sumf[m][row] = 0.f;
    }
  }

  for (int ib = ix; ib < nb; ib += 4) {
    const device uint16_t* sc = scales + first_row * sc_row + ib * Q4K_SC_U16 + iq;
    const device uint16_t* q1 = qs + first_row * qs_row + ib * Q4K_QS_U16 + 16 * iq + 4 * ir;
    const device half* dh = dm + first_row * dm_row + ib * Q4K_DM_F16;

    for (short row = 0; row < nr0; ++row) {
      const uint16_t s0 = sc[0] & kmask1;
      const uint16_t s1 = sc[2] & kmask1;
      const uint16_t s2 = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
      const uint16_t s3 = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
      const float sc0 = float(s0 & 0xff);
      const float sc1 = float(s0 >> 8);
      const float sc2 = float(s1 & 0xff);
      const float sc3 = float(s1 >> 8);
      const float sc4 = float(s2 & 0xff);
      const float sc5 = float(s2 >> 8);
      const float sc6 = float(s3 & 0xff);
      const float sc7 = float(s3 >> 8);
      const float d0 = float(dh[0]);
      const float d1 = float(dh[1]);

      // The block's codes, once.
      float wq1[16];
      float wq2[16];
      for (short i = 0; i < 4; ++i) {
        const uint16_t a = q1[i];
        const uint16_t b = q1[32 + i];
        wq1[4 * i + 0] = float(a & 0x000F);
        wq1[4 * i + 1] = float(a & 0x0F00);
        wq1[4 * i + 2] = float(a & 0x00F0);
        wq1[4 * i + 3] = float(a & 0xF000);
        wq2[4 * i + 0] = float(b & 0x000F);
        wq2[4 * i + 1] = float(b & 0x0F00);
        wq2[4 * i + 2] = float(b & 0x00F0);
        wq2[4 * i + 3] = float(b & 0xF000);
      }

      for (short m = 0; m < MW; ++m) {
        const device T* y4m = x + ybase[m] + ib * QK_K;

        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
          yl[i + 0] = float(y4m[i + 0]);
          sumy[0] += yl[i + 0];
          yl[i + 8] = float(y4m[i + 32]);
          sumy[1] += yl[i + 8];
          yh[i + 0] = float(y4m[i + 128]);
          sumy[2] += yh[i + 0];
          yh[i + 8] = float(y4m[i + 160]);
          sumy[3] += yh[i + 8];
        }

        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 4; ++i) {
          acc1[0] += yl[2 * i + 0] * wq1[4 * i + 0];
          acc1[1] += yl[2 * i + 1] * wq1[4 * i + 1];
          acc1[2] += yl[2 * i + 8] * wq1[4 * i + 2];
          acc1[3] += yl[2 * i + 9] * wq1[4 * i + 3];
          acc2[0] += yh[2 * i + 0] * wq2[4 * i + 0];
          acc2[1] += yh[2 * i + 1] * wq2[4 * i + 1];
          acc2[2] += yh[2 * i + 8] * wq2[4 * i + 2];
          acc2[3] += yh[2 * i + 9] * wq2[4 * i + 3];
        }

        sumf[m][row] +=
            d0 *
                ((acc1[0] + 1.f / 256.f * acc1[1]) * sc0 +
                 (acc1[2] + 1.f / 256.f * acc1[3]) * sc1 * 1.f / 16.f +
                 (acc2[0] + 1.f / 256.f * acc2[1]) * sc4 +
                 (acc2[2] + 1.f / 256.f * acc2[3]) * sc5 * 1.f / 16.f) -
            d1 * (sumy[0] * sc2 + sumy[1] * sc3 + sumy[2] * sc6 + sumy[3] * sc7);
      }

      q1 += qs_row;
      sc += sc_row;
      dh += dm_row;
    }
  }

  for (short m = 0; m < MW; ++m) {
    if (r1 + m >= M) {
      break;
    }
    for (short row = 0; row < nr0 && first_row + row < O; ++row) {
      const float sum_all = simd_sum(sumf[m][row]);
      if (tiisg == 0) {
        out[(int64_t)(r1 + m) * O + first_row + row] = T(sum_all);
      }
    }
  }
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

// The format, as `kquant_mm.h`'s tile wants it: a dequantizer over 16
// consecutive weights, plus the three per-super-block strides.
//
// ggml's `dequantize_q4_K` in `dequantize.h` takes one `block_q4_K *`; this
// takes the three pointers the artifact stores and is otherwise the same
// arithmetic, including the `d / 16` that turns the high nibble's `q & 0xF0`
// back into `q >> 4` exactly. The one change is the type it divides in: ggml
// halves `xb->d` in `half`, where `d` near 6e-5 would denormalise, and this
// halves it in float. That direction is the safe one -- it can only be more
// accurate than the reference it is checked against.
//
// `il` is the il-th run of 16 weights in the 256-weight super-block, which is
// what ggml's `il` means once its `(il/4)*32 + 16*(il&1)` addressing is
// unpicked: chunk `il` is weights `[16*il, 16*il + 16)`.
struct Q4KFmt {
  static constant constexpr const short QK = QK_K;
  static constant constexpr const short NL = 16; // 16-weight chunks a block
  static constant constexpr const int CODES = 128; // bytes of `qs`
  static constant constexpr const int SCALES = 12; // bytes of packed fields
  static constant constexpr const int DM = 2; // halves: d and dmin

  // `get_scale_min_k4_just2`, which reads the pair for one sub-block rather
  // than `get_scale_min_k4`'s two. Same twelve bytes, same packing.
  static inline uchar2 scale_min(int j, int k, const device uint8_t* q) {
    return j < 4
        ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
        : uchar2{
              uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)),
              uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
  }

  static inline void dequantize(
      const device uint8_t* qs,
      const device uint8_t* scales,
      const device half* dm,
      short il,
      thread float4x4& reg) {
    const device uint8_t* q = qs + (il / 4) * 32 + 16 * (il & 1);
    const short is = (il / 4) * 2;
    il = il & 3;
    const uchar2 sc = scale_min(is, il / 2, scales);
    const float d = il < 2 ? float(dm[0]) : float(dm[0]) / 16.f;
    const float dl = d * sc[0];
    const float ml = float(dm[1]) * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (short i = 0; i < 16; ++i) {
      reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
  }
};

// The stand-in that prices the unpack, and nothing in the model dispatches it.
//
// Phase 37 read the quantized gemm at 79% of this machine's matrix ceiling
// and mlx's dense bf16 gemm at 96%, and called the 20% between them "the
// price of the format". That is a difference between **two kernels**, which
// differ in their tiling, their loaders and their threadgroup shape as well
// as in whether they unpack anything -- the same over-attribution
// `REBUILD.md`'s eighth axis is about. This is the control that separates
// them: the same tile, the same loads, the same registers, the same
// everything, with only `Q4KFmt`'s arithmetic taken out.
//
// What it drops, per 16 weights: `scale_min`'s bit-shuffling, the `d / 16`,
// two multiplies for `dl` and `ml`, and per weight the mask, the multiply
// and the subtract. What it keeps: the byte load and the conversion, so the
// memory traffic and the register pressure are unchanged.
//
// **The model produces nonsense while this runs**, exactly like
// `ORNITH_FAKE_SORT`. It exists because a removal cannot price a dispatch
// this one is inside of, and because the difference between two kernels
// cannot price one term of it.
struct Q4KBareFmt {
  static constant constexpr const short QK = QK_K;
  static constant constexpr const short NL = 16;
  static constant constexpr const int CODES = 128;
  static constant constexpr const int SCALES = 12;
  static constant constexpr const int DM = 2;

  static inline void dequantize(
      const device uint8_t* qs,
      const device uint8_t* scales,
      const device half* dm,
      short il,
      thread float4x4& reg) {
    const device uint8_t* q = qs + (il / 4) * 32 + 16 * (il & 1);
    for (short i = 0; i < 16; ++i) {
      reg[i / 4][i % 4] = float(q[i]);
    }
  }
};

// The tile scalar is `float`: see `kquant_mm.h` for why it is not ggml's
// `half`.
#define instantiate_q4k(tname, type)                                           \
  instantiate_kernel("q4k_dequantize_" #tname, q4k_dequantize, type)           \
  instantiate_kernel(                                                          \
      "q4k_qmm_t_bare_r64_" #tname, kquant_qmm_t, type, Q4KBareFmt, float, 64) \
  instantiate_kernel(                                                          \
      "q4k_qmm_t_bare_r32_" #tname, kquant_qmm_t, type, Q4KBareFmt, float, 32) \
  instantiate_kernel(                                                          \
      "q4k_steel_qmm_t_" #tname, kquant_steel_qmm_t, type, Q4KFmt, float, 32, 32, 32) \
  instantiate_kernel("q4k_qmv_" #tname, q4k_qmv, type, 2, 2)                   \
  instantiate_kernel("q4k_qmv_w4x2_" #tname, q4k_qmv, type, 4, 2) \
  instantiate_kernel("q4k_qmv_w2x4_" #tname, q4k_qmv, type, 2, 4) \
  instantiate_kernel("q4k_qmv_w4x4_" #tname, q4k_qmv, type, 4, 4) \
  instantiate_kernel("q4k_qmv_w8x4_" #tname, q4k_qmv, type, 8, 4) \
  instantiate_kernel("q4k_rmsgate_qmv_" #tname, q4k_rmsgate_qmv, type, 2, 2)   \
  instantiate_kernel("q4k_rmsgate_qmv_r4_" #tname, q4k_rmsgate_qmv, type, 4, 4) \
  instantiate_kernel("q4k_rmsgate_qmv_r8_" #tname, q4k_rmsgate_qmv, type, 8, 4) \
  instantiate_kernel("q4k_rmsgate_qmv_r16_" #tname, q4k_rmsgate_qmv, type, 16, 4)\
  instantiate_kernel("q4k_qmv_m2_" #tname, q4k_qmv_m, type, 2, 2, 2)           \
  instantiate_kernel("q4k_qmv_m3_" #tname, q4k_qmv_m, type, 2, 2, 3)           \
  instantiate_kernel("q4k_qmv_m4_" #tname, q4k_qmv_m, type, 2, 2, 4)           \
  instantiate_kernel("q4k_qmv_mh2_" #tname, q4k_qmv_mh, type, 2, 2, 2)         \
  instantiate_kernel("q4k_qmv_mh3_" #tname, q4k_qmv_mh, type, 2, 2, 3)         \
  instantiate_kernel("q4k_qmv_mh4_" #tname, q4k_qmv_mh, type, 2, 2, 4)         \
  instantiate_kernel("q4k_gather_qmv_" #tname, q4k_gather_qmv, type, 2, 2)     \
  instantiate_kernel("q4k_gather_qmv_w4x2_" #tname, q4k_gather_qmv, type, 4, 2) \
  instantiate_kernel("q4k_gather_qmv_w2x4_" #tname, q4k_gather_qmv, type, 2, 4) \
  instantiate_kernel("q4k_gather_qmv_w4x4_" #tname, q4k_gather_qmv, type, 4, 4) \
  instantiate_kernel("q4k_gather_qmv_w8x4_" #tname, q4k_gather_qmv, type, 8, 4) \
  instantiate_kernel(                                                          \
      "q4k_qmm_t_r32_" #tname, kquant_qmm_t, type, Q4KFmt, float, 32)         \
  instantiate_kernel(                                                          \
      "q4k_qmm_t_r32h_" #tname, kquant_qmm_t, type, Q4KFmt, half, 32)         \
  instantiate_kernel(                                                          \
      "q4k_gather_qmm_t_r32h_" #tname, kquant_gather_qmm_t, type, Q4KFmt,     \
      half, 32)                                                                \
  instantiate_kernel(                                                          \
      "q4k_qmm_t_r16_" #tname, kquant_qmm_t, type, Q4KFmt, float, 16)         \
  instantiate_kernel(                                                          \
      "q4k_qmm_t_r64_" #tname, kquant_qmm_t, type, Q4KFmt, float, 64)         \
  instantiate_kernel(                                                          \
      "q4k_gather_qmm_t_r64_" #tname, kquant_gather_qmm_t, type, Q4KFmt,      \
      float, 64)                                                               \
  instantiate_kernel(                                                          \
      "q4k_gather_qmm_t_r32_" #tname, kquant_gather_qmm_t, type, Q4KFmt,      \
      float, 32)                                                               \
  instantiate_kernel(                                                          \
      "q4k_gather_qmm_t_r16_" #tname, kquant_gather_qmm_t, type, Q4KFmt,      \
      float, 16)

instantiate_q4k(bfloat16, bfloat16_t) instantiate_q4k(float32, float)

// The control arm of the spike, at bfloat16 tiles: only for bfloat16, because
// at `type = float` it is the same instantiation as the arm above it.
instantiate_kernel(
    "q4k_steel_qmm_t_h_bfloat16", kquant_steel_qmm_t, bfloat16_t, Q4KFmt, bfloat16_t, 32, 32, 32)
