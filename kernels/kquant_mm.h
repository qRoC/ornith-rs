// ggml's tiled gemm for the two k-quants -- the prefill shape.
//
// Ported from `vendor/ggml/src/ggml-metal/kernels/mul_mm.metal`, the legacy
// `simdgroup_matrix` branch of `kernel_mul_mm` (the `#else` of
// `GGML_METAL_HAS_TENSOR`), which is the one this machine runs:
// `is_nax_available()` is false here, so none of mlx's or ggml's Metal 4
// `mpp::tensor_ops` paths are reachable and none of them are ported.
//
// **What it is for.** `q4k.metal` and `q6k.metal` carry ggml's mat-vec, and at
// M > 1 they walk the token axis -- a mat-vec per token, so the weight matrix
// is re-read once per token instead of being tiled. That is the right shape
// for decoding, where M = 1, and the wrong one for prefill, where M is a
// 2048-token chunk. The debt was measured rather than assumed (`HANDOFF.md`,
// "The debt is prefill"): level with affine at 600 prompt tokens, where L2
// absorbs a 9 MB matrix read 599 times, and -8% / -21% at 3060.
//
// Three departures from ggml's kernel, all forced by this port's contracts:
//
//   the layout   ggml reads one packed block (144 bytes for Q4_K, 210 for
//                Q6_K); the artifact keeps three tensors per module, so the
//                format is a traits struct -- `Q::dequantize` takes the three
//                per-block pointers where ggml's `dequantize_func` takes one
//                `block_q *`. The arithmetic inside is ggml's own.
//
//   the tile type  ggml dequantizes into `half` and multiplies half by half.
//                This port's contract for the gemm is that it agrees with the
//                mat-vec it replaces to within the difference in summation
//                order, and `half` is not that: rounding a dequantized weight
//                to eleven significand bits is a 5e-4 perturbation, three
//                orders above reassociation. So the tile scalar is a template
//                parameter and it is instantiated at `float`. The weights
//                then reach the multiplier exactly as the mat-vec has them
//                and the only difference left is the association.
//
//   the store    ggml writes the accumulator straight to device memory when
//                the tile is full, because its `dst` is float32. Here `dst` is
//                the activation type, so every tile stages through threadgroup
//                memory and converts on the way out. That also gives the
//                routed variant its row slicing for free.
//
// The tile itself is ggml's: 64 output features x NR1 rows x 32 of K, four
// simdgroups, A stored k-major as 8x8 fragments so `simdgroup_load` reads it
// with stride 8.
//
// **NR1 is a template parameter, and it is the one shape ggml's choice does
// not fit.** ggml's 32 is for a gemm whose rows are there; the row counts this
// model produces are a prefill chunk (thousands, where 32 is right) and a
// routed run of ~64 rows an expert (where a 32-row tile straddles an expert
// boundary about half the time, and every straddle is a second pass over K).
// `src/bin/kquant_mm.rs` measures the other half of it: the tile's time is
// *flat* from M = 1 to 32, because it does a 32-row tile's work whether the
// rows exist or not.
//
// The shape is coupled, which is why this is a template and not a #define.
// Four simdgroups split the tile two ways, 2 across the 64 features and 2
// across the NR1 rows, so each owns 32 features by NR1/2 rows -- `mc` is
// 4 feature fragments by NR1/16 row fragments, `lsmb`'s stride is NR1/8
// fragments a K-step, and the output staging offset carries NR1/2. Change one
// and the others have to move with it; `tests/q4k.rs`'s one-hot case is what
// catches it if they do not.

#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#include "mlx/backend/metal/kernels/utils.h"

#define KQ_NR0 64 // output features per threadgroup
#define KQ_NK 32 // K per step
#define KQ_NL0 (KQ_NK / 16) // threads sharing one A row: 2
#define KQ_NL1 (KQ_NK / 8) // threads sharing one B row: 4
#define KQ_NSG 4 // simdgroups per threadgroup; 128 threads

// One 64x32 output tile, over the whole of K.
//
// The caller has already offset the three weight pointers to its expert (or
// not, for a plain weight) and decided which rows of the tile are live:
// `[row_lo, row_hi)`. Rows outside that range are still *loaded* -- a clamped
// read of an in-bounds row -- and simply not stored, which is what lets the
// routed variant re-run the tile per run of equal expert without a second
// addressing scheme.
//
// `GATHER` picks where a tile row's activations come from: row `r` of the
// output reads row `r` of `x` for a plain matmul and row `lhs[r]` for the
// gather, which is `mx.gather_qmm`'s contract and the one `gather_qmv`
// already implements at M = 1.
template <typename T, typename Q, typename S, bool GATHER, short NR1>
static inline void kquant_mm_tile(
    const device uint8_t* qs,
    const device uint8_t* scales,
    const device half* dm,
    const device T* x,
    device T* out,
    const device uint32_t* lhs,
    threadgroup char* shmem,
    int K,
    int O,
    int M,
    int r0,
    int r1,
    short row_lo,
    short row_hi,
    ushort tiitg,
    ushort tiisg,
    ushort sgitg) {
  // 8-row B fragments: NR1/8 in the tile, half of them per simdgroup.
  constexpr short NSY = NR1 / 8;
  constexpr short NRB = NSY / 2;
  constexpr short NMC = 4 * NRB;
  // 128 threads carry 32 B rows at 8 of K each, so a wider tile carries more
  // than one row per thread.
  constexpr short BROWS = (NR1 + 31) / 32;
  static_assert(NR1 == 16 || NR1 == 32 || NR1 == 64,
                "the sgitg split assumes two row halves and NSY fragments a K-slice");

  threadgroup S* sa = (threadgroup S*)shmem;
  threadgroup S* sb = (threadgroup S*)(shmem + sizeof(S) * KQ_NR0 * KQ_NK);

  const int nb = K / Q::QK; // super-blocks a row

  // If this block is of 64x32 shape or smaller.
  const short nr0 = (O - r0 < KQ_NR0) ? short(O - r0) : KQ_NR0;
  const short nr1 = (M - r1 < NR1) ? short(M - r1) : NR1;

  // A thread shouldn't load data outside of the matrix. Both clamps are on the
  // *source* row only: the destination slot in threadgroup memory stays this
  // thread's own, so a short tile holds duplicates of its last row rather than
  // a hole, exactly as ggml leaves it.
  const short lr0 = ((short)tiitg / KQ_NL0) < nr0 ? ((short)tiitg / KQ_NL0) : nr0 - 1;

  const short il0 = (tiitg % KQ_NL0);
  short il = il0;

  // The weight row this thread dequantizes, as three pointers into three
  // tensors where ggml has one into one.
  const int arow = r0 + lr0;
  const device uint8_t* aq = qs + (int64_t)arow * nb * Q::CODES;
  const device uint8_t* as = scales + (int64_t)arow * nb * Q::SCALES;
  const device half* ad = dm + (int64_t)arow * nb * Q::DM;

  // The activation rows, and the eight of K this thread carries within each.
  const short iy = 8 * (tiitg % KQ_NL1);
  short lr1[BROWS];
  const device T* yv[BROWS];
  for (short b = 0; b < BROWS; b++) {
    const short want = (short)(tiitg / KQ_NL1) + 32 * b;
    lr1[b] = want < nr1 ? want : nr1 - 1;
    int brow = r1 + lr1[b];
    if constexpr (GATHER) {
      brow = int(lhs[brow]);
    }
    yv[b] = x + (int64_t)brow * K + iy;
  }

  simdgroup_matrix<S, 8, 8> ma[4];
  simdgroup_matrix<S, 8, 8> mb[NRB];
  simdgroup_float8x8 mc[NMC];

#pragma clang loop unroll(full)
  for (short i = 0; i < NMC; i++) {
    mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
  }

  for (int loop_k = 0; loop_k < K; loop_k += KQ_NK) {
    float4x4 temp_a;
    Q::dequantize(aq, as, ad, il, temp_a);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A is stored as 8x8 fragments, k-major inside a fragment, so that
    // `simdgroup_load(.., 8, 0, false)` reads `ma[k][row]` -- the transpose
    // the outer product wants.
    for (short i = 0; i < 16; i++) {
      const short sx = 2 * il0 + i / 8;
      const short sy = (tiitg / KQ_NL0) / 8;
      const short lx = (tiitg / KQ_NL0) % 8;
      const short ly = i % 8;
      const short ib = 8 * sx + sy;
      sa[64 * ib + 8 * ly + lx] = S(temp_a[i / 4][i % 4]);
    }

    // With NR1 = 16 only half the threads have a row to carry -- 16 rows x 4
    // K-slices is 64 of the 128, and the A tile still wants all of them. With
    // NR1 = 64 each thread carries two.
    for (short b = 0; b < BROWS; b++) {
      const short row = (short)(tiitg / KQ_NL1) + 32 * b;
      if (row < NR1) {
        const short sx = (tiitg % KQ_NL1);
        const short sy = row / 8;
        const short ly = row % 8;
        const short ib = NSY * sx + sy;
        for (short i = 0; i < 8; i++) {
          sb[64 * ib + 8 * ly + i] = S(yv[b][i]);
        }
      }
    }

    il = (il + 2 < Q::NL) ? il + 2 : il % 2;
    if (il < 2) {
      aq += Q::CODES;
      as += Q::SCALES;
      ad += Q::DM;
    }

    for (short b = 0; b < BROWS; b++) {
      yv[b] += KQ_NK;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load matrices from threadgroup memory and conduct outer products.
    threadgroup const S* lsma = (sa + 4 * 64 * (sgitg % 2));
    threadgroup const S* lsmb = (sb + NRB * 64 * (sgitg / 2));

    for (short ik = 0; ik < KQ_NK / 8; ik++) {
      simdgroup_barrier(mem_flags::mem_none);

      for (short i = 0; i < 4; i++) {
        simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
      }

      simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
      for (short i = 0; i < NRB; i++) {
        simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
      }

      simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
      for (short i = 0; i < NMC; i++) {
        simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
      }

      lsma += 8 * 64;
      lsmb += NSY * 64;
    }
  }

  // Stage the accumulators and convert on the way out. ggml stores the full
  // tile straight to device memory when it fits, because its `dst` is float32;
  // here it is the activation type, so the staging is unconditional -- and it
  // is what gives `[row_lo, row_hi)` its slicing.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* temp_str =
      ((threadgroup float*)shmem) + 32 * (sgitg & 1) + ((NR1 / 2) * (sgitg >> 1)) * KQ_NR0;

#pragma clang loop unroll(full)
  for (short i = 0; i < NMC; i++) {
    simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * KQ_NR0 * (i / 4), KQ_NR0, 0, false);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short j = row_lo + sgitg; j < row_hi; j += KQ_NSG) {
    device T* d = out + (int64_t)(r1 + j) * O + r0;
    threadgroup const float* c = ((threadgroup const float*)shmem) + j * KQ_NR0;
    for (short i = tiisg; i < nr0; i += 32) {
      d[i] = T(c[i]);
    }
  }
}

// `x @ w.T` with `x` `[M, K]` and `w` `[O, K]`: ggml's `kernel_mul_mm`.
template <typename T, typename Q, typename S, short NR1>
[[kernel]] void kquant_qmm_t(
    const device uint8_t* qs [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& O [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  const int r1 = int(tgpig.x) * NR1;
  const int r0 = int(tgpig.y) * KQ_NR0;
  const short nr1 = (M - r1 < NR1) ? short(M - r1) : NR1;

  kquant_mm_tile<T, Q, S, false, NR1>(
      qs, scales, dm, x, out, nullptr, shmem, K, O, M, r0, r1, 0, nr1, tiitg, tiisg, sgitg);
}

// `mx.gather_qmm(x, w, lhs_indices, rhs_indices, transpose=True)` at M > 1 --
// ggml's `kernel_mul_mm_id`, with mlx's answer to the ragged part.
//
// ggml solves "which rows belong to which expert" with a separate pass
// (`kernel_mul_mm_id_map0`) that builds a per-expert row list, and then
// dispatches a grid over experts. mlx's `gather_qmm_rhs` solves it inside the
// tile: walk the tile's `rhs_indices`, find each maximal run of one expert,
// and run the whole K loop once per run, storing only that run's rows. This
// takes mlx's, for three reasons -- it needs no second kernel and no scratch
// buffers, it keeps the port's own grid (`out` row `r` is tile row `r`), and
// it is *correct for any ordering*, merely slow when the rows are not sorted.
//
// `switch_glu_sorted` sorts once the batch reaches 64 rows, which is every
// prefill chunk, so a 32-row tile straddles a boundary about half the time at
// this model's ~64 routed rows per expert, and the cost of a straddle is one
// extra pass over K.
template <typename T, typename Q, typename S, short NR1>
[[kernel]] void kquant_gather_qmm_t(
    const device uint8_t* qs [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device half* dm [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& O [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const device uint32_t* lhs [[buffer(8)]],
    const device uint32_t* rhs [[buffer(9)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  const int r1 = int(tgpig.x) * NR1;
  const int r0 = int(tgpig.y) * KQ_NR0;
  const short nr1 = (M - r1 < NR1) ? short(M - r1) : NR1;

  const int64_t nb = K / Q::QK;
  const int64_t estride = (int64_t)O * nb;

  short lo = 0;
  while (lo < nr1) {
    const uint32_t e = rhs[r1 + lo];
    short hi = lo + 1;
    while (hi < nr1 && rhs[r1 + hi] == e) {
      hi++;
    }

    kquant_mm_tile<T, Q, S, true, NR1>(
        qs + (int64_t)e * estride * Q::CODES,
        scales + (int64_t)e * estride * Q::SCALES,
        dm + (int64_t)e * estride * Q::DM,
        x,
        out,
        lhs,
        shmem,
        K,
        O,
        M,
        r0,
        r1,
        lo,
        hi,
        tiitg,
        tiisg,
        sgitg);

    lo = hi;
  }
}
