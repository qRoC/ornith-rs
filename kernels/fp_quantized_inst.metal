// Instantiations of mlx's floating-point quantized kernels, restricted to the
// one mode this port uses: nvfp4.
//
// The kernel bodies are mlx's, verbatim, in
// kernels/mlx/mlx/backend/metal/kernels/fp_quantized.h. This file only decides
// which template specialisations get compiled, mirroring
// vendor/mlx/mlx/backend/metal/kernels/fp_quantized.metal.
//
// mlx has exactly four quantization modes -- affine, mxfp4, nvfp4, mxfp8 --
// and only nvfp4 is instantiated here. REBUILD.md Phase 4 has the arithmetic:
// mxfp4 measures 20-22% weight error against affine int4's 8-9%, and mxfp8
// buys 0.8% of bytes for 4.6x the error of the affine int8 it would replace.
// Both are priced and rejected, so neither is compiled.
//
// nvfp4 layout, as produced by `mx.quantize(w, 16, 4, mode="nvfp4")`:
//
//   w:      [..., O, I / 8]   uint32, 8 E2M1 values per word, low nibble first
//   scales: [..., O, I / 16]  uint8, one E4M3 scale per group of 16
//
// and there is **no biases tensor** -- `nn.QuantizedLinear` leaves
// `self.biases = None`, so a module is two tensors rather than three. The
// optional global scale (mlx's `has_global_scale` / `_hgs` specialisations) is
// only ever produced by `mx.quantize(..., global_scale=...)` or
// `nn.quantize(quantize_input=True)`; `mlx_lm.convert` passes neither, so the
// `_hgs` variants are not instantiated either. src/ops/quantized.rs asserts
// that a loaded nvfp4 module has no third tensor, so an artifact that did
// carry global scales fails loudly instead of being read wrong.
//
// Activations are bfloat16 throughout; float is instantiated alongside so the
// golden tests can drive the same kernels in f32 and separate a kernel bug
// from a bf16 rounding difference.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

#define instantiate_quantized(mode, name, type, group_size, bits)     \
  instantiate_kernel(                                                 \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits,       \
      fp_ ## name,                                                    \
      type,                                                           \
      group_size,                                                     \
      bits)

#define instantiate_quantized_batched(mode, name, type, group_size, bits, batched) \
  instantiate_kernel(                                                               \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched,   \
      fp_ ## name,                                                                   \
      type,                                                                          \
      group_size,                                                                    \
      bits,                                                                          \
      batched)

#define instantiate_quantized_aligned(mode, name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                               \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,     \
      fp_ ## name,                                                                   \
      type,                                                                          \
      group_size,                                                                    \
      bits,                                                                          \
      aligned)

#define instantiate_quantized_aligned_batched(mode, name, type, group_size, bits, aligned, batched) \
  instantiate_kernel(                                                                                \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched,   \
      fp_ ## name,                                                                                   \
      type,                                                                                          \
      group_size,                                                                                    \
      bits,                                                                                          \
      aligned,                                                                                       \
      batched)

#define instantiate_quantized_wide(mode, name, type, group_size, bits, vecs_per_tg, k_lanes, batched) \
  instantiate_kernel(                                                                                  \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched, \
      fp_ ## name,                                                                                     \
      type,                                                                                            \
      group_size,                                                                                      \
      bits,                                                                                            \
      vecs_per_tg,                                                                                     \
      k_lanes,                                                                                         \
      batched)

#define instantiate_gather_qmm_rhs(mode, func, name, type, group_size, bits, bm, bn, bk, wm, wn, transpose) \
  instantiate_kernel(                                                                                        \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,                                                                                                  \
      type,                                                                                                  \
      group_size,                                                                                            \
      bits,                                                                                                  \
      bm,                                                                                                    \
      bn,                                                                                                    \
      bk,                                                                                                    \
      wm,                                                                                                    \
      wn,                                                                                                    \
      transpose)

#define instantiate_quantize_dequantize(mode, type, group_size, bits) \
  instantiate_kernel(                                                 \
      #mode "_dequantize_" #type "_gs_" #group_size "_b_" #bits "_hgs_false", \
      fp_dequantize,                                                  \
      type,                                                           \
      group_size,                                                     \
      bits,                                                           \
      false)

// --- what we dispatch -------------------------------------------------------
//
// The same seven branches the affine half takes, and for the same shapes; see
// kernels/quantized_inst.metal and src/ops/quantized.rs, which reimplements
// mlx's dispatch for both halves out of one function per kernel family.
//
// Decode (M == 1):            fp_qmv_fast / fp_qmv
// Small batch (2 <= M < lim): fp_qmv_wide            (MTP verify, batches)
// Prefill (M >= lim):         fp_qmm_t / fp_qmm_t_splitk
// MoE decode:                 fp_gather_qmv_fast / fp_gather_qmv
// MoE prefill:                fp_gather_qmm_rhs (sorted) / fp_gather_qmm_t
// Round trip (tests):         fp_dequantize
//
// Four families of mlx's are deliberately absent, each for a reason that also
// keeps its affine twin out of quantized_inst.metal: `qmv_quad` (no 64- or
// 128-wide input in this model), `qvm` and `qvm_split_k` (every linear here is
// transposed), and `qmm_n` / `gather_qmm_n` / `gather_qmm_rhs_nn` (likewise).
// Two more are absent for reasons peculiar to the fp half:
//
//   * `_r_2` -- the narrow-output fp_qmv_fast. mlx picks it only when
//     `architecture_gen() == 17 && architecture().back() == 's' && mode ==
//     "nvfp4"` (quantized.cpp's `use_narrow_qmv`). This machine is
//     applegpu_g16s, gen 16, so the branch is unreachable; src/ops/quantized.rs
//     asserts that rather than instantiating a kernel it cannot test.
//   * `_batch_1` -- `quantized_matmul` asserts 2-D weights, so B is always 1
//     and the batched specialisations are never named. (quantized_inst.metal
//     compiles its affine `_batch_1` twins; they are dead there too.)

#define instantiate_quantized_vector(mode, type, group_size, bits)                       \
  instantiate_quantized_batched(mode, qmv_fast, type, group_size, bits, 0)                \
  instantiate_quantized_batched(mode, qmv, type, group_size, bits, 0)                     \
  instantiate_quantized_wide(mode, qmv_wide, type, group_size, bits, 2, 16, 0)            \
  instantiate_quantized_wide(mode, qmv_wide, type, group_size, bits, 3, 16, 0)            \
  instantiate_quantized_wide(mode, qmv_wide, type, group_size, bits, 4, 16, 0)            \
  instantiate_quantized_wide(mode, qmv_wide, type, group_size, bits, 5, 16, 0)

#define instantiate_quantized_matrix(mode, type, group_size, bits)                        \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, group_size, bits, true, 0)      \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, group_size, bits, false, 0)     \
  instantiate_quantized_aligned(mode, qmm_t_splitk, type, group_size, bits, true)          \
  instantiate_quantized_aligned(mode, qmm_t_splitk, type, group_size, bits, false)

#define instantiate_quantized_gather(mode, type, group_size, bits)                         \
  instantiate_quantized(mode, gather_qmv_fast, type, group_size, bits)                      \
  instantiate_quantized(mode, gather_qmv, type, group_size, bits)                           \
  instantiate_quantized_aligned(mode, gather_qmm_t, type, group_size, bits, true)           \
  instantiate_quantized_aligned(mode, gather_qmm_t, type, group_size, bits, false)          \
  instantiate_gather_qmm_rhs(mode, fp_gather_qmm_rhs, gather_qmm_rhs_nt, type, group_size, bits, 16, 32, 32, 1, 2, true)

#define instantiate_quantized_funcs(mode, type, group_size, bits) \
  instantiate_quantize_dequantize(mode, type, group_size, bits)   \
  instantiate_quantized_vector(mode, type, group_size, bits)      \
  instantiate_quantized_matrix(mode, type, group_size, bits)      \
  instantiate_quantized_gather(mode, type, group_size, bits)

#define instantiate_quantized_types(mode, group_size, bits)    \
  instantiate_quantized_funcs(mode, bfloat16_t, group_size, bits) \
  instantiate_quantized_funcs(mode, float, group_size, bits)

instantiate_quantized_types(nvfp4, 16, 4) // clang-format on
