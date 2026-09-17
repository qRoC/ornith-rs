// Instantiations of mlx's split-K steel GEMM.
//
// Kernel bodies are mlx's
// (kernels/mlx/.../steel/gemm/kernels/steel_gemm_splitk.h); this is the
// trimmed equivalent of that kernel's .metal file.
//
// `steel_matmul` takes this path whenever the output is small and K is long:
//
//     batch_size_out == 1 && _tm * _tn <= 2048 && _tk >= 8 && K >= max(M, N)
//
// which is exactly GatedDeltaNet's `in_proj_a` / `in_proj_b` during prefill
// (M = tokens, N = 32 value heads, K = 2048). The partial products accumulate
// in float32 and a second kernel sums them down to bfloat16 -- getting this
// wrong costs an ULP per output, which is how it was found.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h"

#define instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned) \
  instantiate_kernel(                                                                                                                  \
      "steel_gemm_splitk_" #tname "_" #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_MN_" #aname "_K_" #kname, \
      gemm_splitk, itype, otype, bm, bn, bk, wm, wn, trans_a, trans_b, mn_aligned, k_aligned)

#define instantiate_gemm_aligned_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, taligned, true)    \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, naligned, false)   \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, taligned, true)   \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, naligned, false)

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn)             \
    instantiate_gemm_aligned_helper(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gemm_shapes_helper(iname, itype, oname, otype)                  \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 16, 16, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 16, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 16, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2)

instantiate_gemm_shapes_helper(bfloat16, bfloat16_t, float32, float);
instantiate_gemm_shapes_helper(float32, float, float32, float);

// The `_axbpy` epilogue is the addmm form: `steel_gemm_splitk_axpby` appends
// it whenever `alpha != 1 || beta != 0`, and `nn.Linear` with a bias calls
// `mx.addmm(bias, x, w.T)`, whose defaults are alpha = beta = 1. Every biased
// linear in the vision tower whose output tile is small enough for split-K
// lands here.
#define instantiate_accum(oname, otype, aname, atype)      \
  instantiate_kernel(                                      \
    "steel_gemm_splitk_accum_" #oname "_" #aname,          \
    gemm_splitk_accum, atype, otype)                       \
  instantiate_kernel(                                      \
    "steel_gemm_splitk_accum_" #oname "_" #aname "_axbpy", \
    gemm_splitk_accum_axpby, atype, otype)

instantiate_accum(bfloat16, bfloat16_t, float32, float);
instantiate_accum(float32, float, float32, float); // clang-format on
