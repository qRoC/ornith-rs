// Instantiations of mlx's steel GEMM, restricted to what this port dispatches.
//
// The kernel is mlx's (kernels/mlx/.../steel/gemm/kernels/steel_gemm_fused.h);
// this is the trimmed equivalent of
// vendor/mlx/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal.
//
// Only the 64x64x16 / wm=2 / wn=2 tile is built: `GEMM_TPARAM_MACRO` in
// matmul.cpp selects tiles by device class, and an `applegpu_g16s` (M4 Max)
// is a "medium device", for which that macro sets bm=64, bn=64, bk=16, wm=2,
// wn=2 unconditionally. A different Mac class would need the other tiles.
//
// Unquantized matmul is needed in three places, all bfloat16:
//   * attention prefill -- at head_dim 256 mlx's `use_fallback` sends
//     sequences longer than 8 down the unfused softmax(QK^T)V path;
//   * the MTP head, whose sidecar weights are native bfloat16;
//   * the vision tower, likewise bfloat16 throughout.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h"

#define instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                                             \
      "steel_gemm_fused_" #tname "_"  #iname "_" #oname                                           \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,                                          \
  gemm, itype, bm, bn, bk, wm, wn, trans_a, trans_b, float)

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

instantiate_gemm_transpose_helper(bfloat16, bfloat16_t, bfloat16, bfloat16_t, 64, 64, 16, 2, 2)
instantiate_gemm_transpose_helper(float32, float, float32, float, 64, 64, 16, 2, 2)
// clang-format on
