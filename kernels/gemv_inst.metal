// Instantiations of mlx's gemv kernels.
//
// Kernel bodies are mlx's (kernels/mlx/.../gemv.h); this is the trimmed
// equivalent of vendor/mlx/mlx/backend/metal/kernels/gemv.metal, restricted to
// bfloat16 and float32.
//
// `Matmul::eval_gpu` routes to gemv whenever `min(M, N) == 1`, which for this
// model means every unquantized linear during decode: GatedDeltaNet's
// `in_proj_a` / `in_proj_b` (both [32, 2048] bfloat16), and every MTP and
// vision projection. mlx picks the tile from the vector lengths, so the
// full block list is kept rather than guessed at.
//
// The gemv_t (transposed-matrix), gather and wide variants are left out: this
// port never hits them -- gather_mm is only used by unquantized MoE, and
// `gemv_wide` needs 2..5 input vectors with a non-transposed A and transposed
// B, a shape no layer here produces.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/gemv.h"

using namespace metal;

#define instantiate_gemv_helper(                                      \
    name, itype, bm, bn, sm, sn, tm, tn, nc, axpby)                   \
  instantiate_kernel(                                                 \
      "gemv_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm \
      "_tn" #tn "_nc" #nc "_axpby" #axpby,                            \
      gemv, itype, bm, bn, sm, sn, tm, tn, nc, axpby)

#define instantiate_gemv(name, itype, bm, bn, sm, sn, tm, tn)        \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 0) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 0)

#define instantiate_gemv_blocks(name, itype) \
  instantiate_gemv(name, itype, 1,  8, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 1,  8, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 1,  1, 8,  4, 4, 4) \
  instantiate_gemv(name, itype, 1,  1, 8,  4, 1, 4) \
  instantiate_gemv(name, itype, 4,  1, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 4,  1, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 8,  1, 1, 32, 4, 4)

#define instantiate_gemv_t_helper(                                      \
    name, itype, bm, bn, sm, sn, tm, tn, nc, axpby)                     \
  instantiate_kernel(                                                   \
      "gemv_t_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm \
      "_tn" #tn "_nc" #nc "_axpby" #axpby,                              \
      gemv_t, itype, bm, bn, sm, sn, tm, tn, nc, axpby)

#define instantiate_gemv_t(name, itype, bm, bn, sm, sn, tm, tn)        \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 0) \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 0)

#define instantiate_gemv_t_blocks(name, itype)  \
  instantiate_gemv_t(name, itype, 1,  1, 8, 4, 4, 1) \
  instantiate_gemv_t(name, itype, 1,  2, 8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1,  4, 8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 4, 8, 4, 4)

instantiate_gemv_blocks(bfloat16, bfloat16_t);
instantiate_gemv_blocks(float32, float);
instantiate_gemv_t_blocks(bfloat16, bfloat16_t);
instantiate_gemv_t_blocks(float32, float);
