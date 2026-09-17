// Instantiations of mlx's gathering gemv kernels.
//
// Kernel bodies are mlx's (kernels/mlx/.../gemv.h); this is the trimmed
// equivalent of the gather half of
// vendor/mlx/mlx/backend/metal/kernels/gemv.metal.
//
// These are what an *unquantized* MoE dispatches to. The trunk's 256-expert
// blocks are quantized and go through `gather_qmv` / `gather_qmm_rhs`, but the
// MTP head's block is stored bf16 (`mtp_prequantized: false` in
// `mtplx_runtime.json`, and the sidecar's 785 tensors are all BF16), so
// `SwitchLinear.__call__` reaches `mx.gather_mm`, which for M == 1 routes to
// `gather_mv` and from there to these.
//
// Which block: `gather_mv` sees the expert weight as `[E, in, out]` -- a
// `swapaxes(-1, -2)` view of the `[E, out, in]` the checkpoint stores -- so
// `check_transpose` reports it transposed, `is_mv` is false, and the two flips
// cancel: the *non*-transposed `gemv_gather` runs with `marix_ld = K`. Then
//
//   bm = out_vector_len >= 4096 ? 8 : 4;   sn = 32;   sm = 1;   bn = 1;
//   tm = out_vector_len < 4 ? 1 : 4;       tn = 4;
//
// and this head's projections are 512 and 2048 wide, so bm is always 4. The
// 8 and tm=1 forms are instantiated too: they cost nothing and a different
// width would otherwise fail at pipeline lookup rather than in the dispatch.
//
// `gemv_t_gather` is left out. It would need `is_mv` true or a
// non-transposed weight, and neither happens for a SwitchLinear.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/gemv.h"

using namespace metal;

// clang-format off
#define instantiate_gemv_gather(name, itype, bm, bn, sm, sn, tm, tn)   \
  instantiate_kernel(                                                  \
      "gemv_gather_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn     \
      "_tm" #tm "_tn" #tn,                                             \
      gemv_gather, itype, bm, bn, sm, sn, tm, tn)

#define instantiate_gemv_gather_blocks(name, itype)      \
  instantiate_gemv_gather(name, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_gather(name, itype, 8, 1, 1, 32, 4, 4) \
  instantiate_gemv_gather(name, itype, 4, 1, 1, 32, 1, 4) \
  instantiate_gemv_gather(name, itype, 8, 1, 1, 32, 1, 4)

instantiate_gemv_gather_blocks(bfloat16, bfloat16_t);
instantiate_gemv_gather_blocks(float32, float); // clang-format on
