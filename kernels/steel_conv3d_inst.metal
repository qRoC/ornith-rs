// Instantiation of mlx's implicit-gemm 3D convolution, for the vision tower's
// patch embedding.
//
// The kernel body is mlx's
// (kernels/mlx/.../steel/conv/kernels/steel_conv_3d.h); this is the trimmed
// equivalent of
// vendor/mlx/mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_3d.metal.
//
// `PatchEmbed` is an `nn.Conv3d(3, 1152, kernel=(2, 16, 16), stride=(2, 16,
// 16))` over an `[N, 2, 16, 16, 3]` input, so `dispatch_conv_3D_gpu` sees
// `is_idil_one` with `C_per_group = 3`, which is not a multiple of 16. That is
// the `pad_and_slice_conv_3D_gpu` branch: the input and the weight are padded
// from 3 to 16 channels with zeros (`extra_o` is 0, because 1152 already
// divides 16) and `implicit_gemm_conv_3D_gpu` runs on the padded pair.
//
// Its block choice is then fixed:
//
//   implicit_M = N patches, implicit_N = 1152, implicit_K = 2*16*16*16 = 8192
//   bm = (implicit_M >= 8192 && C_per_group >= 64) ? 64 : 32   -> 32, always,
//        because the padded C_per_group is 16
//   bn = (bm == 64 || implicit_N >= 64) ? 64 : 32              -> 64
//   bk = 16, wm = wn = 2, and implicit_N > 16 so the {bn 8, wm 4, wn 1}
//        override does not apply
//   small_filter = all of wS <= 16, and wS is (2, 16, 16)      -> 's'

// clang-format off
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/conv/conv.h"
#include "mlx/backend/metal/kernels/steel/conv/params.h"
#include "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_3d.h"

#define instantiate_implicit_conv_3d(                     \
    name,                                                 \
    itype,                                                \
    bm,                                                   \
    bn,                                                   \
    bk,                                                   \
    wm,                                                   \
    wn,                                                   \
    fn,                                                   \
    f)                                                    \
  instantiate_kernel(                                     \
      "implicit_gemm_conv_3d_" #name "_bm" #bm "_bn" #bn  \
          "_bk" #bk "_wm" #wm "_wn" #wn "_filter_" #fn,   \
      implicit_gemm_conv_3d,                              \
      itype,                                              \
      bm,                                                 \
      bn,                                                 \
      bk,                                                 \
      wm,                                                 \
      wn,                                                 \
      f)

instantiate_implicit_conv_3d(bfloat16, bfloat16_t, 32, 64, 16, 2, 2, s, true)
// clang-format on
