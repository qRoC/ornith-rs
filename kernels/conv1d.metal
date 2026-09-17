// Depthwise 1-D convolution.
//
// Verbatim from vendor/mlx/mlx/backend/metal/kernels/conv.metal. mlx routes
// here from `conv_1D_gpu` when groups == in_channels == out_channels, stride
// and dilation are 1, padding is 0 and the kernel is not flipped -- which is
// exactly GatedDeltaNet's causal conv (`nn.Conv1d(conv_dim, conv_dim,
// kernel_size=4, groups=conv_dim, padding=0)`); the left padding comes from
// the cached conv state, concatenated ahead of the sequence.
//
// Copied rather than pulling in all of conv.metal, which would drag the steel
// conv and Winograd kernels in with it.

#include "mlx/backend/metal/kernels/utils.h"

template <typename T, typename IdxT = int64_t>
[[kernel]] void depthwise_conv_1d(
    const device T* in [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const IdxT strides[3],
    constant const int& kernel_size,
    uint3 tid [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  out += (tid.z * static_cast<IdxT>(grid_dim.y) + tid.y) * grid_dim.x + tid.x;
  in += tid.z * strides[0] + tid.y * strides[1] + tid.x * strides[2];
  w += tid.x * kernel_size;

  float acc = 0.0;
  for (int i = 0; i < kernel_size; ++i) {
    acc += static_cast<float>(in[0]) * w[i];
    in += strides[1];
  }
  *out = static_cast<T>(acc);
}

#define instantiate_depthconv1d(iname, itype)                         \
  instantiate_kernel(                                                 \
      "depthwise_conv_1d_" #iname, depthwise_conv_1d, itype, int32_t) \
      instantiate_kernel(                                             \
          "depthwise_conv_1d_" #iname "_large",                       \
          depthwise_conv_1d,                                          \
          itype,                                                      \
          int64_t)

instantiate_depthconv1d(float32, float);
instantiate_depthconv1d(bfloat16, bfloat16_t);
