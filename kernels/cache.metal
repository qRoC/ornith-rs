// Cache maintenance: zero-fill and the in-place sequence write.
//
// mlx expresses these as `mx.zeros`, `mx.concatenate` and the slice assignment
// `self.keys[..., prev:self.offset, :] = keys` (cache.py). All three are data
// movement, so a direct kernel is equivalent, and writing in place avoids
// re-materialising the whole cache every step.

#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void fill_zero(device T* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = T(0);
}

// Copy `[B, H, used, D]` from a buffer of capacity `src_cap` into one of
// capacity `dst_cap`, starting at `dst_offset` along the sequence axis.
template <typename T>
[[kernel]] void cache_copy(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const constant uint& src_cap [[buffer(2)]],
    const constant uint& dst_cap [[buffer(3)]],
    const constant uint& used [[buffer(4)]],
    const constant uint& dim [[buffer(5)]],
    const constant uint& dst_offset [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
  uint d = i % dim;
  uint t = (i / dim) % used;
  uint bh = i / (dim * used);
  dst[(bh * dst_cap + dst_offset + t) * dim + d] = src[(bh * src_cap + t) * dim + d];
}

#define instantiate_cache(tname, type)                         \
  instantiate_kernel("fill_zero_" #tname, fill_zero, type)     \
  instantiate_kernel("cache_copy_" #tname, cache_copy, type)

instantiate_cache(bfloat16, bfloat16_t)
instantiate_cache(float32, float)
instantiate_kernel("fill_zero_uint32", fill_zero, uint32_t)
