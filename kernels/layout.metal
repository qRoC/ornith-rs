// Layout moves the layers need: slicing along the last axis and the
// head/sequence transpose.
//
// mlx does these with `mx.split`, `mx.reshape` and `mx.transpose`, which
// produce strided views its kernels then consume (or materialise via
// `ensure_row_contiguous`). This port keeps every array row-contiguous, so the
// views become explicit copies. They move data without arithmetic, so they are
// exact; what they cost is bandwidth, which is why the layers below skip them
// whenever a reshape suffices (a [B, 1, H, D] to [B, H, 1, D] transpose is a
// no-op in memory).

#include "mlx/backend/metal/kernels/utils.h"

// `src[..., start:start + dst_row]` over rows of `src_row` elements.
template <typename T>
[[kernel]] void slice_inner(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const constant uint& src_row [[buffer(2)]],
    const constant uint& dst_row [[buffer(3)]],
    const constant uint& start [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  uint r = i / dst_row;
  uint c = i % dst_row;
  dst[i] = src[r * src_row + start + c];
}

// The same slice, sixteen bytes a thread.
//
// A pure copy, so bit-exact by construction. The caller takes this only
// when `start`, `dst_row` and `src_row` are all whole numbers of vectors
// and both bases are 16-byte aligned -- see `layout::vec16`. The narrow
// form stays for every other shape, and `ORNITH_WIDE_ELEM=0` is the
// control for both.
template <typename T>
struct alignas(16) Slice16 {
  uint8_t b[16];
};

template <typename T>
[[kernel]] void slice_inner_wide(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const constant uint& src_row [[buffer(2)]],
    const constant uint& dst_row [[buffer(3)]],
    const constant uint& start [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  constexpr uint L = 16 / sizeof(T);
  const uint r = i / (dst_row / L);
  const uint c = i - r * (dst_row / L);
  const device Slice16<T>* s16 =
      (const device Slice16<T>*)(src + r * src_row + start);
  device Slice16<T>* d16 = (device Slice16<T>*)(dst + (size_t)r * dst_row);
  d16[c] = s16[c];
}

// [B, L, H, D] -> [B, H, L, D]
template <typename T>
[[kernel]] void transpose_lh(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const constant uint& l [[buffer(2)]],
    const constant uint& h [[buffer(3)]],
    const constant uint& d [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  uint dd = i % d;
  uint ll = (i / d) % l;
  uint hh = (i / (d * l)) % h;
  uint b = i / (d * l * h);
  // i indexes the destination [B, H, L, D]
  dst[i] = src[((b * l + ll) * h + hh) * d + dd];
}

// [B, H, L, D] -> [B, L, H, D]
template <typename T>
[[kernel]] void transpose_hl(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const constant uint& l [[buffer(2)]],
    const constant uint& h [[buffer(3)]],
    const constant uint& d [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  uint dd = i % d;
  uint hh = (i / d) % h;
  uint ll = (i / (d * h)) % l;
  uint b = i / (d * h * l);
  // i indexes the destination [B, L, H, D]
  dst[i] = src[((b * h + hh) * l + ll) * d + dd];
}

// `mx.concatenate([a, b], axis=-1)` over rows of `a_row` and `b_row`
// elements. The MTP head's `fc` consumes concat[embedding, hidden], and one
// matmul over the joined row is not the same arithmetic as two matmuls over
// the halves summed -- so the join is materialised rather than factored.
template <typename T>
[[kernel]] void concat_inner(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    device T* dst [[buffer(2)]],
    const constant uint& a_row [[buffer(3)]],
    const constant uint& b_row [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  uint row = a_row + b_row;
  uint r = i / row;
  uint c = i % row;
  dst[i] = (c < a_row) ? a[r * a_row + c] : b[r * b_row + (c - a_row)];
}

#define instantiate_layout(tname, type)                          \
  instantiate_kernel("slice_inner_" #tname, slice_inner, type)                \
  instantiate_kernel("slice_inner_wide_" #tname, slice_inner_wide, type)   \
  instantiate_kernel("concat_inner_" #tname, concat_inner, type) \
  instantiate_kernel("transpose_lh_" #tname, transpose_lh, type) \
  instantiate_kernel("transpose_hl_" #tname, transpose_hl, type)

instantiate_layout(bfloat16, bfloat16_t)
instantiate_layout(float32, float)
