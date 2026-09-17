// Measurement kernels. Nothing in the model dispatches these; they exist so
// `bin/bandwidth` can ask what this machine actually sustains, as opposed to
// what its datasheet says.

#include "mlx/backend/metal/kernels/utils.h"

// A read and nothing else: grid-stride over the buffer in 16-byte chunks, one
// xor per chunk so the loads cannot be optimised away. The xor is a single ALU
// op against 16 bytes of traffic, so what this measures is the memory system.
[[kernel]] void stream_read(
    const device uint4* x [[buffer(0)]],
    device uint* out [[buffer(1)]],
    const constant uint& n4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]]) {
  uint4 acc = uint4(0);
  for (uint i = gid; i < n4; i += gsize) {
    acc ^= x[i];
  }
  out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
