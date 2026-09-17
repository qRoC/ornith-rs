// Instantiations of mlx's binary kernels, restricted to what this port
// dispatches.
//
// mlx's own binary.metal instantiates every op over every type over every
// contiguity variant, which is thousands of kernels; the bodies here are the
// vendored ones (kernels/mlx/.../binary.h) and only the one dispatch the
// generation loop makes is instantiated:
//
//   logprobs = logits - mx.logsumexp(logits, keepdims=True)
//
// `logits` is [1, vocab] and the logsumexp is [1, 1], so `get_binary_op_type`
// says VectorScalar, and `get_work_per_thread(bfloat16, 248320)` is 4 because
// the output is past the 1 << 16 threshold -- so the name picks up an `n`:
// `vsn_Subtractbfloat16`. The plain `vs_` form (work_per_thread 1) is
// instantiated too, for the short rows the tests use.

// clang-format off
#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/binary.h"

#define instantiate_binary_vs(op, tname, itype, otype)                   \
  instantiate_kernel("vs_" #op #tname, binary_vs, itype, otype, op, 1)   \
  instantiate_kernel("vsn_" #op #tname, binary_vs, itype, otype, op)

instantiate_binary_vs(Subtract, bfloat16, bfloat16_t, bfloat16_t)
instantiate_binary_vs(Subtract, float32, float, float) // clang-format on
