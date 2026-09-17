"""Exhaustive activation references over the bfloat16 number line.

The sigmoid-based activations turned out to disagree between mlx's
paths -- `mx.sigmoid` (a plain unary kernel) and the `mx.compile`d
`nn.silu` / `swiglu` give different last bits for some inputs. Rather than
model that, this dumps every finite bfloat16 value in a useful range and lets
the Rust tests check the kernels against all of them at once.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import ref
from mlx_lm.models.activations import precise_swiglu, swiglu


def main():
    # Every bfloat16 bit pattern whose magnitude is in [2^-20, 64]: that
    # covers everything a normalised activation can be, and the extremes
    # saturate anyway.
    bits = np.arange(0, 1 << 16, dtype=np.uint32)
    f32 = (bits << 16).view(np.float32)
    keep = np.isfinite(f32) & (np.abs(f32) <= 64.0) & (
        (np.abs(f32) >= 2.0**-20) | (f32 == 0)
    )
    vals = f32[keep]
    x = mx.array(vals).astype(mx.bfloat16)
    ones = mx.ones(x.shape, dtype=mx.bfloat16)
    # A second operand with a different exponent, so the multiply is not a
    # no-op and rounding in the product is exercised too.
    other = mx.array(np.where(vals == 0, 1.0, 1.0 / np.abs(np.where(vals == 0, 1.0, vals)))).astype(mx.bfloat16)

    case = ref.Case("activations")
    case.put("x", x)
    case.put("other", other)
    case.put("sigmoid", mx.sigmoid(x))          # plain unary kernel
    case.put("silu", nn.silu(x))                # mx.compile'd
    case.put("x_mul_sigmoid", x * mx.sigmoid(x))  # not compiled
    case.put("swiglu", swiglu(x, other))        # mx.compile'd
    case.put("precise_swiglu", precise_swiglu(other, x, other))
    case.put("sigmoid_mul", mx.sigmoid(x) * other)
    case.note(n=int(x.size))
    case.write()


if __name__ == "__main__":
    main()
