"""mlx's affine_quantize, in numpy, with the (scale, bias) encoding swappable."""
import numpy as np

def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)

def e4m3(x):
    """OCP E4M3 magnitude+sign: 3 mantissa bits, normals 2**-6..448, subnormals to 2**-9."""
    a = np.abs(np.asarray(x, np.float32))
    e = np.clip(np.floor(np.log2(np.maximum(a, 1e-30))), -6, 8)
    normal = np.round(a / 2.0**e * 8) / 8 * 2.0**e
    sub = np.round(a / 2.0**-9) * 2.0**-9
    return (np.sign(x) * np.clip(np.where(a < 2.0**-6, sub, normal), 0, 448.0)).astype(np.float32)

def po2_shift(v):
    """A power-of-two per tensor that lands |v| near E4M3's top normal (256).

    Power of two, not an arbitrary fp32, so that e4m3(v/2**k)*2**k keeps four
    significant bits and stays exactly representable in bfloat16 -- which is
    what lets the twin artifact be written in the ordinary affine layout.
    """
    m = np.abs(v).max()
    if m == 0:
        return 1.0
    return float(2.0 ** (np.floor(np.log2(m)) - 8))

def rnd(x):
    """Metal's `round`: half away from zero, where numpy's rounds to even."""
    x = np.asarray(x, np.float32)
    return (np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))).astype(np.float32)


def quantize(w, group_size=32, bits=4, encode=None):
    """Port of `affine_quantize` (quantized.h). `encode` replaces bf16 storage.

    mlx computes the codes against the *unrounded* fp32 scale/bias and only
    then casts them to bf16 for storage, so the dequantizer uses slightly
    different numbers than the quantizer did. With an explicit encoding the
    codes are computed against the values that will actually be stored, which
    is what the kernel would read -- consistent, and very slightly better.
    """
    # float32 throughout: the kernel is float32 and a stray float64 here
    # moves 1.3% of the codes.
    f = np.float32
    n_bins = f((1 << bits) - 1)
    g = w.reshape(-1, group_size).astype(np.float32)
    w_min = g.min(1)
    w_max = np.maximum(g.max(1), f(0))         # the kernel inits w_max to 0
    scale = np.maximum((w_max - w_min) / n_bins, f(1e-7)).astype(np.float32)
    side = np.abs(w_min) > np.abs(w_max)
    scale = np.where(side, scale, -scale).astype(np.float32)
    edge = np.where(side, w_min, w_max).astype(np.float32)
    q0 = rnd(edge / scale)
    at_zero = q0 == f(0)
    scale = np.where(at_zero, scale, edge / np.where(at_zero, f(1), q0)).astype(np.float32)
    bias = np.where(at_zero, f(0), edge).astype(np.float32)

    if encode is None:                          # mlx: codes off fp32, store bf16
        s_use, b_use = scale, bias
        s_st, b_st = bf16(scale), bf16(bias)
        lo = None                               # mlx clamps above only
    else:
        s_st, b_st = encode(scale, bias)
        s_use, b_use = s_st, b_st
        lo = 0                                  # a perturbed scale can underflow
    q = rnd((g - b_use[:, None]) / s_use[:, None])
    q = np.minimum(q, n_bins)
    if lo is not None:
        q = np.maximum(q, 0)
    return q.astype(np.uint8), s_st.astype(np.float32), b_st.astype(np.float32)

def pack4(q, out_cols):
    """8 four-bit codes to a uint32, low nibble first."""
    q = q.reshape(-1, 8).astype(np.uint32)
    word = np.zeros(len(q), np.uint32)
    for i in range(8):
        word |= q[:, i] << (4 * i)
    return word.reshape(-1, out_cols)

# Smallest non-zero E4M3 magnitude: the subnormal step.
E4M3_MIN = 2.0**-9


def fp8_encode(scale, bias):
    """The candidate 4.50-bit format: E4M3 scale and bias, one power-of-two
    exponent each a tensor, so the pair costs 8 bits a group instead of 32.

    A scale can underflow the encoding where the group is nearly constant --
    mlx floors the *unencoded* scale at 1e-7, which says nothing about what
    E4M3 can hold once the tensor's exponent has been taken out. Underflow to
    zero would divide by zero in the quantizer and make the group
    unreconstructable, so it is floored at the smallest subnormal with the
    sign kept. mlx's own eps clamp is the same guard one step earlier.
    """
    ks, kb = po2_shift(scale), po2_shift(bias)
    s = e4m3(scale / ks) * ks
    floor = np.float32(E4M3_MIN * ks)
    s = np.where(s == 0, np.where(scale < 0, -floor, floor), s).astype(np.float32)
    return s, (e4m3(bias / kb) * kb).astype(np.float32)
