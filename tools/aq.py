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
    return pack(q, 4, out_cols)


def pack(q, bits, out_cols):
    """`32 // bits` codes to a uint32, lowest-order field first.

    mlx packs 4-bit and 8-bit affine weights the same way, so one function
    serves both. The routed experts need it: this artifact holds layers 32-39
    at 8 bits / group 64 and the other 32 at 4 / 32, which is invisible until
    something asserts against the base -- `forge_gptq_experts.py --off` did.
    """
    per = 32 // bits
    q = q.reshape(-1, per).astype(np.uint32)
    word = np.zeros(len(q), np.uint32)
    for i in range(per):
        word |= q[:, i] << (bits * i)
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


SUPER = 8       # sub-blocks of `group_size` sharing one power-of-two super-scale
QK_BITS = 6     # magnitude field, plus a sign


def q4k_encode(scale, bias):
    """The k-quant shape: 6-bit integer scale and min against a per-super-block
    power of two. 4 + (7 + 7) / 32 + 16 / 256 = **4.50 bits**.

    llama.cpp's Q4_K stores a sub-block's scale and min as small **integers**
    normalised by two fp16 super-scales per 256 weights, which is why it has no
    subnormal regime -- the failure that cost nvfp4 its quality here, where
    99.6% of group scales fall below E4M3's smallest normal. The trade runs the
    other way from a float: an integer has uniform *absolute* resolution against
    the super-block maximum, so a sub-block far below that maximum is coarse in
    relative terms.

    Standable-in like `fp8_encode`: the super-scales are powers of two, so a
    6-bit integer times one is exactly representable in bfloat16 and the result
    writes out in the ordinary affine layout on mlx's own kernels. Unlike
    IDEAS #1 the sub-block stays 32 wide, which mlx supports -- that is the
    whole reason this one can be built and that one cannot
    (`REBUILD.md` Phase 13).

    `aq.quantize`'s scale carries a sign (it records which edge the group was
    fitted to), so both fields are magnitude-plus-sign rather than unsigned.
    """
    top = np.float32(2**QK_BITS - 1)

    def enc(v):
        n = len(v) // SUPER
        assert len(v) % SUPER == 0, (len(v), SUPER)
        g = np.abs(v[:n * SUPER]).reshape(n, SUPER)
        # A power of two >= max/top: exact in bf16, and nothing clips. Using
        # floor here lets a super-block's own largest field overflow in up to
        # half of all super-blocks, which costs 19% error against 8%.
        k = 2.0 ** np.ceil(np.log2(np.maximum(g.max(1, keepdims=True), 1e-30) / top))
        q = np.clip(np.round(g / k), 0, top) * k
        return (np.sign(v) * q.ravel()).astype(np.float32)

    s = enc(scale)
    # A group whose scale underflows to zero is unreconstructable; floor it at
    # one super-scale step with the sign kept, as `fp8_encode` does.
    step = 2.0 ** np.ceil(np.log2(np.maximum(
        np.abs(scale).reshape(-1, SUPER).max(1, keepdims=True), 1e-30) / top))
    floor = np.repeat(step.ravel(), SUPER).astype(np.float32)
    s = np.where(s == 0, np.where(scale < 0, -floor, floor), s).astype(np.float32)
    return s, enc(bias)
