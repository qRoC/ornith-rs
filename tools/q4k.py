"""ggml's Q4_K, in numpy, ported from the source rather than from the description.

`Q4K.md` step 1. Phase 14 measured a Q4_K-*shaped* grid at the same median KL
as the 5.00-bit base for 7.3% fewer bytes, using a stand-in that lived inside
the affine layout. This is the real format, which is what a kernel has to read.

Ported from `vendor/ggml/src/ggml-quants.c` and `ggml-common.h`. Two things in
there are not what a reimplementation would guess, and both are checked by
`tests/q4k_oracle` rather than trusted:

**1. The 12-byte scale packing is not a straight 6-bit stream.** Eight scales
and eight mins are folded into 12 bytes such that the first four of each sit in
their own byte and the last four borrow their high bits from the first four's
top two bits (`get_scale_min_k4`). Reproduced verbatim below.

**2. The nibble order is not "adjacent elements share a byte".** Within each
64-element half, elements 0..31 are the **low** nibbles of 32 bytes and
elements 32..63 are the **high** nibbles of *those same* bytes. This project's
own `aq.pack4` does the opposite -- eight consecutive elements in one uint32 --
so the two packings are not interchangeable and a `q4k.pack` that reuses it is
silently wrong.

**And the reconstruction subtracts:** `w = d*sc * q - dmin*m`, with `sc` and
`m` unsigned. Affine adds a signed bias. A dequantizer that keeps affine's sign
convention produces fluent text and wrong logits.

    QK_K = 256, K_SCALE_SIZE = 12
    block = d(f16) + dmin(f16) + scales[12] + qs[128] = 144 bytes / 256 weights
          = 4.50 bits a weight
"""

import numpy as np

QK_K = 256
K_SCALE_SIZE = 12
BLOCK_BYTES = 2 + 2 + K_SCALE_SIZE + QK_K // 2      # 144


def f16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def get_scale_min_k4(j, q):
    """`ggml-quants.c:880`, verbatim. `q` is [..., 12] uint8; returns (d, m)."""
    if j < 4:
        return q[..., j] & 63, q[..., j + 4] & 63
    return ((q[..., j + 4] & 0xF) | ((q[..., j - 4] >> 6) << 4),
            (q[..., j + 4] >> 4) | ((q[..., j - 0] >> 6) << 4))


def put_scale_min_k4(sc, mn):
    """The inverse of `get_scale_min_k4`: 8 scales + 8 mins -> 12 bytes.

    ggml writes this inline in `quantize_row_q4_K_impl`; it is stated here as
    one function so the round trip can be asserted against `get_scale_min_k4`.
    """
    sc = np.asarray(sc, np.uint8)
    mn = np.asarray(mn, np.uint8)
    out = np.zeros(sc.shape[:-1] + (K_SCALE_SIZE,), np.uint8)
    for j in range(4):
        out[..., j] = sc[..., j] & 63
        out[..., j + 4] = mn[..., j] & 63
    for j in range(4, 8):
        out[..., j + 4] = (sc[..., j] & 0xF) | ((mn[..., j] & 0xF) << 4)
        out[..., j - 4] |= (sc[..., j] >> 4) << 6
        out[..., j - 0] |= (mn[..., j] >> 4) << 6
    return out


def split(blocks, rows, nb):
    """ggml's 144-byte blocks -> the three tensors an artifact stores.

    `Q4K.md`'s storage decision, in one place because it is a byte layout and
    two copies of a byte layout drift. The bytes are not reinterpreted, only
    sliced: `dm` is the leading `d, dmin` pair, `scales` the twelve packed
    6-bit fields, `qs` the 128 bytes of codes -- so the kernel reading them as
    uint16 sees exactly what ggml's reads out of its packed block.
    """
    b = np.asarray(blocks, np.uint8).reshape(-1, BLOCK_BYTES)
    assert len(b) == rows * nb, (len(b), rows, nb)
    dm = b[:, 0:4].copy().reshape(rows, nb * 4).view(np.float16).reshape(rows, nb, 2)
    scales = b[:, 4:4 + K_SCALE_SIZE].copy().reshape(rows, nb, K_SCALE_SIZE)
    qs = b[:, 4 + K_SCALE_SIZE:].copy().reshape(rows, nb * (QK_K // 2)).view(np.uint32)
    return qs, scales, dm


def dequantize(blocks):
    """`dequantize_row_q4_K`. `blocks` is [n, 144] uint8 -> [n, 256] float32."""
    b = np.asarray(blocks, np.uint8).reshape(-1, BLOCK_BYTES)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).ravel()
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32).ravel()
    sc12 = b[:, 4:4 + K_SCALE_SIZE]
    qs = b[:, 4 + K_SCALE_SIZE:]
    out = np.empty((len(b), QK_K), np.float32)
    is_ = 0
    for j in range(0, QK_K, 64):
        s1, m1 = get_scale_min_k4(is_ + 0, sc12)
        s2, m2 = get_scale_min_k4(is_ + 1, sc12)
        q = qs[:, (j // 64) * 32:(j // 64) * 32 + 32].astype(np.float32)
        d1, o1 = d * s1.astype(np.float32), dmin * m1.astype(np.float32)
        d2, o2 = d * s2.astype(np.float32), dmin * m2.astype(np.float32)
        lo = np.floor_divide(qs[:, (j // 64) * 32:(j // 64) * 32 + 32], 1) & 0xF
        hi = qs[:, (j // 64) * 32:(j // 64) * 32 + 32] >> 4
        out[:, j:j + 32] = d1[:, None] * lo - o1[:, None]
        out[:, j + 32:j + 64] = d2[:, None] * hi - o2[:, None]
        del q
        is_ += 2
    return out


def quantize(w, sub=32):
    """Round-to-nearest Q4_K for a [.., n] array with n divisible by 256.

    This is `quantize_row_q4_K_ref`'s shape -- per sub-block min/max, no search
    and no importance weighting. ggml's `quantize_row_q4_K_impl` does a small
    search and accepts `quant_weights` (that is where an imatrix enters); both
    are deliberately out of scope until the plain path is checked against the
    oracle.
    """
    x = np.asarray(w, np.float32).reshape(-1, QK_K)
    g = x.reshape(-1, QK_K // sub, sub)
    lo, hi = g.min(-1), g.max(-1)
    lo = np.minimum(lo, 0.0)                        # ggml clamps the min at 0
    scale = (hi - lo) / 15.0
    mn = -lo                                        # stored as a positive min
    d = f16(scale.max(-1, keepdims=True) / 63.0)
    dm = f16(mn.max(-1, keepdims=True) / 63.0)
    sc = np.where(d > 0, np.clip(np.rint(scale / np.where(d > 0, d, 1)), 0, 63), 0)
    m = np.where(dm > 0, np.clip(np.rint(mn / np.where(dm > 0, dm, 1)), 0, 63), 0)
    ds, ms = d * sc, dm * m                         # the values the kernel sees
    q = np.clip(np.rint((g + ms[..., None]) / np.where(ds > 0, ds, 1)[..., None]),
                0, 15).astype(np.uint8)
    q = np.where((ds > 0)[..., None], q, 0).astype(np.uint8)

    n = len(x)
    out = np.zeros((n, BLOCK_BYTES), np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(n, 2)
    out[:, 2:4] = dm.astype(np.float16).view(np.uint8).reshape(n, 2)
    out[:, 4:4 + K_SCALE_SIZE] = put_scale_min_k4(sc.astype(np.uint8),
                                                  m.astype(np.uint8))
    # The nibble order: element i and element i+32 of each 64-wide half share a
    # byte, low nibble first. Not `aq.pack4`'s layout -- see the module docstring.
    qs = np.empty((n, QK_K // 2), np.uint8)
    flat = q.reshape(n, QK_K)
    for h in range(QK_K // 64):
        a = flat[:, h * 64:h * 64 + 32]
        b = flat[:, h * 64 + 32:h * 64 + 64]
        qs[:, h * 32:(h + 1) * 32] = (a & 0xF) | (b << 4)
    out[:, 4 + K_SCALE_SIZE:] = qs
    return out


def selfcheck(seed=0, n=64):
    """The packing round trip, before any oracle is available."""
    rng = np.random.default_rng(seed)
    sc = rng.integers(0, 64, (n, 8)).astype(np.uint8)
    mn = rng.integers(0, 64, (n, 8)).astype(np.uint8)
    p = put_scale_min_k4(sc, mn)
    for j in range(8):
        d, m = get_scale_min_k4(j, p)
        assert np.array_equal(d, sc[:, j]), f"scale field {j} round trip"
        assert np.array_equal(m, mn[:, j]), f"min field {j} round trip"
    w = (rng.standard_t(4, (n, QK_K)) * 0.015).astype(np.float32)
    b = quantize(w)
    assert b.shape == (n, BLOCK_BYTES)
    r = dequantize(b)
    err = np.linalg.norm(w - r) / np.linalg.norm(w)
    print(f"packing round trip ok; relative error on t(4) draws {100 * err:.2f}%")
    return err


if __name__ == "__main__":
    selfcheck()
