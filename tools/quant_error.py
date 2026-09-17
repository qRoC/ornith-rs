"""Relative RMS weight error of each quantization format, before spending a day on one.

`REBUILD.md` Phase 4 asks which 4-bit format can replace the 8-bit modules.
That is ultimately a quality question about this model, but the first half of
it is arithmetic about the formats themselves and needs no weights: draw from
a plausible weight distribution, quantize, measure. It is what ranked nvfp4
over mxfp4 and then ranked affine int4 over both.

Scales and biases are rounded to bfloat16, as the artifact stores them, so
affine is not flattered. nvfp4's scale is quantized to E4M3 and mxfp4's to a
power of two, as their specifications require.

    python3 tools/quant_error.py
"""

import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])  # OCP fp4 magnitudes
rng = np.random.default_rng(0)


def bf16(x):
    """Round to bfloat16, which is what a group scale and bias are stored as."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def to_grid(x, grid):
    i = np.abs(np.abs(x)[..., None] - grid).argmin(-1)
    return np.sign(x) * grid[i]


def e4m3(x):
    x = np.clip(np.abs(x), 2**-9, 448.0)
    e = np.floor(np.log2(x))
    return 2**e * np.round(x / 2**e * 8) / 8


def affine(w, g, bits):
    w = w.reshape(-1, g)
    lo, hi = w.min(1, keepdims=True), w.max(1, keepdims=True)
    n = 2**bits - 1
    s = bf16((hi - lo) / n)
    b = bf16(lo)
    q = np.clip(np.round((w - b) / np.where(s == 0, 1, s)), 0, n)
    return (q * s + b).ravel()


def fp4(w, g, po2_scale):
    w = w.reshape(-1, g)
    a = np.abs(w).max(1, keepdims=True) / 6
    a = np.where(a == 0, 1, a)
    s = 2 ** np.floor(np.log2(a)) if po2_scale else e4m3(a)
    return (to_grid(w / s, E2M1) * s).ravel()


def main():
    n = 1 << 20
    formats = [
        ("affine int8 g64", 8.50, lambda w: affine(w, 64, 8)),
        ("affine int4 g32", 5.00, lambda w: affine(w, 32, 4)),
        ("nvfp4 g16", 4.50, lambda w: fp4(w, 16, False)),
        ("mxfp4 g32", 4.25, lambda w: fp4(w, 32, True)),
    ]
    dists = [
        ("gaussian", rng.standard_normal(n)),
        ("laplace", rng.laplace(size=n)),
        ("student-t(4)", rng.standard_t(4, n)),
    ]
    print(f"{'format':<16}{'bits/w':>7}  " + "".join(f"{d:>14}" for d, _ in dists))
    for name, bits, f in formats:
        cells = []
        for _, w in dists:
            base = np.sqrt((w**2).mean())
            cells.append(f"{100 * np.sqrt(((f(w) - w) ** 2).mean()) / base:>13.2f}%")
        print(f"{name:<16}{bits:>7.2f}  " + "".join(cells))
    print("\nrelative RMS weight error, lower is better")


main()
