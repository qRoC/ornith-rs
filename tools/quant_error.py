"""Every quantization number in `IDEAS.md` and `REBUILD.md`, reproducible.

The format half of "which 4-bit scheme can replace the 8-bit modules" is
arithmetic about the formats and needs no weights: draw from a plausible
weight distribution, quantize, measure the relative RMS error. It is what
ranked nvfp4 over mxfp4, then affine int4 over both, then said the grid has
about one bit left in it and vector quantization will not collect it.

Scales and biases are rounded to bfloat16 where the artifact stores them that
way, so affine is never flattered. nvfp4's scale is quantized to E4M3 and
mxfp4's to a power of two, as their specifications require.

    python3 tools/quant_error.py            # the format and metadata tables
    python3 tools/quant_error.py --vq       # and the vector-quantization one (slow)

Numbers are stable to about 0.1 percentage points across seeds at this sample
size; nothing below turns on a tenth.
"""

import sys

import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])  # OCP fp4 magnitudes
N = 1 << 20
rng = np.random.default_rng(0)


def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def fp8(x):
    """E4M3 magnitude, which is how nvfp4 stores a group scale."""
    s = np.sign(x)
    a = np.clip(np.abs(x), 2**-9, 448.0)
    e = np.floor(np.log2(a))
    return s * 2**e * np.round(a / 2**e * 8) / 8


def err(q, w):
    return 100 * np.sqrt(((q - w) ** 2).mean()) / np.sqrt((w**2).mean())


def affine(w, g, bits, scale_q=bf16, zero_point=True):
    w = w.reshape(-1, g)
    n = 2**bits - 1
    if zero_point:
        lo, hi = w.min(1, keepdims=True), w.max(1, keepdims=True)
    else:
        hi = np.abs(w).max(1, keepdims=True)
        lo = -hi
    s = scale_q((hi - lo) / n)
    b = scale_q(lo)
    s = np.where(s == 0, 1, s)
    return (np.clip(np.round((w - b) / s), 0, n) * s + b).ravel()


def fp4(w, g, po2_scale):
    w = w.reshape(-1, g)
    a = np.abs(w).max(1, keepdims=True) / 6
    a = np.where(a == 0, 1, a)
    s = 2 ** np.floor(np.log2(a)) if po2_scale else fp8(a)
    i = np.abs(np.abs(w / s)[..., None] - E2M1).argmin(-1)
    return (np.sign(w) * E2M1[i] * s).ravel()


def lloyd(w, g, bits, scale_q, iters=25):
    """A codebook fitted to the weights, one scale per group -- a palette."""
    w = w.reshape(-1, g)
    s = scale_q(np.abs(w).max(1, keepdims=True))
    s = np.where(s == 0, 1, s)
    x = (w / s).ravel()
    c = np.quantile(x, np.linspace(0, 1, 2**bits))
    for _ in range(iters):
        i = np.abs(x[:, None] - c).argmin(1)
        for k in range(len(c)):
            m = i == k
            if m.any():
                c[k] = x[m].mean()
    i = np.abs(x[:, None] - c).argmin(1)
    return (c[i].reshape(-1, g) * s).ravel()


def kmeans(X, k, iters=8, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        idx = assign(X, C)
        cnt = np.bincount(idx, minlength=k)
        C2 = np.stack(
            [np.bincount(idx, weights=X[:, j], minlength=k) for j in range(X.shape[1])],
            axis=1,
        )
        nz = cnt > 0
        C[nz] = C2[nz] / cnt[nz, None]
    return C


def assign(X, C):
    idx = np.empty(len(X), np.int32)
    Cn = (C * C).sum(1)[None]
    for i in range(0, len(X), 8192):
        b = X[i : i + 8192]
        idx[i : i + 8192] = ((b * b).sum(1)[:, None] - 2 * b @ C.T + Cn).argmin(1)
    return idx


def table(title, rows, dists):
    print(f"\n{title}")
    print(f"  {'scheme':<44}{'bits/w':>7}" + "".join(f"{d:>14}" for d, _ in dists))
    for name, bits, f in rows:
        cells = "".join(f"{err(f(w), w):>13.2f}%" for _, w in dists)
        print(f"  {name:<44}{bits:>7.2f}{cells}")


def main():
    dists = [
        ("gaussian", rng.standard_normal(N).astype(np.float32)),
        ("laplace", rng.laplace(size=N).astype(np.float32)),
        ("student-t(4)", rng.standard_t(4, N).astype(np.float32)),
    ]

    table(
        "the formats on the table (REBUILD.md Phase 4)",
        [
            ("affine int8 g64  -- the 260 modules today", 8.50, lambda w: affine(w, 64, 8)),
            ("affine int4 g32  -- the body today", 5.00, lambda w: affine(w, 32, 4)),
            ("nvfp4 g16", 4.50, lambda w: fp4(w, 16, False)),
            ("mxfp4 g32", 4.25, lambda w: fp4(w, 32, True)),
        ],
        dists,
    )

    table(
        "where the group metadata goes (IDEAS.md) -- 4-bit elements throughout",
        [
            ("g32, bf16 scale+bias  (today)", 5.00, lambda w: affine(w, 32, 4)),
            ("g16, fp8 scale+bias", 5.00, lambda w: affine(w, 16, 4, fp8)),
            ("g64, bf16 scale+bias", 4.50, lambda w: affine(w, 64, 4)),
            ("g32, fp8 scale+bias", 4.50, lambda w: affine(w, 32, 4, fp8)),
            ("g32, bf16 scale, no zero-point", 4.50, lambda w: affine(w, 32, 4, bf16, False)),
            ("g16, fp8 scale, no zero-point", 4.50, lambda w: affine(w, 16, 4, fp8, False)),
            ("Lloyd-Max codebook g32, bf16 scale", 4.50, lambda w: lloyd(w, 32, 4, bf16)),
            ("Lloyd-Max codebook g16, fp8 scale", 4.50, lambda w: lloyd(w, 16, 4, fp8)),
        ],
        dists,
    )

    if "--vq" in sys.argv:
        print("\nvector quantization, gaussian only (IDEAS.md)")
        print(f"  {'scheme':<44}{'bits/w':>7}{'err':>8}")
        w = dists[0][1][: 1 << 19]
        g = 32
        grp = w.reshape(-1, g)
        s = bf16(np.abs(grp).max(1, keepdims=True))
        x = (grp / s).ravel()
        n = 15
        q = np.clip(np.round((x + 1) / 2 * n), 0, n) / n * 2 - 1
        print(f"  {'scalar 4-bit symmetric, g32 bf16 scale':<44}{4.5:>7.2f}"
              f"{err((q.reshape(-1, g) * s).ravel(), w):>7.2f}%")
        for d, k in ((2, 256), (2, 1024), (4, 1024), (4, 4096), (8, 4096)):
            V = np.ascontiguousarray(x.reshape(-1, d))
            sub = V[rng.choice(len(V), min(len(V), 60000), replace=False)]
            C = kmeans(sub, k)
            rec = (C[assign(V, C)].reshape(-1, g) * s).ravel()
            print(f"  {f'VQ dim {d}, {k} codes, g32 bf16 scale':<44}"
                  f"{np.log2(k) / d + 16 / g:>7.2f}{err(rec, w):>7.2f}%")

    print("\nthe floor: no code at this rate can do better on a gaussian")
    for r in (2.5, 3.0, 3.5, 4.0, 4.25, 4.5, 5.0, 8.5):
        print(f"  {r:>5} bits: {100 * 2**-r:>6.2f}%")


main()
