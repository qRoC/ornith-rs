"""GPTQ over mlx's affine layout, in numpy, with `aq.quantize`'s exact grid.

`REBUILD.md` Phase 7. Round-to-nearest picks, for every weight, the closest
point of the group's grid. GPTQ picks a point that costs the layer's *output*
less: it walks the input dimension in order and, after rounding column `j`,
pushes the error it just made into the columns it has not reached yet, in the
direction the inverse Hessian of the calibration input says is cheapest.

    err = (w_j - dequant(q_j)) / Hinv[j, j]
    W[:, j+1:] -= err (x) Hinv[j, j+1:]

The output is an ordinary affine artifact -- same bits, same group size, same
three tensors, same kernels. Only which integer each weight got changes.

**Four decisions, each of which could have gone the other way.**

1. **The block is the group.** The reference implementation processes 128
   columns at a time and re-derives a group's scale from a `W` that the block's
   own in-flight compensation has not reached yet, so a group 96 columns into
   a block is fitted to stale weights. Taking `blocksize == group_size` makes
   every group start on a block boundary, so the scale is always derived from
   the weights that will actually be quantized. Same arithmetic, no staleness.

2. **The error is measured against what the kernel will read.** mlx computes
   codes against an unrounded fp32 scale and then stores bf16, so the
   dequantizer uses slightly different numbers than the quantizer did
   (`aq.quantize`'s docstring). The compensation here is fed
   `w - (q * bf16(scale) + bf16(bias))`, the error the model actually incurs,
   not the error against the fp32 scale.

3. **Codes are clamped on both sides.** mlx clamps only above, because for
   untouched weights `q >= 0` holds by construction. Compensation breaks that:
   a column can be pushed outside the range its own group's scale was fitted
   to. Clamping below is therefore new, and it is the only place this file
   departs from `aq.quantize`'s arithmetic -- it cannot fire when
   `compensate=False`, which is what `forge_gptq.py --off` asserts.

4. **No act-order.** Quantizing columns in order of decreasing `diag(H)` is
   the standard companion trick and it needs a `g_idx` per column, which
   mlx's affine layout has no room for. Permuting the weight instead would
   need the input permuted too, and the input comes out of an RMSNorm shared
   with modules that are not being touched. Same fold-point wall Phase 6 hit.
"""

import numpy as np

import aq

F = np.float32


def group_params(g, bits):
    """`aq.quantize`'s scale/bias for one group block, per row.

    `g` is [rows, group_size] -- the columns of one group, all rows. Returns
    the fp32 pair the codes are computed against and the bf16 pair that is
    stored, which are not the same numbers; see `aq.quantize`.
    """
    n_bins = F((1 << bits) - 1)
    g = g.astype(np.float32)
    w_min = g.min(1)
    w_max = np.maximum(g.max(1), F(0))          # the kernel inits w_max to 0
    scale = np.maximum((w_max - w_min) / n_bins, F(1e-7)).astype(np.float32)
    side = np.abs(w_min) > np.abs(w_max)
    scale = np.where(side, scale, -scale).astype(np.float32)
    edge = np.where(side, w_min, w_max).astype(np.float32)
    q0 = aq.rnd(edge / scale)
    at_zero = q0 == F(0)
    scale = np.where(at_zero, scale, edge / np.where(at_zero, F(1), q0)).astype(np.float32)
    bias = np.where(at_zero, F(0), edge).astype(np.float32)
    return scale, bias, aq.bf16(scale), aq.bf16(bias)


def prepare(H, percdamp=0.01):
    """The upper-triangular Cholesky factor of a damped `H**-1`, and the dead set.

    `H` is `sum x x^T` and the factor 2 in GPTQ's `2 X^T X` is irrelevant: the
    loop only ever uses `Hinv[j, k] / Hinv[j, j]`, and the damping is taken
    relative to the mean of the diagonal, so both are invariant to scaling H.
    """
    H = np.array(H, np.float64, copy=True)      # float64: this is an inverse
    d = np.diag(H).copy()
    dead = d == 0
    if dead.any():
        H[dead, dead] = 1.0
    damp = percdamp * d[~dead].mean() if (~dead).any() else 1.0
    H[np.diag_indices_from(H)] += damp
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T.copy()   # upper factor
    assert np.isfinite(Hinv).all(), "Hessian inverse is not finite"
    return Hinv.astype(np.float64), dead


def quantize(w, Hinv, dead, group_size=32, bits=4, compensate=True):
    """One module. `w` is [out, in] float32; returns (codes, scale, bias).

    With `compensate=False` this is `aq.quantize` exactly, which is the check
    that the grid was not accidentally reimplemented.
    """
    out, cols = w.shape
    assert cols % group_size == 0, (out, cols, group_size)
    n_bins = F((1 << bits) - 1)
    W = w.astype(np.float32, copy=True)
    if dead is not None and dead.any():
        W[:, dead] = F(0)

    Q = np.empty((out, cols), np.float32)
    S = np.empty((out, cols // group_size), np.float32)
    B = np.empty((out, cols // group_size), np.float32)

    for j0 in range(0, cols, group_size):
        j1 = j0 + group_size
        s_use, b_use, s_st, b_st = group_params(W[:, j0:j1], bits)
        S[:, j0 // group_size] = s_st
        B[:, j0 // group_size] = b_st

        blk = W[:, j0:j1].copy()
        err = np.empty((out, group_size), np.float32)
        Hb = Hinv[j0:j1, j0:j1]
        for i in range(group_size):
            col = blk[:, i]
            q = aq.rnd((col - b_use) / s_use)
            q = np.clip(q, F(0), n_bins)
            Q[:, j0 + i] = q
            if not compensate:
                err[:, i] = F(0)
                continue
            # The value the kernel will reconstruct, from the stored bf16 pair.
            deq = (q * s_st + b_st).astype(np.float32)
            e = ((col - deq) / F(Hb[i, i])).astype(np.float32)
            err[:, i] = e
            if i + 1 < group_size:
                blk[:, i + 1:] -= np.outer(e, Hb[i, i + 1:].astype(np.float32))
        if compensate and j1 < cols:
            W[:, j1:] -= err @ Hinv[j0:j1, j1:].astype(np.float32)

    return Q.astype(np.uint8), S, B


def relative_output_error(w, wq, H):
    """`||(W - Wq) X^T||_F / ||W X^T||_F`, via H, without touching X.

    `tr(D H D^T)` with `D = W - Wq`, which is what the objective actually is
    and the currency Phase 6 found wanting as a *chooser*. Reported, not obeyed.
    """
    H32 = np.asarray(H, np.float32)

    def quad(A):
        # float32 matmul, float64 reduction: `lm_head` is 248320 x 2048 and the
        # float64 einsum takes minutes to say the same thing.
        return float(((A @ H32) * A).astype(np.float64).sum())

    num = quad((w - wq).astype(np.float32))
    den = quad(np.asarray(w, np.float32))
    return float(np.sqrt(max(num, 0.0) / max(den, 1e-30)))
