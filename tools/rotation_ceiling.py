"""What a QuaRot/SpinQuant-style rotation could buy here, before writing one.

    cd tools && ORNITH_GGML_BUILD=../../.ggml-build python3 rotation_ceiling.py

`REBUILD.md` Phase 4 surveyed HIGGS and parked it on three blockers, all of
which belong to HIGGS's *grid* rather than to its Hadamard: CUDA runtime, a
non-affine format, and no MLX oracle. QuaRot and SpinQuant are the rotation
without the grid -- an orthogonal `Q` folded offline into the neighbouring
weight matrices, with plain affine (or Q4_K) quantization afterwards -- so the
oracle survives, and the class is the last untried lever in this rebuild.

**This file prices it rather than building it, which is Phase 13's move.** The
premise of a rotation is incoherence: quantization error is dominated by
outlier channels, and an orthogonal mix spreads them. If the weights are
already close to incoherent at the granularity the quantizer actually sees,
the ceiling is low and there is nothing to build.

Three things are measured, on the parent's own weights:

1.  **Per-group spread.** Affine g32 fits a scale and a bias to each run of 32
    input channels, so a global outlier costs nothing -- it costs the 32
    weights that share its group. The quantity that binds is therefore
    `(max - min) / rms` **per group**, not per tensor, and the yardstick is
    what 32 Gaussian draws give (simulated here, not quoted).

2.  **The direct answer: rotate and re-quantize.** `W' = W @ Q` with `Q`
    orthonormal along the *input* axis -- which is the axis the groups run
    along, and the one QuaRot rotates because `x' = Q.T @ x` undoes it in the
    previous op. Quantize both and compare. Two rotations: a Hadamard, which
    is what an implementation would use, and a Haar-random orthogonal, which
    is the best case incoherence theory offers.

3.  **The error that matters, which is not the weight error.** Phases 4, 7,
    14 and 15 falsified weight norms as a selector four times. The closest
    stand-in with a defensible link to the output is the error in the layer's
    *product* under the real activation second moment:

        rel = sqrt( tr(dW H dW.T) / tr(W H W.T) )

    and under rotation `H' = Q.T H Q`, so the two are directly comparable --
    it is the same output error in two bases. `refs/gptq-parent/hessians/` and
    `refs/gptq-attn/hessians/` hold the full `H`, not its diagonal.

What this does **not** answer is whether the rotation is foldable. That is the
lesson AWQ left (`REBUILD.md` Phase 6): the fold point decides scope, not the
spread -- one of three candidate families was reachable, and the dead one had
140x the spread of the live one. Thirty of this model's forty layers are
gated-delta, whose internals are a conv, a recurrent fp32 state and a sigmoid
gate rather than q/k/v/o. So the ceiling here is an upper bound on a scope
that is separately in doubt, and a low ceiling closes the question either way.
"""

import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aq  # noqa: E402
import q4k  # noqa: E402
import q4k_ggml  # noqa: E402

PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"
REFS = pathlib.Path(__file__).resolve().parent.parent / "refs"

# Rows kept for modules too tall to carry a [O, I] x [I, I] product. The
# statistics here are per input group, so rows are samples of the same
# population and truncation costs precision, not validity.
MAX_ROWS = 1024

# (label, tensor key, hessian file or None, bits/group the artifact gives it)
MODULES = [
    ("lm_head", "lm_head.weight",
     "gptq-parent/hessians/language_model.lm_head.npy", 4),
    ("L0.in_proj_qkv", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
     "gptq-parent/hessians/language_model.model.layers.0.linear_attn.in_proj_qkv.npy", 4),
    ("L0.out_proj", "model.language_model.layers.0.linear_attn.out_proj.weight",
     "gptq-parent/hessians/language_model.model.layers.0.linear_attn.out_proj.npy", 4),
    ("L20.in_proj_qkv", "model.language_model.layers.20.linear_attn.in_proj_qkv.weight",
     "gptq-parent/hessians/language_model.model.layers.20.linear_attn.in_proj_qkv.npy", 4),
    ("L11.self_attn.q_proj", "model.language_model.layers.11.self_attn.q_proj.weight",
     "gptq-attn/hessians/language_model.model.layers.11.self_attn.q_proj.npy", 8),
    ("L11.self_attn.o_proj", "model.language_model.layers.11.self_attn.o_proj.weight",
     "gptq-attn/hessians/language_model.model.layers.11.self_attn.o_proj.npy", 8),
    # Layer 20, not layer 0. **Layer 0's experts are 41% dead in the parent**
    # -- over all 256 of them, mean 41.4% of each expert's intermediate
    # channels (median 43.9%, min 0.4%, max 81.8%, none fully alive) are a
    # whole row of `gate_up_proj` at the single bf16 pattern 0x01D4 =
    # 7.7877e-38, thirty-five orders below the mean weight, plus the matching
    # column of `down_proj`. No other layer shows it and no non-expert tensor
    # of layer 0 does. It is 0.38% of a decode step, so it is a measurement
    # hazard rather than a lever: any per-module statistic taken on layer 0's
    # experts is computed on a tensor that is 41% constant, and the first run
    # of this file tripped over exactly that. `REBUILD.md` Phase 20.
    ("L20.experts.gate_up", "model.language_model.layers.20.mlp.experts.gate_up_proj", None, 4),
    ("L20.experts.down", "model.language_model.layers.20.mlp.experts.down_proj", None, 4),
]

EXPERTS = 4  # experts sampled out of 256, for the routed tensors


def hadamard(n):
    """`H / sqrt(n)`, orthonormal. `n` is a power of two for every module here."""
    assert n & (n - 1) == 0, f"{n} is not a power of two"
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def haar(n, seed=0):
    """A Haar-random orthogonal matrix: the best case incoherence theory offers."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


def group_spread(w, group=32):
    """`(max - min) / rms` per group of `group` input channels."""
    g = w.reshape(-1, group).astype(np.float64)
    rms = np.sqrt((g**2).mean(1))
    rms = np.maximum(rms, 1e-30)
    return (g.max(1) - g.min(1)) / rms


def gaussian_baseline(group=32, draws=400000, seed=7):
    """What `group_spread` gives on Gaussian input -- the incoherent floor."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((draws, group))
    s = (g.max(1) - g.min(1)) / np.sqrt((g**2).mean(1))
    return float(np.median(s)), float(np.quantile(s, 0.99))


def affine(w, group, bits):
    rows, cols = w.shape
    q, s, b = aq.quantize(w, group, bits)
    return (q.reshape(rows, cols).astype(np.float32)
            * np.repeat(aq.bf16(s).reshape(rows, -1), group, 1)
            + np.repeat(aq.bf16(b).reshape(rows, -1), group, 1))


def kquant(lib, w, kind):
    return q4k_ggml.dequantize(
        lib, q4k_ggml.quantize(lib, w, kind=kind), w.size, kind=kind
    ).reshape(w.shape)


def rel(w, r):
    return 100.0 * np.linalg.norm(w - r) / np.linalg.norm(w)


def out_rel(w, r, h):
    """`sqrt(tr(dW H dW.T) / tr(W H W.T))`: the error in the layer's product."""
    d = (w - r).astype(np.float64)
    wf = w.astype(np.float64)
    hf = h.astype(np.float64)
    num = float((d @ hf * d).sum())
    den = float((wf @ hf * wf).sum())
    return 100.0 * np.sqrt(num / max(den, 1e-30))


def load(mx, idx, key):
    t = mx.load(str(PARENT / idx[key]))[key]
    mx.eval(t)
    w = np.ascontiguousarray(np.array(t.astype(mx.float32), copy=False))
    del t
    if w.ndim == 3:  # a routed expert tensor: take a few experts, stack rows
        w = w[:EXPERTS].reshape(-1, w.shape[-1])
    if w.shape[0] > MAX_ROWS:
        w = np.ascontiguousarray(w[:MAX_ROWS])
    return w


def main():
    import mlx.core as mx

    lib = q4k_ggml.load()
    idx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    g_med, g_p99 = gaussian_baseline()
    print(f"Gaussian floor for (max-min)/rms over 32: median {g_med:.2f}, p99 {g_p99:.2f}\n")

    rows = []
    for label, key, hpath, bits in MODULES:
        w = load(mx, idx, key)
        o, i = w.shape
        group = 32 if bits == 4 else 64
        h = None
        if hpath is not None and (REFS / hpath).exists():
            h = np.load(REFS / hpath)
            assert h.shape == (i, i), f"{label}: H is {h.shape}, weight is {w.shape}"

        rots = {"none": None, "hadamard": hadamard(i), "haar": haar(i)}
        out = {"label": label, "shape": (o, i), "bits": bits, "has_h": h is not None}
        for name, q in rots.items():
            wr = w if q is None else np.ascontiguousarray((w @ q).astype(np.float32))
            hr = h if (h is None or q is None) else (q.T @ h @ q)

            sp = group_spread(wr, group)
            rec = {
                "spread_med": float(np.median(sp)),
                "spread_p99": float(np.quantile(sp, 0.99)),
            }
            if bits == 4:
                a = affine(wr, 32, 4)
                k = kquant(lib, wr, "q4_K")
                rec["affine"] = rel(wr, a)
                rec["kquant"] = rel(wr, k)
                if hr is not None:
                    rec["affine_out"] = out_rel(wr, a, hr)
                    rec["kquant_out"] = out_rel(wr, k, hr)
            else:
                a = affine(wr, 64, 8)
                k = kquant(lib, wr, "q6_K")
                rec["affine"] = rel(wr, a)
                rec["kquant"] = rel(wr, k)
                if hr is not None:
                    rec["affine_out"] = out_rel(wr, a, hr)
                    rec["kquant_out"] = out_rel(wr, k, hr)
            out[name] = rec
            del wr, hr
        rows.append(out)
        del w, h
        report_one(out)

    print()
    summary(rows, g_med)


def report_one(r):
    o, i = r["shape"]
    fmt = "q4_K " if r["bits"] == 4 else "q6_K "
    aff = "affine int4 g32" if r["bits"] == 4 else "affine int8 g64"
    print(f"{r['label']}  [{o}, {i}]")
    print(f"  {'rotation':<10}{'spread med':>11}{'p99':>8}"
          f"{'  ' + aff:>18}{'  ' + fmt.strip():>10}"
          f"{'   out(affine)':>14}{'  out(' + fmt.strip() + ')':>12}")
    for name in ("none", "hadamard", "haar"):
        c = r[name]
        oa = f"{c['affine_out']:>13.3f}%" if "affine_out" in c else f"{'--':>14}"
        ok = f"{c['kquant_out']:>11.3f}%" if "kquant_out" in c else f"{'--':>12}"
        print(f"  {name:<10}{c['spread_med']:>11.2f}{c['spread_p99']:>8.2f}"
              f"{c['affine']:>17.3f}%{c['kquant']:>9.3f}%{oa}{ok}")
    print()


def summary(rows, g_med):
    print("=" * 78)
    print("ceiling, as a ratio rotated/unrotated (below 1.00 means the rotation helps)\n")
    print(f"{'module':<22}{'spread':>9}{'affine':>9}{'k-quant':>9}"
          f"{'out(aff)':>10}{'out(kq)':>9}   rotation")
    for r in rows:
        for name in ("hadamard", "haar"):
            b, c = r["none"], r[name]
            def ratio(k):
                return f"{c[k] / b[k]:>8.2f}" if k in b and k in c else f"{'--':>9}"
            print(f"{r['label']:<22}{c['spread_med'] / b['spread_med']:>9.2f}"
                  f"{ratio('affine')}{ratio('kquant')}"
                  f"{ratio('affine_out')}{ratio('kquant_out')}   {name}")
    print()
    worst = max(r["none"]["spread_med"] for r in rows)
    print(f"Gaussian floor is {g_med:.2f}; the least incoherent module already sits at "
          f"{min(r['none']['spread_med'] for r in rows):.2f}, the most at {worst:.2f}.")


if __name__ == "__main__":
    main()
