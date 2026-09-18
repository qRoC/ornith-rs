"""Search the activation-aware channel scales, and price what exactness costs.

    PYTHONPATH=../vendor/mlx-lm python3 tools/awq_search.py

Reads `refs/awq/stats.npz` (`dump_awq_stats.py`) and the bf16 parent, and
writes `refs/awq/scales.npz` for `forge_awq.py`. Nothing is built here; this
file only chooses numbers and reports what they buy.

**The objective is output error, measured, not weight error.** For each
folding group -- the linears that share one input tensor, and therefore have
to share one scale -- a candidate `s` is scored by

    sum over the group of  || (W - Q_s(W)) X^T ||_F  ,   Q_s(W) = Q(W diag(s)) diag(1/s)

on the calibration rows `X` the model actually produced. That is the quantity
the idea is about. `weight_error.py`'s currency is deliberately not used:
Phase 4 spent four experiments learning that it ranks formats and not
outcomes.

**`out_proj` is not in the experiment, and the reason is the fold point.**
It has by far the largest activation spread of the three groups -- up to
1319x between its biggest channel and the median -- and none of it is
reachable. Its scale can only fold into `linear_attn.norm`, which is
per-head-dim (128) and shared by all 32 value heads, while the outliers sit
at individual (head, dim) pairs: taking out the per-dim mean leaves 98-114%
of the log-variance behind. The gate is the other candidate and it is
nonlinear -- `silu(z) * normed` -- so nothing folds through it. Spread is
necessary and not sufficient; what decides is whether the preceding op is
indexed the same way the saliency is. That leaves `in_proj_qkv`, `in_proj_z`
and `lm_head`, which is 47.4% of a decode step's traffic.

**Candidates.** AWQ's family, `s = (mean|x|)**alpha` normalised to geometric
mean 1, over an 11-point grid in alpha. `alpha = 0` is `s = 1`, which
reproduces `int4-head-linattn` exactly and is both the baseline and a
correctness check.

**And a power-of-two variant of every candidate**, which is the part that is
this project's rather than AWQ's. Folding `1/s` into an RMSNorm gain, and `s`
into the bf16 `in_proj_a` and `in_proj_b` that read the same tensor, is a
bf16 multiply -- ~0.4% of uncontrolled relative error, applied per channel to
a whole channel. That is the shape of the error that killed nvfp4 (a scale
with 20% error multiplying a whole group), one order down. Restricted to
powers of two every fold is **exact**: a bf16 times 2**k is a bf16, so the
reparametrisation is free of error and everything that moves is attributable
to the quantizer alone -- which is the only way the oracle stays meaningful.
The grid pays for it in resolution, and the report prices exactly that.
"""

import json
import os
import pathlib
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aq  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
AWQ = ROOT / os.environ.get("ORNITH_AWQ_DIR", "refs/awq")
PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"
GROUP_SIZE, BITS = 32, 4
ALPHAS = np.linspace(0.0, 1.0, 21)
SEARCH_ROWS = 32768      # output rows sampled when a matrix is taller than this


def parent_name(mod):
    """Artifact module path -> parent tensor name. Only the two simple cases."""
    if mod == "language_model.lm_head":
        return "lm_head.weight"
    n, rest = mod[len("language_model.model.layers.") :].split(".", 1)
    return f"model.language_model.layers.{n}.{rest}.weight"


def load_parent(pidx, mod):
    name = parent_name(mod)
    t = mx.load(str(PARENT / pidx[name]))[name]
    mx.eval(t)
    w = np.array(t.astype(mx.float32), copy=False)
    del t
    return w


def dequant(q, s, b, shape):
    g = q.astype(np.float32).reshape(-1, GROUP_SIZE) * s[:, None] + b[:, None]
    return g.reshape(shape)


def quant_roundtrip(w):
    q, s, b = aq.quantize(w, GROUP_SIZE, BITS)
    return dequant(q, s, b, w.shape)


def err(ws, x, s):
    """Sum of ||(W - Q_s(W)) X^T||_F**2 over a group sharing the scale s."""
    total = 0.0
    for w in ws:
        what = quant_roundtrip(w * s) / s
        total += float(np.square((w - what) @ x.T).sum())
    return total


def subsample(w, rng_rows):
    return w if w.shape[0] <= rng_rows else w[:: int(np.ceil(w.shape[0] / rng_rows))]


def candidates(sal):
    """AWQ's family plus its power-of-two rounding, keyed by a readable name."""
    sal = np.maximum(sal, sal.max() * 1e-6)
    out = {}
    for a in ALPHAS:
        s = np.power(sal, a).astype(np.float32)
        s = (s / np.sqrt(s.max() * s.min())).astype(np.float32)
        out[f"cont@{a:.1f}"] = s
        out[f"po2@{a:.1f}"] = np.exp2(np.round(np.log2(s))).astype(np.float32)
    return out


def pick(name, ws, x, sal, report):
    """Score every candidate for one folding group and keep the best po2 one."""
    t0 = time.time()
    base = None
    scored = {}
    for key, s in candidates(sal).items():
        e = err(ws, x, s)
        scored[key] = e
        if key == "cont@0.0":
            base = e
    best_c = min((k for k in scored if k.startswith("cont")), key=scored.get)
    best_p = min((k for k in scored if k.startswith("po2")), key=scored.get)
    row = {
        "group": name,
        "baseline": base,
        "best_cont": best_c, "cont_gain": 1.0 - scored[best_c] / base,
        "best_po2": best_p, "po2_gain": 1.0 - scored[best_p] / base,
        "seconds": time.time() - t0,
    }
    report.append(row)
    print(f"{name:<26} base {base:.4e}   cont {best_c:<9} {row['cont_gain']:+7.2%}"
          f"   po2 {best_p:<9} {row['po2_gain']:+7.2%}   {row['seconds']:.0f}s",
          flush=True)
    return candidates(sal)[best_p], best_p


def main():
    stats = np.load(AWQ / "stats.npz")
    pidx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    layers = sorted({int(k.split(".layers.")[1].split(".")[0])
                     for k in stats.files if ".layers." in k})
    print(f"{len(layers)} linear-attention layers, plus lm_head\n")

    scales, report, chosen = {}, [], {}

    for i in layers:
        p = f"language_model.model.layers.{i}.linear_attn."
        sal = stats[p + "in_proj_qkv/s1"] / stats[p + "in_proj_qkv/n"]
        x = stats[p + "in_proj_qkv/rows"]
        ws = [subsample(load_parent(pidx, p + m), SEARCH_ROWS)
              for m in ("in_proj_qkv", "in_proj_z")]
        s, key = pick(f"layer{i:02d}.in", ws, x, sal, report)
        scales[f"layer{i:02d}.in"] = s
        chosen[f"layer{i:02d}.in"] = key
        del ws

    sal_h = stats["language_model.lm_head/s1"] / stats["language_model.lm_head/n"]
    x_h = stats["language_model.lm_head/rows"]
    w_h = subsample(load_parent(pidx, "language_model.lm_head"), SEARCH_ROWS)
    s_h, key = pick("lm_head", [w_h], x_h, sal_h, report)
    scales["lm_head"] = s_h
    chosen["lm_head"] = key

    np.savez(AWQ / "scales.npz", **scales)
    (AWQ / "search.json").write_text(json.dumps(
        {"group_size": GROUP_SIZE, "bits": BITS, "alphas": list(ALPHAS),
         "search_rows": SEARCH_ROWS, "chosen": chosen, "report": report}, indent=1))
    g = np.mean([r["po2_gain"] for r in report])
    c = np.mean([r["cont_gain"] for r in report])
    print(f"\nmean output-error reduction: continuous {c:+.2%}, power-of-two {g:+.2%}")
    print(f"wrote {AWQ}/scales.npz")


main()
