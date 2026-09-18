"""What the router actually does, and what that says about the MoE read.

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 router_stats.py [<artifact>]

`switch_mlp` is 34.5% of what a decode step reads at `int4-head-linattn` --
the largest family once Phase 4 halved the other four -- and its size is set
by `num_experts_per_tok = 8`, a *runtime* constant. Nothing in the artifact
has to change to read fewer experts, so two questions are worth pricing
before anything is built, and this file answers both in one pass.

  mass      how peaked the top-8 gate weights are. If the 8th expert carries
            almost nothing, a cumulative-mass threshold reads fewer of them.
  reuse     how much two adjacent tokens' expert sets overlap. A speculative
            verify of `d` tokens reads the *union*, not `8d`, so this decides
            how the MoE read scales with speculation depth -- and whether
            `switch_glu`'s `rows >= 64` sort threshold (depth 8) is leaving
            anything on the table at the depths actually used.

Both corpora are read: the perplexity corpus and the held-out multilingual
calibration text, so the answer is not a property of one language.
"""

import pathlib
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

CORPORA = ["../refs/artifact/ppl_corpus.txt", "awq_calib.txt"]
CHUNK, TOPK = 256, 8
SWITCH_SHARE, STEP_GB = 0.345, 2.0761
REC, OUT = {}, {}


def patch(cls):
    orig = cls.__call__

    def patched(self, x, *a, **k):
        out = orig(self, x, *a, **k)
        if id(self) in REC:
            logits = out.astype(mx.float32)
            p = mx.softmax(logits, axis=-1)
            top = mx.sort(p, axis=-1)[..., -TOPK:][..., ::-1]
            REC[id(self)].append((top / top.sum(-1, keepdims=True),
                                  mx.argsort(logits, axis=-1)[..., -TOPK:]))
        return out

    cls.__call__ = patched


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else str(
        pathlib.Path.home() / ".mtplx/models/ornith-1.5-35b-a3b-int4-head-linattn")
    patch(nn.QuantizedLinear)
    patch(nn.Linear)
    model, tok = load(d)
    names = {}
    for n, m in model.named_modules():
        if n.endswith("mlp.gate") and isinstance(m, (nn.Linear, nn.QuantizedLinear)):
            REC[id(m)] = []
            names[id(m)] = n
    assert len(names) == 40, len(names)

    text = "\n\n".join(pathlib.Path(c).read_text() for c in CORPORA)
    ids = tok.encode(text, add_special_tokens=False)
    for i in range(0, len(ids) - 1, CHUNK):
        model(mx.array([ids[i : i + CHUNK]]), cache=None)
        for k, v in REC.items():
            mx.eval(*[a for pair in v for a in pair])
            w = np.array(mx.concatenate([a for a, _ in v], axis=1).reshape(-1, TOPK), copy=False)
            e = np.array(mx.concatenate([b for _, b in v], axis=1).reshape(-1, TOPK), copy=False)
            OUT.setdefault(names[k], []).append((w, e))
            REC[k] = []
        mx.clear_cache()

    W = np.concatenate([np.concatenate([w for w, _ in v]) for v in OUT.values()])
    cum = np.cumsum(W, axis=1)
    print(f"{len(ids)} tokens x {len(OUT)} layers = {W.shape[0]:,} routing decisions\n")
    print(f"{'k':>2} {'mean weight':>12} {'mean cum mass':>14}   "
          f"(uniform over 8 would be {1/TOPK:.4f})")
    for k in range(TOPK):
        print(f"{k+1:>2} {W[:,k].mean():>12.4f} {cum[:,k].mean():>14.4f}")

    print(f"\n{'threshold':>10} {'mean k':>8} {'k=8 kept':>9} {'step saving':>12}")
    for tau in (0.80, 0.90, 0.95, 0.99):
        need = np.minimum((cum < tau).sum(1) + 1, TOPK)
        kbar = need.mean()
        print(f"{tau:>10.2f} {kbar:>8.2f} {(need==TOPK).mean():>8.1%} "
              f"{SWITCH_SHARE*(1-kbar/TOPK):>11.1%}")

    dense = STEP_GB * (1 - SWITCH_SHARE)
    print(f"\nexpert reuse across a window of d consecutive tokens")
    print(f"{'d':>3} {'unique':>8} {'vs 8d':>7} {'GB/token':>9} {'vs d=1':>8}   sorted by switch_glu?")
    for dd in (1, 2, 3, 4, 6, 8):
        u = []
        for v in OUT.values():
            e = np.concatenate([x for _, x in v])
            u += [len(np.unique(e[s : s + dd])) for s in range(0, len(e) - dd, dd)]
        m = float(np.mean(u))
        gb = (dense + STEP_GB * SWITCH_SHARE * m / TOPK) / dd
        print(f"{dd:>3} {m:>8.2f} {m/(TOPK*dd):>7.1%} {gb:>9.3f} {gb/STEP_GB-1:>+8.1%}   "
              f"{'yes' if dd*TOPK >= 64 else 'no -- rows < 64'}")


main()
