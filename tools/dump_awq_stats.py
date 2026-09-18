"""Calibration statistics for activation-aware scaling.

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_awq_stats.py [<artifact>]

`IDEAS.md`, "The highest-value conventional idea". Quantization minimises
weight error; what costs the model is **output** error, and the two differ
because activation magnitudes differ per input channel. Scaling a salient
channel up before quantizing gives it more of the grid, and the inverse folds
into the preceding op for free -- `W diag(s)` against `x diag(1/s)` is the
same product.

This is the pass that says which channels are salient. It records, for every
module the forge will scale, the input it actually sees over the perplexity
corpus:

  n, s1, s2   token count and per-channel sums of |x| and x**2. `s1/n` is
              AWQ's saliency statistic; `s2/n` is what a diagonal output-error
              model would use.
  rows        a deterministic stride sample of whole input vectors. The search
              scores a candidate scale by the output error it actually causes
              -- `||(W - Q_s(W)) X^T||` over these -- rather than by any
              diagonal proxy, because this project has already been bitten
              once by ranking formats on a quantity with no demonstrated link
              to the distribution (`REBUILD.md` Phase 4).

**Which modules, and why those.** The four families `int4-head-linattn` moved
to 4 bits, which are 57% of what a decode step reads. They fall into three
folding groups, and the group is what matters: every linear reading one
tensor has to share one scale.

  in_proj_qkv, in_proj_z  read `input_layernorm`, in the 30 linear-attention
                          layers. `in_proj_a` and `in_proj_b` read it too and
                          stay bf16 -- they are compensated, not scaled.
  out_proj                reads `linear_attn.norm` through the gate. That norm
                          is per-head-dim (128) and shared across the 32 value
                          heads, so its scale must be too; the input stat is
                          averaged over heads before the search sees it.
  lm_head                 reads `model.norm` -- and the MTP sidecar reaches
                          the same head through `mtp.norm`, so both norms fold.

Only in_proj_qkv and out_proj are instrumented: in_proj_z sees the identical
tensor, and recording it twice would just double the file.
"""

import json
import os
import pathlib
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

OUT = pathlib.Path(os.environ.get("ORNITH_AWQ_OUT", "../refs/awq"))
CORPUS = pathlib.Path(os.environ.get("ORNITH_AWQ_CORPUS",
                                     "../refs/artifact/ppl_corpus.txt"))
CHUNK = 256
ROWS = 256          # input vectors kept per module, on a fixed stride
SUFFIXES = ("linear_attn.in_proj_qkv", "linear_attn.out_proj", "lm_head")

RECORD = {}


def record(st, x):
    f = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    m = f.shape[0]
    st["s1"] = st["s1"] + mx.abs(f).sum(0)
    st["s2"] = st["s2"] + (f * f).sum(0)
    # Deterministic stride, so two runs of this file agree and a rerun after a
    # crash is the same sample.
    take = [i for i in range(m)
            if (st["n"] + i) % st["stride"] == 0 and len(st["rows"]) + 1 <= ROWS]
    for i in take:
        st["rows"].append(f[i])
    st["n"] += m


def patch(cls):
    orig = cls.__call__

    def patched(self, x, *a, **k):
        st = RECORD.get(id(self))
        if st is not None:
            record(st, x)
        return orig(self, x, *a, **k)

    cls.__call__ = patched


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else (
        os.environ.get("ORNITH_MODEL") or str(
            pathlib.Path.home()
            / ".mtplx/models/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx"))
    OUT.mkdir(parents=True, exist_ok=True)
    patch(nn.QuantizedLinear)
    patch(nn.Linear)
    model, tokenizer = load(model_dir)

    text = CORPUS.read_text()
    ids = tokenizer.encode(text, add_special_tokens=False)
    stride = max(1, (len(ids) - 1) // ROWS)

    names = {}
    for name, mod in model.named_modules():
        if name.endswith(SUFFIXES) and isinstance(mod, (nn.Linear, nn.QuantizedLinear)):
            RECORD[id(mod)] = {"n": 0, "s1": mx.zeros(1), "s2": mx.zeros(1),
                               "rows": [], "stride": stride}
            names[id(mod)] = name
    assert len(names) == 61, f"expected 61 modules, matched {len(names)}"
    print(f"instrumenting {len(names)} modules, stride {stride} over {len(ids)} tokens")

    cache = make_prompt_cache(model)
    for i in range(0, len(ids) - 1, CHUNK):
        model(mx.array([ids[i : i + CHUNK]]), cache=cache)
        for st in RECORD.values():
            mx.eval(st["s1"], st["s2"], *st["rows"][-CHUNK:])
        mx.clear_cache()
        print(f"  {min(i + CHUNK, len(ids))}/{len(ids)} tokens", flush=True)

    out = {}
    for k, st in RECORD.items():
        name = names[k]
        out[name + "/n"] = np.array(st["n"])
        out[name + "/s1"] = np.array(st["s1"], copy=False)
        out[name + "/s2"] = np.array(st["s2"], copy=False)
        out[name + "/rows"] = np.array(mx.stack(st["rows"]), copy=False)
        print(f"{name:<58} n={st['n']:>6} rows={len(st['rows']):>4} "
              f"dim={st['s1'].shape[0]:>5}")
    np.savez(OUT / "stats.npz", **out)
    (OUT / "stats.json").write_text(json.dumps({
        "model_dir": model_dir, "tokens": len(ids), "rows_per_module": ROWS,
        "stride": stride, "corpus": CORPUS.name, "modules": sorted(names.values()),
    }, indent=1))
    print(f"\nwrote {OUT}/stats.npz")


main()
