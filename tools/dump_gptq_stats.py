"""Hessians for GPTQ: the second moment of every input the four families see.

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_gptq_stats.py [<artifact>]

`REBUILD.md` Phase 7. Quantization picks, for every weight, the nearest point
of a grid. GPTQ picks a *different* point when a neighbour costs the layer's
**output** less, and pays for the deviation by adjusting the columns not yet
quantized. The direction of that adjustment is the inverse Hessian of the
layer's input, `H = 2 X^T X`, which is what this file measures.

**Why not `dump_awq_stats.py`'s `rows`.** That file keeps 256 whole input
vectors a module, which is a rank-256 sample. GPTQ inverts `H`, and at
`in_features` 2048 (4096 for `out_proj`) a rank-256 estimate is singular --
the compensation would run entirely on the damping term, which is RTN with
extra steps. So the accumulation here is over every token, and nothing is
kept but the matrix.

**Which modules, and the Hessian they share.** The 91 modules
`int4-head-linattn` moved to 4 bits, in three input groups:

  in_proj_qkv, in_proj_z  both read `input_layernorm`, so **one** 2048x2048
                          Hessian a layer serves both. Asserted, not assumed:
                          the first `VERIFY_CALLS` inputs of `in_proj_z` are
                          compared with `in_proj_qkv`'s element by element.
  out_proj                4096x4096, its own, one a layer.
  lm_head                 2048x2048, reads `model.norm`.

2.53 GB of float32 in total, written one `.npy` a group.

**The corpus is assembled here and written out.** omlx ships a 600-sample
calibration set (code / en / ko / zh / ja / tool_calling / reasoning, 744k
characters, and **no Russian**), and this project has `awq_calib.txt`, which
exists because Phase 6 learned that calibrating on the corpus you then score
on is an in-sample measurement. Both go in; the mix is written to
`refs/gptq/calib.json` with per-category token counts, because the one thing
Phase 5 and 6 agree on is that the damage lands in `arith` and `morph` and
the corpus is the first suspect when it does not move.

**Samples are independent, with a fresh cache each.** `dump_awq_stats.py`
runs one cache over a 6.6k-token corpus, which is fine for a per-channel mean
and wrong here: 30 of the 40 layers carry a recurrent state, and streaming
240k tokens through one would collect second moments from a state no real
prompt ever reaches.
"""

import json
import os
import pathlib
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(os.environ.get("ORNITH_GPTQ_OUT", "../refs/gptq"))
OMLX_CALIB = ROOT / "../vendor/omlx/omlx/oq_calibration_data.json"
OWN_CALIB = ROOT / "tools/awq_calib.txt"
SEQ = int(os.environ.get("ORNITH_GPTQ_SEQ", "1024"))
MAX_SAMPLES = int(os.environ.get("ORNITH_GPTQ_MAX_SAMPLES", "0"))  # 0 = all
VERIFY_CALLS = 4

# id(module) -> {"H": accumulator, "n": tokens}, for the modules that own an input.
RECORD = {}
# id(module) -> (group name, "qkv" | "z"); the first few inputs of each are kept
# so that "in_proj_z sees the identical tensor" is checked rather than trusted.
WITNESS = {}
SEEN = {}


def record(st, x):
    f = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    st["H"] = st["H"] + (f.T @ f)
    st["n"] += f.shape[0]


def patch(cls):
    orig = cls.__call__

    def patched(self, x, *a, **k):
        st = RECORD.get(id(self))
        if st is not None:
            record(st, x)
        w = WITNESS.get(id(self))
        if w is not None:
            seen = SEEN.setdefault(w[0], {"qkv": [], "z": []})
            if len(seen[w[1]]) < VERIFY_CALLS:
                seen[w[1]].append(mx.array(x.reshape(-1, x.shape[-1])[:8]))
        return orig(self, x, *a, **k)

    cls.__call__ = patched


def check_shared_inputs():
    """`in_proj_qkv` and `in_proj_z` must see the same tensor, or one Hessian
    cannot serve both. Compared bit for bit, like everything else here."""
    assert SEEN, "no witnesses recorded"
    for group, seen in sorted(SEEN.items()):
        a, b = seen["qkv"], seen["z"]
        assert len(a) == len(b) >= 1, \
            f"{group}: {len(a)} qkv / {len(b)} z witnesses"
        for i, (x, y) in enumerate(zip(a, b)):
            d = float(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max())
            assert d == 0.0, f"{group}: call {i} differs by {d}"
    n = len(next(iter(SEEN.values()))["qkv"])
    print(f"shared-input check: {len(SEEN)} layers, {n} calls each, max_abs == 0.0")


def build_corpus():
    """The calibration mix, as a list of strings, with a manifest."""
    cats, texts = {}, []
    data = json.loads(OMLX_CALIB.read_text())
    for cat, samples in data.items():
        s = [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
             for x in samples]
        cats[cat] = len(s)
        texts += s
    own = OWN_CALIB.read_text()
    # One paragraph a sample, so the short multilingual passages are not all
    # swallowed by one truncation.
    own_paras = [p.strip() for p in own.split("\n\n") if len(p.strip()) > 120]
    cats["ornith_awq_calib"] = len(own_paras)
    texts += own_paras
    return texts, cats


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else (
        os.environ.get("ORNITH_MODEL") or str(
            pathlib.Path.home()
            / ".mtplx/models/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hessians").mkdir(exist_ok=True)

    texts, cats = build_corpus()
    if MAX_SAMPLES:
        texts = texts[:MAX_SAMPLES]
    print(f"corpus: {len(texts)} samples, {sum(len(t) for t in texts)} chars")

    patch(nn.QuantizedLinear)
    patch(nn.Linear)
    model, tokenizer = load(model_dir)

    names = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, nn.QuantizedLinear)):
            continue
        if name.endswith(("linear_attn.in_proj_qkv", "linear_attn.out_proj")) \
                or name.endswith("lm_head"):
            dim = mod.weight.shape[1] * 32 // mod.bits \
                if isinstance(mod, nn.QuantizedLinear) else mod.weight.shape[1]
            RECORD[id(mod)] = {"H": mx.zeros((dim, dim), mx.float32), "n": 0}
            names[id(mod)] = name
        if name.endswith(("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")):
            WITNESS[id(mod)] = (name.rsplit(".", 1)[0],
                                "qkv" if name.endswith("qkv") else "z")
    assert len(names) == 61, f"expected 61 accumulators, matched {len(names)}"
    assert len(WITNESS) == 60, f"expected 60 in_proj modules, matched {len(WITNESS)}"
    print(f"accumulating {len(names)} Hessians "
          f"({sum(st['H'].size * 4 for st in RECORD.values()) / 2**30:.2f} GB fp32)")

    t0, tok = time.time(), 0
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)[:SEQ]
        if len(ids) < 8:
            continue
        # A fresh cache: each sample is its own sequence.
        cache = make_prompt_cache(model)
        model(mx.array([ids]), cache=cache)
        mx.eval(*[st["H"] for st in RECORD.values()])
        del cache
        mx.clear_cache()
        tok += len(ids)
        if (i + 1) % 25 == 0 or i + 1 == len(texts):
            el = time.time() - t0
            print(f"  {i + 1}/{len(texts)} samples, {tok} tokens, "
                  f"{el:.0f}s ({tok / max(el, 1e-9):.0f} tok/s)", flush=True)

    check_shared_inputs()

    manifest = {"model_dir": model_dir, "samples": len(texts), "tokens": tok,
                "seq": SEQ, "categories": cats,
                "sources": [str(OMLX_CALIB.resolve()), str(OWN_CALIB.resolve())],
                "modules": {}}
    for k, st in RECORD.items():
        name = names[k]
        H = np.array(st["H"], copy=False)
        d = np.diag(H)
        manifest["modules"][name] = {
            "n": st["n"], "dim": int(H.shape[0]),
            "dead": int((d == 0).sum()),
            "diag_mean": float(d.mean()), "diag_max": float(d.max()),
        }
        np.save(OUT / "hessians" / f"{name}.npy", H)
        print(f"{name:<58} n={st['n']:>7} dim={H.shape[0]:>5} "
              f"dead={int((d == 0).sum()):>4} diag_mean={d.mean():.4g}")
    (OUT / "calib.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {OUT}/hessians/ and {OUT}/calib.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
