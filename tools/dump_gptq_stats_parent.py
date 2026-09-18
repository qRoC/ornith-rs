"""The same Hessians as `dump_gptq_stats.py`, but taken from the **bf16 parent**.

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_gptq_stats_parent.py [--check]

`REBUILD.md` Phase 9, and it exists because Phase 8 found a mistake in Phase 7.

GPTQ minimises `||(W - Wq) X^T||` over the inputs `X` it was calibrated on, so
it reconstructs **whatever model produced those inputs**. Phase 7 captured them
from `A` -- the control artifact, itself quantized at 260 modules to 8 bits --
because that was the best model that would load. So the forged variant was
fitted to `A`'s layer outputs, including `A`'s own damage, and was then scored
on its distance from `A`. Against the real reference the margin fell from
38/58, p = 0.025 to **31/57, p = 0.60**.

This file removes that confound: the same corpus, the same 61 accumulators, the
same arithmetic, `X` taken from the parent instead. `parent_stream.py` does the
streaming and owns the sanitize trap.

**It is a falsifiable prediction, not an improvement.** If GPTQ on parent
Hessians beats GPTQ on `A` Hessians *when both are judged against the parent*,
the mechanism above is confirmed and extending GPTQ to the 256 experts of 40
layers -- 90%+ of the parameters, never touched -- is worth the much larger
job it needs. If it does not, the mechanism was wrong, GPTQ's ceiling on this
model is low, and that job should not start.

**Budget.** 254028 tokens of hidden state at bf16 is 1.04 GB and has to persist
across layers, since the loop runs every sample through layer *i* before layer
*i+1* exists. Plus 2.36 GB of accumulators and one 3.13 GiB layer: about 7 GiB
against a 51.8 GiB working set.

**`lm_head` is never run.** Its input is `norm(h)` after the last layer, so its
Hessian is accumulated directly from the stored hidden states at the end --
248320 output rows of a matrix nothing here multiplies by.
"""

import json
import os
import pathlib
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from dump_gptq_stats import build_corpus
from parent_stream import Parent, gib, mem

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(os.environ.get("ORNITH_GPTQ_OUT", "../refs/gptq-parent"))
PARENT = pathlib.Path(os.environ.get(
    "ORNITH_PARENT",
    str(pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B")))
SEQ = int(os.environ.get("ORNITH_GPTQ_SEQ", "1024"))
MAX_SAMPLES = int(os.environ.get("ORNITH_GPTQ_MAX_SAMPLES", "0"))
# `ORNITH_GPTQ_SPLIT=k` keeps k accumulators and routes sample s to s % k, so
# one pass yields k **independent** calibrations whose sum is the single one.
# That is what turns a two-replicate spread estimate (one degree of freedom,
# Phase 10) into something with a shape. Each quarter is a quarter of the data,
# so its Hessian is a noisier estimate than the production one -- the spread it
# measures is an upper bound on the spread at full corpus size, not an estimate
# of it.
SPLIT = int(os.environ.get("ORNITH_GPTQ_SPLIT", "1"))
CHECK_LAYERS, CHECK_SAMPLES = 3, 8
VERIFY_CALLS = 4

RECORD = {}     # id(module) -> [ {"H", "n"} ] * SPLIT
WITNESS = {}    # id(module) -> (group, "qkv" | "z")
SEEN = {}
CUR = [0]       # which split the sample now running belongs to


def record(sts, x):
    st = sts[CUR[0]]
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
    """One Hessian serves `in_proj_qkv` and `in_proj_z` only if they see the
    identical tensor. Compared bit for bit, like everything else here."""
    assert SEEN, "no witnesses recorded"
    for group, seen in sorted(SEEN.items()):
        a, b = seen["qkv"], seen["z"]
        assert len(a) == len(b) >= 1, f"{group}: {len(a)} qkv / {len(b)} z"
        for i, (x, y) in enumerate(zip(a, b)):
            d = float(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max())
            assert d == 0.0, f"{group}: call {i} differs by {d}"
    n = len(next(iter(SEEN.values()))["qkv"])
    print(f"shared-input check: {len(SEEN)} layers, {n} calls each, max_abs == 0.0")


def main():
    check = "--check" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hessians").mkdir(exist_ok=True)

    texts, cats = build_corpus()
    if MAX_SAMPLES or check:
        texts = texts[:(CHECK_SAMPLES if check else MAX_SAMPLES)]
    print(f"corpus: {len(texts)} samples, {sum(len(t) for t in texts)} chars")

    src = Parent(PARENT)
    patch(nn.Linear)
    patch(nn.QuantizedLinear)
    model, tokenizer = load(str(PARENT), lazy=True)
    lm, inner = model.language_model, model.language_model.model
    layers = inner.layers
    print(f"structure built, {len(layers)} layers, {mem()}")
    print(f"norm shift check: layer 0 mean |gain| {src.check_norm_shift():.4f}")

    names, head = {}, None
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, nn.QuantizedLinear)):
            continue
        if name.endswith(("linear_attn.in_proj_qkv", "linear_attn.out_proj")):
            RECORD[id(mod)] = [{"H": mx.zeros((mod.weight.shape[1],) * 2, mx.float32),
                                "n": 0} for _ in range(SPLIT)]
            names[id(mod)] = name
        elif name.endswith("lm_head"):
            head = (name, mod.weight.shape[1])
        if name.endswith(("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")):
            WITNESS[id(mod)] = (name.rsplit(".", 1)[0],
                                "qkv" if name.endswith("qkv") else "z")
    assert len(names) == 60, f"expected 60 layer accumulators, matched {len(names)}"
    assert head is not None and len(WITNESS) == 60, (head, len(WITNESS))
    per = sum(sts[0]["H"].size for sts in RECORD.values()) + head[1] ** 2
    print(f"accumulating {len(names) + 1} Hessians x {SPLIT} split(s) "
          f"({per * 4 * SPLIT / 2**30:.2f} GB)")

    # Tokenize and embed once. The hidden states persist across the whole run:
    # layer i+1 does not exist until every sample has been through layer i.
    seqs = [tokenizer.encode(t, add_special_tokens=False)[:SEQ] for t in texts]
    seqs = [s for s in seqs if len(s) >= 8]
    emb = src.get("model.language_model.embed_tokens.")
    inner.update(tree_unflatten([("embed_tokens.weight",
                                  emb["language_model.model.embed_tokens.weight"])]))
    h = [inner.embed_tokens(mx.array([s])) for s in seqs]
    mx.eval(h)
    del emb
    inner.update(tree_unflatten([("embed_tokens.weight", src.raw(
        "model.language_model.embed_tokens.weight"))]))
    mx.clear_cache()
    ntok = sum(len(s) for s in seqs)
    print(f"embedded {len(seqs)} samples, {ntok} tokens, {mem()}")

    t0 = time.time()
    n_layers = CHECK_LAYERS if check else len(layers)
    for i in range(n_layers):
        layer = layers[i]
        layer.update(tree_unflatten(list(src.layer(i).items())))
        for s in range(len(seqs)):
            # A fresh cache per sample: each one is its own sequence, which is
            # what `dump_gptq_stats.py` does and what a real prompt looks like.
            c = ArraysCache(size=2) if layer.is_linear else KVCache()
            mask = (create_ssm_mask(h[s], c) if layer.is_linear
                    else create_attention_mask(h[s], c))
            CUR[0] = s % SPLIT
            h[s] = layer(h[s], mask=mask, cache=c)
            if s % 64 == 0 or s == len(seqs) - 1:
                mx.eval(h[max(0, s - 63):s + 1],
                        [st["H"] for sts in RECORD.values() for st in sts])
        mx.eval(h, [st["H"] for sts in RECORD.values() for st in sts])
        layer.update(tree_unflatten(list(src.layer(i).items())))
        mx.clear_cache()
        print(f"  layer {i:>2}  {mem()}  {time.time() - t0:7.1f}s", flush=True)

    if check:
        check_shared_inputs()
        print(f"\nCHECK: {CHECK_LAYERS} layers, {len(seqs)} samples, "
              f"peak {gib(mx.get_peak_memory()):.2f} GiB. Active must be flat.")
        return

    check_shared_inputs()

    # `lm_head` reads `norm(h)`, so its Hessian comes from the hidden states
    # rather than from a forward -- the head itself is never run here.
    norm_w = src.get("model.language_model.norm.")
    inner.update(tree_unflatten(
        [("norm.weight", norm_w["language_model.model.norm.weight"])]))
    Hh = [mx.zeros((head[1],) * 2, mx.float32) for _ in range(SPLIT)]
    nh = [0] * SPLIT
    for s in range(len(h)):
        k = s % SPLIT
        f = inner.norm(h[s]).reshape(-1, head[1]).astype(mx.float32)
        Hh[k] = Hh[k] + f.T @ f
        nh[k] += f.shape[0]
        if s % 64 == 0:
            mx.eval(Hh)
    mx.eval(Hh)
    print(f"lm_head Hessians from {sum(nh)} tokens, {mem()}")

    for k in range(SPLIT):
        out = OUT if SPLIT == 1 else OUT.parent / f"{OUT.name}-q{k}"
        (out / "hessians").mkdir(parents=True, exist_ok=True)
        manifest = {"model_dir": str(PARENT), "streamed": True,
                    "split": f"{k} of {SPLIT}", "samples": len(seqs) // SPLIT,
                    "tokens": ntok, "seq": SEQ, "categories": cats, "modules": {}}

        def store(name, H, n):
            A = np.array(H, copy=False)
            d = np.diag(A)
            manifest["modules"][name] = {"n": n, "dim": int(A.shape[0]),
                                         "dead": int((d == 0).sum()),
                                         "diag_mean": float(d.mean()),
                                         "diag_max": float(d.max())}
            np.save(out / "hessians" / f"{name}.npy", A)

        for mid, sts in RECORD.items():
            store(names[mid], sts[k]["H"], sts[k]["n"])
        store(head[0], Hh[k], nh[k])
        manifest["tokens"] = sum(m["n"] for m in manifest["modules"].values()
                                 if m["dim"] == 2048) // (len(names) // 2 + 1)
        (out / "calib.json").write_text(json.dumps(manifest, indent=1))
        print(f"wrote {out}  ({nh[k]} tokens on lm_head)")
    print(f"\ndone in {time.time() - t0:.0f}s, peak {gib(mx.get_peak_memory()):.2f} GiB")


if __name__ == "__main__":
    main()
