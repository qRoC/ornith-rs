"""Hessians for the **routed experts** -- 93.1% of the model, never re-rounded.

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_gptq_experts.py [--check]

`REBUILD.md` Phase 12. Phases 7 to 11 re-rounded 91 modules and bought 23% of
round-to-nearest's damage. Those 91 modules are **1.52 B parameters, 3.7%** of
the artifact. The 256 experts of 40 layers are **38.65 B, 93.1%**, and they have
been round-to-nearest since the parent's own recipe -- their damage sits inside
the 0.0185 nats that even the control `A` pays, which no phase has attacked.

**Two Hessians a layer, not 256.** Each expert sees only the tokens routed to
it, so the exact object is a per-expert `H_e`, and at 2048 x 2048 x 256 experts
x 40 layers that is 172 GB. The reference implementations batch instead: one
Hessian a layer, shared by its experts, which is what makes a single
column-by-column walk serve all 256 at once. Pooling over **(token, expert)
pairs** rather than over tokens is what the hook does naturally -- a token that
routes to eight experts contributes eight rows -- so the pooled estimate is
routing-weighted rather than uniform.

  gate_proj, up_proj   read the same tensor, the MoE block's input. One
                       2048x2048 Hessian a layer serves both, and the identity
                       is asserted rather than assumed, as with in_proj_z.
  down_proj            reads the SwiGLU output, which is expert-specific by
                       construction. 512x512 a layer, pooled over pairs; this
                       is the cruder of the two approximations and is the first
                       thing to refine if the result is close.

714 MB a split, so `ORNITH_GPTQ_SPLIT=4` costs 2.9 GB and gives the four draws
Phase 11 established as the minimum for measuring anything here.

The streaming, the sanitize trap and the `+1.0` norm shift all live in
`parent_stream.py`.
"""

import json
import os
import pathlib
import sys
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.switch_layers import SwitchLinear

from dump_gptq_stats import build_corpus
from parent_stream import Parent, gib, mem

OUT = pathlib.Path(os.environ.get("ORNITH_GPTQ_OUT", "../refs/gptq-experts"))
PARENT = pathlib.Path(os.environ.get(
    "ORNITH_PARENT",
    str(pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B")))
SEQ = int(os.environ.get("ORNITH_GPTQ_SEQ", "1024"))
MAX_SAMPLES = int(os.environ.get("ORNITH_GPTQ_MAX_SAMPLES", "0"))
SPLIT = int(os.environ.get("ORNITH_GPTQ_SPLIT", "1"))
CHECK_LAYERS, CHECK_SAMPLES = 2, 8
VERIFY_CALLS = 4

RECORD = {}     # id(SwitchLinear) -> [ {"H", "n"} ] * SPLIT
WITNESS = {}    # id(SwitchLinear) -> (layer, "up" | "gate")
SEEN = {}
CUR = [0]


def record(sts, x):
    f = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    st = sts[CUR[0]]
    st["H"] = st["H"] + (f.T @ f)
    st["n"] += f.shape[0]


def patch():
    orig = SwitchLinear.__call__

    def patched(self, x, indices, *a, **k):
        sts = RECORD.get(id(self))
        if sts is not None:
            record(sts, x)
        w = WITNESS.get(id(self))
        if w is not None:
            seen = SEEN.setdefault(w[0], {"up": [], "gate": []})
            if len(seen[w[1]]) < VERIFY_CALLS:
                seen[w[1]].append(mx.array(x.reshape(-1, x.shape[-1])[:8]))
        return orig(self, x, indices, *a, **k)

    SwitchLinear.__call__ = patched


def check_shared_inputs():
    """`gate_proj` and `up_proj` are handed the same tensor by `SwitchGLU`, so
    one Hessian serves both. Checked bit for bit, like `in_proj_z` before it."""
    assert SEEN, "no witnesses recorded"
    for layer, seen in sorted(SEEN.items()):
        a, b = seen["up"], seen["gate"]
        assert len(a) == len(b) >= 1, f"layer {layer}: {len(a)} up / {len(b)} gate"
        for i, (x, y) in enumerate(zip(a, b)):
            d = float(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max())
            assert d == 0.0, f"layer {layer}: call {i} differs by {d}"
    n = len(next(iter(SEEN.values()))["up"])
    print(f"shared-input check: {len(SEEN)} layers, {n} calls each, max_abs == 0.0")


def main():
    check = "--check" in sys.argv
    texts, cats = build_corpus()
    if MAX_SAMPLES or check:
        texts = texts[:(CHECK_SAMPLES if check else MAX_SAMPLES)]
    print(f"corpus: {len(texts)} samples")

    src = Parent(PARENT)
    patch()
    model, tokenizer = load(str(PARENT), lazy=True)
    lm, inner = model.language_model, model.language_model.model
    layers = inner.layers
    print(f"structure built, {len(layers)} layers, {mem()}")
    print(f"norm shift check: layer 0 mean |gain| {src.check_norm_shift():.4f}")

    # One accumulator per (layer, input space). `up_proj` owns the 2048 one that
    # `gate_proj` shares; `down_proj` owns the 512 one.
    owner = {}
    for i, layer in enumerate(layers):
        sm = layer.mlp.switch_mlp
        for mod, dim, tag in ((sm.up_proj, sm.up_proj.weight.shape[-1], "hidden"),
                              (sm.down_proj, sm.down_proj.weight.shape[-1], "inter")):
            RECORD[id(mod)] = [{"H": mx.zeros((dim, dim), mx.float32), "n": 0}
                               for _ in range(SPLIT)]
            owner[id(mod)] = (i, tag, dim)
        WITNESS[id(sm.up_proj)] = (i, "up")
        WITNESS[id(sm.gate_proj)] = (i, "gate")
    per = sum(sts[0]["H"].size for sts in RECORD.values())
    print(f"accumulating {len(RECORD)} Hessians x {SPLIT} split(s) "
          f"({per * 4 * SPLIT / 2**30:.2f} GB)")

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
            c = ArraysCache(size=2) if layer.is_linear else KVCache()
            mask = (create_ssm_mask(h[s], c) if layer.is_linear
                    else create_attention_mask(h[s], c))
            CUR[0] = s % SPLIT
            h[s] = layer(h[s], mask=mask, cache=c)
            if s % 32 == 0 or s == len(seqs) - 1:
                mx.eval(h[max(0, s - 31):s + 1],
                        [st["H"] for sts in RECORD.values() for st in sts])
        mx.eval(h, [st["H"] for sts in RECORD.values() for st in sts])
        layer.update(tree_unflatten(list(src.layer(i).items())))
        mx.clear_cache()
        print(f"  layer {i:>2}  {mem()}  {time.time() - t0:7.1f}s", flush=True)

    check_shared_inputs()
    if check:
        for mid, (li, tag, dim) in owner.items():
            if li == 0:
                st = RECORD[mid][0]
                print(f"  layer 0 {tag:<7} dim {dim:>5} rows {st['n']:>9}")
        print(f"\nCHECK ok, peak {gib(mx.get_peak_memory()):.2f} GiB")
        return

    for k in range(SPLIT):
        out = OUT if SPLIT == 1 else OUT.parent / f"{OUT.name}-q{k}"
        (out / "hessians").mkdir(parents=True, exist_ok=True)
        man = {"model_dir": str(PARENT), "streamed": True, "experts": True,
               "split": f"{k} of {SPLIT}", "tokens": ntok, "seq": SEQ,
               "categories": cats, "modules": {}}
        for mid, sts in RECORD.items():
            li, tag, dim = owner[mid]
            name = f"layer{li}.{tag}"
            A = np.array(sts[k]["H"], copy=False)
            d = np.diag(A)
            man["modules"][name] = {"n": sts[k]["n"], "dim": int(dim),
                                    "dead": int((d == 0).sum()),
                                    "diag_mean": float(d.mean())}
            np.save(out / "hessians" / f"{name}.npy", A)
        (out / "calib.json").write_text(json.dumps(man, indent=1))
        print(f"wrote {out}")
    print(f"\ndone in {time.time() - t0:.0f}s, peak {gib(mx.get_peak_memory()):.2f} GiB")


if __name__ == "__main__":
    main()
