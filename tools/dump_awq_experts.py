"""Per-expert channel saliency for the routed experts' `down_proj`.

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_awq_experts.py [--check]

`REBUILD.md` Phase 22. Phase 12 closed GPTQ on the experts and named the
successor: AWQ on `down_proj`, whose fold is exact (`h_c = silu(gate)_c * up_c`
puts the nonlinearity only on the gate path, so scaling channel `c` of `h` is
dividing row `c` of `W_up`) and which does not need the anisotropic Hessian
GPTQ degenerated without.

What AWQ needs instead is **spread in the diagonal**, and the object that
decides it is per-expert. `dump_gptq_experts.py` could not have it -- the exact
per-expert Hessian is 172 GB, so it pools over (token, expert) pairs and shares
one 512x512 matrix across 256 experts, which Phase 12 listed as the honest
alternative explanation for its own null result. **A diagonal is not 172 GB.**
One expert's saliency is 512 numbers, so the whole model's is 40 x 256 x 512 --
21 MB, and the approximation Phase 12 had to make is simply not needed here.

Recorded per (layer, expert, channel), over the same corpus and the same
streamed parent `dump_gptq_experts.py` uses:

    n     rows routed to that expert
    s1    sum |h_c|   -- AWQ's own saliency statistic
    s2    sum h_c**2  -- the diagonal of that expert's own Hessian

Rows are paired with experts inside the hook. `SwitchGLU` sorts once
`indices.size >= 64`, which at any calibration sequence length it always is, and
sorting is what makes the pairing one row to one expert -- asserted rather than
assumed, because unsorted the same row feeds `k` experts and the pairing would
silently be wrong.
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
from parent_stream import Parent, mem

OUT = pathlib.Path(os.environ.get("ORNITH_AWQ_EXPERTS_OUT", "../refs/awq-experts"))
PARENT = pathlib.Path(os.environ.get(
    "ORNITH_PARENT",
    str(pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B")))
SEQ = int(os.environ.get("ORNITH_AWQ_SEQ", "1024"))
MAX_SAMPLES = int(os.environ.get("ORNITH_AWQ_MAX_SAMPLES", "0"))
CHECK_LAYERS, CHECK_SAMPLES = 2, 8

RECORD = {}     # id(SwitchLinear) -> {"n", "s1", "s2"}


def record(st, x, indices):
    f = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    idx = indices.reshape(-1)
    assert f.shape[0] == idx.shape[0], (
        f"{f.shape[0]} rows against {idx.shape[0]} expert ids -- SwitchGLU did "
        "not sort, so a row does not belong to one expert"
    )
    st["s1"] = st["s1"].at[idx].add(mx.abs(f))
    st["s2"] = st["s2"].at[idx].add(f * f)
    st["n"] = st["n"].at[idx].add(mx.ones((f.shape[0],), mx.float32))


def patch():
    orig = SwitchLinear.__call__

    def patched(self, x, indices, *a, **k):
        st = RECORD.get(id(self))
        if st is not None:
            record(st, x, indices)
        return orig(self, x, indices, *a, **k)

    SwitchLinear.__call__ = patched


def main():
    check = "--check" in sys.argv
    texts, cats = build_corpus()
    if MAX_SAMPLES or check:
        texts = texts[:(CHECK_SAMPLES if check else MAX_SAMPLES)]
    print(f"corpus: {len(texts)} samples")

    src = Parent(PARENT)
    patch()
    model, tokenizer = load(str(PARENT), lazy=True)
    inner = model.language_model.model
    layers = inner.layers
    print(f"structure built, {len(layers)} layers, {mem()}")

    owner = {}
    for i, layer in enumerate(layers):
        dp = layer.mlp.switch_mlp.down_proj
        e, d = dp.weight.shape[0], dp.weight.shape[-1]
        RECORD[id(dp)] = {"n": mx.zeros((e,), mx.float32),
                          "s1": mx.zeros((e, d), mx.float32),
                          "s2": mx.zeros((e, d), mx.float32)}
        owner[id(dp)] = (i, e, d)
    print(f"accumulating {len(RECORD)} x [{e}, {d}] saliencies "
          f"({len(RECORD) * e * d * 3 * 4 / 2**20:.0f} MB)")

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
            h[s] = layer(h[s], mask=mask, cache=c)
            if s % 32 == 0 or s == len(seqs) - 1:
                mx.eval(h[max(0, s - 31):s + 1],
                        [v for st in RECORD.values() for v in st.values()])
        mx.eval(h, [v for st in RECORD.values() for v in st.values()])
        layer.update(tree_unflatten(list(src.layer(i).items())))
        mx.clear_cache()
        print(f"  layer {i:>2}  {mem()}  {time.time() - t0:7.1f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for mid, (li, e, d) in owner.items():
        if li >= n_layers:
            continue
        st = RECORD[mid]
        np.savez(OUT / f"layer{li}.npz",
                 n=np.array(st["n"], copy=False),
                 s1=np.array(st["s1"], copy=False),
                 s2=np.array(st["s2"], copy=False))
        written += 1
    (OUT / "manifest.json").write_text(json.dumps(
        {"parent": str(PARENT), "seq": SEQ, "samples": len(seqs),
         "tokens": ntok, "categories": cats, "layers": written}, indent=2))
    print(f"wrote {written} layers to {OUT}")

    st = RECORD[next(k for k, v in owner.items() if v[0] == 0)]
    n = np.array(st["n"], copy=False)
    print(f"layer 0: rows per expert min {n.min():.0f} median "
          f"{np.median(n):.0f} max {n.max():.0f}")


if __name__ == "__main__":
    main()
