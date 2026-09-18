"""The probe set's distributions from the **bf16 parent**, one layer at a time.

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_parent_probes.py [--check]

`REBUILD.md` Phase 8. Every KL in this project is `KL(A || variant)` where the
control `A` is itself a quantized artifact -- 260 modules at 8 bits. The
unquantized parent has never been measured, and `recipes/README.md` has carried
the row as missing since Phase 4, for a stated reason: 67.0 GiB of weights
against a 51.8 GiB working set, so `mlx_lm.load` dies in the loader.

**That reason is about residency, not about the forward.** A layer only has to
be resident while it runs. Measured off the parent's safetensors headers:

    total            66.97 GiB, 1811 tensors, 40 layers
    heaviest layer    3.13 GiB   (mean 1.61)
    lm_head           0.95 GiB
    embed             0.95 GiB
    streamed peak    ~5.02 GiB against a 51.8 GiB working set

So the loop is inverted: instead of walking layers for one probe, walk probes
for one layer. All 64 probes are pushed through layer *i*, then layer *i* is
dropped and layer *i+1* is loaded. One pass reads 67 GiB from disk, once.

**Why it is worth a pass.** `refs/artifact/*/probes.bin` are log-probabilities,
[64, 248320] each, and eight of them are already on disk. Once the parent has
one, `KL(parent || variant)` is computable for **all eight retroactively** --
no rebuild, no re-run. One capture re-scores Phase 4, 5, 6 and 7 together. It
became urgent with Phase 7: GPTQ is the first variant that *fits toward the
parent*, while being scored against A, so the instrument and the method now
disagree about what "closer" means.

**The trap this file has to clear, and it is the one the project already
knows.** The parent is an HF checkpoint carrying `mtp.*` weights and a
torch-layout `conv1d`, so `qwen3_5.TextModel.sanitize` **adds 1.0 to the trunk
norm gains** -- true of a parent, false of a converted artifact. Streaming
means sanitizing per layer rather than once over the whole dict, so
`has_unsanitized_conv1d` is computed **globally, from the full key list**,
before any layer is touched, and asserted to be True. Getting this wrong is
the "double shift" that makes a model that looks loadable and is wrong.
`linear_attn.norm.weight` is deliberately *not* in `norm_keys`; that is
upstream's decision and it is reproduced, not corrected.

**Comparability.** `dump_probes.py` feeds a probe through a prompt cache in
128-token chunks and then forwards the last token alone, because chunking
changes reduction shapes and every artifact must take the same path. This file
reproduces that exactly, per probe, inside the per-layer loop: a probe's chunk
sequence is replayed against that layer's cache in order, so layer *i* sees the
same calls in the same order it would have seen them.

`--check` runs the first `CHECK_LAYERS` layers and reports what memory does
after each, so the assumption that a dropped layer is actually released is
tested before a full pass is spent on it.
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

from parent_stream import Parent, gib, mem
from probe_set import PROBES

OUT = pathlib.Path(os.environ.get("ORNITH_PROBES_OUT", "../refs/artifact/parent"))
PARENT = pathlib.Path(os.environ.get(
    "ORNITH_PARENT",
    str(pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B")))
CHUNK = 128                 # must match dump_probes.py
CHECK_LAYERS = 3

# The streaming machinery and the sanitize trap live in `parent_stream.py`, so
# that `dump_gptq_stats_parent.py` shares one implementation of both.

def main():
    check = "--check" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    src = Parent(PARENT)
    print(f"parent {PARENT.name}: {len(src.map)} tensors, "
          f"norm gains get +1.0 (conv1d is torch layout)")

    # The structure only. `lazy=True` leaves every array unevaluated, so this
    # builds a correctly-typed 40-layer model without materialising weights --
    # which is exactly the thing the loader is documented to die on.
    t0 = time.time()
    model, tokenizer = load(str(PARENT), lazy=True)
    lm, inner = model.language_model, model.language_model.model
    layers = inner.layers
    print(f"structure built in {time.time() - t0:.1f}s, {len(layers)} layers, {mem()}")

    probes = PROBES[:4] if check else PROBES
    ids = [tokenizer.encode(t, add_special_tokens=False) for _, _, t in probes]

    # Each probe's chunk sequence, exactly as dump_probes.py feeds it: the
    # prefix in CHUNK-sized pieces, then the final token on its own.
    chunks = []
    for seq in ids:
        prefix, last = seq[:-1], seq[-1:]
        cs = [prefix[i:i + CHUNK] for i in range(0, len(prefix), CHUNK)] + [last]
        chunks.append([mx.array([c]) for c in cs])

    emb = src.get("model.language_model.embed_tokens.")
    inner.update(tree_unflatten([("embed_tokens.weight",
                                  emb["language_model.model.embed_tokens.weight"])]))
    h = [[inner.embed_tokens(c) for c in cs] for cs in chunks]
    mx.eval([x for hs in h for x in hs])
    del emb
    inner.update(tree_unflatten([("embed_tokens.weight", src.raw(
        "model.language_model.embed_tokens.weight"))]))
    mx.clear_cache()
    print(f"embedded {len(probes)} probes, {mem()}")

    print(f"norm shift check: layer 0 mean |gain| {src.check_norm_shift():.4f}")

    t0 = time.time()
    n_layers = CHECK_LAYERS if check else len(layers)
    for i in range(n_layers):
        layer = layers[i]
        layer.update(tree_unflatten(list(src.layer(i).items())))
        # One cache entry per probe, for this layer only: a probe's state at
        # layer i is dead the moment layer i+1 starts.
        caches = [ArraysCache(size=2) if layer.is_linear else KVCache()
                  for _ in probes]
        for p in range(len(probes)):
            for c, x in enumerate(h[p]):
                mask = (create_ssm_mask(x, caches[p]) if layer.is_linear
                        else create_attention_mask(x, caches[p]))
                h[p][c] = layer(x, mask=mask, cache=caches[p])
            mx.eval(h[p])
        del caches
        # Release: overwrite with freshly loaded, *unevaluated* arrays.
        layer.update(tree_unflatten(list(src.layer(i).items())))
        mx.clear_cache()
        if check or (i + 1) % 5 == 0 or i + 1 == n_layers:
            print(f"  layer {i:>2}  {mem()}  {time.time() - t0:6.1f}s", flush=True)

    if check:
        print(f"\nCHECK: {CHECK_LAYERS} layers on {len(probes)} probes, "
              f"peak {gib(mx.get_peak_memory()):.2f} GiB. "
              f"Active memory must be flat across layers above.")
        return

    # `model.norm.weight` is in `norm_keys`, so `get` has already shifted it.
    norm_w = src.get("model.language_model.norm.")
    inner.update(tree_unflatten(
        [("norm.weight", norm_w["language_model.model.norm.weight"])]))
    lm.update(tree_unflatten([("lm_head.weight", src.raw("lm_head.weight"))]))
    print(f"norm + lm_head loaded, {mem()}")

    rows, meta = [], []
    for p, (group, pid, text) in enumerate(probes):
        logits = lm.lm_head(inner.norm(h[p][-1]))[0][-1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        mx.eval(lp)
        prob = mx.exp(lp)
        top = int(mx.argmax(lp).item())
        entropy = float(-(prob * lp).sum().item())
        rows.append(lp)
        meta.append({"group": group, "id": pid, "text": text,
                     "tokens": len(ids[p]), "entropy": entropy,
                     "top1_p": float(prob[top].item()), "top1_id": top,
                     "top1_text": tokenizer.decode([top])})
        print(f"{p:>3} {group:<10} {pid:<18} {len(ids[p]):>4} tok  "
              f"H {entropy:5.3f}  p1 {meta[-1]['top1_p']:.3f}  "
              f"-> {meta[-1]['top1_text']!r}", flush=True)
        mx.clear_cache()

    dist = mx.stack(rows)
    (OUT / "probes.bin").write_bytes(
        np.array(dist.astype(mx.float32), copy=False).tobytes())
    (OUT / "probes.json").write_text(json.dumps({
        "model_dir": str(PARENT),
        "streamed": {"layers": len(layers), "peak_gib": gib(mx.get_peak_memory()),
                     "norm_shift": src.shift_norms},
        "probes": {"file": "probes.bin", "dtype": "float32",
                   "shape": list(dist.shape), "chunk": CHUNK},
        "set": meta,
    }, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}/probes.json and probes.bin {tuple(dist.shape)} "
          f"in {time.time() - t0:.0f}s, peak {gib(mx.get_peak_memory()):.2f} GiB")


main()
