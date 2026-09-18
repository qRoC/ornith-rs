"""Capture the wide probe set's next-token distributions from one artifact.

`REBUILD.md` Phase 5, and the companion to `dump_artifact_baseline.py`. That
one captures the three things that could not be recomputed once the published
artifact was deleted; this one captures the thing Phase 4 turned out not to
have -- an instrument with enough contexts that a maximum is not a single
sample. `probe_set.py` says which contexts and why.

    cd tools && ORNITH_PROBES_OUT=../refs/artifact/<capture> \
      PYTHONPATH=../../vendor/mlx-lm python3 dump_probes.py <artifact>

Writes `probes.bin` -- [n_probes, vocab] float32 log-probabilities, the same
layout `logprobs.bin` uses -- and `probes.json` beside it. It does not touch
`baseline.json` or `logprobs.bin`, so a capture directory that already holds a
Phase 4 result keeps it, and Phase 2's bit-identity check is unaffected.

Two details that matter for comparability:

  chunked prefill   a long probe is fed through a prompt cache in CHUNK-sized
                    pieces and only the last token is forwarded alone, so the
                    logits that materialise are one row and not eight hundred.
                    Chunking changes reduction shapes, so it must be the same
                    for every artifact -- it is, because they all run this
                    file, and every probe takes the same path regardless of
                    length.
  entropy and top-1 written per probe. A probe whose control entropy is high
                    is not measuring the quantization, it is measuring that
                    the question was open, and the comparison tool drops it
                    from the headline numbers rather than letting it dilute a
                    median.
"""

import json
import os
import pathlib
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from probe_set import PROBES

OUT = pathlib.Path(os.environ.get("ORNITH_PROBES_OUT", "../refs/artifact/new"))
CHUNK = 128


def last_logprobs(model, ids):
    """Log-probabilities of the token after `ids`, without materialising the rest."""
    prefix, last = ids[:-1], ids[-1:]
    cache = make_prompt_cache(model)
    for i in range(0, len(prefix), CHUNK):
        model(mx.array([prefix[i : i + CHUNK]]), cache=cache)
        mx.clear_cache()
    logits = model(mx.array([last]), cache=cache)[0][-1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits)
    mx.eval(lp)
    mx.clear_cache()
    return lp


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if model_dir is None:
        model_dir = os.environ.get("ORNITH_MODEL") or str(
            pathlib.Path.home()
            / ".mtplx/models/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(model_dir)

    rows, meta = [], []
    for n, (group, pid, text) in enumerate(PROBES):
        ids = tokenizer.encode(text, add_special_tokens=False)
        lp = last_logprobs(model, ids)
        rows.append(lp)
        p = mx.exp(lp)
        top = int(mx.argmax(lp).item())
        entropy = float(-(p * lp).sum().item())
        meta.append({
            "group": group,
            "id": pid,
            "text": text,
            "tokens": len(ids),
            "entropy": entropy,
            "top1_p": float(p[top].item()),
            "top1_id": top,
            "top1_text": tokenizer.decode([top]),
        })
        print(f"{n:>3} {group:<10} {pid:<18} {len(ids):>4} tok  "
              f"H {entropy:5.3f}  p1 {meta[-1]['top1_p']:.3f}  "
              f"-> {meta[-1]['top1_text']!r}")

    dist = mx.stack(rows)
    (OUT / "probes.bin").write_bytes(
        np.array(dist.astype(mx.float32), copy=False).tobytes()
    )
    (OUT / "probes.json").write_text(json.dumps({
        "model_dir": str(model_dir),
        "probes": {
            "file": "probes.bin",
            "dtype": "float32",
            "shape": list(dist.shape),
            "chunk": CHUNK,
        },
        "set": meta,
    }, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}/probes.json and probes.bin {tuple(dist.shape)}")


main()
