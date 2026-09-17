"""Capture what cannot be recomputed once the current artifact is deleted.

`REBUILD.md` Phase 0. Every test in this port compares against MLX on the
same artifact, which validates the port and not the artifact -- so a subtly
wrong rebuild would have MLX and this port agreeing on the wrong thing with
the whole suite green. These are the measurements that would catch it, and
they exist only while the artifact does.

Three things, in one model load:

  perplexity   over a fixed corpus kept beside the number, because a
               perplexity without its text is not a measurement. This is what
               a requantization is judged against.
  logprobs     the full next-token distribution for a handful of fixed
               contexts, saved float32. Perplexity averages; a KL against
               these says how far a new artifact's *distribution* moved, which
               is the more sensitive signal.
  anchors      greedy continuations as token ids. They will not match a
               requantized rebuild -- different weights, different tokens --
               so they answer "is it still the same model" by how late the
               streams diverge, not by equality.

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_artifact_baseline.py [<model dir>]

The output directory is ../refs/artifact/old; ORNITH_BASELINE_OUT overrides it.
"""

import json
import math
import os
import pathlib
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# Phase 0 wrote "old"; point ORNITH_BASELINE_OUT at "new" to capture the
# rebuild and get the other side of the comparison.
OUT = pathlib.Path(os.environ.get("ORNITH_BASELINE_OUT", "../refs/artifact/old"))
CORPUS = pathlib.Path("../refs/artifact/ppl_corpus.txt")
CHUNK = 256

# Contexts whose full next-token distribution gets saved, and the prompts whose
# greedy continuation gets saved. Fixed forever: changing them invalidates the
# comparison they exist for.
CONTEXTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "In a memory-bound decode step the bottleneck is",
    "1, 1, 2, 3, 5, 8, 13,",
    "The Rust borrow checker prevents",
    "SELECT name FROM users WHERE",
    "Дважды два равно",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n",
]
ANCHORS = [
    "Write a Python function that merges two sorted lists.",
    "Explain how a memory-bound decode step differs from a compute-bound one.",
    "Name the first eight prime numbers.",
    "In one sentence, what is a neutron star?",
]
ANCHOR_TOKENS = 96


def perplexity(model, ids):
    """Mean NLL over the corpus, chunked so the logits never materialise whole."""
    cache = make_prompt_cache(model)
    total, n = 0.0, 0
    for i in range(0, len(ids) - 1, CHUNK):
        chunk = ids[i : i + CHUNK]
        logits = model(mx.array([chunk]), cache=cache)[0]
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = ids[i + 1 : i + 1 + len(chunk)]
        m = len(targets)
        if m == 0:
            break
        lp = mx.take_along_axis(
            logprobs[:m], mx.array(targets)[:, None], axis=-1
        ).squeeze(-1)
        total += -lp.sum().item()
        n += m
        mx.clear_cache()
    return total / n, n


def save(report):
    (OUT / "baseline.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if model_dir is None:
        model_dir = os.environ.get("ORNITH_MODEL") or str(
            pathlib.Path.home()
            / ".mtplx/models/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(model_dir)
    report = {"model_dir": str(model_dir)}

    text = CORPUS.read_text()
    ids = tokenizer.encode(text, add_special_tokens=False)
    nll, n = perplexity(model, ids)
    report["perplexity"] = {
        "corpus": CORPUS.name,
        "corpus_bytes": len(text.encode()),
        "tokens_scored": n,
        "mean_nll": nll,
        "ppl": math.exp(nll),
    }
    print(f"perplexity {math.exp(nll):.4f} over {n} tokens (mean nll {nll:.6f})")
    save(report)   # write after each stage: a crash late must not lose the rest

    rows = []
    for ctx in CONTEXTS:
        cid = tokenizer.encode(ctx, add_special_tokens=False)
        logits = model(mx.array([cid]))[0][-1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        rows.append(lp)
        mx.clear_cache()
    dist = mx.stack(rows)
    (OUT / "logprobs.bin").write_bytes(
        np.array(dist.astype(mx.float32), copy=False).tobytes()
    )
    report["logprobs"] = {
        "file": "logprobs.bin",
        "dtype": "float32",
        "shape": list(dist.shape),
        "contexts": CONTEXTS,
    }
    print(f"logprobs {tuple(dist.shape)} saved")
    save(report)

    anchors = []
    for p in ANCHORS:
        pid = tokenizer.encode(p, add_special_tokens=False)
        cache = make_prompt_cache(model)
        out, y = [], mx.array([pid])
        for _ in range(ANCHOR_TOKENS):
            logits = model(y, cache=cache)[0][-1]
            t = int(mx.argmax(logits).item())
            out.append(t)
            if t == getattr(tokenizer, "eos_token_id", None):
                break
            y = mx.array([[t]])
        anchors.append({"prompt": p, "ids": out, "text": tokenizer.decode(out)})
        print(f"anchor {len(out):>3} tokens: {p[:48]}")
        mx.clear_cache()
    report["anchors"] = anchors

    save(report)
    print(f"\nwrote {OUT}/baseline.json and logprobs.bin")


main()
