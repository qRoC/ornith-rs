"""mlx's per-token NLL over the same corpus, as the oracle for `bin/ppl`.

`bin/ppl` exists because mlx cannot load a k-quant artifact and the pair that
needs settling -- `q6k-full` against `q6k-perexpert` -- is two of them
(`REBUILD.md` Phase 44). What makes a port-side instrument usable is not an
argument, it is this file: on an **affine** artifact both sides run, so both
write the same 6620 numbers and they are compared element by element.

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_ppl.py <model dir> \
        [<the port's capture to check against>]

The definition is `dump_artifact_baseline.py`'s, unchanged: chunks of 256
tokens through one cache, logits cast to float32 before the reduction,
`logsumexp` over the last axis, and the next token scored for every position
that has one. The corpus's last token is unscored.

Two summaries of one set of numbers, printed side by side because they are
*not* the same arithmetic and neither is the measurement: mlx sums each chunk
in float32 and accumulates the chunk sums in a Python float; the port sums the
whole corpus in f64 in sequence order. `nll.bin` is what the two agree on or
do not.
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = pathlib.Path(
    os.environ.get("ORNITH_PPL_CORPUS", ROOT / "refs/artifact/ppl_corpus.txt")
)
CHUNK = int(os.environ.get("ORNITH_PPL_CHUNK", "256"))


def per_token_nll(model, ids):
    """`(nll per scored token, mlx's own running mean)`, chunked as the baseline is."""
    cache = make_prompt_cache(model)
    out, total, n = [], 0.0, 0
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
        out.append(-np.array(lp, copy=True))
        total += -lp.sum().item()
        n += m
        mx.clear_cache()
    return np.concatenate(out).astype(np.float32), total / n


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_dir = pathlib.Path(sys.argv[1]).expanduser()
    out = pathlib.Path(os.environ.get("ORNITH_PPL_OUT", ROOT / "refs/artifact/mlx-ppl"))
    out.mkdir(parents=True, exist_ok=True)

    text = CORPUS.read_text()
    model, tokenizer = load(str(model_dir))
    ids = tokenizer.encode(text, add_special_tokens=False)
    nll, mlx_mean = per_token_nll(model, ids)

    mean = float(np.float64(nll).sum() / len(nll))     # the port's summary
    (out / "nll.bin").write_bytes(nll.tobytes())
    (out / "ppl.json").write_text(json.dumps({
        "model_dir": str(model_dir),
        "source": "tools/dump_ppl.py (mlx)",
        "corpus": {"file": CORPUS.name, "bytes": len(text.encode()), "tokens": len(ids)},
        "chunk": CHUNK,
        "tokens_scored": len(nll),
        "mean_nll": mean,
        "ppl": math.exp(mean),
        "mean_nll_mlx_accumulation": mlx_mean,
        "ppl_mlx_accumulation": math.exp(mlx_mean),
        "nll": {"file": "nll.bin", "dtype": "float32", "shape": [len(nll)]},
    }, indent=1, ensure_ascii=False))
    print(f"mlx: perplexity {math.exp(mean):.4f} over {len(nll)} tokens "
          f"(mean nll {mean:.6f}; mlx's own accumulation {mlx_mean:.6f})")

    if len(sys.argv) < 3:
        return 0

    # The check the port's instrument has to pass before it is allowed to
    # judge an artifact mlx cannot load: 6620 numbers, not two means.
    port = pathlib.Path(sys.argv[2]).expanduser()
    theirs = np.fromfile(port / "nll.bin", dtype=np.float32)
    pmeta = json.loads((port / "ppl.json").read_text())
    if theirs.shape != nll.shape:
        print(f"  SHAPE {theirs.shape} against {nll.shape} -- not the same corpus")
        return 1
    d = np.abs(theirs.astype(np.float64) - nll.astype(np.float64))
    exact = int((theirs == nll).sum())
    print(f"  against {port}: {exact} of {len(nll)} bit-identical, "
          f"max_abs {d.max():.6e} (at {int(d.argmax())}), mean_abs {d.mean():.6e}")
    print(f"  mean nll  port {pmeta['mean_nll']:.9f}  mlx {mean:.9f}  "
          f"delta {pmeta['mean_nll'] - mean:+.3e}")
    print(f"  ppl       port {pmeta['ppl']:.6f}  mlx {math.exp(mean):.6f}")
    print(f"  port instrument: {pmeta.get('instrument', 'not recorded')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
