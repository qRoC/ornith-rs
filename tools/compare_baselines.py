"""Compare a rebuilt artifact's baseline against the Phase 0 one.

    python3 tools/compare_baselines.py [<new baseline dir>]

`REBUILD.md` Phase 2. The published artifact is gone and a rebuild on this
toolchain cannot be byte-identical to it -- MTPLX 2.9.1 sharded the body into
17 files and 2.11.3 packs it into 5, among other renames. None of that touches
a weight, so the question "is this the same model" has to be asked of the
numbers instead, and `baseline/` holds the only numbers that can still ask it.

What each line means:

  perplexity   averages over 6620 tokens and hides a lot. Equal to the last
               decimal means the arithmetic is the same arithmetic.
  logprobs     8 full next-token distributions, float32, 248320 wide. This is
               the sensitive one: if the rebuild is the same weights, every
               one of the 1986560 numbers is bit-identical, because MLX is
               deterministic and file layout is not an input to a matmul. A
               max_abs of exactly 0.0 is the byte-identity check that survived
               the artifact's deletion.
  anchors      greedy continuations. Identical weights give identical tokens;
               a requantization diverges, and how late it diverges is the
               only cross-format signal there is.

So in Phase 2 the bar is exact equality on all three. In Phase 4 it is not:
there the point is to move the weights, and these become the measurement
rather than the check.
"""

import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OLD = ROOT / "baseline"
NEW = pathlib.Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ROOT / "refs/artifact/new"


def load(d, name="baseline.json"):
    return json.loads((d / name).read_text())


def main():
    if not (NEW / "baseline.json").exists():
        sys.exit(f"no capture at {NEW}\n"
                 "  cd tools && ORNITH_BASELINE_OUT=../refs/artifact/new \\\n"
                 "    PYTHONPATH=../../vendor/mlx-lm python3 dump_artifact_baseline.py <artifact>")
    old, new = load(OLD), load(NEW)
    print(f"old: {old['model_dir']}\nnew: {new['model_dir']}\n")

    po, pn = old["perplexity"], new["perplexity"]
    print("perplexity")
    same_n = po["tokens_scored"] == pn["tokens_scored"]
    print(f"  corpus        {po['corpus']}, {po['tokens_scored']} tokens"
          + ("" if same_n else f" -- the new capture scored {pn['tokens_scored']}"))
    print(f"  mean NLL      {po['mean_nll']:.12f}  ->  {pn['mean_nll']:.12f}"
          f"   delta {pn['mean_nll'] - po['mean_nll']:+.3e}")
    print(f"  perplexity    {po['ppl']:.6f}  ->  {pn['ppl']:.6f}"
          f"   delta {pn['ppl'] - po['ppl']:+.3e}")
    print(f"  {'EXACT' if pn['mean_nll'] == po['mean_nll'] else 'moved'}")

    shape = tuple(old["logprobs"]["shape"])
    a = np.fromfile(OLD / "logprobs.bin", dtype=np.float32).reshape(shape)
    b = np.fromfile(NEW / "logprobs.bin", dtype=np.float32).reshape(shape)
    print(f"\nlogprobs {shape}")
    d = np.abs(a - b)
    print(f"  max_abs       {d.max():.6e}")
    print(f"  identical     {int((a == b).sum())} of {a.size}")
    # KL(old || new) in nats, per context, from log-probabilities.
    kl = (np.exp(a) * (a - b)).sum(axis=-1)
    print(f"  KL(old||new)  max {kl.max():.6e} nats over {shape[0]} contexts")
    top_old, top_new = a.argmax(-1), b.argmax(-1)
    print(f"  top-1 agrees  {int((top_old == top_new).sum())} of {shape[0]}")
    print(f"  {'EXACT -- same weights' if d.max() == 0.0 else 'MOVED'}")

    print("\nanchors")
    for o, n in zip(old["anchors"], new["anchors"]):
        oi, ni = o["ids"], n["ids"]
        k = 0
        while k < min(len(oi), len(ni)) and oi[k] == ni[k]:
            k += 1
        verdict = "identical" if oi == ni else f"diverges at token {k}"
        print(f"  {verdict:<22} {o['prompt'][:52]}")

    mo, mn = old.get("mtp_profile"), new.get("mtp_profile")
    if mo and mn:
        print("\nMTP acceptance")
        print(f"  old {json.dumps(mo)[:160]}")
        print(f"  new {json.dumps(mn)[:160]}")
    elif mo:
        print("\nMTP acceptance -- the new capture has none "
              "(dump_artifact_baseline.py does not write it; see baseline/README.md)")
    return 0


sys.exit(main())
