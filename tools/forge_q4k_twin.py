"""Forge a stand-in for a **Q4_K-shaped** 4.50-bit format, in the 5.00-bit layout.

    PYTHONPATH=../vendor/mlx-lm python3 tools/forge_q4k_twin.py

`REBUILD.md` Phase 14. Phase 4 closed mlx's four modes; Phase 13 priced out
IDEAS #1 because its win lives in a group size mlx will not run. This is the
remaining structure, and it is the one that fits: llama.cpp's k-quants store a
sub-block's scale and min as **6-bit integers** against two super-scales per
256 weights. An integer has **no subnormal regime**, which is exactly what cost
nvfp4 its quality here -- 99.6% of this model's group scales sit below E4M3's
smallest normal, so the format that was supposed to be free measured 10.4-13.1%
instead of 9.5%.

Priced first, as Phase 13 was, on `weight_error.py --schemes`:

    affine g32, bf16 scale+bias -- the body   5.00   8.03  7.81  7.90  8.25  7.94
    **Q4_K-shaped**                           4.50   8.07  7.86  7.93  8.30  7.98
    fp8-scale twin                            4.50   8.31  8.10  8.18  8.56  8.22
    Lloyd-Max g32, bf16 scale                 4.50   8.42  8.20  8.28  8.57  8.98
    nvfp4 g16, + global scale                 4.50   9.45  9.53  9.50  9.44  9.48

**Within 0.05 points of the 5.00-bit body at half a bit less**, and the best of
its rate class by a clear margin. Whether that reaches the distribution is the
open question, and weight error is a chooser this rebuild has falsified twice
(Phase 4: lower error, higher KL; Phase 7: error up 13%, KL down 23%). So this
is a build to find out, not a build because the table says so.

**It is standable-in for the same reason the fp8-scale twin was**, and by the
same arithmetic: the super-scales are **powers of two**, so a 6-bit integer
times one is exactly representable in bfloat16. The sub-block stays 32 wide --
mlx's own group size, unlike IDEAS #1's 16 -- so the result is an ordinary
`affine int4 g32` artifact that unmodified mlx-lm and the unmodified port both
run, and `max_abs == 0.0` means what it has always meant.

What it measures: KL, perplexity, every golden test. What it does not measure:
bytes, which are arithmetic -- 4 + (7 + 7)/32 + 16/256 = 4.50 against the
layout's 5.00, so -10% of the 4-bit side and -7.3% of a decode step. Both
fields are magnitude-plus-sign because `aq.quantize`'s scale carries a sign.
The twin is 5.00 bits on disk by construction; this is a quality experiment.

`aq.q4k_encode` is the encoding; `aq.quantize` with `encode=None` reproduces
`mx.quantize` bit for bit, which is the check that the substitution is the only
thing that moved.
"""

import json
import pathlib
import shutil
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aq  # noqa: E402

MODELS = pathlib.Path.home() / ".mtplx/models"
PARENT = MODELS / "ornith-ai--Ornith-1.5-35B-A3B"
BASE = MODELS / "ornith-1.5-35b-a3b-int4-head-linattn"
OUT = MODELS / "ornith-1.5-35b-a3b-q4k-twin"

GROUP_SIZE, BITS = 32, 4


def parent_of(mod):
    """Artifact module path -> (parent tensor name, which half or None).

    The parent is a HF checkpoint and the artifact is mlx-lm's sanitized form:
    the layer prefix is swapped, and the 256 experts' `gate_proj`/`up_proj`
    are the two halves of one fused `mlp.experts.gate_up_proj`. Verified by
    reproducing `mx.quantize` bit for bit through this mapping.
    """
    if mod == "language_model.lm_head":
        return "lm_head.weight", None
    assert mod.startswith("language_model.model.layers."), mod
    n, rest = mod[len("language_model.model.layers.") :].split(".", 1)
    base = f"model.language_model.layers.{n}."
    if rest.startswith("linear_attn.") or rest.startswith("mlp.shared_expert."):
        return base + rest + ".weight", None
    if rest == "mlp.switch_mlp.gate_proj":
        return base + "mlp.experts.gate_up_proj", "lo"
    if rest == "mlp.switch_mlp.up_proj":
        return base + "mlp.experts.gate_up_proj", "hi"
    if rest == "mlp.switch_mlp.down_proj":
        return base + "mlp.experts.down_proj", None
    raise AssertionError(mod)


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; remove it or pick another name")
    acfg = json.loads((BASE / "config.json").read_text())["quantization"]
    aidx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    pidx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]

    mods = sorted({k.rsplit(".", 1)[0] for k in aidx if k.endswith(".scales")})
    four = [m for m in mods if (acfg.get(m, acfg)).get("bits", acfg["bits"]) == BITS]
    print(f"{len(mods)} quantized modules, {len(four)} at {BITS} bits -> re-encoding those")

    OUT.mkdir(parents=True)
    shards = sorted({v for k, v in aidx.items()})
    touched = {aidx[m + ".weight"] for m in four}
    for f in sorted(BASE.iterdir()):
        if f.is_file() and f.name not in touched:
            shutil.copy2(f, OUT / f.name)
    print(f"copied {len(list(OUT.iterdir()))} files unchanged; "
          f"rewriting {len(touched)} of {len(shards)} shards")

    by_shard = {}
    for m in four:
        by_shard.setdefault(aidx[m + ".weight"], []).append(m)

    t0 = time.time()
    done = under = 0
    for shard in sorted(touched):
        tensors = dict(mx.load(str(BASE / shard)))
        for mod in by_shard[shard]:
            pname, half = parent_of(mod)
            t = mx.load(str(PARENT / pidx[pname]))[pname]
            if half:
                h = t.shape[1] // 2
                t = t[:, :h, :] if half == "lo" else t[:, h:, :]
            mx.eval(t)
            w = np.array(t.astype(mx.float32))
            del t

            q, s, b = aq.quantize(w, GROUP_SIZE, BITS, encode=aq.q4k_encode)
            # The point of the whole exercise: what the candidate format
            # stores must survive bfloat16 exactly, or the twin is not one.
            assert np.array_equal(aq.bf16(s), s), f"{mod}: scale not bf16-exact"
            assert np.array_equal(aq.bf16(b), b), f"{mod}: bias not bf16-exact"

            gshape = list(w.shape[:-1]) + [w.shape[-1] // GROUP_SIZE]
            wshape = list(w.shape[:-1]) + [w.shape[-1] * BITS // 32]
            under += int((s == 0).sum())   # an integer field cannot underflow; assert it
            tensors[mod + ".weight"] = mx.array(
                aq.pack4(q, wshape[-1]).reshape(wshape)
            )
            tensors[mod + ".scales"] = mx.array(s.reshape(gshape)).astype(mx.bfloat16)
            tensors[mod + ".biases"] = mx.array(b.reshape(gshape)).astype(mx.bfloat16)
            del w, q, s, b
            done += 1
            print(f"  [{done:3d}/{len(four)}] {mod}", flush=True)
        mx.eval(*tensors.values())
        mx.save_safetensors(str(OUT / shard), tensors, metadata={"format": "mlx"})
        del tensors
    print(f"\n{done} modules re-encoded in {time.time() - t0:.0f}s; "
          f"{under} groups underflowed to zero (must be 0)")
    print(f"wrote {OUT}")
    print("\nalign_rebuild.py is NOT needed -- the base was already aligned. Next:")
    print(f"  python3 tools/check_rebuild.py {OUT}")
    print(f"  cd tools && ORNITH_BASELINE_OUT=../refs/artifact/q4k-twin \\")
    print(f"    PYTHONPATH=../../vendor/mlx-lm python3 dump_artifact_baseline.py {OUT}")
    print(f"  python3 tools/compare_baselines.py refs/artifact/q4k-twin")


main()
