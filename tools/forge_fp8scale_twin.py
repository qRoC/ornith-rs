"""Forge a stand-in for a 4.50-bit affine format, in the 5.00-bit layout.

    PYTHONPATH=../vendor/mlx-lm python3 tools/forge_fp8scale_twin.py

`REBUILD.md` Phase 4 settled mlx's four modes and `weight_error.py --schemes`
found the best thing left at 4.50 bits is not one of them: **affine int4 g32
with the scale and bias stored as E4M3 instead of bfloat16**, which is
8.1-8.6% weight error against the body's current 7.8-8.3% at 5.00 bits. nvfp4
at the same 4.50 bits is 9.4-9.5%.

That format has no mlx kernel, so measuring it would normally mean writing
one first -- and this project's rule is to measure before building. The way
round it is arithmetic rather than engineering:

**An E4M3 value times a power of two is exactly representable in bfloat16.**
E4M3 keeps four significant bits and bfloat16 keeps eight, so a scale and
bias cut to the candidate format can be *written out* in the ordinary affine
layout with no loss at all. The result is a normal `affine int4 g32` artifact
that reconstructs, bit for bit, what the 4.50-bit format would reconstruct --
and unmodified mlx-lm and the unmodified port both run it.

What it measures: perplexity, KL, and every golden test. What it does not
measure: bytes, which are arithmetic (4 + 8/32 + 8/32 = 4.50 against
4 + 16/32 + 16/32 = 5.00, so -10% of the 4-bit side and -7.3% of a decode
step) and are not in question. The twin is 5.00 bits on disk by construction;
it is a quality experiment, not a speed one.

The quantizer is `aq.quantize`, a port of mlx's `affine_quantize` kernel
(quantized.h) which reproduces `mx.quantize` **bit for bit** with the
encoding left alone -- that equality is the check that the substitution is
the only thing being measured. Its two traps: the kernel is float32
throughout (a stray float64 moves 1.3% of the codes) and Metal's `round` goes
half away from zero where numpy's goes to even.

Source is the bf16 parent, one tensor at a time -- `mx.load` is lazy, so the
67 GiB that cannot be loaded as a model can be read as tensors. The base is
`int4-head-linattn`, already aligned, so `align_rebuild.py` is *not* needed
afterwards; everything it fixes is inherited.
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
OUT = MODELS / "ornith-1.5-35b-a3b-fp8scale-twin"

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

            q, s, b = aq.quantize(w, GROUP_SIZE, BITS, encode=aq.fp8_encode)
            # The point of the whole exercise: what the candidate format
            # stores must survive bfloat16 exactly, or the twin is not one.
            assert np.array_equal(aq.bf16(s), s), f"{mod}: scale not bf16-exact"
            assert np.array_equal(aq.bf16(b), b), f"{mod}: bias not bf16-exact"

            gshape = list(w.shape[:-1]) + [w.shape[-1] // GROUP_SIZE]
            wshape = list(w.shape[:-1]) + [w.shape[-1] * BITS // 32]
            under += int((np.abs(s) <= aq.E4M3_MIN * aq.po2_shift(s) * 1.001).sum())
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
          f"{under} groups hit the E4M3 underflow floor")
    print(f"wrote {OUT}")
    print("\nalign_rebuild.py is NOT needed -- the base was already aligned. Next:")
    print(f"  python3 tools/check_rebuild.py {OUT}")
    print(f"  cd tools && ORNITH_BASELINE_OUT=../refs/artifact/fp8scale-twin \\")
    print(f"    PYTHONPATH=../../vendor/mlx-lm python3 dump_artifact_baseline.py {OUT}")
    print(f"  python3 tools/compare_baselines.py refs/artifact/fp8scale-twin")


main()
