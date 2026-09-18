"""Forge the activation-aware variant: `int4-head-linattn` with scaled in_proj.

    PYTHONPATH=../vendor/mlx-lm python3 tools/forge_awq.py

`IDEAS.md`'s top-ranked idea, built. `W diag(s)` quantized against
`x diag(1/s)` is the same product, so a channel the activations actually use
can be given more of the 4-bit grid at no runtime cost -- provided something
upstream can absorb `1/s`. `awq_search.py` chose the scales by the output
error they cause on calibration activations; this file applies them.

**Only `in_proj_qkv` and `in_proj_z` are scaled, and the reason is the whole
lesson of the experiment.** Three families were candidates and the fold point
decided each one, not the activation statistics:

  in_proj_qkv, in_proj_z  fold into `input_layernorm`, whose only other
                          readers are `in_proj_a` and `in_proj_b`. Those are
                          bf16 and get `s` folded into their weights instead.
                          **Built.** 28.3% of a decode step's traffic, and the
                          search finds 6-51% less output error a layer.
  out_proj                has the *largest* spread of the three -- 1319x
                          between its top channel and the median -- and can
                          only fold into `linear_attn.norm`, which is
                          per-head-dim and shared across all 32 value heads
                          while the outliers sit at individual (head, dim)
                          pairs. Removing the per-dim mean leaves 98-114% of
                          the log-variance. **Unreachable.**
  lm_head                 folds into `model.norm` -- which also feeds the MTP
                          head through `pre_fc_norm_hidden`, an RMSNorm, and
                          an RMSNorm cannot absorb a per-channel rescale
                          because its own denominator moves with the channel
                          pattern. **Unreachable**, and the search says it was
                          worth +1.5% anyway, the smallest of the three.

**Powers of two, so the reparametrisation is exact.** Every fold here is a
bf16 tensor divided or multiplied by `s`: the norm gain by `1/s`, the two
bf16 recurrence projections by `s`. At an arbitrary scale each of those
rounds, ~0.2% of uncontrolled relative error applied per channel to a whole
channel -- the shape of error that killed nvfp4, one order down. At a power
of two every one of them is exact, so the only thing that moves in the whole
artifact is what the quantizer does, and `max_abs == 0.0` against MLX still
means what it has always meant here. `awq_search.py` prices the resolution
this costs: 26.2% mean output-error reduction against 32.6% unrestricted.

Every assertion below is that exactness, checked tensor by tensor rather than
trusted.

The base is `int4-head-linattn`, already aligned, so `align_rebuild.py` is
*not* needed afterwards. `alpha = 0` reproduces the base bit for bit --
verified before this file was written -- which is what makes the scaling the
only variable.
"""

import json
import os
import pathlib
import shutil
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aq  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
AWQ = ROOT / os.environ.get("ORNITH_AWQ_DIR", "refs/awq")
MODELS = pathlib.Path.home() / ".mtplx/models"
PARENT = MODELS / "ornith-ai--Ornith-1.5-35B-A3B"
BASE = MODELS / "ornith-1.5-35b-a3b-int4-head-linattn"
OUT = MODELS / os.environ.get("ORNITH_AWQ_ARTIFACT",
                             "ornith-1.5-35b-a3b-awq-linattn")
GROUP_SIZE, BITS = 32, 4
SCALED = ("in_proj_qkv", "in_proj_z")
FOLDED = ("in_proj_a", "in_proj_b")


def bf16_exact(x, what):
    """A float32 array that survives bf16 storage unchanged, or an assertion."""
    assert np.array_equal(aq.bf16(x), x.astype(np.float32)), what
    return x.astype(np.float32)


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; remove it or pick another name")
    scales = np.load(AWQ / "scales.npz")
    bidx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    pidx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]

    layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in bidx
                     if "linear_attn.in_proj_qkv.weight" in k})
    assert len(layers) == 30, layers
    print(f"{len(layers)} linear-attention layers to rescale")

    # Every tensor this run rewrites, and therefore every shard it must reopen.
    writes = {}
    for i in layers:
        lp = f"language_model.model.layers.{i}"
        names = [f"{lp}.input_layernorm.weight"]
        names += [f"{lp}.linear_attn.{m}.{p}" for m in SCALED
                  for p in ("weight", "scales", "biases")]
        names += [f"{lp}.linear_attn.{m}.weight" for m in FOLDED]
        for n in names:
            writes.setdefault(bidx[n], []).append((i, n))

    OUT.mkdir(parents=True)
    for f in sorted(BASE.iterdir()):
        if f.is_file() and f.name not in writes:
            shutil.copy2(f, OUT / f.name)
    print(f"copied {len(list(OUT.iterdir()))} files unchanged; "
          f"rewriting {len(writes)} shards")

    t0, done = time.time(), 0
    for shard in sorted(writes):
        tensors = dict(mx.load(str(BASE / shard)))
        for i in sorted({i for i, _ in writes[shard]}):
            lp = f"language_model.model.layers.{i}"
            s = scales[f"layer{i:02d}.in"].astype(np.float32)
            assert np.array_equal(s, np.exp2(np.round(np.log2(s)))), f"L{i}: not po2"

            key = f"{lp}.input_layernorm.weight"
            if key in tensors:
                g = np.array(tensors[key].astype(mx.float32), copy=False)
                tensors[key] = mx.array(
                    bf16_exact(g / s, f"L{i}: norm fold not exact")).astype(mx.bfloat16)

            for m in FOLDED:
                key = f"{lp}.linear_attn.{m}.weight"
                if key in tensors:
                    w = np.array(tensors[key].astype(mx.float32), copy=False)
                    tensors[key] = mx.array(
                        bf16_exact(w * s, f"L{i}.{m}: fold not exact")).astype(mx.bfloat16)

            for m in SCALED:
                key = f"{lp}.linear_attn.{m}.weight"
                if key not in tensors:
                    continue
                pn = f"model.language_model.layers.{i}.linear_attn.{m}.weight"
                t = mx.load(str(PARENT / pidx[pn]))[pn]
                mx.eval(t)
                w = np.array(t.astype(mx.float32), copy=False)
                del t
                q, sc, b = aq.quantize(w * s, GROUP_SIZE, BITS)
                gshape = [w.shape[0], w.shape[1] // GROUP_SIZE]
                wshape = [w.shape[0], w.shape[1] * BITS // 32]
                tensors[key] = mx.array(aq.pack4(q, wshape[1]).reshape(wshape))
                tensors[f"{lp}.linear_attn.{m}.scales"] = mx.array(
                    sc.reshape(gshape)).astype(mx.bfloat16)
                tensors[f"{lp}.linear_attn.{m}.biases"] = mx.array(
                    b.reshape(gshape)).astype(mx.bfloat16)
                del w, q, sc, b
                done += 1
                print(f"  [{done:2d}/60] L{i:02d}.{m}", flush=True)
        mx.eval(*tensors.values())
        mx.save_safetensors(str(OUT / shard), tensors, metadata={"format": "mlx"})
        del tensors

    search = json.loads((AWQ / "search.json").read_text())
    (OUT / "awq.json").write_text(json.dumps({
        "base": BASE.name, "parent": PARENT.name,
        "scaled": [f"linear_attn.{m}" for m in SCALED],
        "folded_into": "input_layernorm, with in_proj_a/in_proj_b compensated",
        "scale_family": "power-of-two AWQ, (mean|x|)**alpha normalised",
        "chosen": {k: v for k, v in search["chosen"].items() if k.endswith(".in")},
        "calibration": json.loads((AWQ / "stats.json").read_text()),
    }, indent=1))
    print(f"\n{done} modules rescaled in {time.time() - t0:.0f}s -> {OUT}")
    print("\nalign_rebuild.py is NOT needed -- the base was already aligned. Next:")
    print(f"  python3 tools/check_rebuild.py {OUT}")
    print(f"  cd tools && ORNITH_BASELINE_OUT=../refs/artifact/awq-linattn \\")
    print(f"    PYTHONPATH=../../vendor/mlx-lm python3 dump_artifact_baseline.py {OUT}")
    print(f"  cd tools && ORNITH_PROBES_OUT=../refs/artifact/awq-linattn \\")
    print(f"    PYTHONPATH=../../vendor/mlx-lm python3 dump_probes.py {OUT}")


main()
