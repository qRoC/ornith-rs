"""What each variant's quantization actually cost, against the bf16 parent.

    PYTHONPATH=../vendor/mlx-lm python3 tools/weight_error.py
    PYTHONPATH=../vendor/mlx-lm python3 tools/weight_error.py --decompose

`tools/quant_error.py` predicts a format's error by quantizing a million
synthetic draws -- gaussian, laplace, student-t. That is how REBUILD.md Phase
4 priced nvfp4 at "the same error as affine int4 g32, half a bit cheaper", and
the measured perplexity and KL say the prediction did not hold. This asks the
same question of the weights that are actually in the model: dequantize a
module out of each variant and compare it to the same module in the parent.

It is the one measurement the parent can still give on this host. Loading the
whole 67.0 GiB parent is what `mlx_lm.load` cannot do here (51.8 GiB working
set), but `mx.load` is lazy and a single tensor is at most 1.0 GiB, so one
module at a time is free.

Reported per module:

    rel RMS   ||dequant - parent|| / ||parent||, which is what quant_error.py
              calls the relative RMS weight error, so the two are comparable.

`--decompose` then splits nvfp4's share of that in two, by reconstructing fp4
by hand over the same parent tensors with the shared scale at three
precisions: exact, E4M3 round-to-nearest (what mlx does), and E4M3 rounded
up. The round-to-nearest column reproduces mlx's own nvfp4 numbers to the
last digit, which is what says the exact-scale column is measuring the E2M1
grid rather than a reimplementation of it.

Names differ across the three: the parent is a HF checkpoint
(`model.language_model.layers.N...`), the artifacts are mlx-lm's sanitized
form (`language_model.model.layers.N...`), and the 256 experts are fused in
the parent (`mlp.experts.gate_up_proj`, one tensor) and stacked in the
artifacts (`mlp.switch_mlp.gate_proj`, `.up_proj`).
"""

import json
import pathlib
import sys

import mlx.core as mx
import numpy as np

MODELS = pathlib.Path.home() / ".mtplx/models"
PARENT = MODELS / "ornith-ai--Ornith-1.5-35B-A3B"
VARIANTS = [
    ("int8-everywhere", "ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx"),
    ("int4-head-linattn", "ornith-1.5-35b-a3b-int4-head-linattn"),
    ("nvfp4-head-linattn", "ornith-1.5-35b-a3b-nvfp4-head-linattn"),
    ("int4-except-gates", "ornith-1.5-35b-a3b-int4-except-gates"),
]

L = 0
# (label, artifact module, parent tensor, slice of the parent tensor)
#
# The four byte-heavy families Phase 4 moved, plus one expert slab and the
# router for contrast. `switch_mlp.gate_proj` is the first half of the
# parent's fused `gate_up_proj`, per mlx-lm's sanitize.
CASES = [
    ("lm_head", "language_model.lm_head", "lm_head.weight", None),
    ("in_proj_qkv", f"language_model.model.layers.{L}.linear_attn.in_proj_qkv",
     f"model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight", None),
    ("in_proj_z", f"language_model.model.layers.{L}.linear_attn.in_proj_z",
     f"model.language_model.layers.{L}.linear_attn.in_proj_z.weight", None),
    ("out_proj", f"language_model.model.layers.{L}.linear_attn.out_proj",
     f"model.language_model.layers.{L}.linear_attn.out_proj.weight", None),
    ("switch_mlp.gate_proj[0:8]", f"language_model.model.layers.{L}.mlp.switch_mlp.gate_proj",
     f"model.language_model.layers.{L}.mlp.experts.gate_up_proj", "experts"),
    ("mlp.gate (router)", f"language_model.model.layers.{L}.mlp.gate",
     f"model.language_model.layers.{L}.mlp.gate.weight", None),
]
EXPERTS = 8  # of 256, to keep one case cheap


def shard_of(art, name):
    idx = json.loads((art / "model.safetensors.index.json").read_text())
    return art / idx["weight_map"][name]


def load_one(art, name):
    return mx.load(str(shard_of(art, name)))[name]


def dequantized(art, module):
    """One module out of an artifact, back in bf16, whatever its mode."""
    cfg = json.loads((art / "config.json").read_text())
    q = cfg["quantization"]
    spec = q.get(module, q)
    mode = spec.get("mode", q.get("mode", "affine"))
    gs, bits = spec.get("group_size", q["group_size"]), spec.get("bits", q["bits"])

    w = load_one(art, f"{module}.weight")
    scales = load_one(art, f"{module}.scales")
    biases = None
    if mode == "affine":
        biases = load_one(art, f"{module}.biases")
    return mx.dequantize(
        w, scales=scales, biases=biases, group_size=gs, bits=bits, mode=mode
    ), f"{bits}b g{gs} {mode}"


# --- --decompose: where inside nvfp4 the error is --------------------------

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def e4m3(a, up=False):
    """Round to E4M3 (OCP): 3 mantissa bits, bias 7, max 448."""
    a = np.asarray(a, np.float32)
    out = np.zeros_like(a)
    nz = a != 0
    e = np.clip(np.floor(np.log2(np.abs(a[nz]))), -6, 8)
    r = a[nz] / 2.0**e * 8
    out[nz] = np.clip((np.ceil(r) if up else np.round(r)) / 8 * 2.0**e, -448, 448)
    return out


def fp4(w, g, scale_fn):
    """E2M1 elements, one shared scale a group of `g`, as mlx's nvfp4 cuts it.

    Returns the reconstruction and the fraction of elements the scale pushed
    past the E2M1 clip at 6.0 -- which is the failure mode affine cannot have,
    its scale being min/max by construction.
    """
    w = w.reshape(-1, g).astype(np.float32)
    a = np.abs(w).max(1, keepdims=True) / 6.0
    a = np.where(a == 0, 1, a)
    s = scale_fn(a)
    s = np.where(s == 0, 1, s)
    i = np.abs(np.abs(w / s)[..., None] - E2M1).argmin(-1)
    return (np.sign(w) * E2M1[i] * s).ravel(), float((np.abs(w / s) > 6.0).mean())


def decompose():
    print("nvfp4's error split: the E2M1 grid, then what the E4M3 scale adds\n")
    print(f"{'module':<26}{'exact s':>10}{'E4M3 rn':>10}{'clip':>8}{'E4M3 up':>10}{'clip':>8}")
    for label, _module, parent_name, kind in CASES:
        if parent_name not in json.loads(
            (PARENT / "model.safetensors.index.json").read_text()
        )["weight_map"]:
            continue
        t = load_one(PARENT, parent_name)
        if kind == "experts":
            t = t[:EXPERTS, : t.shape[1] // 2, :]
        mx.eval(t)
        w = np.array(t.astype(mx.float32)).ravel()
        del t
        w = w[: (len(w) // 16) * 16]
        ex, _ = fp4(w, 16, lambda a: a)
        rn, c1 = fp4(w, 16, e4m3)
        up, c2 = fp4(w, 16, lambda a: e4m3(a, up=True))
        print(f"{label:<26}{rel_rms_np(ex, w):9.2f}%{rel_rms_np(rn, w):9.2f}%"
              f"{100 * c1:7.1f}%{rel_rms_np(up, w):9.2f}%{100 * c2:7.1f}%")


def rel_rms_np(a, b):
    return 100 * float(np.sqrt(((a - b) ** 2).sum() / (b**2).sum()))


def rel_rms(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.sqrt(mx.sum((a - b) ** 2) / mx.sum(b**2)))


def main():
    print(f"parent: {PARENT.name}\n")
    width = max(len(n) for n, _ in VARIANTS)
    for label, module, parent_name, kind in CASES:
        ref = load_one(PARENT, parent_name)
        if kind == "experts":
            # gate_up_proj is [E, 2 * inter, in]; gate is the first half, and
            # mlx-lm's sanitize does not transpose it.
            ref = ref[:EXPERTS, : ref.shape[1] // 2, :]
        mx.eval(ref)
        print(f"{label}  parent {list(ref.shape)} {ref.dtype}")
        for name, dirname in VARIANTS:
            art = MODELS / dirname
            try:
                got, how = dequantized(art, module)
            except KeyError:
                print(f"  {name:<{width}}  (not quantized here)")
                continue
            if kind == "experts":
                got = got[:EXPERTS]
            mx.eval(got)
            print(f"  {name:<{width}}  {how:<16}{100 * rel_rms(got, ref):6.2f}%")
            del got
        del ref
        print()


if "--decompose" in sys.argv:
    decompose()
else:
    main()
