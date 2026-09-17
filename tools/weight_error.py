"""What each variant's quantization actually cost, against the bf16 parent.

    PYTHONPATH=../vendor/mlx-lm python3 tools/weight_error.py
    PYTHONPATH=../vendor/mlx-lm python3 tools/weight_error.py --decompose
    PYTHONPATH=../vendor/mlx-lm python3 tools/weight_error.py --schemes

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


# --- --schemes: IDEAS.md's metadata table, on real weights -----------------

# Whole groups of 32, evenly spaced through the tensor, so a g16 scheme and a
# g32 scheme see exactly the same elements and the sample spans every row
# rather than the first few thousand.
SAMPLE_GROUPS = 1 << 16


def sample(w, g=32):
    w = w.ravel()
    w = w[: (len(w) // g) * g].reshape(-1, g)
    if len(w) > SAMPLE_GROUPS:
        w = w[:: len(w) // SAMPLE_GROUPS][:SAMPLE_GROUPS]
    return np.ascontiguousarray(w.ravel())


def schemes():
    """IDEAS.md's metadata table, on real weights, with and without a global
    scale.

    The distinction is the whole result. An 8-bit shared scale is E4M3, whose
    smallest normal is 2**-6; this model's weights have sigma 0.015, so a
    group's max/6 lands near 2**-7.7 and **99.6% of group scales are
    subnormal**, on a grid of 2**-9. That is a ~20% relative scale error, not
    the ~6% three mantissa bits suggest, and it is why nvfp4 measured 10.9%
    where the synthetic study said 9.5%.

    A single fp32 per tensor fixes it, by moving the group scales into E4M3's
    normal range before they are encoded. nvfp4 has exactly that input and
    calls it `global_scale`; `mlx_lm.convert` does not pass one, which is why
    the built artifact pays the full price. Any bespoke scheme would carry the
    same fp32, so the `+global` rows are the honest form of the ideas.
    """
    import quant_error as q

    def enc_global(ref):
        """An E4M3 scale encoding told the tensor's magnitude first: one fp32
        per tensor, exactly what nvfp4's global_scale buys."""
        gs = ref / 448.0
        return lambda x: q.e4m3(x / gs) * gs

    def affine_enc(w, g, bits, enc):
        w = w.reshape(-1, g)
        n = 2**bits - 1
        lo, hi = w.min(1, keepdims=True), w.max(1, keepdims=True)
        s, b = enc((hi - lo) / n), enc(lo)
        s = np.where(s == 0, 1, s)
        return (np.clip(np.round((w - b) / s), 0, n) * s + b).ravel()

    def nvfp4_enc(w, g, enc):
        w = w.reshape(-1, g)
        a = np.abs(w).max(1, keepdims=True) / 6.0
        a = np.where(a == 0, 1, a)
        s = enc(a)
        s = np.where(s == 0, 1, s)
        i = np.abs(np.abs(w / s)[..., None] - q.E2M1).argmin(-1)
        return (np.sign(w) * q.E2M1[i] * s).ravel()

    rows = [
        ("affine g32, bf16 scale+bias   -- the body today", 5.00,
         lambda w: q.affine(w, 32, 4)),
        ("nvfp4 g16                     -- as forged", 4.50,
         lambda w: nvfp4_enc(w, 16, q.e4m3)),
        ("nvfp4 g16, + global scale     -- as mlx allows", 4.50,
         lambda w: nvfp4_enc(w, 16, enc_global(np.abs(w).max()))),
        ("IDEAS #1  affine g16, fp8 scale+bias", 5.00,
         lambda w: affine_enc(w, 16, 4, q.e4m3)),
        ("IDEAS #1  + global scale", 5.00,
         lambda w: affine_enc(w, 16, 4, enc_global(np.abs(w).max()))),
        ("IDEAS #2  Lloyd-Max g16, fp8 scale, + global", 4.50,
         lambda w: q.lloyd(w, 16, 4, enc_global(np.abs(w).max()))),
        ("          Lloyd-Max g32, bf16 scale", 4.50,
         lambda w: q.lloyd(w, 32, 4, q.bf16)),
    ]

    labels = [c[0] for c in CASES if c[0] != "mlp.gate (router)"]
    cols = {}
    for label, _m, parent_name, kind in CASES:
        if label == "mlp.gate (router)":
            continue
        t = load_one(PARENT, parent_name)
        if kind == "experts":
            t = t[:EXPERTS, : t.shape[1] // 2, :]
        mx.eval(t)
        cols[label] = sample(np.array(t.astype(mx.float32)))
        del t

    print("IDEAS.md's metadata table, on the model's own weights")
    print("(a scale is E4M3; +global is one fp32 a tensor, moving the group")
    print(" scales out of E4M3's subnormals -- 99.6% of them land there)\n")
    print(f"  {'scheme':<48}{'bits':>6}" + "".join(f"{l.split('[')[0][:12]:>14}" for l in labels))
    for name, bits, f in rows:
        cells = "".join(f"{rel_rms_np(f(cols[l]), cols[l]):>13.2f}%" for l in labels)
        print(f"  {name:<48}{bits:>6.2f}{cells}")


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
elif "--schemes" in sys.argv:
    schemes()
else:
    main()
