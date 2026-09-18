"""Forge the real Q4_K artifact: the 91 modules, in ggml's own bytes.

    ORNITH_GGML_BUILD=../../.ggml-build python3 tools/forge_q4k.py [--imatrix]

`Q4K.md`, after the kernel. Phase 14 measured a Q4_K-*shaped* grid as a
stand-in -- power-of-two super-scales and magnitude-plus-sign fields, so it
could be written into the affine layout and run without a kernel -- and read
the same median KL as the 5.00-bit base for 7.3% fewer bytes. Both constraints
were scaffolding, and dropping them is what makes llama.cpp the oracle. This
writes the format itself.

**The quantizer is ggml's.** `tools/q4k.py`'s writer is a faithful port of
`quantize_row_q4_K_ref`'s shape and is 0.77 points worse, because ggml runs a
search rather than plain min/max. Nothing that ships should use it;
`q4k_ggml.quantize` calls the real one through ctypes.

**The 91 modules** are the ones `int4-head-linattn` moved to 4 bits: 30
gated-delta layers' `in_proj_qkv`, `in_proj_z` and `out_proj`, plus `lm_head`.
They are 57% of a decode step's traffic, they are what every number since
Phase 4 is measured on, and leaving the experts at affine means one new kernel
rather than two. Everything else is copied from the base artifact byte for
byte, so the only difference between the two is the thing under test.

**Storage**, `Q4K.md`'s three tensors per module -- ggml's bytes, split so the
loader and `check_rebuild.py` see shapes they already know:

    <mod>.weight   uint32  [O, I / 8]        the 4-bit codes
    <mod>.scales   uint8   [O, I / 256, 12]  8 x 6-bit scale + 8 x 6-bit min
    <mod>.dm       float16 [O, I / 256, 2]   d and dmin

and `.biases` is dropped: Q4_K's minimum is one of the packed 6-bit fields.
`config.json` gets `mode: "q4k"`, `group_size: 256` per converted module.

`--imatrix` passes `diag(H)` from `refs/gptq-parent/hessians/` as
`quantize_q4_K`'s importance vector, which measured 8 to 80% off the
H-weighted weight error for 1.3 to 5.2% of the unweighted one. Only 61 of the
91 have a Hessian (the gated-delta pairs and the head, not `in_proj_z`), so
the rest are quantized plain and the count is printed.
"""

import json
import pathlib
import shutil
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import q4k        # noqa: E402
import q4k_ggml   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = pathlib.Path.home() / ".mtplx/models"
PARENT = MODELS / "ornith-ai--Ornith-1.5-35B-A3B"
BASE = MODELS / "ornith-1.5-35b-a3b-int4-head-linattn"
HESSIANS = ROOT / "refs/gptq-parent/hessians"

IMATRIX = "--imatrix" in sys.argv
OUT = MODELS / ("ornith-1.5-35b-a3b-q4k-imatrix" if IMATRIX else "ornith-1.5-35b-a3b-q4k")

# `Q4K.md`'s first cut. A module is in when `int4-head-linattn` moved it to 4
# bits, which is exactly these four suffixes.
SUFFIXES = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")


def parent_of(mod):
    """Artifact module path -> parent tensor name. `forge_q4k_twin.py`'s map,
    minus the expert branches this cut does not touch."""
    if mod == "language_model.lm_head":
        return "lm_head.weight"
    assert mod.startswith("language_model.model.layers."), mod
    n, rest = mod[len("language_model.model.layers.") :].split(".", 1)
    assert rest in SUFFIXES, mod
    return f"model.language_model.layers.{n}.{rest}.weight"


def hessian_for(mod):
    """`diag(H)` for a module, or None when the GPTQ run did not cover it."""
    f = HESSIANS / f"{mod}.npy"
    if not f.exists():
        return None
    # The files hold the whole `H`, [in, in]; the importance vector is its
    # diagonal. Passing the array as it loads hands `quantize_q4_K` H's first
    # row, negatives and all, and the answer is 38589% error.
    h = np.ascontiguousarray(np.diag(np.asarray(np.load(f), np.float32)))
    assert (h >= 0).all(), f"{mod}: diag(H) has negative entries"
    return h


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; remove it or pick another name")
    lib = q4k_ggml.load()
    acfg = json.loads((BASE / "config.json").read_text())
    aidx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    pidx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]

    mods = sorted({k.rsplit(".", 1)[0] for k in aidx if k.endswith(".scales")})
    convert = [
        m
        for m in mods
        if m == "language_model.lm_head" or m.split(".layers.")[-1].split(".", 1)[-1] in SUFFIXES
    ]
    assert len(convert) == 91, f"{len(convert)} modules, expected 91"
    print(f"{len(mods)} quantized modules, converting {len(convert)} to Q4_K"
          + (" with diag(H)" if IMATRIX else ""))

    OUT.mkdir(parents=True)
    touched = {aidx[m + ".weight"] for m in convert}
    for f in sorted(BASE.iterdir()):
        if f.is_file() and f.name not in touched:
            shutil.copy2(f, OUT / f.name)
    print(f"copied {len(list(OUT.iterdir()))} files unchanged; "
          f"rewriting {len(touched)} shards")

    by_shard = {}
    for m in convert:
        by_shard.setdefault(aidx[m + ".weight"], []).append(m)

    t0 = time.time()
    done = with_h = 0
    bytes_before = bytes_after = 0
    for shard in sorted(touched):
        tensors = dict(mx.load(str(BASE / shard)))
        for mod in by_shard[shard]:
            pname = parent_of(mod)
            t = mx.load(str(PARENT / pidx[pname]))[pname]
            mx.eval(t)
            w = np.ascontiguousarray(np.array(t.astype(mx.float32), copy=False))
            del t
            rows, cols = w.shape
            assert cols % q4k.QK_K == 0, f"{mod}: {cols} is not a multiple of 256"
            assert rows % 4 == 0, f"{mod}: {rows} rows, kernel covers 4 at a time"
            nb = cols // q4k.QK_K

            h = hessian_for(mod) if IMATRIX else None
            with_h += h is not None
            qs, scales, dm = q4k.split(q4k_ggml.quantize(lib, w, imatrix=h), rows, nb)

            for suffix in ("weight", "scales", "biases", "dm"):
                tensors.pop(f"{mod}.{suffix}", None)
            bytes_before += rows * cols * 5 // 8      # affine int4 g32 = 5.00 bits
            bytes_after += rows * nb * q4k.BLOCK_BYTES
            tensors[f"{mod}.weight"] = mx.array(qs)
            tensors[f"{mod}.scales"] = mx.array(scales)
            tensors[f"{mod}.dm"] = mx.array(dm)
            del w, qs, scales, dm
            done += 1
            print(f"  [{done:3d}/{len(convert)}] {mod}"
                  + ("  +imatrix" if h is not None else ""), flush=True)
        mx.eval(*tensors.values())
        mx.save_safetensors(str(OUT / shard), tensors, metadata={"format": "mlx"})
        del tensors

    # The index has to lose `.biases` and gain `.dm`, or the loader looks for a
    # tensor no shard holds.
    idx = json.loads((BASE / "model.safetensors.index.json").read_text())
    for m in convert:
        shard = idx["weight_map"].pop(m + ".biases")
        idx["weight_map"][m + ".dm"] = shard
    (OUT / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))

    for m in convert:
        acfg["quantization"][m] = {"group_size": q4k.QK_K, "bits": 4, "mode": "q4k"}
    (OUT / "config.json").write_text(json.dumps(acfg, indent=1))

    print(f"\n{done} modules in {time.time() - t0:.0f}s"
          + (f", {with_h} of them with diag(H)" if IMATRIX else ""))
    print(f"those modules: {bytes_before / 1e9:.4f} GB -> {bytes_after / 1e9:.4f} GB "
          f"({100 * (bytes_after / bytes_before - 1):+.1f}%)")
    print(f"wrote {OUT}\n")
    print("Next -- and note mlx_lm cannot load this, so the port is the instrument:")
    print(f"  python3 tools/check_rebuild.py {OUT}")
    print(f"  ORNITH_MODEL={OUT} ORNITH_PROBES_OUT=refs/artifact/{OUT.name.split('a3b-')[-1]} \\")
    print("    cargo run --release --bin probes")


if __name__ == "__main__":
    main()
