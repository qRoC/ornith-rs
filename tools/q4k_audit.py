"""Audit a forged k-quant artifact against ggml, module by module.

    ORNITH_GGML_BUILD=../../.ggml-build python3 tools/q4k_audit.py <artifact>

`tests/q4k.rs` pins the kernel on two reference matrices. This asks the same
question of the artifact that a verdict is about to be read off: **are the
weights the port will hold the weights ggml wrote?** It covers the forge's
split, the index, the loader and the kernel's read path at once, which the
reference cases cannot -- they never went through a safetensors file.

Two outputs per module:

    rel RMS   against the bf16 parent, so a forge that wrote the right shapes
              and the wrong numbers is visible as a number rather than as a KL
    digest    an order-sensitive 64-bit sum over the dequantized float32
              *bits*, in row-major order. A hash, because the alternative is
              dumping 0.85 GB of floats; exact, because `max_abs == 0.0` is
              the contract and a tolerance would defeat the point. The Rust
              side recomputes it from its own kernel's output and compares.
              Written as hex: a 64-bit integer does not survive a JSON number
              through a parser that reads numbers as float64.

Written to `<artifact>/q4k_audit.json`, which the ignored Rust test reads.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import q4k        # noqa: E402
import q4k_ggml   # noqa: E402

PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"
# An index-dependent multiplier makes the sum order-sensitive, which a plain
# sum is not: a row stride wrong or a block misplaced changes the digest.
# Everything is mod 2**64, where addition is associative, so numpy's pairwise
# summation and Rust's serial fold agree exactly.
MIX = 0x9E3779B97F4A7C15
ADD = 0x165667B19E3779F9


def digest(arr):
    """Order-sensitive 64-bit digest of a float32 array's bits."""
    w = np.ascontiguousarray(arr, np.float32).ravel().view(np.uint32).astype(np.uint64)
    i = np.arange(len(w), dtype=np.uint64)
    return f"{int(np.sum(w * (i * np.uint64(MIX) + np.uint64(ADD)), dtype=np.uint64)):#018x}"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    import mlx.core as mx

    art = pathlib.Path(sys.argv[1]).expanduser()
    lib = q4k_ggml.load()
    cfg = json.loads((art / "config.json").read_text())["quantization"]
    idx = json.loads((art / "model.safetensors.index.json").read_text())["weight_map"]
    pidx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]

    kinds = {m: {"q4k": "q4_K", "q6k": "q6_K"}[v["mode"]]
             for m, v in cfg.items()
             if isinstance(v, dict) and v.get("mode") in ("q4k", "q6k")}
    mods = sorted(kinds)
    if not mods:
        sys.exit(f"{art} has no k-quantized modules")
    from collections import Counter
    print(f"{len(mods)} k-quantized modules in {art.name}: "
          + ", ".join(f"{n} {k}" for k, n in Counter(kinds.values()).most_common()) + "\n")

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from forge_q4k import parent_of

    out = {}
    worst = 0.0
    for n, mod in enumerate(mods):
        shard = idx[f"{mod}.weight"]
        t = mx.load(str(art / shard))
        qs = np.array(t[f"{mod}.weight"], copy=False)
        scales = np.array(t[f"{mod}.scales"], copy=False)
        dm = np.array(t[f"{mod}.dm"], copy=False)
        del t
        # A routed module carries an [E, ...] axis; flatten it, since the block
        # layout below is the same either way.
        shape = qs.shape
        qs = qs.reshape(-1, shape[-1])
        scales = scales.reshape(-1, scales.shape[-2], scales.shape[-1])
        dm = dm.reshape(-1, dm.shape[-2], dm.shape[-1])
        rows, nb = scales.shape[0], scales.shape[1]

        # Reassemble ggml's blocks out of the three tensors -- the inverse of
        # `q4k.split`, so a disagreement here is the forge's.
        kind = kinds[mod]
        nbytes = q4k_ggml.KINDS[kind][0]
        blocks = np.empty((rows * nb, nbytes), np.uint8)
        if kind == "q4_K":
            blocks[:, 0:4] = dm.reshape(rows * nb, 2).view(np.uint8).reshape(-1, 4)
            blocks[:, 4 : 4 + q4k.K_SCALE_SIZE] = scales.reshape(rows * nb, q4k.K_SCALE_SIZE)
            blocks[:, 4 + q4k.K_SCALE_SIZE :] = qs.reshape(rows * nb, 32).view(np.uint8)
        else:
            blocks[:, 0:192] = qs.reshape(rows * nb, 48).view(np.uint8)
            blocks[:, 192:208] = scales.reshape(rows * nb, 16)
            blocks[:, 208:210] = dm.reshape(rows * nb, 1).view(np.uint8).reshape(-1, 2)
        deq = q4k_ggml.dequantize(lib, blocks, rows * nb * q4k.QK_K, kind=kind).reshape(rows, -1)

        pname, half = parent_of(mod)
        pt = mx.load(str(PARENT / pidx[pname]))[pname]
        if half:
            h2 = pt.shape[1] // 2
            pt = pt[:, :h2, :] if half == "lo" else pt[:, h2:, :]
        mx.eval(pt)
        w = np.array(pt.astype(mx.float32), copy=False).reshape(deq.shape)
        del pt
        rel = 100 * float(np.linalg.norm(w - deq) / np.linalg.norm(w))
        worst = max(worst, rel)
        out[mod] = {"rows": rows, "blocks": nb, "kind": kind,
                    "rel_rms": rel, "digest": digest(deq)}
        print(f"  [{n + 1:3d}/{len(mods)}] {mod:<58} {kind}  {rel:6.3f}%", flush=True)
        del w, deq, blocks

    (art / "q4k_audit.json").write_text(json.dumps(out, indent=1))
    rels = [v["rel_rms"] for v in out.values()]
    print(f"\nrel RMS vs the parent: min {min(rels):.3f}%  "
          f"median {float(np.median(rels)):.3f}%  max {max(rels):.3f}%")
    print(f"wrote {art}/q4k_audit.json")


main()
