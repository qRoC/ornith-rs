"""Reference for the Q4_K kernel -- and the only case here whose oracle is not mlx.

`Q4K.md` step 2. mlx has no k-quant mode, so `refs/q4k/` is written from
**ggml**: `quantize_q4_K` produces the bytes, `dequantize_row_q4_K` produces
the values the Metal kernel has to reproduce, and neither learned the format
from this port. Every other `dump_*.py` here writes what MLX did; this one
writes what llama.cpp's own code did.

    cd tools && ORNITH_GGML_BUILD=../../.ggml-build python3 dump_q4k.py

The weights are the parent's, not synthetic: `quantize_q4_K` runs a search
whose behaviour depends on the distribution it is given, and the point of the
reference is to exercise the search's output rather than a uniform grid.

What is written, per case:

    qs / scales / dm    the three tensors the artifact stores, split out of
                        ggml's 144-byte blocks -- the kernel's actual input
    deq                 ggml's `dequantize_row_q4_K` over those blocks. The
                        `q4k_dequantize` kernel must match this at
                        `max_abs == 0.0`; it is the check that covers the
                        12-byte scale packing, the nibble interleave and the
                        subtraction all at once
    x, y                a bfloat16 activation row and the float64 product of
                        it with `deq`. A matvec cannot be pinned bit-exactly
                        -- its summation order belongs to the kernel, not to
                        the reference -- so the test carries a measured bound
    onehot_j, y_onehot  the same product with x = e_j, where the sum has one
                        nonzero term and the answer is a column of `deq`.
                        That one *is* checkable tightly, and it catches an
                        addressing error that a tolerance would hide

And one routed case, `moe`, with the same tensors carrying an `[E, ...]` axis
in front and a `(lhs, rhs)` pair per output row -- `mx.gather_qmm`'s contract
and `SwitchGLU`'s shape. An expert stride is the one thing the two-dimensional
cases cannot exercise.
"""

import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import q4k          # noqa: E402
import q4k_ggml     # noqa: E402
import ref          # noqa: E402

PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"

# Two shapes with different block counts a row, both multiples of the four
# rows a threadgroup covers. `lm_head` is the module the first cut is for.
CASES = [
    ("head", "lm_head.weight", 512, 2048),
    ("qkv", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 256, 2048),
]


def main():
    import mlx.core as mx

    lib = q4k_ggml.load()
    idx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    case = ref.Case("q4k")
    rng = np.random.default_rng(11)
    meta = {}

    for name, key, rows, cols in CASES:
        t = mx.load(str(PARENT / idx[key]))[key]
        mx.eval(t)
        w = np.ascontiguousarray(np.array(t.astype(mx.float32), copy=False)[:rows, :cols])
        del t
        nb = cols // q4k.QK_K

        blocks = q4k_ggml.quantize(lib, w)
        qs, scales, dm = q4k.split(blocks, rows, nb)
        deq = q4k_ggml.dequantize(lib, blocks, w.size).reshape(rows, cols)
        # The numpy reader is exact against ggml (`Q4K.md` step 1); assert it
        # here too, so a reference that disagrees with itself cannot be written.
        assert np.array_equal(deq, q4k.dequantize(blocks).reshape(rows, cols))

        # bfloat16 in, because that is what the model's activations are, and
        # the kernel converts on load exactly as ggml's converts from float.
        x = mx.array(rng.standard_normal(cols).astype(np.float32) * 0.05).astype(mx.bfloat16)
        mx.eval(x)
        xf = np.array(x.astype(mx.float32), copy=False).astype(np.float64)
        y = deq.astype(np.float64) @ xf

        j = int(rng.integers(cols))
        y_onehot = deq[:, j].astype(np.float64)

        case.put(f"{name}_qs", mx.array(qs))
        case.put(f"{name}_scales", mx.array(scales))
        case.put(f"{name}_dm", mx.array(dm))
        case.put(f"{name}_deq", mx.array(deq))
        case.put(f"{name}_x", x)
        case.put(f"{name}_y", mx.array(y.astype(np.float32)))
        case.put(f"{name}_y_onehot", mx.array(y_onehot.astype(np.float32)))
        meta[f"{name}_rows"] = rows
        meta[f"{name}_cols"] = cols
        meta[f"{name}_onehot_j"] = j
        rel = np.linalg.norm(w - deq) / np.linalg.norm(w)
        print(f"{name:<6} [{rows}, {cols}]  {nb} blocks a row  "
              f"ggml Q4_K rel RMS {100 * rel:.3f}%  onehot col {j}")
        del w, deq

    # --- the routed case ------------------------------------------------
    EXPERTS, EROWS, ECOLS, ROWS = 4, 512, 2048, 11
    key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    t = mx.load(str(PARENT / idx[key]))[key]
    mx.eval(t)
    # `gate_proj` is the low half of the parent's fused `gate_up_proj`, which
    # is how `forge_q4k_twin.py` maps it too.
    we = np.ascontiguousarray(
        np.array(t.astype(mx.float32), copy=False)[:EXPERTS, :EROWS, :ECOLS])
    del t
    nb = ECOLS // q4k.QK_K
    qs = np.empty((EXPERTS, EROWS, nb * 32), np.uint32)
    scales = np.empty((EXPERTS, EROWS, nb, 12), np.uint8)
    dm = np.empty((EXPERTS, EROWS, nb, 2), np.float16)
    deq = np.empty_like(we)
    for e in range(EXPERTS):
        blocks = q4k_ggml.quantize(lib, we[e])
        qs[e], scales[e], dm[e] = q4k.split(blocks, EROWS, nb)
        deq[e] = q4k_ggml.dequantize(lib, blocks, we[e].size).reshape(EROWS, ECOLS)

    # Distinct tokens and a repeated expert, so a wrong stride cannot land on
    # the right answer by accident.
    lhs = rng.integers(0, ROWS, ROWS).astype(np.uint32)
    rhs = rng.integers(0, EXPERTS, ROWS).astype(np.uint32)
    rhs[0], rhs[1] = 0, EXPERTS - 1
    xe = mx.array(rng.standard_normal((ROWS, ECOLS)).astype(np.float32) * 0.05)
    mx.eval(xe)
    xf = np.array(xe, copy=False).astype(np.float64)
    ye = np.stack([deq[rhs[r]].astype(np.float64) @ xf[lhs[r]] for r in range(ROWS)])

    case.put("moe_qs", mx.array(qs))
    case.put("moe_scales", mx.array(scales))
    case.put("moe_dm", mx.array(dm))
    case.put("moe_deq", mx.array(deq))
    case.put("moe_x", xe.astype(mx.float32))
    case.put("moe_lhs", mx.array(lhs))
    case.put("moe_rhs", mx.array(rhs))
    case.put("moe_y", mx.array(ye.astype(np.float32)))
    meta.update(moe_experts=EXPERTS, moe_rows=EROWS, moe_cols=ECOLS, moe_n=ROWS)
    print(f"moe    [{EXPERTS}, {EROWS}, {ECOLS}]  {ROWS} routed rows  "
          f"experts {sorted(set(int(v) for v in rhs))}")
    del we, deq

    meta["cases"] = [c[0] for c in CASES]
    meta["module"] = MODULE
    case.note(**meta)
    case.write()
    write_artifact(mx, case.dir)


# A one-module artifact in the layout `WeightStore` reads, so the loader has
# something to open. `Q4K.md`'s three-tensor storage is only a decision until
# something writes it down and something else reads it back.
MODULE = "language_model.model.layers.0.linear_attn.out_proj"


def write_artifact(mx, case_dir):
    import shutil

    d = pathlib.Path(case_dir) / "artifact"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    # Re-read what the case just wrote, so the artifact and the reference can
    # never drift apart.
    tensors = {}
    man = json.loads((pathlib.Path(case_dir) / "manifest.json").read_text())
    dt = {"uint32": np.uint32, "uint8": np.uint8, "float16": np.float16}
    for suffix, name in (("weight", "head_qs"), ("scales", "head_scales"), ("dm", "head_dm")):
        info = man["tensors"][name]
        a = np.fromfile(pathlib.Path(case_dir) / info["file"], dtype=dt[info["dtype"]])
        tensors[f"{MODULE}.{suffix}"] = mx.array(a.reshape(info["shape"]))
    mx.save_safetensors(str(d / "model.safetensors"), tensors)
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model.safetensors" for k in tensors}}, indent=1))
    (d / "config.json").write_text(json.dumps({
        "quantization": {"group_size": q4k.QK_K, "bits": 4, "mode": "q4k"},
    }, indent=1))
    print(f"artifact: {len(tensors)} tensors -> {d}", file=sys.stderr)


if __name__ == "__main__":
    main()
