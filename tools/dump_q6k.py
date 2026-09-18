"""Reference for the Q6_K kernel, from ggml. The sibling of `dump_q4k.py`.

    cd tools && ORNITH_GGML_BUILD=../../.ggml-build python3 dump_q6k.py

Same shape as the Q4_K case and for the same reason -- mlx has no k-quant, so
`quantize_q6_K` writes the bytes and `dequantize_row_q6_K` writes the values
the Metal kernel has to reproduce, and neither learned the format here.

The modules are the ones Q6_K would actually replace: the artifact leaves 169
at affine int8 g64 and Q6_K is 6.5625 bits, so these are `self_attn` and the
shared expert rather than the gated-delta projections `dump_q4k.py` uses. The
routed case is layer 39's experts, which are among the eight layers held at 8
bits.

Q6_K's two traps, neither of them Q4_K's:

  the sign     the sub-block scale is **int8**, not an unsigned field
  the bias     a code is 6 bits biased by 32, so `w = d * sc * (q - 32)` and
               there is no minimum to subtract at all

A reader that keeps Q4_K's conventions is wrong by an offset and a sign, and
still produces fluent text.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import q4k          # noqa: E402
import q4k_ggml     # noqa: E402
import ref          # noqa: E402

PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"

# See `dump_q4k.py`: not a multiple of the gemm's 32-row tile.
MROWS = 37

CASES = [
    ("attn", "model.language_model.layers.3.self_attn.q_proj.weight", 512, 2048),
    ("shared", "model.language_model.layers.39.mlp.shared_expert.down_proj.weight", 1024, 512),
]


def main():
    import mlx.core as mx

    lib = q4k_ggml.load()
    idx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    case = ref.Case("q6k")
    rng = np.random.default_rng(23)
    meta = {}

    for name, key, rows, cols in CASES:
        t = mx.load(str(PARENT / idx[key]))[key]
        mx.eval(t)
        w = np.ascontiguousarray(np.array(t.astype(mx.float32), copy=False)[:rows, :cols])
        del t
        nb = cols // q4k.QK_K

        blocks = q4k_ggml.quantize(lib, w, kind="q6_K")
        qs, scales, dm = q4k.split(blocks, rows, nb, "q6_K")
        deq = q4k_ggml.dequantize(lib, blocks, w.size, kind="q6_K").reshape(rows, cols)

        x = mx.array(rng.standard_normal(cols).astype(np.float32) * 0.05).astype(mx.bfloat16)
        mx.eval(x)
        xf = np.array(x.astype(mx.float32), copy=False).astype(np.float64)
        y = deq.astype(np.float64) @ xf
        j = int(rng.integers(cols))

        case.put(f"{name}_qs", mx.array(qs))
        case.put(f"{name}_scales", mx.array(scales))
        case.put(f"{name}_dm", mx.array(dm))
        case.put(f"{name}_deq", mx.array(deq))
        case.put(f"{name}_x", x)
        case.put(f"{name}_y", mx.array(y.astype(np.float32)))
        case.put(f"{name}_y_onehot", mx.array(deq[:, j].astype(np.float32)))
        meta[f"{name}_rows"] = rows
        meta[f"{name}_cols"] = cols
        meta[f"{name}_onehot_j"] = j

        # And the prefill shape. `qmm_t` tiles 32 rows at a time, so the row
        # count is deliberately not a multiple of 32: the tail tile is where a
        # bounds check that is only nearly right shows up.
        xm = mx.array((rng.standard_normal((MROWS, cols)) * 0.05).astype(np.float32)
                      ).astype(mx.bfloat16)
        mx.eval(xm)
        xmf = np.array(xm.astype(mx.float32), copy=False).astype(np.float64)
        case.put(f"{name}_xm", xm)
        case.put(f"{name}_ym", mx.array((xmf @ deq.astype(np.float64).T).astype(np.float32)))
        meta[f"{name}_m"] = MROWS

        rel = np.linalg.norm(w - deq) / np.linalg.norm(w)
        print(f"{name:<7} [{rows}, {cols}]  {nb} blocks a row  "
              f"ggml Q6_K rel RMS {100 * rel:.3f}%  onehot col {j}")
        del w, deq

    # --- the routed case, from a layer whose experts are held at 8 bits ---
    EXPERTS, EROWS, ECOLS, ROWS = 4, 512, 2048, 11
    key = "model.language_model.layers.39.mlp.experts.gate_up_proj"
    t = mx.load(str(PARENT / idx[key]))[key]
    mx.eval(t)
    we = np.ascontiguousarray(
        np.array(t.astype(mx.float32), copy=False)[:EXPERTS, :EROWS, :ECOLS])
    del t
    nb = ECOLS // q4k.QK_K
    qs = np.empty((EXPERTS, EROWS, nb * 48), np.uint32)
    scales = np.empty((EXPERTS, EROWS, nb, 16), np.uint8)
    dm = np.empty((EXPERTS, EROWS, nb, 1), np.float16)
    deq = np.empty_like(we)
    for e in range(EXPERTS):
        blocks = q4k_ggml.quantize(lib, we[e], kind="q6_K")
        qs[e], scales[e], dm[e] = q4k.split(blocks, EROWS, nb, "q6_K")
        deq[e] = q4k_ggml.dequantize(lib, blocks, we[e].size, kind="q6_K").reshape(EROWS, ECOLS)

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

    # The routed prefill shape: rows grouped by expert, the way
    # `switch_glu_sorted` hands them over once a batch reaches 64. The run
    # lengths are uneven on purpose -- 5, 40, 3, 22 over 70 rows -- so that the
    # 32-row tiles straddle expert boundaries in three different ways, which is
    # the one thing a tiled gather has that a mat-vec per row does not.
    counts = [5, 40, 3, 22]
    srhs = np.concatenate([np.full(c, e, np.uint32) for e, c in enumerate(counts)])
    SN = int(srhs.size)
    # A shuffled lhs, so the activation indirection is exercised rather than
    # collapsing to the identity the sorted path happens to pass.
    slhs = rng.permutation(SN).astype(np.uint32)
    sx = mx.array((rng.standard_normal((SN, ECOLS)) * 0.05).astype(np.float32))
    mx.eval(sx)
    sxf = np.array(sx, copy=False).astype(np.float64)
    sy = np.stack([deq[srhs[r]].astype(np.float64) @ sxf[slhs[r]] for r in range(SN)])
    case.put("moes_x", sx)
    case.put("moes_lhs", mx.array(slhs))
    case.put("moes_rhs", mx.array(srhs))
    case.put("moes_y", mx.array(sy.astype(np.float32)))
    meta["moes_n"] = SN
    meta["moes_counts"] = counts

    meta.update(moe_experts=EXPERTS, moe_rows=EROWS, moe_cols=ECOLS, moe_n=ROWS,
                moe_onehot_j=int(rng.integers(ECOLS)))
    print(f"moe     [{EXPERTS}, {EROWS}, {ECOLS}]  {ROWS} routed rows")

    meta["cases"] = [c[0] for c in CASES]
    case.note(**meta)
    case.write()


if __name__ == "__main__":
    main()
