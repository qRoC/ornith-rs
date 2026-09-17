"""Reference outputs for mx.quantized_matmul / mx.gather_qmm in nvfp4 mode.

The twin of dump_quantized.py for the other mode this port carries. It covers
the same dispatch branches, plus the two gathered ones, because nvfp4 moves
along a few edges affine does not:

    M = 1                -> nvfp4_qmv_fast / nvfp4_qmv
    2 <= M < batch_limit -> nvfp4_qmv_wide          (k_lanes 16, not affine's 8)
    M >= batch_limit     -> nvfp4_qmm_t_splitk      (no third buffer slot)
    gathered, R small    -> nvfp4_gather_qmv_fast
    gathered, R sorted   -> nvfp4_gather_qmm_rhs_nt
    round trip           -> nvfp4_dequantize

**This does not need an nvfp4 artifact.** It requantizes the control
artifact's own weights, so the kernels and the dispatch can be checked before
spending a build on them; the weights and scales go into the reference so the
Rust side reads exactly the tensors mlx computed against, and the test is
independent of which artifact happens to be on disk.

nvfp4 is two tensors -- there is no biases array and no global scale -- so
these references also pin the tensor count the loader asserts.
"""

import mlx.core as mx

import ref

L = 0
NAMES = {
    "in_proj_qkv": f"language_model.model.layers.{L}.linear_attn.in_proj_qkv",
    "out_proj": f"language_model.model.layers.{L}.linear_attn.out_proj",
    "gate": f"language_model.model.layers.{L}.mlp.gate",
}
# The 3-D switch weight, for the two gathered branches. 32 of the 256 experts
# keeps the reference to ~19 MB while still letting R/E >= 4 pick out
# gather_qmm_rhs.
SWITCH = f"language_model.model.layers.{L}.mlp.switch_mlp.gate_proj"
EXPERTS = 32

GROUP_SIZE, BITS, MODE = 16, 4, "nvfp4"

# Row counts: 1 hits qmv, 4 hits qmv_wide (nv=4), 8 hits qmv_wide (nv=4,
# 2 tiles), 16 and 64 cross the batch limit into the matrix kernels.
ROWS = [1, 2, 4, 8, 16, 64, 256]


def requantize(w, scales, biases, group_size, bits):
    """The artifact's affine weight, dequantized and re-cut as nvfp4.

    This is exactly what `mtplx forge build` does to the bf16 parent, one
    quantization step removed: the parent is not loadable on this host (67.0
    GiB against a 51.8 GiB working set), and the point here is to exercise the
    kernels on a realistic weight distribution, not to reproduce the forge.
    """
    dense = mx.dequantize(
        w, scales=scales, biases=biases, group_size=group_size, bits=bits, mode="affine"
    )
    out = mx.quantize(dense, GROUP_SIZE, BITS, mode=MODE)
    assert len(out) == 2, f"nvfp4 must return (weight, scales), got {len(out)} arrays"
    qw, qs = out
    assert qs.dtype == mx.uint8, f"nvfp4 scales must be uint8 E4M3, got {qs.dtype}"
    return dense, qw, qs


def main():
    cfg = ref.config()
    keys = []
    for base in list(NAMES.values()) + [SWITCH]:
        keys += [f"{base}.weight", f"{base}.scales", f"{base}.biases"]
    tensors = ref.load_tensors(keys)

    case = ref.Case("quantized_matmul_nvfp4")
    mx.random.seed(0)

    for tag, base in NAMES.items():
        group_size, bits = ref.quant_config(base, cfg)
        dense, qw, qs = requantize(
            tensors[f"{base}.weight"],
            tensors[f"{base}.scales"],
            tensors[f"{base}.biases"],
            group_size,
            bits,
        )
        out_dims, in_dims = dense.shape
        case.note(**{tag: {"path": base, "from_group_size": group_size, "from_bits": bits,
                           "out_dims": out_dims, "in_dims": in_dims}})
        case.put(f"{tag}_w", qw)
        case.put(f"{tag}_scales", qs)

        for m in ROWS:
            x = mx.random.normal((m, in_dims)).astype(mx.bfloat16)
            y = mx.quantized_matmul(
                x, qw, scales=qs, biases=None,
                transpose=True, group_size=GROUP_SIZE, bits=BITS, mode=MODE,
            )
            case.put(f"{tag}_x_m{m}", x)
            case.put(f"{tag}_y_m{m}", y)

        # The format itself, independent of any matmul: one dequantize of the
        # first 64 rows, which is what `nvfp4_dequantize` has to reproduce.
        case.put(
            f"{tag}_deq",
            mx.dequantize(
                qw[:64], scales=qs[:64], biases=None,
                group_size=GROUP_SIZE, bits=BITS, mode=MODE,
            ),
        )

    # --- the gathered pair --------------------------------------------------
    group_size, bits = ref.quant_config(SWITCH, cfg)
    sw, ss, sb = (tensors[f"{SWITCH}.{k}"] for k in ("weight", "scales", "biases"))
    dense, qw, qs = requantize(sw[:EXPERTS], ss[:EXPERTS], sb[:EXPERTS], group_size, bits)
    _, out_dims, in_dims = dense.shape
    case.note(switch={"path": SWITCH, "experts": EXPERTS,
                      "out_dims": out_dims, "in_dims": in_dims})
    case.put("switch_w", qw)
    case.put("switch_scales", qs)

    # R = 8 over 32 experts fails `B >= 16 && B/E >= 4`, so GatherQMM takes the
    # vector path; R = 256 satisfies it and takes the rhs-sorted gemm. Both are
    # M == 1, which is the only shape SwitchGLU produces.
    #
    # **`lhs_indices` must be withheld for the sorted case.** `gather_qmm` in
    # ops.cpp sets `right_sorted_ = sorted_indices && !lhs_indices_`, so
    # passing an explicit lhs -- even the identity `arange(rows)` -- turns the
    # rhs-sorted branch off and quietly routes the reference through
    # `gather_qmv` instead. The two kernels accumulate in different orders, so
    # the result is a reference that is right to about 1e-3 and wrong for a
    # bit-exact comparison. The port's `gather_qmm_rows` mirrors mlx: it takes
    # lhs and ignores it when `sorted`, which is the same `!lhs_indices` case.
    for rows, sorted_indices in [(8, False), (256, True)]:
        x = mx.random.normal((rows, in_dims)).astype(mx.bfloat16)
        rhs = mx.arange(rows, dtype=mx.uint32) % EXPERTS
        if sorted_indices:
            rhs = mx.sort(rhs)
        lhs = mx.arange(rows, dtype=mx.uint32)
        y = mx.gather_qmm(
            x.reshape(rows, 1, in_dims), qw, scales=qs, biases=None,
            lhs_indices=None if sorted_indices else lhs,
            rhs_indices=rhs, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, mode=MODE,
            sorted_indices=sorted_indices,
        ).reshape(rows, out_dims)
        tag = "sorted" if sorted_indices else "unsorted"
        case.put(f"switch_{tag}_x", x)
        case.put(f"switch_{tag}_lhs", lhs)
        case.put(f"switch_{tag}_rhs", rhs)
        case.put(f"switch_{tag}_y", y)

    case.write()


if __name__ == "__main__":
    main()
