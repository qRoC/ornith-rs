"""Reference outputs for mx.quantized_matmul on this model's real weights.

Covers both quantization settings the artifact uses and all three dispatch
branches mlx picks by row count on an M4 (`applegpu_g16s`, gen 16):

    M = 1                -> affine_qmv_fast / affine_qmv
    2 <= M < batch_limit -> affine_qmv_wide
    M >= batch_limit     -> affine_qmm_t_splitk (then a sum over split_k)

Weights chosen:
  in_proj_qkv  [8192, 2048]  8-bit g64   -- the widest linear_attn projection
  out_proj     [2048, 4096]  8-bit g64   -- K = 4096 crosses the 4096 limit tier
  mlp.gate     [256, 2048]   8-bit g64   -- tiny N, the router
  switch_mlp.gate_proj[0] [512, 2048] 4-bit g32 -- one expert, the 4-bit path
"""

import mlx.core as mx

import ref

L = 0
NAMES = {
    "in_proj_qkv": f"language_model.model.layers.{L}.linear_attn.in_proj_qkv",
    "out_proj": f"language_model.model.layers.{L}.linear_attn.out_proj",
    "gate": f"language_model.model.layers.{L}.mlp.gate",
    "expert0": f"language_model.model.layers.{L}.mlp.switch_mlp.gate_proj",
}

# Row counts: 1 hits qmv, 4 hits qmv_wide (nv=4), 8 hits qmv_wide (nv=4,
# 2 tiles), 16 and 64 cross the batch limit into the matrix kernels.
ROWS = [1, 2, 4, 8, 16, 64, 256]


def main():
    cfg = ref.config()
    keys = []
    for base in NAMES.values():
        keys += [f"{base}.weight", f"{base}.scales", f"{base}.biases"]
    tensors = ref.load_tensors(keys)

    case = ref.Case("quantized_matmul")
    mx.random.seed(0)

    for tag, base in NAMES.items():
        w = tensors[f"{base}.weight"]
        scales = tensors[f"{base}.scales"]
        biases = tensors[f"{base}.biases"]
        group_size, bits = ref.quant_config(base, cfg)

        if tag == "expert0":
            # A 3-D switch weight: take expert 0 so this exercises the plain
            # 2-D quantized_matmul path with 4-bit/g32 numbers.
            w, scales, biases = w[0], scales[0], biases[0]

        out_dims = w.shape[-2]
        in_dims = scales.shape[-1] * group_size
        case.note(**{f"{tag}": {"path": base, "group_size": group_size, "bits": bits,
                                "out_dims": out_dims, "in_dims": in_dims}})

        for m in ROWS:
            x = mx.random.normal((m, in_dims)).astype(mx.bfloat16)
            y = mx.quantized_matmul(
                x, w, scales=scales, biases=biases,
                transpose=True, group_size=group_size, bits=bits, mode="affine",
            )
            case.put(f"{tag}_x_m{m}", x)
            case.put(f"{tag}_y_m{m}", y)

    case.write()


if __name__ == "__main__":
    main()
