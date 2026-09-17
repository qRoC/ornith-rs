"""Reference outputs for the ops the vision tower adds.

Everything here is a primitive the text path never reached:

  layer_norm      mx.fast.layer_norm over hidden 1152 and merged 4608
  gelu / gelu_tanh over *every* bfloat16 value, both approximations
  addmm           nn.Linear's biased form at each of the tower's shapes,
                  covering the split-K and the regular steel gemm
  conv3d          the patch embedding's mx.conv3d, 3 -> 1152 channels
  sdpa            MTPLX's `_fused_sdpa`: pad head dim 72 -> 80, fused full
                  attention, slice back

Run with the vendored mlx-lm on the path:

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_vision_ops.py
"""

import mlx.core as mx
import mlx.nn as nn

import ref

HIDDEN = 1152
HEADS = 16
HEAD_DIM = HIDDEN // HEADS  # 72
INTERMEDIATE = 4304
MERGED = HIDDEN * 4  # 4608
OUT_HIDDEN = 2048
EPS = 1e-6

_FUSED_SDPA_DIMS = (64, 80, 128)


def fused_sdpa(q, k, v, scale):
    """MTPLX qwen3_vl_tower._fused_sdpa, verbatim."""
    d = q.shape[-1]
    target = next((t for t in _FUSED_SDPA_DIMS if d <= t), d)
    if target != d:
        pad = [(0, 0)] * (q.ndim - 1) + [(0, target - d)]
        q, k, v = mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)[..., :d]


def every_bfloat16():
    """All 65536 bfloat16 bit patterns, as a bfloat16 array."""
    return mx.arange(0, 1 << 16, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16)


def main():
    case = ref.Case("vision_ops")
    case.note(
        hidden=HIDDEN,
        heads=HEADS,
        head_dim=HEAD_DIM,
        intermediate=INTERMEDIATE,
        merged=MERGED,
        out_hidden=OUT_HIDDEN,
        eps=EPS,
    )
    mx.random.seed(3)

    # --- layer_norm -------------------------------------------------------
    # Real weights and biases: norm1 of block 0 (1152) and the merger's norm.
    names = [
        "vision_tower.blocks.0.norm1.weight",
        "vision_tower.blocks.0.norm1.bias",
        "vision_tower.merger.norm.weight",
        "vision_tower.merger.norm.bias",
    ]
    t = ref.load_tensors(names)
    case.put("ln_weight", t[names[0]])
    case.put("ln_bias", t[names[1]])
    for n in [1, 256, 280]:
        x = mx.random.normal((n, HIDDEN)).astype(mx.bfloat16)
        case.put(f"ln_x_{n}", x)
        case.put(f"ln_y_{n}", mx.fast.layer_norm(x, t[names[0]], t[names[1]], EPS))

    # --- gelu, both forms, over every bfloat16 input -----------------------
    allbf = every_bfloat16()
    case.put("gelu_x", allbf)
    case.put("gelu_exact", nn.gelu(allbf))
    case.put("gelu_tanh", nn.gelu_approx(allbf))

    # --- addmm: nn.Linear with a bias -------------------------------------
    # The shapes are the tower's, and M is chosen so that both the split-K
    # branch of `steel_matmul_axpby` (small output tile, long K) and the
    # regular steel gemm are exercised. `_tm * _tn <= 2048 && K >= max(M, N)`
    # is the test; for [M, 1152] x [1152, 1152] that is M <= 448.
    shapes = [
        ("qkv", 256, HIDDEN, 3 * HIDDEN),
        ("qkv_odd", 280, HIDDEN, 3 * HIDDEN),
        ("proj_splitk", 256, HIDDEN, HIDDEN),
        ("proj_regular", 1024, HIDDEN, HIDDEN),
        ("fc1", 280, HIDDEN, INTERMEDIATE),
        ("fc2_splitk", 280, INTERMEDIATE, HIDDEN),
        ("fc2_regular", 1024, INTERMEDIATE, HIDDEN),
        ("merge1_splitk", 64, MERGED, MERGED),
        ("merge1_regular", 256, MERGED, MERGED),
        ("merge2", 256, MERGED, OUT_HIDDEN),
    ]
    for tag, m, k, n in shapes:
        x = mx.random.normal((m, k)).astype(mx.bfloat16)
        w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        b = (mx.random.normal((n,)) * 0.02).astype(mx.bfloat16)
        case.put(f"addmm_{tag}_x", x)
        case.put(f"addmm_{tag}_w", w)
        case.put(f"addmm_{tag}_b", b)
        case.put(f"addmm_{tag}_y", mx.addmm(b, x, w.T))

    # --- conv3d: the patch embedding --------------------------------------
    conv_names = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.patch_embed.proj.bias",
    ]
    ct = ref.load_tensors(conv_names)
    cw = ct[conv_names[0]]  # already [O, kD, kH, kW, C] in this checkpoint
    cb = ct[conv_names[1]]
    case.put("conv_w", cw)
    case.put("conv_b", cb)
    for n in [256, 280, 1024]:
        # `PatchEmbed.__call__`'s input: the preprocessor's [N, C*T*P*P] rows.
        flat = mx.random.normal((n, 3 * 2 * 16 * 16)).astype(mx.bfloat16)
        case.put(f"conv_x_{n}", flat)
        h = flat.reshape(-1, 3, 2, 16, 16).moveaxis(1, 4)
        y = mx.conv3d(h, cw, (2, 16, 16), (0, 0, 0), (1, 1, 1))
        case.put(f"conv_y_{n}", y.reshape(n, -1))
        case.put(f"conv_yb_{n}", (y + cb).reshape(n, -1))

    # --- fused sdpa at the vision head dim --------------------------------
    scale = HEAD_DIM ** -0.5
    case.note(sdpa_scale=scale)
    for n in [256, 280, 1024]:
        q = mx.random.normal((1, HEADS, n, HEAD_DIM)).astype(mx.bfloat16)
        k = mx.random.normal((1, HEADS, n, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal((1, HEADS, n, HEAD_DIM)).astype(mx.bfloat16)
        case.put(f"sdpa_q_{n}", q)
        case.put(f"sdpa_k_{n}", k)
        case.put(f"sdpa_v_{n}", v)
        case.put(f"sdpa_y_{n}", fused_sdpa(q, k, v, scale))

    case.write()


if __name__ == "__main__":
    main()
