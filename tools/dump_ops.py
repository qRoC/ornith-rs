"""Reference outputs for the elementwise / norm / rope ops.

Shapes are the ones the model actually evaluates:

  rms_norm   [1, T, 2048]      the decoder residual stream (input_layernorm)
             [1, T, 16, 128]   GatedDeltaNet's q/k normalisation, weightless
             [1, T, 16, 256]   attention q_norm / k_norm, per head_dim
  rope       [1, 16, T, 256]   queries after the head transpose, dims=64
  swiglu etc on MoE-sized activations
"""

import mlx.core as mx
import mlx.nn as nn

import ref
from mlx_lm.models.activations import precise_swiglu, swiglu

L = 0


def main():
    cfg = ref.config()["text_config"]
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_parameters"]["rope_theta"]
    head_dim = cfg["head_dim"]
    dims = int(head_dim * cfg["partial_rotary_factor"])

    names = [
        f"language_model.model.layers.{L}.input_layernorm.weight",
        f"language_model.model.layers.{L}.self_attn.q_norm.weight",
        f"language_model.model.layers.{L+3}.self_attn.q_norm.weight",
        f"language_model.model.layers.{L}.linear_attn.norm.weight",
    ]
    names = [n for n in names if True]
    tensors = ref.load_tensors([n for n in names if n != names[1]])

    case = ref.Case("ops")
    case.note(eps=eps, rope_theta=theta, rope_dims=dims, head_dim=head_dim)
    mx.random.seed(1)

    # --- rms_norm with a real weight -------------------------------------
    w = tensors[f"language_model.model.layers.{L}.input_layernorm.weight"]
    case.put("ln_weight", w)
    for t in [1, 7, 512]:
        x = mx.random.normal((1, t, 2048)).astype(mx.bfloat16)
        case.put(f"rms_x_t{t}", x)
        case.put(f"rms_y_t{t}", mx.fast.rms_norm(x, w, eps))

    # --- rms_norm without a weight (GatedDeltaNet q/k) --------------------
    # qwen3_5.GatedDeltaNet: rms_norm(q, None, 1e-6) over head_k_dim = 128.
    for t in [1, 7]:
        q = mx.random.normal((1, t, 16, 128)).astype(mx.bfloat16)
        case.put(f"rms_nw_x_t{t}", q)
        case.put(f"rms_nw_y_t{t}", mx.fast.rms_norm(q, None, 1e-6))

    # --- rms_norm over the attention head_dim ----------------------------
    qn = tensors[f"language_model.model.layers.{L+3}.self_attn.q_norm.weight"]
    case.put("q_norm_weight", qn)
    for t in [1, 7]:
        q = mx.random.normal((1, t, 16, 256)).astype(mx.bfloat16)
        case.put(f"rms_hd_x_t{t}", q)
        case.put(f"rms_hd_y_t{t}", mx.fast.rms_norm(q, qn, eps))

    # --- rope -------------------------------------------------------------
    rope = nn.RoPE(dims, traditional=False, base=theta, scale=1.0)
    for t, off in [(1, 0), (1, 37), (7, 0), (64, 128)]:
        q = mx.random.normal((1, 16, t, 256)).astype(mx.bfloat16)
        case.put(f"rope_x_t{t}_o{off}", q)
        case.put(f"rope_y_t{t}_o{off}", rope(q, offset=off))

    # --- elementwise ------------------------------------------------------
    a = mx.random.normal((1, 7, 512)).astype(mx.bfloat16)
    b = mx.random.normal((1, 7, 512)).astype(mx.bfloat16)
    case.put("ew_a", a)
    case.put("ew_b", b)
    case.put("ew_swiglu", swiglu(a, b))
    case.put("ew_precise_swiglu", precise_swiglu(a, b, a))
    case.put("ew_add", a + b)

    g = mx.random.normal((1, 7, 512)).astype(mx.bfloat16)
    case.put("ew_gate", g)
    case.put("ew_sigmoid_gate", mx.sigmoid(g) * b)

    g1 = mx.random.normal((1, 7, 1)).astype(mx.bfloat16)
    case.put("ew_gate1", g1)
    case.put("ew_sigmoid_gate1", mx.sigmoid(g1) * b)

    case.write()


if __name__ == "__main__":
    main()
