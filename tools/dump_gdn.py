"""Reference outputs for the gated delta rule and its scalar parts.

Shapes are the model's: Hk=16, Hv=32, Dk=Dv=128, conv kernel 4, conv_dim
= 2*key_dim + value_dim = 2*2048 + 4096 = 8192.
"""

import mlx.core as mx
import mlx.nn as nn

import ref
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, gated_delta_update

L = 0
P = f"language_model.model.layers.{L}.linear_attn"


def main():
    cfg = ref.config()["text_config"]
    hk, hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    kk = cfg["linear_conv_kernel_dim"]

    t = ref.load_tensors([f"{P}.A_log", f"{P}.dt_bias", f"{P}.conv1d.weight"])
    a_log, dt_bias, convw = t[f"{P}.A_log"], t[f"{P}.dt_bias"], t[f"{P}.conv1d.weight"]

    case = ref.Case("gdn")
    case.note(hk=hk, hv=hv, dk=dk, dv=dv, conv_kernel=kk,
              conv_dim=2 * hk * dk + hv * dv)
    case.put("A_log", a_log)
    case.put("dt_bias", dt_bias)
    case.put("conv1d_weight", convw)
    mx.random.seed(2)

    conv_dim = 2 * hk * dk + hv * dv
    for T in [1, 7, 512]:
        # --- conv1d: the cached state is concatenated ahead of the sequence
        state = mx.random.normal((1, kk - 1, conv_dim)).astype(mx.bfloat16)
        qkv = mx.random.normal((1, T, conv_dim)).astype(mx.bfloat16)
        conv_in = mx.concatenate([state, qkv], axis=1)
        conv = nn.Conv1d(conv_dim, conv_dim, bias=False, kernel_size=kk,
                         groups=conv_dim, padding=0)
        conv.weight = convw
        case.put(f"conv_state_t{T}", state)
        case.put(f"conv_qkv_t{T}", qkv)
        case.put(f"conv_in_t{T}", conv_in)
        case.put(f"conv_out_t{T}", nn.silu(conv(conv_in)))

        # --- the gate and beta
        a = mx.random.normal((1, T, hv)).astype(mx.bfloat16)
        b = mx.random.normal((1, T, hv)).astype(mx.bfloat16)
        case.put(f"a_t{T}", a)
        case.put(f"b_t{T}", b)
        case.put(f"g_t{T}", compute_g(a_log, a, dt_bias))
        case.put(f"beta_t{T}", mx.sigmoid(b))

        # --- the recurrence, from a non-zero state so the decay is exercised
        q = mx.random.normal((1, T, hk, dk)).astype(mx.bfloat16)
        k = mx.random.normal((1, T, hk, dk)).astype(mx.bfloat16)
        v = mx.random.normal((1, T, hv, dv)).astype(mx.bfloat16)
        s0 = (mx.random.normal((1, hv, dv, dk)) * 0.1).astype(mx.float32)
        case.put(f"q_t{T}", q)
        case.put(f"k_t{T}", k)
        case.put(f"v_t{T}", v)
        case.put(f"state_in_t{T}", s0)
        y, s1 = gated_delta_update(q, k, v, a, b, a_log, dt_bias, s0, None,
                                   use_kernel=True)
        case.put(f"y_t{T}", y)
        case.put(f"state_out_t{T}", s1)

        # --- and the unpacked kernel, to pin the fallback too
        g = compute_g(a_log, a, dt_bias)
        from mlx_lm.models.gated_delta import gated_delta_kernel_unpacked
        yu, su = gated_delta_kernel_unpacked(q, k, v, g, mx.sigmoid(b), s0, None)
        case.put(f"y_unpacked_t{T}", yu)
        case.put(f"state_out_unpacked_t{T}", su)

    case.write()


if __name__ == "__main__":
    main()
