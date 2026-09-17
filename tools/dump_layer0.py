"""Layer 0 (a gated-delta layer) broken into its steps, with real weights."""

import mlx.core as mx

import ref

T = 24


def main():
    from mlx_lm.utils import load

    model, tokenizer = load(ref.MODEL)
    inner = model.language_model.model
    layer = inner.layers[0]
    gdn = layer.linear_attn

    case = ref.Case("layer0")
    prompt = "The capital of France is Paris, and the capital of Japan is"
    ids = mx.array(tokenizer.encode(prompt))[None, :T]
    ids = mx.concatenate([ids, mx.zeros((1, T - ids.shape[1]), dtype=ids.dtype)], axis=1)[:, :T]

    x = inner.embed_tokens(ids)
    case.put("x", x)
    normed = layer.input_layernorm(x)
    case.put("normed", normed)

    B, S, _ = normed.shape
    qkv = gdn.in_proj_qkv(normed)
    z = gdn.in_proj_z(normed).reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
    b = gdn.in_proj_b(normed)
    a = gdn.in_proj_a(normed)
    case.put("qkv", qkv)
    case.put("z", z)
    case.put("b", b)
    case.put("a", a)

    conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=normed.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    import mlx.nn as nn
    case.put("conv_in", conv_input)
    conv_raw = gdn.conv1d(conv_input)
    case.put("conv_raw", conv_raw)
    conv_out = nn.silu(conv_raw)
    case.put("conv_out", conv_out)

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    case.put("q", q)
    case.put("k", k)
    case.put("v", v)

    from mlx_lm.models.gated_delta import gated_delta_update
    state = mx.zeros((B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32)
    out, state = gated_delta_update(q, k, v, a, b, gdn.A_log, gdn.dt_bias, state, None)
    case.put("gdn_out", out)
    case.put("gdn_state", state)

    out = gdn.norm(out, z)
    case.put("normed_gated", out)
    r = gdn.out_proj(out.reshape(B, S, -1))
    case.put("r", r)
    case.put("h", x + r)
    case.write()


if __name__ == "__main__":
    main()
