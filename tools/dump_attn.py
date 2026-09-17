"""Reference outputs for matmul, softmax and scaled_dot_product_attention.

The attention shapes are the model's full-attention layers: 16 query heads,
2 kv heads, head_dim 256, partial rope over the first 64 channels.

Note which side of mlx's routing each case lands on: at head_dim 256
`use_fallback` sends query lengths > 8 to the unfused softmax(QK^T)V path and
keeps <= 8 on the fused vector kernels.
"""

import mlx.core as mx

import ref

B, HQ, HKV, D = 1, 16, 2, 256


def main():
    case = ref.Case("attn")
    case.note(b=B, hq=HQ, hkv=HKV, d=D)
    mx.random.seed(3)
    scale = D ** -0.5
    case.note(scale=scale)

    # --- plain matmul, including a broadcast batch axis ------------------
    a = mx.random.normal((2, 3, 64, 128)).astype(mx.bfloat16)
    bb = mx.random.normal((2, 1, 128, 96)).astype(mx.bfloat16)
    case.put("mm_a", a)
    case.put("mm_b", bb)
    case.put("mm_out", a @ bb)

    at = mx.random.normal((2, 3, 64, 128)).astype(mx.bfloat16)
    bt = mx.random.normal((2, 3, 96, 128)).astype(mx.bfloat16)
    case.put("mmt_a", at)
    case.put("mmt_b", bt)
    case.put("mmt_out", at @ mx.swapaxes(bt, -1, -2))

    # --- softmax ---------------------------------------------------------
    s = mx.random.normal((8, 512)).astype(mx.bfloat16) * 4
    case.put("sm_x", s)
    case.put("sm_precise", mx.softmax(s, axis=-1, precise=True))
    case.put("sm_plain", mx.softmax(s, axis=-1, precise=False))

    # --- attention -------------------------------------------------------
    # Decode: one query row against a cache. 512 keys crosses mlx's 1024
    # threshold? No -- 1024 is the 2-pass cutoff, so 512 stays 1-pass and
    # 2048 goes 2-pass. Both are covered.
    for lq, lk, causal in [(1, 512, False), (1, 2048, False), (4, 4, True),
                           (7, 7, True), (64, 64, True), (512, 512, True)]:
        q = mx.random.normal((B, HQ, lq, D)).astype(mx.bfloat16)
        k = mx.random.normal((B, HKV, lk, D)).astype(mx.bfloat16)
        v = mx.random.normal((B, HKV, lk, D)).astype(mx.bfloat16)
        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask="causal" if causal else None
        )
        tag = f"q{lq}_k{lk}_{'c' if causal else 'nc'}"
        case.put(f"attn_q_{tag}", q)
        case.put(f"attn_k_{tag}", k)
        case.put(f"attn_v_{tag}", v)
        case.put(f"attn_o_{tag}", o)

    case.write()


if __name__ == "__main__":
    main()
