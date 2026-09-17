"""Reference for the whole language model: per-layer outputs and logits.

Loads the artifact through mlx-lm exactly as the runtime would, then walks the
decoder by hand so every layer's output can be captured. Two passes are dumped:

  prefill  a multi-token step from an empty cache (T tokens at once)
  decode   the next single token, against the cache the prefill left behind

which between them exercise both sides of every shape-dependent branch: the
fused vs unfused attention split at query length 8, the MoE sort threshold at
64 (token, expert) pairs, qmv vs the split-K gemm, and the gated-delta packed
kernel at T and at 1.
"""

import mlx.core as mx

import ref

T = 24
DECODE_STEPS = 3


def main():
    from mlx_lm.utils import load

    model, tokenizer = load(ref.MODEL)
    lm = model.language_model
    inner = lm.model

    case = ref.Case("model")
    mx.random.seed(5)

    prompt = "The capital of France is Paris, and the capital of Japan is"
    ids = mx.array(tokenizer.encode(prompt))[None, :T]
    if ids.shape[1] < T:
        ids = mx.concatenate([ids, mx.zeros((1, T - ids.shape[1]), dtype=ids.dtype)], axis=1)
    case.put("ids", ids.astype(mx.uint32))
    case.note(n_layers=len(inner.layers), seq=int(ids.shape[1]), decode_steps=DECODE_STEPS)

    cache = lm.make_cache()

    def run(tag, tokens):
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        h = inner.embed_tokens(tokens)
        case.put(f"{tag}_embed", h)
        fa_mask = create_attention_mask(h, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)
            # Dumping all 40 would be 40 x seq x 2048; keep the ones that
            # bracket each layer kind plus the last.
            if i in (0, 1, 2, 3, 4, 7, 8, 39):
                case.put(f"{tag}_layer{i}", h)
        pre_norm = h
        h = inner.norm(h)
        case.put(f"{tag}_hidden", h)
        case.put(f"{tag}_prenorm", pre_norm)
        logits = lm.lm_head(h)
        case.put(f"{tag}_logits", logits)
        return logits

    logits = run("prefill", ids)
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    for step in range(DECODE_STEPS):
        case.put(f"decode{step}_id", tok.astype(mx.uint32))
        logits = run(f"decode{step}", tok[:, None])
        tok = mx.argmax(logits[:, -1, :], axis=-1)
    case.put("next_id", tok.astype(mx.uint32))
    print("greedy continuation:", tokenizer.decode([int(t) for t in [tok]]), flush=True)

    case.write()


if __name__ == "__main__":
    main()
