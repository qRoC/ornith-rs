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

import os

import mlx.core as mx

import ref

T = 24
DECODE_STEPS = 3

# The default case is the golden reference `tests/model.rs` reads, and it is
# frozen. `ORNITH_MODEL_PROMPT` dumps the same walk over another text instead:
# the prompt's own tokens, all but the last as the prefill and the last as the
# one decode step -- which is exactly the path `dump_probes.py` puts a probe
# through, so a probe that disagrees between mlx and the port can be bisected
# layer by layer with the test that already exists.
PROMPT = os.environ.get("ORNITH_MODEL_PROMPT")
# The frozen case keeps eight layers -- enough to bracket each layer kind.
# `ORNITH_MODEL_LAYERS=all` keeps every one, which is what finds *which* layer
# a divergence starts in.
ALL_LAYERS = os.environ.get("ORNITH_MODEL_LAYERS") == "all"


def main():
    from mlx_lm.utils import load

    model, tokenizer = load(ref.MODEL)
    lm = model.language_model
    inner = lm.model

    case = ref.Case("model")
    mx.random.seed(5)

    prompt = PROMPT or "The capital of France is Paris, and the capital of Japan is"
    if PROMPT:
        # `dump_probes.py`'s call, and the probe's own path: every token is the
        # prompt's, and the last one is the step that produces the logits --
        # no padding, no argmax continuation.
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        ids = mx.array(encoded)[None, :-1]
        forced = [encoded[-1]]
    else:
        # The frozen case, unchanged: the reference in `refs/model` was written
        # by this branch and nothing here may move it.
        ids = mx.array(tokenizer.encode(prompt))[None, :T]
        forced = None
        if ids.shape[1] < T:
            ids = mx.concatenate([ids, mx.zeros((1, T - ids.shape[1]), dtype=ids.dtype)], axis=1)
    case.put("ids", ids.astype(mx.uint32))
    steps = len(forced) if forced else DECODE_STEPS
    case.note(n_layers=len(inner.layers), seq=int(ids.shape[1]), decode_steps=steps)

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
            if ALL_LAYERS or i in (0, 1, 2, 3, 4, 7, 8, 39):
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
    for step in range(steps):
        if forced:
            tok = mx.array([forced[step]])
        case.put(f"decode{step}_id", tok.astype(mx.uint32))
        logits = run(f"decode{step}", tok[:, None])
        tok = mx.argmax(logits[:, -1, :], axis=-1)
    case.put("next_id", tok.astype(mx.uint32))
    print("greedy continuation:", tokenizer.decode([int(t) for t in [tok]]), flush=True)

    case.write()


if __name__ == "__main__":
    main()
