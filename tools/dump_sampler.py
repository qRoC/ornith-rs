"""Reference for the sampler: every stage of `make_sampler(0.6, 0.95, 20)`.

The chain is long and each link has its own kernel, so each is dumped
separately -- a failure then says which one, instead of just "the token is
wrong":

    logprobs        logits - logsumexp(logits)
    exp             mx.exp(logprobs)
    argsort         mx.argsort(logprobs)                 <- multi-block sort
    gathered        take_along_axis(exp, argsort)
    cumsum          mx.cumsum(gathered)                  <- bfloat16 scan
    top_p           apply_top_p(logprobs, 0.95)
    negated         -top_p
    argsort_k       mx.argsort(negated)
    top_k           apply_top_k(top_p, 20)
    scaled          top_k * (1 / 0.6)
    weights         the where(isinf(m), ...) of categorical_inverse_cdf
    cdf             cumsum(weights, exclusive)           <- float32 scan
    total           sum(weights)
    maxval          max(x)
    token           the sampled id, for a run of draws off one seed

Run it as

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_sampler.py
"""

import pathlib

import mlx.core as mx

import ref

TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
SEED = 5
N_DRAWS = 24


def main():
    from mlx_lm.sample_utils import apply_top_k, apply_top_p, make_sampler
    from mlx_lm.utils import load

    model, tokenizer = load(pathlib.Path(ref.MODEL))
    case = ref.Case("sampler")

    prompt = "The capital of France is Paris, and the capital of Japan is"
    ids = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    logits = model(ids[None])[:, -1, :]
    case.put("logits", logits)

    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    case.put("logprobs", logprobs)

    # --- apply_top_p, stage by stage
    probs = mx.exp(logprobs)
    case.put("exp", probs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    case.put("argsort", sorted_indices.astype(mx.uint32))
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    case.put("gathered", sorted_probs)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    case.put("cumsum", cumulative)
    top_p_out = apply_top_p(logprobs, TOP_P)
    case.put("top_p", top_p_out)

    # --- apply_top_k, stage by stage
    negated = -top_p_out
    case.put("negated", negated)
    case.put("argsort_k", mx.argsort(negated, axis=-1).astype(mx.uint32))
    top_k_out = apply_top_k(top_p_out, TOP_K)
    case.put("top_k", top_k_out)

    # --- categorical_inverse_cdf, stage by stage
    scaled = top_k_out * (1.0 / TEMP)
    case.put("scaled", scaled)
    x = mx.reshape(scaled, (-1,)).astype(mx.float32)
    case.put("x", x)
    m = mx.max(x)
    case.put("maxval", m)
    w = mx.where(mx.isinf(m), mx.equal(x, m).astype(mx.float32), mx.exp(x - m))
    case.put("weights", w)
    case.put("cdf", mx.cumsum(w, axis=0, inclusive=False))
    case.put("total", mx.sum(w))

    # --- the bit generator on its own, so a wrong threefry is not hidden by
    #     a peaked distribution
    k = mx.random.key(SEED)
    keys, subs, uniforms = [k], [], []
    for _ in range(8):
        # `mx.random.split(key, 2)` is `bits({2, 2}, key)`, which is exactly
        # what `KeySequence::next` does before keeping row 0 and handing out
        # row 1.
        pair = mx.random.split(keys[-1], 2)
        keys.append(mx.reshape(pair[0], (2,)))
        subs.append(mx.reshape(pair[1], (2,)))
        uniforms.append(mx.random.uniform(shape=(1,), key=subs[-1]))
    case.put("rng_keys", mx.stack(keys))
    case.put("rng_subs", mx.stack(subs))
    case.put("rng_uniform", mx.concatenate(uniforms))

    # --- the draws themselves, off a seeded key sequence.
    #
    # Two distributions: the model's own, which after top-k and top-p is
    # peaked enough that every draw lands on the same token, and a flat
    # synthetic one where the draw actually depends on the random number.
    sampler = make_sampler(temp=TEMP, top_p=TOP_P, top_k=TOP_K)
    mx.random.seed(SEED)
    draws = []
    for _ in range(N_DRAWS):
        draws.append(int(sampler(logprobs)))
    case.put("tokens", mx.array(draws, dtype=mx.uint32))

    mx.random.seed(11)
    flat = mx.random.normal(logprobs.shape, dtype=mx.float32) * 3.0
    flat = (flat - mx.logsumexp(flat, keepdims=True)).astype(mx.bfloat16)
    case.put("flat_logprobs", flat)
    case.put("flat_top_p", apply_top_p(flat, TOP_P))
    case.put("flat_top_k", apply_top_k(apply_top_p(flat, TOP_P), TOP_K))
    mx.random.seed(SEED)
    flat_draws = [int(sampler(flat)) for _ in range(N_DRAWS)]
    case.put("flat_tokens", mx.array(flat_draws, dtype=mx.uint32))

    # The same sampler driven over a real generation, which is the shape the
    # REPL runs in: a fresh key sequence per run, one draw per step.
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "In one sentence, what is a neutron star?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    mx.random.seed(SEED)
    toks, out_text = [], ""
    for r in stream_generate(
        model,
        tokenizer,
        tokenizer.encode(text, add_special_tokens=False),
        max_tokens=64,
        sampler=make_sampler(temp=TEMP, top_p=TOP_P, top_k=TOP_K),
        prompt_cache=make_prompt_cache(model),
    ):
        toks.append(int(r.token))
        out_text += r.text
    case.put_json("run_prompt", text)
    case.put_json("run_ids", toks)
    case.put_json("run_text", out_text)

    case.note(
        temp=TEMP, top_p=TOP_P, top_k=TOP_K, seed=SEED, n_draws=N_DRAWS, max_tokens=64
    )
    case.write()
    print("draws:", draws, flush=True)
    print("flat draws:", flat_draws, flush=True)


if __name__ == "__main__":
    main()
