"""Reference for the whole generation loop: mlx_lm.generate, greedy.

This is the end-to-end check -- prompt text in, token ids and printed text
out -- so it covers the tokenizer, the chat template, `generate_step`'s
prefill chunking, the sampler and the streaming detokenizer at once.

`stream_generate` is driven directly rather than `generate` so the token ids
can be recorded alongside the text, and the text is accumulated exactly the
way `generate` does it, from `response.text`.

Run it as

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_generate.py
"""

import pathlib

import ref

MAX_TOKENS = 96

RAW_PROMPTS = [
    "The capital of France is Paris, and the capital of Japan is",
    "def fibonacci(n):",
    "1, 1, 2, 3, 5, 8,",
]

CHATS = [
    ([{"role": "user", "content": "In one sentence, what is a neutron star?"}], False),
    ([{"role": "user", "content": "In one sentence, what is a neutron star?"}], True),
    (
        [
            {"role": "system", "content": "You answer in exactly three words."},
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
        False,
    ),
    (
        [
            {"role": "user", "content": "Name three primes."},
            {"role": "assistant", "content": "2, 3, 5."},
            {"role": "user", "content": "And three more?"},
        ],
        False,
    ),
    # long enough to cross the 2048-token prefill step
    (
        [{"role": "user", "content": "Summarize this:\n\n" + ("lorem ipsum dolor sit amet. " * 400)}],
        False,
    ),
]


def main():
    import mlx.core as mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    model, tokenizer = load(pathlib.Path(ref.MODEL))
    case = ref.Case("generate")

    prompts, texts, ids, kinds = [], [], [], []

    def run(prompt_text, tag):
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        text = ""
        toks = []
        cache = make_prompt_cache(model)
        for r in stream_generate(
            model,
            tokenizer,
            prompt_ids,
            max_tokens=MAX_TOKENS,
            sampler=make_sampler(temp=0.0),
            prompt_cache=cache,
        ):
            text += r.text
            toks.append(int(r.token))
        prompts.append(prompt_text)
        texts.append(text)
        # `stream_generate` yields the stopping token in its final response as
        # well, so the last id repeats unless it stopped on length.
        ids.append(toks)
        kinds.append(tag)
        print(f"{tag}: {len(toks)} tokens", flush=True)

    for p in RAW_PROMPTS:
        run(p, "raw")

    for messages, think in CHATS:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )
        run(text, f"chat:thinking={think}")

    case.put_json("prompts", prompts)
    case.put_json("texts", texts)
    case.put_json("ids", ids)
    case.put_json("kinds", kinds)
    case.note(max_tokens=MAX_TOKENS, n=len(prompts))
    case.write()


if __name__ == "__main__":
    main()
