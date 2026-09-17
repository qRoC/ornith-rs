"""Reference for the chat template: `apply_chat_template` over many shapes.

The Rust side hand-compiles `chat_template.jinja`; this renders the same
conversations through the real jinja template, exactly the way mlx-lm does
(`TokenizerWrapper.apply_chat_template`, which fills in `enable_thinking`
when the caller does not).

Run it as

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_chat.py
"""

import ref

SYS = "You are a helpful assistant."
LONG = "  padded   \n\n"


def cases():
    """`(messages, add_generation_prompt, enable_thinking)` triples."""
    u = lambda s: {"role": "user", "content": s}
    a = lambda s, **kw: {"role": "assistant", "content": s, **kw}
    s = lambda t: {"role": "system", "content": t}
    t = lambda x: {"role": "tool", "content": x}

    convos = [
        [u("hi")],
        [u("hi"), a("hello"), u("and again")],
        [s(SYS), u("hi")],
        [s(SYS), u("hi"), a("hello"), u("more")],
        [s(LONG), u(LONG), a(LONG)] + [u("go")],
        [s(""), u("hi")],
        [s(None), u("hi")],
        [u("hi"), a("<think>\nreasoning here\n</think>\n\nthe answer")],
        [u("hi"), a("<think>\nreasoning here\n</think>\n\nthe answer"), u("next")],
        [u("hi"), a("no think block at all"), u("next")],
        [u("hi"), a("</think>\n\nonly a close tag"), u("next")],
        [u("hi"), a("a<think>b</think>c<think>d</think>e"), u("next")],
        [u("hi"), a("the answer", reasoning_content="explicit reasoning"), u("next")],
        [u("hi"), a("<think>x</think>y", reasoning_content="  wins  "), u("next")],
        [u("hi"), a("", reasoning_content=""), u("next")],
        # tool messages, which the template stitches into one user turn
        [u("weather?"), a("calling"), t("{\"t\": 20}"), u("thanks")],
        [u("weather?"), a("calling"), t("a"), t("b"), u("thanks")],
        [t("leading tool message"), u("hi")],
        [u("weather?"), a("calling"), t("a"), t("b")],
        # a user turn that is only a tool_response does not count as a query
        [u("<tool_response>\nx\n</tool_response>"), u("a real question")],
        # content as a list, the vision shape
        [{"role": "user", "content": [{"type": "text", "text": "what is this?"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]}],
        [{"role": "user", "content": [{"image": "a.png"}, {"text": "and this"}]}],
        [{"role": "user", "content": [{"image_url": "http://x/a.png"}]}],
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "?"}]}],
        [{"role": "user", "content": [{"video": "v.mp4"}]}],
        [
            {"role": "system", "content": [{"type": "text", "text": "sys as list"}]},
            u("hi"),
        ],
        [{"role": "user", "content": []}],
        # unicode and whitespace through |trim
        [u(" 　 spaced  "), a("\t\ttabbed\n")],
        [u("यह हिन्दी"), a("\U0001f600")],
    ]

    out = []
    for c in convos:
        for add_gen in (True, False):
            for think in (True, False):
                out.append((c, add_gen, think))
    return out


def main():
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    import pathlib

    tok = load_tokenizer(pathlib.Path(ref.MODEL))

    case = ref.Case("chat")
    ins, outs, ids = [], [], []
    for messages, add_gen, think in cases():
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_gen,
            enable_thinking=think,
        )
        ins.append(
            {"messages": messages, "add_generation_prompt": add_gen, "enable_thinking": think}
        )
        outs.append(text)
        ids.append(tok.encode(text, add_special_tokens=False))

    case.put_json("cases", ins)
    case.put_json("rendered", outs)
    case.put_json("ids", ids)
    case.note(n=len(ins))
    case.write()


if __name__ == "__main__":
    main()
