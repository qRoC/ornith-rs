"""Reference for the chat template: `apply_chat_template` over many shapes.

The Rust side hand-compiles `chat_template.jinja`; this renders the same
conversations through the real jinja template, exactly the way mlx-lm does
(`TokenizerWrapper.apply_chat_template`, which fills in `enable_thinking`
when the caller does not).

It also dumps a second, smaller case: `json.dumps(v, ensure_ascii=False)`
over a list of awkward values. That is the function `transformers` installs
as jinja's `tojson`, so the tools block and the `tool_calls` block are only
as exact as the port's JSON writer, and the writer deserves an oracle of its
own rather than one reached through two layers of template.

Run it as

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_chat.py
"""

import json

import ref

SYS = "You are a helpful assistant."
LONG = "  padded   \n\n"

# A schema with one of everything the writer has to get right: nesting, an
# empty object and an empty list, non-ASCII (`ensure_ascii=False` leaves it
# alone), an int and a float that are numerically equal, a bool and a null.
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "\u041f\u043e\u0433\u043e\u0434\u0430 \u0432 \u0433\u043e\u0440\u043e\u0434\u0435",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": 'quote " backslash \\ tab\t'},
                "unit": {"type": "string", "enum": ["c", "f"]},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
                "scale": {"type": "number", "default": 1.0, "examples": [0.5, 1e16]},
                "flag": {"type": "boolean", "default": False},
                "nothing": {"default": None},
                "empty": {"object": {}, "list": []},
            },
            "required": ["city"],
        },
    },
}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {}}}
# The key order here is not sorted, which is the point: `json.dumps` writes a
# dict in insertion order and so must the port.
UNSORTED = {"z": 1, "a": {"n": 2, "m": 3}, "0": [True, False, None]}

TOJSON_VALUES = [
    None,
    True,
    False,
    0,
    -0,
    2,
    -17,
    123456789012345678,
    2.0,
    -0.0,
    0.1,
    1 / 3,
    1e-4,
    1e-5,
    1e15,
    1e16,
    1e22,
    5e-324,
    1.7976931348623157e308,
    "",
    "plain",
    'quote " backslash \\ slash / newline \n tab\t return \r',
    "control \x00\x01\x1f delete \x7f",
    "bell \b formfeed \f",
    "\u0421\u0430\u043d\u043a\u0442-\u041f\u0435\u0442\u0435\u0440\u0431\u0443\u0440\u0433 \u65e5\u672c\u8a9e \U0001f600",
    "line separator \u2028 paragraph \u2029",
    [],
    {},
    [1, "two", None, [3.5], {"k": []}],
    UNSORTED,
    WEATHER,
]


def cases():
    """`(messages, add_generation_prompt, enable_thinking, tools)` tuples."""
    u = lambda s: {"role": "user", "content": s}
    a = lambda s, **kw: {"role": "assistant", "content": s, **kw}
    s = lambda t: {"role": "system", "content": t}
    t = lambda x: {"role": "tool", "content": x}
    call = lambda name, arguments: {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }

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
                out.append((c, add_gen, think, None))

    # The `tools` system block. Everything above is its `else`; these are the
    # branch, which also swallows the system message differently -- an empty
    # one adds nothing at all, not even the two newlines that separate a
    # non-empty one from the calling convention.
    tool_convos = [
        ([u("weather in Paris?")], [WEATHER]),
        ([u("weather?")], [WEATHER, SEARCH]),
        ([u("hi")], [SEARCH]),
        ([u("hi")], [UNSORTED]),
        ([s(SYS), u("hi")], [WEATHER]),
        ([s(""), u("hi")], [WEATHER]),
        ([s(LONG), u("hi")], [WEATHER]),
        ([s(None), u("hi")], [WEATHER]),
        (
            [{"role": "system", "content": [{"type": "text", "text": "sys as list"}]}, u("hi")],
            [SEARCH],
        ),
    ]

    # The assistant `tool_calls` block. What varies: whether the content in
    # front of it survived `|trim`, how many calls there are, whether the
    # call is wrapped in a `function` key, and what the argument values are.
    args = {
        "city": "Paris",
        "note": "a string goes in raw\nacross lines",
        "days": 3,
        "scale": 2.0,
        "ok": True,
        "none": None,
        "list": [1, "two"],
        "deep": {"a": [1, 2], "b": {}},
        "\u0413\u043e\u0440\u043e\u0434": "\u0421\u0430\u043d\u043a\u0442-\u041f\u0435\u0442\u0435\u0440\u0431\u0443\u0440\u0433",
    }
    call_convos = [
        ([u("weather?"), a("", tool_calls=[call("get_weather", args)]), u("thanks")], None),
        ([u("weather?"), a("calling", tool_calls=[call("get_weather", args)]), u("thanks")], None),
        ([u("weather?"), a("  ", tool_calls=[call("get_weather", {"city": "Paris"})]), u("ok")], None),
        # the flat shape, without the `function` wrapper
        ([u("weather?"), a("", tool_calls=[{"name": "search", "arguments": {"q": "x"}}]), u("ok")], None),
        # two calls, and one with no `arguments` key at all
        ([u("weather?"), a("first", tool_calls=[call("a", {"x": 1}), {"name": "b"}]), u("ok")], None),
        ([u("weather?"), a("", tool_calls=[{"name": "b", "arguments": {}}]), u("ok")], None),
        # what the think-split leaves behind is what decides the gap
        ([u("hi"), a("<think>\nwhy\n</think>\n\n", tool_calls=[call("a", {"x": 1})]), u("ok")], None),
        ([u("hi"), a("<think>\nwhy\n</think>\n\nthen this", tool_calls=[call("a", {"x": 1})]), u("ok")], None),
        ([u("hi"), a("", reasoning_content="explicit", tool_calls=[call("a", {"x": 1})]), u("ok")], None),
        # and both blocks at once, with a tool response in between
        (
            [
                u("weather?"),
                a("", tool_calls=[call("get_weather", {"city": "Paris"})]),
                t('{"t": 20}'),
                u("thanks"),
            ],
            [WEATHER],
        ),
    ]

    for c, tools in tool_convos + call_convos:
        for add_gen in (True, False):
            for think in (True, False):
                out.append((c, add_gen, think, tools))
    return out


def main():
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    import pathlib

    tok = load_tokenizer(pathlib.Path(ref.MODEL))

    case = ref.Case("chat")
    ins, outs, ids = [], [], []
    for messages, add_gen, think, tools in cases():
        kw = {} if tools is None else {"tools": tools}
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_gen,
            enable_thinking=think,
            **kw,
        )
        entry = {
            "messages": messages,
            "add_generation_prompt": add_gen,
            "enable_thinking": think,
        }
        if tools is not None:
            entry["tools"] = tools
        ins.append(entry)
        outs.append(text)
        ids.append(tok.encode(text, add_special_tokens=False))

    case.put_json("cases", ins)
    case.put_json("rendered", outs)
    case.put_json("ids", ids)
    # The writer on its own, one layer below the template.
    case.put_json("tojson_in", TOJSON_VALUES)
    case.put_json("tojson_out", [json.dumps(v, ensure_ascii=False) for v in TOJSON_VALUES])
    case.note(n=len(ins), tojson=len(TOJSON_VALUES))
    case.write()


if __name__ == "__main__":
    main()
