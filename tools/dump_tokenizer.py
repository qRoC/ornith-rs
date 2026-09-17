"""Reference for the tokenizer: character classes, NFC, and `encode`.

Three references, narrow to wide, so a failure says which layer broke:

  `classes`   for every one of the 0x110000 codepoints, which of the split
              regex's character classes oniguruma puts it in.  Derived from
              the *live* pre-tokenizer rather than from a property table, by
              feeding it `a<c><c>b` and reading the tiling back:

                  ["a<c><c>b"]         -> [\\p{L}\\p{M}]
                  ["a", c, c, "b"]     -> \\p{N}
                  ["a", c, "<c>b"]     -> \\s
                  ["a", "<c><c>", "b"] -> none of them

              which is what the four alternatives of the pattern do with a
              character of each class.  `\\r` and `\\n` are spelled in the
              pattern itself and are checked separately.

  `nfc`       the NFC normalizer's output over every codepoint and over a
              pile of combining sequences.  Taken from `tokenizers`' own
              normalizer, not from Python's `unicodedata`, since that is the
              implementation the tokenizer actually runs.

  `encode`    whole-pipeline `encode` over a corpus: prose in several
              scripts, code, whitespace torture, every byte value, added
              tokens in context, and the chat markers.  Dumped twice, because
              the two libraries disagree:

                `encode_json`  the tokenizer `tokenizer.json` describes,
                               driven straight from `tokenizers`.
                `encode_out`   `transformers`' `AutoTokenizer`, which is what
                               mlx-lm loads.  `Qwen2Tokenizer` in transformers
                               5.x rebuilds the tokenizer from `vocab.json`
                               and `merges.txt` and installs its own
                               `PRETOKENIZE_REGEX`, which is the older Qwen2
                               pattern with no `\p{M}` -- so anything with a
                               combining mark tokenizes differently.

Run it as

    cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_tokenizer.py
"""

import json
import os
import random
import sys

import mlx.core as mx

import ref

MAX = 0x110000


def surrogate(c):
    return 0xD800 <= c < 0xE000


def class_bits(model_dir):
    """Classify every codepoint by asking the real pre-tokenizer."""
    from tokenizers import Regex
    from tokenizers.pre_tokenizers import Split

    with open(os.path.join(model_dir, "tokenizer.json")) as f:
        pat = json.load(f)["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    sp = Split(Regex(pat), behavior="isolated", invert=False)

    # 0 = other, 1 = [\p{L}\p{M}], 2 = \p{N}, 3 = \s
    out = bytearray(MAX)
    for c in range(MAX):
        if surrogate(c):
            continue
        ch = chr(c)
        if ch in "\r\n":
            out[c] = 3  # spelled literally in the pattern; \s by inspection
            continue
        toks = [t[0] for t in sp.pre_tokenize_str("a" + ch + ch + "b")]
        if toks == ["a" + ch + ch + "b"]:
            out[c] = 1
        elif toks == ["a", ch, ch, "b"]:
            out[c] = 2
        elif toks == ["a", ch, ch + "b"]:
            out[c] = 3
        elif toks == ["a", ch + ch, "b"]:
            out[c] = 0
        else:
            raise AssertionError(f"U+{c:04X}: unexpected tiling {toks!r}")
        if c % 0x20000 == 0:
            print(f"  classes U+{c:06X}", file=sys.stderr, flush=True)
    return out


COMBINERS = [
    0x0300, 0x0301, 0x0302, 0x0308, 0x030A, 0x0327, 0x031B, 0x0334, 0x0345,
    0x0653, 0x093C, 0x09BE, 0x09D7, 0x0B3E, 0x0F71, 0x0F72, 0x0F80,
    0x102E, 0x1161, 0x11A8, 0x3099, 0x309A, 0x1D165, 0x1D16E, 0x0344, 0x05B0,
]

STARTERS = [
    0x0041, 0x0061, 0x0045, 0x004F, 0x0055, 0x0043, 0x0044, 0x0053, 0x0391,
    0x0410, 0x00C5, 0x212B, 0x1E0A, 0x0915, 0x09C7, 0x0B47, 0x0DD9, 0x1100,
    0x1101, 0xAC00, 0xAC01, 0x304B, 0x30A6, 0x0F90, 0x2126, 0x0958, 0x01D5,
    0x00DC, 0x0F77, 0x0EDC, 0xFB2C, 0x0390, 0x1FD3, 0x0622, 0x0627,
]


def nfc_cases():
    """Inputs for the normalizer: singles, pairs, triples, and long runs."""
    cases = [chr(c) for c in range(MAX) if not surrogate(c)]
    for s in STARTERS:
        for m in COMBINERS:
            cases.append(chr(s) + chr(m))
            for m2 in COMBINERS:
                cases.append(chr(s) + chr(m) + chr(m2))
    rng = random.Random(7)
    for _ in range(4000):
        n = rng.randrange(1, 8)
        s = chr(rng.choice(STARTERS))
        for _ in range(n):
            s += chr(rng.choice(COMBINERS))
        cases.append(s)
    cases += [
        "",
        "plain ascii",
        "é ȩ́ Ḑ̇",
        "각가",
        "ཷक़क़",
    ]
    return cases


PROSE = [
    "The capital of France is Paris, and the capital of Japan is Tokyo.",
    "Don't  you think it's  Peter's?  I'd've said they'll've gone.",
    "В начале было "
    "Слово, и Слово "
    "было у Бога.",
    "人民日报：今天的天气很好。",
    "日本語のテキストです。"
    "カタカナもひらがなも。",
    "السلام عليكم "
    "ورحمة الله.",
    "यह हिन्दी में "
    "लिखा गया है।",
    "Αρχή άνδρα "
    "του παντός.",
    "한국어 문장입니다. 감사합니다.",
    "Emoji: \U0001f600 \U0001f469‍\U0001f4bb \U0001f3f3️‍\U0001f308 "
    "and flags \U0001f1ef\U0001f1f5.",
    "Math: ∫_0^∞ e^{-x²} dx = √π/2, ∀x∈ℝ.",
    "fn main() {\n    let x: Vec<u32> = (0..10).map(|i| i * 2).collect();\n"
    "    println!(\"{x:?}\");\n}\n",
    "def f(a, b=3, *args, **kw):\n\treturn {\n\t\t'a': a,\n\t\t'b': b,\n\t}\n",
    "SELECT * FROM t WHERE a <= 3 AND b >= 'x' -- comment\n;",
    "  leading and trailing   \t\t  \n\n\n  mixed\r\n\r\nline endings  ",
    "a b　c d e  f",
    "​‌‍﻿ zero width",
    "123 4567890 1.5e-3 0xDEADBEEF 1_000_000 ١٢٣ ⅠⅡ ½①",
    "it'ss it'SS it'ſs 'tis 'Tis o'clock",
    "<|im_start|>user\nHello there<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\nHi!<|im_end|>\n",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "</function>\n</tool_call>",
    "<tool_response>\n{\"ok\": true}\n</tool_response>",
    "<|vision_start|><|image_pad|><|vision_end|>Describe this.",
    "<|audio_start|><|audio_pad|><|audio_end|><tts_pad><tts_text_bos>"
    "<tts_text_eod><tts_text_bos_single>",
    "<|fim_prefix|>def f():<|fim_suffix|>    return 1<|fim_middle|>",
    "<|repo_name|>ornith<|file_sep|>src/main.rs<|endoftext|>",
    "not a real token: <|im_start |> <| im_start |> <|im_star|",
    "क़क़ éé Ḑ̇ ÅÅ",
]


def encode_cases():
    cases = list(PROSE)
    # Every byte value, in runs, so the byte-level alphabet is fully exercised.
    for base in range(0, 0x300, 0x40):
        cases.append("".join(chr(c) for c in range(base, base + 0x40)))
    # A sweep of the planes, in short runs, to hit the rarer byte patterns.
    for base in range(0, MAX, 0x800):
        chunk = "".join(
            chr(c) for c in range(base, min(base + 0x30, MAX)) if not surrogate(c)
        )
        if chunk:
            cases.append(chunk)
    # Repetition, which is where the merge queue's tie-break shows up.
    for n in (1, 2, 3, 5, 8, 17, 64, 257):
        cases.append("a" * n)
        cases.append(" " * n)
        cases.append("\n" * n)
        cases.append("ab" * n)
        cases.append("é" * n)
    rng = random.Random(11)
    alphabet = "abc ,.\n\t'é一\U0001f600"
    for _ in range(600):
        cases.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 40))))
    return cases


def main():
    case = ref.Case("tokenizer")

    from tokenizers.normalizers import NFC
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ref.MODEL)

    bits = class_bits(ref.MODEL)
    case.put("classes", mx.array(bytes(bits), dtype=mx.uint8))

    norm = NFC()
    cases = nfc_cases()
    # NFC distributes over a `\n` join: `\n` is a starter that composes with
    # nothing, so it can neither be reordered into a neighbouring mark run nor
    # pulled into a composite.  Batching keeps this dump to seconds.
    outs = [None] * len(cases)
    batched = [i for i, c in enumerate(cases) if "\n" not in c]
    for j in range(0, len(batched), 4096):
        idx = batched[j : j + 4096]
        joined = norm.normalize_str("\n".join(cases[i] for i in idx))
        parts = joined.split("\n")
        assert len(parts) == len(idx), (j, len(parts), len(idx))
        for i, p in zip(idx, parts):
            outs[i] = p
    for i, c in enumerate(cases):
        if outs[i] is None:
            outs[i] = norm.normalize_str(c)
    case.put_json("nfc_in", cases)
    case.put_json("nfc_out", outs)

    from tokenizers import Tokenizer as RawTokenizer

    raw = RawTokenizer.from_file(os.path.join(ref.MODEL, "tokenizer.json"))
    # `tokenizer.json` does not carry the seven tokens only the config knows,
    # so add them the way `transformers` does before comparing.
    with open(os.path.join(ref.MODEL, "tokenizer_config.json")) as f:
        tcfg = json.load(f)
    from tokenizers import AddedToken

    known = {t["content"] for t in json.load(
        open(os.path.join(ref.MODEL, "tokenizer.json")))["added_tokens"]}
    extra = [
        AddedToken(v["content"], special=v["special"], normalized=False)
        for k, v in sorted(tcfg["added_tokens_decoder"].items(), key=lambda kv: int(kv[0]))
        if v["content"] not in known
    ]
    if extra:
        raw.add_tokens(extra)

    enc = encode_cases()
    ids = [tok.encode(t) for t in enc]
    case.put_json("encode_in", enc)
    case.put_json("encode_out", ids)
    case.put_json("encode_json", [raw.encode(t, add_special_tokens=False).ids for t in enc])
    case.put_json("decode_out", [tok.decode(i) for i in ids])
    case.put_json("pattern_json", [
        json.load(open(os.path.join(ref.MODEL, "tokenizer.json")))["pre_tokenizer"][
            "pretokenizers"][0]["pattern"]["Regex"],
        __import__("transformers.models.qwen2.tokenization_qwen2", fromlist=["x"])
        .PRETOKENIZE_REGEX,
    ])

    # The id -> token-string map the detokenizer indexes, so the Rust side can
    # check it built the same one.
    case.put_json("tokenmap", tok.convert_ids_to_tokens(list(range(len(tok)))))
    case.note(n_classes=MAX, n_nfc=len(cases), n_encode=len(enc), n_tokens=len(tok))
    case.write()


if __name__ == "__main__":
    main()
