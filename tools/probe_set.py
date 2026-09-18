"""The wide probe set: 64 contexts chosen to discriminate between quantizations.

`REBUILD.md` Phase 5. Phase 4 judged five artifacts on eight contexts and
6620 tokens of perplexity, and afterwards it turned out that **one** of those
eight was carrying the whole verdict: `Дважды два равно` separates the
variants by a factor of ten, on the other seven all four sit within 0.1 nats
of the control and two within 0.001. The conclusions stood; the instrument
that produced them has about one bit of discriminating power. This is the
replacement instrument.

It is a *second* set, not a wider first one. `dump_artifact_baseline.py`'s
`CONTEXTS` is fixed forever because `baseline/logprobs.bin` -- the Phase 2
check that the rebuilt trunk is bit-identical to the deleted artifact -- is
eight distributions over exactly those eight strings. Adding to that list
would retire the only measurement that cannot be regenerated. So this set
lives beside it, in `probes.bin`, and the old check is untouched.

**What makes a probe discriminate -- and the theory this set was built on,
which was wrong.** The selection below assumed that damage shows where the
model is *confident by a narrow path*: a forced continuation, held up by a
thin slice of a model that saw far more English than Russian. The captures
say that is not the criterion.

  - Entropy correlates *positively* with KL, so "confident" is the wrong
    direction. Against a near-delta distribution there is nowhere for mass to
    go, and four probes here are dead for exactly that reason.
  - Entropy does not select for it either. `The Rust borrow checker prevents`
    has the highest entropy of the old eight and cannot tell the variants
    apart; `Дважды два равно` sits below it and spreads them over 8.4x.

What separates a useful probe from a decorative one is **spread across
variants**, and spread is not visible in the text -- it has to be measured.
`compare_probes.py` reports it, and the measured verdict on these eight
groups is at the bottom of this file. The set survives its own bad theory
(60 of 64 probes are live) because breadth was the real requirement, so the
axes below are kept as the record of how it was chosen:

  arith      a forced number, in six languages. Context 6's own family --
             here to answer whether the signal was that prompt or that shape.
  script     non-Latin writing systems, forced factual completion. Eight
             scripts, so a maximum over them is not one sample.
  morph      agreement that forces an inflection: fusional (ru/pl), 
             agglutinative (fi/tr/hu), and separable-verb German.
  longrange  a token defined early and recalled hundreds of tokens later.
             Induction, not knowledge -- the only group whose answer is in
             its own prompt, so a wrong continuation cannot be excused.
  format     syntax where exactly one token is legal: JSON, XML, SQL, Rust
             turbofish, YAML, a C return.
  rare       nomenclature and proper nouns that appear a handful of times in
             any corpus.
  code       forced continuations in five languages, two with non-ASCII
             identifiers or comments.
  mixed      code-switching and mixed-script, where the model has to hold two
             tokenizations at once.

**Reading the result.** Every probe carries the control's entropy and top-1
mass, written by `dump_probes.py` at capture time, so a probe with no power
is visible as such instead of quietly diluting a median. A probe whose
control entropy is above ~2 nats is not measuring quantization, it is
measuring that the question was open.

Fixed once captured, for the same reason `CONTEXTS` is: the comparison is
between artifacts, and it only means anything if both saw the same strings.
"""

# ---------------------------------------------------------------- long-range
# Built rather than typed, because the point is length: a fact stated once,
# then several hundred tokens of plausible filler, then the recall. The
# filler has to be *boring* -- if it is surprising the probe measures the
# filler instead. Deterministic: no rng, no clock, same bytes every run.

_FILLER = (
    "The build system compiles each module in dependency order. "
    "Artifacts are cached by content hash and reused across runs. "
    "Logs are written to the session directory and rotated daily. "
    "Configuration is read once at startup and never re-read. "
)


def _padded(head: str, tail: str, repeats: int) -> str:
    return head + "\n\n" + (_FILLER * repeats).strip() + "\n\n" + tail


_KV = "\n".join(
    f"key {k} -> {v}"
    for k, v in [
        ("8f3a", "41"), ("c7d1", "92"), ("0b5e", "17"), ("9a2f", "63"),
        ("4e18", "28"), ("bd60", "75"), ("2c93", "50"), ("7a4d", "36"),
    ]
)

_ITEMS = "\n".join(
    f"{i:2d}. {w}"
    for i, w in enumerate(
        ["anchor", "basalt", "cinder", "dolmen", "ember", "fathom", "girder",
         "harrow", "isthmus", "jetty", "keel", "lintel", "mortise", "nadir",
         "obelisk", "plinth", "quarry", "rafter", "sluice", "tundra"],
        start=1,
    )
)

LONGRANGE = [
    ("kv-table",
     _padded(_KV, "Looking up key c7d1 gives the value", 6)),
    ("password",
     _padded("The access code for the east gate is Zyx-4471-Qav.",
             "Asked for it again, the guard repeated the access code: Zyx-", 8)),
    ("numbered-item",
     _padded(_ITEMS, "Item 13 on the list above is the word", 5)),
    ("story-name",
     _padded("Ingrid Solberg arrived at the station before dawn and paid "
             "for her ticket in cash.",
             "Nobody at the station could later recall the name of the woman "
             "who paid in cash, except the clerk, who wrote down: Ingrid", 7)),
    ("code-binding",
     _padded("let stride_bytes = 248320 * 4;  // one row of the lm_head, f32",
             "// and the final read multiplies the row count by\n    "
             "let total = rows * stride_", 6)),
    ("json-field",
     _padded('{"artifact": "ornith-1.5-35b-a3b", "group_size": 32, '
             '"body_mode": "affine"}',
             "The value of the body_mode field in the record above is", 6)),
    ("russian-decl",
     _padded("Экспедицию возглавил капитан Челюскин, и мыс назвали его именем.",
             "Самая северная точка материка носит имя капитана", 7)),
    ("two-facts",
     _padded("Rule A: every request carries a trace id. "
             "Rule B: every response carries a span id.",
             "A response, by rule B, must therefore carry a", 6)),
]

# --------------------------------------------------------------------- probes
# (group, id, text). Ids are stable; order is not load-bearing but is kept.

PROBES = [
    # ---- arith: a forced number. Context 6's family, widened deliberately.
    ("arith", "ru-2x2", "Дважды два равно"),
    ("arith", "ru-3x3", "Трижды три равно"),
    ("arith", "ru-7x8", "Семью восемь равно"),
    ("arith", "zh-2x2", "二二得"),
    ("arith", "de-2x2", "Zwei mal zwei ist"),
    ("arith", "es-2x2", "Dos por dos son"),
    ("arith", "ar-2p2", "اثنان زائد اثنان يساوي"),
    ("arith", "fib", "1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89,"),

    # ---- script: eight writing systems, each with a forced completion.
    ("script", "ja-capital", "日本の首都は"),
    ("script", "ko-capital", "대한민국의 수도는"),
    ("script", "ru-capital", "Столица Российской Федерации —"),
    ("script", "el-capital", "Η πρωτεύουσα της Ελλάδας είναι η"),
    ("script", "hi-capital", "भारत की राजधानी है"),
    ("script", "he-capital", "בירת ישראל היא"),
    ("script", "th-capital", "เมืองหลวงของประเทศไทยคือ"),
    ("script", "hy-capital", "Հայաստանի մայրաքաղաքն է"),

    # ---- morph: the next token is forced by agreement, not by meaning.
    ("morph", "ru-dat", "Он медленно подошёл к большому деревянному"),
    ("morph", "ru-prep", "Вчера мы долго говорили о новых научных"),
    ("morph", "pl-gen", "Nigdy wcześniej nie widziałem tej wysokiej"),
    ("morph", "fi-abl", "Sain kirjeen vanhalta suomalaiselta"),
    ("morph", "tr-agg", "Arkadaşımın evindeki kitaplarımı masanın üstüne"),
    ("morph", "hu-acc", "Az ablakon keresztül jól láttam a kicsi"),
    ("morph", "de-part", "Ich habe dem kleinen Mädchen gestern ein Buch"),
    ("morph", "uk-inst", "Вона пишалася своєю старшою"),

    # ---- longrange: the answer is in the prompt, several hundred tokens back.
    *[("longrange", pid, text) for pid, text in LONGRANGE],

    # ---- format: exactly one token is legal.
    ("format", "json", '{"name": "ornith", "version": "1.5", "quantization":'),
    ("format", "sql-order", "SELECT name, email FROM users WHERE id = 42 ORDER BY"),
    ("format", "rust-assoc",
     "impl<T: Clone + Send> Iterator for MyIter<T> {\n    type Item ="),
    ("format", "xml-close",
     '<?xml version="1.0" encoding="UTF-8"?>\n<root>\n  <item id="1">value</'),
    ("format", "c-return",
     '#include <stdio.h>\n\nint main(void) {\n    printf("hello\\n");\n    return'),
    ("format", "rust-result",
     "fn main() -> Result<(), Box<dyn std::error::"),
    ("format", "yaml-ci",
     "---\nname: probe\non:\n  push:\n    branches: [main]\njobs:\n  build:\n    runs-on:"),
    ("format", "toml-dep",
     '[dependencies]\nserde = { version = "1.0", features = ['),

    # ---- rare: nomenclature and proper nouns, thin in any corpus.
    ("rare", "helicase",
     "The enzyme that unwinds the DNA double helix ahead of the replication "
     "fork is called"),
    ("rare", "topaz",
     "The mineral with the chemical formula Al2SiO4(F,OH)2 is called"),
    ("rare", "schwarzschild",
     "The Schwarzschild radius of a body of mass M equals 2GM divided by the "
     "square of the speed of"),
    ("rare", "e4m3",
     "In the OCP microscaling specification, E4M3 has four exponent bits and "
     "three bits of"),
    ("rare", "belisarius",
     "The Byzantine general Belisarius reconquered North Africa under the "
     "emperor"),
    ("rare", "chelyuskin",
     "Мыс Челюскин — самая северная точка материковой части"),
    ("rare", "halite",
     "Кристаллическая решётка поваренной соли имеет форму"),
    ("rare", "cauchy",
     "The inequality |⟨u, v⟩| ≤ ‖u‖‖v‖ is known as the inequality of Cauchy "
     "and"),

    # ---- code: forced continuations, two of them through non-ASCII source.
    ("code", "py-quicksort",
     "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
     "    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x <"),
    ("code", "rust-derive",
     "#[derive(Debug, Clone, Copy, PartialEq)]\npub struct Quant {\n"
     "    pub bits: u8,\n    pub group_size:"),
    ("code", "rust-ru-comment",
     "// Считаем сумму элементов среза и возвращаем её.\n"
     "fn sum(xs: &[f32]) -> f32 {\n    xs.iter()."),
    ("code", "py-aiohttp",
     "async def fetch(url: str) -> bytes:\n"
     "    async with aiohttp.ClientSession() as session:\n"
     "        async with session.get(url) as resp:\n            return await resp."),
    ("code", "cpp-dtor",
     "template <typename T>\nclass Vector {\npublic:\n"
     "    explicit Vector(size_t n) : size_(n), data_(new T[n]) {}\n"
     "    ~Vector() { delete[]"),
    ("code", "metal-kernel",
     "kernel void add(device const float* a [[buffer(0)]],\n"
     "                device const float* b [[buffer(1)]],\n"
     "                device float* out [[buffer(2)]],\n"
     "                uint gid [[thread_position_in_"),
    ("code", "sql-having",
     "SELECT u.name, COUNT(o.id) AS n\nFROM users u\n"
     "JOIN orders o ON o.user_id = u.id\nGROUP BY u.name\nHAVING COUNT(o.id) >"),
    ("code", "zh-comment",
     "# 计算数组中所有元素的平均值\ndef mean(xs):\n    return sum(xs) / len("),

    # ---- mixed: two tokenizations at once.
    ("mixed", "ru-rust-ownership",
     "В Rust владение (ownership) означает, что у каждого значения есть ровно один"),
    ("mixed", "ja-pytorch",
     "PyTorch の nn.Module を継承したクラスでは、順伝播を forward メソッドに"),
    ("mixed", "es-ml",
     "El término «machine learning» se traduce al español como aprendizaje"),
    ("mixed", "zh-complexity",
     "快速排序的平均时间复杂度是 O(n log"),
    ("mixed", "ru-mlx",
     "Функция `quantized_matmul` принимает упакованные веса, scales и"),
    ("mixed", "ar-python",
     "في لغة البرمجة بايثون، تُستخدم الدالة print من أجل"),
    ("mixed", "fr-quicksort",
     "L'algorithme de tri rapide a une complexité moyenne en O(n log"),
    ("mixed", "ja-mlx-eval",
     "MLX の mx.array は遅延評価なので、実際に値を求めるには mx."),
]

GROUPS = ["arith", "script", "morph", "longrange", "format", "rare", "code", "mixed"]


# ------------------------------------------------------------ what it measured
# From the first full run, five artifacts, 2026-09-18. Median KL of the four
# Phase 4 variants against the control, by group, live probes only:
#
#     arith      0.089 - 0.288     forced numbers, non-English
#     morph      0.048 - 0.109     agreement-driven inflection
#     longrange  0.056 - 0.081     in-context recall, and it hides the worst
#                                  single failures: two variants answer the
#                                  wrong name on `russian-decl`, at 1.6 and
#                                  2.2 nats, which nothing in the old eight
#                                  could see
#     script     0.027 - 0.114     non-Latin, forced factual
#     code       0.012 - 0.086
#     rare       0.008 - 0.040
#     mixed      0.020 - 0.035
#     format     0.012 - 0.024     syntax-locked, and nearly free -- three of
#                                  the four dead probes are here
#
# So a probe set that wants power should be weighted toward arith, morph and
# longrange, and `format` earns at most a couple of slots as a control. The
# pattern under that: damage lands where the answer is forced by something
# learned narrowly (non-English numerals, agreement, in-context recall) and
# not where it is forced by syntax, which every quantization keeps.
