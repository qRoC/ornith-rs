"""Generate `src/tokenizer/tables.rs`: the Unicode data the tokenizer needs.

Two things in `tokenizer.json` need Unicode tables that Rust's standard library
does not carry:

  * the pre-tokenizer's split regex, which uses `\\p{L}`, `\\p{M}`, `\\p{N}`
    and `\\s`.  The `tokenizers` crate runs that pattern through oniguruma, so
    `\\s` is oniguruma's Unicode `CTYPE_SPACE`, i.e. the `White_Space`
    property -- not Ruby's ASCII-only `\\s` and not Python's `str.isspace`.
    Probed against the real pre-tokenizer in `dump_tokenizer.py`.

  * the NFC normalizer: canonical combining classes, canonical decompositions
    and the primary composition pairs.  Hangul is algorithmic (UAX #15, D132)
    and is left out of the tables.

Run it from `tools/`:

    python3 gen_unicode.py

The generated file is checked in; regenerate it only when the Python that
produced it moves to a new Unicode version (the header records which one).
"""

import os
import sys
import unicodedata

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "tokenizer",
    "tables.rs",
)

MAX = 0x110000
SURROGATES = range(0xD800, 0xE000)

# UAX #15, D132: the algorithmic Hangul syllables, handled in code.
S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT

# PropList.txt, White_Space.  Oniguruma's \s for a Unicode encoding is this
# property; `dump_tokenizer.py` re-derives it from the live pre-tokenizer and
# the Rust test compares, so a wrong entry here fails loudly.
WHITE_SPACE = (
    list(range(0x09, 0x0E))
    + [0x20, 0x85, 0xA0, 0x1680]
    + list(range(0x2000, 0x200B))
    + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)


def ranges(codepoints):
    """Collapse a sorted iterable of ints into inclusive [start, end] ranges."""
    out = []
    for c in codepoints:
        if out and c == out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [(a, b) for a, b in out]


def category_ranges(prefix):
    return ranges(
        c
        for c in range(MAX)
        if c not in SURROGATES and unicodedata.category(chr(c)).startswith(prefix)
    )


def canonical_decomposition(c):
    """The single-step canonical decomposition of `c`, or None."""
    d = unicodedata.decomposition(chr(c))
    if not d or d.startswith("<"):
        return None
    return [int(x, 16) for x in d.split()]


def full_decomposition(c):
    """Recursively applied canonical decomposition (UAX #15, D68)."""
    d = canonical_decomposition(c)
    if d is None:
        return [c]
    out = []
    for x in d:
        out.extend(full_decomposition(x))
    return out


def collect():
    letter = category_ranges("L")
    mark = category_ranges("M")
    number = category_ranges("N")
    space = ranges(sorted(WHITE_SPACE))
    # Python's `str.isspace`, which is what jinja's `|trim` and `.strip()`
    # strip in the chat template.  It is White_Space plus the four
    # information separators U+001C..U+001F.
    py_space = ranges(c for c in range(MAX) if c not in SURROGATES and chr(c).isspace())

    ccc = []
    for c in range(MAX):
        if c in SURROGATES:
            continue
        k = unicodedata.combining(chr(c))
        if k:
            if ccc and ccc[-1][1] == c - 1 and ccc[-1][2] == k:
                ccc[-1][1] = c
            else:
                ccc.append([c, c, k])

    decomp = []
    compose = []
    for c in range(MAX):
        if c in SURROGATES or S_BASE <= c < S_BASE + S_COUNT:
            continue
        step = canonical_decomposition(c)
        if step is None:
            continue
        full = full_decomposition(c)
        assert 1 <= len(full) <= 4, (hex(c), full)
        decomp.append((c, full))
        # Primary composites: (first, second) -> composed, off the *single
        # step* decomposition (U+01D5 decomposes to two characters but to
        # three when that is applied recursively).  A pair composes exactly
        # when NFC puts it back together, which is the definition net of the
        # composition exclusions, the singleton decompositions and the
        # non-starter decompositions -- so ask NFC rather than reimplement
        # the three rules.
        if len(step) == 2 and unicodedata.normalize(
            "NFC", chr(step[0]) + chr(step[1])
        ) == chr(c):
            compose.append((step[0], step[1], c))
    compose.sort()

    # Quick check: text made only of "safe" characters is already NFC.  A
    # character is safe when it has ccc 0, is unchanged by NFC (so NFC_QC is
    # not No), and is never the second half of a composition (so NFC_QC is not
    # Maybe).  Hangul syllables are composed already, but their jamo are not,
    # so the jamo fall out of the composition test below.
    seconds = {b for _, b, _ in compose} | set(range(V_BASE, V_BASE + V_COUNT))
    seconds |= set(range(T_BASE + 1, T_BASE + T_COUNT))
    unsafe = []
    for c in range(MAX):
        if c in SURROGATES:
            continue
        ch = chr(c)
        if (
            unicodedata.combining(ch)
            or unicodedata.normalize("NFC", ch) != ch
            or c in seconds
        ):
            unsafe.append(c)
    unsafe = ranges(unsafe)

    return letter, mark, number, space, py_space, ccc, decomp, compose, unsafe


def fmt_ranges(name, doc, rs):
    lines = [f"/// {doc}", f"pub static {name}: &[(u32, u32)] = &["]
    row = []
    for a, b in rs:
        row.append(f"(0x{a:X},0x{b:X}),")
        if len(row) == 6:
            lines.append("    " + "".join(row))
            row = []
    if row:
        lines.append("    " + "".join(row))
    lines.append("];")
    return "\n".join(lines)


def main():
    letter, mark, number, space, py_space, ccc, decomp, compose, unsafe = collect()

    out = []
    out.append("//! Unicode tables, generated by `tools/gen_unicode.py`.")
    out.append("//!")
    out.append(f"//! Unicode {unicodedata.unidata_version}"
               f" (python {sys.version.split()[0]}).  Do not edit by hand.")
    out.append("")
    out.append(fmt_ranges("LETTER", "General category L* -- the regex's `\\p{L}`.", letter))
    out.append("")
    out.append(fmt_ranges("MARK", "General category M* -- the regex's `\\p{M}`.", mark))
    out.append("")
    out.append(fmt_ranges("NUMBER", "General category N* -- the regex's `\\p{N}`.", number))
    out.append("")
    out.append(fmt_ranges(
        "WHITE_SPACE",
        "The `White_Space` property -- oniguruma's `\\s` for a Unicode encoding.",
        space,
    ))
    out.append("")
    out.append(fmt_ranges(
        "PY_WHITESPACE",
        "Python's `str.isspace` -- what jinja's `|trim` strips in the chat "
        "template.",
        py_space,
    ))
    out.append("")
    out.append(fmt_ranges(
        "NFC_UNSAFE",
        "Characters that can take part in a normalization: ccc != 0, or "
        "NFC_QC != Yes.\n/// Text without any of these is already NFC.",
        unsafe,
    ))
    out.append("")

    out.append("/// Canonical combining class, as `(start, end, ccc)` over "
               "non-zero runs.")
    out.append("pub static CCC: &[(u32, u32, u8)] = &[")
    row = []
    for a, b, k in ccc:
        row.append(f"(0x{a:X},0x{b:X},{k}),")
        if len(row) == 4:
            out.append("    " + "".join(row))
            row = []
    if row:
        out.append("    " + "".join(row))
    out.append("];")
    out.append("")

    out.append("/// Full canonical decompositions, excluding the algorithmic "
               "Hangul syllables:")
    out.append("/// `(codepoint, offset, len)` into `DECOMPOSE_DATA`, sorted "
               "by codepoint.")
    out.append("pub static DECOMPOSE: &[(u32, u32, u32)] = &[")
    data = []
    row = []
    for c, full in decomp:
        row.append(f"(0x{c:X},{len(data)},{len(full)}),")
        data.extend(full)
        if len(row) == 3:
            out.append("    " + "".join(row))
            row = []
    if row:
        out.append("    " + "".join(row))
    out.append("];")
    out.append("")
    out.append("/// The decomposed characters `DECOMPOSE` indexes into.")
    out.append("pub static DECOMPOSE_DATA: &[u32] = &[")
    row = []
    for x in data:
        row.append(f"0x{x:X},")
        if len(row) == 8:
            out.append("    " + "".join(row))
            row = []
    if row:
        out.append("    " + "".join(row))
    out.append("];")
    out.append("")

    out.append("/// Primary composites: `(first, second, composed)`, sorted by "
               "`(first, second)`.")
    out.append("pub static COMPOSE: &[(u32, u32, u32)] = &[")
    row = []
    for a, b, c in compose:
        row.append(f"(0x{a:X},0x{b:X},0x{c:X}),")
        if len(row) == 3:
            out.append("    " + "".join(row))
            row = []
    if row:
        out.append("    " + "".join(row))
    out.append("];")
    out.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")

    print(
        f"{OUT}: letter {len(letter)} ranges, mark {len(mark)}, number "
        f"{len(number)}, space {len(space)}, py_space {len(py_space)}, "
        f"unsafe {len(unsafe)}, "
        f"ccc {len(ccc)} runs, decompose {len(decomp)}, compose {len(compose)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
