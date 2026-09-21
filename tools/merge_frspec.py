"""Extend the FR-Spec sub-vocabulary with what a corpus needs and the list lacks.

`REBUILD.md` Phase 43 measured that `--frspec` is not a flat win but a
property of the text: acceptance under it is the full head's acceptance times
the fraction of emitted tokens that fall inside the 65536 rows it drafts from.
MTPLX's list is ranked on a **code** corpus (`qwen38_code_ranked_64k.npy`), so
English prose covers 100.0% and code 99.0% -- and Russian prose covers 65%,
where `--frspec` costs 11-19% instead of paying 4-7%.

**This adds rather than swaps, and that is the whole design.** The ranks are
not in the shipped artifact -- `dump_frspec.py` says the ids were ranked and
*then sorted*, so there is no tail to evict by rank, only by a frequency this
corpus measured, which would fit the list to this corpus and quietly lower its
coverage everywhere else. A superset cannot do that: every id the original had
is still there, so **English and code coverage are unchanged by construction**
and the only cost is that the draft head reads a few per cent more bytes.
Nothing else about FR-Spec changes -- it still cannot alter what the model
emits, because every committed token is the verify forward's own argmax
(`src/frspec.rs`).

**Two ways to choose what to add, and the cheaper one is not the better one.**

    # by what a corpus needs
    cd tools && ORNITH_MODEL=~/.mtplx/models/<artifact> python3 merge_frspec.py \
        --min-count 2 --out ../refs/frspec/ids_64k_plus.bin <corpus paths>

    # by what the vocabulary *is* -- no corpus at all
    cd tools && ORNITH_MODEL=~/.mtplx/models/<artifact> python3 merge_frspec.py \
        --script cyrillic --out ../refs/frspec/ids_64k_cyr.bin

`--min-count 2` drops the hapaxes: an id seen once in the corpus is a poor bet
for a list whose job is the *next* text. On 48k tokens of this project's own
Russian it keeps 2011 of 3317 missing ids, takes coverage 68.2% -> 97.2% and
grows the list by 3.1% -- **but only on text like the corpus**: on an
unrelated Russian prompt the same list reaches 68.2% where the original
reached 43.8%, i.e. half the gap.

`--script cyrillic` does not ask a corpus anything. The vocabulary holds
**18386 Cyrillic-majority ids and MTPLX's list contains 311 of them**, which
is the code corpus seen from the other side. Adding all of them costs +27.6%
of list and reaches **100.0% coverage on both Russian texts**, in-domain and
out, with English and code untouched. A corpus can only tell you about the
text it is; the vocabulary tells you about the language.

Point the port at the result with `ORNITH_FRSPEC=<path>`, and measure before
believing it: the cell is `--mtp-depth 1` with and without `--frspec`, three
interleaved draws, and the emitted bytes must be identical in every arm.

**It is a divergence from MTPLX's artifact**, which is a decision and not a
conclusion -- the same shape as leaving `--mtp` at depth 3. The port's default
stays MTPLX's list; this writes a second one beside it.
"""

import argparse
import collections
import os
import pathlib
import struct
import sys

VOCAB = 248320
DEFAULT_BASE = pathlib.Path(__file__).resolve().parent.parent / "refs/frspec/ids_64k.bin"


def read_ids(path):
    raw = pathlib.Path(path).read_bytes()
    n = struct.unpack("<I", raw[:4])[0]
    if len(raw) != 4 + 4 * n:
        sys.exit(f"{path}: header says {n} ids, file holds {(len(raw) - 4) / 4:.1f}")
    return list(struct.unpack(f"<{n}I", raw[4 : 4 + 4 * n]))


def write_ids(path, ids):
    ids = sorted(set(int(i) for i in ids))
    assert ids, "no ids"
    assert len(ids) < VOCAB, f"{len(ids)} is not actually pruned against {VOCAB}"
    assert 0 <= ids[0] and ids[-1] < VOCAB, "ids out of range"
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        for i in ids:
            f.write(struct.pack("<I", i))
    return len(ids)


SCRIPTS = {"cyrillic": lambda c: "\u0430" <= c.lower() <= "\u044f" or c == "\u0451"}


def ids_in_script(tok, name):
    """Every id whose token is majority `name`, read off the vocabulary.

    The tokens come back in byte-level BPE's own alphabet, so they have to be
    mapped back through it first -- reading them raw finds **zero** Cyrillic
    ids in a vocabulary that holds 18386 of them, which is what the first
    version of this did.
    """
    want = SCRIPTS.get(name)
    if want is None:
        sys.exit(f"--script {name}: known scripts are {sorted(SCRIPTS)}")
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    back = dict(zip([chr(c) for c in cs], bs))

    out = set()
    for i, s in enumerate(tok.convert_ids_to_tokens(list(range(VOCAB)))):
        if s is None:
            continue
        try:
            text = bytes(back[c] for c in s).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
        letters = [c for c in text if c.isalpha()]
        if letters and sum(1 for c in letters if want(c)) / len(letters) > 0.5:
            out.add(i)
    return out


def texts(paths):
    for p in paths:
        p = pathlib.Path(p)
        for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
            if f.is_file():
                try:
                    yield f, f.read_text()
                except (UnicodeDecodeError, OSError):
                    continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="*", help="files or directories of plain text")
    ap.add_argument(
        "--script",
        default=None,
        help="also add every id whose token is majority this script (cyrillic); "
        "needs no corpus at all -- see the note above",
    )
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--out", default=None, help="default: <base>_plus.bin")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--model", default=os.environ.get("ORNITH_MODEL"))
    a = ap.parse_args()
    if not a.model:
        sys.exit("ORNITH_MODEL or --model: the tokenizer comes from the artifact")
    out = a.out or str(pathlib.Path(a.base).with_name(pathlib.Path(a.base).stem + "_plus.bin"))

    if not a.corpus and not a.script:
        sys.exit("nothing to add: give a corpus, or --script, or both")

    base = read_ids(a.base)
    inside = set(base)

    # transformers' own pre-tokenizer, which is this port's default dialect --
    # `--tokenizer-json` is the opt-in and would produce different ids.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)

    script_ids = []
    if a.script:
        script_ids = sorted(ids_in_script(tok, a.script) - inside)
        print(f"  --script {a.script}: {len(script_ids)} ids not already in the list")

    seen, files = collections.Counter(), 0
    total = 0
    for f, text in texts(a.corpus):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        files += 1
        total += len(ids)
        seen.update(ids)
    if a.corpus and not total:
        sys.exit("the corpus tokenized to nothing")

    if not total:
        print(f"{out}: {write_ids(out, base + script_ids)} ids "
              f"(+{100 * len(script_ids) / len(base):.1f}%, script only)")
        return
    covered = sum(v for i, v in seen.items() if i in inside)
    missing = collections.Counter({i: v for i, v in seen.items() if i not in inside})
    keep = sorted(i for i, v in missing.items() if v >= a.min_count)
    after = covered + sum(missing[i] for i in keep)

    n = write_ids(out, base + keep + script_ids)
    print(f"corpus: {files} files, {total} tokens")
    print(f"  coverage with {a.base}: {100 * covered / total:.1f}%")
    print(f"  missing ids: {len(missing)} distinct, {len(keep)} of them seen >= {a.min_count}")
    print(f"  coverage with the merged list: {100 * after / total:.1f}%")
    print(f"{out}: {n} ids ({n / VOCAB:.4f} of the vocabulary, +{100 * (n - len(base)) / len(base):.1f}%)")


main()
