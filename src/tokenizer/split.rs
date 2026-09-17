//! The pre-tokenizer's split pattern, hand-matched.
//!
//! `tokenizer.json` asks for
//!
//! ```text
//! Split(Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)
//!              |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//!              |\p{N}
//!              | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//!              |\s*[\r\n]+
//!              |\s+(?!\S)
//!              |\s+"), behavior = Isolated)
//! ```
//!
//! -- the GPT-4 pattern, in the shape newer Qwen tokenizers use, with
//! `\p{M}` alongside `\p{L}` so a combining mark stays with its base letter.
//!
//! `transformers` 5.x does **not** run that pattern. `Qwen2Tokenizer` there is
//! a `TokenizersBackend` that rebuilds the tokenizer from `vocab.json` and
//! `merges.txt` and installs its own hardcoded `PRETOKENIZE_REGEX`
//! (`tokenization_qwen2.py`), which is the older Qwen2 pattern with no
//! `\p{M}` in either class. mlx-lm reaches the tokenizer through
//! `AutoTokenizer`, so that is the tokenization `mlx_lm.generate` actually
//! runs, and the two disagree on anything with a combining mark in it --
//! Devanagari, Arabic diacritics, an emoji with a variation selector. Both are
//! ported; see `Dialect`.
//!
//! `tokenizers` hands the pattern to oniguruma, an ordered
//! backtracking engine, and the pattern matches at least one character at
//! every position, so `Isolated` tiles the input: repeatedly take the match at
//! the cursor, first alternative that succeeds, greedy quantifiers with
//! backtracking. That is exactly what `match_at` below does, alternative by
//! alternative in source order.
//!
//! Two details of the engine are pinned by probing the live pre-tokenizer
//! (see `tools/dump_tokenizer.py`, which re-derives both and lets the Rust
//! test compare):
//!
//!   * `\s` is oniguruma's Unicode `CTYPE_SPACE`, the `White_Space` property,
//!     not Ruby's ASCII-only `\s`. U+00A0 and U+3000 are `\s`; U+200B and
//!     U+001C are not.
//!   * `(?i:)` folds with Unicode simple case folding, so `'s` also matches
//!     `'S` and `'\u{17f}` (LATIN SMALL LETTER LONG S). No other character
//!     folds onto any of `s t r e v m l d`.
//!
//! `\p{L}` and `\p{M}` are only ever used together, as `[\p{L}\p{M}]`, and the
//! only class that separates them, `[^\r\n\p{L}\p{N}]`, is a single optional
//! character in front of a `[\p{L}\p{M}]+` run -- where a `\p{M}` prefix and a
//! `\p{M}` first run character give the same span either way. The two tables
//! are still kept apart here to stay readable against the pattern.

use super::tables::{LETTER, MARK, NUMBER, WHITE_SPACE};

/// Which of the two patterns to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// The pattern `tokenizer.json` carries, with `\p{M}`:
    /// `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` and `[^\s\p{L}\p{M}\p{N}]+`.
    TokenizerJson,
    /// `transformers`' `PRETOKENIZE_REGEX`, without it:
    /// `[^\r\n\p{L}\p{N}]?\p{L}+` and `[^\s\p{L}\p{N}]+`.
    Transformers,
}

impl Dialect {
    /// The pattern each dialect spells, so `tokenizer.json` can be checked
    /// against them rather than assumed.
    pub const TOKENIZER_JSON_PATTERN: &'static str =
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    pub const TRANSFORMERS_PATTERN: &'static str =
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn marks_with_letters(self) -> bool {
        self == Dialect::TokenizerJson
    }
}

fn in_ranges(table: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    table
        .binary_search_by(|&(lo, hi)| {
            if c < lo {
                std::cmp::Ordering::Greater
            } else if c > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `\p{L}`
pub fn is_letter(c: char) -> bool {
    in_ranges(LETTER, c)
}

/// `\p{M}`
pub fn is_mark(c: char) -> bool {
    in_ranges(MARK, c)
}

/// `\p{N}`
pub fn is_number(c: char) -> bool {
    in_ranges(NUMBER, c)
}

/// `\s`
pub fn is_space(c: char) -> bool {
    in_ranges(WHITE_SPACE, c)
}

/// `[\p{L}\p{M}]`, or just `\p{L}` in the transformers dialect.
fn is_word_char(c: char, d: Dialect) -> bool {
    is_letter(c) || (d.marks_with_letters() && is_mark(c))
}

/// `[^\r\n\p{L}\p{N}]`
fn is_word_prefix(c: char) -> bool {
    c != '\r' && c != '\n' && !is_letter(c) && !is_number(c)
}

/// `[^\s\p{L}\p{M}\p{N}]`, or `[^\s\p{L}\p{N}]` in the transformers dialect.
fn is_punct(c: char, d: Dialect) -> bool {
    !is_space(c) && !is_word_char(c, d) && !is_number(c)
}

/// Unicode simple case folding, restricted to the eight letters the
/// contraction alternative spells. Only U+017F folds onto one of them.
fn folds_to(c: char, lower: char) -> bool {
    debug_assert!("strevmld".contains(lower));
    c == lower || c == lower.to_ascii_uppercase() || (lower == 's' && c == '\u{17f}')
}

/// The pieces the pre-tokenizer hands to the model, in order; concatenated
/// they are `text` again.
pub fn pre_split(text: &str, dialect: Dialect) -> Vec<&str> {
    let chars: Vec<char> = text.chars().collect();
    let mut offs: Vec<usize> = Vec::with_capacity(chars.len() + 1);
    let mut o = 0;
    for c in &chars {
        offs.push(o);
        o += c.len_utf8();
    }
    offs.push(o);

    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = match_at(&chars, i, dialect);
        debug_assert!(end > i, "the split pattern matched nothing at {i}");
        out.push(&text[offs[i]..offs[end]]);
        i = end;
    }
    out
}

/// The end of the match starting at `i`: the first alternative that succeeds.
fn match_at(c: &[char], i: usize, d: Dialect) -> usize {
    let n = c.len();

    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c[i] == '\'' {
        for word in ["s", "t", "re", "ve", "m", "ll", "d"] {
            let w = word.as_bytes();
            if i + 1 + w.len() <= n
                && w.iter().enumerate().all(|(k, &l)| folds_to(c[i + 1 + k], l as char))
            {
                return i + 1 + w.len();
            }
        }
    }

    // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+   (the `?` is greedy, so try the
    //    prefix first and fall back to not taking it)
    for skip in [1, 0] {
        if skip == 1 && !is_word_prefix(c[i]) {
            continue;
        }
        let mut j = i + skip;
        if j < n && is_word_char(c[j], d) {
            while j < n && is_word_char(c[j], d) {
                j += 1;
            }
            return j;
        }
    }

    // 3. \p{N}
    if is_number(c[i]) {
        return i + 1;
    }

    // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    {
        let skip = usize::from(c[i] == ' ');
        let mut j = i + skip;
        if j < n && is_punct(c[j], d) {
            while j < n && is_punct(c[j], d) {
                j += 1;
            }
            while j < n && (c[j] == '\r' || c[j] == '\n') {
                j += 1;
            }
            return j;
        }
    }

    // 5. \s*[\r\n]+   (greedy `\s*`, then back off until `[\r\n]+` can match)
    {
        let mut j = i;
        while j < n && is_space(c[j]) {
            j += 1;
        }
        while j > i {
            if c[j - 1] == '\r' || c[j - 1] == '\n' {
                return j;
            }
            j -= 1;
        }
    }

    // 6. \s+(?!\S)   (give back one character at a time until the next one is
    //    a space, or the string ends)
    {
        let mut j = i;
        while j < n && is_space(c[j]) {
            j += 1;
        }
        while j > i {
            if j == n || is_space(c[j]) {
                return j;
            }
            j -= 1;
        }
    }

    // 7. \s+
    if is_space(c[i]) {
        let mut j = i;
        while j < n && is_space(c[j]) {
            j += 1;
        }
        return j;
    }

    unreachable!("the split pattern always matches at least one character")
}

#[cfg(test)]
mod tests {
    use super::Dialect;

    fn pre_split(text: &str) -> Vec<&str> {
        super::pre_split(text, Dialect::TokenizerJson)
    }

    #[test]
    fn matches_the_documented_shapes() {
        assert_eq!(pre_split("don't"), vec!["don", "'t"]);
        assert_eq!(pre_split("it'ss"), vec!["it", "'s", "s"]);
        assert_eq!(pre_split("it'\u{17f}s"), vec!["it", "'\u{17f}", "s"]);
        assert_eq!(pre_split("123abc"), vec!["1", "2", "3", "abc"]);
        assert_eq!(pre_split("--x"), vec!["--", "x"]);
        assert_eq!(pre_split("  \n\n  x"), vec!["  \n\n", " ", " x"]);
        assert_eq!(pre_split("a   "), vec!["a", "   "]);
        assert_eq!(pre_split("a  b"), vec!["a", " ", " b"]);
        // U+00A0 is \s for oniguruma, so it goes with the following word as a
        // prefix rather than joining a punctuation run.
        assert_eq!(pre_split("a\u{a0}\u{a0}x"), vec!["a", "\u{a0}", "\u{a0}x"]);
        // U+2028 is \s but not [\r\n]
        assert_eq!(pre_split("a \u{2028} b"), vec!["a", " \u{2028}", " b"]);
    }

    #[test]
    fn the_two_dialects_differ_only_on_marks() {
        let hindi = "\u{939}\u{93f}\u{928}";
        assert_eq!(
            super::pre_split(hindi, Dialect::TokenizerJson),
            vec!["\u{939}\u{93f}\u{928}"]
        );
        assert_eq!(
            super::pre_split(hindi, Dialect::Transformers),
            vec!["\u{939}", "\u{93f}\u{928}"]
        );
        for s in ["hello, world", "  \n\n  x", "don't 123"] {
            assert_eq!(
                super::pre_split(s, Dialect::TokenizerJson),
                super::pre_split(s, Dialect::Transformers)
            );
        }
    }
}
