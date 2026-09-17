//! The tokenizer against `tokenizers` / `transformers`.
//!
//! Reference: tools/dump_tokenizer.py
//!
//! Three layers, narrowest first, so a failure names the layer:
//! the character classes the split regex asks oniguruma about, the NFC
//! normalizer, then whole-pipeline `encode` and `decode`.

use ornith_rs::refs::{RefCase, model_dir};
use ornith_rs::tokenizer::nfc::nfc;
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::tokenizer::split::{Dialect, is_letter, is_mark, is_number, is_space};

/// The class oniguruma put each codepoint in, as `dump_tokenizer.py` read it
/// back out of the live pre-tokenizer.
const OTHER: u8 = 0;
const LETTER_OR_MARK: u8 = 1;
const NUMBER: u8 = 2;
const SPACE: u8 = 3;

#[test]
fn character_classes_match_oniguruma() {
    let Some(case) = RefCase::open_or_skip("tokenizer") else { return };
    let want = case.get("classes").bytes;
    assert_eq!(want.len(), 0x110000);

    let mut bad = 0;
    for c in 0..0x110000u32 {
        let Some(ch) = char::from_u32(c) else { continue };
        let got = if is_letter(ch) || is_mark(ch) {
            LETTER_OR_MARK
        } else if is_number(ch) {
            NUMBER
        } else if is_space(ch) {
            SPACE
        } else {
            OTHER
        };
        if got != want[c as usize] {
            if bad < 20 {
                eprintln!(
                    "U+{c:04X}: tables say {got}, the pre-tokenizer says {}",
                    want[c as usize]
                );
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} codepoints are in the wrong class");
}

#[test]
fn nfc_matches_the_tokenizers_normalizer() {
    let Some(case) = RefCase::open_or_skip("tokenizer") else { return };
    let ins = case.strings("nfc_in");
    let outs = case.strings("nfc_out");
    assert_eq!(ins.len(), outs.len());
    for (i, (a, b)) in ins.iter().zip(outs.iter()).enumerate() {
        let got = nfc(a);
        assert_eq!(&got, b, "case {i}: nfc({:?})", cps(a));
    }
}

/// The two split patterns the dump recorded, against the two this port
/// spells. `Dialect` is only honest if these are the strings in play.
#[test]
fn both_split_patterns_are_the_ones_in_play() {
    let Some(case) = RefCase::open_or_skip("tokenizer") else { return };
    let pats = case.strings("pattern_json");
    assert_eq!(pats[0], Dialect::TOKENIZER_JSON_PATTERN, "tokenizer.json's pattern moved");
    assert_eq!(pats[1], Dialect::TRANSFORMERS_PATTERN, "transformers' pattern moved");
}

#[test]
fn encode_matches_transformers() {
    check_encode(Dialect::Transformers, "encode_out", true);
}

#[test]
fn encode_matches_tokenizer_json() {
    check_encode(Dialect::TokenizerJson, "encode_json", false);
}

fn check_encode(dialect: Dialect, want_key: &str, check_decode: bool) {
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let Some(case) = RefCase::open_or_skip("tokenizer") else { return };
    let tk = Tokenizer::load_as(&dir, dialect).unwrap();

    // The id -> token-string map the streaming detokenizer indexes into.
    let tokenmap = case.strings("tokenmap");
    assert_eq!(tk.vocab_size(), tokenmap.len());
    for (id, want) in tokenmap.iter().enumerate() {
        assert_eq!(tk.token(id as u32), want.as_str(), "token {id}");
    }

    let ins = case.strings("encode_in");
    let want = case.json(want_key);
    let want = want.as_arr().unwrap();
    let decoded = case.strings("decode_out");
    assert_eq!(ins.len(), want.len());
    for (i, text) in ins.iter().enumerate() {
        let got = tk.encode(text);
        let expect: Vec<u32> =
            want[i].as_arr().unwrap().iter().map(|v| v.as_usize().unwrap() as u32).collect();
        assert_eq!(got, expect, "{dialect:?} case {i}: encode({:?})", cps(text));
        if check_decode {
            assert_eq!(&tk.decode(&got), &decoded[i], "case {i}: decode");
        }
    }
}

/// Codepoints, so a failing case is readable in the terminal.
fn cps(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i == 40 {
            out.push_str("...");
            break;
        }
        out.push_str(&format!("{:04X} ", c as u32));
    }
    out
}
