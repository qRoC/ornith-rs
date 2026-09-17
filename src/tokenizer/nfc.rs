//! NFC, the normalizer `tokenizer.json` names (`{"type": "NFC"}`).
//!
//! `tokenizers` gets this from the `unicode-normalization` crate; this is the
//! algorithm those and every other implementation follow, UAX #15:
//!
//!   1. canonical decomposition (D68), recursively applied, with the Hangul
//!      syllables done by arithmetic rather than from a table (D132),
//!   2. canonical ordering (D109): a stable sort of each maximal run of
//!      non-starters by combining class,
//!   3. canonical composition (D117): walk the string, and fold each character
//!      into the last starter when a primary composite exists and nothing
//!      blocks it.
//!
//! Text made only of characters that can take no part in any of that -- the
//! overwhelmingly common case -- is returned untouched, which is the same
//! NFC_QC quick check every implementation uses.

use super::tables::{CCC, COMPOSE, DECOMPOSE, DECOMPOSE_DATA, NFC_UNSAFE};

// UAX #15, D132.
const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn in_ranges(table: &[(u32, u32)], c: u32) -> bool {
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

/// Canonical combining class.
pub fn ccc(c: char) -> u8 {
    let c = c as u32;
    match CCC.binary_search_by(|&(lo, hi, _)| {
        if c < lo {
            std::cmp::Ordering::Greater
        } else if c > hi {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => CCC[i].2,
        Err(_) => 0,
    }
}

/// The primary composite of `a` and `b`, if there is one.
fn compose_pair(a: char, b: char) -> Option<char> {
    let (au, bu) = (a as u32, b as u32);

    // Hangul, UAX #15 D133: L + V and LV + T are arithmetic.
    let li = au.wrapping_sub(L_BASE);
    if li < L_COUNT {
        let vi = bu.wrapping_sub(V_BASE);
        if vi < V_COUNT {
            return char::from_u32(S_BASE + (li * V_COUNT + vi) * T_COUNT);
        }
    }
    let si = au.wrapping_sub(S_BASE);
    if si < S_COUNT && si % T_COUNT == 0 {
        let ti = bu.wrapping_sub(T_BASE);
        if ti > 0 && ti < T_COUNT {
            return char::from_u32(au + ti);
        }
    }

    let i = COMPOSE.binary_search_by(|&(x, y, _)| (x, y).cmp(&(au, bu))).ok()?;
    char::from_u32(COMPOSE[i].2)
}

/// Append the full canonical decomposition of `c` to `out`.
fn decompose_into(c: char, out: &mut Vec<char>) {
    let cu = c as u32;
    let si = cu.wrapping_sub(S_BASE);
    if si < S_COUNT {
        let (li, vi, ti) = (si / N_COUNT, (si % N_COUNT) / T_COUNT, si % T_COUNT);
        out.push(char::from_u32(L_BASE + li).unwrap());
        out.push(char::from_u32(V_BASE + vi).unwrap());
        if ti > 0 {
            out.push(char::from_u32(T_BASE + ti).unwrap());
        }
        return;
    }
    match DECOMPOSE.binary_search_by(|&(x, _, _)| x.cmp(&cu)) {
        Ok(i) => {
            let (_, off, len) = DECOMPOSE[i];
            for &d in &DECOMPOSE_DATA[off as usize..(off + len) as usize] {
                out.push(char::from_u32(d).unwrap());
            }
        }
        Err(_) => out.push(c),
    }
}

/// True when `s` contains nothing NFC could change or combine with, so it is
/// already in NFC and can be handed on untouched.
fn is_nfc_quick(s: &str) -> bool {
    s.chars().all(|c| !in_ranges(NFC_UNSAFE, c as u32))
}

pub fn nfc(s: &str) -> String {
    if is_nfc_quick(s) {
        return s.to_string();
    }

    // 1. decompose
    let mut d: Vec<char> = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose_into(c, &mut d);
    }

    // 2. canonical ordering: an insertion sort, stable within equal classes,
    //    over each run of non-starters.
    let mut i = 1;
    while i < d.len() {
        let cc = ccc(d[i]);
        if cc != 0 {
            let mut j = i;
            while j > 0 {
                let prev = ccc(d[j - 1]);
                if prev == 0 || prev <= cc {
                    break;
                }
                d.swap(j - 1, j);
                j -= 1;
            }
        }
        i += 1;
    }

    // 3. compose
    let mut out: Vec<char> = Vec::with_capacity(d.len());
    let mut starter: Option<usize> = None;
    // The combining class of the last character kept after that starter;
    // -1 while the next character would sit directly against the starter.
    let mut last_ccc: i32 = -1;
    for &c in &d {
        let cc = ccc(c) as i32;
        if let Some(sp) = starter {
            if last_ccc < cc || last_ccc == -1 {
                if let Some(x) = compose_pair(out[sp], c) {
                    out[sp] = x;
                    continue;
                }
            }
        }
        if cc == 0 {
            starter = Some(out.len());
            last_ccc = -1;
        } else {
            last_ccc = cc;
        }
        out.push(c);
    }
    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::nfc;

    #[test]
    fn composes_and_orders() {
        assert_eq!(nfc("abc"), "abc");
        // e + combining acute -> e-acute
        assert_eq!(nfc("e\u{301}"), "\u{e9}");
        // already composed
        assert_eq!(nfc("\u{e9}"), "\u{e9}");
        // U+1E0A (D dot above) + cedilla reorders to D + cedilla + dot above,
        // then composes to U+1E10 (D cedilla) + dot above.
        assert_eq!(nfc("\u{1e0a}\u{327}"), "\u{1e10}\u{307}");
        // singleton decomposition: angstrom sign -> A with ring above
        assert_eq!(nfc("\u{212b}"), "\u{c5}");
        // Hangul: jamo compose into a syllable
        assert_eq!(nfc("\u{1100}\u{1161}\u{11a8}"), "\u{ac01}");
        // a composition exclusion decomposes and does not go back together
        assert_eq!(nfc("\u{958}"), "\u{915}\u{93c}");
        assert_eq!(nfc("\u{915}\u{93c}"), "\u{915}\u{93c}");
        // a compatibility decomposition is not NFC's business
        assert_eq!(nfc("\u{f77}"), "\u{f77}");
        // a non-starter decomposition: U+0344 splits and does not recombine
        assert_eq!(nfc("\u{344}"), "\u{308}\u{301}");
    }
}
