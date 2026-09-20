//! Minimal JSON reader, and the writer the chat template needs.
//!
//! Needed in three places: `config.json`, the safetensors header and
//! `tokenizer.json` (12.8 MB, so the parser stays allocation-light and does a
//! single pass).
//!
//! Two properties exist for `chat::apply_chat_template`, which renders tool
//! schemas through jinja's `tojson` -- the one `transformers` replaces with
//! `json.dumps(..., ensure_ascii=False)` -- and so has to come out byte for
//! byte the same as Python's:
//!
//! - **objects keep their source order**, because a Python dict does and
//!   `json.dumps` writes it in that order. `Map` is a vector of pairs, not a
//!   hash map;
//! - **integers stay integers.** Python writes `2` for `2` and `2.0` for
//!   `2.0`, so a single `f64` would lose the difference. `Num` carries the
//!   ones a JSON document writes with a `.` or an exponent, `Int` the rest;
//!   every accessor reads both.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// An integer literal: no `.`, no exponent, and it fits in an `i64`.
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Map),
}

/// A JSON object, in source order.
///
/// Lookup is a linear scan. The two large objects this port reads --
/// `model.vocab` at 248044 entries and a safetensors header at a few
/// thousand -- are only ever iterated; everything looked up by key has a
/// handful of entries.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Map(Vec<(String, Value)>);

impl Map {
    pub fn new() -> Map {
        Map(Vec::new())
    }

    /// The **last** entry under `key`, which is the value Python's reader
    /// would have kept and the value the `HashMap` this replaced did. No
    /// file here has a duplicate key; what differs if one ever appears is
    /// that the writer would echo both entries where Python collapses them.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (String, Value)> {
        self.0.iter()
    }

    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.0.iter().map(|(_, v)| v)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }

    /// Append an entry, in `O(1)`: `model.vocab` is 248044 of them, so a
    /// reader that checked for a duplicate on every key would be quadratic.
    pub fn push(&mut self, key: impl Into<String>, value: Value) {
        self.0.push((key.into(), value));
    }
}

impl<'a> IntoIterator for &'a Map {
    type Item = &'a (String, Value);
    type IntoIter = std::slice::Iter<'a, (String, Value)>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl FromIterator<(String, Value)> for Map {
    fn from_iter<I: IntoIterator<Item = (String, Value)>>(it: I) -> Map {
        Map(it.into_iter().collect())
    }
}

#[derive(Debug)]
pub struct Error {
    pub msg: String,
    pub pos: usize,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json: {} at byte {}", self.msg, self.pos)
    }
}

impl std::error::Error for Error {}

pub fn parse(bytes: &[u8]) -> Result<Value, Error> {
    let mut p = Parser { b: bytes, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(p.err("trailing bytes"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> Error {
        Error { msg: msg.to_string(), pos: self.i }
    }

    fn ws(&mut self) {
        while self.i < self.b.len() {
            match self.b[self.i] {
                b' ' | b'\t' | b'\n' | b'\r' => self.i += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Result<u8, Error> {
        self.b.get(self.i).copied().ok_or_else(|| self.err("unexpected end"))
    }

    fn eat(&mut self, c: u8) -> Result<(), Error> {
        if self.peek()? != c {
            return Err(self.err(&format!("expected {:?}", c as char)));
        }
        self.i += 1;
        Ok(())
    }

    fn lit(&mut self, word: &[u8]) -> Result<(), Error> {
        if self.b.len() - self.i < word.len() || &self.b[self.i..self.i + word.len()] != word {
            return Err(self.err("bad literal"));
        }
        self.i += word.len();
        Ok(())
    }

    fn value(&mut self) -> Result<Value, Error> {
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Value::Str(self.string()?)),
            b't' => {
                self.lit(b"true")?;
                Ok(Value::Bool(true))
            }
            b'f' => {
                self.lit(b"false")?;
                Ok(Value::Bool(false))
            }
            b'n' => {
                self.lit(b"null")?;
                Ok(Value::Null)
            }
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Result<Value, Error> {
        self.eat(b'{')?;
        let mut map = Map::new();
        self.ws();
        if self.peek()? == b'}' {
            self.i += 1;
            return Ok(Value::Obj(map));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value()?;
            map.push(k, v);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(Value::Obj(map));
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self) -> Result<Value, Error> {
        self.eat(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.peek()? == b']' {
            self.i += 1;
            return Ok(Value::Arr(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Ok(Value::Arr(out));
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    fn string(&mut self) -> Result<String, Error> {
        self.eat(b'"')?;
        let start = self.i;
        // Fast path: no escapes, pure copy of the slice.
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if c == b'"' {
                let s = std::str::from_utf8(&self.b[start..self.i])
                    .map_err(|_| self.err("invalid utf-8"))?
                    .to_string();
                self.i += 1;
                return Ok(s);
            }
            if c == b'\\' {
                break;
            }
            self.i += 1;
        }
        // Slow path: rebuild with escapes resolved.
        let mut out = String::from(
            std::str::from_utf8(&self.b[start..self.i]).map_err(|_| self.err("invalid utf-8"))?,
        );
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.peek()?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            // Surrogate pair: \uD800-\uDBFF must be followed by
                            // a low surrogate. tokenizer.json contains these.
                            let ch = if (0xD800..0xDC00).contains(&hi) {
                                if self.peek()? != b'\\' {
                                    return Err(self.err("lone high surrogate"));
                                }
                                self.i += 1;
                                self.eat(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("bad low surrogate"));
                                }
                                let c = 0x10000 + ((hi as u32 - 0xD800) << 10) + (lo as u32 - 0xDC00);
                                char::from_u32(c).ok_or_else(|| self.err("bad surrogate pair"))?
                            } else {
                                char::from_u32(hi as u32).ok_or_else(|| self.err("bad \\u"))?
                            };
                            out.push(ch);
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
                _ => {
                    // Multi-byte UTF-8 passes through verbatim.
                    let start = self.i - 1;
                    let len = utf8_len(c);
                    self.i = start + len;
                    if self.i > self.b.len() {
                        return Err(self.err("truncated utf-8"));
                    }
                    out.push_str(
                        std::str::from_utf8(&self.b[start..self.i])
                            .map_err(|_| self.err("invalid utf-8"))?,
                    );
                }
            }
        }
        Err(self.err("unterminated string"))
    }

    fn hex4(&mut self) -> Result<u16, Error> {
        if self.i + 4 > self.b.len() {
            return Err(self.err("short \\u"));
        }
        let mut v: u16 = 0;
        for _ in 0..4 {
            let c = self.b[self.i];
            self.i += 1;
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return Err(self.err("bad hex digit")),
            };
            v = v * 16 + d as u16;
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<Value, Error> {
        let start = self.i;
        if self.peek()? == b'-' {
            self.i += 1;
        }
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => self.i += 1,
                _ => break,
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| self.err("bad number"))?;
        // An integer literal stays an integer, so the writer can put it
        // back the way Python would. One difference from Python's reader,
        // which has no fixed width: an integer past `i64` falls through to
        // the float and would be written back in exponential form. Nothing
        // this port reads or writes carries one.
        if !s.contains(['.', 'e', 'E'])
            && let Ok(i) = s.parse::<i64>()
        {
            return Ok(Value::Int(i));
        }
        s.parse::<f64>().map(Value::Num).map_err(|_| self.err("bad number"))
    }
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// Typed accessors. These mirror how the Python side reads the same configs:
// `cfg["text_config"]["num_hidden_layers"]` becomes
// `cfg.get("text_config").get("num_hidden_layers").usize()`.
// ---------------------------------------------------------------------------

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn idx(&self, i: usize) -> Option<&Value> {
        match self {
            Value::Arr(a) => a.get(i),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        self.as_f64().map(|v| v as f32)
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Int(i) => usize::try_from(*i).ok(),
            _ => self.as_f64().map(|v| v as usize),
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => self.as_f64().map(|v| v as i64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&Map> {
        match self {
            Value::Obj(m) => Some(m),
            _ => None,
        }
    }

    /// `a.b.c` style lookup, used for the nested model configs.
    pub fn path(&self, dotted: &str) -> Option<&Value> {
        let mut cur = self;
        for part in dotted.split('.') {
            cur = cur.get(part)?;
        }
        Some(cur)
    }

    /// `json.dumps(value, ensure_ascii=False)`, which is what `transformers`
    /// installs as jinja's `tojson` filter -- the chat template's tool
    /// schemas go through it and have to match character for character.
    ///
    /// That means Python's defaults and not JSON's freedoms: `", "` between
    /// items and `": "` after a key, non-ASCII written as itself, and only
    /// `"`, `\\` and the C0 controls escaped (not `/`, not U+2028). Objects
    /// come out in source order, which is why `Map` keeps it.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            Value::Int(i) => out.push_str(&i.to_string()),
            Value::Num(v) => out.push_str(&py_repr(*v)),
            Value::Str(s) => write_str(s, out),
            Value::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_json(out);
                }
                out.push(']');
            }
            Value::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_str(k, out);
                    out.push_str(": ");
                    v.write_json(out);
                }
                out.push('}');
            }
        }
    }
}

/// `py_encode_basestring`: the short escapes, C0 as `\uXXXX`, everything else
/// verbatim.
fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `repr(float)`: the shortest string that reads back, in fixed notation
/// unless the decimal point lands outside `-4 < decpt <= 16`, and with a
/// `.0` glued on when fixed notation would otherwise look like an integer.
/// CPython's `format_float_short` for format code `'r'`.
///
/// Rust's `{:e}` already gives the shortest round-tripping digits; all that
/// is left is Python's choice of where to put the point.
fn py_repr(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let e = format!("{:e}", v);
    let (mantissa, exp) = e.split_once('e').expect("{:e} writes an exponent");
    let exp: i32 = exp.parse().expect("{:e} writes a decimal exponent");
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let decpt = exp + 1;

    if decpt <= -4 || decpt > 16 {
        let mut out = format!("{sign}{}", &digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
        out
    } else if decpt <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat(-decpt as usize))
    } else if decpt as usize >= digits.len() {
        format!("{sign}{digits}{}.0", "0".repeat(decpt as usize - digits.len()))
    } else {
        let (a, b) = digits.split_at(decpt as usize);
        format!("{sign}{a}.{b}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_and_nesting() {
        let v = parse(br#"{"a": 1, "b": [true, null, -2.5e3], "c": {"d": "x"}}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_usize(), Some(1));
        assert_eq!(v.path("c.d").unwrap().as_str(), Some("x"));
        let b = v.get("b").unwrap().as_arr().unwrap();
        assert_eq!(b[0], Value::Bool(true));
        assert_eq!(b[1], Value::Null);
        assert_eq!(b[2].as_f64(), Some(-2500.0));
    }

    #[test]
    fn escapes_and_surrogates() {
        // Written as bytes so the escapes in the JSON survive the Rust
        // source: the input contains a literal backslash-u sequence.
        let src = b"{\"s\": \"a\\\"b\\\\c\\nd\\u00e9\\ud83d\\ude00\"}";
        let v = parse(src).unwrap();
        assert_eq!(
            v.get("s").unwrap().as_str(),
            Some("a\"b\\c\nd\u{e9}\u{1f600}")
        );
    }

    #[test]
    fn integers_survive_the_round_trip() {
        // The chat template's `tojson` is the reason: Python writes `2` for
        // an int and `2.0` for a float, and a single f64 could not tell them
        // apart.
        let v = parse(br#"{"a": 2, "b": 2.0, "c": -0.0, "d": 1e16, "e": 1e-5}"#).unwrap();
        assert_eq!(v.get("a"), Some(&Value::Int(2)));
        assert_eq!(v.get("b"), Some(&Value::Num(2.0)));
        assert_eq!(v.to_json(), r#"{"a": 2, "b": 2.0, "c": -0.0, "d": 1e+16, "e": 1e-05}"#);
    }

    #[test]
    fn objects_keep_their_source_order() {
        let src = br#"{"z": 1, "a": 2, "m": {"y": [1, 2], "b": null}}"#;
        let v = parse(src).unwrap();
        assert_eq!(v.to_json(), std::str::from_utf8(src).unwrap());
    }

    #[test]
    fn writing_escapes_what_python_escapes() {
        // ensure_ascii=False: the emoji and the Cyrillic go out as
        // themselves, the tab and the control do not, and `/` is untouched.
        let v = Value::Str("a\"b\\c\n\td/\u{1}\u{7f}\u{444}\u{1f600}".to_string());
        assert_eq!(v.to_json(), "\"a\\\"b\\\\c\\n\\td/\\u0001\u{7f}\u{444}\u{1f600}\"");
    }

    #[test]
    fn byte_level_token_strings() {
        // The BPE vocab is full of these: U+0120 (G-with-dot) is the
        // byte-level stand-in for a space.
        let v = parse(b"{\"\\u0120the\": 1}").unwrap();
        assert!(v.get("\u{120}the").is_some());
    }
}
