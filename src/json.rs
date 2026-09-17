//! Minimal JSON reader.
//!
//! Needed in three places: `config.json`, the safetensors header and
//! `tokenizer.json` (12.8 MB, so the parser stays allocation-light and does a
//! single pass). Numbers are kept as `f64`; every integer we read out of these
//! files is far below 2^53 so nothing is lost.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(HashMap<String, Value>),
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
        let mut map = HashMap::new();
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
            map.insert(k, v);
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
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        self.as_f64().map(|v| v as f32)
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().map(|v| v as usize)
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|v| v as i64)
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

    pub fn as_obj(&self) -> Option<&HashMap<String, Value>> {
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
    fn byte_level_token_strings() {
        // The BPE vocab is full of these: U+0120 (G-with-dot) is the
        // byte-level stand-in for a space.
        let v = parse(b"{\"\\u0120the\": 1}").unwrap();
        assert!(v.get("\u{120}the").is_some());
    }
}
