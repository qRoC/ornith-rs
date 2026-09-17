//! The tokenizer: `tokenizer.json` read and run by hand.
//!
//! The pipeline `tokenizer.json` describes, in the order `tokenizers` runs it:
//!
//! ```text
//!   added vocabulary   split the raw text on the 33 added tokens
//!   normalizer         NFC, on each piece that is not an added token
//!   pre_tokenizer      Split(the GPT-4 regex, Isolated) then ByteLevel
//!   model              BPE, 248044 entries and 247587 ranked merges
//!   post_processor     ByteLevel -- adds nothing
//! ```
//!
//! The split pattern has two spellings and they are not equivalent: see
//! `split::Dialect`. `load` takes the one `tokenizer.json` declares;
//! `load_as`, and `Dialect::Transformers`, take the one `transformers` 5.x
//! installs over the top of it and so the one `mlx_lm.generate` runs.
//!
//! The added vocabulary is 26 tokens from `tokenizer.json` plus 7 more that
//! only `tokenizer_config.json` carries (`<|audio_start|>` and the `<tts_*>`
//! family, ids 248070..248076). `transformers` adds those at load time, so
//! `tokenizer.encode` knows them and so must this.
//!
//! Every added token here has `normalized: false`, `lstrip: false`,
//! `rstrip: false` and `single_word: false`, which is why the split below
//! runs against the raw text and takes the leftmost-longest match --
//! `AddedVocabulary`'s `split_trie` is an Aho-Corasick with
//! `MatchKind::LeftmostLongest`.

pub mod bpe;
pub mod nfc;
pub mod split;
pub mod tables;

use std::collections::HashMap;
use std::path::Path;

use crate::json::{self, Value};
use split::Dialect;

pub struct Tokenizer {
    pub dialect: Dialect,
    /// Token string -> id, for the 248044 BPE entries only.
    vocab: HashMap<String, u32>,
    /// id -> token string, for every id including the added ones.
    tokens: Vec<String>,
    merges: bpe::Merges,
    /// `(content, id)` for the added tokens, longest first so a left-to-right
    /// scan that takes the first hit is leftmost-longest.
    added: Vec<(String, u32)>,
    /// The subset of `added` that is marked `special`.
    special: Vec<u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub eos_ids: Vec<u32>,
}

/// The GPT-2 byte alphabet: the printable Latin-1 bytes stand for themselves,
/// every other byte gets a codepoint from U+0100 up, in byte order.
fn byte_alphabet() -> [char; 256] {
    let mut out = ['\0'; 256];
    let printable = |b: u8| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    let mut n = 0u32;
    for b in 0..=255u8 {
        out[b as usize] = if printable(b) {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    out
}

impl Tokenizer {
    /// The tokenizer exactly as `tokenizer.json` describes it.
    pub fn load(model_dir: &Path) -> Result<Tokenizer, String> {
        Tokenizer::load_as(model_dir, Dialect::TokenizerJson)
    }

    pub fn load_as(model_dir: &Path, dialect: Dialect) -> Result<Tokenizer, String> {
        let raw = std::fs::read(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("tokenizer.json: {e}"))?;
        let tj = json::parse(&raw).map_err(|e| format!("tokenizer.json: {e}"))?;
        let raw_cfg = std::fs::read(model_dir.join("tokenizer_config.json"))
            .map_err(|e| format!("tokenizer_config.json: {e}"))?;
        let cfg = json::parse(&raw_cfg).map_err(|e| format!("tokenizer_config.json: {e}"))?;

        // The pipeline is hard-coded, so refuse a tokenizer.json that asks for
        // anything else rather than quietly tokenizing differently.
        assert_eq!(tj.path("normalizer.type").and_then(|v| v.as_str()), Some("NFC"));
        assert_eq!(tj.path("model.type").and_then(|v| v.as_str()), Some("BPE"));
        for k in ["dropout", "unk_token"] {
            assert!(
                matches!(tj.path(&format!("model.{k}")), None | Some(Value::Null)),
                "model.{k} is set; the BPE port does not implement it"
            );
        }
        for k in ["fuse_unk", "byte_fallback", "ignore_merges"] {
            assert_eq!(
                tj.path(&format!("model.{k}")).and_then(|v| v.as_bool()),
                Some(false),
                "model.{k} is set; the BPE port does not implement it"
            );
        }
        for k in ["continuing_subword_prefix", "end_of_word_suffix"] {
            assert_eq!(tj.path(&format!("model.{k}")).and_then(|v| v.as_str()), Some(""));
        }
        assert_eq!(tj.path("pre_tokenizer.type").and_then(|v| v.as_str()), Some("Sequence"));
        let pres = tj.path("pre_tokenizer.pretokenizers").and_then(|v| v.as_arr()).unwrap();
        assert_eq!(pres.len(), 2);
        assert_eq!(pres[0].get("type").and_then(|v| v.as_str()), Some("Split"));
        assert_eq!(pres[0].get("behavior").and_then(|v| v.as_str()), Some("Isolated"));
        assert_eq!(
            pres[0].path("pattern.Regex").and_then(|v| v.as_str()),
            Some(Dialect::TOKENIZER_JSON_PATTERN),
            "tokenizer.json carries a split pattern this port does not implement"
        );
        assert_eq!(pres[0].get("invert").and_then(|v| v.as_bool()), Some(false));
        assert_eq!(pres[1].get("type").and_then(|v| v.as_str()), Some("ByteLevel"));
        assert_eq!(pres[1].get("add_prefix_space").and_then(|v| v.as_bool()), Some(false));
        assert_eq!(tj.path("post_processor.type").and_then(|v| v.as_str()), Some("ByteLevel"));
        assert_eq!(tj.path("decoder.type").and_then(|v| v.as_str()), Some("ByteLevel"));
        assert_eq!(
            cfg.get("add_bos_token").and_then(|v| v.as_bool()),
            Some(false),
            "tokenizer_config asks for a BOS token"
        );

        let vocab_obj = match tj.path("model.vocab") {
            Some(Value::Obj(m)) => m,
            _ => return Err("model.vocab is not an object".into()),
        };
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(vocab_obj.len());
        let mut tokens: Vec<String> = vec![String::new(); vocab_obj.len()];
        for (k, v) in vocab_obj {
            let id = v.as_usize().ok_or("vocab value is not an integer")?;
            if id >= tokens.len() {
                return Err(format!("vocab id {id} is past the end of the vocab"));
            }
            tokens[id] = k.clone();
            vocab.insert(k.clone(), id as u32);
        }

        let merge_arr = tj.path("model.merges").and_then(|v| v.as_arr()).unwrap();
        let mut merges: bpe::Merges = HashMap::with_capacity(merge_arr.len());
        for (rank, m) in merge_arr.iter().enumerate() {
            let s = m.as_str().ok_or("a merge is not a string")?;
            let (a, b) = s.split_once(' ').ok_or_else(|| format!("bad merge {s:?}"))?;
            let ia = *vocab.get(a).ok_or_else(|| format!("merge lhs {a:?} not in vocab"))?;
            let ib = *vocab.get(b).ok_or_else(|| format!("merge rhs {b:?} not in vocab"))?;
            let joined = format!("{a}{b}");
            let new_id =
                *vocab.get(&joined).ok_or_else(|| format!("merge result {joined:?} not in vocab"))?;
            merges.insert((ia, ib), (rank as u32, new_id));
        }

        // Added tokens: tokenizer.json first, then the ones only the config
        // knows, in id order.
        let mut added: Vec<(String, u32)> = Vec::new();
        let mut special: Vec<u32> = Vec::new();
        let mut entries: Vec<(usize, &Value)> = tj
            .get("added_tokens")
            .and_then(|v| v.as_arr())
            .unwrap()
            .iter()
            .map(|t| (t.get("id").and_then(|v| v.as_usize()).unwrap(), t))
            .collect();
        if let Some(Value::Obj(dec)) = cfg.get("added_tokens_decoder") {
            let mut extra: Vec<(usize, &Value)> = dec
                .iter()
                .map(|(k, v)| (k.parse::<usize>().expect("added_tokens_decoder key"), v))
                .collect();
            extra.sort_by_key(|(k, _)| *k);
            for (id, t) in extra {
                if !entries.iter().any(|(i, _)| *i == id) {
                    entries.push((id, t));
                }
            }
        }
        entries.sort_by_key(|(id, _)| *id);
        for (id, t) in entries {
            let content = t.get("content").and_then(|v| v.as_str()).unwrap();
            for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
                assert_eq!(
                    t.get(flag).and_then(|v| v.as_bool()),
                    Some(false),
                    "added token {content:?} sets {flag}; the split does not implement it"
                );
            }
            if tokens.len() <= id {
                tokens.resize(id + 1, String::new());
            }
            tokens[id] = content.to_string();
            added.push((content.to_string(), id as u32));
            if t.get("special").and_then(|v| v.as_bool()) == Some(true) {
                special.push(id as u32);
            }
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));

        let byte_to_char = byte_alphabet();
        let char_to_byte: HashMap<char, u8> =
            byte_to_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();

        // `generation_config.json` is what mlx-lm's `load` hands the sampler;
        // `tokenizer_config.eos_token` is the chat one and is already in it.
        let mut eos_ids: Vec<u32> = Vec::new();
        let gen_cfg = std::fs::read(model_dir.join("generation_config.json"))
            .ok()
            .and_then(|r| json::parse(&r).ok());
        if let Some(g) = &gen_cfg {
            match g.get("eos_token_id") {
                Some(Value::Arr(a)) => {
                    eos_ids.extend(a.iter().filter_map(|v| v.as_usize()).map(|v| v as u32))
                }
                Some(v) => eos_ids.extend(v.as_usize().map(|v| v as u32)),
                None => {}
            }
        }
        if eos_ids.is_empty() {
            let eos = cfg.get("eos_token").and_then(|v| v.as_str()).unwrap_or("<|im_end|>");
            eos_ids.extend(added.iter().find(|(c, _)| c == eos).map(|(_, i)| *i));
        }

        Ok(Tokenizer {
            dialect,
            vocab,
            tokens,
            merges,
            added,
            special,
            byte_to_char,
            char_to_byte,
            eos_ids,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|(c, _)| c == token)
            .map(|(_, i)| *i)
            .or_else(|| self.vocab.get(token).copied())
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.contains(&id)
    }

    /// `tokenizer.encode(text, add_special_tokens=False)`.
    ///
    /// The post-processor is ByteLevel, which contributes no tokens, and
    /// `add_bos_token` is false, so this is the whole of `encode`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        let mut plain_from = 0;
        while i < bytes.len() {
            let hit = self
                .added
                .iter()
                .find(|(c, _)| bytes[i..].starts_with(c.as_bytes()))
                .map(|(c, id)| (c.len(), *id));
            match hit {
                Some((len, id)) => {
                    if plain_from < i {
                        self.encode_plain(&text[plain_from..i], &mut out);
                    }
                    out.push(id);
                    i += len;
                    plain_from = i;
                }
                None => i += 1,
            }
        }
        if plain_from < text.len() {
            self.encode_plain(&text[plain_from..], &mut out);
        }
        out
    }

    /// Normalize, pre-tokenize and BPE one stretch of text with no added
    /// tokens in it.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let normalized = nfc::nfc(text);
        let mut word: Vec<u32> = Vec::new();
        let mut buf = [0u8; 4];
        for piece in split::pre_split(&normalized, self.dialect) {
            word.clear();
            for &b in piece.as_bytes() {
                let s: &str = self.byte_to_char[b as usize].encode_utf8(&mut buf);
                word.push(self.vocab[s]);
            }
            out.extend(bpe::merge_all(&word, &self.merges));
        }
    }

    /// `tokenizer.decode(ids)`: the ByteLevel decoder, i.e. map every
    /// character of every token back to the byte it stands for and read the
    /// result as UTF-8.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let s = self.token(id);
            for c in s.chars() {
                match self.char_to_byte.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The token string an id stands for.
    pub fn token(&self, id: u32) -> &str {
        // mlx-lm's fallback for an id past the end of the vocabulary -- this
        // model's config pads `vocab_size` to 248320 while the tokenizer holds
        // 248077 entries.
        self.tokens.get(id as usize).map(|s| s.as_str()).filter(|s| !s.is_empty()).unwrap_or("!")
    }

    pub fn detokenizer(&self) -> Detokenizer<'_> {
        Detokenizer {
            tk: self,
            text: String::new(),
            unflushed: String::new(),
            offset: 0,
            tokens: Vec::new(),
        }
    }
}

/// A port of mlx-lm's `BPEStreamingDetokenizer` (`tokenizer_utils.py`), which
/// is the detokenizer `mlx_lm.generate` prints through for this model -- so
/// its two quirks are part of the output text and are kept:
///
///   * a single leading space at the very start of the generation is dropped
///     (`_maybe_trim_space`),
///   * `_decode_bytes` tests the byte it looked up for truthiness, so the
///     character standing for byte 0 falls through to the "not a byte-level
///     character" branch and is emitted as its own UTF-8.
pub struct Detokenizer<'a> {
    tk: &'a Tokenizer,
    pub text: String,
    unflushed: String,
    offset: usize,
    pub tokens: Vec<u32>,
}

impl<'a> Detokenizer<'a> {
    fn decode_bytes(&self, seq: &str) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for c in seq.chars() {
            match self.tk.char_to_byte.get(&c) {
                Some(&b) if b != 0 => bytes.push(b),
                _ => {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn maybe_trim_space(&self, current: &str) -> String {
        if current.is_empty() || !current.starts_with(' ') || !self.text.is_empty() {
            current.to_string()
        } else {
            current[1..].to_string()
        }
    }

    pub fn add_token(&mut self, token: u32) {
        self.tokens.push(token);
        let v = self.tk.token(token).to_string();
        self.unflushed.push_str(&v);
        let text = self.decode_bytes(&self.unflushed);

        // Hold a partial UTF-8 sequence back until it completes, and hold a
        // lone space back until the next token says whether to trim it.
        let lone_space = v.chars().count() == 1
            && self.tk.char_to_byte.get(&v.chars().next().unwrap()) == Some(&32);
        if !text.ends_with('\u{fffd}') && !lone_space {
            let trimmed = self.maybe_trim_space(&text);
            self.text.push_str(&trimmed);
            self.unflushed.clear();
        }
    }

    pub fn finalize(&mut self) {
        let mut bytes: Vec<u8> = Vec::new();
        for c in self.unflushed.chars() {
            bytes.push(self.tk.char_to_byte[&c]);
        }
        let current = String::from_utf8_lossy(&bytes).into_owned();
        let trimmed = self.maybe_trim_space(&current);
        self.text.push_str(&trimmed);
        self.unflushed.clear();
    }

    /// The text produced since this was last called.
    pub fn last_segment(&mut self) -> String {
        let seg = self.text[self.offset..].to_string();
        self.offset = self.text.len();
        seg
    }
}
