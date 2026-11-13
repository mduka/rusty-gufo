//! The decoding half of `tokenizer.json` (`tokenizer.hpp`, `tokenizer.cpp`):
//! SentencePiece BPE with a Metaspace decoder, token IDs to text.
//! Transcription never encodes text.

use std::path::Path;
use std::sync::Arc;

use gufo_core::json::{self, Value};

use crate::language::tag_locale;

/// U+2581, SentencePiece's word boundary.
const METASPACE: &str = "\u{2581}";

/// The piece with every "▁" as a space (`Spaces`).
fn spaces(piece: &str) -> String {
    piece.replace(METASPACE, " ")
}

/// The whitespace of C's `isspace` in the C locale (`Whitespace`).
fn whitespace(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

fn id(value: &Value) -> Result<u32, String> {
    let number = value.as_double(f64::NAN);
    if !value.is_number() || number < 0.0 || number != number.floor() || number > 1e7 {
        return Err("tokenizer.json has an invalid token ID".into());
    }
    Ok(number as u32)
}

/// Token IDs to their pieces and texts (`Tokenizer`).
#[derive(Debug, Clone, Default)]
pub struct Tokenizer {
    pieces: Vec<String>,
    texts: Vec<String>,
    special: Vec<bool>,
}

impl Tokenizer {
    /// Reads and parses `tokenizer.json` (`Load`).
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read(path).map_err(|_| format!("cannot read {}", path.display()))?;
        Self::parse_bytes(&text)
    }

    /// Parses a `tokenizer.json` text (`Parse`).
    pub fn parse(text: &str) -> Result<Self, String> {
        Self::parse_bytes(text.as_bytes())
    }

    fn parse_bytes(text: &[u8]) -> Result<Self, String> {
        let root = json::parse(text).map_err(|error| error.to_string())?;
        let model = root
            .find("model")
            .filter(|model| model.is_object())
            .ok_or("tokenizer.json lacks a model")?;
        if model
            .find("type")
            .is_none_or(|kind| !kind.is_string() || kind.str() != "BPE")
        {
            return Err("tokenizer.json is not a BPE model".into());
        }
        if root
            .find("decoder")
            .and_then(|decoder| decoder.find("type"))
            .is_none_or(|kind| !kind.is_string() || kind.str() != "Metaspace")
        {
            return Err("tokenizer.json lacks a Metaspace decoder".into());
        }
        let vocab = model
            .find("vocab")
            .filter(|vocab| vocab.is_object())
            .ok_or("tokenizer.json lacks a vocabulary")?;

        let mut tokenizer = Self::default();
        for (piece, value) in vocab.members() {
            tokenizer.put(piece, id(value)?, false);
        }
        if let Some(added) = root.find("added_tokens").filter(|added| added.is_array()) {
            for token in added.items() {
                let content = token.find("content").filter(|content| content.is_string());
                let (Some(content), Some(value)) = (content, token.find("id")) else {
                    return Err("tokenizer.json has a malformed added token".into());
                };
                let special = token
                    .find("special")
                    .is_some_and(|special| special.as_bool(false));
                tokenizer.put(content.str(), id(value)?, special);
            }
        }
        if tokenizer.pieces.is_empty() {
            return Err("tokenizer.json has no tokens".into());
        }
        Ok(tokenizer)
    }

    fn put(&mut self, piece: &str, id: u32, special: bool) {
        let id = id as usize;
        if id >= self.pieces.len() {
            self.pieces.resize(id + 1, String::new());
            self.texts.resize(id + 1, String::new());
            self.special.resize(id + 1, true);
        }
        self.pieces[id] = piece.to_owned();
        self.texts[id] = if special {
            String::new()
        } else {
            spaces(piece)
        };
        self.special[id] = special;
    }

    /// Token IDs up to the largest one (`size`).
    pub fn size(&self) -> usize {
        self.pieces.len()
    }

    /// The token's text with "▁" as a space; empty for special tokens and
    /// IDs beyond the vocabulary (`Text`).
    pub fn text(&self, id: u32) -> &str {
        self.texts.get(id as usize).map_or("", String::as_str)
    }

    /// Special tokens, IDs no token holds and IDs beyond the vocabulary
    /// (`IsSpecial`).
    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).is_none_or(|special| *special)
    }

    /// The token as stored ("<de-DE>", "▁Heute") (`Piece`).
    pub fn piece(&self, id: u32) -> &str {
        self.pieces.get(id as usize).map_or("", String::as_str)
    }
}

/// The transcript of a raw decoding (`CleanText`): runs of whitespace become
/// one space, and none leads or trails. Appending to the raw text only
/// appends to (or completes the trailing space of) its transcript.
pub fn clean_text(raw: &str) -> String {
    let mut out = Vec::with_capacity(raw.len());
    let mut space = false;
    for &c in raw.as_bytes() {
        if whitespace(c) {
            space = !out.is_empty();
            continue;
        }
        if space {
            out.push(b' ');
        }
        space = false;
        out.push(c);
    }
    // Only ASCII whitespace bytes were dropped or added, so the UTF-8
    // sequences stay whole.
    String::from_utf8(out).unwrap_or_default()
}

/// Accumulates emitted tokens into the transcript and remembers the last
/// language tag the model emitted (`TextDecoder`).
#[derive(Debug, Clone)]
pub struct TextDecoder {
    tokenizer: Arc<Tokenizer>,
    ids: Vec<u32>,
    raw: String,
    language: Option<String>,
}

impl TextDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            tokenizer,
            ids: Vec::new(),
            raw: String::new(),
            language: None,
        }
    }

    /// Appends an emitted token (`Push`).
    pub fn push(&mut self, id: u32) {
        self.ids.push(id);
        if self.tokenizer.is_special(id) {
            if let Some(locale) = tag_locale(self.tokenizer.piece(id)) {
                self.language = Some(locale);
            }
            return;
        }
        self.raw.push_str(self.tokenizer.text(id));
    }

    pub fn reset(&mut self) {
        self.ids.clear();
        self.raw.clear();
        self.language = None;
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The decoded pieces with spaces, uncleaned.
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// The transcript (`clean_text` of the raw text).
    pub fn text(&self) -> String {
        clean_text(&self.raw)
    }

    /// The locale of the last language tag ("de-DE"), if any.
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }
}
