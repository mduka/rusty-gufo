//! OmniVoice's Qwen2 byte-level BPE tokenizer, read from `tokenizer.json`
//! (`tokenizer.cpp`). NFC normalization, the byte-level alphabet and the BPE
//! merge loop follow the Qwen3-TTS tokenizer; the pre-tokenizer follows
//! OmniVoice's Qwen2 pattern, which differs from Qwen3-TTS's: letters
//! exclude combining marks and digits split one at a time.

use std::collections::HashMap;
use std::path::Path;

use gufo_core::json::{self, Value};
use unicode_normalization::UnicodeNormalization;

use crate::unicode::{Codepoint, append_utf8, decode_utf8, is_space};

const MAX_TOKENIZER_BYTES: u64 = 64 << 20;
const PAIR_SEPARATOR: char = '\0';

// Character classes of the pattern (`is_space` is Oniguruma's \s)
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
fn is_letter(value: u32) -> bool {
    char::from_u32(value).is_some_and(gufo_core::unicode::is_letter)
}

fn is_number(value: u32) -> bool {
    char::from_u32(value).is_some_and(gufo_core::unicode::is_number)
}

fn is_newline(value: u32) -> bool {
    value == u32::from(b'\r') || value == u32::from(b'\n')
}

fn contraction_length(points: &[Codepoint], index: usize) -> usize {
    const SUFFIXES: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];
    if points[index].value != u32::from(b'\'') {
        return 0;
    }
    for suffix in SUFFIXES {
        if suffix.len() >= points.len() - index {
            continue;
        }
        let matches = suffix.bytes().enumerate().all(|(offset, letter)| {
            let mut value = points[index + 1 + offset].value;
            if (u32::from(b'A')..=u32::from(b'Z')).contains(&value) {
                value += u32::from(b'a' - b'A');
            }
            value == u32::from(letter)
        });
        if matches {
            return suffix.len() + 1;
        }
    }
    0
}

/// Length of the pattern's match at `index`, trying its alternatives in
/// order.
fn match_length(points: &[Codepoint], index: usize) -> usize {
    let count = points.len();
    let value = |at: usize| points[at].value;
    let length = contraction_length(points, index);
    if length != 0 {
        return length;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    let mut letters = index;
    if !is_letter(value(index))
        && !is_newline(value(index))
        && !is_number(value(index))
        && index + 1 < count
        && is_letter(value(index + 1))
    {
        letters = index + 1;
    }
    if is_letter(value(letters)) {
        let mut end = letters + 1;
        while end < count && is_letter(value(end)) {
            end += 1;
        }
        return end - index;
    }
    // \p{N}
    if is_number(value(index)) {
        return 1;
    }
    //  ?[^\s\p{L}\p{N}]+[\r\n]*
    let is_symbol =
        |at: usize| !is_space(value(at)) && !is_letter(value(at)) && !is_number(value(at));
    let mut symbols = index;
    if value(index) == u32::from(b' ') && index + 1 < count && is_symbol(index + 1) {
        symbols = index + 1;
    }
    if is_symbol(symbols) {
        let mut end = symbols + 1;
        while end < count && is_symbol(end) {
            end += 1;
        }
        while end < count && is_newline(value(end)) {
            end += 1;
        }
        return end - index;
    }
    // Every other character is whitespace.
    let mut run_end = index + 1;
    while run_end < count && is_space(value(run_end)) {
        run_end += 1;
    }
    // \s*[\r\n]+ ends after the run's last newline.
    for at in (index + 1..=run_end).rev() {
        if is_newline(value(at - 1)) {
            return at - index;
        }
    }
    // \s+(?!\S) leaves the run's last whitespace before a non-space; \s+
    // takes a single whitespace character.
    if run_end == count {
        return run_end - index;
    }
    if run_end - index > 1 {
        run_end - index - 1
    } else {
        1
    }
}

fn pair_key(left: &str, right: &str) -> String {
    let mut key = String::with_capacity(left.len() + right.len() + 1);
    key.push_str(left);
    key.push(PAIR_SEPARATOR);
    key.push_str(right);
    key
}

fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if metadata.len() > MAX_TOKENIZER_BYTES {
        return Err(format!("{} is implausibly large", path.display()));
    }
    let text = std::fs::read(path).map_err(|_| format!("cannot read {}", path.display()))?;
    if text.len() as u64 != metadata.len() {
        return Err(format!("cannot read {}", path.display()));
    }
    Ok(text)
}

fn token_id(value: &Value) -> Result<u32, String> {
    u32::try_from(value.as_size(usize::MAX))
        .map_err(|_| "token ID is not a 32-bit integer".to_owned())
}

#[derive(Debug, Clone)]
struct AddedToken {
    content: String,
    id: u32,
}

/// OmniVoice's Qwen2 byte-level BPE tokenizer, read from `tokenizer.json`.
///
/// Encodes like Hugging Face's `tokenizer(text, add_special_tokens=False)`:
/// added tokens match verbatim; the text between them is NFC-normalized,
/// split with the Qwen2 pre-tokenizer pattern and byte-pair encoded.
pub struct Tokenizer {
    vocabulary: HashMap<String, u32>,
    merge_ranks: HashMap<String, u32>,
    /// Longest first, so the leftmost match is also the longest.
    added_tokens: Vec<AddedToken>,
    byte_symbols: Vec<String>,
    vocabulary_size: usize,
}

impl Tokenizer {
    /// `Load`.
    pub fn load(tokenizer_json: &Path) -> Result<Self, String> {
        Self::parse(tokenizer_json)
            .map_err(|error| format!("OmniVoice tokenizer {}: {error}", tokenizer_json.display()))
    }

    fn parse(tokenizer_json: &Path) -> Result<Self, String> {
        let root = json::parse(&read_file(tokenizer_json)?).map_err(|error| error.to_string())?;
        if !root.is_object() {
            return Err("root is not an object".into());
        }
        if root
            .find("normalizer")
            .is_none_or(|normalizer| normalizer.member_str("type", "") != "NFC")
        {
            return Err("expected an NFC normalizer".into());
        }
        let model = root
            .find("model")
            .filter(|model| model.member_str("type", "") == "BPE")
            .ok_or("expected a BPE model")?;
        let (Some(vocabulary), Some(merges)) = (
            model.find("vocab").filter(|v| v.is_object()),
            model.find("merges").filter(|m| m.is_array()),
        ) else {
            return Err("BPE model lacks vocab or merges".into());
        };
        let mut tokenizer = Self {
            vocabulary: HashMap::with_capacity(vocabulary.len()),
            merge_ranks: HashMap::with_capacity(merges.len()),
            added_tokens: Vec::new(),
            byte_symbols: Vec::with_capacity(256),
            vocabulary_size: 0,
        };
        let mut limit = 0usize;
        for (symbol, id_value) in vocabulary.members() {
            let id = token_id(id_value)?;
            if tokenizer.vocabulary.insert(symbol.clone(), id).is_some() {
                return Err("duplicate vocabulary entry".into());
            }
            limit = limit.max(id as usize + 1);
        }
        for (rank, merge) in merges.items().iter().enumerate() {
            let items = merge.items();
            let key = if merge.is_array()
                && items.len() == 2
                && items[0].is_string()
                && items[1].is_string()
            {
                pair_key(items[0].str(), items[1].str())
            } else if merge.is_string() {
                let text = merge.str();
                match text.find(' ') {
                    Some(space) if space != 0 && space + 1 != text.len() => {
                        pair_key(&text[..space], &text[space + 1..])
                    }
                    _ => return Err("malformed merge".into()),
                }
            } else {
                return Err("malformed merge".into());
            };
            if tokenizer.merge_ranks.insert(key, rank as u32).is_some() {
                return Err("duplicate merge".into());
            }
        }
        let added = root
            .find("added_tokens")
            .filter(|added| added.is_array())
            .ok_or("missing added_tokens")?;
        for token in added.items() {
            let content = token.member_str("content", "");
            let Some(id) = token.find("id").filter(|_| !content.is_empty()) else {
                return Err("malformed added token".into());
            };
            // Only verbatim matches are implemented.
            for flag in ["normalized", "lstrip", "rstrip", "single_word"] {
                if token.find(flag).is_some_and(|option| option.as_bool(false)) {
                    return Err(format!("added token {content} sets {flag}"));
                }
            }
            let id = token_id(id)?;
            tokenizer.added_tokens.push(AddedToken { content, id });
            limit = limit.max(id as usize + 1);
        }
        tokenizer.added_tokens.sort_by(|left, right| {
            right
                .content
                .len()
                .cmp(&left.content.len())
                .then_with(|| left.content.cmp(&right.content))
        });
        tokenizer.vocabulary_size = limit;
        // GPT-2's byte-to-unicode alphabet.
        let mut shifted = 0u32;
        for byte in 0u32..256 {
            let visible = (u32::from(b'!')..=u32::from(b'~')).contains(&byte)
                || (0xA1..=0xAC).contains(&byte)
                || (0xAE..=0xFF).contains(&byte);
            let value = if visible {
                byte
            } else {
                shifted += 1;
                256 + shifted - 1
            };
            let mut symbol = Vec::new();
            append_utf8(value, &mut symbol);
            tokenizer
                .byte_symbols
                .push(String::from_utf8(symbol).map_err(|error| error.to_string())?);
        }
        Ok(tokenizer)
    }

    /// One past the largest token ID.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// The ID of an added token such as `<|text_start|>` (`AddedTokenId`).
    pub fn added_token_id(&self, content: &str) -> Option<u32> {
        self.added_tokens
            .iter()
            .find(|token| token.content == content)
            .map(|token| token.id)
    }

    fn encode_piece(&self, piece: &[u8], ids: &mut Vec<u32>) -> Result<(), String> {
        let mut symbols: Vec<String> = piece
            .iter()
            .map(|&byte| self.byte_symbols[byte as usize].clone())
            .collect();
        while symbols.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for index in 0..symbols.len() - 1 {
                if let Some(&rank) = self
                    .merge_ranks
                    .get(&pair_key(&symbols[index], &symbols[index + 1]))
                    && best.is_none_or(|(best_rank, _)| rank < best_rank)
                {
                    best = Some((rank, index));
                }
            }
            let Some((_, best)) = best else {
                break;
            };
            let left = symbols[best].clone();
            let right = symbols[best + 1].clone();
            let mut merged = Vec::with_capacity(symbols.len() - 1);
            let mut index = 0;
            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == left
                    && symbols[index + 1] == right
                {
                    merged.push(format!("{left}{right}"));
                    index += 2;
                } else {
                    merged.push(std::mem::take(&mut symbols[index]));
                    index += 1;
                }
            }
            symbols = merged;
        }
        for symbol in &symbols {
            let id = self
                .vocabulary
                .get(symbol)
                .ok_or("OmniVoice tokenizer: BPE symbol outside the vocabulary")?;
            ids.push(*id);
        }
        Ok(())
    }

    fn encode_ordinary(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), String> {
        let normalized: String = text.nfc().collect();
        let bytes = normalized.as_bytes();
        let points = decode_utf8(bytes).ok_or("OmniVoice tokenizer: text is not valid UTF-8")?;
        let mut index = 0;
        while index < points.len() {
            let length = match_length(&points, index);
            let begin = points[index].begin;
            let end = points[index + length - 1].end;
            self.encode_piece(&bytes[begin..end], ids)?;
            index += length;
        }
        Ok(())
    }

    /// `Encode`: the token IDs of `text` (UTF-8 bytes; malformed UTF-8 is
    /// refused).
    pub fn encode(&self, text: impl AsRef<[u8]>) -> Result<Vec<u32>, String> {
        let text = std::str::from_utf8(text.as_ref())
            .ok()
            .filter(|text| decode_utf8(text.as_bytes()).is_some())
            .ok_or("OmniVoice tokenizer: text is not valid UTF-8")?;
        let mut ids = Vec::new();
        let mut start = 0;
        while start < text.len() {
            let mut selected: Option<(&AddedToken, usize)> = None;
            for candidate in &self.added_tokens {
                if let Some(at) = text[start..]
                    .find(candidate.content.as_str())
                    .map(|at| at + start)
                    && selected.is_none_or(|(_, selected_at)| at < selected_at)
                {
                    selected = Some((candidate, at));
                }
            }
            let ordinary_end = selected.map_or(text.len(), |(_, at)| at);
            if ordinary_end > start {
                self.encode_ordinary(&text[start..ordinary_end], &mut ids)?;
            }
            let Some((token, at)) = selected else {
                break;
            };
            ids.push(token.id);
            start = at + token.content.len();
        }
        Ok(ids)
    }
}
