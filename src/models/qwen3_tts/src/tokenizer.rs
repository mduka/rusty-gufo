//! The Qwen2 byte-level BPE tokenizer of Qwen3-TTS (`tokenizer.cpp`):
//! `vocab.json`, `merges.txt` and the added tokens of
//! `tokenizer_config.json`. The pre-tokenizer, the merge loop and the added
//! token search follow the C++ code step by step, with the Unicode
//! predicates of `gufo_core::unicode` (ICU 78's answers) and NFC from
//! `unicode-normalization` (Unicode 17, as ICU 78).

use std::collections::HashMap;
use std::path::Path;

use gufo_core::json;
use gufo_core::unicode::{is_letter_or_mark, is_number};
use unicode_normalization::UnicodeNormalization;

const MAXIMUM_VOCABULARY_BYTES: u64 = 16 << 20;
const MAXIMUM_TOKENIZER_CONFIG_BYTES: u64 = 1 << 20;

struct AddedToken {
    content: String,
    id: u32,
}

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), u32>,
    added_tokens: HashMap<String, u32>,
    /// Longest first, then by content.
    added_alternatives: Vec<AddedToken>,
    byte_encoder: [String; 256],
}

fn read_file(path: &Path, maximum_bytes: u64) -> Result<String, String> {
    let size = std::fs::metadata(path)
        .map_err(|_| format!("cannot open {}", path.display()))?
        .len();
    if size > maximum_bytes {
        return Err(format!("unexpected file size for {}", path.display()));
    }
    std::fs::read_to_string(path).map_err(|_| format!("cannot read {}", path.display()))
}

/// `IsSpace`: Unicode `White_Space`, plus the information separators
/// U+001C..U+001F.
fn is_space(c: char) -> bool {
    gufo_core::unicode::is_space(c) || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `ContractionLength`: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll` or `'d`, the
/// letters compared after ASCII lowercasing.
fn contraction_length(points: &[(char, usize, usize)], index: usize) -> usize {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    if points[index].0 != '\'' {
        return 0;
    }
    for candidate in CONTRACTIONS {
        let length = candidate.len();
        if length > points.len() - index {
            continue;
        }
        let matches = candidate
            .bytes()
            .enumerate()
            .skip(1)
            .all(|(offset, expected)| {
                points[index + offset].0.to_ascii_lowercase() == char::from(expected)
            });
        if matches {
            return length;
        }
    }
    0
}

/// `Pretokenize`: NFC, then the Qwen2 pre-tokenizer pattern.
fn pretokenize(input: &str) -> Result<Vec<String>, String> {
    let text: String = input.nfc().collect();
    let points: Vec<(char, usize, usize)> = text
        .char_indices()
        .map(|(begin, c)| (c, begin, begin + c.len_utf8()))
        .collect();
    let slice = |begin: usize, end: usize| text[points[begin].1..points[end - 1].2].to_owned();
    let mut pieces = Vec::new();
    let mut index = 0;
    while index < points.len() {
        let contraction = contraction_length(&points, index);
        if contraction != 0 {
            pieces.push(slice(index, index + contraction));
            index += contraction;
            continue;
        }

        let value = points[index].0;
        let letter_start = if is_letter_or_mark(value) {
            Some(index)
        } else if value != '\r'
            && value != '\n'
            && !is_number(value)
            && index + 1 < points.len()
            && is_letter_or_mark(points[index + 1].0)
        {
            Some(index + 1)
        } else {
            None
        };
        if let Some(start) = letter_start {
            let mut stop = start;
            while stop < points.len() && is_letter_or_mark(points[stop].0) {
                stop += 1;
            }
            pieces.push(slice(index, stop));
            index = stop;
            continue;
        }

        if is_number(value) {
            let mut stop = index + 1;
            while stop < points.len() && stop - index < 3 && is_number(points[stop].0) {
                stop += 1;
            }
            pieces.push(slice(index, stop));
            index = stop;
            continue;
        }

        let punctuation_start = index
            + usize::from(
                value == ' '
                    && index + 1 < points.len()
                    && !is_space(points[index + 1].0)
                    && !is_letter_or_mark(points[index + 1].0)
                    && !is_number(points[index + 1].0),
            );
        let mut stop = punctuation_start;
        while stop < points.len()
            && !is_space(points[stop].0)
            && !is_letter_or_mark(points[stop].0)
            && !is_number(points[stop].0)
        {
            stop += 1;
        }
        if stop > punctuation_start {
            while stop < points.len() && (points[stop].0 == '\r' || points[stop].0 == '\n') {
                stop += 1;
            }
            pieces.push(slice(index, stop));
            index = stop;
            continue;
        }

        if is_space(value) {
            let mut whitespace_end = index + 1;
            while whitespace_end < points.len() && is_space(points[whitespace_end].0) {
                whitespace_end += 1;
            }
            let newline_end = (index..whitespace_end)
                .rfind(|&cursor| points[cursor].0 == '\r' || points[cursor].0 == '\n')
                .map(|cursor| cursor + 1);
            let piece_end = if let Some(end) = newline_end {
                end
            } else if whitespace_end == points.len() {
                whitespace_end
            } else if whitespace_end - index > 1 {
                whitespace_end - 1
            } else {
                index + 1
            };
            pieces.push(slice(index, piece_end));
            index = piece_end;
            continue;
        }

        return Err("cannot pre-tokenize Qwen3-TTS prompt".into());
    }
    Ok(pieces)
}

impl Tokenizer {
    /// `Tokenizer::Load`.
    pub fn load(model_root: &Path) -> Result<Self, String> {
        Self::load_files(model_root)
            .map_err(|message| format!("Qwen3-TTS tokenizer load failed: {message}"))
    }

    fn load_files(model_root: &Path) -> Result<Self, String> {
        let vocabulary = json::parse_str(&read_file(
            &model_root.join("vocab.json"),
            MAXIMUM_VOCABULARY_BYTES,
        )?)
        .map_err(|e| e.message().to_owned())?;
        let tokenizer_config = json::parse_str(&read_file(
            &model_root.join("tokenizer_config.json"),
            MAXIMUM_TOKENIZER_CONFIG_BYTES,
        )?)
        .map_err(|e| e.message().to_owned())?;
        if !vocabulary.is_object() || !tokenizer_config.is_object() {
            return Err("tokenizer JSON root must be an object".into());
        }

        let mut vocab = HashMap::with_capacity(vocabulary.members().len());
        for (symbol, identifier) in vocabulary.members() {
            if !identifier.is_number() {
                return Err("vocabulary ID must be numeric".into());
            }
            let parsed = identifier.as_size(0);
            let id = u32::try_from(parsed)
                .map_err(|_| "invalid or duplicate vocabulary entry".to_string())?;
            if vocab.insert(symbol.clone(), id).is_some() {
                return Err("invalid or duplicate vocabulary entry".into());
            }
        }

        let merges_path = model_root.join("merges.txt");
        let merges = std::fs::read_to_string(&merges_path)
            .map_err(|_| format!("cannot open {}", merges_path.display()))?;
        let mut merge_ranks = HashMap::new();
        let mut rank = 0u32;
        // `std::getline`: lines end at '\n' only; a trailing '\r' stays.
        for line in merges.split('\n') {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some(separator) = line.find(' ') else {
                return Err("invalid tokenizer merge entry".into());
            };
            if separator == 0 || separator + 1 == line.len() {
                return Err("invalid tokenizer merge entry".into());
            }
            let key = (
                line[..separator].to_owned(),
                line[separator + 1..].to_owned(),
            );
            if merge_ranks.insert(key, rank).is_some() {
                return Err("duplicate tokenizer merge entry".into());
            }
            rank += 1;
        }

        let decoder = tokenizer_config
            .find("added_tokens_decoder")
            .filter(|decoder| decoder.is_object())
            .ok_or("missing added_tokens_decoder")?;
        let mut added_tokens = HashMap::new();
        let mut added_alternatives = Vec::new();
        for (identifier, specification) in decoder.members() {
            if !specification.is_object() {
                return Err("invalid added token specification".into());
            }
            let content = specification.member_str("content", "");
            let id = parse_stoul(identifier)
                .filter(|_| !content.is_empty())
                .and_then(|parsed| u32::try_from(parsed).ok())
                .ok_or("invalid added token")?;
            if added_tokens.insert(content.clone(), id).is_some() {
                return Err("duplicate added token".into());
            }
            added_alternatives.push(AddedToken { content, id });
        }
        added_alternatives.sort_by(|left, right| {
            right
                .content
                .len()
                .cmp(&left.content.len())
                .then_with(|| left.content.cmp(&right.content))
        });

        let mut extra = 0u32;
        let byte_encoder = std::array::from_fn(|byte| {
            let byte = byte as u32;
            let visible = (u32::from(b'!')..=u32::from(b'~')).contains(&byte)
                || (0xA1..=0xAC).contains(&byte)
                || (0xAE..=0xFF).contains(&byte);
            let code = if visible {
                byte
            } else {
                extra += 1;
                256 + extra - 1
            };
            char::from_u32(code).unwrap_or_default().to_string()
        });
        Ok(Self {
            vocab,
            merge_ranks,
            added_tokens,
            added_alternatives,
            byte_encoder,
        })
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocab.len() + self.added_tokens.len()
    }

    /// `ApplyBpe`: repeatedly merges every occurrence of the lowest-ranked
    /// pair, left to right.
    fn apply_bpe(&self, piece: &str, ids: &mut Vec<u32>) -> Result<(), String> {
        let encoded: String = piece
            .bytes()
            .map(|byte| self.byte_encoder[usize::from(byte)].as_str())
            .collect();
        let mut symbols: Vec<String> = encoded.chars().map(String::from).collect();
        while symbols.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for index in 0..symbols.len() - 1 {
                // Owned keys: the map is keyed by pairs of strings.
                let key = (symbols[index].clone(), symbols[index + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&key)
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
                .vocab
                .get(symbol)
                .ok_or("Qwen3-TTS BPE symbol is absent from vocabulary")?;
            ids.push(*id);
        }
        Ok(())
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), String> {
        for piece in pretokenize(text)? {
            self.apply_bpe(&piece, ids)?;
        }
        Ok(())
    }

    /// `Tokenizer::Encode`: added tokens anywhere in the text (the earliest,
    /// the longest on a tie), plain BPE between them.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        let mut start = 0;
        while start < text.len() {
            let mut selected: Option<(usize, &AddedToken)> = None;
            for candidate in &self.added_alternatives {
                let Some(found) = text[start..].find(candidate.content.as_str()) else {
                    continue;
                };
                let offset = start + found;
                if selected.is_none_or(|(best, token)| {
                    offset < best
                        || (offset == best && candidate.content.len() > token.content.len())
                }) {
                    selected = Some((offset, candidate));
                }
            }
            let Some((offset, token)) = selected else {
                break;
            };
            if offset > start {
                self.encode_plain(&text[start..offset], &mut ids)?;
            }
            ids.push(token.id);
            start = offset + token.content.len();
        }
        if start < text.len() {
            self.encode_plain(&text[start..], &mut ids)?;
        }
        Ok(ids)
    }

    /// `EncodeAssistantPrompt`.
    pub fn encode_assistant_prompt(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode(&format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ))
    }

    /// `EncodeInstructionPrompt`.
    pub fn encode_instruction_prompt(&self, instruction: &str) -> Result<Vec<u32>, String> {
        self.encode(&format!("<|im_start|>user\n{instruction}<|im_end|>\n"))
    }

    /// `EncodeReferencePrompt`.
    pub fn encode_reference_prompt(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode(&format!("<|im_start|>assistant\n{text}<|im_end|>\n"))
    }
}

/// `std::stoul` that must consume the whole identifier: optional leading
/// whitespace, an optional sign, decimal digits.
fn parse_stoul(text: &str) -> Option<u64> {
    let trimmed = text.trim_start_matches([' ', '\t', '\n', '\r', '\u{b}', '\u{c}']);
    let (negative, digits) = match trimmed.as_bytes().first() {
        Some(b'+') => (false, &trimmed[1..]),
        Some(b'-') => (true, &trimmed[1..]),
        _ => (false, trimmed),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: u64 = digits.parse().ok()?;
    // `stoul` wraps a negative value; the result then exceeds u32 anyway.
    Some(if negative {
        value.wrapping_neg()
    } else {
        value
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenizer_splits_like_the_qwen2_pattern() {
        let pieces = pretokenize("Hello, world! It's 12345 \u{1c}x\n\n  done").unwrap();
        assert_eq!(
            pieces,
            [
                "Hello", ",", " world", "!", " It", "'s", " ", "123", "45", " ", "\u{1c}x", "\n\n",
                " ", " done"
            ]
        );
    }

    #[test]
    fn contractions_lowercase_only_ascii() {
        assert_eq!(pretokenize("'S'LL").unwrap(), ["'S", "'LL"]);
        assert_eq!(pretokenize("'\u{17f}").unwrap(), ["'\u{17f}"]);
    }

    #[test]
    fn stoul_needs_the_whole_identifier() {
        assert_eq!(parse_stoul("151643"), Some(151_643));
        assert_eq!(parse_stoul(" 7"), Some(7));
        assert_eq!(parse_stoul("7a"), None);
        assert_eq!(parse_stoul(""), None);
    }
}
