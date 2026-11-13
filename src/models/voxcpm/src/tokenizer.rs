//! VoxCPM2's text tokenizer (`tokenizer.hpp`, `tokenizer.cpp`): the
//! SentencePiece-style BPE of `tokenizer.json` (normalizer: prepend "▁",
//! spaces to "▁"; no pre-tokenizer; byte fallback) followed by VoxCPM's
//! `mask_multichar_chinese_tokens`, which replaces every token made of two
//! or more CJK ideographs (U+4E00..U+9FFF) by the tokens of its single
//! characters. Added tokens (`<s>`, `<|im_end|>`, ...) in the text are
//! matched before normalization, as the `tokenizers` library does. No BOS is
//! added.
//!
//! The multi-character Chinese token split follows OpenBMB/VoxCPM's
//! mask_multichar_chinese_tokens (src/voxcpm/model/utils.py; Apache-2.0,
//! Copyright OpenBMB; see licenses/VoxCPM.txt).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use gufo_core::json::{self, Value};

/// U+2581.
const SPACE: &str = "\u{2581}";

/// Length of the UTF-8 sequence at `text[offset]`, 0 if it is invalid.
fn sequence_length(text: &[u8], offset: usize) -> usize {
    let lead = text[offset];
    let length = match lead {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => 0,
    };
    if length == 0 || offset + length > text.len() {
        return 0;
    }
    for index in 1..length {
        let byte = text[offset + index];
        if !(0x80..=0xBF).contains(&byte) {
            return 0;
        }
        if index == 1
            && ((lead == 0xE0 && byte < 0xA0)
                || (lead == 0xED && byte > 0x9F)
                || (lead == 0xF0 && byte < 0x90)
                || (lead == 0xF4 && byte > 0x8F))
        {
            return 0;
        }
    }
    length
}

/// The code-point pieces of valid UTF-8, or `None`.
fn characters(text: &[u8]) -> Option<Vec<&[u8]>> {
    let mut pieces = Vec::new();
    let mut offset = 0;
    while offset < text.len() {
        let length = sequence_length(text, offset);
        if length == 0 {
            return None;
        }
        pieces.push(&text[offset..offset + length]);
        offset += length;
    }
    Some(pieces)
}

/// The code point of a valid sequence.
fn decode(sequence: &[u8]) -> u32 {
    let lead = u32::from(sequence[0]);
    if sequence.len() == 1 {
        return lead;
    }
    let mut value = lead & (0x7F >> sequence.len());
    for &byte in &sequence[1..] {
        value = (value << 6) | (u32::from(byte) & 0x3F);
    }
    value
}

fn is_han(c: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&c)
}

fn pair_key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

#[derive(Clone, Copy)]
struct Merge {
    rank: u32,
    id: u32,
}

struct Added {
    content: Vec<u8>,
    id: u32,
}

/// The tokenizer (`Tokenizer`).
pub struct Tokenizer {
    vocab: HashMap<Vec<u8>, u32>,
    /// Key: left ID << 32 | right ID.
    merges: HashMap<u64, Merge>,
    added: Vec<Added>,
    byte_ids: [u32; 256],
    unk_id: u32,
    /// Multi-character CJK tokens and the IDs of their characters.
    splits: HashMap<u32, Vec<u32>>,
    size: usize,
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(format!("tokenizer.json: {message}"))
    }
}

/// `value[key]` as a string, empty when absent or not a string.
fn member_str<'a>(value: &'a Value, key: &str) -> &'a str {
    value.find(key).map_or("", Value::str)
}

impl Tokenizer {
    /// Reads and parses a tokenizer.json (`Load`).
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read(path).map_err(|_| format!("cannot read {}", path.display()))?;
        Self::parse_bytes(&text)
    }

    /// Parses the text of a tokenizer.json (`Parse`).
    pub fn parse(text: &str) -> Result<Self, String> {
        Self::parse_bytes(text.as_bytes())
    }

    fn parse_bytes(text: &[u8]) -> Result<Self, String> {
        let root = json::parse(text).map_err(|error| error.to_string())?;
        let model = root.find("model");
        require(model.is_some_and(Value::is_object), "lacks the model")?;
        let model = model.ok_or("tokenizer.json: lacks the model")?;
        let ty = model.find("type");
        require(
            ty.is_some_and(|ty| ty.is_string() && ty.str() == "BPE"),
            "the model is not BPE",
        )?;
        let fallback = model.find("byte_fallback");
        require(
            fallback.is_some_and(|value| value.is_bool() && value.as_bool(false)),
            "byte fallback is off",
        )?;

        // Only the normalizer VoxCPM2 ships: Prepend("▁") then Replace(" ",
        // "▁"), with no pre-tokenizer.
        let pre = root.find("pre_tokenizer");
        require(pre.is_none_or(Value::is_null), "has a pre-tokenizer")?;
        let normalizer = root.find("normalizer");
        require(
            normalizer.is_some_and(Value::is_object),
            "lacks the normalizer",
        )?;
        let steps = normalizer.and_then(|normalizer| normalizer.find("normalizers"));
        require(
            steps.is_some_and(|steps| steps.is_array() && steps.len() == 2),
            "has an unexpected normalizer",
        )?;
        let steps = steps.map(Value::items).unwrap_or_default();
        let (prepend, replace) = (&steps[0], &steps[1]);
        require(
            member_str(prepend, "type") == "Prepend"
                && member_str(prepend, "prepend") == SPACE
                && member_str(replace, "type") == "Replace"
                && replace
                    .find("pattern")
                    .is_some_and(|pattern| member_str(pattern, "String") == " ")
                && member_str(replace, "content") == SPACE,
            "has an unexpected normalizer",
        )?;

        let vocab_value = model.find("vocab");
        require(
            vocab_value.is_some_and(Value::is_object),
            "lacks the vocabulary",
        )?;
        let mut vocab = HashMap::new();
        let mut size = 0usize;
        for (token, id) in vocab_value.map(Value::members).unwrap_or_default() {
            let value = id.as_size(0) as u32;
            vocab.entry(token.as_bytes().to_vec()).or_insert(value);
            size = size.max(value as usize + 1);
        }
        let unk = model.find("unk_token");
        require(unk.is_some_and(Value::is_string), "lacks the unknown token")?;
        let unk_id = unk
            .and_then(|unk| vocab.get(unk.str().as_bytes()).copied())
            .ok_or("tokenizer.json: the unknown token is unknown")?;
        let mut byte_ids = [0u32; 256];
        for (byte, slot) in byte_ids.iter_mut().enumerate() {
            let name = format!("<0x{byte:02X}>");
            *slot = *vocab
                .get(name.as_bytes())
                .ok_or_else(|| format!("tokenizer.json: lacks the byte token {name}"))?;
        }

        let merges_value = model.find("merges");
        require(
            merges_value.is_some_and(Value::is_array),
            "lacks the merges",
        )?;
        let mut merges = HashMap::new();
        for (rank, merge) in merges_value
            .map(Value::items)
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            let (left, right) = if merge.is_string() {
                let pair = merge.str();
                let space = pair
                    .find(' ')
                    .ok_or("tokenizer.json: has a malformed merge")?;
                (pair[..space].to_owned(), pair[space + 1..].to_owned())
            } else {
                require(
                    merge.is_array() && merge.len() == 2,
                    "has a malformed merge",
                )?;
                (
                    merge.items()[0].str().to_owned(),
                    merge.items()[1].str().to_owned(),
                )
            };
            let a = vocab.get(left.as_bytes());
            let b = vocab.get(right.as_bytes());
            let merged = vocab.get(format!("{left}{right}").as_bytes());
            let (Some(&a), Some(&b), Some(&merged)) = (a, b, merged) else {
                return Err("tokenizer.json: merges tokens outside the vocabulary".into());
            };
            merges.entry(pair_key(a, b)).or_insert(Merge {
                rank: rank as u32,
                id: merged,
            });
        }

        let mut added = Vec::new();
        if let Some(tokens) = root.find("added_tokens").filter(|tokens| tokens.is_array()) {
            for token in tokens.items() {
                let flag = |key: &str| {
                    token
                        .find(key)
                        .is_some_and(|value| value.is_bool() && value.as_bool(false))
                };
                require(
                    !flag("single_word")
                        && !flag("lstrip")
                        && !flag("rstrip")
                        && !flag("normalized"),
                    "has an added token with matching options",
                )?;
                let id = token.find("id").map_or(0, |id| id.as_size(0)) as u32;
                added.push(Added {
                    content: member_str(token, "content").as_bytes().to_vec(),
                    id,
                });
                size = size.max(id as usize + 1);
            }
        }
        // Leftmost-longest matching tries longer contents first.
        added.sort_by_key(|token| Reverse(token.content.len()));

        // mask_multichar_chinese_tokens: the reference removes every "▁" from
        // a produced token and splits it into characters when the rest is a
        // vocabulary token of two or more Han characters.
        let mut splits = HashMap::new();
        for (token, &id) in &vocab {
            let mut clean = Vec::with_capacity(token.len());
            let mut offset = 0;
            while offset < token.len() {
                if token[offset..].starts_with(SPACE.as_bytes()) {
                    offset += SPACE.len();
                } else {
                    clean.push(token[offset]);
                    offset += 1;
                }
            }
            let Some(pieces) = characters(&clean) else {
                continue;
            };
            if pieces.len() < 2 || !vocab.contains_key(&clean) {
                continue;
            }
            let mut split = Vec::with_capacity(pieces.len());
            let mut han = true;
            for piece in pieces {
                if !is_han(decode(piece)) {
                    han = false;
                    break;
                }
                split.push(vocab.get(piece).copied().unwrap_or(unk_id));
            }
            if han {
                splits.entry(id).or_insert(split);
            }
        }

        Ok(Self {
            vocab,
            merges,
            added,
            byte_ids,
            unk_id,
            splits,
            size,
        })
    }

    /// The ID of a vocabulary or added token (`TokenId`).
    pub fn token_id(&self, token: &str) -> Option<u32> {
        if let Some(&id) = self.vocab.get(token.as_bytes()) {
            return Some(id);
        }
        self.added
            .iter()
            .find(|added| added.content == token.as_bytes())
            .map(|added| added.id)
    }

    /// One past the largest ID.
    pub fn vocabulary_size(&self) -> usize {
        self.size
    }

    /// The unknown token's ID.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Appends the IDs of `text` to `ids` (`Encode`).
    pub fn encode(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), String> {
        self.encode_bytes(text.as_bytes(), ids)
    }

    /// Appends the IDs of UTF-8 `text` to `ids`; fails on invalid UTF-8
    /// (`Encode`).
    pub fn encode_bytes(&self, text: &[u8], ids: &mut Vec<u32>) -> Result<(), String> {
        if characters(text).is_none() {
            return Err("the text is not valid UTF-8".into());
        }
        let mut start = 0;
        let mut offset = 0;
        while offset < text.len() {
            let found = self
                .added
                .iter()
                .find(|added| text[offset..].starts_with(&added.content));
            let Some(found) = found else {
                offset += sequence_length(text, offset);
                continue;
            };
            self.encode_segment(&text[start..offset], ids);
            ids.push(found.id);
            offset += found.content.len();
            start = offset;
        }
        self.encode_segment(&text[start..], ids);
        Ok(())
    }

    /// The IDs of `text` (`Encode` into a new vector).
    pub fn encode_to_vec(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        self.encode(text, &mut ids)?;
        Ok(ids)
    }

    fn encode_segment(&self, segment: &[u8], ids: &mut Vec<u32>) {
        if segment.is_empty() {
            return;
        }
        let mut normalized = SPACE.as_bytes().to_vec();
        for &c in segment {
            if c == b' ' {
                normalized.extend_from_slice(SPACE.as_bytes());
            } else {
                normalized.push(c);
            }
        }

        // Symbols of the word, linked in order; merged-away ones are dead.
        struct Symbol {
            id: u32,
            prev: isize,
            next: isize,
            alive: bool,
        }
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut add = |id: u32| {
            let index = symbols.len() as isize;
            symbols.push(Symbol {
                id,
                prev: index - 1,
                next: index + 1,
                alive: true,
            });
        };
        // The segment came from valid UTF-8 split at sequence boundaries.
        for piece in characters(&normalized).unwrap_or_default() {
            match self.vocab.get(piece) {
                Some(&id) => add(id),
                None => {
                    for &byte in piece {
                        add(self.byte_ids[usize::from(byte)]);
                    }
                }
            }
        }
        let Some(last) = symbols.last_mut() else {
            return;
        };
        last.next = -1;

        // The tokenizers library's Word::merge_all: lowest rank first, then
        // the leftmost position; stale entries are those whose pair no
        // longer yields the recorded token.
        let mut queue: BinaryHeap<Reverse<(u32, isize, u32)>> = BinaryHeap::new();
        let consider = |symbols: &[Symbol],
                        queue: &mut BinaryHeap<Reverse<(u32, isize, u32)>>,
                        left: isize| {
            if left < 0 || symbols[left as usize].next < 0 {
                return;
            }
            let right = symbols[left as usize].next as usize;
            if let Some(merge) = self
                .merges
                .get(&pair_key(symbols[left as usize].id, symbols[right].id))
            {
                queue.push(Reverse((merge.rank, left, merge.id)));
            }
        };
        for index in 0..symbols.len().saturating_sub(1) {
            consider(&symbols, &mut queue, index as isize);
        }
        while let Some(Reverse((_, position, id))) = queue.pop() {
            let left = position as usize;
            if !symbols[left].alive || symbols[left].next < 0 {
                continue;
            }
            let right = symbols[left].next as usize;
            match self
                .merges
                .get(&pair_key(symbols[left].id, symbols[right].id))
            {
                Some(merge) if merge.id == id => {}
                _ => continue,
            }
            symbols[left].id = id;
            symbols[right].alive = false;
            let after = symbols[right].next;
            symbols[left].next = after;
            if after >= 0 {
                symbols[after as usize].prev = position;
            }
            let before = symbols[left].prev;
            consider(&symbols, &mut queue, before);
            consider(&symbols, &mut queue, position);
        }

        let mut index: isize = 0;
        while index >= 0 {
            let id = symbols[index as usize].id;
            match self.splits.get(&id) {
                Some(split) => ids.extend_from_slice(split),
                None => ids.push(id),
            }
            index = symbols[index as usize].next;
        }
    }
}
