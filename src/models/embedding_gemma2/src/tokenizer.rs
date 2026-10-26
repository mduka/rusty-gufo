//! Gemma SentencePiece-style BPE tokenizer for EmbeddingGemma 2
//! (`tokenizer.cpp`, `tokenizer_loader.cpp`). It reproduces the Hugging Face
//! `tokenizers` output for `<model_dir>/tokenizer.json`. Special tokens in
//! the text match first, longest match wins. In the remaining text, every
//! space becomes U+2581 and the whole string merges as one BPE word. Bytes of
//! characters missing from the vocabulary fall back to `<0xXX>` tokens. When
//! requested, `<bos>` and `<eos>` wrap the result.
//!
//! Only the configuration shipped with the checkpoint is accepted. Any other
//! normalizer, pre-tokenizer, model option, or post-processor is an error.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;

use gufo_core::json::{self, Value};

const MAXIMUM_TOKENIZER_BYTES: u64 = 512 << 20;
const MAXIMUM_TOKEN_ID: u64 = 1 << 24;

/// U+2581 LOWER ONE EIGHTH BLOCK, the replacement for every ASCII space.
const SPACE_SYMBOL: &[u8] = "\u{2581}".as_bytes();

/// FxHash's word mixing for the vocabulary and merge tables: their keys are
/// short byte strings and pairs of IDs, for which SipHash would cost more
/// than the lookup.
#[derive(Default, Clone, Copy)]
struct FastHasher(u64);

impl FastHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        // The multiply leaves the low bits, which pick the bucket, weak;
        // fold the high ones in.
        let mut hash = self.0;
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^ (hash >> 33)
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[derive(Debug, Clone, Copy)]
struct Merge {
    rank: u32,
    new_id: u32,
}

struct AddedToken {
    content: Vec<u8>,
    id: u32,
}

pub struct Tokenizer {
    vocab_size: usize,
    ids: FastMap<Box<[u8]>, u32>,
    merges: FastMap<u64, Merge>,
    /// Longest first.
    added: Vec<AddedToken>,
    added_first_byte: [bool; 256],
    byte_tokens: [u32; 256],
    max_token_bytes: usize,
    prefix_ids: Vec<u32>,
    suffix_ids: Vec<u32>,
}

fn pair_key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

/// UTF-8 sequence length of the character at `text[pos]`, or 1 for invalid
/// bytes so that they fall back to a byte token each.
fn utf8_char_length(text: &[u8], pos: usize) -> usize {
    let lead = text[pos];
    if lead < 0x80 {
        return 1;
    }
    let (length, mut code, minimum) = match lead {
        0xC2..=0xDF => (2, u32::from(lead & 0x1F), 0x80),
        0xE0..=0xEF => (3, u32::from(lead & 0x0F), 0x800),
        0xF0..=0xF4 => (4, u32::from(lead & 0x07), 0x10000),
        _ => return 1,
    };
    if pos + length > text.len() {
        return 1;
    }
    for &next in &text[pos + 1..pos + length] {
        if next & 0xC0 != 0x80 {
            return 1;
        }
        code = (code << 6) | u32::from(next & 0x3F);
    }
    if code < minimum || code > 0x10FFFF || (0xD800..=0xDFFF).contains(&code) {
        return 1;
    }
    length
}

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    prev: i32,
    next: i32,
    alive: bool,
}

/// A candidate merge. The heap pops the smallest rank first, then the
/// leftmost position; entries with equal keys describe the same merge.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PendingMerge {
    rank: u32,
    position: u32,
    new_id: u32,
    // The pair this entry was queued for. Comparing it on pop detects stale
    // entries cheaply.
    left_id: u32,
    right_id: u32,
}

impl Tokenizer {
    /// Reads and validates `<model_dir>/tokenizer.json`; the error describes
    /// the problem.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        Loader::run(model_dir)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Longest vocabulary entry in bytes of normalized text. No token covers
    /// more, so a text needs at least `size / max_token_bytes()` tokens.
    pub fn max_token_bytes(&self) -> usize {
        self.max_token_bytes
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.ids.get(token.as_bytes()).copied()
    }

    fn find_merge(&self, left: u32, right: u32) -> Option<Merge> {
        self.merges.get(&pair_key(left, right)).copied()
    }

    /// Length of the added token starting at `text[pos]`, with its ID.
    fn match_added_token(&self, text: &[u8], pos: usize) -> Option<(usize, u32)> {
        self.added
            .iter()
            .find(|token| text[pos..].starts_with(&token.content))
            .map(|token| (token.content.len(), token.id))
    }

    fn encode_segment(&self, text: &[u8], out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // Normalizer: every ASCII space becomes U+2581. The pre-tokenizer
        // then finds no space to split on, so the segment is a single BPE
        // word.
        let mut word = Vec::with_capacity(text.len() + text.len() / 4);
        for &c in text {
            if c == b' ' {
                word.extend_from_slice(SPACE_SYMBOL);
            } else {
                word.push(c);
            }
        }

        let mut symbols: Vec<Symbol> = Vec::with_capacity(word.len());
        let add_symbol = |symbols: &mut Vec<Symbol>, id: u32| {
            let index = symbols.len() as i32;
            if let Some(last) = symbols.last_mut() {
                last.next = index;
            }
            symbols.push(Symbol {
                id,
                prev: index - 1,
                next: -1,
                alive: true,
            });
        };
        let mut i = 0;
        while i < word.len() {
            let length = utf8_char_length(&word, i);
            let piece = &word[i..i + length];
            i += length;
            if let Some(&id) = self.ids.get(piece) {
                add_symbol(&mut symbols, id);
                continue;
            }
            for &byte in piece {
                add_symbol(&mut symbols, self.byte_tokens[usize::from(byte)]);
            }
        }

        // Linked-list BPE. A heap holds candidate merges ordered by rank,
        // then position. Popping detects stale entries.
        let mut heap = BinaryHeap::with_capacity(symbols.len());
        for (index, pair) in symbols.windows(2).enumerate() {
            if let Some(merge) = self.find_merge(pair[0].id, pair[1].id) {
                heap.push(Reverse(PendingMerge {
                    rank: merge.rank,
                    position: index as u32,
                    new_id: merge.new_id,
                    left_id: pair[0].id,
                    right_id: pair[1].id,
                }));
            }
        }
        while let Some(Reverse(top)) = heap.pop() {
            let position = top.position as usize;
            let left = symbols[position];
            if !left.alive || left.next < 0 {
                continue;
            }
            let right_index = left.next as usize;
            let right = symbols[right_index];
            if left.id != top.left_id || right.id != top.right_id {
                // The neighbours changed. The entry is still usable only if
                // the new pair merges into the same token.
                match self.find_merge(left.id, right.id) {
                    Some(current) if current.new_id == top.new_id => {}
                    _ => continue,
                }
            }
            symbols[position].id = top.new_id;
            symbols[position].next = right.next;
            symbols[right_index].alive = false;
            if right.next >= 0 {
                symbols[right.next as usize].prev = top.position as i32;
            }
            let left = symbols[position];
            if left.prev >= 0 {
                let before = symbols[left.prev as usize];
                if let Some(merge) = self.find_merge(before.id, left.id) {
                    heap.push(Reverse(PendingMerge {
                        rank: merge.rank,
                        position: left.prev as u32,
                        new_id: merge.new_id,
                        left_id: before.id,
                        right_id: left.id,
                    }));
                }
            }
            if left.next >= 0 {
                let after = symbols[left.next as usize];
                if let Some(merge) = self.find_merge(left.id, after.id) {
                    heap.push(Reverse(PendingMerge {
                        rank: merge.rank,
                        position: top.position,
                        new_id: merge.new_id,
                        left_id: left.id,
                        right_id: after.id,
                    }));
                }
            }
        }
        out.extend(symbols.iter().filter(|s| s.alive).map(|s| s.id));
    }

    /// Encodes UTF-8 `text`. `add_special_tokens` wraps the IDs with the
    /// post-processor template (`<bos>` ... `<eos>`). Special tokens that
    /// appear in the text itself are always recognized. Invalid UTF-8 never
    /// fails, and the offending bytes fall back to byte tokens.
    pub fn encode(&self, text: &[u8], add_special_tokens: bool) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 4);
        if add_special_tokens {
            ids.extend_from_slice(&self.prefix_ids);
        }
        // Added (special) tokens match on the raw text, leftmost and longest
        // first. The text between them is encoded as BPE segments.
        let mut segment_start = 0;
        let mut pos = 0;
        while pos < text.len() {
            let matched = if self.added_first_byte[usize::from(text[pos])] {
                self.match_added_token(text, pos)
            } else {
                None
            };
            let Some((length, id)) = matched else {
                pos += 1;
                continue;
            };
            self.encode_segment(&text[segment_start..pos], &mut ids);
            ids.push(id);
            pos += length;
            segment_start = pos;
        }
        self.encode_segment(&text[segment_start..], &mut ids);
        if add_special_tokens {
            ids.extend_from_slice(&self.suffix_ids);
        }
        ids
    }
}

static NULL: Value = Value::Null;

/// The member `key` of `object`, or a null value when it is absent.
fn member<'a>(object: &'a Value, key: &str) -> &'a Value {
    object.find(key).unwrap_or(&NULL)
}

/// A non-negative integer below `limit`.
fn read_id(value: &Value, limit: u64) -> Option<u32> {
    let number = value.as_size(limit as usize) as u64;
    (number < limit).then_some(number as u32)
}

/// The tokenizer under construction (`TokenizerLoader`).
struct Loader {
    tokens: Vec<Vec<u8>>,
    tokenizer: Tokenizer,
}

impl Loader {
    fn run(model_dir: &Path) -> Result<Tokenizer, String> {
        let path = model_dir.join("tokenizer.json");
        let root = {
            let size = std::fs::metadata(&path)
                .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
                .len();
            if size > MAXIMUM_TOKENIZER_BYTES {
                return Err(format!("unexpectedly large {}", path.display()));
            }
            let contents =
                std::fs::read(&path).map_err(|_| format!("cannot read {}", path.display()))?;
            json::parse(&contents)
                .map_err(|e| format!("invalid {}: {}", path.display(), e.message()))?
        };
        if !root.is_object() {
            return Err(format!("invalid {}: expected an object", path.display()));
        }
        if !member(&root, "truncation").is_null() || !member(&root, "padding").is_null() {
            return Err("truncation and padding are not supported".into());
        }
        check_normalization(member(&root, "normalizer"), member(&root, "pre_tokenizer"))?;

        let model = member(&root, "model");
        if !model.is_object() {
            return Err(if model.is_null() {
                "model is missing".into()
            } else {
                "invalid model: expected an object".into()
            });
        }
        let mut loader = Self {
            tokens: Vec::new(),
            tokenizer: Tokenizer {
                vocab_size: 0,
                ids: FastMap::default(),
                merges: FastMap::default(),
                added: Vec::new(),
                added_first_byte: [false; 256],
                byte_tokens: [0; 256],
                max_token_bytes: 0,
                prefix_ids: Vec::new(),
                suffix_ids: Vec::new(),
            },
        };
        loader.load_vocab(member(model, "vocab"))?;
        loader.load_added_tokens(member(&root, "added_tokens"))?;
        let tokens = std::mem::take(&mut loader.tokens);
        let t = &mut loader.tokenizer;
        t.vocab_size = tokens.len();
        t.ids.reserve(tokens.len());
        for (id, token) in tokens.into_iter().enumerate() {
            if token.is_empty() {
                continue;
            }
            t.max_token_bytes = t.max_token_bytes.max(token.len());
            let text = String::from_utf8_lossy(&token).into_owned();
            if t.ids.insert(token.into_boxed_slice(), id as u32).is_some() {
                return Err(format!("duplicate token '{text}' in the vocabulary"));
            }
        }
        check_model_options(model, t)?;
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        for byte in 0..256usize {
            let name = [b'<', b'0', b'x', HEX[byte >> 4], HEX[byte & 0xF], b'>'];
            t.byte_tokens[byte] = *t
                .ids
                .get(&name[..])
                .ok_or("the vocabulary lacks the <0xXX> byte-fallback tokens")?;
        }
        load_merges(member(model, "merges"), t)?;
        load_post_processor(member(&root, "post_processor"), t)?;
        Ok(loader.tokenizer)
    }

    fn load_vocab(&mut self, vocab: &Value) -> Result<(), String> {
        if !vocab.is_object() {
            return Err(if vocab.is_null() {
                "model.vocab is missing".into()
            } else {
                "invalid model.vocab: expected an object".into()
            });
        }
        let mut present: Vec<bool> = Vec::new();
        for (token, value) in vocab.members() {
            let id = read_id(value, MAXIMUM_TOKEN_ID).ok_or_else(|| {
                format!(
                    "invalid model.vocab: id of '{token}' is not an integer in [0, \
                     {MAXIMUM_TOKEN_ID})"
                )
            })? as usize;
            if id >= self.tokens.len() {
                self.tokens.resize(id + 1, Vec::new());
                present.resize(self.tokens.len(), false);
            }
            if present[id] {
                return Err(format!("model.vocab repeats id {id}"));
            }
            present[id] = true;
            self.tokens[id] = token.as_bytes().to_vec();
        }
        if self.tokens.is_empty() {
            return Err("model.vocab is empty".into());
        }
        if present.contains(&false) {
            return Err("model.vocab ids are not contiguous".into());
        }
        Ok(())
    }

    fn load_added_tokens(&mut self, added: &Value) -> Result<(), String> {
        if added.is_null() {
            return Ok(());
        }
        if !added.is_array() {
            return Err("added_tokens must be an array".into());
        }
        let t = &mut self.tokenizer;
        for item in added.items() {
            let content = member(item, "content");
            let id = read_id(member(item, "id"), MAXIMUM_TOKEN_ID);
            let (Some(id), Value::String(content)) = (id, content) else {
                return Err("invalid added_tokens entry".into());
            };
            if content.is_empty() {
                return Err("invalid added_tokens entry".into());
            }
            for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
                if member(item, flag).as_bool(false) {
                    return Err(format!("added token option '{flag}' is not supported"));
                }
            }
            let index = id as usize;
            if index >= self.tokens.len() {
                self.tokens.resize(index + 1, Vec::new());
            }
            if self.tokens[index].is_empty() {
                self.tokens[index] = content.as_bytes().to_vec();
            } else if self.tokens[index] != content.as_bytes() {
                return Err(format!(
                    "added token '{content}' conflicts with vocabulary id {id}"
                ));
            }
            t.added.push(AddedToken {
                content: content.as_bytes().to_vec(),
                id,
            });
        }
        // Stable: tokens of equal length keep their file order.
        t.added.sort_by_key(|token| Reverse(token.content.len()));
        for token in &t.added {
            t.added_first_byte[usize::from(token.content[0])] = true;
        }
        Ok(())
    }
}

fn check_model_options(model: &Value, t: &Tokenizer) -> Result<(), String> {
    if member(model, "type").str() != "BPE" || !member(model, "type").is_string() {
        return Err("model.type must be BPE".into());
    }
    if !member(model, "dropout").is_null() {
        return Err("model.dropout is not supported".into());
    }
    for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
        if let Value::String(value) = member(model, key)
            && !value.is_empty()
        {
            return Err(format!("model.{key} is not supported"));
        }
    }
    if member(model, "ignore_merges").as_bool(false) {
        return Err("model.ignore_merges is not supported".into());
    }
    if !member(model, "byte_fallback").as_bool(false) {
        return Err("model.byte_fallback must be true".into());
    }
    // Every non-vocabulary byte becomes a <0xXX> token, so the unknown token
    // is never emitted. Its name must still be valid.
    if let Value::String(unknown) = member(model, "unk_token")
        && !t.ids.contains_key(unknown.as_bytes())
    {
        return Err("model.unk_token is not in the vocabulary".into());
    }
    Ok(())
}

fn check_normalization(normalizer: &Value, pre_tokenizer: &Value) -> Result<(), String> {
    let string_pattern = |value: &Value| -> String {
        match value
            .find("pattern")
            .filter(|p| p.is_object())
            .and_then(|p| p.find("String"))
        {
            Some(Value::String(text)) => text.clone(),
            _ => "\x01".into(),
        }
    };
    if !normalizer.is_object()
        || normalizer.member_str("type", "") != "Replace"
        || string_pattern(normalizer) != " "
        || normalizer.member_str("content", "").as_bytes() != SPACE_SYMBOL
    {
        return Err("normalizer must replace ' ' with U+2581".into());
    }
    // With every space replaced, a split on ' ' never fires, so each text
    // segment stays one BPE word.
    let inverted = pre_tokenizer
        .find("invert")
        .is_some_and(|invert| invert.as_bool(false));
    if !pre_tokenizer.is_object()
        || pre_tokenizer.member_str("type", "") != "Split"
        || string_pattern(pre_tokenizer) != " "
        || pre_tokenizer.member_str("behavior", "") != "MergedWithPrevious"
        || inverted
    {
        return Err("pre_tokenizer must split on ' ' (MergedWithPrevious)".into());
    }
    Ok(())
}

fn load_merges(merges: &Value, t: &mut Tokenizer) -> Result<(), String> {
    if !merges.is_array() {
        return Err(if merges.is_null() {
            "model.merges is missing".into()
        } else {
            "invalid model.merges: expected an array".into()
        });
    }
    t.merges.reserve(merges.len());
    let mut joined = Vec::new();
    for (rank, pair) in merges.items().iter().enumerate() {
        let (Value::String(left), Value::String(right)) = (
            pair.items().first().unwrap_or(&NULL),
            pair.items().get(1).unwrap_or(&NULL),
        ) else {
            return Err(format!(
                "invalid model.merges: entry {rank} is not a pair of strings"
            ));
        };
        if !pair.is_array() || pair.len() != 2 {
            return Err(format!(
                "invalid model.merges: entry {rank} is not a pair of strings"
            ));
        }
        joined.clear();
        joined.extend_from_slice(left.as_bytes());
        joined.extend_from_slice(right.as_bytes());
        let (Some(&left), Some(&right), Some(&joined)) = (
            t.ids.get(left.as_bytes()),
            t.ids.get(right.as_bytes()),
            t.ids.get(&joined[..]),
        ) else {
            return Err(format!(
                "model.merges entry {rank} references a token missing from the vocabulary"
            ));
        };
        // A repeated pair keeps its last rank, as in Hugging Face.
        t.merges.insert(
            pair_key(left, right),
            Merge {
                rank: rank as u32,
                new_id: joined,
            },
        );
    }
    Ok(())
}

fn load_post_processor(processor: &Value, t: &mut Tokenizer) -> Result<(), String> {
    if processor.is_null() {
        return Ok(());
    }
    let single = processor.find("single").filter(|s| s.is_array());
    let specials = processor.find("special_tokens").filter(|s| s.is_object());
    let (Some(single), Some(specials)) = (single, specials) else {
        return Err("post_processor must be TemplateProcessing".into());
    };
    if processor.member_str("type", "") != "TemplateProcessing" {
        return Err("post_processor must be TemplateProcessing".into());
    }
    let mut seen_sequence = false;
    for item in single.items() {
        if let Some(sequence) = item.find("Sequence") {
            if seen_sequence || sequence.member_str("id", "") != "A" {
                return Err("post_processor single template must hold one sequence A".into());
            }
            seen_sequence = true;
        } else if let Some(special) = item.find("SpecialToken") {
            let ids = specials
                .find(&special.member_str("id", ""))
                .and_then(|entry| entry.find("ids"))
                .filter(|ids| ids.is_array())
                .ok_or("post_processor references an unknown special token")?;
            let target = if seen_sequence {
                &mut t.suffix_ids
            } else {
                &mut t.prefix_ids
            };
            for id in ids.items() {
                let value = read_id(id, t.vocab_size as u64)
                    .ok_or("post_processor special token id out of range")?;
                target.push(value);
            }
        } else {
            return Err("unsupported post_processor template item".into());
        }
    }
    if !seen_sequence {
        return Err("post_processor single template lacks sequence A".into());
    }
    Ok(())
}
