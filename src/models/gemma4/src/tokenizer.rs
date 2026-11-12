//! The `gemma4` tokenizer (`tokenizer.cpp`): SentencePiece-style BPE over
//! raw UTF-8.
//!
//! Semantics follow llama.cpp's `gemma4` vocabulary (391fac16): special
//! tokens are partitioned out first (longest text first; control tokens only
//! when parsing special text), spaces become U+2581, text splits into newline
//! and non-newline runs, a newline run that is itself a token is kept whole,
//! BPE merges by rank (ties by leftmost position), and symbols outside the
//! vocabulary fall back to `<0xXX>` byte tokens. Invalid UTF-8 bytes become
//! U+FFFD before merging. End-of-generation tokens are treated as control
//! tokens.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::sync::Mutex;

use gufo_core::gguf::{Gguf, Value};

pub type TokenId = i32;

/// GGUF `tokenizer.ggml.token_type` values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum TokenType {
    #[default]
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_value(value: i64) -> Self {
        match value {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Undefined,
        }
    }
}

/// U+2581, the escaped space.
const ESCAPED_SPACE: &[u8] = "\u{2581}".as_bytes();
const REPLACEMENT: u32 = 0xFFFD;

/// End-of-generation token texts of the Gemma 4 vocabulary.
const END_OF_GENERATION: [&str; 3] = ["<eos>", "<turn|>", "<|tool_response>"];

// Pre-split words from this length on are cached (shorter ones encode faster
// than a lookup pays off), within this many bytes of text and ids.
const CACHED_WORD_BYTES: usize = 64;
const WORD_CACHE_BYTES: usize = 64 << 20;

// No valid pair has this key: token ids stay below 2^31.
const EMPTY_PAIR: u64 = u64::MAX;

fn pair_key(left: TokenId, right: TokenId) -> u64 {
    (u64::from(left as u32) << 32) | u64::from(right as u32)
}

fn pair_slot(key: u64, shift: u32) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift) as usize
}

fn append_utf8(cpt: u32, out: &mut Vec<u8>) {
    if cpt <= 0x7F {
        out.push(cpt as u8);
    } else if cpt <= 0x7FF {
        out.push((0xC0 | (cpt >> 6)) as u8);
        out.push((0x80 | (cpt & 0x3F)) as u8);
    } else if cpt <= 0xFFFF {
        out.push((0xE0 | (cpt >> 12)) as u8);
        out.push((0x80 | ((cpt >> 6) & 0x3F)) as u8);
        out.push((0x80 | (cpt & 0x3F)) as u8);
    } else {
        out.push((0xF0 | (cpt >> 18)) as u8);
        out.push((0x80 | ((cpt >> 12) & 0x3F)) as u8);
        out.push((0x80 | ((cpt >> 6) & 0x3F)) as u8);
        out.push((0x80 | (cpt & 0x3F)) as u8);
    }
}

/// Re-encodes `text` through llama.cpp's permissive decoder: structurally
/// valid sequences round-trip, every byte of an invalid one becomes U+FFFD
/// (`Canonicalize`).
fn canonicalize(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    let cont = |i: usize| i < text.len() && (text[i] & 0xC0) == 0x80;
    let mut i = 0;
    while i < text.len() {
        let b = text[i];
        let mut cpt = REPLACEMENT;
        let mut len = 1;
        if b & 0x80 == 0 {
            cpt = u32::from(b);
        } else if b & 0x40 == 0 {
            // Stray continuation byte.
        } else if b & 0x20 == 0 {
            if cont(i + 1) {
                cpt = ((u32::from(b) & 0x1F) << 6) | (u32::from(text[i + 1]) & 0x3F);
                len = 2;
            }
        } else if b & 0x10 == 0 {
            if cont(i + 1) && cont(i + 2) {
                cpt = ((u32::from(b) & 0x0F) << 12)
                    | ((u32::from(text[i + 1]) & 0x3F) << 6)
                    | (u32::from(text[i + 2]) & 0x3F);
                len = 3;
            }
        } else if b & 0x08 == 0 && cont(i + 1) && cont(i + 2) && cont(i + 3) {
            cpt = ((u32::from(b) & 0x07) << 18)
                | ((u32::from(text[i + 1]) & 0x3F) << 12)
                | ((u32::from(text[i + 2]) & 0x3F) << 6)
                | (u32::from(text[i + 3]) & 0x3F);
            len = 4;
        }
        append_utf8(cpt, &mut out);
        i += len;
    }
    out
}

/// Byte length of the UTF-8 character starting with `lead`, as llama.cpp's
/// `unicode_len_utf8` computes it.
fn utf8_length(lead: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[usize::from(lead >> 4)]
}

/// The byte of a `<0xXX>` token text.
fn byte_value(text: &[u8]) -> Option<u8> {
    if text.len() != 6 || &text[..3] != b"<0x" || text[5] != b'>' {
        return None;
    }
    let digit = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'A'..=b'F' => Some(c - b'A' + 10),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    Some((digit(text[3])? << 4) | digit(text[4])?)
}

/// The vocabulary a tokenizer is built from.
#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    pub tokens: Vec<Vec<u8>>,
    pub types: Vec<TokenType>,
    /// "left right", rank = index.
    pub merges: Vec<Vec<u8>>,
    pub bos: TokenId,
    pub eos: TokenId,
    pub pad: TokenId,
    pub unknown: TokenId,
    pub add_bos: bool,
}

#[derive(Debug, Clone, Copy)]
struct MergeSlot {
    key: u64,
    rank: u32,
    merged: TokenId,
}

/// Recently encoded long pre-split words (whole lines), least recently used
/// first: a conversation resends every earlier line with each turn.
#[derive(Default)]
struct WordCache {
    /// Word -> (its ids, its last use).
    entries: HashMap<Vec<u8>, (Vec<TokenId>, u64)>,
    /// Last use -> word.
    order: BTreeMap<u64, Vec<u8>>,
    clock: u64,
    bytes: usize,
}

impl WordCache {
    fn get(&mut self, word: &[u8]) -> Option<Vec<TokenId>> {
        self.clock += 1;
        let clock = self.clock;
        let (ids, used) = self.entries.get_mut(word)?;
        let key = self.order.remove(used);
        *used = clock;
        if let Some(key) = key {
            self.order.insert(clock, key);
        }
        Some(ids.clone())
    }

    fn insert(&mut self, word: &[u8], ids: Vec<TokenId>) {
        if self.entries.contains_key(word) {
            return; // another request encoded it meanwhile
        }
        self.clock += 1;
        self.bytes += word.len() + ids.len() * size_of::<TokenId>();
        self.order.insert(self.clock, word.to_vec());
        self.entries.insert(word.to_vec(), (ids, self.clock));
        while self.bytes > WORD_CACHE_BYTES {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            if let Some((ids, _)) = self.entries.remove(&oldest) {
                self.bytes -= oldest.len() + ids.len() * size_of::<TokenId>();
            }
        }
    }
}

/// A queued merge of two adjacent symbols (`Bigram`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Bigram {
    rank: i64,
    left: i32,
    right: i32,
    left_id: TokenId,
    right_id: TokenId,
    merged: TokenId,
}

impl Ord for Bigram {
    /// Lowest rank first, then the leftmost pair (llama.cpp's comparator), as
    /// the greatest element of a max-heap.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy)]
struct Symbol {
    offset: usize,
    /// Zero once merged into the left neighbour.
    length: usize,
    /// -1 when the text is outside the vocabulary.
    id: TokenId,
    prev: i32,
    next: i32,
}

/// The `gemma4` tokenizer (`Tokenizer`).
pub struct Tokenizer {
    tokens: Vec<Vec<u8>>,
    types: Vec<TokenType>,
    token_to_id: HashMap<Vec<u8>, TokenId>,
    /// Open-addressing merge table keyed by the (left, right) id pair; the
    /// lowest rank wins for a repeated pair.
    merges: Vec<MergeSlot>,
    merge_shift: u32,
    /// Token of each single byte that is itself a vocabulary entry, else -1.
    char_tokens: [TokenId; 256],
    word_cache: Mutex<WordCache>,
    /// Partition order: longest text first.
    special: Vec<TokenId>,
    eog: Vec<TokenId>,
    byte_tokens: [TokenId; 256],
    bos: TokenId,
    eos: TokenId,
    add_bos: bool,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.tokens.len())
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Reads the vocabulary of a `gemma4` GGUF (`CreateFromGguf`).
    pub fn from_gguf(gguf: &Gguf<'_>) -> Result<Self, String> {
        if gguf.string("tokenizer.ggml.model") != Some(b"gemma4".as_slice()) {
            return Err("tokenizer.ggml.model is not gemma4".into());
        }
        let strings = |key: &str| -> Vec<Vec<u8>> {
            gguf.strings(key)
                .map(|items| items.iter().map(|s| s.to_vec()).collect())
                .unwrap_or_default()
        };
        let types = match gguf.metadata("tokenizer.ggml.token_type") {
            None => return Err("missing tokenizer.ggml.token_type".into()),
            Some(Value::IntArray(values)) => {
                values.iter().map(|&v| TokenType::from_value(v)).collect()
            }
            Some(Value::UIntArray(values)) => values
                .iter()
                .map(|&v| TokenType::from_value(v.min(7) as i64))
                .collect(),
            Some(_) => return Err("tokenizer.ggml.token_type must be an integer array".into()),
        };
        let id = |key: &str| gguf.u32(key).map_or(-1, |v| v as TokenId);
        Self::new(Vocabulary {
            tokens: strings("tokenizer.ggml.tokens"),
            types,
            merges: strings("tokenizer.ggml.merges"),
            bos: id("tokenizer.ggml.bos_token_id"),
            eos: id("tokenizer.ggml.eos_token_id"),
            pad: id("tokenizer.ggml.padding_token_id"),
            unknown: id("tokenizer.ggml.unknown_token_id"),
            add_bos: gguf.bool("tokenizer.ggml.add_bos_token").unwrap_or(false),
        })
    }

    /// Builds a tokenizer from a vocabulary (`Create`).
    pub fn new(v: Vocabulary) -> Result<Self, String> {
        if v.tokens.is_empty() || v.tokens.len() > 0x7fff_ffff || v.types.len() != v.tokens.len() {
            return Err("tokenizer vocabulary and token types must match in size".into());
        }
        let n = v.tokens.len() as TokenId;
        let in_range = |t: TokenId| t >= 0 && t < n;
        if !in_range(v.bos) || !in_range(v.eos) {
            return Err("tokenizer BOS/EOS token ids are out of range".into());
        }
        let mut token_to_id = HashMap::with_capacity(v.tokens.len());
        for (i, token) in v.tokens.iter().enumerate() {
            // The first occurrence wins, as in llama.cpp's text_to_token map.
            token_to_id.entry(token.clone()).or_insert(i as TokenId);
        }
        let mut t = Tokenizer {
            tokens: v.tokens,
            types: v.types,
            token_to_id,
            merges: Vec::new(),
            merge_shift: 64,
            char_tokens: [-1; 256],
            word_cache: Mutex::new(WordCache::default()),
            special: Vec::new(),
            eog: Vec::new(),
            byte_tokens: [-1; 256],
            bos: v.bos,
            eos: v.eos,
            add_bos: v.add_bos,
        };

        // End-of-generation tokens behave as control tokens even when the
        // file types them otherwise.
        let add_eog = |t: &mut Tokenizer, token: TokenId| {
            if in_range(token) && !t.eog.contains(&token) {
                t.eog.push(token);
                t.types[token as usize] = TokenType::Control;
            }
        };
        add_eog(&mut t, v.eos);
        for text in END_OF_GENERATION {
            if let Some(token) = t.find_token(text.as_bytes()) {
                add_eog(&mut t, token);
            }
        }
        t.eog.sort_unstable();

        for i in 0..n {
            let ty = t.types[i as usize];
            if ty == TokenType::Byte {
                let value = byte_value(&t.tokens[i as usize])
                    .ok_or_else(|| format!("byte token {i} is malformed"))?;
                t.byte_tokens[usize::from(value)] = i;
            } else if matches!(
                ty,
                TokenType::Control | TokenType::UserDefined | TokenType::Unknown
            ) && !t.tokens[i as usize].is_empty()
            {
                t.special.push(i);
            }
        }
        // Longest text first; equal lengths keep id order for determinism.
        let tokens = &t.tokens;
        t.special
            .sort_by(|&a, &b| tokens[b as usize].len().cmp(&tokens[a as usize].len()));

        // A power-of-two table at most half full.
        let mut bits = 4u32;
        while (1usize << bits) < v.merges.len() * 2 {
            bits += 1;
        }
        t.merge_shift = 64 - bits;
        t.merges = vec![
            MergeSlot {
                key: EMPTY_PAIR,
                rank: 0,
                merged: -1,
            };
            1usize << bits
        ];
        let mask = t.merges.len() - 1;
        for (rank, merge) in v.merges.iter().enumerate() {
            let split = merge
                .get(1..)
                .and_then(|rest| rest.iter().position(|&c| c == b' '))
                .map(|p| p + 1)
                .ok_or_else(|| format!("tokenizer merge {rank} is malformed"))?;
            let left = t.find_token(&merge[..split]);
            let right = t.find_token(&merge[split + 1..]);
            let mut joined = merge[..split].to_vec();
            joined.extend_from_slice(&merge[split + 1..]);
            let merged = t.find_token(&joined);
            let (Some(left), Some(right), Some(merged)) = (left, right, merged) else {
                // Symbols are tracked by id, so every merge must stay in the
                // vocabulary.
                return Err(format!(
                    "tokenizer merge {rank} references text outside the vocabulary"
                ));
            };
            let key = pair_key(left, right);
            let mut slot = pair_slot(key, t.merge_shift);
            loop {
                let entry = &mut t.merges[slot];
                if entry.key == key {
                    break; // an earlier rank already holds this pair
                }
                if entry.key == EMPTY_PAIR {
                    *entry = MergeSlot {
                        key,
                        rank: rank as u32,
                        merged,
                    };
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
        for byte in 0..=255u8 {
            t.char_tokens[usize::from(byte)] = t.find_token(&[byte]).unwrap_or(-1);
        }
        Ok(t)
    }

    pub fn find_token(&self, text: &[u8]) -> Option<TokenId> {
        self.token_to_id.get(text).copied()
    }

    pub fn token_type(&self, token: TokenId) -> TokenType {
        usize::try_from(token)
            .ok()
            .and_then(|i| self.types.get(i).copied())
            .unwrap_or(TokenType::Undefined)
    }

    /// A token's vocabulary text as stored (`▁` for a space); empty outside
    /// the vocabulary.
    pub fn piece(&self, token: TokenId) -> &[u8] {
        usize::try_from(token)
            .ok()
            .and_then(|i| self.tokens.get(i))
            .map_or(&[], Vec::as_slice)
    }

    /// Per token, the rank of the earliest merge that creates it, or
    /// `u32::MAX` for tokens no merge creates (single symbols, special and
    /// byte tokens).
    pub fn creation_ranks(&self) -> Vec<u32> {
        let mut ranks = vec![u32::MAX; self.tokens.len()];
        for entry in &self.merges {
            if entry.key != EMPTY_PAIR {
                let rank = &mut ranks[entry.merged as usize];
                *rank = (*rank).min(entry.rank);
            }
        }
        ranks
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn bos_token(&self) -> TokenId {
        self.bos
    }

    pub fn eos_token(&self) -> TokenId {
        self.eos
    }

    pub fn is_end_of_generation(&self, token: TokenId) -> bool {
        self.eog.binary_search(&token).is_ok()
    }

    /// Sorted.
    pub fn end_of_generation_tokens(&self) -> &[TokenId] {
        &self.eog
    }

    fn find_merge(&self, left: TokenId, right: TokenId) -> Option<(i64, TokenId)> {
        if left < 0 || right < 0 {
            return None;
        }
        let key = pair_key(left, right);
        let mask = self.merges.len() - 1;
        let mut slot = pair_slot(key, self.merge_shift);
        loop {
            let entry = &self.merges[slot];
            if entry.key == key {
                return Some((i64::from(entry.rank), entry.merged));
            }
            if entry.key == EMPTY_PAIR {
                return None;
            }
            slot = (slot + 1) & mask;
        }
    }

    /// `parse_special` also matches control tokens in the text; user-defined
    /// tokens are always matched. `add_bos` follows the vocabulary flag only
    /// when requested: chat templates emit `<bos>` themselves.
    pub fn encode(&self, text: &[u8], add_bos: bool, parse_special: bool) -> Vec<TokenId> {
        // Fragments are either raw text or an already matched special token.
        enum Fragment<'t> {
            Text(&'t [u8]),
            Token(TokenId),
        }
        let mut fragments = Vec::new();
        if !text.is_empty() {
            fragments.push(Fragment::Text(text));
        }
        for &special in &self.special {
            let ty = self.types[special as usize];
            if !parse_special && matches!(ty, TokenType::Control | TokenType::Unknown) {
                continue;
            }
            let needle = self.tokens[special as usize].as_slice();
            let mut next = Vec::with_capacity(fragments.len());
            for fragment in fragments {
                let Fragment::Text(text) = fragment else {
                    next.push(fragment);
                    continue;
                };
                let mut start = 0;
                while let Some(found) = find(&text[start..], needle) {
                    let at = start + found;
                    if at > start {
                        next.push(Fragment::Text(&text[start..at]));
                    }
                    next.push(Fragment::Token(special));
                    start = at + needle.len();
                }
                if start < text.len() {
                    next.push(Fragment::Text(&text[start..]));
                }
            }
            fragments = next;
        }

        let mut out = Vec::new();
        if add_bos && self.add_bos {
            out.push(self.bos);
        }
        for fragment in fragments {
            match fragment {
                Fragment::Token(token) => out.push(token),
                Fragment::Text(text) => self.encode_text(text, &mut out),
            }
        }
        out
    }

    fn encode_text(&self, raw: &[u8], out: &mut Vec<TokenId>) {
        let mut escaped = Vec::with_capacity(raw.len());
        for &c in raw {
            if c == b' ' {
                escaped.extend_from_slice(ESCAPED_SPACE);
            } else {
                escaped.push(c);
            }
        }
        let text = canonicalize(&escaped);
        // Pre-split into maximal runs of newlines and of everything else.
        let mut start = 0;
        while start < text.len() {
            let newline = text[start] == b'\n';
            let mut end = start;
            while end < text.len() && (text[end] == b'\n') == newline {
                end += 1;
            }
            let word = &text[start..end];
            start = end;
            if word.len() < CACHED_WORD_BYTES {
                self.encode_word(word, out);
                continue;
            }
            if let Some(ids) = self
                .word_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(word)
            {
                out.extend_from_slice(&ids);
                continue;
            }
            let mut ids = Vec::new();
            self.encode_word(word, &mut ids);
            out.extend_from_slice(&ids);
            if word.len() + ids.len() * size_of::<TokenId>() > WORD_CACHE_BYTES {
                continue;
            }
            self.word_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(word, ids);
        }
    }

    fn encode_word(&self, word: &[u8], out: &mut Vec<TokenId>) {
        if word[0] == b'\n' {
            // A newline run that is a token stays whole instead of being
            // merged.
            if let Some(token) = self.find_token(word) {
                out.push(token);
                return;
            }
        }

        let mut symbols: Vec<Symbol> = Vec::new();
        let mut offset = 0;
        while offset < word.len() {
            let length = (word.len() - offset).min(utf8_length(word[offset]));
            let id = if length == 1 {
                self.char_tokens[usize::from(word[offset])]
            } else {
                self.find_token(&word[offset..offset + length])
                    .unwrap_or(-1)
            };
            let index = symbols.len() as i32;
            symbols.push(Symbol {
                offset,
                length,
                id,
                prev: index - 1,
                next: if offset + length == word.len() {
                    -1
                } else {
                    index + 1
                },
            });
            offset += length;
        }

        let mut queue = BinaryHeap::new();
        let add = |queue: &mut BinaryHeap<Bigram>, symbols: &[Symbol], left: i32, right: i32| {
            if left < 0 || right < 0 {
                return;
            }
            let (l, r) = (symbols[left as usize].id, symbols[right as usize].id);
            if let Some((rank, merged)) = self.find_merge(l, r) {
                queue.push(Bigram {
                    rank,
                    left,
                    right,
                    left_id: l,
                    right_id: r,
                    merged,
                });
            }
        };
        for i in 1..symbols.len() as i32 {
            add(&mut queue, &symbols, i - 1, i);
        }
        while let Some(b) = queue.pop() {
            let (left, right) = (symbols[b.left as usize], symbols[b.right as usize]);
            // Stale when either side has changed since the pair was queued.
            if left.length == 0
                || right.length == 0
                || left.id != b.left_id
                || right.id != b.right_id
                || left.next != b.right
            {
                continue;
            }
            {
                let l = &mut symbols[b.left as usize];
                l.length += right.length;
                l.id = b.merged;
                l.next = right.next;
            }
            symbols[b.right as usize].length = 0;
            if right.next >= 0 {
                symbols[right.next as usize].prev = b.left;
            }
            let (prev, next) = (left.prev, right.next);
            add(&mut queue, &symbols, prev, b.left);
            add(&mut queue, &symbols, b.left, next);
        }

        let mut i = 0i32;
        while i >= 0 && (i as usize) < symbols.len() {
            let s = symbols[i as usize];
            if s.id >= 0 {
                out.push(s.id);
            } else {
                for &byte in &word[s.offset..s.offset + s.length] {
                    let token = self.byte_tokens[usize::from(byte)];
                    if token >= 0 {
                        out.push(token);
                    }
                }
            }
            i = s.next;
        }
    }

    /// Text of one token. Control and unknown tokens render only with
    /// `special`; user-defined tokens always render; byte tokens yield one raw
    /// byte, so a multi-byte character may span several calls.
    pub fn token_text(&self, token: TokenId, special: bool) -> Vec<u8> {
        match self.token_type(token) {
            TokenType::Control | TokenType::Unknown => {
                if special {
                    self.tokens[token as usize].clone()
                } else {
                    Vec::new()
                }
            }
            TokenType::UserDefined => self.tokens[token as usize].clone(),
            TokenType::Byte => byte_value(&self.tokens[token as usize])
                .map(|b| vec![b])
                .unwrap_or_default(),
            TokenType::Normal => {
                let text = &self.tokens[token as usize];
                let mut out = Vec::with_capacity(text.len());
                let mut i = 0;
                while i < text.len() {
                    if text[i..].starts_with(ESCAPED_SPACE) {
                        out.push(b' ');
                        i += ESCAPED_SPACE.len();
                    } else {
                        out.push(text[i]);
                        i += 1;
                    }
                }
                out
            }
            TokenType::Undefined | TokenType::Unused => Vec::new(),
        }
    }

    pub fn decode(&self, tokens: &[TokenId], special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &token in tokens {
            out.extend_from_slice(&self.token_text(token, special));
        }
        out
    }
}

/// The first occurrence of `needle` in `haystack` (`std::string_view::find`).
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let (&first, rest) = needle.split_first()?;
    let last = haystack.len().checked_sub(needle.len())?;
    let mut at = 0;
    while at <= last {
        at += haystack[at..=last].iter().position(|&b| b == first)?;
        if &haystack[at + 1..at + needle.len()] == rest {
            return Some(at);
        }
        at += 1;
    }
    None
}
