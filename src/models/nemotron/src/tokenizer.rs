//! The artifact's byte-level BPE tokenizer with the `pixtral` split
//! (`tokenizer.cpp`).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;

use gufo_core::gguf::{Gguf, Value};
use unicode_properties::{GeneralCategory, GeneralCategoryGroup, UnicodeGeneralCategory};

pub type TokenId = u32;
pub const INVALID_TOKEN: TokenId = 0xFFFF_FFFF;

// GGUF token types (llama.cpp's llama_token_type).
const TYPE_NORMAL: i64 = 1;
const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;

// Codepoint classes of the split regex.
/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`
const UPPER: u8 = 1 << 0;
/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`
const LOWER: u8 = 1 << 1;
/// `\p{L}`
const LETTER: u8 = 1 << 2;
/// `\p{N}`
const NUMBER: u8 = 1 << 3;
/// `\s` (White_Space)
const SPACE: u8 = 1 << 4;
/// `[\r\n]`
const CRLF: u8 = 1 << 5;

#[derive(Clone, Copy)]
struct Codepoint {
    offset: usize,
    flags: u8,
    value: u32,
}

/// The classes ICU's general category mask and `u_isUWhiteSpace` give a
/// codepoint; surrogates and values past U+10FFFF have none.
fn classify(value: u32) -> u8 {
    let Some(c) = char::from_u32(value) else {
        return 0;
    };
    let category = c.general_category();
    let group = c.general_category_group();
    let mut flags = 0;
    let modifier_or_other = matches!(
        category,
        GeneralCategory::ModifierLetter | GeneralCategory::OtherLetter
    );
    if matches!(
        category,
        GeneralCategory::UppercaseLetter | GeneralCategory::TitlecaseLetter
    ) || modifier_or_other
        || group == GeneralCategoryGroup::Mark
    {
        flags |= UPPER;
    }
    if category == GeneralCategory::LowercaseLetter
        || modifier_or_other
        || group == GeneralCategoryGroup::Mark
    {
        flags |= LOWER;
    }
    if group == GeneralCategoryGroup::Letter {
        flags |= LETTER;
    }
    if group == GeneralCategoryGroup::Number {
        flags |= NUMBER;
    }
    if c.is_whitespace() {
        flags |= SPACE;
    }
    if c == '\r' || c == '\n' {
        flags |= CRLF;
    }
    flags
}

/// Decodes UTF-8; an invalid byte becomes one U+FFFD codepoint of length 1.
fn decode(text: &[u8]) -> Vec<Codepoint> {
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let b0 = text[i];
        let mut length = 1;
        let mut value = u32::from(b0);
        if b0 >= 0x80 {
            length = if b0 & 0xE0 == 0xC0 {
                2
            } else if b0 & 0xF0 == 0xE0 {
                3
            } else if b0 & 0xF8 == 0xF0 {
                4
            } else {
                0
            };
            let mut valid = length != 0 && i + length <= text.len();
            if valid {
                value = u32::from(b0 & (0x7F >> length));
                for k in 1..length {
                    let b = text[i + k];
                    valid = valid && b & 0xC0 == 0x80;
                    value = (value << 6) | u32::from(b & 0x3F);
                }
            }
            if !valid {
                length = 1;
                value = 0xFFFD;
            }
        }
        out.push(Codepoint {
            offset: i,
            flags: classify(value),
            value,
        });
        i += length;
    }
    out
}

/// End index (in codepoints) of the piece starting at codepoint `i`. Each
/// alternative reproduces the backtracking order of a greedy regex engine.
fn piece_end(c: &[Codepoint], i: usize) -> usize {
    let n = c.len();
    let has = |k: usize, flag: u8| k < n && c[k].flags & flag != 0;
    // [^\r\n\p{L}\p{N}]
    let prefix = |k: usize| k < n && c[k].flags & (CRLF | LETTER | NUMBER) == 0;

    // Alternative 1: U*L+. U* first takes its longest run, then gives back
    // codepoints until L+ can start; L+ then runs as far as it can.
    let upper_then_lower = |j: usize| -> Option<usize> {
        let mut k = j;
        while has(k, UPPER) {
            k += 1;
        }
        let mut p = k;
        loop {
            if has(p, LOWER) {
                let mut e = p;
                while has(e, LOWER) {
                    e += 1;
                }
                return Some(e);
            }
            if p == j {
                return None;
            }
            p -= 1;
        }
    };
    if prefix(i)
        && let Some(e) = upper_then_lower(i + 1)
    {
        return e;
    }
    if let Some(e) = upper_then_lower(i) {
        return e;
    }

    // Alternative 2: U+L*; L* cannot fail, so no backtracking.
    let upper_run = |j: usize| -> Option<usize> {
        if !has(j, UPPER) {
            return None;
        }
        let mut k = j;
        while has(k, UPPER) {
            k += 1;
        }
        while has(k, LOWER) {
            k += 1;
        }
        Some(k)
    };
    if prefix(i)
        && let Some(e) = upper_run(i + 1)
    {
        return e;
    }
    if let Some(e) = upper_run(i) {
        return e;
    }

    // Alternative 3: one number codepoint.
    if has(i, NUMBER) {
        return i + 1;
    }

    // Alternative 4: optional space, punctuation run, then [\r\n/]*.
    let punctuation = |j: usize| -> Option<usize> {
        let other = |k: usize| k < n && c[k].flags & (SPACE | LETTER | NUMBER) == 0;
        if !other(j) {
            return None;
        }
        let mut k = j;
        while other(k) {
            k += 1;
        }
        while k < n && matches!(c[k].value, 0x0D | 0x0A | 0x2F) {
            k += 1;
        }
        Some(k)
    };
    if c[i].value == u32::from(b' ')
        && let Some(e) = punctuation(i + 1)
    {
        return e;
    }
    if let Some(e) = punctuation(i) {
        return e;
    }

    if has(i, SPACE) {
        let mut k = i;
        while has(k, SPACE) {
            k += 1;
        }
        // Alternative 5: \s*[\r\n]+ ends after the run's last CR/LF.
        for p in (i..k).rev() {
            if has(p, CRLF) {
                return p + 1;
            }
        }
        // Alternative 6: \s+(?!\S) leaves the last space before non-space to
        // the next piece.
        if k == n {
            return k;
        }
        if k - 1 > i {
            return k - 1;
        }
        // Alternative 7: \s+.
        return k;
    }
    i + 1 // Unreachable: every codepoint falls in some alternative.
}

/// Splits text the way the artifact's `pixtral` pre-tokenizer regex does,
/// evaluated with Unicode general categories as in Hugging Face tokenizers:
///
/// ```text
/// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
/// |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
/// |\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// llama.cpp collapses non-ASCII letters to one class and so keeps a
/// lowercase-to-uppercase change across non-ASCII letters ("aÄb") in one
/// piece that this split separates ("a", "Äb"); on the golden corpus BPE
/// still yields the same token IDs. Returns byte ranges of `text`
/// (`PreTokenize`).
pub fn pre_tokenize(text: &[u8]) -> Vec<Range<usize>> {
    let c = decode(text);
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let e = piece_end(&c, i);
        let begin = c[i].offset;
        let end = if e < c.len() { c[e].offset } else { text.len() };
        pieces.push(begin..end);
        i = e;
    }
    pieces
}

/// GPT-2 byte-to-codepoint map used by byte-level BPE vocabularies.
fn byte_level_codepoints() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut next = 256u32;
    for (b, slot) in table.iter_mut().enumerate() {
        let printable =
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        *slot = if printable {
            b as u32
        } else {
            let value = next;
            next += 1;
            value
        };
    }
    table
}

/// Raw bytes of a byte-level vocabulary string; `None` when a codepoint is
/// outside the byte alphabet.
fn unescape_byte_level(token: &[u8], to_byte: &HashMap<u32, u8>) -> Option<Vec<u8>> {
    decode(token)
        .iter()
        .map(|cp| {
            if cp.value == 0xFFFD {
                None
            } else {
                to_byte.get(&cp.value).copied()
            }
        })
        .collect()
}

fn pair_key(a: TokenId, b: TokenId) -> u64 {
    (u64::from(a) << 32) | u64::from(b)
}

fn int_array(gguf: &Gguf<'_>, key: &str) -> Vec<i64> {
    match gguf.metadata(key) {
        Some(Value::IntArray(values)) => values.clone(),
        Some(Value::UIntArray(values)) => values
            .iter()
            .map(|&v| v.min(i64::MAX as u64) as i64)
            .collect(),
        _ => Vec::new(),
    }
}

#[derive(Clone, Copy)]
struct Merge {
    rank: u32,
    token: TokenId,
}

#[derive(Clone, Copy)]
struct Added {
    id: TokenId,
    control: bool,
}

/// Byte-level BPE tokenizer of the pinned artifact: added tokens are matched
/// before the split, whole pieces found in the vocabulary skip merging
/// (`ignore_merges`), and no BOS is prepended.
pub struct Tokenizer {
    /// Decoded bytes per token id.
    bytes: Vec<Vec<u8>>,
    /// Normal tokens.
    by_bytes: HashMap<Vec<u8>, TokenId>,
    merges: HashMap<u64, Merge>,
    byte_token: [TokenId; 256],
    added: HashMap<Vec<u8>, Added>,
    /// Distinct, longest first.
    added_lengths: Vec<usize>,
    added_first_byte: [bool; 256],
    /// Per id: control added token.
    control: Vec<bool>,
    eos: TokenId,
    end_of_sequence: TokenId,
}

impl Tokenizer {
    /// Loads the vocabulary, merges and token types of the GGUF
    /// (`FromGguf`).
    pub fn from_gguf(gguf: &Gguf<'_>) -> Result<Self, String> {
        if gguf.string("tokenizer.ggml.model") != Some(b"gpt2".as_slice())
            || gguf.string("tokenizer.ggml.pre") != Some(b"pixtral".as_slice())
        {
            return Err("tokenizer is not the byte-level BPE with the pixtral split".into());
        }
        if gguf.bool("tokenizer.ggml.add_bos_token").unwrap_or(false) {
            return Err("tokenizer requests a BOS token; the chat format has none".into());
        }
        let tokens = gguf.strings("tokenizer.ggml.tokens").unwrap_or(&[]);
        let merges = gguf.strings("tokenizer.ggml.merges").unwrap_or(&[]);
        let types = int_array(gguf, "tokenizer.ggml.token_type");
        if tokens.is_empty() || merges.is_empty() || types.len() != tokens.len() {
            return Err("tokenizer vocabulary, merges or token types are missing".into());
        }
        let to_byte: HashMap<u32, u8> = byte_level_codepoints()
            .iter()
            .enumerate()
            .map(|(b, &cp)| (cp, b as u8))
            .collect();

        let mut t = Tokenizer {
            bytes: vec![Vec::new(); tokens.len()],
            by_bytes: HashMap::with_capacity(tokens.len()),
            merges: HashMap::with_capacity(merges.len()),
            byte_token: [0; 256],
            added: HashMap::new(),
            added_lengths: Vec::new(),
            added_first_byte: [false; 256],
            control: vec![false; tokens.len()],
            eos: INVALID_TOKEN,
            end_of_sequence: INVALID_TOKEN,
        };
        for (id, &token) in tokens.iter().enumerate() {
            let token_id = id as TokenId;
            if types[id] == TYPE_CONTROL || types[id] == TYPE_USER_DEFINED {
                t.bytes[id] = token.to_vec();
                t.control[id] = types[id] == TYPE_CONTROL;
                t.added.entry(token.to_vec()).or_insert(Added {
                    id: token_id,
                    control: types[id] == TYPE_CONTROL,
                });
                continue;
            }
            if types[id] != TYPE_NORMAL {
                return Err(format!("unsupported token type at id {id}"));
            }
            let bytes = unescape_byte_level(token, &to_byte)
                .filter(|b| !b.is_empty())
                .ok_or_else(|| format!("token {id} is not byte-level"))?;
            t.bytes[id] = bytes.clone();
            t.by_bytes.entry(bytes).or_insert(token_id);
        }
        for b in 0..=255u8 {
            let id = t
                .by_bytes
                .get([b].as_slice())
                .ok_or_else(|| format!("vocabulary lacks a token for byte {b}"))?;
            t.byte_token[usize::from(b)] = *id;
        }

        for (rank, &merge) in merges.iter().enumerate() {
            let space = merge
                .iter()
                .position(|&b| b == b' ')
                .ok_or_else(|| format!("malformed merge {rank}"))?;
            let left = unescape_byte_level(&merge[..space], &to_byte);
            let right = unescape_byte_level(&merge[space + 1..], &to_byte);
            let (Some(left), Some(right)) = (left, right) else {
                return Err(format!("merge {rank} is not byte-level"));
            };
            let joined = [left.as_slice(), right.as_slice()].concat();
            let (Some(&a), Some(&b), Some(&merged)) = (
                t.by_bytes.get(&left),
                t.by_bytes.get(&right),
                t.by_bytes.get(&joined),
            ) else {
                return Err(format!("merge {rank} names unknown tokens"));
            };
            t.merges.entry(pair_key(a, b)).or_insert(Merge {
                rank: rank as u32,
                token: merged,
            });
        }

        for text in t.added.keys() {
            if !t.added_lengths.contains(&text.len()) {
                t.added_lengths.push(text.len());
            }
            if let Some(&first) = text.first() {
                t.added_first_byte[usize::from(first)] = true;
            }
        }
        t.added_lengths.sort_unstable_by(|a, b| b.cmp(a));

        let eos = gguf
            .u32("tokenizer.ggml.eos_token_id")
            .filter(|&eos| (eos as usize) < tokens.len())
            .ok_or("tokenizer has no valid EOS token")?;
        t.eos = eos;
        t.end_of_sequence = t.added_token("</s>").unwrap_or(INVALID_TOKEN);
        Ok(t)
    }

    fn encode_piece(&self, piece: &[u8], out: &mut Vec<TokenId>) {
        // ignore_merges: a piece that is itself a vocabulary entry is emitted
        // whole.
        if let Some(&whole) = self.by_bytes.get(piece) {
            out.push(whole);
            return;
        }
        // Merge the lowest-ranked adjacent pair repeatedly, leftmost first on
        // a tie, with the symbols in a linked list and candidate pairs in a
        // heap.
        const NONE: usize = usize::MAX;
        let n = piece.len();
        let mut symbol: Vec<TokenId> = piece
            .iter()
            .map(|&b| self.byte_token[usize::from(b)])
            .collect();
        let mut prev: Vec<usize> = (0..n).map(|i| if i == 0 { NONE } else { i - 1 }).collect();
        let mut next: Vec<usize> = (0..n)
            .map(|i| if i + 1 < n { i + 1 } else { NONE })
            .collect();
        // (rank, left) ascending: the smallest rank, then the leftmost.
        let mut heap: BinaryHeap<Reverse<(u32, usize, TokenId, TokenId)>> = BinaryHeap::new();
        let offer = |heap: &mut BinaryHeap<_>, symbol: &[TokenId], next: &[usize], left: usize| {
            let right = next[left];
            if right == NONE {
                return;
            }
            if let Some(m) = self.merges.get(&pair_key(symbol[left], symbol[right])) {
                heap.push(Reverse((m.rank, left, symbol[left], symbol[right])));
            }
        };
        for i in 0..n.saturating_sub(1) {
            offer(&mut heap, &symbol, &next, i);
        }
        while let Some(Reverse((_, left, a, b))) = heap.pop() {
            let right = next[left];
            if symbol[left] != a || right == NONE || symbol[right] != b {
                continue; // A neighbouring merge already consumed this pair.
            }
            let Some(m) = self.merges.get(&pair_key(a, b)) else {
                continue;
            };
            symbol[left] = m.token;
            next[left] = next[right];
            if next[right] != NONE {
                prev[next[right]] = left;
            }
            symbol[right] = INVALID_TOKEN;
            if prev[left] != NONE {
                offer(&mut heap, &symbol, &next, prev[left]);
            }
            offer(&mut heap, &symbol, &next, left);
        }
        let mut i = 0;
        while i != NONE {
            out.push(symbol[i]);
            i = next[i];
        }
    }

    fn encode_text(&self, text: &[u8], out: &mut Vec<TokenId>) {
        for piece in pre_tokenize(text) {
            self.encode_piece(&text[piece], out);
        }
    }

    /// Control tokens (special added tokens such as `<|im_start|>`) are
    /// matched only with `parse_control`; the other added tokens (`<think>`,
    /// `<tool_call>`, ...) are always matched, as in Hugging Face and
    /// llama.cpp (`Encode`).
    pub fn encode(&self, text: impl AsRef<[u8]>, parse_control: bool) -> Vec<TokenId> {
        let text = text.as_ref();
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        // Added tokens split the text first: leftmost match, longest at a
        // position.
        let mut start = 0;
        let mut pos = 0;
        while pos < text.len() {
            if self.added_first_byte[usize::from(text[pos])] {
                for &length in &self.added_lengths {
                    if length > text.len() - pos {
                        continue;
                    }
                    let Some(added) = self.added.get(&text[pos..pos + length]) else {
                        continue;
                    };
                    if added.control && !parse_control {
                        continue;
                    }
                    self.encode_text(&text[start..pos], &mut out);
                    out.push(added.id);
                    start = pos + length;
                    pos = start - 1;
                    break;
                }
            }
            pos += 1;
        }
        self.encode_text(&text[start..], &mut out);
        out
    }

    /// Concatenated bytes of `tokens`; a token may end inside a UTF-8
    /// sequence.
    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        tokens
            .iter()
            .flat_map(|&id| self.token_bytes(id).iter().copied())
            .collect()
    }

    /// Bytes of one token; added tokens decode to their literal text.
    pub fn token_bytes(&self, id: TokenId) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    pub fn added_token(&self, text: &str) -> Option<TokenId> {
        self.added.get(text.as_bytes()).map(|a| a.id)
    }

    pub fn is_control(&self, id: TokenId) -> bool {
        self.control.get(id as usize).copied().unwrap_or(false)
    }

    /// Whether `text` contains the spelling of a control token.
    pub fn spells_control(&self, text: &[u8]) -> bool {
        for pos in 0..text.len() {
            if !self.added_first_byte[usize::from(text[pos])] {
                continue;
            }
            for &length in &self.added_lengths {
                if length > text.len() - pos {
                    continue;
                }
                if self
                    .added
                    .get(&text[pos..pos + length])
                    .is_some_and(|a| a.control)
                {
                    return true;
                }
            }
        }
        false
    }

    /// `<|im_end|>` and `</s>`, the end-of-generation tokens of the
    /// generation config.
    pub fn is_stop(&self, id: TokenId) -> bool {
        id != INVALID_TOKEN && (id == self.eos || id == self.end_of_sequence)
    }

    pub fn eos(&self) -> TokenId {
        self.eos
    }

    pub fn vocab_size(&self) -> usize {
        self.bytes.len()
    }
}
