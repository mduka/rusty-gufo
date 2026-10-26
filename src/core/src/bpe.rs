//! Byte-level BPE over Qwen vocabularies, a port of the C++ `BpeTokenizer`
//! (Qwen3-ASR) and `QwenTokenizer` (Qwen3.8 text models). The generated token
//! IDs are part of each model's numerical contract, so the algorithm is
//! reproduced as is, including its quirks: later merge lines overriding
//! earlier ranks of the same pair, and the lenient code point decoding of
//! byte-escaped tokens. Without a pre-tokenizer (Qwen3-ASR) the text between
//! added tokens is merged as one chunk; the `qwen35` pre-tokenizer normalises
//! it to NFC and splits it like the tokenizer's Unicode regex first (`qwen2`
//! likewise, with the older regex).

use unicode_normalization::UnicodeNormalization;

use crate::unicode::{folds_to, is_letter, is_letter_or_mark, is_newline, is_number, is_space};

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

pub type TokenId = u32;

/// rustc's FxHash. The vocabulary maps hash short byte strings and token
/// pairs from trusted model files; SipHash dominated building them.
#[derive(Default, Clone, Copy)]
struct FxHasher(u64);

impl FxHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.add(u64::from_le_bytes(*word));
        }
        if !rest.is_empty() {
            let mut buffer = [0; 8];
            buffer[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buffer));
        }
    }

    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type FxBuild = BuildHasherDefault<FxHasher>;

/// How text between added tokens is split before merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizer {
    /// Merge the whole span (Qwen3-ASR).
    None,
    /// NFC, then the pieces of Qwen's split regex (GGUF `tokenizer.ggml.pre`
    /// `qwen35`).
    Qwen35,
    /// NFC, then the pieces of the Qwen2 split regex of `tokenizer.json`,
    /// whose letter runs exclude combining marks (Qwen-Image-2.1, where the
    /// C++ tokenizer runs the regex through ICU).
    Qwen2,
}

/// GPT-2's printable stand-in for every byte, as UTF-8.
fn byte_to_gpt2(byte: u8) -> String {
    let printable = |b: u32| {
        (u32::from(b'!')..=u32::from(b'~')).contains(&b)
            || (161..=172).contains(&b)
            || (174..=255).contains(&b)
    };
    let b = u32::from(byte);
    let code_point = if printable(b) {
        b
    } else {
        256 + (0..b).filter(|&earlier| !printable(earlier)).count() as u32
    };
    char::from_u32(code_point)
        .map(String::from)
        .unwrap_or_default()
}

/// The inverse map: GPT-2 code point → byte.
fn gpt2_to_byte() -> &'static HashMap<u32, u8> {
    static MAP: std::sync::OnceLock<HashMap<u32, u8>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        (0..=255u8)
            .filter_map(|b| Some((u32::from(byte_to_gpt2(b).chars().next()?), b)))
            .collect()
    })
}

/// Maps GPT-2 escaped code points back to bytes; others pass through. Code
/// points are decoded leniently, like the C++ code: a truncated multi-byte
/// sequence is read as its lead byte alone.
fn unescape_gpt2(text: &[u8]) -> Vec<u8> {
    let map = gpt2_to_byte();
    let mut result = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let b0 = u32::from(text[i]);
        let continuation = |k: usize| u32::from(text[i + k]) & 0x3F;
        let (code_point, len) = if b0 & 0xE0 == 0xC0 && i + 1 < text.len() {
            (((b0 & 0x1F) << 6) | continuation(1), 2)
        } else if b0 & 0xF0 == 0xE0 && i + 2 < text.len() {
            (
                ((b0 & 0x0F) << 12) | (continuation(1) << 6) | continuation(2),
                3,
            )
        } else if b0 & 0xF8 == 0xF0 && i + 3 < text.len() {
            (
                ((b0 & 0x07) << 18)
                    | (continuation(1) << 12)
                    | (continuation(2) << 6)
                    | continuation(3),
                4,
            )
        } else {
            (b0, 1)
        };
        match map.get(&code_point) {
            Some(&byte) => result.push(byte),
            None => result.extend_from_slice(&text[i..i + len]),
        }
        i += len;
    }
    result
}

/// `<0xHH>` byte tokens.
fn hex_byte_token(token: &[u8]) -> Option<u8> {
    if token.len() != 6 || !token.starts_with(b"<0x") || token[5] != b'>' {
        return None;
    }
    let digit = |c: u8| char::from(c).to_digit(16);
    Some((digit(token[3])? * 16 + digit(token[4])?) as u8)
}

#[derive(Clone)]
pub struct BpeTokenizer {
    id_to_token: Vec<Vec<u8>>,
    /// The rank of each mergeable pair and the token it merges into (`None`
    /// when the vocabulary lacks the merged text, which stops merging).
    merge_ranks: HashMap<(TokenId, TokenId), (u32, Option<TokenId>), FxBuild>,
    special_to_id: Vec<(Vec<u8>, TokenId)>,
    special_ids: HashSet<TokenId, FxBuild>,
    byte_tokens: [Option<TokenId>; 256],
    pre_tokenizer: PreTokenizer,
}

impl BpeTokenizer {
    /// `tokens[id]` is the (GPT-2 escaped) token text; `merges` are
    /// `"left right"` lines in rank order; added tokens are matched whole
    /// before merging.
    pub fn new<T: AsRef<[u8]>, M: AsRef<[u8]>>(
        tokens: &[T],
        merges: &[M],
        added: &[(Vec<u8>, TokenId)],
    ) -> Result<Self, String> {
        if tokens.is_empty() {
            return Err("Vocabulary cannot be empty".into());
        }
        let mut token_to_id: HashMap<&[u8], TokenId, FxBuild> =
            HashMap::with_capacity_and_hasher(tokens.len(), FxBuild::default());
        for (id, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.as_ref(), id as TokenId);
        }
        let mut id_to_token: Vec<&[u8]> = tokens.iter().map(AsRef::as_ref).collect();
        // Added tokens that already name their vocabulary entry leave both
        // maps as they are; merged tokens can then be looked up in the same
        // pass as the pairs.
        let unchanged = added.iter().all(|(name, id)| {
            id_to_token
                .get(*id as usize)
                .is_none_or(|text| *text == name.as_slice() && token_to_id.get(text) == Some(id))
        });
        let mut merge_ranks = HashMap::with_capacity_and_hasher(merges.len(), FxBuild::default());
        let mut text = Vec::new();
        for (rank, merge) in merges.iter().enumerate() {
            let merge = merge.as_ref();
            let Some(space) = merge.iter().position(|&c| c == b' ') else {
                continue;
            };
            let (left, right) = (&merge[..space], &merge[space + 1..]);
            if let (Some(&a), Some(&b)) = (token_to_id.get(left), token_to_id.get(right)) {
                let merged = if unchanged {
                    text.clear();
                    text.extend_from_slice(left);
                    text.extend_from_slice(right);
                    token_to_id.get(text.as_slice()).copied()
                } else {
                    None
                };
                merge_ranks.insert((a, b), (rank as u32, merged));
            }
        }
        let mut special_ids = HashSet::default();
        for (name, id) in added {
            special_ids.insert(*id);
            if let Some(slot) = id_to_token.get_mut(*id as usize) {
                *slot = name;
                token_to_id.insert(name, *id);
            }
        }
        if !unchanged {
            // The token each pair merges into, with the final vocabulary (as
            // the merge loop used to look it up at every merge).
            for (&(a, b), (_, merged)) in merge_ranks.iter_mut() {
                text.clear();
                text.extend_from_slice(id_to_token[a as usize]);
                text.extend_from_slice(id_to_token[b as usize]);
                *merged = token_to_id.get(text.as_slice()).copied();
            }
        }
        let mut byte_tokens = [None; 256];
        for (byte, slot) in byte_tokens.iter_mut().enumerate() {
            let byte = byte as u8;
            *slot = token_to_id
                .get(byte_to_gpt2(byte).as_bytes())
                .or_else(|| token_to_id.get(&[byte][..]))
                .or_else(|| token_to_id.get(format!("<0x{byte:02X}>").as_bytes()))
                .copied();
        }
        Ok(Self {
            id_to_token: id_to_token.into_iter().map(<[u8]>::to_vec).collect(),
            merge_ranks,
            special_to_id: added.to_vec(),
            special_ids,
            byte_tokens,
            pre_tokenizer: PreTokenizer::None,
        })
    }

    pub fn with_pre_tokenizer(mut self, pre_tokenizer: PreTokenizer) -> Self {
        self.pre_tokenizer = pre_tokenizer;
        self
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Merges the lowest-ranked adjacent pair repeatedly, the leftmost on a
    /// tie. Pre-tokenizer pieces are unbounded (a run of letters or spaces is
    /// one piece), so the symbols form a linked list and the candidate pairs
    /// a heap, instead of a rescan per merge; stale candidates are skipped
    /// when popped (`BpeMergeChunk`).
    fn merge_chunk(&self, chunk: &[u8], out: &mut Vec<TokenId>) {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        const NONE: usize = usize::MAX;
        let mut word: Vec<TokenId> = chunk
            .iter()
            .filter_map(|&b| self.byte_tokens[usize::from(b)])
            .collect();
        let count = word.len();
        if count < 2 {
            out.extend_from_slice(&word);
            return;
        }
        let mut previous: Vec<usize> = (0..count).map(|i| i.wrapping_sub(1)).collect();
        let mut following: Vec<usize> = (1..=count).collect();
        following[count - 1] = NONE;
        // (rank, left index, left token, right token), smallest first.
        let mut candidates = BinaryHeap::new();
        let offer =
            |candidates: &mut BinaryHeap<_>, word: &[TokenId], left: usize, right: usize| {
                if let Some(&(rank, _)) = self.merge_ranks.get(&(word[left], word[right])) {
                    candidates.push(Reverse((rank, left, word[left], word[right])));
                }
            };
        for i in 0..count - 1 {
            offer(&mut candidates, &word, i, i + 1);
        }
        while let Some(Reverse((_, left, a, b))) = candidates.pop() {
            let right = following[left];
            if word[left] != a || right == NONE || word[right] != b {
                continue; // a neighbouring merge replaced this pair
            }
            let Some(&(_, Some(merged))) = self.merge_ranks.get(&(a, b)) else {
                break; // a ranked pair without a vocabulary entry stops merging
            };
            word[left] = merged;
            let after = following[right];
            following[left] = after;
            if after != NONE {
                previous[after] = left;
            }
            word[right] = TokenId::MAX;
            if previous[left] != NONE {
                offer(&mut candidates, &word, previous[left], left);
            }
            if after != NONE {
                offer(&mut candidates, &word, left, after);
            }
        }
        let mut i = 0;
        while i != NONE {
            out.push(word[i]);
            i = following[i];
        }
    }

    fn encode_text(&self, text: &str, out: &mut Vec<TokenId>) {
        if self.pre_tokenizer == PreTokenizer::None {
            self.merge_chunk(text.as_bytes(), out);
            return;
        }
        let normalized: String;
        let text = if text.is_ascii() {
            text
        } else {
            normalized = text.nfc().collect();
            &normalized
        };
        let marks = self.pre_tokenizer == PreTokenizer::Qwen35;
        let mut offset = 0;
        while offset < text.len() {
            let end = qwen_piece_end(text, offset, marks);
            self.merge_chunk(&text.as_bytes()[offset..end], out);
            offset = end;
        }
    }

    /// Encodes text; added tokens are recognised anywhere in it. Each added
    /// token keeps its next occurrence and searches again only once the
    /// cursor passed it, so the text is walked once overall instead of once
    /// per turn of a conversation.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut occurrence: Vec<Option<usize>> = self
            .special_to_id
            .iter()
            .map(|(name, _)| find(bytes, name))
            .collect();
        let mut pos = 0;
        while pos < bytes.len() {
            // The earliest added token, the longest one on a tie.
            let mut next: Option<(usize, usize, TokenId)> = None;
            for ((name, id), found) in self.special_to_id.iter().zip(occurrence.iter_mut()) {
                if found.is_some_and(|at| at < pos) {
                    *found = find(&bytes[pos..], name).map(|at| pos + at);
                }
                let Some(at) = *found else { continue };
                if next.is_none_or(|(best, len, _)| at < best || (at == best && name.len() > len)) {
                    next = Some((at, name.len(), *id));
                }
            }
            match next {
                None => {
                    self.encode_text(&text[pos..], &mut tokens);
                    break;
                }
                Some((at, len, id)) => {
                    // Matches of UTF-8 needles fall on character boundaries.
                    if at > pos {
                        self.encode_text(&text[pos..at], &mut tokens);
                    }
                    tokens.push(id);
                    pos = at + len;
                }
            }
        }
        tokens
    }

    /// Encodes text without recognising added tokens: a spelled control
    /// token stays its characters.
    pub fn encode_plain(&self, text: &str) -> Vec<TokenId> {
        let mut tokens = Vec::new();
        self.encode_text(text, &mut tokens);
        tokens
    }

    /// The bytes of one token: added tokens verbatim, `<0xHH>` as its byte,
    /// everything else with the GPT-2 byte escaping undone.
    pub fn decode_token(&self, id: TokenId) -> Vec<u8> {
        let Some(token) = self.id_to_token.get(id as usize) else {
            return Vec::new();
        };
        if self.special_ids.contains(&id) {
            return token.clone();
        }
        if let Some(byte) = hex_byte_token(token) {
            return vec![byte];
        }
        unescape_gpt2(token)
    }
}

fn char_at(text: &str, offset: usize) -> char {
    text[offset..].chars().next().unwrap_or('\0')
}

/// The letter class of the split regex: `[\p{L}\p{M}]` for `qwen35`,
/// `\p{L}` for `qwen2`.
fn is_word(c: char, marks: bool) -> bool {
    if marks {
        is_letter_or_mark(c)
    } else {
        is_letter(c)
    }
}

/// End of the run of word characters starting at `offset`.
fn letters_end(text: &str, mut offset: usize, marks: bool) -> usize {
    while offset < text.len() {
        let c = char_at(text, offset);
        if !is_word(c, marks) {
            break;
        }
        offset += c.len_utf8();
    }
    offset
}

/// `'s|'t|'re|'ve|'m|'ll|'d`, case-insensitively.
fn contraction_end(text: &str, offset: usize) -> usize {
    if text.as_bytes()[offset] != b'\'' {
        return offset;
    }
    'suffixes: for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
        let mut end = offset + 1;
        for letter in suffix.chars() {
            if end == text.len() {
                continue 'suffixes;
            }
            let c = char_at(text, end);
            if !folds_to(c, letter) {
                continue 'suffixes;
            }
            end += c.len_utf8();
        }
        return end;
    }
    offset
}

/// End of the piece at `offset`. With `marks`, per `Qwen35PieceEnd` in the
/// C++ tokenizer, which hand-codes the split regex
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;
/// without, the ICU matches of the Qwen2 regex, which has `\p{L}` in place
/// of `[\p{L}\p{M}]`.
fn qwen_piece_end(text: &str, offset: usize, marks: bool) -> usize {
    let contraction = contraction_end(text, offset);
    if contraction != offset {
        return contraction;
    }
    let first = char_at(text, offset);
    let after_first = offset + first.len_utf8();
    if is_word(first, marks) {
        return letters_end(text, after_first, marks);
    }
    if !is_newline(first) && !is_letter(first) && !is_number(first) && after_first < text.len() {
        let next = char_at(text, after_first);
        if is_word(next, marks) {
            return letters_end(text, after_first + next.len_utf8(), marks);
        }
    }
    if is_number(first) {
        return after_first;
    }
    let mut punctuation_start = offset;
    if first == ' ' && after_first < text.len() {
        let next = char_at(text, after_first);
        if !is_space(next) && !is_word(next, marks) && !is_number(next) {
            punctuation_start = after_first;
        }
    }
    let punctuation = char_at(text, punctuation_start);
    if !is_space(punctuation) && !is_word(punctuation, marks) && !is_number(punctuation) {
        let mut end = punctuation_start;
        while end < text.len() {
            let c = char_at(text, end);
            if is_space(c) || is_word(c, marks) || is_number(c) {
                break;
            }
            end += c.len_utf8();
        }
        while end < text.len() {
            let c = char_at(text, end);
            if !is_newline(c) {
                break;
            }
            end += c.len_utf8();
        }
        return end;
    }
    if is_space(first) {
        let (mut end, mut last_start, mut last_newline_end) = (offset, offset, offset);
        while end < text.len() {
            let c = char_at(text, end);
            if !is_space(c) {
                break;
            }
            last_start = end;
            end += c.len_utf8();
            if is_newline(c) {
                last_newline_end = end;
            }
        }
        if last_newline_end != offset {
            return last_newline_end;
        }
        // `\s+(?!\S)` leaves the last space before non-space text to the
        // next piece.
        return if end < text.len() && last_start > offset {
            last_start
        } else {
            end
        };
    }
    after_first
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_bytes_round_trip() {
        assert_eq!(byte_to_gpt2(b'a'), "a");
        assert_eq!(byte_to_gpt2(b' '), "\u{120}");
        assert_eq!(byte_to_gpt2(b'\n'), "\u{10A}");
        assert_eq!(byte_to_gpt2(0xAD), "\u{143}");
        for b in 0..=255u8 {
            assert_eq!(unescape_gpt2(byte_to_gpt2(b).as_bytes()), [b]);
        }
        assert_eq!(hex_byte_token(b"<0x0A>"), Some(10));
        assert_eq!(hex_byte_token(b"<0xZZ>"), None);
    }

    fn toy() -> BpeTokenizer {
        let tokens: Vec<Vec<u8>> = ["a", "b", "c", "ab", "abc", "\u{120}", "<x>"]
            .iter()
            .map(|t| t.as_bytes().to_vec())
            .collect();
        let merges: Vec<Vec<u8>> = ["b c", "a b", "ab c"]
            .iter()
            .map(|m| m.as_bytes().to_vec())
            .collect();
        BpeTokenizer::new(&tokens, &merges, &[(b"<x>".to_vec(), 6)]).unwrap()
    }

    #[test]
    fn merges_follow_rank_and_added_tokens_split_chunks() {
        let bpe = toy();
        // "b c" has rank 0 but "bc" is not a token: merging stops there.
        assert_eq!(bpe.encode("abc"), [0, 1, 2]);
        assert_eq!(bpe.encode("ab<x>a b"), [3, 6, 0, 5, 1]);
        assert_eq!(bpe.decode_token(5), b" ");
        assert_eq!(bpe.decode_token(6), b"<x>");
        assert_eq!(bpe.decode_token(99), b"");
    }

    /// The merge loop before the heap: rescan the word, merge the lowest
    /// rank (leftmost on a tie), stop at a pair without a merged token.
    fn rescan_merge(bpe: &BpeTokenizer, chunk: &[u8]) -> Vec<TokenId> {
        let token_to_id: HashMap<&[u8], TokenId> = bpe
            .id_to_token
            .iter()
            .enumerate()
            .map(|(id, token)| (token.as_slice(), id as TokenId))
            .chain(
                bpe.special_to_id
                    .iter()
                    .map(|(name, id)| (name.as_slice(), *id)),
            )
            .collect();
        let mut word: Vec<TokenId> = chunk
            .iter()
            .filter_map(|&b| bpe.byte_tokens[usize::from(b)])
            .collect();
        while word.len() >= 2 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..word.len() - 1 {
                if let Some(&(rank, _)) = bpe.merge_ranks.get(&(word[i], word[i + 1]))
                    && best.is_none_or(|(r, _)| rank < r)
                {
                    best = Some((rank, i));
                }
            }
            let Some((_, i)) = best else { break };
            let mut merged = bpe.id_to_token[word[i] as usize].clone();
            merged.extend_from_slice(&bpe.id_to_token[word[i + 1] as usize]);
            let Some(&id) = token_to_id.get(merged.as_slice()) else {
                break;
            };
            word[i] = id;
            word.remove(i + 1);
        }
        word
    }

    #[test]
    fn heap_merges_equal_the_rescan_and_stay_linear() {
        // Every string of up to three letters of "abcd" is a token; merges
        // in a scrambled rank order, some without a merged token.
        let letters = *b"abcd";
        let mut tokens: Vec<Vec<u8>> = letters.iter().map(|&l| vec![l]).collect();
        for &x in &letters {
            for &y in &letters {
                tokens.push(vec![x, y]);
                for &z in &letters {
                    if (u32::from(x) + u32::from(y) + u32::from(z)) % 3 != 0 {
                        tokens.push(vec![x, y, z]);
                    }
                }
            }
        }
        let mut merges = Vec::new();
        let mut seed = 7u64;
        for token in &tokens[4..] {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let split = 1 + (seed >> 40) as usize % (token.len() - 1);
            merges.push([&token[..split], b" ", &token[split..]].concat());
        }
        merges.push(b"dd dd".to_vec()); // "dddd" is not a token: merging stops
        let len = merges.len();
        for i in 0..len {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            merges.swap(i, (seed >> 33) as usize % len);
        }
        let bpe = BpeTokenizer::new(&tokens, &merges, &[]).unwrap();
        for n in 0..400usize {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let word: Vec<u8> = (0..n % 37)
                .map(|i| letters[((seed >> (i % 60)) as usize + i * 7) % 4])
                .collect();
            let mut heap = Vec::new();
            bpe.merge_chunk(&word, &mut heap);
            assert_eq!(
                heap,
                rescan_merge(&bpe, &word),
                "{:?}",
                String::from_utf8_lossy(&word)
            );
        }
        // A 200,000-letter piece: quadratic merging took minutes.
        let long = vec![b'a'; 200_000];
        let start = std::time::Instant::now();
        let mut out = Vec::new();
        bpe.merge_chunk(&long, &mut out);
        assert!(start.elapsed().as_secs_f64() < 2.0, "merging is not linear");
    }

    #[test]
    fn added_tokens_are_found_in_one_pass_with_the_longest_on_a_tie() {
        let tokens: Vec<Vec<u8>> = ["a", "<s>", "<s>x", "<t>"]
            .iter()
            .map(|t| t.as_bytes().to_vec())
            .collect();
        let added = [
            (b"<s>".to_vec(), 1),
            (b"<s>x".to_vec(), 2),
            (b"<t>".to_vec(), 3),
        ];
        let bpe = BpeTokenizer::new::<_, &[u8]>(&tokens, &[], &added).unwrap();
        assert_eq!(bpe.encode("a<s>xa<s>a<t><s>x"), [0, 2, 0, 1, 0, 3, 2]);
        let long = "a<t>".repeat(50_000);
        let start = std::time::Instant::now();
        assert_eq!(bpe.encode(&long).len(), 100_000);
        assert!(
            start.elapsed().as_secs_f64() < 2.0,
            "the added-token scan is not linear"
        );
    }

    fn split(text: &str, marks: bool) -> Vec<&str> {
        let mut out = Vec::new();
        let mut offset = 0;
        while offset < text.len() {
            let end = qwen_piece_end(text, offset, marks);
            out.push(&text[offset..end]);
            offset = end;
        }
        out
    }

    fn pieces(text: &str) -> Vec<&str> {
        split(text, true)
    }

    #[test]
    fn qwen35_pieces_follow_the_split_regex() {
        assert_eq!(pieces("Hello world"), ["Hello", " world"]);
        assert_eq!(
            pieces("I'm here, don'T"),
            ["I", "'m", " here", ",", " don", "'T"]
        );
        assert_eq!(pieces("abc123"), ["abc", "1", "2", "3"]);
        assert_eq!(pieces("a  b"), ["a", " ", " b"]);
        assert_eq!(pieces("x\n\ny"), ["x", "\n\n", "y"]);
        assert_eq!(pieces("end.  "), ["end", ".", "  "]);
        assert_eq!(pieces(" !!?\nnext"), [" !!?\n", "next"]);
        assert_eq!(pieces("Grüße, 東京"), ["Grüße", ",", " 東京"]);
        assert_eq!(pieces("e\u{301}t\u{17F}"), ["e\u{301}t\u{17F}"]);
        assert_eq!(pieces("'\u{17F}"), ["'\u{17F}"], "long s folds to s");
    }

    #[test]
    fn qwen2_pieces_leave_marks_out_of_letter_runs() {
        let qwen2 = |text| split(text, false);
        assert_eq!(qwen2("I'm here, don'T"), pieces("I'm here, don'T"));
        assert_eq!(qwen2("a  b\n\n c"), pieces("a  b\n\n c"));
        // U+094D VIRAMA and U+0947 VOWEL SIGN E are marks (Mn): a mark may
        // lead a letter run, as the optional first character, but not join
        // one.
        assert_eq!(
            qwen2("\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947}"),
            ["\u{928}\u{92E}\u{938}", "\u{94D}\u{924}", "\u{947}"]
        );
        assert_eq!(qwen2(" \u{301}x"), [" \u{301}", "x"]);
        assert_eq!(qwen2("e\u{301}t"), ["e", "\u{301}t"]);
        assert_eq!(qwen2("'\u{17F}"), ["'\u{17F}"]);
    }

    #[test]
    fn qwen35_normalises_to_nfc_before_splitting() {
        // Byte tokens only: token IDs are the bytes.
        let tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| byte_level(&[b])).collect();
        let bpe = BpeTokenizer::new::<_, &[u8]>(&tokens, &[], &[]).unwrap();
        assert_eq!(bpe.encode("e\u{301}"), [0x65, 0xCC, 0x81]);
        let qwen = bpe.with_pre_tokenizer(PreTokenizer::Qwen35);
        assert_eq!(qwen.encode("e\u{301}"), [0xC3, 0xA9], "U+00E9 after NFC");
    }

    fn byte_level(bytes: &[u8]) -> Vec<u8> {
        bytes
            .iter()
            .flat_map(|&b| byte_to_gpt2(b).into_bytes())
            .collect()
    }
}
