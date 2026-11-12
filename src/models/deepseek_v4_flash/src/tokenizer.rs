//! The GPT-2 byte-level BPE of DeepSeek V4 Flash (`runtime/tokenizer.cpp`):
//! token strings and merge ranks from the GGUF, the JoyAI pre-tokenizer
//! (`tokenizer.ggml.pre = "joyai-llm"`) written out by hand, and chat
//! special tokens inserted by ID.
//!
//! Like C++, text ends at its first NUL byte, a later duplicate token
//! string or merge wins, a symbol without a token falls back to the tokens
//! of its single bytes (which exist only for ASCII), and merges take the
//! lowest rank, the leftmost pair among equals.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::gguf::Model;

/// FNV-1a, the hash of the C++ tables.
#[derive(Default)]
struct Fnv(u64);

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = if self.0 == 0 {
            1_469_598_103_934_665_603
        } else {
            self.0
        };
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
        self.0 = hash;
    }
}

type Table = HashMap<Vec<u8>, i32, BuildHasherDefault<Fnv>>;

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const DSML: &str = "｜DSML｜";

/// The tokenizer of a loaded model (`ds4_vocab`).
pub struct Vocab {
    tokens: Vec<Vec<u8>>,
    token_to_id: Table,
    merge_rank: Table,
    pub bos: i32,
    pub eos: i32,
    pub user: i32,
    pub assistant: i32,
    pub think_start: i32,
    pub think_end: i32,
    pub dsml: i32,
}

impl Vocab {
    /// `ds4_vocab_create`: token strings, merges and special token IDs.
    pub fn load(model: &Model) -> Result<Self, String> {
        let tokens = model
            .string_array("tokenizer.ggml.tokens")
            .filter(|tokens| tokens.remaining() <= i32::MAX as u64)
            .ok_or("GGUF tokenizer token table is missing or invalid")?;
        let merges = model
            .string_array("tokenizer.ggml.merges")
            .ok_or("GGUF tokenizer merge table is missing or invalid")?;
        let n_vocab = tokens.remaining() as usize;
        let mut strings = Vec::with_capacity(n_vocab);
        let mut token_to_id = Table::with_capacity_and_hasher(n_vocab, Default::default());
        let mut tokens = tokens;
        for i in 0..n_vocab {
            let token = tokens
                .next()
                .ok_or("GGUF tokenizer token table ended early")?;
            token_to_id.insert(token.to_vec(), i as i32);
            strings.push(token.to_vec());
        }
        let mut merge_rank =
            Table::with_capacity_and_hasher(merges.remaining() as usize, Default::default());
        for (rank, merge) in merges.enumerate() {
            merge_rank.insert(merge.to_vec(), rank as i32);
        }
        let lookup = |text: &str| {
            token_to_id.get(text.as_bytes()).copied().ok_or_else(|| {
                eprintln!("ds4: required tokenizer token is missing: {text}");
                "required tokenizer token is missing".to_owned()
            })
        };
        Ok(Self {
            bos: lookup(BOS)?,
            eos: lookup(EOS)?,
            user: lookup(USER)?,
            assistant: lookup(ASSISTANT)?,
            think_start: lookup(THINK_START)?,
            think_end: lookup(THINK_END)?,
            dsml: lookup(DSML)?,
            tokens: strings,
            token_to_id,
            merge_rank,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// `ds4_vocab_tokenize`: BPE over `text` up to its first NUL byte.
    pub fn tokenize(&self, text: &[u8], out: &mut Vec<i32>) {
        let text = &text[..text.iter().position(|&b| b == 0).unwrap_or(text.len())];
        let len = text.len();
        let mut pos = 0;
        while pos < len {
            let start = pos;
            let c = text[pos];
            if c.is_ascii_digit() {
                let mut digits = 0;
                while pos < len && text[pos].is_ascii_digit() && digits < 3 {
                    pos += 1;
                    digits += 1;
                }
            } else if cjk_at(text, pos) {
                loop {
                    pos = next_utf8_char(text, pos);
                    if !(pos < len && cjk_at(text, pos)) {
                        break;
                    }
                }
            } else if punct_symbol(c) && pos + 1 < len && text[pos + 1].is_ascii_alphabetic() {
                pos += 1;
                while pos < len && text[pos].is_ascii_alphabetic() {
                    pos += 1;
                }
            } else if letter_like_at(text, pos) {
                pos = consume_letters(text, pos);
            } else if !newline(c)
                && !punct_symbol(c)
                && pos + 1 < len
                && letter_like_at(text, pos + 1)
            {
                pos = consume_letters(text, pos + 1);
            } else if c == b' ' && pos + 1 < len && punct_symbol(text[pos + 1]) {
                pos += 1;
                while pos < len && punct_symbol(text[pos]) {
                    pos += 1;
                }
                while pos < len && newline(text[pos]) {
                    pos += 1;
                }
            } else if punct_symbol(c) {
                while pos < len && punct_symbol(text[pos]) {
                    pos += 1;
                }
                while pos < len && newline(text[pos]) {
                    pos += 1;
                }
            } else if space(c) {
                let mut p = pos;
                let mut last_newline_end = 0;
                while p < len && space(text[p]) {
                    let sc = text[p];
                    p += 1;
                    if newline(sc) {
                        last_newline_end = p;
                    }
                }
                if last_newline_end != 0 {
                    pos = last_newline_end;
                } else if p < len
                    && p > pos + 1
                    && (letter_like_at(text, p) || punct_symbol(text[p]))
                {
                    // A single leading space joins the following word or
                    // punctuation run: "    int" is "   " then " int".
                    pos = p - 1;
                } else {
                    pos = p;
                }
            } else {
                pos = next_utf8_char(text, pos);
            }
            if pos == start {
                pos = next_utf8_char(text, pos);
            }
            self.emit_piece(&text[start..pos], out);
        }
    }

    /// BPE of one pre-tokenized piece (`bpe_emit_piece`).
    fn emit_piece(&self, raw: &[u8], out: &mut Vec<i32>) {
        let encoded = byte_encode(raw);
        let mut symbols: Vec<Vec<u8>> = Vec::new();
        let mut off = 0;
        while off < encoded.len() {
            let mut n = utf8_len(encoded[off]);
            if off + n > encoded.len() {
                n = 1;
            }
            symbols.push(encoded[off..off + n].to_vec());
            off += n;
        }
        let mut pair = Vec::new();
        loop {
            let mut best: Option<(usize, i32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                pair.clear();
                pair.extend_from_slice(&symbols[i]);
                pair.push(b' ');
                pair.extend_from_slice(&symbols[i + 1]);
                if let Some(&rank) = self.merge_rank.get(&pair)
                    && rank >= 0
                    && best.is_none_or(|(_, best_rank)| rank < best_rank)
                {
                    best = Some((i, rank));
                }
            }
            let Some((i, _)) = best else { break };
            let next = symbols.remove(i + 1);
            symbols[i].extend_from_slice(&next);
        }
        for symbol in &symbols {
            if let Some(&token) = self.token_to_id.get(symbol) {
                out.push(token);
            } else {
                for &byte in symbol {
                    if let Some(&token) = self.token_to_id.get([byte].as_slice()) {
                        out.push(token);
                    }
                }
            }
        }
    }

    /// `ds4_vocab_tokenize_rendered_chat`: special tokens by ID, the text
    /// between them through BPE.
    pub fn tokenize_rendered_chat(&self, text: &[u8], out: &mut Vec<i32>) {
        let text = &text[..text.iter().position(|&b| b == 0).unwrap_or(text.len())];
        let specials = [
            (BOS, self.bos),
            (EOS, self.eos),
            (USER, self.user),
            (ASSISTANT, self.assistant),
            (THINK_START, self.think_start),
            (THINK_END, self.think_end),
            (DSML, self.dsml),
        ];
        let mut span = 0;
        let mut p = 0;
        while p < text.len() {
            if let Some(&(special, token)) = specials
                .iter()
                .find(|(special, _)| text[p..].starts_with(special.as_bytes()))
            {
                self.tokenize(&text[span..p], out);
                out.push(token);
                p += special.len();
                span = p;
                continue;
            }
            p += 1;
        }
        self.tokenize(&text[span..p], out);
    }

    /// `encode_chat_prompt`: BOS, the system text, the user turn and the
    /// assistant prefix with thinking closed.
    pub fn encode_chat(&self, system: &[u8], prompt: &[u8], out: &mut Vec<i32>) {
        out.push(self.bos);
        if system.first().is_some_and(|&b| b != 0) {
            self.tokenize(system, out);
        }
        out.push(self.user);
        self.tokenize(prompt, out);
        out.push(self.assistant);
        out.push(self.think_end);
    }

    /// `ds4_vocab_chat_begin`.
    pub fn chat_begin(&self, out: &mut Vec<i32>) {
        out.push(self.bos);
    }

    /// `ds4_vocab_chat_append_message`.
    pub fn chat_append_message(&self, out: &mut Vec<i32>, role: &str, content: &[u8]) {
        match role {
            "system" | "developer" => self.tokenize(content, out),
            "assistant" => {
                out.push(self.assistant);
                if !content.starts_with(b"<think>") && !content.starts_with(b"</think>") {
                    out.push(self.think_end);
                }
                self.tokenize(content, out);
            }
            _ => {
                out.push(self.user);
                if role == "tool" || role == "function" {
                    self.tokenize(b"Tool: ", out);
                }
                self.tokenize(content, out);
            }
        }
    }

    /// `ds4_vocab_chat_append_assistant_prefix`.
    pub fn chat_append_assistant_prefix(&self, out: &mut Vec<i32>) {
        out.push(self.assistant);
        out.push(self.think_end);
    }

    /// `ds4_vocab_token_text`: the bytes of `token`; literal for special
    /// tokens (those containing U+FF5C), empty outside the vocabulary.
    pub fn token_text(&self, token: i32) -> Vec<u8> {
        let Some(s) = usize::try_from(token).ok().and_then(|t| self.tokens.get(t)) else {
            return Vec::new();
        };
        const BAR: &[u8] = "｜".as_bytes();
        if s.windows(BAR.len()).any(|w| w == BAR) {
            return s.clone();
        }
        let mut out = Vec::with_capacity(s.len());
        let mut pos = 0;
        while pos < s.len() {
            let cp = utf8_decode_one(s, &mut pos);
            if let Some(b) = gpt2_codepoint_to_byte(cp) {
                out.push(b);
            }
        }
        out
    }

    /// `ds4_vocab_is_stop`.
    pub fn is_stop(&self, token: i32) -> bool {
        token == self.eos
    }
}

fn gpt2_printable(b: u32) -> bool {
    (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174
}

fn gpt2_byte_to_codepoint(b: u8) -> u32 {
    let b = u32::from(b);
    if gpt2_printable(b) {
        return b;
    }
    256 + (0..b).filter(|&x| !gpt2_printable(x)).count() as u32
}

fn gpt2_codepoint_to_byte(cp: u32) -> Option<u8> {
    if gpt2_printable(cp) && cp <= 255 {
        return Some(cp as u8);
    }
    let mut n = 0;
    for b in 0..256u32 {
        if gpt2_printable(b) {
            continue;
        }
        if cp == 256 + n {
            return Some(b as u8);
        }
        n += 1;
    }
    None
}

/// The printable code points of `raw`'s bytes, as UTF-8 (`byte_encode`).
fn byte_encode(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() * 2);
    for &b in raw {
        let cp = gpt2_byte_to_codepoint(b);
        let c = char::from_u32(cp).unwrap_or('\0');
        let mut buf = [0u8; 4];
        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
    }
    out
}

fn utf8_len(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if c & 0xe0 == 0xc0 {
        2
    } else if c & 0xf0 == 0xe0 {
        3
    } else if c & 0xf8 == 0xf0 {
        4
    } else {
        1
    }
}

fn next_utf8_char(s: &[u8], pos: usize) -> usize {
    let mut n = utf8_len(s[pos]);
    if pos + n > s.len() {
        n = 1;
    }
    pos + n
}

fn newline(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}

fn space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn punct_symbol(c: u8) -> bool {
    (b'!'..=b'/').contains(&c)
        || (b':'..=b'@').contains(&c)
        || (b'['..=b'`').contains(&c)
        || (b'{'..=b'~').contains(&c)
}

/// ASCII letters, and every non-ASCII byte (`joyai_letter_like_at`).
fn letter_like_at(s: &[u8], pos: usize) -> bool {
    let c = s[pos];
    c >= 128 || c.is_ascii_alphabetic()
}

fn consume_letters(s: &[u8], mut pos: usize) -> usize {
    while pos < s.len() && letter_like_at(s, pos) {
        pos = next_utf8_char(s, pos);
    }
    pos
}

/// The code point at `pos` as `utf8_peek_one` decodes it (no validation).
fn peek_one(s: &[u8], pos: usize) -> u32 {
    let c0 = s[pos];
    let mut n = utf8_len(c0);
    if pos + n > s.len() {
        n = 1;
    }
    let cont = |i: usize| u32::from(s[pos + i] & 0x3f);
    match n {
        1 => u32::from(c0),
        2 => (u32::from(c0 & 0x1f) << 6) | cont(1),
        3 => (u32::from(c0 & 0x0f) << 12) | (cont(1) << 6) | cont(2),
        _ => (u32::from(c0 & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
    }
}

fn cjk_at(s: &[u8], pos: usize) -> bool {
    if s[pos] < 128 {
        return false;
    }
    let cp = peek_one(s, pos);
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

/// `utf8_decode_one` of the token text decoder.
fn utf8_decode_one(s: &[u8], pos: &mut usize) -> u32 {
    let len = s.len();
    let c = s[*pos];
    let cont = |i: usize| u32::from(s[*pos + i] & 0x3f);
    if c < 0x80 || *pos + 1 >= len {
        *pos += 1;
        return u32::from(c);
    }
    if c & 0xe0 == 0xc0 && *pos + 1 < len {
        let cp = (u32::from(c & 0x1f) << 6) | cont(1);
        *pos += 2;
        return cp;
    }
    if c & 0xf0 == 0xe0 && *pos + 2 < len {
        let cp = (u32::from(c & 0x0f) << 12) | (cont(1) << 6) | cont(2);
        *pos += 3;
        return cp;
    }
    if c & 0xf8 == 0xf0 && *pos + 3 < len {
        let cp = (u32::from(c & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3);
        *pos += 4;
        return cp;
    }
    *pos += 1;
    u32::from(c)
}
