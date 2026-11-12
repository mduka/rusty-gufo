//! Generated token log probabilities (`logprobs.hpp`,
//! `responses_logprobs.hpp`): the request controls, the decoded entries a
//! token piece carries, and their Chat Completions, Completions and
//! Responses JSON.

use std::ops::Deref;

use gufo_core::json::Value;
use gufo_core::sampling::{SamplingConfig, TokenId, TokenLogprob, TokenLogprobs};
use gufo_core::utf8::{Utf8Decoder, lossy};

use crate::sampling_request::{SamplingRequestError, read_integer_or_null};

/// The lowest log probability the JSON reports (`-inf` included).
const MIN_LOGPROB: f64 = -9999.0;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecodedLogprob {
    pub id: TokenId,
    pub token: String,
    pub logprob: f64,
}

/// Sampled token first, then the unique top candidates in descending order,
/// as in vLLM v0.32.0's `append_logprobs_for_next_position`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecodedLogprobs {
    pub entries: Vec<DecodedLogprob>,
}

impl DecodedLogprobs {
    /// The bytes these entries hold, for output buffer accounting.
    pub fn storage_bytes(&self) -> usize {
        size_of::<Self>()
            + self.entries.capacity() * size_of::<DecodedLogprob>()
            + self
                .entries
                .iter()
                .map(|entry| entry.token.capacity())
                .sum::<usize>()
    }
}

/// A streamed token piece: its text bytes, which it dereferences to, and the
/// log probabilities of a request that asks for them.
#[derive(Debug, Clone, Copy)]
pub struct TokenPiece<'a> {
    pub text: &'a [u8],
    pub logprobs: Option<&'a DecodedLogprobs>,
}

impl<'a> TokenPiece<'a> {
    pub fn new(text: &'a [u8]) -> Self {
        Self {
            text,
            logprobs: None,
        }
    }

    pub fn with_logprobs(text: &'a [u8], logprobs: Option<&'a DecodedLogprobs>) -> Self {
        Self { text, logprobs }
    }
}

impl Deref for TokenPiece<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.text
    }
}

/// `logprobs` and `top_logprobs` of a Chat Completions body (`chat`), or the
/// legacy Completions `logprobs` count (`ParseLogprobs`).
pub fn parse_logprobs(
    body: &Value,
    chat: bool,
    config: &mut SamplingConfig,
) -> Result<(), SamplingRequestError> {
    config.logprobs = None;
    let enabled = body.find("logprobs");
    if chat {
        if enabled.is_some_and(|value| !value.is_null() && !value.is_bool()) {
            return Err(SamplingRequestError {
                message: "'logprobs' must be a boolean".into(),
                code: "invalid_logprobs".into(),
            });
        }
        let mut count = 0u32;
        read_integer_or_null(body, "top_logprobs", 0, 20, &mut count, true)?;
        if enabled.is_some_and(|value| value.as_bool(false)) {
            config.logprobs = Some(count);
        } else if count != 0 {
            return Err(SamplingRequestError {
                message: "'top_logprobs' requires 'logprobs': true".into(),
                code: "invalid_top_logprobs".into(),
            });
        }
    } else if enabled.is_some_and(|value| !value.is_null()) {
        let mut count = 0u32;
        read_integer_or_null(body, "logprobs", 0, 20, &mut count, true)?;
        config.logprobs = Some(count);
    }
    Ok(())
}

/// Corrects byte-fallback tokens against the last four sampled token pieces,
/// independently for every alternative (vLLM v0.32.0 `LogprobsProcessor`).
#[derive(Debug, Default)]
pub struct LogprobDecoder {
    history: Vec<Vec<u8>>,
}

impl LogprobDecoder {
    /// The entries of `values`, each token's text from `decode`.
    pub fn decode_token(
        &mut self,
        values: &TokenLogprobs,
        mut decode: impl FnMut(TokenId) -> Vec<u8>,
    ) -> DecodedLogprobs {
        let mut result = DecodedLogprobs::default();
        let raw = decode(values.selected.token);
        let mut append = |entry: &TokenLogprob, bytes: &[u8]| {
            result.entries.push(DecodedLogprob {
                id: entry.token,
                token: self.correct(bytes),
                logprob: entry.logprob,
            });
        };
        append(&values.selected, &raw);
        for entry in &values.top {
            if entry.token != values.selected.token {
                append(entry, &decode(entry.token));
            }
        }
        self.history.push(raw);
        if self.history.len() > 4 {
            self.history.remove(0);
        }
        result
    }

    fn correct(&self, raw: &[u8]) -> String {
        const REPLACEMENT: char = '\u{FFFD}';
        let text = lossy(raw);
        if !text.ends_with(REPLACEMENT) {
            return text;
        }
        let history = &self.history;
        for count in 1..=history.len() {
            let begin = history.len() - count;
            let mut combined: Vec<u8> = history[begin..].concat();
            combined.extend_from_slice(raw);
            let decoded = lossy(&combined);
            if decoded.ends_with(REPLACEMENT) {
                continue;
            }
            let mut clean_end = history.len();
            while clean_end > begin && lossy(&history[clean_end - 1]).ends_with(REPLACEMENT) {
                clean_end -= 1;
            }
            let clean = lossy(&history[begin..clean_end].concat());
            let (clean, bytes) = (clean.as_bytes(), decoded.as_bytes());
            let mut common = 0;
            while common < clean.len() && common < bytes.len() && clean[common] == bytes[common] {
                common += 1;
            }
            // Keep the substring on a Unicode scalar boundary.
            while common < bytes.len() && common > 0 && bytes[common] & 0xc0 == 0x80 {
                common -= 1;
            }
            return decoded[common..].to_owned();
        }
        String::new()
    }
}

/// One entry's `token`, `logprob` and UTF-8 `bytes` (`LogprobJson`).
pub fn logprob_json(value: &DecodedLogprob) -> Value {
    let bytes: Vec<Value> = value
        .token
        .bytes()
        .map(|byte| Value::from(u32::from(byte)))
        .collect();
    Value::object()
        .with("token", value.token.as_str())
        .with("logprob", value.logprob.max(MIN_LOGPROB))
        .with("bytes", Value::Array(bytes))
}

/// The Chat Completions `logprobs` object of `tokens` with `count` top
/// alternatives each (`ChatLogprobs`).
pub fn chat_logprobs(tokens: &[DecodedLogprobs], count: u32) -> Value {
    let mut content = Value::array();
    for token in tokens {
        let mut entry = logprob_json(&token.entries[0]);
        let mut top = Value::array();
        for alternative in token.entries.iter().take(count as usize) {
            top.push(logprob_json(alternative));
        }
        entry.set("top_logprobs", top);
        content.push(entry);
    }
    Value::object().with("content", content)
}

/// Unicode scalars of UTF-8 `text` (`UnicodeLength`).
pub fn unicode_length(text: &[u8]) -> usize {
    text.iter().filter(|&&byte| byte & 0xc0 != 0x80).count()
}

/// The legacy Completions `logprobs` object of `tokens`, whose text starts
/// `offset` scalars into the choice (`CompletionLogprobs`).
pub fn completion_logprobs(tokens: &[DecodedLogprobs], mut offset: usize) -> Value {
    let mut result = Value::object();
    for name in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
        result.set(name, Value::array());
    }
    for token in tokens {
        let selected = &token.entries[0];
        result.member_mut("tokens").push(selected.token.as_str());
        result
            .member_mut("token_logprobs")
            .push(selected.logprob.max(MIN_LOGPROB));
        result.member_mut("text_offset").push(offset);
        offset += unicode_length(selected.token.as_bytes());
        let mut top = Value::object();
        for entry in &token.entries {
            top.set(&entry.token, entry.logprob.max(MIN_LOGPROB));
        }
        result.member_mut("top_logprobs").push(top);
    }
    result
}

/// `include: ["message.output_text.logprobs"]` and `top_logprobs` of a
/// Responses body (`ParseResponsesLogprobs`).
pub fn parse_responses_logprobs(
    body: &Value,
    config: &mut SamplingConfig,
) -> Result<(), SamplingRequestError> {
    config.logprobs = None;
    let invalid = || SamplingRequestError {
        message: "'include' must be an array of strings".into(),
        code: "invalid_include".into(),
    };
    let mut enabled = false;
    if let Some(include) = body.find("include").filter(|include| !include.is_null()) {
        if !include.is_array() {
            return Err(invalid());
        }
        for value in include.items() {
            if !value.is_string() {
                return Err(invalid());
            }
            enabled |= value.str() == "message.output_text.logprobs";
        }
    }
    let mut count = 0u32;
    read_integer_or_null(body, "top_logprobs", 0, 20, &mut count, true)?;
    if enabled {
        config.logprobs = Some(count);
    }
    Ok(())
}

/// A range in the UTF-8-normalized, unfiltered model output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TextSourceSpan {
    pub begin: usize,
    pub size: usize,
}

/// Streaming text events omit the `bytes` field that `output_text` parts
/// carry (`ResponsesStreamLogprobs`).
pub fn responses_stream_logprobs(tokens: &Value) -> Value {
    let field = |value: &Value, name: &str| value.find(name).cloned().unwrap_or_default();
    let mut result = Value::array();
    for token in tokens.items() {
        let mut top = Value::array();
        if let Some(alternatives) = token.find("top_logprobs") {
            for alternative in alternatives.items() {
                top.push(
                    Value::object()
                        .with("token", field(alternative, "token"))
                        .with("logprob", field(alternative, "logprob")),
                );
            }
        }
        result.push(
            Value::object()
                .with("token", field(token, "token"))
                .with("logprob", field(token, "logprob"))
                .with("top_logprobs", top),
        );
    }
    result
}

/// Joins filtered output spans to generated tokens by source position, never
/// by matching text (a tool argument can repeat the visible answer
/// verbatim). A token crossing a text/framing boundary belongs to its first
/// visible text span; pure reasoning, tool markup and hidden stop tokens are
/// omitted (`ResponsesLogprobs`).
pub struct ResponsesLogprobs {
    count: u32,
    decoder: Utf8Decoder,
    tokens: Vec<SourceToken>,
    input_bytes: usize,
    consumed_bytes: usize,
    decoded_bytes: usize,
    mapped: usize,
}

#[derive(Debug)]
struct SourceToken {
    input_begin: usize,
    input_end: usize,
    values: DecodedLogprobs,
    begin: usize,
    end: usize,
    mapped: bool,
    emitted: bool,
}

impl ResponsesLogprobs {
    pub fn new(count: u32) -> Self {
        Self {
            count,
            decoder: Utf8Decoder::default(),
            tokens: Vec::new(),
            input_bytes: 0,
            consumed_bytes: 0,
            decoded_bytes: 0,
            mapped: 0,
        }
    }

    /// Records `piece` at its position in the decoded output (`last` ends
    /// the output).
    pub fn push(&mut self, piece: &TokenPiece<'_>, last: bool) {
        if let Some(values) = piece.logprobs
            && !piece.text.is_empty()
        {
            self.tokens.push(SourceToken {
                input_begin: self.input_bytes,
                input_end: self.input_bytes + piece.text.len(),
                values: values.clone(),
                begin: 0,
                end: 0,
                mapped: false,
                emitted: false,
            });
        }
        self.input_bytes += piece.text.len();
        let Self {
            decoder,
            tokens,
            consumed_bytes,
            decoded_bytes,
            mapped,
            ..
        } = self;
        decoder.push_observed(piece.text, last, |input, output| {
            let end = *consumed_bytes + input;
            for token in tokens[*mapped..]
                .iter_mut()
                .take_while(|token| token.input_begin < end)
            {
                if !token.mapped {
                    token.begin = *decoded_bytes;
                }
                token.end = *decoded_bytes + output;
                token.mapped = true;
            }
            while *mapped < tokens.len() && tokens[*mapped].input_end <= end {
                *mapped += 1;
            }
            *consumed_bytes = end;
            *decoded_bytes += output;
        });
    }

    /// The entries of the tokens behind `spans` not yet taken.
    pub fn take(&mut self, spans: &[TextSourceSpan]) -> Value {
        let mut result = Value::array();
        for span in spans {
            if span.size == 0 {
                continue;
            }
            // Unmapped tokens are only the trailing incomplete UTF-8 scalar.
            let mapped_end = self.mapped
                + usize::from(self.mapped < self.tokens.len() && self.tokens[self.mapped].mapped);
            let tokens = &mut self.tokens[..mapped_end];
            let first = tokens.partition_point(|token| token.end <= span.begin);
            for token in &mut tokens[first..] {
                if token.begin >= span.begin + span.size {
                    break;
                }
                if token.emitted {
                    continue;
                }
                let entries = chat_logprobs(std::slice::from_ref(&token.values), self.count);
                if let Some(entry) = entries
                    .find("content")
                    .and_then(|content| content.items().first())
                {
                    result.push(entry.clone());
                }
                token.emitted = true;
            }
        }
        result
    }
}
