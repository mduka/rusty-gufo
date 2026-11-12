//! Chat prompt around the audio embeddings, and parsing of the model's
//! `language X<asr_text>transcript` answer.

use crate::tokenizer::Tokenizer;
use gufo_core::bpe::TokenId;

pub const IM_START: TokenId = 151_644;
pub const IM_END: TokenId = 151_645;
pub const ENDOFTEXT: TokenId = 151_643;
pub const AUDIO_START: TokenId = 151_669;
pub const AUDIO_END: TokenId = 151_670;
pub const AUDIO_PAD: TokenId = 151_676;
const NEWLINE: TokenId = 198;
const SYSTEM: TokenId = 8948;
const USER: TokenId = 872;
const ASSISTANT: TokenId = 77_091;
const ASR_TEXT_TAG: &[u8] = b"<asr_text>";
const LANGUAGE_PREFIX: &[u8] = b"language ";

/// `<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>`
/// `{pad × audio_tokens}<|audio_end|><|im_end|>\n<|im_start|>assistant\n`,
/// followed by `language {language}<asr_text>` when the language is forced.
pub fn build_prompt(
    tokenizer: &Tokenizer,
    context: &str,
    language: Option<&str>,
    audio_tokens: usize,
) -> Result<Vec<TokenId>, String> {
    if audio_tokens == 0 {
        return Err("Qwen3-ASR prompt requires at least one audio embedding token".into());
    }
    let mut prompt = Vec::with_capacity(15 + audio_tokens + context.len() / 2);
    prompt.extend_from_slice(&[IM_START, SYSTEM, NEWLINE]);
    if !context.is_empty() {
        prompt.extend(tokenizer.encode(context)?);
    }
    prompt.extend_from_slice(&[IM_END, NEWLINE, IM_START, USER, NEWLINE, AUDIO_START]);
    prompt.resize(prompt.len() + audio_tokens, AUDIO_PAD);
    prompt.extend_from_slice(&[AUDIO_END, IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE]);
    if let Some(language) = language {
        prompt.extend(tokenizer.encode(&format!("language {language}<asr_text>"))?);
    }
    Ok(prompt)
}

/// Language and transcript bytes of a decoded answer.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Transcription {
    pub language: Vec<u8>,
    pub text: Vec<u8>,
}

/// C `isspace` in the "C" locale.
fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

fn trim(mut bytes: &[u8]) -> Vec<u8> {
    while let [first, rest @ ..] = bytes
        && is_space(*first)
    {
        bytes = rest;
    }
    while let [rest @ .., last] = bytes
        && is_space(*last)
    {
        bytes = rest;
    }
    bytes.to_vec()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Splits `language X<asr_text>text`; a forced language keeps the whole
/// answer as text, and `language None` means no speech.
pub fn parse_transcription(decoded: &[u8], forced_language: Option<&str>) -> Transcription {
    if let Some(language) = forced_language {
        return Transcription {
            language: language.as_bytes().to_vec(),
            text: trim(decoded),
        };
    }
    let Some(tag) = find(decoded, ASR_TEXT_TAG) else {
        return Transcription {
            language: Vec::new(),
            text: trim(decoded),
        };
    };
    let metadata = &decoded[..tag];
    let mut language = Vec::new();
    if let Some(prefix) = find(metadata, LANGUAGE_PREFIX) {
        language = trim(&metadata[prefix + LANGUAGE_PREFIX.len()..]);
        if language.eq_ignore_ascii_case(b"none") {
            return Transcription::default();
        }
    }
    Transcription {
        language,
        text: trim(&decoded[tag + ASR_TEXT_TAG.len()..]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_split_into_language_and_text() {
        let parsed = parse_transcription(b"language English<asr_text> Hello there.\n", None);
        assert_eq!(parsed.language, b"English");
        assert_eq!(parsed.text, b"Hello there.");
        assert_eq!(
            parse_transcription(b"language None<asr_text>", None),
            Transcription::default()
        );
        let forced = parse_transcription(b" Bonjour ", Some("French"));
        assert_eq!(
            (forced.language.as_slice(), forced.text.as_slice()),
            (&b"French"[..], &b"Bonjour"[..])
        );
        assert_eq!(parse_transcription(b"  untagged ", None).text, b"untagged");
    }
}
