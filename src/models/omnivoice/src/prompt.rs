//! The text tokens of one generation (`prompt.cpp`, ported from
//! k2-fsa/OmniVoice's `omnivoice/models/omnivoice.py`
//! `_prepare_inference_inputs` and `_tokenize_with_nonverbal_tags`,
//! Apache-2.0).

use crate::text::combine_text;
use crate::tokenizer::Tokenizer;

const NONVERBAL_TAGS: [&str; 13] = [
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
];

/// The text side of one generation: what to say, the reference transcript
/// when cloning, the resolved language ID and voice-design instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptText {
    pub text: String,
    pub reference_text: Option<String>,
    pub language: Option<String>,
    pub instruct: Option<String>,
    /// Adds `<|denoise|>` when a reference voice is present.
    pub denoise: bool,
    pub has_reference: bool,
}

impl Default for PromptText {
    fn default() -> Self {
        Self {
            text: String::new(),
            reference_text: None,
            language: None,
            instruct: None,
            denoise: true,
            has_reference: false,
        }
    }
}

/// `_tokenize_with_nonverbal_tags`: encodes the text between non-verbal
/// tags and each tag separately (`EncodeWithNonverbalTags`).
pub fn encode_with_nonverbal_tags(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    let mut ids = Vec::new();
    let mut start = 0;
    loop {
        let mut found: Option<(usize, &str)> = None;
        for candidate in NONVERBAL_TAGS {
            if let Some(at) = text[start..].find(candidate).map(|at| at + start)
                && found.is_none_or(|(tag_at, _)| at < tag_at)
            {
                found = Some((at, candidate));
            }
        }
        let end = found.map_or(text.len(), |(at, _)| at);
        if end > start {
            ids.extend(tokenizer.encode(&text[start..end])?);
        }
        let Some((at, tag)) = found else {
            return Ok(ids);
        };
        ids.extend(tokenizer.encode(tag)?);
        start = at + tag.len();
    }
}

/// `_prepare_inference_inputs`' text tokens, which precede the reference
/// and target audio frames (`EncodePromptText`):
///   `[<|denoise|>]<|lang_start|>LANG<|lang_end|>`
///   `<|instruct_start|>INSTRUCT<|instruct_end|>`
///   `<|text_start|>REFERENCE TEXT<|text_end|>`
/// Non-verbal tags such as `[laughter]` are encoded on their own.
pub fn encode_prompt_text(tokenizer: &Tokenizer, prompt: &PromptText) -> Result<Vec<u32>, String> {
    let or_none = |value: &Option<String>| match value {
        Some(value) if !value.is_empty() => value.clone(),
        _ => "None".to_owned(),
    };
    let mut style = String::new();
    if prompt.denoise && prompt.has_reference {
        style += "<|denoise|>";
    }
    style += &format!("<|lang_start|>{}<|lang_end|>", or_none(&prompt.language));
    style += &format!(
        "<|instruct_start|>{}<|instruct_end|>",
        or_none(&prompt.instruct)
    );
    let mut ids = tokenizer.encode(&style)?;
    let text = format!(
        "<|text_start|>{}<|text_end|>",
        combine_text(&prompt.text, prompt.reference_text.as_deref())
    );
    ids.extend(encode_with_nonverbal_tags(tokenizer, &text)?);
    Ok(ids)
}
