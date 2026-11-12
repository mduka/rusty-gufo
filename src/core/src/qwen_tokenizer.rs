//! The Qwen3.x text tokenizer from GGUF metadata, a port of
//! `QwenTokenizer::CreateFromGguf` (`src/models/qwen/tokenizer.cpp`).

use crate::bpe::{BpeTokenizer, PreTokenizer, TokenId};
use crate::gguf::Gguf;
use crate::json_constraint::{ConstraintVocabulary, Piece};

/// Added tokens the C++ loader recognises by name, besides every `<|...|>`.
const NAMED_SPECIAL: [&str; 21] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|quad_start|>",
    "<|quad_end|>",
];

fn is_special(token: &[u8]) -> bool {
    NAMED_SPECIAL.iter().any(|s| s.as_bytes() == token)
        || (token.len() >= 4 && token.starts_with(b"<|") && token.ends_with(b"|>"))
}

/// The tokens that end generation (`QwenTokenizer::IsStopToken`): the EOS
/// and the vocabulary's `<|endoftext|>`, whichever exist.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StopTokens {
    eos: Option<TokenId>,
    endoftext: Option<TokenId>,
}

impl StopTokens {
    pub fn contains(self, token: TokenId) -> bool {
        Some(token) == self.eos || Some(token) == self.endoftext
    }
}

#[derive(Clone)]
pub struct QwenTokenizer {
    bpe: BpeTokenizer,
    special: Vec<(Vec<u8>, TokenId)>,
    pub eos: Option<TokenId>,
    pub bos: Option<TokenId>,
    pub pad: Option<TokenId>,
    endoftext: Option<TokenId>,
}

impl QwenTokenizer {
    pub fn from_gguf(gguf: &Gguf<'_>) -> Result<Self, String> {
        let tokens = gguf
            .strings("tokenizer.ggml.tokens")
            .filter(|t| !t.is_empty())
            .ok_or("GGUF metadata lacks tokenizer.ggml.tokens")?;
        let merges = gguf.strings("tokenizer.ggml.merges").unwrap_or_default();
        let special: Vec<(Vec<u8>, TokenId)> = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| is_special(t))
            .map(|(id, t)| (t.to_vec(), id as TokenId))
            .collect();
        let find = |name: &str| {
            special
                .iter()
                .find(|(t, _)| t == name.as_bytes())
                .map(|&(_, id)| id)
        };
        let mut bpe = BpeTokenizer::new(tokens, merges, &special)?;
        if gguf.string("tokenizer.ggml.pre") == Some(b"qwen35") {
            bpe = bpe.with_pre_tokenizer(PreTokenizer::Qwen35);
        }
        // The vocabulary names the stop tokens; fixed IDs of one Qwen
        // generation are ordinary tokens in another.
        let endoftext = find("<|endoftext|>");
        let eos = gguf
            .u32("tokenizer.ggml.eos_token_id")
            .or_else(|| find("<|im_end|>"))
            .or(endoftext);
        let bos = gguf
            .u32("tokenizer.ggml.bos_token_id")
            .or_else(|| find("<|im_start|>"));
        let pad = gguf.u32("tokenizer.ggml.padding_token_id").or(endoftext);
        Ok(Self {
            bpe,
            special,
            eos,
            bos,
            pad,
            endoftext,
        })
    }

    pub fn stop_tokens(&self) -> StopTokens {
        StopTokens {
            eos: self.eos,
            endoftext: self.endoftext,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.bpe.vocab_size()
    }

    /// Encodes text; added tokens are recognised anywhere in it.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        self.bpe.encode(text)
    }

    /// Encodes text as text: a spelled added token stays its characters.
    pub fn encode_text(&self, text: &str) -> Vec<TokenId> {
        self.bpe.encode_plain(text)
    }

    /// Whether `text` spells one of the added tokens [`Self::encode`]
    /// recognises.
    pub fn spells_special_token(&self, text: &str) -> bool {
        // Every added token starts with '<': only those positions are checked.
        let bytes = text.as_bytes();
        bytes.iter().enumerate().any(|(at, &byte)| {
            byte == b'<'
                && self
                    .special
                    .iter()
                    .any(|(token, _)| bytes[at..].starts_with(token))
        })
    }

    /// Each token's bytes and whether it stops generation, for grammar
    /// masks (`BuildConstraintVocabulary` over `DecodeTokenCopy` and
    /// `IsStopToken`).
    pub fn constraint_vocabulary(&self) -> crate::text::Result<ConstraintVocabulary> {
        let size = u32::try_from(self.vocab_size())
            .map_err(|_| crate::text::Error::invalid("vocabulary exceeds the token ID range"))?;
        let stops = self.stop_tokens();
        ConstraintVocabulary::new(size, |id| {
            Ok(Piece {
                text: self.bpe.decode_token(id),
                stop: stops.contains(id),
            })
        })
    }

    /// Raw bytes of `ids`; may end inside a UTF-8 sequence.
    pub fn decode(&self, ids: &[TokenId]) -> Vec<u8> {
        ids.iter()
            .flat_map(|&id| self.bpe.decode_token(id))
            .collect()
    }

    /// The IDs of the special tokens, in vocabulary order.
    pub fn special_ids(&self) -> impl Iterator<Item = TokenId> + '_ {
        self.special.iter().map(|&(_, id)| id)
    }

    pub fn special(&self, name: &str) -> Option<TokenId> {
        self.special
            .iter()
            .find(|(t, _)| t == name.as_bytes())
            .map(|&(_, id)| id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::{image, string_array};

    #[test]
    fn special_tokens_and_ids_come_from_the_metadata() {
        let tokens = [
            "a",
            "b",
            "ab",
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "<odd>",
            "<||>",
        ];
        let bytes = image(
            &[
                ("tokenizer.ggml.tokens", 9, string_array(&tokens)),
                ("tokenizer.ggml.merges", 9, string_array(&["a b"])),
                (
                    "tokenizer.ggml.pre",
                    8,
                    [&6u64.to_le_bytes()[..], b"qwen35"].concat(),
                ),
                (
                    "tokenizer.ggml.eos_token_id",
                    4,
                    4u32.to_le_bytes().to_vec(),
                ),
            ],
            &[],
            0,
        );
        let gguf = Gguf::parse(&bytes).unwrap();
        let t = QwenTokenizer::from_gguf(&gguf).unwrap();
        assert_eq!((t.eos, t.bos, t.pad), (Some(4), Some(3), None));
        let stops = t.stop_tokens();
        assert!(stops.contains(4) && !(0..8).any(|id| id != 4 && stops.contains(id)));
        assert_eq!(t.special("<think>"), Some(5));
        assert_eq!(t.special("<odd>"), None);
        assert_eq!(t.special("<||>"), Some(7));
        assert_eq!(t.encode("ab<|im_end|>"), [2, 4]);
        assert_eq!(t.decode(&[2, 5]), b"ab<think>");
    }
}
