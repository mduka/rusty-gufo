//! The provider options `extension.json` declares (`provider_options.hpp`,
//! `prefill_weights.hpp`): parsing and validation of their values. Gemma's
//! default applies to every option that was not given.

use crate::vision_prompt::{DEFAULT_SOFT_TOKENS, is_soft_token_budget};

/// Prefill copies of the dense projections, kept beside the GGUF weights
/// that decode reads. Prefill results are bit-identical in every mode
/// (`PrefillWeights`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum PrefillWeights {
    /// GGUF weights only.
    #[default]
    Off,
    /// K-quant weights in the compact prefill layout.
    Compact,
    /// The fastest layout per projection, binary16 where it pays.
    Decoded,
}

impl PrefillWeights {
    /// `--prefill-weights`: off, compact or decoded; empty is off
    /// (`ParsePrefillWeightsOption`).
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "" | "off" => Ok(Self::Off),
            "compact" => Ok(Self::Compact),
            "decoded" => Ok(Self::Decoded),
            _ => Err("--prefill-weights must be off, compact or decoded".into()),
        }
    }
}

/// The soft-token budget per image (`--image-tokens`, default 280); fails
/// for a budget the checkpoint does not support (`ImageTokens`).
pub fn image_tokens(given: Option<i64>) -> Result<u32, String> {
    let Some(tokens) = given else {
        return Ok(DEFAULT_SOFT_TOKENS);
    };
    match u32::try_from(tokens) {
        Ok(tokens) if is_soft_token_budget(tokens) => Ok(tokens),
        _ => Err("--image-tokens must be 70, 140, 280, 560 or 1120".into()),
    }
}
