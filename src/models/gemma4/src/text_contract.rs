//! The text contract of the Gemma 4 provider apart from its tool dialect and
//! reasoning markers (`text_contract.cpp`): the sampling defaults and the
//! image prefill stops. The tool-call syntax and the reasoning delimiters
//! live with the serving layer.

use gufo_core::gguf::Gguf;
use gufo_core::sampling::{SamplingConfig, SamplingOverrides};

use crate::vision_prompt::Prompt;

/// The generation_config (and GGUF `general.sampling.*`) defaults: the same
/// values with and without thinking (`SamplingDefaults`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
}

pub const SAMPLING_DEFAULTS: SamplingDefaults = SamplingDefaults {
    temperature: 1.0,
    top_k: 64,
    top_p: 0.95,
};

impl SamplingDefaults {
    /// Provider defaults below explicit server/CLI settings (`supplied`,
    /// values in `configured`), for any thinking mode
    /// (`ResolveTextSampling` with provider defaults): a supplied control
    /// keeps `configured`'s value, an unsupplied one takes the default, and
    /// controls without a default start from `SamplingConfig::default()`.
    /// Defaults never change the caller's grammar or reasoning budget; the
    /// seed is never a default.
    pub fn resolve(
        &self,
        configured: &SamplingConfig,
        supplied: &SamplingOverrides,
    ) -> SamplingConfig {
        let base = SamplingConfig::default();
        fn pick<T>(set: bool, value: T, default: T) -> T {
            if set { value } else { default }
        }
        SamplingConfig {
            logprobs: None,
            temperature: pick(
                supplied.temperature,
                configured.temperature,
                self.temperature,
            ),
            top_k: pick(supplied.top_k, configured.top_k, self.top_k),
            top_p: pick(supplied.top_p, configured.top_p, self.top_p),
            min_p: pick(supplied.min_p, configured.min_p, base.min_p),
            min_keep: pick(supplied.min_keep, configured.min_keep, base.min_keep),
            seed: pick(supplied.seed, configured.seed, base.seed),
            repeat_penalty: pick(
                supplied.repeat_penalty,
                configured.repeat_penalty,
                base.repeat_penalty,
            ),
            repeat_last_n: pick(
                supplied.repeat_last_n,
                configured.repeat_last_n,
                base.repeat_last_n,
            ),
            frequency_penalty: pick(
                supplied.frequency_penalty,
                configured.frequency_penalty,
                base.frequency_penalty,
            ),
            presence_penalty: pick(
                supplied.presence_penalty,
                configured.presence_penalty,
                base.presence_penalty,
            ),
            constraint: configured.constraint.clone(),
            reasoning_budget: configured.reasoning_budget.clone(),
        }
    }
}

/// `TextProvider::sampling_defaults`: the same for every Gemma 4 file
/// (`ProbeSamplingDefaults`).
pub fn probe_sampling_defaults(_gguf: &Gguf<'_>) -> Option<SamplingDefaults> {
    Some(SAMPLING_DEFAULTS)
}

/// Where a prefill may stop near `position` (`ImageContext::PrefillStop`):
/// image soft tokens attend each other, so prefill stops before an image or
/// after its closing token. `up` moves a position inside an image past its
/// closing token, otherwise to the image's first soft token.
pub fn prefill_stop(prompt: Option<&Prompt>, position: usize, up: bool) -> usize {
    if let Some(prompt) = prompt {
        for image in &prompt.images {
            let first = image.span.offset as usize;
            let last = first + image.span.rows as usize; // closing token
            if position > first && position <= last {
                return if up { last + 1 } else { first };
            }
        }
    }
    position
}
