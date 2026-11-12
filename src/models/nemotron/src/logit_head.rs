//! A logit row summarized on the device (`logit_head.cpp`), so that top-k
//! sampling need not pass over the whole vocabulary.

use gufo_core::sampling::{LogitCandidate, SamplerState, SamplingConfig, TokenId};
use gufo_core::text::Result;

/// What a [`LogitHead`] must hold (`LogitHeadFilter`): with `keep` nonzero,
/// the keep best tokens; otherwise at least `minimum` tokens and `top_p` of
/// the softmax mass at `temperature`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitHeadFilter {
    pub temperature: f32,
    pub keep: usize,
    pub top_p: f32,
    pub minimum: usize,
}

impl Default for LogitHeadFilter {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            keep: 0,
            top_p: 1.0,
            minimum: 1,
        }
    }
}

/// A logit row summarized where it was computed (`LogitHead`): its largest
/// finite logit, the sum over every finite logit of
/// `exp((logit - maximum) / temperature)`, and every finite token whose
/// logit reaches some floor, ties included, in any order. No candidates
/// means no head.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LogitHead {
    pub maximum: f32,
    pub temperature: f32,
    pub softmax_sum: f64,
    pub candidates: Vec<LogitCandidate>,
}

/// The head [`sample_with_head`] can read for `config` (`HeadFilterFor`):
/// top-k sampling at a positive temperature without penalties, keeping top-k
/// raised to min-keep. Other configurations sample from the full row.
pub fn head_filter_for(config: &SamplingConfig) -> Option<LogitHeadFilter> {
    // Gufo's sampler takes a compact distribution only for bounded top-k; it
    // filters top-p and min-p inside the kept tokens, so the head needs no
    // more than them. Penalties can lift a token the head left out.
    // NaN temperatures take the full row too.
    let positive = config.temperature.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater);
    if !positive || config.top_k <= 0 || config.penalties_enabled() {
        return None;
    }
    let minimum = config.min_keep.max(1);
    Some(LogitHeadFilter {
        temperature: config.temperature,
        keep: (config.top_k as usize).max(minimum),
        top_p: config.top_p,
        minimum,
    })
}

/// Draws the token `sampler.sample(logits)` draws and advances the sampler
/// the same way, reading `head` instead of the full row when the sampler's
/// next draw is plain top-k sampling and the head holds candidates
/// (`SampleWithHead`).
pub fn sample_with_head(
    sampler: &mut SamplerState,
    logits: &[f32],
    head: &LogitHead,
) -> Result<TokenId> {
    // A forced or deferred token, a grammar, greedy decoding and penalties
    // all take the full row, as does a head the device could not fill.
    if head.candidates.is_empty()
        || head_filter_for(sampler.config()).is_none()
        || sampler.needs_constraint_mask()?
        || sampler.forced_token().is_some()
        || sampler.save_draw_state().pending.is_some()
    {
        return sampler.sample(logits);
    }
    let distribution = sampler.distribution_from_candidates(&head.candidates)?;
    distribution.sample(sampler.mutable_rng_state())
}
