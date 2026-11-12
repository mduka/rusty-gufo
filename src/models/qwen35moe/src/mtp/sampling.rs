//! Sampled MTP proposals, a port of `SampleMtpProposal` (`fnext/mtp_sampling.hpp`):
//! a draft token drawn from the full head's top candidates at the request's
//! temperature, with the exact compact distribution it was drawn from, which
//! the verifier's rejection sampling corrects against.

use gufo_core::sampling::{self, SamplerState, SamplingConfig};
use gufo_qwen35moe_kernels::sizes::MTP_CANDIDATES;

/// The full head's best logits of one step, descending with lowest-ID ties
/// (`MtpCandidateLogits`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidates {
    pub ids: [u32; MTP_CANDIDATES],
    pub logits: [f32; MTP_CANDIDATES],
    pub size: usize,
}

/// A drawn token and the distribution it came from (`MtpProposal`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drawn {
    pub ids: [u32; MTP_CANDIDATES],
    pub probabilities: [f32; MTP_CANDIDATES],
    pub size: usize,
    pub token: u32,
    pub probability: f32,
}

/// The temperature-only sampler of a proposal (`SamplingConfig` with the
/// request's temperature, every other control at its default).
pub(crate) fn proposal_sampler(temperature: f32) -> Result<SamplerState, String> {
    let config = SamplingConfig {
        temperature,
        ..SamplingConfig::default()
    };
    SamplerState::new(config, &[]).map_err(|e| e.to_string())
}

/// Draws from the candidates. Integer masses sum to 2^24, so the exported
/// F32 q sums to exactly one and is exactly the distribution drawn from
/// (`SampleMtpProposal`).
pub fn sample_proposal(
    candidates: &Candidates,
    sampler: &SamplerState,
    rng: &mut u64,
) -> Result<Drawn, String> {
    if candidates.size == 0 || candidates.size > MTP_CANDIDATES {
        return Err("invalid MTP candidate count".into());
    }
    let distribution = sampler
        .distribution_compact(
            &candidates.logits[..candidates.size],
            &candidates.ids[..candidates.size],
        )
        .map_err(|e| e.to_string())?;
    const UNITS: u32 = 1 << 24;
    let entries = distribution.entries();
    let mut mass = [0u32; MTP_CANDIDATES];
    let mut drawn = Drawn {
        ids: [0; MTP_CANDIDATES],
        probabilities: [0.0; MTP_CANDIDATES],
        size: entries.len(),
        token: 0,
        probability: 0.0,
    };
    let mut total = 0u32;
    for (i, entry) in entries.iter().enumerate() {
        drawn.ids[i] = candidates.ids[entry.token as usize];
        mass[i] = (entry.value * f64::from(UNITS)).floor() as u32;
        total = total.wrapping_add(mass[i]);
    }
    mass[0] = mass[0].wrapping_add(UNITS.wrapping_sub(total));
    let draw = (sampling::uniform(rng) * f64::from(UNITS)) as u32;
    let mut cumulative = 0u32;
    let mut selected = None;
    let size = drawn.size;
    for ((probability, &id), &m) in drawn
        .probabilities
        .iter_mut()
        .zip(&drawn.ids)
        .zip(&mass)
        .take(size)
    {
        *probability = m as f32 / UNITS as f32;
        cumulative = cumulative.wrapping_add(m);
        if selected.is_none() && draw < cumulative {
            selected = Some((id, *probability));
        }
    }
    if let Some((token, probability)) = selected {
        drawn.token = token;
        drawn.probability = probability;
    }
    Ok(drawn)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The masses sum to one, the draw is one of the candidates and the
    /// same RNG state draws the same token.
    #[test]
    fn proposals_carry_the_distribution_they_were_drawn_from() {
        let mut candidates = Candidates {
            ids: [0; MTP_CANDIDATES],
            logits: [0.0; MTP_CANDIDATES],
            size: 4,
        };
        candidates.ids[..4].copy_from_slice(&[7, 3, 11, 5]);
        candidates.logits[..4].copy_from_slice(&[4.0, 3.5, 1.0, -2.0]);
        let sampler = proposal_sampler(0.8).unwrap_or_else(|e| panic!("{e}"));
        let (mut a, mut b) = (42u64, 42u64);
        let first =
            sample_proposal(&candidates, &sampler, &mut a).unwrap_or_else(|e| panic!("{e}"));
        let second =
            sample_proposal(&candidates, &sampler, &mut b).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(first, second);
        let sum: f64 = first.probabilities[..first.size]
            .iter()
            .map(|&p| f64::from(p))
            .sum();
        assert_eq!(sum, 1.0);
        assert!(candidates.ids[..4].contains(&first.token));
        assert!(first.probability > 0.0);
    }
}
