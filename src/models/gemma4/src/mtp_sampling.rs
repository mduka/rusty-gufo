//! How sampled MTP proposals are drawn and verified (`mtp_sampling.hpp`,
//! which the extension copied from Gufo's Flash-Next at upstream 1c75887b,
//! unchanged since v0.28.1; the functions equal the Flash-Next port's
//! `policy.rs`). Every float operation follows the C++ order, so seeded
//! requests replay its proposals.

use gufo_core::sampling::{self, LogitCandidate, SamplerState, SamplingDistribution, TokenId};

/// The decision on one deterministic draft (`DraftDecision`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftDecision {
    Accept,
    Reject,
    Stop,
}

/// Top drafter logits a draft step reports (`kMtpCandidates`).
pub const MTP_CANDIDATES: usize = 64;

/// The best draft logits, descending with the lowest ID first on ties
/// (`MtpCandidateLogits`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpCandidateLogits {
    pub ids: [TokenId; MTP_CANDIDATES],
    pub logits: [f32; MTP_CANDIDATES],
    pub size: usize,
}

impl Default for MtpCandidateLogits {
    fn default() -> Self {
        Self {
            ids: [0; MTP_CANDIDATES],
            logits: [0.0; MTP_CANDIDATES],
            size: 0,
        }
    }
}

/// A sampled draft token and the compact distribution it was drawn from
/// (`MtpProposal`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpProposal {
    pub ids: [TokenId; MTP_CANDIDATES],
    pub probabilities: [f32; MTP_CANDIDATES],
    pub size: usize,
    pub token: TokenId,
    pub probability: f32,
}

impl Default for MtpProposal {
    fn default() -> Self {
        Self {
            ids: [0; MTP_CANDIDATES],
            probabilities: [0.0; MTP_CANDIDATES],
            size: 0,
            token: 0,
            probability: 0.0,
        }
    }
}

/// Draws a proposal from the candidates. The proposal has bounded support;
/// the target distribution stays complete. Integer masses sum to 2^24, so
/// the exported F32 q sums to exactly one and is exactly the distribution
/// used to draw the proposal (`SampleMtpProposal`).
pub fn sample_mtp_proposal(
    candidates: &MtpCandidateLogits,
    sampler: &SamplerState,
    rng: &mut u64,
) -> Result<MtpProposal, String> {
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
    let mut mass = [0u32; MTP_CANDIDATES];
    let mut proposal = MtpProposal {
        size: distribution.entries().len(),
        ..MtpProposal::default()
    };
    let mut total = 0u32;
    for (i, entry) in distribution.entries().iter().enumerate() {
        proposal.ids[i] = candidates.ids[entry.token as usize];
        mass[i] = (entry.value * f64::from(UNITS)).floor() as u32;
        total = total.wrapping_add(mass[i]);
    }
    mass[0] = mass[0].wrapping_add(UNITS.wrapping_sub(total));
    let draw = (sampling::uniform(rng) * f64::from(UNITS)) as u32;
    let mut cumulative = 0u32;
    let mut selected = None;
    let size = proposal.size;
    for ((probability, &id), &m) in proposal
        .probabilities
        .iter_mut()
        .zip(&proposal.ids)
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
        proposal.token = token;
        proposal.probability = probability;
    }
    Ok(proposal)
}

/// The verified token and whether it is the proposal (`MtpVerification`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpVerification {
    pub token: TokenId,
    pub accepted: bool,
}

/// Rejection sampling with the FP64 target distribution and a residual
/// correction; only q uses the compact F32 masses (`VerifyMtpProposal`).
pub fn verify_mtp_proposal(
    logits: &[f32],
    proposal: &MtpProposal,
    sampler: &mut SamplerState,
) -> Result<MtpVerification, String> {
    verify_mtp_target(logits.len(), proposal, sampler, |sampler| {
        sampler.distribution(logits)
    })
}

/// [`verify_mtp_proposal`] against a compact target row; the same FP64
/// distribution decides (`VerifyMtpProposal` over `LogitCandidate`s).
pub fn verify_mtp_proposal_candidates(
    candidates: &[LogitCandidate],
    vocab: usize,
    proposal: &MtpProposal,
    sampler: &mut SamplerState,
) -> Result<MtpVerification, String> {
    verify_mtp_target(vocab, proposal, sampler, |sampler| {
        sampler.distribution_from_candidates(candidates)
    })
}

/// The same FP64 target distribution drives AR draws, support, acceptance
/// and residual correction (`VerifyMtpTarget`).
pub fn verify_mtp_target(
    vocab: usize,
    proposal: &MtpProposal,
    sampler: &mut SamplerState,
    distribution: impl FnOnce(&SamplerState) -> gufo_core::text::Result<SamplingDistribution>,
) -> Result<MtpVerification, String> {
    if proposal.size == 0
        || proposal.size > MTP_CANDIDATES
        || proposal.token as usize >= vocab
        || !proposal.probability.is_finite()
        || proposal.probability <= 0.0
        || proposal.probability > 1.0
    {
        return Err("invalid MTP proposal".into());
    }
    let (mut total, mut selected) = (0.0f64, 0.0f64);
    for i in 0..proposal.size {
        if proposal.ids[i] as usize >= vocab
            || !proposal.probabilities[i].is_finite()
            || proposal.probabilities[i] < 0.0
        {
            return Err("invalid MTP proposal mass".into());
        }
        total += f64::from(proposal.probabilities[i]);
        if proposal.ids[i] == proposal.token {
            selected += f64::from(proposal.probabilities[i]);
        }
    }
    if total != 1.0 || selected != f64::from(proposal.probability) {
        return Err("MTP proposal masses do not match the draw".into());
    }
    let target = distribution(sampler).map_err(|e| e.to_string())?;
    if sampler.uniform() * f64::from(proposal.probability) < target.probability(proposal.token) {
        return Ok(MtpVerification {
            token: proposal.token,
            accepted: true,
        });
    }
    let token = target
        .sample_residual(
            &proposal.ids[..proposal.size],
            &proposal.probabilities[..proposal.size],
            sampler.mutable_rng_state(),
        )
        .map_err(|e| e.to_string())?;
    Ok(MtpVerification {
        token,
        accepted: false,
    })
}

/// The draft is deterministic, so an exact target sample decides
/// acceptance. A rejected draw is deferred to the next cycle: its RNG is
/// restored, so that stopping at this prefix consumes exactly the draws made
/// by ordinary decode (`VerifyDraft`).
pub fn verify_draft(
    logits: &[f32],
    draft: i32,
    sampler: &mut SamplerState,
    is_stop: impl Fn(i32) -> bool,
) -> Result<DraftDecision, String> {
    let rng = sampler.rng_state();
    let token = sampler.sample(logits).map_err(|e| e.to_string())?;
    if is_stop(token as i32) {
        return Ok(DraftDecision::Stop);
    }
    if token != draft as TokenId {
        sampler.set_rng_state(rng);
        return Ok(DraftDecision::Reject);
    }
    sampler.accept(token).map_err(|e| e.to_string())?;
    Ok(DraftDecision::Accept)
}
