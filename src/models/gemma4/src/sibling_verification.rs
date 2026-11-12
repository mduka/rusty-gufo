//! Sampled verification of a draft with a sibling
//! (`sibling_verification.hpp`).

use gufo_core::sampling::{Probability, SamplerState, SamplingDistribution, TokenId};

use crate::mtp_sampling::MtpProposal;

/// The verified token, whether it is the chain's draft, and whether it is
/// the sibling, after the draft's rejection (`SiblingVerification`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiblingVerification {
    pub token: TokenId,
    pub accepted: bool,
    pub sibling: bool,
}

/// p/q verification of a draft with a sibling drawn from q without it:
/// recursive rejection sampling (`VerifyWithSibling`). The draft is accepted
/// with min(1, p/q); else the sibling, against the residual
/// r = norm(max(p - q, 0)) and its own proposal q' = q without the draft,
/// with min(1, r/q'); else the token comes from norm(max(r - q', 0)). The
/// emitted token follows p exactly.
pub fn verify_with_sibling(
    target: &SamplingDistribution,
    q: &MtpProposal,
    sibling: TokenId,
    sampler: &mut SamplerState,
) -> Result<SiblingVerification, String> {
    if sampler.uniform() * f64::from(q.probability) < target.probability(q.token) {
        return Ok(SiblingVerification {
            token: q.token,
            accepted: true,
            sibling: false,
        });
    }
    let draft = |token: TokenId| {
        q.ids[..q.size]
            .iter()
            .position(|&id| id == token)
            .map_or(0.0, |i| f64::from(q.probabilities[i]))
    };
    let mut residual = Vec::new();
    let mut total = 0.0f64;
    for entry in target.entries() {
        let value = (entry.value - draft(entry.token)).max(0.0);
        if value > 0.0 {
            residual.push(Probability {
                token: entry.token,
                value,
            });
            total += value;
        }
    }
    let rejected = |token| SiblingVerification {
        token,
        accepted: false,
        sibling: false,
    };
    if !total.is_finite() || total <= 0.0 {
        return target
            .sample(sampler.mutable_rng_state())
            .map(rejected)
            .map_err(|e| e.to_string());
    }
    let rest = 1.0 - f64::from(q.probability);
    let own = draft(sibling) / rest;
    let r = (target.probability(sibling) - draft(sibling)).max(0.0) / total;
    if sampler.uniform() * own < r {
        return Ok(SiblingVerification {
            token: sibling,
            accepted: false,
            sibling: true,
        });
    }
    let mut second = Vec::new();
    for entry in &residual {
        let other = if entry.token == q.token {
            0.0
        } else {
            draft(entry.token) / rest
        };
        let value = (entry.value / total - other).max(0.0);
        if value > 0.0 {
            second.push(Probability {
                token: entry.token,
                value,
            });
        }
    }
    let remaining = if second.is_empty() { residual } else { second };
    SamplingDistribution::new(remaining)
        .and_then(|distribution| distribution.sample(sampler.mutable_rng_state()))
        .map(rejected)
        .map_err(|e| e.to_string())
}
