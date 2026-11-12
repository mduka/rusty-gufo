//! The sampler for speculative proposals (`proposal_sampler.hpp`).

use gufo_core::sampling::SamplerState;

/// The sampler with top-p disabled, for speculative proposals: rejection
/// sampling is exact for any proposal distribution (`WithoutTopP`). Built
/// from the public sampler state (config, repetition history and draw
/// state), so the copy keeps the request's temperature, top-k, min-p,
/// repetition window and RNG. The public state does not expose the
/// generated-token counts or the reasoning-budget phase, so
/// frequency/presence penalties count the history window only and the copy
/// carries no reasoning budget. Only proposals see the difference; the
/// target's distribution is unchanged.
pub fn without_top_p(sampler: &SamplerState) -> Result<SamplerState, String> {
    let mut config = sampler.config().clone();
    config.top_p = 1.0;
    config.reasoning_budget = None;
    let mut copy = SamplerState::new(config, sampler.history()).map_err(|e| e.to_string())?;
    copy.copy_draw_state_from(sampler);
    Ok(copy)
}
