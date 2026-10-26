//! How many tokens the draft block proposes per cycle, and how sampled
//! proposals are drawn and verified: ports of `mtp_policy.hpp`,
//! `mtp_costs.hpp` and `mtp_sampling.hpp`. Every float operation follows
//! the C++ order, so seeded requests replay its proposals and lengths.

use gufo_core::sampling::{self, LogitCandidate, SamplerState, SamplingDistribution, TokenId};

use crate::mtp::{MTP_CANDIDATES, MtpCandidates};

/// Longest proposal chain (`kMaxMtpDraftTokens`).
pub const MAX_DRAFT_TOKENS: u32 = 7;
const DEPTHS: usize = MAX_DRAFT_TOKENS as usize;

/// gfx1151, pinned Flash-Next Q4 target and Q8 MTP, 2026-09-20:
/// milliseconds per cohort, including catch-up, recursive proposals and
/// target verification, at contexts 0, 4096 and 32768; rows are the cohorts
/// C1, C2, C4, C6 and C8, columns widths 1..=8 (`kMtpCycleMilliseconds`).
/// C1 at context 0 was remeasured with invariant draft cache projections
///.
#[rustfmt::skip]
const CYCLE_MS: [[[f32; 8]; 5]; 3] = [
    [
        [38.5, 47.3, 56.1, 65.7, 74.1, 82.2, 89.7, 96.8],
        [46.1, 59.6, 74.0, 90.5, 109.0, 121.2, 135.9, 145.8],
        [57.6, 80.6, 108.6, 131.4, 161.9, 183.5, 213.5, 234.3],
        [68.4, 104.7, 140.2, 175.5, 211.6, 254.2, 293.7, 325.8],
        [78.7, 121.9, 167.4, 215.7, 264.7, 311.2, 364.1, 411.1],
    ],
    [
        [39.7, 49.3, 59.8, 69.7, 79.1, 89.1, 98.6, 108.8],
        [49.1, 64.3, 80.5, 97.1, 116.6, 130.9, 145.3, 157.2],
        [62.3, 87.7, 118.6, 142.4, 173.0, 197.9, 228.4, 255.4],
        [75.1, 115.4, 156.5, 193.1, 233.5, 278.9, 320.1, 358.1],
        [87.9, 134.3, 189.7, 238.6, 296.0, 342.8, 402.1, 456.0],
    ],
    [
        [42.5, 51.9, 61.5, 71.1, 82.5, 92.5, 103.6, 112.6],
        [54.9, 69.0, 83.2, 99.6, 120.5, 135.3, 151.4, 163.7],
        [73.6, 96.9, 125.9, 150.1, 185.3, 212.3, 243.9, 266.5],
        [92.0, 128.6, 167.0, 206.3, 249.5, 298.3, 339.1, 370.7],
        [110.1, 152.5, 201.8, 256.0, 321.9, 376.1, 432.0, 490.9],
    ],
];

/// C1 cycles of a sampled request drafting through the small head, same
/// weights, 2026-10-23: fixed token histories at contexts 0, 4096 and 32768,
/// the median of three warmed complete cycles; columns widths 1..=8
/// (`kMtpSampledSubsetMilliseconds`). Full-head and concurrent cycles keep
/// [`CYCLE_MS`]: pricing the smaller head as a full projection shortens
/// profitable draft chains.
#[rustfmt::skip]
const SAMPLED_SUBSET_MS: [[f32; 8]; 3] = [
    [34.4, 42.3, 47.1, 53.1, 59.3, 64.5, 72.0, 78.9],
    [35.1, 42.9, 48.7, 54.8, 60.8, 66.5, 74.3, 81.5],
    [35.2, 42.9, 48.9, 55.1, 60.9, 66.5, 74.4, 81.5],
];

/// The cycle cost per width at `context` for a physical batch of
/// `concurrency` sessions: the next measured cohort, interpolated in the
/// context and extrapolated past 32768 (`MtpCycleCosts`). `sampled_subset`
/// prices a single sampled session on the small head.
pub fn cycle_costs(context: u32, concurrency: u32, sampled_subset: bool) -> [f32; 8] {
    let cohort = match concurrency {
        0..=1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=6 => 3,
        _ => 4,
    };
    let interval = usize::from(context > 4096);
    let fraction = if interval == 0 {
        context as f32 / 4096.0
    } else {
        (context.min(262_144) - 4096) as f32 / 28672.0
    };
    let (low_row, high_row) = if sampled_subset && cohort == 0 {
        (
            &SAMPLED_SUBSET_MS[interval],
            &SAMPLED_SUBSET_MS[interval + 1],
        )
    } else {
        (&CYCLE_MS[interval][cohort], &CYCLE_MS[interval + 1][cohort])
    };
    let mut costs = [0.0f32; 8];
    for i in 0..costs.len() {
        let low = low_row[i];
        let high = high_row[i];
        costs[i] = low + fraction * (high - low).max(0.0);
        if fraction > 1.0 && i != 0 {
            // Independently extrapolated slopes can cross at long contexts:
            // keep at least the last measured marginal cost of a draft.
            let marginal = high - high_row[i - 1];
            costs[i] = costs[i].max(costs[i - 1] + marginal);
        }
    }
    costs
}

/// The occupancy an unplanned draft cycle is priced at, or 0 for the
/// configured session count (`MtpPolicyConcurrency`). Greedy output does not
/// depend on the width, and an unseeded sampled request has nothing to
/// replay, so both price at the requests decoding now. A seeded sampled
/// request keeps the configured count so it replays the same widths, and a
/// planned batch keeps its plan's row.
pub const fn policy_concurrency(
    sampled: bool,
    seeded: bool,
    planned_batch: bool,
    occupancy: u32,
) -> u32 {
    if (sampled && seeded) || planned_batch {
        0
    } else {
        occupancy
    }
}

/// The acceptance history a snapshot carries (`MtpLengthState`, 72 bytes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LengthState {
    pub successes: [f32; DEPTHS],
    pub failures: [f32; DEPTHS],
    pub retry_tokens: u32,
    pub probe_depth: u32,
    pub explored_depth: u32,
    pub probe_delay: u32,
    pub failed_depths: u32,
}

impl LengthState {
    pub const BYTES: usize = 2 * DEPTHS * 4 + 5 * 4;

    pub fn write(&self, out: &mut [u8]) {
        let mut at = 0;
        let mut put = |bytes: [u8; 4]| {
            out[at..at + 4].copy_from_slice(&bytes);
            at += 4;
        };
        for v in self.successes.iter().chain(&self.failures) {
            put(v.to_le_bytes());
        }
        for v in [
            self.retry_tokens,
            self.probe_depth,
            self.explored_depth,
            self.probe_delay,
            self.failed_depths,
        ] {
            put(v.to_le_bytes());
        }
    }

    pub fn read(bytes: &[u8]) -> Self {
        let word =
            |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap_or_default());
        let mut state = Self {
            successes: [0.0; DEPTHS],
            failures: [0.0; DEPTHS],
            retry_tokens: word(2 * DEPTHS),
            probe_depth: word(2 * DEPTHS + 1),
            explored_depth: word(2 * DEPTHS + 2),
            probe_delay: word(2 * DEPTHS + 3),
            failed_depths: word(2 * DEPTHS + 4),
        };
        for d in 0..DEPTHS {
            state.successes[d] = f32::from_bits(word(d));
            state.failures[d] = f32::from_bits(word(DEPTHS + d));
        }
        state
    }
}

/// Proposal length from committed acceptance history, never timings. Only
/// the first rejected draft is an observed failure; its unverified tail and
/// a fully accepted, budget-limited chain are censored
/// (`MtpLengthController`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LengthController {
    limit: u32,
    concurrency: u32,
    state: LengthState,
}

impl LengthController {
    pub fn new(limit: u32, concurrency: u32) -> Self {
        let mut controller = Self {
            limit: limit.clamp(1, MAX_DRAFT_TOKENS),
            concurrency: concurrency.clamp(1, 8),
            state: LengthState {
                successes: [0.0; DEPTHS],
                failures: [0.0; DEPTHS],
                retry_tokens: 0,
                probe_depth: 0,
                explored_depth: 0,
                probe_delay: 0,
                failed_depths: 0,
            },
        };
        controller.reset();
        controller
    }

    /// `concurrency` prices this cycle at another occupancy than the
    /// configured one; 0 keeps the configured value ([`policy_concurrency`]).
    /// `sampled_subset` prices a sampled cycle on the small head.
    pub fn choose(&self, budget: u32, context: u32, concurrency: u32, sampled_subset: bool) -> u32 {
        let cap = budget.min(self.limit);
        if cap == 0 || self.state.retry_tokens != 0 {
            return 0;
        }
        // The first cycle keeps the conservative width until acceptance is
        // observed: the smaller head's curve would otherwise widen short
        // replies from the prior alone, paying for a new verifier shape
        // before it can help.
        let sampled_subset = sampled_subset && self.state.explored_depth != 0;
        // After a bounded AR interval, one proposal detects recovery.
        cap.min(
            1.max(self.best_length(cap, context, concurrency, sampled_subset))
                .max(self.state.probe_depth),
        )
    }

    pub fn expected_tokens(&self, drafts: u32) -> f32 {
        let (mut expected, mut survival) = (1.0f32, 1.0f32);
        for depth in 0..drafts.min(self.limit) {
            survival *= self.probability(depth);
            expected += survival;
        }
        expected
    }

    pub fn observe(
        &mut self,
        accepted: usize,
        drafted: usize,
        context: u32,
        concurrency: u32,
        sampled_subset: bool,
    ) {
        if drafted == 0 {
            return;
        }
        let drafted = drafted.min(self.limit as usize);
        let accepted = accepted.min(drafted);
        let observed = accepted + usize::from(accepted < drafted);
        let s = &mut self.state;
        for depth in 0..observed {
            s.successes[depth] =
                0.75 * s.successes[depth] + if depth < accepted { 1.0 } else { 0.0 };
            s.failures[depth] =
                0.75 * s.failures[depth] + if depth == accepted { 1.0 } else { 0.0 };
        }
        // A fully accepted chain supplies no observations for its tail:
        // probe farther rather than keeping the prior there.
        s.probe_depth = 0;
        if accepted == drafted && drafted < self.limit as usize {
            if drafted as u32 >= s.explored_depth || s.probe_delay == 0 {
                s.probe_depth = (2 * drafted as u32).min(self.limit);
                s.probe_delay = 16;
            } else {
                s.probe_delay -= 1;
            }
        }
        s.explored_depth = s.explored_depth.max(observed as u32);
        if accepted < drafted {
            s.failed_depths |= 1 << accepted;
        }
        if self.best_length(self.limit, context, concurrency, sampled_subset) == 0 {
            self.state.retry_tokens = 16;
        }
    }

    pub fn observe_ar_token(&mut self) {
        self.state.retry_tokens = self.state.retry_tokens.saturating_sub(1);
    }

    pub fn reset(&mut self) {
        let s = &mut self.state;
        s.successes = [1.5; DEPTHS];
        s.failures = [0.5; DEPTHS];
        s.retry_tokens = 0;
        s.probe_depth = 0;
        s.explored_depth = 0;
        s.probe_delay = 0;
        s.failed_depths = 0;
    }

    pub fn state(&self) -> LengthState {
        self.state
    }

    /// Adopts a snapshot's history if it is valid for this limit.
    pub fn restore(&mut self, state: &LengthState) -> bool {
        if state.retry_tokens > 16
            || state.probe_depth > self.limit
            || state.explored_depth > self.limit
            || state.probe_delay > 16
            || state.failed_depths >= (1 << self.limit)
        {
            return false;
        }
        for depth in 0..DEPTHS {
            let (success, failure) = (state.successes[depth], state.failures[depth]);
            if !success.is_finite()
                || !failure.is_finite()
                || success < 0.0
                || failure < 0.0
                || !(success + failure).is_finite()
                || success + failure <= 0.0
            {
                return false;
            }
        }
        self.state = *state;
        true
    }

    fn probability(&self, depth: u32) -> f32 {
        let s = &self.state;
        if depth < s.explored_depth && s.failed_depths & (1 << depth) == 0 {
            1.0
        } else {
            let d = depth as usize;
            s.successes[d] / (s.successes[d] + s.failures[d])
        }
    }

    fn best_length(&self, cap: u32, context: u32, concurrency: u32, sampled_subset: bool) -> u32 {
        let concurrency = if concurrency == 0 {
            self.concurrency
        } else {
            concurrency
        };
        let costs = cycle_costs(context, concurrency, sampled_subset);
        let mut survival = 1.0f32;
        let mut expected_tokens = 1.0f32; // the sampled target anchor
        let mut best_score = 1.0f32; // ordinary decoding
        let mut best = 0;
        for length in 1..=cap {
            // An explored depth without a rejection stays optimistic;
            // otherwise the prior alone would shorten perfect chains.
            survival *= self.probability(length - 1);
            expected_tokens += survival;
            let cost = costs[length as usize] / costs[0];
            let score = expected_tokens / cost;
            if score > best_score {
                best_score = score;
                best = length;
            }
        }
        best
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Cell {
    ms: [f32; 8],
    samples: [u32; 8],
    cycles: u64,
    last_drafts: u32,
}

/// Proposal width of a physical batch of greedy sessions from measured
/// cycle costs per occupancy and context bin (`MtpBatchController`).
/// Greedy output does not depend on the width; sampled requests keep the
/// per-session policy for seeded replay.
#[derive(Debug, Clone, Default)]
pub struct BatchController {
    cells: [[Cell; 8]; 8],
    last_concurrency: usize,
    last_context_bin: usize,
    stable_cycles: u32,
}

/// One session of a batch: its acceptance history and budget.
pub struct BatchRow<'a> {
    pub acceptance: &'a LengthController,
    pub budget: u32,
}

impl BatchController {
    pub fn choose(&self, rows: &[BatchRow<'_>], context: u32) -> u32 {
        if rows.is_empty() || rows.len() > 8 {
            return 0;
        }
        let cell = self.cell(rows.len(), context);
        let cap = rows
            .iter()
            .map(|r| r.budget)
            .fold(MAX_DRAFT_TOKENS, u32::min);
        if cap == 0 {
            return 0;
        }
        // Periodic headless cycles measure the actual AR alternative,
        // predictor catch-up included.
        let stable = self.last_concurrency == rows.len()
            && self.last_context_bin == context_bin(context)
            && self.stable_cycles >= 2;
        if stable && (cell.samples[0] == 0 || cell.cycles % 32 < 2) {
            return 0;
        }
        // Copies may leave every neural cost unmeasured: qualify the
        // cheapest width before widening. Transitions include the previous
        // catch-up, and the first usable sample can still contain lazy
        // allocation.
        if stable && cell.samples[0] != 0 && cell.samples[1] < 2 {
            return 1;
        }
        let mut costs = cycle_costs(context, rows.len() as u32, false);
        let scale = if cell.samples[0] != 0 {
            cell.ms[0] / costs[0]
        } else {
            1.0
        };
        for (width, cost) in costs.iter_mut().enumerate().take(cap as usize + 1) {
            *cost = if cell.samples[width] >= 2 {
                cell.ms[width]
            } else {
                *cost * scale
            };
        }
        let mut best_score = rows.len() as f32 / costs[0];
        let mut best = 0u32;
        for width in 1..=cap {
            let mut expected = 0.0f32;
            for row in rows {
                expected += row.acceptance.expected_tokens(width);
            }
            let score = expected / costs[width as usize];
            if score > best_score * 1.01 {
                best_score = score;
                best = width;
            }
        }
        // Bounded exploration learns wider costs and lets an AR-only batch
        // notice a change in the workload.
        if cell.cycles % 16 == 2 {
            return cap.min(1.max(best + 1));
        }
        // Observe excludes the transition's catch-up: a new probe holds for
        // two usable samples so its measured cost informs the next choice.
        let probe_phase = cell.cycles % 16;
        if stable
            && (probe_phase == 3 || probe_phase == 4)
            && cell.last_drafts != 0
            && cell.samples[cell.last_drafts as usize] < 2
        {
            return cap.min(cell.last_drafts);
        }
        best
    }

    /// Lookup can widen verification and shorten neural drafting: neither
    /// that cycle nor its following catch-up measures a neural width's cost
    /// (`SkipCycle`).
    pub fn skip_cycle(&mut self) {
        self.last_concurrency = 0;
        self.stable_cycles = 0;
    }

    pub fn observe(&mut self, concurrency: usize, context: u32, drafts: u32, ms: f32) {
        if concurrency == 0
            || concurrency > 8
            || drafts > MAX_DRAFT_TOKENS
            || !ms.is_finite()
            || ms <= 0.0
        {
            return;
        }
        let bin = context_bin(context);
        let same_cohort = self.last_concurrency == concurrency && self.last_context_bin == bin;
        self.stable_cycles = if same_cohort {
            (self.stable_cycles + 1).min(1_000_000)
        } else {
            1
        };
        self.last_concurrency = concurrency;
        self.last_context_bin = bin;
        let cell = &mut self.cells[concurrency - 1][bin];
        // Catch-up belongs to the preceding chain: a change of width cannot
        // measure the steady cost of the new one.
        if !same_cohort || drafts != cell.last_drafts {
            cell.last_drafts = drafts;
            cell.cycles += 1;
            return;
        }
        cell.last_drafts = drafts;
        let d = drafts as usize;
        // Drop first-use graph and allocation costs once warmed.
        if cell.samples[d] < 2 {
            cell.ms[d] = ms;
        } else {
            cell.ms[d] += 0.25 * (ms - cell.ms[d]);
        }
        cell.samples[d] = (cell.samples[d] + 1).min(1_000_000);
        cell.cycles += 1;
    }

    fn cell(&self, concurrency: usize, context: u32) -> &Cell {
        &self.cells[concurrency - 1][context_bin(context)]
    }
}

fn context_bin(context: u32) -> usize {
    let mut bin = 0;
    let mut bound = 4096u64;
    while u64::from(context) >= bound && bin < 7 {
        bin += 1;
        bound *= 2;
    }
    bin
}

/// A sampled draft token and the compact distribution it was drawn from
/// (`MtpProposal`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Proposal {
    pub ids: [TokenId; MTP_CANDIDATES],
    pub probabilities: [f32; MTP_CANDIDATES],
    pub size: usize,
    pub token: TokenId,
    pub probability: f32,
}

/// Draws a proposal from the candidates. Integer masses sum to 2^24, so the
/// exported F32 q sums to exactly one and is the distribution drawn from
/// (`SampleMtpProposal`).
pub fn sample_proposal(
    candidates: &MtpCandidates,
    sampler: &SamplerState,
    rng: &mut u64,
) -> Result<Proposal, String> {
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
    let mut proposal = Proposal {
        ids: [0; MTP_CANDIDATES],
        probabilities: [0.0; MTP_CANDIDATES],
        size: distribution.entries().len(),
        token: 0,
        probability: 0.0,
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
pub struct Verification {
    pub token: TokenId,
    pub accepted: bool,
}

/// Rejection sampling with the FP64 target distribution and a residual
/// correction; only q uses the compact F32 masses (`VerifyMtpProposal`).
pub fn verify_proposal(
    logits: &[f32],
    proposal: &Proposal,
    sampler: &mut SamplerState,
) -> Result<Verification, String> {
    verify_target(logits.len(), proposal, sampler, |sampler| {
        sampler.distribution(logits)
    })
}

/// [`verify_proposal`] against a compact target row, the candidate union
/// of `Executor::sampled_mtp_candidates`; the same FP64 distribution
/// decides (`VerifyMtpProposal` over `LogitCandidate`s).
pub fn verify_proposal_candidates(
    candidates: &[LogitCandidate],
    vocab: usize,
    proposal: &Proposal,
    sampler: &mut SamplerState,
) -> Result<Verification, String> {
    verify_target(vocab, proposal, sampler, |sampler| {
        sampler.distribution_from_candidates(candidates)
    })
}

/// The same FP64 target distribution drives acceptance and the residual
/// correction (`VerifyMtpTarget`).
fn verify_target(
    vocab: usize,
    proposal: &Proposal,
    sampler: &mut SamplerState,
    distribution: impl FnOnce(&SamplerState) -> gufo_core::text::Result<SamplingDistribution>,
) -> Result<Verification, String> {
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
        return Ok(Verification {
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
    Ok(Verification {
        token,
        accepted: false,
    })
}

/// The decision on one deterministic draft (`DraftDecision`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftDecision {
    Accept,
    Reject,
    Stop,
}

/// An exact target sample decides a deterministic draft. A rejected draw is
/// deferred to the next cycle: its RNG is restored, so that stopping at this
/// prefix consumes exactly ordinary decoding's draws (`VerifyDraft`).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn costs_interpolate_and_extrapolate_like_the_table() {
        assert_eq!(cycle_costs(0, 1, false), CYCLE_MS[0][0]);
        assert_eq!(cycle_costs(4096, 8, false), CYCLE_MS[1][4]);
        let far = cycle_costs(262_144, 1, false);
        for w in 1..8 {
            assert!(far[w] > far[w - 1]);
        }
    }

    #[test]
    fn perfect_chains_probe_deeper_and_failures_back_off() {
        let mut c = LengthController::new(7, 1);
        let first = c.choose(7, 0, 0, false);
        assert!(first >= 1);
        c.observe(first as usize, first as usize, 0, 0, false);
        assert!(c.choose(7, 0, 0, false) >= first);
        for _ in 0..8 {
            c.observe(0, 3, 0, 0, false);
        }
        assert_eq!(c.state().retry_tokens, 16);
        assert_eq!(c.choose(7, 0, 0, false), 0);
        let state = c.state();
        let mut bytes = [0u8; LengthState::BYTES];
        state.write(&mut bytes);
        assert_eq!(LengthState::read(&bytes), state);
        let mut other = LengthController::new(7, 1);
        assert!(other.restore(&state));
        assert_eq!(other, c);
    }

    // The checks of `tests/models/qwen38_flash_next/mtp_sampling_test.cpp`.

    use gufo_core::sampling::SamplingConfig;

    /// `QwenSamplingCases()` (`tests/models/qwen27b/sampling_cases.hpp`).
    fn sampling_cases() -> Vec<SamplingConfig> {
        let t = |temperature| SamplingConfig {
            temperature,
            ..SamplingConfig::default()
        };
        let hot = t(0.8);
        let mut cases = vec![
            SamplingConfig::default(),
            t(0.05),
            t(1.0),
            t(2.0),
            SamplingConfig {
                top_k: 1,
                ..hot.clone()
            },
            SamplingConfig {
                top_k: 3,
                ..hot.clone()
            },
            SamplingConfig {
                top_p: 0.7,
                ..hot.clone()
            },
            SamplingConfig {
                top_p: 0.0,
                ..hot.clone()
            },
            SamplingConfig {
                top_p: 0.0,
                min_keep: 3,
                ..hot.clone()
            },
            SamplingConfig {
                min_p: 0.3,
                ..hot.clone()
            },
            SamplingConfig {
                min_p: 1.0,
                ..hot.clone()
            },
            SamplingConfig {
                top_k: 1,
                min_keep: 3,
                ..hot.clone()
            },
            SamplingConfig {
                top_p: 0.01,
                min_keep: 3,
                ..hot.clone()
            },
            SamplingConfig {
                min_p: 1.0,
                min_keep: 3,
                ..hot.clone()
            },
            SamplingConfig {
                repeat_penalty: 1.5,
                ..hot.clone()
            },
            SamplingConfig {
                repeat_penalty: 0.7,
                ..hot.clone()
            },
            SamplingConfig {
                frequency_penalty: 0.4,
                ..hot.clone()
            },
            SamplingConfig {
                frequency_penalty: -0.4,
                ..hot.clone()
            },
            SamplingConfig {
                presence_penalty: 0.4,
                ..hot.clone()
            },
            SamplingConfig {
                presence_penalty: -0.4,
                ..hot.clone()
            },
            SamplingConfig {
                repeat_penalty: 1.5,
                repeat_last_n: 1,
                ..hot.clone()
            },
            SamplingConfig {
                repeat_penalty: 1.5,
                repeat_last_n: 0,
                frequency_penalty: 0.4,
                presence_penalty: 0.4,
                ..hot.clone()
            },
            SamplingConfig {
                repeat_penalty: 1.5,
                repeat_last_n: 3,
                frequency_penalty: 0.4,
                presence_penalty: -0.4,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                top_k: 4,
                top_p: 0.7,
                min_p: 0.3,
                min_keep: 2,
                repeat_penalty: 1.2,
                repeat_last_n: 3,
                frequency_penalty: 0.2,
                presence_penalty: -0.1,
                ..hot.clone()
            },
            SamplingConfig {
                top_k: 1,
                top_p: 0.01,
                min_p: 1.0,
                min_keep: 3,
                ..hot.clone()
            },
        ];
        for case in &mut cases {
            case.seed = 73;
        }
        cases[2].seed = 0;
        cases[3].seed = 808;
        cases
    }

    fn candidates() -> MtpCandidates {
        let mut c = MtpCandidates {
            size: 3,
            ..MtpCandidates::default()
        };
        c.ids[..3].copy_from_slice(&[3, 1, 4]);
        c.logits[..3].copy_from_slice(&[1.2, 0.1, -0.7]);
        c
    }

    fn zero_state() -> LengthState {
        LengthState {
            successes: [0.0; DEPTHS],
            failures: [0.0; DEPTHS],
            retry_tokens: 0,
            probe_depth: 0,
            explored_depth: 0,
            probe_delay: 0,
            failed_depths: 0,
        }
    }

    #[test]
    fn the_length_controller_grows_backs_off_and_replays() {
        let mut controller = LengthController::new(7, 1);
        let initial = controller.choose(7, 0, 0, false);
        assert_eq!(
            controller.choose(0, 0, 0, false),
            0,
            "empty draft budget was ignored"
        );
        assert_eq!(
            controller.choose(1, 0, 0, false),
            1,
            "one-draft budget was ignored"
        );
        controller.observe(0, 0, 0, 0, false);
        assert_eq!(
            controller.choose(7, 0, 0, false),
            initial,
            "empty cycle changed policy"
        );
        for _ in 0..12 {
            let length = controller.choose(7, 0, 0, false) as usize;
            controller.observe(length, length, 0, 0, false);
        }
        assert_eq!(
            controller.choose(7, 0, 0, false),
            7,
            "accepted prefixes did not grow to the maximum"
        );
        assert_eq!(
            controller.choose(2, 0, 0, false),
            2,
            "remaining output budget was ignored"
        );
        for _ in 0..12 {
            let length = controller.choose(7, 0, 0, false) as usize;
            controller.observe(0, length, 0, 0, false);
        }
        assert_eq!(
            controller.choose(7, 0, 0, false),
            0,
            "unprofitable speculation did not back off to AR"
        );
        let mut backed_off = LengthController::new(7, 1);
        assert!(
            backed_off.restore(&controller.state()),
            "backoff restore failed"
        );
        for _ in 0..16 {
            assert!(
                controller.choose(7, 0, 0, false) == 0 && backed_off.choose(7, 0, 0, false) == 0,
                "AR interval ended early"
            );
            controller.observe_ar_token();
            backed_off.observe_ar_token();
        }
        assert!(
            controller.choose(7, 0, 0, false) == 1 && backed_off.choose(7, 0, 0, false) == 1,
            "AR backoff did not retry a proposal"
        );
        controller.reset();
        assert_eq!(
            controller.choose(7, 0, 0, false),
            initial,
            "reset retained old acceptance"
        );
        controller.observe(5, 6, 0, 0, false);
        let mut restored = LengthController::new(7, 1);
        assert!(
            restored.restore(&controller.state()),
            "controller restore failed"
        );
        for budget in 0..=8 {
            assert_eq!(
                restored.choose(budget, 0, 0, false),
                controller.choose(budget, 0, 0, false),
                "restored policy changes widths"
            );
        }
        let mut invalid = controller.state();
        invalid.successes[0] = -1.0;
        assert!(
            !restored.restore(&invalid) && !restored.restore(&zero_state()),
            "invalid controller state accepted"
        );
        // The unverified suffix is not additional evidence of failure.
        let (mut short_tail, mut long_tail) =
            (LengthController::new(7, 1), LengthController::new(7, 1));
        for accepted in [2usize, 0, 1, 2, 1, 0] {
            short_tail.observe(accepted, accepted + 1, 0, 0, false);
            long_tail.observe(accepted, 7, 0, 0, false);
            for budget in 0..=9 {
                assert_eq!(
                    short_tail.choose(budget, 0, 0, false),
                    long_tail.choose(budget, 0, 0, false),
                    "unverified suffix biased the acceptance estimate"
                );
            }
        }
        for limit in 1..=7 {
            let (mut a, mut b) = (
                LengthController::new(limit, 1),
                LengthController::new(limit, 1),
            );
            for cycle in 0..32u32 {
                let budget = cycle % 10;
                let length = a.choose(budget, 0, 0, false);
                assert!(
                    length == b.choose(budget, 0, 0, false) && length <= budget && length <= limit,
                    "controller violated a limit or changed on replay"
                );
                let accepted = if length == 0 { 0 } else { cycle % (length + 1) } as usize;
                a.observe(accepted, length as usize, 0, 0, false);
                b.observe(accepted, length as usize, 0, 0, false);
                if length == 0 {
                    a.observe_ar_token();
                    b.observe_ar_token();
                }
            }
        }
        let (mut early, mut late) = (LengthController::new(7, 1), LengthController::new(7, 1));
        for _ in 0..12 {
            early.observe(0, 7, 0, 0, false);
            late.observe(4, 7, 0, 0, false);
        }
        assert!(
            early.choose(7, 0, 0, false) == 0 && late.choose(7, 0, 0, false) > 1,
            "conditional acceptance by depth was collapsed to one probability"
        );
    }

    #[test]
    fn calibrated_costs_shorten_concurrent_chains() {
        let (mut single, mut concurrent) =
            (LengthController::new(7, 1), LengthController::new(7, 8));
        let mut state = single.state();
        state.successes = [6.0; DEPTHS];
        state.failures = [4.0; DEPTHS];
        state.explored_depth = 7;
        state.failed_depths = 127;
        assert!(
            single.restore(&state) && concurrent.restore(&state),
            "calibrated state rejected"
        );
        assert!(
            single.choose(7, 0, 0, false) > concurrent.choose(7, 0, 0, false),
            "concurrent verification cost did not shorten the draft chain"
        );
        single.observe(0, 1, 0, 0, false);
        concurrent.observe(0, 1, 0, 0, false);
        assert!(
            single.choose(7, 0, 0, false) > 0 && concurrent.choose(7, 0, 0, false) == 0,
            "measured verification cost did not change the AR break-even point"
        );
        for c in 1..=8 {
            let mut perfect = LengthController::new(7, c);
            for _ in 0..8 {
                let length = perfect.choose(7, 0, 0, false) as usize;
                perfect.observe(length, length, 0, 0, false);
            }
            assert_eq!(
                perfect.choose(7, 0, 0, false),
                7,
                "perfect acceptance lost the fastest width"
            );
            for depth in [0, 4096, 32_768, 131_072, 262_144] {
                let costs = cycle_costs(depth, c, false);
                assert!(costs[0] > 0.0, "AR baseline cost is not positive");
                for i in 1..costs.len() {
                    assert!(
                        costs[i].is_finite() && costs[i] > costs[i - 1],
                        "costs do not increase with verification work"
                    );
                }
            }
        }
    }

    /// The sampled small-head curve: the first cycle keeps its width before
    /// any acceptance, cheaper proposals widen a profitable chain, the curve
    /// keeps its relative saving and only replaces C1, and restoring after a
    /// head fallback replays the same bounded decisions.
    /// Greedy and unseeded sampled cycles price at the live occupancy;
    /// seeded sampled cycles and planned batches keep the configured row.
    #[test]
    fn policy_concurrency_follows_replay_needs() {
        assert!(
            policy_concurrency(false, false, false, 1) == 1
                && policy_concurrency(false, true, false, 3) == 3,
            "an unplanned greedy cycle did not price at its occupancy"
        );
        assert_eq!(
            policy_concurrency(true, false, false, 1),
            1,
            "an unseeded sampled cycle did not price at its occupancy"
        );
        assert_eq!(
            policy_concurrency(true, true, false, 1),
            0,
            "a seeded sampled cycle did not keep the configured row"
        );
        assert!(
            policy_concurrency(false, false, true, 2) == 0
                && policy_concurrency(true, false, true, 2) == 0,
            "a planned batch did not keep its plan's row"
        );
    }

    #[test]
    fn sampled_subset_costs() {
        let mut single = LengthController::new(7, 1);
        let mut configured_8 = LengthController::new(7, 8);
        assert_eq!(
            single.choose(7, 4096, 0, true),
            single.choose(7, 4096, 0, false),
            "sampled prior widened the first cycle before any acceptance"
        );
        let mut state = single.state();
        state.successes = [4.0; DEPTHS];
        state.failures = [1.0; DEPTHS];
        state.explored_depth = 7;
        state.failed_depths = 127;
        assert!(
            single.restore(&state) && configured_8.restore(&state),
            "sampled-head controller state rejected"
        );
        assert!(
            single.choose(7, 0, 0, true) > single.choose(7, 0, 0, false),
            "cheaper sampled proposals did not widen a profitable chain"
        );
        for context in [0, 2048, 4096, 32_768, 131_072, 262_144] {
            let full = cycle_costs(context, 1, false);
            let subset = cycle_costs(context, 1, true);
            for width in 1..subset.len() {
                assert!(
                    subset[width].is_finite() && subset[width] > subset[width - 1],
                    "sampled-head extrapolation crosses verification widths"
                );
                if context <= 32_768 && width >= 2 {
                    assert!(
                        subset[width] / subset[0] < full[width] / full[0],
                        "sampled-head cost lost its measured relative saving"
                    );
                }
            }
            for c in 2..=8 {
                assert_eq!(
                    cycle_costs(context, c, true),
                    cycle_costs(context, c, false),
                    "C1 sampled-head calibration changed a concurrent cost"
                );
            }
            for budget in 0..=8 {
                assert_eq!(
                    configured_8.choose(budget, context, 0, true),
                    configured_8.choose(budget, context, 0, false),
                    "seeded concurrency used lone-decoder costs"
                );
                assert_eq!(
                    configured_8.choose(budget, context, 1, true),
                    single.choose(budget, context, 0, true),
                    "lone unseeded decoder did not use the sampled-head curve"
                );
            }
        }
        // Restoring after a head fallback replays the same decisions; neither
        // switching heads nor changing the budget adds RNG state.
        let mut replay = LengthController::new(7, 1);
        for cycle in 0..64u32 {
            assert!(
                replay.restore(&single.state()),
                "sampled-head restore failed"
            );
            let subset = cycle % 3 != 0;
            let budget = cycle % 9;
            let context = cycle * 2048;
            let length = single.choose(budget, context, 0, subset);
            assert!(
                length == replay.choose(budget, context, 0, subset)
                    && length <= budget
                    && length <= 7,
                "head fallback or snapshot restore changed bounded replay"
            );
            let accepted = if length == 0 { 0 } else { cycle % (length + 1) };
            single.observe(accepted as usize, length as usize, context, 0, subset);
            replay.observe(accepted as usize, length as usize, context, 0, subset);
            if length == 0 {
                single.observe_ar_token();
                replay.observe_ar_token();
            }
            assert_eq!(
                single.choose(7, context, 0, subset),
                replay.choose(7, context, 0, subset),
                "restored acceptance history changed the next sampled width"
            );
        }
    }

    #[test]
    fn the_batch_controller_follows_measured_cycles() {
        let (mut a, mut b) = (LengthController::new(7, 1), LengthController::new(7, 1));
        for _ in 0..12 {
            a.observe(7, 7, 0, 0, false);
            b.observe(7, 7, 0, 0, false);
        }
        let mut policy = BatchController::default();
        let rows = |budget: u32| {
            [
                BatchRow {
                    acceptance: &a,
                    budget: 7,
                },
                BatchRow {
                    acceptance: &b,
                    budget,
                },
            ]
        };
        assert!(
            policy.choose(&rows(7), 0) > 0,
            "transient occupancy paid an unnecessary plain control"
        );
        for _ in 0..2 {
            let width = policy.choose(&rows(7), 0);
            policy.observe(2, 0, width, 100.0);
        }
        assert_eq!(
            policy.choose(&rows(7), 0),
            0,
            "stable occupancy omitted its plain control"
        );
        // Perfect proposals are unprofitable when measured cycles are too slow.
        for w in 0..=7 {
            for _ in 0..3 {
                policy.observe(2, 0, w, if w == 0 { 50.0 } else { 1000.0 });
            }
        }
        assert_eq!(
            policy.choose(&rows(7), 0),
            0,
            "perfect acceptance overrode measured cost"
        );
        policy.observe(2, 0, 0, 1000.0); // previous speculative catch-up debt
        policy.observe(2, 0, 0, 50.0);
        assert_eq!(
            policy.choose(&rows(7), 0),
            0,
            "catch-up debt inflated the plain baseline"
        );
        // Context and occupancy samples stay apart.
        assert!(
            policy.choose(&rows(7), 32_768) > 0,
            "deep batch inherited shallow timings"
        );
        for _ in 0..32 {
            policy.observe(2, 0, 3, 60.0);
        }
        assert_eq!(
            policy.choose(&rows(7), 0),
            3,
            "batch did not select the profitable width"
        );
        policy.observe(2, 0, 7, 500.0);
        policy.observe(2, 0, 3, 2000.0);
        assert_eq!(
            policy.choose(&rows(7), 0),
            3,
            "a width transition charged the new width"
        );
        assert!(
            policy.choose(&rows(1), 0) <= 1,
            "batch exceeded one member's budget"
        );
        policy.observe(2, 0, 0, f32::NAN);
        assert!(
            policy.choose(&rows(1), 0) <= 1,
            "invalid timing corrupted the policy"
        );
    }

    #[test]
    fn compact_proposals_are_exact_distributions() {
        let target = [0.3f32, -0.2, 1.1, 0.7, -1.0];
        let history = [2u32, 3, 2, 0];
        let c = candidates();
        for config in sampling_cases() {
            let sampler = SamplerState::new(config, &history).unwrap();
            let p = sampler.distribution(&target).unwrap();
            for seed in 0..16u64 {
                let (mut rng, mut replay_rng) = (seed, seed);
                let q = sample_proposal(&c, &sampler, &mut rng).unwrap();
                let replay = sample_proposal(&c, &sampler, &mut replay_rng).unwrap();
                assert!(
                    q == replay && rng == replay_rng,
                    "compact proposal is not reproducible"
                );
                let (mut sum, mut draft) = (0.0f64, [0.0f64; 5]);
                for i in 0..q.size {
                    let mass = f64::from(q.probabilities[i]) * f64::from(1u32 << 24);
                    assert!(
                        mass >= 0.0 && mass == mass.floor(),
                        "proposal mass is not discrete"
                    );
                    draft[q.ids[i] as usize] += f64::from(q.probabilities[i]);
                    sum += f64::from(q.probabilities[i]);
                }
                assert!(
                    sum == 1.0
                        && q.probability > 0.0
                        && draft[q.token as usize] == f64::from(q.probability),
                    "sampled q differs from the exported distribution"
                );
                let (mut accepted, mut residual) = (0.0, 0.0);
                for (i, &d) in draft.iter().enumerate() {
                    let probability = p.probability(i as u32);
                    accepted += probability.min(d);
                    residual += (probability - d).max(0.0);
                }
                assert!(
                    (accepted + residual - 1.0).abs() < 1e-12,
                    "mass is not conserved"
                );
                for (i, &d) in draft.iter().enumerate() {
                    let probability = p.probability(i as u32);
                    let correction = if residual > 0.0 {
                        (1.0 - accepted) * (probability - d).max(0.0) / residual
                    } else {
                        0.0
                    };
                    assert!(
                        (probability.min(d) + correction - probability).abs() < 1e-12,
                        "bounded draft support changes the target probability"
                    );
                }
            }
        }
        let mut counted = SamplerState::new(
            SamplingConfig {
                temperature: 1.0,
                repeat_penalty: 1.5,
                repeat_last_n: 1,
                frequency_penalty: 0.25,
                presence_penalty: 0.5,
                ..SamplingConfig::default()
            },
            &[1, 1],
        )
        .unwrap();
        counted.accept_tokens(&[3, 3]).unwrap();
        counted.accept_tokens(&[0; 80]).unwrap();
        counted.accept(4).unwrap();
        let mut rng = 17;
        let proposal = sample_proposal(&c, &counted, &mut rng).unwrap();
        // Token 3 is outside the repetition window but keeps both generated
        // counts; token 1 occurs only in the prompt; token 4 gets all three.
        let adjusted = [1.2 - 2.0 * 0.25 - 0.5, 0.1, -0.7 * 1.5 - 0.25 - 0.5];
        let sum: f64 = adjusted.iter().map(|&l: &f64| l.exp()).sum();
        for i in 0..proposal.size {
            let index = c.ids[..3]
                .iter()
                .position(|&id| id == proposal.ids[i])
                .unwrap();
            assert!(
                (f64::from(proposal.probabilities[i]) - adjusted[index].exp() / sum).abs()
                    < 2.0 / f64::from(1u32 << 24),
                "compact MTP proposal lost full generated penalty counts"
            );
        }
    }

    #[test]
    fn proposals_and_corrections_keep_the_target_frequencies() {
        let logits = [0.3f32, -0.2, 1.1, 0.7, -1.0];
        let c = candidates();
        let trials = 100_000;
        for temperature in [0.7f32, 1.0] {
            let mut sampler = SamplerState::new(
                SamplingConfig {
                    temperature,
                    seed: 73,
                    ..SamplingConfig::default()
                },
                &[],
            )
            .unwrap();
            let p = sampler.distribution(&logits).unwrap();
            let mut counts = [0u32; 5];
            for _ in 0..trials {
                let mut draft_rng = sampling::next_random(sampler.mutable_rng_state());
                let q = sample_proposal(&c, &sampler, &mut draft_rng).unwrap();
                counts[verify_proposal(&logits, &q, &mut sampler).unwrap().token as usize] += 1;
            }
            for (token, &count) in counts.iter().enumerate() {
                assert!(
                    (f64::from(count) / f64::from(trials) - p.probability(token as u32)).abs()
                        < 0.008,
                    "proposal and correction draws distort target frequencies"
                );
            }
        }
    }

    #[test]
    fn verification_uses_the_canonical_distribution_at_cutoffs() {
        let logits = [0.0f32, -1.0, -2.0];
        let history = [1u32, 1, 2];
        for base in sampling_cases() {
            let mut config = base;
            config.temperature = 0.5;
            config.seed = 6;
            // top-p and min-p cutoffs around the finite FP64 boundary.
            for top_p in [0.8f32, 0.866_813_3, 0.866_813_4, 1.0] {
                config.top_p = top_p;
                for min_p in [0.0f32, 0.135_335_27, 0.135_335_3] {
                    config.min_p = min_p;
                    let mut sampler = SamplerState::new(config.clone(), &history).unwrap();
                    sampler.accept(1).unwrap();
                    let target = sampler.distribution(&logits).unwrap();
                    for draft in 0..logits.len() as u32 {
                        let mut proposal = Proposal {
                            ids: [0; MTP_CANDIDATES],
                            probabilities: [0.0; MTP_CANDIDATES],
                            size: 1,
                            token: draft,
                            probability: 1.0,
                        };
                        proposal.ids[0] = draft;
                        proposal.probabilities[0] = 1.0;
                        for _ in 0..16 {
                            let (mut reference, mut replay) = (sampler.clone(), sampler.clone());
                            let accepted = reference.uniform() < target.probability(draft);
                            let expected = if accepted {
                                draft
                            } else {
                                target
                                    .sample_residual(
                                        &proposal.ids[..1],
                                        &proposal.probabilities[..1],
                                        reference.mutable_rng_state(),
                                    )
                                    .unwrap()
                            };
                            let actual = verify_proposal(&logits, &proposal, &mut sampler).unwrap();
                            let repeated =
                                verify_proposal(&logits, &proposal, &mut replay).unwrap();
                            assert!(
                                actual.accepted == accepted
                                    && actual.token == expected
                                    && sampler.rng_state() == reference.rng_state()
                                    && actual == repeated
                                    && sampler.rng_state() == replay.rng_state(),
                                "MTP acceptance/residual must use the canonical AR distribution"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn deterministic_drafts_defer_rejections_and_keep_stops() {
        let logits = [0.3f32, -0.2, 1.1, 0.7, -1.0];
        let history = [2u32, 3, 2, 0];
        for config in sampling_cases()
            .into_iter()
            .filter(|c| !c.uses_random_sampling())
        {
            for accepted in 0..4 {
                let mut ar = SamplerState::new(config.clone(), &history).unwrap();
                let mut speculative = ar.clone();
                for _ in 0..accepted {
                    let token = ar.sample(&logits).unwrap();
                    ar.accept(token).unwrap();
                    assert_eq!(
                        verify_draft(&logits, token as i32, &mut speculative, |_| false).unwrap(),
                        DraftDecision::Accept,
                        "matching target sample was rejected"
                    );
                }
                let (before, saved) = (speculative.rng_state(), speculative.history().to_vec());
                let token = ar.sample(&logits).unwrap();
                assert_eq!(
                    verify_draft(&logits, ((token + 1) % 5) as i32, &mut speculative, |_| {
                        false
                    })
                    .unwrap(),
                    DraftDecision::Reject,
                    "mismatching target sample was accepted"
                );
                assert!(
                    speculative.rng_state() == before && speculative.history() == saved,
                    "rejected suffix consumed RNG or history"
                );
                assert_eq!(
                    speculative.sample(&logits).unwrap(),
                    token,
                    "deferred rejection lost the AR token"
                );
                ar.accept(token).unwrap();
                speculative.accept(token).unwrap();
                assert!(
                    ar.rng_state() == speculative.rng_state()
                        && ar.history() == speculative.history(),
                    "accepted prefix differs from AR state"
                );
                let mut stop_reference = speculative.clone();
                let stop = stop_reference.sample(&logits).unwrap();
                assert_eq!(
                    verify_draft(&logits, ((stop + 1) % 5) as i32, &mut speculative, |t| t
                        == stop as i32)
                    .unwrap(),
                    DraftDecision::Stop,
                    "target stop must end verification even when the draft differs"
                );
                assert_eq!(
                    speculative.rng_state(),
                    stop_reference.rng_state(),
                    "stop draw was discarded"
                );
            }
        }
    }
}
