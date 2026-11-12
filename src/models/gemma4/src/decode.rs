//! Decoding cycles (`engine.cpp`: `DecodeStep`, `BeginCycle`, `CycleDraft`,
//! `FinishDraft`, `FinishCycle`, `DecodeBatch`, `EvaluateBatch`). One cycle
//! emits the pending token if needed, drafts with the MTP drafter or copies
//! from the prompt (prompt lookup), verifies the rows in one forward and
//! accepts while the target's own sample agrees, so greedy decoding
//! reproduces autoregressive output and sampled decoding keeps the target
//! distribution.

use std::sync::Arc;

use gufo_core::prompt_lookup::{MIN_MATCH, PromptLookup};
use gufo_core::sampling::{
    self, LogitCandidate, SamplerState, SamplingDistribution, TokenId as SampleToken,
    collect_logprobs,
};

use crate::cycle_trace::{CycleRecord, CycleTrace, Sparse, Stop as TraceStop};
use crate::draft_policy::{
    CalibratedChain, DraftBatch, DraftCalibration, DraftCalibrationScope, DraftCosts, DraftPolicy,
    draft_costs_at, draft_signal,
};
use crate::draft_vocab::DRAFT_HEAD_ROWS;
use crate::engine::{BatchDecode, DecodeResult, Exec, Model, Session};
use crate::executor::{Candidates, DraftHead, DraftJob, DraftProposal, SPLIT_ROWS, Segment};
use crate::mtp_sampling::{
    MtpCandidateLogits, MtpProposal, sample_mtp_proposal, verify_mtp_target,
};
use crate::proposal_sampler::without_top_p;
use crate::sibling_verification::verify_with_sibling;
use crate::tokenizer::TokenId;

/// The confidence policy: a chain continues while the drafter's estimate
/// that every draft so far is accepted stays at or above a floor; greedy
/// chains use the top-1 share of the top-64 candidates, sampled chains the
/// probability of the sampled proposal, capped at four drafts.
const GREEDY_CHAIN_FLOOR: f32 = 0.5;
const SAMPLED_CHAIN_FLOOR: f32 = 0.3;
const SAMPLED_DRAFT_CAP: u32 = 4;

fn text(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The top candidate's softmax share among the top-64 logits (`TopShare`).
fn top_share(c: &MtpCandidateLogits) -> f32 {
    let mut top = c.logits[0];
    for &l in &c.logits[1..c.size] {
        top = top.max(l);
    }
    let mut total = 0.0f64;
    for &l in &c.logits[..c.size] {
        total += f64::from(l - top).exp();
    }
    (1.0 / total) as f32
}

/// The two largest softmax shares among the top-64 logits (`TopTwoShares`).
fn top_two_shares(c: &MtpCandidateLogits) -> (f32, f32) {
    let mut top = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &l in &c.logits[..c.size] {
        if l > top {
            second = top;
            top = l;
        } else if l > second {
            second = l;
        }
    }
    let mut total = 0.0f64;
    for &l in &c.logits[..c.size] {
        total += f64::from(l - top).exp();
    }
    let runner = if c.size > 1 {
        f64::from(second - top).exp()
    } else {
        0.0
    };
    ((1.0 / total) as f32, (runner / total) as f32)
}

/// The argmax candidate, lowest token id on ties (`TopToken`).
fn top_token(c: &MtpCandidateLogits) -> i32 {
    let mut best = 0usize;
    for i in 1..c.size {
        if c.logits[i] > c.logits[best] || (c.logits[i] == c.logits[best] && c.ids[i] < c.ids[best])
        {
            best = i;
        }
    }
    c.ids[best] as i32
}

/// The drafter's choice after the chosen draft: a sibling at the same
/// depth, and its calibration signal (`Alternative`).
#[derive(Debug, Clone, Copy)]
struct Alternative {
    token: i32,
    signal: f32,
}

impl Default for Alternative {
    fn default() -> Self {
        Self {
            token: -1,
            signal: 0.0,
        }
    }
}

/// Greedy: the runner-up candidate (`GreedyAlternative`).
fn greedy_alternative(c: &MtpCandidateLogits, chosen: i32) -> Alternative {
    let mut best = c.size;
    let mut top = c.size;
    for i in 0..c.size {
        if c.ids[i] as i32 == chosen {
            top = i;
            continue;
        }
        if best == c.size
            || c.logits[i] > c.logits[best]
            || (c.logits[i] == c.logits[best] && c.ids[i] < c.ids[best])
        {
            best = i;
        }
    }
    if best == c.size || top == c.size {
        return Alternative::default();
    }
    let mut total = 0.0f64;
    for i in 0..c.size {
        total += f64::from(c.logits[i] - c.logits[top]).exp();
    }
    let second = f64::from(c.logits[best] - c.logits[top]).exp() / total;
    let rest = 1.0 - 1.0 / total;
    Alternative {
        token: c.ids[best] as i32,
        signal: if rest > 0.0 {
            (second / rest) as f32
        } else {
            0.0
        },
    }
}

/// Sampled: a draw from the proposal without its token
/// (`SampledAlternative`).
fn sampled_alternative(p: &MtpProposal, rng: &mut u64) -> Alternative {
    let rest = 1.0 - f64::from(p.probability);
    if rest.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Alternative::default();
    }
    let draw = sampling::uniform(rng) * rest;
    let mut cumulative = 0.0f64;
    let mut last = p.size;
    for i in 0..p.size {
        if p.ids[i] == p.token || p.probabilities[i] <= 0.0 {
            continue;
        }
        last = i;
        cumulative += f64::from(p.probabilities[i]);
        if draw < cumulative {
            break;
        }
    }
    if last == p.size {
        return Alternative::default();
    }
    Alternative {
        token: p.ids[last] as i32,
        signal: (f64::from(p.probabilities[last]) / rest) as f32,
    }
}

/// Whether a sampler's target distributions can come from the
/// verification rows' candidates (`CandidateSampling`).
fn candidate_sampling(sampler: &SamplerState) -> Result<bool, String> {
    let c = sampler.config();
    Ok(c.uses_random_sampling()
        && c.top_k > 0
        && c.top_k as usize <= gufo_gemma4_kernels::sizes::CANDIDATE_TOP
        && c.min_keep <= gufo_gemma4_kernels::sizes::CANDIDATE_TOP
        && !sampler.needs_constraint_mask().map_err(text)?)
}

/// Adds to `extra` the tokens a cycle's sampler may penalize in any of its
/// rows (`AddPenalizable`).
fn add_penalizable(
    sampler: &SamplerState,
    rows: &[TokenId],
    siblings: &[TokenId],
    extra: &mut Vec<u32>,
) {
    if !sampler.config().penalties_enabled() {
        return;
    }
    extra.extend(sampler.penalties().iter().map(|p| p.token));
    extra.extend(rows.iter().map(|&t| t as u32));
    extra.extend(siblings.iter().map(|&t| t as u32));
}

fn sort_unique(tokens: &mut Vec<u32>) {
    tokens.sort_unstable();
    tokens.dedup();
}

/// The sessions of a batch drafting together (`DraftShare`).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DraftShare {
    sessions: u32,
    rows: u32,
    expected: f32,
    /// Each session's even share of the forward past its pending row, and
    /// the rows the even split leaves over.
    fair: u32,
    spare: u32,
}

/// One decode cycle (`Session::Cycle`).
#[derive(Default)]
pub(crate) struct Cycle {
    max_tokens: usize,
    stop_at_eos: bool,
    /// Nothing to verify: the result is complete.
    done: bool,
    position: u32,
    rows: Vec<TokenId>,
    sampled: bool,
    proposals: Vec<MtpProposal>,
    copied: usize,
    copy_start: usize,
    small_head: bool,
    siblings: Vec<TokenId>,
    sibling_signals: Vec<f32>,
    signals: Vec<f32>,
    expected: f32,
    draft_ms: f32,
    trace: Option<CycleRecord>,
    result: DecodeResult,
}

/// Where a cycle's drafting starts: drafter steps and slots.
#[derive(Debug, Clone, Copy)]
struct DraftSetup {
    steps: u32,
    max_drafts: u32,
}

/// One cycle's drafting: the length policy, sampled proposals and
/// prompt-lookup copies, asked for one proposal per drafter step
/// (`Session::CycleDraft`).
struct CycleDraft<'c> {
    /// Whether this cycle may copy (not a request's first).
    lookup: bool,
    policy: DraftPolicy,
    min_drafts: u32,
    steps: u32,
    max_drafts: u32,
    alone: bool,
    sampled: bool,
    stop_at_eos: bool,
    calibrated: CalibratedChain<'c>,
    /// The table the chain prices with (trace estimates).
    table: &'c DraftCalibration,
    chain: f32,
    /// Rows past the pending one taken from the batch (`SharesSpare`).
    used: u32,
    /// The context the next token continues: pending plus kept drafts.
    context: Vec<i32>,
    copies: Vec<i32>,
    drafts: Vec<i32>,
    small_head: bool,
    alternatives: Vec<Alternative>,
    draft_sampler: Option<SamplerState>,
    draft_rng: u64,
    sibling_rng: u64,
    proposals: Vec<MtpProposal>,
    signals: Vec<f32>,
    copy_start: usize,
    trace: Option<CycleRecord>,
}

/// What a cycle draft reads of its session.
struct DraftContext<'a> {
    lookup: &'a PromptLookup<i32>,
    tokens: &'a [TokenId],
    model: &'a Model,
}

impl<'c> CycleDraft<'c> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        session_lookup: &mut PromptLookup<i32>,
        tokens: &[TokenId],
        model: &Model,
        cycle: &mut Cycle,
        sampler: &mut SamplerState,
        setup: DraftSetup,
        table: &'c DraftCalibration,
        costs: &'c DraftCosts,
        alone: bool,
        others: DraftBatch,
    ) -> Result<Self, String> {
        let options = &model.options;
        let min_drafts = options.min_draft_tokens;
        let calibrated = CalibratedChain::new(
            table,
            costs,
            min_drafts,
            setup.steps,
            if alone { DraftBatch::default() } else { others },
        );
        let mut trace = cycle.trace.take();
        if let Some(t) = trace.as_mut() {
            t.step_cap = setup.steps;
        }
        // A request's first cycle copies nothing (`PromptLookup::Step`).
        let lookup = session_lookup.step(cycle.position as usize);
        if lookup {
            session_lookup.extend(tokens, false);
        }
        let mut draft_sampler = None;
        let mut draft_rng = 0;
        let mut sibling_rng = 0;
        if cycle.sampled {
            // A cycle-local proposal stream; target draws keep the
            // sampler's. Siblings draw from their own stream.
            draft_rng = sampling::next_random(sampler.mutable_rng_state());
            sibling_rng = draft_rng ^ 0x9E37_79B9_7F4A_7C15;
            // Drafts are proposals: only the target's verification is
            // constrained. Dense targets' drafts skip top-p too.
            let mut s = sampler.without_constraint();
            if !model.config().has_experts() {
                s = without_top_p(&s)?;
            }
            draft_sampler = Some(s);
        }
        let pending = cycle
            .rows
            .first()
            .copied()
            .ok_or("gemma4 cycle without its pending row")?;
        Ok(Self {
            lookup,
            policy: options.draft_policy,
            min_drafts,
            steps: setup.steps,
            max_drafts: setup.max_drafts,
            alone,
            sampled: cycle.sampled,
            stop_at_eos: cycle.stop_at_eos,
            calibrated,
            table,
            chain: 1.0,
            used: 0,
            context: vec![pending],
            copies: Vec::new(),
            drafts: Vec::new(),
            small_head: false,
            alternatives: Vec::new(),
            draft_sampler,
            draft_rng,
            sibling_rng,
            proposals: Vec::new(),
            signals: Vec::new(),
            copy_start: 0,
            trace,
        })
    }

    fn worthwhile(&self) -> bool {
        self.policy != DraftPolicy::Calibrated || self.calibrated.first_draft_can_pay()
    }

    /// Whether this cycle may go past its even share into the batch's spare
    /// rows.
    fn shares_spare(&self, share: Option<&DraftShare>) -> bool {
        share.is_some_and(|s| s.spare + s.fair != 0)
            && !self.alone
            && self.policy == DraftPolicy::Calibrated
    }

    /// Rows this cycle may still add: what remains of its share, then
    /// spare.
    fn free_rows(&self, share: &DraftShare) -> u32 {
        share.fair.saturating_sub(self.used) + share.spare
    }

    /// Counts one more row, past the share from the spare rows.
    fn take_row(&mut self, share: Option<&mut DraftShare>) {
        let Some(share) = share else {
            return;
        };
        if !self.shares_spare(Some(share)) {
            return;
        }
        if self.used >= share.fair && share.spare > 0 {
            share.spare -= 1;
        }
        self.used += 1;
    }

    /// Fills the remaining slots with the tokens that followed an earlier
    /// occurrence of the context, as far as the other occurrences matching
    /// about as well agree (`Copy`).
    fn copy(
        &mut self,
        minimum: usize,
        mut share: Option<&mut DraftShare>,
        s: &DraftContext<'_>,
    ) -> bool {
        if !self.lookup {
            return false;
        }
        if let Some(&back) = self.context.last()
            && self.stop_at_eos
            && s.model.is_stop_token(back)
        {
            return false;
        }
        let mut room = (self.max_drafts as usize).saturating_sub(self.context.len() - 1);
        if self.shares_spare(share.as_deref()) {
            let free = share.as_deref().map_or(0, |sh| self.free_rows(sh));
            room = room.min(free as usize);
        }
        if room < minimum {
            return false;
        }
        let (found, agreed) =
            s.lookup
                .find_agreed(s.tokens, &self.context, MIN_MATCH, |_| true, false);
        if found.length == 0 {
            return false;
        }
        let available = room.min(agreed.unwrap_or(0));
        if available < minimum {
            return false;
        }
        self.copies.clear();
        for i in 0..available {
            let token = s.tokens[found.start + i];
            self.copies.push(token);
            self.take_row(share.as_deref_mut());
            if self.stop_at_eos && s.model.is_stop_token(token) {
                break;
            }
        }
        self.copy_start = found.start;
        true
    }

    /// Whether the draft just proposed ends the chain unverified (`Stop`).
    fn stop(
        &mut self,
        kept: usize,
        confidence: f32,
        c: &MtpCandidateLogits,
        mut share: Option<&mut DraftShare>,
    ) -> bool {
        match self.policy {
            DraftPolicy::Calibrated => {
                if self.shares_spare(share.as_deref())
                    && share.as_deref().is_some_and(|sh| self.free_rows(sh) == 0)
                {
                    return true;
                }
                let signal = draft_signal(c);
                let before = self.calibrated.expected();
                if let Some(sh) = share.as_deref()
                    && !self.alone
                {
                    let own_rows = kept as u32 + 1;
                    self.calibrated.set_others(DraftBatch {
                        rows: sh.rows - own_rows,
                        expected: sh.expected - before,
                        draft_ms: 0.0,
                    });
                }
                let estimate = if self.trace.is_some() {
                    self.table.estimate(signal)
                } else {
                    0.0
                };
                if !self.calibrated.include(signal) {
                    let d = self.calibrated.last_decision();
                    if let Some(t) = self.trace.as_mut()
                        && d.compared
                    {
                        t.stop = Some(TraceStop {
                            signal,
                            estimate,
                            gain: d.gain,
                            cost: d.cost,
                        });
                    }
                    return true;
                }
                if let Some(t) = self.trace.as_mut() {
                    let d = self.calibrated.last_decision();
                    t.estimates.push(estimate);
                    t.margins
                        .push(if d.compared { d.gain - d.cost } else { 0.0 });
                }
                if let Some(sh) = share.as_deref_mut() {
                    sh.rows += 1;
                    sh.expected += self.calibrated.expected() - before;
                }
                self.take_row(share);
                self.signals.push(signal);
                false
            }
            DraftPolicy::Confidence => {
                self.chain *= confidence;
                kept >= self.min_drafts as usize
                    && self.chain
                        < if self.sampled {
                            SAMPLED_CHAIN_FLOOR
                        } else {
                            GREEDY_CHAIN_FLOOR
                        }
            }
            DraftPolicy::Fixed => false,
        }
    }

    fn propose(
        &mut self,
        c: &MtpCandidateLogits,
        mut share: Option<&mut DraftShare>,
        s: &DraftContext<'_>,
    ) -> Result<DraftProposal, String> {
        if let Some(t) = self.trace.as_mut() {
            t.drafter_steps += 1;
            t.proposal_signals.push(draft_signal(c));
            let (top, second) = top_two_shares(c);
            t.proposal_top.push(top);
            t.proposal_margin.push(top - second);
        }
        if self.sampled {
            let sampler = self
                .draft_sampler
                .as_mut()
                .ok_or("gemma4 sampled cycle without its sampler")?;
            let proposal = sample_mtp_proposal(c, sampler, &mut self.draft_rng)?;
            if let Some(t) = self.trace.as_mut() {
                t.proposal_q.push(proposal.probability);
                t.draft_candidates.push(Sparse {
                    ids: c.ids[..c.size].iter().map(|&i| i as i32).collect(),
                    values: c.logits[..c.size].to_vec(),
                });
            }
            if self.stop(
                self.proposals.len(),
                proposal.probability,
                c,
                share.as_deref_mut(),
            ) {
                let _ = self.copy(1, share, s);
                return Ok(DraftProposal::default());
            }
            if let Some(sampler) = self.draft_sampler.as_mut() {
                sampler.accept(proposal.token).map_err(text)?;
            }
            let alternative = sampled_alternative(&proposal, &mut self.sibling_rng);
            self.alternatives.push(alternative);
            self.proposals.push(proposal);
            self.context.push(proposal.token as i32);
            let last = self.copy(1, share, s);
            return Ok(DraftProposal {
                token: Some(proposal.token as i32),
                last,
            });
        }
        if let Some(t) = self.trace.as_mut()
            && let Some(&top) = t.proposal_top.last()
        {
            t.proposal_q.push(top);
        }
        if self.stop(
            self.context.len() - 1,
            top_share(c),
            c,
            share.as_deref_mut(),
        ) {
            let _ = self.copy(1, share, s);
            return Ok(DraftProposal::default());
        }
        let token = top_token(c);
        self.alternatives.push(greedy_alternative(c, token));
        self.context.push(token);
        let last = self.copy(1, share, s);
        Ok(DraftProposal {
            token: Some(token),
            last,
        })
    }
}

/// Drafter steps shared by the sessions of a batch cost each a share
/// (`SharedCosts`).
fn shared_costs(position: u32, experts: bool, share: Option<&DraftShare>) -> DraftCosts {
    let mut costs = draft_costs_at(position, experts);
    if let Some(sh) = share
        && sh.sessions > 1
    {
        for ms in &mut costs.draft {
            *ms /= sh.sessions as f32;
        }
    }
    costs
}

/// A cycle's view of its verification forward's candidates
/// (`VerifyCandidates`).
struct VerifyView<'a> {
    rows: &'a Candidates,
    first: usize,
    extra: &'a [u32],
}

/// A cycle's full logit rows: given, or read from the forward on first use.
struct Rows<'a> {
    given: Option<&'a [f32]>,
    owned: Option<Vec<f32>>,
    /// The cycle's first row in the forward, its rows, the forward's rows.
    first: usize,
    count: usize,
    total: usize,
    vocab: usize,
}

impl Rows<'_> {
    fn ensure(&mut self, exec: &mut Exec) -> Result<(), String> {
        if self.given.is_none() && self.owned.is_none() {
            let all = exec.read_logits(self.total)?;
            self.owned =
                Some(all[self.first * self.vocab..(self.first + self.count) * self.vocab].to_vec());
        }
        Ok(())
    }

    fn get(&self) -> Option<&[f32]> {
        self.given.or(self.owned.as_deref())
    }

    /// Row `r` of the cycle.
    fn row(&mut self, exec: &mut Exec, r: usize) -> Result<&[f32], String> {
        self.ensure(exec)?;
        let vocab = self.vocab;
        self.get()
            .map(|l| &l[r * vocab..(r + 1) * vocab])
            .ok_or_else(|| "gemma4 cycle has no logits".to_string())
    }
}

/// The calibration table of a cycle (`Session::Calibration`).
fn table<'a>(
    scope: DraftCalibrationScope,
    session: &'a mut [DraftCalibration; 4],
    model: &'a mut [DraftCalibration; 4],
    sampled: bool,
    sibling: bool,
) -> &'a mut DraftCalibration {
    let tables = if scope == DraftCalibrationScope::Request {
        session
    } else {
        model
    };
    &mut tables[usize::from(sampled) + if sibling { 2 } else { 0 }]
}

impl Session {
    /// A copy of the cycle's calibration table.
    fn calibration_of(
        &mut self,
        exec: &mut Exec,
        sampled: bool,
        sibling: bool,
    ) -> DraftCalibration {
        *table(
            self.model.options.draft_calibration,
            &mut self.calibration,
            &mut exec.calibration,
            sampled,
            sibling,
        )
    }

    /// Brings the logits up to date (`EnsureLogits`).
    fn ensure_logits(&mut self, exec: &mut Exec) -> Result<(), String> {
        if self.logits_on_device {
            exec.copy_cache_logits(&self.cache, &mut self.logits)?;
            self.logits_on_device = false;
        }
        Ok(())
    }

    /// Emits the pending token if needed and decides the drafting
    /// (`BeginCycle` up to the drafter); `None` when nothing is drafted.
    fn begin_cycle(
        &mut self,
        exec: &mut Exec,
        cycle: &mut Cycle,
        sampler: &mut SamplerState,
        draft_limit: u32,
        share: Option<&DraftShare>,
    ) -> Result<Option<DraftSetup>, String> {
        cycle.result = DecodeResult::default();
        cycle.done = true;
        cycle.trace = None;
        if self.model.trace.is_some() {
            cycle.trace = Some(CycleRecord {
                begin_ns: CycleTrace::now(),
                ..CycleRecord::default()
            });
        }
        if cycle.max_tokens == 0 {
            return Ok(None);
        }
        if self.pending.is_none() {
            if !self.valid {
                return Err("session has no logits to decode from".into());
            }
            self.ensure_logits(exec)?;
            let token = sampler.sample(&self.logits).map_err(text)?;
            if let Some(lp) =
                collect_logprobs(&self.logits, token, sampler.config().logprobs).map_err(text)?
            {
                cycle.result.logprobs.push(lp);
            }
            sampler.accept(token).map_err(text)?;
            let token = token as TokenId;
            cycle.result.tokens.push(token);
            cycle.result.stop = cycle.stop_at_eos && self.model.is_stop_token(token);
            self.pending = Some(token);
            if cycle.result.stop || cycle.result.tokens.len() >= cycle.max_tokens {
                return Ok(None);
            }
        }
        let pending = self
            .pending
            .ok_or("gemma4 cycle without its pending token")?;
        let position = self.tokens.len() as u32;
        if position + 1 >= self.cache.max_context {
            return Err("session context is full".into());
        }
        cycle.done = false;
        cycle.position = position;
        cycle.rows = vec![pending];
        let budget = cycle.max_tokens - cycle.result.tokens.len();
        let mut steps = 0u32;
        if self.model.has_mtp() && budget > 1 {
            steps = draft_limit
                .min((budget - 1).min(u32::MAX as usize) as u32)
                .min(self.cache.max_context - position - 1);
        }
        // With random sampling the drafter samples its proposals too.
        cycle.sampled = steps > 0 && sampler.config().uses_random_sampling();
        let options = &self.model.options;
        let policy = options.draft_policy;
        if let Some(sh) = share
            && (policy != DraftPolicy::Calibrated
                || (cycle.sampled && options.draft_calibration == DraftCalibrationScope::Request))
        {
            steps = steps.min(sh.fair);
        }
        let max_drafts = steps;
        if cycle.sampled && policy == DraftPolicy::Confidence {
            steps = steps.min(SAMPLED_DRAFT_CAP);
        }
        cycle.signals.clear();
        if steps == 0 {
            return Ok(None);
        }
        Ok(Some(DraftSetup { steps, max_drafts }))
    }

    /// Whether the cycle prices its drafts as if alone (`CycleDraft::Alone`).
    fn alone(&self, cycle: &Cycle) -> bool {
        cycle.sampled && self.model.options.draft_calibration == DraftCalibrationScope::Request
    }

    /// Brings the draft vocabulary up to the tokens (loading new table
    /// rows) and picks the head for a chain after `anchor`
    /// (`PrepareDraftHead`); returns whether the chain may score the small
    /// head.
    fn prepare_draft_head(&mut self, exec: &mut Exec, anchor: TokenId) -> Result<bool, String> {
        let model = Arc::clone(&self.model);
        let rows = &model.device.draft_head_rows;
        if rows.is_empty() {
            return Ok(false);
        }
        let common = |token: i32| {
            usize::try_from(token)
                .ok()
                .and_then(|t| rows.get(t))
                .is_some_and(|&row| row < DRAFT_HEAD_ROWS)
        };
        self.draft_added.clear();
        self.draft_vocab
            .catch_up(&self.tokens, common, Some(&mut self.draft_added));
        let first = (self.draft_vocab.rare().len() - self.draft_added.len()) as u32;
        exec.load_draft_table(&model.device, &mut self.cache, first, &self.draft_added)?;
        Ok(self.draft_vocab.fits(anchor, common))
    }

    /// Appends a finished chain's drafts and copies to the cycle's rows and
    /// decides its siblings (`FinishDraft`).
    fn finish_draft(
        &self,
        cycle: &mut Cycle,
        mut draft: CycleDraft<'_>,
        hits: &DraftCalibration,
        mut share: Option<&mut DraftShare>,
    ) {
        if cycle.sampled {
            // A copied token is a point-mass proposal: accepted with the
            // target's probability, a rejection resampling without it.
            for &token in &draft.copies {
                let mut proposal = MtpProposal::default();
                proposal.ids[0] = token as SampleToken;
                proposal.probabilities[0] = 1.0;
                proposal.size = 1;
                proposal.token = proposal.ids[0];
                proposal.probability = 1.0;
                draft.proposals.push(proposal);
            }
        }
        cycle.proposals = std::mem::take(&mut draft.proposals);
        cycle.signals = std::mem::take(&mut draft.signals);
        cycle.copy_start = draft.copy_start;
        cycle.trace = draft.trace.take();
        cycle.rows.extend_from_slice(&draft.drafts);
        cycle.rows.extend_from_slice(&draft.copies);
        cycle.copied = draft.copies.len();
        if draft.policy == DraftPolicy::Calibrated {
            // Siblings, shallowest first, while each pays for its row.
            let mut rows = cycle.rows.len() as u32;
            if let Some(sh) = share.as_deref()
                && !draft.alone
            {
                let own = draft.drafts.len() as u32 + 1;
                let expected = draft.calibrated.expected();
                draft.calibrated.set_others(DraftBatch {
                    rows: sh.rows - own,
                    expected: sh.expected - expected,
                    draft_ms: 0.0,
                });
            }
            let mut i = 0;
            while !draft.alone
                && i < draft.drafts.len()
                && i < draft.alternatives.len()
                && rows < draft.max_drafts + 1
                && (!draft.shares_spare(share.as_deref())
                    || share.as_deref().is_some_and(|sh| draft.free_rows(sh) > 0))
            {
                let alt = draft.alternatives[i];
                let before = draft.calibrated.expected();
                if alt.token < 0
                    || !draft
                        .calibrated
                        .include_sibling(i as u32, rows, hits.estimate(alt.signal))
                {
                    break;
                }
                if let Some(sh) = share.as_deref_mut() {
                    sh.rows += 1;
                    sh.expected += draft.calibrated.expected() - before;
                }
                draft.take_row(share.as_deref_mut());
                cycle.siblings.push(alt.token);
                cycle.sibling_signals.push(alt.signal);
                if let Some(t) = cycle.trace.as_mut() {
                    t.sibling_estimates.push(hits.estimate(alt.signal));
                }
                rows += 1;
                i += 1;
            }
            cycle.rows.extend_from_slice(&cycle.siblings.clone());
            cycle.expected = draft.calibrated.expected();
            cycle.draft_ms = draft.calibrated.draft_ms();
            if let Some(t) = cycle.trace.as_mut() {
                t.priced_draft_ms = cycle.draft_ms;
                t.priced_verify_ms = draft.calibrated.verify(rows);
                t.expected = cycle.expected;
            }
        }
        if let Some(t) = cycle.trace.as_mut() {
            for i in 0..draft.drafts.len() {
                let alt = draft.alternatives.get(i).copied().unwrap_or_default();
                t.runner_ups.push(alt.token);
                t.runner_up_signals.push(alt.signal);
            }
        }
        cycle.small_head = draft.small_head;
    }

    /// Records each chain row's acceptance probability and the greedy
    /// runner-up hits in the cycle's trace, before verification
    /// (`TraceRows`).
    fn trace_rows(
        &self,
        cycle: &mut Cycle,
        sampler: &SamplerState,
        logits: &[f32],
        argmax: Option<&[u32]>,
        chain: usize,
    ) -> Result<(), String> {
        let vocab = self.model.vocab_size() as usize;
        let Some(trace) = cycle.trace.as_mut() else {
            return Ok(());
        };
        let target = |r: usize| -> i32 {
            if let Some(a) = argmax {
                return a[r] as i32;
            }
            let row = &logits[r * vocab..(r + 1) * vocab];
            let mut best = 0;
            for (i, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = i;
                }
            }
            best as i32
        };
        for r in 0..chain.saturating_sub(1) {
            let beside = r < trace.runner_ups.len();
            let up = if beside { trace.runner_ups[r] } else { -1 };
            if !cycle.sampled {
                let t = target(r);
                trace
                    .row_acceptance
                    .push(if t == cycle.rows[r + 1] { 1.0 } else { 0.0 });
                if beside {
                    trace
                        .runner_up_acceptance
                        .push(if up >= 0 && t == up { 1.0 } else { 0.0 });
                }
                continue;
            }
            let q = &cycle.proposals[r];
            let p = sampler
                .distribution(&logits[r * vocab..(r + 1) * vocab])
                .map_err(text)?;
            trace
                .row_acceptance
                .push((p.probability(q.token) / f64::from(q.probability)).min(1.0) as f32);
            trace.proposal_dist.push(Sparse {
                ids: q.ids[..q.size].iter().map(|&i| i as i32).collect(),
                values: q.probabilities[..q.size].to_vec(),
            });
            trace.target_dist.push(Sparse {
                ids: p.entries().iter().map(|e| e.token as i32).collect(),
                values: p.entries().iter().map(|e| e.value as f32).collect(),
            });
            if !beside {
                continue;
            }
            // VerifyWithSibling's second stage.
            let mass = |token: SampleToken| {
                q.ids[..q.size]
                    .iter()
                    .position(|&id| id == token)
                    .map_or(0.0, |i| f64::from(q.probabilities[i]))
            };
            let total: f64 = p
                .entries()
                .iter()
                .map(|e| (e.value - mass(e.token)).max(0.0))
                .sum();
            let rest = 1.0 - f64::from(q.probability);
            let mut hit = 0.0;
            if up >= 0 && total > 0.0 && rest > 0.0 {
                let sibling = up as SampleToken;
                let own = mass(sibling) / rest;
                let residual = (p.probability(sibling) - mass(sibling)).max(0.0) / total;
                hit = if own > 0.0 {
                    (residual / own).min(1.0)
                } else {
                    0.0
                };
            }
            trace.runner_up_acceptance.push(hit as f32);
        }
        Ok(())
    }

    /// Accepts from the verification rows' argmax ids (greedy), candidates
    /// (sampled) or full logits (`FinishCycle`). `rows` of the last forward
    /// from `first_row` are the cycle's; `total_rows` the forward's.
    #[allow(clippy::too_many_arguments)]
    fn finish_cycle(
        &mut self,
        exec: &mut Exec,
        cycle: &mut Cycle,
        sampler: &mut SamplerState,
        given: Option<&[f32]>,
        argmax: Option<&[u32]>,
        first_row: u32,
        total_rows: usize,
        candidates: Option<&VerifyView<'_>>,
    ) -> Result<(), String> {
        let model = Arc::clone(&self.model);
        let vocab = model.vocab_size() as usize;
        let rows = cycle.rows.clone();
        let siblings = cycle.siblings.len();
        let chain = rows.len() - siblings;
        let first = first_row as usize;
        // The cycle's full logit rows, copied on first use when not given.
        let mut rows_of = Rows {
            given,
            owned: None,
            first,
            count: rows.len(),
            total: total_rows,
            vocab,
        };
        if cycle.trace.is_some() {
            if argmax.is_none() {
                rows_of.ensure(exec)?;
            }
            let logits: &[f32] = rows_of.get().unwrap_or(&[]);
            let sampler_view = sampler.clone();
            self.trace_rows(cycle, &sampler_view, logits, argmax, chain)?;
        }
        // Row r's candidates give its exact target distribution while the
        // sampler needs no constraint mask and holds no deferred draw.
        let compact = |sampler: &SamplerState, r: usize| -> Result<bool, String> {
            Ok(match candidates {
                Some(v) => {
                    v.rows.complete(v.first + r)
                        && !sampler.needs_constraint_mask().map_err(text)?
                        && sampler.save_draw_state().pending.is_none()
                }
                None => false,
            })
        };
        let candidate_row = |r: usize| -> Vec<LogitCandidate> {
            let Some(v) = candidates else {
                return Vec::new();
            };
            let row = v.first + r;
            let mut out: Vec<LogitCandidate> = v
                .rows
                .top(row)
                .iter()
                .filter(|c| v.extra.binary_search(&c.token).is_err())
                .copied()
                .collect();
            out.extend_from_slice(v.rows.extra(row));
            out
        };
        let mut row = 0usize;
        let mut accepted = 0u64;
        let mut agrees;
        let mut sibling = false;
        let is_stop = |t: TokenId| model.is_stop_token(t);
        loop {
            agrees = false;
            let beside = row < siblings && row + 1 < chain;
            let token: TokenId;
            if cycle.sampled && row + 1 < chain {
                let is_compact = compact(sampler, row)?;
                let target: SamplingDistribution = if is_compact {
                    sampler
                        .distribution_from_candidates(&candidate_row(row))
                        .map_err(text)?
                } else {
                    sampler
                        .distribution(rows_of.row(exec, row)?)
                        .map_err(text)?
                };
                if beside {
                    let v = verify_with_sibling(
                        &target,
                        &cycle.proposals[row],
                        cycle.siblings[row] as SampleToken,
                        sampler,
                    )?;
                    token = v.token as TokenId;
                    agrees = v.accepted;
                    sibling = v.sibling;
                } else {
                    let v =
                        verify_mtp_target(vocab, &cycle.proposals[row], sampler, |_| Ok(target))?;
                    token = v.token as TokenId;
                    agrees = v.accepted;
                }
            } else {
                token = if let Some(a) = argmax {
                    let t = a[row];
                    if t as usize >= vocab {
                        return Err("logit distribution contains no finite values".into());
                    }
                    t as TokenId
                } else if compact(sampler, row)? {
                    let target = sampler
                        .distribution_from_candidates(&candidate_row(row))
                        .map_err(text)?;
                    target.sample(sampler.mutable_rng_state()).map_err(text)? as TokenId
                } else {
                    sampler.sample(rows_of.row(exec, row)?).map_err(text)? as TokenId
                };
            }
            // Emit: row's distribution produced the token; logprobs read
            // its full row.
            if sampler.config().logprobs.is_some()
                && let Some(lp) = collect_logprobs(
                    rows_of.row(exec, row)?,
                    token as SampleToken,
                    sampler.config().logprobs,
                )
                .map_err(text)?
            {
                cycle.result.logprobs.push(lp);
            }
            sampler.accept(token as SampleToken).map_err(text)?;
            cycle.result.tokens.push(token);
            if cycle.stop_at_eos && is_stop(token) {
                cycle.result.stop = true;
            }
            if !(cycle.sampled && row + 1 < chain) {
                agrees = row + 1 < chain && token == rows[row + 1];
                sibling = beside && !agrees && token == cycle.siblings[row];
            }
            if !agrees || cycle.result.stop || cycle.result.tokens.len() >= cycle.max_tokens {
                break;
            }
            accepted += 1;
            row += 1;
        }
        let scope = model.options.draft_calibration;
        if row < siblings && row + 1 < chain && !agrees {
            table(
                scope,
                &mut self.calibration,
                &mut exec.calibration,
                cycle.sampled,
                true,
            )
            .observe(cycle.sibling_signals[row], sibling);
            if let Some(t) = cycle.trace.as_mut() {
                t.sibling_observed = Some(sibling);
            }
        }
        // An accepted sibling (the target's own token) commits in place of
        // the rejected draft, and its row's token follows.
        let mut last = row;
        if sibling && !cycle.result.stop && cycle.result.tokens.len() < cycle.max_tokens {
            last = chain + row;
            let token = if let Some(a) = argmax {
                let t = a[last];
                if t as usize >= vocab {
                    return Err("logit distribution contains no finite values".into());
                }
                t as TokenId
            } else if compact(sampler, last)? {
                let target = sampler
                    .distribution_from_candidates(&candidate_row(last))
                    .map_err(text)?;
                target.sample(sampler.mutable_rng_state()).map_err(text)? as TokenId
            } else {
                sampler.sample(rows_of.row(exec, last)?).map_err(text)? as TokenId
            };
            if sampler.config().logprobs.is_some()
                && let Some(lp) = collect_logprobs(
                    rows_of.row(exec, last)?,
                    token as SampleToken,
                    sampler.config().logprobs,
                )
                .map_err(text)?
            {
                cycle.result.logprobs.push(lp);
            }
            sampler.accept(token as SampleToken).map_err(text)?;
            cycle.result.tokens.push(token);
            if cycle.stop_at_eos && is_stop(token) {
                cycle.result.stop = true;
            }
            self.tokens.extend_from_slice(&rows[..=row]);
            self.tokens.push(cycle.siblings[row]);
            exec.move_key(
                &model.device,
                &mut self.cache,
                cycle.position + (chain + row) as u32,
                cycle.position + (row + 1) as u32,
            )?;
            self.stats.siblings_accepted += 1;
        } else {
            // Rows [0, row] are committed; the last emitted token becomes
            // pending.
            self.tokens.extend_from_slice(&rows[..=row]);
        }
        if cycle.trace.is_some()
            && cycle.sampled
            && let Some(logits) = rows_of.get()
        {
            // The distribution the pending token was drawn from.
            let p = sampler
                .distribution(&logits[last * vocab..(last + 1) * vocab])
                .map_err(text)?;
            let mut top = 0.0f64;
            let mut entropy = 0.0f64;
            for e in p.entries() {
                top = top.max(e.value);
                if e.value > 0.0 {
                    entropy -= e.value * e.value.ln();
                }
            }
            if let Some(t) = cycle.trace.as_mut() {
                t.pending_top = top as f32;
                t.pending_entropy = entropy as f32;
            }
        }
        let committed = first_row + last as u32;
        exec.commit_hidden(&model.device, &mut self.cache, committed)?;
        // Greedy and candidate-sampled decode keep the row on the device:
        // nothing reads it on the host before the next cycle.
        if argmax.is_some() || candidates.is_some() {
            exec.commit_logits(&mut self.cache, committed)?;
            self.logits_on_device = true;
        } else {
            self.logits = rows_of.row(exec, last)?.to_vec();
            self.logits_on_device = false;
        }
        self.valid = true;
        self.pending = cycle.result.tokens.last().copied();
        self.stats.cycles += 1;
        self.stats.verified += u64::from(chain > 1);
        self.stats.drafted += (chain - 1) as u64;
        self.stats.accepted += accepted;
        self.stats.siblings += siblings as u64;
        // Copies trail the MTP drafts.
        let mtp = chain - 1 - cycle.copied;
        self.stats.copied += cycle.copied as u64;
        // The calibrated policy learns from every draft verification judged.
        if !cycle.signals.is_empty() {
            let judged = (row + usize::from(row + 1 < chain)).min(cycle.signals.len());
            let t = table(
                scope,
                &mut self.calibration,
                &mut exec.calibration,
                cycle.sampled,
                false,
            );
            for j in 0..judged {
                t.observe(cycle.signals[j], j < row || agrees);
            }
            if let Some(trace) = cycle.trace.as_mut() {
                for j in 0..judged {
                    trace.observed.push(j < row || agrees);
                }
            }
        }
        let copies_kept = (accepted as usize).saturating_sub(mtp);
        self.stats.copied_accepted += copies_kept as u64;
        if cycle.copied != 0 {
            self.lookup
                .observe(cycle.copy_start, cycle.copied, copies_kept);
        }
        if cycle.small_head {
            self.stats.small_head_cycles += 1;
            self.stats.small_head_drafted += mtp as u64;
            self.stats.small_head_accepted += (accepted as usize).min(mtp) as u64;
        }
        if let Some(mut trace) = cycle.trace.take() {
            trace.position = cycle.position;
            trace.sampled = cycle.sampled;
            if cycle.sampled {
                let config = sampler.config();
                trace.temperature = config.temperature;
                trace.top_k = config.top_k;
                trace.top_p = config.top_p;
            }
            trace.rows = rows.len() as u32;
            trace.copied = cycle.copied as u32;
            trace.siblings = siblings as u32;
            trace.accepted = accepted as u32;
            trace.sibling_accepted = sibling;
            trace.small_head = cycle.small_head;
            trace.emitted = cycle.result.tokens.len() as u32;
            trace.signals = cycle.signals.clone();
            trace.end_ns = CycleTrace::now();
            if let Some(out) = &model.trace {
                out.write(&trace).map_err(text)?;
            }
        }
        Ok(())
    }

    /// Emits the next tokens (`DecodeStep`). With the MTP drafter one call
    /// drafts, verifies and emits the accepted prefix plus the target's own
    /// next sample; without it (or with a one-token budget) one token is
    /// emitted. The last emitted token stays pending until the next call or
    /// a sync.
    pub fn decode_step(
        &mut self,
        max_tokens: usize,
        sampler: &mut SamplerState,
        stop_at_eos: bool,
    ) -> Result<DecodeResult, String> {
        self.drop_lookahead();
        let model = Arc::clone(&self.model);
        let mut guard = model.exec();
        let exec = &mut *guard;
        let mut cycle = Cycle {
            max_tokens,
            stop_at_eos,
            ..Cycle::default()
        };
        let result = self.decode_cycle(exec, &mut cycle, sampler);
        if let Err(e) = result {
            self.reset();
            return Err(e);
        }
        Ok(std::mem::take(&mut cycle.result))
    }

    fn decode_cycle(
        &mut self,
        exec: &mut Exec,
        cycle: &mut Cycle,
        sampler: &mut SamplerState,
    ) -> Result<(), String> {
        let model = Arc::clone(&self.model);
        let setup = self.begin_cycle(exec, cycle, sampler, model.draft_tokens(), None)?;
        if let Some(setup) = setup {
            let alone = self.alone(cycle);
            let calibration = self.calibration_of(exec, cycle.sampled, false);
            let hits = self.calibration_of(exec, cycle.sampled, true);
            let costs = shared_costs(cycle.position, model.config().has_experts(), None);
            let mut draft = CycleDraft::new(
                &mut self.lookup,
                &self.tokens,
                &model,
                cycle,
                sampler,
                setup,
                &calibration,
                &costs,
                alone,
                DraftBatch::default(),
            )?;
            if !draft.worthwhile() {
                // Not even a certain draft would pay: verify the pending
                // token alone without running the drafter.
                cycle.trace = draft.trace.take();
            } else {
                let ctx = DraftContext {
                    lookup: &self.lookup,
                    tokens: &self.tokens,
                    model: &model,
                };
                // A context that continues an earlier passage verifies at
                // least three copies without running the drafter.
                if draft.copy(3, None, &ctx) {
                    self.finish_draft(cycle, draft, &hits, None);
                } else {
                    let pending = cycle.rows[0];
                    let small = self.prepare_draft_head(exec, pending)?;
                    draft.small_head = small;
                    let ctx = DraftContext {
                        lookup: &self.lookup,
                        tokens: &self.tokens,
                        model: &model,
                    };
                    let job = DraftJob {
                        cache: &self.cache,
                        token: pending,
                        position: cycle.position,
                        steps: draft.steps,
                        head: DraftHead {
                            small,
                            table: self.draft_vocab.rare(),
                        },
                    };
                    let drafts = exec.draft_chains(&model.device, &[job], &mut |_, c| {
                        draft.propose(c, None, &ctx)
                    })?;
                    draft.drafts = drafts.into_iter().next().unwrap_or_default();
                    self.finish_draft(cycle, draft, &hits, None);
                }
            }
        }
        if cycle.done {
            return Ok(());
        }
        if let Some(t) = cycle.trace.as_mut() {
            t.drafted_ns = CycleTrace::now();
            t.forward = exec.forwards;
            exec.forwards += 1;
        }
        // Logprobs need every emitted token's full row on the host.
        let logprobs = sampler.config().logprobs.is_some();
        let use_argmax =
            !cycle.sampled && !logprobs && sampler.config().can_use_unmodified_argmax();
        let mut use_candidates = !use_argmax
            && !logprobs
            && cycle.trace.is_none()
            && exec.can_read_candidates()
            && candidate_sampling(sampler)?;
        let mut extra = Vec::new();
        if use_candidates {
            add_penalizable(sampler, &cycle.rows, &cycle.siblings, &mut extra);
            sort_unique(&mut extra);
            use_candidates = extra.len() <= gufo_gemma4_kernels::sizes::MAX_CANDIDATE_EXTRA;
        }
        let n = cycle.rows.len();
        let logit_rows: Vec<u32> = (0..n as u32).collect();
        exec.forward_one(
            &model.device,
            &mut self.cache,
            &cycle.rows,
            cycle.position,
            &logit_rows,
            &[],
            cycle.siblings.len() as u32,
        )?;
        if use_argmax {
            let argmax = exec.read_argmax(n)?;
            if let Some(t) = cycle.trace.as_mut() {
                t.verified_ns = CycleTrace::now();
            }
            self.finish_cycle(exec, cycle, sampler, None, Some(&argmax), 0, n, None)
        } else if use_candidates {
            let candidates = exec.read_candidates(n, &extra)?;
            let view = VerifyView {
                rows: &candidates,
                first: 0,
                extra: &extra,
            };
            self.finish_cycle(exec, cycle, sampler, None, None, 0, n, Some(&view))
        } else {
            let logits = exec.read_logits(n)?.to_vec();
            if let Some(t) = cycle.trace.as_mut() {
                t.verified_ns = CycleTrace::now();
            }
            self.finish_cycle(exec, cycle, sampler, Some(&logits), None, 0, n, None)
        }
    }

    /// `decode_step` for several sessions of one model with a single
    /// verification forward of at most `SPLIT_ROWS` rows, so every row keeps
    /// the arithmetic of its own session's decode (`DecodeBatch`). Returns
    /// false if any session failed (see [`BatchDecode::error`]); the others
    /// completed.
    pub fn decode_batch(decodes: &mut [BatchDecode<'_>]) -> bool {
        if decodes.is_empty() {
            return true;
        }
        for d in decodes.iter_mut() {
            d.session.drop_lookahead();
            d.error = None;
        }
        let model = Arc::clone(&decodes[0].session.model);
        if decodes
            .iter()
            .any(|d| !Arc::ptr_eq(&d.session.model, &model))
        {
            for d in decodes.iter_mut() {
                d.error = Some("gemma4 batch mixes models".into());
            }
            return false;
        }
        let mut guard = model.exec();
        let exec = &mut *guard;
        let count = decodes.len() as u32;
        // Every session takes its pending row and an even share of the
        // rest; calibrated sessions may also take the rows that split
        // leaves over.
        let fair = (SPLIT_ROWS / count).max(1) - 1;
        let spare = if count > 1 {
            SPLIT_ROWS.saturating_sub(count * (fair + 1))
        } else {
            0
        };
        let draft_limit = model.draft_tokens().min(fair + spare);
        let mut share = DraftShare {
            sessions: count,
            rows: count,
            expected: count as f32,
            fair,
            spare,
        };
        let others = DraftBatch {
            rows: count - 1,
            expected: (count - 1) as f32,
            draft_ms: 0.0,
        };
        let batched = count > 1;
        let mut ok = true;
        let mut cycles: Vec<Cycle> = Vec::with_capacity(decodes.len());
        let mut setups: Vec<Option<DraftSetup>> = Vec::with_capacity(decodes.len());
        for d in decodes.iter_mut() {
            let mut cycle = Cycle {
                max_tokens: d.max_tokens,
                stop_at_eos: d.stop_at_eos,
                ..Cycle::default()
            };
            let setup = d.session.begin_cycle(
                exec,
                &mut cycle,
                d.sampler,
                draft_limit,
                batched.then_some(&share),
            );
            match setup {
                Ok(setup) => setups.push(setup),
                Err(e) => {
                    d.session.reset();
                    d.error = Some(e);
                    cycle.done = true;
                    setups.push(None);
                    ok = false;
                }
            }
            cycles.push(cycle);
        }
        // Calibration tables and costs, fixed for the drafting.
        let mut tables = Vec::with_capacity(decodes.len());
        for (i, d) in decodes.iter_mut().enumerate() {
            let alone = d.session.alone(&cycles[i]);
            let calibration = d.session.calibration_of(exec, cycles[i].sampled, false);
            let hits = d.session.calibration_of(exec, cycles[i].sampled, true);
            let costs = shared_costs(
                cycles[i].position,
                model.config().has_experts(),
                if alone || !batched {
                    None
                } else {
                    Some(&share)
                },
            );
            tables.push((calibration, hits, costs, alone));
        }
        // Each drafting session's chain state.
        let mut drafts: Vec<Option<CycleDraft<'_>>> = Vec::with_capacity(decodes.len());
        let mut heads = vec![false; decodes.len()];
        for (i, d) in decodes.iter_mut().enumerate() {
            let Some(setup) = setups[i] else {
                drafts.push(None);
                continue;
            };
            let (calibration, _, costs, alone) = &tables[i];
            let session = &mut *d.session;
            let created = CycleDraft::new(
                &mut session.lookup,
                &session.tokens,
                &model,
                &mut cycles[i],
                d.sampler,
                setup,
                calibration,
                costs,
                *alone,
                others,
            );
            let mut draft = match created {
                Ok(draft) => draft,
                Err(e) => {
                    session.reset();
                    d.error = Some(e);
                    cycles[i].done = true;
                    ok = false;
                    drafts.push(None);
                    continue;
                }
            };
            if !draft.worthwhile() {
                // Beside other sessions not even a certain draft would pay.
                cycles[i].trace = draft.trace.take();
                drafts.push(None);
                continue;
            }
            let ctx = DraftContext {
                lookup: &session.lookup,
                tokens: &session.tokens,
                model: &model,
            };
            if draft.copy(3, batched.then_some(&mut share), &ctx) {
                session.finish_draft(
                    &mut cycles[i],
                    draft,
                    &tables[i].1,
                    batched.then_some(&mut share),
                );
                drafts.push(None);
                continue;
            }
            let pending = cycles[i].rows[0];
            match session.prepare_draft_head(exec, pending) {
                Ok(small) => heads[i] = small,
                Err(e) => {
                    session.reset();
                    d.error = Some(e);
                    cycles[i].done = true;
                    ok = false;
                    drafts.push(None);
                    continue;
                }
            }
            drafts.push(Some(draft));
        }
        let drafting: Vec<usize> = (0..drafts.len()).filter(|&i| drafts[i].is_some()).collect();
        if !drafting.is_empty() {
            // A lone chain drafts on its small head.
            if drafting.len() == 1
                && let Some(draft) = drafts[drafting[0]].as_mut()
            {
                draft.small_head = heads[drafting[0]];
            }
            let result = {
                let jobs: Vec<DraftJob<'_>> = drafting
                    .iter()
                    .map(|&i| {
                        let s = &*decodes[i].session;
                        DraftJob {
                            cache: &s.cache,
                            token: cycles[i].rows[0],
                            position: cycles[i].position,
                            steps: drafts[i].as_ref().map_or(0, |d| d.steps),
                            head: DraftHead {
                                small: heads[i],
                                table: s.draft_vocab.rare(),
                            },
                        }
                    })
                    .collect();
                let contexts: Vec<DraftContext<'_>> = drafting
                    .iter()
                    .map(|&i| {
                        let s = &*decodes[i].session;
                        DraftContext {
                            lookup: &s.lookup,
                            tokens: &s.tokens,
                            model: &model,
                        }
                    })
                    .collect();
                let share_ref = &mut share;
                exec.draft_chains(&model.device, &jobs, &mut |j, c| {
                    let i = drafting[j];
                    let draft = drafts[i]
                        .as_mut()
                        .ok_or("gemma4 draft job without its chain")?;
                    draft.propose(c, batched.then_some(&mut *share_ref), &contexts[j])
                })
            };
            match result {
                Ok(chains) => {
                    for (j, chain) in chains.into_iter().enumerate() {
                        let i = drafting[j];
                        if let Some(mut draft) = drafts[i].take() {
                            draft.drafts = chain;
                            decodes[i].session.finish_draft(
                                &mut cycles[i],
                                draft,
                                &tables[i].1,
                                batched.then_some(&mut share),
                            );
                        }
                    }
                }
                Err(e) => {
                    for &i in &drafting {
                        drafts[i] = None;
                        decodes[i].session.reset();
                        decodes[i].error = Some(e.clone());
                        cycles[i].done = true;
                    }
                    ok = false;
                }
            }
        }
        let active: Vec<usize> = (0..decodes.len())
            .filter(|&i| !cycles[i].done && decodes[i].error.is_none())
            .collect();
        for i in 0..decodes.len() {
            if cycles[i].done && decodes[i].error.is_none() {
                decodes[i].result = std::mem::take(&mut cycles[i].result);
            }
        }
        if active.is_empty() {
            return ok;
        }
        if model.trace.is_some() {
            let drafted = CycleTrace::now();
            for &i in &active {
                if let Some(t) = cycles[i].trace.as_mut() {
                    t.drafted_ns = drafted;
                    t.forward = exec.forwards;
                    t.sessions = active.len() as u32;
                }
            }
            exec.forwards += 1;
        }
        // Each cycle reads argmax ids (greedy) or sampling candidates; if
        // any needs full logits, all read them.
        let mut sample = vec![false; decodes.len()];
        let mut any_argmax = false;
        let mut any_candidates = false;
        let mut full = model.trace.is_some() || !exec.can_read_candidates();
        let mut extra = Vec::new();
        for &i in &active {
            let sampler = &*decodes[i].sampler;
            if sampler.config().logprobs.is_some() {
                full = true;
            } else if !cycles[i].sampled && sampler.config().can_use_unmodified_argmax() {
                any_argmax = true;
            } else if candidate_sampling(sampler).unwrap_or(false) {
                sample[i] = true;
                any_candidates = true;
                add_penalizable(sampler, &cycles[i].rows, &cycles[i].siblings, &mut extra);
            } else {
                full = true;
            }
        }
        sort_unique(&mut extra);
        full = full || extra.len() > gufo_gemma4_kernels::sizes::MAX_CANDIDATE_EXTRA;
        if full {
            any_argmax = false;
            any_candidates = false;
        }
        let mut tokens: Vec<TokenId> = Vec::new();
        for &i in &active {
            tokens.extend_from_slice(&cycles[i].rows);
        }
        let verified = (|| -> Result<(), String> {
            {
                let mut caches: Vec<(&mut Session, &Cycle)> = Vec::new();
                let mut rest: &mut [BatchDecode<'_>] = decodes;
                let mut index = 0;
                for &i in &active {
                    let (_, tail) = rest.split_at_mut(i - index);
                    let (head, tail) = tail.split_at_mut(1);
                    caches.push((&mut *head[0].session, &cycles[i]));
                    rest = tail;
                    index = i + 1;
                }
                let mut segments: Vec<Segment<'_>> = caches
                    .into_iter()
                    .map(|(s, c)| Segment {
                        cache: &mut s.cache,
                        first_position: c.position,
                        rows: c.rows.len() as u32,
                        siblings: c.siblings.len() as u32,
                    })
                    .collect();
                let logit_rows: Vec<u32> = (0..tokens.len() as u32).collect();
                exec.forward(&model.device, &mut segments, &tokens, &logit_rows, &[])?;
            }
            let total = tokens.len();
            let logits = if full {
                Some(exec.read_logits(total)?.to_vec())
            } else {
                None
            };
            let argmax = if any_argmax {
                Some(exec.read_argmax(total)?)
            } else {
                None
            };
            let candidates = if any_candidates {
                Some(exec.read_candidates(total, &extra)?)
            } else {
                None
            };
            if model.trace.is_some() {
                let now = CycleTrace::now();
                for &i in &active {
                    if let Some(t) = cycles[i].trace.as_mut() {
                        t.verified_ns = now;
                    }
                }
            }
            let vocab = model.vocab_size() as usize;
            let mut row0 = 0usize;
            for &i in &active {
                let n = cycles[i].rows.len();
                let d = &mut decodes[i];
                let given = logits
                    .as_ref()
                    .map(|l| &l[row0 * vocab..(row0 + n) * vocab]);
                let am = if !full && !sample[i] {
                    argmax.as_ref().map(|a| &a[row0..row0 + n])
                } else {
                    None
                };
                let view = candidates.as_ref().map(|c| VerifyView {
                    rows: c,
                    first: row0,
                    extra: &extra,
                });
                let view = if !full && sample[i] {
                    view.as_ref()
                } else {
                    None
                };
                d.session.finish_cycle(
                    exec,
                    &mut cycles[i],
                    d.sampler,
                    given,
                    am,
                    row0 as u32,
                    total,
                    view,
                )?;
                d.result = std::mem::take(&mut cycles[i].result);
                row0 += n;
            }
            Ok(())
        })();
        if let Err(e) = verified {
            for &i in &active {
                decodes[i].session.reset();
                decodes[i].error = Some(e.clone());
            }
            return false;
        }
        ok
    }

    /// `evaluate` for several sessions of one model in a single forward
    /// (`EvaluateBatch`).
    pub fn evaluate_batch(sessions: &mut [&mut Session], tokens: &[TokenId]) -> Result<(), String> {
        for s in sessions.iter_mut() {
            s.drop_lookahead();
        }
        if sessions.len() != tokens.len() {
            return Err("gemma4 batch needs one token per session".into());
        }
        if sessions.is_empty() {
            return Ok(());
        }
        if sessions.len() > SPLIT_ROWS as usize {
            return Err("gemma4 batch is wider than one decode forward".into());
        }
        let model = Arc::clone(&sessions[0].model);
        if sessions.iter().any(|s| !Arc::ptr_eq(&s.model, &model)) {
            return Err("gemma4 batch mixes models".into());
        }
        if sessions
            .iter()
            .any(|s| s.tokens.len() >= s.cache.max_context as usize)
        {
            return Err("session context is full".into());
        }
        let mut guard = model.exec();
        let exec = &mut *guard;
        let result = (|| -> Result<Vec<f32>, String> {
            {
                let mut segments: Vec<Segment<'_>> = sessions
                    .iter_mut()
                    .map(|s| Segment {
                        first_position: s.tokens.len() as u32,
                        cache: &mut s.cache,
                        rows: 1,
                        siblings: 0,
                    })
                    .collect();
                let logit_rows: Vec<u32> = (0..tokens.len() as u32).collect();
                exec.forward(&model.device, &mut segments, tokens, &logit_rows, &[])?;
            }
            for (i, s) in sessions.iter_mut().enumerate() {
                exec.commit_hidden(&model.device, &mut s.cache, i as u32)?;
            }
            Ok(exec.read_logits(tokens.len())?.to_vec())
        })();
        let logits = match result {
            Ok(logits) => logits,
            Err(e) => {
                for s in sessions.iter_mut() {
                    s.reset();
                }
                return Err(e);
            }
        };
        let vocab = model.vocab_size() as usize;
        for (i, s) in sessions.iter_mut().enumerate() {
            s.pending = None;
            s.tokens.push(tokens[i]);
            s.logits = logits[i * vocab..(i + 1) * vocab].to_vec();
            s.logits_on_device = false;
            s.valid = true;
        }
        Ok(())
    }
}
