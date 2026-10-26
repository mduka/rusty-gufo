//! Speculative decoding for serving, a port of the `SpeculativeVerifier`
//! path the Qwen runner drives (`core/speculative/speculative_verifier.cpp`)
//! with its serving options: batched verification, retained frontier logits,
//! a fixed draft length. The draft proposes a block, the target verifies it
//! with decode arithmetic, and the longest agreeing prefix plus a correction
//! is emitted; greedy requests compare argmax tokens, sampled requests use
//! the device sampler (penalties only) or lossless rejection sampling of the
//! drafted candidates. Several sessions verify in one pass, in chunks for
//! requests whose acceptance is low.

use gufo_core::qwen_tokenizer::StopTokens;
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_hip::Stream;

use crate::dflash::draft::{Draft, DraftConfig, DraftContext, DraftSnapshot, ProposalRequest};
use crate::dflash::executor::{DraftPenalties, DraftScratch, Proposal};
use crate::dflash::model::DFlashModel;
use crate::engine::{Model, Session, VerifyRequest};
use crate::persistent::{VerifierHeader, VerifierLimits};

/// Rolling acceptance entries a snapshot may carry (`rolling_window`).
const ROLLING_WINDOW: usize = 16;

/// The target and draft models with the draft scratch: what a verification
/// step runs on.
pub struct SpecExec<'a> {
    pub model: &'a mut Model,
    pub draft_model: &'a DFlashModel,
    pub draft_scratch: &'a mut DraftScratch,
    /// The stream of the draft scratch and sessions. Every draft call waits
    /// for its own work only, so on a stream of its own the next proposal
    /// overlaps the target's recurrence replay, as with C++'s draft stream.
    pub draft_stream: &'a Stream,
}

impl SpecExec<'_> {
    fn draft(&mut self) -> DraftContext<'_> {
        DraftContext {
            stream: self.draft_stream,
            model: self.draft_model,
            scratch: &mut *self.draft_scratch,
        }
    }

    /// The draft context beside the target, read-only: the prompt features
    /// the target's prefill left go straight to the draft.
    fn target_and_draft(&mut self) -> (&Model, DraftContext<'_>) {
        (
            &*self.model,
            DraftContext {
                stream: self.draft_stream,
                model: self.draft_model,
                scratch: &mut *self.draft_scratch,
            },
        )
    }
}

/// `SpeculativeStats`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub draft_tokens: u64,
    pub accepted_tokens: u64,
    pub verification_steps: u64,
    pub emitted_tokens: u64,
}

impl Stats {
    pub fn acceptance_rate(&self) -> f32 {
        if self.draft_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.draft_tokens as f32
        }
    }
}

/// One verification step's outcome (`StepResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepResult {
    pub emitted: Vec<u32>,
    pub next_token: u32,
    pub next_token_logits: Vec<f32>,
    pub draft_count: usize,
    pub accepted_count: usize,
    pub hit_eos: bool,
    pub physical_width: usize,
    /// Of the drafted and accepted tokens: those copied from the history.
    pub lookup_tokens: usize,
    pub lookup_accepted_tokens: usize,
}

/// One session's side of speculative decoding.
pub struct Verifier {
    draft: Draft,
    stats: Stats,
    min_draft_tokens: u32,
    max_draft_tokens: u32,
    /// The fixed draft length (`current_draft_length_`, adaptive off).
    draft_length: u32,
    rolling_acceptance: Vec<f32>,
    /// Steps return the logits after their last token
    /// (`retain_frontier_logits`, on for serving).
    retain_frontier_logits: bool,
}

/// A verifier state at a committed boundary (`SpeculativeVerifierSnapshot`).
pub struct VerifierSnapshot {
    draft: DraftSnapshot,
    stats: Stats,
    draft_length: u32,
    rolling_acceptance: Vec<f32>,
}

impl VerifierSnapshot {
    pub fn payload_bytes(&self) -> usize {
        self.draft.payload_bytes()
            + self.rolling_acceptance.len() * 4
            + std::mem::size_of::<Stats>()
            + 4
    }
}

/// One session's request of a verification step (`StepRequest`).
pub struct StepRequest<'a> {
    pub verifier: &'a mut Verifier,
    pub session: &'a mut Session,
    pub sequence: &'a [u32],
    pub position: u32,
    pub current_token: u32,
    /// The target's stop tokens (`IsStopToken`).
    pub stops: StopTokens,
    pub max_emitted: u32,
    pub sampler: &'a mut SamplerState,
}

/// A step prepared before its target rows run (`PreparedStep`).
#[derive(Default)]
struct Prepared {
    proposal: Proposal,
    inputs: Vec<u32>,
    token_probabilities: Vec<f64>,
    working: Option<SamplerState>,
    immediate: Option<StepResult>,
    position: u32,
    stops: StopTokens,
    sampled: bool,
    random: bool,
    target_only: bool,
    max_draft_tokens: u32,
    chunk_limit: usize,
    verified_rows: usize,
    accepted: usize,
    physical_width: usize,
}

/// What a verification pass returned for one session.
struct Chunk {
    predictions: Vec<u32>,
    hidden: Vec<f32>,
}

/// The target's token for verification row `row`: its already computed
/// argmax when the sampler can take it unchanged (greedy, no penalties,
/// legal under the grammar, so also the exact masked argmax), else a
/// selection from the row's logits.
fn select_row(
    model: &mut Model,
    session: &Session,
    predictions: &[u32],
    row: usize,
    working: &mut SamplerState,
) -> Result<u32, String> {
    if let Some(&prediction) = predictions.get(row)
        && working
            .can_select_argmax(prediction, false)
            .map_err(|e| e.to_string())?
    {
        return Ok(prediction);
    }
    model.sample_verification(session, row, working)
}

fn err(message: &str) -> String {
    message.to_string()
}

fn validate(request: &StepRequest<'_>) -> Result<(), String> {
    request
        .sampler
        .config()
        .validate()
        .map_err(|e| e.to_string())
}

/// The numeric penalties a request's draft q follows: those of the working
/// sampler of a sampled step, else the request's.
fn draft_penalties(step: &Prepared, request: &StepRequest<'_>) -> DraftPenalties {
    DraftPenalties::of(step.working.as_ref().unwrap_or(request.sampler))
}

impl Verifier {
    /// A verifier over a new draft; the limits are the serving options
    /// before the draft clamps them to its block
    /// (`ConfigureAdaptiveDraftPolicy`).
    pub fn new(
        ctx: &DraftContext<'_>,
        config: DraftConfig,
        min_draft_tokens: u32,
        max_draft_tokens: u32,
    ) -> Result<Self, String> {
        let draft = Draft::new(ctx, config)?;
        let max_draft_tokens = max_draft_tokens.max(1);
        let min_draft_tokens = min_draft_tokens.clamp(1, max_draft_tokens);
        Ok(Self {
            draft,
            stats: Stats::default(),
            min_draft_tokens,
            max_draft_tokens,
            draft_length: max_draft_tokens,
            rolling_acceptance: Vec::new(),
            retain_frontier_logits: true,
        })
    }

    /// Whether steps return the logits after their last token; the command
    /// line leaves them out (`retain_frontier_logits`).
    pub fn set_retain_frontier_logits(&mut self, retain: bool) {
        self.retain_frontier_logits = retain;
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn draft(&self) -> &Draft {
        &self.draft
    }

    pub fn draft_mut(&mut self) -> &mut Draft {
        &mut self.draft
    }

    /// `Reset`.
    pub fn reset(&mut self) {
        self.stats = Stats::default();
        self.rolling_acceptance.clear();
        self.draft_length = self.max_draft_tokens;
        self.draft.reset();
    }

    /// `BeginRequest`.
    pub fn begin_request(&mut self) {
        self.stats = Stats::default();
        self.rolling_acceptance.clear();
        self.draft_length = self.max_draft_tokens;
        self.draft.begin_request();
    }

    fn capture(exec: &SpecExec<'_>, st: &mut Session) -> Result<(), String> {
        let taps = exec.draft_model.config.target_layer_ids.clone();
        st.set_hidden_capture(exec.model.config(), true, &taps)
    }

    /// Resets both sides, prefills `prompt` and primes the draft with its
    /// features; the first target token (`Prime`).
    pub fn prime(
        &mut self,
        exec: &mut SpecExec<'_>,
        st: &mut Session,
        prompt: &[u32],
        compute_logits: bool,
    ) -> Result<u32, String> {
        if prompt.is_empty() {
            return Err("speculative prompt must not be empty".into());
        }
        self.reset();
        exec.model.reset(st)?;
        Self::capture(exec, st)?;
        let first = exec
            .model
            .forward_prompt(st, prompt, 0, compute_logits)?
            .unwrap_or(0);
        let (model, mut draft) = exec.target_and_draft();
        let hidden = model.prompt_hidden()?;
        if hidden.is_empty() || !hidden.len().is_multiple_of(prompt.len()) {
            return Err("target executor did not capture complete prompt hidden states".into());
        }
        self.draft
            .prime(&mut draft, hidden, prompt.len())
            .map_err(|_| err("draft backend failed to prime target context"))?;
        Ok(first)
    }

    /// Prefills `tokens` at `position` behind the retained prefix and
    /// appends their features to the draft (`ExtendPrompt`).
    pub fn extend_prompt(
        &mut self,
        exec: &mut SpecExec<'_>,
        st: &mut Session,
        tokens: &[u32],
        position: u32,
        compute_logits: bool,
    ) -> Result<u32, String> {
        if tokens.is_empty() || position == 0 {
            return Err("prompt extension needs a nonempty retained prefix".into());
        }
        Self::capture(exec, st)?;
        let next = exec
            .model
            .forward_prompt(st, tokens, position, compute_logits)?
            .unwrap_or(0);
        let (model, mut draft) = exec.target_and_draft();
        let hidden = model.prompt_hidden()?;
        if hidden.is_empty() || !hidden.len().is_multiple_of(tokens.len()) {
            return Err("draft failed to append target prompt features".into());
        }
        self.draft
            .append(&mut draft, hidden, tokens.len(), position)
            .map_err(|_| err("draft failed to append target prompt features"))?;
        Ok(next)
    }

    /// A plain decode step whose features go straight to the draft
    /// (`AdvanceCommittedToken`).
    pub fn advance_committed_token(
        &mut self,
        exec: &mut SpecExec<'_>,
        st: &mut Session,
        token: u32,
        position: u32,
    ) -> Result<u32, String> {
        let next = exec
            .model
            .forward_token(st, token, position, true)?
            .ok_or("decode returned no token")?;
        let hidden = exec.model.last_hidden(st)?;
        self.draft
            .append(&mut exec.draft(), &hidden, 1, position)
            .map_err(|_| err("draft failed to append committed target token"))?;
        Ok(next)
    }

    /// Generates up to `max_new_tokens` after `prompt` under `sampling`
    /// (`SpeculativeVerifier::Generate`): primes both sides, then verifies
    /// drafted blocks until a stop token or the budget. `on_token` sees each
    /// emitted token and stops generation by returning `false`.
    pub fn generate(
        &mut self,
        exec: &mut SpecExec<'_>,
        st: &mut Session,
        prompt: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingConfig,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, String> {
        let mut output = Vec::new();
        if prompt.is_empty() {
            return Ok(output);
        }
        sampling.validate().map_err(|e| e.to_string())?;
        let mut first = self.prime(exec, st, prompt, true)?;
        let mut sampler = SamplerState::new(sampling.clone(), prompt).map_err(|e| e.to_string())?;
        if !sampling.can_use_unmodified_argmax() {
            first = exec.model.sample(st, None, &mut sampler)?;
        }
        let stops = exec.model.weights().tokenizer.stop_tokens();
        if max_new_tokens == 0 || stops.contains(first) {
            return Ok(output);
        }
        let mut sequence = prompt.to_vec();
        sequence.push(first);
        let mut position = u32::try_from(prompt.len()).map_err(|_| "prompt exceeds u32")?;
        let mut next = first;
        output.push(first);
        sampler.accept(first).map_err(|e| e.to_string())?;
        if !on_token(first) {
            return Ok(output);
        }
        while output.len() < max_new_tokens {
            let remaining = u32::try_from(max_new_tokens - output.len()).unwrap_or(u32::MAX);
            let step = Self::verify_step(
                exec,
                &mut StepRequest {
                    verifier: &mut *self,
                    session: &mut *st,
                    sequence: &sequence,
                    position,
                    current_token: next,
                    stops,
                    max_emitted: remaining,
                    sampler: &mut sampler,
                },
            )?;
            let mut stop = false;
            for &token in &step.emitted {
                if stops.contains(token) {
                    stop = true;
                    break;
                }
                output.push(token);
                sampler.accept(token).map_err(|e| e.to_string())?;
                sequence.push(token);
                position += 1;
                if !on_token(token) || output.len() >= max_new_tokens {
                    stop = true;
                    break;
                }
            }
            if stop || step.hit_eos {
                break;
            }
            next = step.next_token;
        }
        Ok(output)
    }

    /// One step (`VerifyStep` with batched verification): a single target
    /// pass over the whole block.
    pub fn verify_step(
        exec: &mut SpecExec<'_>,
        request: &mut StepRequest<'_>,
    ) -> Result<StepResult, String> {
        validate(request)?;
        let mut step = Self::prepare(exec, request, false, false)?;
        if let Some(immediate) = step.immediate.take() {
            return Ok(immediate);
        }
        let mut items = [VerifyRequest {
            session: &mut *request.session,
            tokens: &step.inputs,
            position: step.position,
            capture_logits: false,
        }];
        let verification = exec.model.verify_batch(&mut items)?.remove(0);
        let chunk = Chunk {
            predictions: verification.tokens,
            hidden: request.session.verification_hidden().to_vec(),
        };
        Self::process_chunk(exec, request, &mut step, chunk)?
            .ok_or_else(|| err("complete verification still requires target rows"))
    }

    /// Steps of one to eight sessions (`VerifyBatch`). Proposals are drawn
    /// together; verification passes run until every step completes, a
    /// request with low acceptance two rows at a time while four or more
    /// requests remain.
    pub fn verify_batch(
        exec: &mut SpecExec<'_>,
        requests: &mut [StepRequest<'_>],
    ) -> Result<Vec<StepResult>, String> {
        if let [request] = requests {
            return Ok(vec![Self::verify_step(exec, request)?]);
        }
        for request in requests.iter() {
            validate(request)?;
        }
        let mut prepared = Vec::with_capacity(requests.len());
        for request in requests.iter_mut() {
            prepared.push(Self::prepare(exec, request, true, true)?);
        }
        let drafting: Vec<usize> = (0..prepared.len())
            .filter(|&index| prepared[index].max_draft_tokens > 0)
            .collect();
        if !drafting.is_empty() {
            let proposals = {
                let mut batch = Vec::with_capacity(drafting.len());
                let penalties: Vec<DraftPenalties> = requests
                    .iter()
                    .zip(prepared.iter())
                    .filter(|(_, step)| step.max_draft_tokens > 0)
                    .map(|(request, step)| draft_penalties(step, request))
                    .collect();
                let mut penalties = penalties.iter();
                for (request, step) in requests.iter_mut().zip(prepared.iter_mut()) {
                    if step.max_draft_tokens == 0 {
                        continue;
                    }
                    let temperature = request.sampler.config().temperature;
                    let sampling = if step.random {
                        step.working
                            .as_mut()
                            .map(|working| (temperature, working.mutable_rng_state()))
                    } else {
                        None
                    };
                    batch.push(ProposalRequest {
                        draft: &mut request.verifier.draft,
                        tokens: request.sequence,
                        position: request.position,
                        max_tokens: step.max_draft_tokens,
                        sampling,
                        penalties: penalties.next(),
                    });
                }
                Draft::propose_batch(&mut exec.draft(), &mut batch)?
            };
            if proposals.len() != drafting.len() {
                return Err("draft returned an incomplete proposal batch".into());
            }
            for (&index, proposal) in drafting.iter().zip(proposals) {
                prepared[index].proposal = proposal;
                Self::prepare_proposal(exec, &mut requests[index], &mut prepared[index], true)?;
            }
        }
        let mut results: Vec<Option<StepResult>> = vec![None; requests.len()];
        let mut indices = Vec::with_capacity(requests.len());
        for (index, step) in prepared.iter_mut().enumerate() {
            if let Some(immediate) = step.immediate.take() {
                results[index] = Some(immediate);
                continue;
            }
            indices.push(index);
            let stats = requests[index].verifier.stats;
            if stats.draft_tokens > 0 && stats.acceptance_rate() < 0.5 {
                step.chunk_limit = 2;
            }
        }
        while !indices.is_empty() {
            let mut spans = Vec::with_capacity(indices.len());
            for &index in &indices {
                let step = &mut prepared[index];
                // Small surviving groups cannot amortize another pass.
                if indices.len() < 4 {
                    step.chunk_limit = usize::MAX;
                }
                let begin = step.verified_rows;
                spans.push(
                    begin
                        ..step
                            .inputs
                            .len()
                            .min(begin.saturating_add(step.chunk_limit)),
                );
            }
            let rows: usize = spans.iter().map(|span| span.len()).sum();
            // One pass for two to eight sessions within the prompt batch
            // (`QwenGpuSpeculativeTarget::ForwardVerificationBatch`), else
            // one per session.
            let batched = indices.len() >= 2 && rows <= exec.model.max_prompt_batch() as usize;
            let mut sessions: Vec<Option<&mut Session>> = requests
                .iter_mut()
                .map(|request| Some(&mut *request.session))
                .collect();
            let mut items: Vec<VerifyRequest<'_>> = indices
                .iter()
                .zip(&spans)
                .map(|(&index, span)| {
                    let step = &prepared[index];
                    VerifyRequest {
                        session: sessions[index].take().expect("sessions are distinct"),
                        tokens: &step.inputs[span.clone()],
                        position: step.position + span.start as u32,
                        capture_logits: false,
                    }
                })
                .collect();
            let verified = if batched {
                exec.model.verify_batch(&mut items)?
            } else {
                let mut verified = Vec::with_capacity(items.len());
                for item in items.chunks_mut(1) {
                    verified.extend(exec.model.verify_batch(item)?);
                }
                verified
            };
            drop(items);
            drop(sessions);
            let physical_width = if batched { indices.len() } else { 1 };
            let mut remaining = Vec::with_capacity(indices.len());
            for (&index, verification) in indices.iter().zip(verified) {
                let step = &mut prepared[index];
                step.physical_width = step.physical_width.max(physical_width);
                let chunk = Chunk {
                    predictions: verification.tokens,
                    hidden: requests[index].session.verification_hidden().to_vec(),
                };
                match Self::process_chunk(exec, &mut requests[index], step, chunk)? {
                    Some(mut result) => {
                        result.physical_width = step.physical_width;
                        results[index] = Some(result);
                    }
                    None => remaining.push(index),
                }
            }
            indices = remaining;
        }
        Ok(results.into_iter().map(Option::unwrap_or_default).collect())
    }

    /// `PrepareStep`.
    fn prepare(
        exec: &mut SpecExec<'_>,
        request: &mut StepRequest<'_>,
        defer_target_only: bool,
        defer_proposal: bool,
    ) -> Result<Prepared, String> {
        if request.max_emitted == 0 {
            return Err("speculative verification must emit at least one token".into());
        }
        let config = request.sampler.config().clone();
        let random = config.uses_random_sampling();
        // A reasoning budget can force a token the target never predicts, so
        // its greedy rows go through the sampler like penalties and grammars
        //.
        let sampled = random
            || config.penalties_enabled()
            || config.constraint.is_some()
            || config.reasoning_budget.is_some();
        let mut step = Prepared {
            position: request.position,
            stops: request.stops,
            random,
            sampled,
            working: sampled.then(|| request.sampler.clone()),
            chunk_limit: usize::MAX,
            physical_width: 1,
            ..Prepared::default()
        };
        step.max_draft_tokens = if request.max_emitted > 1 {
            request.verifier.draft_length.min(request.max_emitted - 1)
        } else {
            0
        };
        if step.max_draft_tokens == 0 {
            Self::prepare_target_only(exec, request, &mut step, defer_target_only)?;
            return Ok(step);
        }
        if defer_proposal {
            return Ok(step);
        }
        let penalties = draft_penalties(&step, request);
        let working = step.working.as_mut().unwrap_or(request.sampler);
        let sampling = random.then(|| (config.temperature, working.mutable_rng_state()));
        step.proposal = request.verifier.draft.propose(
            &mut exec.draft(),
            request.sequence,
            request.position,
            step.max_draft_tokens,
            sampling,
            Some(&penalties),
        )?;
        Self::prepare_proposal(exec, request, &mut step, defer_target_only)?;
        Ok(step)
    }

    /// `PrepareTargetOnlyStep`.
    fn prepare_target_only(
        exec: &mut SpecExec<'_>,
        request: &mut StepRequest<'_>,
        step: &mut Prepared,
        defer: bool,
    ) -> Result<(), String> {
        if defer {
            step.target_only = true;
            step.inputs = vec![request.current_token];
            return Ok(());
        }
        let mut next = request.verifier.advance_committed_token(
            exec,
            request.session,
            request.current_token,
            request.position,
        )?;
        if let Some(working) = step.working.as_mut() {
            if !working
                .can_select_argmax(next, false)
                .map_err(|e| e.to_string())?
            {
                next = exec.model.sample(request.session, None, working)?;
            }
            request.sampler.set_rng_state(working.rng_state());
        }
        let logits = if request.verifier.retain_frontier_logits {
            exec.model.logits(request.session)?
        } else {
            Vec::new()
        };
        let verifier = &mut *request.verifier;
        verifier.stats.verification_steps += 1;
        verifier.stats.emitted_tokens += 1;
        step.immediate = Some(StepResult {
            emitted: vec![next],
            next_token: next,
            next_token_logits: logits,
            hit_eos: request.stops.contains(next),
            physical_width: 1,
            ..StepResult::default()
        });
        Ok(())
    }

    /// `PrepareProposalVerification`.
    fn prepare_proposal(
        exec: &mut SpecExec<'_>,
        request: &mut StepRequest<'_>,
        step: &mut Prepared,
        defer_target_only: bool,
    ) -> Result<(), String> {
        let drafts = step.proposal.tokens.len();
        if drafts == 0 {
            return Self::prepare_target_only(exec, request, step, defer_target_only);
        }
        if drafts > step.max_draft_tokens as usize {
            return Err("draft backend returned a malformed proposal".into());
        }
        if step.random {
            let proposal = &step.proposal;
            let per_token = proposal
                .candidate_ids
                .len()
                .checked_div(drafts)
                .unwrap_or(0);
            if per_token == 0
                || proposal.candidate_ids.len() != drafts * per_token
                || proposal.candidate_probabilities.len() != proposal.candidate_ids.len()
            {
                return Err("draft backend returned a malformed sampled proposal".into());
            }
            let vocab = exec.model.config().vocab_size;
            step.token_probabilities = Vec::with_capacity(drafts);
            for row in 0..drafts {
                let ids = &proposal.candidate_ids[row * per_token..(row + 1) * per_token];
                let probabilities =
                    &proposal.candidate_probabilities[row * per_token..(row + 1) * per_token];
                let (mut row_sum, mut token_probability) = (0.0f64, 0.0f64);
                for (index, (&id, &probability)) in ids.iter().zip(probabilities).enumerate() {
                    if !probability.is_finite()
                        || !(0.0..=1.0).contains(&probability)
                        || id >= vocab
                        || ids[..index].contains(&id)
                    {
                        return Err("draft backend returned an invalid sampled candidate".into());
                    }
                    row_sum += f64::from(probability);
                    if id == proposal.tokens[row] {
                        token_probability += f64::from(probability);
                    }
                }
                if (row_sum - 1.0).abs() > 1e-5 || token_probability <= 0.0 {
                    return Err("draft backend returned an unnormalized sampled proposal".into());
                }
                step.token_probabilities.push(token_probability);
            }
        }
        step.inputs = Vec::with_capacity(drafts + 1);
        step.inputs.push(request.current_token);
        step.inputs.extend(&step.proposal.tokens);
        exec.model.save_state(request.session, request.position)
    }

    /// `ProcessVerificationChunk`: `Some` once the step is complete.
    fn process_chunk(
        exec: &mut SpecExec<'_>,
        request: &mut StepRequest<'_>,
        step: &mut Prepared,
        chunk: Chunk,
    ) -> Result<Option<StepResult>, String> {
        let drafts = step.proposal.tokens.len();
        let (position, stops) = (step.position, step.stops);
        let chunk_begin = step.verified_rows;
        let chunk_rows = step.chunk_limit.min(step.inputs.len() - chunk_begin);
        let chunk_end = chunk_begin + chunk_rows;
        if (!step.sampled || step.target_only) && chunk.predictions.len() != chunk_rows {
            return Err("target executor returned an incomplete verification chunk".into());
        }
        let width = exec.draft_model.config.target_features();
        if chunk.hidden.len() != chunk_rows * width {
            return Err("target executor returned incomplete verification hidden states".into());
        }
        let session = &mut *request.session;
        let verifier = &mut *request.verifier;
        if step.target_only {
            exec.model.finish_verification(session)?;
            let prediction = chunk.predictions[0];
            verifier
                .draft
                .append(
                    &mut exec.draft(),
                    &chunk.hidden,
                    step.inputs.len(),
                    position,
                )
                .map_err(|_| err("draft failed to append committed target token"))?;
            let mut next = prediction;
            if let Some(working) = step.working.as_mut() {
                next = select_row(exec.model, session, &chunk.predictions, 0, working)?;
            }
            let logits = if verifier.retain_frontier_logits {
                exec.model.verification_logits(session, 0)?
            } else {
                Vec::new()
            };
            if let Some(working) = &step.working {
                request.sampler.set_rng_state(working.rng_state());
            }
            verifier.stats.verification_steps += 1;
            verifier.stats.emitted_tokens += 1;
            return Ok(Some(StepResult {
                emitted: vec![next],
                next_token: next,
                next_token_logits: logits,
                hit_eos: stops.contains(next),
                physical_width: 1,
                ..StepResult::default()
            }));
        }
        let per_token = step
            .proposal
            .candidate_ids
            .len()
            .checked_div(drafts)
            .unwrap_or(0);
        let mut correction = 0u32;
        while step.accepted < drafts.min(chunk_end) {
            let row = step.accepted;
            let local = row - chunk_begin;
            let token = step.proposal.tokens[row];
            if !step.sampled {
                correction = chunk.predictions[local];
                if correction != token || stops.contains(correction) {
                    break;
                }
                step.accepted += 1;
                continue;
            }
            let Some(working) = step.working.as_mut() else {
                return Err("sampled verification lost its sampler".into());
            };
            if !step.random {
                // Penalties change the target argmax after each committed
                // token; verify against that same evolving history.
                let target = select_row(exec.model, session, &chunk.predictions, local, working)?;
                if target != token || stops.contains(target) {
                    correction = target;
                    break;
                }
                working.accept(target).map_err(|e| e.to_string())?;
                step.accepted += 1;
                continue;
            }
            let ids = &step.proposal.candidate_ids[row * per_token..(row + 1) * per_token];
            let probabilities =
                &step.proposal.candidate_probabilities[row * per_token..(row + 1) * per_token];
            let (drawn, accepted) = exec.model.verify_sampled_token(
                session,
                local,
                token,
                ids,
                probabilities,
                step.token_probabilities[row],
                working,
            )?;
            if accepted {
                if stops.contains(token) {
                    correction = token;
                    break;
                }
                step.accepted += 1;
                continue;
            }
            correction = drawn;
            break;
        }
        let accepted = step.accepted;
        let complete = accepted < chunk_end;
        let committed = if complete { accepted + 1 } else { chunk_end };
        // Only committed rows reach the draft, in their order.
        for row in chunk_begin..committed {
            let local = row - chunk_begin;
            verifier
                .draft
                .update_hidden(&chunk.hidden[local * width..(local + 1) * width])?;
        }
        step.verified_rows = chunk_end;
        if !complete {
            return Ok(None);
        }
        exec.model.finish_verification(session)?;
        let correction_row = accepted - chunk_begin;
        if accepted == drafts {
            correction = match step.working.as_mut() {
                None => chunk.predictions[correction_row],
                Some(working) => select_row(
                    exec.model,
                    session,
                    &chunk.predictions,
                    correction_row,
                    working,
                )?,
            };
        }
        let frontier = if verifier.retain_frontier_logits {
            exec.model.verification_logits(session, correction_row)?
        } else {
            Vec::new()
        };
        if chunk_end > committed {
            exec.model.restore_state(session)?;
            exec.model
                .commit_verified(session, &step.inputs[..committed], position)?;
        }
        let mut emitted = step.proposal.tokens[..accepted].to_vec();
        emitted.push(correction);
        let hit_eos = emitted.iter().any(|&t| stops.contains(t));
        verifier.stats.draft_tokens += drafts as u64;
        verifier.stats.accepted_tokens += accepted as u64;
        verifier.stats.verification_steps += 1;
        verifier.stats.emitted_tokens += emitted.len() as u64;
        if let Some(working) = &step.working {
            request.sampler.set_rng_state(working.rng_state());
        }
        verifier.draft.accept(accepted)?;
        let copied = step.proposal.from_lookup;
        Ok(Some(StepResult {
            emitted,
            next_token: correction,
            next_token_logits: frontier,
            draft_count: drafts,
            accepted_count: accepted,
            hit_eos,
            physical_width: 1,
            lookup_tokens: if copied { drafts } else { 0 },
            lookup_accepted_tokens: if copied { accepted } else { 0 },
        }))
    }

    /// Bytes a snapshot takes (`SnapshotPayloadBytes`).
    pub fn snapshot_payload_bytes(&self, model: &DFlashModel) -> usize {
        self.draft.snapshot_payload_bytes(model)
            + self.rolling_acceptance.len() * 4
            + std::mem::size_of::<Stats>()
            + 4
    }

    /// Captures the verifier on `transfer` (`Snapshot`).
    pub fn snapshot(
        &self,
        transfer: &Stream,
        model: &DFlashModel,
    ) -> Result<VerifierSnapshot, String> {
        Ok(VerifierSnapshot {
            draft: self.draft.snapshot(transfer, model)?,
            stats: self.stats,
            draft_length: self.draft_length,
            rolling_acceptance: self.rolling_acceptance.clone(),
        })
    }

    /// Returns to `snapshot`; the target's hidden capture is set up again
    /// first (`RestoreSnapshot`).
    pub fn restore(
        &mut self,
        exec: &mut SpecExec<'_>,
        st: &mut Session,
        snapshot: &VerifierSnapshot,
    ) -> Result<(), String> {
        Self::capture(exec, st)?;
        self.draft.restore(&exec.draft(), &snapshot.draft)?;
        self.stats = snapshot.stats;
        self.draft_length = snapshot.draft_length;
        self.rolling_acceptance = snapshot.rolling_acceptance.clone();
        Ok(())
    }

    /// `RestorePersistentSnapshot` of a `GSPVER01` payload.
    pub fn restore_persistent(
        &mut self,
        exec: &mut SpecExec<'_>,
        st: &mut Session,
        payload: &[u8],
    ) -> Result<(), String> {
        let header = VerifierHeader::parse(
            payload,
            VerifierLimits {
                rolling_window: ROLLING_WINDOW,
                min_draft_tokens: self.min_draft_tokens,
                max_draft_tokens: self.max_draft_tokens,
            },
        )?;
        Self::capture(exec, st)?;
        self.draft
            .restore_persistent(&exec.draft(), &payload[header.draft()])?;
        self.stats = Stats {
            draft_tokens: header.total_draft_tokens,
            accepted_tokens: header.total_accepted_tokens,
            verification_steps: header.total_verification_steps,
            emitted_tokens: header.total_emitted_tokens,
        };
        self.draft_length = header.current_draft_length;
        self.rolling_acceptance = header.rolling_acceptance;
        Ok(())
    }
}

impl VerifierSnapshot {
    fn header(&self, model: &DFlashModel, history_capacity: u32) -> Result<VerifierHeader, String> {
        Ok(VerifierHeader {
            draft_bytes: self.draft.persistent_bytes(model, history_capacity)?,
            rolling_acceptance: self.rolling_acceptance.clone(),
            total_draft_tokens: self.stats.draft_tokens,
            total_accepted_tokens: self.stats.accepted_tokens,
            total_verification_steps: self.stats.verification_steps,
            total_emitted_tokens: self.stats.emitted_tokens,
            current_draft_length: self.draft_length,
        })
    }

    /// `PersistentPayloadBytes`.
    pub fn persistent_bytes(
        &self,
        model: &DFlashModel,
        history_capacity: u32,
    ) -> Result<usize, String> {
        self.header(model, history_capacity)?.total_bytes()
    }

    /// `SerializePersistent` into exactly sized `destination`.
    pub fn serialize(
        &self,
        transfer: &Stream,
        model: &DFlashModel,
        history_capacity: u32,
        destination: &mut [u8],
    ) -> Result<(), String> {
        let header = self.header(model, history_capacity)?;
        header.write(destination)?;
        self.draft.serialize(
            transfer,
            model,
            history_capacity,
            &mut destination[header.draft()],
        )
    }
}
