//! Greedy speculative decoding with batched verification, a port of the
//! `SpeculativeVerifier` path that `gufo bench` and, with the MTP head,
//! `gufo prompt` drive (`core/speculative/speculative_verifier.cpp`): the
//! draft proposes a block, the target verifies it with decode arithmetic in
//! one pass, the longest agreeing prefix plus the target's correction is
//! emitted, and a rejected tail is rolled back by replaying the recurrent
//! state. The emitted tokens are those of greedy autoregressive decoding.
//! The DFlash2 draft keeps a fixed block (its own policy shortens it); the
//! MTP head drafts an adaptive length and nothing for sampled requests.

use std::collections::VecDeque;
use std::sync::Arc;

use gufo_core::sampling::{SamplerState, SamplingConfig};

use crate::dflash::{DFlash, DFlashState};
use crate::engine::Engine;
use crate::mtp::{MtpContext, MtpDraft, MtpModel, Rows};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub draft_tokens: usize,
    pub accepted_tokens: usize,
    pub verification_steps: usize,
    pub emitted_tokens: usize,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub emitted: Vec<u32>,
    pub accepted: usize,
    pub drafted: usize,
    pub next_token: u32,
    pub hit_eos: bool,
}

enum Draft {
    DFlash(Box<DFlash>),
    Mtp {
        model: Arc<MtpModel>,
        draft: Box<MtpDraft>,
    },
}

/// The adaptive draft length (`UpdateAdaptiveDraftLength` with
/// `rolling_window` 16 and `target_acceptance_rate` 0.7).
struct Adaptive {
    min: u32,
    max: u32,
    initial: u32,
    rolling: VecDeque<f32>,
}

const ROLLING_WINDOW: usize = 16;
const TARGET_ACCEPTANCE: f32 = 0.70;

impl Adaptive {
    /// `ConfigureAdaptiveDraftPolicy`.
    fn new(min: u32, max: u32, initial: u32) -> Self {
        let max = max.max(1);
        let min = min.clamp(1, max);
        Self {
            min,
            max,
            initial: initial.clamp(min, max),
            rolling: VecDeque::new(),
        }
    }

    fn update(&mut self, length: &mut u32, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        self.rolling.push_back(accepted as f32 / drafted as f32);
        if self.rolling.len() > ROLLING_WINDOW {
            self.rolling.pop_front();
        }
        let average =
            self.rolling.iter().fold(0.0f32, |sum, &rate| sum + rate) / self.rolling.len() as f32;
        if average > TARGET_ACCEPTANCE && *length < self.max {
            *length += 1;
        } else if average < TARGET_ACCEPTANCE * 0.5 && *length > self.min {
            *length -= 1;
        }
    }
}

pub struct Speculative {
    draft: Draft,
    stats: Stats,
    /// Drafts per step before the draft clamps them (`current_draft_length_`).
    draft_length: u32,
    adaptive: Option<Adaptive>,
}

/// Verifier state at a committed boundary (`SpeculativeVerifierSnapshot`).
pub struct SpeculativeState {
    draft: DFlashState,
    stats: Stats,
}

impl Speculative {
    pub fn new(draft: DFlash) -> Self {
        let draft_length = draft.max_draft_tokens();
        Self {
            draft: Draft::DFlash(Box::new(draft)),
            stats: Stats::default(),
            draft_length,
            adaptive: None,
        }
    }

    /// A verifier over the MTP head with the adaptive draft length between
    /// `min_draft_tokens` and `max_draft_tokens`, starting at the maximum, as
    /// the C++ commands configure it.
    pub fn mtp(
        model: Arc<MtpModel>,
        draft: MtpDraft,
        min_draft_tokens: u32,
        max_draft_tokens: u32,
    ) -> Self {
        let adaptive = Adaptive::new(min_draft_tokens, max_draft_tokens, max_draft_tokens);
        Self {
            draft: Draft::Mtp {
                model,
                draft: Box::new(draft),
            },
            stats: Stats::default(),
            draft_length: adaptive.initial,
            adaptive: Some(adaptive),
        }
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The DFlash2 draft, if this verifier has one.
    pub fn dflash(&self) -> Option<&DFlash> {
        match &self.draft {
            Draft::DFlash(draft) => Some(draft.as_ref()),
            Draft::Mtp { .. } => None,
        }
    }

    /// `Reset`: statistics, the adaptive length and the draft.
    fn reset(&mut self) {
        self.stats = Stats::default();
        if let Some(adaptive) = &mut self.adaptive {
            adaptive.rolling.clear();
            self.draft_length = adaptive.initial;
        }
        match &mut self.draft {
            Draft::DFlash(draft) => draft.reset(),
            Draft::Mtp { draft, .. } => draft.reset(),
        }
    }

    /// Hidden-row capture for the draft: DFlash2's taps, or the final row,
    /// whose verified copies the MTP head takes on the device.
    fn capture(&self, target: &mut Engine) -> Result<(), String> {
        match &self.draft {
            Draft::DFlash(draft) => {
                let taps = draft.target_layers().to_vec();
                target.set_device_rows(false);
                target.set_hidden_capture(true, &taps)
            }
            Draft::Mtp { .. } => {
                target.set_device_rows(true);
                target.set_hidden_capture(true, &[])
            }
        }
    }

    /// Resets both models, prefills `prompt` and primes the draft with its
    /// features; returns the first target token (`Prime`).
    pub fn prime(
        &mut self,
        target: &mut Engine,
        prompt: &[u32],
        compute_logits: bool,
    ) -> Result<u32, String> {
        if prompt.is_empty() {
            return Err("speculative prompt must not be empty".into());
        }
        self.reset();
        target.reset()?;
        self.capture(target)?;
        let first = target
            .forward_prompt(prompt, 0, compute_logits)?
            .unwrap_or(0);
        match &mut self.draft {
            Draft::DFlash(draft) => draft.prime(target.prompt_hidden()?, prompt.len())?,
            Draft::Mtp { model, draft } => {
                let ctx = MtpContext {
                    stream: target.model().stream(),
                    model,
                    vision: Some(target.vision()),
                };
                draft
                    .prime(&ctx, prompt, target.prompt_hidden()?)
                    .map_err(|_| "draft backend failed to prime target context".to_string())?;
            }
        }
        Ok(first)
    }

    /// Prefills `tokens` at `position` behind the committed sequence and
    /// appends their features to the draft (`ExtendPrompt`).
    pub fn extend_prompt(
        &mut self,
        target: &mut Engine,
        tokens: &[u32],
        position: u32,
        compute_logits: bool,
    ) -> Result<u32, String> {
        self.capture(target)?;
        let next = target
            .forward_prompt(tokens, position, compute_logits)?
            .unwrap_or(0);
        match &mut self.draft {
            Draft::DFlash(draft) => {
                draft.append(target.prompt_hidden()?, tokens.len(), position)?
            }
            Draft::Mtp { model, draft } => {
                let ctx = MtpContext {
                    stream: target.model().stream(),
                    model,
                    vision: Some(target.vision()),
                };
                draft
                    .append(&ctx, tokens, Rows::Host(target.prompt_hidden()?), position)
                    .map_err(|_| "draft failed to append target prompt features".to_string())?;
            }
        }
        Ok(next)
    }

    /// One step from `current` at `position`, emitting at most
    /// `max_emitted` tokens (`VerifyStep`). `sequence` is the committed
    /// history ending with `current`; with all `position + 1` tokens the
    /// DFlash2 draft can copy repeated passages from it. `eos` and the
    /// target's stop tokens end a step.
    pub fn verify_step(
        &mut self,
        target: &mut Engine,
        sequence: &[u32],
        current: u32,
        position: u32,
        eos: u32,
        max_emitted: u32,
    ) -> Result<Step, String> {
        if sequence.last() != Some(&current) {
            return Err("speculative sequence must end with the current token".into());
        }
        let stops = target.weights().tokenizer.stop_tokens();
        self.step(
            target,
            sequence,
            position,
            |token| token == eos || stops.contains(token),
            max_emitted,
        )
    }

    fn step(
        &mut self,
        target: &mut Engine,
        sequence: &[u32],
        position: u32,
        is_stop: impl Fn(u32) -> bool,
        max_emitted: u32,
    ) -> Result<Step, String> {
        let current = *sequence
            .last()
            .ok_or("speculative sequence must end with the current token")?;
        if max_emitted == 0 {
            return Err("speculative verification must emit at least one token".into());
        }
        let max_draft = if max_emitted > 1 {
            self.draft_length.min(max_emitted - 1)
        } else {
            0
        };
        let proposal = if max_draft == 0 {
            Vec::new()
        } else {
            match &mut self.draft {
                Draft::DFlash(draft) => draft.propose(sequence, position, max_draft)?,
                Draft::Mtp { model, draft } => {
                    let ctx = MtpContext {
                        stream: target.model().stream(),
                        model,
                        vision: Some(target.vision()),
                    };
                    draft.propose(&ctx, current, position, max_draft)?
                }
            }
        };
        if proposal.is_empty() {
            return self.target_only(target, current, position, is_stop);
        }
        let mut inputs = Vec::with_capacity(proposal.len() + 1);
        inputs.push(current);
        inputs.extend(&proposal);
        target.save_state(position)?;
        let predictions = target.verify(&inputs, position)?;
        let mut accepted = 0;
        let mut correction = 0;
        while accepted < proposal.len() {
            correction = predictions[accepted];
            if correction != proposal[accepted] || is_stop(correction) {
                break;
            }
            accepted += 1;
        }
        let committed = accepted + 1;
        // Only committed rows reach the draft, in their order.
        match &mut self.draft {
            Draft::DFlash(draft) => {
                let width = target.verification_hidden().len() / inputs.len();
                for row in 0..committed {
                    draft.update_hidden(
                        &target.verification_hidden()[row * width..(row + 1) * width],
                    )?;
                }
            }
            Draft::Mtp { model, draft } => {
                let ctx = MtpContext {
                    stream: target.model().stream(),
                    model,
                    vision: Some(target.vision()),
                };
                let rows = target.verification_rows()?;
                let width = rows.len() / inputs.len();
                for row in 0..committed {
                    let features = rows.range(row * width..(row + 1) * width);
                    draft.update_hidden(&ctx, Rows::Device(features))?;
                }
            }
        }
        target.finish_verification()?;
        if accepted == proposal.len() {
            correction = predictions[accepted];
        }
        if inputs.len() > committed {
            target.restore_state()?;
            target.commit_verified(&inputs[..committed], position)?;
        }
        let mut emitted = proposal[..accepted].to_vec();
        emitted.push(correction);
        let hit_eos = emitted.iter().any(|&t| is_stop(t));
        self.stats.draft_tokens += proposal.len();
        self.stats.accepted_tokens += accepted;
        self.stats.verification_steps += 1;
        self.stats.emitted_tokens += emitted.len();
        if let Some(adaptive) = &mut self.adaptive {
            adaptive.update(&mut self.draft_length, accepted, proposal.len());
        }
        match &mut self.draft {
            Draft::DFlash(draft) => draft.accept(accepted)?,
            Draft::Mtp { model, draft } => {
                let ctx = MtpContext {
                    stream: target.model().stream(),
                    model,
                    vision: Some(target.vision()),
                };
                draft.accept(&ctx, &proposal[..accepted])?;
            }
        }
        Ok(Step {
            emitted,
            accepted,
            drafted: proposal.len(),
            next_token: correction,
            hit_eos,
        })
    }

    /// A plain decode step whose features go straight to the draft
    /// (`AdvanceCommittedToken`).
    fn advance(&mut self, target: &mut Engine, current: u32, position: u32) -> Result<u32, String> {
        let next = target
            .forward_token(current, position, true)?
            .ok_or("decode returned no token")?;
        match &mut self.draft {
            Draft::DFlash(draft) => draft.append(&target.last_hidden()?, 1, position)?,
            Draft::Mtp { model, draft } => {
                let ctx = MtpContext {
                    stream: target.model().stream(),
                    model,
                    vision: Some(target.vision()),
                };
                draft
                    .append(
                        &ctx,
                        &[current],
                        Rows::Device(target.final_row()?),
                        position,
                    )
                    .map_err(|_| "draft failed to append committed target token".to_string())?;
            }
        }
        Ok(next)
    }

    fn target_only(
        &mut self,
        target: &mut Engine,
        current: u32,
        position: u32,
        is_stop: impl Fn(u32) -> bool,
    ) -> Result<Step, String> {
        let next = self.advance(target, current, position)?;
        self.stats.verification_steps += 1;
        self.stats.emitted_tokens += 1;
        Ok(Step {
            emitted: vec![next],
            accepted: 0,
            drafted: 0,
            next_token: next,
            hit_eos: is_stop(next),
        })
    }

    /// Generates up to `max_new_tokens` after `prompt` under `sampling`
    /// (`SpeculativeVerifier::Generate`): primes both models, then verifies
    /// drafted blocks until a stop token or the budget. A sampled request
    /// (random sampling or penalties) drafts nothing, as the MTP head has no
    /// sampled proposals: each step decodes one token, samples it and keeps
    /// the head in step. `on_token` sees each emitted token and stops
    /// generation by returning `false`.
    pub fn generate(
        &mut self,
        target: &mut Engine,
        prompt: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingConfig,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, String> {
        let mut output = Vec::new();
        if prompt.is_empty() {
            return Ok(output);
        }
        if matches!(self.draft, Draft::DFlash(_)) {
            return Err("DFlash2 generation runs on the serving verifier".into());
        }
        sampling.validate().map_err(|e| e.to_string())?;
        let mut first = self.prime(target, prompt, true)?;
        let mut sampler = SamplerState::new(sampling.clone(), prompt).map_err(|e| e.to_string())?;
        if !sampling.can_use_unmodified_argmax() {
            first = target.sample(&mut sampler)?;
        }
        let sampled = sampling.uses_random_sampling() || sampling.penalties_enabled();
        let stops = target.weights().tokenizer.stop_tokens();
        if max_new_tokens == 0 || stops.contains(first) {
            return Ok(output);
        }
        let mut position = u32::try_from(prompt.len()).map_err(|_| "prompt exceeds u32")?;
        let mut next = first;
        output.push(first);
        sampler.accept(first).map_err(|e| e.to_string())?;
        if !on_token(first) {
            return Ok(output);
        }
        while output.len() < max_new_tokens {
            let remaining = u32::try_from(max_new_tokens - output.len()).unwrap_or(u32::MAX);
            let step = if sampled {
                // `PrepareTargetOnlyStep` with a working copy of the sampler,
                // of which only the random state carries over.
                let mut working = sampler.clone();
                self.advance(target, next, position)?;
                let token = target.sample(&mut working)?;
                sampler.set_rng_state(working.rng_state());
                self.stats.verification_steps += 1;
                self.stats.emitted_tokens += 1;
                Step {
                    emitted: vec![token],
                    accepted: 0,
                    drafted: 0,
                    next_token: token,
                    hit_eos: stops.contains(token),
                }
            } else {
                // The MTP head drafts from the anchor alone.
                self.step(target, &[next], position, |t| stops.contains(t), remaining)?
            };
            let mut stop = false;
            for &token in &step.emitted {
                if stops.contains(token) {
                    stop = true;
                    break;
                }
                output.push(token);
                sampler.accept(token).map_err(|e| e.to_string())?;
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

    pub fn snapshot(&self) -> Result<SpeculativeState, String> {
        match &self.draft {
            Draft::DFlash(draft) => Ok(SpeculativeState {
                draft: draft.snapshot()?,
                stats: self.stats,
            }),
            Draft::Mtp { .. } => Err("draft backend does not support snapshots".into()),
        }
    }

    /// Restores the verifier; like C++, the target's hidden capture is set up
    /// again (clearing its host rows) before the draft state returns.
    pub fn restore(&mut self, target: &mut Engine, state: &SpeculativeState) -> Result<(), String> {
        self.capture(target)?;
        match &mut self.draft {
            Draft::DFlash(draft) => draft.restore(&state.draft)?,
            Draft::Mtp { .. } => {
                return Err("draft backend does not support snapshot restore".into());
            }
        }
        self.stats = state.stats;
        Ok(())
    }
}
