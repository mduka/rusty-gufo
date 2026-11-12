//! Greedy DFlash2 speculative decoding with batched verification, a port of
//! the `SpeculativeVerifier` path that `gufo bench` drives
//! (`core/speculative/speculative_verifier.cpp`): the draft proposes a
//! block, the target verifies it with decode arithmetic in one pass, the
//! longest agreeing prefix plus the target's correction is emitted, and a
//! rejected tail is rolled back by replaying the recurrent state. The emitted
//! tokens are those of greedy autoregressive decoding. The draft keeps a
//! fixed block (its own policy shortens it). The MTP head runs on the
//! serving verifier (`Verifier<MtpBackend>`) in every command.

use crate::dflash::{DFlash, DFlashState};
use crate::engine::Engine;

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

pub struct Speculative {
    draft: Box<DFlash>,
    stats: Stats,
    /// Drafts per step before the draft clamps them (`current_draft_length_`).
    draft_length: u32,
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
            draft: Box::new(draft),
            stats: Stats::default(),
            draft_length,
        }
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The DFlash2 draft.
    pub fn dflash(&self) -> &DFlash {
        &self.draft
    }

    /// `Reset`: statistics and the draft.
    fn reset(&mut self) {
        self.stats = Stats::default();
        self.draft.reset();
    }

    /// Hidden-row capture for the draft: DFlash2's taps.
    fn capture(&self, target: &mut Engine) -> Result<(), String> {
        let taps = self.draft.target_layers().to_vec();
        target.set_device_rows(false);
        target.set_hidden_capture(true, &taps)
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
        self.draft.prime(target.prompt_hidden()?, prompt.len())?;
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
        self.draft
            .append(target.prompt_hidden()?, tokens.len(), position)?;
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
            self.draft.propose(sequence, position, max_draft)?
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
        let width = target.verification_hidden().len() / inputs.len();
        for row in 0..committed {
            self.draft
                .update_hidden(&target.verification_hidden()[row * width..(row + 1) * width])?;
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
        self.draft.accept(accepted)?;
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
        self.draft.append(&target.last_hidden()?, 1, position)?;
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

    pub fn snapshot(&self) -> Result<SpeculativeState, String> {
        Ok(SpeculativeState {
            draft: self.draft.snapshot()?,
            stats: self.stats,
        })
    }

    /// Restores the verifier; like C++, the target's hidden capture is set up
    /// again (clearing its host rows) before the draft state returns.
    pub fn restore(&mut self, target: &mut Engine, state: &SpeculativeState) -> Result<(), String> {
        self.capture(target)?;
        self.draft.restore(&state.draft)?;
        self.stats = state.stats;
        Ok(())
    }
}
