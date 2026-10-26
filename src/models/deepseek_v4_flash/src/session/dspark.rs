//! DSpark speculative cycles of sessions (`session_dspark_step`,
//! `ds4_sessions_dspark_step_batch` and their policy in `session.cpp`),
//! greedy. A cycle emits the target-known anchor plus the accepted tail the
//! support model drafted after it; accepted verifier prefixes become cache
//! state, and replay is only a fallback when a prefix cannot be committed.
//! Sampled cycles come with the request sampler.

use gufo_deepseek_v4_flash_kernels::Result as GpuResult;

use super::{Session, SessionError, argmax, error, interrupted};
use crate::dspark_model::{MAX_BLOCK, RETRY_TOKENS};
use crate::dspark_policy::cycle_cost;
use crate::dspark_sampler::DsparkSampler;
use crate::engine::Engine;
use crate::graph::{
    BatchItem, DraftItem, VerifyItem, dspark_draft_head_batch, eval_sessions_batch, greedy_accept,
    verify_sessions_batch,
};

/// One request of a DSpark step (`ds4_session_dspark_batch_item`).
pub struct DsparkItem<'a> {
    pub session: &'a mut Session,
    /// At most this many tokens are emitted (up to 32).
    pub max_tokens: usize,
    /// At most this many support tokens are verified.
    pub max_draft_tokens: u32,
    /// The request sampler of a sampled cycle; `None` runs the greedy one.
    pub sampler: Option<&'a mut DsparkSampler>,
    /// EOS ends the request: it is emitted but not evaluated.
    pub stop_at_eos: bool,
    /// The tokens the step emitted.
    pub emitted: Vec<i32>,
}

/// `Session::DsparkStats`: the request's speculative statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DsparkStats {
    pub verifier_rows: u64,
    pub verifier_accepted: u64,
    pub support_drafted: u64,
    pub support_accepted: u64,
    pub positional_accepted: u64,
    pub anchors: u64,
    pub full_blocks: u64,
    pub steps: u64,
    pub skipped: u64,
    pub context_tokens: u32,
}

const SAMPLER_FAILED: &str = "DSpark request sampler failed";

/// The emitted-token cap of a step (`emitted_cap`).
fn emitted_cap(max_tokens: usize) -> u32 {
    max_tokens.min(32) as u32
}

impl Session {
    /// `ds4_session_dspark_stats`.
    pub fn dspark_stats(&self) -> DsparkStats {
        let s = &self.dspark_state;
        DsparkStats {
            verifier_rows: s.drafted,
            verifier_accepted: s.accepted,
            support_drafted: s.support_drafted,
            support_accepted: s.support_accepted,
            positional_accepted: s.positional_accepted,
            anchors: s.anchors,
            full_blocks: s.full_blocks,
            steps: s.steps,
            skipped: s.skipped,
            context_tokens: self.graph.dspark_context_len(),
        }
    }

    /// `session_dspark_seed_pending`: injects the latest captured batch (a
    /// prompt or a resumed suffix) into the support ring when it is needed
    /// to cover `required_context`.
    pub(super) fn seed_pending(&mut self, engine: &mut Engine, required_context: u32) -> bool {
        if self.graph.dspark_block_size() == 0 {
            return false;
        }
        let context = self.graph.dspark_context_len();
        if context >= required_context {
            return true;
        }
        let Some((start, rows)) = self.graph.dspark_batch_capture() else {
            return false;
        };
        if rows == 0 || start > context || start + rows < required_context {
            return false;
        }
        if self.dspark_inject(engine, start, rows).is_err() {
            return false;
        }
        self.graph.dspark_context_len() >= required_context
    }

    /// `session_dspark_note_cycle`: one request alone keeps the four-probe,
    /// 64-token acceptance backoff; concurrent requests use their recent
    /// cost window instead.
    fn note_cycle(&mut self, concurrent_batch: bool) {
        if concurrent_batch {
            return;
        }
        const EARLY_CYCLES: u64 = 4;
        const EARLY_REJECT_RATE: f64 = 0.25;
        const WARMUP_CYCLES: u64 = 8;
        const BREAK_EVEN_ACCEPTANCE: f64 = 0.48;
        let s = &mut self.dspark_state;
        let drafted = s.support_drafted - s.policy_start_drafted;
        let accepted = s.support_accepted - s.policy_start_accepted;
        let steps = s.steps - s.policy_start_steps;
        if drafted == 0 || steps < EARLY_CYCLES {
            return;
        }
        let acceptance = accepted as f64 / drafted as f64;
        if steps < WARMUP_CYCLES && acceptance >= EARLY_REJECT_RATE {
            return;
        }
        s.skip_remaining = if acceptance >= BREAK_EVEN_ACCEPTANCE {
            0
        } else {
            RETRY_TOKENS
        };
    }

    /// `session_dspark_cycle_cost`.
    fn cycle_cost(&self, tail: u32, concurrency: usize) -> u32 {
        cycle_cost(self.checkpoint.len() as u32, tail, concurrency)
    }

    /// `session_dspark_profitable_tail`: the tail with the lowest cost per
    /// emitted token under full acceptance; 0 for ordinary decoding.
    fn profitable_tail(&self, maximum: u32, concurrency: usize) -> u32 {
        let (mut best_tail, mut best_cost) = (0u32, 100u32);
        for tail in 1..=maximum.min(3) {
            let cost = self.cycle_cost(tail, concurrency);
            if cost * (best_tail + 1) < best_cost * (tail + 1) {
                best_tail = tail;
                best_cost = cost;
            }
        }
        best_tail
    }

    /// `session_dspark_use_concurrency`: a new cohort size restarts the
    /// policy window.
    fn use_concurrency(&mut self, concurrency: usize) {
        let s = &mut self.dspark_state;
        if s.policy_concurrency as usize == concurrency {
            return;
        }
        s.policy_concurrency = concurrency as u32;
        s.policy_start_drafted = s.support_drafted;
        s.policy_start_accepted = s.support_accepted;
        s.policy_start_steps = s.steps;
        s.concurrent_width = 3;
        s.probe_cycles = 0;
        s.probe_emitted = 0;
        s.probe_cost = 0;
        s.skip_remaining = 0;
    }

    /// `session_dspark_note_concurrent_cycle`.
    fn note_concurrent_cycle(&mut self, drafted: u32, accepted: u32, concurrency: usize) {
        if drafted == 0 {
            return;
        }
        // Two requests build their adaptive tail before a cold start counts.
        if concurrency <= 2 && self.dspark_state.steps - self.dspark_state.policy_start_steps <= 4 {
            return;
        }
        let cost = self.cycle_cost(drafted, concurrency);
        let s = &mut self.dspark_state;
        // One probe suffices when even full acceptance cannot repay the
        // cycle; a bounded retry lets a later cohort resume drafting.
        if cost >= 100 * (drafted + 1) {
            s.skip_remaining = RETRY_TOKENS;
            s.probe_cycles = 0;
            s.probe_emitted = 0;
            s.probe_cost = 0;
            return;
        }
        s.probe_cycles += 1;
        s.probe_emitted += 1 + accepted;
        s.probe_cost += cost;
        if s.probe_cycles < 4 {
            return;
        }
        if 100 * s.probe_emitted < s.probe_cost {
            s.skip_remaining = RETRY_TOKENS;
        }
        s.probe_cycles = 0;
        s.probe_emitted = 0;
        s.probe_cost = 0;
    }

    /// `session_dspark_adaptive_width`.
    fn adaptive_width(&self, maximum: u32) -> u32 {
        if maximum == 0 {
            return 0;
        }
        self.dspark_state.concurrent_width.clamp(1, maximum)
    }

    /// `session_dspark_usable_tail`: the tail that fits the width, the
    /// emitted-token cap and the context.
    fn usable_tail(&self, maximum: u32, emitted_cap: u32) -> u32 {
        let room = self.context_size as i64 - self.checkpoint.len() as i64;
        if emitted_cap <= 1 || room <= 2 {
            return 0;
        }
        self.adaptive_width(maximum)
            .min(emitted_cap - 1)
            .min((room - 2) as u32)
    }

    /// `session_dspark_update_adaptive_width`: one wider after a full
    /// block, one narrower after accepting less than half.
    fn update_adaptive_width(&mut self, drafted: u32, accepted: u32, maximum: u32) {
        if drafted == 0 || maximum == 0 {
            return;
        }
        let previous = self.adaptive_width(maximum);
        let next = if accepted >= drafted {
            (previous + 1).min(maximum)
        } else if accepted * 2 < drafted {
            previous.saturating_sub(1).max(1)
        } else {
            previous
        };
        self.dspark_state.concurrent_width = next;
    }

    /// `session_dspark_anchor`: the target-known token that starts the next
    /// block: the frontier argmax for greedy cycles; for sampled ones the
    /// token the previous verification rejected (already drawn, so the RNG
    /// does not advance again) or a draw from the frontier logits.
    fn anchor(
        &mut self,
        sampler: Option<&mut DsparkSampler>,
        vocabulary: usize,
    ) -> Result<i32, SessionError> {
        let Some(sampler) = sampler else {
            return Ok(argmax(&self.logits[..vocabulary]));
        };
        let pending = std::mem::replace(&mut self.dspark_state.pending_anchor, -1);
        if pending >= 0 {
            sampler.accept(pending).map_err(|_| error(SAMPLER_FAILED))?;
            return Ok(pending);
        }
        sampler
            .sample(&self.logits[..vocabulary])
            .map_err(|_| error(SAMPLER_FAILED))
    }

    /// `ds4_session_dspark_take_pending_anchor`: the token a sampled cycle
    /// drew but has not emitted, or -1; an exact target step emits it
    /// instead of sampling again.
    pub fn take_pending_dspark_token(&mut self) -> i32 {
        std::mem::replace(&mut self.dspark_state.pending_anchor, -1)
    }

    /// `session_dspark_accept_prefix`: the accepted prefix of a verified
    /// block. Greedy requests accept while the target argmax reproduces the
    /// draft; sampled ones verify each row (p/q, or a point mass) and keep
    /// the first rejected draw as the next anchor. Row `i` is the target
    /// after draft `i`, from `row_logits` (concurrent cycles) or read from
    /// the graph, where the last read row is the frontier (`frontier_read`).
    /// 0 when a row cannot be read.
    #[allow(clippy::too_many_arguments)]
    fn accept_prefix(
        &mut self,
        engine: &Engine,
        sampler: Option<&mut DsparkSampler>,
        row_logits: Option<&[f32]>,
        drafts: &[i32],
        row_tops: &[i32],
        stop_at_eos: bool,
        frontier_read: &mut bool,
    ) -> Result<u32, SessionError> {
        let mut accepted = 1;
        *frontier_read = false;
        let Some(sampler) = sampler else {
            let stop = if stop_at_eos { engine.vocab.eos } else { -1 };
            return Ok(greedy_accept(row_tops, drafts, drafts.len() as u32, stop));
        };
        let vocabulary = engine.vocab.len();
        while (accepted as usize) < drafts.len() {
            let row = accepted - 1;
            let draft = drafts[accepted as usize];
            let top = row_tops[row as usize];
            let argmax = usize::try_from(top).is_ok_and(|top| top < vocabulary)
                && sampler
                    .try_accept_argmax(top)
                    .map_err(|_| error(SAMPLER_FAILED))?;
            let drawn = if argmax {
                // A legal greedy argmax needs no vocabulary download or full
                // mask; a rejection below still fetches its row as the
                // reusable frontier.
                *frontier_read = false;
                top
            } else {
                let logits = match row_logits {
                    Some(rows) => &rows[row as usize * vocabulary..(row as usize + 1) * vocabulary],
                    None => {
                        if self
                            .graph
                            .read_spec_logits_row(&engine.backend, row, &mut self.logits)
                            .is_err()
                        {
                            return Ok(0);
                        }
                        *frontier_read = true;
                        &self.logits[..vocabulary]
                    }
                };
                if sampler.stochastic() {
                    sampler.verify(row, logits, draft)
                } else {
                    sampler.sample(logits)
                }
                .map_err(|_| error(SAMPLER_FAILED))?
            };
            if drawn != draft || (stop_at_eos && engine.vocab.is_stop(drawn)) {
                // EOS is a completed draw that must not enter the reusable
                // checkpoint: the next anchor, not drawn again.
                self.dspark_state.pending_anchor = drawn;
                return Ok(accepted);
            }
            accepted += 1;
        }
        *frontier_read = false;
        Ok(accepted)
    }

    /// `session_dspark_finish_verified`: commits the accepted prefix of a
    /// verified block at `length` — the whole block, a prefix snapshot (up
    /// to four rows, or four and one decode step), or else the saved
    /// frontier and a replay — and records the cycle.
    #[allow(clippy::too_many_arguments)]
    fn finish_verified(
        &mut self,
        engine: &mut Engine,
        length: usize,
        concurrent_batch: bool,
        sampler: Option<&mut DsparkSampler>,
        row_logits: Option<&[f32]>,
        drafts: &[i32],
        row_tops: &[i32],
        stop_at_eos: bool,
        emitted: &mut Vec<i32>,
    ) -> Result<(), SessionError> {
        let vocabulary = engine.vocab.len();
        let drafted = drafts.len() as u32;
        let sampled = sampler.is_some();
        let mut frontier_read = false;
        let accepted = self.accept_prefix(
            engine,
            sampler,
            row_logits,
            drafts,
            row_tops,
            stop_at_eos,
            &mut frontier_read,
        )?;
        if accepted == 0 {
            self.checkpoint_valid = false;
            return Err(error("DSpark verified row read failed"));
        }
        let positional_accepted = (1..drafts.len())
            .filter(|&i| row_tops[i - 1] == drafts[i])
            .count() as u64;
        let record = |s: &mut crate::dspark_policy::RequestState, committed: u32, full: bool| {
            s.drafted += u64::from(drafted);
            s.accepted += u64::from(committed);
            s.support_drafted += u64::from(drafted - 1);
            s.support_accepted += u64::from(committed.saturating_sub(1));
            s.positional_accepted += positional_accepted;
            s.anchors += 1;
            if full {
                s.full_blocks += 1;
            }
            s.steps += 1;
        };
        let row = |index: u32| index as usize * vocabulary..(index as usize + 1) * vocabulary;
        if accepted == drafted {
            self.checkpoint.truncate(length);
            self.checkpoint.extend_from_slice(drafts);
            if let Some(rows) = row_logits {
                self.logits[..vocabulary].copy_from_slice(&rows[row(drafted - 1)]);
            } else if !concurrent_batch
                && self
                    .graph
                    .read_spec_logits_row(&engine.backend, drafted - 1, &mut self.logits)
                    .is_err()
            {
                self.checkpoint_valid = false;
                return Err(error("DSpark frontier logits read failed"));
            }
            self.checkpoint_valid = true;
            if self.graph.dspark_capture_ready() {
                let _ = self.dspark_inject(engine, length as u32, drafted);
            }
            emitted.extend_from_slice(drafts);
            record(&mut self.dspark_state, accepted, true);
            self.note_cycle(concurrent_batch);
            return Ok(());
        }
        if let Some(rows) = row_logits {
            // The rejected draw came from this row: the frontier an exact
            // target step or a restored prefix sees.
            self.logits[..vocabulary].copy_from_slice(&rows[row(accepted - 1)]);
        }
        let mut committed_prefix = false;
        if accepted <= 4 {
            committed_prefix = (concurrent_batch
                || frontier_read
                || self
                    .graph
                    .read_spec_logits_row(&engine.backend, accepted - 1, &mut self.logits)
                    .is_ok())
                && self
                    .graph
                    .spec_frontier_commit_prefix(&engine.backend, accepted)
                    .is_ok();
            if committed_prefix {
                self.checkpoint.truncate(length);
                self.checkpoint
                    .extend_from_slice(&drafts[..accepted as usize]);
                if self.graph.dspark_capture_ready() {
                    committed_prefix = self.dspark_inject(engine, length as u32, accepted).is_ok();
                }
            }
        } else if accepted == 5 {
            committed_prefix = self
                .graph
                .spec_frontier_commit_prefix(&engine.backend, 4)
                .is_ok();
            if committed_prefix {
                self.checkpoint.truncate(length);
                self.checkpoint.extend_from_slice(&drafts[..4]);
                if self.graph.dspark_capture_ready() {
                    committed_prefix = self.dspark_inject(engine, length as u32, 4).is_ok();
                }
            }
            if committed_prefix {
                committed_prefix = self.commit(engine, drafts[4]).is_ok();
            }
        }
        if committed_prefix {
            self.checkpoint_valid = true;
            emitted.extend_from_slice(&drafts[..accepted as usize]);
            record(&mut self.dspark_state, accepted, false);
            self.note_cycle(concurrent_batch);
            return Ok(());
        }
        if self.graph.spec_frontier_restore(&engine.backend).is_err() {
            self.checkpoint_valid = false;
            return Err(error("DSpark rollback failed"));
        }
        self.checkpoint.truncate(length);
        self.graph.dspark_truncate_context(length as u32);
        // Greedy replay re-checks each row against the sequential argmax; a
        // sampled prefix was accepted by draws the request already consumed.
        let mut committed = 0;
        for &draft in &drafts[..accepted as usize] {
            let sequential_top = if sampled {
                draft
            } else {
                argmax(&self.logits[..vocabulary])
            };
            if sequential_top < 0 {
                self.checkpoint_valid = false;
                return Err(error("DSpark replay logits contain no finite values"));
            }
            if sequential_top != draft {
                break;
            }
            if self.commit(engine, draft).is_err() {
                self.checkpoint_valid = false;
                return Err(error("DSpark accepted-prefix replay failed"));
            }
            emitted.push(draft);
            committed += 1;
        }
        record(&mut self.dspark_state, committed, committed == drafted);
        self.note_cycle(concurrent_batch);
        Ok(())
    }

    /// `session_dspark_step`, greedy: one cycle of this session alone.
    fn dspark_step_one(
        &mut self,
        engine: &mut Engine,
        max_draft_tokens: u32,
        emitted_cap: u32,
        mut sampler: Option<&mut DsparkSampler>,
        stop_at_eos: bool,
    ) -> Result<Vec<i32>, SessionError> {
        if !self.checkpoint_valid || emitted_cap < 1 {
            return Err(error("invalid DSpark step request"));
        }
        if self.checkpoint.len() >= self.context_size as usize {
            return Err(error("DSpark decode exceeds session context"));
        }
        // Alone, probabilistic proposals did not repay their overhead: the
        // point-mass path keeps the autoregressive seed identity.
        if let Some(sampler) = sampler.as_deref_mut() {
            sampler.scalar();
        }
        self.use_concurrency(1);
        let block = self.graph.dspark_block_size();
        let maximum = if max_draft_tokens == 0 {
            block
        } else {
            block.min(max_draft_tokens)
        };
        let requested_tail = self.usable_tail(maximum, emitted_cap);
        let length = self.checkpoint.len();
        let vocabulary = engine.vocab.len() as i32;
        let target_first = self.anchor(sampler.as_deref_mut(), vocabulary as usize)?;
        if target_first < 0 || target_first >= vocabulary {
            self.checkpoint_valid = false;
            return Err(error("DSpark target logits contain no finite values"));
        }
        let mut emitted = Vec::new();
        if stop_at_eos && engine.vocab.is_stop(target_first) {
            emitted.push(target_first);
            return Ok(emitted);
        }
        let _ = self.seed_pending(engine, length as u32);
        // One ordinary token when a block cannot be drafted or verified.
        let mut can_draft = !self.dspark_state.plain_only
            && requested_tail != 0
            && length >= 2
            && self.graph.dspark_context_len() as usize >= length;
        if can_draft && self.dspark_state.skip_remaining != 0 {
            self.dspark_state.skip_remaining -= 1;
            self.dspark_state.skipped += 1;
            can_draft = false;
        }
        let plain = |session: &mut Self,
                     engine: &mut Engine,
                     message: &'static str|
         -> Result<Vec<i32>, SessionError> {
            if session.commit(engine, target_first).is_err() {
                session.checkpoint_valid = false;
                return Err(error(message));
            }
            Ok(vec![target_first])
        };
        if !can_draft {
            return plain(self, engine, "DeepSeek decode failed");
        }
        let proposed = engine.lend(&self.graph.workspace).and_then(|(target, ws)| {
            self.graph.dspark_draft(
                &target,
                ws,
                target_first,
                length as u32,
                sampler.as_deref_mut(),
                requested_tail,
                1,
            )
        });
        let mut drafts = vec![target_first];
        let mut tail_drafted = 0;
        if let Ok(tail) = &proposed
            && tail.len() <= MAX_BLOCK as usize
        {
            tail_drafted = (tail.len() as u32).min(requested_tail);
            drafts.extend_from_slice(&tail[..tail_drafted as usize]);
        }
        if proposed.is_err() || drafts.len() <= 1 || drafts.len() > MAX_BLOCK as usize + 1 {
            return plain(self, engine, "DeepSeek decode failed after draft failure");
        }
        // Row 0 is the target-known anchor, so every pass commits it.
        let drafted = drafts.len() as u32;
        if self
            .graph
            .spec_prepare(&engine.backend, &engine.weights, drafted)
            .is_err()
            || self.graph.spec_frontier_save(&engine.backend).is_err()
        {
            return Err(error("DSpark verification setup failed"));
        }
        self.checkpoint.extend_from_slice(&drafts);
        let verified = engine.lend(&self.graph.workspace).and_then(|(target, ws)| {
            self.graph
                .verify_suffix(&target, ws, &self.checkpoint, length as u32, drafted)
        });
        self.checkpoint.truncate(length);
        let Ok(row_tops) = verified else {
            self.checkpoint_valid = false;
            return Err(error("DSpark verification failed"));
        };
        let accepted_before = self.dspark_state.support_accepted;
        self.finish_verified(
            engine,
            length,
            false,
            sampler,
            None,
            &drafts,
            &row_tops,
            stop_at_eos,
            &mut emitted,
        )?;
        let accepted = (self.dspark_state.support_accepted - accepted_before) as u32;
        self.update_adaptive_width(tail_drafted, accepted, maximum);
        Ok(emitted)
    }

    /// `Session::DsparkStep`: one cycle, at most `max_tokens` emitted and
    /// `max_draft_tokens` drafted; greedy without a sampler.
    pub fn dspark_step(
        &mut self,
        engine: &mut Engine,
        max_tokens: usize,
        max_draft_tokens: u32,
        sampler: Option<&mut DsparkSampler>,
        stop_at_eos: bool,
    ) -> Result<Vec<i32>, SessionError> {
        let mut items = [DsparkItem {
            session: self,
            max_tokens,
            max_draft_tokens,
            sampler,
            stop_at_eos,
            emitted: Vec::new(),
        }];
        Session::dspark_step_batch(engine, &mut items)?;
        Ok(std::mem::take(&mut items[0].emitted))
    }

    /// `sessions_commit_and_extend_batch`: one token for each session as a
    /// batch, extending each support ring.
    fn commit_batch(engine: &mut Engine, items: &mut [(&mut Session, i32)]) -> GpuResult<()> {
        let mut batch: Vec<BatchItem<'_>> = items
            .iter_mut()
            .map(|(session, token)| BatchItem {
                position: session.checkpoint.len() as u32,
                token: *token,
                graph: &mut session.graph,
                logits: &mut session.logits,
            })
            .collect();
        let (target, workspaces) = engine.lend_all();
        eval_sessions_batch(&target, workspaces, &mut batch)?;
        drop(batch);
        for (session, token) in items.iter_mut() {
            let position = session.checkpoint.len() as u32;
            session.checkpoint.push(*token);
            if session.graph.dspark_block_size() != 0 && session.graph.dspark_capture_ready() {
                session.dspark_inject(engine, position, 1)?;
            }
        }
        Ok(())
    }

    /// `Model::DsparkStepBatch` and `ds4_sessions_dspark_step_batch`,
    /// greedy: one cycle of one to eight sessions. Drafting runs as one
    /// support batch and verification as one target batch; requests that
    /// cannot draft profitably commit their anchors.
    pub fn dspark_step_batch(
        engine: &mut Engine,
        items: &mut [DsparkItem<'_>],
    ) -> Result<(), SessionError> {
        let count = items.len();
        if count == 0 || count > 8 {
            return Err(error(
                "DeepSeek DSpark batch requires one to eight sessions",
            ));
        }
        if items
            .iter()
            .any(|i| i.max_tokens == 0 || i.max_draft_tokens == 0)
        {
            return Err(error("DeepSeek DSpark batch needs positive token budgets"));
        }
        for item in items.iter_mut() {
            item.emitted.clear();
        }
        if count == 1 {
            let item = &mut items[0];
            if !item.session.checkpoint_valid {
                return Err(error("DSpark batch contains an invalid session"));
            }
            if item.session.cancelled() {
                return Err(interrupted("DSpark batch decode cancelled"));
            }
            item.emitted = item.session.dspark_step_one(
                engine,
                item.max_draft_tokens,
                emitted_cap(item.max_tokens),
                item.sampler.as_deref_mut(),
                item.stop_at_eos,
            )?;
            return Ok(());
        }
        let block = items[0].session.graph.dspark_block_size();
        for item in items.iter() {
            if !item.session.checkpoint_valid {
                return Err(error("DSpark batch contains an invalid session"));
            }
            if item.session.checkpoint.len() >= item.session.context_size as usize {
                return Err(error("DSpark batch decode exceeds session context"));
            }
            if item.session.cancelled() {
                return Err(interrupted("DSpark batch decode cancelled"));
            }
        }
        let vocabulary = engine.vocab.len() as i32;
        // Validate the whole cohort before any request's policy or KV moves.
        let mut lengths = vec![0usize; count];
        let mut target_first = vec![0i32; count];
        let mut can_draft = vec![false; count];
        let mut verify_tails = vec![0u32; count];
        let mut policy_tails = vec![0u32; count];
        let mut any_can_draft = false;
        let mut any_stopped = false;
        for (index, item) in items.iter_mut().enumerate() {
            let session = &mut *item.session;
            session.use_concurrency(count);
            let length = session.checkpoint.len();
            lengths[index] = length;
            target_first[index] =
                session.anchor(item.sampler.as_deref_mut(), vocabulary as usize)?;
            if target_first[index] < 0 || target_first[index] >= vocabulary {
                session.checkpoint_valid = false;
                return Err(error("DSpark target logits contain no finite values"));
            }
            if item.stop_at_eos && engine.vocab.is_stop(target_first[index]) {
                any_stopped = true;
                continue;
            }
            let _ = session.seed_pending(engine, length as u32);
            let cap = emitted_cap(item.max_tokens);
            let maximum = item.max_draft_tokens.min(block).min(MAX_BLOCK).min(3);
            policy_tails[index] = session.profitable_tail(maximum, count);
            verify_tails[index] = session.usable_tail(policy_tails[index], cap);
            let mut draft = !session.dspark_state.plain_only
                && session.usable_tail(maximum, cap) != 0
                && length >= 2
                && session.graph.dspark_context_len() as usize >= length;
            if draft && (policy_tails[index] == 0 || session.dspark_state.skip_remaining != 0) {
                // Cost is reconsidered every token, and an acceptance
                // backoff keeps aging while the draft limit is unprofitable.
                if session.dspark_state.skip_remaining != 0 {
                    session.dspark_state.skip_remaining -= 1;
                }
                session.dspark_state.skipped += 1;
                draft = false;
            }
            can_draft[index] = draft;
            any_can_draft |= draft;
        }
        let fallback = |engine: &mut Engine,
                        items: &mut [DsparkItem<'_>],
                        message: &'static str,
                        skip_stopped: bool|
         -> Result<(), SessionError> {
            let mut active: Vec<(&mut Session, i32)> = Vec::new();
            for (index, item) in items.iter_mut().enumerate() {
                if skip_stopped && item.stop_at_eos && engine.vocab.is_stop(target_first[index]) {
                    continue;
                }
                active.push((&mut *item.session, target_first[index]));
            }
            let committed = match active.len() {
                0 => Ok(()),
                1 => {
                    let (session, token) = &mut active[0];
                    session.commit(engine, *token)
                }
                _ => Session::commit_batch(engine, &mut active),
            };
            if committed.is_err() {
                for (session, _) in active.iter_mut() {
                    session.checkpoint_valid = false;
                }
                return Err(error(message));
            }
            drop(active);
            for (index, item) in items.iter_mut().enumerate() {
                item.emitted = vec![target_first[index]];
            }
            Ok(())
        };
        if any_stopped || !any_can_draft {
            // Completed requests stay at the frontier before EOS.
            return fallback(
                engine,
                items,
                "DeepSeek DSpark fallback batch decode failed",
                true,
            );
        }
        let mut tails: Vec<Vec<i32>> = vec![Vec::new(); count];
        let drafting: Vec<usize> = (0..count).filter(|&i| can_draft[i]).collect();
        let proposed = if drafting.len() >= 2 {
            let mut draft_items: Vec<DraftItem<'_>> = items
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| can_draft[*index])
                .map(|(index, item)| DraftItem {
                    graph: &mut item.session.graph,
                    target_next_token: target_first[index],
                    position: lengths[index] as u32,
                    max_draft_tokens: verify_tails[index],
                    sampler: item.sampler.as_deref_mut(),
                    concurrency: count,
                    tokens: Vec::new(),
                })
                .collect();
            let (target, workspaces) = engine.lend_all();
            let result = dspark_draft_head_batch(&target, workspaces, &mut draft_items);
            for (item, &index) in draft_items.into_iter().zip(&drafting) {
                tails[index] = item.tokens;
            }
            result.is_ok()
        } else {
            let index = drafting[0];
            let item = &mut items[index];
            let graph = &mut item.session.graph;
            let drafted = engine.lend(&graph.workspace).and_then(|(target, ws)| {
                graph.dspark_draft(
                    &target,
                    ws,
                    target_first[index],
                    lengths[index] as u32,
                    item.sampler.as_deref_mut(),
                    verify_tails[index],
                    count,
                )
            });
            match drafted {
                Ok(tail) => {
                    tails[index] = tail;
                    true
                }
                Err(_) => false,
            }
        };
        let mut drafted_counts = vec![0u32; count];
        let mut proposed_all = proposed;
        for index in 0..count {
            if proposed_all && tails[index].len() <= MAX_BLOCK as usize {
                let verify_tail = (tails[index].len() as u32).min(verify_tails[index]);
                drafted_counts[index] = if can_draft[index] { verify_tail + 1 } else { 1 };
            }
            proposed_all = proposed_all
                && drafted_counts[index] != 0
                && drafted_counts[index] <= MAX_BLOCK + 1;
        }
        if proposed_all {
            // Confidence describes this block, so a proposing request that
            // drafted nothing retries after one budget; acceptance keeps its
            // longer backoff.
            for (index, item) in items.iter_mut().enumerate() {
                if can_draft[index]
                    && item.sampler.as_ref().is_some_and(|s| s.stochastic())
                    && tails[index].is_empty()
                {
                    item.session.dspark_state.skip_remaining = verify_tails[index];
                }
            }
            if drafted_counts.iter().all(|&c| c == 1) {
                proposed_all = false;
            }
        }
        if !proposed_all {
            return fallback(
                engine,
                items,
                "DeepSeek DSpark draft fallback batch decode failed",
                false,
            );
        }
        let drafts: Vec<Vec<i32>> = (0..count)
            .map(|index| {
                let mut block = vec![target_first[index]];
                block.extend_from_slice(&tails[index][..drafted_counts[index] as usize - 1]);
                block
            })
            .collect();
        let vocabulary = vocabulary as usize;
        for (index, (item, block)) in items.iter_mut().zip(&drafts).enumerate() {
            item.session.checkpoint.extend_from_slice(block);
            // Sampled requests decide on every verified row
            // (`session_spec_rows`).
            if item.sampler.is_some() {
                item.session
                    .spec_rows
                    .resize(drafted_counts[index] as usize * vocabulary, 0.0);
            }
        }
        let mut verify_items: Vec<VerifyItem<'_>> = items
            .iter_mut()
            .enumerate()
            .map(|(index, item)| {
                let sampled = item.sampler.is_some();
                let session = &mut *item.session;
                VerifyItem {
                    graph: &mut session.graph,
                    tokens: &session.checkpoint,
                    start: lengths[index] as u32,
                    n_tokens: drafted_counts[index],
                    logical_n_tokens: drafted_counts[index],
                    row_tops: Vec::new(),
                    frontier_logits: Some(&mut session.logits),
                    row_logits: sampled.then_some(&mut session.spec_rows[..]),
                    stop_token: if item.stop_at_eos {
                        engine.vocab.eos
                    } else {
                        -1
                    },
                }
            })
            .collect();
        let (target, workspaces) = engine.lend_all();
        let verified = verify_sessions_batch(&target, workspaces, &mut verify_items);
        let row_tops: Vec<Vec<i32>> = verify_items.into_iter().map(|v| v.row_tops).collect();
        for (item, &length) in items.iter_mut().zip(&lengths) {
            item.session.checkpoint.truncate(length);
        }
        if verified.is_err() {
            for item in items.iter_mut() {
                item.session.checkpoint_valid = false;
            }
            return Err(error("DSpark batch verification failed"));
        }
        for index in 0..count {
            let item = &mut items[index];
            let mut emitted = Vec::new();
            let sampled = item.sampler.is_some();
            let rows = std::mem::take(&mut item.session.spec_rows);
            let result = item.session.finish_verified(
                engine,
                lengths[index],
                true,
                item.sampler.as_deref_mut(),
                sampled.then_some(&rows[..]),
                &drafts[index],
                &row_tops[index],
                item.stop_at_eos,
                &mut emitted,
            );
            item.session.spec_rows = rows;
            item.emitted = emitted;
            if let Err(error) = result {
                for remaining in items.iter_mut().skip(index + 1) {
                    remaining.session.checkpoint_valid = false;
                }
                return Err(error);
            }
            if can_draft[index] {
                let maximum = item.max_draft_tokens.min(block).min(MAX_BLOCK);
                let policy_maximum = maximum.min(policy_tails[index]);
                let support_drafted = drafted_counts[index].saturating_sub(1);
                let support_accepted = (item.emitted.len() as u32).saturating_sub(1);
                item.session
                    .note_concurrent_cycle(support_drafted, support_accepted, count);
                item.session.update_adaptive_width(
                    support_drafted,
                    support_accepted,
                    policy_maximum,
                );
            }
        }
        Ok(())
    }
}

/// The EOS boundary of sampled cohorts (`CheckDsparkEosBoundary` in
/// `tests/models/deepseek_v4_flash/serving_test.cpp`), with hooks whose
/// draws are fixed: an EOS draw ends its request without entering the
/// checkpoint or being drawn again, alone, beside one survivor and in a
/// compacted batch, and a fixed-length step consumes EOS as a token.
#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used)]
mod eos {
    use std::path::PathBuf;

    use super::*;

    /// The artifact `variable` names; unset, the variable itself, so the
    /// failure says what to set.
    fn artifact(variable: &str) -> PathBuf {
        std::env::var_os(variable)
            .filter(|path| !path.is_empty())
            .map_or_else(|| PathBuf::from(format!("${variable}")), PathBuf::from)
    }

    /// One cycle of `sessions` with draws `tokens`: the emitted tokens and
    /// the draws of each hook.
    fn step(
        engine: &mut Engine,
        sessions: &mut [Session],
        tokens: &[i32],
    ) -> (Vec<Vec<i32>>, Vec<u32>) {
        let mut samplers: Vec<DsparkSampler> =
            tokens.iter().map(|&t| DsparkSampler::fixed(t)).collect();
        let mut items: Vec<DsparkItem<'_>> = sessions
            .iter_mut()
            .zip(samplers.iter_mut())
            .map(|(session, sampler)| DsparkItem {
                session,
                max_tokens: 1,
                max_draft_tokens: 1,
                sampler: Some(sampler),
                stop_at_eos: true,
                emitted: Vec::new(),
            })
            .collect();
        Session::dspark_step_batch(engine, &mut items).unwrap();
        let emitted = items.into_iter().map(|item| item.emitted).collect();
        (
            emitted,
            samplers.iter().map(DsparkSampler::fixed_draws).collect(),
        )
    }

    #[test]
    #[ignore = "QUALITY check on the DeepSeek V4 Flash artifact"]
    fn eos_draws_end_requests_without_resampling() {
        let model = artifact("GUFO_DEEPSEEK_V4_FLASH_MODEL");
        let support = artifact("GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL");
        let mut engine = Engine::open(&model, Some(&support)).unwrap();
        let mut prompt = Vec::new();
        engine
            .vocab
            .tokenize(b"Continue the pattern: red, blue, red,", &mut prompt);
        let eos = engine.vocab.eos;
        let last = *prompt.last().unwrap();
        let tokens = [eos, last, last];
        let mut sessions: Vec<Session> = (0..3)
            .map(|_| {
                let mut session = engine.create_session(4096, true).unwrap();
                session.sync(&mut engine, &prompt).unwrap();
                session
            })
            .collect();
        let eos_logits = sessions[0].logits().unwrap().to_vec();
        // No survivors, one survivor and a compacted GPU batch.
        for width in 1..=3 {
            let positions: Vec<usize> = sessions[..width].iter().map(Session::position).collect();
            let (emitted, draws) = step(&mut engine, &mut sessions[..width], &tokens[..width]);
            for i in 0..width {
                assert!(
                    emitted[i] == [tokens[i]] && draws[i] == 1,
                    "EOS compaction emits each request's draw without resampling"
                );
                assert_eq!(
                    sessions[i].position(),
                    positions[i] + usize::from(i != 0),
                    "only non-EOS anchors enter the request checkpoint"
                );
            }
            assert!(
                sessions[0].logits().unwrap() == eos_logits,
                "EOS keeps the preceding frontier logits"
            );
        }
        let positions: Vec<usize> = sessions.iter().map(Session::position).collect();
        let (emitted, draws) = step(&mut engine, &mut sessions, &[eos; 3]);
        for i in 0..3 {
            assert!(
                sessions[i].position() == positions[i] && draws[i] == 1 && emitted[i] == [eos],
                "an all-EOS cohort completes without committing tokens"
            );
        }
        let mut sampler = DsparkSampler::fixed(eos);
        let emitted = sessions[0]
            .dspark_step(&mut engine, 1, 1, Some(&mut sampler), false)
            .unwrap();
        assert!(
            sessions[0].position() == prompt.len() + 1 && emitted == [eos],
            "fixed-length decoding still consumes EOS as an ordinary token"
        );
    }
}
