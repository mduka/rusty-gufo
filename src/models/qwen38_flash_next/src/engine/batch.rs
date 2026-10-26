//! Decoding and evaluating several sessions in one batch.

use super::*;

impl Model {
    /// Decodes independent sessions together: shared projections, private
    /// state, proposal distributions and RNG streams. A failed batch step
    /// retries every session it did not touch on its own; a session whose
    /// state was partly changed becomes invalid until its next sync
    /// (`DecodeBatch`).
    pub fn decode_batch(
        &mut self,
        requests: &mut [DecodeRequest<'_>],
    ) -> Vec<Result<DecodeResult, String>> {
        if requests.is_empty() || requests.len() > BATCH_SESSIONS {
            return requests
                .iter()
                .map(|_| Err("batch must contain 1..8 sessions".into()))
                .collect();
        }
        let mut outcomes: Vec<Outcome> = (0..requests.len()).map(|_| Outcome::default()).collect();
        let mut epochs = Vec::with_capacity(requests.len());
        let mut draws = Vec::with_capacity(requests.len());
        let mut lookups = Vec::with_capacity(requests.len());
        let mut active = vec![false; requests.len()];
        for (i, r) in requests.iter_mut().enumerate() {
            epochs.push(r.session.native.mutation_epoch());
            draws.push(r.sampler.save_draw_state());
            lookups.push(r.session.lookup.save());
            let valid = r.session.valid && r.max_tokens > 0 && !r.session.tokens.is_empty();
            if !valid {
                outcomes[i].error = "invalid or non-independent batch request".into();
            } else if r.session.native.check_cancelled() {
                outcomes[i].error = "generation cancelled".into();
            } else {
                active[i] = true;
            }
        }
        let shared = if active.iter().any(|&a| a) {
            self.decode_batch_active(requests, &active, &mut outcomes)
                .err()
                .unwrap_or_default()
        } else {
            String::new()
        };
        let mut results = Vec::with_capacity(requests.len());
        for (i, r) in requests.iter_mut().enumerate() {
            let outcome = &mut outcomes[i];
            if !outcome.completed && outcome.error.is_empty() {
                if r.session.native.cancelled() {
                    outcome.error = "generation cancelled".into();
                } else if r.session.native.mutation_epoch() == epochs[i] {
                    // A preparation or allocation failure did not touch this
                    // session: decode it alone from the same frontier.
                    r.session.valid = true;
                    r.sampler.restore_draw_state(draws[i]);
                    r.session.lookup.restore(lookups[i]);
                    match self.decode_step(r.session, r.max_tokens, r.sampler, r.stop_at_eos) {
                        Ok(result) => {
                            outcome.completed = true;
                            outcome.result = result;
                        }
                        Err(e) => outcome.error = e,
                    }
                } else if shared.is_empty() {
                    outcome.error = "batch execution failed".into();
                } else {
                    outcome.error = shared.clone();
                }
            }
            if outcome.completed {
                r.session.valid = true;
                results.push(Ok(std::mem::take(&mut outcome.result)));
            } else {
                if outcome.error.is_empty() {
                    outcome.error = "batch request failed".into();
                }
                if r.session.native.mutation_epoch() != epochs[i] {
                    r.session.valid = false;
                }
                results.push(Err(std::mem::take(&mut outcome.error)));
            }
        }
        results
    }

    /// Appends one token to each of several sessions (`EvaluateBatch`), with
    /// the failure handling of [`Model::decode_batch`].
    pub fn evaluate_batch(
        &mut self,
        requests: &mut [AdvanceRequest<'_>],
    ) -> Vec<Result<(), String>> {
        if requests.is_empty() || requests.len() > BATCH_SESSIONS {
            return requests
                .iter()
                .map(|_| Err("batch must contain 1..8 sessions".into()))
                .collect();
        }
        let vocab = self.vocab_size();
        let mut outcomes: Vec<Outcome> = (0..requests.len()).map(|_| Outcome::default()).collect();
        let mut epochs = Vec::with_capacity(requests.len());
        let mut active = vec![false; requests.len()];
        for (i, r) in requests.iter_mut().enumerate() {
            epochs.push(r.session.native.mutation_epoch());
            let valid = r.session.valid
                && r.token >= 0
                && (r.token as u32) < vocab
                && r.session.position() < r.session.context_size();
            if !valid {
                outcomes[i].error = "invalid or non-independent batch request".into();
            } else if r.session.native.check_cancelled() {
                outcomes[i].error = "generation cancelled".into();
            } else {
                active[i] = true;
            }
        }
        let shared = if active.iter().any(|&a| a) {
            self.evaluate_batch_active(requests, &active, &mut outcomes)
                .err()
                .unwrap_or_default()
        } else {
            String::new()
        };
        let mut results = Vec::with_capacity(requests.len());
        for (i, r) in requests.iter_mut().enumerate() {
            let outcome = &mut outcomes[i];
            if !outcome.completed && outcome.error.is_empty() {
                if r.session.native.cancelled() {
                    outcome.error = "generation cancelled".into();
                } else if r.session.native.mutation_epoch() == epochs[i] {
                    r.session.valid = true;
                    match self.evaluate(r.session, r.token) {
                        Ok(()) => outcome.completed = true,
                        Err(e) => outcome.error = e,
                    }
                } else if shared.is_empty() {
                    outcome.error = "batch execution failed".into();
                } else {
                    outcome.error = shared.clone();
                }
            }
            if outcome.completed {
                r.session.valid = true;
                results.push(Ok(()));
            } else {
                if outcome.error.is_empty() {
                    outcome.error = "batch request failed".into();
                }
                if r.session.native.mutation_epoch() != epochs[i] {
                    r.session.valid = false;
                }
                results.push(Err(std::mem::take(&mut outcome.error)));
            }
        }
        results
    }

    /// The draft block of each MTP session catches up to its next token in
    /// one batch (`DraftCatchUpBatch`); `propose[i]` says a proposal follows
    /// for session i, whose residual is kept (none given: K/V only).
    pub(super) fn draft_catch_up_batch(
        &mut self,
        sessions: &mut [(&mut Session, i32)],
        propose: &[bool],
    ) -> Result<(), String> {
        if !propose.is_empty() && propose.len() != sessions.len() {
            return Err("draft catch-up head count differs from requests".into());
        }
        let mut replays = Vec::with_capacity(sessions.len());
        for (s, token) in sessions.iter() {
            replays.push(if s.mtp_enabled() {
                self.draft_replay(s, *token)?
            } else {
                None
            });
        }
        let mut items: Vec<MtpBatchItem<'_>> = sessions
            .iter_mut()
            .zip(&replays)
            .enumerate()
            .filter_map(|(i, ((s, _), replay))| {
                replay.as_ref().map(|(tokens, hidden_row)| MtpBatchItem {
                    session: &mut s.native,
                    tokens,
                    hidden_row: *hidden_row,
                    kv_only: propose.get(i).is_none_or(|&p| !p),
                })
            })
            .collect();
        if items.is_empty() {
            return Ok(());
        }
        self.executor.mtp_forward_batch(&mut items)
    }

    /// `DecodeBatchImpl` over the active requests.
    pub(super) fn decode_batch_active(
        &mut self,
        requests: &mut [DecodeRequest<'_>],
        active: &[bool],
        outcomes: &mut [Outcome],
    ) -> Result<(), String> {
        let ids: Vec<usize> = (0..requests.len()).filter(|&i| active[i]).collect();
        if let [only] = ids[..] {
            let r = &mut requests[only];
            match self.decode_step(r.session, r.max_tokens, r.sampler, r.stop_at_eos) {
                Ok(result) => {
                    outcomes[only].completed = true;
                    outcomes[only].result = result;
                }
                Err(e) => outcomes[only].error = e,
            }
            return Ok(());
        }
        let max_speculative = self.executor.options().max_speculative as usize;
        let mut batch_context = 0;
        let batch_drafts = if ids.iter().all(|&i| requests[i].session.mtp_enabled())
            && ids
                .iter()
                .all(|&i| !requests[i].sampler.config().uses_random_sampling())
        {
            let rows: Vec<BatchRow<'_>> = ids
                .iter()
                .map(|&i| {
                    let r = &requests[i];
                    let cap = r
                        .max_tokens
                        .min((r.session.context_size() - r.session.position()) as usize);
                    batch_context = batch_context.max(r.session.position());
                    BatchRow {
                        acceptance: &r.session.draft_length,
                        budget: cap.min(max_speculative) as u32 - u32::from(cap != 0),
                    }
                })
                .collect();
            Some(self.batch_policy.choose(&rows, batch_context))
        } else {
            None
        };
        let cycle_start = std::time::Instant::now();
        let mut pending: Vec<PendingDecode> = (0..requests.len())
            .map(|_| PendingDecode::default())
            .collect();
        for &i in &ids {
            let r = &mut requests[i];
            let prepared = self.prepare_decode(
                r.session,
                r.max_tokens,
                r.sampler,
                r.stop_at_eos,
                &mut pending[i],
                &mut outcomes[i].result,
                true,
                batch_drafts,
                ids.len() as u32,
            );
            if let Err(e) = prepared {
                outcomes[i].error = e;
                pending[i] = PendingDecode::default();
            }
        }
        {
            let mut catch_up: Vec<(&mut Session, i32)> = requests
                .iter_mut()
                .zip(&pending)
                .filter(|(_, p)| !p.chain.is_empty())
                .map(|(r, p)| (&mut *r.session, p.chain[0]))
                .collect();
            // A proposal follows where the chain still has room for drafts.
            let propose: Vec<bool> = pending
                .iter()
                .filter(|p| !p.chain.is_empty())
                .map(|p| p.speculative && p.chain.len() < p.width)
                .collect();
            self.draft_catch_up_batch(&mut catch_up, &propose)?;
        }
        // Each round shares the predictor's projections across the ready
        // sessions.
        loop {
            let mut heads: Vec<MtpHeadItem<'_>> = requests
                .iter_mut()
                .zip(pending.iter_mut())
                .filter(|(r, p)| {
                    !r.session.native.cancelled() && p.speculative && p.chain.len() < p.width
                })
                .map(|(r, p)| {
                    let output = if p.sampled {
                        MtpOutput {
                            token: None,
                            candidates: Some(&mut p.candidates),
                            kv_only: false,
                            draft_subset: p.draft_subset,
                        }
                    } else {
                        MtpOutput {
                            token: Some(&mut p.draft),
                            candidates: None,
                            kv_only: false,
                            draft_subset: p.draft_subset,
                        }
                    };
                    MtpHeadItem {
                        session: &mut r.session.native,
                        output,
                    }
                })
                .collect();
            if heads.is_empty() {
                break;
            }
            self.executor.mtp_heads(&mut heads)?;
            drop(heads);
            let mut drafts = [0i32; BATCH_SESSIONS];
            let mut body = [false; BATCH_SESSIONS];
            for (i, r) in requests.iter().enumerate() {
                let p = &mut pending[i];
                if r.session.native.cancelled() || !p.speculative || p.chain.len() >= p.width {
                    continue;
                }
                p.append_draft()?;
                if r.stop_at_eos && p.chain.last().is_some_and(|&t| self.is_stop_token(t)) {
                    p.width = p.chain.len();
                    continue;
                }
                if self.append_lookup(r.session, p, r.stop_at_eos, 1, false) {
                    continue;
                }
                if p.chain.len() < p.width {
                    drafts[i] = p.draft;
                    body[i] = true;
                }
            }
            let mut bodies: Vec<MtpBatchItem<'_>> = requests
                .iter_mut()
                .zip(drafts.iter())
                .enumerate()
                .filter(|(i, _)| body[*i])
                .map(|(_, (r, draft))| MtpBatchItem {
                    session: &mut r.session.native,
                    tokens: std::slice::from_ref(draft),
                    hidden_row: -1,
                    kv_only: false,
                })
                .collect();
            if !bodies.is_empty() {
                self.executor.mtp_forward_batch(&mut bodies)?;
            }
        }
        // In an all-greedy cohort full copies may widen verification beyond
        // the planned shape and delay peers producing new text: rows keep
        // their full width when every active row copies; in a mixed cohort
        // copies are bounded to half the verifier, or the planned width if
        // larger. Only copied suffixes are trimmed; sampled proposal and RNG
        // streams stay fixed.
        if let Some(drafts) = batch_drafts
            && pending.iter().any(|p| !p.chain.is_empty() && p.lookup == 0)
        {
            for p in pending.iter_mut().filter(|p| p.lookup != 0) {
                let cap = (1 + drafts as usize).max(p.lookup_cap.div_ceil(2));
                if p.chain.len() > cap {
                    p.lookup -= (p.chain.len() - cap) as u32;
                    p.chain.truncate(cap);
                    p.width = cap;
                }
            }
        }
        let included: Vec<bool> = requests
            .iter()
            .zip(&pending)
            .map(|(r, p)| !r.session.native.cancelled() && !p.chain.is_empty())
            .collect();
        {
            let mut items: Vec<BatchItem<'_>> = requests
                .iter_mut()
                .zip(&pending)
                .zip(&included)
                .filter(|(_, inc)| **inc)
                .map(|((r, p), _)| BatchItem {
                    session: &mut r.session.native,
                    tokens: &p.chain,
                    speculative: p.speculative,
                })
                .collect();
            if !items.is_empty() {
                self.executor.forward_batch(&mut items)?;
            }
        }
        let mut offset = 0u32;
        for (i, r) in requests.iter_mut().enumerate() {
            let p = &pending[i];
            let row_offset = offset;
            if included[i] {
                offset += p.chain.len() as u32;
            }
            let outcome = &mut outcomes[i];
            if !active[i] || !outcome.error.is_empty() {
                continue;
            }
            if r.session.native.cancelled() {
                outcome.error = "generation cancelled".into();
                continue;
            }
            if p.chain.is_empty() {
                outcome.completed = true;
                continue;
            }
            let s = &mut *r.session;
            let logits = if !p.speculative {
                Some(&mut s.logits[..])
            } else if p.gpu_verification {
                None
            } else {
                Some(&mut s.verify_logits[..])
            };
            let finished = self
                .executor
                .select_batch_logits(row_offset, p.chain.len() as u32, logits)
                .and_then(|()| {
                    self.finish_decode(s, r.sampler, r.stop_at_eos, p, &mut outcome.result)
                });
            match finished {
                Ok(()) => outcome.completed = true,
                Err(e) => outcome.error = e,
            }
        }
        let used_lookup = pending.iter().any(|p| p.lookup != 0);
        if used_lookup {
            self.batch_policy.skip_cycle();
        }
        if let Some(drafts) = batch_drafts
            && !used_lookup
            && included.iter().filter(|&&inc| inc).count() == ids.len()
            && ids.iter().all(|&i| outcomes[i].completed)
        {
            let ms = cycle_start.elapsed().as_secs_f32() * 1000.0;
            self.batch_policy
                .observe(ids.len(), batch_context, drafts, ms);
        }
        Ok(())
    }

    /// `EvaluateBatchImpl` over the active requests.
    pub(super) fn evaluate_batch_active(
        &mut self,
        requests: &mut [AdvanceRequest<'_>],
        active: &[bool],
        outcomes: &mut [Outcome],
    ) -> Result<(), String> {
        let ids: Vec<usize> = (0..requests.len()).filter(|&i| active[i]).collect();
        if let [only] = ids[..] {
            let r = &mut requests[only];
            match self.evaluate(r.session, r.token) {
                Ok(()) => outcomes[only].completed = true,
                Err(e) => outcomes[only].error = e,
            }
            return Ok(());
        }
        {
            let mut catch_up: Vec<(&mut Session, i32)> = requests
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| active[*i])
                .map(|(_, r)| (&mut *r.session, r.token))
                .collect();
            self.draft_catch_up_batch(&mut catch_up, &[])?;
        }
        let tokens: Vec<i32> = requests.iter().map(|r| r.token).collect();
        {
            let mut items: Vec<BatchItem<'_>> = requests
                .iter_mut()
                .zip(&tokens)
                .enumerate()
                .filter(|(i, _)| active[*i])
                .map(|(_, (r, token))| BatchItem {
                    session: &mut r.session.native,
                    tokens: std::slice::from_ref(token),
                    speculative: false,
                })
                .collect();
            self.executor.forward_batch(&mut items)?;
        }
        for (row, &i) in ids.iter().enumerate() {
            let r = &mut requests[i];
            let outcome = &mut outcomes[i];
            if r.session.native.cancelled() {
                outcome.error = "generation cancelled".into();
                continue;
            }
            let s = &mut *r.session;
            match self
                .executor
                .select_batch_logits(row as u32, 1, Some(&mut s.logits[..]))
            {
                Ok(()) => {
                    s.hidden_base = s.tokens.len() as u32;
                    s.tokens.push(r.token);
                    outcome.completed = true;
                }
                Err(e) => outcome.error = e,
            }
        }
        Ok(())
    }
}
