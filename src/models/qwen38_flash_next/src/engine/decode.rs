//! One session's decode step: preparing it, the prompt-lookup rows, and
//! finishing it.

use super::*;

impl PendingDecode {
    /// Appends the current draft, drawing a sampled proposal first
    /// (`AppendDraft`).
    pub(super) fn append_draft(&mut self) -> Result<(), String> {
        if self.sampled {
            let sampler = self.draft_sampler.as_mut().ok_or("missing draft sampler")?;
            let proposal = policy::sample_proposal(&self.candidates, sampler, &mut self.draft_rng)?;
            self.draft = proposal.token as i32;
            sampler.accept(proposal.token).map_err(|e| e.to_string())?;
            self.proposals.push(proposal);
        }
        self.chain.push(self.draft);
        Ok(())
    }
}

impl Model {
    /// Samples the anchor and prepares the draft chain (`PrepareDecode`).
    /// `defer_head` leaves catch-up and proposals to a batched caller;
    /// `batch_drafts` overrides the session's length policy.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_decode(
        &mut self,
        s: &mut Session,
        max_tokens: usize,
        sampler: &mut SamplerState,
        stop_at_eos: bool,
        pending: &mut PendingDecode,
        result: &mut DecodeResult,
        defer_head: bool,
        batch_drafts: Option<u32>,
        occupancy: u32,
    ) -> Result<(), String> {
        if max_tokens == 0 || s.tokens.is_empty() {
            return Err("decode needs an output, a positive budget and a synced prompt".into());
        }
        *result = DecodeResult::default();
        let max_speculative = self.executor.options().max_speculative as usize;
        let room = s.context_size() as usize - s.tokens.len();
        let cap = max_tokens.min(room).min(max_speculative);
        pending.policy_concurrency = policy::policy_concurrency(
            sampler.config().uses_random_sampling(),
            sampler.config().seed >= 0,
            batch_drafts.is_some(),
            occupancy,
        );
        let width = if s.mtp_enabled() && cap > 1 {
            1 + match batch_drafts {
                Some(drafts) => (drafts as usize).min(cap - 1),
                None => s.draft_length.choose(
                    (cap - 1) as u32,
                    s.tokens.len() as u32,
                    pending.policy_concurrency,
                    false,
                ) as usize,
            }
        } else {
            cap
        };
        if width == 0 {
            result.stop = true;
            return Ok(());
        }
        // Greedy verification left the device argmax of these logits.
        let epoch = s.native.mutation_epoch();
        let hint = s
            .anchor_hint
            .take()
            .filter(|&(_, at, tokens)| at == epoch && tokens == s.tokens.len())
            .map(|(token, ..)| token);
        let anchor = sampler
            .sample_with_argmax(&s.logits, hint)
            .map_err(|e| e.to_string())? as i32;
        if stop_at_eos && self.is_stop_token(anchor) {
            result.stop = true;
            return Ok(());
        }
        if !s.mtp_enabled() || width < 2 {
            if s.mtp_enabled() && !defer_head {
                self.draft_catch_up(s, anchor, false, None, false)?;
            }
            pending.chain = vec![anchor];
            pending.base = s.tokens.len() as u32;
            return Ok(());
        }
        let base = s.tokens.len() as u32;
        let sampled = sampler.config().uses_random_sampling();
        // Greedy rows stay on the device, penalties included; a grammar
        // takes each legal argmax and downloads the rows only at the first
        // forbidden one.
        let gpu_greedy = sampler.config().temperature == 0.0;
        // Whether the cycle may draft through the small head; the table then
        // decides per cycle (`DraftVocab::fits`). Penalties and constraints
        // never matter: drafts ignore both, and verification applies them
        // over the full target vocabulary. Sampled requests may at any
        // session count: verification corrects their proposals against the
        // full target distribution, and the choice reads only the request's
        // own tokens, so seeded output does not depend on overlapping
        // requests. Greedy requests use the small head on one session, when
        // they decode alone (`occupancy` 1: no peer shares the step), or in
        // a planned batch, so changed proposal chains cannot delay unplanned
        // peers.
        let allow_subset = draft_subset_allowed(
            sampler.config().uses_random_sampling(),
            self.options.decode_concurrency,
            occupancy,
            batch_drafts.is_some(),
        );
        // A cycle-local proposal stream keeps pending RNG state out of
        // snapshots; target verification keeps its own draws.
        pending.draft_rng = if sampled {
            sampling::next_random(sampler.mutable_rng_state())
        } else {
            0
        };
        // The draft proposes without the grammar; target verification keeps
        // it.
        let mut draft_sampler = sampler.without_constraint();
        draft_sampler
            .accept(anchor as TokenId)
            .map_err(|e| e.to_string())?;
        pending.draft_sampler = Some(draft_sampler);
        pending.chain = vec![anchor];
        pending.width = width;
        pending.base = base;
        pending.speculative = true;
        pending.sampled = sampled;
        pending.gpu_greedy = gpu_greedy;
        // Sampled rows also stay on the device when their target top-k can
        // travel as compact candidate sets.
        pending.gpu_verification = gpu_greedy || mtp::can_compact_mtp_verification(sampler);
        let rows = max_speculative * self.vocab_size() as usize;
        if !pending.gpu_verification && s.verify_logits.len() < rows {
            s.verify_logits = vec![0.0; rows];
        }
        // Both execution paths use the same request-local copy rule:
        // scheduling a peer must not change a sampled request's proposals or
        // RNG sequence. A request's first speculative step copies nothing.
        pending.lookup_cap = if s.lookup.step(base as usize) { cap } else { 0 };
        if pending.lookup_cap != 0 {
            s.lookup.extend(&s.tokens, sampled);
        }
        // Copies can supply the whole chain without the neural head. A
        // sampled copy may only after an accepted copy established its source
        // and a full sampled match still holds: new or short matches keep the
        // neural proposal, since shared syntax alone predicts poorly at
        // temperature.
        if self.append_lookup(s, pending, stop_at_eos, 3, sampled) {
            if !defer_head {
                self.draft_catch_up(s, anchor, false, None, false)?;
            }
            return Ok(());
        }
        if allow_subset {
            self.sync_draft_vocab(s)?;
        }
        pending.draft_subset = allow_subset
            && self.draft_specials.is_some()
            && s.draft_vocab.fits(anchor, |t| self.in_draft_subset(t));
        // Wider verification slowed short resumed replies, so cycles within
        // two blocks of the remaining budget keep the established costs, as
        // do full target rows (an active tool constraint): the small-head
        // curve does not price their transfers.
        pending.sampled_subset_cost = sampled
            && pending.gpu_verification
            && pending.draft_subset
            && max_tokens.min(room) > 2 * max_speculative;
        if pending.sampled_subset_cost && batch_drafts.is_none() {
            // Price the head actually used, after vocabulary fallback and the
            // copy bypass; the policy reads only committed history.
            pending.width =
                1 + s
                    .draft_length
                    .choose((cap - 1) as u32, base, pending.policy_concurrency, true)
                    as usize;
        }
        if !defer_head {
            let candidates = sampled.then_some(&mut pending.candidates);
            self.draft_catch_up(s, anchor, true, candidates, pending.draft_subset)?;
        }
        pending.draft = s.draft_token;
        if !defer_head {
            while pending.chain.len() < pending.width {
                pending.append_draft()?;
                // Drafting stops at a drafted end of generation.
                if stop_at_eos && pending.chain.last().is_some_and(|&t| self.is_stop_token(t)) {
                    pending.width = pending.chain.len();
                    break;
                }
                if self.append_lookup(s, pending, stop_at_eos, 1, false) {
                    break;
                }
                if pending.chain.len() < pending.width {
                    let draft = pending.draft;
                    let mut token = pending.draft;
                    let output = if sampled {
                        MtpOutput {
                            token: None,
                            candidates: Some(&mut pending.candidates),
                            kv_only: false,
                            draft_subset: pending.draft_subset,
                        }
                    } else {
                        MtpOutput {
                            token: Some(&mut token),
                            candidates: None,
                            kv_only: false,
                            draft_subset: pending.draft_subset,
                        }
                    };
                    self.executor
                        .mtp_forward(&mut s.native, &[draft], -1, output)?;
                    if !sampled {
                        pending.draft = token;
                    }
                }
            }
        }
        Ok(())
    }

    /// Appends prompt-lookup copies within the prepared budget, stopping at
    /// an end of generation when enabled (`AppendLookup`); true when it
    /// appended any. A sampled copy is a point-mass proposal: accepted with
    /// the target's p(token), and a rejection resamples from the target
    /// without it.
    pub(super) fn append_lookup(
        &self,
        s: &Session,
        pending: &mut PendingDecode,
        stop_at_eos: bool,
        minimum: usize,
        require_resume: bool,
    ) -> bool {
        let cap = pending.lookup_cap;
        let Some(&last) = pending.chain.last() else {
            return false;
        };
        if pending.chain.len() >= cap || (stop_at_eos && self.is_stop_token(last)) {
            return false;
        }
        let min_match = if pending.sampled {
            lookup::SAMPLED_MIN_MATCH
        } else {
            lookup::MIN_MATCH
        };
        let found = s
            .lookup
            .find(&s.tokens, &pending.chain, min_match, require_resume);
        if found.length == 0 {
            return false;
        }
        let available = (cap - pending.chain.len()).min(s.lookup.indexed_tokens() - found.start);
        if available < minimum {
            return false;
        }
        let mut count = 0;
        while count < available {
            let token = s.tokens[found.start + count];
            pending.chain.push(token);
            count += 1;
            if pending.sampled {
                let mut proposal = Proposal {
                    ids: [0; MTP_CANDIDATES],
                    probabilities: [0.0; MTP_CANDIDATES],
                    size: 1,
                    token: token as TokenId,
                    probability: 1.0,
                };
                proposal.ids[0] = token as TokenId;
                proposal.probabilities[0] = 1.0;
                pending.proposals.push(proposal);
            }
            if stop_at_eos && self.is_stop_token(token) {
                break;
            }
        }
        pending.lookup = count as u32;
        pending.lookup_start = found.start;
        pending.width = pending.chain.len();
        count != 0
    }

    /// Verifies the chain against the target rows, rolls back to the kept
    /// prefix and commits it (`FinishDecode`).
    pub(super) fn finish_decode(
        &mut self,
        s: &mut Session,
        sampler: &mut SamplerState,
        stop_at_eos: bool,
        pending: &PendingDecode,
        result: &mut DecodeResult,
    ) -> Result<(), String> {
        let chain = &pending.chain;
        let anchor = chain[0];
        let k = chain.len() as u32;
        let vocab = self.vocab_size() as usize;
        let stops = self.tokenizer.stop_tokens();
        let is_stop = |token: i32| {
            stop_at_eos && u32::try_from(token).is_ok_and(|token| stops.contains(token))
        };
        if !pending.speculative {
            s.draft_length.observe_ar_token();
            s.hidden_base = pending.base;
            s.tokens.push(anchor);
            sampler
                .accept(anchor as TokenId)
                .map_err(|e| e.to_string())?;
            result.tokens.push(anchor);
            return Ok(());
        }
        sampler
            .accept(anchor as TokenId)
            .map_err(|e| e.to_string())?;
        // Every row's argmax: the kept row's is the next anchor. Under
        // penalties the selection takes at most seven rows (C++ verifies only
        // the drafts' rows); without the kept row there is no anchor hint.
        let mut greedy = [ArgmaxCandidate::default(); MAX_DRAFT_TOKENS as usize + 1];
        let greedy_rows = if sampler.config().penalties_enabled() {
            (k as usize).min(crate::state::PENALTY_ROWS)
        } else {
            k as usize
        };
        if pending.gpu_greedy {
            self.executor.greedy_mtp_predictions(
                &mut greedy[..greedy_rows],
                sampler,
                &chain[1..],
            )?;
        }
        let mut keep = 1u32;
        let mut correction = None;
        let mut cpu_rows = false;
        // Downloads rows `first..k` once, for exact selection on the host.
        let max_rows = self.executor.options().max_speculative as usize * vocab;
        let download_remaining =
            |executor: &Executor, s: &mut Session, first: usize| -> Result<(), String> {
                if s.verify_logits.len() < max_rows {
                    s.verify_logits = vec![0.0; max_rows];
                }
                executor.read_verification_rows(
                    first,
                    &mut s.verify_logits[first * vocab..k as usize * vocab],
                )
            };
        let mut candidates = Vec::new();
        if pending.gpu_verification && !pending.gpu_greedy {
            // Accepting the anchor may activate a tool grammar or a
            // reasoning budget, or exceed the bounded penalty workspace:
            // those keep the complete rows.
            if mtp::can_compact_mtp_verification(sampler) {
                candidates =
                    self.executor
                        .sampled_mtp_candidates(k as usize - 1, sampler, &chain[1..])?;
            } else {
                download_remaining(&self.executor, s, 0)?;
                cpu_rows = true;
            }
        }
        while keep < k {
            let row = (keep - 1) as usize;
            if pending.gpu_greedy && !cpu_rows {
                let prediction = greedy[row];
                if !prediction.value.is_finite() {
                    return Err("logit distribution contains no finite values".into());
                }
                if !sampler
                    .can_select_argmax(prediction.index as TokenId, true)
                    .map_err(|e| e.to_string())?
                {
                    // Most native tool tokens obey the grammar already. At
                    // the first forbidden argmax the remaining rows download
                    // once for exact masked selection.
                    download_remaining(&self.executor, s, row)?;
                    cpu_rows = true;
                    continue;
                }
                if is_stop(prediction.index) {
                    result.stop = true;
                    break;
                }
                if prediction.index != chain[keep as usize] {
                    break;
                }
                sampler
                    .accept(prediction.index as TokenId)
                    .map_err(|e| e.to_string())?;
                keep += 1;
                continue;
            }
            if pending.sampled {
                if pending.gpu_verification
                    && !cpu_rows
                    && sampler.needs_constraint_mask().map_err(|e| e.to_string())?
                {
                    download_remaining(&self.executor, s, row)?;
                    cpu_rows = true;
                }
                let proposal = &pending.proposals[row];
                let verified = if pending.gpu_verification && !cpu_rows {
                    policy::verify_proposal_candidates(&candidates[row], vocab, proposal, sampler)?
                } else {
                    let logits = &s.verify_logits[row * vocab..(row + 1) * vocab];
                    policy::verify_proposal(logits, proposal, sampler)?
                };
                let token = verified.token as i32;
                if is_stop(token) {
                    result.stop = true;
                    break;
                }
                if !verified.accepted {
                    correction = Some(token);
                    break;
                }
                sampler.accept(verified.token).map_err(|e| e.to_string())?;
                keep += 1;
                continue;
            }
            let logits = &s.verify_logits[row * vocab..(row + 1) * vocab];
            let decision = policy::verify_draft(logits, chain[keep as usize], sampler, is_stop)?;
            if decision != DraftDecision::Accept {
                result.stop = decision == DraftDecision::Stop;
                break;
            }
            keep += 1;
        }
        let logits = if pending.gpu_verification && !cpu_rows {
            Some(&mut s.logits[..])
        } else {
            None
        };
        self.executor.rollback(&mut s.native, keep, logits)?;
        if !pending.gpu_verification || cpu_rows {
            let at = (keep - 1) as usize * vocab;
            s.logits.copy_from_slice(&s.verify_logits[at..at + vocab]);
        }
        s.hidden_base = pending.base;
        s.tokens.extend_from_slice(&chain[..keep as usize]);
        result.tokens = chain[..keep as usize].to_vec();
        s.stats.cycles += 1;
        s.stats.drafted += u64::from(k - 1);
        s.stats.accepted += u64::from(keep - 1);
        s.stats.small_head_cycles += u64::from(pending.draft_subset);
        // Copied proposals trail the MTP drafts. After a fully accepted MTP
        // part the length controller also counts the accepted copies as
        // accepted drafts: a copy that ended drafting early would otherwise
        // make a perfect chain look short. A rejected copy says nothing about
        // the MTP head, so it is not a failure.
        let mtp_drafts = k - 1 - pending.lookup;
        let mtp_accepted = (keep - 1).min(mtp_drafts);
        s.stats.lookup += u64::from(pending.lookup);
        s.stats.lookup_accepted += u64::from(keep - 1 - mtp_accepted);
        if pending.lookup != 0 {
            s.lookup.observe(
                pending.lookup_start,
                pending.lookup as usize,
                (keep - 1 - mtp_accepted) as usize,
            );
        }
        let mtp_all = mtp_accepted == mtp_drafts;
        let observed = if mtp_all { keep - 1 } else { mtp_accepted };
        // A target stop ends the request without failing the remaining
        // proposals.
        let drafted = if result.stop || mtp_all {
            observed
        } else {
            mtp_drafts
        };
        s.draft_length.observe(
            observed as usize,
            drafted as usize,
            pending.base,
            pending.policy_concurrency,
            pending.sampled_subset_cost,
        );
        // The next call knows the next anchor: catch-up waits until then.
        self.executor.mtp_rewind(&mut s.native, pending.base);
        let kept = greedy[keep as usize - 1];
        s.anchor_hint = (pending.gpu_greedy
            && !cpu_rows
            && correction.is_none()
            && (keep as usize) <= greedy_rows
            && kept.value.is_finite())
        .then(|| (kept.index as u32, s.native.mutation_epoch(), s.tokens.len()));
        if let Some(token) = correction {
            // The residual draw becomes the next anchor; resampling p would
            // bias it.
            sampler
                .defer_sample(token as TokenId)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Decodes up to `max_tokens` tokens: greedy decoding verifies
    /// deterministic drafts, sampled decoding draws compact proposals with
    /// rejection and residual correction; a one-token budget or a model
    /// without MTP decodes one token (`DecodeStep`).
    pub fn decode_step(
        &mut self,
        s: &mut Session,
        max_tokens: usize,
        sampler: &mut SamplerState,
        stop_at_eos: bool,
    ) -> Result<DecodeResult, String> {
        if !s.valid {
            return Err("session needs a successful Sync after a failed operation".into());
        }
        s.valid = false;
        let mut pending = PendingDecode::default();
        let mut result = DecodeResult::default();
        self.prepare_decode(
            s,
            max_tokens,
            sampler,
            stop_at_eos,
            &mut pending,
            &mut result,
            false,
            None,
            1,
        )?;
        if pending.chain.is_empty() {
            s.valid = true;
            return Ok(result);
        }
        if pending.lookup != 0 {
            self.batch_policy.skip_cycle();
        }
        let (mode, rows) = if pending.speculative {
            (ForwardMode::Verify, pending.chain.len() as u32)
        } else {
            (ForwardMode::Decode, pending.chain.len() as u32)
        };
        let logits = if !pending.speculative {
            Some(&mut s.logits[..])
        } else if pending.gpu_verification {
            None
        } else {
            Some(&mut s.verify_logits[..])
        };
        self.executor
            .forward(&mut s.native, &pending.chain, rows, logits, mode)?;
        self.finish_decode(s, sampler, stop_at_eos, &pending, &mut result)?;
        s.valid = true;
        Ok(result)
    }
}
