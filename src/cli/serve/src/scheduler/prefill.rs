//! Prefill steps and the hand-over to decoding.

use super::*;

impl<R: TextModelRunner> Worker<R> {
    pub(super) fn step_prefill(
        &mut self,
        work: &mut Work<R>,
        decoder_runnable: bool,
        peer_waiting: bool,
    ) {
        if self.complete_if_stopped(work) {
            return;
        }
        if let Err(failure) = self.try_step_prefill(work, decoder_runnable, peer_waiting) {
            self.fail_unless_stopped(work, failure);
        }
    }

    pub(super) fn try_step_prefill(
        &mut self,
        work: &mut Work<R>,
        decoder_runnable: bool,
        peer_waiting: bool,
    ) -> Result<()> {
        work.shared.set_phase(TextRequestPhase::Prefilling);
        self.last_prefill_request_id = work.shared.id;
        // Bound the work for short waiting prefills and captures as well as
        // for decoders. Long prefills alone keep the model's efficient chunk
        // size.
        // Bounded chunks also while a GPU grant limits the share, so realtime
        // servers get the device between them.
        let bounded = self.incremental_prefill_supported
            && (decoder_runnable || peer_waiting || self.gpu_limited());
        let budget = if bounded {
            self.prefill_policy.decode_active_tokens
        } else {
            work.runner()?.prompt_tokens()
        };
        if (decoder_runnable || self.pool.capacity() > 1) && !self.incremental_prefill_supported {
            work.result.prefill_fallback_reason = "incremental_prefill_unavailable".into();
        }
        let start = Instant::now();
        let step = work.runner()?.prefill(budget)?;
        let step_ms = elapsed_ms(start);
        // Only a step that ran alone measures the cost a wait saves.
        if !decoder_runnable && !peer_waiting {
            self.publication_estimate
                .observe(step.consumed_tokens, step_ms);
        }
        let result = &mut work.result;
        result.prefill_ms += step_ms;
        result.prefill_tokens += step.consumed_tokens;
        work.shared
            .live_prefill_tokens
            .store(result.prefill_tokens, Ordering::Relaxed);
        metrics::add_prompt_tokens(step.consumed_tokens);
        result.prefill_chunks += 1;
        result.max_prefill_chunk_tokens = result.max_prefill_chunk_tokens.max(step.consumed_tokens);
        self.log_prefill_progress(work, step.consumed_tokens, step_ms);
        work.publish_prompt_progress();
        let result = &mut work.result;
        if decoder_runnable {
            result.active_decode_prefill_chunks += 1;
            self.consecutive_active_prefill_chunks += 1;
            result.max_consecutive_active_prefill_chunks = result
                .max_consecutive_active_prefill_chunks
                .max(self.consecutive_active_prefill_chunks);
        } else {
            self.consecutive_active_prefill_chunks = 0;
        }
        if self.complete_if_stopped(work) {
            return Ok(());
        }
        work.shared.set_phase(if step.decode_ready {
            TextRequestPhase::DecodeReady
        } else {
            TextRequestPhase::Prefilling
        });
        Ok(())
    }

    /// Selects and publishes the next token, leaving it to advance; `None`
    /// when the request finished or waits for its prompt snapshot.
    pub(super) fn prepare_decode(&mut self, work: &mut Work<R>) -> Result<Option<Instant>> {
        loop {
            if self.complete_if_stopped(work) {
                return Ok(None);
            }
            if work.advance_pending {
                if !work.runner()?.prepare_prompt_snapshot() {
                    return Ok(None);
                }
                work.advance_pending = false;
                if !self.final_token_advance_required
                    && work.result.tokens.len() >= work.token_limit
                {
                    self.complete_success(work, FinishReason::Length)?;
                    return Ok(None);
                }
                // Snapshot capture and other requests' work are outside this
                // step.
                return Ok(Some(Instant::now()));
            }
            work.shared.set_phase(TextRequestPhase::Decoding);
            let decode_start = Instant::now();
            let selection = work.runner()?.select_next()?;
            if selection.stop {
                work.result.decode_ms += elapsed_ms(decode_start);
                self.complete_success(work, FinishReason::Stop)?;
                return Ok(None);
            }
            if !self.publish_selection(work, &selection) {
                return Ok(None);
            }
            work.result.decode_ms += elapsed_ms(decode_start);
            if work.stop_filter.stopped() {
                self.complete_success(work, FinishReason::StopSequence)?;
                return Ok(None);
            }
            work.advance_pending = true;
        }
    }

    pub(super) fn publish_selection(
        &self,
        work: &mut Work<R>,
        selection: &TextDecodeSelection,
    ) -> bool {
        let now = Instant::now();
        match work.previous_token {
            None => {
                work.result.ttft_ms = (now - work.request_start).as_secs_f64() * 1e3;
            }
            Some(previous) => {
                let inter_token = now - previous;
                work.inter_token_total += inter_token;
                work.result.max_inter_token_ms = work
                    .result
                    .max_inter_token_ms
                    .max(inter_token.as_secs_f64() * 1e3);
                work.inter_token_samples += 1;
            }
        }
        work.previous_token = Some(now);
        if selection.piece.len() > work.max_output_bytes - work.generated_output_bytes {
            self.complete_failure(
                work,
                Error::generation(
                    GenerationErrorCode::OutputLimit,
                    "text generation output byte limit exceeded",
                ),
            );
            return false;
        }
        work.generated_output_bytes += selection.piece.len();
        work.result.tokens.push(selection.token);
        work.shared
            .live_generated_tokens
            .store(work.result.tokens.len(), Ordering::Relaxed);
        metrics::add_generated_tokens(1);
        let piece = if work.stop_filter.enabled() {
            work.stop_filter.push(&selection.piece)
        } else {
            selection.piece.clone()
        };
        // Without a stop filter every token is an event, also one without
        // text; a filter publishes only released bytes.
        if (!piece.is_empty() || !work.stop_filter.enabled())
            && !work.shared.publish_piece(piece.clone())
        {
            self.complete_failure(
                work,
                Error::generation(
                    GenerationErrorCode::OutputBackpressure,
                    "text generation buffered output limit exceeded",
                ),
            );
            return false;
        }
        if self.incremental_text_is_exact {
            work.result.text.extend_from_slice(&piece);
        }
        if let Some(sequence) = work.stop_filter.matched_sequence() {
            work.result.stop_sequence = sequence.to_owned();
        }
        !self.complete_if_stopped(work)
    }

    pub(super) fn finish_advanced(&self, work: &mut Work<R>, decode_start: Instant) -> Result<()> {
        work.result.decode_ms += elapsed_ms(decode_start);
        self.log_decode_progress(work, false);
        if self.complete_if_stopped(work) {
            return Ok(());
        }
        if work.result.tokens.len() >= work.token_limit {
            self.complete_success(work, FinishReason::Length)?;
        }
        Ok(())
    }
}
