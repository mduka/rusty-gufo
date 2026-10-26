//! Decode steps: single tokens, multi-token drafts and batches.

use super::*;

impl<R: TextModelRunner> Worker<R> {
    pub(super) fn step_decode(&mut self, work: &mut Work<R>) {
        self.consecutive_active_prefill_chunks = 0;
        if self.multi_token_decode {
            self.step_multi_token_decode(work);
            return;
        }
        let stepped = (|| {
            let Some(decode_start) = self.prepare_decode(work)? else {
                return Ok(());
            };
            work.runner()?.advance()?;
            self.finish_advanced(work, decode_start)
        })();
        if let Err(failure) = stepped {
            self.fail_unless_stopped(work, failure);
        }
    }

    pub(super) fn step_multi_token_decode(&mut self, work: &mut Work<R>) {
        self.consecutive_active_prefill_chunks = 0;
        if let Err(failure) = self.try_step_multi_token_decode(work) {
            self.fail_unless_stopped(work, failure);
        }
    }

    pub(super) fn try_step_multi_token_decode(&mut self, work: &mut Work<R>) -> Result<()> {
        if self.complete_if_stopped(work) {
            return Ok(());
        }
        work.shared.set_phase(TextRequestPhase::Decoding);
        if !self.prepare_first_snapshot(work)? {
            return Ok(());
        }
        let decode_start = Instant::now();
        let remaining =
            work.token_limit - work.result.tokens.len() + usize::from(work.preview_token.is_some());
        let step = work.runner()?.decode_step(remaining)?;
        work.result.draft_rounds += step.draft_rounds;
        work.result.draft_tokens += step.draft_tokens;
        work.result.draft_accepted_tokens += step.draft_accepted_tokens;
        work.result.lookup_tokens += step.lookup_tokens;
        work.result.lookup_accepted_tokens += step.lookup_accepted_tokens;
        check_preview_result(work, &step)?;
        for selection in &step.selections {
            if !self.publish_decoded_selection(work, selection) {
                return Ok(());
            }
            if work.stop_filter.stopped() {
                break;
            }
        }
        work.result.decode_ms += elapsed_ms(decode_start);
        self.log_decode_progress(work, false);
        if work.stop_filter.stopped() {
            return self.complete_success(work, FinishReason::StopSequence);
        }
        if step.stop {
            return self.complete_success(work, FinishReason::Stop);
        }
        if work.result.tokens.len() >= work.token_limit {
            return self.complete_success(work, FinishReason::Length);
        }
        Ok(())
    }

    pub(super) fn prepare_first_snapshot(&mut self, work: &mut Work<R>) -> Result<bool> {
        if work.result.tokens.is_empty() && !work.preview_stops {
            let preview_start = Instant::now();
            let preview = work.runner()?.preview_first_token()?;
            if let Some(preview) = preview.as_ref().filter(|p| !p.stop) {
                if !self.publish_selection(work, preview) {
                    return Ok(false);
                }
                work.preview_token = Some(preview.token);
            }
            work.preview_stops = preview.is_some_and(|p| p.stop);
            work.result.decode_ms += elapsed_ms(preview_start);
            self.log_decode_progress(work, false);
        }
        if work.stop_filter.stopped() {
            self.complete_success(work, FinishReason::StopSequence)?;
            return Ok(false);
        }
        if !work.runner()?.prepare_prompt_snapshot() {
            return Ok(false);
        }
        if work.preview_stops {
            self.complete_success(work, FinishReason::Stop)?;
            return Ok(false);
        }
        Ok(!self.complete_if_stopped(work))
    }

    pub(super) fn publish_decoded_selection(
        &self,
        work: &mut Work<R>,
        selection: &TextDecodeSelection,
    ) -> bool {
        if work.preview_token.take().is_some() {
            return true;
        }
        self.publish_selection(work, selection)
    }

    /// Fails the prepared lanes of `batch` once the device is lost.
    pub(super) fn fail_all_if_device_lost(
        &self,
        batch: &mut Batch<R>,
        prepared: impl Iterator<Item = usize>,
    ) -> bool {
        if !self.device_lost() {
            return false;
        }
        for index in prepared {
            self.complete_classified_failure(&mut batch[index], device_lost_error());
        }
        true
    }

    pub(super) fn step_decode_batch(&mut self, batch: &mut Batch<R>) {
        let selected_plan = match self.pool.select_decode_plan(batch.len()) {
            Ok(plan) => plan,
            Err(failure) => {
                for work in batch.iter_mut() {
                    self.complete_failure(work, failure.clone());
                }
                return;
            }
        };
        if self.batched_multi_token_decode
            && (self.batched_multi_token_decode_max_width == 0
                || selected_plan.physical_width <= self.batched_multi_token_decode_max_width)
        {
            self.step_multi_token_decode_batch(batch);
            return;
        }
        if batch.len() == 1 {
            self.step_decode(&mut batch[0]);
            return;
        }
        self.consecutive_active_prefill_chunks = 0;
        let mut prepared: Vec<(usize, Instant)> = Vec::with_capacity(batch.len());
        for (index, work) in batch.iter_mut().enumerate() {
            match self.prepare_decode(work) {
                Ok(Some(decode_start)) => prepared.push((index, decode_start)),
                Ok(None) => {}
                Err(failure) => self.complete_failure(work, failure),
            }
        }
        let advance_one = |this: &Self, work: &mut Work<R>, decode_start: Instant| {
            let advanced = work
                .runner()
                .and_then(|r| r.advance())
                .and_then(|()| this.finish_advanced(work, decode_start));
            if let Err(failure) = advanced {
                this.complete_failure(work, failure);
            }
        };
        match prepared.len() {
            0 => return,
            1 => {
                let (index, decode_start) = prepared[0];
                advance_one(self, &mut batch[index], decode_start);
                return;
            }
            _ => {}
        }
        if self.fail_all_if_device_lost(batch, prepared.iter().map(|p| p.0)) {
            return;
        }
        let plan = match self.pool.select_decode_plan(prepared.len()) {
            Ok(plan) if plan.kind == TextExecutionPlanKind::Batched => plan,
            _ => {
                for &(index, decode_start) in &prepared {
                    advance_one(self, &mut batch[index], decode_start);
                }
                return;
            }
        };
        let results = {
            let mut requests: Vec<&mut PoolRequest<R>> = Vec::with_capacity(prepared.len());
            let mut next = prepared.iter().map(|(i, _)| *i).peekable();
            for (index, work) in batch.iter_mut().enumerate() {
                if next.peek() == Some(&index) {
                    next.next();
                    if let Some(request) = work.runner_request.as_mut() {
                        requests.push(request);
                    }
                }
            }
            if requests.len() == prepared.len() {
                self.pool.advance_batch(&mut requests, &plan)
            } else {
                Err(Error::logic("invalid request in batched text advance"))
            }
        };
        let results = match results {
            Ok(results) => results,
            Err(failure) => {
                for &(index, _) in &prepared {
                    self.fail_unless_stopped(&mut batch[index], failure.clone());
                }
                return;
            }
        };
        // Classify every failed lane before committing a successful peer: its
        // commit may touch the context that just failed.
        let results: Vec<Result<()>> = results
            .into_iter()
            .map(|result| result.map_err(|failure| self.classify_failure(failure)))
            .collect();
        if self.fail_all_if_device_lost(batch, prepared.iter().map(|p| p.0)) {
            return;
        }
        let execution_plan = format!("batched-w{}", plan.physical_width);
        for (&(index, decode_start), result) in prepared.iter().zip(results) {
            let work = &mut batch[index];
            if let Err(failure) = result {
                if !self.complete_if_stopped(work) {
                    self.complete_classified_failure(work, failure);
                }
                continue;
            }
            work.result.physical_execution_width = work
                .result
                .physical_execution_width
                .max(plan.physical_width);
            work.result.execution_plan = execution_plan.clone();
            if let Err(failure) = self.finish_advanced(work, decode_start) {
                self.complete_failure(work, failure);
            }
        }
    }

    pub(super) fn step_multi_token_decode_batch(&mut self, batch: &mut Batch<R>) {
        if batch.len() == 1 {
            self.step_multi_token_decode(&mut batch[0]);
            return;
        }
        self.consecutive_active_prefill_chunks = 0;
        let mut prepared: Vec<(usize, Instant, usize)> = Vec::with_capacity(batch.len());
        for (index, work) in batch.iter_mut().enumerate() {
            if self.complete_if_stopped(work) {
                continue;
            }
            work.shared.set_phase(TextRequestPhase::Decoding);
            match self.prepare_first_snapshot(work) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(failure) => {
                    self.complete_failure(work, failure);
                    continue;
                }
            }
            let remaining = work.token_limit - work.result.tokens.len()
                + usize::from(work.preview_token.is_some());
            prepared.push((index, Instant::now(), remaining));
        }
        match prepared.len() {
            0 => return,
            1 => {
                self.step_multi_token_decode(&mut batch[prepared[0].0]);
                return;
            }
            _ => {}
        }
        if self.fail_all_if_device_lost(batch, prepared.iter().map(|p| p.0)) {
            return;
        }
        let plan = match self.pool.select_decode_plan(prepared.len()) {
            Ok(plan) if plan.kind == TextExecutionPlanKind::Batched => plan,
            _ => {
                for &(index, _, _) in &prepared {
                    self.step_multi_token_decode(&mut batch[index]);
                }
                return;
            }
        };
        let max_tokens: Vec<usize> = prepared.iter().map(|p| p.2).collect();
        let steps = {
            let mut requests: Vec<&mut PoolRequest<R>> = Vec::with_capacity(prepared.len());
            let mut next = prepared.iter().map(|p| p.0).peekable();
            for (index, work) in batch.iter_mut().enumerate() {
                if next.peek() == Some(&index) {
                    next.next();
                    if let Some(request) = work.runner_request.as_mut() {
                        requests.push(request);
                    }
                }
            }
            if requests.len() == prepared.len() {
                self.pool.decode_batch(&mut requests, &max_tokens, &plan)
            } else {
                Err(Error::logic("invalid request in batched text decode"))
            }
        };
        let steps = match steps {
            Ok(steps) => steps,
            Err(failure) => {
                for &(index, _, _) in &prepared {
                    self.fail_unless_stopped(&mut batch[index], failure.clone());
                }
                return;
            }
        };
        let steps: Vec<Result<TextDecodeStep>> = steps
            .into_iter()
            .map(|step| step.map_err(|failure| self.classify_failure(failure)))
            .collect();
        if self.fail_all_if_device_lost(batch, prepared.iter().map(|p| p.0)) {
            return;
        }
        let capacity = self.pool.capacity();
        for (&(index, decode_start, _), step) in prepared.iter().zip(steps) {
            let work = &mut batch[index];
            let step = match step {
                Ok(step) => step,
                Err(failure) => {
                    if !self.complete_if_stopped(work) {
                        self.complete_classified_failure(work, failure);
                    }
                    continue;
                }
            };
            let width = step.execution_plan.physical_width;
            if width >= work.result.physical_execution_width {
                work.result.physical_execution_width = width;
                work.result.execution_plan =
                    if step.execution_plan.kind == TextExecutionPlanKind::Batched {
                        format!("batched-w{width}")
                    } else if capacity == 1 {
                        "serial-c1".into()
                    } else {
                        "serial-fallback".into()
                    };
            }
            work.result.draft_rounds += step.draft_rounds;
            work.result.draft_tokens += step.draft_tokens;
            work.result.draft_accepted_tokens += step.draft_accepted_tokens;
            work.result.lookup_tokens += step.lookup_tokens;
            work.result.lookup_accepted_tokens += step.lookup_accepted_tokens;
            if let Err(failure) = check_preview_result(work, &step) {
                self.complete_failure(work, failure);
                continue;
            }
            let mut published = true;
            for selection in &step.selections {
                if !self.publish_decoded_selection(work, selection) {
                    published = false;
                    break;
                }
                if work.stop_filter.stopped() {
                    break;
                }
            }
            work.result.decode_ms += elapsed_ms(decode_start);
            self.log_decode_progress(work, false);
            if !published || work.is_terminal() {
                continue;
            }
            let finished = if work.stop_filter.stopped() {
                self.complete_success(work, FinishReason::StopSequence)
            } else if step.stop {
                self.complete_success(work, FinishReason::Stop)
            } else if work.result.tokens.len() >= work.token_limit {
                self.complete_success(work, FinishReason::Length)
            } else {
                Ok(())
            };
            if let Err(failure) = finished {
                self.complete_failure(work, failure);
            }
        }
    }
}
