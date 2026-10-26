//! The worker thread: its loop, progress logs, the queue and how requests
//! complete.

use super::*;

/// The scheduler thread.
pub(super) struct Worker<R: TextModelRunner> {
    pub(super) shared: Arc<SchedulerShared<R>>,
    pub(super) pool: Arc<TextRunnerPool<R>>,
    pub(super) prefill_policy: TextPrefillPolicy,
    pub(super) incremental_prefill_supported: bool,
    pub(super) final_token_advance_required: bool,
    pub(super) incremental_text_is_exact: bool,
    pub(super) multi_token_decode: bool,
    pub(super) batched_multi_token_decode: bool,
    pub(super) batched_multi_token_decode_max_width: usize,
    pub(super) log_progress: bool,
    pub(super) device_probe_interval: Duration,
    pub(super) consecutive_active_prefill_chunks: usize,
    /// The request whose prefill ran last.
    pub(super) last_prefill_request_id: u64,
    /// The cost of recent prefill steps that ran alone.
    pub(super) publication_estimate: PublicationWaitEstimate,
    /// The arbiter's GPU grant; `None` without `--arbiter`.
    pub(super) gpu_share: Option<Arc<GpuShare>>,
    /// The requests resident when the device was lost, kept with their
    /// states, snapshots and workers until process exit: resetting, freeing
    /// or joining on the dead context can block or crash first.
    pub(super) retained: RefCell<Vec<PoolRequest<R>>>,
}

/// Decoded tokens between two progress lines.
/// The longest single sleep of a pause for a GPU grant.
pub(super) const GPU_YIELD_SLICE: Duration = Duration::from_millis(5);

pub(super) const DECODE_PROGRESS_INTERVAL: usize = 50;

pub(super) fn tokens_per_second(tokens: usize, elapsed_ms: f64) -> f64 {
    if elapsed_ms > 0.0 {
        tokens as f64 * 1000.0 / elapsed_ms
    } else {
        0.0
    }
}

pub(super) fn check_preview_result<R: TextModelRunner>(
    work: &Work<R>,
    step: &TextDecodeStep,
) -> Result<()> {
    if let Some(preview) = work.preview_token
        && step.selections.first().is_none_or(|s| s.token != preview)
    {
        return Err(Error::runtime(
            "first-token preview disagrees with decoding",
        ));
    }
    Ok(())
}

/// The next due decoder (rotating the others to the back), or the front.
pub(super) fn pop_decoder<R: TextModelRunner>(
    decoding: &mut VecDeque<Box<Work<R>>>,
    require_due: bool,
) -> Option<Box<Work<R>>> {
    for _ in 0..decoding.len() {
        let work = decoding.pop_front()?;
        if !require_due || work.decode_due {
            return Some(work);
        }
        decoding.push_back(work);
    }
    decoding.pop_front()
}

impl<R: TextModelRunner> Worker<R> {
    pub(super) fn stop_requested(&self) -> bool {
        self.shared.stop.load(Ordering::Acquire)
    }

    /// `--log-progress`: one line per prefill chunk.
    pub(super) fn log_prefill_progress(&self, work: &Work<R>, chunk_tokens: usize, chunk_ms: f64) {
        if !self.log_progress || !logging::enabled(LogLevel::Info) {
            return;
        }
        let result = &work.result;
        let total = result.prompt_tokens - result.cached_prompt_tokens.min(result.prompt_tokens);
        let processed = result.prefill_tokens.min(total);
        let percentage = if total > 0 {
            processed as f64 * 100.0 / total as f64
        } else {
            100.0
        };
        logging::info(
            "progress",
            &format!(
                "request={} phase=prefill tokens={processed}/{total} percentage={percentage:.1} \
                 chunk_tps={:.1} avg_tps={:.1}",
                work.shared.id,
                tokens_per_second(chunk_tokens, chunk_ms),
                tokens_per_second(result.prefill_tokens, result.prefill_ms)
            ),
        );
    }

    /// `--log-progress`: a line at every 50-token boundary and, when
    /// `last`, for the remainder.
    pub(super) fn log_decode_progress(&self, work: &mut Work<R>, last: bool) {
        if !self.log_progress || !logging::enabled(LogLevel::Info) {
            return;
        }
        let generated = work.result.tokens.len();
        let logged = work.last_decode_progress_tokens;
        if generated == 0
            || generated == logged
            || (!last && generated / DECODE_PROGRESS_INTERVAL == logged / DECODE_PROGRESS_INTERVAL)
        {
            return;
        }
        let result = &work.result;
        let percentage = if work.token_limit > 0 {
            generated as f64 * 100.0 / work.token_limit as f64
        } else {
            100.0
        };
        let mut line = format!(
            "request={} phase=decode tokens={generated}/{} percentage={percentage:.1} \
             chunk_tps={:.1} avg_tps={:.1}",
            work.shared.id,
            work.token_limit,
            tokens_per_second(
                generated - logged,
                result.decode_ms - work.last_decode_progress_ms
            ),
            tokens_per_second(generated, result.decode_ms)
        );
        if result.draft_tokens > 0 {
            let _ = write!(
                line,
                " draft_accepted={} draft_proposed={} acceptance_percentage={:.1}",
                result.draft_accepted_tokens,
                result.draft_tokens,
                result.draft_accepted_tokens as f64 * 100.0 / result.draft_tokens as f64
            );
        }
        logging::info("progress", &line);
        work.last_decode_progress_tokens = generated;
        work.last_decode_progress_ms = result.decode_ms;
    }

    /// Admits the next queued request into a session.
    pub(super) fn pop_queued(&self) -> Option<Box<Work<R>>> {
        let mut queue = self.shared.queue();
        let mut client = queue.clients.pop_front()?;
        let mut work = client.requests.pop_front();
        if let Some(work) = work.as_mut() {
            // Admission reserves a session before potentially slow cache
            // preparation: publish the transfer before the deferred count
            // drops. Admit holds fewer than capacity requests here, so a
            // session is free.
            work.counted_processing = true;
            metrics::add_requests_processing(1);
            let count = queue.count - 1;
            queue.set_count(count);
            let mut sessions = lock(&self.shared.sessions);
            if let Some(free) = sessions.iter().position(Option::is_none) {
                sessions[free] = Some(work.shared.clone());
                work.session = Some(free);
            }
        }
        if !client.requests.is_empty() {
            queue.clients.push_back(client);
        }
        work
    }

    pub(super) fn remove_queued(&self, id: u64) -> Option<Box<Work<R>>> {
        let mut queue = self.shared.queue();
        for index in 0..queue.clients.len() {
            let client = &mut queue.clients[index];
            let Some(position) = client.requests.iter().position(|w| w.shared.id == id) else {
                continue;
            };
            let work = client.requests.remove(position);
            let emptied = client.requests.is_empty();
            let count = queue.count - 1;
            queue.set_count(count);
            if emptied {
                queue.clients.remove(index);
            }
            return work;
        }
        None
    }

    pub(super) fn queued_snapshot(&self) -> Vec<Arc<RequestShared>> {
        let queue = self.shared.queue();
        queue
            .clients
            .iter()
            .flat_map(|c| c.requests.iter().map(|w| w.shared.clone()))
            .collect()
    }

    pub(super) fn finalize_result(
        &self,
        work: &mut Work<R>,
        finish_reason: FinishReason,
    ) -> Result<()> {
        let result = &mut work.result;
        result.completion_tokens = result.tokens.len();
        result.finish_reason = finish_reason;
        if work.inter_token_samples > 0 {
            result.mean_inter_token_ms =
                work.inter_token_total.as_secs_f64() * 1e3 / work.inter_token_samples as f64;
        }
        if !self.incremental_text_is_exact {
            result.text = self.pool.runner().decode(&result.tokens)?;
        }
        Ok(())
    }

    pub(super) fn complete_cancelled(&self, work: &mut Work<R>) {
        if let Some(request) = work.runner_request.as_mut().filter(|r| r.is_active()) {
            let retained = request.cancel();
            work.result.cache_snapshot_bytes = retained.snapshot_bytes;
            work.result.cache_snapshot_ms = retained.snapshot_ms;
            work.result.cache_disk_queued_bytes = retained.disk_queued_bytes;
            work.result.cache_disk_enqueue_ms = retained.disk_enqueue_ms;
        }
        work.result.cancelled = true;
        match self.finalize_result(work, FinishReason::Cancelled) {
            Ok(()) => {
                self.log_decode_progress(work, true);
                metrics::record_request_metrics(&work.result);
                work.publish_terminal(None, true);
            }
            Err(error) => work.publish_terminal(Some(error), true),
        }
    }

    pub(super) fn device_lost(&self) -> bool {
        self.shared.device_lost.load(Ordering::Acquire)
    }

    /// A GPU reset leaves this process's context unusable for good, and every
    /// later work unit then fails with a raw driver message. After a model
    /// failure, on this thread between work units, a bounded probe tells the
    /// two apart.
    pub(super) fn classify_failure(&self, failure: Error) -> Error {
        if self.device_lost() {
            return device_lost_error();
        }
        if failure.generation_code().is_some() {
            return failure;
        }
        let deadline = Instant::now() + DEVICE_PROBE_TIMEOUT;
        loop {
            match self.pool.runner().poll_device() {
                DeviceStatus::Usable => return failure,
                DeviceStatus::Lost => break,
                // Possibly queued behind long kernels: counts as usable.
                DeviceStatus::Pending if Instant::now() >= deadline => {
                    logging::warn(
                        "scheduler",
                        &format!(
                            "event=device_probe_timeout timeout_s={}",
                            DEVICE_PROBE_TIMEOUT.as_secs()
                        ),
                    );
                    return failure;
                }
                DeviceStatus::Pending => std::thread::sleep(Duration::from_millis(1)),
            }
        }
        self.mark_device_lost(failure.message());
        device_lost_error()
    }

    pub(super) fn mark_device_lost(&self, reason: &str) {
        self.shared.mark_device_lost(reason);
    }

    pub(super) fn advance_progress(&self) {
        self.shared.progress.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn complete_failure(&self, work: &mut Work<R>, failure: Error) {
        let failure = self.classify_failure(failure);
        self.complete_classified_failure(work, failure);
    }

    pub(super) fn complete_classified_failure(&self, work: &mut Work<R>, failure: Error) {
        if work.is_terminal() {
            return;
        }
        if self.device_lost() {
            if let Some(request) = work.runner_request.take() {
                self.retained.borrow_mut().push(request);
            }
        } else if let Some(request) = work.runner_request.as_mut() {
            request.invalidate();
        }
        work.result.completion_tokens = work.result.tokens.len();
        self.log_decode_progress(work, true);
        work.publish_terminal(Some(failure), true);
    }

    pub(super) fn complete_deadline(&self, work: &mut Work<R>) {
        self.complete_failure(
            work,
            Error::generation(
                GenerationErrorCode::DeadlineExceeded,
                "text generation request deadline exceeded",
            ),
        );
    }

    pub(super) fn complete_if_stopped(&self, work: &mut Work<R>) -> bool {
        if work.is_terminal() {
            return true;
        }
        if self.device_lost() {
            self.complete_classified_failure(work, device_lost_error());
            return true;
        }
        if work.shared.deadline_exceeded() {
            self.complete_deadline(work);
            return true;
        }
        if work.shared.cancellation_requested() {
            self.complete_cancelled(work);
            return true;
        }
        false
    }

    pub(super) fn complete_success(
        &self,
        work: &mut Work<R>,
        finish_reason: FinishReason,
    ) -> Result<()> {
        if work.stop_filter.enabled() {
            let tail = work.stop_filter.finish();
            work.result.text.extend_from_slice(&tail);
            if !tail.is_empty() && !work.shared.publish_piece(tail) {
                return Err(Error::generation(
                    GenerationErrorCode::OutputBackpressure,
                    "text generation buffered output limit exceeded",
                ));
            }
        }
        // A stop may land on a preview, an unadvanced token or inside an
        // accepted speculative block: retain only correctly labelled
        // executed state, as an interrupted request does.
        let commit = if finish_reason == FinishReason::StopSequence {
            work.runner()?.cancel()
        } else {
            work.runner()?.commit()?
        };
        let result = &mut work.result;
        result.cache_snapshot_bytes = commit.snapshot_bytes;
        result.cache_snapshot_ms = commit.snapshot_ms;
        result.cache_disk_queued_bytes = commit.disk_queued_bytes;
        result.cache_disk_enqueue_ms = commit.disk_enqueue_ms;
        result.cache_shared_prefix_snapshots = commit.shared_prefix_snapshots;
        result.cache_shared_prefix_bytes = commit.shared_prefix_bytes;
        result.cache_shared_prefix_ms = commit.shared_prefix_ms;
        self.finalize_result(work, finish_reason)?;
        self.log_decode_progress(work, true);
        // Recorded here rather than by the HTTP adapter, so a client that
        // leaves before reading the result still counts.
        metrics::record_request_metrics(&work.result);
        work.publish_terminal(None, false);
        Ok(())
    }

    /// `if (!CompleteIfStopped(request)) CompleteFailure(request, failure)`.
    pub(super) fn fail_unless_stopped(&self, work: &mut Work<R>, failure: Error) {
        if !self.complete_if_stopped(work) {
            self.complete_failure(work, failure);
        }
    }

    pub(super) fn process_queued_cancellations(&self) {
        for shared in self.queued_snapshot() {
            let deadline_exceeded = shared.deadline_exceeded();
            let cancelled = !deadline_exceeded && shared.cancellation_requested();
            if !(deadline_exceeded || cancelled) {
                continue;
            }
            let Some(mut work) = self.remove_queued(shared.id) else {
                continue;
            };
            if deadline_exceeded {
                self.complete_deadline(&mut work);
            } else {
                self.complete_cancelled(&mut work);
            }
        }
    }

    pub(super) fn cancel_remaining(
        &self,
        prefilling: &mut VecDeque<Box<Work<R>>>,
        decoding: &mut VecDeque<Box<Work<R>>>,
        waiting: &mut VecDeque<Box<Work<R>>>,
    ) {
        let remaining: Vec<Box<Work<R>>> = {
            let mut queue = self.shared.queue();
            queue.set_count(0);
            queue
                .clients
                .drain(..)
                .flat_map(|client| client.requests)
                .collect()
        };
        for mut work in remaining
            .into_iter()
            .chain(prefilling.drain(..))
            .chain(decoding.drain(..))
            .chain(waiting.drain(..))
        {
            if self.device_lost() {
                self.complete_classified_failure(&mut work, device_lost_error());
            } else {
                self.complete_cancelled(&mut work);
            }
        }
        if self.device_lost() {
            // The process exits; dropping these would touch the dead device.
            std::mem::forget(self.retained.take());
        }
    }

    pub(super) fn requeue_after_prefill(
        work: Box<Work<R>>,
        prefilling: &mut VecDeque<Box<Work<R>>>,
        decoding: &mut VecDeque<Box<Work<R>>>,
        capturing: &mut VecDeque<Box<Work<R>>>,
        mark_due: bool,
    ) {
        if work.is_terminal() {
            return;
        }
        if work.snapshot_pending() {
            capturing.push_back(work);
        } else if work.prefill_complete() {
            let mut work = work;
            if mark_due {
                work.decode_due = true;
            }
            decoding.push_back(work);
        } else {
            prefilling.push_back(work);
        }
    }

    /// Pauses for the arbiter's GPU grant after a step that began at
    /// `started`.
    pub(super) fn yield_gpu(&self, started: Instant) {
        if let Some(share) = &self.gpu_share {
            share.pause_after(started.elapsed(), GPU_YIELD_SLICE, &|| {
                self.stop_requested()
            });
        }
    }

    /// Whether a grant currently limits the GPU share.
    pub(super) fn gpu_limited(&self) -> bool {
        self.gpu_share
            .as_ref()
            .is_some_and(|share| share.current() < 1.0)
    }

    pub(super) fn run(mut self) {
        let mut prefilling: VecDeque<Box<Work<R>>> = VecDeque::new();
        let mut decoding: VecDeque<Box<Work<R>>> = VecDeque::new();
        let mut capturing: VecDeque<Box<Work<R>>> = VecDeque::new();
        // Cold requests parked behind a peer's prefill; they hold no session.
        let mut waiting: VecDeque<Box<Work<R>>> = VecDeque::new();
        // A lost device ends the loop: the remaining requests fail without
        // touching it, and new submissions are refused.
        while !self.stop_requested() && !self.device_lost() {
            self.shared.loop_working.store(true, Ordering::Relaxed);
            self.process_queued_cancellations();

            for _ in 0..capturing.len() {
                let Some(mut work) = capturing.pop_front() else {
                    break;
                };
                if work.snapshot_pending() {
                    capturing.push_back(work);
                    continue;
                }
                self.advance_progress();
                if !self.complete_if_stopped(&mut work) {
                    if work.prefill_complete() {
                        // Its capture already waited through peer prefill.
                        // Resume it before another bounded chunk, as for any
                        // due decoder.
                        work.decode_due = true;
                        decoding.push_back(work);
                    } else {
                        prefilling.push_back(work);
                    }
                }
            }
            self.admit(&mut prefilling, &mut decoding, &mut capturing, &mut waiting);
            if self.device_lost() {
                break;
            }
            if prefilling.is_empty() && decoding.is_empty() {
                let capacity = self.pool.capacity();
                let mut queue = self.shared.queue();
                let wake = |queue: &Queue<R>| {
                    self.stop_requested()
                        || queue.stopping
                        || (queue.count != 0 && capturing.len() + waiting.len() < capacity)
                };
                if capturing.is_empty() && waiting.is_empty() {
                    // Only the idle wait has a timer: an arrival interrupts it
                    // at once, and active work never polls the device.
                    while !wake(&queue) {
                        self.shared.loop_working.store(false, Ordering::Relaxed);
                        let (guard, timeout) = self
                            .shared
                            .queue_condition
                            .wait_timeout(queue, self.device_probe_interval)
                            .unwrap_or_else(|e| e.into_inner());
                        queue = guard;
                        self.advance_progress();
                        self.shared.loop_working.store(true, Ordering::Relaxed);
                        if !timeout.timed_out() || wake(&queue) {
                            continue;
                        }
                        drop(queue);
                        if self.pool.runner().poll_device() == DeviceStatus::Lost {
                            self.mark_device_lost("idle device probe failed");
                        }
                        queue = self.shared.queue();
                        if self.device_lost() {
                            break;
                        }
                    }
                } else if !wake(&queue) {
                    let _ = self
                        .shared
                        .queue_condition
                        .wait_timeout(queue, Duration::from_millis(1));
                }
                continue;
            }

            self.advance_progress();
            let due_decoder = decoding.iter().any(|w| w.decode_due);
            let resident = prefilling.len() + decoding.len();
            let preparing_multi_token_batch = self.multi_token_decode
                && resident > 1
                && !prefilling.is_empty()
                && self
                    .pool
                    .select_decode_plan(resident)
                    .is_ok_and(|p| p.kind == TextExecutionPlanKind::Batched);
            if preparing_multi_token_batch {
                for work in prefilling.iter_mut().chain(decoding.iter_mut()) {
                    if let Ok(request) = work.runner() {
                        // A failure here surfaces on the request's next step.
                        let _ = request.prepare_batch_execution();
                    }
                }
            }
            // Give a simultaneous new request whose rest fits one bounded
            // chunk that chunk to form the first batch; a longer one does
            // not hold ready decoders back. Once decoding starts, every due
            // decoder runs before more prefill.
            let short_prefill_first = prefilling
                .front()
                .is_some_and(|w| w.prefill_remaining() <= self.prefill_policy.decode_active_tokens);
            let assemble_initial_batch = preparing_multi_token_batch
                && self.consecutive_active_prefill_chunks == 0
                && short_prefill_first
                && decoding.iter().all(|w| w.result.tokens.is_empty());
            if !prefilling.is_empty()
                && !decoding.is_empty()
                && (!due_decoder || assemble_initial_batch)
            {
                if let Some(mut work) = prefilling.pop_front() {
                    let started = Instant::now();
                    self.step_prefill(&mut work, true, false);
                    self.yield_gpu(started);
                    Self::requeue_after_prefill(
                        work,
                        &mut prefilling,
                        &mut decoding,
                        &mut capturing,
                        false,
                    );
                }
                for work in decoding.iter_mut() {
                    work.decode_due = true;
                }
                continue;
            }

            if !decoding.is_empty() {
                let candidates = if due_decoder {
                    decoding.iter().filter(|w| w.decode_due).count()
                } else {
                    decoding.len()
                };
                let batch_size = match self.pool.select_decode_plan(candidates) {
                    Ok(plan) if plan.kind == TextExecutionPlanKind::Batched => {
                        candidates.min(plan.physical_width)
                    }
                    _ => 1,
                };
                let mut batch: Batch<R> = Vec::with_capacity(batch_size);
                for _ in 0..batch_size {
                    if let Some(mut work) = pop_decoder(&mut decoding, due_decoder) {
                        work.decode_due = false;
                        batch.push(work);
                    }
                }
                let started = Instant::now();
                self.step_decode_batch(&mut batch);
                self.yield_gpu(started);
                for work in batch {
                    if work.is_terminal() {
                        continue;
                    }
                    if work.snapshot_pending() {
                        capturing.push_back(work);
                    } else {
                        decoding.push_back(work);
                    }
                }
                continue;
            }

            if let Some(mut work) = prefilling.pop_front() {
                let short_peer_waiting = prefilling.iter().any(|peer| {
                    peer.prefill_remaining() <= self.prefill_policy.decode_active_tokens
                });
                let started = Instant::now();
                self.step_prefill(
                    &mut work,
                    false,
                    !capturing.is_empty() || short_peer_waiting,
                );
                self.yield_gpu(started);
                Self::requeue_after_prefill(
                    work,
                    &mut prefilling,
                    &mut decoding,
                    &mut capturing,
                    true,
                );
            }
        }
        decoding.extend(capturing.drain(..));
        self.cancel_remaining(&mut prefilling, &mut decoding, &mut waiting);
    }
}
