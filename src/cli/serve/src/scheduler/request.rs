//! A request's shared state and its handle: published output and phase,
//! waiting and streaming, cancellation.

use super::*;

impl OutputBudget {
    pub(super) fn try_reserve(&self, bytes: usize) -> bool {
        let reserved =
            self.buffered_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |current| {
                    (current <= self.limit && bytes <= self.limit - current)
                        .then_some(current + bytes)
                });
        match reserved {
            Ok(previous) => {
                self.max_buffered_bytes
                    .fetch_max(previous + bytes, Ordering::Relaxed);
                true
            }
            Err(_) => false,
        }
    }

    pub(super) fn release(&self, bytes: usize) {
        if bytes > 0 {
            self.buffered_bytes.fetch_sub(bytes, Ordering::AcqRel);
        }
    }
}

impl RequestShared {
    pub(super) fn output(&self) -> MutexGuard<'_, Output> {
        self.output.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn phase(&self) -> TextRequestPhase {
        TextRequestPhase::from_u8(self.phase.load(Ordering::Acquire))
    }

    pub(super) fn set_phase(&self, phase: TextRequestPhase) {
        self.phase.store(phase as u8, Ordering::Release);
    }

    pub(super) fn is_terminal(&self) -> bool {
        self.phase() == TextRequestPhase::Terminal
    }

    pub(super) fn cancellation_requested(&self) -> bool {
        self.cancellation_requested.load(Ordering::Acquire)
            || self.external_cancellation.as_ref().is_some_and(|c| c())
    }

    pub(super) fn deadline_exceeded(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    pub(super) fn release_buffered(&self, output: &mut Output) {
        output.pieces.clear();
        self.output_budget.release(output.buffered_bytes);
        output.buffered_bytes = 0;
    }

    /// False when the per-request or server-wide output bound is reached.
    pub(super) fn publish_piece(&self, piece: Vec<u8>) -> bool {
        if !self.publish_token_pieces {
            return true;
        }
        let bytes = queued_piece_cost(&piece);
        let refused = {
            let mut output = self.output();
            if output.buffered_bytes > self.max_buffered_output_bytes
                || bytes > self.max_buffered_output_bytes - output.buffered_bytes
            {
                Some(("request_buffer", output.buffered_bytes))
            } else if !self.output_budget.try_reserve(bytes) {
                Some(("total_buffer", output.buffered_bytes))
            } else {
                output.buffered_bytes += bytes;
                output.max_buffered_bytes = output.max_buffered_bytes.max(output.buffered_bytes);
                output.pieces.push_back(piece);
                None
            }
        };
        if let Some((reason, buffered)) = refused {
            self.log_backpressure(reason, buffered, bytes);
            return false;
        }
        self.output_condition.notify_one();
        true
    }

    /// Names the budget that refused a piece, the question a stalled stream
    /// raises; once per request, since a stalled consumer refuses every
    /// token. Logged after the output lock is released.
    pub(super) fn log_backpressure(&self, reason: &str, buffered_bytes: usize, piece_bytes: usize) {
        if !logging::enabled(LogLevel::Debug)
            || self.backpressure_logged.swap(true, Ordering::Relaxed)
        {
            return;
        }
        logging::debug(
            "scheduler",
            &format!(
                "event=backpressure request={} reason={reason} buffered_bytes={buffered_bytes} \
                 piece_bytes={piece_bytes} request_limit_bytes={} total_bytes={} \
                 total_limit_bytes={}",
                self.id,
                self.max_buffered_output_bytes,
                self.output_budget.buffered_bytes.load(Ordering::Relaxed),
                self.output_budget.limit
            ),
        );
    }
}

/// The waiter's handle: consumes output pieces on its thread and waits for
/// the request to become terminal. Dropping it cancels the request.
pub struct SchedulerRequest {
    pub(super) shared: Arc<RequestShared>,
    pub(super) waited: bool,
}

impl SchedulerRequest {
    pub fn id(&self) -> u64 {
        self.shared.id
    }

    pub fn phase(&self) -> TextRequestPhase {
        self.shared.phase()
    }

    pub fn cancel(&self) {
        if !self.shared.is_terminal() {
            self.shared
                .cancellation_requested
                .store(true, Ordering::Release);
        }
    }

    /// Delivers each output piece to `on_token` (returning false cancels)
    /// and returns the result, or the scheduler's failure.
    pub fn wait(&mut self, on_token: Option<TokenCallback<'_>>) -> Result<GenerationResult> {
        self.wait_streaming(StreamCallbacks {
            on_token,
            ..StreamCallbacks::default()
        })
    }

    /// `wait` that also reports a stream's start, once at admission or
    /// after `stream_start_delay` in the queue, and then the latest prompt
    /// progress of a `return_progress` request, both before any piece.
    pub fn wait_streaming(&mut self, callbacks: StreamCallbacks<'_>) -> Result<GenerationResult> {
        let StreamCallbacks {
            mut on_token,
            mut on_progress,
            mut on_start,
        } = callbacks;
        if self.waited {
            return Err(Error::logic("text scheduler request was already consumed"));
        }
        self.waited = true;
        let mut deliver = true;
        let mut consumer_cancelled = false;
        // Only streams with a start consumer use the queue deadline.
        let mut started = on_start.is_none() || !self.shared.publish_token_pieces;
        loop {
            let (start, progress, piece, terminal) = {
                let mut output = self.shared.output();
                let ready = |output: &Output| {
                    output.terminal
                        || !output.pieces.is_empty()
                        || output.progress.is_some()
                        || (!started && output.admitted)
                };
                let mut start = false;
                if started {
                    while !ready(&output) {
                        output = self
                            .shared
                            .output_condition
                            .wait(output)
                            .unwrap_or_else(|e| e.into_inner());
                    }
                } else {
                    // A request still queued at the deadline starts as well.
                    let deadline = self.shared.stream_start_deadline;
                    while !ready(&output) {
                        let Some(remaining) = deadline.checked_duration_since(Instant::now())
                        else {
                            break;
                        };
                        output = self
                            .shared
                            .output_condition
                            .wait_timeout(output, remaining)
                            .map_or_else(|e| e.into_inner().0, |(output, _)| output);
                    }
                    // Admission precedes progress and output, even when the
                    // request also failed before this consumer woke.
                    start = !ready(&output) || output.admitted;
                }
                if start {
                    (true, None, None, None)
                } else if let Some(progress) = output.progress.take() {
                    (false, Some(progress), None, None)
                } else if let Some(piece) = output.pieces.pop_front() {
                    let cost = queued_piece_cost(&piece);
                    output.buffered_bytes -= cost;
                    self.shared.output_budget.release(cost);
                    (false, None, Some(piece), None)
                } else {
                    (
                        false,
                        None,
                        None,
                        Some((output.result.take(), output.failure.take())),
                    )
                }
            };
            if start {
                started = true;
                if let (true, Some(callback)) = (deliver, on_start.as_mut())
                    && !callback()
                {
                    consumer_cancelled = true;
                    deliver = false;
                    self.cancel();
                }
                continue;
            }
            if let (Some(progress), true, Some(callback)) =
                (&progress, deliver, on_progress.as_mut())
                && !callback(progress)
            {
                consumer_cancelled = true;
                deliver = false;
                self.cancel();
            }
            // A token without text still gets its event.
            if let (Some(piece), true, Some(callback)) = (&piece, deliver, on_token.as_mut())
                && !callback(piece)
            {
                consumer_cancelled = true;
                deliver = false;
                self.cancel();
            }
            if let Some((result, failure)) = terminal {
                if let Some(failure) = failure {
                    return Err(failure);
                }
                let mut result = result.unwrap_or_default();
                if consumer_cancelled {
                    result.cancelled = true;
                    result.finish_reason = FinishReason::Cancelled;
                }
                return Ok(result);
            }
        }
    }
}

impl Drop for SchedulerRequest {
    fn drop(&mut self) {
        self.cancel();
    }
}
