//! A leased request: checkpoint sharing, prefill, decode, commit and
//! cancellation.

use super::*;

impl<R: TextModelRunner> Drop for PoolRequest<R> {
    fn drop(&mut self) {
        // A capture worker owns the state: join it before the lease goes.
        if let Some(inner) = self.inner.as_mut() {
            let _ = inner.join_capture();
            inner.clear_cancellation();
        }
    }
}

impl<R: TextModelRunner> PoolRequest<R> {
    pub(super) fn empty() -> Self {
        Self { inner: None }
    }

    pub fn is_active(&self) -> bool {
        self.inner.is_some()
    }

    pub(super) fn inner_mut(&mut self) -> Result<&mut RequestInner<R>> {
        self.inner.as_mut().ok_or_else(empty_request)
    }

    pub fn cache_hit(&self) -> bool {
        self.inner.as_ref().is_some_and(|i| i.lease.cache_hit())
    }

    pub fn cache_lookup(&self) -> ContinuationLookup {
        self.inner
            .as_ref()
            .map_or_else(ContinuationLookup::default, |i| i.lease.lookup())
    }

    pub fn cached_prompt_tokens(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.lease.cached_tokens())
    }

    pub fn cache_restore_bytes(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(0, |i| i.lease.restored_snapshot_bytes())
    }

    pub fn cache_restore_ms(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, |i| i.lease.restore_ms())
    }

    pub fn cache_disk_hit(&self) -> bool {
        self.inner
            .as_ref()
            .is_some_and(|i| i.lease.restored_from_disk())
    }

    pub fn prompt_tokens(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.prompt.len())
    }

    /// The prompt this request prefills.
    pub fn prompt(&self) -> &[Token] {
        self.inner.as_ref().map_or(&[], |inner| &inner.prompt)
    }

    /// Prompt tokens already in the model state, restored or prefilled.
    pub fn prefill_position(&self) -> usize {
        self.inner.as_ref().map_or(0, |inner| inner.prefill_offset)
    }

    /// The input identity of the prompt's first `token_count` tokens.
    pub fn input_identity(&self, token_count: usize) -> &[u8] {
        self.inner
            .as_ref()
            .map_or(&[], |inner| inner.input_identity(token_count))
    }

    /// Lets another request reuse this prompt's first `common_tokens`: the
    /// position, at most that, where this request publishes a RAM
    /// checkpoint before prefilling past it, reusing a planned checkpoint
    /// within [`SHARED_CHECKPOINT_SLACK`] tokens. Zero when the runner
    /// cannot snapshot or this request is past that position. Other
    /// requests depend on it, waiting now or arriving later, so it is
    /// retained with continuation priority rather than as an optional copy.
    pub fn share_checkpoint(&mut self, common_tokens: usize) -> usize {
        let Some(inner) = self.inner.as_mut() else {
            return 0;
        };
        let capabilities = inner.capabilities;
        if inner.decode_ready
            || inner.stopped
            || !capabilities.incremental_prefill
            || !capabilities.prefix_reuse
            || !capabilities.snapshot
            || !capabilities.fork
        {
            return 0;
        }
        // The complete prompt is retained only after decoding starts: share
        // the last prefill position instead, so peers start sooner.
        let target = common_tokens.min(inner.prompt.len() - 1);
        let offset = inner.prefill_offset;
        if target <= offset {
            return 0;
        }
        // Prefill already publishes the fallback, the turn's stable boundary
        // and the intermediate checkpoints before advancing past them.
        let mut planned = 0;
        let mut consider = |position: usize| {
            if position > offset && position <= target && position > planned {
                planned = position;
            }
        };
        if inner.snapshot_tokens.len() < inner.prompt.len() {
            consider(inner.snapshot_tokens.len());
        }
        consider(inner.stable_prefix_position);
        for &position in &inner.checkpoints {
            consider(position);
        }
        let position = if planned != 0 && target - planned <= SHARED_CHECKPOINT_SLACK {
            planned
        } else {
            target
        };
        if position == target
            && let Err(index) = inner.checkpoints.binary_search(&target)
        {
            inner.checkpoints.insert(index, target);
        }
        // Prompt snapshots are retained as continuation boundaries already.
        if inner.checkpoints.binary_search(&position).is_ok() {
            inner.shared_checkpoints.push(position);
        }
        position
    }

    pub fn prefill_complete(&self) -> bool {
        self.inner.as_ref().is_some_and(|i| i.decode_ready)
    }

    /// Captures the prompt snapshot now if it is due, and waits for it.
    pub fn capture_prompt_snapshot(&mut self) {
        if let Some(inner) = self.inner.as_mut() {
            inner.capture_prompt_snapshot(true);
        }
    }

    /// Starts a due capture and polls it without blocking other requests;
    /// true once no capture is outstanding.
    pub fn prepare_prompt_snapshot(&mut self) -> bool {
        match self.inner.as_mut() {
            Some(inner) => {
                inner.capture_prompt_snapshot(false);
                inner.capture.is_none()
            }
            None => true,
        }
    }

    pub fn snapshot_pending(&self) -> bool {
        self.inner
            .as_ref()
            .and_then(|i| i.capture.as_ref())
            .is_some_and(Capture::pending)
    }

    pub fn preview_first_token(&mut self) -> Result<Option<TextDecodeSelection>> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };
        if !inner.decode_ready
            || !inner.generated.is_empty()
            || inner.pending_selection.is_some()
            || inner.stopped
        {
            return Ok(None);
        }
        // The scheduler previews before any capture starts; join one anyway
        // rather than race it for the state.
        if inner.capture.is_some() {
            inner.finish_capture();
        }
        let mut sampler = inner.sampler.clone();
        let runner = inner.runner.clone();
        runner.preview_first_token(inner.state_mut()?, &mut sampler)
    }

    pub fn prepare_batch_execution(&mut self) -> Result<()> {
        let inner = self.inner_mut()?;
        let runner = inner.runner.clone();
        runner.prepare_batch_execution(inner.state_mut()?)
    }

    pub fn prefill(&mut self, max_input_tokens: usize) -> Result<TextPrefillStep> {
        let inner = self.inner_mut()?;
        if inner.pending_selection.is_some() {
            return Err(Error::logic(
                "text runner cannot prefill with a pending decode token",
            ));
        }
        if inner.stopped {
            return Err(Error::logic("text runner request already stopped"));
        }
        if inner.decode_ready {
            return Ok(TextPrefillStep {
                consumed_tokens: 0,
                decode_ready: true,
            });
        }
        if max_input_tokens == 0 {
            return Err(Error::invalid(
                "text runner prefill budget must be at least one token",
            ));
        }
        // Stop exactly on the next shared-prefix boundary so it can be
        // persisted; join any capture before mutating its state.
        inner.capture_prompt_snapshot(true);
        let mut max_input_tokens = max_input_tokens;
        let snapshot_position = inner.snapshot_tokens.len();
        let offset = inner.prefill_offset;
        inner.boundaries.retain(|&b| b > offset);
        if let Some(&boundary) = inner.boundaries.first() {
            max_input_tokens = max_input_tokens.min(boundary - inner.prefill_offset);
        }
        inner.checkpoints.retain(|&c| c > offset);
        if let Some(&checkpoint) = inner.checkpoints.first() {
            max_input_tokens = max_input_tokens.min(checkpoint - inner.prefill_offset);
        }
        // A runner that captures the prompt target inside the pass crossing
        // it need not split the pass there.
        let in_pass = inner.in_pass_checkpoint(max_input_tokens, snapshot_position);
        if !in_pass && inner.prefill_offset < snapshot_position {
            max_input_tokens = max_input_tokens.min(snapshot_position - inner.prefill_offset);
        }

        let remaining = inner.prompt.len() - inner.prefill_offset;
        inner.state_reusable = false;
        // Complete the model frontier at the cache boundary (the prompt
        // snapshot or the next learned shared prefix), including logits and
        // draft catch-up, so its snapshot restores independently.
        let mut model_len = if !in_pass && inner.prefill_offset < snapshot_position {
            snapshot_position
        } else {
            inner.prompt.len()
        };
        if let Some(&boundary) = inner.boundaries.first() {
            model_len = model_len.min(boundary);
        }
        if let Some(&checkpoint) = inner.checkpoints.first() {
            model_len = model_len.min(checkpoint);
        }
        // A step that ends at the prompt target lets the runner prepare the
        // capture's storage while it runs.
        let prepare = !in_pass
            && offset + max_input_tokens == snapshot_position
            && inner.expects_prompt_snapshot();
        let runner = inner.runner.clone();
        let RequestInner { lease, prompt, .. } = &mut *inner;
        let mut captured = None;
        let mut step = if in_pass {
            let through = runner.prefill_through(
                lease.state_mut()?,
                &prompt[..model_len],
                offset,
                max_input_tokens,
                snapshot_position,
            )?;
            captured = Some((through.checkpoint, through.checkpoint_ms));
            through.step
        } else {
            let state = lease.state_mut()?;
            if prepare {
                // Best effort: a capture without it allocates as before.
                let _ = runner.prepare_snapshot(state);
            }
            runner.prefill(state, &prompt[..model_len], offset, max_input_tokens)?
        };
        let maximum_consumed = remaining.min(max_input_tokens);
        if step.consumed_tokens == 0 || step.consumed_tokens > maximum_consumed {
            return Err(Error::runtime(
                "text runner returned an invalid prefill token count",
            ));
        }
        inner.prefill_offset += step.consumed_tokens;
        let reached_frontier = inner.prefill_offset == inner.prompt.len();
        if step.decode_ready != (inner.prefill_offset == model_len) {
            return Err(Error::runtime(
                "text runner returned an inconsistent prefill boundary",
            ));
        }
        inner.decode_ready = reached_frontier;
        step.decode_ready = reached_frontier;
        inner.state_reusable = true;
        if let Some((checkpoint, checkpoint_ms)) = captured {
            inner.snapshot_metrics.snapshot_ms += checkpoint_ms;
            let payload = TextRunnerSnapshot::payload_bytes(&checkpoint);
            if inner.prefill_offset <= snapshot_position || payload != inner.snapshot_bytes {
                inner.lease.skip_snapshot(
                    SnapshotEventReason::ReservationMismatch,
                    payload,
                    snapshot_position,
                );
                inner.retain_snapshot = false;
            } else {
                inner.prompt_snapshot = Some(Arc::new(checkpoint));
            }
            inner.advance_prompt_target();
        }
        if !reached_frontier && inner.prefill_offset == snapshot_position {
            inner.capture_prompt_snapshot(false);
        }
        let position = inner.prefill_offset;
        let history = inner.checkpoints.first() == Some(&position);
        let shared = inner.boundaries.first() == Some(&position);
        if history {
            inner.checkpoints.remove(0);
        }
        if shared {
            inner.boundaries.remove(0);
        }
        // A prompt capture at this position persists the boundary too: do
        // not start a second copy of the same state.
        if (history || shared) && position != snapshot_position {
            // A boundary shared with peers is a learned branch point.
            let purpose = if history && inner.shared_checkpoints.contains(&position) {
                SnapshotPurpose::BranchPoint
            } else {
                SnapshotPurpose::History
            };
            inner.capture_boundary_snapshot(position, history, shared, purpose);
        }
        Ok(step)
    }

    pub fn select_next(&mut self) -> Result<TextDecodeSelection> {
        let inner = self.inner_mut()?;
        if !inner.decode_ready {
            return Err(Error::logic(
                "text runner cannot decode before prefill completes",
            ));
        }
        if inner.pending_selection.is_some() {
            return Err(Error::logic(
                "text runner decode token must be advanced before selecting another",
            ));
        }
        if inner.stopped {
            return Err(Error::logic("text runner request already stopped"));
        }
        inner.state_reusable = false;
        let runner = inner.runner.clone();
        let RequestInner { lease, sampler, .. } = &mut *inner;
        let selection = runner.select_next(lease.state_mut()?, sampler)?;
        inner.state_reusable = true;
        if selection.stop {
            inner.stopped = true;
            return Ok(selection);
        }
        inner.sampler.accept(selection.token);
        inner.generated.push(selection.token);
        inner.pending_selection = Some(selection.clone());
        Ok(selection)
    }

    pub fn advance(&mut self) -> Result<()> {
        let inner = self.inner_mut()?;
        let Some(token) = inner.pending_selection.as_ref().map(|s| s.token) else {
            return Err(Error::logic(
                "text runner request has no selected token to advance",
            ));
        };
        inner.capture_prompt_snapshot(true);
        inner.state_reusable = false;
        let runner = inner.runner.clone();
        runner.advance(inner.state_mut()?, token)?;
        inner.pending_selection = None;
        inner.state_reusable = true;
        Ok(())
    }

    pub fn decode_step(&mut self, max_tokens: usize) -> Result<TextDecodeStep> {
        let inner = self.inner_mut()?;
        if !inner.decode_ready {
            return Err(Error::logic(
                "text runner cannot decode before prefill completes",
            ));
        }
        if inner.pending_selection.is_some() {
            return Err(Error::logic(
                "text runner cannot decode a step with a pending token",
            ));
        }
        if inner.stopped {
            return Err(Error::logic("text runner request already stopped"));
        }
        if max_tokens == 0 {
            return Err(Error::invalid(
                "text runner decode step budget must be at least one token",
            ));
        }
        inner.capture_prompt_snapshot(true);
        inner.state_reusable = false;
        let runner = inner.runner.clone();
        let RequestInner { lease, sampler, .. } = &mut *inner;
        let step = runner.decode_step(lease.state_mut()?, max_tokens, sampler)?;
        if step.selections.len() > max_tokens || (step.selections.is_empty() && !step.stop) {
            return Err(Error::runtime(
                "text runner returned an invalid decode step",
            ));
        }
        for selection in &step.selections {
            if selection.stop {
                return Err(Error::runtime(
                    "text runner decode step contains an embedded stop selection",
                ));
            }
            inner.sampler.accept(selection.token);
            inner.generated.push(selection.token);
        }
        inner.stopped = step.stop;
        inner.state_reusable = true;
        Ok(step)
    }

    /// Publishes a reusable continuation boundary: snapshot-capable runners
    /// keep the immutable prompt frontier and the final live state, others
    /// their reported checkpoint.
    pub fn commit(&mut self) -> Result<CommitMetrics> {
        let inner = self.inner_mut()?;
        if !inner.decode_ready {
            return Err(Error::logic(
                "text runner cannot commit before prefill completes",
            ));
        }
        if inner.pending_selection.is_some() {
            if inner.capabilities.final_token_advance_required {
                return Err(Error::logic(
                    "text runner cannot commit an unadvanced decode token",
                ));
            }
            inner.pending_selection = None;
        }
        inner.capture_prompt_snapshot(true);
        inner.clear_cancellation();
        let capabilities = inner.capabilities;
        if !capabilities.prefix_reuse {
            if let Some(inner) = self.inner.take() {
                inner.lease.invalidate();
            }
            return Ok(CommitMetrics::default());
        }
        let mut checkpoint = inner.prompt.clone();
        checkpoint.extend_from_slice(&inner.generated);
        let position = inner.runner.checkpoint_position(inner.lease.state()?)?;
        if position < inner.lease.cached_tokens() || position > checkpoint.len() {
            return Err(Error::runtime(
                "text runner checkpoint is outside executed token history",
            ));
        }
        checkpoint.truncate(position);
        let inner = self.inner.take().ok_or_else(empty_request)?;
        if capabilities.snapshot && capabilities.fork {
            let mut metrics = inner.snapshot_metrics;
            metrics.snapshot_bytes += inner.lease.commit_with(
                inner.snapshot_tokens,
                inner.prompt_snapshot,
                checkpoint,
            )?;
            return Ok(metrics);
        }
        Ok(CommitMetrics {
            snapshot_bytes: inner.lease.commit(checkpoint)?,
            ..CommitMetrics::default()
        })
    }

    /// Releases a cancelled request, keeping only completed model work and an
    /// already captured prompt snapshot. Neither advances nor samples.
    pub fn cancel(&mut self) -> CommitMetrics {
        let Some(inner) = self.inner.as_mut() else {
            return CommitMetrics::default();
        };
        // Join only an existing capture; cancellation must not start a new
        // copy or execute the pending token merely to retain the session.
        if inner.capture.is_some() {
            inner.finish_capture();
        }
        match self.cancel_inner() {
            Ok(metrics) => metrics,
            Err(_) => {
                self.invalidate();
                CommitMetrics::default()
            }
        }
    }

    pub(super) fn cancel_inner(&mut self) -> Result<CommitMetrics> {
        let inner = self.inner_mut()?;
        inner.clear_cancellation();
        let capabilities = inner.capabilities;
        if !capabilities.prefix_reuse {
            self.invalidate();
            return Ok(CommitMetrics::default());
        }
        let mut checkpoint = Vec::new();
        if inner.state_reusable
            && (inner.decode_ready
                || inner.prefill_offset == inner.snapshot_tokens.len()
                || inner.fallback_position == Some(inner.prefill_offset))
        {
            let runner = inner.runner.clone();
            if inner.decode_ready {
                runner.prepare_cancellation(inner.state_mut()?)?;
            }
            checkpoint.extend_from_slice(&inner.prompt[..inner.prefill_offset]);
            if inner.decode_ready {
                checkpoint.extend_from_slice(&inner.generated);
            }
            let position = runner.checkpoint_position(inner.lease.state()?)?;
            if position < inner.lease.cached_tokens() || position > checkpoint.len() {
                return Err(Error::runtime(
                    "cancelled checkpoint is outside completed work",
                ));
            }
            checkpoint.truncate(position);
        } else {
            TextRunnerState::invalidate(inner.state_mut()?);
        }
        let inner = self.inner.take().ok_or_else(empty_request)?;
        let mut metrics = inner.snapshot_metrics;
        metrics.snapshot_bytes += if capabilities.snapshot && capabilities.fork {
            inner
                .lease
                .commit_with(inner.snapshot_tokens, inner.prompt_snapshot, checkpoint)?
        } else {
            inner.lease.commit(checkpoint)?
        };
        Ok(metrics)
    }

    /// Discards the request's state.
    pub fn invalidate(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            let _ = inner.join_capture();
            inner.clear_cancellation();
            inner.lease.invalidate();
        }
    }
}
