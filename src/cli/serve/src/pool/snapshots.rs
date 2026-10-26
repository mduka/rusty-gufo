//! A request's prompt and boundary snapshots: capture on a worker thread
//! while the request is frozen, and their publication.

use super::*;

impl<R: TextModelRunner> RequestInner<R> {
    /// The input identity of the first `token_count` tokens.
    pub(super) fn input_identity(&self, token_count: usize) -> &[u8] {
        self.context
            .as_ref()
            .map_or(&[], |c| c.cache_identity_for(token_count))
    }

    pub(super) fn state_mut(&mut self) -> Result<&mut R::State> {
        self.lease.state_mut()
    }

    pub(super) fn capture_prompt_snapshot(&mut self, wait: bool) {
        if self.boundary.is_some() {
            if !wait && self.capture.as_ref().is_some_and(Capture::pending) {
                return;
            }
            self.finish_boundary_snapshot();
        }
        if !self.prompt_snapshot_attempted {
            self.start_prompt_snapshot(wait);
        }
        if let Some(capture) = &self.capture {
            if !wait && capture.pending() {
                return;
            }
            self.finish_prompt_snapshot();
        }
        if self.prompt_snapshot_attempted
            && self.prefill_offset == self.snapshot_tokens.len()
            && self.snapshot_tokens.len() < self.prompt.len()
        {
            self.advance_prompt_target();
        }
    }

    /// Whether the next prefill step of at most `max_input_tokens` captures
    /// the prompt target at `snapshot_position` in the same pass; reserves
    /// its bytes when it does. Disk persistence keeps the split pass.
    pub(super) fn in_pass_checkpoint(
        &mut self,
        max_input_tokens: usize,
        snapshot_position: usize,
    ) -> bool {
        if self.disk_store.is_some()
            || self.prompt_snapshot_attempted
            || self.prefill_offset >= snapshot_position
            || snapshot_position >= self.prompt.len()
            || !self.capabilities.in_pass_checkpoint
        {
            return false;
        }
        let bytes = self.lease.state().and_then(|state| {
            self.runner.prefill_checkpoint_bytes(
                state,
                &self.prompt,
                self.prefill_offset,
                max_input_tokens,
                snapshot_position,
            )
        });
        let Ok(Some(bytes)) = bytes else {
            return false;
        };
        let reserved = self.lease.try_reserve_snapshot(
            bytes,
            snapshot_position,
            self.retain_fallback && self.fallback_position.is_none(),
            SnapshotPurpose::Continuation,
            &self.snapshot_tokens,
        );
        if !matches!(reserved, Ok(true)) {
            return false;
        }
        self.snapshot_bytes = bytes;
        self.retain_snapshot = true;
        self.prompt_snapshot_attempted = true;
        true
    }

    /// Publishes the captured prompt target and moves on to the next one
    /// (`AdvancePromptTarget`).
    pub(super) fn advance_prompt_target(&mut self) {
        // Keep both the branching fallback and the complete prompt. The
        // latter is captured after first-token publication, as for ordinary
        // requests; admission may refuse it without discarding the fallback.
        if let Some(snapshot) = self.prompt_snapshot.take()
            && self.retain_snapshot
        {
            match self.lease.publish_snapshot(
                self.snapshot_tokens.clone(),
                snapshot,
                self.fallback_position.is_some(),
            ) {
                Ok(bytes) => self.snapshot_metrics.snapshot_bytes += bytes,
                Err(_) => self.lease.skip_snapshot(
                    SnapshotEventReason::CaptureFailure,
                    self.snapshot_bytes,
                    self.snapshot_tokens.len(),
                ),
            }
        }
        self.retain_snapshot = false;
        self.fallback_position = Some(self.snapshot_tokens.len());
        // Take this turn's stable boundary next, before the complete prompt:
        // the boundary is a prefix of whatever the client sends next, while
        // the complete prompt ends in assistant framing the client may
        // rewrite.
        let stable = std::mem::take(&mut self.stable_prefix_position);
        self.snapshot_tokens = if stable > self.snapshot_tokens.len() && stable < self.prompt.len()
        {
            self.prompt[..stable].to_vec()
        } else {
            self.prompt.clone()
        };
        self.prompt_snapshot_attempted = false;
    }

    /// Whether [`start_prompt_snapshot`](Self::start_prompt_snapshot) will
    /// capture once the prefill reaches the prompt target: its conditions
    /// before sizing, and room to retain or persist the copy.
    pub(super) fn expects_prompt_snapshot(&self) -> bool {
        let capabilities = self.capabilities;
        let tokens = self.snapshot_tokens.len();
        !self.prompt_snapshot_attempted
            && capabilities.prefix_reuse
            && capabilities.snapshot
            && capabilities.fork
            && (self.lease.snapshot_capacity_bytes() != 0 || self.disk_store.is_some())
            && !(self.lease.cache_hit()
                && !self.lease.restored_from_disk()
                && self.lease.cached_tokens() == tokens
                && self.lease.has_snapshot_for(&self.snapshot_tokens))
    }

    pub(super) fn start_prompt_snapshot(&mut self, wait: bool) {
        if self.prompt_snapshot_attempted || self.prefill_offset != self.snapshot_tokens.len() {
            return;
        }
        self.prompt_snapshot_attempted = true;
        let capabilities = self.capabilities;
        if !capabilities.prefix_reuse || !capabilities.snapshot || !capabilities.fork {
            return;
        }
        let tokens = self.snapshot_tokens.len();
        if self.lease.cache_hit()
            && !self.lease.restored_from_disk()
            && self.lease.cached_tokens() == tokens
            && self.lease.has_snapshot_for(&self.snapshot_tokens)
        {
            return;
        }
        self.snapshot_bytes = 0;
        let sized = self.lease.state().and_then(|state| {
            if self.runner.checkpoint_position(state)? != tokens {
                return Err(Error::logic("checkpoint position mismatch"));
            }
            self.runner.snapshot_payload_bytes(state)
        });
        match sized {
            Ok(bytes) => self.snapshot_bytes = bytes,
            Err(_) => {
                self.lease
                    .skip_snapshot(SnapshotEventReason::CaptureFailure, 0, tokens);
                return;
            }
        }
        // The complete prompt behind a stable boundary is a retry copy; the
        // fallback and the turn's boundary are continuations.
        let complete = tokens == self.prompt.len();
        let purpose = if self.retain_fallback && complete {
            SnapshotPurpose::Retry
        } else {
            SnapshotPurpose::Continuation
        };
        self.retain_snapshot = match self.lease.try_reserve_snapshot(
            self.snapshot_bytes,
            tokens,
            self.retain_fallback && (self.fallback_position.is_none() || complete),
            purpose,
            &self.snapshot_tokens,
        ) {
            Ok(admitted) => admitted,
            Err(_) => return,
        };
        if !self.retain_snapshot {
            let Some(store) = self.disk_store.clone() else {
                return;
            };
            // Copying device state only for disk is costly on the request
            // path: skip it when the store would discard the checkpoint.
            if store.within_checkpoint_step(
                &*self.runner,
                &self.snapshot_tokens,
                self.input_identity(tokens),
            ) {
                return;
            }
            match store.reserve_capture(
                &*self.runner,
                tokens,
                self.snapshot_bytes,
                self.input_identity(tokens),
            ) {
                Ok(Some(reservation)) => self.disk_capture = Some(reservation),
                _ => return,
            }
        }
        self.snapshot_start = Instant::now();
        let persist = self.disk_store.is_some();
        if wait {
            // A warm prefill must await its fallback before mutating the
            // recurrent state: copy on this thread rather than start a worker
            // only to join it. Post-token captures stay asynchronous.
            let snapshot = self.lease.state().and_then(|state| {
                catch_unwind(AssertUnwindSafe(|| capture(&*self.runner, state, persist)))
                    .unwrap_or_else(|_| Err(Error::runtime("snapshot capture panicked")))
            });
            self.capture = Some(Capture::Taken(snapshot));
            return;
        }
        let Some(state) = self.lease.take_state() else {
            self.disk_capture = None;
            if self.retain_snapshot {
                self.lease.skip_snapshot(
                    SnapshotEventReason::CaptureFailure,
                    self.snapshot_bytes,
                    tokens,
                );
            }
            return;
        };
        let runner = self.runner.clone();
        let spawned = std::thread::Builder::new()
            .name("gufo-snapshot".into())
            .spawn(move || {
                let snapshot =
                    catch_unwind(AssertUnwindSafe(|| capture(&*runner, &state, persist)))
                        .unwrap_or_else(|_| Err(Error::runtime("snapshot capture panicked")));
                (state, snapshot)
            });
        match spawned {
            Ok(handle) => self.capture = Some(Capture::Running(handle)),
            Err(_) => {
                // The closure (and with it the state) is gone only if the
                // thread could not be created; the lease is then unusable.
                self.disk_capture = None;
                if self.retain_snapshot {
                    self.lease.skip_snapshot(
                        SnapshotEventReason::CaptureFailure,
                        self.snapshot_bytes,
                        tokens,
                    );
                }
            }
        }
    }

    /// Joins a running capture and returns the state to the lease.
    pub(super) fn join_capture(&mut self) -> Option<Result<R::Snapshot>> {
        match self.capture.take()? {
            Capture::Taken(snapshot) => Some(snapshot),
            Capture::Running(handle) => match handle.join() {
                Ok((state, snapshot)) => {
                    self.lease.put_state(state);
                    Some(snapshot)
                }
                Err(_) => Some(Err(Error::runtime("snapshot capture thread failed"))),
            },
        }
    }

    pub(super) fn finish_prompt_snapshot(&mut self) {
        let snapshot = self.join_capture().and_then(|r| r.ok()).map(Arc::new);
        self.snapshot_metrics.snapshot_ms += elapsed_ms(self.snapshot_start);
        let tokens = self.snapshot_tokens.len();
        let Some(snapshot) = snapshot else {
            self.prompt_snapshot = None;
            self.disk_capture = None;
            if self.retain_snapshot {
                self.lease.skip_snapshot(
                    SnapshotEventReason::CaptureFailure,
                    self.snapshot_bytes,
                    tokens,
                );
            }
            return;
        };
        let payload = TextRunnerSnapshot::payload_bytes(&*snapshot);
        if self.retain_snapshot && payload != self.snapshot_bytes {
            self.lease
                .skip_snapshot(SnapshotEventReason::ReservationMismatch, payload, tokens);
            self.retain_snapshot = false;
        }
        // Charge disk-only snapshots now: holding them until commit would let
        // concurrent requests pass the same queue-capacity preflight.
        let already_persisted =
            self.lease.restored_from_disk() && self.lease.cached_tokens() == tokens;
        if let Some(store) = self.disk_store.clone()
            && !already_persisted
        {
            let started = Instant::now();
            // A failure keeps the bytes of earlier queued checkpoints.
            if let Ok(queued) = store.save_async(
                self.runner.clone(),
                self.snapshot_tokens.clone(),
                snapshot.clone(),
                self.input_identity(self.snapshot_tokens.len()).to_vec(),
                self.disk_capture.take(),
                false,
            ) {
                self.snapshot_metrics.disk_queued_bytes += queued;
            }
            self.snapshot_metrics.disk_enqueue_ms += elapsed_ms(started);
        }
        self.disk_capture = None;
        self.prompt_snapshot = self.retain_snapshot.then_some(snapshot);
    }

    /// Captures the state at a boundary prefill reached: an intermediate RAM
    /// checkpoint (`history`) and/or a learned shared prefix for the disk
    /// cache (`shared`), from one copy when both coincide. The history copy
    /// must not displace the frontier this request branched from, including
    /// its stable image or reasoning fallback. With more than one execution
    /// state the copy runs on a worker, so it does not stall other requests
    /// (the scheduler parks this lease meanwhile); a single state has no
    /// peer work to overlap with it.
    pub(super) fn capture_boundary_snapshot(
        &mut self,
        position: usize,
        history: bool,
        shared: bool,
        purpose: SnapshotPurpose,
    ) {
        let capabilities = self.capabilities;
        if !capabilities.prefix_reuse
            || !capabilities.snapshot
            || !capabilities.fork
            || position == 0
            || position >= self.prompt.len()
        {
            return;
        }
        let mut boundary = BoundaryCapture {
            position,
            reserved: 0,
            persistence: None,
            history,
            shared,
            started: Instant::now(),
        };
        let identity = self.input_identity(position).to_vec();
        let prepared = (|| -> Result<bool> {
            let state = self.lease.state()?;
            if self.runner.checkpoint_position(state)? != position {
                return Ok(false);
            }
            let prefix = &self.prompt[..position];
            let bytes = self.runner.snapshot_payload_bytes(state)?;
            if history
                && !self.lease.has_snapshot_for(prefix)
                // A copy waiting requests depend on competes like a
                // continuation boundary; the rest give way under pressure.
                && self
                    .lease
                    .try_reserve_snapshot(bytes, position, true, purpose, &[])?
            {
                boundary.reserved = bytes;
            }
            // Another request may have stored this prefix meanwhile.
            if shared
                && let Some(store) = &self.disk_store
                && !store.touch(&*self.runner, prefix, &identity)?
            {
                boundary.persistence =
                    store.reserve_capture(&*self.runner, position, bytes, &identity)?;
            }
            Ok(boundary.reserved != 0 || boundary.persistence.is_some())
        })();
        match prepared {
            Ok(true) => {}
            Ok(false) => return,
            Err(_) => {
                self.boundary = Some(boundary);
                self.capture = Some(Capture::Taken(Err(Error::runtime(
                    "boundary capture failed",
                ))));
                self.finish_boundary_snapshot();
                return;
            }
        }
        let persist = self.disk_store.is_some();
        if self.overlap_snapshots
            && let Some(state) = self.lease.take_state()
        {
            let runner = self.runner.clone();
            let spawned = std::thread::Builder::new()
                .name("gufo-snapshot".into())
                .spawn(move || {
                    let snapshot =
                        catch_unwind(AssertUnwindSafe(|| capture(&*runner, &state, persist)))
                            .unwrap_or_else(|_| Err(Error::runtime("snapshot capture panicked")));
                    (state, snapshot)
                });
            self.boundary = Some(boundary);
            match spawned {
                Ok(handle) => self.capture = Some(Capture::Running(handle)),
                // The state went with the closure: the lease is unusable.
                Err(_) => {
                    self.capture = Some(Capture::Taken(Err(Error::runtime(
                        "snapshot capture thread failed",
                    ))));
                    self.finish_boundary_snapshot();
                }
            }
            return;
        }
        let snapshot = self.lease.state().and_then(|state| {
            catch_unwind(AssertUnwindSafe(|| capture(&*self.runner, state, persist)))
                .unwrap_or_else(|_| Err(Error::runtime("snapshot capture panicked")))
        });
        self.boundary = Some(boundary);
        self.capture = Some(Capture::Taken(snapshot));
        self.finish_boundary_snapshot();
    }

    /// Joins a boundary capture and publishes its copy: the RAM checkpoint
    /// and the disk shared prefix share the immutable snapshot.
    pub(super) fn finish_boundary_snapshot(&mut self) {
        let Some(boundary) = self.boundary.take() else {
            return;
        };
        let snapshot = self.join_capture().and_then(|r| r.ok()).map(Arc::new);
        let position = boundary.position;
        let mut failed = snapshot.is_none();
        if let Some(snapshot) = &snapshot {
            if boundary.reserved != 0 {
                match self.lease.publish_snapshot(
                    self.prompt[..position].to_vec(),
                    snapshot.clone(),
                    true,
                ) {
                    Ok(bytes) => self.snapshot_metrics.snapshot_bytes += bytes,
                    Err(_) => self.lease.skip_snapshot(
                        SnapshotEventReason::CaptureFailure,
                        boundary.reserved,
                        position,
                    ),
                }
            }
            if let Some(persistence) = boundary.persistence
                && let Some(store) = self.disk_store.clone()
            {
                match store.save_async(
                    self.runner.clone(),
                    self.prompt[..position].to_vec(),
                    snapshot.clone(),
                    self.input_identity(position).to_vec(),
                    Some(persistence),
                    true,
                ) {
                    Ok(0) => {}
                    Ok(saved) => {
                        self.snapshot_metrics.shared_prefix_snapshots += 1;
                        self.snapshot_metrics.shared_prefix_bytes += saved;
                    }
                    Err(_) => failed = true,
                }
            }
        } else if boundary.reserved != 0 {
            self.lease.skip_snapshot(
                SnapshotEventReason::CaptureFailure,
                boundary.reserved,
                position,
            );
        }
        let elapsed = elapsed_ms(boundary.started);
        if boundary.history {
            self.snapshot_metrics.snapshot_ms += elapsed;
        }
        if boundary.shared {
            self.snapshot_metrics.shared_prefix_ms += elapsed;
            // Best effort: the request itself is unaffected.
            self.snapshot_metrics.shared_prefix_failures += usize::from(failed);
        }
    }

    /// Finishes whichever capture is outstanding.
    pub(super) fn finish_capture(&mut self) {
        if self.boundary.is_some() {
            self.finish_boundary_snapshot();
        } else {
            self.finish_prompt_snapshot();
        }
    }

    pub(super) fn clear_cancellation(&mut self) {
        if let Ok(state) = self.lease.state_mut() {
            state.set_cancellation_check(None);
        }
    }
}
