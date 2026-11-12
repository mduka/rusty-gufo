//! Saving, restoring and touching entries, and the persistence queue.

use super::*;

impl Inner {
    /// Whether the longest stored prefix of `tokens` lies less than the
    /// minimum step before them (`WithinCheckpointStep`).
    pub(super) fn within_checkpoint_step<R: TextModelRunner + ?Sized>(
        &self,
        metadata: &Metadata,
        runner: &R,
        tokens: &[Token],
        input_identity: &[u8],
    ) -> bool {
        let step = self.options.min_checkpoint_step_tokens;
        if step == 0 || persistence_for_input(runner, input_identity).is_none() {
            return false;
        }
        metadata
            .find_longest_input(runner, tokens, InputIdentity::from(input_identity))
            .and_then(|id| metadata.entries.get(&id))
            .is_some_and(|base| {
                base.tokens.len() < tokens.len() && tokens.len() - base.tokens.len() < step
            })
    }

    pub(super) fn save<R: TextModelRunner>(
        &self,
        runner: &R,
        checkpoint_tokens: &[Token],
        snapshot: &R::Snapshot,
        input_identity: &[u8],
        shared_prefix: bool,
    ) -> Result<SaveResult> {
        use ContinuationDiskEventAction::Skipped;
        use ContinuationDiskEventReason as Reason;

        // Serialise writers, but keep existing entries readable during
        // payload serialisation, hashing and filesystem durability work.
        let mut counter = lock(&self.writer);
        let mut metadata = lock(&self.metadata);
        let token_count = checkpoint_tokens.len();
        let Some(persistence) = persistence_for_input(runner, input_identity) else {
            self.emit(Skipped, Reason::Unsupported, 0, 0, token_count, 0.0);
            return Ok(SaveResult::default());
        };
        if checkpoint_tokens.is_empty() {
            self.emit(Skipped, Reason::SerializationFailure, 0, 0, 0, 0.0);
            return Ok(SaveResult::default());
        }
        // The same prefix is offered again whenever a restored continuation
        // is committed or conversations share a system prompt; the stored
        // file already holds it, so only its recency is refreshed.
        let digest = self.hash_key(&persistence, checkpoint_tokens)?;
        if let Some(existing) = metadata.find_exact(&digest, &persistence, checkpoint_tokens) {
            self.touch_entry(&mut metadata, existing);
            if let Some(entry) = metadata.entries.get(&existing) {
                self.emit(
                    Skipped,
                    Reason::ExactReplacement,
                    entry.file_bytes,
                    entry.payload_bytes,
                    token_count,
                    0.0,
                );
            }
            return Ok(SaveResult::default());
        }
        // A continuation advances by a few hundred tokens per turn, so
        // persisting every turn rewrites a largely identical snapshot;
        // skipping writes that barely pass a stored prefix trades them for
        // re-prefilling the short gap. Shared prefixes are exempt: they give
        // new conversations a start and sit a few tokens past a deeper entry
        // by construction.
        if !shared_prefix
            && self.within_checkpoint_step(&metadata, runner, checkpoint_tokens, input_identity)
        {
            self.emit(Skipped, Reason::MinStep, 0, 0, token_count, 0.0);
            return Ok(SaveResult::default());
        }

        let started = Instant::now();
        let mut payload_bytes = 0;
        let mut file_bytes = 0;
        let header = 'prepare: {
            payload_bytes = match runner.persistent_snapshot_payload_bytes(snapshot) {
                Ok(0) | Err(_) => break 'prepare None,
                Ok(bytes) => bytes,
            };
            file_bytes = match checked_file_bytes(
                persistence.compatibility_identity.len(),
                token_count,
                payload_bytes,
            ) {
                Ok(bytes) => bytes,
                Err(_) => break 'prepare None,
            };
            if file_bytes > self.options.staging_capacity_bytes {
                self.emit(
                    Skipped,
                    Reason::StagingCapacity,
                    file_bytes,
                    payload_bytes,
                    token_count,
                    0.0,
                );
                return Ok(SaveResult::default());
            }
            if file_bytes > self.options.capacity_bytes {
                self.emit(
                    Skipped,
                    Reason::ByteCapacity,
                    file_bytes,
                    payload_bytes,
                    token_count,
                    0.0,
                );
                return Ok(SaveResult::default());
            }
            build_header(&persistence, checkpoint_tokens, payload_bytes).ok()
        };
        let Some(header) = header else {
            self.emit(
                Skipped,
                Reason::SerializationFailure,
                file_bytes,
                payload_bytes,
                token_count,
                0.0,
            );
            return Ok(SaveResult::default());
        };

        let filename = self.new_final_filename(&digest, &mut counter)?;
        drop(metadata);
        let published = self.publish_image(
            &filename,
            &header,
            runner,
            snapshot,
            payload_bytes,
            &mut counter,
        );
        let mut metadata = lock(&self.metadata);
        if !published {
            self.emit(
                Skipped,
                Reason::IoFailure,
                file_bytes,
                payload_bytes,
                token_count,
                0.0,
            );
            return Ok(SaveResult::default());
        }
        // No visible entry is evicted until its replacement is durable.
        if !self.make_capacity(&mut metadata, file_bytes) {
            self.remove_file_only(OsStr::new(&filename));
            self.emit(
                Skipped,
                Reason::ByteCapacity,
                file_bytes,
                payload_bytes,
                token_count,
                0.0,
            );
            return Ok(SaveResult::default());
        }

        metadata.add(Entry {
            filename: filename.into(),
            key_hash: digest,
            persistence,
            tokens: checkpoint_tokens.to_vec(),
            file_bytes,
            payload_bytes,
            last_access: now(),
        });
        self.retained.fetch_add(file_bytes, Ordering::Relaxed);
        self.retained_entries.fetch_add(1, Ordering::Relaxed);
        self.emit(
            ContinuationDiskEventAction::Stored,
            Reason::Saved,
            file_bytes,
            payload_bytes,
            token_count,
            started.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(SaveResult {
            stored: true,
            file_bytes,
            payload_bytes,
        })
    }

    /// The longest entry `prompt` may reuse: past `stable_prefix_tokens`
    /// only while an entry at or before that boundary remains.
    pub(super) fn usable_longest<R: TextModelRunner>(
        metadata: &Metadata,
        runner: &R,
        prompt: &[Token],
        input: InputIdentity<'_>,
        stable_prefix_tokens: usize,
    ) -> Option<EntryId> {
        let candidate = metadata.find_longest_input(runner, prompt, input)?;
        let entry = metadata.entries.get(&candidate)?;
        if stable_prefix_tokens != 0
            && entry.tokens.len() > stable_prefix_tokens
            && metadata
                .find_longest_input(runner, &prompt[..stable_prefix_tokens], input)
                .is_none()
        {
            return None;
        }
        Some(candidate)
    }

    /// Restores the longest usable entry deeper than `deeper_than` tokens;
    /// a rejected file makes way for the next one.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_longest_prefix<R: TextModelRunner>(
        &self,
        metadata: &mut Metadata,
        runner: &R,
        state: &mut R::State,
        prompt: &[Token],
        input: InputIdentity<'_>,
        stable_prefix_tokens: usize,
        deeper_than: usize,
    ) -> RestoreResult {
        use ContinuationDiskEventAction::Miss;
        use ContinuationDiskEventReason as Reason;

        if persistence_for_input(runner, input.complete).is_none() {
            self.emit(Miss, Reason::Unsupported, 0, 0, 0, 0.0);
            return RestoreResult::default();
        }
        loop {
            let Some(candidate) =
                Self::usable_longest(metadata, runner, prompt, input, stable_prefix_tokens)
            else {
                self.emit(Miss, Reason::NotFound, 0, 0, 0, 0.0);
                return RestoreResult::default();
            };
            let Some(entry) = metadata.entries.get(&candidate) else {
                return RestoreResult::default();
            };
            // A shallower RAM checkpoint is the caller's to restore.
            if entry.tokens.len() <= deeper_than {
                self.emit(Miss, Reason::NotFound, 0, 0, 0, 0.0);
                return RestoreResult::default();
            }
            let (file_bytes, payload_bytes, token_count) =
                (entry.file_bytes, entry.payload_bytes, entry.tokens.len());

            let mut image = match self.read_image(&entry.filename) {
                Ok((image, _)) => image,
                Err(reason) => {
                    let removed = self.remove_entry(metadata, candidate, reason);
                    self.emit(Miss, reason, file_bytes, payload_bytes, token_count, 0.0);
                    if !removed {
                        return RestoreResult::default();
                    }
                    continue;
                }
            };
            let parsed = match parse_and_verify_image(&mut image) {
                Ok(parsed)
                    if parsed.persistence == entry.persistence
                        && parsed.tokens == entry.tokens
                        && parsed.payload_bytes == entry.payload_bytes =>
                {
                    parsed
                }
                failure => {
                    let reason = if matches!(failure, Err(ParseFailure::Checksum)) {
                        Reason::ChecksumMismatch
                    } else {
                        Reason::Corrupt
                    };
                    let removed = self.remove_entry(metadata, candidate, reason);
                    self.emit(Miss, reason, file_bytes, payload_bytes, token_count, 0.0);
                    if !removed {
                        return RestoreResult::default();
                    }
                    continue;
                }
            };

            let payload =
                &image[parsed.payload_offset..parsed.payload_offset + parsed.payload_bytes];
            if runner.restore_persistent_snapshot(state, payload).is_err() {
                state.invalidate();
                self.remove_entry(metadata, candidate, Reason::RestoreFailure);
                self.emit(
                    Miss,
                    Reason::RestoreFailure,
                    file_bytes,
                    payload_bytes,
                    token_count,
                    0.0,
                );
                return RestoreResult::default();
            }

            self.touch_entry(metadata, candidate);
            self.emit(
                ContinuationDiskEventAction::Restored,
                Reason::Hit,
                file_bytes,
                payload_bytes,
                token_count,
                0.0,
            );
            return RestoreResult {
                restored: true,
                token_count,
                file_bytes,
                payload_bytes,
            };
        }
    }

    /// Marks an entry as used now, in memory and as the file's times.
    pub(super) fn touch_entry(&self, metadata: &mut Metadata, id: EntryId) {
        let Some(entry) = metadata.entries.get_mut(&id) else {
            return;
        };
        entry.last_access = now();
        let current = Timespec {
            tv_sec: 0,
            tv_nsec: UTIME_NOW,
        };
        let _ = rustix::fs::utimensat(
            &self.directory,
            entry.filename.as_os_str(),
            &Timestamps {
                last_access: current,
                last_modification: current,
            },
            AtFlags::SYMLINK_NOFOLLOW,
        );
    }

    pub(super) fn shared_prefix_boundaries<R: TextModelRunner>(
        &self,
        metadata: &Metadata,
        runner: &R,
        prompt: &[Token],
        min_tokens: usize,
        max_boundaries: usize,
        input: InputIdentity<'_>,
    ) -> Vec<usize> {
        if persistence_for_input(runner, input.complete).is_none() || max_boundaries == 0 {
            return Vec::new();
        }
        let mut boundaries = Vec::new();
        visit_input_ranges(prompt.len(), input, |begin, end, identity| {
            let Some(persistence) = persistence_for_input(runner, identity) else {
                return;
            };
            let key = (
                persistence.payload_version,
                persistence.compatibility_identity,
            );
            let Some(root) = metadata.prefixes.get(&key) else {
                return;
            };
            boundaries.extend(
                root.find(&prompt[..end], true)
                    .shared_boundaries
                    .into_iter()
                    .filter(|&count| count >= begin && count < prompt.len()),
            );
        });
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.retain(|&length| length >= min_tokens);
        // Keep the longest ones: they save the most prefill when they hit.
        if boundaries.len() > max_boundaries {
            boundaries.drain(..boundaries.len() - max_boundaries);
        }
        boundaries
    }

    pub(super) fn touch<R: TextModelRunner>(
        &self,
        metadata: &mut Metadata,
        runner: &R,
        tokens: &[Token],
        input_identity: &[u8],
    ) -> Result<bool> {
        let Some(persistence) = persistence_for_input(runner, input_identity) else {
            return Ok(false);
        };
        let digest = self.hash_key(&persistence, tokens)?;
        let Some(entry) = metadata.find_exact(&digest, &persistence, tokens) else {
            return Ok(false);
        };
        self.touch_entry(metadata, entry);
        Ok(true)
    }

    /// The single writer: drains the queue, also after `stopping` is set.
    pub(super) fn persist_queued(&self) {
        loop {
            let PendingSave {
                job,
                token_count,
                retained_bytes,
                publication,
            } = {
                let mut queue = lock(&self.queue);
                while !queue.stopping && queue.pending.is_empty() {
                    queue = self
                        .queue_changed
                        .wait(queue)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                match queue.pending.pop_front() {
                    Some(pending) => pending,
                    None => return,
                }
            };
            lock(&self.queue).in_flight = Some(publication);
            // The job consumes and drops the runner and snapshot references:
            // snapshot storage is released before its budget becomes free.
            let saved = catch_unwind(AssertUnwindSafe(|| job(self)));
            if !matches!(saved, Ok(Ok(_))) {
                self.emit(
                    ContinuationDiskEventAction::Skipped,
                    ContinuationDiskEventReason::IoFailure,
                    0,
                    0,
                    token_count,
                    0.0,
                );
            }
            {
                let mut queue = lock(&self.queue);
                queue.queued_bytes -= retained_bytes;
                queue.in_flight = None;
            }
            self.queue_changed.notify_all();
        }
    }

    /// Whether `bytes` fit the disk and staging budgets (with the queue
    /// locked). A refusal is reported before anything allocates, where it
    /// would otherwise look like a successful capture.
    pub(super) fn admits(
        &self,
        queue: &Queue,
        bytes: usize,
        payload_bytes: usize,
        token_count: usize,
    ) -> bool {
        if queue.stopping {
            return false;
        }
        let used = queue.queued_bytes + self.capture_bytes.load(Ordering::Relaxed);
        let limit = self.staging_limit();
        let reason = if bytes > self.options.capacity_bytes {
            ContinuationDiskEventReason::ByteCapacity
        } else if bytes > limit || used > limit - bytes {
            ContinuationDiskEventReason::StagingCapacity
        } else {
            return true;
        };
        self.emit_staging(
            ContinuationDiskEventAction::Skipped,
            reason,
            bytes,
            payload_bytes,
            token_count,
            0.0,
            used,
        );
        false
    }
}
