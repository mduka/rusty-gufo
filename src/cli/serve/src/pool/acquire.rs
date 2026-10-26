//! Acquiring a state for a request: the longest retained prefix from RAM
//! or disk, and freezing a live frontier for reuse.

use super::*;

impl<R: TextModelRunner> TextRunnerPool<R> {
    /// Acquires a state for `prompt` with default options.
    /// The longest prefix of `prompt` retained in RAM or on disk that an
    /// acquire could reuse, without leasing a state; an upper bound, since
    /// acquire may still prefer a shorter one.
    pub fn cached_prefix_tokens(
        &self,
        prompt: &[Token],
        context: Option<&R::PromptContext>,
    ) -> usize {
        let input = context.map_or_else(InputIdentity::default, |context| InputIdentity {
            complete: context.cache_identity(),
            prefixes: context.cache_prefixes(),
        });
        let disk = self.disk_store.as_ref().map_or(0, |store| {
            store.longest_prefix_tokens(&*self.validated.runner, prompt, input, 0)
        });
        self.cache.cached_prefix_tokens(prompt, input).max(disk)
    }

    /// The deepest checkpoint still being written to disk that `prompt`
    /// extends; a first query (`checkpoint` `None`) wants a large saving
    /// over the prefix [`cached_prefix_tokens`](Self::cached_prefix_tokens)
    /// reuses now, a follow-up names its checkpoint.
    pub fn pending_publication(
        &self,
        prompt: &[Token],
        context: Option<&R::PromptContext>,
        checkpoint: Option<u64>,
    ) -> Option<PendingPublication> {
        let store = self.disk_store.as_ref()?;
        let input = context.map_or_else(InputIdentity::default, |context| InputIdentity {
            complete: context.cache_identity(),
            prefixes: context.cache_prefixes(),
        });
        let available = self.cached_prefix_tokens(prompt, context);
        store.pending_publication(prompt, input, available, checkpoint)
    }

    pub fn acquire(&self, prompt: Vec<Token>) -> Result<PoolRequest<R>> {
        self.acquire_with(prompt, AcquireOptions::default())
    }

    /// Leases a state, reusing the longest retained prefix of the prompt
    /// from RAM or disk; past `cache_prefix_tokens` only while a fallback at
    /// or before that boundary remains. Returns an inactive request when
    /// `is_cancelled` fires while waiting.
    pub fn acquire_with(
        &self,
        prompt: Vec<Token>,
        options: AcquireOptions<R::PromptContext>,
    ) -> Result<PoolRequest<R>> {
        let AcquireOptions {
            sampling,
            is_cancelled,
            context,
            reuse_prompt,
            cache_prefix_tokens,
            stop_at_eos,
        } = options;
        sampling.validate()?;
        if prompt.is_empty() {
            return Err(Error::invalid("text runner prompt must not be empty"));
        }
        if prompt.len() > self.validated.descriptor.max_context as usize {
            return Err(Error::length("text runner prompt exceeds model context"));
        }
        if cache_prefix_tokens > prompt.len() {
            return Err(Error::invalid("cache prefix exceeds prompt length"));
        }
        let Some(_admission) = self.admission.lock(is_cancelled.as_ref()) else {
            return Ok(PoolRequest::empty());
        };
        let cancelled = || is_cancelled.as_ref().is_some_and(|c| c());
        let runner = &self.validated.runner;
        // Checkpoints before a request's later images keep the identity they
        // were computed with.
        let input_context = context.clone();
        let input = input_context
            .as_deref()
            .map_or_else(InputIdentity::default, |c| InputIdentity {
                complete: c.cache_identity(),
                prefixes: c.cache_prefixes(),
            });
        let mut prepare = |state: &mut R::State| {
            state.set_cancellation_check(is_cancelled.clone());
            state.set_stop_at_eos(stop_at_eos)?;
            runner.set_prompt_context(state, context.as_ref())
        };
        let cancel_check: &dyn Fn() -> bool = &cancelled;
        // The deepest checkpoint wins across RAM and disk, RAM on equal
        // lengths (upstream's tiered lookup); the probe reads metadata only.
        let disk_tokens = match &self.disk_store {
            Some(store) if reuse_prompt => {
                store.longest_prefix_tokens(&**runner, &prompt, input, cache_prefix_tokens)
            }
            _ => 0,
        };
        // Full checkpoints are reused as well as branching fallbacks; both
        // tiers make unmarked older entries leave room for the stable
        // boundary.
        let Some(mut lease) = self.cache.acquire_tiered(
            &prompt,
            Some(cancel_check),
            input,
            Some(&mut prepare),
            reuse_prompt,
            cache_prefix_tokens,
            disk_tokens,
        )?
        else {
            return Ok(PoolRequest::empty());
        };
        let capabilities = self.validated.descriptor.capabilities;
        // A live frontier can hold generated tokens beyond the prompt
        // snapshot. Freeze it before the first branch mutates it, so peers
        // restore the same decode history instead of prefilling generated
        // output. C1 captures any required fallback just before prefill.
        if self.cache.capacity() > 1
            && lease.cache_hit()
            && capabilities.snapshot
            && capabilities.fork
            && !cancelled()
        {
            let prefix = &prompt[..lease.cached_tokens()];
            if !lease.has_snapshot_for(prefix) {
                self.freeze_live_frontier(&mut lease, prefix, input.at(prefix.len()));
            }
        }
        if reuse_prompt
            && !lease.cache_hit()
            && !cancelled()
            && let Some(store) = &self.disk_store
        {
            let restore_start = Instant::now();
            let ram_tokens = lease.fallback_tokens();
            let restored = store.restore_longest_prefix(
                &**runner,
                lease.state_mut()?,
                &prompt,
                input,
                cache_prefix_tokens,
                ram_tokens,
            );
            match restored {
                Ok(restored) if restored.restored => {
                    // The restore replaced the position layout: reattach the
                    // request's inputs, which may add images after it.
                    if context.is_some() {
                        runner.set_prompt_context(lease.state_mut()?, context.as_ref())?;
                    }
                    lease.adopt_restored_prefix(
                        restored.token_count,
                        restored.file_bytes,
                        elapsed_ms(restore_start),
                    )?;
                }
                restored => {
                    // A failed or refused restore leaves the state for the
                    // next shallower checkpoint: the RAM one, if any.
                    if restored.is_err() {
                        TextRunnerState::invalidate(lease.state_mut()?);
                    }
                    prepare(lease.state_mut()?)?;
                    lease.restore_ram_fallback(&prompt, &mut prepare)?;
                }
            }
        }
        let mut boundaries = Vec::new();
        if reuse_prompt
            && !cancelled()
            && let Some(store) = &self.disk_store
        {
            boundaries = store
                .shared_prefix_boundaries(
                    &**runner,
                    &prompt,
                    self.shared_prefix_min_tokens,
                    self.shared_prefix_max_boundaries,
                    input,
                )
                .unwrap_or_default();
            let cached = lease.cached_tokens();
            boundaries.retain(|&b| b > cached);
        }
        let sampler = runner.create_sampler(&sampling, &prompt)?;
        let prefill_offset = lease.cached_tokens();
        let retain_fallback = cache_prefix_tokens != 0;
        let mut count = if retain_fallback {
            cache_prefix_tokens
        } else {
            prompt.len()
        };
        // On a warm continuation the reused frontier already is a safe
        // fallback: freeze it before prefill, then retain this turn's own
        // stable boundary before the assistant framing. Cold requests
        // checkpoint before that framing, and so does a newly consumed
        // image: otherwise the next turn's framing can invalidate the
        // complete prompt and send the image through prefill again from the
        // older text frontier.
        let new_images = input.at(prefill_offset) != input.at(count);
        let mut stable_prefix_position = 0;
        if retain_fallback && lease.cache_hit() && prefill_offset <= count && !new_images {
            // Freezing the reused frontier must not cost this turn its own
            // stable boundary. A client that rewrites the assistant turn, as
            // one dropping reasoning does, diverges after that boundary:
            // keeping only the frontier would pin every later turn to the
            // same early position, with a re-prefilled tail that grows for
            // the rest of the conversation.
            if prefill_offset < count {
                stable_prefix_position = count;
            }
            count = prefill_offset;
        }
        if lease.restored_from_disk() && prefill_offset == prompt.len() {
            count = prompt.len();
        }
        let mut checkpoints = Vec::new();
        if capabilities.incremental_prefill
            && capabilities.prefix_reuse
            && capabilities.snapshot
            && capabilities.fork
        {
            // A reused frontier already bounds the work an edit loses: a short
            // continuation is not split just to copy a nearby grid point,
            // while cold long prompts keep the last grid point for late edits.
            // A grid point just before the prompt end saves little on an edit
            // but would split the final prefill pass and store another
            // snapshot.
            let grid_points = (prompt.len() - 1) / CHECKPOINT_INTERVAL;
            let points = grid_points.min(INTERMEDIATE_CHECKPOINTS);
            for point in 1..=points {
                let position = grid_points * point / points * CHECKPOINT_INTERVAL;
                if position > prefill_offset
                    && position - prefill_offset >= CHECKPOINT_INTERVAL
                    && prompt.len() - position > CHECKPOINT_TAIL
                    && position != count
                    && position != stable_prefix_position
                    && !lease.has_snapshot_for(&prompt[..position])
                {
                    checkpoints.push(position);
                }
            }
        }
        let mut inner = RequestInner {
            runner: runner.clone(),
            capabilities,
            disk_store: self.disk_store.clone(),
            snapshot_tokens: prompt[..count].to_vec(),
            // A live continuation can already be beyond the snapshot point:
            // keep its existing checkpoint, never snapshot mismatched state.
            prompt_snapshot_attempted: prefill_offset > count,
            decode_ready: prefill_offset == prompt.len(),
            prompt,
            lease,
            boundaries,
            checkpoints,
            boundary: None,
            shared_checkpoints: Vec::new(),
            overlap_snapshots: self.cache.capacity() > 1,
            generated: Vec::new(),
            prefill_offset,
            state_reusable: true,
            stopped: false,
            pending_selection: None,
            sampler,
            context,
            prompt_snapshot: None,
            snapshot_metrics: CommitMetrics::default(),
            snapshot_bytes: 0,
            retain_snapshot: false,
            snapshot_start: Instant::now(),
            disk_capture: None,
            capture: None,
            retain_fallback,
            fallback_position: None,
            stable_prefix_position,
        };
        if inner.lease.cache_hit() {
            let RequestInner { lease, prompt, .. } = &mut inner;
            runner.prepare_prefix_reuse(lease.state_mut()?, &prompt[..prefill_offset])?;
        }
        let mut request = PoolRequest { inner: Some(inner) };
        // A prompt diverging from a retained one after a long common prefix,
        // such as a new conversation under one system prompt, publishes a
        // checkpoint at the divergence, which later prompts sharing it
        // restore exactly instead of the nearest grid point. A divergence in
        // the last few tokens (an edited final message) is covered by the
        // request's own stable checkpoint already.
        if reuse_prompt {
            let prompt = request.prompt();
            let shared = self.cache.common_prefix_tokens(prompt, input);
            if shared >= request.prefill_position() + SHARED_PREFIX_MIN_TOKENS
                && shared + SHARED_CHECKPOINT_SLACK < prompt.len()
            {
                request.share_checkpoint(shared);
            }
        }
        Ok(request)
    }

    pub(super) fn freeze_live_frontier(
        &self,
        lease: &mut Lease<R::State, R::Snapshot>,
        prefix: &[Token],
        identity: &[u8],
    ) {
        let runner = &self.validated.runner;
        let started = Instant::now();
        let mut bytes = 0;
        let captured = (|| -> Result<Option<Arc<R::Snapshot>>> {
            let state = lease.state()?;
            if runner.checkpoint_position(state)? != prefix.len() {
                return Err(Error::logic("live checkpoint position mismatch"));
            }
            bytes = runner.snapshot_payload_bytes(state)?;
            if !lease.try_reserve_snapshot(
                bytes,
                prefix.len(),
                true,
                SnapshotPurpose::Continuation,
                &[],
            )? {
                return Ok(None);
            }
            let snapshot = Arc::new(runner.snapshot(lease.state()?)?);
            if lease.publish_snapshot(prefix.to_vec(), snapshot.clone(), false)? == 0 {
                return Ok(None);
            }
            Ok(Some(snapshot))
        })();
        let snapshot = match captured {
            Ok(snapshot) => snapshot,
            Err(_) => {
                lease.skip_snapshot(SnapshotEventReason::CaptureFailure, bytes, prefix.len());
                None
            }
        };
        let Some(snapshot) = snapshot else {
            return;
        };
        if let Some(store) = &self.disk_store {
            // RAM branching does not depend on optional disk admission.
            let _ = store.save_async(
                runner.clone(),
                prefix.to_vec(),
                snapshot,
                identity.to_vec(),
                None,
                false,
            );
        }
        logging::info(
            "cache",
            &format!(
                "event=live_checkpoint tokens={} bytes={bytes} capture_ms={}",
                prefix.len(),
                format_general(elapsed_ms(started))
            ),
        );
    }
}
