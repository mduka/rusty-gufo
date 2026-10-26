//! Admitting queued requests: shared prefixes, waiting for published
//! checkpoints, and acquiring a session.

use super::*;

impl<R: TextModelRunner> Worker<R> {
    /// Parks a cold request behind a resident one still prefilling the
    /// longest prefix both prompts share: that request publishes a checkpoint
    /// there, which this one restores instead of prefilling the same tokens
    /// in lockstep. Prefill runs one request at a time, so waiting costs
    /// nothing.
    pub(super) fn wait_for_shared_prefix(
        &self,
        work: &mut Work<R>,
        prefilling: &mut VecDeque<Box<Work<R>>>,
        capturing: &mut VecDeque<Box<Work<R>>>,
        waiting: usize,
    ) -> bool {
        if std::mem::replace(&mut work.prefix_considered, true)
            || !work.cache_prompt
            // Parked requests hold no runner lease; bound their reservations
            // like admitted ones.
            || waiting + 1 >= self.pool.capacity()
        {
            return false;
        }
        let input = work
            .prompt_context
            .as_deref()
            .map_or_else(InputIdentity::default, |context| InputIdentity {
                complete: context.cache_identity(),
                prefixes: context.cache_prefixes(),
            });
        let mut leader: Option<&mut Box<Work<R>>> = None;
        let mut common = 0;
        for resident in prefilling.iter_mut().chain(capturing.iter_mut()) {
            let Some(request) = resident
                .runner_request
                .as_ref()
                .filter(|r| r.is_active() && !r.prefill_complete())
            else {
                continue;
            };
            if resident.shared.is_terminal() {
                continue;
            }
            let shared = work
                .prompt
                .iter()
                .zip(request.prompt())
                .take_while(|(a, b)| a == b)
                .count();
            if shared <= common
                || shared <= request.prefill_position()
                || input.at(shared) != request.input_identity(shared)
            {
                continue;
            }
            common = shared;
            leader = Some(resident);
        }
        let Some(leader) = leader.filter(|_| common >= SHARED_PREFIX_MIN_TOKENS) else {
            return false;
        };
        // A longer retained prefix, such as the conversation's previous
        // turn, already beats waiting for a peer.
        let cached = self
            .pool
            .cached_prefix_tokens(&work.prompt, work.prompt_context.as_deref());
        if common < cached + SHARED_PREFIX_MIN_TOKENS {
            return false;
        }
        let Some(request) = leader.runner_request.as_mut() else {
            return false;
        };
        let position = request.share_checkpoint(common);
        if position < cached + SHARED_PREFIX_MIN_TOKENS {
            return false;
        }
        work.prefix_leader = Some(leader.shared.clone());
        work.prefix_position = position;
        work.prefix_wait_start = Some(Instant::now());
        true
    }

    /// Whether a parked request may resume: its leader's checkpoint is
    /// retained, or the leader can no longer publish it.
    pub(super) fn shared_prefix_settled(
        &self,
        work: &Work<R>,
        prefilling: &VecDeque<Box<Work<R>>>,
        capturing: &VecDeque<Box<Work<R>>>,
    ) -> bool {
        let Some(leader) = &work.prefix_leader else {
            return true;
        };
        let Some(request) = prefilling
            .iter()
            .chain(capturing.iter())
            .find(|w| Arc::ptr_eq(&w.shared, leader))
            .and_then(|w| w.runner_request.as_ref())
            .filter(|r| r.is_active())
        else {
            return true;
        };
        if leader.is_terminal()
            || request.prefill_complete()
            || request.prefill_position() > work.prefix_position
        {
            return true;
        }
        request.prefill_position() == work.prefix_position
            && self
                .pool
                .cached_prefix_tokens(&work.prompt, work.prompt_context.as_deref())
                >= work.prefix_position
    }

    /// Drops stopped parked requests and releases settled ones; released
    /// requests are admitted before newer queued ones, in arrival order.
    /// Parks a request whose prompt extends a checkpoint still being
    /// written to disk when the saving exceeds
    /// [`PendingPublication::MINIMUM_GAIN_TOKENS`] (`PendingPublication`):
    /// at most as long as prefilling that saving takes at the recent rate.
    pub(super) fn wait_for_publication(&self, work: &mut Work<R>) -> bool {
        // Unlike a shared-prefix wait no resident leader is needed: admission
        // already left this request a session to reserve while parked.
        if std::mem::replace(&mut work.publication_considered, true) || !work.cache_prompt {
            return false;
        }
        let context = work.prompt_context.as_deref();
        let Some(pending) = self.pool.pending_publication(&work.prompt, context, None) else {
            return false;
        };
        let id = work.shared.id;
        let Some(limit) = self.publication_estimate.limit(pending.gain()) else {
            logging::info(
                "cache",
                &format!(
                    "event=publication_wait request={id} stage=skip checkpoint_id={} \
                     pending_tokens={} available_tokens={} release=no_estimate",
                    pending.checkpoint_id, pending.pending_tokens, pending.available_tokens
                ),
            );
            return false;
        };
        logging::info(
            "cache",
            &format!(
                "event=publication_wait request={id} stage=start checkpoint_id={} \
                 pending_tokens={} available_tokens={} limit_ms={:.1}",
                pending.checkpoint_id,
                pending.pending_tokens,
                pending.available_tokens,
                limit.as_secs_f64() * 1e3
            ),
        );
        work.publication = Some(PublicationWait {
            checkpoint_id: pending.checkpoint_id,
            pending_tokens: pending.pending_tokens,
            limit,
            start: Instant::now(),
        });
        true
    }

    /// Releases a request parked for a publication once its checkpoint is
    /// written (`not_pending`), the prefix is available otherwise
    /// (`available`), or the wait costs as much as the prefill it saves
    /// (`cost_bound`, at most `cap`).
    pub(super) fn update_publication_wait(&self, work: &mut Work<R>) {
        let Some(wait) = &mut work.publication else {
            return;
        };
        let context = work.prompt_context.as_deref();
        let available = self.pool.cached_prefix_tokens(&work.prompt, context);
        let pending =
            self.pool
                .pending_publication(&work.prompt, context, Some(wait.checkpoint_id));
        if let Some(limit) = pending.and_then(|p| {
            self.publication_estimate.limit(
                PendingPublication {
                    available_tokens: available,
                    ..p
                }
                .gain(),
            )
        }) {
            wait.limit = wait.limit.min(limit);
        }
        let waited = wait.start.elapsed();
        let release = if pending.is_none() {
            "not_pending"
        } else if available >= wait.pending_tokens {
            "available"
        } else if waited >= wait.limit {
            if wait.limit >= PublicationWaitEstimate::MAXIMUM_WAIT {
                "cap"
            } else {
                "cost_bound"
            }
        } else {
            return;
        };
        logging::info(
            "cache",
            &format!(
                "event=publication_wait request={} stage=end checkpoint_id={} \
                 pending_tokens={} available_tokens={available} limit_ms={:.1} wait_ms={:.1} \
                 release={release}",
                work.shared.id,
                wait.checkpoint_id,
                wait.pending_tokens,
                wait.limit.as_secs_f64() * 1e3,
                waited.as_secs_f64() * 1e3
            ),
        );
        work.publication = None;
    }

    pub(super) fn update_waiting(
        &self,
        waiting: &mut VecDeque<Box<Work<R>>>,
        prefilling: &VecDeque<Box<Work<R>>>,
        capturing: &VecDeque<Box<Work<R>>>,
    ) {
        let mut index = 0;
        while index < waiting.len() {
            if self.complete_if_stopped(&mut waiting[index]) {
                waiting.remove(index);
                continue;
            }
            if waiting[index].prefix_leader.is_some()
                && self.shared_prefix_settled(&waiting[index], prefilling, capturing)
            {
                let work = &mut waiting[index];
                work.result.shared_prefix_wait_ms = work.prefix_wait_start.map_or(0.0, elapsed_ms);
                work.prefix_leader = None;
            }
            self.update_publication_wait(&mut waiting[index]);
            index += 1;
        }
    }

    pub(super) fn admit(
        &mut self,
        prefilling: &mut VecDeque<Box<Work<R>>>,
        decoding: &mut VecDeque<Box<Work<R>>>,
        capturing: &mut VecDeque<Box<Work<R>>>,
        waiting: &mut VecDeque<Box<Work<R>>>,
    ) {
        self.update_waiting(waiting, prefilling, capturing);
        // Captures keep their runner lease until decoding resumes.
        while !self.stop_requested()
            && prefilling.len() + decoding.len() + capturing.len() < self.pool.capacity()
        {
            let released = waiting
                .iter()
                .position(|w| w.prefix_leader.is_none() && w.publication.is_none());
            let mut work = match released.and_then(|index| waiting.remove(index)) {
                Some(work) => work,
                None => {
                    // Parked requests already reserve a visible session; they
                    // resume in place, but new arrivals respect it.
                    if prefilling.len() + decoding.len() + capturing.len() + waiting.len()
                        >= self.pool.capacity()
                    {
                        return;
                    }
                    let Some(work) = self.pop_queued() else {
                        return;
                    };
                    work
                }
            };
            if self.complete_if_stopped(&mut work) {
                continue;
            }
            if self.wait_for_shared_prefix(&mut work, prefilling, capturing, waiting.len())
                || self.wait_for_publication(&mut work)
            {
                waiting.push_back(work);
                continue;
            }
            if let Err(failure) = self.acquire(
                &mut work,
                prefilling.len() + decoding.len() + capturing.len(),
            ) {
                self.complete_failure(&mut work, failure);
                continue;
            }
            if work.is_terminal() {
                continue;
            }
            if work.prefill_complete() {
                work.decode_due = true;
                decoding.push_back(work);
            } else {
                // The last-served prefill just moved to the back of its
                // round. Give an arrival its first turn before repeating that
                // work, while keeping the order of peers already waiting.
                let at = match prefilling.back() {
                    Some(last) if last.shared.id == self.last_prefill_request_id => {
                        prefilling.len() - 1
                    }
                    _ => prefilling.len(),
                };
                prefilling.insert(at, work);
            }
        }
    }

    pub(super) fn acquire(&self, work: &mut Work<R>, resident: usize) -> Result<()> {
        let weak: Weak<RequestShared> = Arc::downgrade(&work.shared);
        let stop = self.shared.stop.clone();
        let check: CancellationCheck = Arc::new(move || {
            stop.load(Ordering::Acquire)
                || weak
                    .upgrade()
                    .is_none_or(|r| r.cancellation_requested() || r.deadline_exceeded())
        });
        let request = self.pool.acquire_with(
            std::mem::take(&mut work.prompt),
            AcquireOptions {
                sampling: work.sampling.clone(),
                is_cancelled: Some(check),
                context: work.prompt_context.take(),
                reuse_prompt: work.cache_prompt,
                cache_prefix_tokens: work.cache_prefix_tokens,
                stop_at_eos: work.stop_at_eos,
            },
        )?;
        // Each admission may restore or capture a checkpoint of its own.
        self.advance_progress();
        if !request.is_active() {
            self.complete_cancelled(work);
            return Ok(());
        }
        let result = &mut work.result;
        result.cache_hit = request.cache_hit();
        let lookup = request.cache_lookup();
        result.cache_miss_reason = lookup.miss_reason.into();
        result.cache_common_prefix_tokens = lookup.common_prefix_tokens;
        result.cache_checkpoint_tokens = lookup.checkpoint_tokens;
        result.cached_prompt_tokens = request.cached_prompt_tokens();
        work.shared
            .live_cached_prompt_tokens
            .store(result.cached_prompt_tokens, Ordering::Relaxed);
        result.cache_restore_bytes = request.cache_restore_bytes();
        result.cache_restore_ms = request.cache_restore_ms();
        result.cache_disk_hit = request.cache_disk_hit();
        result.incremental_prefill_supported = self.incremental_prefill_supported;
        result.queue_ms = elapsed_ms(work.request_start);
        result.resident_requests_at_admission = resident + 1;
        let prefill_complete = request.prefill_complete();
        work.runner_request = Some(request);
        work.publish_admission();
        work.publish_prompt_progress();
        work.shared.set_phase(TextRequestPhase::Admitted);
        work.shared.set_phase(if prefill_complete {
            TextRequestPhase::DecodeReady
        } else {
            TextRequestPhase::Prefilling
        });
        Ok(())
    }
}
