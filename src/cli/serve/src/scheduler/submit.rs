//! The scheduler's interface: start-up, state queries and submission.

use super::*;

impl<R: TextModelRunner> TextGenerationScheduler<R> {
    pub fn new(
        pool: Arc<TextRunnerPool<R>>,
        prefill_policy: TextPrefillPolicy,
        policy: TextSchedulerPolicy,
    ) -> Result<Self> {
        if prefill_policy.decode_active_tokens == 0 {
            return Err(Error::invalid(
                "active-decode prefill budget must be at least one token",
            ));
        }
        if policy.max_pending_requests == 0
            || policy.max_pending_requests_per_client == 0
            || policy.max_pending_requests_per_client > policy.max_pending_requests
            || policy.max_output_bytes_per_request == 0
            || policy.max_buffered_output_bytes_per_request == 0
            || policy.max_buffered_output_bytes_total == 0
        {
            return Err(Error::invalid("invalid text scheduler limits"));
        }
        let shared = Arc::new(SchedulerShared {
            queue: Mutex::new(Queue {
                clients: VecDeque::new(),
                count: 0,
                stopping: false,
            }),
            sessions: Arc::new(Mutex::new(vec![None; pool.capacity()])),
            queue_condition: Condvar::new(),
            stop: Arc::new(AtomicBool::new(false)),
            device_lost: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            progress: AtomicU64::new(0),
            loop_working: AtomicBool::new(false),
        });
        let capabilities = pool.descriptor().capabilities;
        let worker = Worker {
            shared: shared.clone(),
            pool: pool.clone(),
            prefill_policy,
            incremental_prefill_supported: capabilities.incremental_prefill,
            final_token_advance_required: capabilities.final_token_advance_required,
            incremental_text_is_exact: capabilities.incremental_text_is_exact,
            multi_token_decode: capabilities.multi_token_decode,
            batched_multi_token_decode: capabilities.batched_multi_token_decode,
            batched_multi_token_decode_max_width: capabilities.batched_multi_token_decode_max_width,
            log_progress: policy.log_progress,
            device_probe_interval: policy.device_probe_interval,
            consecutive_active_prefill_chunks: 0,
            last_prefill_request_id: 0,
            publication_estimate: PublicationWaitEstimate::default(),
            gpu_share: policy.gpu_share.clone(),
            retained: RefCell::new(Vec::new()),
        };
        let failure = shared.clone();
        let worker = std::thread::Builder::new()
            .name("gufo-scheduler".into())
            .spawn(move || {
                // The C++ worker terminates the process on an escaped
                // exception; here the requests fail and the server reports
                // the failure until a restart.
                if let Err(payload) = catch_unwind(AssertUnwindSafe(|| worker.run())) {
                    failure.fail_after_panic(payload.as_ref());
                }
            })
            .map_err(|e| Error::runtime(format!("cannot start the text scheduler: {e}")))?;
        let stall_monitor = if policy.stall_timeout.is_zero() {
            None
        } else {
            let shared = shared.clone();
            let timeout = policy.stall_timeout;
            Some(
                std::thread::Builder::new()
                    .name("gufo-stall-monitor".into())
                    .spawn(move || shared.monitor_stalls(timeout))
                    .map_err(|e| Error::runtime(format!("cannot start the stall monitor: {e}")))?,
            )
        };
        let output_budget_limit = policy.max_buffered_output_bytes_total;
        Ok(Self {
            shared,
            pool,
            prefill_policy,
            policy,
            output_budget: Arc::new(OutputBudget {
                limit: output_budget_limit,
                buffered_bytes: AtomicUsize::new(0),
                max_buffered_bytes: AtomicUsize::new(0),
            }),
            next_request_id: AtomicU64::new(1),
            worker: Some(worker),
            stall_monitor,
        })
    }

    pub fn runner(&self) -> &Arc<R> {
        self.pool.runner()
    }

    /// The runner's session context.
    pub fn max_context(&self) -> u32 {
        self.pool.descriptor().max_context
    }

    /// Whether a failed generation showed that the device can no longer run
    /// work, for example after a GPU reset; permanent.
    pub fn device_lost(&self) -> bool {
        self.shared.device_lost.load(Ordering::Acquire)
    }

    /// The permanent failure that stops all work, if any: a lost device, or
    /// a panic on the scheduler thread.
    pub fn fatal_failure(&self) -> Option<GenerationErrorCode> {
        if self.device_lost() {
            Some(GenerationErrorCode::DeviceLost)
        } else if self.shared.failed.load(Ordering::Acquire) {
            Some(GenerationErrorCode::SchedulerFailed)
        } else {
            None
        }
    }

    /// One entry per session; admitted requests hold one until terminal,
    /// parked ones included.
    pub fn session_states(&self) -> Vec<SessionState> {
        let speculative = self.pool.descriptor().capabilities.multi_token_decode;
        lock(&self.shared.sessions)
            .iter()
            .map(|request| {
                let Some(request) = request else {
                    return SessionState {
                        speculative,
                        ..SessionState::default()
                    };
                };
                // Identity, prompt size and limit are fixed before admission.
                let generated = request.live_generated_tokens.load(Ordering::Relaxed);
                SessionState {
                    processing: true,
                    speculative,
                    request_id: request.id,
                    prompt_tokens: request.prompt_tokens,
                    cached_prompt_tokens: request.live_cached_prompt_tokens.load(Ordering::Relaxed),
                    processed_prompt_tokens: request.live_prefill_tokens.load(Ordering::Relaxed),
                    generated_tokens: generated,
                    remaining_tokens: request.token_limit - generated.min(request.token_limit),
                }
            })
            .collect()
    }

    pub fn capacity(&self) -> usize {
        self.pool.capacity()
    }

    pub fn buffered_output_bytes(&self) -> usize {
        self.output_budget.buffered_bytes.load(Ordering::Relaxed)
    }

    pub fn max_buffered_output_bytes(&self) -> usize {
        self.output_budget
            .max_buffered_bytes
            .load(Ordering::Relaxed)
    }

    /// Queues a request or rejects it (queue limits, stopping) before any
    /// state is admitted, so streaming responses fail before their headers.
    pub fn submit(
        &self,
        prompt: Vec<Token>,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<CancellationCheck>,
        publish_token_pieces: bool,
        metadata: RequestMetadata<R::PromptContext>,
    ) -> Result<SchedulerRequest> {
        if prompt.is_empty() {
            return Err(Error::invalid("text scheduler prompt must not be empty"));
        }
        let context = self.pool.descriptor().max_context as usize;
        if prompt.len() >= context {
            return Err(Error::length(format!(
                "prompt has {} tokens but the context is {context}; increase --context or \
                 shorten the conversation",
                prompt.len()
            )));
        }
        // Zero generates until EOS or the context is full.
        let available = context - prompt.len();
        let token_limit = if max_tokens > 0 {
            max_tokens.min(available)
        } else {
            available
        };
        sampling.validate()?;
        validate_stop_sequences(&metadata.stop_sequences, MAX_STOP_SEQUENCES)
            .map_err(Error::invalid)?;
        if !metadata.stop_sequences.is_empty()
            && !self
                .pool
                .descriptor()
                .capabilities
                .incremental_text_is_exact
        {
            return Err(Error::invalid(
                "stop sequences require exact incremental token decoding",
            ));
        }
        let stop_filter =
            StopSequenceFilter::new(metadata.stop_sequences).map_err(Error::invalid)?;
        let client_id = if metadata.client_id.is_empty() {
            "anonymous".to_owned()
        } else {
            metadata.client_id
        };
        let deadline = metadata.deadline.or_else(|| {
            (!self.policy.request_timeout.is_zero())
                .then(|| metadata.request_start + self.policy.request_timeout)
        });
        let shared = Arc::new(RequestShared {
            id: self.next_request_id.fetch_add(1, Ordering::Relaxed),
            cancellation_requested: AtomicBool::new(false),
            phase: AtomicU8::new(TextRequestPhase::Queued as u8),
            external_cancellation: is_cancelled,
            deadline,
            stream_start_deadline: metadata.request_start + self.policy.stream_start_delay,
            publish_token_pieces,
            publish_prompt_progress: publish_token_pieces && metadata.return_progress,
            backpressure_logged: AtomicBool::new(false),
            max_buffered_output_bytes: self.policy.max_buffered_output_bytes_per_request,
            output_budget: self.output_budget.clone(),
            output: Mutex::new(Output {
                admitted: false,
                progress: None,
                pieces: VecDeque::new(),
                buffered_bytes: 0,
                max_buffered_bytes: 0,
                result: None,
                failure: None,
                terminal: false,
            }),
            output_condition: Condvar::new(),
            prompt_tokens: prompt.len(),
            token_limit,
            live_cached_prompt_tokens: AtomicUsize::new(0),
            live_prefill_tokens: AtomicUsize::new(0),
            live_generated_tokens: AtomicUsize::new(0),
        });
        let capacity = self.pool.capacity();
        let result = GenerationResult {
            prompt_tokens: prompt.len(),
            client_id: client_id.clone(),
            token_metrics_recorded: true,
            configured_active_prefill_tokens: self.prefill_policy.decode_active_tokens,
            requested_logical_concurrency: capacity,
            execution_plan: if capacity == 1 {
                "serial-c1"
            } else {
                "serial-fallback"
            }
            .into(),
            ..GenerationResult::default()
        };
        let mut work = Box::new(Work {
            shared: shared.clone(),
            client_id,
            prompt,
            prompt_context: metadata.prompt_context,
            cache_prompt: metadata.cache_prompt,
            cache_prefix_tokens: metadata.cache_prefix_tokens,
            stop_at_eos: metadata.stop_at_eos,
            token_limit,
            sampling: sampling.clone(),
            request_start: metadata.request_start,
            max_output_bytes: self.policy.max_output_bytes_per_request,
            runner_request: None,
            result,
            previous_token: None,
            inter_token_total: Duration::ZERO,
            inter_token_samples: 0,
            decode_due: false,
            preview_token: None,
            advance_pending: false,
            preview_stops: false,
            generated_output_bytes: 0,
            stop_filter,
            last_decode_progress_tokens: 0,
            last_decode_progress_ms: 0.0,
            prefix_leader: None,
            prefix_position: 0,
            prefix_considered: false,
            prefix_wait_start: None,
            publication_considered: false,
            publication: None,
            counted_processing: false,
            session: None,
            sessions: self.shared.sessions.clone(),
            prefill_start: None,
        });
        let prompt_tokens = work.prompt.len();
        let client_id = work.client_id.clone();
        let policy = &self.policy;
        // The admission decision is logged after the queue lock is released,
        // so its stderr write never serialises other submissions.
        let decision = {
            let mut queue = self.shared.queue();
            if queue.stopping {
                return Err(Error::generation(
                    GenerationErrorCode::SchedulerStopping,
                    "text generation scheduler is stopping",
                ));
            }
            if self.device_lost() {
                return Err(device_lost_error());
            }
            if self.shared.failed.load(Ordering::Acquire) {
                return Err(scheduler_failed_error());
            }
            let count = queue.count;
            let client = queue
                .clients
                .iter()
                .position(|c| c.client_id == work.client_id);
            if count >= policy.max_pending_requests {
                Err(("queue_full", count, None))
            } else if let Some(index) = client.filter(|&index| {
                queue.clients[index].requests.len() >= policy.max_pending_requests_per_client
            }) {
                Err((
                    "client_quota",
                    count,
                    Some(queue.clients[index].requests.len()),
                ))
            } else {
                let client = client.unwrap_or_else(|| {
                    queue.clients.push_back(PendingClient {
                        client_id: work.client_id.clone(),
                        requests: VecDeque::new(),
                    });
                    queue.clients.len() - 1
                });
                let client_queued = queue.clients[client].requests.len() + 1;
                work.result.queue_depth_at_submit = count + 1;
                work.result.client_queue_depth_at_submit = client_queued;
                queue.clients[client].requests.push_back(work);
                queue.set_count(count + 1);
                Ok((count + 1, client_queued))
            }
        };
        let (queued, client_queued) = match decision {
            Ok(depths) => depths,
            Err((reason, queued, client_queued)) => {
                if logging::enabled(LogLevel::Debug) {
                    let (client_depth, limit) = match client_queued {
                        Some(depth) => (
                            format!(" client_queued={depth}"),
                            policy.max_pending_requests_per_client,
                        ),
                        None => (String::new(), policy.max_pending_requests),
                    };
                    logging::debug(
                        "scheduler",
                        &format!(
                            "event=admission_refused reason={reason} queued={queued}{client_depth} \
                             limit={limit} client_id={client_id}"
                        ),
                    );
                }
                return Err(if client_queued.is_some() {
                    Error::generation(
                        GenerationErrorCode::ClientQueueFull,
                        "text generation client pending queue is full",
                    )
                } else {
                    Error::generation(
                        GenerationErrorCode::QueueFull,
                        "text generation pending queue is full",
                    )
                });
            }
        };
        self.shared.queue_condition.notify_one();
        if logging::enabled(LogLevel::Debug) {
            logging::debug(
                "scheduler",
                &format!(
                    "event=admitted request={} client_id={client_id} queued={queued} \
                     client_queued={client_queued} prompt_tokens={prompt_tokens} max_tokens={}",
                    shared.id, shared.token_limit
                ),
            );
        }
        Ok(SchedulerRequest {
            shared,
            waited: false,
        })
    }

    /// `submit` with greedy sampling at `temperature` and default metadata.
    pub fn submit_simple(
        &self,
        prompt: Vec<Token>,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<SchedulerRequest> {
        self.submit(
            prompt,
            max_tokens,
            &SamplingConfig::with_temperature(temperature),
            None,
            false,
            RequestMetadata::default(),
        )
    }
}

impl<R: TextModelRunner> Drop for TextGenerationScheduler<R> {
    fn drop(&mut self) {
        self.shared.queue().stopping = true;
        self.shared.stop.store(true, Ordering::Release);
        self.shared.queue_condition.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        if let Some(monitor) = self.stall_monitor.take() {
            monitor.thread().unpark();
            let _ = monitor.join();
        }
    }
}
