//! The text generation boundary of the HTTP transport and its results
//! (`src/cli/serve/text_generation_backend.hpp`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub use gufo_core::json_constraint::ToolFormat;
use gufo_core::sampling::{
    SamplingConfig, SamplingOverrides, TextModelPreset, resolve_text_sampling,
};
pub use gufo_core::text::InitialOutputState;
use gufo_core::text::{
    CancellationCheck, ChatRequest, Error, GenerationErrorCode, ReasoningOptions, Result, Token,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FinishReason {
    #[default]
    Stop,
    /// A requested stop sequence matched the output.
    StopSequence,
    Length,
    Cancelled,
}

/// Server defaults applied to requests that omit them.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingDefaults {
    /// Zero generates until EOS or the remaining context is exhausted.
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    /// The model's recommended sampling, under the `supplied` fields of
    /// `sampling`.
    pub model: TextModelPreset,
    pub supplied: SamplingOverrides,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            sampling: SamplingConfig::default(),
            model: TextModelPreset::Unspecified,
            supplied: SamplingOverrides::ALL,
        }
    }
}

impl SamplingDefaults {
    /// The request defaults for the effective thinking mode.
    pub fn resolve(&self, thinking: Option<bool>) -> SamplingConfig {
        resolve_text_sampling(self.model, thinking, &self.sampling, &self.supplied)
    }
}

/// The generated text and aggregate request metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResult {
    /// Decoded bytes; the transport repairs incomplete UTF-8.
    pub text: Vec<u8>,
    pub tokens: Vec<Token>,
    pub prompt_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub cache_restore_bytes: usize,
    pub cache_snapshot_bytes: usize,
    /// Bytes admitted to background persistence; the cache logs completion.
    pub cache_disk_queued_bytes: usize,
    pub cache_shared_bytes: usize,
    /// Shared-prefix snapshots queued while prefilling this request.
    pub cache_shared_prefix_snapshots: usize,
    pub cache_shared_prefix_bytes: usize,
    pub completion_tokens: usize,
    /// Generated reasoning tokens, without the closing delimiter.
    pub reasoning_tokens: usize,
    /// Speculative verification rounds actually executed.
    pub draft_rounds: usize,
    pub draft_tokens: usize,
    pub draft_accepted_tokens: usize,
    /// Of `draft_tokens`: proposals copied from the context (prompt
    /// lookup).
    pub lookup_tokens: usize,
    pub lookup_accepted_tokens: usize,
    pub prefill_tokens: usize,
    pub prefill_chunks: usize,
    pub active_decode_prefill_chunks: usize,
    pub max_prefill_chunk_tokens: usize,
    pub max_consecutive_active_prefill_chunks: usize,
    pub configured_active_prefill_tokens: usize,
    pub queue_depth_at_submit: usize,
    pub client_queue_depth_at_submit: usize,
    pub resident_requests_at_admission: usize,
    pub requested_logical_concurrency: usize,
    pub physical_execution_width: usize,
    pub max_buffered_output_bytes: usize,
    pub queue_ms: f64,
    /// The part of `queue_ms` spent waiting for a concurrent request to
    /// publish the prompt prefix both share.
    pub shared_prefix_wait_ms: f64,
    pub cache_restore_ms: f64,
    pub cache_snapshot_ms: f64,
    pub cache_disk_enqueue_ms: f64,
    pub cache_shared_prefix_ms: f64,
    pub ttft_ms: f64,
    pub mean_inter_token_ms: f64,
    pub max_inter_token_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub client_id: String,
    pub execution_plan: String,
    pub prefill_fallback_reason: String,
    pub cache_miss_reason: String,
    pub cache_common_prefix_tokens: usize,
    pub cache_checkpoint_tokens: usize,
    pub finish_reason: FinishReason,
    /// The matched stop sequence of a `StopSequence` finish.
    pub stop_sequence: String,
    pub incremental_prefill_supported: bool,
    pub cache_hit: bool,
    pub cache_disk_hit: bool,
    pub cancelled: bool,
    /// Internal: the scheduler recorded this request's `/metrics` counters.
    pub token_metrics_recorded: bool,
}

impl Default for GenerationResult {
    fn default() -> Self {
        Self {
            text: Vec::new(),
            tokens: Vec::new(),
            prompt_tokens: 0,
            cached_prompt_tokens: 0,
            cache_restore_bytes: 0,
            cache_snapshot_bytes: 0,
            cache_disk_queued_bytes: 0,
            cache_shared_bytes: 0,
            cache_shared_prefix_snapshots: 0,
            cache_shared_prefix_bytes: 0,
            completion_tokens: 0,
            reasoning_tokens: 0,
            draft_rounds: 0,
            draft_tokens: 0,
            draft_accepted_tokens: 0,
            lookup_tokens: 0,
            lookup_accepted_tokens: 0,
            prefill_tokens: 0,
            prefill_chunks: 0,
            active_decode_prefill_chunks: 0,
            max_prefill_chunk_tokens: 0,
            max_consecutive_active_prefill_chunks: 0,
            configured_active_prefill_tokens: 0,
            queue_depth_at_submit: 0,
            client_queue_depth_at_submit: 0,
            resident_requests_at_admission: 0,
            requested_logical_concurrency: 1,
            physical_execution_width: 1,
            max_buffered_output_bytes: 0,
            queue_ms: 0.0,
            shared_prefix_wait_ms: 0.0,
            cache_restore_ms: 0.0,
            cache_snapshot_ms: 0.0,
            cache_disk_enqueue_ms: 0.0,
            cache_shared_prefix_ms: 0.0,
            ttft_ms: 0.0,
            mean_inter_token_ms: 0.0,
            max_inter_token_ms: 0.0,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            client_id: "anonymous".into(),
            execution_plan: "serial-c1".into(),
            prefill_fallback_reason: String::new(),
            cache_miss_reason: String::new(),
            cache_common_prefix_tokens: 0,
            cache_checkpoint_tokens: 0,
            finish_reason: FinishReason::Stop,
            stop_sequence: String::new(),
            incremental_prefill_supported: false,
            cache_hit: false,
            cache_disk_hit: false,
            cancelled: false,
            token_metrics_recorded: false,
        }
    }
}

/// One execution session for llama-server `/slots`. Idle sessions report
/// zero counts rather than the previous request's.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionState {
    pub processing: bool,
    pub speculative: bool,
    pub request_id: u64,
    pub prompt_tokens: usize,
    pub cached_prompt_tokens: usize,
    /// Prompt tokens prefilled so far, excluding cached tokens.
    pub processed_prompt_tokens: usize,
    pub generated_tokens: usize,
    pub remaining_tokens: usize,
}

/// llama-server `prompt_progress`; `processed` includes cached tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptProgress {
    pub total: usize,
    pub cache: usize,
    pub processed: usize,
    pub time_ms: i64,
}

/// Receives prompt progress before the first piece; returning false cancels
/// the request.
pub type ProgressCallback<'a> = &'a mut dyn FnMut(&PromptProgress) -> bool;

/// Receives each generated piece; returning false cancels the request.
pub type TokenCallback<'a> = &'a mut dyn FnMut(&[u8]) -> bool;

/// Called when a streaming request starts; returning false cancels it.
pub type StartCallback<'a> = &'a mut dyn FnMut() -> bool;

/// What a streaming waiter receives besides the result.
#[derive(Default)]
pub struct StreamCallbacks<'a> {
    pub on_token: Option<TokenCallback<'a>>,
    /// The latest prompt progress before the first piece, for requests
    /// started with `return_progress`.
    pub on_progress: Option<ProgressCallback<'a>>,
    /// At most once, before progress or output: the request started work or
    /// has queued for a bounded time. Requests without a queue may never
    /// start.
    pub on_start: Option<StartCallback<'a>>,
}

/// A reserved generation (`TextGenerationBackend::GenerationRequest`).
pub trait GenerationRequest: Send {
    /// Delivers output pieces to `on_token` (returning false cancels) and
    /// returns the result; callable once.
    fn wait(&mut self, on_token: Option<TokenCallback<'_>>) -> Result<GenerationResult>;

    /// `wait` that also reports a streaming request's start and prompt
    /// progress; requests without a scheduler report neither.
    fn wait_streaming(&mut self, callbacks: StreamCallbacks<'_>) -> Result<GenerationResult> {
        self.wait(callbacks.on_token)
    }

    fn cancel(&self);

    /// The tool-call format the request was admitted with, the JSON fallback
    /// for a schema included; known before `wait`. `None` without
    /// constrained tool calls, and from backends without this metadata,
    /// whose output is parsed for every dialect.
    fn tool_format(&self) -> Option<ToolFormat> {
        None
    }
}

/// The model-agnostic text generation boundary the HTTP transport uses
/// (`TextGenerationBackend`). Implementations own tokenization, templates,
/// continuation state and execution; the transport sees generated pieces
/// and aggregate metrics only.
pub trait TextGenerationBackend: Send + Sync + 'static {
    fn model_id(&self) -> String;
    fn ready(&self) -> bool;

    /// The session context of the loaded model; zero when unknown.
    fn max_context(&self) -> u32 {
        0
    }

    fn sampling_defaults(&self) -> SamplingDefaults {
        SamplingDefaults::default()
    }

    fn reasoning_defaults(&self) -> ReasoningOptions {
        ReasoningOptions::default()
    }

    fn initial_output_state(&self, request: &ChatRequest) -> InitialOutputState {
        let _ = request;
        InitialOutputState::Auto
    }

    /// Whether the loaded model accepts image inputs in chat requests.
    fn supports_images(&self) -> bool {
        false
    }

    /// The served model file, as llama-server's `/props` reports it; empty
    /// when unknown.
    fn model_path(&self) -> String {
        String::new()
    }

    /// The build as `gufo --version` prints it; empty when unknown.
    fn build_info(&self) -> String {
        String::new()
    }

    /// True once a failed generation showed that the execution device can no
    /// longer run work, for example after a GPU reset; permanent.
    fn device_lost(&self) -> bool {
        false
    }

    /// The permanent failure that keeps this backend from running work, if
    /// any: a lost device, or a scheduler whose thread panicked. Only a new
    /// process recovers ([`GenerationErrorCode::requires_restart`]).
    fn fatal_failure(&self) -> Option<GenerationErrorCode> {
        self.device_lost()
            .then_some(GenerationErrorCode::DeviceLost)
    }

    /// Live execution sessions; empty when the backend has no session pool.
    fn session_states(&self) -> Vec<SessionState> {
        Vec::new()
    }

    /// A raw prompt; `stop_sequences` end generation where they match the
    /// output and are excluded from it. Without `cache_prompt` the request
    /// skips the cache lookup (its result may still populate the cache).
    #[allow(clippy::too_many_arguments)]
    fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<&CancellationCheck>,
        on_token: Option<TokenCallback<'_>>,
        client_id: &str,
        stop_sequences: &[String],
        cache_prompt: bool,
    ) -> Result<GenerationResult>;

    fn chat(
        &self,
        request: &ChatRequest,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<&CancellationCheck>,
        on_token: Option<TokenCallback<'_>>,
    ) -> Result<GenerationResult>;

    /// Reserves admission before a streaming response commits its headers.
    /// Backends with an asynchronous scheduler return the request; `None`
    /// (the default) defers to a blocking `chat` on wait, see [`start_chat`].
    fn start_chat(
        &self,
        request: &ChatRequest,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<CancellationCheck>,
        stream_output: bool,
    ) -> Result<Option<Box<dyn GenerationRequest>>> {
        let _ = (request, max_tokens, sampling, is_cancelled, stream_output);
        Ok(None)
    }

    /// Reserves admission for a raw prompt before a streaming response
    /// commits its headers. `ignore_eos` generates exactly `max_tokens`
    /// (bounded by the context); `None` (the default) defers to a blocking
    /// `complete` on wait, see [`start_complete`].
    #[allow(clippy::too_many_arguments)]
    fn start_complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<CancellationCheck>,
        stream_output: bool,
        ignore_eos: bool,
        client_id: &str,
        stop_sequences: &[String],
        return_progress: bool,
        cache_prompt: bool,
    ) -> Result<Option<Box<dyn GenerationRequest>>> {
        let _ = (
            prompt,
            max_tokens,
            sampling,
            is_cancelled,
            stream_output,
            client_id,
            stop_sequences,
            return_progress,
            cache_prompt,
        );
        if ignore_eos {
            return Err(Error::invalid("this backend does not support ignore_eos"));
        }
        Ok(None)
    }

    fn count_tokens(&self, text: &str) -> usize;

    /// The prompt tokens a chat request would use after template rendering,
    /// without generating (`count_chat_tokens`); `None` when this backend
    /// cannot tell.
    fn count_chat_tokens(&self, _request: &ChatRequest) -> Result<Option<usize>> {
        Ok(None)
    }
}

/// `backend.start_complete(...)`, falling back to a request that runs
/// `complete` on wait when the backend has no scheduler.
#[allow(clippy::too_many_arguments)]
pub fn start_complete(
    backend: &Arc<dyn TextGenerationBackend>,
    prompt: &str,
    max_tokens: usize,
    sampling: &SamplingConfig,
    is_cancelled: Option<CancellationCheck>,
    stream_output: bool,
    ignore_eos: bool,
    client_id: &str,
    stop_sequences: &[String],
    return_progress: bool,
    cache_prompt: bool,
) -> Result<Box<dyn GenerationRequest>> {
    if let Some(started) = backend.start_complete(
        prompt,
        max_tokens,
        sampling,
        is_cancelled.clone(),
        stream_output,
        ignore_eos,
        client_id,
        stop_sequences,
        return_progress,
        cache_prompt,
    )? {
        return Ok(started);
    }
    Ok(Box::new(DeferredCompletion {
        backend: backend.clone(),
        prompt: prompt.to_owned(),
        max_tokens,
        sampling: sampling.clone(),
        client_id: client_id.to_owned(),
        stop_sequences: stop_sequences.to_vec(),
        cache_prompt,
        external_cancellation: is_cancelled,
        waited: false,
        cancelled: Arc::new(AtomicBool::new(false)),
    }))
}

struct DeferredCompletion {
    backend: Arc<dyn TextGenerationBackend>,
    prompt: String,
    max_tokens: usize,
    sampling: SamplingConfig,
    client_id: String,
    stop_sequences: Vec<String>,
    cache_prompt: bool,
    external_cancellation: Option<CancellationCheck>,
    waited: bool,
    cancelled: Arc<AtomicBool>,
}

impl GenerationRequest for DeferredCompletion {
    fn wait(&mut self, on_token: Option<TokenCallback<'_>>) -> Result<GenerationResult> {
        if std::mem::replace(&mut self.waited, true) {
            return Err(Error::logic("generation request was already consumed"));
        }
        let cancelled = self.cancelled.clone();
        let external = self.external_cancellation.clone();
        let check: CancellationCheck = Arc::new(move || {
            cancelled.load(Ordering::Acquire) || external.as_ref().is_some_and(|c| c())
        });
        self.backend.complete(
            &self.prompt,
            self.max_tokens,
            &self.sampling,
            Some(&check),
            on_token,
            &self.client_id,
            &self.stop_sequences,
            self.cache_prompt,
        )
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

/// `backend.start_chat(...)`, falling back to a request that runs `chat` on
/// wait when the backend has no scheduler.
pub fn start_chat(
    backend: &Arc<dyn TextGenerationBackend>,
    request: &ChatRequest,
    max_tokens: usize,
    sampling: &SamplingConfig,
    is_cancelled: Option<CancellationCheck>,
    stream_output: bool,
) -> Result<Box<dyn GenerationRequest>> {
    if let Some(started) = backend.start_chat(
        request,
        max_tokens,
        sampling,
        is_cancelled.clone(),
        stream_output,
    )? {
        return Ok(started);
    }
    Ok(Box::new(DeferredGenerationRequest {
        backend: backend.clone(),
        request: request.clone(),
        max_tokens,
        sampling: sampling.clone(),
        external_cancellation: is_cancelled,
        waited: false,
        cancelled: Arc::new(AtomicBool::new(false)),
    }))
}

struct DeferredGenerationRequest {
    backend: Arc<dyn TextGenerationBackend>,
    request: ChatRequest,
    max_tokens: usize,
    sampling: SamplingConfig,
    external_cancellation: Option<CancellationCheck>,
    waited: bool,
    cancelled: Arc<AtomicBool>,
}

impl GenerationRequest for DeferredGenerationRequest {
    fn wait(&mut self, on_token: Option<TokenCallback<'_>>) -> Result<GenerationResult> {
        if std::mem::replace(&mut self.waited, true) {
            return Err(Error::logic("generation request was already consumed"));
        }
        let cancelled = self.cancelled.clone();
        let external = self.external_cancellation.clone();
        let check: CancellationCheck = Arc::new(move || {
            cancelled.load(Ordering::Acquire) || external.as_ref().is_some_and(|c| c())
        });
        self.backend.chat(
            &self.request,
            self.max_tokens,
            &self.sampling,
            Some(&check),
            on_token,
        )
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}
