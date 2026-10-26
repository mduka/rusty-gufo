//! The single-owner scheduler for opaque runner states
//! (`TextGenerationScheduler`, `src/cli/serve/text_generation_scheduler.cpp`).
//!
//! Submitters never execute model code. One scheduler thread owns
//! admission, runner leases, prefill and decode work units, cancellation
//! and reclamation. Pending requests queue per client and are admitted in
//! round-robin order; prefill runs one chunk at a time, bounded while any
//! request decodes, and every due decoder runs before further prefill.
//! Decode is batched across requests at the smallest plan that covers them.
//! Output pieces reach the waiting thread through bounded per-request
//! queues. A panic on the scheduler thread fails every request, resident
//! and queued, and the scheduler then refuses work for good
//! ([`GenerationErrorCode::SchedulerFailed`]).

use std::any::Any;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    CancellationCheck, DEVICE_LOST_MESSAGE, DeviceStatus, Error, GenerationErrorCode,
    PromptContext, Result, SCHEDULER_FAILED_MESSAGE, TextDecodeSelection, TextDecodeStep,
    TextExecutionPlanKind, TextModelRunner, Token,
};

use crate::backend::{
    FinishReason, GenerationResult, PromptProgress, SessionState, StreamCallbacks, TokenCallback,
};
use crate::continuation_cache::InputIdentity;
use crate::gpu_share::GpuShare;
use crate::logging::{self, LogLevel};
use crate::metrics;
use crate::pool::{AcquireOptions, PoolRequest, SHARED_PREFIX_MIN_TOKENS, TextRunnerPool};
use crate::publication_wait::{PendingPublication, PublicationWaitEstimate};
use crate::stop_sequences::{MAX_STOP_SEQUENCES, StopSequenceFilter, validate_stop_sequences};

mod admission;
mod decode;
mod prefill;
mod request;
mod submit;
mod worker;

pub use request::*;
use worker::*;

pub const DEFAULT_DECODE_ACTIVE_PREFILL_TOKENS: usize = 512;
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 1024 * 1024;
pub const DEFAULT_MAX_BUFFERED_OUTPUT_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_BUFFERED_OUTPUT_BYTES_TOTAL: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextPrefillPolicy {
    /// The prefill chunk while any request decodes (`--prefill-chunk`).
    pub decode_active_tokens: usize,
}

impl Default for TextPrefillPolicy {
    fn default() -> Self {
        Self {
            decode_active_tokens: DEFAULT_DECODE_ACTIVE_PREFILL_TOKENS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextSchedulerPolicy {
    pub max_pending_requests: usize,
    pub max_pending_requests_per_client: usize,
    pub max_output_bytes_per_request: usize,
    pub max_buffered_output_bytes_per_request: usize,
    pub max_buffered_output_bytes_total: usize,
    /// Zero disables the deadline.
    pub request_timeout: Duration,
    /// Logs each prefill chunk and every 50 decoded tokens.
    pub log_progress: bool,
    /// How often an idle scheduler polls the device; inference never does.
    pub device_probe_interval: Duration,
    /// How long a queued stream waits before it reports its start anyway,
    /// so transports can keep the connection alive.
    pub stream_start_delay: Duration,
    /// A scheduler that has work but starts no work unit and completes no
    /// pending capture for this long reports a lost device; zero disables
    /// the watchdog.
    pub stall_timeout: Duration,
    /// The arbiter's GPU grant (`--arbiter`); `None` runs unthrottled.
    pub gpu_share: Option<Arc<GpuShare>>,
}

/// Two policies are equal with the same limits and the same grant (by
/// identity).
impl PartialEq for TextSchedulerPolicy {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            max_pending_requests,
            max_pending_requests_per_client,
            max_output_bytes_per_request,
            max_buffered_output_bytes_per_request,
            max_buffered_output_bytes_total,
            request_timeout,
            log_progress,
            device_probe_interval,
            stream_start_delay,
            stall_timeout,
            gpu_share,
        } = self;
        *max_pending_requests == other.max_pending_requests
            && *max_pending_requests_per_client == other.max_pending_requests_per_client
            && *max_output_bytes_per_request == other.max_output_bytes_per_request
            && *max_buffered_output_bytes_per_request == other.max_buffered_output_bytes_per_request
            && *max_buffered_output_bytes_total == other.max_buffered_output_bytes_total
            && *request_timeout == other.request_timeout
            && *log_progress == other.log_progress
            && *device_probe_interval == other.device_probe_interval
            && *stream_start_delay == other.stream_start_delay
            && *stall_timeout == other.stall_timeout
            && match (gpu_share, &other.gpu_share) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for TextSchedulerPolicy {}

impl Default for TextSchedulerPolicy {
    fn default() -> Self {
        Self {
            max_pending_requests: 16,
            max_pending_requests_per_client: 4,
            max_output_bytes_per_request: DEFAULT_MAX_OUTPUT_BYTES,
            max_buffered_output_bytes_per_request: DEFAULT_MAX_BUFFERED_OUTPUT_BYTES,
            max_buffered_output_bytes_total: DEFAULT_MAX_BUFFERED_OUTPUT_BYTES_TOTAL,
            request_timeout: Duration::ZERO,
            log_progress: false,
            device_probe_interval: Duration::from_secs(5),
            stream_start_delay: Duration::from_secs(5),
            stall_timeout: Duration::ZERO,
            gpu_share: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TextRequestPhase {
    Queued,
    Admitted,
    Prefilling,
    DecodeReady,
    Decoding,
    Terminal,
}

impl TextRequestPhase {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Queued,
            1 => Self::Admitted,
            2 => Self::Prefilling,
            3 => Self::DecodeReady,
            4 => Self::Decoding,
            _ => Self::Terminal,
        }
    }
}

/// Per-request scheduling inputs besides prompt and sampling.
pub struct RequestMetadata<C> {
    pub client_id: String,
    pub deadline: Option<Instant>,
    pub request_start: Instant,
    pub prompt_context: Option<Arc<C>>,
    pub cache_prompt: bool,
    pub cache_prefix_tokens: usize,
    /// Output sequences that end the request; they require exact
    /// incremental text.
    pub stop_sequences: Vec<String>,
    /// False (`ignore_eos`) generates through the model's stop tokens up to
    /// the token limit.
    pub stop_at_eos: bool,
    /// Publishes prompt progress to a streaming waiter.
    pub return_progress: bool,
}

impl<C> Default for RequestMetadata<C> {
    fn default() -> Self {
        Self {
            client_id: "anonymous".into(),
            deadline: None,
            request_start: Instant::now(),
            prompt_context: None,
            cache_prompt: true,
            cache_prefix_tokens: 0,
            stop_sequences: Vec::new(),
            stop_at_eos: true,
            return_progress: false,
        }
    }
}

/// The queue cost of a streamed piece: a token without text still takes a
/// slot, so it costs one byte, or a slow consumer could let any number of
/// them pile up.
fn queued_piece_cost(piece: &[u8]) -> usize {
    piece.len().max(1)
}

/// The server-wide bound on output pieces not yet taken by their waiters.
struct OutputBudget {
    limit: usize,
    buffered_bytes: AtomicUsize,
    max_buffered_bytes: AtomicUsize,
}

struct Output {
    /// Set at a stream's admission; its waiter reports the start once.
    admitted: bool,
    /// The newest unread prompt progress; a slow waiter holds at most one.
    progress: Option<PromptProgress>,
    pieces: VecDeque<Vec<u8>>,
    buffered_bytes: usize,
    max_buffered_bytes: usize,
    result: Option<GenerationResult>,
    failure: Option<Error>,
    terminal: bool,
}

/// The part of a request shared by its waiter and the scheduler.
struct RequestShared {
    id: u64,
    cancellation_requested: AtomicBool,
    phase: AtomicU8,
    external_cancellation: Option<CancellationCheck>,
    deadline: Option<Instant>,
    /// When a still-queued stream reports its start.
    stream_start_deadline: Instant,
    publish_token_pieces: bool,
    publish_prompt_progress: bool,
    backpressure_logged: AtomicBool,
    max_buffered_output_bytes: usize,
    output_budget: Arc<OutputBudget>,
    output: Mutex<Output>,
    output_condition: Condvar,
    /// `/slots` progress; the scheduler thread updates the counters while
    /// the session table holds the request.
    prompt_tokens: usize,
    token_limit: usize,
    live_cached_prompt_tokens: AtomicUsize,
    live_prefill_tokens: AtomicUsize,
    live_generated_tokens: AtomicUsize,
}

/// The requests holding each execution session, for `/slots`. No other lock
/// is taken while it is held.
type SessionTable = Mutex<Vec<Option<Arc<RequestShared>>>>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl Drop for RequestShared {
    /// A completed stream may be dropped without its output ever being read;
    /// its last owner returns the remaining charge.
    fn drop(&mut self) {
        let output = self.output.get_mut().unwrap_or_else(|e| e.into_inner());
        self.output_budget.release(output.buffered_bytes);
        output.buffered_bytes = 0;
    }
}

/// A request parked for a pending disk checkpoint.
struct PublicationWait {
    checkpoint_id: u64,
    pending_tokens: usize,
    /// Only ever shrinks, as the cost estimate and the saving fall.
    limit: Duration,
    start: Instant,
}

/// The scheduler-owned part of a request.
struct Work<R: TextModelRunner> {
    shared: Arc<RequestShared>,
    client_id: String,
    prompt: Vec<Token>,
    prompt_context: Option<Arc<R::PromptContext>>,
    cache_prompt: bool,
    cache_prefix_tokens: usize,
    stop_at_eos: bool,
    token_limit: usize,
    sampling: SamplingConfig,
    request_start: Instant,
    max_output_bytes: usize,
    runner_request: Option<PoolRequest<R>>,
    result: GenerationResult,
    previous_token: Option<Instant>,
    inter_token_total: Duration,
    inter_token_samples: usize,
    decode_due: bool,
    preview_token: Option<Token>,
    advance_pending: bool,
    preview_stops: bool,
    generated_output_bytes: usize,
    stop_filter: StopSequenceFilter,
    last_decode_progress_tokens: usize,
    last_decode_progress_ms: f64,
    /// Admission waits at most once for a resident request to publish the
    /// prompt prefix both share; only the scheduler thread touches these.
    prefix_leader: Option<Arc<RequestShared>>,
    prefix_position: usize,
    prefix_considered: bool,
    prefix_wait_start: Option<Instant>,
    /// Admission waits at most once for a checkpoint still being written to
    /// disk that saves more than its prefill costs.
    publication_considered: bool,
    publication: Option<PublicationWait>,
    /// Admitted, holding `session` in the table until terminal; only the
    /// scheduler thread changes these.
    counted_processing: bool,
    session: Option<usize>,
    sessions: Arc<SessionTable>,
    /// When prompt processing started, for published progress.
    prefill_start: Option<Instant>,
}

impl<R: TextModelRunner> Work<R> {
    fn is_terminal(&self) -> bool {
        self.shared.is_terminal()
    }

    fn runner(&mut self) -> Result<&mut PoolRequest<R>> {
        self.runner_request
            .as_mut()
            .filter(|r| r.is_active())
            .ok_or_else(|| Error::logic("text runner request is empty"))
    }

    fn snapshot_pending(&self) -> bool {
        self.runner_request
            .as_ref()
            .is_some_and(|r| r.snapshot_pending())
    }

    fn prefill_complete(&self) -> bool {
        self.runner_request
            .as_ref()
            .is_some_and(|r| r.prefill_complete())
    }

    /// Prompt tokens still to prefill.
    fn prefill_remaining(&self) -> usize {
        self.runner_request
            .as_ref()
            .map_or(0, |r| r.prompt_tokens() - r.prefill_position())
    }

    /// Lets a streaming waiter commit its response before prompt processing.
    fn publish_admission(&self) {
        if !self.shared.publish_token_pieces {
            return;
        }
        self.shared.output().admitted = true;
        self.shared.output_condition.notify_one();
    }

    /// Replaces unread prompt progress with the current one.
    fn publish_prompt_progress(&mut self) {
        if !self.shared.publish_prompt_progress {
            return;
        }
        let start = *self.prefill_start.get_or_insert_with(Instant::now);
        let total = self.result.prompt_tokens;
        let cache = self.result.cached_prompt_tokens.min(total);
        let progress = PromptProgress {
            total,
            cache,
            processed: (cache + self.result.prefill_tokens).min(total),
            time_ms: start.elapsed().as_millis() as i64,
        };
        self.shared.output().progress = Some(progress);
        self.shared.output_condition.notify_one();
    }

    /// Makes the request terminal and wakes its waiter.
    fn publish_terminal(&mut self, failure: Option<Error>, discard_pending_output: bool) {
        // Retired before the waiter wakes, so its result never sees itself
        // still processing.
        if std::mem::take(&mut self.counted_processing) {
            metrics::add_requests_processing(-1);
        }
        if let Some(session) = self.session.take() {
            lock(&self.sessions)[session] = None;
        }
        {
            let mut output = self.shared.output();
            if discard_pending_output {
                self.shared.release_buffered(&mut output);
            }
            let mut result = std::mem::take(&mut self.result);
            result.max_buffered_output_bytes = output.max_buffered_bytes;
            output.result = Some(result);
            output.failure = failure;
            output.terminal = true;
            self.shared.set_phase(TextRequestPhase::Terminal);
        }
        self.shared.output_condition.notify_all();
    }
}

/// A request is always terminal once the scheduler lets go of it: work that
/// a panic unwinds, or that a failed scheduler drains, fails instead of
/// leaving its waiter blocked for good.
impl<R: TextModelRunner> Drop for Work<R> {
    fn drop(&mut self) {
        if !self.is_terminal() {
            self.publish_terminal(Some(scheduler_failed_error()), true);
        }
    }
}

struct PendingClient<R: TextModelRunner> {
    client_id: String,
    requests: VecDeque<Box<Work<R>>>,
}

struct Queue<R: TextModelRunner> {
    clients: VecDeque<PendingClient<R>>,
    count: usize,
    stopping: bool,
}

impl<R: TextModelRunner> Queue<R> {
    /// Keeps the process-wide deferred gauge in step with `count`.
    fn set_count(&mut self, count: usize) {
        metrics::add_requests_deferred(count as i64 - self.count as i64);
        self.count = count;
    }
}

struct SchedulerShared<R: TextModelRunner> {
    queue: Mutex<Queue<R>>,
    sessions: Arc<SessionTable>,
    queue_condition: Condvar,
    stop: Arc<AtomicBool>,
    /// Set once a failed work unit showed the device lost; permanent.
    device_lost: AtomicBool,
    /// Set, under the queue lock, once the scheduler thread panicked;
    /// permanent.
    failed: AtomicBool,
    /// Written by the scheduler thread, read by the stall monitor:
    /// `progress` advances when a request is admitted, a work unit starts, a
    /// pending capture completes or an idle wait returns; `loop_working` is
    /// false only inside that idle wait. A loop that only polls a capture
    /// which never completes makes no progress.
    progress: AtomicU64,
    loop_working: AtomicBool,
}

/// How long a failure's device probe may stay queued behind other kernels
/// before the device counts as usable.
const DEVICE_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

fn device_lost_error() -> Error {
    Error::generation(GenerationErrorCode::DeviceLost, DEVICE_LOST_MESSAGE)
}

fn scheduler_failed_error() -> Error {
    Error::generation(
        GenerationErrorCode::SchedulerFailed,
        SCHEDULER_FAILED_MESSAGE,
    )
}

/// The message of a panic payload, as `std` prints it.
fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

impl<R: TextModelRunner> SchedulerShared<R> {
    fn mark_device_lost(&self, reason: &str) {
        if self.device_lost.swap(true, Ordering::AcqRel) {
            return;
        }
        metrics::add_device_lost();
        logging::error(
            "scheduler",
            &format!("event=device_lost remedy=restart reason={reason}"),
        );
    }

    /// The stall watchdog (`MonitorStalls`): a device call that never
    /// returns blocks the scheduler thread, so nothing on that thread can
    /// report it. This thread only reads atomics; it never touches the device
    /// or the queues. Stalled time is counted in wake-ups, not read from the
    /// clock, so a process paused as a whole (SIGSTOP, a cgroup freeze) is
    /// credited one interval for the pause however long it lasted.
    fn monitor_stalls(&self, timeout: Duration) {
        let interval = (timeout / 4).clamp(Duration::from_millis(1), Duration::from_secs(60));
        let mut seen = self.progress.load(Ordering::Relaxed);
        let mut stalled = Duration::ZERO;
        loop {
            std::thread::park_timeout(interval);
            if self.stop.load(Ordering::Acquire) || self.device_lost.load(Ordering::Acquire) {
                return;
            }
            let now = self.progress.load(Ordering::Relaxed);
            if now != seen || !self.loop_working.load(Ordering::Relaxed) {
                seen = now;
                stalled = Duration::ZERO;
                continue;
            }
            stalled += interval;
            if stalled >= timeout {
                self.mark_device_lost(&format!(
                    "scheduler made no progress for {} ms",
                    timeout.as_millis()
                ));
                return;
            }
        }
    }

    fn queue(&self) -> MutexGuard<'_, Queue<R>> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// After the scheduler thread panicked: its own work has failed while
    /// unwinding ([`Work`]'s `Drop`); the queued work fails here, and later
    /// submissions are refused. The thread's model state is not reused.
    fn fail_after_panic(&self, payload: &(dyn Any + Send)) {
        logging::error(
            "scheduler",
            &format!(
                "event=scheduler_failed remedy=restart reason={:?}",
                panic_message(payload)
            ),
        );
        let queued: Vec<Box<Work<R>>> = {
            let mut queue = self.queue();
            self.failed.store(true, Ordering::Release);
            queue.set_count(0);
            queue
                .clients
                .drain(..)
                .flat_map(|client| client.requests)
                .collect()
        };
        drop(queued);
    }
}

pub struct TextGenerationScheduler<R: TextModelRunner> {
    shared: Arc<SchedulerShared<R>>,
    pool: Arc<TextRunnerPool<R>>,
    prefill_policy: TextPrefillPolicy,
    policy: TextSchedulerPolicy,
    output_budget: Arc<OutputBudget>,
    next_request_id: AtomicU64,
    worker: Option<JoinHandle<()>>,
    stall_monitor: Option<JoinHandle<()>>,
}

type Batch<R> = Vec<Box<Work<R>>>;

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

impl<R: TextModelRunner> Worker<R> {}
