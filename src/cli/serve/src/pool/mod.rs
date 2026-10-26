//! A bounded pool of opaque runner states with exact-prefix continuation
//! reuse (`TextRunnerPool`, `src/cli/serve/text_model_runner.cpp`).
//!
//! A request leases one state for its prompt, reusing the longest retained
//! prefix from RAM (live frontier or immutable snapshot) or disk, then drives
//! prefill and decode work units. On commit it publishes a snapshot at the
//! stable prompt boundary plus its live frontier; on cancellation it keeps
//! only completed work. Prompt snapshots are captured on a worker thread
//! while the request is frozen, so other requests keep running.

use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    CancellationCheck, Error, PromptContext, Result, Sampler, TextDecodeSelection, TextDecodeStep,
    TextExecutionPlan, TextExecutionPlanKind, TextModelRunner, TextPrefillStep, TextRunnerAdvance,
    TextRunnerCapabilities, TextRunnerDecode, TextRunnerDescriptor, TextRunnerResourceClaim,
    TextRunnerSnapshot, TextRunnerState, Token,
};

use crate::continuation_cache::{
    ContinuationCache, ContinuationLookup, InputIdentity, Lease, SnapshotEvent,
    SnapshotEventAction, SnapshotEventReason, SnapshotPurpose, SnapshotSupport,
};
use crate::disk_store::{
    CaptureReservation, ContinuationDiskEvent, ContinuationDiskEventAction,
    ContinuationDiskEventReason, ContinuationDiskStore, ContinuationDiskStoreOptions,
    DEFAULT_CAPACITY_BYTES,
};
use crate::logging;
use crate::metrics::format_general;
use crate::publication_wait::PendingPublication;

mod acquire;
mod batch;
mod events;
mod request;
mod snapshots;

use events::*;

const ADMISSION_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextRunnerDiskCacheOptions {
    pub directory: PathBuf,
    pub capacity_bytes: usize,
    /// Zero selects an eighth of the available host RAM, at most
    /// `capacity_bytes`.
    pub staging_capacity_bytes: usize,
    /// Shared prefixes shorter than this are cheaper to prefill than to
    /// restore.
    pub shared_prefix_min_tokens: usize,
    /// Bound on shared-prefix snapshots written while prefilling one request.
    pub shared_prefix_max_boundaries: usize,
    /// The least advance over a stored prefix worth persisting: a
    /// continuation moves a few hundred tokens per turn, and the tokens
    /// between two checkpoints are cheaper to prefill again than the
    /// rewrite. Zero disables the gate.
    pub min_checkpoint_step_tokens: usize,
}

impl Default for TextRunnerDiskCacheOptions {
    fn default() -> Self {
        Self {
            directory: PathBuf::new(),
            capacity_bytes: DEFAULT_CAPACITY_BYTES,
            staging_capacity_bytes: 0,
            shared_prefix_min_tokens: 128,
            shared_prefix_max_boundaries: 4,
            min_checkpoint_step_tokens: 2048,
        }
    }
}

/// The RAM snapshot cache (`TextRunnerRamCacheOptions`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TextRunnerRamCacheOptions {
    /// Zero selects min(the model's snapshot budget, 32 GiB) after session
    /// allocation; an explicit limit may exceed that budget up to the
    /// model's ceiling, which keeps 4 GiB of host RAM free.
    pub capacity_bytes: usize,
}

impl TextRunnerRamCacheOptions {
    pub const AUTOMATIC_MAX_BYTES: usize = 32 << 30;
    /// Checkpoint records, independent of the execution states.
    pub const MAX_ENTRIES: usize = 128;
}

/// What a finished request published.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CommitMetrics {
    pub snapshot_bytes: usize,
    pub snapshot_ms: f64,
    pub disk_queued_bytes: usize,
    pub disk_enqueue_ms: f64,
    /// Shared-prefix snapshots queued for persistence during prefill.
    pub shared_prefix_snapshots: usize,
    pub shared_prefix_bytes: usize,
    pub shared_prefix_failures: usize,
    pub shared_prefix_ms: f64,
}

/// Options of `TextRunnerPool::acquire`.
pub struct AcquireOptions<C> {
    pub sampling: SamplingConfig,
    pub is_cancelled: Option<CancellationCheck>,
    pub context: Option<Arc<C>>,
    /// False bypasses prompt reuse (`cache_prompt: false`); the finished
    /// state may still be retained.
    pub reuse_prompt: bool,
    /// The stable prompt boundary before mutable assistant framing; zero
    /// means the whole prompt.
    pub cache_prefix_tokens: usize,
    /// False (`ignore_eos`) generates through the model's stop tokens.
    pub stop_at_eos: bool,
}

impl<C> Default for AcquireOptions<C> {
    fn default() -> Self {
        Self {
            sampling: SamplingConfig::default(),
            is_cancelled: None,
            context: None,
            reuse_prompt: true,
            cache_prefix_tokens: 0,
            stop_at_eos: true,
        }
    }
}

struct Validated<R> {
    runner: Arc<R>,
    descriptor: TextRunnerDescriptor,
    resources: TextRunnerResourceClaim,
    plans: Vec<TextExecutionPlan>,
}

/// The retained-snapshot limits selected after session allocation: the
/// automatic budget and the most an explicit limit may claim.
#[derive(Default)]
struct SnapshotLimits {
    automatic_bytes: AtomicUsize,
    max_bytes: AtomicUsize,
}

fn per_state_reservation_bytes(resources: &TextRunnerResourceClaim) -> Result<Option<usize>> {
    match (
        resources.per_request_state_bytes,
        resources.temporary_scratch_bytes,
    ) {
        (Some(state), Some(scratch)) => state
            .checked_add(scratch)
            .map(Some)
            .ok_or_else(|| Error::invalid("text runner resource claim overflows")),
        _ => Ok(None),
    }
}

/// `std::timed_mutex` with cancellation polling: publishes a live
/// checkpoint before another admission can fall back to an older snapshot.
struct AdmissionGate {
    held: Mutex<bool>,
    released: Condvar,
}

struct AdmissionGuard<'a>(&'a AdmissionGate);

impl AdmissionGate {
    fn lock(&self, is_cancelled: Option<&CancellationCheck>) -> Option<AdmissionGuard<'_>> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while *held {
            let (guard, timeout) = self
                .released
                .wait_timeout(held, ADMISSION_POLL_INTERVAL)
                .unwrap_or_else(|e| e.into_inner());
            held = guard;
            if *held && timeout.timed_out() && is_cancelled.is_some_and(|c| c()) {
                return None;
            }
        }
        *held = true;
        Some(AdmissionGuard(self))
    }
}

impl Drop for AdmissionGuard<'_> {
    fn drop(&mut self) {
        *self.0.held.lock().unwrap_or_else(|e| e.into_inner()) = false;
        self.0.released.notify_one();
    }
}

type CaptureResult<R> = (
    <R as TextModelRunner>::State,
    Result<<R as TextModelRunner>::Snapshot>,
);

/// A boundary capture's reservations, published when its copy is joined.
struct BoundaryCapture {
    position: usize,
    /// Bytes reserved for the RAM history checkpoint; zero for none.
    reserved: usize,
    /// Disk staging reserved for the shared prefix.
    persistence: Option<CaptureReservation>,
    history: bool,
    shared: bool,
    started: Instant,
}

/// A prompt capture: running on a worker that owns the lease's state until
/// joined, or already taken on the request's own thread.
enum Capture<R: TextModelRunner> {
    Running(JoinHandle<CaptureResult<R>>),
    Taken(Result<R::Snapshot>),
}

impl<R: TextModelRunner> Capture<R> {
    fn pending(&self) -> bool {
        matches!(self, Self::Running(handle) if !handle.is_finished())
    }
}

pub struct TextRunnerPool<R: TextModelRunner> {
    validated: Arc<Validated<R>>,
    cache: ContinuationCache<R::State, R::Snapshot>,
    admission: AdmissionGate,
    disk_store: Option<Arc<ContinuationDiskStore>>,
    shared_prefix_min_tokens: usize,
    shared_prefix_max_boundaries: usize,
}

/// One request's lease on a runner state.
pub struct PoolRequest<R: TextModelRunner> {
    inner: Option<RequestInner<R>>,
}

/// Intermediate prefill checkpoints a request keeps at most, on a grid of
/// [`CHECKPOINT_INTERVAL`] tokens.
const INTERMEDIATE_CHECKPOINTS: usize = 4;
const CHECKPOINT_INTERVAL: usize = 2048;
/// No grid checkpoint within this many tokens of the prompt end.
const CHECKPOINT_TAIL: usize = 128;
/// A planned checkpoint this close to a shared position is cheaper to use
/// than capturing another: followers prefill the gap themselves.
pub const SHARED_CHECKPOINT_SLACK: usize = 64;
/// A shared prefix must add at least this many tokens to what a request can
/// already restore before it waits for a peer or captures an extra
/// checkpoint for later requests; shorter gaps cost less to prefill again.
pub const SHARED_PREFIX_MIN_TOKENS: usize = 512;

struct RequestInner<R: TextModelRunner> {
    runner: Arc<R>,
    capabilities: TextRunnerCapabilities,
    disk_store: Option<Arc<ContinuationDiskStore>>,
    lease: Lease<R::State, R::Snapshot>,
    prompt: Vec<Token>,
    snapshot_tokens: Vec<Token>,
    /// Ascending prefill positions to persist, all inside (cached, prompt).
    boundaries: Vec<usize>,
    /// Bounded intermediate positions retained in RAM before the state
    /// advances, so an edit of earlier history restores near it.
    checkpoints: Vec<usize>,
    /// The boundary the outstanding capture belongs to, if not the prompt.
    boundary: Option<BoundaryCapture>,
    /// Checkpoints shared with other requests, retained like continuations.
    shared_checkpoints: Vec<usize>,
    /// Several execution states: boundary copies overlap peers' work.
    overlap_snapshots: bool,
    generated: Vec<Token>,
    prefill_offset: usize,
    decode_ready: bool,
    /// A failed model call may have partially mutated device state; never
    /// publish that state as a live continuation.
    state_reusable: bool,
    stopped: bool,
    pending_selection: Option<TextDecodeSelection>,
    sampler: R::Sampler,
    context: Option<Arc<R::PromptContext>>,
    prompt_snapshot_attempted: bool,
    prompt_snapshot: Option<Arc<R::Snapshot>>,
    snapshot_metrics: CommitMetrics,
    snapshot_bytes: usize,
    retain_snapshot: bool,
    snapshot_start: Instant,
    disk_capture: Option<CaptureReservation>,
    capture: Option<Capture<R>>,
    /// Chat prompts checkpoint the stable prefix before the assistant
    /// suffix as a branching fallback, then the complete prompt.
    retain_fallback: bool,
    /// Where that fallback was taken, once prefill has passed it.
    fallback_position: Option<usize>,
    /// This turn's stable boundary when the reused frontier sits before it,
    /// so it is checkpointed after the frozen fallback instead of being
    /// skipped; zero once taken, or when the two coincide.
    stable_prefix_position: usize,
}

fn empty_request() -> Error {
    Error::logic("text runner request is empty")
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

impl<R: TextModelRunner> TextRunnerPool<R> {
    pub fn new(
        runner: Arc<R>,
        state_count: usize,
        disk_cache: Option<TextRunnerDiskCacheOptions>,
    ) -> Result<Self> {
        Self::with_ram_cache(
            runner,
            state_count,
            disk_cache,
            TextRunnerRamCacheOptions::default(),
        )
    }

    pub fn with_ram_cache(
        runner: Arc<R>,
        state_count: usize,
        disk_cache: Option<TextRunnerDiskCacheOptions>,
        ram_cache: TextRunnerRamCacheOptions,
    ) -> Result<Self> {
        let validated = Arc::new(validate_runner(runner, state_count)?);
        let factory_validated = validated.clone();
        let factory = move || {
            let state = factory_validated.runner.create_state()?;
            reconcile_state_bytes(&factory_validated.resources, &state)?;
            Ok(state)
        };
        let capabilities = validated.descriptor.capabilities;
        let limits = Arc::new(SnapshotLimits::default());
        let support = if capabilities.snapshot && capabilities.fork {
            let restore_validated = validated.clone();
            let capacity_runner = validated.runner.clone();
            let capacity_limits = limits.clone();
            SnapshotSupport {
                restore: Some(Box::new(
                    move |state: &mut R::State, snapshot: &R::Snapshot| {
                        restore_validated.runner.restore_or_fork(state, snapshot)?;
                        reconcile_state_bytes(&restore_validated.resources, state)
                    },
                )),
                capacity_bytes: Some(Box::new(move || {
                    let claim = capacity_runner.resource_claim();
                    let automatic = claim.retained_snapshot_capacity_bytes.unwrap_or(0);
                    let automatic_bytes =
                        automatic.min(TextRunnerRamCacheOptions::AUTOMATIC_MAX_BYTES);
                    let max_bytes = claim.retained_snapshot_ceiling_bytes.unwrap_or(automatic);
                    capacity_limits
                        .automatic_bytes
                        .store(automatic_bytes, Ordering::Relaxed);
                    capacity_limits
                        .max_bytes
                        .store(max_bytes, Ordering::Relaxed);
                    if ram_cache.capacity_bytes == 0 {
                        automatic_bytes
                    } else {
                        // An explicit limit is the operator's choice to trade
                        // headroom for retention, bounded so the host keeps a
                        // fixed reserve.
                        ram_cache.capacity_bytes.min(max_bytes)
                    }
                })),
                on_event: Some(Box::new(emit_snapshot_event)),
            }
        } else {
            SnapshotSupport::default()
        };
        // Checkpoint records are independent of the execution states; the
        // byte budget bounds their payloads.
        let cache = ContinuationCache::new(
            state_count,
            factory,
            support,
            TextRunnerRamCacheOptions::MAX_ENTRIES,
        )?;
        if capabilities.snapshot {
            // The same post-allocation limits that selected the capacity.
            logging::info(
                "cache",
                &format!(
                    "event=snapshot_cache_configured sessions={state_count} snapshot_entries={} \
                     capacity_bytes={} automatic_bytes={} max_bytes={}",
                    cache.entry_capacity(),
                    cache.snapshot_capacity_bytes(),
                    limits.automatic_bytes.load(Ordering::Relaxed),
                    limits.max_bytes.load(Ordering::Relaxed)
                ),
            );
        }
        let mut pool = Self {
            validated,
            cache,
            admission: AdmissionGate {
                held: Mutex::new(false),
                released: Condvar::new(),
            },
            disk_store: None,
            shared_prefix_min_tokens: 0,
            shared_prefix_max_boundaries: 0,
        };
        if let Some(options) = disk_cache {
            if pool.validated.descriptor.persistence.is_none() {
                return Err(Error::invalid(
                    "text runner does not support persistent snapshots",
                ));
            }
            let store = ContinuationDiskStore::new(
                ContinuationDiskStoreOptions {
                    directory: options.directory,
                    capacity_bytes: options.capacity_bytes,
                    staging_capacity_bytes: options.staging_capacity_bytes,
                    min_checkpoint_step_tokens: options.min_checkpoint_step_tokens,
                },
                Some(Box::new(emit_disk_event)),
                None,
            )?;
            logging::info(
                "cache",
                &format!(
                    "event=disk_cache_configured capacity_bytes={} staging_capacity_bytes={}",
                    store.capacity_bytes(),
                    store.staging_capacity_bytes()
                ),
            );
            pool.disk_store = Some(Arc::new(store));
            pool.shared_prefix_min_tokens = options.shared_prefix_min_tokens;
            pool.shared_prefix_max_boundaries = options.shared_prefix_max_boundaries;
        }
        Ok(pool)
    }

    pub fn runner(&self) -> &Arc<R> {
        &self.validated.runner
    }

    /// The descriptor validated at construction.
    pub fn descriptor(&self) -> &TextRunnerDescriptor {
        &self.validated.descriptor
    }

    pub fn capacity(&self) -> usize {
        self.cache.capacity()
    }

    /// Waits until every accepted disk write finished.
    pub fn flush_disk(&self) {
        if let Some(store) = &self.disk_store {
            store.flush();
        }
    }
}

/// A capture of `state`: complete host bytes when a disk store may export it
/// (`SnapshotForPersistence`), else whatever the runner prefers.
fn capture<R: TextModelRunner>(runner: &R, state: &R::State, persist: bool) -> Result<R::Snapshot> {
    if persist {
        runner.snapshot_for_persistence(state)
    } else {
        runner.snapshot(state)
    }
}
