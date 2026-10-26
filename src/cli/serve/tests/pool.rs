//! Port of `tests/cli/text_model_runner_test.cpp`.

mod common;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    ChatRequest, ContinuationInputPrefix, Error, ErrorKind, PromptContext, Result, Sampler,
    TextDecodeSelection, TextExecutionPlan, TextModelRunner, TextPrefillStep, TextPrefillThrough,
    TextRunnerAdvance, TextRunnerCapabilities, TextRunnerDescriptor, TextRunnerMeasuredResources,
    TextRunnerPersistenceDescriptor, TextRunnerResourceClaim, TextRunnerSnapshot, TextRunnerState,
    Token,
};
use gufo_serve::logging;
use gufo_serve::pool::{
    AcquireOptions, PoolRequest, TextRunnerDiskCacheOptions, TextRunnerPool,
    TextRunnerRamCacheOptions,
};

use common::Semaphore;

/// `sizeof(FakeSnapshot)` of the C++ test (three words and a vtable), which
/// the disk staging limits below are sized against.
const SNAPSHOT_BYTES: usize = 32;
const PERSISTENT_BYTES: usize = 4 * 8;

#[derive(Default)]
struct FakeStats {
    fail_after_advance: bool,
    /// The in-pass runner fails after its prefill changed the state.
    fail_in_pass: bool,
    states_created: usize,
    invalidations: usize,
    snapshot_restores: usize,
    snapshot_size_queries: usize,
    snapshot_captures: usize,
    cancellation_bindings: usize,
    cancellation_clears: usize,
    prepared_prefixes: Vec<Vec<Token>>,
    prefill_spans: Vec<usize>,
    /// The prompt length each prefill completes its frontier at.
    prefill_frontiers: Vec<usize>,
    advanced_tokens: Vec<Token>,
    advanced_batches: Vec<Vec<Token>>,
}

type Stats = Arc<Mutex<FakeStats>>;

fn lock(stats: &Stats) -> MutexGuard<'_, FakeStats> {
    stats.lock().expect("stats")
}

struct FakeState {
    stats: Stats,
    measured_bytes: usize,
    position: usize,
    decode_count: usize,
    frontier: Option<Token>,
}

impl TextRunnerState for FakeState {
    fn invalidate(&mut self) {
        lock(&self.stats).invalidations += 1;
        self.position = 0;
        self.decode_count = 0;
        self.frontier = None;
    }

    fn set_cancellation_check(&mut self, check: Option<gufo_core::text::CancellationCheck>) {
        let mut stats = lock(&self.stats);
        if check.is_some() {
            stats.cancellation_bindings += 1;
        } else {
            stats.cancellation_clears += 1;
        }
    }

    fn measured_resources(&self) -> TextRunnerMeasuredResources {
        TextRunnerMeasuredResources {
            per_request_state_bytes: Some(self.measured_bytes),
            temporary_scratch_bytes: Some(16),
        }
    }
}

struct FakeSnapshot {
    position: usize,
    decode_count: usize,
    frontier: Option<Token>,
}

impl TextRunnerSnapshot for FakeSnapshot {
    fn payload_bytes(&self) -> usize {
        SNAPSHOT_BYTES
    }
}

/// An image request's input identity (`TextPromptContext`); the fake model
/// accepts it without reading it.
#[derive(Debug, Default)]
struct FakeContext {
    identity: Vec<u8>,
    prefixes: Vec<ContinuationInputPrefix>,
}

impl PromptContext for FakeContext {
    fn cache_identity(&self) -> &[u8] {
        &self.identity
    }

    fn cache_prefixes(&self) -> &[ContinuationInputPrefix] {
        &self.prefixes
    }
}

#[derive(Clone)]
struct FakeSampler;

impl Sampler for FakeSampler {
    fn accept(&mut self, _token: Token) {}
}

/// `FakeRunner`, `SnapshotRunner`, `PersistentSnapshotRunner`,
/// `FlakySnapshotRunner` and `InPassRunner` in one type.
struct FakeRunner {
    stats: Stats,
    measured_bytes: usize,
    state_capacity_bytes: usize,
    retained_snapshot_capacity_bytes: usize,
    snapshots: bool,
    max_context: u32,
    identity: Option<Vec<u8>>,
    flaky: bool,
    /// Captures the prompt target inside the pass crossing it.
    in_pass: bool,
    before_serialize: Option<Box<dyn Fn() + Send + Sync>>,
}

impl FakeRunner {
    fn plain(stats: &Stats) -> Self {
        Self::sized(stats, 64, 256, 256)
    }

    fn sized(
        stats: &Stats,
        measured_bytes: usize,
        state_capacity_bytes: usize,
        retained_snapshot_capacity_bytes: usize,
    ) -> Self {
        Self {
            stats: stats.clone(),
            measured_bytes,
            state_capacity_bytes,
            retained_snapshot_capacity_bytes,
            snapshots: false,
            max_context: 64,
            identity: None,
            flaky: false,
            in_pass: false,
            before_serialize: None,
        }
    }

    /// `InPassRunner` with `retained_snapshot_capacity_bytes` of cache.
    fn in_pass(stats: &Stats, retained_snapshot_capacity_bytes: usize) -> Self {
        Self {
            snapshots: true,
            in_pass: true,
            ..Self::sized(stats, 64, 256, retained_snapshot_capacity_bytes)
        }
    }

    fn snapshot(stats: &Stats) -> Self {
        Self {
            snapshots: true,
            ..Self::plain(stats)
        }
    }

    /// A persistent snapshot runner with a 32K context.
    fn long_persistent(
        stats: &Stats,
        identity: &str,
        retained_snapshot_capacity_bytes: usize,
    ) -> Self {
        Self {
            identity: Some(identity.as_bytes().to_vec()),
            ..Self::long(stats, retained_snapshot_capacity_bytes)
        }
    }

    /// A snapshot runner with a 32K context (`LongSnapshotRunner`).
    fn long(stats: &Stats, retained_snapshot_capacity_bytes: usize) -> Self {
        Self {
            snapshots: true,
            max_context: 32768,
            ..Self::sized(stats, 64, 256, retained_snapshot_capacity_bytes)
        }
    }

    fn persistent(stats: &Stats, identity: &str, retained_snapshot_capacity_bytes: usize) -> Self {
        Self {
            snapshots: true,
            identity: Some(identity.as_bytes().to_vec()),
            ..Self::sized(stats, 64, 256, retained_snapshot_capacity_bytes)
        }
    }
}

impl TextModelRunner for FakeRunner {
    type State = FakeState;
    type Snapshot = FakeSnapshot;
    type Sampler = FakeSampler;
    type PromptContext = FakeContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        TextRunnerDescriptor {
            model_id: "fake-model".into(),
            state_abi: "fake-state-v1".into(),
            max_context: self.max_context,
            capabilities: TextRunnerCapabilities {
                incremental_prefill: true,
                snapshot: self.snapshots,
                fork: self.snapshots,
                in_pass_checkpoint: self.in_pass,
                ..TextRunnerCapabilities::default()
            },
            persistence: self
                .identity
                .clone()
                .map(|identity| TextRunnerPersistenceDescriptor {
                    compatibility_identity: identity,
                    payload_version: 1,
                }),
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        TextRunnerResourceClaim {
            resident_weights_bytes: None,
            state_capacity_bytes: Some(self.state_capacity_bytes),
            per_request_state_bytes: Some(64),
            temporary_scratch_bytes: Some(16),
            retained_snapshot_capacity_bytes: Some(self.retained_snapshot_capacity_bytes),
            retained_snapshot_ceiling_bytes: None,
            requires_device_runtime_lock: false,
        }
    }

    fn supported_plans(&self) -> Vec<TextExecutionPlan> {
        vec![
            TextExecutionPlan::SERIAL,
            TextExecutionPlan::batched(2),
            TextExecutionPlan::batched(4),
        ]
    }

    fn set_prompt_context(
        &self,
        _state: &mut FakeState,
        _context: Option<&Arc<FakeContext>>,
    ) -> Result<()> {
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        Ok(text.bytes().map(Token::from).collect())
    }

    fn render_and_tokenize(&self, request: &ChatRequest) -> Result<Option<Vec<Token>>> {
        request
            .messages
            .first()
            .map(|m| self.tokenize(&m.content))
            .transpose()
    }

    fn decode(&self, tokens: &[Token]) -> Result<Vec<u8>> {
        Ok(tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<String>()
            .into_bytes())
    }

    fn create_state(&self) -> Result<FakeState> {
        lock(&self.stats).states_created += 1;
        Ok(FakeState {
            stats: self.stats.clone(),
            measured_bytes: self.measured_bytes,
            position: 0,
            decode_count: 0,
            frontier: None,
        })
    }

    fn create_sampler(&self, _config: &SamplingConfig, _prompt: &[Token]) -> Result<FakeSampler> {
        Ok(FakeSampler)
    }

    fn prepare_prefix_reuse(&self, state: &mut FakeState, prefix: &[Token]) -> Result<()> {
        if state.position != prefix.len() {
            return Err(Error::logic("fake retained prefix position mismatch"));
        }
        lock(&self.stats).prepared_prefixes.push(prefix.to_vec());
        Ok(())
    }

    fn prefill(
        &self,
        state: &mut FakeState,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
    ) -> Result<TextPrefillStep> {
        if offset != state.position || offset >= prompt.len() {
            return Err(Error::logic("invalid fake prefill position"));
        }
        let consumed = max_input_tokens.min(prompt.len() - offset);
        lock(&self.stats).prefill_spans.push(consumed);
        lock(&self.stats).prefill_frontiers.push(prompt.len());
        state.position += consumed;
        let ready = state.position == prompt.len();
        if ready {
            state.frontier = Some(90);
        }
        Ok(TextPrefillStep {
            consumed_tokens: consumed,
            decode_ready: ready,
        })
    }

    fn prefill_checkpoint_bytes(
        &self,
        _state: &FakeState,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
        boundary: usize,
    ) -> Result<Option<usize>> {
        Ok((self.in_pass
            && boundary > offset
            && boundary < prompt.len().min(offset + max_input_tokens))
        .then_some(SNAPSHOT_BYTES))
    }

    fn prefill_through(
        &self,
        state: &mut FakeState,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
        boundary: usize,
    ) -> Result<TextPrefillThrough<FakeSnapshot>> {
        lock(&self.stats).snapshot_captures += 1;
        let checkpoint = FakeSnapshot {
            position: boundary,
            decode_count: 0,
            frontier: Some(90),
        };
        let step = self.prefill(state, prompt, offset, max_input_tokens)?;
        if lock(&self.stats).fail_in_pass {
            state.decode_count = 7;
            state.frontier = Some(777);
            return Err(Error::runtime(
                "injected in-pass failure after state mutation",
            ));
        }
        Ok(TextPrefillThrough {
            step,
            checkpoint,
            checkpoint_ms: 2.0,
        })
    }

    fn select_next(
        &self,
        state: &mut FakeState,
        _sampler: &mut FakeSampler,
    ) -> Result<TextDecodeSelection> {
        let frontier = state
            .frontier
            .ok_or_else(|| Error::logic("fake state has no decode frontier"))?;
        if state.decode_count == 2 {
            return Ok(TextDecodeSelection {
                stop: true,
                ..TextDecodeSelection::default()
            });
        }
        let token = frontier + state.decode_count as Token;
        Ok(TextDecodeSelection {
            stop: false,
            token,
            piece: token.to_string().into_bytes(),
        })
    }

    fn advance(&self, state: &mut FakeState, token: Token) -> Result<()> {
        let fail = {
            let mut stats = lock(&self.stats);
            stats.advanced_tokens.push(token);
            stats.fail_after_advance
        };
        state.position += 1;
        state.decode_count += 1;
        if fail {
            return Err(Error::runtime("injected failure after state mutation"));
        }
        Ok(())
    }

    fn advance_batch(
        &self,
        advances: &mut [TextRunnerAdvance<'_, FakeState>],
    ) -> Result<Vec<Result<()>>> {
        lock(&self.stats)
            .advanced_batches
            .push(advances.iter().map(|a| a.token).collect());
        for advance in advances.iter_mut() {
            self.advance(advance.state, advance.token)?;
        }
        Ok(advances.iter().map(|_| Ok(())).collect())
    }

    fn checkpoint_position(&self, state: &FakeState) -> Result<usize> {
        Ok(state.position)
    }

    fn snapshot_payload_bytes(&self, _state: &FakeState) -> Result<usize> {
        if !self.snapshots {
            return Err(Error::logic("text runner does not support snapshot sizing"));
        }
        lock(&self.stats).snapshot_size_queries += 1;
        Ok(SNAPSHOT_BYTES)
    }

    fn snapshot(&self, state: &FakeState) -> Result<FakeSnapshot> {
        if !self.snapshots {
            return Err(Error::logic("text runner does not support snapshots"));
        }
        let captures = {
            let mut stats = lock(&self.stats);
            stats.snapshot_captures += 1;
            stats.snapshot_captures
        };
        if self.flaky && captures == 1 {
            return Err(Error::runtime("synthetic snapshot capture failure"));
        }
        Ok(FakeSnapshot {
            position: state.position,
            decode_count: state.decode_count,
            frontier: state.frontier,
        })
    }

    fn restore_or_fork(&self, state: &mut FakeState, snapshot: &FakeSnapshot) -> Result<()> {
        if !self.snapshots {
            return Err(Error::logic(
                "text runner does not support snapshot restore/fork",
            ));
        }
        state.position = snapshot.position;
        state.decode_count = snapshot.decode_count;
        state.frontier = snapshot.frontier;
        lock(&self.stats).snapshot_restores += 1;
        Ok(())
    }

    fn persistent_snapshot_payload_bytes(&self, _snapshot: &FakeSnapshot) -> Result<usize> {
        Ok(PERSISTENT_BYTES)
    }

    fn serialize_persistent_snapshot(
        &self,
        snapshot: &FakeSnapshot,
        destination: &mut [u8],
    ) -> Result<usize> {
        if let Some(hook) = &self.before_serialize {
            hook();
        }
        if destination.len() != PERSISTENT_BYTES {
            return Err(Error::invalid("fake persistent payload size mismatch"));
        }
        let fields = [
            snapshot.position as u64,
            snapshot.decode_count as u64,
            u64::from(snapshot.frontier.is_some()),
            u64::from(snapshot.frontier.unwrap_or(0)),
        ];
        for (chunk, field) in destination.as_chunks_mut::<8>().0.iter_mut().zip(fields) {
            *chunk = field.to_le_bytes();
        }
        Ok(destination.len())
    }

    fn restore_persistent_snapshot(&self, state: &mut FakeState, payload: &[u8]) -> Result<()> {
        if payload.len() != PERSISTENT_BYTES {
            return Err(Error::invalid("fake persistent payload size mismatch"));
        }
        let fields: Vec<u64> = payload
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| u64::from_le_bytes(*word))
            .collect();
        state.position = fields[0] as usize;
        state.decode_count = fields[1] as usize;
        state.frontier = (fields[2] != 0).then_some(fields[3] as Token);
        lock(&self.stats).snapshot_restores += 1;
        Ok(())
    }
}

fn stats() -> Stats {
    Arc::default()
}

fn pool(runner: FakeRunner, states: usize) -> TextRunnerPool<FakeRunner> {
    TextRunnerPool::new(Arc::new(runner), states, None).expect("pool")
}

fn disk_pool(
    runner: &Arc<FakeRunner>,
    states: usize,
    disk: &TextRunnerDiskCacheOptions,
) -> TextRunnerPool<FakeRunner> {
    TextRunnerPool::new(runner.clone(), states, Some(disk.clone())).expect("pool")
}

fn disk(
    directory: &tempfile::TempDir,
    capacity: usize,
    staging: usize,
) -> TextRunnerDiskCacheOptions {
    TextRunnerDiskCacheOptions {
        directory: directory.path().to_path_buf(),
        capacity_bytes: capacity,
        staging_capacity_bytes: staging,
        ..TextRunnerDiskCacheOptions::default()
    }
}

fn no_reuse() -> AcquireOptions<FakeContext> {
    AcquireOptions {
        reuse_prompt: false,
        ..AcquireOptions::default()
    }
}

fn stable_prefix(tokens: usize) -> AcquireOptions<FakeContext> {
    AcquireOptions {
        cache_prefix_tokens: tokens,
        ..AcquireOptions::default()
    }
}

#[test]
fn bounded_prefill_decode_and_prefix_reuse() {
    let stats = stats();
    let pool = pool(FakeRunner::plain(&stats), 1);
    assert_eq!(
        pool.descriptor().model_id,
        "fake-model",
        "pool exposes the validated runner"
    );
    assert_eq!(pool.capacity(), 1, "pool exposes its bounded capacity");
    {
        let mut request = pool.acquire(vec![1, 2, 3, 4]).expect("acquire");
        assert!(request.is_active(), "cold request acquires state");
        assert!(!request.cache_hit(), "first request is a miss");
        assert_eq!(
            request.cached_prompt_tokens(),
            0,
            "cold request starts at token zero"
        );
        let first = request.prefill(2).expect("prefill");
        assert!(
            first.consumed_tokens == 2 && !first.decode_ready,
            "first prefill obeys its input budget"
        );
        let second = request.prefill(8).expect("prefill");
        assert!(
            second.consumed_tokens == 2 && second.decode_ready,
            "second prefill reaches the decode frontier"
        );
        let token = request.select_next().expect("select");
        assert!(
            !token.stop && token.token == 90,
            "first frontier token is selected"
        );
        request.advance().expect("advance");
        let token = request.select_next().expect("select");
        assert!(
            !token.stop && token.token == 91,
            "second frontier token is selected"
        );
        request.advance().expect("advance");
        assert!(
            request.select_next().expect("select").stop,
            "runner reports an explicit stop boundary"
        );
        request.commit().expect("commit");
    }
    {
        let mut extension = pool
            .acquire(vec![1, 2, 3, 4, 90, 91, 7, 8])
            .expect("acquire");
        assert!(extension.cache_hit(), "exact extension reuses opaque state");
        assert_eq!(
            extension.cached_prompt_tokens(),
            6,
            "cache boundary includes advanced decode tokens"
        );
        assert_eq!(
            lock(&stats).prepared_prefixes,
            vec![vec![1, 2, 3, 4, 90, 91]],
            "runner prepares retained metadata before suffix prefill"
        );
        let suffix = extension.prefill(16).expect("prefill");
        assert!(
            suffix.consumed_tokens == 2 && suffix.decode_ready,
            "only the uncached suffix is prefetched"
        );
        extension.invalidate();
    }
    let stats = lock(&stats);
    assert_eq!(
        stats.prefill_spans,
        vec![2, 2, 2],
        "runner receives deterministic bounded prefill work units"
    );
    assert_eq!(
        stats.advanced_tokens,
        vec![90, 91],
        "runner receives one decode advance per emitted token"
    );
}

#[test]
fn abandoned_request_rolls_back_state() {
    let stats = stats();
    let pool = pool(FakeRunner::plain(&stats), 1);
    {
        let mut request = pool.acquire(vec![4, 5, 6]).expect("acquire");
        request.prefill(3).expect("prefill");
    }
    assert_eq!(
        lock(&stats).invalidations,
        1,
        "abandoned request invalidates partially executed state"
    );
    let mut retry = pool.acquire(vec![4, 5, 6, 7]).expect("acquire");
    assert!(!retry.cache_hit(), "rolled-back state is not reusable");
    retry.invalidate();
}

#[test]
fn request_binds_and_clears_cancellation() {
    let stats = stats();
    let pool = pool(FakeRunner::plain(&stats), 1);
    let mut request = pool
        .acquire_with(
            vec![4, 5, 6],
            AcquireOptions {
                is_cancelled: Some(Arc::new(|| false)),
                ..AcquireOptions::default()
            },
        )
        .expect("acquire");
    assert_eq!(
        lock(&stats).cancellation_bindings,
        1,
        "request binds its cancellation check to opaque state"
    );
    request.invalidate();
    assert_eq!(
        lock(&stats).cancellation_clears,
        1,
        "request clears its cancellation check before releasing state"
    );
}

#[test]
fn batched_advance_preserves_independent_requests() {
    let stats = stats();
    let pool = pool(FakeRunner::plain(&stats), 2);
    let mut first = pool.acquire(vec![1]).expect("acquire");
    let mut second = pool.acquire(vec![2]).expect("acquire");
    assert!(
        first.prefill(1).expect("prefill").decode_ready
            && second.prefill(1).expect("prefill").decode_ready,
        "independent requests reach their decode frontiers"
    );
    assert!(
        first.select_next().expect("select").token == 90
            && second.select_next().expect("select").token == 90,
        "independent requests select their pending tokens"
    );
    let plan = pool.select_decode_plan(2).expect("plan");
    assert_eq!(
        plan,
        TextExecutionPlan::batched(2),
        "two ready requests select W=2"
    );
    let results = pool
        .advance_batch(&mut [&mut first, &mut second], &plan)
        .expect("advance batch");
    assert!(results.iter().all(|r| r.is_ok()));
    assert_eq!(
        lock(&stats).advanced_batches,
        vec![vec![90, 90]],
        "one batched runner call receives both request tokens"
    );
    assert!(
        first.select_next().expect("select").token == 91
            && second.select_next().expect("select").token == 91,
        "batched advance independently updates both request states"
    );
    first.invalidate();
    second.invalidate();
}

#[test]
fn resource_claims_are_validated_before_allocation() {
    let stats = stats();
    let result = TextRunnerPool::new(Arc::new(FakeRunner::sized(&stats, 64, 159, 256)), 2, None);
    assert!(
        matches!(result, Err(ref e) if e.kind() == ErrorKind::InvalidArgument),
        "aggregate request-state claim must fit capacity"
    );
    assert_eq!(
        lock(&stats).states_created,
        0,
        "invalid resource claim is rejected before state allocation"
    );
}

#[test]
fn snapshot_fork_and_unsupported_capabilities() {
    let stats = stats();
    let runner = FakeRunner::snapshot(&stats);
    let mut state = runner.create_state().expect("state");
    state.position = 7;
    state.frontier = Some(90);
    let snapshot = runner.snapshot(&state).expect("snapshot");
    assert_eq!(
        snapshot.payload_bytes(),
        SNAPSHOT_BYTES,
        "snapshot reports its opaque payload bytes"
    );
    let mut fork = runner.create_state().expect("state");
    runner
        .restore_or_fork(&mut fork, &snapshot)
        .expect("restore");
    assert!(
        fork.position == 7 && fork.frontier == Some(90),
        "fork restores the exact runner-owned boundary"
    );
    let unsupported = FakeRunner::plain(&stats);
    assert!(
        matches!(unsupported.snapshot(&state), Err(ref e) if e.kind() == ErrorKind::Logic),
        "unsupported snapshots fail explicitly"
    );
}

#[test]
fn snapshot_cache_branches_one_prefix_into_independent_states() {
    let stats = stats();
    let pool = pool(FakeRunner::snapshot(&stats), 2);
    {
        let mut root = pool.acquire(vec![1, 2, 3, 4]).expect("acquire");
        assert!(
            root.prefill(4).expect("prefill").decode_ready,
            "root prefix reaches its snapshot boundary"
        );
        let commit = root.commit().expect("commit");
        assert!(
            commit.snapshot_bytes == SNAPSHOT_BYTES && commit.snapshot_ms >= 0.0,
            "root commit reports retained full-copy snapshot cost"
        );
    }
    let mut first = pool.acquire(vec![1, 2, 3, 4, 5]).expect("acquire");
    let mut second = pool.acquire(vec![1, 2, 3, 4, 6]).expect("acquire");
    assert!(
        first.cache_hit() && second.cache_hit(),
        "two simultaneous requests restore one retained snapshot"
    );
    assert!(
        first.cached_prompt_tokens() == 4 && second.cached_prompt_tokens() == 4,
        "both branches report the same immutable root prefix"
    );
    assert!(
        first.cache_restore_bytes() == 0
            && second.cache_restore_bytes() == SNAPSHOT_BYTES
            && first.cache_restore_ms() >= 0.0
            && second.cache_restore_ms() >= 0.0,
        "one branch uses the live frontier and the other restores the snapshot"
    );
    {
        let stats = lock(&stats);
        assert_eq!(
            stats.snapshot_restores, 1,
            "only the simultaneous branch needs a snapshot copy"
        );
        assert_eq!(
            stats.states_created, 2,
            "snapshot branching reuses preallocated request states"
        );
        assert!(
            stats.snapshot_size_queries == 1 && stats.snapshot_captures == 1,
            "root snapshot reserves its exact payload before capture"
        );
    }
    assert!(
        first.prefill(1).expect("prefill").decode_ready
            && second.prefill(1).expect("prefill").decode_ready,
        "each branch prefills only its divergent suffix"
    );
    assert!(
        first.select_next().expect("select").token == 90
            && second.select_next().expect("select").token == 90,
        "both restored branches retain an exact frontier"
    );
    first.advance().expect("advance");
    second.advance().expect("advance");
    first.commit().expect("commit");
    second.commit().expect("commit");
    let mut third = pool.acquire(vec![1, 2, 3, 4, 7]).expect("acquire");
    assert!(
        third.cache_hit() && third.cached_prompt_tokens() == 4,
        "branch commits preserve the shared root while capacity permits"
    );
    third.invalidate();
}

fn snapshot_retention_uses_prompt_boundary() {
    let stats = stats();
    let pool = pool(FakeRunner::snapshot(&stats), 1);
    {
        let mut root = pool.acquire(vec![1, 2, 3, 4]).expect("acquire");
        assert!(
            root.prefill(4).expect("prefill").decode_ready,
            "root prompt reaches its decode frontier"
        );
        assert_eq!(
            lock(&stats).snapshot_captures,
            0,
            "prefill returns before optional snapshot capture"
        );
        assert_eq!(
            root.select_next().expect("select").token,
            90,
            "root selects its prompt-boundary frontier"
        );
        root.advance().expect("advance");
        assert_eq!(
            lock(&stats).snapshot_captures,
            1,
            "snapshot is captured before generated tokens mutate the state"
        );
        assert_eq!(
            root.select_next().expect("select").token,
            91,
            "root advances beyond the reusable prompt boundary"
        );
        root.advance().expect("advance");
        let commit = root.commit().expect("commit");
        assert_eq!(
            commit.snapshot_bytes, SNAPSHOT_BYTES,
            "root commit publishes the held prompt snapshot"
        );
    }
    {
        let mut continuation = pool.acquire(vec![1, 2, 3, 4, 90, 91, 7]).expect("acquire");
        assert!(
            continuation.cache_hit()
                && continuation.cached_prompt_tokens() == 6
                && continuation.cache_restore_bytes() == 0,
            "conversation reuses all executed assistant tokens without restoration"
        );
        continuation.invalidate();
    }
    {
        let mut repeated = pool.acquire(vec![1, 2, 3, 4]).expect("acquire");
        assert!(
            repeated.cache_hit() && repeated.cached_prompt_tokens() == 4,
            "an identical prompt restores the prompt-boundary snapshot"
        );
        assert!(
            repeated.prefill_complete(),
            "an exact snapshot hit retains its decode frontier"
        );
        assert_eq!(
            lock(&stats).snapshot_captures,
            1,
            "an exact in-memory hit avoids another snapshot copy"
        );
        assert_eq!(
            repeated.select_next().expect("select").token,
            90,
            "exact reuse starts from the prompt rather than post-generation"
        );
        repeated.invalidate();
    }
    {
        let mut extension = pool.acquire(vec![1, 2, 3, 4, 7, 8]).expect("acquire");
        assert!(
            extension.cache_hit() && extension.cached_prompt_tokens() == 4,
            "a prompt that omits generated tokens still reuses the root"
        );
        let suffix = extension.prefill(8).expect("prefill");
        assert!(
            suffix.consumed_tokens == 2 && suffix.decode_ready,
            "only the extension after the stable prompt is prefetched"
        );
        extension.capture_prompt_snapshot();
        assert_eq!(
            lock(&stats).snapshot_captures,
            2,
            "the extended prompt captures its own reusable boundary"
        );
        assert_eq!(
            extension.select_next().expect("select").token,
            90,
            "extended reuse preserves the rebuilt decode frontier"
        );
        extension.invalidate();
    }
}

#[test]
fn snapshot_retention_uses_prompt_boundary_quietly() {
    let ((), log) = logging::capture(snapshot_retention_uses_prompt_boundary);
    let noise: Vec<&str> = log
        .lines()
        .filter(|line| !line.contains("event=snapshot_cache_configured"))
        .collect();
    assert!(
        noise.is_empty(),
        "routine cache replacement stays quiet beyond the startup line: {log}"
    );
}

#[test]
fn generated_frontier_forks_before_mutation() {
    let stats = stats();
    let pool = pool(FakeRunner::snapshot(&stats), 2);
    {
        let mut root = pool.acquire(vec![1, 2, 3, 4]).expect("acquire");
        root.prefill(4).expect("prefill");
        assert_eq!(
            root.select_next().expect("select").token,
            90,
            "generated root token"
        );
        root.advance().expect("advance");
        root.commit().expect("commit");
    }
    let mut first = pool.acquire(vec![1, 2, 3, 4, 90, 5]).expect("acquire");
    let mut second = pool.acquire(vec![1, 2, 3, 4, 90, 6]).expect("acquire");
    assert!(
        first.cached_prompt_tokens() == 5 && second.cached_prompt_tokens() == 5,
        "concurrent forks must reuse the generated frontier, not re-prefill its output"
    );
    assert!(
        first.cache_restore_bytes() == 0 && second.cache_restore_bytes() == SNAPSHOT_BYTES,
        "one live frontier is frozen for its peer"
    );
    {
        let stats = lock(&stats);
        assert!(
            stats.snapshot_captures == 2 && stats.snapshot_restores == 1,
            "the generated frontier is copied once while the prompt is retained"
        );
    }
    let blocked = pool
        .acquire_with(
            vec![9],
            AcquireOptions {
                is_cancelled: Some(Arc::new(|| true)),
                ..AcquireOptions::default()
            },
        )
        .expect("acquire");
    assert!(
        !blocked.is_active(),
        "publishing a snapshot must not release either live lease"
    );
    assert!(
        first.prefill(64).expect("prefill").consumed_tokens == 1
            && second.prefill(64).expect("prefill").consumed_tokens == 1,
        "neither fork puts generated history into a prefill chunk"
    );
    first.invalidate();
    second.invalidate();
    let root_branch = pool.acquire(vec![1, 2, 3, 4, 7]).expect("acquire");
    assert_eq!(
        root_branch.cached_prompt_tokens(),
        4,
        "freezing a generated frontier preserves the original prompt branch"
    );
}

#[test]
fn generated_frontier_persists_for_forks() {
    // Persist every frontier; spacing is checked below and by the disk store.
    let directory = tempfile::tempdir().expect("tempdir");
    let disk = TextRunnerDiskCacheOptions {
        min_checkpoint_step_tokens: 0,
        ..disk(&directory, 4096, 4096)
    };
    let stats = stats();
    let runner = Arc::new(FakeRunner::persistent(&stats, "artifact-A", 256));
    {
        let pool = disk_pool(&runner, 2, &disk);
        let mut root = pool.acquire(vec![1, 2, 3]).expect("acquire");
        root.prefill(3).expect("prefill");
        root.select_next().expect("select");
        root.advance().expect("advance");
        root.commit().expect("commit");
        let mut continuation = pool.acquire(vec![1, 2, 3, 90, 4]).expect("acquire");
        assert_eq!(
            continuation.cached_prompt_tokens(),
            4,
            "the live frontier includes generated output"
        );
        continuation.invalidate();
    }
    let restarted = disk_pool(&runner, 1, &disk);
    let mut restored = restarted.acquire(vec![1, 2, 3, 90, 5]).expect("acquire");
    assert!(
        restored.cache_disk_hit() && restored.cached_prompt_tokens() == 4,
        "disk forks restore the generated checkpoint before suffix prefill"
    );
    assert_eq!(
        restored.prefill(16).expect("prefill").consumed_tokens,
        1,
        "disk restoration does not re-prefill known generated tokens"
    );

    // With the default step the generated frontier one token past the
    // prompt is not written: a restart resumes from the prompt and prefills
    // it again.
    let spaced_directory = tempfile::tempdir().expect("tempdir");
    let spaced = self::disk(&spaced_directory, 4096, 4096);
    {
        let pool = disk_pool(&runner, 2, &spaced);
        let mut root = pool.acquire(vec![1, 2, 3]).expect("acquire");
        root.prefill(3).expect("prefill");
        root.select_next().expect("select");
        root.advance().expect("advance");
        root.commit().expect("commit");
    }
    let spaced_restart = disk_pool(&runner, 1, &spaced);
    let mut resumed = spaced_restart
        .acquire(vec![1, 2, 3, 90, 5])
        .expect("acquire");
    assert!(
        resumed.cache_disk_hit()
            && resumed.cached_prompt_tokens() == 3
            && resumed.prefill(16).expect("prefill").consumed_tokens == 2,
        "spaced disk checkpoints resume from the nearest stored prefix"
    );
}

#[test]
fn cancellation_retains_only_completed_work() {
    for pending in [false, true] {
        let stats = stats();
        let pool = pool(FakeRunner::snapshot(&stats), 1);
        let mut request = pool.acquire(vec![1, 2]).expect("acquire");
        request.prefill(2).expect("prefill");
        assert_eq!(
            request.select_next().expect("select").token,
            90,
            "first cancellation token"
        );
        request.advance().expect("advance");
        if pending {
            assert_eq!(
                request.select_next().expect("select").token,
                91,
                "selected but unexecuted token"
            );
        }
        request.cancel();
        {
            let stats = lock(&stats);
            assert!(
                stats.advanced_tokens.len() == 1 && stats.snapshot_captures == 1,
                "cancellation neither advances a pending token nor starts a snapshot"
            );
        }
        let mut continuation = pool.acquire(vec![1, 2, 90, 91, 7]).expect("acquire");
        assert!(
            continuation.cached_prompt_tokens() == 3 && continuation.cache_restore_bytes() == 0,
            "cancelled conversation retains the exact completed live frontier"
        );
        continuation.invalidate();
        let mut branch = pool.acquire(vec![1, 2, 8]).expect("acquire");
        assert_eq!(
            branch.cached_prompt_tokens(),
            2,
            "the original prompt remains available for branching after cancellation"
        );
        branch.invalidate();
    }

    let stats = stats();
    let pool = pool(FakeRunner::snapshot(&stats), 1);
    let mut request = pool.acquire(vec![1, 2]).expect("acquire");
    request.prefill(2).expect("prefill");
    request.select_next().expect("select");
    lock(&stats).fail_after_advance = true;
    let failure = request.advance().expect_err("mutating failure must fail");
    assert_eq!(failure.kind(), ErrorKind::Runtime);
    request.cancel();
    let mut continuation = pool.acquire(vec![1, 2, 90, 7]).expect("acquire");
    assert!(
        continuation.cached_prompt_tokens() == 2 && continuation.cache_restore_bytes() > 0,
        "a failed operation restores the immutable prompt, never mutated state"
    );
    continuation.invalidate();
}

#[test]
fn persistent_snapshot_restores_across_pools() {
    let directory = tempfile::tempdir().expect("tempdir");
    let disk = disk(&directory, 4096, 4096);
    {
        let writer_stats = stats();
        let writer = Arc::new(FakeRunner::persistent(
            &writer_stats,
            "artifact-A",
            SNAPSHOT_BYTES - 1,
        ));
        let pool = disk_pool(&writer, 1, &disk);
        let mut request = pool.acquire(vec![1, 2, 3]).expect("acquire");
        assert!(
            request.prefill(3).expect("prefill").decode_ready,
            "writer reaches persistent checkpoint"
        );
        let commit = request.commit().expect("commit");
        assert!(
            commit.snapshot_bytes == 0
                && commit.disk_queued_bytes > 0
                && commit.disk_enqueue_ms >= 0.0,
            "disk publication is independent of RAM snapshot admission"
        );
    }
    let reader_stats = stats();
    let reader = Arc::new(FakeRunner::persistent(&reader_stats, "artifact-A", 256));
    let restarted = disk_pool(&reader, 1, &disk);
    {
        let mut cold = restarted
            .acquire_with(vec![1, 2, 3, 4, 5], no_reuse())
            .expect("acquire");
        assert!(
            !cold.cache_hit()
                && !cold.cache_disk_hit()
                && cold.cached_prompt_tokens() == 0
                && lock(&reader_stats).snapshot_restores == 0,
            "cache_prompt=false bypasses disk restoration"
        );
        cold.invalidate();
    }
    let mut extension = restarted.acquire(vec![1, 2, 3, 4, 5]).expect("acquire");
    assert!(
        extension.cache_hit() && extension.cache_disk_hit(),
        "clean pool restores compatible prefix from disk"
    );
    assert!(
        extension.cached_prompt_tokens() == 3
            && extension.cache_restore_bytes() > 0
            && extension.cache_restore_ms() >= 0.0,
        "disk restore reports exact prefix and actual read metrics"
    );
    assert_eq!(
        extension.prefill(8).expect("prefill").consumed_tokens,
        2,
        "restarted request prefills only its unmatched suffix"
    );
    extension.invalidate();

    let incompatible_stats = stats();
    let incompatible = Arc::new(FakeRunner::persistent(
        &incompatible_stats,
        "artifact-B",
        256,
    ));
    let incompatible_pool = disk_pool(&incompatible, 1, &disk);
    let mut miss = incompatible_pool
        .acquire(vec![1, 2, 3, 4])
        .expect("acquire");
    assert!(
        !miss.cache_hit() && !miss.cache_disk_hit(),
        "changed compatibility identity is a cold miss"
    );
    miss.invalidate();
}

#[test]
fn prompt_reuse_can_be_disabled_per_request() {
    let stats = stats();
    let pool = pool(FakeRunner::snapshot(&stats), 1);
    {
        let mut initial = pool.acquire(vec![1, 2]).expect("acquire");
        initial.prefill(2).expect("prefill");
        initial.select_next().expect("select");
        initial.advance().expect("advance");
        initial.commit().expect("commit");
    }
    let restores = lock(&stats).snapshot_restores;
    let mut cold = pool
        .acquire_with(vec![1, 2, 90, 7], no_reuse())
        .expect("acquire");
    assert!(
        !cold.cache_hit()
            && cold.cached_prompt_tokens() == 0
            && lock(&stats).snapshot_restores == restores,
        "cache_prompt=false bypasses both live and immutable RAM frontiers"
    );
    assert_eq!(
        cold.prefill(8).expect("prefill").consumed_tokens,
        4,
        "disabled reuse processes every prompt token"
    );
    cold.commit().expect("commit");
    let mut retained = pool.acquire(vec![1, 2, 90, 7, 8]).expect("acquire");
    assert_eq!(
        retained.cached_prompt_tokens(),
        4,
        "a no-reuse request can populate the cache for later requests"
    );
    retained.invalidate();
}

#[test]
fn stable_chat_prefix_survives_interrupted_framing() {
    let directory = tempfile::tempdir().expect("tempdir");
    let disk = disk(&directory, 8192, 4096);
    let stats = stats();
    let runner = Arc::new(FakeRunner::persistent(&stats, "artifact-A", 256));
    {
        let pool = disk_pool(&runner, 1, &disk);
        // {1,2,3} is stable history; {40,41} opens assistant reasoning.
        let mut initial = pool
            .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
            .expect("acquire");
        let step = initial.prefill(64).expect("prefill");
        assert!(
            step.consumed_tokens == 3 && !step.decode_ready,
            "prefill stops before mutable assistant framing"
        );
        assert!(
            initial.prefill(64).expect("prefill").decode_ready,
            "assistant suffix completes prefill"
        );
        initial.select_next().expect("select");
        initial.advance().expect("advance");
        initial.cancel();
        // The client omits the unfinished reasoning and closes the turn.
        let mut resumed = pool
            .acquire_with(vec![1, 2, 3, 50, 51, 60], stable_prefix(5))
            .expect("acquire");
        assert_eq!(
            resumed.cached_prompt_tokens(),
            3,
            "changed assistant framing retains all stable prompt tokens"
        );
        resumed.invalidate();
    }
    let restarted = disk_pool(&runner, 1, &disk);
    let mut restored = restarted
        .acquire_with(vec![1, 2, 3, 50, 51, 60], stable_prefix(5))
        .expect("acquire");
    assert!(
        restored.cache_disk_hit() && restored.cached_prompt_tokens() == 3,
        "interrupted stable chat prefix survives server restart"
    );
    restored.invalidate();

    // Full-prompt entries written by older servers must not prevent creating
    // the shorter, stable checkpoint on the next request.
    let legacy = TextRunnerPool::new(runner.clone(), 1, None).expect("pool");
    let mut old = legacy.acquire(vec![1, 2, 3, 40, 41]).expect("acquire");
    old.prefill(64).expect("prefill");
    old.commit().expect("commit");
    let mut migrated = legacy
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    assert!(
        !migrated.cache_hit(),
        "legacy mutable-suffix snapshot cannot bypass the stable boundary"
    );
    assert_eq!(
        migrated.prefill(64).expect("prefill").consumed_tokens,
        3,
        "legacy entry is replaced by a stable chat checkpoint"
    );
    migrated.cancel();
    let mut next = legacy
        .acquire_with(vec![1, 2, 3, 50, 51, 60], stable_prefix(5))
        .expect("acquire");
    assert_eq!(
        next.cached_prompt_tokens(),
        3,
        "migrated checkpoint survives changed assistant framing"
    );
    next.invalidate();
}

#[test]
fn warm_chat_checkpoints_stop_at_the_stable_boundary() {
    let stats = stats();
    let pool = pool(FakeRunner::snapshot(&stats), 1);
    assert!(
        pool.capacity() == 1 && lock(&stats).states_created == 1,
        "extra checkpoint entries do not allocate execution sessions"
    );
    let mut root = pool
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    assert_eq!(
        root.prefill(64).expect("prefill").consumed_tokens,
        3,
        "cold chat retains the stable fallback before assistant framing"
    );
    assert!(
        root.prefill(64).expect("prefill").decode_ready,
        "cold chat completes the suffix"
    );
    let first = root.select_next().expect("select").token;
    root.advance().expect("advance");
    root.commit().expect("commit");

    let prompt = vec![1, 2, 3, 40, 41, first, 7, 40, 41];
    let mut continuation = pool
        .acquire_with(prompt.clone(), stable_prefix(7))
        .expect("acquire");
    assert!(
        continuation.cached_prompt_tokens() == 6 && continuation.cache_restore_bytes() == 0,
        "retained reasoning reuses generated tokens without a restore"
    );
    // The warm turn stops once at its own stable boundary so that boundary
    // is checkpointed; otherwise every later turn that rewrites the
    // assistant keeps falling back to this turn's frontier.
    let step = continuation.prefill(64).expect("prefill");
    assert!(
        step.consumed_tokens == 1 && !step.decode_ready,
        "warm prefill stops at the stable boundary before assistant framing"
    );
    let framing = continuation.prefill(64).expect("prefill");
    assert!(
        framing.consumed_tokens == 2 && framing.decode_ready,
        "assistant framing completes in the following pass"
    );
    let second = continuation.select_next().expect("select").token;
    continuation.advance().expect("advance");
    continuation.commit().expect("commit");

    let mut repeated = pool
        .acquire_with(prompt, stable_prefix(7))
        .expect("acquire");
    assert!(
        repeated.cached_prompt_tokens() == 9 && repeated.prefill_complete(),
        "identical thinking prompt reuses the full prompt checkpoint"
    );
    assert_eq!(
        repeated.select_next().expect("select").token,
        second,
        "the full checkpoint preserves greedy selection"
    );
    repeated.advance().expect("advance");
    repeated.commit().expect("commit");

    // Empty reasoning changes 40,41 into 50,51 in the replayed assistant.
    let mut dropped = pool
        .acquire_with(
            vec![1, 2, 3, 40, 41, first, 7, 50, 51, 8, 40, 41],
            stable_prefix(10),
        )
        .expect("acquire");
    assert_eq!(
        dropped.cached_prompt_tokens(),
        7,
        "omitted reasoning resumes from the previous turn's stable boundary \
         rather than the frontier before it"
    );
    assert!(
        !dropped.prefill(64).expect("prefill").decode_ready,
        "the rewritten turn stops at its own boundary in turn"
    );
    assert!(
        dropped.prefill(64).expect("prefill").decode_ready,
        "omitted reasoning prefills the remaining suffix"
    );
    dropped.cancel();
}

/// A client that rewrites the assistant turn, as one dropping reasoning
/// does, diverges after the previous turn's stable boundary, so each warm
/// turn checkpoints its own boundary and reuse keeps advancing.
#[test]
fn warm_hit_advances_the_stable_checkpoint() {
    let stats = stats();
    let pool = pool(FakeRunner::snapshot(&stats), 1);
    // Cold turn: the stable boundary is 3; 40, 41 is the assistant framing.
    let mut first = pool
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    assert!(!first.prefill(64).expect("prefill").decode_ready);
    assert!(first.prefill(64).expect("prefill").decode_ready);
    let reply = first.select_next().expect("select").token;
    first.advance().expect("advance");
    first.commit().expect("commit");
    // Warm turn replaying the assistant verbatim, then a new user turn: its
    // stable boundary 7 lies past the frontier 6 it reuses.
    let mut second = pool
        .acquire_with(vec![1, 2, 3, 40, 41, reply, 7, 40, 41], stable_prefix(7))
        .expect("acquire");
    assert_eq!(second.cached_prompt_tokens(), 6);
    assert!(
        !second.prefill(64).expect("prefill").decode_ready,
        "the warm turn stops to checkpoint its stable boundary"
    );
    assert!(second.prefill(64).expect("prefill").decode_ready);
    second.select_next().expect("select");
    second.advance().expect("advance");
    second.commit().expect("commit");
    // The client drops that turn's reasoning: 40, 41 becomes 50, 51 and the
    // prompt diverges at 7, exactly the previous turn's boundary.
    let mut third = pool
        .acquire_with(
            vec![1, 2, 3, 40, 41, reply, 7, 50, 51, 8, 40, 41],
            stable_prefix(10),
        )
        .expect("acquire");
    assert_eq!(
        third.cached_prompt_tokens(),
        7,
        "a warm turn checkpoints its own stable boundary, not only the frontier it reused"
    );
    assert!(!third.prefill(64).expect("prefill").decode_ready);
    assert!(third.prefill(64).expect("prefill").decode_ready);
    third.select_next().expect("select");
    third.advance().expect("advance");
    third.commit().expect("commit");
    // Dropping reasoning again keeps advancing.
    let mut fourth = pool
        .acquire_with(
            vec![1, 2, 3, 40, 41, reply, 7, 50, 51, 8, 60, 61, 11, 40, 41],
            stable_prefix(13),
        )
        .expect("acquire");
    assert_eq!(
        fourth.cached_prompt_tokens(),
        10,
        "successive rewritten turns resume from the latest boundary"
    );
    fourth.cancel();
}

/// Long prefills keep intermediate checkpoints on a 2048-token grid, so an
/// edit of earlier history restores near it.
#[test]
fn history_edits_restore_intermediate_checkpoints() {
    let complete = |request: &mut PoolRequest<FakeRunner>| {
        while !request.prefill_complete() {
            request.prefill(32768).expect("prefill");
        }
    };
    let stats = stats();
    let pool = pool(FakeRunner::long(&stats, 4096), 1);
    let mut prompt = vec![1; 10000];
    let mut original = pool
        .acquire_with(prompt.clone(), stable_prefix(9900))
        .expect("acquire");
    complete(&mut original);
    original.commit().expect("commit");
    let captures = lock(&stats).snapshot_captures;
    let mut unchanged = pool
        .acquire_with(prompt.clone(), stable_prefix(9900))
        .expect("acquire");
    assert!(
        unchanged.prefill_complete() && unchanged.cached_prompt_tokens() == prompt.len(),
        "complete prompt retries still reuse every token"
    );
    unchanged.commit().expect("commit");
    assert_eq!(
        lock(&stats).snapshot_captures,
        captures,
        "an unchanged retry does not recapture intermediate state"
    );
    prompt[9000] = 2;
    let mut late_edit = pool
        .acquire_with(prompt.clone(), stable_prefix(9900))
        .expect("acquire");
    assert!(
        late_edit.cache_hit() && late_edit.cached_prompt_tokens() == 8192,
        "late edits resume from the nearest earlier intermediate checkpoint"
    );
    complete(&mut late_edit);
    late_edit.commit().expect("commit");
    prompt[4500] = 3;
    let mut earlier_edit = pool
        .acquire_with(prompt.clone(), stable_prefix(9900))
        .expect("acquire");
    assert!(
        earlier_edit.cache_hit() && earlier_edit.cached_prompt_tokens() == 4096,
        "committing a late edit preserves earlier usable checkpoints"
    );
    earlier_edit.invalidate();
    assert_eq!(
        lock(&stats).states_created,
        1,
        "intermediate checkpoint entries allocate no extra execution states"
    );

    let limited_stats = self::stats();
    let limited = self::pool(FakeRunner::long(&limited_stats, 3 * SNAPSHOT_BYTES), 1);
    let root_prompt = vec![1; 1000];
    let mut root = limited.acquire(root_prompt.clone()).expect("acquire");
    root.prefill(32768).expect("prefill");
    root.commit().expect("commit");
    let mut extension = vec![1; 10000];
    let mut continued = limited
        .acquire_with(extension.clone(), stable_prefix(9900))
        .expect("acquire");
    complete(&mut continued);
    continued.commit().expect("commit");
    extension[1500] = 2;
    let mut branched = limited
        .acquire_with(extension, stable_prefix(9900))
        .expect("acquire");
    assert_eq!(
        branched.cached_prompt_tokens(),
        root_prompt.len(),
        "intermediate admission never evicts the reused branching fallback"
    );
    branched.invalidate();

    let cancel_stats = self::stats();
    let cancel = self::pool(FakeRunner::long(&cancel_stats, 4096), 1);
    let mut interrupted = cancel.acquire(vec![1; 10000]).expect("acquire");
    interrupted.prefill(32768).expect("prefill");
    interrupted.prefill(1024).expect("prefill");
    interrupted.cancel();
    let mut resumed = cancel.acquire(vec![1; 10000]).expect("acquire");
    assert_eq!(
        resumed.cached_prompt_tokens(),
        2048,
        "cancellation retains completed intermediate state, not partial work"
    );
    resumed.invalidate();

    let short_stats = self::stats();
    let short = self::pool(FakeRunner::long(&short_stats, 4096), 1);
    let mut short_prompt = vec![1; 7900];
    let mut short_root = short.acquire(short_prompt.clone()).expect("acquire");
    complete(&mut short_root);
    short_root.commit().expect("commit");
    let before_short = lock(&short_stats).snapshot_captures;
    lock(&short_stats).prefill_spans.clear();
    short_prompt.resize(8400, 1);
    let mut short_turn = short.acquire(short_prompt.clone()).expect("acquire");
    assert_eq!(
        short_turn.cached_prompt_tokens(),
        7900,
        "short continuation starts from the previous frontier"
    );
    assert!(
        short_turn.prefill(32768).expect("prefill").decode_ready,
        "crossing a nearby grid point does not split a short continuation"
    );
    short_turn.commit().expect("commit");
    assert!(
        lock(&short_stats).prefill_spans == [500]
            && lock(&short_stats).snapshot_captures == before_short + 1,
        "short continuation captures only its completed prompt"
    );
    short_prompt[7800] = 2;
    let mut short_edit = short.acquire(short_prompt).expect("acquire");
    assert_eq!(
        short_edit.cached_prompt_tokens(),
        6144,
        "skipping a redundant warm checkpoint retains earlier edit recovery"
    );
    short_edit.invalidate();

    let aligned_stats = self::stats();
    let aligned_pool = self::pool(FakeRunner::long(&aligned_stats, 4096), 1);
    let mut aligned_root = aligned_pool.acquire(vec![1; 1000]).expect("acquire");
    aligned_root.prefill(32768).expect("prefill");
    aligned_root.commit().expect("commit");
    let mut aligned_prompt = vec![1; 5000];
    let mut aligned = aligned_pool
        .acquire_with(aligned_prompt.clone(), stable_prefix(4096))
        .expect("acquire");
    complete(&mut aligned);
    aligned.commit().expect("commit");
    assert_eq!(
        lock(&aligned_stats).snapshot_captures,
        3,
        "a nearby grid point is skipped and the stable boundary is captured only once"
    );
    aligned_prompt[4096] = 2;
    let mut aligned_next = aligned_pool
        .acquire_with(aligned_prompt, stable_prefix(4096))
        .expect("acquire");
    assert_eq!(
        aligned_next.cached_prompt_tokens(),
        4096,
        "an aligned stable boundary remains reusable after assistant changes"
    );
    aligned_next.invalidate();

    // No grid point within 128 tokens of the prompt end.
    let tail_stats = self::stats();
    let tail_pool = self::pool(FakeRunner::long(&tail_stats, 4096), 1);
    let mut near_end = tail_pool.acquire(vec![1; 2100]).expect("acquire");
    assert!(
        near_end.prefill(32768).expect("prefill").decode_ready
            && lock(&tail_stats).prefill_spans == [2100],
        "a grid point near the prompt end does not split the final prefill"
    );
    near_end.invalidate();
    lock(&tail_stats).prefill_spans.clear();
    let mut past_tail = tail_pool.acquire(vec![2; 2300]).expect("acquire");
    complete(&mut past_tail);
    assert_eq!(
        lock(&tail_stats).prefill_spans,
        [2048, 252],
        "a grid point farther from the prompt end is still retained"
    );
    past_tail.invalidate();
}

#[test]
fn chat_fallback_survives_snapshot_budget_pressure() {
    let stats = stats();
    let runner = FakeRunner {
        snapshots: true,
        ..FakeRunner::sized(&stats, 64, 256, SNAPSHOT_BYTES)
    };
    let pool = pool(runner, 1);
    let mut root = pool
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    assert!(
        !root.prefill(64).expect("prefill").decode_ready,
        "cold chat stops at its fallback"
    );
    assert!(
        root.prefill(64).expect("prefill").decode_ready,
        "cold chat completes prefill"
    );
    root.capture_prompt_snapshot();
    assert_eq!(
        lock(&stats).snapshot_captures,
        1,
        "a full prompt is not copied when only the fallback fits"
    );
    assert_eq!(
        root.commit().expect("commit").snapshot_bytes,
        SNAPSHOT_BYTES,
        "checkpoint retention respects the original one-snapshot budget"
    );
    let mut dropped = pool
        .acquire_with(vec![1, 2, 3, 50, 51, 9], stable_prefix(5))
        .expect("acquire");
    assert_eq!(
        dropped.cached_prompt_tokens(),
        3,
        "the optional full checkpoint never evicts the required fallback"
    );
    dropped.invalidate();
}

#[test]
fn full_chat_checkpoint_restores_without_suffix_prefill() {
    let directory = tempfile::tempdir().expect("tempdir");
    let disk_cache = TextRunnerDiskCacheOptions {
        min_checkpoint_step_tokens: 0,
        ..disk(&directory, 8192, 4096)
    };
    let stats = stats();
    let runner = Arc::new(FakeRunner::persistent(&stats, "artifact-A", 256));
    {
        let writer = disk_pool(&runner, 1, &disk_cache);
        let mut prompt = writer
            .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
            .expect("acquire");
        assert!(
            !prompt.prefill(64).expect("prefill").decode_ready,
            "writer saves the stable prefix"
        );
        assert!(
            prompt.prefill(64).expect("prefill").decode_ready,
            "writer saves the complete prompt"
        );
        prompt.select_next().expect("select");
        prompt.advance().expect("advance");
        prompt.commit().expect("commit");
    }
    let reader = disk_pool(&runner, 1, &disk_cache);
    let mut exact = reader
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    assert!(
        exact.cache_disk_hit() && exact.cached_prompt_tokens() == 5 && exact.prefill_complete(),
        "disk restores the exact prompt instead of recomputing its suffix"
    );
    let token = exact.select_next().expect("select").token;
    exact.advance().expect("advance");
    exact.commit().expect("commit");
    let mut repeated = reader
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    assert!(
        !repeated.cache_disk_hit() && repeated.prefill_complete(),
        "a disk-restored full prompt is retained for cheap RAM retries"
    );
    assert_eq!(
        repeated.select_next().expect("select").token,
        token,
        "disk and RAM checkpoints select the same token"
    );
    repeated.advance().expect("advance");
    repeated.commit().expect("commit");
    let mut changed = reader
        .acquire_with(vec![1, 2, 3, 50, 51, 60], stable_prefix(5))
        .expect("acquire");
    assert!(
        changed.cache_disk_hit() && changed.cached_prompt_tokens() == 3,
        "the earlier disk fallback survives exact full-prompt restoration"
    );
    changed.invalidate();

    let legacy_directory = tempfile::tempdir().expect("tempdir");
    let legacy_disk = TextRunnerDiskCacheOptions {
        min_checkpoint_step_tokens: 0,
        ..disk(&legacy_directory, 8192, 4096)
    };
    {
        let writer = disk_pool(&runner, 1, &legacy_disk);
        let mut old = writer.acquire(vec![1, 2, 3, 40, 41]).expect("acquire");
        assert!(
            old.prefill(64).expect("prefill").decode_ready,
            "legacy writer saves only the full prompt"
        );
        old.commit().expect("commit");
    }
    let migrated = disk_pool(&runner, 1, &legacy_disk);
    let mut old = migrated
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    assert!(
        !old.cache_hit(),
        "a legacy full-prompt file without a fallback still migrates safely"
    );
    old.invalidate();
}

#[test]
fn new_image_gets_a_stable_checkpoint() {
    let stats = stats();
    let pool = pool(
        FakeRunner {
            snapshots: true,
            ..FakeRunner::sized(&stats, 64, 256, 1024)
        },
        1,
    );
    let mut text = pool
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    while !text.prefill_complete() {
        text.prefill(64).expect("prefill");
    }
    text.commit().expect("commit");

    let image = Arc::new(FakeContext {
        identity: vec![10],
        prefixes: vec![ContinuationInputPrefix {
            token_count: 5,
            identity: Vec::new(),
        }],
    });
    let with_image = |prompt: Vec<Token>, tokens: usize| {
        pool.acquire_with(
            prompt,
            AcquireOptions {
                context: Some(image.clone()),
                ..stable_prefix(tokens)
            },
        )
        .expect("acquire")
    };
    let prompt = vec![1, 2, 3, 50, 51, 60, 61, 40, 41];
    let mut appended = with_image(prompt.clone(), 7);
    assert_eq!(
        appended.cached_prompt_tokens(),
        3,
        "new image restores the preceding text checkpoint"
    );
    assert!(
        !appended.prefill(64).expect("prefill").decode_ready,
        "new image stops before mutable assistant framing"
    );
    assert!(
        appended.prefill(64).expect("prefill").decode_ready,
        "assistant framing completes after saving the image checkpoint"
    );
    appended.commit().expect("commit");

    let mut retry = with_image(prompt, 7);
    assert!(
        retry.prefill_complete() && retry.cached_prompt_tokens() == 9,
        "exact image retry keeps its complete prompt checkpoint"
    );
    retry.commit().expect("commit");

    let mut followup = with_image(vec![1, 2, 3, 50, 51, 60, 61, 52, 53, 40, 41], 9);
    assert_eq!(
        followup.cached_prompt_tokens(),
        7,
        "rewritten assistant framing does not reprocess the earlier image"
    );
    assert!(
        !followup.prefill(64).expect("prefill").decode_ready,
        "the warm turn stops to checkpoint its own stable boundary"
    );
    assert!(
        followup.prefill(64).expect("prefill").decode_ready,
        "assistant framing completes after the boundary is saved"
    );
    followup.commit().expect("commit");
}

#[test]
fn shared_prefix_is_learned_and_restored_across_conversations() {
    let directory = tempfile::tempdir().expect("tempdir");
    let disk = TextRunnerDiskCacheOptions {
        shared_prefix_min_tokens: 2,
        shared_prefix_max_boundaries: 4,
        ..disk(&directory, 8192, 4096)
    };
    let stats = stats();
    let runner = Arc::new(FakeRunner::persistent(
        &stats,
        "artifact-A",
        SNAPSHOT_BYTES - 1,
    ));
    // Conversation A: system prefix {7, 7, 7} plus its own turn.
    {
        let pool = disk_pool(&runner, 1, &disk);
        let mut request = pool.acquire(vec![7, 7, 7, 1, 2]).expect("acquire");
        assert!(!request.cache_hit(), "first conversation is cold");
        assert_eq!(
            request.prefill(8).expect("prefill").consumed_tokens,
            5,
            "nothing is shared yet, so prefill runs uninterrupted"
        );
        let commit = request.commit().expect("commit");
        assert_eq!(
            commit.shared_prefix_snapshots, 0,
            "no shared prefix exists after one conversation"
        );
    }
    // Conversation B shares only the system prefix; prefill stops there so
    // the prefix is persisted for the next conversation.
    {
        let pool = disk_pool(&runner, 1, &disk);
        let mut request = pool.acquire(vec![7, 7, 7, 3, 4, 5]).expect("acquire");
        assert!(
            !request.cache_hit(),
            "second conversation still has no prefix"
        );
        lock(&stats).prefill_frontiers.clear();
        let first = request.prefill(8).expect("prefill");
        assert!(
            first.consumed_tokens == 3 && !first.decode_ready,
            "prefill stops at the learned shared prefix"
        );
        assert_eq!(
            lock(&stats).prefill_frontiers,
            [3],
            "the model completes its frontier at the learned boundary, so the \
             snapshot there holds that prefix's logits"
        );
        let second = request.prefill(8).expect("prefill");
        assert!(
            second.consumed_tokens == 3 && second.decode_ready,
            "prefill resumes after the boundary"
        );
        let commit = request.commit().expect("commit");
        assert!(
            commit.shared_prefix_snapshots == 1
                && commit.shared_prefix_bytes > 0
                && commit.shared_prefix_ms >= 0.0,
            "the shared prefix was written once during prefill"
        );
    }
    // Conversation C restores the shared prefix and prefills only its turn.
    {
        let pool = disk_pool(&runner, 1, &disk);
        let mut request = pool.acquire(vec![7, 7, 7, 9]).expect("acquire");
        assert!(
            request.cache_hit() && request.cache_disk_hit() && request.cached_prompt_tokens() == 3,
            "third conversation restores the shared prefix from disk"
        );
        let step = request.prefill(8).expect("prefill");
        assert!(
            step.consumed_tokens == 1 && step.decode_ready,
            "only the conversation's own turn is prefilled"
        );
        let commit = request.commit().expect("commit");
        assert_eq!(
            commit.shared_prefix_snapshots, 0,
            "a restored prefix is not written again"
        );
    }
}

#[test]
fn disk_only_capture_reserves_budget_before_commit() {
    let directory = tempfile::tempdir().expect("tempdir");
    let stats = stats();
    let entered = Arc::new(Semaphore::default());
    let release = Arc::new(Semaphore::default());
    let mut runner = FakeRunner::persistent(&stats, "artifact-A", SNAPSHOT_BYTES - 1);
    {
        let (entered, release) = (entered.clone(), release.clone());
        runner.before_serialize = Some(Box::new(move || {
            entered.release();
            release.acquire();
        }));
    }
    let runner = Arc::new(runner);
    let pool = disk_pool(&runner, 2, &disk(&directory, 4096, 192));
    let mut first = pool.acquire(vec![1, 2, 3]).expect("acquire");
    first.prefill(3).expect("prefill");
    first.capture_prompt_snapshot();
    assert!(
        entered.try_acquire_for(Duration::from_secs(2)),
        "disk-only prompt is queued immediately after capture"
    );
    let mut second = pool.acquire(vec![4, 5, 6]).expect("acquire");
    second.prefill(3).expect("prefill");
    second.capture_prompt_snapshot();
    let second_commit = second.commit().expect("commit");
    assert!(
        lock(&stats).snapshot_captures == 1 && second_commit.disk_queued_bytes == 0,
        "pending disk-only state prevents another unbudgeted capture"
    );
    release.release();
    assert_ne!(
        first.commit().expect("commit").disk_queued_bytes,
        0,
        "request reports its earlier background persistence admission"
    );
}

#[test]
fn disk_preflight_avoids_unusable_capture() {
    let directory = tempfile::tempdir().expect("tempdir");
    let stats = stats();
    let runner = Arc::new(FakeRunner::persistent(
        &stats,
        "artifact-A",
        SNAPSHOT_BYTES - 1,
    ));
    let pool = disk_pool(&runner, 1, &disk(&directory, 4096, 96));
    let mut request = pool.acquire(vec![1, 2, 3]).expect("acquire");
    request.prefill(3).expect("prefill");
    let commit = request.commit().expect("commit");
    assert!(
        lock(&stats).snapshot_captures == 0 && commit.disk_queued_bytes == 0,
        "disk staging refusal happens before snapshot capture"
    );
}

#[test]
fn measured_state_is_reconciled_with_claim() {
    let stats = stats();
    let result = TextRunnerPool::new(Arc::new(FakeRunner::sized(&stats, 65, 256, 256)), 1, None);
    assert!(
        matches!(result, Err(ref e) if e.kind() == ErrorKind::Runtime),
        "measured state cannot exceed its proposed allocation"
    );
    assert_eq!(
        lock(&stats).states_created,
        1,
        "measured resource check runs immediately after allocation"
    );
}

#[test]
fn snapshot_budget_refusal_does_not_fail_completed_request() {
    let stats = stats();
    let runner = FakeRunner {
        snapshots: true,
        ..FakeRunner::sized(&stats, 64, 256, SNAPSHOT_BYTES - 1)
    };
    let pool = pool(runner, 1);
    let mut request = pool.acquire(vec![1, 2, 3]).expect("acquire");
    assert!(
        request.prefill(3).expect("prefill").decode_ready,
        "request completes before optional snapshot retention"
    );
    let commit = request.commit().expect("commit");
    assert!(
        commit.snapshot_bytes == 0 && commit.snapshot_ms == 0.0,
        "budget refusal succeeds without reporting a retained snapshot"
    );
    {
        let stats = lock(&stats);
        assert!(
            stats.snapshot_size_queries == 1 && stats.snapshot_captures == 0,
            "cache admission happens before snapshot allocation"
        );
    }
    let mut extension = pool.acquire(vec![1, 2, 3, 4]).expect("acquire");
    assert!(
        extension.cache_hit()
            && extension.cached_prompt_tokens() == 3
            && extension.cache_restore_bytes() == 0,
        "live state remains reusable when snapshot admission is refused"
    );
    extension.invalidate();
}

#[test]
fn snapshot_capture_failure_releases_reservation_and_keeps_request_successful() {
    let ((), log) = logging::capture(|| {
        let stats = stats();
        let runner = FakeRunner {
            flaky: true,
            ..FakeRunner::snapshot(&stats)
        };
        let pool = pool(runner, 1);
        {
            let mut first = pool.acquire(vec![1, 2, 3]).expect("acquire");
            assert!(
                first.prefill(3).expect("prefill").decode_ready,
                "first request reaches checkpoint"
            );
            let commit = first.commit().expect("commit");
            assert!(
                commit.snapshot_bytes == 0 && commit.snapshot_ms >= 0.0,
                "snapshot failure does not fail the completed request"
            );
        }
        {
            let mut second = pool.acquire(vec![4, 5]).expect("acquire");
            assert!(
                !second.cache_hit(),
                "failed capture retained no partial entry"
            );
            assert!(
                second.prefill(2).expect("prefill").decode_ready,
                "second request executes normally after capture failure"
            );
            let commit = second.commit().expect("commit");
            assert_eq!(
                commit.snapshot_bytes, SNAPSHOT_BYTES,
                "released reservation admits a later successful snapshot"
            );
        }
        let mut extension = pool.acquire(vec![4, 5, 6]).expect("acquire");
        assert!(
            extension.cache_hit() && extension.cached_prompt_tokens() == 2,
            "later retained snapshot restores after the failed attempt"
        );
        extension.invalidate();
    });
    assert!(
        log.contains("[WARN] [cache]") && log.contains("reason=capture_failure"),
        "cache fallback retains an actionable warning: {log}"
    );
}

/// The startup line reports the selected snapshot limits; an explicit RAM
/// limit applies up to the runner's ceiling.
#[test]
fn snapshot_startup_reports_selected_limits() {
    for (requested, selected) in [(0, 256), (100, 100), (1000, 256)] {
        let stats = stats();
        let ((), log) = logging::capture(|| {
            let _pool = TextRunnerPool::with_ram_cache(
                Arc::new(FakeRunner::snapshot(&stats)),
                2,
                None,
                TextRunnerRamCacheOptions {
                    capacity_bytes: requested,
                },
            )
            .expect("pool");
        });
        let expected = format!(
            "event=snapshot_cache_configured sessions=2 snapshot_entries=128 \
             capacity_bytes={selected} automatic_bytes=256 max_bytes=256"
        );
        assert_eq!(log.matches(&expected).count(), 1, "{requested}: {log}");
    }
}

/// Losing a retained prefix to record pressure is logged, unlike routine
/// replacement.
#[test]
fn entry_capacity_eviction_is_logged() {
    let ((), log) = logging::capture(|| {
        let stats = stats();
        // Bytes for every snapshot, so the 128 records bind.
        let pool = TextRunnerPool::new(Arc::new(FakeRunner::long(&stats, 1 << 20)), 1, None)
            .expect("pool");
        for prefix in 0..130u32 {
            let mut request = pool.acquire(vec![prefix + 1, 7]).expect("acquire");
            assert!(request.prefill(2).expect("prefill").decode_ready);
            request.select_next().expect("select");
            request.advance().expect("advance");
            request.commit().expect("commit");
        }
    });
    assert!(
        log.contains("event=snapshot action=removed reason=entry_capacity"),
        "{log}"
    );
}

/// A learned disk boundary that coincides with a history or stable
/// checkpoint is served by one immutable copy, which survives a restart
///.
#[test]
fn coincident_cache_boundaries_share_one_copy() {
    for stable in [false, true] {
        let directory = tempfile::tempdir().expect("tempdir");
        let disk_cache = TextRunnerDiskCacheOptions {
            shared_prefix_min_tokens: 2048,
            ..disk(&directory, 1024 * 1024, 64 * 1024)
        };
        let stats = stats();
        let runner = Arc::new(FakeRunner::long_persistent(&stats, "artifact-A", 4096));
        let mut prompt = vec![7; 2050];
        prompt[2048] = 1;
        {
            let pool = disk_pool(&runner, 1, &disk_cache);
            let mut source = pool.acquire(prompt.clone()).expect("acquire");
            while !source.prefill_complete() {
                source.prefill(4096).expect("prefill");
            }
            source.commit().expect("commit");
        }
        prompt[2048] = 2;
        {
            let pool = disk_pool(&runner, 1, &disk_cache);
            let mut branch = pool
                .acquire_with(prompt.clone(), stable_prefix(if stable { 2048 } else { 0 }))
                .expect("acquire");
            let before = lock(&stats).snapshot_captures;
            assert_eq!(
                branch.prefill(4096).expect("prefill").consumed_tokens,
                2048,
                "disk boundary coincides with a history or stable checkpoint"
            );
            assert!(
                branch.prefill(4096).expect("prefill").consumed_tokens == 2
                    && lock(&stats).snapshot_captures == before + 1,
                "one immutable copy serves both coincident boundaries"
            );
            branch.commit().expect("commit");
        }
        prompt[2048] = 3;
        {
            let pool = disk_pool(&runner, 1, &disk_cache);
            let mut restored = pool.acquire(prompt.clone()).expect("acquire");
            assert!(
                restored.cache_disk_hit()
                    && restored.cached_prompt_tokens() == 2048
                    && restored.prefill(4096).expect("prefill").consumed_tokens == 2,
                "the shared copy survives a restart with its exact position"
            );
            restored.commit().expect("commit");
        }
    }
}

/// With several execution states, intermediate checkpoints are copied on a
/// worker while the scheduler could run peers; the next prefill joins the
/// copy before it changes the state.
#[test]
fn concurrent_sessions_capture_checkpoints_on_a_worker() {
    let stats = stats();
    let pool = pool(FakeRunner::long(&stats, 4096), 2);
    let mut prompt = vec![1; 10000];
    let mut original = pool
        .acquire_with(prompt.clone(), stable_prefix(9900))
        .expect("acquire");
    while !original.prefill_complete() {
        original.prefill(32768).expect("prefill");
        if original.snapshot_pending() || !original.prefill_complete() {
            // A parked lease resumes once its copy is done.
            while original.snapshot_pending() {
                std::thread::yield_now();
            }
        }
    }
    original.commit().expect("commit");
    prompt[9000] = 2;
    let mut late_edit = pool
        .acquire_with(prompt.clone(), stable_prefix(9900))
        .expect("acquire");
    assert!(
        late_edit.cache_hit() && late_edit.cached_prompt_tokens() == 8192,
        "worker-captured checkpoints restore like synchronous ones"
    );
    late_edit.invalidate();
}

/// Without a disk cache, prompts diverging from a retained one after a long
/// shared prefix publish a RAM checkpoint at the divergence.
#[test]
fn ram_learns_divergence_boundaries() {
    let stats = stats();
    let pool = pool(
        FakeRunner {
            max_context: 4096,
            ..FakeRunner::persistent(&stats, "ram-boundary", 1 << 20)
        },
        1,
    );
    // Conversations share a system prompt, then each has its own tail. Below
    // 2048 tokens no grid checkpoint exists, so only learning can help.
    let conversation = |shared: usize, tail: Token| -> Vec<Token> {
        (0..shared)
            .map(|index| 100 + (index % 50) as Token)
            .chain((0..600).map(|index| tail + index))
            .collect()
    };
    let run = |prompt: Vec<Token>| {
        let mut request = pool.acquire(prompt).expect("acquire");
        let cached = request.cached_prompt_tokens();
        let mut steps = Vec::new();
        while !request.prefill_complete() {
            steps.push(request.prefill(4096).expect("prefill").consumed_tokens);
        }
        request.commit().expect("commit");
        (cached, steps)
    };
    assert_eq!(
        run(conversation(1000, 10000)),
        (0, vec![1600]),
        "the first conversation has nothing to share"
    );
    assert_eq!(
        run(conversation(1000, 20000)),
        (0, vec![1000, 600]),
        "the second conversation stops at the divergence point to retain it"
    );
    assert_eq!(
        run(conversation(1000, 30000)),
        (1000, vec![600]),
        "later conversations restore the learned boundary exactly"
    );
    assert_eq!(
        run(conversation(300, 40000)),
        (0, vec![900]),
        "a short shared prefix is not worth an extra checkpoint"
    );
    // An edit near the end diverges inside this request's own final tokens;
    // its stable checkpoint covers that, so no extra copy is taken.
    let mut edited = conversation(1000, 30000);
    *edited.last_mut().expect("token") += 1;
    let mut request = pool.acquire(edited).expect("acquire");
    assert!(
        request.cached_prompt_tokens() == 1000
            && request.prefill(4096).expect("prefill").consumed_tokens == 600,
        "a late divergence does not stop prefill for another checkpoint"
    );
    request.invalidate();
}

/// A request for exactly the learned prefix publishes the same tokens again.
/// Uncached, it replaces the learned checkpoint, in place or, with a full
/// budget, after admission reclaims the old copy; either way the copy stays a
/// branch point, or it ranks as redundant once the older branch is gone and
/// the next new branch retires it.
#[test]
fn learned_boundary_survives_exact_republication() {
    for (full, reuse) in [(false, true), (false, false), (true, true), (true, false)] {
        let stats = stats();
        // Room for five snapshots, so unrelated prompts force evictions, or
        // for three, so republication itself must evict.
        let pool = pool(
            FakeRunner {
                max_context: 4096,
                ..FakeRunner::persistent(
                    &stats,
                    "republished-boundary",
                    (if full { 3 } else { 5 }) * SNAPSHOT_BYTES,
                )
            },
            1,
        );
        let shared: Vec<Token> = vec![100; 1000];
        let run = |prompt: Vec<Token>, options: AcquireOptions<FakeContext>| {
            let mut request = pool.acquire_with(prompt, options).expect("acquire");
            let cached = request.cached_prompt_tokens();
            while !request.prefill_complete() {
                request.prefill(4096).expect("prefill");
            }
            request.commit().expect("commit");
            cached
        };
        let branch = |tail: Token| -> Vec<Token> {
            shared
                .iter()
                .copied()
                .chain(std::iter::repeat_n(tail, 600))
                .collect()
        };
        run(branch(10000), AcquireOptions::default());
        // Learns the branch point after 1000 tokens.
        run(branch(20000), AcquireOptions::default());
        if full {
            // Fills the budget and evicts the older branch.
            run(vec![9, 9, 9], AcquireOptions::default());
        }
        run(
            shared.clone(),
            if reuse {
                AcquireOptions::default()
            } else {
                no_reuse()
            },
        );
        if !full {
            for token in [9, 8, 7, 6] {
                run(vec![token; 3], AcquireOptions::default());
            }
        }
        assert_eq!(
            run(branch(30000), AcquireOptions::default()),
            1000,
            "full {full} reuse {reuse}: republishing the learned prefix keeps it a branch point"
        );
        assert_eq!(
            run(branch(40000), AcquireOptions::default()),
            1000,
            "full {full} reuse {reuse}: a new branch does not retire the republished branch point"
        );
    }
}

/// The stable boundary of a chat turn is captured inside the pass that
/// completes the framing after it.
#[test]
fn in_pass_stable_checkpoint() {
    let stats = stats();
    let pool = pool(FakeRunner::in_pass(&stats, 256), 1);
    let mut first = pool
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    let step = first.prefill(64).expect("prefill");
    assert!(
        step.consumed_tokens == 5 && step.decode_ready,
        "in-pass checkpoint completes framing in the same forward"
    );
    assert!(
        lock(&stats).snapshot_captures == 1 && lock(&stats).prefill_spans == [5],
        "one prefill captures the stable boundary before mutation"
    );
    let commit = first.commit().expect("commit");
    assert!(
        commit.snapshot_ms >= 2.0,
        "in-pass capture contributes to snapshot phase timing"
    );
    let mut second = pool
        .acquire_with(vec![1, 2, 3, 50, 51, 7, 40, 41], stable_prefix(6))
        .expect("acquire");
    assert_eq!(
        second.cached_prompt_tokens(),
        3,
        "rewritten framing restores the in-pass boundary"
    );
    assert!(
        second.prefill(64).expect("prefill").decode_ready,
        "warm rewritten turn captures its next boundary in one pass"
    );
    second.commit().expect("commit");
    let mut third = pool
        .acquire_with(vec![1, 2, 3, 50, 51, 7, 60, 61], stable_prefix(6))
        .expect("acquire");
    assert_eq!(
        third.cached_prompt_tokens(),
        6,
        "the stable checkpoint advances on warm turns"
    );
    third.invalidate();
}

/// A failed in-pass step leaves no reservation and no partial work: the
/// old fallback survives, and a retry recomputes everything.
#[test]
fn in_pass_failure_retains_only_completed_checkpoints() {
    let stats = stats();
    // Both the old fallback and the next stable boundary fit, with no spare
    // reservation: a leaked one would force the old fallback out.
    let pool = pool(FakeRunner::in_pass(&stats, 2 * SNAPSHOT_BYTES), 1);
    let mut seed = pool
        .acquire_with(vec![1, 2, 3, 40, 41], stable_prefix(3))
        .expect("acquire");
    assert!(
        seed.prefill(64).expect("prefill").decode_ready,
        "seed reaches its decode frontier"
    );
    // Retain the stable checkpoint without a full-prompt copy.
    seed.cancel();

    let mut failed = pool
        .acquire_with(vec![1, 2, 3, 50, 51, 7, 40, 41], stable_prefix(6))
        .expect("acquire");
    assert_eq!(
        failed.cached_prompt_tokens(),
        3,
        "rewritten framing starts from the immutable fallback"
    );
    lock(&stats).fail_in_pass = true;
    let error = failed.prefill(64).err().map(|e| e.to_string());
    assert!(
        error
            .as_deref()
            .is_some_and(|e| e.contains("injected in-pass failure after state mutation")),
        "in-pass prefill fails after changing the fake state: {error:?}"
    );
    failed.cancel();
    lock(&stats).fail_in_pass = false;

    let mut retry = pool
        .acquire_with(vec![1, 2, 3, 50, 51, 7, 40, 41], stable_prefix(6))
        .expect("acquire");
    assert!(
        retry.cached_prompt_tokens() == 3 && retry.cache_restore_bytes() == SNAPSHOT_BYTES,
        "cancel restores the immutable fallback, not the failed frontier"
    );
    let step = retry.prefill(64).expect("prefill");
    assert!(
        step.consumed_tokens == 5 && step.decode_ready,
        "retry recomputes all work from the failed in-pass operation"
    );
    assert_eq!(
        retry.select_next().expect("select").token,
        90,
        "retry does not inherit the failed operation's decode state"
    );
    retry.advance().expect("advance");
    retry.commit().expect("commit");

    let mut old_branch = pool
        .acquire_with(vec![1, 2, 3, 70, 71], stable_prefix(3))
        .expect("acquire");
    assert!(
        old_branch.cached_prompt_tokens() == 3
            && old_branch.cache_restore_bytes() == SNAPSHOT_BYTES,
        "released reservation lets the old fallback survive the healthy turn"
    );
    old_branch.invalidate();
    let mut new_branch = pool
        .acquire_with(vec![1, 2, 3, 50, 51, 7, 60, 61], stable_prefix(6))
        .expect("acquire");
    assert!(
        new_branch.cached_prompt_tokens() == 6
            && new_branch.cache_restore_bytes() == SNAPSHOT_BYTES,
        "healthy retry also retains its newly completed stable boundary"
    );
    new_branch.invalidate();
}

/// The `.kvc` files of a disk cache directory.
fn kvc_files(directory: &tempfile::TempDir) -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(directory.path())
        .expect("directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "kvc"))
        .collect();
    files.sort();
    files
}

/// The deepest checkpoint wins across RAM and disk, RAM on equal lengths,
/// and a rejected deeper file falls back to the RAM checkpoint (upstream's
/// tiered `PrefixIndex` lookup).
#[test]
fn deepest_checkpoint_across_ram_and_disk() {
    let directory = tempfile::tempdir().expect("tempdir");
    let disk = disk(&directory, 4096, 4096);
    {
        // RAM refuses this writer's snapshots: its checkpoint lives on disk
        // only.
        let writer_stats = stats();
        let writer = Arc::new(FakeRunner::persistent(
            &writer_stats,
            "artifact-A",
            SNAPSHOT_BYTES - 1,
        ));
        let pool = disk_pool(&writer, 1, &disk);
        let mut request = pool.acquire(vec![1, 2, 3, 4, 5]).expect("acquire");
        request.prefill(5).expect("prefill");
        assert!(request.commit().expect("commit").disk_queued_bytes > 0);
    }
    let deep_file = kvc_files(&directory);
    assert_eq!(deep_file.len(), 1, "the writer published one checkpoint");

    let reader_stats = stats();
    let reader = Arc::new(FakeRunner::persistent(&reader_stats, "artifact-A", 256));
    let pool = disk_pool(&reader, 1, &disk);
    {
        // A shallower RAM checkpoint of the same conversation.
        let mut request = pool.acquire(vec![1, 2, 3]).expect("acquire");
        assert!(!request.cache_hit(), "the disk checkpoint is not a prefix");
        request.prefill(3).expect("prefill");
        assert!(request.commit().expect("commit").snapshot_bytes > 0);
    }
    // A store busy with a write keeps the RAM path; this test waits.
    pool.flush_disk();
    assert_eq!(
        pool.cached_prefix_tokens(&[1, 2, 3, 4, 5, 6], None),
        5,
        "the cached prefix counts disk checkpoints too"
    );
    {
        let mut deeper = pool.acquire(vec![1, 2, 3, 4, 5, 6]).expect("acquire");
        assert!(
            deeper.cache_hit() && deeper.cache_disk_hit() && deeper.cached_prompt_tokens() == 5,
            "a deeper disk checkpoint beats the RAM one"
        );
        deeper.invalidate();
    }
    {
        let mut tie = pool.acquire(vec![1, 2, 3, 9]).expect("acquire");
        assert!(
            tie.cache_hit() && !tie.cache_disk_hit() && tie.cached_prompt_tokens() == 3,
            "RAM wins on equal lengths"
        );
        tie.invalidate();
    }
    // A payload that fails its checksum gives way to the RAM checkpoint.
    let mut bytes = std::fs::read(&deep_file[0]).expect("read");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&deep_file[0], bytes).expect("write");
    let mut fallback = pool.acquire(vec![1, 2, 3, 4, 5, 7]).expect("acquire");
    assert!(
        fallback.cache_hit() && !fallback.cache_disk_hit() && fallback.cached_prompt_tokens() == 3,
        "a rejected deeper file falls back to the RAM checkpoint"
    );
    assert_eq!(
        fallback.prefill(8).expect("prefill").consumed_tokens,
        3,
        "the fallback prefills only past the RAM checkpoint"
    );
    fallback.invalidate();
    assert!(!deep_file[0].exists(), "the rejected file is removed");
}
