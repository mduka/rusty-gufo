//! Bounded exact-prefix cache over opaque model continuation states
//! (`src/cli/serve/continuation_cache.cpp`).
//!
//! Each entry holds one mutable state, at most one immutable snapshot with
//! the tokens it represents, and the live frontier the state reached when
//! its last request finished. A lease owns its state exclusively until it is
//! committed or dropped; the cache never inspects state or snapshot bytes.
//! Without snapshot support ("mutable-state mode") an entry's tokens describe
//! its state directly.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use gufo_core::text::{
    ContinuationInputPrefix, Error, Result, TextRunnerSnapshot, TextRunnerState, Token,
    prefix_input_identity,
};

use crate::logging::{self, LogLevel};

const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Model-private continuation state retained by the cache.
pub trait ContinuationState: Send + 'static {
    /// Discards any partially or fully computed continuation.
    fn invalidate(&mut self);

    /// [`TextRunnerState::snapshot_owner`].
    fn snapshot_owner(&self) -> Option<u64> {
        None
    }
}

impl<T: TextRunnerState> ContinuationState for T {
    fn invalidate(&mut self) {
        TextRunnerState::invalidate(self);
    }

    fn snapshot_owner(&self) -> Option<u64> {
        TextRunnerState::snapshot_owner(self)
    }
}

/// Immutable model-private continuation payload.
pub trait ContinuationSnapshot: Send + Sync + 'static {
    fn payload_bytes(&self) -> usize;

    /// [`TextRunnerSnapshot::source`].
    fn source(&self) -> Option<u64> {
        None
    }
}

impl<T: TextRunnerSnapshot> ContinuationSnapshot for T {
    fn payload_bytes(&self) -> usize {
        TextRunnerSnapshot::payload_bytes(self)
    }

    fn source(&self) -> Option<u64> {
        TextRunnerSnapshot::source(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotEventAction {
    Removed,
    Skipped,
}

impl SnapshotEventAction {
    pub fn name(self) -> &'static str {
        match self {
            Self::Removed => "removed",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotEventReason {
    ByteCapacity,
    EntryCapacity,
    ExactReplacement,
    CaptureFailure,
    ReservationMismatch,
}

/// What a retained checkpoint is for: continuation boundaries take
/// precedence over optional history and retry copies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SnapshotPurpose {
    /// A frontier a conversation resumes from.
    #[default]
    Continuation,
    /// An intermediate prefill checkpoint for edits of earlier history.
    History,
    /// A complete prompt behind a stable boundary, for exact retries.
    Retry,
    /// A continuation boundary learned where two prompts diverged; it stays a
    /// branch point after the older prompt's checkpoints are gone.
    BranchPoint,
}

/// Continuation boundaries, learned branch points included.
fn is_continuation(purpose: SnapshotPurpose) -> bool {
    matches!(
        purpose,
        SnapshotPurpose::Continuation | SnapshotPurpose::BranchPoint
    )
}

/// The highest removal priority a checkpoint of `purpose` may evict: an
/// optional copy never displaces a family's last useful checkpoint.
fn max_removal_priority(purpose: SnapshotPurpose) -> i32 {
    match purpose {
        SnapshotPurpose::Retry => 0,
        SnapshotPurpose::History => 1,
        SnapshotPurpose::Continuation | SnapshotPurpose::BranchPoint => 3,
    }
}

impl SnapshotEventReason {
    pub fn name(self) -> &'static str {
        match self {
            Self::ByteCapacity => "byte_capacity",
            Self::EntryCapacity => "entry_capacity",
            Self::ExactReplacement => "exact_replacement",
            Self::CaptureFailure => "capture_failure",
            Self::ReservationMismatch => "reservation_mismatch",
        }
    }
}

/// Sanitized snapshot-retention event; token values and prompt contents are
/// deliberately absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotEvent {
    pub action: SnapshotEventAction,
    pub reason: SnapshotEventReason,
    pub snapshot_bytes: usize,
    pub token_count: usize,
    pub retained_snapshot_bytes: usize,
    pub reserved_snapshot_bytes: usize,
    pub capacity_bytes: usize,
}

/// Explains a miss without exposing prompt or token contents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContinuationLookup {
    /// Empty on a hit; otherwise `no_checkpoint`, `disabled`,
    /// `input_changed` or `prefix_changed`.
    pub miss_reason: &'static str,
    pub common_prefix_tokens: usize,
    pub checkpoint_tokens: usize,
}

pub type SnapshotRestore<S, P> = Box<dyn Fn(&mut S, &P) -> Result<()> + Send + Sync>;
pub type SnapshotCapacity = Box<dyn Fn() -> usize + Send + Sync>;
pub type SnapshotEventSink = Box<dyn Fn(&SnapshotEvent) + Send + Sync>;
/// Runs on an acquired state before any snapshot restore.
pub type PrepareState<'a, S> = &'a mut dyn FnMut(&mut S) -> Result<()>;

/// Snapshot mode is on when `restore` is set.
pub struct SnapshotSupport<S, P> {
    pub restore: Option<SnapshotRestore<S, P>>,
    pub capacity_bytes: Option<SnapshotCapacity>,
    pub on_event: Option<SnapshotEventSink>,
}

impl<S, P> Default for SnapshotSupport<S, P> {
    fn default() -> Self {
        Self {
            restore: None,
            capacity_bytes: None,
            on_event: None,
        }
    }
}

struct Entry<S, P> {
    /// `None` while leased.
    state: Option<S>,
    snapshot: Option<Arc<P>>,
    tokens: Vec<Token>,
    input_identity: Vec<u8>,
    live_tokens: Vec<Token>,
    live_identity: Vec<u8>,
    snapshot_bytes: usize,
    /// The stable boundary a full-prompt checkpoint was captured behind;
    /// zero when unmarked.
    stable_prefix_tokens: usize,
    purpose: SnapshotPurpose,
    state_last_used: u64,
    snapshot_last_used: u64,
    available: bool,
    valid: bool,
    dirty: bool,
}

struct Inner<S, P> {
    /// The execution states first, then (in snapshot mode) the checkpoint
    /// records, which hold every retained snapshot.
    entries: Vec<Entry<S, P>>,
    /// Entries below this index are execution states.
    state_count: usize,
    clock: u64,
    snapshot_capacity_bytes: usize,
    retained_snapshot_bytes: usize,
    reserved_snapshot_bytes: usize,
}

impl<S, P> Inner<S, P> {
    /// Whether two retained continuations extend this continuation and then
    /// diverge: it is the prefix they share, such as a system prompt, which
    /// no copy replaces, so it is never ranked as redundant. A learned
    /// branch point stays one after the older branch is gone, since a client
    /// that rewrote the prompt there (a chat bridge replaying its last user
    /// message) diverges there again; a deeper learned branch point of the
    /// same family supersedes it.
    fn is_branch_point(&self, candidate: usize) -> bool {
        let entry = &self.entries[candidate];
        if !is_continuation(entry.purpose) {
            return false;
        }
        let mut learned = entry.purpose == SnapshotPurpose::BranchPoint;
        let mut next = None;
        for (other, peer) in self.entries.iter().enumerate().skip(self.state_count) {
            if other == candidate
                || !peer.valid
                || !is_continuation(peer.purpose)
                || peer.input_identity != entry.input_identity
                || peer.tokens.len() <= entry.tokens.len()
                || !is_prefix(&entry.tokens, &peer.tokens)
            {
                continue;
            }
            if peer.purpose == SnapshotPurpose::BranchPoint {
                learned = false;
            }
            let token = peer.tokens[entry.tokens.len()];
            if next.is_some_and(|next| next != token) {
                return true;
            }
            next = Some(token);
        }
        learned
    }

    /// Whether a prompt that restores `source` then differs from a deeper
    /// retained checkpoint or live frontier of the same family: it branches
    /// at `source`, as conversations sharing a system prompt do. Later
    /// conversations restore it as well, so it is learned like a published
    /// branch point; otherwise a grid checkpoint stays history and yields to
    /// any frontier frozen later. A prompt inside the deeper tokens retries
    /// it and teaches nothing.
    fn restore_diverges(&self, source: usize, prompt: &[Token]) -> bool {
        let entry = &self.entries[source];
        if !matches!(
            entry.purpose,
            SnapshotPurpose::History | SnapshotPurpose::Continuation
        ) {
            return false;
        }
        let diverges = |tokens: &[Token], identity: &[u8]| {
            if identity != entry.input_identity.as_slice()
                || tokens.len() <= entry.tokens.len()
                || !is_prefix(&entry.tokens, tokens)
            {
                return false;
            }
            let common = tokens
                .iter()
                .zip(prompt)
                .take_while(|(a, b)| a == b)
                .count();
            common < tokens.len().min(prompt.len())
        };
        self.entries.iter().any(|peer| {
            (peer.valid && peer.snapshot.is_some() && diverges(&peer.tokens, &peer.input_identity))
                || diverges(&peer.live_tokens, &peer.live_identity)
        })
    }

    /// Ranks a record for removal, lowest first: extra copies before the last
    /// useful checkpoint of a prefix family (3). Exact tokens and input
    /// identity, not client IDs or hashes, establish that another
    /// continuation keeps the family reusable after the removal.
    fn removal_priority(
        &self,
        candidate: usize,
        incoming: &[Token],
        incoming_identity: &[u8],
    ) -> i32 {
        let entry = &self.entries[candidate];
        if self.is_branch_point(candidate) {
            return 3;
        }
        // A new continuation makes its own earlier copies redundant too:
        // during a cold prefill those may all be history checkpoints, with
        // no retained continuation yet to protect another family's last copy.
        if entry.tokens.len() < incoming.len()
            && entry.input_identity == incoming_identity
            && is_prefix(&entry.tokens, incoming)
        {
            return match entry.purpose {
                SnapshotPurpose::Retry => 0,
                SnapshotPurpose::History => 1,
                SnapshotPurpose::Continuation | SnapshotPurpose::BranchPoint => 2,
            };
        }
        for (other, peer) in self.entries.iter().enumerate().skip(self.state_count) {
            if other == candidate
                || !peer.valid
                || !is_continuation(peer.purpose)
                || peer.input_identity != entry.input_identity
            {
                continue;
            }
            match entry.purpose {
                SnapshotPurpose::Retry
                    if peer.tokens.len() <= entry.stable_prefix_tokens
                        && is_prefix(&peer.tokens, &entry.tokens) =>
                {
                    return 0;
                }
                SnapshotPurpose::History
                    if is_prefix(&entry.tokens, &peer.tokens)
                        || is_prefix(&peer.tokens, &entry.tokens) =>
                {
                    return 1;
                }
                SnapshotPurpose::Continuation | SnapshotPurpose::BranchPoint
                    if peer.tokens.len() > entry.tokens.len()
                        && is_prefix(&entry.tokens, &peer.tokens) =>
                {
                    return 2;
                }
                _ => {}
            }
        }
        3
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn event(
        &self,
        action: SnapshotEventAction,
        reason: SnapshotEventReason,
        snapshot_bytes: usize,
        token_count: usize,
    ) -> SnapshotEvent {
        SnapshotEvent {
            action,
            reason,
            snapshot_bytes,
            token_count,
            retained_snapshot_bytes: self.retained_snapshot_bytes,
            reserved_snapshot_bytes: self.reserved_snapshot_bytes,
            capacity_bytes: self.snapshot_capacity_bytes,
        }
    }

    fn release_reservation(&mut self, bytes: usize) -> bool {
        if bytes <= self.reserved_snapshot_bytes {
            self.reserved_snapshot_bytes -= bytes;
            true
        } else {
            self.reserved_snapshot_bytes = 0;
            false
        }
    }
}

struct Shared<S, P> {
    inner: Mutex<Inner<S, P>>,
    condition: Condvar,
    support: SnapshotSupport<S, P>,
    capacity: usize,
}

impl<S, P> Shared<S, P> {
    fn lock(&self) -> MutexGuard<'_, Inner<S, P>> {
        // A panic while holding the lock leaves consistent bookkeeping
        // behind (every update completes before any call that can panic).
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn snapshot_mode(&self) -> bool {
        self.support.restore.is_some()
    }

    fn emit(&self, events: &[SnapshotEvent]) {
        if let Some(sink) = &self.support.on_event {
            for event in events {
                sink(event);
            }
        }
    }
}

pub struct ContinuationCache<S, P> {
    shared: Arc<Shared<S, P>>,
}

/// A request's input identity: the complete one and, for prefixes ending
/// before later inputs (images), the identity each was computed with.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputIdentity<'a> {
    pub complete: &'a [u8],
    pub prefixes: &'a [ContinuationInputPrefix],
}

impl<'a> InputIdentity<'a> {
    /// The identity of the first `token_count` tokens.
    pub fn at(&self, token_count: usize) -> &'a [u8] {
        prefix_input_identity(self.complete, self.prefixes, token_count)
    }

    /// Boundaries must be ordered and inside the prompt.
    pub fn validate(&self, prompt_tokens: usize) -> Result<()> {
        let mut previous = 0;
        for prefix in self.prefixes {
            if prefix.token_count < previous || prefix.token_count > prompt_tokens {
                return Err(Error::invalid("input identity boundaries are invalid"));
            }
            previous = prefix.token_count;
        }
        Ok(())
    }
}

impl<'a> From<&'a [u8]> for InputIdentity<'a> {
    fn from(complete: &'a [u8]) -> Self {
        Self {
            complete,
            prefixes: &[],
        }
    }
}

/// Exclusive use of one state, with the reuse it was acquired with.
pub struct Lease<S: ContinuationState, P: ContinuationSnapshot> {
    cache: Option<Arc<Shared<S, P>>>,
    /// `None` only while the state is lent out (`take_state`).
    state: Option<S>,
    index: usize,
    cache_hit: bool,
    cached_tokens: usize,
    source_index: Option<usize>,
    /// The RAM snapshot to restore when a deeper lower-tier checkpoint
    /// fails: its entry, the snapshot and its token count.
    fallback: Option<(usize, Arc<P>, usize)>,
    restored_snapshot_bytes: usize,
    restore_ms: f64,
    restored_from_disk: bool,
    reserved_snapshot_bytes: usize,
    snapshot_purpose: SnapshotPurpose,
    prompt_tokens: usize,
    stable_prefix_tokens: usize,
    input_identity: Vec<u8>,
    input_prefixes: Vec<ContinuationInputPrefix>,
    lookup: ContinuationLookup,
}

fn is_prefix(prefix: &[Token], tokens: &[Token]) -> bool {
    tokens.starts_with(prefix)
}

fn empty_lease() -> Error {
    Error::logic("continuation cache lease is empty")
}

impl<S: ContinuationState, P: ContinuationSnapshot> Lease<S, P> {
    pub fn state(&self) -> Result<&S> {
        self.state.as_ref().ok_or_else(empty_lease)
    }

    pub fn state_mut(&mut self) -> Result<&mut S> {
        self.state.as_mut().ok_or_else(empty_lease)
    }

    /// Lends the state out (to a capture worker); `put_state` returns it.
    pub(crate) fn take_state(&mut self) -> Option<S> {
        self.state.take()
    }

    pub(crate) fn put_state(&mut self, state: S) {
        self.state = Some(state);
    }

    pub fn cache_hit(&self) -> bool {
        self.cache_hit
    }

    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub fn restored_snapshot_bytes(&self) -> usize {
        self.restored_snapshot_bytes
    }

    pub fn restore_ms(&self) -> f64 {
        self.restore_ms
    }

    pub fn restored_from_disk(&self) -> bool {
        self.restored_from_disk
    }

    pub fn lookup(&self) -> ContinuationLookup {
        self.lookup
    }

    fn shared(&self) -> Result<&Arc<Shared<S, P>>> {
        self.cache.as_ref().ok_or_else(empty_lease)
    }

    /// The input identity of the first `token_count` tokens.
    fn identity_at(&self, token_count: usize) -> &[u8] {
        prefix_input_identity(&self.input_identity, &self.input_prefixes, token_count)
    }

    /// Whether an entry already retains a snapshot of exactly `tokens` for
    /// this lease's input.
    pub fn has_snapshot_for(&self, tokens: &[Token]) -> bool {
        let Some(shared) = &self.cache else {
            return false;
        };
        let inner = shared.lock();
        let identity = self.identity_at(tokens.len());
        inner.entries.iter().any(|source| {
            source.valid
                && source.snapshot.is_some()
                && source.input_identity == identity
                && source.tokens == tokens
        })
    }

    /// Records a successful restore by a lower cache tier.
    /// The token count of the RAM fallback under a deeper lower-tier
    /// checkpoint; 0 without one.
    pub fn fallback_tokens(&self) -> usize {
        self.fallback.as_ref().map_or(0, |(_, _, tokens)| *tokens)
    }

    /// Restores the RAM fallback when the deeper lower-tier checkpoint could
    /// not be used (upstream walks to the next shallower candidate);
    /// `prepare` runs again after the restore, which replaces the request's
    /// later inputs. False without a fallback.
    pub fn restore_ram_fallback(
        &mut self,
        prompt: &[Token],
        prepare: &mut dyn FnMut(&mut S) -> Result<()>,
    ) -> Result<bool> {
        let shared = self.shared()?.clone();
        let Some((source, snapshot, tokens)) = self.fallback.take() else {
            return Ok(false);
        };
        if self.cache_hit || tokens > prompt.len() {
            return Ok(false);
        }
        let Some(restore) = &shared.support.restore else {
            return Ok(false);
        };
        let start = Instant::now();
        let state = self.state.as_mut().ok_or_else(empty_lease)?;
        restore(state, &snapshot)?;
        if !self.input_prefixes.is_empty() {
            prepare(state)?;
        }
        {
            let mut inner = shared.lock();
            let clock = inner.tick();
            let branches = inner.restore_diverges(source, prompt);
            if let Some(entry) = inner.entries.get_mut(source)
                && entry
                    .snapshot
                    .as_ref()
                    .is_some_and(|retained| Arc::ptr_eq(retained, &snapshot))
            {
                entry.snapshot_last_used = clock;
                if branches {
                    entry.purpose = SnapshotPurpose::BranchPoint;
                }
            }
        }
        self.cache_hit = true;
        self.cached_tokens = tokens;
        self.source_index = Some(source);
        self.restored_snapshot_bytes = snapshot.payload_bytes();
        self.restore_ms = start.elapsed().as_secs_f64() * 1e3;
        self.lookup = ContinuationLookup::default();
        Ok(true)
    }

    pub fn adopt_restored_prefix(
        &mut self,
        cached_tokens: usize,
        restored_bytes: usize,
        restore_ms: f64,
    ) -> Result<()> {
        self.shared()?;
        if self.cache_hit || cached_tokens == 0 || restored_bytes == 0 || restore_ms < 0.0 {
            return Err(Error::invalid(
                "invalid lower-tier continuation restore metrics",
            ));
        }
        self.cache_hit = true;
        self.cached_tokens = cached_tokens;
        self.restored_snapshot_bytes = restored_bytes;
        self.restore_ms = restore_ms;
        self.restored_from_disk = true;
        self.lookup = ContinuationLookup::default();
        Ok(())
    }

    /// Reserves retained-snapshot capacity before the model allocates the
    /// snapshot; byte-pressure evictions happen before this returns true.
    /// Evictions rank optional copies before a family's last useful
    /// checkpoint; a verified `replacement_prefix` (one extending the
    /// source) may retire this family's old boundary before another
    /// family's last copy, unless `preserve_source`.
    /// Bytes the cache may retain in RAM snapshots: zero when none can be
    /// reserved.
    pub fn snapshot_capacity_bytes(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, |shared| shared.lock().snapshot_capacity_bytes)
    }

    pub fn try_reserve_snapshot(
        &mut self,
        snapshot_bytes: usize,
        token_count: usize,
        preserve_source: bool,
        purpose: SnapshotPurpose,
        replacement_prefix: &[Token],
    ) -> Result<bool> {
        let shared = self.shared()?.clone();
        if self.reserved_snapshot_bytes != 0 {
            return Err(Error::logic(
                "continuation cache lease already has a snapshot reservation",
            ));
        }
        // Reclaiming a retained copy of the same prefix keeps its learned
        // branch point for this publication.
        let Some(purpose) = reserve_snapshot(
            &shared,
            ReserveArgs {
                source_index: self.source_index,
                snapshot_bytes,
                token_count,
                preserve_source,
                purpose,
                replacement_prefix,
                input_identity: self.identity_at(token_count),
            },
        ) else {
            return Ok(false);
        };
        self.reserved_snapshot_bytes = snapshot_bytes;
        self.snapshot_purpose = purpose;
        Ok(true)
    }

    /// Releases an admitted reservation and records a sanitized skip.
    pub fn skip_snapshot(
        &mut self,
        reason: SnapshotEventReason,
        snapshot_bytes: usize,
        token_count: usize,
    ) {
        let Some(shared) = &self.cache else {
            return;
        };
        let event = {
            let mut inner = shared.lock();
            inner.release_reservation(self.reserved_snapshot_bytes);
            inner.event(
                SnapshotEventAction::Skipped,
                reason,
                snapshot_bytes,
                token_count,
            )
        };
        shared.emit(&[event]);
        self.reserved_snapshot_bytes = 0;
    }

    /// Publishes a reserved immutable checkpoint without releasing the
    /// state, so peers can fork it before this lease advances.
    /// The published entry becomes this lease's source, which later
    /// reservations with `preserve_source` never evict; a capture past the
    /// fallback (`preserve_source`) keeps the original source instead, so
    /// later admission cannot replace the required fallback with it.
    pub fn publish_snapshot(
        &mut self,
        tokens: Vec<Token>,
        snapshot: Arc<P>,
        preserve_source: bool,
    ) -> Result<usize> {
        let shared = self.shared()?.clone();
        let input_identity = self.identity_at(tokens.len()).to_vec();
        let (retained, published) = commit(
            &shared,
            CommitArgs {
                index: self.index,
                source_index: self.source_index,
                reservation_bytes: self.reserved_snapshot_bytes,
                tokens,
                snapshot: Some(snapshot),
                input_identity,
                live_tokens: Vec::new(),
                live_identity: Vec::new(),
                released_state: None,
                stable_prefix_tokens: 0,
                purpose: self.snapshot_purpose,
            },
        );
        self.reserved_snapshot_bytes = 0;
        if published.is_some() && !preserve_source {
            self.source_index = published;
        }
        Ok(retained)
    }

    /// Publishes the state and the exact tokens it represents.
    pub fn commit(self, tokens: Vec<Token>) -> Result<usize> {
        self.commit_with(tokens, None, Vec::new())
    }

    /// Publishes the state with its live frontier and, in snapshot mode, an
    /// immutable checkpoint of `tokens`. Returns the snapshot bytes actually
    /// retained; a failed admission or capture retains none.
    pub fn commit_with(
        mut self,
        tokens: Vec<Token>,
        snapshot: Option<Arc<P>>,
        live_tokens: Vec<Token>,
    ) -> Result<usize> {
        let shared = self.cache.take().ok_or_else(empty_lease)?;
        let state = self
            .state
            .take()
            .ok_or_else(|| Error::logic("continuation cache state is lent out"))?;
        // Only a checkpoint of the complete prompt sits behind the stable
        // boundary this lease was acquired with.
        let stable_prefix_tokens = if tokens.len() == self.prompt_tokens {
            self.stable_prefix_tokens
        } else {
            0
        };
        let input_identity = self.identity_at(tokens.len()).to_vec();
        let live_identity = self.identity_at(live_tokens.len()).to_vec();
        let (retained, _) = commit(
            &shared,
            CommitArgs {
                index: self.index,
                source_index: self.source_index,
                reservation_bytes: std::mem::take(&mut self.reserved_snapshot_bytes),
                tokens,
                snapshot,
                input_identity,
                live_tokens,
                live_identity,
                released_state: Some(state),
                stable_prefix_tokens,
                purpose: self.snapshot_purpose,
            },
        );
        Ok(retained)
    }

    /// Discards partial state; dropping an uncommitted lease does the same.
    pub fn invalidate(mut self) {
        self.release_invalidated();
    }

    fn release_invalidated(&mut self) {
        let Some(shared) = self.cache.take() else {
            return;
        };
        let mut state = self.state.take();
        if let Some(state) = state.as_mut() {
            state.invalidate();
        }
        {
            let mut inner = shared.lock();
            inner.release_reservation(self.reserved_snapshot_bytes);
            let clock = inner.tick();
            let snapshot_mode = shared.snapshot_mode();
            let entry = &mut inner.entries[self.index];
            // A state that is still lent out cannot come back; the entry is
            // then unusable, which only happens after a capture panic.
            entry.available = state.is_some();
            entry.state = state;
            if !snapshot_mode {
                entry.tokens.clear();
                entry.valid = false;
            }
            entry.live_tokens.clear();
            entry.live_identity.clear();
            entry.dirty = false;
            entry.state_last_used = clock;
        }
        self.reserved_snapshot_bytes = 0;
        shared.condition.notify_one();
    }
}

impl<S: ContinuationState, P: ContinuationSnapshot> Drop for Lease<S, P> {
    fn drop(&mut self) {
        self.release_invalidated();
    }
}

struct ReserveArgs<'a> {
    source_index: Option<usize>,
    snapshot_bytes: usize,
    token_count: usize,
    preserve_source: bool,
    purpose: SnapshotPurpose,
    replacement_prefix: &'a [Token],
    input_identity: &'a [u8],
}

/// The admitted publication's purpose: a learned branch point when it
/// reclaims a retained copy of the same prefix that was one.
fn reserve_snapshot<S, P>(shared: &Shared<S, P>, args: ReserveArgs<'_>) -> Option<SnapshotPurpose> {
    let ReserveArgs {
        source_index,
        snapshot_bytes,
        token_count,
        preserve_source,
        mut purpose,
        replacement_prefix,
        input_identity,
    } = args;
    let max_priority = max_removal_priority(purpose);
    let incoming: &[Token] = if is_continuation(purpose) {
        replacement_prefix
    } else {
        &[]
    };
    let mut removed: Vec<Arc<P>> = Vec::new();
    let mut events = Vec::new();
    let admitted = {
        let mut inner = shared.lock();
        let records = inner.state_count..inner.entries.len();
        let fits = |inner: &Inner<S, P>| {
            let used = inner.retained_snapshot_bytes + inner.reserved_snapshot_bytes;
            snapshot_bytes != 0
                && used <= inner.snapshot_capacity_bytes
                && snapshot_bytes <= inner.snapshot_capacity_bytes - used
        };
        // An optional copy is skipped when every record is taken by one it
        // may not displace.
        let crowded = !is_continuation(purpose)
            && records.clone().all(|i| inner.entries[i].valid)
            && records.clone().all(|i| {
                Some(i) == source_index
                    || inner.removal_priority(i, incoming, input_identity) > max_priority
            });
        if crowded {
            events.push(inner.event(
                SnapshotEventAction::Skipped,
                SnapshotEventReason::EntryCapacity,
                snapshot_bytes,
                token_count,
            ));
            false
        } else {
            // The lowest priority first, then the least recently used.
            let lowest = |inner: &Inner<S, P>, allow_source: bool| {
                records
                    .clone()
                    .filter(|&i| {
                        let e = &inner.entries[i];
                        e.valid && e.snapshot.is_some() && (allow_source || Some(i) != source_index)
                    })
                    .map(|i| (i, inner.removal_priority(i, incoming, input_identity)))
                    .filter(|&(_, rank)| rank <= max_priority)
                    .min_by_key(|&(i, rank)| (rank, inner.entries[i].snapshot_last_used))
            };
            // Another lease may have replaced the source record: verify it.
            // A branch point other prompts still diverge at is not replaced.
            let can_replace_source = |inner: &Inner<S, P>, purpose: SnapshotPurpose| {
                if preserve_source || !is_continuation(purpose) || replacement_prefix.is_empty() {
                    return false;
                }
                source_index.is_some_and(|s| {
                    inner.entries.get(s).is_some_and(|source| {
                        source.valid
                            && source.snapshot.is_some()
                            && inner.removal_priority(s, incoming, input_identity) < 3
                            && source.tokens.len() < replacement_prefix.len()
                            && source.input_identity == input_identity
                            && is_prefix(&source.tokens, replacement_prefix)
                    })
                })
            };
            // Publishing a retained prefix again replaces that copy: reclaim
            // it before another family's.
            let exact = (replacement_prefix.len() == token_count)
                .then(|| {
                    records.clone().find(|&i| {
                        let e = &inner.entries[i];
                        Some(i) != source_index
                            && e.valid
                            && e.snapshot.is_some()
                            && e.tokens == replacement_prefix
                            && e.input_identity == input_identity
                    })
                })
                .flatten();
            let can_fit_after_eviction = snapshot_bytes != 0
                && inner.reserved_snapshot_bytes <= inner.snapshot_capacity_bytes
                && snapshot_bytes <= inner.snapshot_capacity_bytes - inner.reserved_snapshot_bytes;
            while can_fit_after_eviction && !fits(&inner) {
                let reason;
                let target = if let Some(exact) = exact.filter(|&i| inner.entries[i].valid) {
                    if inner.entries[exact].purpose == SnapshotPurpose::BranchPoint {
                        purpose = SnapshotPurpose::BranchPoint;
                    }
                    reason = SnapshotEventReason::ExactReplacement;
                    exact
                } else {
                    reason = SnapshotEventReason::ByteCapacity;
                    let mut target = lowest(&inner, false);
                    // Advance this family before sacrificing another family's
                    // last copy.
                    if can_replace_source(&inner, purpose)
                        && target.is_none_or(|(_, rank)| rank == 3)
                    {
                        target = source_index.map(|s| (s, 0));
                    }
                    if target.is_none() && !preserve_source {
                        target = lowest(&inner, true);
                    }
                    let Some((target, _)) = target else {
                        break;
                    };
                    target
                };
                let entry = &mut inner.entries[target];
                let removed_bytes = entry.snapshot_bytes;
                let removed_tokens = entry.tokens.len();
                removed.extend(entry.snapshot.take());
                entry.tokens.clear();
                entry.snapshot_bytes = 0;
                entry.valid = false;
                inner.retained_snapshot_bytes -= removed_bytes;
                events.push(inner.event(
                    SnapshotEventAction::Removed,
                    reason,
                    removed_bytes,
                    removed_tokens,
                ));
            }
            if fits(&inner) {
                inner.reserved_snapshot_bytes += snapshot_bytes;
                true
            } else {
                events.push(inner.event(
                    SnapshotEventAction::Skipped,
                    SnapshotEventReason::ByteCapacity,
                    snapshot_bytes,
                    token_count,
                ));
                false
            }
        }
    };
    // Freeing snapshots can be expensive (device memory); never under the
    // lock.
    drop(removed);
    shared.emit(&events);
    admitted.then_some(purpose)
}

struct CommitArgs<S, P> {
    index: usize,
    source_index: Option<usize>,
    reservation_bytes: usize,
    tokens: Vec<Token>,
    snapshot: Option<Arc<P>>,
    input_identity: Vec<u8>,
    live_tokens: Vec<Token>,
    live_identity: Vec<u8>,
    /// The state returned to its entry; `None` keeps the lease alive.
    released_state: Option<S>,
    stable_prefix_tokens: usize,
    purpose: SnapshotPurpose,
}

/// The snapshot bytes retained and the entry that retains them.
fn commit<S, P: ContinuationSnapshot>(
    shared: &Shared<S, P>,
    args: CommitArgs<S, P>,
) -> (usize, Option<usize>) {
    let CommitArgs {
        index,
        source_index,
        reservation_bytes,
        mut tokens,
        snapshot,
        mut input_identity,
        live_tokens,
        live_identity,
        released_state,
        stable_prefix_tokens,
        purpose,
    } = args;
    let max_priority = max_removal_priority(purpose);
    let release_state = released_state.is_some();
    let token_count = tokens.len();
    let snapshot_bytes = snapshot.as_ref().map_or(0, |s| s.payload_bytes());
    let mut skip_reason = SnapshotEventReason::CaptureFailure;
    let mut retain_snapshot = shared.snapshot_mode()
        && snapshot.is_some()
        && !tokens.is_empty()
        && reservation_bytes != 0
        && snapshot_bytes != 0
        && snapshot_bytes == reservation_bytes;
    if snapshot.is_some()
        && (reservation_bytes == 0
            || snapshot_bytes == 0
            || snapshot_bytes != reservation_bytes
            || tokens.is_empty())
    {
        skip_reason = SnapshotEventReason::ReservationMismatch;
    }
    let had_snapshot = snapshot.is_some();
    let (mut retained_snapshot, mut dropped_snapshot) = if retain_snapshot {
        (snapshot, None)
    } else {
        (None, snapshot)
    };

    let mut removed: Vec<Arc<P>> = Vec::new();
    let mut events = Vec::new();
    let mut retained_bytes = 0;
    let mut published = None;
    {
        let mut inner = shared.lock();
        if !inner.release_reservation(reservation_bytes) {
            retain_snapshot = false;
            skip_reason = SnapshotEventReason::ReservationMismatch;
            dropped_snapshot = dropped_snapshot.or(retained_snapshot.take());
        }
        if shared.snapshot_mode() {
            if release_state {
                let entry = &mut inner.entries[index];
                entry.live_tokens = live_tokens;
                entry.live_identity = live_identity;
            }
            if retain_snapshot {
                let records = inner.state_count..inner.entries.len();
                let incoming: &[Token] = if is_continuation(purpose) {
                    &tokens
                } else {
                    &[]
                };
                let priority = |i: usize| inner.removal_priority(i, incoming, &input_identity);
                let entries = &inner.entries;
                let exact = records.clone().find(|&i| {
                    let e = &entries[i];
                    e.valid && e.tokens == tokens && e.input_identity == input_identity
                });
                let exact_replacement = exact.is_some();
                let mut last_copy_eviction = false;
                let mut target = exact
                    .or_else(|| records.clone().find(|&i| !entries[i].valid))
                    .or_else(|| {
                        // An edited branch replaces its old, incompatible
                        // tail before evicting checkpoints of the shared
                        // prefix, keeping a new stable boundary and complete
                        // prompt within the bounded record pool.
                        let source = source_index.and_then(|s| entries.get(s)).filter(|s| {
                            s.valid && !s.tokens.is_empty() && tokens.starts_with(&s.tokens)
                        })?;
                        records
                            .clone()
                            .filter(|&i| {
                                let e = &entries[i];
                                e.valid
                                    && Some(i) != source_index
                                    && e.input_identity == input_identity
                                    && e.tokens.len() > source.tokens.len()
                                    && !tokens.starts_with(&e.tokens)
                                    && e.tokens.starts_with(&source.tokens)
                                    && priority(i) <= max_priority
                            })
                            .min_by_key(|&i| entries[i].snapshot_last_used)
                    })
                    .or_else(|| {
                        let lowest = records
                            .clone()
                            .filter(|&i| {
                                Some(i) != source_index
                                    || (records.len() <= 1 && is_continuation(purpose))
                            })
                            .map(|i| (i, priority(i)))
                            .filter(|&(_, rank)| rank <= max_priority)
                            .min_by_key(|&(i, rank)| (rank, entries[i].snapshot_last_used));
                        last_copy_eviction = lowest.is_some_and(|(_, rank)| rank == 3);
                        lowest.map(|(i, _)| i)
                    });
                // Before evicting another family's last copy, a continuation
                // replaces its own source when it extends it.
                if last_copy_eviction
                    && is_continuation(purpose)
                    && let Some(source) = source_index.filter(|&s| {
                        let e = &entries[s];
                        e.valid
                            && priority(s) < 3
                            && e.input_identity == input_identity
                            && e.tokens.len() < tokens.len()
                            && tokens.starts_with(&e.tokens)
                    })
                {
                    target = Some(source);
                }
                // Optional copies cannot displace a last useful checkpoint,
                // even when another publication took a free record after
                // the reservation.
                match target {
                    None => {
                        retain_snapshot = false;
                        skip_reason = SnapshotEventReason::EntryCapacity;
                        dropped_snapshot = retained_snapshot.take();
                    }
                    Some(target) => {
                        // Republishing the same prefix, as an uncached
                        // request for exactly a learned branch point does,
                        // keeps the learned divergence.
                        let learned = exact_replacement
                            && inner.entries[target].purpose == SnapshotPurpose::BranchPoint;
                        if let Some(old) = inner.entries[target].snapshot.take() {
                            let entry = &mut inner.entries[target];
                            let removed_bytes = entry.snapshot_bytes;
                            let removed_tokens = entry.tokens.len();
                            entry.tokens.clear();
                            entry.snapshot_bytes = 0;
                            entry.valid = false;
                            inner.retained_snapshot_bytes -= removed_bytes;
                            removed.push(old);
                            events.push(inner.event(
                                SnapshotEventAction::Removed,
                                if exact_replacement {
                                    SnapshotEventReason::ExactReplacement
                                } else {
                                    SnapshotEventReason::EntryCapacity
                                },
                                removed_bytes,
                                removed_tokens,
                            ));
                        }
                        let used = inner.retained_snapshot_bytes + inner.reserved_snapshot_bytes;
                        if used > inner.snapshot_capacity_bytes
                            || snapshot_bytes > inner.snapshot_capacity_bytes - used
                        {
                            retain_snapshot = false;
                            skip_reason = SnapshotEventReason::ReservationMismatch;
                            dropped_snapshot = retained_snapshot.take();
                        } else {
                            let clock = inner.tick();
                            let entry = &mut inner.entries[target];
                            entry.tokens = std::mem::take(&mut tokens);
                            entry.input_identity = std::mem::take(&mut input_identity);
                            entry.snapshot = retained_snapshot.take();
                            entry.snapshot_bytes = snapshot_bytes;
                            entry.stable_prefix_tokens = stable_prefix_tokens;
                            entry.purpose = if learned {
                                SnapshotPurpose::BranchPoint
                            } else {
                                purpose
                            };
                            entry.valid = !entry.tokens.is_empty();
                            entry.snapshot_last_used = clock;
                            inner.retained_snapshot_bytes += snapshot_bytes;
                            retained_bytes = snapshot_bytes;
                            published = Some(target);
                        }
                    }
                }
                if !retain_snapshot {
                    events.push(inner.event(
                        SnapshotEventAction::Skipped,
                        skip_reason,
                        snapshot_bytes,
                        token_count,
                    ));
                }
            } else if had_snapshot || reservation_bytes != 0 {
                events.push(inner.event(
                    SnapshotEventAction::Skipped,
                    skip_reason,
                    snapshot_bytes,
                    token_count,
                ));
            }
        } else {
            let entry = &mut inner.entries[index];
            entry.tokens = tokens;
            entry.input_identity = input_identity;
            entry.valid = !entry.tokens.is_empty();
        }
        if let Some(state) = released_state {
            let clock = inner.tick();
            let entry = &mut inner.entries[index];
            entry.state = Some(state);
            entry.dirty = true;
            entry.available = true;
            entry.state_last_used = clock;
        }
    }
    drop(removed);
    drop(dropped_snapshot);
    shared.emit(&events);
    shared.condition.notify_all();
    (retained_bytes, published)
}

impl<S: ContinuationState, P: ContinuationSnapshot> ContinuationCache<S, P> {
    /// Creates `capacity` states up front. In snapshot mode the cache holds
    /// up to `snapshot_capacity` checkpoints; the extra entries allocate no
    /// execution state, and the byte budget still bounds every retained and
    /// in-flight snapshot.
    pub fn new(
        capacity: usize,
        mut factory: impl FnMut() -> Result<S>,
        support: SnapshotSupport<S, P>,
        snapshot_capacity: usize,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::invalid(
                "continuation cache capacity must be at least one",
            ));
        }
        // Checkpoint records are separate from execution states; zero
        // records default to the state count.
        let records = if snapshot_capacity == 0 {
            capacity
        } else {
            snapshot_capacity
        };
        let entry_count = if support.restore.is_some() {
            capacity
                .checked_add(records)
                .ok_or_else(|| Error::invalid("continuation checkpoint capacity overflows"))?
        } else {
            capacity
        };
        let mut entries = Vec::with_capacity(entry_count);
        for index in 0..entry_count {
            // Checkpoint-only entries are never selected as execution slots.
            let state = if index < capacity {
                Some(factory()?)
            } else {
                None
            };
            entries.push(Entry {
                available: state.is_some(),
                state,
                snapshot: None,
                tokens: Vec::new(),
                input_identity: Vec::new(),
                live_tokens: Vec::new(),
                live_identity: Vec::new(),
                snapshot_bytes: 0,
                stable_prefix_tokens: 0,
                purpose: SnapshotPurpose::Continuation,
                state_last_used: 0,
                snapshot_last_used: 0,
                valid: false,
                dirty: false,
            });
        }
        let snapshot_capacity_bytes = match (&support.restore, &support.capacity_bytes) {
            (Some(_), Some(capacity_bytes)) => capacity_bytes(),
            _ => 0,
        };
        Ok(Self {
            shared: Arc::new(Shared {
                inner: Mutex::new(Inner {
                    entries,
                    state_count: capacity,
                    clock: 0,
                    snapshot_capacity_bytes,
                    retained_snapshot_bytes: 0,
                    reserved_snapshot_bytes: 0,
                }),
                condition: Condvar::new(),
                support,
                capacity,
            }),
        })
    }

    /// The longest retained prefix of `prompt` with a matching input
    /// identity, ignoring the stable-prefix fallback rules; leases nothing.
    /// The same candidates as `acquire`: retained checkpoints and the live
    /// frontiers of states not leased.
    pub fn cached_prefix_tokens(&self, prompt: &[Token], input: InputIdentity<'_>) -> usize {
        let matches = |tokens: &[Token], identity: &[u8]| {
            is_prefix(tokens, prompt) && identity == input.at(tokens.len())
        };
        let snapshot_mode = self.shared.support.restore.is_some();
        let inner = self.shared.lock();
        let mut longest = 0;
        for entry in &inner.entries {
            if entry.valid
                && (snapshot_mode || entry.available)
                && entry.tokens.len() > longest
                && matches(&entry.tokens, &entry.input_identity)
            {
                longest = entry.tokens.len();
            }
            if snapshot_mode
                && entry.available
                && entry.live_tokens.len() > longest
                && matches(&entry.live_tokens, &entry.live_identity)
            {
                longest = entry.live_tokens.len();
            }
        }
        longest
    }

    /// The longest prefix `prompt` shares with any retained checkpoint or
    /// live frontier of the same input identity, whether or not that record
    /// ends there: where the prompts diverge, so a checkpoint captured there
    /// serves later prompts sharing it. Leases nothing. A record's one
    /// identity must match the prompt's at the record's length, which
    /// conservatively skips records whose images differ anywhere: a boundary
    /// never spans an image change.
    pub fn common_prefix_tokens(&self, prompt: &[Token], input: InputIdentity<'_>) -> usize {
        let common = |tokens: &[Token], identity: &[u8]| {
            if tokens.is_empty() || identity != input.at(tokens.len()) {
                return 0;
            }
            tokens
                .iter()
                .zip(prompt)
                .take_while(|(a, b)| a == b)
                .count()
        };
        let inner = self.shared.lock();
        let mut longest = 0;
        for entry in &inner.entries {
            if entry.valid {
                longest = longest.max(common(&entry.tokens, &entry.input_identity));
            }
            longest = longest.max(common(&entry.live_tokens, &entry.live_identity));
        }
        longest
    }

    /// Immutable checkpoint records, independent of the execution states.
    pub fn entry_capacity(&self) -> usize {
        let inner = self.shared.lock();
        if self.shared.support.restore.is_some() {
            inner.entries.len() - inner.state_count
        } else {
            inner.entries.len()
        }
    }

    /// Acquires a state for `prompt`, reusing the longest retained prefix.
    pub fn acquire(&self, prompt: &[Token]) -> Result<Option<Lease<S, P>>> {
        self.acquire_with(prompt, None, InputIdentity::default(), None, true, 0)
    }

    /// Waits for a free state (polling `is_cancelled` every 10 ms; `None`
    /// when it fires), reuses the longest prefix of `prompt` retained with
    /// the input identity at its length unless `reuse_prompt` is false, and
    /// runs `prepare_state` on the state before any snapshot restore (and
    /// again after one when `input` has prefix identities, since the
    /// restore replaces the request's later inputs). A nonzero
    /// `stable_prefix_tokens` reuses a checkpoint past that boundary
    /// (including an exact retry) only while a fallback at or before it
    /// remains.
    pub fn acquire_with(
        &self,
        prompt: &[Token],
        is_cancelled: Option<&dyn Fn() -> bool>,
        input: InputIdentity<'_>,
        prepare_state: Option<PrepareState<'_, S>>,
        reuse_prompt: bool,
        stable_prefix_tokens: usize,
    ) -> Result<Option<Lease<S, P>>> {
        self.acquire_tiered(
            prompt,
            is_cancelled,
            input,
            prepare_state,
            reuse_prompt,
            stable_prefix_tokens,
            0,
        )
    }

    /// [`acquire_with`](Self::acquire_with) when a lower tier holds a usable
    /// checkpoint of `deeper_tier_tokens`: the deepest checkpoint wins
    /// across tiers, RAM and live states on equal lengths (upstream's
    /// `PrefixIndex` order). A strictly deeper lower tier leaves the lease
    /// a cold miss for the caller to restore, keeping the best RAM snapshot
    /// as its [`restore_ram_fallback`](Lease::restore_ram_fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn acquire_tiered(
        &self,
        prompt: &[Token],
        is_cancelled: Option<&dyn Fn() -> bool>,
        input: InputIdentity<'_>,
        mut prepare_state: Option<PrepareState<'_, S>>,
        reuse_prompt: bool,
        stable_prefix_tokens: usize,
        deeper_tier_tokens: usize,
    ) -> Result<Option<Lease<S, P>>> {
        if stable_prefix_tokens > prompt.len() {
            return Err(Error::invalid("stable cache prefix exceeds prompt length"));
        }
        input.validate(prompt.len())?;
        let matches_input = |identity: &[u8], count: usize| identity == input.at(count);
        let shared = &self.shared;
        let snapshot_mode = shared.snapshot_mode();
        // Debug records are collected under the lock and logged after it is
        // released; a wait reports its first record set once, not per poll.
        let debug_cache = logging::enabled(LogLevel::Debug);
        let mut skipped: Vec<(usize, usize, &'static str)> = Vec::new();
        let mut wait_reported = false;
        let log_skips = |skipped: &[(usize, usize, &str)]| {
            for (index, tokens, reason) in skipped {
                logging::debug(
                    "cache",
                    &format!(
                        "event=candidate_skip index={index} tokens={tokens} prompt_tokens={} \
                         reason={reason}",
                        prompt.len()
                    ),
                );
            }
        };
        let mut inner = shared.lock();
        loop {
            skipped.clear();
            // Full-prompt hits are safe while a shorter checkpoint remains for
            // clients that rewrite the assistant's opening tokens, or when the
            // full checkpoint was captured behind that boundary (its fallback
            // may now live on disk). Unmarked older entries migrate.
            let has_fallback = stable_prefix_tokens == 0
                || inner.entries.iter().any(|entry| {
                    entry.valid
                        && entry.snapshot.is_some()
                        && (entry.tokens.len() <= stable_prefix_tokens
                            || (entry.stable_prefix_tokens != 0
                                && entry.stable_prefix_tokens <= stable_prefix_tokens))
                        && matches_input(&entry.input_identity, entry.tokens.len())
                        && is_prefix(&entry.tokens, prompt)
                });
            let can_reuse = |count: usize| has_fallback || count <= stable_prefix_tokens;
            let mut source: Option<usize> = None;
            let mut cached_tokens = 0;
            if reuse_prompt {
                for (index, entry) in inner.entries.iter().enumerate() {
                    // In order, so the first failing guard is the reason.
                    let skip = if (!snapshot_mode && !entry.available) || !entry.valid {
                        Some("unavailable")
                    } else if !matches_input(&entry.input_identity, entry.tokens.len()) {
                        Some("input_identity")
                    } else if !can_reuse(entry.tokens.len()) {
                        Some("stable_prefix_boundary")
                    } else if !is_prefix(&entry.tokens, prompt) {
                        Some("token_prefix")
                    } else {
                        None
                    };
                    if let Some(reason) = skip {
                        if debug_cache {
                            skipped.push((index, entry.tokens.len(), reason));
                        }
                        continue;
                    }
                    if source.is_none() || entry.tokens.len() > cached_tokens {
                        source = Some(index);
                        cached_tokens = entry.tokens.len();
                    }
                }
            }
            // A live final frontier can extend the immutable prompt snapshot;
            // prefer it on equal lengths too, since it needs no restore.
            let mut live_source = None;
            if reuse_prompt && snapshot_mode {
                for (index, entry) in inner.entries.iter().enumerate() {
                    if entry.available
                        && !entry.live_tokens.is_empty()
                        && entry.live_tokens.len() >= cached_tokens
                        && can_reuse(entry.live_tokens.len())
                        && matches_input(&entry.live_identity, entry.live_tokens.len())
                        && is_prefix(&entry.live_tokens, prompt)
                    {
                        live_source = Some(index);
                        cached_tokens = entry.live_tokens.len();
                    }
                }
            }
            // Without any RAM candidate this is an ordinary miss.
            let deeper_tier =
                deeper_tier_tokens > cached_tokens && (source.is_some() || live_source.is_some());
            let live_hit = live_source.is_some() && !deeper_tier;
            let cache_hit = live_hit || (source.is_some() && !deeper_tier);
            let mut lookup = ContinuationLookup::default();
            if deeper_tier {
                lookup = ContinuationLookup {
                    miss_reason: "deeper_tier",
                    common_prefix_tokens: cached_tokens,
                    checkpoint_tokens: cached_tokens,
                };
            } else if !cache_hit {
                lookup.miss_reason = if reuse_prompt {
                    "no_checkpoint"
                } else {
                    "disabled"
                };
                if reuse_prompt {
                    // Token agreement is reported even where the input
                    // identity makes the checkpoint unusable.
                    let mut inspect = |tokens: &[Token], identity: &[u8]| {
                        if tokens.is_empty() {
                            return;
                        }
                        let common = tokens
                            .iter()
                            .zip(prompt)
                            .take_while(|(a, b)| a == b)
                            .count();
                        if lookup.checkpoint_tokens == 0 || common > lookup.common_prefix_tokens {
                            lookup = ContinuationLookup {
                                miss_reason: if matches_input(identity, tokens.len()) {
                                    "prefix_changed"
                                } else {
                                    "input_changed"
                                },
                                common_prefix_tokens: common,
                                checkpoint_tokens: tokens.len(),
                            };
                        }
                    };
                    for entry in &inner.entries {
                        if entry.valid {
                            inspect(&entry.tokens, &entry.input_identity);
                        }
                        inspect(&entry.live_tokens, &entry.live_identity);
                    }
                }
            }
            let mut selected = if live_hit { live_source } else { source };
            if (snapshot_mode && !live_hit) || !cache_hit {
                // An available state that still holds the snapshot's borrowed
                // rows restores it with the fewest copies; a busy one leaves the choice to recency.
                let owner = source
                    .filter(|_| cache_hit)
                    .and_then(|s| inner.entries[s].snapshot.as_ref())
                    .and_then(|s| s.source());
                let owned = owner.and_then(|owner| {
                    inner.entries.iter().position(|e| {
                        e.available
                            && e.state.as_ref().and_then(|s| s.snapshot_owner()) == Some(owner)
                    })
                });
                selected = owned.or_else(|| {
                    inner
                        .entries
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| e.available)
                        .min_by_key(|(_, e)| e.state_last_used)
                        .map(|(i, _)| i)
                });
            }

            if let Some(selected) = selected {
                // The RAM snapshot under a deeper lower-tier checkpoint.
                let fallback = source
                    .filter(|_| deeper_tier && snapshot_mode)
                    .and_then(|s| {
                        let entry = &inner.entries[s];
                        entry
                            .snapshot
                            .clone()
                            .map(|snapshot| (s, snapshot, entry.tokens.len()))
                    });
                let evicts = (debug_cache && inner.entries[selected].valid).then(|| {
                    let entry = &inner.entries[selected];
                    (entry.tokens.len(), entry.stable_prefix_tokens)
                });
                let clock = inner.tick();
                let entry = &mut inner.entries[selected];
                entry.available = false;
                let needs_invalidation = entry.dirty && !cache_hit;
                entry.dirty = true;
                entry.state_last_used = clock;
                entry.live_tokens.clear();
                entry.live_identity.clear();
                let mut state = entry
                    .state
                    .take()
                    .ok_or_else(|| Error::logic("continuation cache state is missing"))?;
                let mut snapshot = None;
                if snapshot_mode {
                    if cache_hit
                        && !live_hit
                        && let Some(source) = source
                    {
                        let clock = inner.tick();
                        let branches = inner.restore_diverges(source, prompt);
                        let source_entry = &mut inner.entries[source];
                        snapshot = source_entry.snapshot.clone();
                        source_entry.snapshot_last_used = clock;
                        if branches {
                            source_entry.purpose = SnapshotPurpose::BranchPoint;
                        }
                    }
                } else {
                    entry.valid = false;
                    entry.tokens.clear();
                }
                drop(inner);
                if debug_cache {
                    log_skips(&skipped);
                    if let Some((tokens, stable_prefix_tokens)) = evicts {
                        logging::debug(
                            "cache",
                            &format!(
                                "event=capture_evicts index={selected} tokens={tokens} \
                                 stable_prefix_tokens={stable_prefix_tokens} prompt_tokens={}",
                                prompt.len()
                            ),
                        );
                    }
                }

                let prepared = (|| {
                    if needs_invalidation {
                        state.invalidate();
                    }
                    if let Some(prepare) = prepare_state.as_mut() {
                        prepare(&mut state)?;
                    }
                    let mut restored = (0, 0.0);
                    if cache_hit && snapshot_mode && !live_hit {
                        let snapshot = snapshot.as_ref().ok_or_else(|| {
                            Error::runtime("continuation cache snapshot entry is empty")
                        })?;
                        let start = Instant::now();
                        if let Some(restore) = &shared.support.restore {
                            restore(&mut state, snapshot)?;
                        }
                        // The restore replaced the saved position layout:
                        // reattach this request's inputs, which may add
                        // images after the checkpoint.
                        if !input.prefixes.is_empty()
                            && let Some(prepare) = prepare_state.as_mut()
                        {
                            prepare(&mut state)?;
                        }
                        restored = (
                            snapshot.payload_bytes(),
                            start.elapsed().as_secs_f64() * 1e3,
                        );
                    }
                    Ok(restored)
                })();
                return match prepared {
                    Ok((restored_snapshot_bytes, restore_ms)) => Ok(Some(Lease {
                        cache: Some(shared.clone()),
                        state: Some(state),
                        index: selected,
                        cache_hit,
                        cached_tokens: if deeper_tier { 0 } else { cached_tokens },
                        source_index: source.filter(|_| !deeper_tier),
                        fallback,
                        restored_snapshot_bytes,
                        restore_ms,
                        restored_from_disk: false,
                        reserved_snapshot_bytes: 0,
                        snapshot_purpose: SnapshotPurpose::Continuation,
                        prompt_tokens: prompt.len(),
                        stable_prefix_tokens,
                        input_identity: input.complete.to_vec(),
                        input_prefixes: input.prefixes.to_vec(),
                        lookup,
                    })),
                    Err(error) => {
                        state.invalidate();
                        {
                            let mut inner = shared.lock();
                            let clock = inner.tick();
                            let entry = &mut inner.entries[selected];
                            entry.state = Some(state);
                            entry.available = true;
                            entry.dirty = false;
                            entry.state_last_used = clock;
                        }
                        shared.condition.notify_one();
                        Err(error)
                    }
                };
            }

            drop(inner);
            // No state is free: these records answer why a checkpoint was not
            // reused, the question contention raises.
            if debug_cache && !wait_reported && !skipped.is_empty() {
                log_skips(&skipped);
                wait_reported = true;
            }
            if is_cancelled.is_some_and(|cancelled| cancelled()) {
                return Ok(None);
            }
            inner = shared.lock();
            inner = shared
                .condition
                .wait_timeout(inner, CANCELLATION_POLL_INTERVAL)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|e| e.into_inner().0);
        }
    }

    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }

    pub fn snapshot_capacity_bytes(&self) -> usize {
        self.shared.lock().snapshot_capacity_bytes
    }

    pub fn retained_snapshot_bytes(&self) -> usize {
        self.shared.lock().retained_snapshot_bytes
    }

    pub fn reserved_snapshot_bytes(&self) -> usize {
        self.shared.lock().reserved_snapshot_bytes
    }
}
