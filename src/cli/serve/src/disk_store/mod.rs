//! Restart-safe, model-neutral exact-prefix snapshot store
//! (`src/cli/serve/continuation_disk_store.cpp`).
//!
//! Model code owns the compatibility identity and the payload
//! serialisation. The store owns checksums, exact token verification, byte
//! limits, atomic publication, startup indexing, LRU replacement, permissions
//! and collision-safe lookup. Persistence uses one bounded worker; payload
//! writes and fsync run outside the metadata gate, so existing entries stay
//! readable while a snapshot is written.
//!
//! A file `<key sha256>-<16 hex>.kvc` holds, little endian:
//!
//! | Offset | Bytes | Field |
//! | --- | --- | --- |
//! | 0 | 8 | magic `GUFOKVC1` |
//! | 8 | 4 | file version (1) |
//! | 12 | 4 | payload version |
//! | 16 | 4 | identity bytes |
//! | 20 | 4 | token count |
//! | 24 | 8 | payload bytes |
//! | 32 | 64 | lowercase hex SHA-256 of the file with this field zeroed |
//! | 96 | | identity, `u32` tokens, payload |
//!
//! Files are written as `.tmp-<16 hex>`, fsynced, renamed and the directory
//! is fsynced; the directory must be a private directory of the effective
//! user and every file owner-only.

use std::collections::hash_map::RandomState;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::hash::BuildHasher;
use std::io::{ErrorKind as IoErrorKind, Read, Seek, SeekFrom, Write};
use std::os::fd::OwnedFd;
use std::os::unix::ffi::OsStrExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, TryLockError};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use gufo_core::text::{
    Error, Result, TextModelRunner, TextRunnerPersistenceDescriptor, TextRunnerSnapshot,
    TextRunnerState, Token, host_snapshot_budget_bytes,
};
use rustix::fs::{AtFlags, FileType, Mode, OFlags, Stat, Timespec, Timestamps, UTIME_NOW};

use crate::continuation_cache::InputIdentity;
use crate::publication_wait::PendingPublication;
use rustix::io::Errno;
use sha2::{Digest, Sha256};

mod entries;
mod files;
mod format;
mod index;

use format::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuationDiskEventAction {
    Stored,
    Restored,
    Miss,
    Removed,
    Skipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuationDiskEventReason {
    Saved,
    Hit,
    NotFound,
    Unsupported,
    ByteCapacity,
    StagingCapacity,
    Lru,
    ExactReplacement,
    /// A stored prefix already covers this checkpoint within the minimum
    /// step.
    MinStep,
    Corrupt,
    ChecksumMismatch,
    UnsafeFile,
    IoFailure,
    SerializationFailure,
    RestoreFailure,
    Busy,
}

/// Sanitized disk-cache event: prompt contents, token values, paths, model
/// identities and compatibility fingerprints are deliberately absent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContinuationDiskEvent {
    pub action: ContinuationDiskEventAction,
    pub reason: ContinuationDiskEventReason,
    pub file_bytes: usize,
    pub payload_bytes: usize,
    pub token_count: usize,
    pub retained_bytes: usize,
    pub capacity_bytes: usize,
    pub staging_capacity_bytes: usize,
    /// Queued and reserved capture bytes when a snapshot is skipped.
    pub staging_used_bytes: usize,
    pub elapsed_ms: f64,
}

/// The retained disk budget of `--cache-disk`.
pub const DEFAULT_CAPACITY_BYTES: usize = 8 << 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuationDiskStoreOptions {
    pub directory: PathBuf,
    pub capacity_bytes: usize,
    /// RAM for queued snapshots and each disk read; zero selects a quarter
    /// of the host snapshot budget (an eighth of available RAM), at most the
    /// disk budget. Nothing is allocated upfront.
    pub staging_capacity_bytes: usize,
    /// The least advance over the longest stored prefix of a checkpoint
    /// before it is written; zero disables the gate.
    pub min_checkpoint_step_tokens: usize,
}

impl Default for ContinuationDiskStoreOptions {
    fn default() -> Self {
        Self {
            directory: PathBuf::new(),
            capacity_bytes: DEFAULT_CAPACITY_BYTES,
            staging_capacity_bytes: 0,
            min_checkpoint_step_tokens: 0,
        }
    }
}

pub type DiskEventSink = Box<dyn Fn(&ContinuationDiskEvent) + Send + Sync>;
pub type KeyHashFunction = Box<dyn Fn(&[u8]) -> String + Send + Sync>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SaveResult {
    pub stored: bool,
    pub file_bytes: usize,
    pub payload_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RestoreResult {
    pub restored: bool,
    pub token_count: usize,
    pub file_bytes: usize,
    pub payload_bytes: usize,
}

/// Holds disk-only capture memory against the staging budget until it is
/// queued or dropped; may outlive the store.
#[derive(Debug)]
pub struct CaptureReservation {
    counter: Arc<AtomicUsize>,
    bytes: usize,
}

impl Drop for CaptureReservation {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

const MAGIC: &[u8; 8] = b"GUFOKVC1";
const FILE_VERSION: u32 = 1;
const CHECKSUM_BYTES: usize = 64;
const FILE_VERSION_OFFSET: usize = 8;
const PAYLOAD_VERSION_OFFSET: usize = 12;
const IDENTITY_BYTES_OFFSET: usize = 16;
const TOKEN_COUNT_OFFSET: usize = 20;
const PAYLOAD_BYTES_OFFSET: usize = 24;
const CHECKSUM_OFFSET: usize = 32;
const HEADER_BYTES: usize = CHECKSUM_OFFSET + CHECKSUM_BYTES;
const MAX_IDENTITY_BYTES: usize = 64 * 1024;
const MAX_TOKEN_COUNT: usize = 16 * 1024 * 1024;
const FILE_SUFFIX: &str = ".kvc";
const TEMPORARY_PREFIX: &str = ".tmp-";

/// A held `flock` on the cache directory, released on drop.
struct DirectoryLock<'a>(&'a OwnedFd);

impl<'a> DirectoryLock<'a> {
    /// `None` when `operation` fails or, non-blocking, would wait.
    fn acquire(directory: &'a OwnedFd, operation: rustix::fs::FlockOperation) -> Option<Self> {
        loop {
            match rustix::fs::flock(directory, operation) {
                Ok(()) => return Some(Self(directory)),
                Err(rustix::io::Errno::INTR) => {}
                Err(_) => return None,
            }
        }
    }
}

impl Drop for DirectoryLock<'_> {
    fn drop(&mut self) {
        while rustix::fs::flock(self.0, rustix::fs::FlockOperation::Unlock)
            == Err(rustix::io::Errno::INTR)
        {}
    }
}
/// Bounded write size; each byte is checksummed exactly once.
const WRITE_CHUNK_BYTES: usize = 1024 * 1024;

/// Lowercase hex SHA-256, the default key hash and the file checksum.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(digest: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(digest.len() * 2);
    for &byte in digest {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    text
}

fn is_lower_hex_digest(digest: &[u8]) -> bool {
    digest.len() == CHECKSUM_BYTES
        && digest
            .iter()
            .all(|&c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The metadata gate taken without waiting: lookups report "busy" rather
/// than stall behind a writer (`ScopedOperationPermit(gate, false)`).
fn try_lock<T>(mutex: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

/// `strerror`-style text after `operation`.
fn errno_message(operation: &str, error: Errno) -> String {
    let text = std::io::Error::from(error).to_string();
    let reason = text.split(" (os error").next().unwrap_or(&text);
    format!("{operation}: {reason}")
}

/// Nanoseconds since the Unix epoch, the recency clock of entries.
type FileTime = i128;

fn now() -> FileTime {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => i128::try_from(elapsed.as_nanos()).unwrap_or(i128::MAX),
        Err(before) => -i128::try_from(before.duration().as_nanos()).unwrap_or(i128::MAX),
    }
}

fn modification_time(status: &Stat) -> FileTime {
    i128::from(status.st_mtime) * 1_000_000_000 + i128::from(status.st_mtime_nsec)
}

type EntryId = u64;

/// A token radix tree over entries of one persistence key. Compressed edges
/// keep lookup proportional to the prompt length; the full compatibility
/// bytes and tokens stay the keys, never only a hash.
#[derive(Default)]
struct PrefixNode {
    edge: Vec<Token>,
    entry: Option<EntryId>,
    children: BTreeMap<Token, PrefixNode>,
}

#[derive(Default)]
struct PrefixMatch {
    longest: Option<EntryId>,
    shared_boundaries: Vec<usize>,
}

struct Entry {
    filename: OsString,
    key_hash: String,
    persistence: TextRunnerPersistenceDescriptor,
    tokens: Vec<Token>,
    file_bytes: usize,
    payload_bytes: usize,
    last_access: FileTime,
}

type PersistenceKey = (u32, Vec<u8>);

/// Entries and their indexes, guarded by the metadata gate.
#[derive(Default)]
struct Metadata {
    entries: HashMap<EntryId, Entry>,
    next_id: EntryId,
    /// Key hash to entries; a multimap because collisions stay possible.
    index: HashMap<String, Vec<EntryId>>,
    prefixes: BTreeMap<PersistenceKey, PrefixNode>,
}

type SaveJob = Box<dyn FnOnce(&Inner) -> Result<SaveResult> + Send>;

struct PendingSave {
    job: SaveJob,
    token_count: usize,
    retained_bytes: usize,
    publication: Publication,
}

/// What a queued or running save will make restorable.
#[derive(Clone)]
struct Publication {
    id: u64,
    tokens: Arc<[Token]>,
    input_identity: Arc<[u8]>,
}

#[derive(Default)]
struct Queue {
    pending: VecDeque<PendingSave>,
    /// The save the writer runs now.
    in_flight: Option<Publication>,
    next_id: u64,
    queued_bytes: usize,
    stopping: bool,
}

struct Inner {
    options: ContinuationDiskStoreOptions,
    event_sink: Option<DiskEventSink>,
    key_hash: Option<KeyHashFunction>,
    directory: OwnedFd,
    /// The metadata gate: writers hold it except while publishing a file,
    /// lookups only try it.
    metadata: Mutex<Metadata>,
    /// Serialises writers; holds the file-name counter.
    writer: Mutex<u64>,
    retained: AtomicUsize,
    retained_entries: AtomicUsize,
    queue: Mutex<Queue>,
    queue_changed: Condvar,
    capture_bytes: Arc<AtomicUsize>,
}

/// A fresh file-name suffix: 16 hex digits from a randomly keyed hash of a
/// counter, the time and the process ID.
fn unique_suffix(counter: &mut u64) -> String {
    *counter += 1;
    let value = RandomState::new().hash_one((*counter, now(), std::process::id()));
    format!("{value:016x}")
}

impl Inner {
    fn hash_key(
        &self,
        persistence: &TextRunnerPersistenceDescriptor,
        tokens: &[Token],
    ) -> Result<String> {
        self.hash(&make_key_bytes(persistence, tokens)?)
    }

    fn remove_file_only(&self, filename: &OsStr) {
        if !filename.is_empty() {
            let _ = rustix::fs::unlinkat(&self.directory, filename, AtFlags::empty());
        }
    }

    fn evict_to_capacity(&self, metadata: &mut Metadata) {
        while self.retained.load(Ordering::Relaxed) > self.options.capacity_bytes {
            let Some(victim) = metadata.least_recently_used() else {
                break;
            };
            if !self.remove_entry(metadata, victim, ContinuationDiskEventReason::Lru) {
                break;
            }
        }
    }

    /// The staging budget shared by queued writes and in-flight captures.
    fn staging_limit(&self) -> usize {
        self.options
            .capacity_bytes
            .min(self.options.staging_capacity_bytes)
    }
}

/// Restart-safe, model-neutral exact-prefix snapshot store.
pub struct ContinuationDiskStore {
    inner: Arc<Inner>,
    writer: Option<JoinHandle<()>>,
}

impl ContinuationDiskStore {
    /// Opens (creating) the private directory, removes interrupted and
    /// invalid files, indexes the rest, evicts to capacity and starts the
    /// writer.
    pub fn new(
        options: ContinuationDiskStoreOptions,
        event_sink: Option<DiskEventSink>,
        key_hash: Option<KeyHashFunction>,
    ) -> Result<Self> {
        if options.directory.as_os_str().is_empty() {
            return Err(Error::invalid(
                "continuation disk directory must not be empty",
            ));
        }
        if options.capacity_bytes == 0 {
            return Err(Error::invalid("continuation disk capacity must be nonzero"));
        }
        let mut options = options;
        if options.staging_capacity_bytes == 0 {
            // The host snapshot budget already leaves half of the available
            // RAM free; staging takes a quarter of it. It allocates nothing
            // upfront, so a fixed cap would only refuse checkpoints larger
            // than it.
            options.staging_capacity_bytes =
                options.capacity_bytes.min(host_snapshot_budget_bytes() / 4);
        }
        if options.staging_capacity_bytes < HEADER_BYTES {
            return Err(Error::invalid(
                "continuation disk staging capacity is too small",
            ));
        }
        let directory = Inner::initialize_directory(&options)?;
        let inner = Arc::new(Inner {
            options,
            event_sink,
            key_hash,
            directory,
            metadata: Mutex::new(Metadata::default()),
            writer: Mutex::new(0),
            retained: AtomicUsize::new(0),
            retained_entries: AtomicUsize::new(0),
            queue: Mutex::new(Queue::default()),
            queue_changed: Condvar::new(),
            capture_bytes: Arc::new(AtomicUsize::new(0)),
        });
        {
            let mut metadata = lock(&inner.metadata);
            inner.index_existing_files(&mut metadata)?;
            inner.evict_to_capacity(&mut metadata);
        }
        let worker = Arc::clone(&inner);
        let writer = std::thread::Builder::new()
            .name("gufo-disk-cache".into())
            .spawn(move || worker.persist_queued())
            .map_err(|_| Error::runtime("failed to start the continuation disk writer"))?;
        Ok(Self {
            inner,
            writer: Some(writer),
        })
    }

    /// Stores `snapshot` for `checkpoint_tokens` synchronously.
    pub fn save<R: TextModelRunner>(
        &self,
        runner: &R,
        checkpoint_tokens: &[Token],
        snapshot: &R::Snapshot,
        input_identity: &[u8],
    ) -> Result<SaveResult> {
        self.inner
            .save(runner, checkpoint_tokens, snapshot, input_identity, false)
    }

    /// True when a stored prefix lies less than the minimum step before
    /// `tokens`, so saving them as a continuation would be skipped: callers
    /// need not capture a snapshot only to have it discarded.
    pub fn within_checkpoint_step<R: TextModelRunner>(
        &self,
        runner: &R,
        tokens: &[Token],
        input_identity: &[u8],
    ) -> bool {
        try_lock(&self.inner.metadata).is_some_and(|metadata| {
            self.inner
                .within_checkpoint_step(&metadata, runner, tokens, input_identity)
        })
    }

    /// Reserves staging for a disk-only capture before it allocates.
    pub fn reserve_capture<R: TextModelRunner>(
        &self,
        runner: &R,
        token_count: usize,
        snapshot_bytes: usize,
        input_identity: &[u8],
    ) -> Result<Option<CaptureReservation>> {
        let Some(persistence) = persistence_for_input(runner, input_identity) else {
            return Ok(None);
        };
        if token_count == 0 || snapshot_bytes == 0 {
            return Ok(None);
        }
        let bytes = checked_file_bytes(
            persistence.compatibility_identity.len(),
            token_count,
            snapshot_bytes,
        )?;
        let queue = lock(&self.inner.queue);
        if !self
            .inner
            .admits(&queue, bytes, snapshot_bytes, token_count)
        {
            return Ok(None);
        }
        self.inner.capture_bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(Some(CaptureReservation {
            counter: Arc::clone(&self.inner.capture_bytes),
            bytes,
        }))
    }

    /// Conservative admission before capturing a snapshot, counting queued
    /// and in-flight bytes; `false` means a harmless cache skip.
    pub fn can_save<R: TextModelRunner>(
        &self,
        runner: &R,
        token_count: usize,
        snapshot_bytes: usize,
        input_identity: &[u8],
    ) -> Result<bool> {
        let Some(persistence) = persistence_for_input(runner, input_identity) else {
            return Ok(false);
        };
        if token_count == 0 || snapshot_bytes == 0 {
            return Ok(false);
        }
        let bytes = checked_file_bytes(
            persistence.compatibility_identity.len(),
            token_count,
            snapshot_bytes,
        )?;
        let queue = lock(&self.inner.queue);
        Ok(self
            .inner
            .admits(&queue, bytes, snapshot_bytes, token_count))
    }

    /// Queues an immutable snapshot for the writer, keeping the runner and
    /// the snapshot alive until it is written. Returns the file bytes queued
    /// (not yet written), or 0 when the queue is full. The charge is the
    /// larger of the in-memory and the serialised size.
    pub fn save_async<R: TextModelRunner>(
        &self,
        runner: Arc<R>,
        checkpoint_tokens: Vec<Token>,
        snapshot: Arc<R::Snapshot>,
        input_identity: Vec<u8>,
        reservation: Option<CaptureReservation>,
        shared_prefix: bool,
    ) -> Result<usize> {
        if checkpoint_tokens.is_empty() {
            return Ok(0);
        }
        let Some(persistence) = persistence_for_input(&*runner, &input_identity) else {
            return Ok(0);
        };
        let identity_bytes = persistence.compatibility_identity.len();
        let persistent_bytes = runner.persistent_snapshot_payload_bytes(&snapshot)?;
        let file_bytes =
            checked_file_bytes(identity_bytes, checkpoint_tokens.len(), persistent_bytes)?;
        let charge = checked_file_bytes(
            identity_bytes,
            checkpoint_tokens.len(),
            snapshot.payload_bytes().max(persistent_bytes),
        )?;
        {
            let mut queue = lock(&self.inner.queue);
            if let Some(reservation) = reservation {
                if !Arc::ptr_eq(&reservation.counter, &self.inner.capture_bytes) {
                    return Ok(0);
                }
                drop(reservation);
            }
            if !self.inner.admits(
                &queue,
                charge,
                snapshot.payload_bytes(),
                checkpoint_tokens.len(),
            ) {
                return Ok(0);
            }
            let token_count = checkpoint_tokens.len();
            queue.next_id += 1;
            let publication = Publication {
                id: queue.next_id,
                tokens: checkpoint_tokens.as_slice().into(),
                input_identity: input_identity.as_slice().into(),
            };
            let job: SaveJob = Box::new(move |inner: &Inner| {
                inner.save(
                    &*runner,
                    &checkpoint_tokens,
                    &*snapshot,
                    &input_identity,
                    shared_prefix,
                )
            });
            queue.pending.push_back(PendingSave {
                job,
                token_count,
                retained_bytes: charge,
                publication,
            });
            queue.queued_bytes += charge;
        }
        self.inner.queue_changed.notify_one();
        Ok(file_bytes)
    }

    /// The deepest queued or running save that `prompt` extends with the
    /// same input identity (`PendingPublication`), the lower ID on equal
    /// lengths. A first query (`checkpoint` `None`) wants one saving more
    /// than [`PendingPublication::MINIMUM_GAIN_TOKENS`] over the
    /// `available_tokens` a lookup reuses now; a follow-up asks whether that
    /// checkpoint is still pending.
    pub fn pending_publication(
        &self,
        prompt: &[Token],
        input: InputIdentity<'_>,
        available_tokens: usize,
        checkpoint: Option<u64>,
    ) -> Option<PendingPublication> {
        let queue = lock(&self.inner.queue);
        queue
            .in_flight
            .iter()
            .chain(queue.pending.iter().map(|pending| &pending.publication))
            .filter(|publication| checkpoint.is_none_or(|id| publication.id == id))
            .filter(|publication| {
                prompt.starts_with(&publication.tokens)
                    && *publication.input_identity == *input.at(publication.tokens.len())
            })
            .map(|publication| PendingPublication {
                checkpoint_id: publication.id,
                available_tokens,
                pending_tokens: publication.tokens.len(),
            })
            .filter(|pending| {
                checkpoint.is_some() || pending.gain() > PendingPublication::MINIMUM_GAIN_TOKENS
            })
            .max_by(|a, b| {
                a.pending_tokens
                    .cmp(&b.pending_tokens)
                    .then(b.checkpoint_id.cmp(&a.checkpoint_id))
            })
    }

    /// Waits until every accepted write finished; dropping the store drains
    /// too.
    pub fn flush(&self) {
        let mut queue = lock(&self.inner.queue);
        while queue.queued_bytes != 0 {
            queue = self
                .inner
                .queue_changed
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Restores the longest exact saved prefix of `prompt` into `state`,
    /// saved with the input identity at its length; a busy store is a miss.
    /// With a nonzero stable boundary, a checkpoint past it needs a saved
    /// prefix at or before it too, so full-prompt files of older servers
    /// cannot bypass the migration.
    /// The token count of the longest entry
    /// [`restore_longest_prefix`](Self::restore_longest_prefix) would try
    /// first, from metadata alone; 0 without one or while the store is busy.
    pub fn longest_prefix_tokens<R: TextModelRunner>(
        &self,
        runner: &R,
        prompt: &[Token],
        input: InputIdentity<'_>,
        stable_prefix_tokens: usize,
    ) -> usize {
        if stable_prefix_tokens > prompt.len()
            || input.validate(prompt.len()).is_err()
            || persistence_for_input(runner, input.complete).is_none()
        {
            return 0;
        }
        let Some(metadata) = try_lock(&self.inner.metadata) else {
            return 0;
        };
        Inner::usable_longest(&metadata, runner, prompt, input, stable_prefix_tokens)
            .and_then(|candidate| metadata.entries.get(&candidate))
            .map_or(0, |entry| entry.tokens.len())
    }

    /// Restores the longest stored prefix of `prompt` deeper than
    /// `deeper_than` tokens, which a RAM checkpoint already covers.
    pub fn restore_longest_prefix<R: TextModelRunner>(
        &self,
        runner: &R,
        state: &mut R::State,
        prompt: &[Token],
        input: InputIdentity<'_>,
        stable_prefix_tokens: usize,
        deeper_than: usize,
    ) -> Result<RestoreResult> {
        if stable_prefix_tokens > prompt.len() {
            return Err(Error::invalid("stable cache prefix exceeds prompt length"));
        }
        input.validate(prompt.len())?;
        let Some(mut metadata) = try_lock(&self.inner.metadata) else {
            self.inner.emit(
                ContinuationDiskEventAction::Miss,
                ContinuationDiskEventReason::Busy,
                0,
                0,
                prompt.len(),
                0.0,
            );
            return Ok(RestoreResult::default());
        };
        Ok(self.inner.restore_longest_prefix(
            &mut metadata,
            runner,
            state,
            prompt,
            input,
            stable_prefix_tokens,
            deeper_than,
        ))
    }

    /// Prefix lengths `prompt` shares with stored entries of the same
    /// compatibility identity that no entry holds exactly: at least
    /// `min_tokens`, below the prompt length, the longest `max_boundaries`
    /// in ascending order. Snapshots there make a prefix shared across
    /// conversations (system prompt, tool schemas) restorable next time.
    pub fn shared_prefix_boundaries<R: TextModelRunner>(
        &self,
        runner: &R,
        prompt: &[Token],
        min_tokens: usize,
        max_boundaries: usize,
        input: InputIdentity<'_>,
    ) -> Result<Vec<usize>> {
        input.validate(prompt.len())?;
        let Some(metadata) = try_lock(&self.inner.metadata) else {
            return Ok(Vec::new());
        };
        Ok(self.inner.shared_prefix_boundaries(
            &metadata,
            runner,
            prompt,
            min_tokens,
            max_boundaries,
            input,
        ))
    }

    /// Marks the exact entry for `tokens` as recently used without reading
    /// it; `false` when there is none (or the store is busy).
    pub fn touch<R: TextModelRunner>(
        &self,
        runner: &R,
        tokens: &[Token],
        input_identity: &[u8],
    ) -> Result<bool> {
        let Some(mut metadata) = try_lock(&self.inner.metadata) else {
            return Ok(false);
        };
        self.inner
            .touch(&mut metadata, runner, tokens, input_identity)
    }

    pub fn entry_count(&self) -> usize {
        self.inner.retained_entries.load(Ordering::Relaxed)
    }

    pub fn retained_bytes(&self) -> usize {
        self.inner.retained.load(Ordering::Relaxed)
    }

    pub fn capacity_bytes(&self) -> usize {
        self.inner.options.capacity_bytes
    }

    pub fn staging_capacity_bytes(&self) -> usize {
        self.inner.options.staging_capacity_bytes
    }
}

impl Drop for ContinuationDiskStore {
    fn drop(&mut self) {
        lock(&self.inner.queue).stopping = true;
        self.inner.queue_changed.notify_all();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn header_layout_matches_the_format() {
        let persistence = TextRunnerPersistenceDescriptor {
            compatibility_identity: b"id".to_vec(),
            payload_version: 7,
        };
        let header = build_header(&persistence, &[1, 0x0102_0304], 16).expect("header");
        assert_eq!(header.len(), HEADER_BYTES + 2 + 8);
        assert_eq!(&header[..8], b"GUFOKVC1");
        assert_eq!(le_u32(&header, 8), 1);
        assert_eq!(le_u32(&header, 12), 7);
        assert_eq!(le_u32(&header, 16), 2);
        assert_eq!(le_u32(&header, 20), 2);
        assert_eq!(le_u64(&header, 24), 16);
        assert!(header[32..96].iter().all(|&b| b == 0));
        assert_eq!(&header[96..98], b"id");
        assert_eq!(&header[98..106], &[1, 0, 0, 0, 4, 3, 2, 1]);
    }

    #[test]
    fn key_bytes_follow_the_cpp_layout() {
        let persistence = TextRunnerPersistenceDescriptor {
            compatibility_identity: b"ab".to_vec(),
            payload_version: 3,
        };
        let key = make_key_bytes(&persistence, &[5]).expect("key");
        assert_eq!(
            key,
            [2, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, b'a', b'b', 5, 0, 0, 0]
        );
    }

    #[test]
    fn radix_tree_splits_edges_and_reports_boundaries() {
        let mut root = PrefixNode::default();
        root.insert(&[7, 1, 8, 9], 1);
        root.insert(&[7, 2, 8, 9], 2);
        root.insert(&[7, 1], 3);
        assert_eq!(root.find(&[7, 1, 8, 9, 10], false).longest, Some(1));
        assert_eq!(root.find(&[7, 1, 5], false).longest, Some(3));
        assert_eq!(root.find(&[7, 3, 8], true).shared_boundaries, vec![1]);
        root.erase(&[7, 1]);
        assert_eq!(root.find(&[7, 1, 5], false).longest, None);
        root.erase(&[7, 1, 8, 9]);
        root.erase(&[7, 2, 8, 9]);
        assert!(root.is_empty());
    }
}
