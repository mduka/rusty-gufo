//! The C++ `tests/cli/continuation_disk_store_test.cpp`, case by case.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    ChatRequest, ContinuationInputPrefix, Error, ErrorKind, NoPromptContext, Result, Sampler,
    TextDecodeSelection, TextExecutionPlan, TextModelRunner, TextPrefillStep,
    TextRunnerCapabilities, TextRunnerDescriptor, TextRunnerPersistenceDescriptor,
    TextRunnerResourceClaim, TextRunnerSnapshot, TextRunnerState, Token,
};
use gufo_serve::continuation_cache::InputIdentity;
use gufo_serve::disk_store::{
    ContinuationDiskEvent, ContinuationDiskEventReason, ContinuationDiskStore,
    ContinuationDiskStoreOptions, DiskEventSink, KeyHashFunction, RestoreResult, SaveResult,
    sha256_hex,
};

const FAKE_PAYLOAD_BYTES: usize = 16;
const DISK_HEADER_BYTES: usize = 96;
/// `sizeof(FakeSnapshot)` in C++: a vtable pointer and two `u64`.
const CPP_SIZEOF_FAKE_SNAPSHOT: usize = 24;

#[derive(Default)]
struct FakeState {
    value: u64,
    position: u64,
}

impl TextRunnerState for FakeState {
    fn invalidate(&mut self) {
        self.value = 0;
        self.position = 0;
    }
}

struct FakeSnapshot {
    value: u64,
    position: u64,
}

impl TextRunnerSnapshot for FakeSnapshot {
    fn payload_bytes(&self) -> usize {
        FAKE_PAYLOAD_BYTES
    }
}

#[derive(Clone)]
struct FakeSampler;

impl Sampler for FakeSampler {
    fn accept(&mut self, _: Token) {}
}

type Hook = Arc<dyn Fn() + Send + Sync>;

struct FakeRunner {
    identity: Vec<u8>,
    model_id: String,
    before_stream: Mutex<Option<Hook>>,
}

impl FakeRunner {
    fn new(identity: &str) -> Self {
        Self::with_model(identity, "fake-model")
    }

    fn with_model(identity: &str, model_id: &str) -> Self {
        Self {
            identity: identity.as_bytes().to_vec(),
            model_id: model_id.into(),
            before_stream: Mutex::new(None),
        }
    }

    fn set_before_stream(&self, hook: Option<Hook>) {
        *self.before_stream.lock().expect("hook") = hook;
    }
}

fn put_u64(destination: &mut [u8], offset: usize, value: u64) {
    destination[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u64(source: &[u8], offset: usize) -> u64 {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&source[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

impl TextModelRunner for FakeRunner {
    type State = FakeState;
    type Snapshot = FakeSnapshot;
    type Sampler = FakeSampler;
    type PromptContext = NoPromptContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        TextRunnerDescriptor {
            model_id: self.model_id.clone(),
            state_abi: "fake-state-v1".into(),
            max_context: 4096,
            capabilities: TextRunnerCapabilities {
                incremental_prefill: true,
                snapshot: true,
                fork: true,
                ..TextRunnerCapabilities::default()
            },
            persistence: Some(TextRunnerPersistenceDescriptor {
                compatibility_identity: self.identity.clone(),
                payload_version: 7,
            }),
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        TextRunnerResourceClaim::default()
    }

    fn supported_plans(&self) -> Vec<TextExecutionPlan> {
        vec![TextExecutionPlan::SERIAL]
    }

    fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        Ok(text.bytes().map(Token::from).collect())
    }

    fn render_and_tokenize(&self, _: &ChatRequest) -> Result<Option<Vec<Token>>> {
        Ok(None)
    }

    fn decode(&self, _: &[Token]) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }

    fn create_state(&self) -> Result<FakeState> {
        Ok(FakeState::default())
    }

    fn create_sampler(&self, _: &SamplingConfig, _: &[Token]) -> Result<FakeSampler> {
        Ok(FakeSampler)
    }

    fn prefill(
        &self,
        _: &mut FakeState,
        _: &[Token],
        _: usize,
        _: usize,
    ) -> Result<TextPrefillStep> {
        Err(Error::logic("fake persistence test does not prefill"))
    }

    fn select_next(&self, _: &mut FakeState, _: &mut FakeSampler) -> Result<TextDecodeSelection> {
        Err(Error::logic("fake persistence test does not decode"))
    }

    fn advance(&self, _: &mut FakeState, _: Token) -> Result<()> {
        Err(Error::logic("fake persistence test does not advance"))
    }

    fn checkpoint_position(&self, state: &FakeState) -> Result<usize> {
        Ok(state.position as usize)
    }

    fn snapshot_payload_bytes(&self, _: &FakeState) -> Result<usize> {
        Ok(FAKE_PAYLOAD_BYTES)
    }

    fn snapshot(&self, state: &FakeState) -> Result<FakeSnapshot> {
        Ok(FakeSnapshot {
            value: state.value,
            position: state.position,
        })
    }

    fn restore_or_fork(&self, state: &mut FakeState, snapshot: &FakeSnapshot) -> Result<()> {
        state.value = snapshot.value;
        state.position = snapshot.position;
        Ok(())
    }

    fn persistent_snapshot_payload_bytes(&self, _: &FakeSnapshot) -> Result<usize> {
        Ok(FAKE_PAYLOAD_BYTES)
    }

    fn serialize_persistent_snapshot(
        &self,
        snapshot: &FakeSnapshot,
        destination: &mut [u8],
    ) -> Result<usize> {
        if destination.len() != FAKE_PAYLOAD_BYTES {
            return Err(Error::invalid("fake destination size mismatch"));
        }
        put_u64(destination, 0, snapshot.value);
        put_u64(destination, 8, snapshot.position);
        Ok(destination.len())
    }

    fn stream_persistent_snapshot(
        &self,
        snapshot: &FakeSnapshot,
        sink: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let hook = self.before_stream.lock().expect("hook").clone();
        if let Some(hook) = hook {
            hook();
        }
        let mut bytes = [0; FAKE_PAYLOAD_BYTES];
        self.serialize_persistent_snapshot(snapshot, &mut bytes)?;
        sink(&bytes[..7])?;
        sink(&bytes[7..])
    }

    fn restore_persistent_snapshot(&self, state: &mut FakeState, payload: &[u8]) -> Result<()> {
        if payload.len() != FAKE_PAYLOAD_BYTES {
            return Err(Error::invalid("fake payload size mismatch"));
        }
        state.value = get_u64(payload, 0);
        state.position = get_u64(payload, 8);
        Ok(())
    }
}

/// A counting semaphore (`std::binary_semaphore` in C++).
#[derive(Default)]
struct Semaphore {
    count: Mutex<usize>,
    changed: Condvar,
}

impl Semaphore {
    fn release(&self) {
        *self.count.lock().expect("semaphore") += 1;
        self.changed.notify_all();
    }

    fn acquire(&self) {
        let mut count = self.count.lock().expect("semaphore");
        while *count == 0 {
            count = self.changed.wait(count).expect("semaphore");
        }
        *count -= 1;
    }

    fn try_acquire_for(&self, timeout: Duration) -> bool {
        let count = self.count.lock().expect("semaphore");
        let (mut count, _) = self
            .changed
            .wait_timeout_while(count, timeout, |count| *count == 0)
            .expect("semaphore");
        if *count == 0 {
            return false;
        }
        *count -= 1;
        true
    }
}

fn make_snapshot(runner: &FakeRunner, value: u64, position: u64) -> FakeSnapshot {
    let state = FakeState { value, position };
    runner.snapshot(&state).expect("snapshot")
}

fn store_options(
    directory: &Path,
    capacity_bytes: usize,
    staging_bytes: usize,
) -> ContinuationDiskStoreOptions {
    ContinuationDiskStoreOptions {
        directory: directory.to_path_buf(),
        capacity_bytes,
        staging_capacity_bytes: staging_bytes,
        min_checkpoint_step_tokens: 0,
    }
}

fn default_options(directory: &Path) -> ContinuationDiskStoreOptions {
    store_options(directory, 1024 * 1024, 1024 * 1024)
}

fn open(options: ContinuationDiskStoreOptions) -> ContinuationDiskStore {
    ContinuationDiskStore::new(options, None, None).expect("store")
}

type Events = Arc<Mutex<Vec<ContinuationDiskEvent>>>;

fn recording(options: ContinuationDiskStoreOptions) -> (ContinuationDiskStore, Events) {
    let events = Events::default();
    let sink_events = Arc::clone(&events);
    let sink: DiskEventSink =
        Box::new(move |event| sink_events.lock().expect("events").push(*event));
    let store = ContinuationDiskStore::new(options, Some(sink), None).expect("store");
    (store, events)
}

fn has_reason(events: &Events, reason: ContinuationDiskEventReason) -> bool {
    events
        .lock()
        .expect("events")
        .iter()
        .any(|e| e.reason == reason)
}

fn cache_files(directory: &Path) -> Vec<PathBuf> {
    fs::read_dir(directory)
        .expect("directory")
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().is_some_and(|e| e == "kvc"))
        .collect()
}

fn expected_file_bytes(identity_bytes: usize, token_count: usize) -> usize {
    DISK_HEADER_BYTES + identity_bytes + token_count * 4 + FAKE_PAYLOAD_BYTES
}

fn save_tokens(
    store: &ContinuationDiskStore,
    runner: &FakeRunner,
    tokens: &[Token],
    snapshot: &FakeSnapshot,
) -> SaveResult {
    store.save(runner, tokens, snapshot, &[]).expect("save")
}

fn restore_tokens(
    store: &ContinuationDiskStore,
    runner: &FakeRunner,
    state: &mut FakeState,
    tokens: &[Token],
) -> RestoreResult {
    store
        .restore_longest_prefix(runner, state, tokens, InputIdentity::default(), 0, 0)
        .expect("restore")
}

fn temporary_directory() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("gufo-continuation-disk-")
        .tempdir()
        .expect("temporary directory")
}

fn flip_last_byte(path: &Path) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open");
    file.seek(SeekFrom::End(-1)).expect("seek");
    let mut value = [0];
    file.read_exact(&mut value).expect("read");
    value[0] ^= 0x5A;
    file.seek(SeekFrom::End(-1)).expect("seek");
    file.write_all(&value).expect("write");
}

#[test]
fn sha256_known_vector() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "shared SHA-256 implementation matches the standard vector"
    );
}

#[test]
fn restart_restore_and_compatibility_identity() {
    let directory = temporary_directory();
    let writer = FakeRunner::with_model("artifact-A", "fake-old-executable");
    let snapshot = make_snapshot(&writer, 0x1234_5678_9ABC_DEF0, 3);
    {
        let store = open(default_options(directory.path()));
        let saved = save_tokens(&store, &writer, &[1, 2, 3], &snapshot);
        assert!(
            saved.stored && saved.payload_bytes == FAKE_PAYLOAD_BYTES,
            "first process atomically stores a persistent snapshot"
        );
        assert!(
            store.entry_count() == 1 && store.retained_bytes() == saved.file_bytes,
            "store accounts exact retained file bytes"
        );
    }

    let compatible_reader = FakeRunner::with_model("artifact-A", "fake-new-executable");
    let mut compatible_state = FakeState::default();
    let restarted = open(default_options(directory.path()));
    let restored = restore_tokens(
        &restarted,
        &compatible_reader,
        &mut compatible_state,
        &[1, 2, 3, 4, 5],
    );
    assert!(
        restored.restored && restored.token_count == 3,
        "new executable restores the longest compatible saved prefix"
    );
    assert!(
        compatible_state.value == 0x1234_5678_9ABC_DEF0 && compatible_state.position == 3,
        "restart restore reconstructs exact model-owned state"
    );

    let incompatible_reader = FakeRunner::new("artifact-B");
    let mut incompatible_state = FakeState::default();
    assert!(
        !restore_tokens(
            &restarted,
            &incompatible_reader,
            &mut incompatible_state,
            &[1, 2, 3, 4]
        )
        .restored,
        "different compatibility identity is a deterministic cache miss"
    );
}

#[test]
fn longest_prefix_and_forced_hash_collision() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("collision-model");
    let forced_collision: KeyHashFunction = Box::new(|_| "0".repeat(64));
    let store = ContinuationDiskStore::new(
        default_options(directory.path()),
        None,
        Some(forced_collision),
    )
    .expect("store");

    let short_snapshot = make_snapshot(&runner, 11, 2);
    let long_snapshot = make_snapshot(&runner, 22, 4);
    let unrelated_snapshot = make_snapshot(&runner, 33, 3);
    assert!(
        save_tokens(&store, &runner, &[1, 2], &short_snapshot).stored,
        "collision bucket stores first key"
    );
    assert!(
        save_tokens(&store, &runner, &[1, 2, 3, 4], &long_snapshot).stored,
        "collision bucket stores longer exact key"
    );
    assert!(
        save_tokens(&store, &runner, &[9, 8, 7], &unrelated_snapshot).stored,
        "collision bucket stores unrelated exact key"
    );

    let mut state = FakeState::default();
    let restored = restore_tokens(&store, &runner, &mut state, &[1, 2, 3, 4, 5]);
    assert!(
        restored.restored && restored.token_count == 4 && state.value == 22,
        "full token verification selects longest exact key after collision"
    );

    let mut unrelated_state = FakeState::default();
    assert!(
        restore_tokens(&store, &runner, &mut unrelated_state, &[9, 8, 7, 6]).restored
            && unrelated_state.value == 33,
        "forced collision cannot return another prompt's state"
    );

    let mut miss_state = FakeState::default();
    assert!(
        !restore_tokens(&store, &runner, &mut miss_state, &[5, 5, 5]).restored,
        "forced collision without exact tokens remains a miss"
    );
}

#[test]
fn corruption_becomes_deterministic_miss_and_removal() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("corruption-model");
    let (store, events) = recording(default_options(directory.path()));
    let snapshot = make_snapshot(&runner, 44, 3);
    assert!(
        save_tokens(&store, &runner, &[4, 4, 4], &snapshot).stored,
        "corruption test stores an entry"
    );
    let files = cache_files(directory.path());
    assert_eq!(files.len(), 1, "one published cache file exists");
    flip_last_byte(&files[0]);

    if rustix::process::geteuid().as_raw() != 0 {
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o500))
            .expect("cache directory can simulate an unlink failure");
        let (returned, result) = std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            let (store, runner) = (&store, &runner);
            scope.spawn(move || {
                let mut state = FakeState::default();
                let result = restore_tokens(store, runner, &mut state, &[4, 4, 4, 5]);
                let _ = sender.send(result);
            });
            let returned = receiver.recv_timeout(Duration::from_secs(1));
            fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))
                .expect("cache directory permissions restored");
            match returned {
                Ok(result) => (true, result),
                Err(_) => (false, receiver.recv().expect("lookup result")),
            }
        });
        assert!(
            returned && !result.restored && store.entry_count() == 1,
            "unremovable corrupt entries miss without an infinite retry"
        );
    }

    let mut state = FakeState::default();
    assert!(
        !restore_tokens(&store, &runner, &mut state, &[4, 4, 4, 5]).restored,
        "checksum-invalid entry becomes a cache miss"
    );
    assert!(
        store.entry_count() == 0 && cache_files(directory.path()).is_empty(),
        "checksum-invalid entry is removed from index and disk"
    );
    assert!(
        has_reason(&events, ContinuationDiskEventReason::ChecksumMismatch),
        "sanitized event reports checksum removal reason"
    );
}

#[test]
fn byte_and_staging_limits() {
    let staging_directory = temporary_directory();
    let runner = FakeRunner::new("limits");
    let file_bytes = expected_file_bytes(6, 2);
    let snapshot = make_snapshot(&runner, 1, 2);

    let staging_limited = open(store_options(
        staging_directory.path(),
        4096,
        file_bytes - 1,
    ));
    assert!(
        !save_tokens(&staging_limited, &runner, &[1, 2], &snapshot).stored
            && staging_limited.entry_count() == 0,
        "full file image must fit the bounded staging allocation"
    );

    let byte_directory = temporary_directory();
    let byte_limited = open(store_options(byte_directory.path(), file_bytes - 1, 4096));
    assert!(
        !save_tokens(&byte_limited, &runner, &[1, 2], &snapshot).stored
            && byte_limited.entry_count() == 0,
        "single entry larger than total disk budget is skipped"
    );
}

#[test]
fn automatic_staging_and_admission_diagnostics() {
    let default_directory = temporary_directory();
    let defaults = open(ContinuationDiskStoreOptions {
        directory: default_directory.path().to_path_buf(),
        ..ContinuationDiskStoreOptions::default()
    });
    assert!(
        defaults.capacity_bytes() == 8 << 30
            && defaults.staging_capacity_bytes() > DISK_HEADER_BYTES
            && defaults.staging_capacity_bytes() <= defaults.capacity_bytes(),
        "defaults bound RAM staging by available RAM and disk retention"
    );

    let directory = temporary_directory();
    let runner = FakeRunner::new("automatic");
    let (store, events) = recording(ContinuationDiskStoreOptions {
        directory: directory.path().to_path_buf(),
        capacity_bytes: 1024 * 1024,
        staging_capacity_bytes: 0,
        min_checkpoint_step_tokens: 0,
    });
    let limit = store.staging_capacity_bytes();
    assert!(
        limit > DISK_HEADER_BYTES && limit <= store.capacity_bytes(),
        "automatic staging resolves to a usable bounded RAM budget"
    );
    let payload = limit / 2;
    let first = store
        .reserve_capture(&runner, 2, payload, &[])
        .expect("reserve");
    assert!(first.is_some(), "automatic staging admits a snapshot");
    assert!(
        !store.can_save(&runner, 2, payload, &[]).expect("can save")
            && store
                .reserve_capture(&runner, 2, payload, &[])
                .expect("reserve")
                .is_none(),
        "automatic staging still accounts for all in-flight captures"
    );
    let last = *events.lock().expect("events").last().expect("event");
    assert!(
        last.reason == ContinuationDiskEventReason::StagingCapacity
            && last.staging_capacity_bytes == limit
            && last.staging_used_bytes > payload
            && last.payload_bytes == payload,
        "preflight rejection reports staging usage and required size"
    );
    drop(first);
    assert!(
        store.can_save(&runner, 2, payload, &[]).expect("can save"),
        "released staging is reusable without allocating the whole budget"
    );
    assert!(
        !store
            .can_save(&runner, 2, store.capacity_bytes(), &[])
            .expect("can save")
            && events.lock().expect("events").last().expect("event").reason
                == ContinuationDiskEventReason::ByteCapacity,
        "disk retention rejection is distinguished from staging pressure"
    );

    let explicit_directory = temporary_directory();
    let (explicit, events) = recording(store_options(explicit_directory.path(), 4096, 256));
    assert!(
        explicit.staging_capacity_bytes() == 256
            && !explicit.can_save(&runner, 2, 512, &[]).expect("can save"),
        "an explicit too-small staging limit remains enforced"
    );
    let last = *events.lock().expect("events").last().expect("event");
    assert!(
        last.reason == ContinuationDiskEventReason::StagingCapacity && last.staging_used_bytes == 0,
        "and visible"
    );
}

#[test]
fn smaller_staging_preserves_existing_files() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("staging-restart");
    let file_bytes = expected_file_bytes(15, 2);
    {
        let store = open(default_options(directory.path()));
        assert!(
            save_tokens(&store, &runner, &[1, 2], &make_snapshot(&runner, 42, 2)).stored,
            "restart fixture stores"
        );
    }
    {
        let (store, events) = recording(store_options(directory.path(), 4096, file_bytes - 1));
        assert!(
            store.entry_count() == 1
                && store.retained_bytes() == file_bytes
                && cache_files(directory.path()).len() == 1,
            "a smaller RAM budget preserves and accounts for existing files"
        );
        let mut state = runner.create_state().expect("state");
        assert!(
            !restore_tokens(&store, &runner, &mut state, &[1, 2, 3]).restored,
            "unverified oversized files are not exposed to lookup"
        );
        assert!(
            events.lock().expect("events").iter().any(|event| {
                event.reason == ContinuationDiskEventReason::StagingCapacity
                    && event.file_bytes == file_bytes
            }),
            "oversized startup files report a staging skip, not corruption"
        );
    }
    {
        let store = open(ContinuationDiskStoreOptions {
            directory: directory.path().to_path_buf(),
            capacity_bytes: 4096,
            staging_capacity_bytes: 0,
            min_checkpoint_step_tokens: 0,
        });
        let mut state = runner.create_state().expect("state");
        assert!(
            restore_tokens(&store, &runner, &mut state, &[1, 2, 3]).restored && state.value == 42,
            "automatic staging validates and restores the preserved file"
        );
    }
    {
        let store = open(store_options(
            directory.path(),
            file_bytes - 1,
            file_bytes - 1,
        ));
        assert!(
            store.retained_bytes() == 0 && cache_files(directory.path()).is_empty(),
            "unindexed files remain subject to the disk retention budget"
        );
    }
}

#[test]
fn lru_eviction_uses_actual_file_bytes() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("12345678");
    let entry_bytes = expected_file_bytes(8, 2);
    let store = open(store_options(directory.path(), entry_bytes * 2, 4096));

    let first = make_snapshot(&runner, 1, 2);
    let second = make_snapshot(&runner, 2, 2);
    let third = make_snapshot(&runner, 3, 2);
    assert_eq!(
        save_tokens(&store, &runner, &[1, 1], &first).file_bytes,
        entry_bytes,
        "store reports exact first file size"
    );
    assert!(
        save_tokens(&store, &runner, &[2, 2], &second).stored,
        "second entry fills exact two-entry budget"
    );

    let mut touched = FakeState::default();
    assert!(
        restore_tokens(&store, &runner, &mut touched, &[1, 1, 9]).restored,
        "restore refreshes first entry's LRU position"
    );
    assert!(
        save_tokens(&store, &runner, &[3, 3], &third).stored,
        "third entry evicts one least-recently-used file"
    );
    assert!(
        store.entry_count() == 2 && store.retained_bytes() == entry_bytes * 2,
        "retained bytes never exceed configured total budget"
    );

    let mut first_state = FakeState::default();
    let mut second_state = FakeState::default();
    let mut third_state = FakeState::default();
    assert!(
        restore_tokens(&store, &runner, &mut first_state, &[1, 1, 8]).restored,
        "recently touched entry survives LRU pressure"
    );
    assert!(
        !restore_tokens(&store, &runner, &mut second_state, &[2, 2, 8]).restored,
        "oldest entry is the deterministic LRU victim"
    );
    assert!(
        restore_tokens(&store, &runner, &mut third_state, &[3, 3, 8]).restored,
        "new entry survives LRU pressure"
    );
}

#[test]
fn atomic_publication_and_private_permissions() {
    let directory = temporary_directory();
    let orphan = directory.path().join(".tmp-orphan");
    fs::write(&orphan, "partial").expect("orphan");

    let runner = FakeRunner::new("permissions");
    let store = open(default_options(directory.path()));
    assert!(
        !orphan.exists(),
        "startup removes an interrupted private temporary file"
    );
    let snapshot = make_snapshot(&runner, 77, 2);
    assert!(
        save_tokens(&store, &runner, &[7, 7], &snapshot).stored,
        "permission test stores one entry"
    );

    let files = cache_files(directory.path());
    assert_eq!(files.len(), 1, "atomic publication exposes one final file");
    for entry in fs::read_dir(directory.path()).expect("directory") {
        let name = entry.expect("entry").file_name();
        assert!(
            !name.to_string_lossy().starts_with(".tmp-"),
            "successful publication leaves no temporary file"
        );
    }
    let status = fs::symlink_metadata(&files[0]).expect("status");
    let mode = status.permissions().mode();
    assert!(
        status.file_type().is_file() && mode & 0o077 == 0 && mode & 0o400 != 0 && mode & 0o200 != 0,
        "published cache file is private and owner read-write"
    );
}

#[test]
fn startup_rejects_unsafe_and_invalid_files() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("startup-valid");
    let incompatible = FakeRunner::new("startup-incompatible");
    let truncated_file;
    let checksum_file;
    {
        let store = open(default_options(directory.path()));
        let save_and_find_file =
            |entry_runner: &FakeRunner, tokens: &[Token], snapshot: &FakeSnapshot| {
                let before = cache_files(directory.path());
                assert!(
                    save_tokens(&store, entry_runner, tokens, snapshot).stored,
                    "startup fixture publishes one immutable entry"
                );
                cache_files(directory.path())
                    .into_iter()
                    .find(|candidate| !before.contains(candidate))
                    .expect("startup fixture identifies the newly published file")
            };
        let first = make_snapshot(&runner, 1, 2);
        let second = make_snapshot(&runner, 2, 2);
        let third = make_snapshot(&runner, 3, 2);
        let other = make_snapshot(&incompatible, 4, 2);
        truncated_file = save_and_find_file(&runner, &[1, 1], &first);
        checksum_file = save_and_find_file(&runner, &[2, 2], &second);
        save_and_find_file(&runner, &[3, 3], &third);
        save_and_find_file(&incompatible, &[4, 4], &other);
    }

    assert_eq!(
        cache_files(directory.path()).len(),
        4,
        "startup fixture exposes four immutable files"
    );
    OpenOptions::new()
        .write(true)
        .open(&truncated_file)
        .and_then(|file| file.set_len(12))
        .expect("truncate");
    flip_last_byte(&checksum_file);
    let target = directory.path().join("external-target");
    fs::write(&target, "must survive").expect("target");
    let unsafe_link = directory.path().join("unsafe.kvc");
    std::os::unix::fs::symlink("external-target", &unsafe_link).expect("symlink");

    let (restarted, events) = recording(default_options(directory.path()));
    // Startup reads metadata only: the payload checksum is checked when the
    // entry is first restored.
    assert!(
        restarted.entry_count() == 3 && cache_files(directory.path()).len() == 3,
        "startup indexes regular entries with valid metadata"
    );
    assert!(
        target.exists() && fs::symlink_metadata(&unsafe_link).is_err(),
        "unsafe symlink is removed without following its target"
    );
    assert!(
        has_reason(&events, ContinuationDiskEventReason::Corrupt)
            && !has_reason(&events, ContinuationDiskEventReason::ChecksumMismatch)
            && has_reason(&events, ContinuationDiskEventReason::UnsafeFile),
        "startup reports sanitized invalid-file reasons"
    );

    let mut compatible_hits = 0;
    for prompt in [[1, 1, 9], [2, 2, 9], [3, 3, 9]] {
        let mut state = FakeState::default();
        if restore_tokens(&restarted, &runner, &mut state, &prompt).restored {
            compatible_hits += 1;
        }
    }
    assert_eq!(
        compatible_hits, 1,
        "the remaining compatible startup entry restores exactly"
    );
    assert!(
        has_reason(&events, ContinuationDiskEventReason::ChecksumMismatch)
            && restarted.entry_count() == 2
            && cache_files(directory.path()).len() == 2,
        "the first restore rejects and removes the corrupt payload"
    );
    let mut incompatible_state = FakeState::default();
    assert!(
        restore_tokens(
            &restarted,
            &incompatible,
            &mut incompatible_state,
            &[4, 4, 9]
        )
        .restored,
        "a valid different compatibility identity remains independently usable"
    );
}

#[test]
fn root_symlink_is_rejected() {
    let directory = temporary_directory();
    let target = directory.path().join("target");
    let link = directory.path().join("cache-link");
    fs::create_dir(&target).expect("target");
    std::os::unix::fs::symlink("target", &link).expect("symlink");
    let rejected = ContinuationDiskStore::new(default_options(&link), None, None)
        .err()
        .is_some_and(|error| error.kind() == ErrorKind::Runtime);
    assert!(rejected, "configured cache-directory symlinks are rejected");
}

#[test]
fn concurrent_callers_remain_bounded_and_exact() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("concurrent");
    const THREAD_COUNT: u64 = 8;
    let store = open(store_options(directory.path(), 1024 * 1024, 4096));
    std::thread::scope(|scope| {
        for index in 0..THREAD_COUNT {
            let (store, runner) = (&store, &runner);
            scope.spawn(move || {
                let snapshot = make_snapshot(runner, 100 + index, 2);
                let tokens = [index as Token, (index + 100) as Token];
                assert!(
                    store
                        .save(runner, &tokens, &snapshot, &[])
                        .expect("save")
                        .stored,
                    "concurrent save succeeds under serialized I/O bound"
                );
            });
        }
    });
    assert_eq!(
        store.entry_count(),
        THREAD_COUNT as usize,
        "all concurrent saves publish distinct exact entries"
    );
    for index in 0..THREAD_COUNT {
        let mut state = FakeState::default();
        let prompt = [index as Token, (index + 100) as Token, 999];
        assert!(
            restore_tokens(&store, &runner, &mut state, &prompt).restored
                && state.value == 100 + index,
            "concurrent store preserves exact per-key payloads"
        );
    }
}

#[test]
fn concurrent_store_instances_publish_safely() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("multi-process");
    {
        let first = open(default_options(directory.path()));
        let second = open(default_options(directory.path()));
        let first_snapshot = make_snapshot(&runner, 101, 2);
        let second_snapshot = make_snapshot(&runner, 202, 2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                assert!(
                    save_tokens(&first, &runner, &[1, 0], &first_snapshot).stored,
                    "first store instance publishes atomically"
                );
            });
            scope.spawn(|| {
                assert!(
                    save_tokens(&second, &runner, &[2, 0], &second_snapshot).stored,
                    "second store instance publishes atomically"
                );
            });
        });
    }

    let restarted = open(default_options(directory.path()));
    assert_eq!(
        restarted.entry_count(),
        2,
        "restart indexes files from concurrent store instances"
    );
    let mut first_state = FakeState::default();
    let mut second_state = FakeState::default();
    assert!(
        restore_tokens(&restarted, &runner, &mut first_state, &[1, 0, 9]).restored
            && first_state.value == 101
            && restore_tokens(&restarted, &runner, &mut second_state, &[2, 0, 9]).restored
            && second_state.value == 202,
        "concurrent publishers expose no partial or confused entry"
    );
}

#[test]
fn shared_prefix_boundaries_and_exact_dedup() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("shared-prefix-model");
    let store = open(default_options(directory.path()));
    let boundaries = |prompt: &[Token], min_tokens: usize, max_boundaries: usize| {
        store
            .shared_prefix_boundaries(
                &runner,
                prompt,
                min_tokens,
                max_boundaries,
                InputIdentity::default(),
            )
            .expect("boundaries")
    };

    // Conversation A: shared system prefix {7, 7, 7} plus its own turn.
    let first = make_snapshot(&runner, 11, 5);
    assert!(
        save_tokens(&store, &runner, &[7, 7, 7, 1, 2], &first).stored,
        "first conversation frontier is stored"
    );

    // Conversation B shares the first three tokens only.
    let second_prompt = [7, 7, 7, 3, 4, 5];
    assert_eq!(
        boundaries(&second_prompt, 1, 4),
        vec![3],
        "common prefix with a stored entry becomes a boundary"
    );
    assert!(
        boundaries(&second_prompt, 4, 4).is_empty(),
        "boundaries below the minimum length are dropped"
    );
    assert!(
        boundaries(&[7, 7, 7, 1, 2], 1, 4).is_empty(),
        "an exact stored prompt yields no boundary"
    );

    // Persist the shared prefix; a third conversation restores it.
    let shared = make_snapshot(&runner, 77, 3);
    assert!(
        save_tokens(&store, &runner, &[7, 7, 7], &shared).stored,
        "shared prefix snapshot is stored"
    );
    assert!(
        boundaries(&second_prompt, 1, 4).is_empty(),
        "a stored shared prefix is no longer a boundary"
    );
    let mut state = FakeState::default();
    let restored = restore_tokens(&store, &runner, &mut state, &[7, 7, 7, 9]);
    assert!(
        restored.restored && restored.token_count == 3 && state.value == 77,
        "new conversation restores the shared prefix"
    );

    // Saving the same tokens again touches instead of rewriting.
    let retained_before = store.retained_bytes();
    let entries_before = store.entry_count();
    let duplicate = make_snapshot(&runner, 78, 3);
    assert!(
        !save_tokens(&store, &runner, &[7, 7, 7], &duplicate).stored,
        "exact duplicate save is skipped"
    );
    assert!(
        store.retained_bytes() == retained_before && store.entry_count() == entries_before,
        "exact duplicate save leaves the store unchanged"
    );
    assert!(
        store.touch(&runner, &[7, 7, 7], &[]).expect("touch"),
        "touch finds the exact entry"
    );
    assert!(
        !store.touch(&runner, &[7, 7], &[]).expect("touch"),
        "touch misses a non-stored prefix"
    );

    // Longest boundaries win when more than max_boundaries are shared.
    let deeper = make_snapshot(&runner, 12, 5);
    assert!(
        save_tokens(&store, &runner, &[7, 7, 7, 3, 8], &deeper).stored,
        "deeper entry is stored"
    );
    let third_prompt = [7, 7, 7, 3, 4, 6];
    assert_eq!(
        boundaries(&third_prompt, 1, 4),
        vec![4],
        "only the unstored shared length is reported"
    );
    assert_eq!(
        boundaries(&third_prompt, 1, 1),
        vec![4],
        "capping keeps the longest boundary"
    );
}

#[test]
fn bounded_async_persistence_does_not_block_lookup() {
    let directory = temporary_directory();
    let runner = Arc::new(FakeRunner::new("async-store"));
    let entered = Arc::new(Semaphore::default());
    let release = Arc::new(Semaphore::default());
    let identity_bytes = runner
        .descriptor()
        .persistence
        .expect("persistence")
        .compatibility_identity
        .len();
    let bytes = expected_file_bytes(identity_bytes, 2);
    {
        let store = open(store_options(
            directory.path(),
            4096,
            bytes + CPP_SIZEOF_FAKE_SNAPSHOT,
        ));
        assert!(
            store
                .save(&*runner, &[7, 8], &make_snapshot(&runner, 99, 2), &[])
                .expect("save")
                .stored,
            "an existing entry is available before the new write"
        );
        let (hook_entered, hook_release) = (Arc::clone(&entered), Arc::clone(&release));
        runner.set_before_stream(Some(Arc::new(move || {
            hook_entered.release();
            hook_release.acquire();
        })));
        let snapshot = make_snapshot(&runner, 42, 2);
        let payload = snapshot.payload_bytes();
        let can_save = |bytes: usize| store.can_save(&*runner, 2, bytes, &[]).expect("can save");
        assert!(
            can_save(payload),
            "empty persistence queue admits the snapshot before capture"
        );
        let capture = store
            .reserve_capture(&*runner, 2, payload, &[])
            .expect("reserve");
        assert!(
            capture.is_some()
                && !can_save(payload)
                && store
                    .reserve_capture(&*runner, 2, payload, &[])
                    .expect("reserve")
                    .is_none(),
            "in-flight captures reserve staging before any allocation"
        );
        drop(capture);
        assert!(
            can_save(payload),
            "discarded captures release their reservation"
        );
        let capture = store
            .reserve_capture(&*runner, 2, payload, &[])
            .expect("reserve");
        assert_eq!(
            store
                .save_async(
                    Arc::clone(&runner),
                    vec![1, 2],
                    Arc::new(snapshot),
                    Vec::new(),
                    capture,
                    false,
                )
                .expect("save async"),
            bytes,
            "immutable snapshot is queued without waiting for serialization"
        );
        let started = entered.try_acquire_for(Duration::from_secs(2));
        assert!(started, "persistence worker starts independently");
        assert!(
            !can_save(CPP_SIZEOF_FAKE_SNAPSHOT)
                && store
                    .save_async(
                        Arc::clone(&runner),
                        vec![3, 4],
                        Arc::new(make_snapshot(&runner, 43, 2)),
                        Vec::new(),
                        None,
                        false,
                    )
                    .expect("save async")
                    == 0,
            "active writes count against the bounded queue"
        );
        let (ready, looked_up) = std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            let (store, runner) = (&store, &*runner);
            scope.spawn(move || {
                let mut state = FakeState::default();
                let result = restore_tokens(store, runner, &mut state, &[7, 8, 9]).restored
                    && state.value == 99
                    && store.touch(runner, &[7, 8], &[]).expect("touch")
                    && !restore_tokens(store, runner, &mut state, &[1, 2, 9]).restored
                    && store
                        .shared_prefix_boundaries(runner, &[1, 9], 1, 4, InputIdentity::default())
                        .expect("boundaries")
                        .is_empty()
                    && !store.touch(runner, &[1, 2], &[]).expect("touch");
                let _ = sender.send(result);
            });
            let ready = receiver.recv_timeout(Duration::from_secs(1));
            release.release();
            match ready {
                Ok(result) => (true, result),
                Err(_) => (false, receiver.recv().expect("lookup result")),
            }
        });
        assert!(
            ready && looked_up,
            "existing entries remain readable while a writer is blocked"
        );
        store.flush();
        runner.set_before_stream(None);
        let mut state = FakeState::default();
        assert!(
            restore_tokens(&store, &runner, &mut state, &[1, 2, 9]).restored && state.value == 42,
            "streamed chunks form a checksummed restorable snapshot"
        );
        assert_ne!(
            store
                .save_async(
                    Arc::clone(&runner),
                    vec![3, 4],
                    Arc::new(make_snapshot(&runner, 43, 2)),
                    Vec::new(),
                    None,
                    false,
                )
                .expect("save async"),
            0,
            "completed writes release queue admission"
        );
    }
    let restarted = open(default_options(directory.path()));
    assert_eq!(
        restarted.entry_count(),
        3,
        "shutdown drains accepted persistence jobs"
    );
}

#[test]
fn indexed_prefix_lookup() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("prefix-index");
    let store = open(default_options(directory.path()));
    // Many independent branches, with terminal prefixes inside compressed
    // edges.
    for branch in 0..96 {
        assert!(
            save_tokens(
                &store,
                &runner,
                &[7, branch, 8, 9],
                &make_snapshot(&runner, u64::from(branch), 4)
            )
            .stored,
            "indexed branch stores"
        );
    }
    assert!(
        save_tokens(&store, &runner, &[7, 45], &make_snapshot(&runner, 999, 2)).stored,
        "shorter prefix splits an existing compressed edge"
    );
    let mut state = FakeState::default();
    assert!(
        restore_tokens(&store, &runner, &mut state, &[7, 45, 8, 9, 10]).token_count == 4
            && state.value == 45,
        "longest terminal on the matching path wins"
    );
    assert!(
        restore_tokens(&store, &runner, &mut state, &[7, 45, 10]).token_count == 2
            && state.value == 999,
        "a mismatched edge falls back to its terminal ancestor"
    );
    assert_eq!(
        store
            .shared_prefix_boundaries(&runner, &[7, 46, 8, 10], 1, 4, InputIdentity::default())
            .expect("boundaries"),
        vec![1, 3],
        "index returns branch and partial-edge boundaries in prompt order"
    );
}

#[test]
fn image_identity_survives_restart() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("image-cache-test");
    let prompt: [Token; 3] = [1, 248_056, 3];
    let (image_a, image_b): (&[u8], &[u8]) = (&[1, 2], &[1, 3]);
    {
        let store = open(default_options(directory.path()));
        assert!(
            store
                .save(&runner, &prompt, &make_snapshot(&runner, 10, 3), image_a)
                .expect("save")
                .stored,
            "first image is persisted"
        );
        assert!(
            store
                .save(&runner, &prompt, &make_snapshot(&runner, 20, 3), image_b)
                .expect("save")
                .stored,
            "second image has an independent disk entry"
        );
    }
    let store = open(default_options(directory.path()));
    let mut state = FakeState::default();
    assert!(
        store
            .restore_longest_prefix(&runner, &mut state, &prompt, image_a.into(), 0, 0)
            .expect("restore")
            .restored
            && state.value == 10,
        "restart restores first image exactly"
    );
    assert!(
        store
            .restore_longest_prefix(&runner, &mut state, &prompt, image_b.into(), 0, 0)
            .expect("restore")
            .restored
            && state.value == 20,
        "restart restores second image exactly"
    );
    assert!(
        !store
            .restore_longest_prefix(&runner, &mut state, &prompt, InputIdentity::default(), 0, 0)
            .expect("restore")
            .restored,
        "text cannot restore a disk image continuation"
    );
    assert!(
        !store.touch(&runner, &prompt, &[]).expect("touch"),
        "disk dedup respects image identity"
    );
}

#[test]
fn appended_image_prefixes_survive_restart() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("image-prefix-cache");
    let prompt: [Token; 8] = [1, 2, 3, 248_056, 4, 5, 248_056, 6];
    let (a, b, ab, ba, ac) = (
        &[10u8][..],
        &[20u8][..],
        &[30u8][..],
        &[40u8][..],
        &[50u8][..],
    );
    let prefix = |token_count: usize, identity: &[u8]| ContinuationInputPrefix {
        token_count,
        identity: identity.to_vec(),
    };
    let input = |complete, prefixes| InputIdentity { complete, prefixes };
    let prefixes = [prefix(3, &[]), prefix(6, a)];
    {
        let store = open(default_options(directory.path()));
        let save = |tokens: &[Token], value: u64, identity: &[u8]| {
            store
                .save(
                    &runner,
                    tokens,
                    &make_snapshot(&runner, value, tokens.len() as u64),
                    identity,
                )
                .expect("save")
                .stored
        };
        assert!(
            save(&prompt[..3], 10, &[]),
            "text prefix persists without future image identity"
        );
        assert!(
            save(&prompt[..5], 20, a),
            "first-image prefix persists independently of the second image"
        );
        assert!(
            save(&prompt, 30, ab),
            "full two-image checkpoint is persisted"
        );
    }
    let store = open(default_options(directory.path()));
    let mut state = FakeState::default();
    let mut restore = |input: InputIdentity<'_>, stable: usize| {
        let result = store
            .restore_longest_prefix(&runner, &mut state, &prompt, input, stable, 0)
            .expect("restore");
        (result.restored, result.token_count, state.value)
    };
    assert_eq!(
        restore(input(ab, &prefixes), 7),
        (true, 8, 30),
        "exact disk retry finds its fallback under an earlier image identity"
    );
    assert_eq!(
        restore(input(ac, &prefixes), 7),
        (true, 5, 20),
        "changing only the second image restores the first-image checkpoint"
    );
    let reordered = [prefix(3, &[]), prefix(6, b)];
    assert_eq!(
        restore(input(ba, &reordered), 7),
        (true, 3, 10),
        "reordered or changed first image restores only the text prefix"
    );
    let (restored, tokens, _) = restore(InputIdentity::default(), 0);
    assert!(
        restored && tokens == 3,
        "literal image-pad text cannot recover visual state"
    );
    let moved = [prefix(2, &[])];
    assert!(
        !restore(input(b, &moved), 0).0,
        "moving the first image before the checkpoint invalidates reuse"
    );
}

#[test]
fn file_layout_is_the_cpp_format() {
    let directory = temporary_directory();
    let runner = FakeRunner::new("layout");
    let store = open(default_options(directory.path()));
    let saved = save_tokens(&store, &runner, &[1, 2], &make_snapshot(&runner, 5, 2));
    let files = cache_files(directory.path());
    let mut image = Vec::new();
    File::open(&files[0])
        .and_then(|mut file| file.read_to_end(&mut image))
        .expect("read");
    assert_eq!(image.len(), saved.file_bytes);
    assert_eq!(&image[..8], b"GUFOKVC1");
    assert_eq!(&image[8..12], &1u32.to_le_bytes());
    assert_eq!(&image[12..16], &7u32.to_le_bytes());
    assert_eq!(&image[16..20], &6u32.to_le_bytes());
    assert_eq!(&image[20..24], &2u32.to_le_bytes());
    assert_eq!(&image[24..32], &16u64.to_le_bytes());
    let checksum = image[32..96].to_vec();
    image[32..96].fill(0);
    assert_eq!(sha256_hex(&image).as_bytes(), checksum.as_slice());
    assert_eq!(&image[96..102], b"layout");
    // The key hash names the file: identity length, payload version, token
    // count, identity and tokens.
    let mut key = Vec::new();
    key.extend_from_slice(&6u32.to_le_bytes());
    key.extend_from_slice(&7u32.to_le_bytes());
    key.extend_from_slice(&2u32.to_le_bytes());
    key.extend_from_slice(b"layout");
    key.extend_from_slice(&1u32.to_le_bytes());
    key.extend_from_slice(&2u32.to_le_bytes());
    let name = files[0]
        .file_name()
        .expect("name")
        .to_string_lossy()
        .into_owned();
    assert!(name.starts_with(&format!("{}-", sha256_hex(&key))));
    assert!(name.ends_with(".kvc") && name.len() == 64 + 1 + 16 + 4);
}

/// Checkpoints less than one step past a stored prefix are skipped, shared
/// prefixes are exempt, and zero disables the gate.
#[test]
fn minimum_checkpoint_step() {
    let directory = temporary_directory();
    let runner = Arc::new(FakeRunner::new("min-step-model"));
    let (store, events) = recording(ContinuationDiskStoreOptions {
        min_checkpoint_step_tokens: 3,
        ..store_options(directory.path(), 4096, 4096)
    });
    assert!(
        save_tokens(&store, &runner, &[1, 2], &make_snapshot(&runner, 1, 2)).stored,
        "the first checkpoint has no covering prefix and is stored"
    );
    assert!(
        store.within_checkpoint_step(&*runner, &[1, 2, 3, 4], &[])
            && !store.within_checkpoint_step(&*runner, &[1, 2, 3, 4, 5], &[])
            && !store.within_checkpoint_step(&*runner, &[8, 8], &[]),
        "callers can skip a capture the step would discard"
    );
    assert!(
        !save_tokens(
            &store,
            &runner,
            &[1, 2, 3, 4],
            &make_snapshot(&runner, 2, 4)
        )
        .stored
            && events.lock().expect("events").last().map(|e| e.reason)
                == Some(ContinuationDiskEventReason::MinStep),
        "a checkpoint less than one step past a stored prefix is skipped"
    );
    assert!(
        save_tokens(&store, &runner, &[9, 9], &make_snapshot(&runner, 3, 2)).stored,
        "an unrelated checkpoint is not covered by another prompt"
    );
    assert!(
        save_tokens(
            &store,
            &runner,
            &[1, 2, 3, 4, 5],
            &make_snapshot(&runner, 4, 5)
        )
        .stored,
        "a checkpoint one full step past the prefix is stored"
    );
    assert_ne!(
        store
            .save_async(
                Arc::clone(&runner),
                vec![1, 2, 3, 4, 5, 6],
                Arc::new(make_snapshot(&runner, 5, 6)),
                Vec::new(),
                None,
                true,
            )
            .expect("queue"),
        0,
        "a shared prefix is queued"
    );
    store.flush();
    let mut state = FakeState {
        value: 0,
        position: 0,
    };
    assert!(
        restore_tokens(&store, &runner, &mut state, &[1, 2, 3, 4, 5, 6, 7]).restored
            && state.value == 5,
        "shared prefixes are exempt from the step"
    );
    let ungated_directory = temporary_directory();
    let ungated = open(store_options(ungated_directory.path(), 4096, 4096));
    assert!(
        save_tokens(&ungated, &runner, &[1, 2], &make_snapshot(&runner, 1, 2)).stored
            && save_tokens(&ungated, &runner, &[1, 2, 3], &make_snapshot(&runner, 2, 3)).stored,
        "zero disables the step"
    );
}

/// A store starting while another store publishes neither waits for that
/// write nor deletes its temporary file; without a publisher, startup
/// removes abandoned temporary files.
#[test]
fn startup_preserves_an_active_publisher() {
    let directory = temporary_directory();
    let runner = Arc::new(FakeRunner::new("startup-publisher"));
    let (entered, release) = (
        Arc::new(Semaphore::default()),
        Arc::new(Semaphore::default()),
    );
    let orphan = directory.path().join(".tmp-orphan");
    {
        let writer = open(default_options(directory.path()));
        let (hook_entered, hook_release) = (Arc::clone(&entered), Arc::clone(&release));
        runner.set_before_stream(Some(Arc::new(move || {
            hook_entered.release();
            hook_release.acquire();
        })));
        assert_ne!(
            writer
                .save_async(
                    Arc::clone(&runner),
                    vec![1, 2],
                    Arc::new(make_snapshot(&runner, 42, 2)),
                    Vec::new(),
                    None,
                    false,
                )
                .expect("save async"),
            0,
            "the writer accepts a checkpoint before another store starts"
        );
        let started = entered.try_acquire_for(Duration::from_secs(2));
        if !started {
            release.release();
        }
        assert!(started, "the writer reaches its temporary file");
        fs::write(&orphan, b"interrupted older write").expect("orphan");
        let (sender, receiver) = mpsc::channel();
        let path = directory.path().to_path_buf();
        let startup = std::thread::spawn(move || {
            drop(open(default_options(&path)));
            let _ = sender.send(());
        });
        let ready = receiver.recv_timeout(Duration::from_secs(1)).is_ok();
        release.release();
        startup.join().expect("startup");
        writer.flush();
        assert!(
            ready,
            "startup does not wait for an active checkpoint writer"
        );
        assert!(
            writer.entry_count() == 1 && cache_files(directory.path()).len() == 1,
            "an overlapping startup keeps the active writer's checkpoint"
        );
    }
    runner.set_before_stream(None);
    let restarted = open(default_options(directory.path()));
    let mut state = FakeState::default();
    assert!(
        restore_tokens(&restarted, &runner, &mut state, &[1, 2, 3]).restored && state.value == 42,
        "a checkpoint published across a startup restores its exact payload"
    );
    assert!(
        !orphan.exists(),
        "startup without a publisher removes abandoned temporary files"
    );
}

/// A queued or running save is a pending publication for prompts that
/// extend it with the same input, once it saves enough; a follow-up names
/// its checkpoint.
#[test]
fn pending_publications_track_queued_and_running_saves() {
    let directory = temporary_directory();
    let runner = Arc::new(FakeRunner::new("pending-publication"));
    let entered = Arc::new(Semaphore::default());
    let release = Arc::new(Semaphore::default());
    let store = open(store_options(directory.path(), 1 << 20, 1 << 20));
    let (hook_entered, hook_release) = (Arc::clone(&entered), Arc::clone(&release));
    runner.set_before_stream(Some(Arc::new(move || {
        hook_entered.release();
        hook_release.acquire();
    })));
    let long: Vec<Token> = (0..5000).collect();
    assert!(
        store
            .save_async(
                Arc::clone(&runner),
                long.clone(),
                Arc::new(make_snapshot(&runner, 7, long.len() as u64)),
                Vec::new(),
                None,
                false,
            )
            .expect("save async")
            > 0
    );
    assert!(
        entered.try_acquire_for(Duration::from_secs(2)),
        "the writer runs the save"
    );
    let mut prompt = long.clone();
    prompt.push(9);
    let pending = store
        .pending_publication(&prompt, InputIdentity::default(), 0, None)
        .expect("the running save is pending");
    assert!(pending.pending_tokens == 5000 && pending.gain() == 5000);
    assert!(
        store
            .pending_publication(&prompt, InputIdentity::default(), 1000, None)
            .is_none(),
        "a saving below the minimum gain does not wait"
    );
    assert_eq!(
        store
            .pending_publication(
                &prompt,
                InputIdentity::default(),
                1000,
                Some(pending.checkpoint_id)
            )
            .map(|p| p.checkpoint_id),
        Some(pending.checkpoint_id),
        "a follow-up skips the gain check"
    );
    assert!(
        store
            .pending_publication(&[1, 2, 3], InputIdentity::default(), 0, None)
            .is_none(),
        "only prompts extending the checkpoint wait"
    );
    let image = InputIdentity::from(&b"image"[..]);
    assert!(
        store.pending_publication(&prompt, image, 0, None).is_none(),
        "another input identity does not wait"
    );
    release.release();
    store.flush();
    runner.set_before_stream(None);
    assert!(
        store
            .pending_publication(
                &prompt,
                InputIdentity::default(),
                0,
                Some(pending.checkpoint_id)
            )
            .is_none(),
        "a finished save is no longer pending"
    );
}
