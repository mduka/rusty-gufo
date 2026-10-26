//! The MiniMax H3 text-to-video job queue (`src/cli/serve/video_jobs.cpp`):
//! bounded admission, one worker thread, job state persisted atomically to
//! `<storage_root>/<id>/job.json`, recovery of completed jobs at startup,
//! expiry after the artifact TTL and cancellation by deletion. Prompts are
//! never persisted and are wiped as soon as generation took them.
//!
//! Generation and the checkpoint inventory check are injected
//! ([`VideoRunner`], [`InventoryCheck`]). No model implements them yet: the
//! service waits for a GGUF-based MiniMax H3 port (docs/RUST.md).

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gufo_core::json::Value;

use crate::logging::{self, LogLevel};
pub use crate::video_model::{
    CancellationToken, GenerationParameters, GenerationRequest, GenerationTelemetry,
    InventoryCheck, VideoRunner,
};
use crate::video_model::{
    generation_parameters_json, generation_telemetry_json, resolve_duration_frames, sha256_hex,
    validate_generation_parameters,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VideoJobStatus {
    #[default]
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl VideoJobStatus {
    pub fn name(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoJobResult {
    Ok,
    Invalid,
    QueueFull,
    NotFound,
    NotReady,
    Expired,
    IoError,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoJobRequest {
    pub model: String,
    pub prompt: String,
    pub size: String,
    pub seconds: String,
    pub output_format: String,
    pub parameters: GenerationParameters,
    pub seed: u64,
}

impl Default for VideoJobRequest {
    fn default() -> Self {
        Self {
            model: "minimax-h3".into(),
            prompt: String::new(),
            size: "512x512".into(),
            seconds: "1".into(),
            output_format: "mp4".into(),
            parameters: GenerationParameters::default(),
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VideoJobSnapshot {
    pub id: String,
    pub status: VideoJobStatus,
    pub progress: i32,
    pub created_at: i64,
    pub completed_at: i64,
    pub expires_at: i64,
    pub model: String,
    pub size: String,
    pub seconds: String,
    pub output_format: String,
    pub error_code: String,
    pub error_message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VideoJobContent {
    pub path: PathBuf,
    pub media_type: String,
    pub bytes: u64,
}

pub struct VideoJobServiceOptions {
    pub model_root: PathBuf,
    pub source_manifest: PathBuf,
    pub storage_root: PathBuf,
    pub queue_capacity: usize,
    pub artifact_ttl: Duration,
    /// Inspects the checkpoint before admitting requests; needs
    /// `inventory_check`.
    pub validate_model_inventory: bool,
    pub id_factory: Option<Box<dyn Fn() -> String + Send + Sync>>,
    /// Unix seconds.
    pub now: Option<Box<dyn Fn() -> i64 + Send + Sync>>,
    pub runner: Option<VideoRunner>,
    pub inventory_check: Option<InventoryCheck>,
}

impl Default for VideoJobServiceOptions {
    fn default() -> Self {
        Self {
            model_root: PathBuf::new(),
            source_manifest: PathBuf::new(),
            storage_root: PathBuf::new(),
            queue_capacity: 1,
            artifact_ttl: Duration::from_secs(3600),
            validate_model_inventory: true,
            id_factory: None,
            now: None,
            runner: None,
            inventory_check: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoJobCreateResult {
    pub result: VideoJobResult,
    pub job: Option<VideoJobSnapshot>,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoJobLookupResult {
    pub result: VideoJobResult,
    pub job: Option<VideoJobSnapshot>,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoJobContentResult {
    pub result: VideoJobResult,
    pub content: Option<VideoJobContent>,
    pub error: String,
}

fn is_h3_model(model: &str) -> bool {
    matches!(
        model,
        "minimax-h3"
            | "minimax-h3-exact"
            | "minimax-h3-fast"
            | "minimax-h3-aggressive"
            | "minimax-h3-dev"
            | "minimax-h3-fullres"
    )
}

fn model_matches_preset(model: &str, preset: &str) -> bool {
    model == "minimax-h3"
        || matches!(
            (model, preset),
            ("minimax-h3-exact", "exact-512")
                | ("minimax-h3-fast", "fast-384")
                | ("minimax-h3-aggressive", "aggressive-320")
                | ("minimax-h3-dev", "development-256")
                | ("minimax-h3-fullres", "exact-1344x768")
        )
}

fn matches_output_contract(request: &VideoJobRequest) -> bool {
    let p = &request.parameters;
    match resolve_duration_frames(&request.seconds) {
        Ok(frames) if p.frames == frames && model_matches_preset(&request.model, &p.preset) => {}
        _ => return false,
    }
    if request.output_format == "mp4" {
        let square = request.size == "512x512"
            && p.output_width == 512
            && p.output_height == 512
            && matches!(
                p.preset.as_str(),
                "exact-512" | "fast-384" | "aggressive-320"
            );
        let released_full_resolution = request.size == "1344x768"
            && p.preset == "exact-1344x768"
            && p.internal_width == 1344
            && p.internal_height == 768
            && p.output_width == 1344
            && p.output_height == 768;
        return (square || released_full_resolution)
            && p.mux
            && p.decode_audio
            && p.selected_frames.is_empty();
    }
    request.output_format == "ppm"
        && request.size == "256x256"
        && p.preset == "development-256"
        && p.internal_width == 256
        && p.internal_height == 256
        && p.output_width == 256
        && p.output_height == 256
        && !p.mux
        && !p.decode_audio
        && p.selected_frames.len() == 1
}

/// Overwrites and clears a prompt (`WipeString`).
fn wipe(value: &mut String) {
    let mut bytes = std::mem::take(value).into_bytes();
    bytes.fill(0);
    std::hint::black_box(&bytes);
}

/// 64 unpredictable bits from the standard library's randomly keyed hasher.
fn random_u64() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
    );
    hasher.finish()
}

/// `video_` plus 24 random lowercase hex digits (`RandomVideoId`).
fn random_video_id() -> String {
    let (a, b) = (random_u64(), random_u64());
    format!("video_{a:016x}{:08x}", b & 0xffff_ffff)
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

fn is_video_id(id: &str) -> bool {
    id.len() == 30
        && id.starts_with("video_")
        && id[6..]
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// A regular, non-symlinked file whose canonical path lies strictly inside
/// `storage_root`; returns its size (`IsSafeRegularContent`).
fn safe_regular_content(storage_root: &Path, candidate: &Path) -> Option<u64> {
    let metadata = fs::symlink_metadata(candidate).ok()?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return None;
    }
    let canonical = fs::canonicalize(candidate).ok()?;
    if canonical == storage_root || !canonical.starts_with(storage_root) {
        return None;
    }
    fs::metadata(candidate).ok().map(|m| m.len())
}

fn next_file_nonce() -> u64 {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    NONCE.fetch_add(1, Ordering::Relaxed)
}

/// Writes `contents` through a sibling partial file and a rename.
fn write_atomic(path: &Path, contents: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create video job directory: {e}"))?;
    }
    let partial = PathBuf::from(format!(
        "{}.gufo-partial-{}-{}",
        path.display(),
        std::process::id(),
        next_file_nonce()
    ));
    let written = fs::File::create(&partial).and_then(|mut file| file.write_all(contents));
    if written.is_err() {
        let _ = fs::remove_file(&partial);
        return Err("cannot write video job metadata".into());
    }
    if let Err(error) = fs::rename(&partial, path) {
        let _ = fs::remove_file(&partial);
        return Err(format!("cannot publish video job metadata: {error}"));
    }
    Ok(())
}

fn snapshot_json(snapshot: &VideoJobSnapshot) -> String {
    let error = if snapshot.error_code.is_empty() {
        Value::Null
    } else {
        Value::object()
            .with("code", snapshot.error_code.as_str())
            .with("message", snapshot.error_message.as_str())
    };
    let root = Value::object()
        .with("schema", "gufo.video-job.v1")
        .with("id", snapshot.id.as_str())
        .with("status", snapshot.status.name())
        .with("progress", snapshot.progress)
        .with("created_at", snapshot.created_at)
        .with("completed_at", snapshot.completed_at)
        .with("expires_at", snapshot.expires_at)
        .with("model", snapshot.model.as_str())
        .with("size", snapshot.size.as_str())
        .with("seconds", snapshot.seconds.as_str())
        .with("output_format", snapshot.output_format.as_str())
        .with("error", error);
    // Every member is finite, so serialisation cannot fail.
    root.dump().unwrap_or_default() + "\n"
}

fn progress_percent(phase: &str, completed: i32, total: i32) -> i32 {
    let scaled = |begin: i32, end: i32| {
        if total <= 0 {
            return begin;
        }
        let bounded = completed.clamp(0, total);
        begin + ((end - begin) * bounded) / total
    };
    match phase {
        "inventory" => scaled(1, 3),
        "prompt" => scaled(3, 10),
        "denoiser" => scaled(10, 80),
        "video_vae" => scaled(80, 90),
        "audio_vae" => scaled(90, 97),
        "media" => scaled(97, 99),
        _ => 1,
    }
}

struct JobData {
    snapshot: VideoJobSnapshot,
    request: VideoJobRequest,
    content_path: PathBuf,
    media_type: String,
    deleted: bool,
    logged_phase: String,
    logged_bucket: i32,
}

/// Job fields change only while the service lock is held; the job's own
/// lock is taken after it and is therefore uncontended.
struct Job {
    id: String,
    directory: PathBuf,
    cancellation: CancellationToken,
    data: Mutex<JobData>,
}

impl Job {
    fn data(&self) -> MutexGuard<'_, JobData> {
        self.data.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[derive(Default)]
struct State {
    jobs: BTreeMap<String, Arc<Job>>,
    queue: VecDeque<Arc<Job>>,
    active: Option<Arc<Job>>,
}

struct Shared {
    options: VideoJobServiceOptions,
    canonical_storage_root: PathBuf,
    state: Mutex<State>,
    condition: Condvar,
    stop: AtomicBool,
}

pub struct VideoJobService {
    shared: Option<Arc<Shared>>,
    initialization_error: String,
    worker: Option<JoinHandle<()>>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn now(&self) -> i64 {
        self.options.now.as_ref().map_or_else(unix_now, |now| now())
    }

    fn ttl_seconds(&self) -> i64 {
        self.options.artifact_ttl.as_secs() as i64
    }

    fn persist(&self, snapshot: &VideoJobSnapshot) -> Result<(), String> {
        write_atomic(
            &self
                .options
                .storage_root
                .join(&snapshot.id)
                .join("job.json"),
            snapshot_json(snapshot).as_bytes(),
        )
    }

    fn sweep_expired(&self, state: &mut State) {
        let now = self.now();
        state.jobs.retain(|_, job| {
            let data = job.data();
            let expired = data.snapshot.expires_at > 0
                && data.snapshot.expires_at <= now
                && !matches!(
                    data.snapshot.status,
                    VideoJobStatus::InProgress | VideoJobStatus::Queued
                );
            if expired {
                let _ = fs::remove_dir_all(&job.directory);
            }
            !expired
        });
    }

    fn recover_jobs(&self, state: &mut State) -> Result<(), String> {
        let now = self.now();
        let entries = fs::read_dir(&self.options.storage_root)
            .map_err(|e| format!("cannot inspect video job storage: {e}"))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("cannot inspect video job storage: {e}"))?;
            let file_type = entry
                .file_type()
                .map_err(|e| format!("cannot inspect video job entry: {e}"))?;
            let directory_id = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            if file_type.is_symlink() || !file_type.is_dir() || !is_video_id(&directory_id) {
                continue;
            }
            if let Ok(children) = fs::read_dir(&path) {
                for child in children.flatten() {
                    if child
                        .file_name()
                        .to_string_lossy()
                        .contains(".gufo-partial-")
                    {
                        let child = child.path();
                        let _ = fs::remove_dir_all(&child).or_else(|_| fs::remove_file(&child));
                    }
                }
            }
            let text = fs::read(path.join("job.json")).unwrap_or_default();
            match self.recover_job(&path, &directory_id, &text, now) {
                Some(job) => {
                    state.jobs.insert(job.id.clone(), Arc::new(job));
                }
                None => {
                    let _ = fs::remove_dir_all(&path);
                }
            }
        }
        Ok(())
    }

    fn recover_job(&self, path: &Path, directory_id: &str, text: &[u8], now: i64) -> Option<Job> {
        if text.is_empty() {
            return None;
        }
        let value = gufo_core::json::parse(text).ok()?;
        let mut snapshot = VideoJobSnapshot {
            id: value.member_str("id", ""),
            model: value.member_str("model", ""),
            size: value.member_str("size", ""),
            seconds: value.member_str("seconds", ""),
            output_format: value.member_str("output_format", ""),
            ..VideoJobSnapshot::default()
        };
        let progress = value.member_double("progress", -1.0);
        let created_at = value.member_double("created_at", -1.0);
        let completed_at = value.member_double("completed_at", -1.0);
        let expires_at = value.member_double("expires_at", -1.0);
        let valid_output = (snapshot.output_format == "mp4"
            && (snapshot.size == "512x512" || snapshot.size == "1344x768"))
            || (snapshot.output_format == "ppm" && snapshot.size == "256x256");
        let limit = i64::MAX as f64;
        let integral = |v: f64| v.is_finite() && v.floor() == v;
        let valid_numbers = integral(progress)
            && progress == 100.0
            && integral(created_at)
            && created_at > 0.0
            && created_at <= limit
            && integral(completed_at)
            && completed_at >= created_at
            && completed_at <= limit
            && integral(expires_at)
            && expires_at > completed_at
            && expires_at <= limit;
        if value.member_str("schema", "") != "gufo.video-job.v1"
            || !is_video_id(&snapshot.id)
            || snapshot.id != directory_id
            || !is_h3_model(&snapshot.model)
            || resolve_duration_frames(&snapshot.seconds).is_err()
            || !valid_output
            || !valid_numbers
            || expires_at <= now as f64
            || value.member_str("status", "") != "completed"
        {
            return None;
        }
        snapshot.progress = progress as i32;
        snapshot.created_at = created_at as i64;
        snapshot.completed_at = completed_at as i64;
        snapshot.expires_at = expires_at as i64;
        snapshot.status = VideoJobStatus::Completed;
        let (content_path, media_type) = if snapshot.output_format == "mp4" {
            (path.join("output.mp4"), "video/mp4")
        } else {
            (path.join("output.ppm"), "image/x-portable-pixmap")
        };
        safe_regular_content(&self.canonical_storage_root, &content_path)?;
        Some(Job {
            id: snapshot.id.clone(),
            directory: path.to_path_buf(),
            cancellation: CancellationToken::default(),
            data: Mutex::new(JobData {
                snapshot,
                request: VideoJobRequest::default(),
                content_path,
                media_type: media_type.into(),
                deleted: false,
                logged_phase: String::new(),
                logged_bucket: -1,
            }),
        })
    }

    fn update_progress(&self, job: &Job, phase: &str, completed: i32, total: i32) {
        let _state = self.state();
        let mut data = job.data();
        if data.deleted || data.snapshot.status != VideoJobStatus::InProgress {
            return;
        }
        data.snapshot.progress = data
            .snapshot
            .progress
            .max(progress_percent(phase, completed, total).clamp(1, 99));
        let bucket = if total > 0 {
            (10 * i64::from(completed) / i64::from(total)) as i32
        } else {
            0
        };
        if data.logged_phase != phase || bucket > data.logged_bucket {
            data.logged_phase = phase.to_owned();
            data.logged_bucket = bucket;
            logging::info(
                "video",
                &format!(
                    "job={} event=progress phase={phase} completed={completed} total={total} {}",
                    job.id,
                    logging::memory_status()
                ),
            );
        }
    }

    /// Records a failed transition under the lock.
    fn fail_locked(&self, job: &Job, state: &mut State, message: &str) {
        state.active = None;
        let mut data = job.data();
        wipe(&mut data.request.prompt);
        data.snapshot.status = VideoJobStatus::Failed;
        data.snapshot.completed_at = self.now();
        data.snapshot.expires_at = data.snapshot.completed_at + self.ttl_seconds();
        data.snapshot.error_code = "storage_error".into();
        data.snapshot.error_message = message.into();
        let _ = self.persist(&data.snapshot);
    }

    fn run(self: &Arc<Self>) {
        while !self.stop.load(Ordering::Acquire) {
            let job = {
                let mut state = self.state();
                while !self.stop.load(Ordering::Acquire) && state.queue.is_empty() {
                    state = self
                        .condition
                        .wait(state)
                        .unwrap_or_else(|e| e.into_inner());
                }
                if self.stop.load(Ordering::Acquire) {
                    break;
                }
                let Some(job) = state.queue.pop_front() else {
                    continue;
                };
                if job.data().deleted {
                    continue;
                }
                state.active = Some(job.clone());
                let persisted = {
                    let mut data = job.data();
                    data.snapshot.status = VideoJobStatus::InProgress;
                    data.snapshot.progress = 1;
                    self.persist(&data.snapshot)
                };
                if persisted.is_err() {
                    self.fail_locked(&job, &mut state, "cannot persist video job transition");
                    drop(state);
                    self.condition.notify_all();
                    continue;
                }
                job
            };
            self.process(&job);
            self.condition.notify_all();
        }
    }

    fn process(&self, job: &Arc<Job>) {
        let start = Instant::now();
        // Everything but the prompt, which moves under the service lock.
        let (parameters, size, output_format, seed) = {
            let data = job.data();
            (
                data.request.parameters.clone(),
                data.request.size.clone(),
                data.request.output_format.clone(),
                data.request.seed,
            )
        };
        logging::info(
            "video",
            &format!(
                "job={} event=started preset={} size={size} frames={} steps={} seed={seed} {}",
                job.id,
                parameters.preset,
                parameters.frames,
                parameters.evaluations,
                logging::memory_status()
            ),
        );
        let mut generation = GenerationRequest {
            model_root: self.options.model_root.clone(),
            source_manifest: self.options.source_manifest.clone(),
            output_path: job.directory.join("output.mp4"),
            frames_directory: if output_format == "ppm" {
                job.directory.join("frames")
            } else {
                PathBuf::new()
            },
            latents_directory: PathBuf::new(),
            prompt: String::new(),
            seed,
            parameters,
        };
        {
            // Delete can clear a queued or active prompt concurrently.
            let _state = self.state();
            let mut data = job.data();
            generation.prompt = std::mem::take(&mut data.request.prompt);
            wipe(&mut data.request.prompt);
        }
        let prompt_hash = sha256_hex(generation.prompt.as_bytes());
        let parameters_json =
            generation_parameters_json(&generation.parameters, seed, &prompt_hash);
        if write_atomic(
            &job.directory.join("parameters.json"),
            parameters_json.as_bytes(),
        )
        .is_err()
        {
            wipe(&mut generation.prompt);
            let mut state = self.state();
            self.fail_locked(job, &mut state, "cannot persist video job parameters");
            return;
        }
        let mut telemetry = GenerationTelemetry::default();
        let outcome = match &self.options.runner {
            Some(runner) => {
                let mut progress = |phase: &str, completed: i32, total: i32| {
                    self.update_progress(job, phase, completed, total);
                };
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    runner(
                        &generation,
                        &job.cancellation,
                        &mut progress,
                        &mut telemetry,
                    )
                }))
                .unwrap_or_else(|_| Err("video generation backend failed unexpectedly".into()))
            }
            None => Err("video generation backend is unavailable".into()),
        };
        wipe(&mut generation.prompt);
        let success = outcome.is_ok();
        if let Err(error) = &outcome
            && !job.cancellation.is_cancelled()
        {
            logging::error(
                "video",
                &format!("job={} event=backend_failed detail={error}", job.id),
            );
        }

        let mut content_path = PathBuf::new();
        let mut media_type = String::new();
        let mut output_ready = success && !job.cancellation.is_cancelled();
        if output_ready && output_format == "mp4" {
            content_path = job.directory.join("output.mp4");
            media_type = "video/mp4".into();
        } else if output_ready {
            let frame = generation
                .parameters
                .selected_frames
                .first()
                .copied()
                .unwrap_or(0);
            let generated = job
                .directory
                .join("frames")
                .join(format!("frame-{frame:04}.ppm"));
            content_path = job.directory.join("output.ppm");
            if fs::rename(&generated, &content_path).is_ok() {
                let _ = fs::remove_dir_all(job.directory.join("frames"));
            } else {
                output_ready = false;
            }
            media_type = "image/x-portable-pixmap".into();
        }
        if output_ready {
            output_ready =
                safe_regular_content(&self.canonical_storage_root, &content_path).is_some();
        }
        let mut storage_failure = false;
        if output_ready
            && write_atomic(
                &job.directory.join("telemetry.json"),
                generation_telemetry_json(&telemetry).as_bytes(),
            )
            .is_err()
        {
            output_ready = false;
            storage_failure = true;
        }

        let (snapshot, deleted) = {
            let mut state = self.state();
            state.active = None;
            let mut data = job.data();
            let deleted = data.deleted;
            let mut snapshot = VideoJobSnapshot::default();
            if !deleted {
                let completed_at = self.now();
                snapshot = data.snapshot.clone();
                snapshot.completed_at = completed_at;
                snapshot.expires_at = completed_at + self.ttl_seconds();
                if output_ready {
                    snapshot.status = VideoJobStatus::Completed;
                    snapshot.progress = 100;
                } else if job.cancellation.is_cancelled() {
                    snapshot.status = VideoJobStatus::Cancelled;
                    snapshot.error_code = "cancelled".into();
                    snapshot.error_message = "video generation was cancelled".into();
                } else if storage_failure {
                    snapshot.status = VideoJobStatus::Failed;
                    snapshot.error_code = "storage_error".into();
                    snapshot.error_message = "cannot persist video job telemetry".into();
                } else {
                    snapshot.status = VideoJobStatus::Failed;
                    snapshot.error_code = "generation_failed".into();
                    snapshot.error_message = "video generation failed".into();
                }
                if self.persist(&snapshot).is_err() {
                    snapshot.status = VideoJobStatus::Failed;
                    snapshot.progress = snapshot.progress.min(99);
                    snapshot.error_code = "storage_error".into();
                    snapshot.error_message = "cannot persist video job completion".into();
                    let _ = self.persist(&snapshot);
                }
                data.snapshot = snapshot.clone();
                if snapshot.status == VideoJobStatus::Completed {
                    data.content_path = std::mem::take(&mut content_path);
                    data.media_type = std::mem::take(&mut media_type);
                }
            }
            (snapshot, deleted)
        };
        logging::log(
            if snapshot.status == VideoJobStatus::Failed {
                LogLevel::Error
            } else {
                LogLevel::Info
            },
            "video",
            &format!(
                "job={} event=finished state={} elapsed_ms={} error_code={} {}",
                job.id,
                if deleted {
                    "deleted"
                } else {
                    snapshot.status.name()
                },
                start.elapsed().as_millis(),
                snapshot.error_code,
                logging::memory_status()
            ),
        );
        if deleted {
            let _ = fs::remove_dir_all(&job.directory);
        } else if snapshot.status != VideoJobStatus::Completed {
            let _ = fs::remove_file(job.directory.join("output.mp4"));
            let _ = fs::remove_file(job.directory.join("output.ppm"));
            let _ = fs::remove_dir_all(job.directory.join("frames"));
            let _ = fs::remove_file(job.directory.join("telemetry.json"));
        }
    }
}

impl VideoJobService {
    pub fn new(options: VideoJobServiceOptions) -> Self {
        let failed = |error: String| Self {
            shared: None,
            initialization_error: error,
            worker: None,
        };
        if options.model_root.as_os_str().is_empty()
            || options.source_manifest.as_os_str().is_empty()
            || options.storage_root.as_os_str().is_empty()
            || options.queue_capacity == 0
            || options.artifact_ttl.as_secs() == 0
        {
            return failed("invalid MiniMax H3 video job configuration".into());
        }
        if options.runner.is_none() {
            return failed(
                "MiniMax H3 video serving requires a ROCm HIP-enabled build; no CPU inference \
                 fallback is provided"
                    .into(),
            );
        }
        if options.validate_model_inventory {
            let Some(check) = &options.inventory_check else {
                return failed(
                    "MiniMax H3 video serving requires a ROCm HIP-enabled build; no CPU inference \
                     fallback is provided"
                        .into(),
                );
            };
            if let Err(error) = check(&options.model_root, &options.source_manifest) {
                return failed(error);
            }
        }
        if let Err(error) = fs::create_dir_all(&options.storage_root) {
            return failed(format!("cannot create video job storage: {error}"));
        }
        let canonical_storage_root = match fs::canonicalize(&options.storage_root) {
            Ok(root) => root,
            Err(error) => return failed(format!("cannot resolve video job storage: {error}")),
        };
        let mut options = options;
        options.storage_root = canonical_storage_root.clone();
        let shared = Arc::new(Shared {
            options,
            canonical_storage_root,
            state: Mutex::new(State::default()),
            condition: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        {
            let mut state = shared.state();
            if let Err(error) = shared.recover_jobs(&mut state) {
                return failed(error);
            }
        }
        let worker_shared = shared.clone();
        let worker = std::thread::Builder::new()
            .name("gufo-video".into())
            .spawn(move || worker_shared.run());
        match worker {
            Ok(worker) => Self {
                shared: Some(shared),
                initialization_error: String::new(),
                worker: Some(worker),
            },
            Err(error) => failed(format!("cannot start the video worker: {error}")),
        }
    }

    pub fn ready(&self) -> bool {
        self.shared.is_some()
    }

    pub fn initialization_error(&self) -> String {
        self.initialization_error.clone()
    }

    pub fn create(&self, request: &VideoJobRequest) -> VideoJobCreateResult {
        let reject = |result: VideoJobResult, error: &str| VideoJobCreateResult {
            result,
            job: None,
            error: error.to_owned(),
        };
        let Some(shared) = &self.shared else {
            return reject(VideoJobResult::IoError, &self.initialization_error);
        };
        if !is_h3_model(&request.model)
            || request.prompt.is_empty()
            || request.prompt.contains('\0')
            || (request.output_format != "mp4" && request.output_format != "ppm")
            || validate_generation_parameters(&request.parameters).is_err()
            || !matches_output_contract(request)
        {
            return reject(
                VideoJobResult::Invalid,
                "invalid MiniMax H3 video job request",
            );
        }
        let id = shared
            .options
            .id_factory
            .as_ref()
            .map_or_else(random_video_id, |factory| factory());
        let mut job_request = request.clone();
        if !is_video_id(&id) {
            wipe(&mut job_request.prompt);
            return reject(
                VideoJobResult::Invalid,
                "video job ID factory returned an invalid ID",
            );
        }
        let snapshot = VideoJobSnapshot {
            id: id.clone(),
            status: VideoJobStatus::Queued,
            created_at: shared.now(),
            model: request.model.clone(),
            size: request.size.clone(),
            seconds: request.seconds.clone(),
            output_format: request.output_format.clone(),
            ..VideoJobSnapshot::default()
        };
        let directory = shared.options.storage_root.join(&id);
        let admitted = {
            let mut state = shared.state();
            shared.sweep_expired(&mut state);
            if state.queue.len() >= shared.options.queue_capacity {
                wipe(&mut job_request.prompt);
                return reject(VideoJobResult::QueueFull, "MiniMax H3 video queue is full");
            }
            if state.jobs.contains_key(&id) {
                wipe(&mut job_request.prompt);
                return reject(VideoJobResult::Invalid, "duplicate video job ID");
            }
            match directory.try_exists() {
                Err(_) => {
                    wipe(&mut job_request.prompt);
                    return reject(
                        VideoJobResult::IoError,
                        "cannot inspect video job destination",
                    );
                }
                Ok(true) => {
                    wipe(&mut job_request.prompt);
                    return reject(
                        VideoJobResult::Invalid,
                        "video job destination already exists",
                    );
                }
                Ok(false) => {}
            }
            let job = Arc::new(Job {
                id: id.clone(),
                directory: directory.clone(),
                cancellation: CancellationToken::default(),
                data: Mutex::new(JobData {
                    snapshot: snapshot.clone(),
                    request: job_request,
                    content_path: PathBuf::new(),
                    media_type: String::new(),
                    deleted: false,
                    logged_phase: String::new(),
                    logged_bucket: -1,
                }),
            });
            state.jobs.insert(id.clone(), job.clone());
            state.queue.push_back(job.clone());
            if let Err(error) = shared.persist(&snapshot) {
                state.queue.pop_back();
                state.jobs.remove(&id);
                let _ = fs::remove_dir_all(&directory);
                wipe(&mut job.data().request.prompt);
                return reject(VideoJobResult::IoError, &error);
            }
            snapshot
        };
        logging::info(
            "video",
            &format!(
                "job={} event=queued preset={}",
                admitted.id, request.parameters.preset
            ),
        );
        shared.condition.notify_one();
        VideoJobCreateResult {
            result: VideoJobResult::Ok,
            job: Some(admitted),
            error: String::new(),
        }
    }

    fn not_found() -> VideoJobLookupResult {
        VideoJobLookupResult {
            result: VideoJobResult::NotFound,
            job: None,
            error: "video job not found".into(),
        }
    }

    pub fn get(&self, id: &str) -> VideoJobLookupResult {
        let Some(shared) = &self.shared else {
            return Self::not_found();
        };
        if !is_video_id(id) {
            return Self::not_found();
        }
        let mut state = shared.state();
        shared.sweep_expired(&mut state);
        match state.jobs.get(id) {
            Some(job) => VideoJobLookupResult {
                result: VideoJobResult::Ok,
                job: Some(job.data().snapshot.clone()),
                error: String::new(),
            },
            None => Self::not_found(),
        }
    }

    pub fn content(&self, id: &str) -> VideoJobContentResult {
        let failure = |result: VideoJobResult, error: &str| VideoJobContentResult {
            result,
            content: None,
            error: error.to_owned(),
        };
        let Some(shared) = &self.shared else {
            return failure(VideoJobResult::NotFound, "video job not found");
        };
        let (path, media_type) = {
            let mut state = shared.state();
            shared.sweep_expired(&mut state);
            let Some(job) = state.jobs.get(id) else {
                return failure(VideoJobResult::NotFound, "video job not found");
            };
            let data = job.data();
            if data.snapshot.status != VideoJobStatus::Completed {
                return failure(VideoJobResult::NotReady, "video content is not ready");
            }
            (data.content_path.clone(), data.media_type.clone())
        };
        match safe_regular_content(&shared.canonical_storage_root, &path) {
            Some(bytes) => VideoJobContentResult {
                result: VideoJobResult::Ok,
                content: Some(VideoJobContent {
                    path,
                    media_type,
                    bytes,
                }),
                error: String::new(),
            },
            None => failure(VideoJobResult::IoError, "video content is unavailable"),
        }
    }

    pub fn delete(&self, id: &str) -> VideoJobLookupResult {
        let Some(shared) = &self.shared else {
            return Self::not_found();
        };
        let (job, snapshot, was_active) = {
            let mut state = shared.state();
            shared.sweep_expired(&mut state);
            let Some(job) = state.jobs.remove(id) else {
                return Self::not_found();
            };
            let snapshot = {
                let mut data = job.data();
                data.deleted = true;
                wipe(&mut data.request.prompt);
                data.snapshot.clone()
            };
            job.cancellation.cancel();
            let was_active = state.active.as_ref().is_some_and(|a| Arc::ptr_eq(a, &job));
            state.queue.retain(|queued| !Arc::ptr_eq(queued, &job));
            (job, snapshot, was_active)
        };
        if !was_active {
            let _ = fs::remove_dir_all(&job.directory);
        }
        shared.condition.notify_all();
        VideoJobLookupResult {
            result: VideoJobResult::Ok,
            job: Some(snapshot),
            error: String::new(),
        }
    }
}

impl Drop for VideoJobService {
    fn drop(&mut self) {
        let Some(shared) = &self.shared else {
            return;
        };
        shared.stop.store(true, Ordering::Release);
        {
            let state = shared.state();
            if let Some(active) = &state.active {
                active.cancellation.cancel();
            }
            for job in &state.queue {
                job.cancellation.cancel();
                wipe(&mut job.data().request.prompt);
            }
        }
        shared.condition.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
