//! Port of `tests/cli/video_jobs_test.cpp`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use gufo_serve::video_jobs::{
    CancellationToken, GenerationRequest, GenerationTelemetry, VideoJobRequest, VideoJobResult,
    VideoJobService, VideoJobServiceOptions, VideoJobStatus, VideoRunner,
};
use gufo_serve::video_model::{GenerationProgress, resolve_generation_preset};

#[derive(Default)]
struct RunnerState {
    calls: usize,
    release: bool,
    requests: Vec<GenerationRequest>,
}

#[derive(Default)]
struct FakeRunner {
    state: Mutex<RunnerState>,
    condition: Condvar,
}

impl FakeRunner {
    fn released() -> Arc<Self> {
        let runner = Self::default();
        runner.state.lock().expect("runner").release = true;
        Arc::new(runner)
    }

    fn run(
        &self,
        request: &GenerationRequest,
        cancellation: &CancellationToken,
        progress: &mut GenerationProgress<'_>,
        telemetry: &mut GenerationTelemetry,
    ) -> Result<(), String> {
        {
            let mut state = self.state.lock().expect("runner");
            state.calls += 1;
            state.requests.push(request.clone());
        }
        self.condition.notify_all();
        progress("inventory", 1, 1);
        progress("prompt", 1, 1);
        progress("denoiser", 2, 4);
        progress("video_vae", 1, 1);
        {
            let mut state = self.state.lock().expect("runner");
            while !state.release && !cancellation.is_cancelled() {
                state = self
                    .condition
                    .wait_timeout(state, Duration::from_millis(5))
                    .expect("runner")
                    .0;
            }
        }
        if cancellation.is_cancelled() {
            return Err("cancelled".into());
        }
        if request.parameters.mux {
            fs::create_dir_all(request.output_path.parent().expect("parent"))
                .map_err(|e| e.to_string())?;
            fs::write(&request.output_path, "fake-mp4-content").map_err(|e| e.to_string())?;
        } else {
            fs::create_dir_all(&request.frames_directory).map_err(|e| e.to_string())?;
            let frame = request.parameters.selected_frames[0];
            fs::write(
                request
                    .frames_directory
                    .join(format!("frame-{frame:04}.ppm")),
                "P6\n1 1\n255\nabc",
            )
            .map_err(|e| e.to_string())?;
        }
        telemetry.prompt_tokens = 6;
        telemetry.denoiser.forward_ms = vec![1.0, 2.0];
        Ok(())
    }

    fn wait_for_calls(&self, expected: usize) {
        let state = self.state.lock().expect("runner");
        let (_state, timeout) = self
            .condition
            .wait_timeout_while(state, Duration::from_secs(5), |s| s.calls < expected)
            .expect("runner");
        assert!(
            !timeout.timed_out(),
            "timed out waiting for fake video runner"
        );
    }

    fn release(&self) {
        self.state.lock().expect("runner").release = true;
        self.condition.notify_all();
    }

    fn boxed(self: &Arc<Self>) -> VideoRunner {
        let runner = self.clone();
        Box::new(move |request, cancellation, progress, telemetry| {
            runner.run(request, cancellation, progress, telemetry)
        })
    }
}

fn next_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("video_{:024x}", COUNTER.fetch_add(1, Ordering::Relaxed) + 1)
}

fn mp4_request(prompt: &str) -> VideoJobRequest {
    VideoJobRequest {
        model: "minimax-h3".into(),
        prompt: prompt.into(),
        size: "512x512".into(),
        seconds: "1".into(),
        output_format: "mp4".into(),
        parameters: resolve_generation_preset("fast").expect("fast preset resolves"),
        seed: 42,
    }
}

fn wait_until(mut predicate: impl FnMut() -> bool, message: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if predicate() {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("{message}");
}

fn status_is(service: &VideoJobService, id: &str, status: VideoJobStatus) -> bool {
    service.get(id).job.is_some_and(|job| job.status == status)
}

fn write_completed_job(directory: &Path, id: &str, format: &str) {
    fs::create_dir_all(directory).expect("directory");
    fs::write(
        directory.join("job.json"),
        format!(
            "{{\"schema\":\"gufo.video-job.v1\",\"id\":\"{id}\",\"status\":\"completed\",\"progress\":100,\
             \"created_at\":3000,\"completed_at\":3001,\"expires_at\":4000,\"model\":\"minimax-h3\",\
             \"size\":\"{}\",\"seconds\":\"1\",\"output_format\":\"{format}\",\"error\":null}}\n",
            if format == "mp4" { "512x512" } else { "256x256" }
        ),
    )
    .expect("metadata");
}

fn options(root: &Path, now: i64, runner: VideoRunner) -> VideoJobServiceOptions {
    VideoJobServiceOptions {
        model_root: "/models/h3".into(),
        source_manifest: "/manifest.json".into(),
        storage_root: root.to_path_buf(),
        queue_capacity: 1,
        artifact_ttl: Duration::from_secs(3600),
        validate_model_inventory: false,
        id_factory: Some(Box::new(next_id)),
        now: Some(Box::new(move || now)),
        runner: Some(runner),
        inventory_check: None,
    }
}

fn failing_runner() -> VideoRunner {
    Box::new(|_, _, _, _| Err(String::new()))
}

#[test]
fn queue_lifecycle_and_recovery() {
    let base = tempfile::tempdir().expect("tempdir");
    let root = base.path().join("lifecycle");
    let runner = Arc::new(FakeRunner::default());
    let now = Arc::new(AtomicI64::new(1000));
    let make_options = || {
        let now = now.clone();
        VideoJobServiceOptions {
            model_root: "/private/models/MiniMax-H3".into(),
            source_manifest: "/private/manifest.json".into(),
            storage_root: root.clone(),
            queue_capacity: 1,
            artifact_ttl: Duration::from_secs(60),
            validate_model_inventory: false,
            id_factory: Some(Box::new(next_id)),
            now: Some(Box::new(move || now.load(Ordering::Relaxed))),
            runner: Some(runner.boxed()),
            inventory_check: None,
        }
    };
    let completed_id;
    {
        let service = VideoJobService::new(make_options());
        assert!(service.ready(), "{}", service.initialization_error());
        let first = service.create(&mp4_request("private prompt one"));
        assert!(
            first.result == VideoJobResult::Ok && first.job.is_some(),
            "first video job is admitted"
        );
        let first_id = first.job.expect("job").id;
        runner.wait_for_calls(1);
        wait_until(
            || {
                service.get(&first_id).job.is_some_and(|job| {
                    job.status == VideoJobStatus::InProgress && job.progress >= 80
                })
            },
            "first job did not publish expected progress",
        );
        let running = service.get(&first_id).job.expect("job");
        assert_eq!(
            running.status,
            VideoJobStatus::InProgress,
            "first job enters in_progress"
        );
        assert!(
            running.progress >= 80 && running.progress < 100,
            "progress is monotonic and bounded before completion"
        );

        let mut longer = mp4_request("private prompt two");
        longer.seconds = "14.375".into();
        longer.parameters.frames = 345;
        let second = service.create(&longer);
        assert!(
            second.result == VideoJobResult::Ok && second.job.is_some(),
            "one queued video job is admitted"
        );
        completed_id = second.job.expect("job").id;
        assert_eq!(
            service.create(&mp4_request("third prompt")).result,
            VideoJobResult::QueueFull,
            "bounded queue rejects saturation"
        );
        assert_eq!(
            service.delete(&first_id).result,
            VideoJobResult::Ok,
            "active job deletion requests cancellation"
        );
        assert_eq!(
            service.get(&first_id).result,
            VideoJobResult::NotFound,
            "deleted active job disappears immediately"
        );
        runner.wait_for_calls(2);
        runner.release();
        wait_until(
            || status_is(&service, &completed_id, VideoJobStatus::Completed),
            "queued job did not complete",
        );
        assert_eq!(
            service.get(&completed_id).job.expect("job").progress,
            100,
            "completed progress reaches 100"
        );
        let content = service.content(&completed_id);
        assert!(
            content.result == VideoJobResult::Ok
                && content
                    .content
                    .as_ref()
                    .is_some_and(|c| c.media_type == "video/mp4"
                        && c.bytes == "fake-mp4-content".len() as u64),
            "completed MP4 content is retrievable"
        );
        {
            let state = runner.state.lock().expect("runner");
            assert_eq!(state.requests.len(), 2, "runner request count");
            let request = state.requests.last().expect("request");
            let p = &request.parameters;
            assert!(
                p.internal_width == 384
                    && p.output_width == 512
                    && p.evaluations == 19
                    && p.active_blocks == 45
                    && p.reuse_interval == 2
                    && p.frames == 345
                    && request.seed == 42,
                "server request matches direct fast CLI parameters"
            );
        }
        let directory = root.canonicalize().expect("root").join(&completed_id);
        let metadata = fs::read_to_string(directory.join("job.json")).expect("metadata");
        let parameters = fs::read_to_string(directory.join("parameters.json")).expect("parameters");
        let telemetry = fs::read_to_string(directory.join("telemetry.json")).expect("telemetry");
        assert!(
            metadata.contains("\"status\":\"completed\"")
                && telemetry.contains("gufo.minimax-h3-generation-telemetry.v1"),
            "completed state is published with telemetry"
        );
        assert!(
            !metadata.contains("private prompt")
                && !metadata.contains("/private/")
                && !parameters.contains("private prompt")
                && !parameters.contains("/private/"),
            "persisted job data excludes prompts and private paths"
        );
        fs::write(
            directory.join("job.json.gufo-partial-injected"),
            "incomplete",
        )
        .expect("partial");
    }
    {
        let recovered = VideoJobService::new(make_options());
        assert!(recovered.ready(), "{}", recovered.initialization_error());
        let directory = root.canonicalize().expect("root").join(&completed_id);
        assert!(
            !directory.join("job.json.gufo-partial-injected").exists(),
            "restart removes partial metadata files"
        );
        assert!(
            status_is(&recovered, &completed_id, VideoJobStatus::Completed),
            "completed job survives service restart"
        );
        assert_eq!(
            recovered.content(&completed_id).result,
            VideoJobResult::Ok,
            "recovered content remains retrievable"
        );
        assert_eq!(
            recovered.delete(&completed_id).result,
            VideoJobResult::Ok,
            "recovered job can be deleted"
        );
        assert!(!directory.exists(), "delete reclaims completed artifacts");
    }
}

#[test]
fn ppm_and_expiry() {
    let base = tempfile::tempdir().expect("tempdir");
    let root = base.path().join("ppm");
    let runner = FakeRunner::released();
    let now = Arc::new(AtomicI64::new(2000));
    let clock = now.clone();
    let service = VideoJobService::new(VideoJobServiceOptions {
        artifact_ttl: Duration::from_secs(5),
        now: Some(Box::new(move || clock.load(Ordering::Relaxed))),
        ..options(&root, 0, runner.boxed())
    });
    let mut parameters = resolve_generation_preset("dev").expect("development preset resolves");
    parameters.selected_frames = vec![11];
    let created = service.create(&VideoJobRequest {
        model: "minimax-h3".into(),
        prompt: "diagnostic".into(),
        size: "256x256".into(),
        seconds: "1".into(),
        output_format: "ppm".into(),
        parameters,
        seed: 7,
    });
    let id = created.job.expect("single-frame PPM job admitted").id;
    wait_until(
        || status_is(&service, &id, VideoJobStatus::Completed),
        "PPM job did not complete",
    );
    let content = service.content(&id);
    assert!(
        content
            .content
            .is_some_and(|c| c.media_type == "image/x-portable-pixmap"),
        "diagnostic PPM content type"
    );
    now.fetch_add(6, Ordering::Relaxed);
    assert_eq!(
        service.get(&id).result,
        VideoJobResult::NotFound,
        "expired job is no longer visible"
    );
    assert!(
        !root.canonicalize().expect("root").join(&id).exists(),
        "expired artifacts are reclaimed"
    );
}

#[test]
fn validation() {
    let base = tempfile::tempdir().expect("tempdir");
    let root = base.path().join("validation");
    let unavailable = VideoJobService::new(VideoJobServiceOptions {
        model_root: root.join("missing-model"),
        source_manifest: root.join("missing-manifest.json"),
        storage_root: root.join("unavailable"),
        queue_capacity: 1,
        artifact_ttl: Duration::from_secs(3600),
        validate_model_inventory: true,
        id_factory: None,
        now: None,
        runner: None,
        inventory_check: None,
    });
    assert!(
        !unavailable.ready(),
        "invalid checkpoint fails before video request admission"
    );

    let service = VideoJobService::new(VideoJobServiceOptions {
        now: None,
        ..options(&root, 0, failing_runner())
    });
    assert_eq!(
        service.create(&mp4_request("")).result,
        VideoJobResult::Invalid,
        "empty prompt is rejected"
    );
    assert_eq!(
        service.create(&mp4_request("A\0B")).result,
        VideoJobResult::Invalid,
        "embedded NUL in prompt is rejected"
    );
    let mut invalid = mp4_request("A red fox");
    invalid.seconds = "10".into();
    assert_eq!(
        service.create(&invalid).result,
        VideoJobResult::Invalid,
        "job admission rejects seconds/frame mismatch independently of HTTP"
    );
    invalid.parameters.frames = 362;
    invalid.seconds = "15".into();
    assert_eq!(
        service.create(&invalid).result,
        VideoJobResult::Invalid,
        "job admission rejects alignment beyond the released limit"
    );
    let mut invalid = mp4_request("A red fox");
    invalid.model = "../escape".into();
    assert_eq!(
        service.create(&invalid).result,
        VideoJobResult::Invalid,
        "invalid model and traversal-like value rejected"
    );
    let mut invalid = mp4_request("A red fox");
    invalid.model = "minimax-h3-exact".into();
    assert_eq!(
        service.create(&invalid).result,
        VideoJobResult::Invalid,
        "model alias cannot disagree with the frozen preset"
    );
    let mut development = resolve_generation_preset("dev").expect("development preset resolves");
    development.selected_frames = vec![11];
    let wrong_ppm_size = VideoJobRequest {
        model: "minimax-h3-dev".into(),
        prompt: "diagnostic".into(),
        size: "512x512".into(),
        seconds: "1".into(),
        output_format: "ppm".into(),
        parameters: development,
        seed: 42,
    };
    assert_eq!(
        service.create(&wrong_ppm_size).result,
        VideoJobResult::Invalid,
        "direct PPM jobs enforce development output geometry"
    );

    let collision_id = "video_0000000000000000000000d1";
    let collision_root = root.join("collision");
    let collision_service = VideoJobService::new(VideoJobServiceOptions {
        id_factory: Some(Box::new(move || collision_id.to_owned())),
        now: None,
        ..options(&collision_root, 0, failing_runner())
    });
    assert!(
        collision_service.ready(),
        "{}",
        collision_service.initialization_error()
    );
    fs::create_dir_all(collision_root.join(collision_id)).expect("collision");
    fs::write(
        collision_root.join(collision_id).join("sentinel"),
        "preserve",
    )
    .expect("sentinel");
    assert_eq!(
        collision_service.create(&mp4_request("A red fox")).result,
        VideoJobResult::Invalid,
        "pre-existing job destinations are never reused"
    );
    assert!(
        collision_root.join(collision_id).join("sentinel").is_file(),
        "destination collision preserves existing data"
    );
}

#[test]
fn recovery_hardening() {
    let base = tempfile::tempdir().expect("tempdir");
    let root = base.path().join("recovery-hardening");
    let directory_id = "video_0000000000000000000000a1";
    let claimed_id = "video_0000000000000000000000a2";
    let mismatched = root.join(directory_id);
    write_completed_job(&mismatched, claimed_id, "mp4");
    fs::write(mismatched.join("output.mp4"), "wrong-id").expect("output");

    let linked_content_id = "video_0000000000000000000000b1";
    let linked_content = root.join(linked_content_id);
    write_completed_job(&linked_content, linked_content_id, "mp4");
    let secret = root.join("outside-secret.mp4");
    fs::write(&secret, "must-not-be-served").expect("secret");
    std::os::unix::fs::symlink(&secret, linked_content.join("output.mp4"))
        .expect("content symlink fixture created");

    let linked_directory_id = "video_0000000000000000000000c1";
    let external = root.join("external-job");
    write_completed_job(&external, linked_directory_id, "mp4");
    fs::write(external.join("output.mp4"), "outside-directory").expect("external");
    std::os::unix::fs::symlink(&external, root.join(linked_directory_id))
        .expect("directory symlink fixture created");

    let service = VideoJobService::new(options(&root, 3100, failing_runner()));
    assert!(service.ready(), "{}", service.initialization_error());
    assert_eq!(
        service.get(claimed_id).result,
        VideoJobResult::NotFound,
        "recovery rejects metadata/directory ID mismatch"
    );
    assert_eq!(
        service.get(linked_content_id).result,
        VideoJobResult::NotFound,
        "recovery rejects symlinked content"
    );
    assert_eq!(
        service.get(linked_directory_id).result,
        VideoJobResult::NotFound,
        "recovery rejects symlinked job directory"
    );
    assert!(secret.is_file(), "recovery preserves the symlink target");
    assert!(
        external.join("output.mp4").is_file(),
        "recovery does not traverse the directory symlink"
    );
}

#[test]
fn runner_exception_recovery() {
    let base = tempfile::tempdir().expect("tempdir");
    let root = base.path().join("runner-exception");
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let runner: VideoRunner = Box::new(move |request, _, _, _| {
        if counter.fetch_add(1, Ordering::SeqCst) == 0 {
            panic!("injected backend exception");
        }
        fs::create_dir_all(request.output_path.parent().expect("parent"))
            .map_err(|e| e.to_string())?;
        fs::write(&request.output_path, "recovered-worker-output").map_err(|e| e.to_string())?;
        Ok(())
    });
    let service = VideoJobService::new(options(&root, 5000, runner));
    assert!(service.ready(), "{}", service.initialization_error());
    let first = service.create(&mp4_request("throwing request"));
    let first_id = first.job.expect("throwing backend request admitted").id;
    wait_until(
        || status_is(&service, &first_id, VideoJobStatus::Failed),
        "throwing backend did not become a failed job",
    );
    let second = service.create(&mp4_request("recovery request"));
    let second_id = second.job.expect("post-exception request admitted").id;
    wait_until(
        || status_is(&service, &second_id, VideoJobStatus::Completed),
        "worker did not recover after backend exception",
    );
    assert_eq!(
        service.content(&second_id).result,
        VideoJobResult::Ok,
        "post-exception worker publishes content"
    );
}

#[test]
fn runner_output_boundary() {
    let base = tempfile::tempdir().expect("tempdir");
    let root = base.path().join("runner-output-boundary");
    let external: PathBuf = base.path().join("runner-output-secret.mp4");
    fs::write(&external, "must-not-be-published").expect("external");
    let target = external.clone();
    let runner: VideoRunner = Box::new(move |request, _, _, _| {
        fs::create_dir_all(request.output_path.parent().expect("parent"))
            .map_err(|e| e.to_string())?;
        std::os::unix::fs::symlink(&target, &request.output_path).map_err(|e| e.to_string())
    });
    let service = VideoJobService::new(options(&root, 6000, runner));
    assert!(service.ready(), "{}", service.initialization_error());
    let created = service.create(&mp4_request("boundary request"));
    let id = created.job.expect("output-boundary request admitted").id;
    wait_until(
        || status_is(&service, &id, VideoJobStatus::Failed),
        "symlinked runner output did not fail",
    );
    assert_eq!(
        service.content(&id).result,
        VideoJobResult::NotReady,
        "symlinked runner output is not published"
    );
    assert_eq!(
        fs::read_to_string(&external).expect("external"),
        "must-not-be-published",
        "runner output rejection preserves the external target"
    );
}
