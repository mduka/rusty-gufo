//! Port of `tests/cli/video_api_test.cpp`.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use gufo_core::json::Value;
use gufo_serve::http::{HttpRequest, HttpResponse};
use gufo_serve::video_api::handle_video_api_request;
use gufo_serve::video_jobs::{
    CancellationToken, GenerationRequest, VideoJobService, VideoJobServiceOptions, VideoRunner,
};
use gufo_serve::video_model::GenerationProgress;

fn next_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("video_{:024x}", COUNTER.fetch_add(1, Ordering::Relaxed) + 1)
}

#[derive(Default)]
struct RunnerState {
    released: bool,
    calls: usize,
    latest: GenerationRequest,
}

struct FakeRunner {
    blocked: bool,
    state: Mutex<RunnerState>,
    condition: Condvar,
}

impl FakeRunner {
    fn new(blocked: bool) -> Arc<Self> {
        Arc::new(Self {
            blocked,
            state: Mutex::default(),
            condition: Condvar::new(),
        })
    }

    fn run(
        &self,
        request: &GenerationRequest,
        cancellation: &CancellationToken,
        progress: &mut GenerationProgress<'_>,
    ) -> Result<(), String> {
        {
            let mut state = self.state.lock().expect("runner");
            state.calls += 1;
            state.latest = request.clone();
        }
        self.condition.notify_all();
        progress("denoiser", 1, 2);
        {
            let mut state = self.state.lock().expect("runner");
            while self.blocked && !state.released && !cancellation.is_cancelled() {
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
            fs::write(&request.output_path, "fake-video-content").map_err(|e| e.to_string())
        } else {
            fs::create_dir_all(&request.frames_directory).map_err(|e| e.to_string())?;
            let frame = request.parameters.selected_frames[0];
            fs::write(
                request
                    .frames_directory
                    .join(format!("frame-{frame:04}.ppm")),
                "P6\n1 1\n255\nabc",
            )
            .map_err(|e| e.to_string())
        }
    }

    fn wait_for_calls(&self, expected: usize) {
        let state = self.state.lock().expect("runner");
        let (_state, timeout) = self
            .condition
            .wait_timeout_while(state, Duration::from_secs(5), |s| s.calls < expected)
            .expect("runner");
        assert!(!timeout.timed_out(), "timed out waiting for fake runner");
    }

    fn release(&self) {
        self.state.lock().expect("runner").released = true;
        self.condition.notify_all();
    }

    fn latest(&self) -> GenerationRequest {
        self.state.lock().expect("runner").latest.clone()
    }
}

fn service(root: &Path, runner: &Arc<FakeRunner>) -> Arc<VideoJobService> {
    let runner = runner.clone();
    let boxed: VideoRunner = Box::new(move |request, cancellation, progress, _| {
        runner.run(request, cancellation, progress)
    });
    let service = VideoJobService::new(VideoJobServiceOptions {
        model_root: "/private/models/MiniMax-H3".into(),
        source_manifest: "/private/manifest.json".into(),
        storage_root: root.to_path_buf(),
        queue_capacity: 1,
        artifact_ttl: Duration::from_secs(60),
        validate_model_inventory: false,
        id_factory: Some(Box::new(next_id)),
        now: Some(Box::new(|| 1000)),
        runner: Some(boxed),
        inventory_check: None,
    });
    assert!(service.ready(), "{}", service.initialization_error());
    Arc::new(service)
}

fn send(
    service: &Arc<VideoJobService>,
    method: &str,
    path: &str,
    body: &[u8],
    range: &str,
    content_type: &str,
) -> HttpResponse {
    let mut request = HttpRequest {
        method: method.into(),
        path: path.into(),
        body: body.to_vec(),
        ..HttpRequest::default()
    };
    if !range.is_empty() {
        request.headers.push(("RaNgE".into(), range.into()));
    }
    if !content_type.is_empty() {
        request
            .headers
            .push(("CoNtEnT-TyPe".into(), content_type.into()));
    }
    handle_video_api_request(&request, service).expect("video API")
}

fn post(service: &Arc<VideoJobService>, body: &str) -> HttpResponse {
    send(service, "POST", "/v1/videos", body.as_bytes(), "", "")
}

fn get(service: &Arc<VideoJobService>, path: &str) -> HttpResponse {
    send(service, "GET", path, b"", "", "")
}

fn text(response: &HttpResponse) -> String {
    String::from_utf8_lossy(&response.body).into_owned()
}

fn multipart(boundary: &str, fields: &[(&str, &str)]) -> Vec<u8> {
    let mut body = String::new();
    for (name, value) in fields {
        body.push_str(&format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        ));
    }
    body.push_str(&format!("--{boundary}--\r\n"));
    body.into_bytes()
}

fn created_id(response: &HttpResponse) -> String {
    gufo_core::json::parse(&response.body)
        .expect("JSON")
        .member_str("id", "")
}

fn wait_completed(service: &Arc<VideoJobService>, id: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if text(&get(service, &format!("/v1/videos/{id}"))).contains("\"status\":\"completed\"") {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("video API job did not complete");
}

#[test]
fn lifecycle_and_range() {
    let base = tempfile::tempdir().expect("tempdir");
    let runner = FakeRunner::new(false);
    let service = service(&base.path().join("lifecycle"), &runner);
    let request = r#"{"model":"minimax-h3-dev","prompt":"private fox prompt","size":"256x256","seconds":"1","gufo":{"seed":7,"frames":22,"output_format":"ppm","selected_frame":4}}"#;
    let created = send(
        &service,
        "POST",
        "/v1/videos",
        request.as_bytes(),
        "",
        "application/json; charset=utf-8",
    );
    assert!(
        created.status == 202
            && !text(&created).contains("private fox prompt")
            && text(&created).contains("\"status\":\"queued\""),
        "create returns a private, queued video object"
    );
    let id = created_id(&created);
    assert!(!id.is_empty(), "create response includes an ID");
    assert!(
        created.log_details.contains(&format!("job={id}"))
            && !created.log_details.contains("private fox prompt"),
        "request diagnostics link the worker job without logging the prompt"
    );
    runner.wait_for_calls(1);
    wait_completed(&service, &id);
    let generation = runner.latest();
    assert!(
        generation.seed == 7
            && generation.parameters.preset == "development-256"
            && generation.parameters.frames == 22
            && generation.parameters.selected_frames == vec![4]
            && !generation.parameters.mux,
        "HTTP adapter preserves direct generation parameters"
    );
    let content = get(&service, &format!("/v1/videos/{id}/content"));
    assert!(
        content.status == 200 && content.body == b"P6\n1 1\n255\nabc",
        "completed diagnostic frame is returned"
    );
    let partial = send(
        &service,
        "GET",
        &format!("/v1/videos/{id}/content"),
        b"",
        "bytes=0-4",
        "",
    );
    assert!(
        partial.status == 206 && partial.body == b"P6\n1 ",
        "single HTTP byte range is returned"
    );
    assert!(
        partial
            .headers
            .iter()
            .any(|(name, value)| name == "Content-Range" && value == "bytes 0-4/14"),
        "partial content includes a stable Content-Range"
    );
    let invalid_range = send(
        &service,
        "GET",
        &format!("/v1/videos/{id}/content"),
        b"",
        "bytes=99-100",
        "",
    );
    assert_eq!(
        invalid_range.status, 416,
        "unsatisfiable range returns HTTP 416"
    );
    let deleted = send(&service, "DELETE", &format!("/v1/videos/{id}"), b"", "", "");
    assert!(
        deleted.status == 200 && text(&deleted).contains("\"deleted\":true"),
        "DELETE returns a deletion object"
    );
    assert_eq!(
        get(&service, &format!("/v1/videos/{id}")).status,
        404,
        "deleted jobs are no longer visible"
    );
}

#[test]
fn validation() {
    let base = tempfile::tempdir().expect("tempdir");
    let runner = FakeRunner::new(false);
    let service = service(&base.path().join("validation"), &runner);
    let fullres = post(
        &service,
        r#"{"model":"minimax-h3","prompt":"A goalkeeper saves a shot","size":"1344x768","seconds":5}"#,
    );
    assert_eq!(
        fullres.status, 202,
        "released full-resolution API request accepted"
    );
    runner.wait_for_calls(1);
    let p = runner.latest().parameters;
    assert!(
        p.preset == "exact-1344x768"
            && p.internal_width == 1344
            && p.internal_height == 768
            && p.frames == 124
            && p.evaluations == 49,
        "API maps 1344x768 five-second request to the exact preset"
    );
    wait_completed(&service, &created_id(&fullres));

    let expect_error = |body: &str, code: &str, message: &str| {
        let response = post(&service, body);
        assert!(
            response.status == 400 && text(&response).contains(code),
            "{message}: {}",
            text(&response)
        );
        response
    };
    expect_error(
        r#"{"model":"minimax-h3","prompt":"x","seconds":"1","input_reference":"secret.png"}"#,
        "unsupported_input_reference",
        "future conditioning inputs fail closed",
    );
    assert_eq!(
        post(
            &service,
            r#"{"model":"minimax-h3","prompt":"x","size":"512x512","seconds":"1","gufo":{"preset":"dev","output_format":"mp4"}}"#
        )
        .status,
        400,
        "development preset cannot silently produce MP4"
    );
    expect_error(
        r#"{"model":"minimax-h3-fast","prompt":"x","size":"512x512","seconds":"1","gufo":{"preset":"exact"}}"#,
        "preset_model_mismatch",
        "model aliases cannot drift from the frozen preset",
    );
    let too_short = expect_error(
        r#"{"model":"minimax-h3-dev","prompt":"x","size":"256x256","seconds":"1","gufo":{"frames":5,"output_format":"ppm","selected_frame":4}}"#,
        "invalid_frames",
        "sub-chunk requests fail before the released VisualVAE is invoked",
    );
    assert!(text(&too_short).contains("at least 22 aligned frames"));
    expect_error(
        r#"{"model":"minimax-h3","prompt":"x","seconds":"1","silent_typo":true}"#,
        "unsupported_field",
        "unknown top-level fields fail closed",
    );
    expect_error(
        r#"{"model":"minimax-h3","model":"minimax-h3-fast","prompt":"x","seconds":"1"}"#,
        "parse_error",
        "duplicate JSON fields fail closed",
    );
    expect_error(
        r#"{"model":"minimax-h3","prompt":"x","seconds":"1"} trailing"#,
        "parse_error",
        "trailing JSON input fails closed",
    );
    expect_error(
        r#"{"model":"minimax-h3","prompt":"x","seconds":"1","gufo":{"seed":1e}}"#,
        "parse_error",
        "malformed JSON numbers fail closed",
    );
    expect_error(
        r#"{"model":"minimax-h3","prompt":"x","size":512,"seconds":"1"}"#,
        "invalid_parameter_type",
        "wrongly typed optional fields do not fall back to defaults",
    );
    assert_eq!(
        get(&service, "/v1/videos/../../private").status,
        404,
        "path traversal-like IDs fail closed"
    );

    let boundary = "gufo-openai-video-boundary";
    let accepted = send(
        &service,
        "POST",
        "/v1/videos",
        &multipart(
            boundary,
            &[
                ("model", "minimax-h3-fast"),
                ("prompt", "A red fox walking through snow"),
                ("size", "512x512"),
                ("seconds", "1"),
            ],
        ),
        "",
        "multipart/form-data; boundary=\"gufo-openai-video-boundary\"",
    );
    assert_eq!(
        accepted.status, 202,
        "OpenAI-style multipart text-to-video request is admitted"
    );
    runner.wait_for_calls(2);
    assert_eq!(
        runner.latest().parameters.preset,
        "fast-384",
        "multipart request preserves the selected H3 mode"
    );
    let duplicate = send(
        &service,
        "POST",
        "/v1/videos",
        &multipart(
            boundary,
            &[
                ("model", "minimax-h3"),
                ("prompt", "first"),
                ("prompt", "second"),
                ("seconds", "1"),
            ],
        ),
        "",
        "multipart/form-data; boundary=gufo-openai-video-boundary",
    );
    assert!(
        duplicate.status == 400 && text(&duplicate).contains("invalid_multipart_form"),
        "multipart duplicate fields fail closed"
    );
    let reference_body = "--gufo-openai-video-boundary\r\n\
         Content-Disposition: form-data; name=\"model\"\r\n\r\n\
         minimax-h3\r\n\
         --gufo-openai-video-boundary\r\n\
         Content-Disposition: form-data; name=\"prompt\"\r\n\r\n\
         x\r\n\
         --gufo-openai-video-boundary\r\n\
         Content-Disposition: form-data; name=\"seconds\"\r\n\r\n\
         1\r\n\
         --gufo-openai-video-boundary\r\n\
         Content-Disposition: form-data; name=\"input_reference\"; filename=\"frame.png\"\r\n\
         Content-Type: image/png\r\n\r\n\
         not-an-image\r\n\
         --gufo-openai-video-boundary--\r\n";
    let multipart_reference = send(
        &service,
        "POST",
        "/v1/videos",
        reference_body.as_bytes(),
        "",
        "multipart/form-data; boundary=gufo-openai-video-boundary",
    );
    assert!(
        multipart_reference.status == 400
            && text(&multipart_reference).contains("unsupported_input_reference"),
        "multipart first-frame conditioning remains explicit future work"
    );
    let unsupported_media = send(
        &service,
        "POST",
        "/v1/videos",
        b"model=minimax-h3",
        "",
        "application/x-www-form-urlencoded",
    );
    assert!(
        unsupported_media.status == 415
            && text(&unsupported_media).contains("unsupported_media_type"),
        "unsupported create content types fail explicitly"
    );
}

#[test]
fn durations() {
    let base = tempfile::tempdir().expect("tempdir");
    let runner = FakeRunner::new(false);
    let service = service(&base.path().join("durations"), &runner);
    let mut calls = 0;
    for size in ["512x512", "1344x768"] {
        for (seconds, frames) in [("5", 124), ("8", 192), ("10", 243), ("14.375", 345)] {
            let response = post(
                &service,
                &format!(
                    r#"{{"model":"minimax-h3","prompt":"x","size":"{size}","seconds":{seconds}}}"#
                ),
            );
            assert_eq!(
                response.status,
                202,
                "numeric aligned duration accepted: {}",
                text(&response)
            );
            calls += 1;
            runner.wait_for_calls(calls);
            wait_completed(&service, &created_id(&response));
            assert_eq!(
                runner.latest().parameters.frames,
                frames,
                "duration reaches generation with official aligned frames"
            );
        }
    }
    let boundary = "h3-duration";
    let response = send(
        &service,
        "POST",
        "/v1/videos",
        &multipart(
            boundary,
            &[
                ("model", "minimax-h3"),
                ("prompt", "x"),
                ("size", "1344x768"),
                ("seconds", "14.375"),
            ],
        ),
        "",
        &format!("multipart/form-data; boundary={boundary}"),
    );
    assert_eq!(
        response.status, 202,
        "multipart accepts fractional duration"
    );
    calls += 1;
    runner.wait_for_calls(calls);
    wait_completed(&service, &created_id(&response));
    assert_eq!(
        runner.latest().parameters.frames,
        345,
        "multipart resolves the maximum aligned duration"
    );
    for seconds in ["0", "4", "14.376", "15", "16"] {
        let response = post(
            &service,
            &format!(r#"{{"model":"minimax-h3","prompt":"x","seconds":"{seconds}"}}"#),
        );
        assert!(
            response.status == 400 && text(&response).contains("invalid_seconds"),
            "unsupported duration fails before job admission"
        );
    }
    let mismatch = post(
        &service,
        r#"{"model":"minimax-h3","prompt":"x","seconds":10,"gufo":{"frames":124}}"#,
    );
    assert!(
        mismatch.status == 400 && text(&mismatch).contains("invalid_frames"),
        "explicit frames cannot disagree with duration"
    );
}

#[test]
fn long_prompts() {
    let base = tempfile::tempdir().expect("tempdir");
    let runner = FakeRunner::new(false);
    let service = service(&base.path().join("long-prompts"), &runner);
    let unicode = "e\u{301}".repeat(2048);
    let mut calls = 0;
    for prompt in ["A".repeat(8192), unicode] {
        for use_multipart in [false, true] {
            let (body, content_type) = if use_multipart {
                (
                    multipart(
                        "long-prompt",
                        &[
                            ("model", "minimax-h3"),
                            ("prompt", &prompt),
                            ("seconds", "1"),
                        ],
                    ),
                    "multipart/form-data; boundary=long-prompt",
                )
            } else {
                let json = Value::object()
                    .with("model", "minimax-h3")
                    .with("prompt", prompt.as_str())
                    .with("seconds", "1");
                (json.dump().expect("JSON").into_bytes(), "application/json")
            };
            let response = send(&service, "POST", "/v1/videos", &body, "", content_type);
            assert_eq!(
                response.status, 202,
                "byte length does not replace the encoder's token limit"
            );
            calls += 1;
            runner.wait_for_calls(calls);
            assert_eq!(
                runner.latest().prompt,
                prompt,
                "HTTP and job admission preserve the entire UTF-8 prompt"
            );
            wait_completed(&service, &created_id(&response));
        }
    }
}

#[test]
fn queue_saturation() {
    let base = tempfile::tempdir().expect("tempdir");
    let runner = FakeRunner::new(true);
    let service = service(&base.path().join("queue"), &runner);
    let body = r#"{"model":"minimax-h3","prompt":"x","size":"512x512","seconds":"1","gufo":{"preset":"fast"}}"#;
    let first = post(&service, body);
    runner.wait_for_calls(1);
    let second = post(&service, body);
    let third = post(&service, body);
    assert!(
        first.status == 202 && second.status == 202 && third.status == 429,
        "one active and one queued request enforce bounded admission"
    );
    assert!(
        third.headers.iter().any(|(name, _)| name == "Retry-After"),
        "queue saturation supplies Retry-After"
    );
    assert_eq!(
        send(
            &service,
            "DELETE",
            &format!("/v1/videos/{}", created_id(&first)),
            b"",
            "",
            ""
        )
        .status,
        200,
        "active request can be cancelled through DELETE"
    );
    runner.wait_for_calls(2);
    runner.release();
}
