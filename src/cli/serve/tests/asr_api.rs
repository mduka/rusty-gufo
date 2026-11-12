//! Port of `src/models/qwen3_asr/tests/audio_api_test.cpp` (the API cases;
//! the chunking and token-budget cases test the engine, `gufo-qwen3-asr`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gufo_serve::asr_model::{
    TextSink, TranscriptionRequest, TranscriptionResult, TranscriptionTimings,
};
use gufo_serve::asr_service::{AsrService, AsrServiceOptions};
use gufo_serve::audio_asr_api::handle_audio_asr_api_request;
use gufo_serve::http::{HttpRequest, HttpResponse};

#[derive(Default)]
struct FakeRunner {
    calls: usize,
    wav_bytes: usize,
    context: String,
    language: Option<String>,
    maximum_tokens: usize,
}

type Fake = Arc<Mutex<FakeRunner>>;

fn run(
    fake: &Fake,
    request: &TranscriptionRequest,
    on_text: Option<TextSink<'_>>,
) -> Result<TranscriptionResult, String> {
    {
        let mut fake = fake.lock().expect("fake");
        fake.calls += 1;
        fake.wav_bytes = request.wav.len();
        fake.context = request.context.clone();
        fake.language = request.language.clone();
        fake.maximum_tokens = request.max_new_tokens;
    }
    let result = TranscriptionResult {
        language: "English".into(),
        text: "A native transcript.".into(),
        audio_samples: 32000,
        audio_tokens: 25,
        generated_ids: vec![1, 2, 3],
        timings: TranscriptionTimings {
            audio_encoder_ms: 10.0,
            text_decoder_ms: 20.0,
            ..TranscriptionTimings::default()
        },
        ..TranscriptionResult::default()
    };
    if let Some(sink) = on_text
        && !sink("A native")
    {
        return Err("cancelled".into());
    }
    Ok(result)
}

fn make_service(fake: &Fake) -> Arc<AsrService> {
    let fake = fake.clone();
    Arc::new(AsrService::new(AsrServiceOptions {
        validate_model: false,
        model_id: "qwen3-asr-1.7b".into(),
        runner: Some(Box::new(move |request, on_text, _cancelled| {
            run(&fake, request, on_text)
        })),
        ..AsrServiceOptions::default()
    }))
}

fn multipart(response_format: &str, include_unknown: bool, stream: &str) -> Vec<u8> {
    const BOUNDARY: &str = "gufo-test-boundary";
    let mut body = Vec::new();
    let mut append = |name: &str, value: &str, filename: &str| {
        body.extend_from_slice(
            format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"").as_bytes(),
        );
        if !filename.is_empty() {
            body.extend_from_slice(format!("; filename=\"{filename}\"").as_bytes());
        }
        body.extend_from_slice(b"\r\n\r\n");
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    };
    append("file", "RIFFfake", "sample.wav");
    append("model", "qwen3-asr", "");
    append("language", "en", "");
    append("prompt", "Names: Gufo.", "");
    append("response_format", response_format, "");
    append("max_tokens", "42", "");
    append("temperature", "0", "");
    if !stream.is_empty() {
        append("stream", stream, "");
    }
    if include_unknown {
        append("typo", "true", "");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    body
}

fn send_as(service: &Arc<AsrService>, body: Vec<u8>, content_type: &str) -> HttpResponse {
    handle_audio_asr_api_request(
        &HttpRequest {
            method: "POST".into(),
            path: "/v1/audio/transcriptions".into(),
            body,
            headers: vec![("Content-Type".into(), content_type.into())],
            ..HttpRequest::default()
        },
        service,
    )
    .expect("handler")
}

fn send(service: &Arc<AsrService>, body: Vec<u8>) -> HttpResponse {
    send_as(
        service,
        body,
        "multipart/form-data; boundary=gufo-test-boundary",
    )
}

fn has_header(response: &HttpResponse, name: &str, value: &str) -> bool {
    response
        .headers
        .iter()
        .any(|(n, v)| n == name && v == value)
}

fn body_text(response: &HttpResponse) -> String {
    String::from_utf8_lossy(&response.body).into_owned()
}

#[test]
fn json_request() {
    let fake = Fake::default();
    let service = make_service(&fake);
    let response = send(&service, multipart("json", false, ""));
    assert!(
        response.status == 200
            && body_text(&response).contains("\"text\":\"A native transcript.\""),
        "valid multipart request returns OpenAI JSON"
    );
    {
        let fake = fake.lock().expect("fake");
        assert!(
            fake.calls == 1
                && fake.wav_bytes == 8
                && fake.context == "Names: Gufo."
                && fake.language.as_deref() == Some("en")
                && fake.maximum_tokens == 42,
            "multipart fields reach the model contract"
        );
    }
    assert!(
        has_header(&response, "X-Gufo-ASR-Backend", "native-hip"),
        "native backend is exposed"
    );
    assert!(
        response.log_details.contains("encoder_ms=")
            && response.log_details.contains("decoder_ms=")
            && !response.log_details.contains("A native transcript.")
            && !response.log_details.contains("Names: Gufo."),
        "transcription diagnostics report work without transcript or context"
    );
}

#[test]
fn formats_and_validation() {
    let fake = Fake::default();
    let service = make_service(&fake);
    let text = send(&service, multipart("text", false, ""));
    assert!(
        text.status == 200
            && text.body == b"A native transcript."
            && has_header(&text, "Content-Type", "text/plain; charset=utf-8"),
        "text response format is supported"
    );
    let verbose = send(&service, multipart("verbose_json", false, ""));
    assert!(
        verbose.status == 200
            && body_text(&verbose).contains("\"duration\":2")
            && body_text(&verbose).contains("\"language\":\"english\""),
        "verbose_json exposes duration and language"
    );
    assert_eq!(
        send(&service, multipart("srt", false, "")).status,
        400,
        "unsupported response formats fail closed"
    );
    assert_eq!(
        send(&service, multipart("json", true, "")).status,
        400,
        "unknown multipart fields fail closed"
    );
    assert_eq!(
        send_as(&service, multipart("json", false, ""), "application/json").status,
        400,
        "non-multipart requests fail closed"
    );
    assert_eq!(
        fake.lock().expect("fake").calls,
        2,
        "invalid requests never invoke inference"
    );
}

#[test]
fn streaming() {
    let fake = Fake::default();
    let service = make_service(&fake);
    let response = send(&service, multipart("json", false, "true"));
    assert!(
        response.status == 200
            && response.streaming_body.is_some()
            && fake.lock().expect("fake").calls == 0,
        "streaming defers inference until the response is consumed"
    );
    let mut events = Vec::new();
    (response.streaming_body.expect("streaming body"))(&mut |part: &[u8]| {
        events.extend_from_slice(part);
        true
    })
    .expect("stream");
    let events = String::from_utf8_lossy(&events);
    assert!(
        fake.lock().expect("fake").calls == 1
            && events.contains("\"delta\":\"A native\"")
            && events.contains("\"delta\":\" transcript.\"")
            && events.contains("\"type\":\"transcript.text.done\""),
        "OpenAI SSE emits incremental deltas and a complete transcript"
    );
    let disconnected = send(&service, multipart("json", false, "true"));
    let log = disconnected.stream_log.clone().expect("stream log");
    let mut writes = 0;
    (disconnected.streaming_body.expect("streaming body"))(&mut |_: &[u8]| {
        writes += 1;
        false
    })
    .expect("stream");
    assert!(
        writes == 1 && log.lock().expect("log").error_code == "cancelled",
        "disconnected stream cancels inference without a done event"
    );
    assert!(
        send(&service, multipart("json", false, "false"))
            .streaming_body
            .is_none(),
        "stream=false retains the ordinary JSON contract"
    );
    assert!(
        send(&service, multipart("json", false, "yes")).status == 400
            && send(&service, multipart("text", false, "true")).status == 400,
        "invalid streaming configurations fail closed"
    );
}

#[test]
fn transcript_prefix_changes_fail_the_stream() {
    let service = Arc::new(AsrService::new(AsrServiceOptions {
        validate_model: false,
        runner: Some(Box::new(|_, on_text, _| {
            if let Some(sink) = on_text {
                sink("Hello");
                sink("Help");
            }
            Ok(TranscriptionResult::default())
        })),
        ..AsrServiceOptions::default()
    }));
    let response = send(&service, multipart("json", false, "true"));
    let outcome = (response.streaming_body.expect("streaming body"))(&mut |_: &[u8]| true);
    assert_eq!(
        outcome.expect_err("retracted prefix").message(),
        "ASR transcript changed its committed prefix"
    );
}

#[test]
fn the_native_runtime_is_created_after_validation_and_admits_per_chunk() {
    let created = Arc::new(AtomicBool::new(false));
    let factory = |error: Option<&'static str>| -> gufo_serve::asr_service::AsrRuntimeFactory {
        let created = created.clone();
        Box::new(move |root: &Path, context| {
            created.store(true, Ordering::SeqCst);
            assert_eq!((root, context), (Path::new("asr"), 512));
            match error {
                Some(error) => Err(error.into()),
                None => Ok(Box::new(|_, _, _| Ok(TranscriptionResult::default()))),
            }
        })
    };
    let options = |valid: bool, factory| AsrServiceOptions {
        model_root: PathBuf::from("asr"),
        native_context_tokens: 512,
        model_validator: Some(Box::new(move |_: &Path| valid)),
        runtime_factory: Some(factory),
        ..AsrServiceOptions::default()
    };
    let unsupported = AsrService::new(options(false, factory(None)));
    assert!(!unsupported.ready() && !created.load(Ordering::SeqCst));
    assert_eq!(
        unsupported.initialization_error(),
        "Qwen3-ASR service requires the supported 1.7B model"
    );
    let failed = AsrService::new(options(true, factory(Some("load failed"))));
    assert!(!failed.ready() && created.load(Ordering::SeqCst));
    assert_eq!(failed.initialization_error(), "load failed");
    assert_eq!(failed.model_id(), "qwen3-asr-1.7b");
    assert_eq!(
        failed
            .transcribe(&TranscriptionRequest::default(), None, &|| false)
            .expect_err("unready"),
        "load failed"
    );

    // The native runtime admits per chunk itself: two requests run in it
    // at once, where an injected runner is admitted one at a time.
    let concurrent = |native: bool| {
        let inside = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let runner: gufo_serve::asr_model::AsrRunner = {
            let (inside, peak) = (inside.clone(), peak.clone());
            Box::new(move |_, _, _| {
                peak.fetch_max(inside.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                let deadline = Instant::now() + Duration::from_millis(500);
                while peak.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
                    std::thread::yield_now();
                }
                let together = peak.load(Ordering::SeqCst) == 2;
                inside.fetch_sub(1, Ordering::SeqCst);
                Ok(TranscriptionResult {
                    text: together.to_string(),
                    ..TranscriptionResult::default()
                })
            })
        };
        let service = if native {
            let mut runner = Some(runner);
            AsrService::new(AsrServiceOptions {
                validate_model: false,
                runtime_factory: Some(Box::new(move |_: &Path, _| {
                    runner.take().ok_or_else(|| "created twice".to_string())
                })),
                ..AsrServiceOptions::default()
            })
        } else {
            AsrService::new(AsrServiceOptions {
                validate_model: false,
                runner: Some(runner),
                ..AsrServiceOptions::default()
            })
        };
        std::thread::scope(|scope| {
            let requests: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        service
                            .transcribe(&TranscriptionRequest::default(), None, &|| false)
                            .expect("transcribed")
                            .text
                    })
                })
                .collect();
            requests
                .into_iter()
                .map(|request| request.join().expect("request thread"))
                .collect::<Vec<_>>()
        })
    };
    assert_eq!(concurrent(true), ["true", "true"]);
    assert_eq!(concurrent(false), ["false", "false"]);
}
