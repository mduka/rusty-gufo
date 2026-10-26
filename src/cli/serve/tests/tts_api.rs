//! Port of `tests/models/qwen3_tts/audio_api_test.cpp` (the API and service
//! cases; the codec sampler cases belong to the TTS model port).

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gufo_serve::audio_stream::{decode_audio_base64, encode_audio_base64, encode_pcm16};
use gufo_serve::audio_tts_api::handle_audio_tts_api_request;
use gufo_serve::http::{HttpRequest, HttpResponse};
use gufo_serve::tts_model::{
    AudioSink, ModelVariant, SynthesisRequest, SynthesisResult, take_speech_segments,
};
use gufo_serve::tts_service::{TtsService, TtsServiceOptions};

use common::Semaphore;

#[derive(Default)]
struct FakeRunner {
    calls: usize,
    latest: SynthesisRequest,
}

type Fake = Arc<Mutex<FakeRunner>>;

fn run(
    fake: &Fake,
    request: &SynthesisRequest,
    on_audio: Option<AudioSink<'_>>,
) -> Result<SynthesisResult, String> {
    {
        let mut fake = fake.lock().expect("fake");
        fake.calls += 1;
        fake.latest = request.clone();
    }
    let result = SynthesisResult {
        sample_rate: 24000,
        code_groups: 16,
        codes: vec![0; 32],
        samples: vec![-1.0, -0.5, 0.0, 0.5, 1.0],
        sample_count: 0,
    };
    if let Some(sink) = on_audio {
        for sample in &result.samples {
            if !sink(std::slice::from_ref(sample)) {
                return Err("cancelled".into());
            }
        }
    }
    Ok(result)
}

fn service_with(fake: &Fake, options: TtsServiceOptions) -> Arc<TtsService> {
    let fake = fake.clone();
    Arc::new(TtsService::new(TtsServiceOptions {
        validate_model: false,
        runner: Some(Box::new(move |request, on_audio, _cancelled| {
            run(&fake, request, on_audio)
        })),
        ..options
    }))
}

fn make_service(fake: &Fake) -> Arc<TtsService> {
    service_with(
        fake,
        TtsServiceOptions {
            model_id: "qwen3-tts-12hz-1.7b-customvoice".into(),
            voices: vec!["vivian".into(), "ryan".into()],
            ..TtsServiceOptions::default()
        },
    )
}

fn make_variant_service(fake: &Fake, variant: ModelVariant) -> Arc<TtsService> {
    service_with(
        fake,
        TtsServiceOptions {
            variant,
            ..TtsServiceOptions::default()
        },
    )
}

fn send(service: &Arc<TtsService>, method: &str, path: &str, body: &str) -> HttpResponse {
    handle_audio_tts_api_request(
        &HttpRequest {
            method: method.into(),
            path: path.into(),
            body: body.as_bytes().to_vec(),
            ..HttpRequest::default()
        },
        service,
    )
    .expect("handler")
}

fn speech(service: &Arc<TtsService>, body: &str) -> HttpResponse {
    send(service, "POST", "/v1/audio/speech", body)
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

/// Runs a streamed body, collecting what it writes.
fn consume(response: HttpResponse) -> (Vec<u8>, String) {
    let mut output = Vec::new();
    let log = response.stream_log.clone();
    let body = response.streaming_body.expect("streaming body");
    body(&mut |bytes: &[u8]| {
        output.extend_from_slice(bytes);
        true
    })
    .expect("stream");
    let code = log
        .map(|log| log.lock().expect("log").error_code.clone())
        .unwrap_or_default();
    (output, code)
}

#[test]
fn speech_reaches_the_model_contract() {
    let fake = Fake::default();
    let service = make_service(&fake);
    assert!(service.ready(), "injected service is ready");
    let response = speech(
        &service,
        r#"{"model":"qwen3-tts","input":"The boy who lived.","voice":"vivian","language":"english","instructions":"calm","seed":7,"max_new_tokens":12,"greedy":true,"response_format":"wav","speed":1.0}"#,
    );
    assert_eq!(response.status, 200, "valid request succeeds");
    assert!(
        response.log_details.contains("codec_steps=2")
            && response.log_details.contains("synthesis_ms=")
            && !response.log_details.contains("The boy who lived."),
        "speech diagnostics report work without input text"
    );
    assert!(
        response.body.len() == 54
            && &response.body[..4] == b"RIFF"
            && &response.body[8..16] == b"WAVEfmt ",
        "response is a mono PCM16 WAV"
    );
    assert!(
        has_header(&response, "Content-Type", "audio/wav"),
        "WAV content type is explicit"
    );
    assert!(
        has_header(&response, "X-Gufo-Codec-Steps", "2"),
        "codec step count is exposed"
    );
    {
        let fake = fake.lock().expect("fake");
        let latest = &fake.latest;
        assert!(
            fake.calls == 1
                && latest.text == "The boy who lived."
                && latest.speaker == "vivian"
                && latest.language == "english"
                && latest.instruct == "calm"
                && latest.sampling.seed == 7
                && latest.max_new_tokens == 12
                && !latest.sampling.sample
                && !latest.sampling.predictor_sample,
            "API fields reach the model contract without drift"
        );
    }
    let sampled = speech(
        &service,
        r#"{"model":"qwen3-tts","input":"The boy who lived.","voice":"vivian"}"#,
    );
    let fake = fake.lock().expect("fake");
    assert!(
        sampled.status == 200
            && fake.calls == 2
            && fake.latest.sampling.sample
            && fake.latest.sampling.predictor_sample,
        "checkpoint-quality sampling is the API default"
    );
}

#[test]
fn sampling_controls_reach_both_distributions() {
    let fake = Fake::default();
    let service = make_service(&fake);
    let response = speech(
        &service,
        r#"{"model":"qwen3-tts","input":"x","voice":"vivian","temperature":0.5,"top_k":0,"top_p":0.8,"repetition_penalty":1.2,"subtalker_dosample":false,"subtalker_temperature":0.7,"subtalker_top_k":13,"subtalker_top_p":0.9}"#,
    );
    {
        let fake = fake.lock().expect("fake");
        let sampling = &fake.latest.sampling;
        assert!(
            response.status == 200
                && sampling.temperature == 0.5
                && sampling.top_k == 0
                && sampling.top_p == 0.8
                && sampling.repetition_penalty == 1.2
                && !sampling.predictor_sample
                && sampling.predictor_temperature == 0.7
                && sampling.predictor_top_k == 13
                && sampling.predictor_top_p == 0.9,
            "main and predictor sampling controls reach their own distributions"
        );
    }
    for field in [
        r#""top_p":0"#,
        r#""top_p":1.1"#,
        r#""temperature":0"#,
        r#""temperature":1e100"#,
        r#""top_k":-1"#,
        r#""top_k":1.5"#,
        r#""subtalker_top_p":2"#,
        r#""subtalker_dosample":1"#,
        r#""repetition_penalty":0"#,
    ] {
        let invalid = speech(
            &service,
            &format!(r#"{{"model":"qwen3-tts","input":"x","voice":"vivian",{field}}}"#),
        );
        assert!(
            invalid.status == 400 && body_text(&invalid).contains("invalid_sampling"),
            "invalid sampling rejected: {field}"
        );
    }
    assert_eq!(
        fake.lock().expect("fake").calls,
        1,
        "invalid sampling never invokes inference"
    );
}

#[test]
fn streaming_pcm_and_sse() {
    let fake = Fake::default();
    let service = make_service(&fake);
    let pcm_request =
        r#"{"model":"qwen3-tts","input":"x","voice":"vivian","response_format":"pcm"}"#;
    let response = speech(&service, pcm_request);
    assert!(
        response.status == 200
            && response.streaming_body.is_some()
            && fake.lock().expect("fake").calls == 0,
        "PCM synthesis starts when the response is consumed"
    );
    let (pcm, _) = consume(response);
    assert_eq!(
        pcm,
        encode_pcm16(&[-1.0, -0.5, 0.0, 0.5, 1.0]),
        "PCM chunks retain buffered sample encoding"
    );
    let (events, _) = consume(speech(
        &service,
        r#"{"model":"qwen3-tts","input":"x","voice":"vivian","stream_format":"sse"}"#,
    ));
    let events = String::from_utf8_lossy(&events);
    assert!(
        events.contains("speech.audio.delta")
            && events.contains("\"audio\":")
            && events.contains("speech.audio.done"),
        "SSE uses the OpenAI audio field and terminal event"
    );
    // C++ replays the same std::function; a streamed body runs once here, so
    // a second request carries the disconnect.
    let disconnected = speech(&service, pcm_request);
    let log = disconnected.stream_log.clone().expect("stream log");
    let mut writes = 0;
    (disconnected.streaming_body.expect("streaming body"))(&mut |_: &[u8]| {
        writes += 1;
        false
    })
    .expect("stream");
    assert!(
        writes == 1 && log.lock().expect("log").error_code == "cancelled",
        "output disconnect cancels native synthesis immediately"
    );
    for value in [r#""bogus""#, "true"] {
        assert_eq!(
            speech(
                &service,
                &format!(r#"{{"model":"qwen3-tts","input":"x","voice":"vivian","stream_format":{value}}}"#),
            )
            .status,
            400,
            "unsupported stream formats fail closed"
        );
    }
    assert_eq!(
        decode_audio_base64(encode_audio_base64(&pcm).as_bytes(), pcm.len()).as_deref(),
        Some(pcm.as_slice()),
        "audio base64 round trip"
    );
    assert!(
        decode_audio_base64(b"AB==", 16).is_none()
            && decode_audio_base64(b"YQ==AAAA", 16).is_none()
            && decode_audio_base64(b"AAAA", 2).is_none(),
        "base64 rejects noncanonical padding and oversized audio"
    );
    let text = "Value 3.14 is fine. Next phrase! Then 中文。";
    let mut pending = text.to_owned();
    let expected = take_speech_segments(&mut pending, "sentence", true);
    for split in (0..=text.len()).filter(|&s| text.is_char_boundary(s)) {
        pending = text[..split].to_owned();
        let mut actual = take_speech_segments(&mut pending, "sentence", false);
        pending.push_str(&text[split..]);
        actual.extend(take_speech_segments(&mut pending, "sentence", true));
        assert_eq!(
            actual, expected,
            "text splitting is independent of packet boundaries"
        );
    }
}

#[test]
fn queued_cancellation() {
    let entered = Arc::new(Semaphore::default());
    let release = Arc::new(Semaphore::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let service = {
        let (entered, release, calls) = (entered.clone(), release.clone(), calls.clone());
        TtsService::new(TtsServiceOptions {
            validate_model: false,
            runner: Some(Box::new(move |_, _, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                entered.release();
                release.acquire();
                Ok(SynthesisResult::default())
            })),
            ..TtsServiceOptions::default()
        })
    };
    let cancel = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let active = scope.spawn(|| {
            service
                .synthesize(&SynthesisRequest::default(), None, &|| false)
                .is_ok()
        });
        entered.acquire();
        let (sender, receiver) = mpsc::channel();
        let cancel = &cancel;
        let service = &service;
        scope.spawn(move || {
            let outcome = service.synthesize(&SynthesisRequest::default(), None, &|| {
                cancel.load(Ordering::SeqCst)
            });
            let _ = sender.send(outcome.is_err());
        });
        cancel.store(true, Ordering::SeqCst);
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(1)),
            Ok(true),
            "queued cancellation does not wait for active inference"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "cancelled waiter never enters the mutable runtime"
        );
        release.release();
        assert!(
            active.join().expect("active"),
            "queued cancellation does not invalidate the active request"
        );
    });
}

#[test]
fn validation_and_voices() {
    let fake = Fake::default();
    let service = make_service(&fake);
    let voices = send(&service, "GET", "/v1/audio/voices", "");
    assert!(
        voices.status == 200
            && body_text(&voices).contains("\"id\":\"vivian\"")
            && body_text(&voices).contains("\"id\":\"ryan\""),
        "voice discovery lists configured CustomVoice speakers"
    );
    assert_eq!(
        speech(
            &service,
            r#"{"model":"qwen3-tts","input":"x","voice":"unknown"}"#
        )
        .status,
        400,
        "unknown voice fails closed"
    );
    assert_eq!(
        speech(
            &service,
            r#"{"model":"qwen3-tts","input":"x","voice":"vivian","response_format":"mp3"}"#
        )
        .status,
        400,
        "unsupported format fails closed"
    );
    assert_eq!(
        speech(
            &service,
            r#"{"model":"qwen3-tts","input":"x","voice":"vivian","seed":1.5}"#
        )
        .status,
        400,
        "fractional seed fails closed"
    );
    assert_eq!(
        speech(
            &service,
            r#"{"model":"qwen3-tts","input":"x","voice":"vivian","typo":true}"#
        )
        .status,
        400,
        "unknown fields fail closed"
    );
    assert_eq!(
        fake.lock().expect("fake").calls,
        0,
        "invalid requests never invoke inference"
    );
}

#[test]
fn voice_design_contract() {
    let fake = Fake::default();
    let service = make_variant_service(&fake, ModelVariant::VoiceDesign);
    assert!(
        service.ready() && service.model_id() == "qwen3-tts-12hz-1.7b-voice-design",
        "VoiceDesign service derives its model identity"
    );
    let response = speech(
        &service,
        r#"{"model":"qwen3-tts","input":"A lantern in the rain.","language":"english","instructions":"A calm, warm adult voice."}"#,
    );
    {
        let fake = fake.lock().expect("fake");
        assert!(
            response.status == 200
                && fake.calls == 1
                && fake.latest.speaker == "voice-design"
                && fake.latest.instruct == "A calm, warm adult voice.",
            "VoiceDesign uses instructions without CustomVoice speaker semantics"
        );
    }
    assert_eq!(
        speech(&service, r#"{"model":"qwen3-tts","input":"x"}"#).status,
        400,
        "VoiceDesign rejects a missing instruction"
    );
    assert_eq!(
        speech(
            &service,
            r#"{"model":"qwen3-tts","input":"x","voice":"vivian","instructions":"calm"}"#
        )
        .status,
        400,
        "VoiceDesign rejects CustomVoice speaker names"
    );
}

#[test]
fn base_contract() {
    const REFERENCE_WAV: &str =
        "UklGRiwAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQgAAAAAAOgDGPwAAA==";
    let fake = Fake::default();
    let service = make_variant_service(&fake, ModelVariant::Base);
    assert!(
        service.ready() && service.model_id() == "qwen3-tts-12hz-1.7b-base",
        "Base service derives its model identity"
    );
    let response = speech(
        &service,
        &format!(
            r#"{{"model":"qwen3-tts","input":"Clone me.","reference_audio":"{REFERENCE_WAV}","reference_text":"Reference phrase.","language":"auto"}}"#
        ),
    );
    {
        let fake = fake.lock().expect("fake");
        let latest = &fake.latest;
        assert!(
            response.status == 200
                && fake.calls == 1
                && latest.speaker == "voice-clone"
                && latest.reference_audio.sample_rate == 24000
                && latest.reference_audio.channels == 1
                && latest.reference_audio.samples.len() == 4
                && latest.reference_text == "Reference phrase."
                && !latest.speaker_embedding_only,
            "Base ICL reference audio reaches the synthesis contract"
        );
    }
    assert_eq!(
        speech(&service, r#"{"model":"qwen3-tts","input":"x"}"#).status,
        400,
        "Base rejects a missing reference WAV"
    );
    assert_eq!(
        speech(
            &service,
            &format!(r#"{{"model":"qwen3-tts","input":"x","reference_audio":"{REFERENCE_WAV}"}}"#)
        )
        .status,
        400,
        "Base ICL rejects missing reference text"
    );
    let embedding_only = speech(
        &service,
        &format!(
            r#"{{"model":"qwen3-tts","input":"x","reference_audio":"{REFERENCE_WAV}","voice_clone_mode":"speaker_embedding_only"}}"#
        ),
    );
    assert!(
        embedding_only.status == 200 && fake.lock().expect("fake").latest.speaker_embedding_only,
        "Base speaker-embedding-only mode does not require reference text"
    );
}

#[test]
fn invalid_waveform() {
    for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let service = Arc::new(TtsService::new(TtsServiceOptions {
            validate_model: false,
            voices: vec!["vivian".into()],
            runner: Some(Box::new(move |_, on_audio, _| {
                let result = SynthesisResult {
                    sample_rate: 24000,
                    samples: vec![0.0, sample],
                    ..SynthesisResult::default()
                };
                if let Some(sink) = on_audio
                    && !sink(&result.samples)
                {
                    return Err(String::new());
                }
                Ok(result)
            })),
            ..TtsServiceOptions::default()
        }));
        let wav = speech(
            &service,
            r#"{"model":"qwen3-tts","input":"x","voice":"vivian"}"#,
        );
        assert!(
            wav.status == 500 && body_text(&wav).contains("invalid_audio"),
            "nonfinite WAV samples fail before conversion"
        );
        for format in ["audio", "sse"] {
            let (body, code) = consume(speech(
                &service,
                &format!(
                    r#"{{"model":"qwen3-tts","input":"x","voice":"vivian","response_format":"pcm","stream_format":"{format}"}}"#
                ),
            ));
            assert_eq!(
                code, "invalid_audio",
                "nonfinite streaming samples report invalid audio"
            );
            let body = String::from_utf8_lossy(&body);
            assert!(
                !body.contains("speech.audio.delta")
                    && !body.contains("speech.audio.done")
                    && if format == "sse" {
                        body.contains("event: error")
                    } else {
                        body.is_empty()
                    },
                "invalid audio never becomes PCM or a successful terminal event"
            );
        }
    }
}

/// The service creates the runtime only after its own validation and reports
/// the creation error as its initialization error.
#[test]
fn the_runtime_is_created_after_validation() {
    let missing = std::path::Path::new("/nonexistent/qwen3-tts");
    let service = TtsService::new(TtsServiceOptions {
        model_root: missing.to_path_buf(),
        validate_model: false,
        runtime_factory: Some(Box::new(|_, _, _| Err("creation failed".into()))),
        ..TtsServiceOptions::default()
    });
    assert!(!service.ready());
    assert_eq!(service.initialization_error(), "creation failed");
    let service = TtsService::new(TtsServiceOptions {
        model_root: missing.to_path_buf(),
        runtime_factory: Some(Box::new(|_, _, _| panic!("created before validation"))),
        ..TtsServiceOptions::default()
    });
    assert_eq!(
        service.initialization_error(),
        "Qwen3-TTS service requires a supported 1.7B model"
    );
}
