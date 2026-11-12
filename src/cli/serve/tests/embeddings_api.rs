//! `POST /v1/embeddings` against a fake encoder: no model files, no device
//! and no socket; the handler is called directly
//! (`tests/models/embedding_gemma2/embeddings_api_test.cpp`).

#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gufo_core::json::{self, Value};
use gufo_core::text::CancellationCheck;
use gufo_serve::embedding_service::{
    EmbeddingBatch, EmbeddingEngine, EmbeddingInput, EmbeddingPartType, EmbeddingRequestOptions,
    EmbeddingRunner, EmbeddingService, EmbeddingServiceOptions, EmbeddingStatus,
    EmbeddingVideoOptions, EngineMedia, EngineSequence, EngineVideo, EngineVideoError,
    MAX_INPUT_TOKENS, load_sentence_transformer_prompts, truncate_keeping_images,
};
use gufo_serve::embeddings_api::{handle_embeddings_api_request, is_embeddings_api_path};
use gufo_serve::http::{HttpRequest, HttpResponse};

const WIDTH: usize = 768;

/// FNV-1a, a deterministic stand-in for `std::hash<std::string>`.
fn hash(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A deterministic unit vector per text. Components differ in magnitude so a
/// truncated prefix is not already normalized.
fn fake_embedding(text: &str) -> Vec<f32> {
    let seed = hash(text);
    let mut vector: Vec<f32> = (0..WIDTH)
        .map(|i| (((seed % 1009) as f64 + 0.37 * i as f64).sin() * (1.0 + (i % 7) as f64)) as f32)
        .collect();
    let sum: f64 = vector.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let scale = 1.0 / sum.sqrt();
    for value in &mut vector {
        *value = (f64::from(*value) * scale) as f32;
    }
    vector
}

/// One token per byte plus `<bos>` and `<eos>`.
fn fake_tokens(text: &str) -> usize {
    text.len() + 2
}

#[derive(Default)]
struct FakeRunner {
    calls: AtomicI32,
    last_texts: Mutex<Vec<String>>,
    last_truncate: AtomicBool,
    last_image_soft_tokens: AtomicU32,
}

/// The text of an input: its parts joined, as the service tokenizes them.
fn joined(input: &EmbeddingInput) -> String {
    input
        .content
        .iter()
        .map(|part| part.value.as_str())
        .collect()
}

impl FakeRunner {
    fn run(
        &self,
        inputs: &[EmbeddingInput],
        request: &EmbeddingRequestOptions,
        batch: &mut EmbeddingBatch,
        error: &mut String,
    ) -> EmbeddingStatus {
        let texts: Vec<String> = inputs.iter().map(joined).collect();
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last_texts.lock().unwrap() = texts.clone();
        self.last_truncate.store(request.truncate, Ordering::SeqCst);
        self.last_image_soft_tokens
            .store(request.image_soft_tokens, Ordering::SeqCst);
        for (i, text) in texts.iter().enumerate() {
            if text.starts_with("TOOLONG") {
                if request.truncate {
                    batch.embeddings.push(fake_embedding(text));
                    batch.token_counts.push(MAX_INPUT_TOKENS);
                    continue;
                }
                batch.failed_index = i;
                return EmbeddingStatus::InputTooLong;
            }
            if text == "BROKEN" {
                *error = "device exploded".into();
                return EmbeddingStatus::Failed;
            }
            if text == "LOST" {
                *error = "device reset".into();
                return EmbeddingStatus::DeviceLost;
            }
            batch.embeddings.push(fake_embedding(text));
            batch.token_counts.push(fake_tokens(text));
        }
        EmbeddingStatus::Ok
    }
}

fn service_with(runner: EmbeddingRunner) -> Arc<EmbeddingService> {
    Arc::new(EmbeddingService::new(EmbeddingServiceOptions {
        validate_model: false,
        model_id: "embeddinggemma-2".into(),
        runner: Some(runner),
        ..EmbeddingServiceOptions::default()
    }))
}

fn make_service(runner: &Arc<FakeRunner>) -> Arc<EmbeddingService> {
    let runner = runner.clone();
    service_with(Box::new(move |inputs, request, batch, error| {
        runner.run(inputs, request, batch, error)
    }))
}

fn send_with(
    service: &Arc<EmbeddingService>,
    body: &str,
    method: &str,
    cancelled: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
) -> HttpResponse {
    handle_embeddings_api_request(
        &HttpRequest {
            method: method.into(),
            path: "/v1/embeddings".into(),
            body: body.as_bytes().to_vec(),
            headers: vec![("Content-Type".into(), "application/json".into())],
            is_cancelled: cancelled,
            ..HttpRequest::default()
        },
        service,
    )
    .unwrap()
}

fn send(service: &Arc<EmbeddingService>, body: &str) -> HttpResponse {
    send_with(service, body, "POST", None)
}

fn send_to(service: &Arc<EmbeddingService>, path: &str, body: &str) -> HttpResponse {
    handle_embeddings_api_request(
        &HttpRequest {
            method: "POST".into(),
            path: path.into(),
            body: body.as_bytes().to_vec(),
            headers: vec![("Content-Type".into(), "application/json".into())],
            ..HttpRequest::default()
        },
        service,
    )
    .unwrap()
}

/// A service whose runner records the inputs it receives.
fn capturing_service() -> (
    Arc<EmbeddingService>,
    Arc<Mutex<Vec<EmbeddingInput>>>,
    Arc<AtomicI32>,
) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicI32::new(0));
    let (c, n) = (captured.clone(), calls.clone());
    let service = service_with(Box::new(move |inputs, _, batch, _| {
        n.fetch_add(1, Ordering::SeqCst);
        *c.lock().unwrap() = inputs.to_vec();
        for input in inputs {
            let text = joined(input);
            batch.embeddings.push(fake_embedding(&text));
            batch.token_counts.push(fake_tokens(&text));
        }
        EmbeddingStatus::Ok
    }));
    (service, captured, calls)
}

fn parse(response: &HttpResponse) -> Value {
    json::parse(&response.body).unwrap_or_else(|e| {
        panic!(
            "response is not JSON: {}: {}",
            e.message(),
            String::from_utf8_lossy(&response.body)
        )
    })
}

fn error_code(response: &HttpResponse) -> String {
    let body = parse(response);
    let error = body
        .find("error")
        .filter(|e| e.is_object())
        .expect("error object present");
    assert!(
        error.contains("message") && error.contains("type"),
        "error carries message and type"
    );
    error.member_str("code", "")
}

fn expect_error(response: &HttpResponse, status: u16, code: &str, what: &str) {
    assert_eq!(
        response.status,
        status,
        "{what}: body {}",
        String::from_utf8_lossy(&response.body)
    );
    assert_eq!(error_code(response), code, "{what}");
}

fn floats_of(embedding: &Value) -> Vec<f32> {
    assert!(embedding.is_array(), "embedding is an array");
    embedding
        .items()
        .iter()
        .map(|item| {
            assert!(item.is_number(), "embedding component is a number");
            item.as_double(0.0) as f32
        })
        .collect()
}

fn norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt()
}

fn decode_base64(text: &str) -> Vec<u8> {
    let value = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    assert_eq!(text.len() % 4, 0, "base64 length is a multiple of four");
    let mut bytes = Vec::new();
    for quad in text.as_bytes().chunks(4) {
        let mut word = 0u32;
        let mut padding = 0;
        for &c in quad {
            word <<= 6;
            if c == b'=' {
                padding += 1;
            } else {
                assert!(padding == 0, "valid base64 character");
                word |= value(c).expect("valid base64 character");
            }
        }
        bytes.push((word >> 16) as u8);
        if padding < 2 {
            bytes.push((word >> 8) as u8);
        }
        if padding < 1 {
            bytes.push(word as u8);
        }
    }
    bytes
}

fn embedding_of(body: &Value, index: usize) -> &Value {
    body.find("data").unwrap().items()[index]
        .find("embedding")
        .unwrap()
}

#[test]
fn string_input() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    let response = send(
        &service,
        r#"{"model":"embeddinggemma-2","input":"hello world"}"#,
    );
    assert_eq!(response.status, 200, "string input succeeds");
    assert!(
        response
            .headers
            .iter()
            .any(|(n, v)| n == "Content-Type" && v == "application/json"),
        "response is JSON"
    );
    let body = parse(&response);
    assert_eq!(body.member_str("object", ""), "list", "list object");
    assert_eq!(
        body.member_str("model", ""),
        "embeddinggemma-2",
        "model echoes id"
    );
    let data = body.find("data").unwrap().items();
    assert_eq!(data.len(), 1, "one embedding");
    assert_eq!(
        data[0].member_str("object", ""),
        "embedding",
        "embedding object"
    );
    assert_eq!(data[0].member_size("index", 99), 0, "index zero");
    let values = floats_of(data[0].find("embedding").unwrap());
    assert_eq!(
        values,
        fake_embedding("hello world"),
        "floats round-trip exactly through the JSON text"
    );
    assert!(
        (norm(&values) - 1.0).abs() < 1e-5,
        "full vector is unit length"
    );
    let usage = body.find("usage").unwrap();
    assert!(
        usage.member_size("prompt_tokens", 0) == fake_tokens("hello world")
            && usage.member_size("total_tokens", 0) == fake_tokens("hello world"),
        "usage counts the markers"
    );
    assert_eq!(
        *runner.last_texts.lock().unwrap(),
        vec!["hello world".to_owned()],
        "service receives the text unchanged"
    );
}

#[test]
fn array_input_and_order() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    let inputs: Vec<String> = ["alpha", "b", "gamma gamma", "δελτα \"quoted\"\n"]
        .map(String::from)
        .to_vec();
    // Omitting `model` is accepted.
    let request = Value::object()
        .with(
            "input",
            Value::from(inputs.iter().map(Value::from).collect::<Vec<_>>()),
        )
        .with("user", "someone");
    let response = send(&service, &request.dump().unwrap());
    assert_eq!(response.status, 200, "array input succeeds");
    assert_eq!(
        *runner.last_texts.lock().unwrap(),
        inputs,
        "texts reach the service in order"
    );
    let body = parse(&response);
    let data = body.find("data").unwrap().items();
    assert_eq!(data.len(), inputs.len(), "one embedding per input");
    let mut tokens = 0;
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(
            data[i].member_size("index", 99),
            i,
            "index follows input order"
        );
        assert_eq!(
            floats_of(data[i].find("embedding").unwrap()),
            fake_embedding(input),
            "embedding {i} belongs to its input"
        );
        tokens += fake_tokens(input);
    }
    let usage = body.find("usage").unwrap();
    assert!(
        usage.member_size("prompt_tokens", 0) == tokens
            && usage.member_size("total_tokens", 0) == tokens,
        "usage sums every input"
    );
}

#[test]
fn dimensions() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    let full = fake_embedding("matryoshka");
    for dimensions in [128usize, 256, 512, 768] {
        let response = send(
            &service,
            &format!(r#"{{"input":"matryoshka","dimensions":{dimensions}}}"#),
        );
        assert_eq!(response.status, 200, "dimensions accepted");
        let values = floats_of(embedding_of(&parse(&response), 0));
        assert_eq!(values.len(), dimensions, "dimension count honored");
        assert!(
            (norm(&values) - 1.0).abs() < 1e-5,
            "truncated vector is renormalized"
        );
        if dimensions == WIDTH {
            assert_eq!(values, full, "768 leaves the vector untouched");
            continue;
        }
        // The truncation is a scaled prefix of the full vector.
        let scale = f64::from(values[0]) / f64::from(full[0]);
        assert!(scale > 1.0, "a prefix of a unit vector scales up");
        for i in 0..dimensions {
            assert!(
                (f64::from(values[i]) - f64::from(full[i]) * scale).abs() < 1e-5,
                "truncation keeps the leading components"
            );
        }
    }
    for bad in [
        "100", "0", "1024", "-128", "128.5", "\"256\"", "true", "[128]",
    ] {
        expect_error(
            &send(&service, &format!(r#"{{"input":"x","dimensions":{bad}}}"#)),
            400,
            "invalid_dimensions",
            &format!("dimensions {bad}"),
        );
    }
    // null means unspecified, as several client libraries send it.
    assert_eq!(
        send(&service, r#"{"input":"x","dimensions":null}"#).status,
        200,
        "null dimensions is the default"
    );
}

#[test]
fn base64_encoding() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    let response = send(
        &service,
        r#"{"input":["one","two"],"encoding_format":"base64","dimensions":256}"#,
    );
    assert_eq!(response.status, 200, "base64 succeeds");
    let body = parse(&response);
    let data = body.find("data").unwrap().items();
    assert_eq!(data.len(), 2, "two embeddings");
    for (i, input) in ["one", "two"].iter().enumerate() {
        let embedding = data[i].find("embedding").unwrap();
        assert!(embedding.is_string(), "base64 embedding is a string");
        let bytes = decode_base64(embedding.str());
        assert_eq!(bytes.len(), 256 * 4, "256 little-endian floats");
        let decoded: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        let full = fake_embedding(input);
        assert!(
            (norm(&decoded) - 1.0).abs() < 1e-5,
            "decoded vector is unit"
        );
        let scale = f64::from(decoded[0]) / f64::from(full[0]);
        for j in 0..decoded.len() {
            assert!(
                (f64::from(decoded[j]) - f64::from(full[j]) * scale).abs() < 1e-5,
                "base64 carries the truncated components"
            );
        }
    }
    // The same request in float form must agree bit for bit.
    let floats = send(
        &service,
        r#"{"input":["one","two"],"encoding_format":"float","dimensions":256}"#,
    );
    let from_float = floats_of(embedding_of(&parse(&floats), 0));
    let bytes = decode_base64(data[0].find("embedding").unwrap().str());
    let from_base64: Vec<u8> = from_float.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        from_base64, bytes,
        "float and base64 encodings are the same numbers"
    );
    expect_error(
        &send(&service, r#"{"input":"x","encoding_format":"hex"}"#),
        400,
        "invalid_encoding_format",
        "unknown encoding_format",
    );
    expect_error(
        &send(&service, r#"{"input":"x","encoding_format":1}"#),
        400,
        "invalid_encoding_format",
        "non-string encoding_format",
    );
}

#[test]
fn input_validation() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    for (body, status, code, what) in [
        (r#"{"input":""}"#, 400, "empty_input", "empty string"),
        (r#"{"input":[]}"#, 400, "empty_input", "empty array"),
        (
            r#"{"input":["fine",""]}"#,
            400,
            "empty_input",
            "empty array item",
        ),
        (
            r#"{"model":"embeddinggemma-2"}"#,
            400,
            "missing_input",
            "missing input",
        ),
        (r#"{"input":null}"#, 400, "missing_input", "null input"),
        (
            r#"{"input":[[1,2],[3]]}"#,
            400,
            "unsupported_input",
            "nested token arrays",
        ),
        (
            r#"{"input":["text",7]}"#,
            400,
            "unsupported_input",
            "mixed array",
        ),
        (
            r#"{"input":{"a":1}}"#,
            400,
            "invalid_parameter_type",
            "object input",
        ),
        (
            r#"{"input":[true]}"#,
            400,
            "invalid_parameter_type",
            "boolean item",
        ),
        (
            r#"{"input":"x","user":3}"#,
            400,
            "invalid_parameter_type",
            "non-string user",
        ),
        (
            r#"{"input":"x","normalize":true}"#,
            400,
            "unsupported_field",
            "unknown member",
        ),
    ] {
        expect_error(&send(&service, body), status, code, what);
    }
    let tokens = send(&service, r#"{"input":[1,2,3]}"#);
    expect_error(&tokens, 400, "unsupported_input", "token array");
    assert_eq!(
        parse(&tokens)
            .find("error")
            .unwrap()
            .member_str("message", ""),
        "token-array input is not supported; send strings",
        "token-array message is explicit"
    );

    let items = |count: usize| format!(r#"{{"input":[{}]}}"#, vec![r#""a""#; count].join(","));
    expect_error(
        &send(&service, &items(2049)),
        400,
        "too_many_inputs",
        "2049 inputs",
    );
    let accepted = send(&service, &items(2048));
    assert!(
        accepted.status == 200 && parse(&accepted).find("data").unwrap().items().len() == 2048,
        "2048 inputs are accepted"
    );
    assert_eq!(
        runner.calls.load(Ordering::SeqCst),
        1,
        "rejected requests never reach the encoder"
    );
}

#[test]
fn model_and_json() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    expect_error(
        &send(&service, r#"{"model":"other","input":"x"}"#),
        404,
        "model_not_found",
        "wrong model",
    );
    for (body, code, what) in [
        (
            r#"{"model":7,"input":"x"}"#,
            "invalid_parameter_type",
            "non-string model",
        ),
        ("", "parse_error", "empty body"),
        ("{\"input\":", "parse_error", "truncated JSON"),
        ("not json", "parse_error", "not JSON"),
        (r#"["x"]"#, "invalid_request_body", "array body"),
        (r#""x""#, "invalid_request_body", "string body"),
    ] {
        expect_error(&send(&service, body), 400, code, what);
    }
    expect_error(
        &send_with(&service, r#"{"input":"x"}"#, "GET", None),
        405,
        "method_not_allowed",
        "GET",
    );
    assert_eq!(
        runner.calls.load(Ordering::SeqCst),
        0,
        "invalid requests never reach the encoder"
    );
    assert!(
        is_embeddings_api_path("/v1/embeddings")
            && !is_embeddings_api_path("/v1/embeddings/")
            && !is_embeddings_api_path("/v1/chat/completions"),
        "path predicate is exact"
    );
}

#[test]
fn limits_and_failures() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    let long_input = send(&service, r#"{"input":["short","TOOLONG input"]}"#);
    expect_error(
        &long_input,
        400,
        "context_length_exceeded",
        "over-length input",
    );
    let message = parse(&long_input)
        .find("error")
        .unwrap()
        .member_str("message", "");
    assert!(
        message.contains("index 1") && message.contains("8192"),
        "message names the input index and the limit: {message}"
    );

    let broken = send(&service, r#"{"input":"BROKEN"}"#);
    expect_error(&broken, 500, "embedding_failed", "runtime failure");
    let error = parse(&broken).find("error").unwrap().clone();
    assert_eq!(
        error.member_str("message", ""),
        "device exploded",
        "runtime error text is returned"
    );
    assert_eq!(
        error.member_str("type", ""),
        "server_error",
        "runtime failures are server errors"
    );
    assert!(
        !service.device_lost(),
        "an ordinary failure keeps the device"
    );

    let lost = send(&service, r#"{"input":"LOST"}"#);
    expect_error(&lost, 503, "device_lost", "lost device");
    assert_eq!(
        parse(&lost).find("error").unwrap().member_str("type", ""),
        "server_error",
        "a lost device is a server error"
    );
    assert!(
        service.device_lost(),
        "the service remembers the lost device"
    );

    // A runner that returns fewer vectors than inputs is a server bug.
    let short_service = service_with(Box::new(|_, _, batch, _| {
        batch.embeddings = vec![fake_embedding("only")];
        batch.token_counts = vec![3];
        EmbeddingStatus::Ok
    }));
    expect_error(
        &send(&short_service, r#"{"input":["a","b"]}"#),
        500,
        "embedding_failed",
        "inconsistent batch",
    );
}

#[test]
fn cancellation() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    expect_error(
        &send_with(
            &service,
            r#"{"input":"x"}"#,
            "POST",
            Some(Arc::new(|| true)),
        ),
        499,
        "cancelled",
        "cancelled before encoding",
    );
    assert_eq!(
        runner.calls.load(Ordering::SeqCst),
        0,
        "a cancelled request never reaches the encoder"
    );

    // Cancelled while the encoder runs. The client is gone, so no body is
    // sent.
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = cancelled.clone();
    let cancelling = service_with(Box::new(move |inputs, _, batch, _| {
        flag.store(true, Ordering::SeqCst);
        batch.embeddings = vec![fake_embedding(&joined(&inputs[0]))];
        batch.token_counts = vec![3];
        EmbeddingStatus::Ok
    }));
    expect_error(
        &send_with(
            &cancelling,
            r#"{"input":"x"}"#,
            "POST",
            Some(Arc::new(move || cancelled.load(Ordering::SeqCst))),
        ),
        499,
        "cancelled",
        "cancelled during encoding",
    );
}

#[test]
fn service_lifecycle() {
    // An injected runner needs no model files.
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    assert!(
        service.ready() && service.initialization_error().is_empty(),
        "fake service is ready"
    );
    assert_eq!(service.model_id(), "embeddinggemma-2", "model id kept");
    let defaulted = EmbeddingService::new(EmbeddingServiceOptions {
        validate_model: false,
        runner: Some(Box::new(|_, _, _, _| EmbeddingStatus::Ok)),
        ..EmbeddingServiceOptions::default()
    });
    assert_eq!(defaulted.model_id(), "embeddinggemma-2", "default model id");

    // Validation rejects a directory that is not an EmbeddingGemma 2
    // checkpoint.
    let invalid = EmbeddingService::new(EmbeddingServiceOptions {
        model_root: "/nonexistent/embeddinggemma".into(),
        validate_model: true,
        model_validator: Some(Box::new(|root| root.join("config.json").is_file())),
        runner: Some(Box::new(|_, _, _, _| EmbeddingStatus::Ok)),
        ..EmbeddingServiceOptions::default()
    });
    assert!(
        !invalid.ready() && !invalid.initialization_error().is_empty(),
        "an invalid model directory is refused"
    );
    let mut batch = EmbeddingBatch::default();
    let mut error = String::new();
    assert!(
        invalid.embed(&["x".into()], None, &mut batch, &mut error) == EmbeddingStatus::Failed
            && !error.is_empty(),
        "a service that is not ready refuses work"
    );

    // Without a runner or a native engine nothing can run.
    let engineless = EmbeddingService::new(EmbeddingServiceOptions {
        model_root: "/nonexistent".into(),
        validate_model: false,
        ..EmbeddingServiceOptions::default()
    });
    assert!(
        !engineless.ready()
            && engineless.initialization_error() == "embedding service requires the HIP backend",
        "builds without an engine refuse to start"
    );

    // GPU access is serialized.
    let active = Arc::new(AtomicI32::new(0));
    let overlapped = Arc::new(AtomicI32::new(0));
    let (a, o) = (active.clone(), overlapped.clone());
    let serial = Arc::new(EmbeddingService::new(EmbeddingServiceOptions {
        validate_model: false,
        runner: Some(Box::new(move |inputs, _, out, _| {
            if a.fetch_add(1, Ordering::SeqCst) != 0 {
                o.fetch_add(1, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_millis(5));
            out.embeddings = vec![fake_embedding(&joined(&inputs[0]))];
            out.token_counts = vec![3];
            a.fetch_sub(1, Ordering::SeqCst);
            EmbeddingStatus::Ok
        })),
        ..EmbeddingServiceOptions::default()
    }));
    let threads: Vec<_> = (0..6)
        .map(|_| {
            let serial = serial.clone();
            std::thread::spawn(move || {
                let mut result = EmbeddingBatch::default();
                let mut error = String::new();
                assert_eq!(
                    serial.embed(&["x".into()], None, &mut result, &mut error),
                    EmbeddingStatus::Ok,
                    "concurrent embed succeeds"
                );
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(
        overlapped.load(Ordering::SeqCst),
        0,
        "the encoder never runs concurrently"
    );
}

/// The native engine: tokenizing before the device, the byte and token
/// limits, and the status a failed encode maps to.
struct FakeEngine {
    lost: AtomicBool,
    fail: bool,
    sequences: Mutex<Vec<Vec<u32>>>,
    /// What `process_video` reports; `None` without video support.
    video: Option<EngineVideoError>,
}

impl EmbeddingEngine for FakeEngine {
    fn max_token_bytes(&self) -> usize {
        4
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        // One token per word between the markers.
        let mut ids = vec![2];
        ids.extend(text.split(' ').map(|word| word.len() as u32));
        ids.push(1);
        ids
    }

    fn encode_truncated(&self, text: &str, max_tokens: usize) -> Vec<u32> {
        let mut ids = self.encode(text);
        if ids.len() > max_tokens && max_tokens >= 2 {
            let end = ids.len() - 1;
            ids.drain(max_tokens - 1..end);
        }
        ids
    }

    fn supports_vision(&self) -> bool {
        false
    }

    fn supports_video(&self) -> bool {
        self.video.is_some()
    }

    fn supports_audio(&self) -> bool {
        false
    }

    fn image_processor_error(&self) -> String {
        String::new()
    }

    fn audio_processor_error(&self) -> String {
        String::new()
    }

    fn video_processor_error(&self) -> String {
        String::new()
    }

    fn process_image(&self, _: &[u8], _: u32) -> Result<EngineMedia, String> {
        Err("no images".into())
    }

    fn process_video(
        &self,
        _: &[u8],
        _: &EmbeddingVideoOptions,
        _: Option<&CancellationCheck>,
        _: usize,
    ) -> Result<EngineVideo, EngineVideoError> {
        Err(self
            .video
            .clone()
            .unwrap_or_else(|| EngineVideoError::Invalid("no video".into())))
    }

    fn process_audio(
        &self,
        _: &[u8],
        _: Option<&CancellationCheck>,
    ) -> Result<EngineMedia, String> {
        Err("no audio".into())
    }

    fn embed(
        &self,
        sequences: Vec<EngineSequence>,
        is_cancelled: Option<CancellationCheck>,
    ) -> Result<Vec<Vec<f32>>, String> {
        *self.sequences.lock().unwrap() = sequences.iter().map(|s| s.tokens.clone()).collect();
        if is_cancelled.is_some_and(|cancelled| cancelled()) {
            return Err("embedding_gemma2: request cancelled".into());
        }
        if self.fail {
            self.lost.store(true, Ordering::SeqCst);
            return Err("embedding_gemma2: GPU device lost: reset".into());
        }
        Ok(sequences.iter().map(|_| fake_embedding("native")).collect())
    }

    fn device_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }
}

fn native_service(fail: bool) -> (Arc<EmbeddingService>, Arc<FakeEngine>) {
    native_service_with(fail, None)
}

fn native_service_with(
    fail: bool,
    video: Option<EngineVideoError>,
) -> (Arc<EmbeddingService>, Arc<FakeEngine>) {
    let engine = Arc::new(FakeEngine {
        lost: AtomicBool::new(false),
        fail,
        sequences: Mutex::new(Vec::new()),
        video,
    });
    struct Shared(Arc<FakeEngine>);
    impl EmbeddingEngine for Shared {
        fn max_token_bytes(&self) -> usize {
            self.0.max_token_bytes()
        }
        fn encode(&self, text: &str) -> Vec<u32> {
            self.0.encode(text)
        }
        fn encode_truncated(&self, text: &str, max_tokens: usize) -> Vec<u32> {
            self.0.encode_truncated(text, max_tokens)
        }
        fn supports_vision(&self) -> bool {
            self.0.supports_vision()
        }
        fn supports_video(&self) -> bool {
            self.0.supports_video()
        }
        fn supports_audio(&self) -> bool {
            self.0.supports_audio()
        }
        fn image_processor_error(&self) -> String {
            self.0.image_processor_error()
        }
        fn audio_processor_error(&self) -> String {
            self.0.audio_processor_error()
        }
        fn video_processor_error(&self) -> String {
            self.0.video_processor_error()
        }
        fn process_image(&self, bytes: &[u8], max_soft_tokens: u32) -> Result<EngineMedia, String> {
            self.0.process_image(bytes, max_soft_tokens)
        }
        fn process_video(
            &self,
            bytes: &[u8],
            options: &EmbeddingVideoOptions,
            is_cancelled: Option<&CancellationCheck>,
            frame_budget: usize,
        ) -> Result<EngineVideo, EngineVideoError> {
            self.0
                .process_video(bytes, options, is_cancelled, frame_budget)
        }
        fn process_audio(
            &self,
            bytes: &[u8],
            is_cancelled: Option<&CancellationCheck>,
        ) -> Result<EngineMedia, String> {
            self.0.process_audio(bytes, is_cancelled)
        }
        fn embed(
            &self,
            sequences: Vec<EngineSequence>,
            is_cancelled: Option<CancellationCheck>,
        ) -> Result<Vec<Vec<f32>>, String> {
            self.0.embed(sequences, is_cancelled)
        }
        fn device_lost(&self) -> bool {
            self.0.device_lost()
        }
    }
    let shared = engine.clone();
    let service = Arc::new(EmbeddingService::new(EmbeddingServiceOptions {
        model_root: "/models/embeddinggemma-2".into(),
        model_validator: Some(Box::new(|_| true)),
        engine_factory: Some(Box::new(move |_| Ok(Box::new(Shared(shared))))),
        ..EmbeddingServiceOptions::default()
    }));
    (service, engine)
}

#[test]
fn the_native_engine_tokenizes_limits_and_reports_device_loss() {
    let unvalidated = EmbeddingService::new(EmbeddingServiceOptions {
        model_root: "/nonexistent".into(),
        validate_model: false,
        engine_factory: Some(Box::new(|_| Err("never created".into()))),
        ..EmbeddingServiceOptions::default()
    });
    assert_eq!(
        unvalidated.initialization_error(),
        "embedding service requires an EmbeddingGemma 2 model directory",
        "the native engine is always validated"
    );
    let failing = EmbeddingService::new(EmbeddingServiceOptions {
        model_validator: Some(Box::new(|_| true)),
        engine_factory: Some(Box::new(|_| {
            Err("embedding encoder failed to load: no device".into())
        })),
        ..EmbeddingServiceOptions::default()
    });
    assert!(!failing.ready());
    assert_eq!(
        failing.initialization_error(),
        "embedding encoder failed to load: no device"
    );

    let (service, engine) = native_service(false);
    assert!(service.ready());
    let response = send(&service, r#"{"input":["a bb","ccc"]}"#);
    assert_eq!(response.status, 200);
    assert_eq!(
        *engine.sequences.lock().unwrap(),
        vec![vec![2, 1, 2, 1], vec![2, 3, 1]],
        "texts are tokenized with their markers"
    );
    let body = parse(&response);
    assert_eq!(
        body.find("usage").unwrap().member_size("prompt_tokens", 0),
        7
    );

    // Longer than 8,192 tokens of at most four bytes: refused untokenized.
    let oversized = "x".repeat(8192 * 4 + 1);
    let response = send(&service, &format!(r#"{{"input":["ok","{oversized}"]}}"#));
    expect_error(&response, 400, "context_length_exceeded", "byte limit");
    // 8,191 words and the markers: one token too many.
    let words = vec!["y"; 8191].join(" ");
    let response = send(&service, &format!(r#"{{"input":["{words}"]}}"#));
    expect_error(&response, 400, "context_length_exceeded", "token limit");
    let words = vec!["y"; 8190].join(" ");
    assert_eq!(
        send(&service, &format!(r#"{{"input":["{words}"]}}"#)).status,
        200
    );

    let cancelled = send_with(
        &service,
        r#"{"input":"x"}"#,
        "POST",
        Some(Arc::new(|| true)),
    );
    expect_error(&cancelled, 499, "cancelled", "cancelled on the device");

    let (lost, _) = native_service(true);
    expect_error(
        &send(&lost, r#"{"input":"x"}"#),
        503,
        "device_lost",
        "lost device",
    );
    assert!(lost.device_lost());
}

#[test]
fn typed_input() {
    let (service, captured, _) = capturing_service();
    let response = send(
        &service,
        r#"{"input":["plain",{"content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"text","text":"after"}]},{"content":[{"type":"text","text":"typed"}]}]}"#,
    );
    assert_eq!(response.status, 200, "typed input succeeds");
    {
        let captured = captured.lock().unwrap();
        assert!(
            captured.len() == 3 && captured[1].content.len() == 3,
            "typed inputs preserve item and part counts"
        );
        assert!(
            captured[0].content[0].value == "plain"
                && captured[1].content[0].value == "before"
                && captured[1].content[1].kind == EmbeddingPartType::Image
                && captured[1].content[1].value == "data:image/png;base64,AAAA"
                && captured[1].content[2].value == "after"
                && captured[2].content[0].value == "typed",
            "typed inputs preserve order and image URL"
        );
    }
    for body in [
        r#"{"input":[{"content":[]}]}"#,
        r#"{"input":[{"content":[{"type":"text","text":""}]}]}"#,
    ] {
        expect_error(
            &send(&service, body),
            400,
            "empty_input",
            "empty typed input",
        );
    }
    for body in [
        r#"{"input":[{"content":"x"}]}"#,
        r#"{"input":[{"content":[true]}]}"#,
        r#"{"input":[{"content":[{"type":"text","text":3}]}]}"#,
        r#"{"input":[{"content":[{"type":"image_url","image_url":"x"}]}]}"#,
        r#"{"input":[{"content":[{"type":"image_url","image_url":{"url":""}}]}]}"#,
        r#"{"input":[{"content":[{"type":"image_url","image_url":{"url":"x","detail":"low"}}]}]}"#,
    ] {
        expect_error(
            &send(&service, body),
            400,
            "invalid_parameter_type",
            "malformed typed input",
        );
    }
    expect_error(
        &send(
            &service,
            r#"{"input":[{"content":[{"type":"input_audio","input_audio":{}}]}]}"#,
        ),
        400,
        "invalid_parameter_type",
        "audio source is required",
    );
    let audio = send(
        &service,
        r#"{"input":["plain",{"content":[{"type":"input_audio","input_audio":{"data":"AAAA","url":"ignored","format":"ignored"}},{"type":"text","text":"middle"},{"type":"input_audio","input_audio":{"url":"https://example.com/test.wav"}},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}"#,
    );
    {
        let captured = captured.lock().unwrap();
        assert!(
            audio.status == 200 && captured.len() == 2 && captured[1].content.len() == 4,
            "audio and mixed content are accepted"
        );
        let parts = &captured[1].content;
        assert!(
            parts[0].kind == EmbeddingPartType::Audio
                && parts[0].value == "AAAA"
                && parts[1].value == "middle"
                && parts[2].kind == EmbeddingPartType::Audio
                && parts[2].value == "https://example.com/test.wav"
                && parts[3].kind == EmbeddingPartType::Image,
            "audio data precedence and part order match the request"
        );
    }
    for body in [
        r#"{"input":[{"content":[{"type":"input_audio","input_audio":null}]}]}"#,
        r#"{"input":[{"content":[{"type":"input_audio","input_audio":{"data":7}}]}]}"#,
        r#"{"input":[{"content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":7}}]}]}"#,
        r#"{"input":[{"content":[{"type":"input_audio","input_audio":{"data":"AAAA","other":"x"}}]}]}"#,
    ] {
        expect_error(
            &send(&service, body),
            400,
            "invalid_parameter_type",
            "malformed audio part",
        );
    }
    let many = |part: &str| {
        format!(
            r#"{{"input":[{{"content":[{}]}}]}}"#,
            vec![part; 33].join(",")
        )
    };
    expect_error(
        &send(
            &service,
            &many(r#"{"type":"input_audio","input_audio":{"data":"AAAA"}}"#),
        ),
        400,
        "too_many_audio",
        "audio count bounded before decoding",
    );
    expect_error(
        &send(
            &service,
            &many(r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}"#),
        ),
        400,
        "too_many_images",
        "image count bounded before reading",
    );
}

#[test]
fn native_embeddings() {
    assert!(
        is_embeddings_api_path("/embeddings")
            && is_embeddings_api_path("/embedding")
            && !is_embeddings_api_path("/embeddings/")
            && !is_embeddings_api_path("/v1/embedding"),
        "native embeddings paths are recognized exactly"
    );
    let (service, captured, calls) = capturing_service();

    // Frigate's text request.
    let text = send_to(
        &service,
        "/embeddings",
        r#"{"model":"embeddinggemma-2","content":[{"prompt_string":"a red car"},{"prompt_string":"a dog"}]}"#,
    );
    assert_eq!(text.status, 200, "native text request succeeds");
    let body = parse(&text);
    assert!(
        body.is_array() && body.len() == 2,
        "native response is a bare array"
    );
    for (i, item) in body.items().iter().enumerate() {
        assert!(
            item.len() == 2 && item.member_size("index", 99) == i,
            "native items carry only index and embedding, in order"
        );
        let nested = item.find("embedding").unwrap();
        assert!(
            nested.is_array() && nested.len() == 1,
            "the pooled vector is nested"
        );
        assert_eq!(
            floats_of(&nested.items()[0]),
            fake_embedding(if i == 0 { "a red car" } else { "a dog" }),
            "native floats round-trip exactly"
        );
    }
    {
        let captured = captured.lock().unwrap();
        assert!(
            captured.len() == 2
                && captured[0].content.len() == 1
                && captured[0].content[0].kind == EmbeddingPartType::Text
                && captured[0].content[0].value == "a red car",
            "a prompt_string without media is one text part"
        );
    }

    // Frigate's image request: the marker, a newline, and one base64 JPEG.
    let image = send_to(
        &service,
        "/embeddings",
        r#"{"model":"embeddinggemma-2","content":[{"prompt_string":"<__media__>\n","multimodal_data":["/9j/AAAA"]}]}"#,
    );
    assert_eq!(image.status, 200, "native image request succeeds");
    {
        let captured = captured.lock().unwrap();
        let parts = &captured[0].content;
        assert!(
            captured.len() == 1
                && parts.len() == 2
                && parts[0].kind == EmbeddingPartType::Image
                && parts[0].value == "data:image/jpeg;base64,/9j/AAAA"
                && parts[1].kind == EmbeddingPartType::Text
                && parts[1].value == "\n",
            "the marker becomes the image part, in place, before the newline"
        );
    }
    let around = send_to(
        &service,
        "/embedding",
        r#"{"content":{"prompt_string":"before <__media__> mid <__media__>","multimodal_data":["AAAA","BBBB"]}}"#,
    );
    assert_eq!(
        around.status, 200,
        "legacy path and a single prompt object work"
    );
    {
        let captured = captured.lock().unwrap();
        let values: Vec<&str> = captured[0]
            .content
            .iter()
            .map(|p| p.value.as_str())
            .collect();
        assert_eq!(
            values,
            [
                "before ",
                "data:image/jpeg;base64,AAAA",
                " mid ",
                "data:image/jpeg;base64,BBBB"
            ],
            "text around markers keeps its order and whitespace"
        );
    }

    // Like llama.cpp's mtmd helper, files that look like WAV, MP3 or FLAC
    // (and Ogg) become audio parts; WebP's RIFF container and PNG stay
    // images.
    let media = send_to(
        &service,
        "/embeddings",
        r#"{"content":{"prompt_string":"<__media__><__media__><__media__><__media__><__media__><__media__><__media__>","multimodal_data":["UklGRiQAAABXQVZFZm10IA==","SUQzBAAAAAAAAAAA","//uQAAAAAAAAAAAA","ZkxhQwAAACIAAAAA","T2dnUwACAAAAAAAA","UklGRiQAAABXRUJQVlA4IA==","iVBORw0KGgoAAAANSUhEUg=="]}}"#,
    );
    assert_eq!(media.status, 200, "native audio files succeed");
    {
        use EmbeddingPartType::{Audio, Image};
        let captured = captured.lock().unwrap();
        let kinds: Vec<EmbeddingPartType> = captured[0].content.iter().map(|p| p.kind).collect();
        assert_eq!(kinds, [Audio, Audio, Audio, Audio, Audio, Image, Image]);
        assert_eq!(
            captured[0].content[0].value, "UklGRiQAAABXQVZFZm10IA==",
            "multimodal_data audio is kept as raw base64"
        );
    }

    // The same text embeds identically on both routes; `input` wins over
    // `content`, as in llama.cpp.
    let plain = send_to(&service, "/embeddings", r#"{"input":"x","content":"y"}"#);
    assert_eq!(plain.status, 200, "input and content together succeed");
    assert_eq!(captured.lock().unwrap()[0].content[0].value, "x");
    let oai = send(&service, r#"{"input":"x"}"#);
    assert_eq!(
        floats_of(&parse(&plain).items()[0].find("embedding").unwrap().items()[0]),
        floats_of(
            parse(&oai).find("data").unwrap().items()[0]
                .find("embedding")
                .unwrap()
        ),
        "both routes return the same vector"
    );
    assert_eq!(
        send_to(
            &service,
            "/embeddings",
            r#"{"content":"y","embd_normalize":2,"encoding_format":"float"}"#
        )
        .status,
        200
    );
    let base64 = send_to(
        &service,
        "/embeddings",
        r#"{"content":"y","encoding_format":"base64"}"#,
    );
    assert!(
        base64.status == 200
            && parse(&base64).items()[0].find("embedding").unwrap().items()[0].is_array(),
        "the native form returns floats for base64 too"
    );
    assert_eq!(
        send_to(
            &service,
            "/embeddings",
            r#"{"content":[{"content":[{"type":"text","text":"typed"}]}]}"#
        )
        .status,
        200,
        "typed content objects work here too"
    );

    let before = calls.load(Ordering::SeqCst);
    expect_error(
        &send_to(&service, "/embeddings", r#"{"model":"embeddinggemma-2"}"#),
        400,
        "missing_input",
        "native request without input or content",
    );
    for (body, what) in [
        (
            r#"{"content":[{"prompt_string":"<__media__><__media__>","multimodal_data":["AAAA"]}]}"#,
            "more markers than files",
        ),
        (
            r#"{"content":[{"prompt_string":"x","multimodal_data":["AAAA"]}]}"#,
            "a file without a marker",
        ),
    ] {
        expect_error(
            &send_to(&service, "/embeddings", body),
            400,
            "invalid_multimodal_data",
            what,
        );
    }
    for bad in [
        r#"{"content":[{"prompt_string":3}]}"#,
        r#"{"content":[{"prompt_string":"x","extra":1}]}"#,
        r#"{"content":[{"prompt_string":"x","multimodal_data":"AAAA"}]}"#,
        r#"{"content":[{"prompt_string":"<__media__>","multimodal_data":[""]}]}"#,
        r#"{"content":[{"prompt_string":"<__media__>","multimodal_data":[7]}]}"#,
    ] {
        expect_error(
            &send_to(&service, "/embeddings", bad),
            400,
            "invalid_parameter_type",
            bad,
        );
    }
    expect_error(
        &send_to(
            &service,
            "/embeddings",
            r#"{"content":[{"prompt_string":""}]}"#,
        ),
        400,
        "empty_input",
        "empty prompt_string",
    );
    expect_error(
        &send_to(&service, "/embeddings", r#"{"content":[1,2,3]}"#),
        400,
        "unsupported_input",
        "token arrays stay unsupported",
    );
    for normalize in ["-1", "0", "1", "3", r#""2""#] {
        expect_error(
            &send_to(
                &service,
                "/embeddings",
                &format!(r#"{{"content":"y","embd_normalize":{normalize}}}"#),
            ),
            400,
            "unsupported_embd_normalize",
            normalize,
        );
    }
    expect_error(
        &send_to(
            &service,
            "/embeddings",
            r#"{"content":"y","encoding_format":"int8"}"#,
        ),
        400,
        "invalid_encoding_format",
        "encoding_format is validated",
    );
    expect_error(
        &send_to(
            &service,
            "/embeddings",
            r#"{"content":"y","pooling":"none"}"#,
        ),
        400,
        "unsupported_field",
        "unknown native fields are rejected",
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        before,
        "rejected native requests never reach the runner"
    );
    // The OpenAI route keeps its own field set.
    expect_error(
        &send(&service, r#"{"content":"y"}"#),
        400,
        "unsupported_field",
        "no content",
    );
    expect_error(
        &send(&service, r#"{"input":"y","embd_normalize":2}"#),
        400,
        "unsupported_field",
        "no embd_normalize",
    );
    expect_error(
        &send(&service, r#"{"input":[{"prompt_string":"y"}]}"#),
        400,
        "invalid_parameter_type",
        "no prompt objects",
    );
}

/// A response's structure: array lengths, member names and float-vector
/// widths, in the fixture's form.
fn skeleton(value: &Value) -> Value {
    match value {
        Value::Array(items) if !items.is_empty() && items.iter().all(Value::is_number) => {
            Value::object().with("floats", items.len())
        }
        Value::Array(items) => {
            let mut list = Value::object().with("list", items.len());
            if let Some(first) = items.first() {
                list.set("item", skeleton(first));
            }
            list
        }
        Value::Object(members) => {
            let mut object = Value::object();
            for (name, member) in members {
                object.set(name, skeleton(member));
            }
            object
        }
        other => other.clone(),
    }
}

/// llama.cpp's captured responses and Frigate's requests
/// (`TestNativeReferenceFixture`).
#[test]
fn native_reference_fixture() {
    let text = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama_cpp_native_embeddings.json"
    ))
    .unwrap();
    let fixture = json::parse(&text).unwrap();
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    let cases = fixture.find("llama_cpp").unwrap().find("cases").unwrap();
    assert!(cases.len() >= 6, "fixture holds the llama.cpp cases");
    for item in cases.items() {
        let mut request = item.find("request").unwrap().clone();
        // The capture served another model name.
        if request.contains("model") {
            request.set("model", "embeddinggemma-2");
        }
        let request = request.dump().unwrap();
        let response = send_to(&service, &item.member_str("path", ""), &request);
        assert_eq!(
            usize::from(response.status),
            item.member_size("status", 0),
            "{request}"
        );
        assert_eq!(
            skeleton(&parse(&response)).dump().unwrap(),
            item.find("response").unwrap().dump().unwrap(),
            "llama.cpp response shape for {request}"
        );
    }
    let frigate = fixture.find("frigate").unwrap();
    let path = frigate.member_str("path", "");
    let texts = send_to(
        &service,
        &path,
        &frigate.find("text").unwrap().dump().unwrap(),
    );
    assert!(texts.status == 200 && parse(&texts).len() == 2);
    let image = send_to(
        &service,
        &path,
        &frigate.find("image").unwrap().dump().unwrap(),
    );
    assert!(image.status == 200 && parse(&image).len() == 1);
}

#[test]
fn truncate() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    expect_error(
        &send(&service, r#"{"input":["short","TOOLONG input"]}"#),
        400,
        "context_length_exceeded",
        "default refuses",
    );
    assert!(
        !runner.last_truncate.load(Ordering::SeqCst),
        "truncation is off by default"
    );
    expect_error(
        &send(&service, r#"{"input":"TOOLONG input","truncate":false}"#),
        400,
        "context_length_exceeded",
        "truncate false refuses",
    );
    let cut = send(
        &service,
        r#"{"input":["short","TOOLONG input"],"truncate":true}"#,
    );
    assert_eq!(cut.status, 200, "truncate true embeds an over-length input");
    assert!(
        runner.last_truncate.load(Ordering::SeqCst),
        "truncate true reaches the service"
    );
    let body = parse(&cut);
    assert_eq!(
        body.find("data").unwrap().len(),
        2,
        "both inputs are embedded"
    );
    assert_eq!(
        body.find("usage").unwrap().member_size("prompt_tokens", 0),
        fake_tokens("short") + MAX_INPUT_TOKENS,
        "usage counts the tokens embedded after truncation"
    );
    expect_error(
        &send(&service, r#"{"input":"a","truncate":"yes"}"#),
        400,
        "invalid_parameter_type",
        "truncate must be a boolean",
    );
    let calls = runner.calls.load(Ordering::SeqCst);
    expect_error(
        &send(&service, r#"{"input":"a","truncate":1}"#),
        400,
        "invalid_parameter_type",
        "a numeric truncate is refused",
    );
    assert_eq!(runner.calls.load(Ordering::SeqCst), calls);
    assert_eq!(
        send(&service, r#"{"input":"a","truncate":null}"#).status,
        200
    );
}

/// The native path's cut for inputs with images: 2 = <bos>, 1 = <eos>,
/// 9 = image placeholder, 8 = end-of-image marker, 5 = text.
#[test]
fn truncation_keeps_images_whole() {
    let mut text = vec![2, 5, 5, 5, 5, 5, 1];
    assert!(truncate_keeping_images(&mut text, 4, 9) && text == [2, 5, 5, 1]);
    let mut image = vec![2, 5, 9, 9, 8, 5, 5, 5, 1];
    assert!(truncate_keeping_images(&mut image, 6, 9) && image == [2, 5, 9, 9, 8, 1]);
    let mut marker = vec![2, 5, 9, 9, 8, 5, 5, 5, 1];
    assert!(!truncate_keeping_images(&mut marker, 5, 9) && marker == [2, 5, 9, 9, 8, 5, 5, 5, 1]);
    let mut inside = vec![2, 9, 9, 9, 8, 5, 1];
    assert!(!truncate_keeping_images(&mut inside, 3, 9));
    let mut fits = vec![2, 9, 8, 1];
    assert!(truncate_keeping_images(&mut fits, 4, 9) && fits == [2, 9, 8, 1]);
}

fn prompt_service(runner: &Arc<FakeRunner>, default_name: &str) -> Arc<EmbeddingService> {
    let runner = runner.clone();
    Arc::new(EmbeddingService::new(EmbeddingServiceOptions {
        validate_model: false,
        model_id: "embeddinggemma-2".into(),
        prompts: [
            (
                "query".to_owned(),
                "task: search result | query: ".to_owned(),
            ),
            ("document".to_owned(), "title: none | text: ".to_owned()),
        ]
        .into(),
        default_prompt_name: default_name.into(),
        runner: Some(Box::new(move |inputs, request, batch, error| {
            runner.run(inputs, request, batch, error)
        })),
        ..EmbeddingServiceOptions::default()
    }))
}

#[test]
fn prompt_names() {
    let runner = Arc::new(FakeRunner::default());
    let texts = || runner.last_texts.lock().unwrap().clone();
    let service = prompt_service(&runner, "");
    assert_eq!(send(&service, r#"{"input":"a"}"#).status, 200);
    assert_eq!(
        texts(),
        ["a"],
        "without a default, inputs reach the encoder unchanged"
    );
    assert_eq!(
        send(&service, r#"{"input":["a","b"],"prompt_name":"query"}"#).status,
        200
    );
    assert_eq!(
        texts(),
        [
            "task: search result | query: a",
            "task: search result | query: b"
        ],
        "prompt_name prefixes every input"
    );
    let calls = runner.calls.load(Ordering::SeqCst);
    expect_error(
        &send(&service, r#"{"input":"a","prompt_name":"Query"}"#),
        400,
        "invalid_prompt_name",
        "prompt names are case-sensitive",
    );
    expect_error(
        &send(&service, r#"{"input":"a","prompt_name":1}"#),
        400,
        "invalid_parameter_type",
        "a non-string prompt_name",
    );
    assert_eq!(runner.calls.load(Ordering::SeqCst), calls);

    let with_default = prompt_service(&runner, "document");
    assert!(with_default.ready(), "a known default prompt");
    assert_eq!(send(&with_default, r#"{"input":"a"}"#).status, 200);
    assert_eq!(texts(), ["title: none | text: a"]);
    assert_eq!(
        send(&with_default, r#"{"input":"a","prompt_name":"query"}"#).status,
        200
    );
    assert_eq!(texts(), ["task: search result | query: a"]);
    assert_eq!(
        send(&with_default, r#"{"input":"a","prompt_name":""}"#).status,
        200
    );
    assert_eq!(texts(), ["a"], "an empty prompt_name applies no prompt");
    assert_eq!(
        send(&with_default, r#"{"input":"a","prompt_name":null}"#).status,
        200
    );
    assert_eq!(
        texts(),
        ["title: none | text: a"],
        "a null prompt_name keeps the default"
    );
    assert_eq!(
        send_to(
            &with_default,
            "/embeddings",
            r#"{"content":"a","prompt_name":"query"}"#
        )
        .status,
        200
    );
    assert_eq!(texts(), ["task: search result | query: a"]);

    // The model card passes images and audio without a prefix: the default
    // skips inputs without text, but a requested prompt applies to all.
    let image = r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}"#;
    let audio = r#"{"type":"input_audio","input_audio":{"data":"AAAA"}}"#;
    let caption = r#"{"type":"text","text":"cap"}"#;
    let media = format!(
        r#"{{"input":[{{"content":[{image}]}},{{"content":[{audio}]}},{{"content":[{image},{caption}]}},"a"]}}"#
    );
    assert_eq!(send(&with_default, &media).status, 200);
    let last = texts();
    assert_eq!(last.len(), 4);
    assert_eq!(last[0], "data:image/png;base64,AAAA");
    assert!(!last[1].starts_with("title: none | text: "));
    assert_eq!(last[2], "title: none | text: data:image/png;base64,AAAAcap");
    assert_eq!(last[3], "title: none | text: a");
    assert_eq!(
        send_to(
            &with_default,
            "/embeddings",
            r#"{"content":{"prompt_string":"<__media__>","multimodal_data":["AAAA"]}}"#
        )
        .status,
        200
    );
    assert!(!texts()[0].starts_with("title: none | text: "));
    assert_eq!(
        send(
            &with_default,
            &format!(r#"{{"input":[{{"content":[{image}]}}],"prompt_name":"query"}}"#)
        )
        .status,
        200
    );
    assert_eq!(
        texts(),
        ["task: search result | query: data:image/png;base64,AAAA"]
    );

    let unknown = prompt_service(&runner, "missing");
    assert!(!unknown.ready() && unknown.initialization_error().contains("missing"));

    let dir = tempfile::tempdir().unwrap();
    assert!(
        load_sentence_transformer_prompts(dir.path())
            .unwrap()
            .is_empty()
    );
    let file = dir.path().join("config_sentence_transformers.json");
    std::fs::write(
        &file,
        r#"{"default_prompt_name":null,"prompts":{"query":"q: ","document":"d: "}}"#,
    )
    .unwrap();
    let prompts = load_sentence_transformer_prompts(dir.path()).unwrap();
    assert!(prompts.len() == 2 && prompts["query"] == "q: " && prompts["document"] == "d: ");
    std::fs::write(&file, r#"{"prompts":{"query":3}}"#).unwrap();
    assert!(
        load_sentence_transformer_prompts(dir.path())
            .unwrap_err()
            .contains("query")
    );
    std::fs::write(&file, r#"{"prompts":["q: "]}"#).unwrap();
    assert!(
        load_sentence_transformer_prompts(dir.path())
            .unwrap_err()
            .contains("must be an object")
    );
    std::fs::write(&file, r#"{"prompts":"#).unwrap();
    assert!(
        load_sentence_transformer_prompts(dir.path())
            .unwrap_err()
            .contains("config_sentence_transformers.json")
    );
}

/// Video parts and `video_options` reach the service in request order
/// (`TestTypedInput`, Gufo #869).
#[test]
fn typed_video_input() {
    let captured = Arc::new(Mutex::new((Vec::new(), EmbeddingRequestOptions::default())));
    let c = captured.clone();
    let service = service_with(Box::new(move |inputs, options, batch, _| {
        *c.lock().unwrap() = (inputs.to_vec(), *options);
        for input in inputs {
            let text = joined(input);
            batch.embeddings.push(fake_embedding(&text));
            batch.token_counts.push(fake_tokens(&text));
        }
        EmbeddingStatus::Ok
    }));
    let response = send(
        &service,
        r#"{"input":[{"content":[{"type":"text","text":"before"},{"type":"input_video","input_video":{"data":"AAAA","url":"ignored"}},{"type":"text","text":"after"}]}],"video_options":{"fps":0.5,"max_frames":3,"add_timestamps":true,"overflow_strategy":"truncate"}}"#,
    );
    assert_eq!(response.status, 200, "video content and controls accepted");
    {
        let (inputs, options) = &*captured.lock().unwrap();
        let parts = &inputs[0].content;
        assert!(
            parts.len() == 3
                && parts[1].kind == EmbeddingPartType::Video
                && parts[1].value == "AAAA"
                && parts[0].value == "before"
                && parts[2].value == "after",
            "video data precedence and interleaving"
        );
        assert_eq!(
            options.video,
            EmbeddingVideoOptions {
                fps: 0.5,
                max_frames: 3,
                timestamps: true,
                truncate_overflow: true,
            },
            "video controls reach runtime"
        );
    }
    for body in [
        r#"{"input":[{"content":[{"type":"input_video","input_video":{}}]}]}"#,
        r#"{"input":[{"content":[{"type":"input_video","input_video":{"data":3}}]}]}"#,
        r#"{"input":[{"content":[{"type":"input_video","input_video":{"url":"x","extra":"x"}}]}]}"#,
    ] {
        expect_error(
            &send(&service, body),
            400,
            "invalid_parameter_type",
            "invalid video schema",
        );
    }
    for settings in [
        r#"{"fps":0}"#,
        r#"{"fps":121}"#,
        r#"{"max_frames":0}"#,
        r#"{"max_frames":33}"#,
        r#"{"max_frames":1.5}"#,
        r#"{"add_timestamps":1}"#,
        r#"{"overflow_strategy":"other"}"#,
        r#"{"unknown":true}"#,
    ] {
        expect_error(
            &send(
                &service,
                &format!(r#"{{"input":"hello","video_options":{settings}}}"#),
            ),
            400,
            "invalid_video_options",
            "invalid video control",
        );
    }
    let many = format!(
        r#"{{"input":[{{"content":[{}]}}]}}"#,
        vec![r#"{"type":"input_video","input_video":{"data":"AAAA"}}"#; 33].join(",")
    );
    expect_error(
        &send(&service, &many),
        400,
        "too_many_videos",
        "video request count bound",
    );
}

/// `mm_processor_kwargs.max_soft_tokens` selects the image budget, as in
/// vLLM's embedding requests; the default is the checkpoint's 280
/// (`TestImageSoftTokens`, Gufo #870).
#[test]
fn image_soft_tokens() {
    let runner = Arc::new(FakeRunner::default());
    let service = make_service(&runner);
    let last = || runner.last_image_soft_tokens.load(Ordering::SeqCst);
    assert_eq!(
        send(&service, r#"{"input":"a"}"#).status,
        200,
        "default request"
    );
    assert_eq!(last(), 280, "the default budget is 280");
    for budget in [70u32, 140, 280, 560, 1120] {
        let response = send(
            &service,
            &format!(r#"{{"input":"a","mm_processor_kwargs":{{"max_soft_tokens":{budget}}}}}"#),
        );
        assert_eq!(response.status, 200, "a supported budget is accepted");
        assert_eq!(last(), budget, "the budget reaches the service");
    }
    assert_eq!(
        send_to(
            &service,
            "/embeddings",
            r#"{"content":"a","mm_processor_kwargs":{"max_soft_tokens":70}}"#
        )
        .status,
        200,
        "the native form accepts the budget"
    );
    assert_eq!(last(), 70, "the native budget is applied");
    assert_eq!(
        send(&service, r#"{"input":"a","mm_processor_kwargs":null}"#).status,
        200,
        "null mm_processor_kwargs is absent"
    );
    assert_eq!(last(), 280, "null keeps the default");
    assert_eq!(
        send(&service, r#"{"input":"a","mm_processor_kwargs":{}}"#).status,
        200,
        "empty mm_processor_kwargs keeps the default"
    );
    let calls = runner.calls.load(Ordering::SeqCst);
    expect_error(
        &send(&service, r#"{"input":"a","mm_processor_kwargs":[]}"#),
        400,
        "invalid_parameter_type",
        "mm_processor_kwargs is an object",
    );
    for body in [
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":281}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":0}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":-70}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":null}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":true}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":[]}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":{}}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":1121}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":280.5}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":"280"}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":4294967576}}"#,
        r#"{"input":"a","mm_processor_kwargs":{"fps":2}}"#,
    ] {
        expect_error(
            &send(&service, body),
            400,
            "invalid_mm_processor_kwargs",
            &format!("refused: {body}"),
        );
    }
    assert_eq!(
        runner.calls.load(Ordering::SeqCst),
        calls,
        "invalid kwargs never reach the service"
    );
    assert_eq!(
        send(
            &service,
            r#"{"input":"a","mm_processor_kwargs":{"max_soft_tokens":70},"video_options":{"fps":0.5,"max_frames":3}}"#
        )
        .status,
        200,
        "image and video processor options coexist"
    );
    assert_eq!(last(), 70, "video options do not override the image budget");
}

/// The native service refuses video without the model's support, reports a
/// video it cannot decode as the client's error and missing video libraries
/// as its own.
#[test]
fn native_video_failures() {
    let video = r#"{"input":[{"content":[{"type":"input_video","input_video":{"data":"AAAA"}}]}]}"#;
    let (unsupported, _) = native_service(false);
    let response = send(&unsupported, video);
    expect_error(&response, 400, "invalid_video_input", "video unsupported");
    assert_eq!(
        parse(&response)
            .find("error")
            .unwrap()
            .member_str("message", ""),
        "unsupported embedding vision configuration"
    );
    let (invalid, _) = native_service_with(
        false,
        Some(EngineVideoError::Invalid(
            "embedding video has no video stream".into(),
        )),
    );
    let response = send(&invalid, video);
    expect_error(&response, 400, "invalid_video_input", "undecodable video");
    assert_eq!(
        parse(&response)
            .find("error")
            .unwrap()
            .member_str("message", ""),
        "embedding video has no video stream"
    );
    expect_error(
        &send(
            &invalid,
            r#"{"input":[{"content":[{"type":"input_video","input_video":{"data":"data:video/avi;base64,AAAA"}}]}]}"#,
        ),
        400,
        "invalid_video_input",
        "unsupported video media type",
    );
    let (missing, _) = native_service_with(
        false,
        Some(EngineVideoError::Library(
            "cannot load runtime library libavformat.so.62: not found".into(),
        )),
    );
    expect_error(
        &send(&missing, video),
        500,
        "embedding_failed",
        "missing video libraries",
    );
}
