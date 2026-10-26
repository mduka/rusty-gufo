//! `POST /v1/embeddings` against a fake encoder: no model files, no device
//! and no socket; the handler is called directly
//! (`tests/models/embedding_gemma2/embeddings_api_test.cpp`).

#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gufo_core::json::{self, Value};
use gufo_serve::embedding_service::{
    EmbeddingBatch, EmbeddingEngine, EmbeddingRunner, EmbeddingService, EmbeddingServiceOptions,
    EmbeddingStatus,
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
}

impl FakeRunner {
    fn run(
        &self,
        texts: &[String],
        batch: &mut EmbeddingBatch,
        error: &mut String,
    ) -> EmbeddingStatus {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last_texts.lock().unwrap() = texts.to_vec();
        for (i, text) in texts.iter().enumerate() {
            if text.starts_with("TOOLONG") {
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
    service_with(Box::new(move |texts, batch, error| {
        runner.run(texts, batch, error)
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
            r#"{"input":"x","truncate":true}"#,
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
    let short_service = service_with(Box::new(|_, batch, _| {
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
    let cancelling = service_with(Box::new(move |texts, batch, _| {
        flag.store(true, Ordering::SeqCst);
        batch.embeddings = vec![fake_embedding(&texts[0])];
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
        runner: Some(Box::new(|_, _, _| EmbeddingStatus::Ok)),
        ..EmbeddingServiceOptions::default()
    });
    assert_eq!(defaulted.model_id(), "embeddinggemma-2", "default model id");

    // Validation rejects a directory that is not an EmbeddingGemma 2
    // checkpoint.
    let invalid = EmbeddingService::new(EmbeddingServiceOptions {
        model_root: "/nonexistent/embeddinggemma".into(),
        validate_model: true,
        model_validator: Some(Box::new(|root| root.join("config.json").is_file())),
        runner: Some(Box::new(|_, _, _| EmbeddingStatus::Ok)),
        ..EmbeddingServiceOptions::default()
    });
    assert!(
        !invalid.ready() && !invalid.initialization_error().is_empty(),
        "an invalid model directory is refused"
    );
    let mut batch = EmbeddingBatch::default();
    let mut error = String::new();
    assert!(
        invalid.embed(&["x".into()], &|| false, &mut batch, &mut error) == EmbeddingStatus::Failed
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
        runner: Some(Box::new(move |texts, out, _| {
            if a.fetch_add(1, Ordering::SeqCst) != 0 {
                o.fetch_add(1, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_millis(5));
            out.embeddings = vec![fake_embedding(&texts[0])];
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
                    serial.embed(&["x".into()], &|| false, &mut result, &mut error),
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

    fn embed(
        &self,
        sequences: &[Vec<u32>],
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Vec<f32>>, String> {
        *self.sequences.lock().unwrap() = sequences.to_vec();
        if is_cancelled() {
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
    let engine = Arc::new(FakeEngine {
        lost: AtomicBool::new(false),
        fail,
        sequences: Mutex::new(Vec::new()),
    });
    struct Shared(Arc<FakeEngine>);
    impl EmbeddingEngine for Shared {
        fn max_token_bytes(&self) -> usize {
            self.0.max_token_bytes()
        }
        fn encode(&self, text: &str) -> Vec<u32> {
            self.0.encode(text)
        }
        fn embed(
            &self,
            sequences: &[Vec<u32>],
            is_cancelled: &dyn Fn() -> bool,
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
