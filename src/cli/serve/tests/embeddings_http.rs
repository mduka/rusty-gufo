//! `/v1/embeddings` through the HTTP server
//! (`TestEmbeddingService` of `tests/cli/http_server_test.cpp`): a server with
//! only an embedding service and no text backend answers it behind the API
//! key, lists the model and reports readiness; a lost device shuts it down.

#![allow(clippy::unwrap_used)]

mod common;
mod http_fixture;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use gufo_serve::backend::TextGenerationBackend;
use gufo_serve::embedding_service::{EmbeddingService, EmbeddingServiceOptions, EmbeddingStatus};
use gufo_serve::http::HttpServerOptions;
use gufo_serve::http::server::{HttpServer, HttpServices};
use http_fixture::{RunningServer, expect_status};

const REQUEST: &str = r#"{"model":"embedding-test","input":"hi"}"#;

/// A service over a fake runner; `ready` false makes validation fail, since
/// there is no checkpoint.
fn make_service(ready: bool) -> Arc<EmbeddingService> {
    Arc::new(EmbeddingService::new(EmbeddingServiceOptions {
        model_root: "/nonexistent/embeddinggemma".into(),
        validate_model: !ready,
        model_validator: Some(Box::new(|root| root.join("config.json").is_file())),
        model_id: "embedding-test".into(),
        runner: Some(Box::new(|texts, batch, _| {
            for text in texts {
                batch.embeddings.push(vec![1.0, 0.0]);
                batch.token_counts.push(text.len() + 2);
            }
            EmbeddingStatus::Ok
        })),
        ..EmbeddingServiceOptions::default()
    }))
}

/// A running server without a text backend.
struct EmbeddingServer {
    server: Arc<HttpServer>,
    worker: Option<JoinHandle<()>>,
}

impl EmbeddingServer {
    fn new(service: Option<Arc<EmbeddingService>>, options: HttpServerOptions) -> Self {
        Self::with_hook(service, options, None)
    }

    fn with_hook(
        service: Option<Arc<EmbeddingService>>,
        options: HttpServerOptions,
        hook_calls: Option<Arc<AtomicUsize>>,
    ) -> Self {
        let server = Arc::new(
            HttpServer::new(
                "127.0.0.1",
                0,
                None,
                HttpServices {
                    embeddings: service,
                    ..HttpServices::default()
                },
                options,
            )
            .unwrap(),
        );
        if let Some(calls) = hook_calls {
            server.on_fatal_failure(move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
            });
        }
        server.start().unwrap();
        let running = server.clone();
        let worker = std::thread::spawn(move || running.run(false));
        Self {
            server,
            worker: Some(worker),
        }
    }

    fn send(&self, request: &str) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", self.server.port())).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = Vec::new();
        let _ = stream.read_to_end(&mut response);
        String::from_utf8_lossy(&response).into_owned()
    }

    fn post(&self, body: &str, extra_headers: &str) -> String {
        self.send(&format!(
            "POST /v1/embeddings HTTP/1.1\r\n{extra_headers}Content-Length: {}\r\n\r\n{body}",
            body.len()
        ))
    }
}

impl Drop for EmbeddingServer {
    fn drop(&mut self) {
        self.server.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[test]
fn an_embedding_only_server_serves_lists_and_reports_ready() {
    let service = make_service(true);
    assert!(service.ready());
    let server = EmbeddingServer::new(
        Some(service),
        HttpServerOptions {
            api_key: "test-secret".into(),
            ..HttpServerOptions::default()
        },
    );
    let auth = "Authorization: Bearer test-secret\r\n";
    expect_status(&server.post(REQUEST, ""), 401);
    expect_status(&server.post(REQUEST, "Authorization: Bearer nope\r\n"), 401);
    let response = server.post(REQUEST, auth);
    expect_status(&response, 200);
    assert!(response.contains(r#""object":"list""#), "{response}");
    assert!(response.contains(r#""embedding":[1,0]"#), "{response}");
    assert!(response.contains(r#""prompt_tokens":4"#), "{response}");
    assert!(
        response.contains(r#""model":"embedding-test""#),
        "{response}"
    );
    let missing_model = server.post(r#"{"model":"other","input":"hi"}"#, auth);
    expect_status(&missing_model, 404);
    assert!(missing_model.contains(r#""code":"model_not_found""#));
    let models = server.send(&format!("GET /v1/models HTTP/1.1\r\n{auth}\r\n"));
    expect_status(&models, 200);
    assert!(models.contains(r#""id":"embedding-test""#), "{models}");
    assert!(models.contains(r#""capability":"embedding""#), "{models}");
    let ready = server.send(&format!("GET /ready HTTP/1.1\r\n{auth}\r\n"));
    expect_status(&ready, 200);
    assert!(ready.contains(r#""model":"embedding-test""#), "{ready}");
    // Routes of other modalities stay absent.
    expect_status(
        &server.send(&format!(
            "POST /v1/chat/completions HTTP/1.1\r\n{auth}Content-Length: 2\r\n\r\n{{}}"
        )),
        404,
    );
}

#[test]
fn a_service_that_failed_to_start_answers_503() {
    let service = make_service(false);
    assert!(!service.ready());
    let server = EmbeddingServer::new(Some(service), HttpServerOptions::default());
    let response = server.post(REQUEST, "");
    expect_status(&response, 503);
    assert!(response.contains(r#""code":"embedding_service_unavailable""#));
    expect_status(&server.send("GET /ready HTTP/1.1\r\n\r\n"), 503);
    let models = server.send("GET /v1/models HTTP/1.1\r\n\r\n");
    assert!(!models.contains("embedding-test"), "{models}");
}

#[test]
fn without_an_embedding_service_the_route_is_absent_or_the_text_stub() {
    // A server with no text backend does not serve the route.
    let server = EmbeddingServer::new(None, HttpServerOptions::default());
    expect_status(&server.post(REQUEST, ""), 404);
    // One with a backend keeps its stub.
    let text = RunningServer::new(HttpServerOptions::default());
    let response = text.post("/v1/embeddings", r#"{"input":"hi"}"#);
    assert!(!response.starts_with("HTTP/1.1 200 "), "{response}");
    assert!(
        !response.contains("embedding_service_unavailable"),
        "{response}"
    );
}

#[test]
fn a_lost_device_answers_503_runs_the_hook_once_and_fails_health() {
    let service = Arc::new(EmbeddingService::new(EmbeddingServiceOptions {
        model_root: "/nonexistent/embeddinggemma".into(),
        validate_model: false,
        model_id: "embedding-test".into(),
        runner: Some(Box::new(|_, _, _| EmbeddingStatus::DeviceLost)),
        ..EmbeddingServiceOptions::default()
    }));
    let hook_calls = Arc::new(AtomicUsize::new(0));
    let server = EmbeddingServer::with_hook(
        Some(service),
        HttpServerOptions::default(),
        Some(hook_calls.clone()),
    );
    expect_status(&server.send("GET /health HTTP/1.1\r\n\r\n"), 200);
    let lost = server.post(REQUEST, "");
    expect_status(&lost, 503);
    assert!(lost.contains(r#""code":"device_lost""#), "{lost}");
    assert_eq!(hook_calls.load(Ordering::SeqCst), 1);
    expect_status(&server.send("GET /health HTTP/1.1\r\n\r\n"), 503);
    expect_status(&server.post(REQUEST, ""), 503);
    assert_eq!(hook_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn with_a_text_backend_too_the_service_answers_instead_of_the_stub() {
    let text = RunningServer::new(HttpServerOptions::default());
    let backend: Arc<dyn TextGenerationBackend> = text.backend.clone();
    let server = Arc::new(
        HttpServer::new(
            "127.0.0.1",
            0,
            Some(backend),
            HttpServices {
                embeddings: Some(make_service(true)),
                ..HttpServices::default()
            },
            HttpServerOptions::default(),
        )
        .unwrap(),
    );
    server.start().unwrap();
    let running = server.clone();
    let worker = std::thread::spawn(move || running.run(false));
    let mut stream = TcpStream::connect(("127.0.0.1", server.port())).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream
        .write_all(
            format!(
                "POST /v1/embeddings HTTP/1.1\r\nContent-Length: {}\r\n\r\n{REQUEST}",
                REQUEST.len()
            )
            .as_bytes(),
        )
        .unwrap();
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    expect_status(&String::from_utf8_lossy(&response), 200);
    server.stop();
    worker.join().unwrap();
}
