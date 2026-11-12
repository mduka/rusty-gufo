//! `TestRequestLogging`, `TestConnectionLimitIgnoresClosingConnections` and
//! the log-level tests of `tests/cli/http_server_test.cpp`, alone in their
//! binary because they capture or intercept the log of every thread and set
//! the process-wide level; they run one at a time.

mod common;
mod http_fixture;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use gufo_serve::http::HttpServerOptions;
use gufo_serve::logging::{self, LogLevel};

use common::Semaphore;
use http_fixture::{RunningServer, expect_status};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// One pass of traffic at `level`: a health poll, a streamed chat request
/// with private text, an unknown route and a failing stream. Returns the
/// chat request's ID and every logged line.
fn capture_traffic(level: LogLevel) -> (String, String) {
    let previous = logging::level();
    logging::set_level(level);
    let captured = logging::capture_all(|| {
        let request_id = {
            let server = RunningServer::new(HttpServerOptions::default());
            expect_status(&server.send("GET /health HTTP/1.1\r\n\r\n", false), 200);
            let response = server.post(
                "/v1/chat/completions?private-query",
                r#"{"model":"test","messages":[{"role":"user","content":"private-prompt"}],"stream":true}"#,
            );
            expect_status(&response, 200);
            let begin =
                response.find("X-Request-ID: ").expect("request ID") + "X-Request-ID: ".len();
            let end = begin + response[begin..].find("\r\n").expect("header end");
            let request_id = response[begin..end].to_owned();
            expect_status(
                &server.send("GET /not-a-route HTTP/1.1\r\n\r\n", false),
                404,
            );
            let failed = server.post("/stream-error", "");
            expect_status(&failed, 200);
            assert!(failed.contains("first chunk"));
            assert!(!failed[1..].contains("HTTP/1.1"));
            request_id
        };
        logging::info("test", "escaped\n\x1b[31m");
        request_id
    });
    logging::set_level(previous);
    captured
}

#[test]
fn request_logging() {
    let _serial = serial();
    let (request_id, log) = capture_traffic(LogLevel::Info);
    assert!(
        log.contains(&format!("request={request_id} event=received")),
        "{log}"
    );
    assert!(
        log.contains(&format!("request={request_id} event=completed")),
        "{log}"
    );
    assert!(log.contains("cache=memory"), "{log}");
    assert!(log.contains("acceptance_pct=50.0"), "{log}");
    assert!(log.contains("rss_mib="), "{log}");
    assert!(log.contains("error_code=server_exception"), "{log}");
    // A successful poll is quiet at the default level; statuses never are.
    assert!(!log.contains("path=/health"), "{log}");
    assert!(log.contains("[WARN] [http] request="), "{log}");
    assert!(log.contains("path=/not-a-route status=404"), "{log}");
    assert!(log.contains("event=listening"), "{log}");
    assert!(!log.contains("private-query"), "{log}");
    assert!(!log.contains("private-prompt"), "{log}");
    assert!(log.contains("escaped\\x0a\\x1b[31m"), "{log}");
}

/// Debug adds the quiet polls without downgrading inference completions or
/// revealing request text.
#[test]
fn debug_tier_shows_polls() {
    let _serial = serial();
    let (request_id, log) = capture_traffic(LogLevel::Debug);
    assert!(log.contains("[DEBUG]"), "{log}");
    assert!(
        log.contains("event=received method=GET path=/health"),
        "{log}"
    );
    assert!(log.contains("path=/health status=200"), "{log}");
    assert!(
        log.contains(&format!(
            "[INFO] [http] request={request_id} event=completed"
        )),
        "{log}"
    );
    assert!(log.contains("path=/not-a-route status=404"), "{log}");
    assert!(!log.contains("private-query"), "{log}");
    assert!(!log.contains("private-prompt"), "{log}");
    // `gufo serve` echoes the options, not the HTTP layer.
    assert!(!log.contains("event=options"), "{log}");
}

/// The quiet tiers boot silently, while an escalation keeps its own tier
///.
#[test]
fn quiet_tiers_suppress_lifecycle() {
    let _serial = serial();
    let (_, warn) = capture_traffic(LogLevel::Warn);
    assert!(!warn.contains("event=listening"), "{warn}");
    assert!(!warn.contains("event=received"), "{warn}");
    assert!(warn.contains("path=/not-a-route status=404"), "{warn}");
    let (_, error) = capture_traffic(LogLevel::Error);
    assert!(!error.contains("event=listening"), "{error}");
    assert!(!error.contains("event=received"), "{error}");
    assert!(!error.contains("path=/not-a-route status=404"), "{error}");
    assert!(error.contains("error_code=server_exception"), "{error}");
}

/// A generic failure keeps its cause in the server log, not only in the
/// client's response, whether it happens before or after streaming starts.
#[test]
fn generation_failure_logs_cause() {
    let _serial = serial();
    let ((), log) = logging::capture_all(|| {
        let server = RunningServer::new(HttpServerOptions::default());
        for failure in [1, 6] {
            server
                .backend
                .failure
                .store(failure, std::sync::atomic::Ordering::SeqCst);
            for (path, body) in [
                ("/v1/completions", r#"{"prompt":"hello","stream":true}"#),
                (
                    "/v1/chat/completions",
                    r#"{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
                ),
                ("/v1/responses", r#"{"input":"hello","stream":true}"#),
            ] {
                expect_status(
                    &server.post(path, body),
                    if failure == 6 { 200 } else { 500 },
                );
            }
        }
    });
    // Before streaming, the HTTP handler logs the cause with the request ID.
    assert_eq!(
        log.matches("event=server_exception reason=context exceeded")
            .count(),
        3,
        "{log}"
    );
    // After streaming starts, each endpoint logs it once beside its stable code.
    assert_eq!(
        log.matches("[ERROR] [chat] context exceeded").count(),
        3,
        "{log}"
    );
}

/// Stops the thread that writes a request's completion record, which the
/// server emits after the response is on the wire (`CompletionGate`). This
/// holds a worker in its closing phase for as long as the test wants. The
/// gate blocks under the logger's lock, so a probe request must log nothing
/// at the default level; `GET /health` qualifies.
#[derive(Default)]
struct CompletionGate {
    open: AtomicBool,
    held: Semaphore,
    released: Semaphore,
}

impl CompletionGate {
    fn pass(&self, line: &str) {
        if !self.open.load(Ordering::SeqCst) && line.contains("event=completed") {
            self.held.release();
            self.released.try_acquire_for(Duration::from_secs(5));
        }
    }

    fn wait_until_held(&self) -> bool {
        self.held.try_acquire_for(Duration::from_secs(5))
    }

    fn release(&self) {
        self.open.store(true, Ordering::SeqCst);
        self.released.release();
    }
}

/// One response, read up to the end its `Content-Length` declares.
fn read_framed_response(stream: &mut TcpStream) -> String {
    let mut response = Vec::new();
    let mut buffer = [0u8; 4096];
    let mut total = None;
    loop {
        if total.is_none()
            && let Some(head) = response.windows(4).position(|w| w == b"\r\n\r\n")
        {
            let text = String::from_utf8_lossy(&response[..head]).into_owned();
            let length = text
                .find("Content-Length: ")
                .expect("Content-Length before the body");
            let digits: String = text[length + 16..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            total = Some(head + 4 + digits.parse::<usize>().expect("length"));
        }
        if total.is_some_and(|total| response.len() >= total) {
            break;
        }
        match stream.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(count) => response.extend_from_slice(&buffer[..count]),
        }
    }
    String::from_utf8_lossy(&response).into_owned()
}

#[test]
fn connection_limit_ignores_closing_connections() {
    let _serial = serial();
    let gate = Arc::new(CompletionGate::default());
    let hook = gate.clone();
    logging::intercept_all(
        move |line| hook.pass(line),
        || {
            let server = RunningServer::new(HttpServerOptions {
                max_connections: 2,
                ..HttpServerOptions::default()
            });
            let idle = server.connect();
            let mut answered = server.connect();

            // Both slots hold a connection the server is still reading from.
            let mut refused = server.connect();
            let refusal = read_framed_response(&mut refused);
            expect_status(&refusal, 503);
            assert!(refusal.contains("connection limit reached"), "{refusal}");
            drop(refused);

            answered
                .write_all(b"POST /echo HTTP/1.1\r\nContent-Length: 4\r\n\r\nping")
                .expect("send");
            assert!(read_framed_response(&mut answered).ends_with("ping"));
            // The response is complete but the server has not closed the
            // connection: its worker is parked writing the completion record.
            assert!(gate.wait_until_held());

            // A client that reads its response and reconnects at once must
            // be served.
            let mut next = server.connect();
            next.write_all(b"GET /health HTTP/1.1\r\n\r\n")
                .expect("send");
            expect_status(&read_framed_response(&mut next), 200);

            gate.release();
            drop(next);
            drop(answered);
            drop(idle);
        },
    );
}
