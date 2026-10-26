//! `TestRequestLogging` and the log-level tests of
//! `tests/cli/http_server_test.cpp`, alone in their binary because they
//! capture the log of every thread and set the process-wide level; they run
//! one at a time.

mod common;
mod http_fixture;

use std::sync::{Mutex, MutexGuard};

use gufo_serve::http::HttpServerOptions;
use gufo_serve::logging::{self, LogLevel};

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
