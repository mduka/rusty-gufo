//! Request deadlines of the eval client against a loopback fixture
//! (`tests/quality/eval_http_client_test.cpp`), and the base URL checks.

#![allow(clippy::unwrap_used)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;

use gufo_eval::http::{HttpClient, normalize_base_url, validate_base_url};

/// One request against a server that answers `response` and then either
/// closes or stalls for up to two seconds.
fn check_transfer(post: bool, response: &'static str, stalls: bool) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (release, released) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut request = [0u8; 4096];
        assert!(peer.read(&mut request).unwrap() > 0, "client sends request");
        if !response.is_empty() {
            peer.write_all(response.as_bytes()).unwrap();
        }
        // Also bounds this test if the timeout regresses.
        if stalls {
            let _ = released.recv_timeout(Duration::from_secs(2));
        }
    });
    let client = HttpClient::new(
        &format!("http://127.0.0.1:{port}"),
        "",
        Duration::from_millis(100),
    );
    let result = if post {
        client.post_json("/completion", "{}")
    } else {
        client.get("/models")
    };
    let _ = release.send(());
    server.join().unwrap();
    if stalls {
        assert!(
            !result.transport_ok && result.transport_code == "request_timeout",
            "stalled request reports a bounded transport timeout: {result:?}"
        );
        assert!(
            (50.0..1500.0).contains(&result.elapsed_ms),
            "deadline applies after connection, including partial response: {result:?}"
        );
    } else {
        assert!(
            result.transport_ok && result.status_code == 200 && result.body == b"{}",
            "complete response remains successful: {result:?}"
        );
    }
}

#[test]
fn request_deadlines() {
    for post in [false, true] {
        check_transfer(post, "", true);
        check_transfer(
            post,
            "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{",
            true,
        );
        check_transfer(
            post,
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}",
            false,
        );
    }
}

#[test]
fn base_urls_are_validated_and_normalized() {
    assert!(validate_base_url("http://127.0.0.1:8080/v1").is_ok());
    assert!(validate_base_url("https://example.com").is_ok());
    for (url, error) in [
        ("ftp://host/v1", "--base-url must use http:// or https://"),
        (
            "http://host/v1?x=1",
            "--base-url must not contain a query or fragment",
        ),
        (
            "http://host/v1#top",
            "--base-url must not contain a query or fragment",
        ),
        ("http:///v1", "--base-url is missing a host"),
        (
            "http://user:secret@host/v1",
            "--base-url must not embed credentials",
        ),
    ] {
        assert_eq!(validate_base_url(url).unwrap_err(), error, "{url}");
    }
    assert_eq!(normalize_base_url("http://host/v1///"), "http://host/v1");
}
