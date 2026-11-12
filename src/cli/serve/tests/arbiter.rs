//! The arbiter end to end on loopback: servers connect over its WebSocket,
//! a streamed TTS whose playback buffer runs low throttles text
//! generation, a full buffer, client pressure, manual shares and a lost
//! arbiter change the grant as the rules say.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gufo_core::json::{self, Value};
use gufo_serve::arbiter::{
    ArbiterClient, ArbiterConfig, ArbiterService, Identity, PlaybackBuffer, Presence,
};
use gufo_serve::gpu_share::GpuShare;
use gufo_serve::http::HttpServerOptions;
use gufo_serve::http::server::{HttpServer, HttpServices};

struct Arbiter {
    server: Arc<HttpServer>,
    thread: Option<JoinHandle<()>>,
    address: String,
}

impl Arbiter {
    fn start(config: ArbiterConfig) -> Self {
        let services = HttpServices {
            arbiter: Some(ArbiterService::start(config).expect("arbiter")),
            ..HttpServices::default()
        };
        let server = Arc::new(
            HttpServer::new("127.0.0.1", 0, None, services, HttpServerOptions::default())
                .expect("server"),
        );
        server.start().expect("start");
        let address = format!("127.0.0.1:{}", server.port());
        let running = server.clone();
        let thread = std::thread::spawn(move || running.run(false));
        Self {
            server,
            thread: Some(thread),
            address,
        }
    }

    fn stop(&mut self) {
        self.server.stop();
        if let Some(thread) = self.thread.take() {
            thread.join().expect("server thread");
        }
    }

    fn request(&self, method: &str, path: &str, body: &str) -> (u16, String) {
        let mut stream = TcpStream::connect(&self.address).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("timeout");
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            self.address,
            body.len()
        );
        stream.write_all(request.as_bytes()).expect("send");
        let mut response = Vec::new();
        let _ = stream.read_to_end(&mut response);
        let text = String::from_utf8_lossy(&response).into_owned();
        let status = text
            .split(' ')
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let body = text
            .split_once("\r\n\r\n")
            .map_or(String::new(), |(_, b)| b.to_owned());
        (status, body)
    }

    fn instances(&self) -> Value {
        let (status, body) = self.request("GET", "/v1/instances", "");
        assert_eq!(status, 200, "{body}");
        json::parse_str(&body).expect("instances json")
    }
}

impl Drop for Arbiter {
    fn drop(&mut self) {
        self.stop();
    }
}

fn identity(modality: &str, model: &str, port: u16) -> Identity {
    Identity {
        modality: modality.into(),
        model: model.into(),
        address: format!("127.0.0.1:{port}"),
        pid: u32::from(port),
    }
}

fn wait_until(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn connected(arbiter: &Arbiter) -> usize {
    arbiter
        .instances()
        .find("instances")
        .map_or(0, |i| i.items().len())
}

#[test]
fn a_streamed_tts_behind_throttles_text_generation() {
    let mut arbiter = Arbiter::start(ArbiterConfig::default());
    let share = Arc::new(GpuShare::default());
    let _llm = ArbiterClient::start(
        arbiter.address.clone(),
        identity("llm", "text", 1),
        share.clone(),
        Some(share.clone()),
    )
    .expect("llm client");
    let playback = Arc::new(PlaybackBuffer::default());
    let _tts = ArbiterClient::start(
        arbiter.address.clone(),
        identity("tts", "speech", 2),
        playback.clone(),
        None,
    )
    .expect("tts client");
    wait_until("two instances", || connected(&arbiter) == 2);
    assert_eq!(share.current(), 1.0, "no pressure without synthesis");

    // A synthesis starts with an empty buffer: full pressure, the least share.
    playback.begin();
    wait_until("the least share", || (share.current() - 0.25).abs() < 1e-3);
    // Ten buffered seconds are far above the target: the grant ends.
    playback.add(10.0, Instant::now());
    wait_until("the full share", || share.current() == 1.0);
    playback.end();

    // Pressure from any client, and a manual share, apply the same way.
    let pressure = r#"{"source": "assistant", "pressure": 1.0, "ttl_ms": 60000}"#;
    assert_eq!(arbiter.request("POST", "/v1/pressure", pressure).0, 200);
    wait_until("client pressure", || (share.current() - 0.25).abs() < 1e-3);
    let released = r#"{"source": "assistant", "pressure": 0.0, "ttl_ms": 60000}"#;
    assert_eq!(arbiter.request("POST", "/v1/pressure", released).0, 200);
    wait_until("released pressure", || share.current() == 1.0);
    let path = "/v1/instances/llm:text@127.0.0.1:1/share";
    let manual = r#"{"share": 0.5, "ttl_ms": 60000}"#;
    assert_eq!(arbiter.request("POST", path, manual).0, 200);
    wait_until("the manual share", || (share.current() - 0.5).abs() < 1e-3);
    let metrics = arbiter.request("GET", "/metrics", "").1;
    assert!(
        metrics.contains(
            "gufo_arbiter_share{instance=\"llm:text@127.0.0.1:1\",modality=\"llm\",model=\"text\"} 0.5"
        ),
        "{metrics}"
    );
    assert_eq!(arbiter.request("POST", path, r#"{"share": null}"#).0, 200);
    wait_until("no manual share", || share.current() == 1.0);

    // A grant never outlives its arbiter.
    assert_eq!(arbiter.request("POST", "/v1/pressure", pressure).0, 200);
    wait_until("pressure again", || share.current() < 1.0);
    arbiter.stop();
    wait_until("the release after the arbiter stopped", || {
        share.current() == 1.0
    });
}

#[test]
fn per_instance_rules_and_unique_ids() {
    let config = ArbiterConfig::parse(
        "[[instances]]\nmodality = \"llm\"\nmodel = \"big\"\nmin_share = 0.5\n",
    )
    .expect("config");
    let arbiter = Arbiter::start(config);
    let big = Arc::new(GpuShare::default());
    let small = Arc::new(GpuShare::default());
    let _big = ArbiterClient::start(
        arbiter.address.clone(),
        identity("llm", "big", 3),
        big.clone(),
        Some(big.clone()),
    )
    .expect("big");
    let _small = ArbiterClient::start(
        arbiter.address.clone(),
        identity("llm", "small", 4),
        small.clone(),
        Some(small.clone()),
    )
    .expect("small");
    wait_until("both servers", || connected(&arbiter) == 2);
    // The same identity a second time is refused; the first keeps its place.
    let _twin = ArbiterClient::start(
        arbiter.address.clone(),
        identity("llm", "small", 4),
        Arc::new(Presence),
        None,
    )
    .expect("twin");
    let pressure = r#"{"source": "test", "pressure": 1.0, "ttl_ms": 60000}"#;
    assert_eq!(arbiter.request("POST", "/v1/pressure", pressure).0, 200);
    wait_until("the rule's share", || (big.current() - 0.5).abs() < 1e-3);
    wait_until("the default share", || {
        (small.current() - 0.25).abs() < 1e-3
    });
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(connected(&arbiter), 2, "the twin never registered");
}

#[test]
fn routes_report_and_refuse() {
    let arbiter = Arbiter::start(ArbiterConfig::default());
    let (status, body) = arbiter.request("GET", "/v1/config", "");
    assert_eq!(status, 200);
    let config = json::parse_str(&body).expect("config json");
    assert_eq!(
        config
            .find("tts")
            .expect("tts")
            .member_double("buffer_target_s", 0.0),
        2.5
    );
    assert_eq!(arbiter.request("GET", "/health", "").0, 200);
    assert_eq!(arbiter.request("GET", "/v1/models", "").0, 404);
    for body in [
        "[]",
        r#"{"source": "x", "pressure": 2, "ttl_ms": 100}"#,
        r#"{"source": "", "pressure": 1, "ttl_ms": 100}"#,
        r#"{"source": "x", "pressure": 1, "ttl_ms": 0}"#,
    ] {
        assert_eq!(
            arbiter.request("POST", "/v1/pressure", body).0,
            400,
            "{body}"
        );
    }
    let unknown = arbiter.request(
        "POST",
        "/v1/instances/nobody/share",
        r#"{"share": 0.5, "ttl_ms": 100}"#,
    );
    assert_eq!(unknown.0, 404, "{}", unknown.1);
}
