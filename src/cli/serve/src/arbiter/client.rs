//! A server's connection to the arbiter (`--arbiter`): one thread that
//! connects, introduces the server, reports its status and applies the
//! arbiter's grants, reconnecting with backoff when the arbiter is away.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gufo_core::json::{self, Value};

use super::playback::PlaybackBuffer;
use super::websocket_client::WebSocketClient;
use super::{CONNECT_PATH, Identity};
use crate::gpu_share::GpuShare;
use crate::logging;

/// How long a read waits; also the finest reporting interval.
const POLL: Duration = Duration::from_millis(50);
/// Status interval while a server is busy (a streamed synthesis runs).
const BUSY_INTERVAL: Duration = Duration::from_millis(50);
const IDLE_INTERVAL: Duration = Duration::from_millis(250);
const FIRST_BACKOFF: Duration = Duration::from_millis(500);
const LAST_BACKOFF: Duration = Duration::from_secs(5);

/// What a server reports in `status`.
pub trait StatusSource: Send + Sync {
    /// The report's fields besides `type`, as an object.
    fn status(&self) -> Value;
    /// Whether to report at the busy interval.
    fn busy(&self) -> bool {
        false
    }
}

/// Text generation reports its grant and the time it yielded.
impl StatusSource for GpuShare {
    fn status(&self) -> Value {
        Value::object()
            .with("share", f64::from(self.current()))
            .with("yielded_s", self.yielded().as_secs_f64())
    }
}

/// Streamed TTS reports its playback buffer.
impl StatusSource for PlaybackBuffer {
    fn status(&self) -> Value {
        let snapshot = self.snapshot(Instant::now());
        Value::object()
            .with("active", snapshot.active)
            .with("buffer_s", snapshot.buffer_s)
            .with("audio_s", snapshot.audio_s)
    }

    fn busy(&self) -> bool {
        self.snapshot(Instant::now()).active
    }
}

/// A server that only shows up in the arbiter's view.
pub struct Presence;

impl StatusSource for Presence {
    fn status(&self) -> Value {
        Value::object()
    }
}

pub struct ArbiterClient {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ArbiterClient {
    /// Connects to the arbiter at `address` in the background. `share`
    /// receives the arbiter's grants; servers without one only report.
    pub fn start(
        address: String,
        identity: Identity,
        source: Arc<dyn StatusSource>,
        share: Option<Arc<GpuShare>>,
    ) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let thread = std::thread::Builder::new()
            .name("gufo-arbiter".into())
            .spawn(move || {
                Connection {
                    address,
                    identity,
                    source,
                    share,
                    stop: thread_stop,
                }
                .run();
            })?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for ArbiterClient {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Connection {
    address: String,
    identity: Identity,
    source: Arc<dyn StatusSource>,
    share: Option<Arc<GpuShare>>,
    stop: Arc<AtomicBool>,
}

impl Connection {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn run(&self) {
        let mut backoff = FIRST_BACKOFF;
        let mut reported = false;
        while !self.stopped() {
            match self.session() {
                Ok(()) => return,
                Err((message, connected)) => {
                    // A grant must not outlive its arbiter.
                    if let Some(share) = &self.share {
                        share.release();
                    }
                    if connected {
                        logging::warn(
                            "arbiter",
                            &format!("lost the arbiter at {}: {message}", self.address),
                        );
                        backoff = FIRST_BACKOFF;
                        reported = false;
                    } else if !reported {
                        logging::warn(
                            "arbiter",
                            &format!(
                                "arbiter at {} unavailable: {message}; retrying",
                                self.address
                            ),
                        );
                        reported = true;
                    }
                }
            }
            let resume = Instant::now() + backoff;
            while !self.stopped() && Instant::now() < resume {
                std::thread::sleep(POLL);
            }
            backoff = (backoff * 2).min(LAST_BACKOFF);
        }
    }

    /// One connection until it fails (`Err` with whether it got as far as
    /// the welcome) or the client stops (`Ok`).
    fn session(&self) -> Result<(), (String, bool)> {
        let mut socket =
            WebSocketClient::connect(&self.address, CONNECT_PATH, POLL).map_err(|e| (e, false))?;
        let hello = self
            .identity
            .hello()
            .dump()
            .map_err(|e| (e.to_string(), false))?;
        socket.send_text(&hello).map_err(|e| (e, false))?;
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match socket.receive().map_err(|e| (e, false))? {
                Some(text) => {
                    let message = json::parse_str(&text).map_err(|e| (e.to_string(), false))?;
                    match message.member_str("type", "").as_str() {
                        "welcome" => break,
                        "error" => return Err((message.member_str("message", ""), false)),
                        _ => {}
                    }
                }
                None if Instant::now() >= deadline => {
                    return Err(("no welcome from the arbiter".into(), false));
                }
                None => {}
            }
        }
        logging::info(
            "arbiter",
            &format!(
                "connected to the arbiter at {} as {}",
                self.address,
                self.identity.id()
            ),
        );
        let mut next_status = Instant::now();
        let mut was_busy = false;
        while !self.stopped() {
            // A server that turns busy reports at once: the arbiter reacts
            // to the start of a stream within a poll, not an idle interval.
            let busy = self.source.busy();
            if Instant::now() >= next_status || busy != was_busy {
                was_busy = busy;
                let mut status = self.source.status();
                status.set("type", "status");
                let text = status.dump().map_err(|e| (e.to_string(), true))?;
                socket.send_text(&text).map_err(|e| (e, true))?;
                next_status = Instant::now() + if busy { BUSY_INTERVAL } else { IDLE_INTERVAL };
            }
            if let Some(text) = socket.receive().map_err(|e| (e, true))? {
                self.apply(&text).map_err(|e| (e, true))?;
            }
        }
        socket.close();
        Ok(())
    }

    fn apply(&self, text: &str) -> Result<(), String> {
        let message = json::parse_str(text).map_err(|e| e.to_string())?;
        match message.member_str("type", "").as_str() {
            "share" => {
                let Some(share) = &self.share else {
                    return Ok(());
                };
                let value = message.member_double("share", 1.0) as f32;
                let ttl =
                    Duration::from_millis(message.member_double("ttl_ms", 0.0).max(0.0) as u64);
                if value >= 1.0 || ttl.is_zero() {
                    share.release();
                } else {
                    share.grant(value, ttl);
                }
                Ok(())
            }
            "error" => Err(message.member_str("message", "arbiter error")),
            _ => Ok(()),
        }
    }
}
