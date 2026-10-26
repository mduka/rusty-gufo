//! The bounded HTTP/1.1 server for trusted-LAN model serving
//! (`src/cli/serve/http_server.cpp`): one blocking thread per connection up
//! to `max_connections` (503 beyond), exactly one request per connection,
//! bearer authentication on every route except `OPTIONS`, `Content-Length`
//! framing only (`Transfer-Encoding` is rejected), chunked streaming for
//! HTTP/1.1 and close-delimited streaming for HTTP/1.0.

use std::io::{Read, Write};
use std::net::{
    IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, SocketAddrV4, SocketAddrV6, TcpListener,
    TcpStream,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gufo_core::json::{self, Value};
use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    CancellationCheck, ChatMessage, ChatRequest, ChatRole, Error, ErrorKind as TextErrorKind,
    GenerationErrorCode, ReasoningOptions, Result,
};
use gufo_core::utf8::Utf8Decoder;
use rustix::event::{PollFd, PollFlags};
use rustix::net::RecvFlags;
use sha2::{Digest, Sha256};

use super::websocket::{WebSocket, is_websocket_upgrade};
use super::{Handler, HttpRequest, HttpResponse, HttpServerOptions, StreamLog};
use crate::arbiter::ArbiterService;
use crate::asr_service::AsrService;
use crate::audio_asr_api::{handle_audio_asr_api_request, is_audio_asr_api_path};
use crate::audio_tts_api::{handle_audio_tts_api_request, is_audio_tts_api_path};
use crate::audio_websocket::{handle_asr_websocket, handle_tts_websocket};
use crate::backend::{
    FinishReason, GenerationRequest, GenerationResult, PromptProgress, SessionState,
    StreamCallbacks, TextGenerationBackend, start_complete,
};
use crate::embedding_service::EmbeddingService;
use crate::embeddings_api::{handle_embeddings_api_request, is_embeddings_api_path};
use crate::generation_metrics::{
    generation_log_details, generation_timings, prefill_tokens_per_second, prompt_progress_json,
    server_timing,
};
use crate::image_api::{ImageService, handle_image_api_request};
use crate::image_url::ImageReadBudget;
use crate::logging::{self, LogLevel};
use crate::metrics::{prometheus_text, record_server_metrics};
use crate::openai_chat::{
    ApiCompat, ResponsesReasoningTarget, handle_openai_chat, parse_anthropic_image,
    parse_anthropic_tool_controls, parse_anthropic_tool_message, parse_openai_response_format,
    parse_openai_response_message, parse_reasoning_budget, parse_reasoning_effort_name,
    parse_reasoning_options, parse_response_function_call, parse_response_function_output,
    parse_response_tool_controls, sse,
};
use crate::sampling_request::{parse_sampling_config, read_integer};
use crate::stop_sequences::{StopSequenceFormat, parse_stop_sequences};
use crate::tts_service::TtsService;
use crate::util::{random_id, unix_time};
use crate::video_api::{handle_video_api_request, is_video_api_path};
use crate::video_jobs::VideoJobService;

// Before the submodules, which use it.
macro_rules! try_response {
    ($expression:expr) => {
        match $expression {
            Ok(value) => value,
            Err(response) => return Ok(response),
        }
    };
}

mod completions;
mod connection;
mod llama;
mod requests;

use completions::*;
pub use connection::*;
use llama::*;
use requests::*;

const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(120);
const LISTEN_BACKLOG: i32 = 16;

/// The model services a server fronts besides the text backend.
#[derive(Default, Clone)]
pub struct HttpServices {
    /// `gufo arbiter`: serves only the arbiter's routes.
    pub arbiter: Option<Arc<ArbiterService>>,
    pub video_jobs: Option<Arc<VideoJobService>>,
    pub tts: Option<Arc<TtsService>>,
    pub asr: Option<Arc<AsrService>>,
    pub images: Option<Arc<ImageService>>,
    pub embeddings: Option<Arc<EmbeddingService>>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn send_all(stream: &mut TcpStream, data: &[u8]) -> bool {
    stream.write_all(data).is_ok()
}

fn send_chunk(stream: &mut TcpStream, data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }
    send_all(stream, format!("{:x}\r\n", data.len()).as_bytes())
        && send_all(stream, data)
        && send_all(stream, b"\r\n")
}

/// Records SIGINT and SIGTERM in `signal` (a second one exits with
/// `128 + signal`) and makes `listener` non-blocking for the polling loop.
fn watch_shutdown_signals(
    listener: &TcpListener,
    signal: &Arc<AtomicUsize>,
) -> std::io::Result<()> {
    use signal_hook::consts::{SIGINT, SIGTERM};
    let received = Arc::new(AtomicBool::new(false));
    for number in [SIGINT, SIGTERM] {
        // Checked before the flag is set: only a repeated signal exits.
        signal_hook::flag::register_conditional_shutdown(number, 128 + number, received.clone())?;
        signal_hook::flag::register(number, received.clone())?;
        signal_hook::flag::register_usize(number, signal.clone(), number as usize)?;
    }
    listener.set_nonblocking(true)
}

/// Whether the client closed its side: a readable socket with nothing to
/// peek, or an error or hang-up (a non-blocking check).
fn is_peer_disconnected(stream: &TcpStream) -> bool {
    let mut descriptor = [PollFd::new(
        stream,
        PollFlags::IN | PollFlags::ERR | PollFlags::HUP | PollFlags::RDHUP,
    )];
    let zero = rustix::event::Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    match rustix::event::poll(&mut descriptor, Some(&zero)) {
        Ok(ready) if ready > 0 => {}
        _ => return false,
    }
    let events = descriptor[0].revents();
    if events.intersects(PollFlags::ERR | PollFlags::HUP | PollFlags::NVAL | PollFlags::RDHUP) {
        return true;
    }
    if !events.contains(PollFlags::IN) {
        return false;
    }
    let mut byte = [0u8; 1];
    match rustix::net::recv(stream, &mut byte[..], RecvFlags::PEEK | RecvFlags::DONTWAIT) {
        Ok((count, _)) => count == 0,
        Err(errno) => {
            errno != rustix::io::Errno::AGAIN
                && errno != rustix::io::Errno::WOULDBLOCK
                && errno != rustix::io::Errno::INTR
        }
    }
}

/// `Content-Length`: `Some(0)` when absent, `None` for `Transfer-Encoding`,
/// a malformed value or disagreeing duplicates.
fn parse_content_length(request: &HttpRequest) -> Option<usize> {
    let mut length: Option<usize> = None;
    for (name, value) in &request.headers {
        let lowered = name.to_ascii_lowercase();
        // Chunked transfer is not implemented; never interpret its encoded
        // bytes as an empty or partial inference request.
        if lowered == "transfer-encoding" {
            return None;
        }
        if lowered != "content-length" {
            continue;
        }
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let parsed: usize = value.parse().ok()?;
        if length.is_some_and(|l| l != parsed) {
            return None;
        }
        length = Some(parsed);
    }
    Some(length.unwrap_or(0))
}

fn is_authorized(request: &HttpRequest, key_hash: &str) -> bool {
    if key_hash.is_empty() {
        return true;
    }
    let mut credential: Option<&str> = None;
    for (name, value) in &request.headers {
        if name.eq_ignore_ascii_case("authorization") {
            if credential.is_some() {
                return false;
            }
            credential = Some(value);
        }
    }
    let credential = credential.unwrap_or("");
    let Some(space) = credential.find(' ') else {
        return false;
    };
    if !credential[..space].eq_ignore_ascii_case("bearer") {
        return false;
    }
    // Comparing digests does not disclose matching prefixes of the key.
    sha256_hex(credential[space + 1..].trim_start_matches(' ').as_bytes()) == key_hash
}

fn build_response_head(response: &HttpResponse, content_length: Option<usize>) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {} {}\r\n", response.status, response.reason);
    if response.status != 101 && !response.has_header("content-type") {
        out.push_str("Content-Type: application/json\r\n");
    }
    if let Some(length) = content_length {
        out.push_str(&format!("Content-Length: {length}\r\n"));
    }
    out.push_str("Access-Control-Allow-Origin: *\r\n");
    out.push_str("Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n");
    out.push_str("Access-Control-Allow-Headers: Content-Type, Authorization, Range\r\n");
    for (name, value) in &response.headers {
        out.push_str(name);
        out.push_str(": ");
        out.push_str(value);
        out.push_str("\r\n");
    }
    if !response.has_header("connection") {
        out.push_str("Connection: close\r\n");
    }
    out.push_str("\r\n");
    out.into_bytes()
}

fn build_response(response: &HttpResponse) -> Vec<u8> {
    let mut out = build_response_head(response, Some(response.body.len()));
    out.extend_from_slice(&response.body);
    out
}

fn dump(value: &Value) -> Vec<u8> {
    value.dump().unwrap_or_default().into_bytes()
}

fn ok(value: &Value) -> HttpResponse {
    HttpResponse::new(200, "OK", dump(value))
}

/// The error body of this file (`Err`).
fn err(status: u16, reason: &str, message: &str, kind: &str, code: &str) -> HttpResponse {
    let body = Value::object().with(
        "error",
        Value::object()
            .with("message", message)
            .with("type", kind)
            .with("code", code),
    );
    HttpResponse::new(status, reason, dump(&body))
}

fn usage_json(result: &GenerationResult) -> Value {
    let predicted_per_second = if result.decode_ms > 0.0 && result.completion_tokens > 0 {
        result.completion_tokens as f64 / (result.decode_ms / 1000.0)
    } else {
        0.0
    };
    Value::object()
        .with("prompt_tokens", result.prompt_tokens)
        .with("completion_tokens", result.completion_tokens)
        .with(
            "total_tokens",
            result.prompt_tokens + result.completion_tokens,
        )
        .with(
            "prompt_tokens_details",
            Value::object().with("cached_tokens", result.cached_prompt_tokens),
        )
        .with("cached_tokens", result.cached_prompt_tokens)
        .with(
            "prompt_tokens_per_second",
            prefill_tokens_per_second(result),
        )
        .with("completion_tokens_per_second", predicted_per_second)
        .with("draft_tokens", result.draft_tokens)
        .with("draft_tokens_accepted", result.draft_accepted_tokens)
}

fn with_timing(mut response: HttpResponse, result: &GenerationResult) -> HttpResponse {
    record_server_metrics(result);
    response
        .headers
        .push(("Server-Timing".into(), server_timing(result)));
    response.log_details = generation_log_details(result);
    response
}

struct Worker {
    /// A clone of the connection for shutdown; the connection thread drops
    /// it when done, so the socket closes when the response ends.
    stream: Arc<Mutex<Option<TcpStream>>>,
    done: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

struct Shared {
    backend: Option<Arc<dyn TextGenerationBackend>>,
    services: HttpServices,
    options: HttpServerOptions,
    api_key_hash: String,
    routes: RwLock<Vec<(String, String, Handler)>>,
    stopped: AtomicBool,
    workers: Mutex<Vec<Worker>>,
    next_request: AtomicU64,
    /// Runs once when the backend reports a failure only a restart recovers.
    fatal_failure_hook: OnceLock<Box<dyn Fn(GenerationErrorCode) + Send + Sync>>,
    fatal_failure_reported: AtomicBool,
}

/// The backend's permanent failure (a lost device, a failed scheduler) or
/// the embedding service's lost device, if any; the first observation runs
/// the hook. Health, readiness and generation answer 503 with its code from
/// then on.
fn fatal_failure(shared: &Shared) -> Option<GenerationErrorCode> {
    let code = shared
        .backend
        .as_ref()
        .and_then(|b| b.fatal_failure())
        .or_else(|| {
            let embeddings = shared.services.embeddings.as_ref();
            embeddings
                .is_some_and(|e| e.device_lost())
                .then_some(GenerationErrorCode::DeviceLost)
        })?;
    if !shared.fatal_failure_reported.swap(true, Ordering::AcqRel)
        && let Some(hook) = shared.fatal_failure_hook.get()
        && catch_unwind(AssertUnwindSafe(|| hook(code))).is_err()
    {
        logging::error(
            "server",
            &format!(
                "event=fatal_failure_hook_failed code={}",
                code.stable_code()
            ),
        );
    }
    Some(code)
}

/// Health, readiness and generation share one body after a permanent
/// failure: `status` as health and readiness report it, `error` as the API.
fn fatal_failure_response(code: GenerationErrorCode) -> HttpResponse {
    let body = Value::object().with("status", code.stable_code()).with(
        "error",
        Value::object()
            .with(
                "message",
                code.restart_message().unwrap_or("restart required"),
            )
            .with("type", "server_error")
            .with("code", code.stable_code()),
    );
    HttpResponse::new(503, "Service Unavailable", dump(&body))
        .with_header("Content-Type", "application/json")
}

pub struct HttpServer {
    host: String,
    requested_port: i32,
    port: AtomicU16,
    listener: OnceLock<TcpListener>,
    shared: Arc<Shared>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl HttpServer {
    pub fn new(
        host: impl Into<String>,
        port: i32,
        backend: Option<Arc<dyn TextGenerationBackend>>,
        services: HttpServices,
        options: HttpServerOptions,
    ) -> Result<Self> {
        if options.max_request_body_bytes == 0 || options.max_connections == 0 {
            return Err(Error::invalid("HTTP server limits must be positive"));
        }
        let api_key_hash = if options.api_key.is_empty() {
            String::new()
        } else {
            sha256_hex(options.api_key.as_bytes())
        };
        let server = Self {
            host: host.into(),
            requested_port: port,
            port: AtomicU16::new(u16::try_from(port).unwrap_or(0)),
            listener: OnceLock::new(),
            shared: Arc::new(Shared {
                backend,
                services,
                options,
                api_key_hash,
                routes: RwLock::new(Vec::new()),
                stopped: AtomicBool::new(false),
                workers: Mutex::new(Vec::new()),
                next_request: AtomicU64::new(0),
                fatal_failure_hook: OnceLock::new(),
                fatal_failure_reported: AtomicBool::new(false),
            }),
        };
        server.register_routes();
        Ok(server)
    }

    /// Runs `hook` once, on the listener or a request thread, when the
    /// backend reports a failure only a restart recovers (a lost device, a
    /// failed scheduler); it decides how the process exits.
    pub fn on_fatal_failure(&self, hook: impl Fn(GenerationErrorCode) + Send + Sync + 'static) {
        let _ = self.shared.fatal_failure_hook.set(Box::new(hook));
    }

    pub fn add(&self, method: &str, path: &str, handler: Handler) {
        self.shared
            .routes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push((method.into(), path.into(), handler));
    }

    fn add_fn(
        &self,
        method: &str,
        path: &str,
        handler: fn(&HttpRequest, &Arc<dyn TextGenerationBackend>) -> Result<HttpResponse>,
    ) {
        self.add(method, path, Arc::new(handler));
    }

    fn register_routes(&self) {
        if self.shared.backend.is_none() {
            return;
        }
        // OpenAI
        self.add_fn("POST", "/v1/completions", openai_completions);
        self.add_fn("POST", "/v1/chat/completions", handle_openai_chat);
        let semantics = Semantics {
            compat: self.shared.options.api_compat,
            target: self.shared.options.responses_reasoning_target,
        };
        self.add(
            "POST",
            "/v1/responses",
            Arc::new(move |request, backend| openai_responses(request, backend, semantics)),
        );
        self.add_fn("POST", "/v1/embeddings", not_implemented);
        // Anthropic
        let compat = semantics.compat;
        self.add(
            "POST",
            "/v1/messages",
            Arc::new(move |request, backend| anthropic_messages(request, backend, compat)),
        );
        self.add(
            "POST",
            "/v1/messages/count_tokens",
            Arc::new(move |request, backend| anthropic_count_tokens(request, backend, compat)),
        );
        // llama-server
        self.add_fn("POST", "/v1/rerank", not_implemented);
        self.add_fn("POST", "/v1/reranking", not_implemented);
        self.add_fn("POST", "/rerank", not_implemented);
        self.add_fn("POST", "/infill", not_implemented);
        self.add_fn("POST", "/completion", llama_completion);
        self.add_fn("GET", "/props", llama_props);
        self.add_fn("GET", "/slots", llama_slots);
        self.add_fn("GET", "/v1/slots", llama_slots);
        self.add_fn("GET", "/metrics", llama_metrics);
        self.add_fn("GET", "/v1/metrics", llama_metrics);
        // sdapi
        self.add_fn("POST", "/sdapi/v1/txt2img", not_implemented);
        self.add_fn("POST", "/sdapi/v1/img2img", not_implemented);
        self.add_fn("GET", "/sdapi/v1/loras", not_implemented);
    }

    /// Binds and listens (backlog 16).
    pub fn start(&self) -> std::result::Result<(), String> {
        if self.listener.get().is_some() || self.shared.stopped.load(Ordering::Acquire) {
            return Err("HTTP server has already been started or stopped".into());
        }
        let address = u16::try_from(self.requested_port)
            .ok()
            .and_then(|port| parse_listen_address(&self.host, port));
        let Some(address) = address else {
            return Err(
                "host must be an IPv4 or IPv6 address and port must be between 0 and 65535".into(),
            );
        };
        let listener = bind_listener(&address).map_err(|step| match step {
            "bind()" => format!("bind() failed on {}:{}", self.host, self.requested_port),
            step => format!("{step} failed"),
        })?;
        let bound = listener
            .local_addr()
            .map_err(|_| "getsockname() failed".to_owned())?;
        self.port.store(bound.port(), Ordering::Release);
        self.listener
            .set(listener)
            .map_err(|_| "HTTP server has already been started or stopped".to_owned())
    }

    pub fn port(&self) -> u16 {
        self.port.load(Ordering::Acquire)
    }

    /// The blocking accept loop, until `stop`. With `handle_signals` (the
    /// CLI), SIGINT or SIGTERM stops the server and cancels active requests,
    /// so dropping the backend afterwards drains accepted disk writes; the
    /// handlers stay for the rest of the process, and a second signal exits
    /// at once. Embedded callers keep their own handlers and call `stop`.
    pub fn run(&self, handle_signals: bool) {
        let Some(listener) = self.listener.get() else {
            return;
        };
        let signal = Arc::new(AtomicUsize::new(0));
        if handle_signals && let Err(error) = watch_shutdown_signals(listener, &signal) {
            logging::error(
                "server",
                &format!("event=signal_setup_failed error=\"{error}\""),
            );
            return;
        }
        // Polled even without signal handling, so a lost device is seen while
        // a request worker blocks writing to a slow client; non-blocking, so
        // a connection that vanishes between poll and accept cannot block.
        if let Err(error) = listener.set_nonblocking(true) {
            logging::error(
                "server",
                &format!("event=listener_setup_failed error=\"{error}\""),
            );
            return;
        }
        let options = &self.shared.options;
        logging::info(
            "server",
            &format!(
                "event=listening address=http://{}:{} auth={} max_connections={} max_body_bytes={}",
                url_host(&self.host),
                self.port(),
                if options.api_key.is_empty() {
                    "off"
                } else {
                    "bearer"
                },
                options.max_connections,
                options.max_request_body_bytes
            ),
        );
        while !self.shared.stopped.load(Ordering::Acquire) {
            fatal_failure(&self.shared);
            if handle_signals {
                let received = signal.load(Ordering::Relaxed);
                if received != 0 {
                    logging::info(
                        "server",
                        &format!("event=shutdown_requested signal={received}"),
                    );
                    self.stop();
                    break;
                }
            }
            let mut descriptor = [PollFd::new(listener, PollFlags::IN)];
            let tick = rustix::event::Timespec {
                tv_sec: 0,
                tv_nsec: 100_000_000,
            };
            if !matches!(rustix::event::poll(&mut descriptor, Some(&tick)), Ok(ready) if ready > 0)
            {
                continue;
            }
            let stream = match listener.accept() {
                Ok((stream, _)) if stream.set_nonblocking(false).is_ok() => stream,
                Ok(_) => continue,
                Err(_) => {
                    if self.shared.stopped.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
            };
            if self.shared.stopped.load(Ordering::Acquire) {
                break;
            }
            self.reap_workers();
            self.dispatch(stream);
        }
        self.reap_workers();
    }

    fn dispatch(&self, mut stream: TcpStream) {
        let mut workers = lock(&self.shared.workers);
        let overloaded = self.shared.stopped.load(Ordering::Acquire)
            || workers.len() >= self.shared.options.max_connections;
        let clone = if overloaded {
            None
        } else {
            stream.try_clone().ok()
        };
        let Some(clone) = clone else {
            drop(workers);
            let response = err(
                503,
                "Service Unavailable",
                "connection limit reached",
                "server_error",
                "overloaded",
            )
            .with_header("Retry-After", "1");
            let _ = stream.write_all(&build_response(&response));
            return;
        };
        let handle = Arc::new(Mutex::new(Some(clone)));
        let done = Arc::new(AtomicBool::new(false));
        let shared = self.shared.clone();
        let thread_handle = handle.clone();
        let thread_done = done.clone();
        let spawned = std::thread::Builder::new()
            .name("gufo-http".into())
            .spawn(move || {
                handle_connection(&shared, stream);
                lock(&thread_handle).take();
                thread_done.store(true, Ordering::Release);
            });
        if let Ok(thread) = spawned {
            workers.push(Worker {
                stream: handle,
                done,
                thread,
            });
        }
    }

    pub fn stop(&self) {
        let was_stopped = self.shared.stopped.swap(true, Ordering::AcqRel);
        if !was_stopped && let Some(listener) = self.listener.get() {
            // Keep the listener owned until drop: `run` may still be inside
            // accept.
            let _ = rustix::net::shutdown(listener, rustix::net::Shutdown::Both);
        }
        let workers: Vec<Worker> = std::mem::take(&mut *lock(&self.shared.workers));
        for worker in &workers {
            if let Some(stream) = lock(&worker.stream).as_ref() {
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
        for worker in workers {
            let _ = worker.thread.join();
        }
    }

    fn reap_workers(&self) {
        let finished: Vec<Worker> = {
            let mut workers = lock(&self.shared.workers);
            let (finished, running) = std::mem::take(&mut *workers)
                .into_iter()
                .partition(|w| w.done.load(Ordering::Acquire));
            *workers = running;
            finished
        };
        for worker in finished {
            let _ = worker.thread.join();
        }
    }
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn service_unavailable(message: &str, code: &str) -> HttpResponse {
    err(503, "Service Unavailable", message, "server_error", code)
}

fn handle_request(shared: &Shared, request: &HttpRequest) -> Result<HttpResponse> {
    if !is_authorized(request, &shared.api_key_hash) {
        return Ok(err(
            401,
            "Unauthorized",
            "missing or invalid API key",
            "authentication_error",
            "invalid_api_key",
        )
        .with_header("WWW-Authenticate", "Bearer"));
    }
    let services = &shared.services;
    let backend_ready = shared.backend.as_ref().is_some_and(|b| b.ready());
    let method = request.method.as_str();
    let path = request.path.as_str();
    // `/health` and `/ready` are native and match llama-server; `/healthz`
    // and `/readyz` serve Kubernetes-style probes unmodified.
    if method == "GET" && matches!(path, "/health" | "/v1/health" | "/healthz") {
        if let Some(code) = fatal_failure(shared) {
            return Ok(fatal_failure_response(code));
        }
        return Ok(ok(&Value::object().with("status", "ok")));
    }
    if let Some(arbiter) = &services.arbiter {
        return Ok(arbiter.handle(request));
    }
    if method == "GET" && matches!(path, "/ready" | "/v1/ready" | "/readyz") {
        if let Some(code) = fatal_failure(shared) {
            return Ok(fatal_failure_response(code));
        }
        let tts_ready = services.tts.as_ref().is_some_and(|t| t.ready());
        let asr_ready = services.asr.as_ref().is_some_and(|a| a.ready());
        let video_ready = services.video_jobs.as_ref().is_some_and(|v| v.ready());
        let embeddings_ready = services.embeddings.as_ref().is_some_and(|e| e.ready());
        if !(backend_ready
            || video_ready
            || tts_ready
            || asr_ready
            || embeddings_ready
            || services.images.is_some())
        {
            return Ok(service_unavailable(
                "model service is not ready",
                "not_ready",
            ));
        }
        let model = match (
            &shared.backend,
            &services.asr,
            &services.tts,
            &services.images,
        ) {
            (Some(backend), _, _, _) if backend_ready => backend.model_id(),
            (_, Some(asr), _, _) if asr_ready => asr.model_id(),
            (_, _, Some(tts), _) if tts_ready => tts.model_id(),
            _ if embeddings_ready => services
                .embeddings
                .as_ref()
                .map_or_else(String::new, |e| e.model_id()),
            (_, _, _, Some(images)) => images.model_id().to_owned(),
            _ => "minimax-h3".to_owned(),
        };
        return Ok(ok(&Value::object()
            .with("status", "ready")
            .with("model", model)));
    }
    if method == "GET" && matches!(path, "/v1/models" | "/models") {
        return Ok(list_models(shared.backend.as_ref(), services));
    }
    if matches!(path, "/v1/images/generations" | "/v1/images/edits") {
        return match &services.images {
            Some(images) => handle_image_api_request(request, images),
            None => Ok(service_unavailable(
                "image service is not configured",
                "image_service_unavailable",
            )),
        };
    }
    if path == "/v1/audio/speech/stream" {
        return match services.tts.as_ref().filter(|t| t.ready()) {
            Some(tts) => handle_tts_websocket(request, tts),
            None => Ok(service_unavailable(
                "TTS is not configured",
                "tts_service_unavailable",
            )),
        };
    }
    if path == "/v1/realtime" {
        return match services.asr.as_ref().filter(|a| a.ready()) {
            Some(asr) => handle_asr_websocket(request, asr),
            None => Ok(service_unavailable(
                "ASR is not configured",
                "asr_service_unavailable",
            )),
        };
    }
    if is_video_api_path(path) {
        return match services.video_jobs.as_ref().filter(|v| v.ready()) {
            Some(video) => handle_video_api_request(request, video),
            None => Ok(service_unavailable(
                "MiniMax H3 video service is not configured",
                "video_service_unavailable",
            )),
        };
    }
    if is_audio_tts_api_path(path) {
        return match services.tts.as_ref().filter(|t| t.ready()) {
            Some(tts) => handle_audio_tts_api_request(request, tts),
            None => Ok(service_unavailable(
                "Qwen3-TTS audio service is not configured",
                "tts_service_unavailable",
            )),
        };
    }
    if is_audio_asr_api_path(path) {
        return match services.asr.as_ref().filter(|a| a.ready()) {
            Some(asr) => handle_audio_asr_api_request(request, asr),
            None => Ok(service_unavailable(
                "Qwen3-ASR service is not configured",
                "asr_service_unavailable",
            )),
        };
    }
    // Handled before the route table. An embedding-only server has no text
    // backend, and the stub route must not shadow the service when a backend
    // exists.
    if is_embeddings_api_path(path)
        && let Some(embeddings) = &services.embeddings
    {
        if let Some(code) = fatal_failure(shared) {
            return Ok(fatal_failure_response(code));
        }
        if !embeddings.ready() {
            return Ok(service_unavailable(
                "EmbeddingGemma 2 service is not ready",
                "embedding_service_unavailable",
            ));
        }
        let response = handle_embeddings_api_request(request, embeddings)?;
        // The request that finds the device lost runs the shutdown hook.
        if let Some(code) = fatal_failure(shared) {
            return Ok(fatal_failure_response(code));
        }
        return Ok(response);
    }
    if let Some(backend) = &shared.backend {
        let handler = shared
            .routes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|(m, p, _)| m == method && p == path)
            .map(|(_, _, handler)| handler.clone());
        if let Some(handler) = handler {
            // Unimplemented stubs too: nothing reaches the failed backend.
            if method == "POST"
                && let Some(code) = fatal_failure(shared)
            {
                return Ok(fatal_failure_response(code));
            }
            return handler(request, backend);
        }
    }
    Ok(err(
        404,
        "Not Found",
        "no route for this path",
        "invalid_request_error",
        "not_found",
    ))
}

/// A listening socket on `address` (backlog 16), or the step that failed.
fn bind_listener(address: &SocketAddr) -> std::result::Result<TcpListener, &'static str> {
    use rustix::net::{AddressFamily, SocketType};
    let family = if address.is_ipv6() {
        AddressFamily::INET6
    } else {
        AddressFamily::INET
    };
    let socket = rustix::net::socket(family, SocketType::STREAM, None).map_err(|_| "socket()")?;
    let _ = rustix::net::sockopt::set_socket_reuseaddr(&socket, true);
    // `::` serves IPv4 clients too. Set it explicitly: the kernel default
    // follows net.ipv6.bindv6only, which differs between hosts.
    if address.is_ipv6() {
        rustix::net::sockopt::set_ipv6_v6only(&socket, false)
            .map_err(|_| "setsockopt(IPV6_V6ONLY)")?;
    }
    rustix::net::bind(&socket, address).map_err(|_| "bind()")?;
    rustix::net::listen(&socket, LISTEN_BACKLOG).map_err(|_| "listen()")?;
    Ok(TcpListener::from(socket))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with(headers: &[(&str, &str)]) -> HttpRequest {
        HttpRequest {
            headers: headers
                .iter()
                .map(|(n, v)| ((*n).to_owned(), (*v).to_owned()))
                .collect(),
            ..HttpRequest::default()
        }
    }

    #[test]
    fn content_length_framing() {
        assert_eq!(parse_content_length(&request_with(&[])), Some(0));
        assert_eq!(
            parse_content_length(&request_with(&[("Content-Length", "12")])),
            Some(12)
        );
        for bad in ["nope", "-1", "+1", "4junk", "18446744073709551616", ""] {
            assert_eq!(
                parse_content_length(&request_with(&[("Content-Length", bad)])),
                None,
                "{bad}"
            );
        }
        assert_eq!(
            parse_content_length(&request_with(&[
                ("Content-Length", "4"),
                ("content-length", "3")
            ])),
            None
        );
        assert_eq!(
            parse_content_length(&request_with(&[("Transfer-Encoding", "chunked")])),
            None
        );
    }
}
