#![allow(dead_code)]

//! A running server over a fake text backend (`RunningServer` and
//! `FakeBackend` of `tests/cli/http_server_test.cpp`).

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    CancellationCheck, ChatRequest, DEVICE_LOST_MESSAGE, Error, GenerationErrorCode,
    ReasoningOptions, Result,
};
use gufo_serve::backend::InitialOutputState;
use gufo_serve::backend::{
    FinishReason, GenerationRequest, GenerationResult, PromptProgress, SamplingDefaults,
    SessionState, StreamCallbacks, TextGenerationBackend, TokenCallback, ToolFormat,
};
use gufo_serve::http::server::{HttpServer, HttpServices};
use gufo_serve::http::{HttpResponse, HttpServerOptions, StreamLog};

use crate::common::Semaphore;

#[derive(Clone, Default)]
pub struct Call {
    pub prompt: String,
    pub chat: ChatRequest,
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    pub client_id: String,
    pub stop_sequences: Vec<String>,
}

pub struct FakeBackend {
    pub calls: AtomicI32,
    pub failure: AtomicI32,
    pub wait_for_disconnect: AtomicBool,
    pub disconnected: AtomicBool,
    pub entered: Semaphore,
    pub finished: Semaphore,
    /// Reports a `StopSequence` finish on this sequence when nonempty.
    pub forced_stop_sequence: Mutex<String>,
    pub reasoning: Mutex<ReasoningOptions>,
    pub defaults: Mutex<SamplingDefaults>,
    pub last_ignore_eos: AtomicBool,
    /// `cache_prompt` of the last raw completion.
    pub last_cache_prompt: AtomicBool,
    pub sessions: Mutex<Vec<SessionState>>,
    /// Prompt progress that started requests report before generating.
    pub progress: Mutex<Vec<PromptProgress>>,
    pub image_support: AtomicBool,
    /// What `/props` reports as `model_path` and `build_info`.
    pub served_path: Mutex<String>,
    pub served_build: Mutex<String>,
    pub lost: AtomicBool,
    /// Reports a panicked scheduler ([`GenerationErrorCode::SchedulerFailed`]).
    pub scheduler_failed: AtomicBool,
    /// Started requests report an admission first, as a scheduler does.
    pub admit: AtomicBool,
    pub token_delay: Mutex<Duration>,
    /// Streams the output as these pieces when nonempty.
    pub pieces: Mutex<Vec<Vec<u8>>>,
    /// The tool format chat requests report as admitted.
    pub tool_format: Mutex<Option<ToolFormat>>,
    /// What `count_chat_tokens` reports; `None` cannot count.
    pub chat_token_count: Mutex<Option<usize>>,
    /// The messages and tools of the last counted request.
    pub counted: Mutex<(usize, usize)>,
    me: Weak<FakeBackend>,
    last: Mutex<Call>,
    output: Mutex<Vec<u8>>,
    initial: Mutex<InitialOutputState>,
}

impl FakeBackend {
    fn new(me: Weak<FakeBackend>) -> Self {
        Self {
            calls: AtomicI32::new(0),
            failure: AtomicI32::new(0),
            wait_for_disconnect: AtomicBool::new(false),
            disconnected: AtomicBool::new(false),
            entered: Semaphore::default(),
            finished: Semaphore::default(),
            forced_stop_sequence: Mutex::new(String::new()),
            reasoning: Mutex::new(ReasoningOptions::default()),
            defaults: Mutex::new(SamplingDefaults::default()),
            last_ignore_eos: AtomicBool::new(false),
            last_cache_prompt: AtomicBool::new(true),
            sessions: Mutex::new(Vec::new()),
            progress: Mutex::new(Vec::new()),
            image_support: AtomicBool::new(false),
            served_path: Mutex::new(String::new()),
            served_build: Mutex::new(String::new()),
            lost: AtomicBool::new(false),
            scheduler_failed: AtomicBool::new(false),
            admit: AtomicBool::new(false),
            token_delay: Mutex::new(Duration::ZERO),
            pieces: Mutex::new(Vec::new()),
            tool_format: Mutex::new(None),
            chat_token_count: Mutex::new(None),
            counted: Mutex::new((0, 0)),
            me,
            last: Mutex::new(Call::default()),
            output: Mutex::new(b"ok".to_vec()),
            initial: Mutex::new(InitialOutputState::Auto),
        }
    }

    pub fn last_call(&self) -> Call {
        self.last.lock().expect("last").clone()
    }

    pub fn set_output(&self, text: Vec<u8>) {
        *self.output.lock().expect("output") = text;
    }

    /// Whether chat output starts inside a reasoning block, as after a
    /// template that opens `<think>`.
    pub fn set_initial_output_state(&self, state: InitialOutputState) {
        *self.initial.lock().expect("initial") = state;
    }
}

type Deferred = Box<
    dyn for<'a> FnOnce(&FakeBackend, Option<TokenCallback<'a>>) -> Result<GenerationResult> + Send,
>;

impl FakeBackend {
    /// A request reporting the configured admission and prompt progress
    /// before `run`, and `tool_format`; without any, the default deferred
    /// request.
    fn with_progress(
        &self,
        run: Deferred,
        tool_format: Option<ToolFormat>,
    ) -> Option<Box<dyn GenerationRequest>> {
        let progress = self.progress.lock().expect("progress").clone();
        let admit = self.admit.load(Ordering::SeqCst);
        if progress.is_empty() && !admit && tool_format.is_none() {
            return None;
        }
        Some(Box::new(ProgressRequest {
            backend: self.me.upgrade().expect("backend"),
            progress,
            admit,
            run: Some(run),
            tool_format,
        }))
    }
}

/// Reports an optional admission and fixed prompt progress, then generates.
struct ProgressRequest {
    backend: Arc<FakeBackend>,
    progress: Vec<PromptProgress>,
    admit: bool,
    run: Option<Deferred>,
    tool_format: Option<ToolFormat>,
}

impl GenerationRequest for ProgressRequest {
    fn wait(&mut self, on_token: Option<TokenCallback<'_>>) -> Result<GenerationResult> {
        self.wait_streaming(StreamCallbacks {
            on_token,
            ..StreamCallbacks::default()
        })
    }

    fn wait_streaming(&mut self, callbacks: StreamCallbacks<'_>) -> Result<GenerationResult> {
        let StreamCallbacks {
            on_token,
            mut on_progress,
            on_start,
        } = callbacks;
        let mut cancelled = false;
        if self.admit
            && let Some(on_start) = on_start
        {
            cancelled = !on_start();
        }
        for progress in &self.progress {
            if let Some(callback) = on_progress.as_mut()
                && !callback(progress)
            {
                cancelled = true;
                break;
            }
        }
        let run = self
            .run
            .take()
            .ok_or_else(|| Error::logic("request consumed"))?;
        let mut result = run(&self.backend, on_token)?;
        result.cancelled |= cancelled;
        Ok(result)
    }

    fn cancel(&self) {}

    fn tool_format(&self) -> Option<ToolFormat> {
        self.tool_format
    }
}

impl TextGenerationBackend for FakeBackend {
    fn model_id(&self) -> String {
        "test".into()
    }

    fn ready(&self) -> bool {
        true
    }

    fn count_tokens(&self, text: &str) -> usize {
        text.len()
    }

    fn count_chat_tokens(&self, request: &ChatRequest) -> Result<Option<usize>> {
        *self.counted.lock().expect("counted") = (request.messages.len(), request.tools.len());
        Ok(*self.chat_token_count.lock().expect("count"))
    }

    fn reasoning_defaults(&self) -> ReasoningOptions {
        *self.reasoning.lock().expect("reasoning")
    }

    fn max_context(&self) -> u32 {
        65536
    }

    fn session_states(&self) -> Vec<SessionState> {
        self.sessions.lock().expect("sessions").clone()
    }

    fn supports_images(&self) -> bool {
        self.image_support.load(Ordering::SeqCst)
    }

    fn model_path(&self) -> String {
        self.served_path.lock().expect("path").clone()
    }

    fn build_info(&self) -> String {
        self.served_build.lock().expect("build").clone()
    }

    fn device_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    fn fatal_failure(&self) -> Option<GenerationErrorCode> {
        if self.scheduler_failed.load(Ordering::SeqCst) {
            Some(GenerationErrorCode::SchedulerFailed)
        } else {
            self.device_lost()
                .then_some(GenerationErrorCode::DeviceLost)
        }
    }

    fn sampling_defaults(&self) -> SamplingDefaults {
        self.defaults.lock().expect("defaults").clone()
    }

    /// Records `ignore_eos` and runs the default deferred completion, after
    /// any configured prompt progress.
    fn start_complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        _: Option<CancellationCheck>,
        _: bool,
        ignore_eos: bool,
        client_id: &str,
        stop_sequences: &[String],
        _: bool,
        cache_prompt: bool,
    ) -> Result<Option<Box<dyn GenerationRequest>>> {
        self.last_ignore_eos.store(ignore_eos, Ordering::SeqCst);
        let (prompt, sampling) = (prompt.to_owned(), sampling.clone());
        let (client_id, stop_sequences) = (client_id.to_owned(), stop_sequences.to_vec());
        Ok(self.with_progress(
            Box::new(move |backend, on_token| {
                backend.complete(
                    &prompt,
                    max_tokens,
                    &sampling,
                    None,
                    on_token,
                    &client_id,
                    &stop_sequences,
                    cache_prompt,
                )
            }),
            None,
        ))
    }

    fn start_chat(
        &self,
        request: &ChatRequest,
        max_tokens: usize,
        sampling: &SamplingConfig,
        _: Option<CancellationCheck>,
        _: bool,
    ) -> Result<Option<Box<dyn GenerationRequest>>> {
        let (request, sampling) = (request.clone(), sampling.clone());
        let tool_format = *self.tool_format.lock().expect("tool format");
        Ok(self.with_progress(
            Box::new(move |backend, on_token| {
                backend.chat(&request, max_tokens, &sampling, None, on_token)
            }),
            tool_format,
        ))
    }

    fn initial_output_state(&self, _: &ChatRequest) -> InitialOutputState {
        *self.initial.lock().expect("initial")
    }

    fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<&CancellationCheck>,
        on_token: Option<&mut dyn FnMut(&[u8]) -> bool>,
        client_id: &str,
        stop_sequences: &[String],
        cache_prompt: bool,
    ) -> Result<GenerationResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.last_cache_prompt.store(cache_prompt, Ordering::SeqCst);
        match self.failure.load(Ordering::SeqCst) {
            1 => return Err(Error::length("context exceeded")),
            2 => return Err(Error::invalid("invalid prompt")),
            // A failed probe: the backend reports device loss from then on.
            3 => {
                self.lost.store(true, Ordering::SeqCst);
                return Err(Error::generation(
                    GenerationErrorCode::DeviceLost,
                    DEVICE_LOST_MESSAGE,
                ));
            }
            // A failure on a device that still accepts work.
            4 => return Err(Error::runtime("unspecified launch failure")),
            5 => return Err(Error::runtime("")),
            _ => {}
        }
        let mut result = GenerationResult::default();
        if self.wait_for_disconnect.load(Ordering::SeqCst) {
            self.entered.release();
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                let disconnected = is_cancelled.is_some_and(|c| c());
                self.disconnected.store(disconnected, Ordering::SeqCst);
                if disconnected || Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            self.finished.release();
            result.cancelled = self.disconnected.load(Ordering::SeqCst);
            return Ok(result);
        }
        {
            let mut last = self.last.lock().expect("last");
            *last = Call {
                prompt: prompt.to_owned(),
                chat: ChatRequest::default(),
                max_tokens,
                sampling: sampling.clone(),
                client_id: client_id.to_owned(),
                stop_sequences: stop_sequences.to_vec(),
            };
            result.text = self.output.lock().expect("output").clone();
        }
        result.prompt_tokens = 10;
        result.cached_prompt_tokens = 8;
        result.cache_hit = true;
        result.draft_accepted_tokens = 4;
        result.draft_rounds = 2;
        result.draft_tokens = 8;
        result.prefill_tokens = 2;
        result.prefill_ms = 4.0;
        result.completion_tokens = 1;
        result.decode_ms = 2.0;
        result.finish_reason = if max_tokens == 1 {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        let forced = self.forced_stop_sequence.lock().expect("stop").clone();
        if !forced.is_empty() {
            result.finish_reason = FinishReason::StopSequence;
            result.stop_sequence = forced;
        }
        // 8 ends successfully without a token; 6 and 7 fail after one.
        let failure = self.failure.load(Ordering::SeqCst);
        let pieces = self.pieces.lock().expect("pieces").clone();
        if !pieces.is_empty() {
            result.text = pieces.concat();
        }
        if let Some(on_token) = on_token
            && failure != 8
        {
            std::thread::sleep(*self.token_delay.lock().expect("delay"));
            if pieces.is_empty() {
                let _ = on_token(&result.text);
            }
            for piece in &pieces {
                if !on_token(piece) {
                    break;
                }
            }
        }
        match failure {
            6 => Err(Error::runtime("context exceeded")),
            7 => {
                self.lost.store(true, Ordering::SeqCst);
                Err(Error::generation(
                    GenerationErrorCode::DeviceLost,
                    DEVICE_LOST_MESSAGE,
                ))
            }
            _ => Ok(result),
        }
    }

    fn chat(
        &self,
        request: &ChatRequest,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<&CancellationCheck>,
        on_token: Option<&mut dyn FnMut(&[u8]) -> bool>,
    ) -> Result<GenerationResult> {
        let result = self.complete(
            "",
            max_tokens,
            sampling,
            is_cancelled,
            on_token,
            &request.client_id,
            &request.stop_sequences,
            request.cache_prompt,
        )?;
        self.last.lock().expect("last").chat = request.clone();
        Ok(result)
    }
}

pub struct RunningServer {
    pub backend: Arc<FakeBackend>,
    pub server: Arc<HttpServer>,
    /// Released when the accept loop returns.
    pub run_finished: Arc<Semaphore>,
    worker: Option<JoinHandle<()>>,
}

impl RunningServer {
    pub fn new(options: HttpServerOptions) -> Self {
        Self::start("127.0.0.1", options, false)
    }

    /// A server listening on `host` (an IPv6 address, say).
    pub fn on_host(host: &str) -> Self {
        Self::start(host, HttpServerOptions::default(), false)
    }

    /// A server whose accept loop handles SIGINT and SIGTERM, as the CLI's.
    pub fn with_signals(options: HttpServerOptions) -> Self {
        Self::start("127.0.0.1", options, true)
    }

    fn start(host: &str, options: HttpServerOptions, handle_signals: bool) -> Self {
        let backend = Arc::new_cyclic(|me| FakeBackend::new(me.clone()));
        let dynamic: Arc<dyn TextGenerationBackend> = backend.clone();
        let server = Arc::new(
            HttpServer::new(host, 0, Some(dynamic), HttpServices::default(), options)
                .expect("server"),
        );
        server.add(
            "POST",
            "/echo",
            Arc::new(|request, _| Ok(HttpResponse::new(200, "OK", request.body.clone()))),
        );
        server.add(
            "POST",
            "/stream-error",
            Arc::new(|_, _| {
                Ok(HttpResponse {
                    streaming_body: Some(Box::new(|write| {
                        let _ = write(b"first chunk");
                        Err(Error::runtime("injected stream failure"))
                    })),
                    ..HttpResponse::default()
                })
            }),
        );
        server.add(
            "POST",
            "/stream",
            Arc::new(|request, _| {
                let log = Arc::new(Mutex::new(StreamLog::default()));
                let fail = !request.body.is_empty();
                let reported = request.body == b"reported";
                let body_log = log.clone();
                Ok(HttpResponse {
                    streaming_body: Some(Box::new(move |write| {
                        let _ = write(b"a\0b");
                        let _ = write(b"");
                        let _ = write(b"end");
                        let mut log = body_log.lock().expect("log");
                        if fail {
                            log.error_code = "injected".into();
                        }
                        log.error_event_sent = reported;
                        Ok(())
                    })),
                    stream_log: Some(log),
                    ..HttpResponse::default()
                })
            }),
        );
        let event_stream = || vec![("Content-Type".to_owned(), "text/event-stream".to_owned())];
        server.add(
            "POST",
            "/sse-idle",
            Arc::new(move |_, _| {
                Ok(HttpResponse {
                    headers: event_stream(),
                    streaming_body: Some(Box::new(|write| {
                        let _ = write(b"data: first\n\n");
                        std::thread::sleep(Duration::from_millis(90));
                        let _ = write(b"data: last\n\n");
                        let _ = write(b"data: [DONE]\n\n");
                        Ok(())
                    })),
                    ..HttpResponse::default()
                })
            }),
        );
        server.add(
            "POST",
            "/sse-finish",
            Arc::new(move |request, _| {
                let fail = !request.body.is_empty();
                Ok(HttpResponse {
                    headers: event_stream(),
                    streaming_body: Some(Box::new(move |write| {
                        let _ = write(b"data: first\n\n");
                        if fail {
                            return Err(Error::runtime("injected SSE failure"));
                        }
                        let _ = write(b"data: [DONE]\n\n");
                        Ok(())
                    })),
                    ..HttpResponse::default()
                })
            }),
        );
        server.start().expect("start");
        let running = server.clone();
        let run_finished = Arc::new(Semaphore::default());
        let finished = run_finished.clone();
        let worker = std::thread::spawn(move || {
            running.run(handle_signals);
            finished.release();
        });
        Self {
            backend,
            server,
            run_finished,
            worker: Some(worker),
        }
    }

    pub fn connect(&self) -> TcpStream {
        let stream = TcpStream::connect(("127.0.0.1", self.server.port())).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("timeout");
        stream
    }

    /// Sends raw bytes and reads the whole response; a half-close lets
    /// truncated-body requests finish without timing-dependent sleeps.
    pub fn send(&self, request: &str, half_close: bool) -> String {
        let mut stream = self.connect();
        stream.write_all(request.as_bytes()).expect("send");
        if half_close {
            let _ = stream.shutdown(Shutdown::Write);
        }
        let mut response = Vec::new();
        let _ = stream.read_to_end(&mut response);
        String::from_utf8_lossy(&response).into_owned()
    }

    pub fn post(&self, path: &str, body: &str) -> String {
        self.send(
            &format!(
                "POST {path} HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            ),
            false,
        )
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        self.server.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub fn expect_status(response: &str, status: u16) {
    assert!(
        response.starts_with(&format!("HTTP/1.1 {status} ")),
        "expected {status}:\n{response}"
    );
}

pub fn body_of(response: &str) -> &str {
    response
        .find("\r\n\r\n")
        .map_or("", |position| &response[position + 4..])
}
