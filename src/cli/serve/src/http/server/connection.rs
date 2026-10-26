//! Listening addresses, reading a request, serving one connection, and the
//! responder that answers while the model loads.

use super::*;

/// Holds the listen port while a model loads and answers every request with
/// llama-server's loading reply, 503 `Loading model`, `/health` included
/// (upstream `LoadingResponder`): a poller can tell a loading
/// server from a dead one, and a client that treats a 200 health check as
/// readiness keeps waiting instead of sending work to a server that cannot
/// serve it. Best effort: when the address cannot be bound, or for port 0,
/// nothing listens and the caller proceeds as without it. Stopping (or
/// dropping) releases the port for [`HttpServer::start`].
pub struct LoadingResponder {
    pub(super) stopped: Arc<AtomicBool>,
    pub(super) worker: Option<JoinHandle<()>>,
}

impl LoadingResponder {
    pub fn start(host: &str, port: i32) -> Self {
        let stopped = Arc::new(AtomicBool::new(false));
        let address = u16::try_from(port)
            .ok()
            .filter(|&port| port != 0)
            .and_then(|port| parse_listen_address(host, port));
        let listener = match address.map(|address| bind_listener(&address)) {
            Some(Ok(listener)) => listener,
            Some(Err(step)) => {
                logging::debug(
                    "server",
                    &format!(
                        "event=loading_listener_unavailable host={host} port={port} step={step}"
                    ),
                );
                return Self {
                    stopped,
                    worker: None,
                };
            }
            None => {
                return Self {
                    stopped,
                    worker: None,
                };
            }
        };
        let flag = stopped.clone();
        let worker = std::thread::Builder::new()
            .name("gufo-loading".into())
            .spawn(move || serve_loading(&listener, &flag))
            .ok();
        Self { stopped, worker }
    }

    /// Whether the responder owns the listen socket.
    pub fn listening(&self) -> bool {
        self.worker.is_some()
    }

    /// Stops answering and releases the port, within one poll interval plus
    /// the bounded wait for a request that is still arriving.
    pub fn stop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for LoadingResponder {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The loading responder's accept loop: one connection at a time is enough
/// for probes. It reads the request head before replying, so closing does
/// not reset an unread request, but bounds that wait: a client that connects
/// and sends nothing must not delay `stop`.
pub(super) fn serve_loading(listener: &TcpListener, stopped: &AtomicBool) {
    const REQUEST_WAIT: Duration = Duration::from_millis(500);
    let reply = build_response(
        &err(
            503,
            "Service Unavailable",
            "Loading model",
            "unavailable_error",
            "loading",
        )
        .with_header("Retry-After", "1"),
    );
    while !stopped.load(Ordering::Acquire) {
        let mut fds = [PollFd::new(listener, PollFlags::IN)];
        let timeout = rustix::event::Timespec {
            tv_sec: 0,
            tv_nsec: 100_000_000,
        };
        if !matches!(rustix::event::poll(&mut fds, Some(&timeout)), Ok(1..)) {
            continue;
        }
        let Ok((mut client, _)) = listener.accept() else {
            continue;
        };
        let deadline = Instant::now() + REQUEST_WAIT;
        let mut head = Vec::new();
        let mut buffer = [0u8; 2048];
        while !head.windows(4).any(|w| w == b"\r\n\r\n")
            && head.len() < 8192
            && !stopped.load(Ordering::Acquire)
        {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() || client.set_read_timeout(Some(remaining)).is_err() {
                break;
            }
            match client.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(count) => head.extend_from_slice(&buffer[..count]),
            }
        }
        let _ = client.write_all(&reply);
        let _ = client.shutdown(Shutdown::Write);
    }
}

/// A numeric listen address (upstream `ParseListenAddress`), so the
/// listener never depends on name resolution: IPv4 strictly, without
/// shorthand such as `127.1`, or IPv6 with an optional `%scope` (interface
/// name or index) for link-local addresses.
pub fn parse_listen_address(host: &str, port: u16) -> Option<SocketAddr> {
    if let Ok(address) = host.parse::<Ipv4Addr>() {
        return Some(SocketAddr::V4(SocketAddrV4::new(address, port)));
    }
    let (address, scope) = match host.split_once('%') {
        Some((address, scope)) => (address, Some(scope)),
        None => (host, None),
    };
    let address: Ipv6Addr = address.parse().ok()?;
    let scope_id = match scope {
        None => 0,
        Some(scope) => scope.parse().ok().or_else(|| interface_index(scope))?,
    };
    Some(SocketAddr::V6(SocketAddrV6::new(
        address, port, 0, scope_id,
    )))
}

/// Whether `host` is an address [`HttpServer::start`] can listen on
/// (`IsListenAddress`).
pub fn is_listen_address(host: &str) -> bool {
    parse_listen_address(host, 0).is_some()
}

/// The index of network interface `name`, from sysfs.
pub(super) fn interface_index(name: &str) -> Option<u32> {
    if name.is_empty() || name.contains('/') || name.starts_with('.') {
        return None;
    }
    std::fs::read_to_string(format!("/sys/class/net/{name}/ifindex"))
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// `host` as the host part of a URL: IPv6 in brackets.
pub(super) fn url_host(host: &str) -> std::borrow::Cow<'_, str> {
    if host.contains(':') {
        format!("[{host}]").into()
    } else {
        host.into()
    }
}

/// A peer's client ID, which keys per-client scheduling: one client keeps
/// one ID whichever listener it reaches, so an IPv4 peer of a dual-stack
/// `::` listener, `::ffff:a.b.c.d`, is `a.b.c.d` (upstream `PeerAddress`).
pub(super) fn client_address(peer: IpAddr) -> String {
    match peer {
        IpAddr::V4(address) => address.to_string(),
        IpAddr::V6(address) => match address.to_ipv4_mapped() {
            Some(address) => address.to_string(),
            None => address.to_string(),
        },
    }
}

/// What reading the request produced.
pub(super) enum Parsed {
    Complete,
    Malformed,
    TooLarge,
}

/// Reads and parses one request into `request`.
pub(super) fn read_request(
    shared: &Shared,
    stream: &mut TcpStream,
    request: &mut HttpRequest,
    http11: &mut bool,
) -> Parsed {
    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    let delimiter = loop {
        if let Some(position) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break position;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return Parsed::Malformed,
            Ok(count) => buffer.extend_from_slice(&chunk[..count]),
        }
        if buffer.len() > MAX_HEADER_BYTES {
            return Parsed::Malformed;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..delimiter]).into_owned();
    let mut body = buffer[delimiter + 4..].to_vec();
    let mut lines = headers.split("\r\n");
    let line = lines.next().unwrap_or("");
    let tokens: Vec<&str> = line
        .split(|c: char| c.is_ascii_whitespace() || c == '\u{b}')
        .filter(|t| !t.is_empty())
        .collect();
    let (method, target, version) = match tokens.as_slice() {
        [method, target] => (*method, *target, ""),
        [method, target, version, ..] => (*method, *target, *version),
        [method] => (*method, "", ""),
        [] => ("", "", ""),
    };
    request.method = method.to_owned();
    *http11 = version == "HTTP/1.1";
    let mut valid_headers = !request.method.is_empty()
        && target.starts_with('/')
        && (version == "HTTP/1.0" || version == "HTTP/1.1")
        && tokens.len() == 3;
    match target.find('?') {
        Some(q) => {
            request.query = target[q + 1..].to_owned();
            request.path = target[..q].to_owned();
        }
        None => request.path = target.to_owned(),
    }
    // RFC 9112 5.1: whitespace before the colon and obsolete line folding
    // are rejected instead of storing a name no lookup would match.
    let token = |c: u8| c.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&c);
    for header_line in lines {
        match header_line.find(':') {
            Some(colon) if colon > 0 && header_line.bytes().take(colon).all(token) => {
                let value = header_line[colon + 1..]
                    .trim_matches(|c: char| c.is_ascii_whitespace() || c == '\u{b}');
                request
                    .headers
                    .push((header_line[..colon].to_owned(), value.to_owned()));
            }
            _ => valid_headers = false,
        }
    }
    let parsed_length = parse_content_length(request);
    let content_length = parsed_length.unwrap_or(0);
    let remaining = content_length.saturating_sub(body.len());
    if !valid_headers || parsed_length.is_none() {
        return Parsed::Malformed;
    }
    if content_length > shared.options.max_request_body_bytes {
        return Parsed::TooLarge;
    }
    if content_length > 0 {
        if remaining > 0 {
            // RFC 9110 10.1.1: a client that sends `Expect: 100-continue`
            // (curl does for bodies over 1 MiB) waits for the interim
            // response before the body; HTTP/1.0 expectations are ignored.
            if *http11
                && request.headers.iter().any(|(name, value)| {
                    name.eq_ignore_ascii_case("expect")
                        && value.eq_ignore_ascii_case("100-continue")
                })
                && stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").is_err()
            {
                return Parsed::Malformed;
            }
            let start = body.len();
            body.resize(start + remaining, 0);
            if stream.read_exact(&mut body[start..]).is_err() {
                return Parsed::Malformed;
            }
        }
        body.truncate(content_length);
    } else if !is_websocket_upgrade(request) {
        body.clear();
    }
    request.body = body;
    Parsed::Complete
}

/// A streamed body's connection, shared with the SSE heartbeat.
pub(super) struct StreamLink<'a> {
    pub(super) stream: &'a mut TcpStream,
    pub(super) chunked: bool,
    pub(super) connected: bool,
    pub(super) last_write: Instant,
    pub(super) finished: bool,
    /// Deferred response headers, sent with the first write, empty or not.
    pub(super) head: Option<Vec<u8>>,
}

impl StreamLink<'_> {
    pub(super) fn send(&mut self, piece: &[u8]) -> bool {
        if !self.connected {
            return false;
        }
        self.connected = match self.head.take() {
            // The headers and the first piece travel in one send.
            Some(mut first) => {
                if self.chunked && !piece.is_empty() {
                    first.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
                    first.extend_from_slice(piece);
                    first.extend_from_slice(b"\r\n");
                } else {
                    first.extend_from_slice(piece);
                }
                send_all(self.stream, &first)
            }
            None if self.chunked => send_chunk(self.stream, piece),
            None => send_all(self.stream, piece),
        };
        if self.connected && !piece.is_empty() {
            self.last_write = Instant::now();
        }
        self.connected
    }
}

pub(super) fn is_event_stream(response: &HttpResponse) -> bool {
    response.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-type")
            && value.to_ascii_lowercase().starts_with("text/event-stream")
    })
}

pub(super) fn handle_connection(shared: &Shared, stream: TcpStream) {
    serve_connection(shared, stream);
    // Act on a failure this request found now, not at the next poll.
    fatal_failure(shared);
}

pub(super) fn serve_connection(shared: &Shared, mut stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(SOCKET_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
    let start = Instant::now();
    let elapsed_ms = || start.elapsed().as_secs_f64() * 1e3;
    let mut request = HttpRequest {
        client_id: match stream.peer_addr() {
            Ok(peer) => client_address(peer.ip()),
            Err(_) => "anonymous".into(),
        },
        request_id: format!(
            "r{}",
            shared.next_request.fetch_add(1, Ordering::Relaxed) + 1
        ),
        ..HttpRequest::default()
    };
    let mut http11 = false;
    let parsed = read_request(shared, &mut stream, &mut request, &mut http11);
    if matches!(parsed, Parsed::Complete)
        && let Ok(peer) = stream.try_clone()
    {
        let check: CancellationCheck = Arc::new(move || is_peer_disconnected(&peer));
        request.is_cancelled = Some(check);
    }
    // Successful health, metrics and video status polling stay quiet at the
    // default level; debug shows them, to tell whether anything arrives.
    let log_request = matches!(request.method.as_str(), "POST" | "DELETE")
        || request.path == "/v1/models"
        || request.path.ends_with("/content")
        || request.path == "/v1/realtime"
        || request.path == "/v1/audio/speech/stream";
    let request_level = if log_request {
        LogLevel::Info
    } else {
        LogLevel::Debug
    };
    let complete = matches!(parsed, Parsed::Complete);
    if complete && logging::enabled(request_level) {
        logging::log(
            request_level,
            "http",
            &format!(
                "request={} event=received method={} path={} body_bytes={}",
                request.request_id,
                request.method,
                request.path,
                request.body.len()
            ),
        );
    }
    let outcome = match parsed {
        Parsed::TooLarge => Ok(err(
            413,
            "Payload Too Large",
            "request body is too large",
            "invalid_request_error",
            "payload_too_large",
        )),
        Parsed::Malformed => Ok(err(
            400,
            "Bad Request",
            "malformed request",
            "invalid_request_error",
            "bad_request",
        )),
        Parsed::Complete if request.method == "OPTIONS" => {
            Ok(HttpResponse::new(204, "No Content", Vec::new()))
        }
        Parsed::Complete => catch_unwind(AssertUnwindSafe(|| handle_request(shared, &request)))
            .unwrap_or_else(|_| Err(Error::runtime("unknown server error"))),
    };
    let mut response = match outcome {
        Ok(response) => response,
        Err(error) => {
            fail(&mut stream, &request, &error, false, 0, elapsed_ms());
            return;
        }
    };
    response
        .headers
        .push(("X-Request-ID".into(), request.request_id.clone()));
    let status = response.status;
    let mut connected: bool;
    if let Some(handler) = response.websocket.take() {
        connected = send_all(&mut stream, &build_response_head(&response, None));
        if connected
            && let Ok(socket_stream) = stream.try_clone()
            && let Ok(socket) = WebSocket::new(socket_stream, std::mem::take(&mut request.body))
        {
            handler(&socket);
        }
    } else if let Some(body) = response.streaming_body.take() {
        let chunked = http11 && !response.has_header("content-length");
        if chunked {
            response
                .headers
                .push(("Transfer-Encoding".into(), "chunked".into()));
        }
        // Text streams commit their headers when generation starts, so an
        // earlier failure still answers with a real status.
        let head = build_response_head(&response, None);
        let deferred = response.defer_stream_headers;
        connected = deferred || send_all(&mut stream, &head);
        if connected {
            let heartbeat = Some(shared.options.sse_heartbeat_interval)
                .filter(|interval| !interval.is_zero() && is_event_stream(&response));
            let link = Mutex::new(StreamLink {
                stream: &mut stream,
                chunked,
                connected,
                last_write: Instant::now(),
                finished: false,
                head: deferred.then_some(head),
            });
            let finished = Condvar::new();
            let streamed = std::thread::scope(|scope| {
                if let Some(interval) = heartbeat {
                    let (link, finished) = (&link, &finished);
                    scope.spawn(move || {
                        let mut link = lock(link);
                        while !link.finished && link.connected {
                            // Heartbeats start once the headers commit.
                            if link.head.is_some() {
                                link = finished.wait(link).unwrap_or_else(|e| e.into_inner());
                                continue;
                            }
                            // Writes only move the deadline; they need not
                            // wake this thread for every generated token.
                            let deadline = link.last_write + interval;
                            let now = Instant::now();
                            if now < deadline {
                                link = finished
                                    .wait_timeout(link, deadline - now)
                                    .map_or_else(|e| e.into_inner().0, |(link, _)| link);
                                continue;
                            }
                            // An SSE comment carries bytes without changing
                            // the API event stream.
                            link.send(b": ping\n\n");
                        }
                    });
                }
                let mut writer = |piece: &[u8]| {
                    let mut link = lock(&link);
                    let committing = link.head.is_some();
                    let sent = link.send(piece);
                    if committing {
                        finished.notify_all();
                    }
                    sent
                };
                let streamed = catch_unwind(AssertUnwindSafe(|| body(&mut writer)))
                    .unwrap_or_else(|_| Err(Error::runtime("unknown server error")));
                lock(&link).finished = true;
                finished.notify_all();
                streamed
            });
            let link = link.into_inner().unwrap_or_else(|e| e.into_inner());
            let (started, pending_head) = (link.head.is_none(), link.head);
            connected = link.connected;
            if let Err(error) = streamed {
                fail(&mut stream, &request, &error, started, status, elapsed_ms());
                return;
            }
            // A stream that ended without writing still commits a valid,
            // empty response.
            if let Some(head) = pending_head {
                connected = connected && send_all(&mut stream, &head);
            }
            // An SSE error is a complete protocol response; a failed raw PCM
            // stream must stay incomplete, or it looks like shorter audio.
            let stream_failed = response.stream_log.as_ref().is_some_and(|log| {
                let log = lock(log);
                !log.error_code.is_empty() && !log.error_event_sent
            });
            if connected && chunked && !stream_failed {
                connected = send_all(&mut stream, b"0\r\n\r\n");
            }
        }
    } else {
        connected = send_all(&mut stream, &build_response(&response));
    }
    let mut outcome = if connected {
        "completed"
    } else {
        "disconnected"
    };
    if let Some(log) = &response.stream_log {
        let log = lock(log);
        response.log_details = log.details.clone();
        if !log.error_code.is_empty() {
            outcome = "stream_error";
            response.log_details += &format!(" error_code={}", log.error_code);
        }
    }
    if response.status >= 400
        && let Ok(body) = json::parse(&response.body)
        && let Some(error) = body.find("error")
    {
        response.log_details += &format!(" error_code={}", error.member_str("code", ""));
    }
    logging::log_request(
        &request.request_id,
        &request.method,
        &request.path,
        response.status,
        elapsed_ms(),
        &response.log_details,
        outcome,
        request_level,
    );
}

/// A failure of the handler or of a started stream (a C++ exception at the
/// connection): logs it and, unless the response already started, answers
/// with the generation status or 500.
pub(super) fn fail(
    stream: &mut TcpStream,
    request: &HttpRequest,
    error: &Error,
    response_started: bool,
    started_status: u16,
    duration_ms: f64,
) {
    let message = if error.message().is_empty() {
        "generation failed"
    } else {
        error.message()
    };
    let (mut response, code) = match error.generation_code() {
        Some(code) => {
            let status = code.http_status();
            let reason = match status {
                408 => "Request Timeout",
                429 => "Too Many Requests",
                _ => "Service Unavailable",
            };
            let mut response = err(status, reason, message, "server_error", code.stable_code());
            if code.retryable() {
                response = response.with_header("Retry-After", "1");
            }
            (response, code.stable_code())
        }
        None => {
            // Retain the cause in logs alongside the client's stable code.
            logging::error(
                "http",
                &format!(
                    "request={} event=server_exception reason={message}",
                    request.request_id
                ),
            );
            (
                err(
                    500,
                    "Internal Server Error",
                    message,
                    "internal_error",
                    "server_exception",
                ),
                "server_exception",
            )
        }
    };
    response = response.with_header("X-Request-ID", request.request_id.clone());
    logging::log_request(
        &request.request_id,
        &request.method,
        &request.path,
        if response_started {
            started_status
        } else {
            response.status
        },
        duration_ms,
        &format!("error_code={code}"),
        if response_started {
            "stream_error"
        } else {
            "failed"
        },
        LogLevel::Info,
    );
    if !response_started {
        let _ = send_all(stream, &build_response(&response));
    }
}
