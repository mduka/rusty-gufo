//! The bounded HTTP/1.1 transport (`src/cli/serve/http_server.{hpp,cpp}`):
//! request and response types shared by every API, the server with its
//! routes, and WebSocket upgrades.

pub mod server;
pub mod websocket;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use gufo_core::text::{CancellationCheck, Result};

use crate::backend::TextGenerationBackend;
use crate::openai_chat::{ApiCompat, ResponsesReasoningTarget};
use websocket::WebSocket;

#[derive(Default)]
pub struct HttpRequest {
    /// `GET`, `POST`, ...
    pub method: String,
    /// The target without its query, e.g. `/v1/chat/completions`.
    pub path: String,
    /// The raw query string without the leading `?`.
    pub query: String,
    pub body: Vec<u8>,
    pub headers: Vec<(String, String)>,
    /// Polls the client socket for a disconnect.
    pub is_cancelled: Option<CancellationCheck>,
    pub request_id: String,
    /// The transport-derived peer address, never read from client headers.
    pub client_id: String,
}

impl HttpRequest {
    /// The URL-decoded value of query parameter `key`, empty if absent.
    pub fn query_param(&self, key: &str) -> String {
        for parameter in self.query.split('&') {
            let (name, value) = match parameter.find('=') {
                Some(equals) => (&parameter[..equals], Some(&parameter[equals + 1..])),
                None => (parameter, None),
            };
            if url_decode(name) == key {
                return value.map(url_decode).unwrap_or_default();
            }
        }
        String::new()
    }

    /// Case-insensitive header lookup, empty if absent.
    pub fn header(&self, name: &str) -> String {
        self.headers
            .iter()
            .find(|(header, _)| header.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
            .unwrap_or_default()
    }
}

/// Decodes `+` and `%XX`; malformed escapes stay literal. Invalid UTF-8 in
/// the decoded bytes becomes U+FFFD.
pub fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let hex = |h: u8| match h {
        b'0'..=b'9' => Some(h - b'0'),
        b'a'..=b'f' => Some(h - b'a' + 10),
        b'A'..=b'F' => Some(h - b'A' + 10),
        _ => None,
    };
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'+' {
            out.push(b' ');
        } else if c == b'%' && i + 2 < bytes.len() {
            match (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                (Some(high), Some(low)) => {
                    out.push((high << 4) | low);
                    i += 2;
                }
                _ => out.push(c),
            }
        } else {
            out.push(c);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Writes one piece of a streamed body; false once the client is gone.
pub type BodyWriter<'a> = dyn FnMut(&[u8]) -> bool + 'a;
/// Produces a streamed body on the connection thread after the head is
/// sent. An error ends the stream without its terminating chunk.
pub type StreamingBody = Box<dyn FnOnce(&mut BodyWriter<'_>) -> Result<()> + Send>;
/// Runs an upgraded WebSocket connection.
pub type WebSocketHandler = Box<dyn FnOnce(&WebSocket) + Send>;

/// What a streamed body reports for the request log.
#[derive(Debug, Clone, Default)]
pub struct StreamLog {
    pub details: String,
    /// Set on a stream error; the terminating chunk is then omitted unless
    /// the error reached the client in the stream's protocol.
    pub error_code: String,
    /// A terminal error event was written: a complete protocol response.
    pub error_event_sent: bool,
}

pub struct HttpResponse {
    pub status: u16,
    pub reason: String,
    pub body: Vec<u8>,
    pub headers: Vec<(String, String)>,
    pub streaming_body: Option<StreamingBody>,
    /// Text streams commit their headers on the first write, when
    /// generation starts or has queued for a bounded time, so a failure
    /// before that returns a real error status while keepalives still cover
    /// prefill and queue waits.
    pub defer_stream_headers: bool,
    pub log_details: String,
    pub stream_log: Option<Arc<Mutex<StreamLog>>>,
    pub websocket: Option<WebSocketHandler>,
}

impl Default for HttpResponse {
    fn default() -> Self {
        Self {
            status: 200,
            reason: "OK".into(),
            body: Vec::new(),
            headers: Vec::new(),
            streaming_body: None,
            defer_stream_headers: false,
            log_details: String::new(),
            stream_log: None,
            websocket: None,
        }
    }
}

impl HttpResponse {
    /// A response with `status`, `reason` and `body`.
    pub fn new(status: u16, reason: &str, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            reason: reason.into(),
            body: body.into(),
            ..Self::default()
        }
    }

    pub fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn has_header(&self, name: &str) -> bool {
        self.headers
            .iter()
            .any(|(h, _)| h.eq_ignore_ascii_case(name))
    }
}

/// A route handler. It borrows nothing from the request beyond the call, so
/// streamed bodies own what they need (the backend is shared by `Arc`).
pub type Handler = Arc<
    dyn Fn(&HttpRequest, &Arc<dyn TextGenerationBackend>) -> Result<HttpResponse> + Send + Sync,
>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpServerOptions {
    pub max_request_body_bytes: usize,
    pub max_connections: usize,
    pub api_key: String,
    /// An SSE stream idle this long gets a `: ping` comment, so proxies and
    /// clients do not drop it during a long prefill; zero disables it.
    pub sse_heartbeat_interval: Duration,
    /// Where `/v1/responses` puts the raw reasoning.
    pub responses_reasoning_target: ResponsesReasoningTarget,
    /// Where the compatibility routes follow llama.cpp instead of Gufo.
    pub api_compat: ApiCompat,
}

impl Default for HttpServerOptions {
    fn default() -> Self {
        Self {
            max_request_body_bytes: 8 * 1024 * 1024,
            max_connections: 16,
            api_key: String::new(),
            sse_heartbeat_interval: Duration::from_secs(15),
            responses_reasoning_target: ResponsesReasoningTarget::default(),
            api_compat: ApiCompat::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_parameters() {
        let mut request = HttpRequest {
            query: "notafter=wrong&note=after=wrong&after=right+value%26x".into(),
            ..HttpRequest::default()
        };
        assert_eq!(request.query_param("after"), "right value&x");
        request.query = "notafter=wrong&note=after=wrong".into();
        assert!(request.query_param("after").is_empty());
        request.query = "%61fter=encoded&broken=%xz&empty".into();
        assert_eq!(request.query_param("after"), "encoded");
        assert_eq!(request.query_param("broken"), "%xz");
        assert!(request.query_param("empty").is_empty());
    }
}
