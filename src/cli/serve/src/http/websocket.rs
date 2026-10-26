//! RFC 6455 server connections (`src/cli/serve/websocket.cpp`): one reader
//! thread parses frames and answers ping and close while the handler
//! consumes a bounded queue of complete text messages.

use std::collections::VecDeque;
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gufo_core::utf8::Utf8Decoder;
use sha1_smol::Sha1;

use super::{HttpRequest, HttpResponse, WebSocketHandler};
use crate::audio_stream::{decode_audio_base64, encode_audio_base64};

const MAXIMUM_MESSAGE: usize = 4 << 20;
const MAXIMUM_QUEUE: usize = 8 << 20;
const MAXIMUM_QUEUED_MESSAGES: usize = 128;

fn valid_utf8(bytes: &[u8]) -> bool {
    Utf8Decoder::default().push(bytes, true).as_bytes() == bytes
}

/// Whether the comma-separated header `value` contains `token`
/// (case-insensitive).
fn has_token(value: &str, token: &str) -> bool {
    value
        .to_ascii_lowercase()
        .split(',')
        .any(|item| item.trim_matches([' ', '\t']) == token)
}

pub fn is_websocket_upgrade(request: &HttpRequest) -> bool {
    has_token(&request.header("Upgrade"), "websocket")
}

/// The 101 response that hands the connection to `handler`, or 400 for an
/// invalid handshake.
pub fn upgrade_websocket(request: &HttpRequest, handler: WebSocketHandler) -> HttpResponse {
    let key = request.header("Sec-WebSocket-Key");
    let content_length = request.header("Content-Length");
    let key_ok = decode_audio_base64(key.as_bytes(), 16).is_some_and(|k| k.len() == 16);
    if request.method != "GET"
        || !is_websocket_upgrade(request)
        || !has_token(&request.header("Connection"), "upgrade")
        || request.header("Sec-WebSocket-Version") != "13"
        || (!content_length.is_empty() && content_length != "0")
        || !key_ok
    {
        return HttpResponse::new(
            400,
            "Bad Request",
            r#"{"error":{"message":"invalid WebSocket handshake","code":"invalid_websocket"}}"#,
        );
    }
    let mut digest = Sha1::new();
    digest.update(key.as_bytes());
    digest.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let mut response = HttpResponse::new(101, "Switching Protocols", Vec::new())
        .with_header("Upgrade", "websocket")
        .with_header("Connection", "Upgrade")
        .with_header(
            "Sec-WebSocket-Accept",
            encode_audio_base64(&digest.digest().bytes()),
        );
    response.websocket = Some(handler);
    response
}

struct Queue {
    messages: VecDeque<String>,
    bytes: usize,
}

struct Shared {
    closed: AtomicBool,
    send: Mutex<TcpStream>,
    queue: Mutex<Queue>,
    ready: Condvar,
    opened: Instant,
    /// Microseconds after `opened` when the last frame was sent: the session
    /// is idle only when neither side sent anything within the server's
    /// receive timeout.
    last_send: AtomicU64,
}

impl Shared {
    fn queue(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn now(&self) -> u64 {
        u64::try_from(self.opened.elapsed().as_micros()).unwrap_or(u64::MAX)
    }

    fn since_send(&self) -> Duration {
        Duration::from_micros(
            self.now()
                .saturating_sub(self.last_send.load(Ordering::SeqCst)),
        )
    }

    /// True for the call that closed the connection. Changing the flag
    /// under the queue lock keeps a close from racing a waiting receive.
    fn mark_closed(&self) -> bool {
        let changed = {
            let _queue = self.queue();
            !self.closed.swap(true, Ordering::SeqCst)
        };
        self.ready.notify_all();
        changed
    }

    fn send(&self, opcode: u8, bytes: &[u8]) -> bool {
        let mut stream = self.send.lock().unwrap_or_else(|e| e.into_inner());
        if self.closed.load(Ordering::SeqCst) && opcode != 8 {
            return false;
        }
        let mut header = vec![0x80 | opcode];
        if bytes.len() < 126 {
            header.push(bytes.len() as u8);
        } else if bytes.len() <= 65535 {
            header.push(126);
            header.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
        } else {
            header.push(127);
            header.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        }
        let sent = stream
            .write_all(&header)
            .and_then(|()| stream.write_all(bytes));
        if sent.is_err() {
            drop(stream);
            self.mark_closed();
            let stream = self.send.lock().unwrap_or_else(|e| e.into_inner());
            let _ = stream.shutdown(Shutdown::Both);
            return false;
        }
        self.last_send.store(self.now(), Ordering::SeqCst);
        true
    }

    fn close(&self, code: u16) {
        if !self.mark_closed() {
            return;
        }
        self.send(8, &code.to_be_bytes());
        let stream = self.send.lock().unwrap_or_else(|e| e.into_inner());
        let _ = stream.shutdown(Shutdown::Read);
    }
}

/// One upgraded connection.
pub struct WebSocket {
    shared: Arc<Shared>,
    reader: Option<JoinHandle<()>>,
}

impl WebSocket {
    /// Takes over `stream`; `buffered` holds bytes read past the handshake.
    pub fn new(stream: TcpStream, buffered: Vec<u8>) -> std::io::Result<Self> {
        // Frame headers and audio payloads are separate writes: do not wait
        // for a delayed acknowledgement. Bound backpressure from a client
        // that stops reading; the read timeout stays the server's.
        stream.set_nodelay(true)?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        let reader_stream = stream.try_clone()?;
        let shared = Arc::new(Shared {
            closed: AtomicBool::new(false),
            send: Mutex::new(stream),
            queue: Mutex::new(Queue {
                messages: VecDeque::new(),
                bytes: 0,
            }),
            ready: Condvar::new(),
            opened: Instant::now(),
            last_send: AtomicU64::new(0),
        });
        let reader_shared = shared.clone();
        let reader = std::thread::Builder::new()
            .name("gufo-websocket".into())
            .spawn(move || {
                let mut reader = FrameReader {
                    shared: reader_shared,
                    stream: reader_stream,
                    buffered,
                    offset: 0,
                };
                reader.run();
                reader.shared.mark_closed();
            })?;
        Ok(Self {
            shared,
            reader: Some(reader),
        })
    }

    /// The next complete text message; `None` once the connection closed.
    pub fn receive(&self) -> Option<String> {
        let mut queue = self.shared.queue();
        while !self.shared.closed.load(Ordering::SeqCst) && queue.messages.is_empty() {
            queue = self
                .shared
                .ready
                .wait(queue)
                .unwrap_or_else(|e| e.into_inner());
        }
        if self.shared.closed.load(Ordering::SeqCst) {
            return None;
        }
        let message = queue.messages.pop_front()?;
        queue.bytes -= message.len();
        Some(message)
    }

    pub fn send_text(&self, text: &str) -> bool {
        self.shared.send(1, text.as_bytes())
    }

    pub fn send_binary(&self, bytes: &[u8]) -> bool {
        self.shared.send(2, bytes)
    }

    pub fn close(&self, code: u16) {
        self.shared.close(code);
    }

    pub fn cancelled(&self) -> bool {
        self.shared.closed.load(Ordering::SeqCst)
    }
}

impl Drop for WebSocket {
    fn drop(&mut self) {
        self.shared.close(1000);
        // Cancellation can wake the consumer before the reader finished its
        // close reply: let that write complete before closing the send half.
        {
            let stream = self.shared.send.lock().unwrap_or_else(|e| e.into_inner());
            let _ = stream.shutdown(Shutdown::Read);
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        let stream = self.shared.send.lock().unwrap_or_else(|e| e.into_inner());
        let _ = stream.shutdown(Shutdown::Both);
    }
}

struct FrameReader {
    shared: Arc<Shared>,
    stream: TcpStream,
    buffered: Vec<u8>,
    offset: usize,
}

impl FrameReader {
    fn read(&mut self, data: &mut [u8]) -> bool {
        let mut filled = 0;
        while filled < data.len() && !self.shared.closed.load(Ordering::SeqCst) {
            if self.offset < self.buffered.len() {
                let count = (data.len() - filled).min(self.buffered.len() - self.offset);
                data[filled..filled + count]
                    .copy_from_slice(&self.buffered[self.offset..self.offset + count]);
                self.offset += count;
                filled += count;
                continue;
            }
            match self.stream.read(&mut data[filled..]) {
                Ok(0) => return false,
                Ok(count) => filled += count,
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    // The receive timeout is the server's idle limit; a
                    // client that only listens to streamed output is not
                    // idle.
                    let idle = self.stream.read_timeout().ok().flatten();
                    if idle.is_some_and(|idle| self.shared.since_send() < idle) {
                        continue;
                    }
                    self.shared.close(1001);
                    return false;
                }
                Err(_) => return false,
            }
        }
        filled == data.len()
    }

    fn run(&mut self) {
        let mut message: Vec<u8> = Vec::new();
        let mut fragmented = false;
        while !self.shared.closed.load(Ordering::SeqCst) {
            let mut header = [0u8; 2];
            if !self.read(&mut header) {
                break;
            }
            let last = header[0] & 128 != 0;
            let opcode = header[0] & 15;
            let control = opcode >= 8;
            if header[0] & 112 != 0
                || header[1] & 128 == 0
                || !matches!(opcode, 0 | 1 | 8 | 9 | 10)
                || (control && !last)
            {
                self.shared.close(1002);
                return;
            }
            let mut length = u64::from(header[1] & 127);
            if length >= 126 {
                if control {
                    self.shared.close(1002);
                    return;
                }
                let count = if length == 126 { 2 } else { 8 };
                let mut extended = [0u8; 8];
                if !self.read(&mut extended[..count]) {
                    break;
                }
                length = extended[..count]
                    .iter()
                    .fold(0, |acc, &b| (acc << 8) | u64::from(b));
                if (count == 8 && extended[0] & 128 != 0)
                    || (count == 2 && length < 126)
                    || (count == 8 && length <= 65535)
                {
                    self.shared.close(1002);
                    return;
                }
            }
            if length > MAXIMUM_MESSAGE as u64
                || (!control && length > (MAXIMUM_MESSAGE - message.len()) as u64)
            {
                self.shared.close(1009);
                return;
            }
            let mut mask = [0u8; 4];
            if !self.read(&mut mask) {
                break;
            }
            let mut payload = vec![0u8; length as usize];
            if !self.read(&mut payload) {
                break;
            }
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i % 4];
            }
            match opcode {
                8 => {
                    if payload.len() == 1 {
                        self.shared.close(1002);
                        return;
                    }
                    if payload.len() >= 2 {
                        let code = u32::from(u16::from_be_bytes([payload[0], payload[1]]));
                        if !(1000..5000).contains(&code)
                            || matches!(code, 1004 | 1005 | 1006 | 1015)
                            || (1016..3000).contains(&code)
                        {
                            self.shared.close(1002);
                            return;
                        }
                        if !valid_utf8(&payload[2..]) {
                            self.shared.close(1007);
                            return;
                        }
                    }
                    if self.shared.mark_closed() {
                        self.shared.send(8, &payload);
                    }
                    break;
                }
                9 => {
                    if !self.shared.send(10, &payload) {
                        break;
                    }
                    continue;
                }
                10 => continue,
                _ => {}
            }
            if (opcode == 0) != fragmented {
                self.shared.close(1002);
                return;
            }
            message.extend_from_slice(&payload);
            fragmented = !last;
            if !last {
                continue;
            }
            let Ok(text) = String::from_utf8(std::mem::take(&mut message)) else {
                self.shared.close(1007);
                return;
            };
            {
                let mut queue = self.shared.queue();
                if queue.messages.len() >= MAXIMUM_QUEUED_MESSAGES
                    || text.len() > MAXIMUM_QUEUE - queue.bytes
                {
                    drop(queue);
                    self.shared.close(1009);
                    return;
                }
                queue.bytes += text.len();
                queue.messages.push_back(text);
            }
            self.shared.ready.notify_one();
        }
    }
}
