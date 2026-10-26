//! A minimal WebSocket client (RFC 6455) for the arbiter connection: text
//! messages in single frames, client frames masked as the protocol
//! requires, pings answered and a close ending the connection.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use sha1_smol::Sha1;

use crate::audio_stream::encode_audio_base64;

/// The largest message the arbiter sends.
const MAXIMUM_MESSAGE: usize = 1 << 20;
const MAXIMUM_HANDSHAKE: usize = 16 * 1024;

pub(super) struct WebSocketClient {
    stream: TcpStream,
    /// Received bytes not yet parsed into frames.
    buffer: Vec<u8>,
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    for chunk in bytes.chunks_mut(8) {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(chunk.as_ptr() as u64);
        let word = hasher.finish().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    bytes
}

fn accept_key(key: &str) -> String {
    let mut digest = Sha1::new();
    digest.update(key.as_bytes());
    digest.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    encode_audio_base64(&digest.digest().bytes())
}

impl WebSocketClient {
    /// Connects to `ws://address{path}`; reads then wait at most `poll`.
    pub(super) fn connect(address: &str, path: &str, poll: Duration) -> Result<Self, String> {
        let target = address
            .to_socket_addrs()
            .map_err(|e| format!("{address}: {e}"))?
            .next()
            .ok_or_else(|| format!("{address}: no address"))?;
        let mut stream = TcpStream::connect_timeout(&target, Duration::from_secs(2))
            .map_err(|e| format!("{address}: {e}"))?;
        stream.set_nodelay(true).map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .map_err(|e| e.to_string())?;
        stream
            .set_read_timeout(Some(Duration::from_millis(200)))
            .map_err(|e| e.to_string())?;
        let key = encode_audio_base64(&random_bytes::<16>());
        let request = format!(
            "GET {path} HTTP/1.1\r\nHost: {address}\r\nUpgrade: websocket\r\n\
             Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|e| format!("{address}: {e}"))?;
        // The response head; anything after it already belongs to frames.
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut received = Vec::new();
        let end = loop {
            if let Some(end) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                break end;
            }
            if received.len() > MAXIMUM_HANDSHAKE || Instant::now() >= deadline {
                return Err(format!("{address}: no WebSocket handshake response"));
            }
            let mut chunk = [0u8; 1024];
            match stream.read(&mut chunk) {
                Ok(0) => return Err(format!("{address}: connection closed during the handshake")),
                Ok(n) => received.extend_from_slice(&chunk[..n]),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                Err(e) => return Err(format!("{address}: {e}")),
            }
        };
        let head = String::from_utf8_lossy(&received[..end]).into_owned();
        let mut lines = head.split("\r\n");
        let status = lines.next().unwrap_or_default();
        if status.split(' ').nth(1) != Some("101") {
            return Err(format!(
                "{address}: the arbiter refused the WebSocket: {status}"
            ));
        }
        let expected = accept_key(&key);
        let accepted = lines.any(|line| {
            line.split_once(':').is_some_and(|(name, value)| {
                name.trim().eq_ignore_ascii_case("sec-websocket-accept") && value.trim() == expected
            })
        });
        if !accepted {
            return Err(format!("{address}: invalid Sec-WebSocket-Accept"));
        }
        stream
            .set_read_timeout(Some(poll))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            stream,
            buffer: received[end + 4..].to_vec(),
        })
    }

    pub(super) fn send_text(&mut self, text: &str) -> Result<(), String> {
        self.send(1, text.as_bytes())
    }

    /// Sends a close frame; the connection ends either way.
    pub(super) fn close(&mut self) {
        let _ = self.send(8, &1000u16.to_be_bytes());
    }

    fn send(&mut self, opcode: u8, payload: &[u8]) -> Result<(), String> {
        let mut frame = Vec::with_capacity(payload.len() + 14);
        frame.push(0x80 | opcode);
        let length = payload.len();
        if length < 126 {
            frame.push(0x80 | length as u8);
        } else if length <= 0xffff {
            frame.push(0x80 | 126);
            frame.extend_from_slice(&(length as u16).to_be_bytes());
        } else {
            frame.push(0x80 | 127);
            frame.extend_from_slice(&(length as u64).to_be_bytes());
        }
        let mask = random_bytes::<4>();
        frame.extend_from_slice(&mask);
        frame.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        self.stream.write_all(&frame).map_err(|e| e.to_string())
    }

    /// The next text message; `Ok(None)` when none completed within the
    /// poll interval. A close or a protocol error is an `Err`.
    pub(super) fn receive(&mut self) -> Result<Option<String>, String> {
        loop {
            if let Some(message) = self.parse()? {
                return Ok(Some(message));
            }
            let mut chunk = [0u8; 4096];
            match self.stream.read(&mut chunk) {
                Ok(0) => return Err("the arbiter closed the connection".into()),
                Ok(n) => self.buffer.extend_from_slice(&chunk[..n]),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    return Ok(None);
                }
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    /// Takes complete frames off the buffer until a text message; control
    /// frames are handled here.
    fn parse(&mut self) -> Result<Option<String>, String> {
        loop {
            let Some(Frame {
                last,
                opcode,
                payload,
                used,
            }) = frame(&self.buffer)?
            else {
                return Ok(None);
            };
            self.buffer.drain(..used);
            match opcode {
                1 if last => {
                    return String::from_utf8(payload)
                        .map(Some)
                        .map_err(|_| "the arbiter sent invalid UTF-8".into());
                }
                8 => {
                    let _ = self.send(8, &payload);
                    let code = payload
                        .get(..2)
                        .map_or(1005, |c| u16::from_be_bytes([c[0], c[1]]));
                    return Err(format!("the arbiter closed the connection ({code})"));
                }
                9 => self.send(10, &payload)?,
                10 => {}
                _ => return Err("the arbiter sent an unsupported frame".into()),
            }
        }
    }
}

/// A parsed frame and the bytes it occupied.
#[derive(Debug, PartialEq, Eq)]
struct Frame {
    last: bool,
    opcode: u8,
    payload: Vec<u8>,
    used: usize,
}

/// The first complete frame of `bytes`. Server frames are unmasked.
fn frame(bytes: &[u8]) -> Result<Option<Frame>, String> {
    if bytes.len() < 2 {
        return Ok(None);
    }
    if bytes[1] & 0x80 != 0 || bytes[0] & 0x70 != 0 {
        return Err("the arbiter sent a masked or extended frame".into());
    }
    let (length, offset) = match bytes[1] & 0x7f {
        126 if bytes.len() >= 4 => (usize::from(u16::from_be_bytes([bytes[2], bytes[3]])), 4),
        127 if bytes.len() >= 10 => {
            let mut length = [0u8; 8];
            length.copy_from_slice(&bytes[2..10]);
            (
                usize::try_from(u64::from_be_bytes(length)).unwrap_or(usize::MAX),
                10,
            )
        }
        126 | 127 => return Ok(None),
        length => (usize::from(length), 2),
    };
    if length > MAXIMUM_MESSAGE {
        return Err("the arbiter sent an oversized frame".into());
    }
    if bytes.len() < offset + length {
        return Ok(None);
    }
    Ok(Some(Frame {
        last: bytes[0] & 0x80 != 0,
        opcode: bytes[0] & 0x0f,
        payload: bytes[offset..offset + length].to_vec(),
        used: offset + length,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_accept_key_matches_the_rfc_example() {
        assert_eq!(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn frames_parse_once_complete() {
        assert_eq!(frame(&[0x81]).unwrap(), None);
        assert_eq!(frame(&[0x81, 3, b'a', b'b']).unwrap(), None);
        assert_eq!(
            frame(&[0x81, 3, b'a', b'b', b'c', 0x89]).unwrap(),
            Some(Frame {
                last: true,
                opcode: 1,
                payload: b"abc".to_vec(),
                used: 5
            })
        );
        let mut long = vec![0x81, 126, 0x01, 0x00];
        long.extend(std::iter::repeat_n(b'x', 256));
        assert_eq!(frame(&long).unwrap().unwrap().payload.len(), 256);
        assert!(
            frame(&[0x81, 0x83, 0, 0, 0, 0, b'a', b'b', b'c']).is_err(),
            "masked"
        );
    }
}
