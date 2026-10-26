//! Port of `tests/cli/audio_websocket_test.cpp`: both speech WebSockets end
//! to end through a running server.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gufo_core::json::{self, Value};
use gufo_serve::asr_model::TranscriptionResult;
use gufo_serve::asr_service::{AsrService, AsrServiceOptions};
use gufo_serve::audio_stream::{encode_audio_base64, encode_pcm16};
use gufo_serve::http::HttpServerOptions;
use gufo_serve::http::server::{HttpServer, HttpServices};
use gufo_serve::tts_model::SynthesisResult;
use gufo_serve::tts_service::{TtsService, TtsServiceOptions};

const KEY: &str = "dGhlIHNhbXBsZSBub25jZQ==";

struct Client {
    stream: TcpStream,
    headers: String,
}

impl Client {
    fn new(port: u16, path: &str, first: &[u8], key: &str, authorization: &str) -> Self {
        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("timeout");
        let mut client = Self {
            stream,
            headers: String::new(),
        };
        let mut request = format!(
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nUpgrade: other,\twebsocket\r\n\
             Connection: keep-alive,\tUpgrade\r\nSec-WebSocket-Version: 13\r\n\
             Sec-WebSocket-Key: {key}\r\nAuthorization: {authorization}\r\n\r\n"
        )
        .into_bytes();
        request.extend_from_slice(first);
        client.send(&request);
        while !client.headers.ends_with("\r\n\r\n") {
            let mut byte = [0u8; 1];
            client.read(&mut byte);
            client.headers.push(char::from(byte[0]));
        }
        client
    }

    fn connect(port: u16, path: &str) -> Self {
        Self::new(port, path, b"", KEY, "Bearer test")
    }

    fn send(&mut self, bytes: &[u8]) {
        self.stream.write_all(bytes).expect("send");
    }

    fn frame(payload: &[u8], opcode: u8, last: bool, masked: bool) -> Vec<u8> {
        let mut frame = vec![if last { 128 } else { 0 } | opcode];
        let mask = if masked { 128 } else { 0 };
        if payload.len() < 126 {
            frame.push(mask | payload.len() as u8);
        } else {
            assert!(payload.len() <= 65535);
            frame.push(mask | 126);
            frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        }
        if masked {
            frame.extend_from_slice(b"abcd");
        }
        for (i, &byte) in payload.iter().enumerate() {
            frame.push(if masked { byte ^ b"abcd"[i % 4] } else { byte });
        }
        frame
    }

    fn text(&mut self, text: &str) {
        let frame = Self::frame(text.as_bytes(), 1, true, true);
        self.send(&frame);
    }

    fn read(&mut self, bytes: &mut [u8]) {
        self.stream.read_exact(bytes).expect("read");
    }

    fn receive(&mut self) -> (u8, Vec<u8>) {
        let mut head = [0u8; 2];
        self.read(&mut head);
        assert!(head[0] & 128 != 0 && head[1] & 128 == 0);
        let mut size = usize::from(head[1] & 127);
        if size >= 126 {
            let count = if size == 126 { 2 } else { 8 };
            size = 0;
            for _ in 0..count {
                let mut byte = [0u8; 1];
                self.read(&mut byte);
                size = size * 256 + usize::from(byte[0]);
            }
        }
        assert!(size < 4 << 20);
        let mut payload = vec![0u8; size];
        self.read(&mut payload);
        (head[0] & 15, payload)
    }

    fn event(&mut self, kind: &str) -> Value {
        let (opcode, payload) = self.receive();
        assert_eq!(opcode, 1);
        let event = json::parse(&payload).expect("event");
        assert_eq!(
            event.member_str("type", ""),
            kind,
            "{}",
            String::from_utf8_lossy(&payload)
        );
        event
    }
}

#[derive(Default)]
struct AsrCalls {
    pcm_samples: Vec<usize>,
    languages: Vec<Option<String>>,
}

#[test]
fn speech_websockets() {
    let tts_calls = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(AtomicBool::new(false));
    let cancelled = Arc::new(AtomicBool::new(false));
    let asr_calls = Arc::new(Mutex::new(AsrCalls::default()));
    let tts = {
        let (tts_calls, entered, cancelled) =
            (tts_calls.clone(), entered.clone(), cancelled.clone());
        Arc::new(TtsService::new(TtsServiceOptions {
            validate_model: false,
            voices: vec!["vivian".into()],
            runner: Some(Box::new(move |request, on_audio, check| {
                tts_calls.fetch_add(1, Ordering::SeqCst);
                if request.text == "wait" {
                    entered.store(true, Ordering::SeqCst);
                    let deadline = Instant::now() + Duration::from_secs(2);
                    while !check() && Instant::now() < deadline {
                        std::thread::yield_now();
                    }
                    cancelled.store(check(), Ordering::SeqCst);
                    return Err(String::new());
                }
                let result = SynthesisResult {
                    sample_rate: 24000,
                    code_groups: 16,
                    samples: vec![-0.5, 0.0, 0.5],
                    ..SynthesisResult::default()
                };
                if let Some(sink) = on_audio
                    && !(sink(&result.samples) && sink(&result.samples))
                {
                    return Err(String::new());
                }
                Ok(result)
            })),
            ..TtsServiceOptions::default()
        }))
    };
    let asr = {
        let asr_calls = asr_calls.clone();
        Arc::new(AsrService::new(AsrServiceOptions {
            validate_model: false,
            runner: Some(Box::new(move |request, on_text, _| {
                {
                    let mut calls = asr_calls.lock().expect("calls");
                    calls.pcm_samples.push(request.pcm16k.len());
                    calls.languages.push(request.language.clone());
                }
                let result = TranscriptionResult {
                    text: "Hello world.".into(),
                    chunks: 1,
                    ..TranscriptionResult::default()
                };
                if let Some(sink) = on_text
                    && !(sink("Hello") && sink(&result.text))
                {
                    return Err(String::new());
                }
                Ok(result)
            })),
            ..AsrServiceOptions::default()
        }))
    };
    let server = HttpServer::new(
        "127.0.0.1",
        0,
        None,
        HttpServices {
            tts: Some(tts),
            asr: Some(asr),
            ..HttpServices::default()
        },
        HttpServerOptions {
            api_key: "test".into(),
            ..HttpServerOptions::default()
        },
    )
    .expect("server");
    server.start().expect("start");
    let port = server.port();
    std::thread::scope(|scope| {
        scope.spawn(|| server.run(false));
        {
            let denied = Client::new(port, "/v1/realtime", b"", KEY, "");
            assert!(
                denied.headers.starts_with("HTTP/1.1 401"),
                "{}",
                denied.headers
            );
            let invalid = Client::new(port, "/v1/realtime", b"", "bad", "Bearer test");
            assert!(
                invalid.headers.starts_with("HTTP/1.1 400"),
                "{}",
                invalid.headers
            );
        }
        {
            let first = Client::frame(
                br#"{"type":"session.update","session":{"type":"transcription","audio":{"input":{"transcription":{"model":"qwen3-asr","language":"en"},"turn_detection":null}}}}"#,
                1,
                true,
                true,
            );
            let mut client = Client::new(
                port,
                "/v1/realtime?intent=transcription",
                &first,
                KEY,
                "Bearer test",
            );
            assert!(
                client.headers.starts_with("HTTP/1.1 101"),
                "{}",
                client.headers
            );
            assert!(client.headers.contains("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
            assert!(!client.headers.contains("Connection: close"));
            client.event("session.created");
            client.event("session.updated");
            let mut previous = String::new();
            for _ in 0..2 {
                let encoded = encode_audio_base64(&[0u8; 4800]);
                client.text(&format!(
                    r#"{{"type":"input_audio_buffer.append","audio":"{encoded}"}}"#
                ));
                client.text(r#"{"type":"input_audio_buffer.commit"}"#);
                let committed = client.event("input_audio_buffer.committed");
                assert_eq!(committed.member_str("previous_item_id", ""), previous);
                previous = committed.member_str("item_id", "");
                client.event("conversation.item.created");
                assert_eq!(
                    client
                        .event("conversation.item.input_audio_transcription.delta")
                        .member_str("delta", ""),
                    "Hello"
                );
                assert_eq!(
                    client
                        .event("conversation.item.input_audio_transcription.delta")
                        .member_str("delta", ""),
                    " world."
                );
                assert_eq!(
                    client
                        .event("conversation.item.input_audio_transcription.completed")
                        .member_str("transcript", ""),
                    "Hello world."
                );
            }
            client.text(
                r#"{"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"server_vad"}}}}}"#,
            );
            client.event("error");
            client.text(r#"{"type":"input_audio_buffer.commit"}"#);
            client.event("error");
            let calls = asr_calls.lock().expect("calls");
            assert_eq!(calls.pcm_samples, [1600, 1600]);
            assert!(calls.languages.iter().all(|l| l.as_deref() == Some("en")));
        }
        {
            let mut client = Client::connect(port, "/v1/audio/speech/stream");
            client.text(
                r#"{"type":"session.config","voice":"vivian","response_format":"pcm","stream_audio":true,"split_granularity":"sentence"}"#,
            );
            let message = r#"{"type":"input.text","text":"Hello é. "}"#.as_bytes();
            let split = message
                .windows(2)
                .position(|w| w == "é".as_bytes())
                .expect("é")
                + 1;
            client.send(&Client::frame(&message[..split], 1, false, true));
            client.send(&Client::frame(b"ping", 9, true, true));
            assert_eq!(client.receive(), (10, b"ping".to_vec()));
            client.send(&Client::frame(&message[split..], 0, true, true));
            assert_eq!(
                client.event("audio.start").member_str("sentence_text", ""),
                "Hello é."
            );
            let expected = encode_pcm16(&[-0.5, 0.0, 0.5]);
            assert_eq!(client.receive(), (2, expected.clone()));
            assert_eq!(client.receive(), (2, expected));
            client.event("audio.done");
            client.text(r#"{"type":"input.done"}"#);
            client.event("session.done");
            client.text(
                r#"{"type":"session.config","voice":"vivian","response_format":"pcm","stream_audio":true}"#,
            );
            client.text(r#"{"type":"input.text","text":"Next utterance"}"#);
            client.text(r#"{"type":"input.done"}"#);
            client.event("audio.start");
            client.receive();
            client.receive();
            client.event("audio.done");
            client.event("session.done");
            assert_eq!(tts_calls.load(Ordering::SeqCst), 2);
            client.send(&Client::frame(b"unmasked", 1, true, false));
            let (opcode, close) = client.receive();
            assert!(opcode == 8 && close == [0x03, 0xEA]);
        }
        {
            let mut client = Client::connect(port, "/v1/audio/speech/stream");
            client.text(
                r#"{"type":"session.config","voice":"vivian","response_format":"pcm","stream_audio":true}"#,
            );
            client.text(r#"{"type":"input.text","text":"wait"}"#);
            client.text(r#"{"type":"input.done"}"#);
            client.event("audio.start");
            let deadline = Instant::now() + Duration::from_secs(2);
            while !entered.load(Ordering::SeqCst) && Instant::now() < deadline {
                std::thread::yield_now();
            }
            assert!(entered.load(Ordering::SeqCst));
            client.send(&Client::frame(&[0x03, 0xE8], 8, true, true));
            assert_eq!(client.receive().0, 8);
            while !cancelled.load(Ordering::SeqCst) && Instant::now() < deadline {
                std::thread::yield_now();
            }
            assert!(cancelled.load(Ordering::SeqCst));
        }
        // Closure before or while the consumer waits for its first message:
        // every connection must release its worker and admission slot.
        for _ in 0..32 {
            let mut client = Client::connect(port, "/v1/audio/speech/stream");
            client.send(&Client::frame(&[0x03, 0xE8], 8, true, true));
            let reply = client.receive();
            assert!(reply.0 == 8 && reply.1 == [0x03, 0xE8]);
        }
        server.stop();
    });
}

/// The server's receive timeout is an idle limit. A client that only listens
/// to streamed output must stay connected; a silent session closes cleanly.
#[test]
fn listening_client_is_not_idle() {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind");
    let mut client = TcpStream::connect(listener.local_addr().expect("address")).expect("connect");
    let (server, _) = listener.accept().expect("accept");
    server
        .set_read_timeout(Some(Duration::from_millis(300)))
        .expect("idle limit");
    client
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("client timeout");
    let socket = gufo_serve::http::websocket::WebSocket::new(server, Vec::new()).expect("socket");
    // Unmasked final binary frames with a five-byte payload.
    let streaming_until = Instant::now() + Duration::from_millis(1200);
    while Instant::now() < streaming_until {
        assert!(socket.send_binary(b"audio"));
        let mut frame = [0u8; 7];
        client.read_exact(&mut frame).expect("frame");
        assert_eq!(&frame, b"\x82\x05audio");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!socket.cancelled());
    let mut close = [0u8; 4];
    client.read_exact(&mut close).expect("close frame");
    assert_eq!(&close, b"\x88\x02\x03\xe9");
    assert!(socket.cancelled());
}
