//! Streaming speech WebSockets (`src/cli/serve/tts_websocket.cpp`,
//! `asr_websocket.cpp`): the vLLM-Omni text-to-speech protocol on
//! `/v1/audio/speech/stream` and OpenAI realtime transcription on
//! `/v1/realtime`.

use std::sync::Arc;

use gufo_core::json::{self, Value};
use gufo_core::text::Result;

use crate::asr_model::{self, AudioBuffer, TranscriptionRequest};
use crate::asr_service::AsrService;
use crate::audio_stream::decode_audio_base64;
use crate::audio_tts_api::{
    buffered_speech, prepare_speech, stream_speech, validate_audio_tts_request,
};
use crate::http::websocket::{WebSocket, upgrade_websocket};
use crate::http::{HttpRequest, HttpResponse};
use crate::logging;
use crate::tts_model::{ModelVariant, take_speech_segments};
use crate::tts_service::TtsService;

fn dump(value: &Value) -> String {
    value.dump().unwrap_or_default()
}

fn speech_request(body: &Value) -> HttpRequest {
    HttpRequest {
        method: "POST".into(),
        path: "/v1/audio/speech".into(),
        body: dump(body).into_bytes(),
        ..HttpRequest::default()
    }
}

/// What the message loop does next.
enum Flow {
    Continue,
    /// `session.close`: end the session normally.
    Break,
    /// The socket is gone.
    Return,
    /// Synthesis failed: close with 1011.
    Fail,
}

struct TtsSession<'a> {
    socket: &'a WebSocket,
    service: &'a TtsService,
    id: &'a str,
    config: Value,
    configured: bool,
    in_utterance: bool,
    streaming: bool,
    granularity: String,
    pending: String,
    input_bytes: usize,
    utterance: usize,
    sentence: usize,
}

impl TtsSession<'_> {
    fn error(&self, message: &str) -> bool {
        let event = Value::object()
            .with("type", "error")
            .with("message", message);
        self.socket.send_text(&dump(&event))
    }

    fn synthesize(&mut self, text: String) -> bool {
        if text.trim_start_matches([' ', '\r', '\n', '\t']).is_empty() {
            return true;
        }
        if self.sentence >= 64 {
            self.error("utterance exceeds 64 speech segments");
            return false;
        }
        let mut body = self.config.clone();
        body.set("input", text.as_str());
        let speech = speech_request(&body);
        let validation = validate_audio_tts_request(&speech, self.service);
        if validation.status != 200 {
            self.error(&String::from_utf8_lossy(&validation.body));
            return false;
        }
        let start = Value::object()
            .with("type", "audio.start")
            .with("utterance_index", self.utterance)
            .with("sentence_index", self.sentence)
            .with("sentence_text", text)
            .with("format", body.member_str("response_format", ""))
            .with("sample_rate", 24000);
        if !self.socket.send_text(&dump(&start)) {
            return false;
        }
        let prepared = match prepare_speech(&speech, self.service) {
            Ok(prepared) => prepared,
            Err(response) => {
                self.error(&String::from_utf8_lossy(&response.body));
                return false;
            }
        };
        let socket = self.socket;
        let cancelled = || socket.cancelled();
        if prepared.pcm {
            let streaming = self.streaming;
            let mut buffered = Vec::new();
            let mut writer = |bytes: &[u8]| {
                if streaming {
                    return socket.send_binary(bytes);
                }
                if socket.cancelled() {
                    return false;
                }
                buffered.extend_from_slice(bytes);
                true
            };
            let log = stream_speech(
                self.service,
                &prepared.synthesis,
                prepared.sse,
                &cancelled,
                &mut writer,
            );
            if !log.error_code.is_empty() {
                self.error(&log.error_code);
                return false;
            }
            if !streaming && !socket.send_binary(&buffered) {
                return false;
            }
        } else {
            let response = buffered_speech(self.service, &prepared.synthesis, &cancelled);
            if response.status != 200 {
                self.error(&String::from_utf8_lossy(&response.body));
                return false;
            }
            if !socket.send_binary(&response.body) {
                return false;
            }
        }
        let done = Value::object()
            .with("type", "audio.done")
            .with("utterance_index", self.utterance)
            .with("sentence_index", self.sentence);
        self.sentence += 1;
        socket.send_text(&dump(&done))
    }

    fn configure(&mut self, message: &Value) -> std::result::Result<(), String> {
        if self.in_utterance {
            return Err("finish the utterance before reconfiguring".into());
        }
        let mut next = Value::object();
        let mut split = "none".to_owned();
        let mut stream_audio = false;
        for (name, value) in message.members() {
            match name.as_str() {
                "type" => {}
                "split_granularity" => {
                    if !value.is_string() || !matches!(value.str(), "none" | "sentence" | "clause")
                    {
                        return Err("split_granularity must be none, sentence or clause".into());
                    }
                    split = value.str().to_owned();
                }
                "stream_audio" => {
                    if !value.is_bool() {
                        return Err("stream_audio must be boolean".into());
                    }
                    stream_audio = value.as_bool(false);
                }
                "word_timestamps" => {
                    if !value.is_bool() || value.as_bool(false) {
                        return Err("timestamps are not supported".into());
                    }
                }
                "input" | "stream_format" => return Err("use input.text and stream_audio".into()),
                "ref_audio" => next.set("reference_audio", value.clone()),
                "ref_text" => next.set("reference_text", value.clone()),
                "x_vector_only_mode" => {
                    if !value.is_bool() {
                        return Err("x_vector_only_mode must be boolean".into());
                    }
                    next.set(
                        "voice_clone_mode",
                        if value.as_bool(false) {
                            "speaker_embedding_only"
                        } else {
                            "icl"
                        },
                    );
                }
                "task_type" => {
                    let expected = match self.service.variant() {
                        ModelVariant::Base => "Base",
                        ModelVariant::VoiceDesign => "VoiceDesign",
                        _ => "CustomVoice",
                    };
                    if !value.is_string() || value.str() != expected {
                        return Err("task_type does not match the loaded checkpoint".into());
                    }
                }
                _ => next.set(name, value.clone()),
            }
        }
        if next.find("model").is_none() {
            next.set("model", self.service.model_id());
        }
        if next.find("response_format").is_none() {
            next.set("response_format", "wav");
        }
        if stream_audio && next.member_str("response_format", "") != "pcm" {
            return Err("stream_audio requires response_format pcm".into());
        }
        next.set("input", "validate");
        let validation = validate_audio_tts_request(&speech_request(&next), self.service);
        if validation.status != 200 {
            return Err(String::from_utf8_lossy(&validation.body).into_owned());
        }
        self.config = next;
        self.granularity = split;
        self.streaming = stream_audio;
        self.configured = true;
        Ok(())
    }

    fn handle(&mut self, raw: &str) -> std::result::Result<Flow, String> {
        let message = json::parse_str(raw).map_err(|e| e.message().to_owned())?;
        if !message.is_object() {
            return Err("message must be an object".into());
        }
        let kind = message.member_str("type", "");
        if kind != "session.config" {
            for (name, _) in message.members() {
                if name != "type" && !(kind == "input.text" && name == "text") {
                    return Err(format!("unsupported speech stream field: {name}"));
                }
            }
        }
        match kind.as_str() {
            "session.close" => Ok(Flow::Break),
            "session.config" => {
                self.configure(&message)?;
                Ok(Flow::Continue)
            }
            "input.text" => {
                if !self.configured {
                    return Err("send session.config first".into());
                }
                let Some(text) = message.find("text").filter(|t| t.is_string()) else {
                    return Err("input.text requires text".into());
                };
                if text.str().len() > (16 << 10) - self.input_bytes {
                    return Err("utterance exceeds 16384 UTF-8 bytes".into());
                }
                self.input_bytes += text.str().len();
                self.in_utterance = true;
                self.pending.push_str(text.str());
                let granularity = self.granularity.clone();
                for part in take_speech_segments(&mut self.pending, &granularity, false) {
                    if !self.synthesize(part) {
                        return Ok(Flow::Fail);
                    }
                }
                Ok(Flow::Continue)
            }
            "input.done" => {
                if !self.configured {
                    return Err("send session.config first".into());
                }
                let granularity = self.granularity.clone();
                for part in take_speech_segments(&mut self.pending, &granularity, true) {
                    if !self.synthesize(part) {
                        return Ok(Flow::Fail);
                    }
                }
                let done = Value::object()
                    .with("type", "session.done")
                    .with("utterance_index", self.utterance)
                    .with("total_sentences", self.sentence);
                self.utterance += 1;
                if !self.socket.send_text(&dump(&done)) {
                    return Ok(Flow::Return);
                }
                logging::info(
                    "tts",
                    &format!(
                        "request={} event=utterance_complete segments={} input_bytes={}",
                        self.id, self.sentence, self.input_bytes
                    ),
                );
                self.sentence = 0;
                self.input_bytes = 0;
                self.in_utterance = false;
                Ok(Flow::Continue)
            }
            _ => Err("unsupported speech stream event".into()),
        }
    }
}

fn tts_session(socket: &WebSocket, service: &TtsService, id: &str) {
    let mut session = TtsSession {
        socket,
        service,
        id,
        config: Value::object(),
        configured: false,
        in_utterance: false,
        streaming: false,
        granularity: "none".into(),
        pending: String::new(),
        input_bytes: 0,
        utterance: 0,
        sentence: 0,
    };
    while let Some(raw) = socket.receive() {
        match session.handle(&raw) {
            Ok(Flow::Continue) => {}
            Ok(Flow::Break) => break,
            Ok(Flow::Return) => return,
            Ok(Flow::Fail) => {
                socket.close(1011);
                return;
            }
            Err(message) => {
                if !session.error(&message) {
                    return;
                }
            }
        }
    }
}

/// `/v1/audio/speech/stream`.
pub fn handle_tts_websocket(
    request: &HttpRequest,
    service: &Arc<TtsService>,
) -> Result<HttpResponse> {
    let service = service.clone();
    let id = request.request_id.clone();
    Ok(upgrade_websocket(
        request,
        Box::new(move |socket: &WebSocket| tts_session(socket, &service, &id)),
    ))
}

/// Rejects members of `object` outside `allowed`.
fn members(object: &Value, allowed: &[&str]) -> std::result::Result<(), String> {
    if !object.is_object() {
        return Err("configuration must be an object".into());
    }
    match object
        .members()
        .iter()
        .find(|(name, _)| !allowed.contains(&name.as_str()))
    {
        Some((name, _)) => Err(format!("unsupported transcription field: {name}")),
        None => Ok(()),
    }
}

const MAXIMUM_AUDIO_BYTES: usize = 120 * 24000 * 2;

struct AsrSession<'a> {
    socket: &'a WebSocket,
    service: &'a AsrService,
    id: &'a str,
    sequence: usize,
    item_sequence: usize,
    previous_item: String,
    audio_bytes: Vec<u8>,
    language: Option<String>,
    prompt: String,
    config: Value,
}

impl AsrSession<'_> {
    fn send(&mut self, mut event: Value) -> bool {
        self.sequence += 1;
        event.set("event_id", format!("event_{}_{}", self.id, self.sequence));
        self.socket.send_text(&dump(&event))
    }

    fn session_event(&mut self, kind: &str) -> bool {
        let event = Value::object()
            .with("type", kind)
            .with("session", self.config.clone());
        self.send(event)
    }

    fn error(&mut self, message: &str, client_event: &str) -> bool {
        let mut detail = Value::object()
            .with("type", "invalid_request_error")
            .with("code", "invalid_audio_event")
            .with("message", message);
        if !client_event.is_empty() {
            detail.set("event_id", client_event);
        }
        let event = Value::object().with("type", "error").with("error", detail);
        self.send(event)
    }

    fn previous_item_value(&self) -> Value {
        if self.previous_item.is_empty() {
            Value::Null
        } else {
            Value::from(self.previous_item.as_str())
        }
    }

    fn update(&mut self, message: &Value) -> std::result::Result<bool, String> {
        members(message, &["type", "event_id", "session"])?;
        if !self.audio_bytes.is_empty() {
            return Err("commit or clear audio before session.update".into());
        }
        let Some(session) = message.find("session") else {
            return Err("session.update requires session".into());
        };
        members(session, &["type", "audio", "include"])?;
        if session
            .find("type")
            .is_some_and(|k| !k.is_string() || k.str() != "transcription")
        {
            return Err("only transcription sessions are supported".into());
        }
        if session
            .find("include")
            .is_some_and(|i| !i.is_array() || !i.items().is_empty())
        {
            return Err("transcription logprobs are not supported".into());
        }
        let mut next = self.config.clone();
        let mut next_language = self.language.clone();
        let mut next_prompt = self.prompt.clone();
        if let Some(audio) = session.find("audio") {
            members(audio, &["input"])?;
            if let Some(input) = audio.find("input") {
                members(
                    input,
                    &[
                        "format",
                        "transcription",
                        "turn_detection",
                        "noise_reduction",
                    ],
                )?;
                if let Some(format) = input.find("format") {
                    members(format, &["type", "rate"])?;
                    if format.member_str("type", "") != "audio/pcm"
                        || format
                            .find("rate")
                            .is_some_and(|r| !r.is_number() || r.as_double(0.0) != 24000.0)
                    {
                        return Err("realtime input requires mono PCM16 at 24000 Hz".into());
                    }
                }
                for name in ["turn_detection", "noise_reduction"] {
                    if input.find(name).is_some_and(|o| !o.is_null()) {
                        return Err(format!("{name} must be null"));
                    }
                }
                if let Some(transcription) = input.find("transcription") {
                    members(transcription, &["model", "language", "prompt"])?;
                    for (name, value) in transcription.members() {
                        if !value.is_string() {
                            return Err(format!("{name} must be a string"));
                        }
                        if name == "model"
                            && value.str() != self.service.model_id()
                            && value.str() != "qwen3-asr"
                        {
                            return Err("unsupported transcription model".into());
                        }
                        if name == "language" {
                            next_language =
                                (!value.str().is_empty()).then(|| value.str().to_owned());
                        }
                        if name == "prompt" {
                            next_prompt = value.str().to_owned();
                        }
                        next.member_mut("audio")
                            .member_mut("input")
                            .member_mut("transcription")
                            .set(name, value.clone());
                    }
                }
            }
        }
        if next_prompt.len() > 16384 || next_language.as_ref().is_some_and(|l| l.len() > 64) {
            return Err("transcription configuration exceeds its text limit".into());
        }
        self.config = next;
        self.prompt = next_prompt;
        self.language = next_language;
        Ok(self.session_event("session.updated"))
    }

    fn commit(&mut self, message: &Value) -> std::result::Result<bool, String> {
        members(message, &["type", "event_id"])?;
        if self.audio_bytes.len() < 2400 * 2 {
            return Err("commit requires at least 100 ms of audio".into());
        }
        self.item_sequence += 1;
        let item = format!("item_{}_{}", self.id, self.item_sequence);
        let committed = Value::object()
            .with("type", "input_audio_buffer.committed")
            .with("item_id", item.as_str())
            .with("previous_item_id", self.previous_item_value());
        if !self.send(committed) {
            return Ok(false);
        }
        let mut created_item = json::parse_str(
            r#"{"type":"message","role":"user","status":"completed","content":[{"type":"input_audio","transcript":null}]}"#,
        )
        .map_err(|e| e.message().to_owned())?;
        created_item.set("id", item.as_str());
        let created = Value::object()
            .with("type", "conversation.item.created")
            .with("previous_item_id", self.previous_item_value())
            .with("item", created_item);
        if !self.send(created) {
            return Ok(false);
        }
        self.previous_item = item.clone();
        let audio = AudioBuffer {
            sample_rate: 24000,
            channels: 1,
            samples: self
                .audio_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32768.0)
                .collect(),
        };
        self.audio_bytes.clear();
        let pcm = asr_model::resample_mono_16k(&audio)?;
        let samples = pcm.len();
        let transcription = TranscriptionRequest {
            context: self.prompt.clone(),
            language: self.language.clone(),
            pcm16k: pcm,
            ..TranscriptionRequest::default()
        };
        let socket = self.socket;
        let service = self.service;
        let mut published = String::new();
        let outcome = {
            let mut on_text = |text: &str| {
                if !text.starts_with(published.as_str()) {
                    return false;
                }
                if text.len() == published.len() {
                    return !socket.cancelled();
                }
                let delta = Value::object()
                    .with("type", "conversation.item.input_audio_transcription.delta")
                    .with("item_id", item.as_str())
                    .with("content_index", 0)
                    .with("delta", &text[published.len()..]);
                published = text.to_owned();
                self.send(delta)
            };
            service.transcribe(&transcription, Some(&mut on_text), &|| socket.cancelled())
        };
        if socket.cancelled() {
            return Ok(false);
        }
        let ok = outcome.is_ok();
        let mut completed = Value::object()
            .with(
                "type",
                if ok {
                    "conversation.item.input_audio_transcription.completed"
                } else {
                    "conversation.item.input_audio_transcription.failed"
                },
            )
            .with("item_id", item.as_str())
            .with("content_index", 0);
        let chunks = match &outcome {
            Ok(result) => {
                completed.set("transcript", result.text.as_str());
                result.chunks
            }
            Err(failure) => {
                completed.set(
                    "error",
                    Value::object()
                        .with("type", "server_error")
                        .with("code", "transcription_failed")
                        .with("message", failure.as_str()),
                );
                0
            }
        };
        if !self.send(completed) {
            return Ok(false);
        }
        logging::info(
            "asr",
            &format!(
                "request={} event=transcription_complete item={item} chunks={chunks} \
                 audio_samples={samples} outcome={}",
                self.id,
                if ok { "completed" } else { "failed" }
            ),
        );
        Ok(true)
    }

    /// Handles one client event; `Ok(false)` once the socket is gone.
    fn handle(&mut self, message: &Value) -> std::result::Result<bool, String> {
        match message.member_str("type", "").as_str() {
            "session.update" => self.update(message),
            "input_audio_buffer.append" => {
                members(message, &["type", "event_id", "audio"])?;
                let bytes = message
                    .find("audio")
                    .filter(|a| a.is_string())
                    .and_then(|a| {
                        decode_audio_base64(
                            a.str().as_bytes(),
                            MAXIMUM_AUDIO_BYTES - self.audio_bytes.len(),
                        )
                    })
                    .filter(|b| b.len() % 2 == 0)
                    .ok_or("invalid PCM16 audio or 120-second input buffer exceeded")?;
                self.audio_bytes.extend_from_slice(&bytes);
                Ok(true)
            }
            "input_audio_buffer.clear" => {
                members(message, &["type", "event_id"])?;
                self.audio_bytes.clear();
                Ok(self.send(Value::object().with("type", "input_audio_buffer.cleared")))
            }
            "input_audio_buffer.commit" => self.commit(message),
            _ => Err("unsupported realtime transcription event".into()),
        }
    }
}

fn asr_session(socket: &WebSocket, service: &AsrService, id: &str) {
    let config = json::parse_str(
        r#"{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{},"turn_detection":null,"noise_reduction":null}}}"#,
    );
    let Ok(mut config) = config else {
        return;
    };
    config.set("id", format!("sess_{id}"));
    config
        .member_mut("audio")
        .member_mut("input")
        .member_mut("transcription")
        .set("model", service.model_id());
    let mut session = AsrSession {
        socket,
        service,
        id,
        sequence: 0,
        item_sequence: 0,
        previous_item: String::new(),
        audio_bytes: Vec::new(),
        language: None,
        prompt: String::new(),
        config,
    };
    if !session.session_event("session.created") {
        return;
    }
    while let Some(raw) = socket.receive() {
        let mut client_event = String::new();
        let outcome = json::parse_str(&raw)
            .map_err(|e| e.message().to_owned())
            .and_then(|message| {
                if !message.is_object() {
                    return Err("event must be an object".into());
                }
                client_event = message.member_str("event_id", "");
                session.handle(&message)
            });
        match outcome {
            Ok(true) => {}
            Ok(false) => return,
            Err(message) => {
                if !session.error(&message, &client_event) {
                    return;
                }
            }
        }
    }
}

/// `/v1/realtime?intent=transcription`.
pub fn handle_asr_websocket(
    request: &HttpRequest,
    service: &Arc<AsrService>,
) -> Result<HttpResponse> {
    let model = request.query_param("model");
    let intent = request.query_param("intent");
    if (!model.is_empty() && model != service.model_id() && model != "qwen3-asr")
        || (!intent.is_empty() && intent != "transcription")
    {
        return Ok(HttpResponse::new(
            400,
            "Bad Request",
            r#"{"error":{"message":"unsupported realtime model or intent","code":"invalid_model"}}"#,
        ));
    }
    let service = service.clone();
    let id = request.request_id.clone();
    Ok(upgrade_websocket(
        request,
        Box::new(move |socket: &WebSocket| asr_session(socket, &service, &id)),
    ))
}
