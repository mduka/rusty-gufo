//! `/v1/audio/speech` and `/v1/audio/voices` over Qwen3-TTS
//! (`src/cli/serve/audio_tts_api.cpp`): request validation, buffered WAV,
//! streamed PCM and OpenAI-style SSE audio events.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use gufo_core::json::{self, Value};
use gufo_core::text::Result;

use crate::audio_stream::{encode_audio_base64, encode_pcm16};
use crate::http::{BodyWriter, HttpRequest, HttpResponse, StreamLog};
use crate::tts_model::{self, AudioBuffer, ModelVariant, SamplingOptions, SynthesisRequest};
use crate::tts_service::TtsService;

pub const AUDIO_TTS_API_SCHEMA: &str = "gufo.audio-tts-api.v1";

const SPEECH_PATH: &str = "/v1/audio/speech";
const VOICES_PATH: &str = "/v1/audio/voices";
const MAXIMUM_INPUT_BYTES: usize = 16 << 10;
const MAXIMUM_NEW_TOKENS: usize = 8192;
const ALLOWED_FIELDS: [&str; 22] = [
    "model",
    "input",
    "voice",
    "response_format",
    "speed",
    "language",
    "instructions",
    "seed",
    "max_new_tokens",
    "greedy",
    "reference_audio",
    "reference_text",
    "voice_clone_mode",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "subtalker_dosample",
    "subtalker_temperature",
    "subtalker_top_k",
    "subtalker_top_p",
    "stream_format",
];

fn dump(value: &Value) -> String {
    value.dump().unwrap_or_default()
}

fn error(status: u16, reason: &str, message: &str, code: &str) -> HttpResponse {
    let root = Value::object().with(
        "error",
        Value::object()
            .with("message", message)
            .with("type", "invalid_request_error")
            .with("code", code),
    );
    HttpResponse::new(status, reason, dump(&root))
}

/// `default` when absent; `None` unless a nonnegative integer up to
/// `maximum`.
fn read_unsigned(body: &Value, name: &str, default: usize, maximum: usize) -> Option<usize> {
    let Some(value) = body.find(name) else {
        return Some(default);
    };
    if !value.is_number() {
        return None;
    }
    let number = value.as_double(0.0);
    if !number.is_finite() || number < 0.0 || number.floor() != number || number > maximum as f64 {
        return None;
    }
    Some(number as usize)
}

/// Leaves `output` unchanged when absent; false unless a finite `f32`.
fn read_float(body: &Value, name: &str, output: &mut f32) -> bool {
    let Some(value) = body.find(name) else {
        return true;
    };
    if !value.is_number() {
        return false;
    }
    *output = value.as_double(0.0) as f32;
    output.is_finite()
}

/// Mono PCM16 WAV; empty for a non-finite sample or an impossible size.
fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    const BLOCK_ALIGN: u32 = 2;
    if sample_rate == 0 || samples.len() > (u32::MAX / BLOCK_ALIGN) as usize {
        return Vec::new();
    }
    let data_bytes = samples.len() as u32 * BLOCK_ALIGN;
    let mut output = Vec::with_capacity(44 + data_bytes as usize);
    output.extend_from_slice(b"RIFF");
    output.extend_from_slice(&(36u32.wrapping_add(data_bytes)).to_le_bytes());
    output.extend_from_slice(b"WAVEfmt ");
    output.extend_from_slice(&16u32.to_le_bytes());
    output.extend_from_slice(&1u16.to_le_bytes());
    output.extend_from_slice(&1u16.to_le_bytes());
    output.extend_from_slice(&sample_rate.to_le_bytes());
    output.extend_from_slice(&sample_rate.wrapping_mul(BLOCK_ALIGN).to_le_bytes());
    output.extend_from_slice(&(BLOCK_ALIGN as u16).to_le_bytes());
    output.extend_from_slice(&16u16.to_le_bytes());
    output.extend_from_slice(b"data");
    output.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        if !sample.is_finite() {
            return Vec::new();
        }
        let pcm = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        output.extend_from_slice(&pcm.to_le_bytes());
    }
    output
}

fn voices(service: &TtsService) -> HttpResponse {
    let mut data = Value::array();
    let mut names = Value::array();
    for voice in service.voices() {
        data.push(
            Value::object()
                .with("id", voice.as_str())
                .with("object", "voice")
                .with("model", service.model_id()),
        );
        names.push(voice);
    }
    // `data` is the OpenAI-style envelope the rest of this API uses; `voices`
    // repeats the names as a flat array for clients (llama-swap, for one)
    // that accept only a bare array or a `voices` key. Keep both.
    let root = Value::object()
        .with("object", "list")
        .with("schema", AUDIO_TTS_API_SCHEMA)
        .with("data", data)
        .with("voices", names);
    HttpResponse::new(200, "OK", dump(&root))
}

/// A validated speech request.
pub(crate) struct PreparedSpeech {
    pub synthesis: SynthesisRequest,
    /// Streamed PCM (with or without SSE framing) instead of a WAV body.
    pub pcm: bool,
    pub sse: bool,
}

/// Validates a speech request into a synthesis request, or the error
/// response.
pub(crate) fn prepare_speech(
    request: &HttpRequest,
    service: &TtsService,
) -> std::result::Result<PreparedSpeech, Box<HttpResponse>> {
    let Ok(body) = json::parse(&request.body) else {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "request body must be valid JSON",
            "parse_error",
        )));
    };
    if !body.is_object()
        || !body
            .members()
            .iter()
            .all(|(name, _)| ALLOWED_FIELDS.contains(&name.as_str()))
    {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "request contains unsupported audio speech fields",
            "unsupported_field",
        )));
    }
    let model = body.find("model");
    let input = body.find("input");
    let voice = body.find("voice");
    let (Some(model), Some(input)) = (
        model.filter(|m| m.is_string()),
        input.filter(|i| i.is_string()),
    ) else {
        return Err(Box::new(invalid_types()));
    };
    if voice.is_some_and(|v| !v.is_string()) {
        return Err(Box::new(invalid_types()));
    }
    if model.str() != service.model_id() && model.str() != "qwen3-tts" {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "unsupported audio speech model",
            "invalid_model",
        )));
    }
    if input.str().is_empty() {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "'input' must not be empty",
            "missing_input",
        )));
    }
    if input.str().len() > MAXIMUM_INPUT_BYTES {
        return Err(Box::new(error(
            413,
            "Payload Too Large",
            "audio speech input exceeds 16384 UTF-8 bytes",
            "input_too_large",
        )));
    }
    let voices = service.voices();
    if service.variant() == ModelVariant::CustomVoice && voice.is_none() {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "CustomVoice requests require a 'voice' speaker",
            "missing_voice",
        )));
    }
    let selected_voice = match voice {
        Some(voice) => voice.str().to_owned(),
        None => voices.first().cloned().unwrap_or_default(),
    };
    if !voices.contains(&selected_voice) {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "unsupported Qwen3-TTS voice",
            "invalid_voice",
        )));
    }
    if service.variant() == ModelVariant::VoiceDesign
        && body.member_str("instructions", "").is_empty()
    {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "VoiceDesign requests require non-empty 'instructions'",
            "missing_instruction",
        )));
    }
    let mut reference_audio = AudioBuffer::default();
    let mut reference_text = String::new();
    let mut speaker_embedding_only = false;
    let preset = service.voice_preset(&selected_voice);
    let has_reference_fields = body.contains("reference_audio")
        || body.contains("reference_text")
        || body.contains("voice_clone_mode");
    match (service.variant(), preset) {
        (ModelVariant::Base, Some(preset)) => {
            // The preset already carries reference audio: accepting
            // per-request reference fields too would leave which one wins
            // ambiguous.
            if has_reference_fields {
                return Err(Box::new(error(
                    400,
                    "Bad Request",
                    &format!(
                        "voice '{selected_voice}' is a preset that already supplies reference audio"
                    ),
                    "unsupported_field",
                )));
            }
            reference_audio = preset.reference_audio.clone();
            reference_text = preset.reference_text.clone();
            speaker_embedding_only = preset.speaker_embedding_only;
        }
        (ModelVariant::Base, None) => {
            let encoded_audio = body.find("reference_audio");
            let clone_text = body.find("reference_text");
            let clone_mode = body.find("voice_clone_mode");
            let Some(encoded_audio) = encoded_audio.filter(|a| a.is_string()) else {
                return Err(Box::new(error(
                    400,
                    "Bad Request",
                    "Base voice cloning requires base64 WAV 'reference_audio'",
                    "missing_reference_audio",
                )));
            };
            if clone_mode.is_some_and(|m| {
                !m.is_string() || (m.str() != "icl" && m.str() != "speaker_embedding_only")
            }) {
                return Err(Box::new(error(
                    400,
                    "Bad Request",
                    "'voice_clone_mode' must be 'icl' or 'speaker_embedding_only'",
                    "invalid_voice_clone_mode",
                )));
            }
            speaker_embedding_only =
                clone_mode.is_some_and(|m| m.str() == "speaker_embedding_only");
            if !speaker_embedding_only
                && !clone_text.is_some_and(|t| t.is_string() && !t.str().is_empty())
            {
                return Err(Box::new(error(
                    400,
                    "Bad Request",
                    "Base ICL voice cloning requires non-empty 'reference_text'",
                    "missing_reference_text",
                )));
            }
            if clone_text.is_some_and(|t| !t.is_string()) {
                return Err(Box::new(error(
                    400,
                    "Bad Request",
                    "'reference_text' must be a string",
                    "invalid_parameter_type",
                )));
            }
            reference_audio =
                tts_model::decode_base64_wav(encoded_audio.str()).map_err(|message| {
                    Box::new(error(
                        400,
                        "Bad Request",
                        &message,
                        "invalid_reference_audio",
                    ))
                })?;
            reference_text = clone_text.map(|t| t.str().to_owned()).unwrap_or_default();
        }
        _ if has_reference_fields => {
            return Err(Box::new(error(
                400,
                "Bad Request",
                "reference audio fields are valid only for the Base model",
                "unsupported_field",
            )));
        }
        _ => {}
    }
    let stream_format = body.find("stream_format");
    if stream_format.is_some_and(|f| !f.is_string() || (f.str() != "audio" && f.str() != "sse")) {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "'stream_format' must be 'audio' or 'sse'",
            "invalid_stream_format",
        )));
    }
    let sse = stream_format.is_some_and(|f| f.str() == "sse");
    let format = body.find("response_format");
    let pcm = format.map_or(sse, |f| f.str() == "pcm");
    if format.is_some_and(|f| !f.is_string() || (f.str() != "wav" && f.str() != "pcm")) {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "'response_format' must be 'wav' or 'pcm'",
            "invalid_response_format",
        )));
    }
    if sse && !pcm {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "SSE audio requires response_format 'pcm'",
            "invalid_response_format",
        )));
    }
    if body
        .find("speed")
        .is_some_and(|s| !s.is_number() || s.as_double(0.0) != 1.0)
    {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "Qwen3-TTS currently supports speed 1.0 only",
            "invalid_speed",
        )));
    }
    for name in ["language", "instructions"] {
        if body.find(name).is_some_and(|v| !v.is_string()) {
            return Err(Box::new(error(
                400,
                "Bad Request",
                &format!("'{name}' must be a string"),
                "invalid_parameter_type",
            )));
        }
    }
    let Some(max_new_tokens) =
        read_unsigned(&body, "max_new_tokens", 3000, MAXIMUM_NEW_TOKENS).filter(|&n| n != 0)
    else {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "'max_new_tokens' must be between 1 and 8192",
            "invalid_max_new_tokens",
        )));
    };
    let Some(seed) = read_unsigned(&body, "seed", 42, u32::MAX as usize) else {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "'seed' must be a uint32",
            "invalid_seed",
        )));
    };
    let mut greedy = false;
    if let Some(value) = body.find("greedy") {
        if !value.is_bool() {
            return Err(Box::new(error(
                400,
                "Bad Request",
                "'greedy' must be a boolean",
                "invalid_greedy",
            )));
        }
        greedy = value.as_bool(false);
    }
    let mut sampling = SamplingOptions {
        seed: seed as u32,
        sample: !greedy,
        predictor_sample: !greedy,
        ..SamplingOptions::default()
    };
    if let Some(value) = body.find("subtalker_dosample") {
        if !value.is_bool() {
            return Err(Box::new(error(
                400,
                "Bad Request",
                "'subtalker_dosample' must be boolean",
                "invalid_sampling",
            )));
        }
        sampling.predictor_sample = value.as_bool(false);
    }
    let top_k = read_unsigned(&body, "top_k", sampling.top_k, u32::MAX as usize);
    let predictor_top_k = top_k.and_then(|_| {
        read_unsigned(
            &body,
            "subtalker_top_k",
            sampling.predictor_top_k,
            u32::MAX as usize,
        )
    });
    let valid = match (top_k, predictor_top_k) {
        (Some(top_k), Some(predictor_top_k)) => {
            sampling.top_k = top_k;
            sampling.predictor_top_k = predictor_top_k;
            read_float(&body, "temperature", &mut sampling.temperature)
                && read_float(&body, "top_p", &mut sampling.top_p)
                && read_float(
                    &body,
                    "repetition_penalty",
                    &mut sampling.repetition_penalty,
                )
                && read_float(
                    &body,
                    "subtalker_temperature",
                    &mut sampling.predictor_temperature,
                )
                && read_float(&body, "subtalker_top_p", &mut sampling.predictor_top_p)
                && tts_model::valid_sampling_options(&sampling)
        }
        _ => false,
    };
    if !valid {
        return Err(Box::new(error(
            400,
            "Bad Request",
            "invalid Qwen3-TTS sampling options",
            "invalid_sampling",
        )));
    }
    // A registered voice may declare the language it speaks, sparing callers
    // from repeating it; an explicit request language still wins.
    let mut language = body.member_str("language", "");
    if language.is_empty() {
        language = match preset {
            Some(preset) if !preset.language.is_empty() => preset.language.clone(),
            _ => "english".into(),
        };
    }
    Ok(PreparedSpeech {
        synthesis: SynthesisRequest {
            text: input.str().to_owned(),
            speaker: selected_voice,
            language,
            instruct: body.member_str("instructions", ""),
            reference_audio,
            reference_text,
            speaker_embedding_only,
            max_new_tokens,
            sampling,
        },
        pcm,
        sse,
    })
}

fn invalid_types() -> HttpResponse {
    error(
        400,
        "Bad Request",
        "'model' and 'input' must be strings; 'voice' must be a string when supplied",
        "invalid_parameter_type",
    )
}

/// Synthesizes into `write` as raw PCM16 or SSE audio events and returns
/// what the request log reports (`error_code` set on failure).
pub(crate) fn stream_speech(
    service: &TtsService,
    synthesis: &SynthesisRequest,
    sse: bool,
    cancelled: &dyn Fn() -> bool,
    write: &mut BodyWriter<'_>,
) -> StreamLog {
    let started = Instant::now();
    let disconnected = std::cell::Cell::new(false);
    let invalid_audio = std::cell::Cell::new(false);
    let samples = std::cell::Cell::new(0usize);
    let check = || disconnected.get() || cancelled();
    let outcome = {
        let mut on_audio = |chunk: &[f32]| {
            if cancelled() {
                return false;
            }
            let mut bytes = encode_pcm16(chunk);
            if !chunk.is_empty() && bytes.is_empty() {
                invalid_audio.set(true);
                return false;
            }
            if sse {
                let event = Value::object()
                    .with("type", "speech.audio.delta")
                    .with("audio", encode_audio_base64(&bytes))
                    .with("response_format", "pcm");
                bytes =
                    format!("event: speech.audio.delta\ndata: {}\n\n", dump(&event)).into_bytes();
            }
            disconnected.set(!write(&bytes));
            samples.set(samples.get() + chunk.len());
            !disconnected.get()
        };
        service.synthesize(synthesis, Some(&mut on_audio), &check)
    };
    let mut log = StreamLog::default();
    match outcome {
        Err(mut message) => {
            log.error_code = if invalid_audio.get() {
                "invalid_audio"
            } else if check() {
                "cancelled"
            } else {
                "generation_failed"
            }
            .into();
            if invalid_audio.get() {
                message = "Qwen3-TTS produced invalid audio".into();
            }
            if sse && !check() {
                let event = Value::object()
                    .with("type", "error")
                    .with("message", message);
                log.error_event_sent =
                    write(format!("event: error\ndata: {}\n\n", dump(&event)).as_bytes());
            }
        }
        Ok(_) if sse => {
            write(b"event: speech.audio.done\ndata: {\"type\":\"speech.audio.done\"}\n\n");
        }
        Ok(_) => {}
    }
    log.details = format!(
        "model={} stream={} audio_ms={} synthesis_ms={}",
        service.model_id(),
        if sse { "sse" } else { "pcm" },
        samples.get() / 24,
        started.elapsed().as_millis()
    );
    log
}

/// Synthesizes a whole WAV response.
pub(crate) fn buffered_speech(
    service: &TtsService,
    synthesis: &SynthesisRequest,
    cancelled: &dyn Fn() -> bool,
) -> HttpResponse {
    let started = Instant::now();
    let result = match service.synthesize(synthesis, None, cancelled) {
        Ok(result) => result,
        Err(message) => {
            if cancelled() {
                return error(
                    499,
                    "Client Closed Request",
                    "audio generation cancelled",
                    "cancelled",
                );
            }
            return error(500, "Internal Server Error", &message, "generation_failed");
        }
    };
    let wav = encode_wav(&result.samples, result.sample_rate);
    if wav.is_empty() {
        return error(
            500,
            "Internal Server Error",
            "Qwen3-TTS produced invalid audio",
            "invalid_audio",
        );
    }
    let codec_steps = result.codes.len() / result.code_groups.max(1) as usize;
    let audio_ms = 1000 * result.samples.len() as u64 / u64::from(result.sample_rate.max(1));
    let mut response = HttpResponse::new(200, "OK", wav)
        .with_header("Content-Type", "audio/wav")
        .with_header("X-Gufo-Schema", AUDIO_TTS_API_SCHEMA)
        .with_header("X-Gufo-TTS-Backend", service.backend_name())
        .with_header("X-Gufo-Codec-Steps", codec_steps.to_string());
    response.log_details = format!(
        "model={} codec_steps={codec_steps} audio_ms={audio_ms} synthesis_ms={}",
        service.model_id(),
        started.elapsed().as_millis()
    );
    response
}

fn speech(request: &HttpRequest, service: &Arc<TtsService>) -> HttpResponse {
    let prepared = match prepare_speech(request, service) {
        Ok(prepared) => prepared,
        Err(response) => return *response,
    };
    let cancelled = request.is_cancelled.clone();
    if !prepared.pcm {
        return buffered_speech(service, &prepared.synthesis, &|| {
            cancelled.as_ref().is_some_and(|c| c())
        });
    }
    let sse = prepared.sse;
    let log = Arc::new(Mutex::new(StreamLog::default()));
    let mut response = HttpResponse::new(200, "OK", Vec::new())
        .with_header(
            "Content-Type",
            if sse {
                "text/event-stream"
            } else {
                "audio/pcm"
            },
        )
        .with_header("Cache-Control", "no-cache")
        .with_header("X-Gufo-TTS-Backend", service.backend_name());
    response.stream_log = Some(log.clone());
    let service = service.clone();
    let synthesis = prepared.synthesis;
    response.streaming_body = Some(Box::new(move |write: &mut BodyWriter<'_>| {
        let check = || cancelled.as_ref().is_some_and(|c| c());
        let outcome = stream_speech(&service, &synthesis, sse, &check, write);
        *log.lock().unwrap_or_else(|e| e.into_inner()) = outcome;
        Ok(())
    }));
    response
}

/// Validates a speech request without synthesizing (`ValidateAudioTtsRequest`):
/// status 200 with an empty body when valid.
pub fn validate_audio_tts_request(request: &HttpRequest, service: &TtsService) -> HttpResponse {
    match prepare_speech(request, service) {
        Ok(_) => HttpResponse::default(),
        Err(response) => *response,
    }
}

pub fn is_audio_tts_api_path(path: &str) -> bool {
    path == SPEECH_PATH || path == VOICES_PATH
}

pub fn handle_audio_tts_api_request(
    request: &HttpRequest,
    service: &Arc<TtsService>,
) -> Result<HttpResponse> {
    if request.path == VOICES_PATH && request.method == "GET" {
        return Ok(voices(service));
    }
    if request.path == SPEECH_PATH && request.method == "POST" {
        return Ok(speech(request, service));
    }
    Ok(error(
        405,
        "Method Not Allowed",
        "HTTP method is not supported for this audio endpoint",
        "method_not_allowed",
    ))
}
