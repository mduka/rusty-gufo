//! `/v1/audio/transcriptions` over Qwen3-ASR
//! (`src/cli/serve/audio_asr_api.cpp`): multipart validation, JSON, text and
//! verbose JSON responses, and streamed SSE transcript deltas.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use gufo_core::json::Value;
use gufo_core::text::{Error, Result};

use crate::asr_model::{TranscriptionRequest, TranscriptionTimings};
use crate::asr_service::AsrService;
use crate::http::{BodyWriter, HttpRequest, HttpResponse, StreamLog};
use crate::multipart::asr::{self as multipart, Part};

pub const AUDIO_ASR_API_SCHEMA: &str = "gufo.audio-asr-api.v1";

const TRANSCRIPTIONS_PATH: &str = "/v1/audio/transcriptions";
const MAXIMUM_TEXT_FIELD_BYTES: usize = 16 << 10;
const MAXIMUM_NEW_TOKENS: usize = 512;
const KNOWN_FIELDS: [&str; 9] = [
    "file",
    "model",
    "language",
    "prompt",
    "response_format",
    "temperature",
    "max_tokens",
    "stream",
    "timestamp_granularities[]",
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

/// `std::from_chars` into `size_t`: digits only, nonzero, at most `maximum`.
fn parse_positive_size(value: &[u8], maximum: usize) -> Option<usize> {
    if value.is_empty() || !value.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let parsed: usize = std::str::from_utf8(value).ok()?.parse().ok()?;
    (parsed != 0 && parsed <= maximum).then_some(parsed)
}

/// `std::from_chars` into `double` over the whole field (no leading `+` or
/// whitespace).
fn parse_double(value: &[u8]) -> Option<f64> {
    let text = std::str::from_utf8(value).ok()?;
    if text.is_empty()
        || text.starts_with('+')
        || text.starts_with(|c: char| c.is_ascii_whitespace())
    {
        return None;
    }
    text.parse().ok()
}

/// C++ `ostream << double` (`%g`, six significant digits).
pub(crate) fn format_general(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".into()
        } else if value < 0.0 {
            "-inf".into()
        } else {
            "inf".into()
        };
    }
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let trim = |digits: &str| -> String {
        if digits.contains('.') {
            digits
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_owned()
        } else {
            digits.to_owned()
        }
    };
    if !(-4..6).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exponent.unsigned_abs())
    } else {
        trim(&format!("{value:.*}", (5 - exponent) as usize))
    }
}

fn server_timing(timings: &TranscriptionTimings) -> String {
    format!(
        "decode_audio;dur={}, features;dur={}, audio_encoder;dur={}, text_decoder;dur={}",
        format_general(timings.decode_audio_ms),
        format_general(timings.feature_extraction_ms),
        format_general(timings.audio_encoder_ms),
        format_general(timings.text_decoder_ms)
    )
}

fn text(part: Option<&Part>) -> String {
    part.map(|p| String::from_utf8_lossy(&p.body).into_owned())
        .unwrap_or_default()
}

fn transcribe(request: &HttpRequest, service: &Arc<AsrService>) -> Result<HttpResponse> {
    let Some(boundary) = multipart::boundary(&request.header("content-type")) else {
        return Ok(error(
            400,
            "Bad Request",
            "Content-Type must be multipart/form-data with a boundary",
            "invalid_content_type",
        ));
    };
    let parts: HashMap<String, Part> = match multipart::parse(&request.body, &boundary) {
        Ok(parts) => parts,
        Err(message) => return Ok(error(400, "Bad Request", &message, "invalid_multipart")),
    };
    if let Some(name) = parts
        .keys()
        .find(|name| !KNOWN_FIELDS.contains(&name.as_str()))
    {
        return Ok(error(
            400,
            "Bad Request",
            &format!("unsupported transcription field: {name}"),
            "unsupported_field",
        ));
    }
    let Some(file) = parts.get("file").filter(|f| !f.body.is_empty()) else {
        return Ok(error(
            400,
            "Bad Request",
            "a non-empty WAV 'file' is required",
            "missing_file",
        ));
    };
    if let Some(model) = parts.get("model") {
        let model = model.body.as_slice();
        if model != service.model_id().as_bytes()
            && model != b"qwen3-asr"
            && model != b"qwen3-asr-1.7b"
        {
            return Ok(error(
                400,
                "Bad Request",
                "unsupported transcription model",
                "invalid_model",
            ));
        }
    }
    let language = parts.get("language");
    let prompt = parts.get("prompt");
    if language.is_some_and(|l| l.body.len() > MAXIMUM_TEXT_FIELD_BYTES)
        || prompt.is_some_and(|p| p.body.len() > MAXIMUM_TEXT_FIELD_BYTES)
    {
        return Ok(error(
            400,
            "Bad Request",
            "transcription text fields exceed 16384 bytes",
            "field_too_large",
        ));
    }
    let format = parts.get("response_format").map_or_else(
        || "json".to_owned(),
        |f| String::from_utf8_lossy(&f.body).into_owned(),
    );
    if format != "json" && format != "text" && format != "verbose_json" {
        return Ok(error(
            400,
            "Bad Request",
            "response_format must be json, text, or verbose_json",
            "invalid_response_format",
        ));
    }
    if let Some(temperature) = parts.get("temperature")
        && !parse_double(&temperature.body).is_some_and(|v| v.is_finite() && v == 0.0)
    {
        return Ok(error(
            400,
            "Bad Request",
            "native Qwen3-ASR supports temperature 0 only",
            "invalid_temperature",
        ));
    }
    if parts.contains_key("timestamp_granularities[]") {
        return Ok(error(
            400,
            "Bad Request",
            "timestamp granularities are not supported",
            "unsupported_option",
        ));
    }
    let mut stream = false;
    if let Some(value) = parts.get("stream") {
        if value.body != b"true" && value.body != b"false" {
            return Ok(error(
                400,
                "Bad Request",
                "'stream' must be true or false",
                "invalid_parameter_type",
            ));
        }
        stream = value.body == b"true";
    }
    if stream && format != "json" {
        return Ok(error(
            400,
            "Bad Request",
            "streaming transcription requires json",
            "invalid_response_format",
        ));
    }
    let mut maximum_tokens = 256;
    if let Some(value) = parts.get("max_tokens") {
        match parse_positive_size(&value.body, MAXIMUM_NEW_TOKENS) {
            Some(parsed) => maximum_tokens = parsed,
            None => {
                return Ok(error(
                    400,
                    "Bad Request",
                    "max_tokens must be between 1 and 512",
                    "invalid_max_tokens",
                ));
            }
        }
    }
    let input = TranscriptionRequest {
        wav: file.body.clone(),
        context: text(prompt),
        language: language
            .filter(|l| !l.body.is_empty())
            .map(|l| String::from_utf8_lossy(&l.body).into_owned()),
        max_new_tokens: maximum_tokens,
        pcm16k: Vec::new(),
    };
    if stream {
        return Ok(streaming_transcription(request, service, input));
    }
    let cancelled = || request.is_cancelled.as_ref().is_some_and(|c| c());
    let result = match service.transcribe(&input, None, &cancelled) {
        Ok(result) => result,
        Err(message) => {
            if cancelled() {
                return Ok(error(
                    499,
                    "Client Closed Request",
                    "transcription cancelled",
                    "cancelled",
                ));
            }
            return Ok(error(
                500,
                "Internal Server Error",
                &message,
                "transcription_failed",
            ));
        }
    };
    let mut response = if format == "text" {
        HttpResponse::new(200, "OK", result.text.clone())
            .with_header("Content-Type", "text/plain; charset=utf-8")
    } else {
        let mut body = Value::object().with("text", result.text.as_str());
        if format == "verbose_json" {
            body.set("task", "transcribe");
            body.set("language", result.language.to_ascii_lowercase());
            body.set(
                "duration",
                result.audio_samples as f64 / f64::from(result.sample_rate),
            );
            body.set("segments", Value::array());
        }
        HttpResponse::new(200, "OK", body.dump()?)
    };
    response = response
        .with_header("X-Gufo-Schema", AUDIO_ASR_API_SCHEMA)
        .with_header("X-Gufo-ASR-Backend", service.backend_name())
        .with_header("Server-Timing", server_timing(&result.timings));
    response.log_details = format!(
        "model={} chunks={} audio_tokens={} generated_tokens={} audio_ms={} feature_ms={:.6} \
         encoder_ms={:.6} decoder_ms={:.6}",
        service.model_id(),
        result.chunks,
        result.audio_tokens,
        result.generated_ids.len(),
        1000 * result.audio_samples as u64 / u64::from(result.sample_rate.max(1)),
        result.timings.feature_extraction_ms,
        result.timings.audio_encoder_ms,
        result.timings.text_decoder_ms
    );
    Ok(response)
}

fn streaming_transcription(
    request: &HttpRequest,
    service: &Arc<AsrService>,
    input: TranscriptionRequest,
) -> HttpResponse {
    let log = Arc::new(Mutex::new(StreamLog::default()));
    let mut response = HttpResponse::new(200, "OK", Vec::new())
        .with_header("Content-Type", "text/event-stream")
        .with_header("Cache-Control", "no-cache")
        .with_header("X-Accel-Buffering", "no")
        .with_header("X-Gufo-Schema", AUDIO_ASR_API_SCHEMA);
    response.stream_log = Some(log.clone());
    let service = service.clone();
    let cancelled = request.is_cancelled.clone();
    response.streaming_body = Some(Box::new(move |write: &mut BodyWriter<'_>| {
        let connected = Cell::new(true);
        let write = RefCell::new(write);
        let send = |event: Value| {
            if !connected.get() {
                return false;
            }
            let frame = format!(
                "event: {}\ndata: {}\n\n",
                event.member_str("type", ""),
                dump(&event)
            );
            let sent = (write.borrow_mut())(frame.as_bytes());
            connected.set(sent);
            sent
        };
        // Committed SSE deltas cannot be retracted: a transcript that changes
        // its committed prefix fails the stream (C++ throws).
        let prefix_changed = Cell::new(false);
        let mut published = String::new();
        let mut progress = |text: &str| {
            if !text.starts_with(published.as_str()) {
                prefix_changed.set(true);
                return false;
            }
            if text.len() == published.len() {
                return connected.get();
            }
            let event = Value::object()
                .with("type", "transcript.text.delta")
                .with("delta", &text[published.len()..]);
            published = text.to_owned();
            send(event)
        };
        let stop = || !connected.get() || cancelled.as_ref().is_some_and(|c| c());
        let outcome = service.transcribe(&input, Some(&mut progress), &stop);
        if prefix_changed.get() {
            return Err(Error::runtime(
                "ASR transcript changed its committed prefix",
            ));
        }
        let result = match outcome {
            Ok(result) => result,
            Err(message) => {
                let code = if stop() {
                    "cancelled"
                } else {
                    "transcription_failed"
                };
                log.lock().unwrap_or_else(|e| e.into_inner()).error_code = code.into();
                if !stop() {
                    let sent = send(
                        Value::object()
                            .with("type", "error")
                            .with("code", code)
                            .with("message", message),
                    );
                    log.lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .error_event_sent = sent;
                }
                return Ok(());
            }
        };
        let committed = progress(&result.text);
        if prefix_changed.get() {
            return Err(Error::runtime(
                "ASR transcript changed its committed prefix",
            ));
        }
        if !committed {
            return Ok(());
        }
        send(
            Value::object()
                .with("type", "transcript.text.done")
                .with("text", result.text.as_str()),
        );
        log.lock().unwrap_or_else(|e| e.into_inner()).details = format!(
            "model={} chunks={} generated_tokens={}",
            service.model_id(),
            result.chunks,
            result.generated_ids.len()
        );
        Ok(())
    }));
    response
}

pub fn is_audio_asr_api_path(path: &str) -> bool {
    path == TRANSCRIPTIONS_PATH
}

pub fn handle_audio_asr_api_request(
    request: &HttpRequest,
    service: &Arc<AsrService>,
) -> Result<HttpResponse> {
    if request.path == TRANSCRIPTIONS_PATH && request.method == "POST" {
        return transcribe(request, service);
    }
    Ok(error(
        405,
        "Method Not Allowed",
        "HTTP method is not supported for this transcription endpoint",
        "method_not_allowed",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_format_matches_ostream() {
        for (value, expected) in [
            (0.0, "0"),
            (10.0, "10"),
            (20.5, "20.5"),
            (0.1, "0.1"),
            (123_456.0, "123456"),
            (1_234_567.0, "1.23457e+06"),
            (0.000_123_456, "0.000123456"),
            (0.000_012_345_6, "1.23456e-05"),
            (1.234_567, "1.23457"),
            (999_999.5, "1e+06"),
        ] {
            assert_eq!(format_general(value), expected, "{value}");
        }
    }

    #[test]
    fn numeric_fields_follow_from_chars() {
        assert_eq!(parse_positive_size(b"42", 512), Some(42));
        for invalid in [&b""[..], b"0", b"513", b"+1", b"-1", b" 1", b"1x"] {
            assert_eq!(parse_positive_size(invalid, 512), None);
        }
        assert_eq!(parse_double(b"0"), Some(0.0));
        assert_eq!(parse_double(b"-0.0"), Some(-0.0));
        assert_eq!(parse_double(b"+0"), None);
        assert_eq!(parse_double(b" 0"), None);
    }
}
