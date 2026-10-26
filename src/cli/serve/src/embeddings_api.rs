//! OpenAI-compatible `POST /v1/embeddings` over EmbeddingGemma 2
//! (`src/cli/serve/embeddings_api.cpp`): request validation, Matryoshka
//! truncation, float and base64 encodings.

use std::sync::Arc;

use gufo_core::json::{self, Value};
use gufo_core::text::Result;

use crate::embedding_service::{
    EmbeddingBatch, EmbeddingService, EmbeddingStatus, MAX_INPUT_TOKENS,
};
use crate::http::{HttpRequest, HttpResponse};

pub const EMBEDDINGS_API_SCHEMA: &str = "gufo.embeddings-api.v1";

const EMBEDDINGS_PATH: &str = "/v1/embeddings";
const MAXIMUM_INPUTS: usize = 2048;
const DIMENSIONS: [usize; 4] = [128, 256, 512, 768];

fn reason_phrase(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        499 => "Client Closed Request",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    }
}

fn error(status: u16, message: &str, code: &str) -> HttpResponse {
    let root = Value::object().with(
        "error",
        Value::object()
            .with("message", message)
            .with(
                "type",
                if status < 500 {
                    "invalid_request_error"
                } else {
                    "server_error"
                },
            )
            .with("code", code)
            .with("param", Value::Null),
    );
    HttpResponse::new(
        status,
        reason_phrase(status),
        root.dump().unwrap_or_default(),
    )
    .with_header("Content-Type", "application/json")
}

fn bad_request(message: &str, code: &str) -> HttpResponse {
    error(400, message, code)
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let word = (u32::from(chunk[0]) << 16)
            | chunk.get(1).map_or(0, |&b| u32::from(b) << 8)
            | chunk.get(2).map_or(0, |&b| u32::from(b));
        output.push(char::from(ALPHABET[(word >> 18) as usize & 63]));
        output.push(char::from(ALPHABET[(word >> 12) as usize & 63]));
        output.push(if chunk.len() > 1 {
            char::from(ALPHABET[(word >> 6) as usize & 63])
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            char::from(ALPHABET[word as usize & 63])
        } else {
            '='
        });
    }
    output
}

/// Little-endian IEEE-754 binary32, as OpenAI's base64 format specifies.
fn encode_base64_floats(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    base64(&bytes)
}

/// `std::to_chars(float)`: the shortest digits that parse back to the same
/// float, in fixed or scientific notation, whichever is shorter (fixed on a
/// tie); exponents have a sign and at least two digits.
fn format_f32(value: f32) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    let sign = if value < 0.0 { "-" } else { "" };
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .unwrap_or((scientific.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };

    let mut sci = String::from(sign);
    sci.push_str(&digits[..1]);
    if digits.len() > 1 {
        sci.push('.');
        sci.push_str(&digits[1..]);
    }
    sci.push('e');
    sci.push(if exponent < 0 { '-' } else { '+' });
    sci.push_str(&format!("{:02}", exponent.unsigned_abs()));

    let count = digits.len() as i32;
    let mut fixed = String::from(sign);
    if exponent >= count - 1 {
        // An integer: its exact value, which f32 holds below 2^128.
        fixed.push_str(&format!("{}", value.abs() as f64 as u128));
    } else if exponent >= 0 {
        let split = (exponent + 1) as usize;
        fixed.push_str(&digits[..split]);
        fixed.push('.');
        fixed.push_str(&digits[split..]);
    } else {
        fixed.push_str("0.");
        for _ in 0..(-exponent - 1) {
            fixed.push('0');
        }
        fixed.push_str(digits);
    }
    if fixed.len() <= sci.len() { fixed } else { sci }
}

/// The shortest decimal text of every value, as a JSON array.
fn append_floats(out: &mut String, values: &[f32]) {
    out.push('[');
    for (index, &value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format_f32(value));
    }
    out.push(']');
}

/// Matryoshka truncation keeps the leading components and renormalises
/// them; `None` when the prefix has no usable norm.
fn truncate_and_normalize(full: &[f32], dimensions: usize) -> Option<Vec<f32>> {
    let sum: f64 = full[..dimensions]
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    // A NaN sum is not finite either.
    if sum <= 0.0 || !sum.is_finite() {
        return None;
    }
    let scale = 1.0 / sum.sqrt();
    Some(
        full[..dimensions]
            .iter()
            .map(|&v| (f64::from(v) * scale) as f32)
            .collect(),
    )
}

fn is_known_field(name: &str) -> bool {
    matches!(
        name,
        "input" | "model" | "encoding_format" | "dimensions" | "user"
    )
}

/// The member's value, or `None` when it is absent or null.
fn optional<'a>(root: &'a Value, name: &str) -> Option<&'a Value> {
    root.find(name).filter(|value| !value.is_null())
}

fn embed(request: &HttpRequest, service: &EmbeddingService) -> HttpResponse {
    let root = match json::parse(&request.body) {
        Ok(root) => root,
        Err(e) => return bad_request(e.message(), "parse_error"),
    };
    if !root.is_object() {
        return bad_request("request body must be a JSON object", "invalid_request_body");
    }
    if let Some((name, _)) = root
        .members()
        .iter()
        .find(|(name, _)| !is_known_field(name))
    {
        return bad_request(
            &format!("unsupported embeddings parameter: {name}"),
            "unsupported_field",
        );
    }

    if let Some(model) = optional(&root, "model") {
        if !model.is_string() {
            return bad_request("'model' must be a string", "invalid_parameter_type");
        }
        if model.str() != service.model_id() {
            return error(
                404,
                "requested embedding model is not loaded",
                "model_not_found",
            );
        }
    }

    let Some(input) = optional(&root, "input") else {
        return bad_request("'input' is required", "missing_input");
    };
    let mut texts: Vec<String> = Vec::new();
    match input {
        Value::String(text) => texts.push(text.clone()),
        Value::Array(items) => {
            if items.is_empty() {
                return bad_request("'input' must not be an empty array", "empty_input");
            }
            if items.len() > MAXIMUM_INPUTS {
                return bad_request(
                    &format!("'input' accepts at most {MAXIMUM_INPUTS} items"),
                    "too_many_inputs",
                );
            }
            for item in items {
                match item {
                    Value::Number(_) | Value::Array(_) => {
                        return bad_request(
                            "token-array input is not supported; send strings",
                            "unsupported_input",
                        );
                    }
                    Value::String(text) => texts.push(text.clone()),
                    _ => {
                        return bad_request(
                            "'input' items must be strings",
                            "invalid_parameter_type",
                        );
                    }
                }
            }
        }
        _ => {
            return bad_request(
                "'input' must be a string or an array of strings",
                "invalid_parameter_type",
            );
        }
    }
    if let Some(index) = texts.iter().position(String::is_empty) {
        return bad_request(
            &format!("'input' must not contain empty strings (index {index})"),
            "empty_input",
        );
    }

    let mut base64 = false;
    if let Some(format) = optional(&root, "encoding_format") {
        if !matches!(format.str(), "float" | "base64") || !format.is_string() {
            return bad_request(
                "'encoding_format' must be \"float\" or \"base64\"",
                "invalid_encoding_format",
            );
        }
        base64 = format.str() == "base64";
    }

    // 0 means the model's full width.
    let mut dimensions = 0;
    if let Some(value) = optional(&root, "dimensions") {
        dimensions = value.as_size(0);
        if !DIMENSIONS.contains(&dimensions) {
            return bad_request(
                "'dimensions' must be one of 128, 256, 512, or 768",
                "invalid_dimensions",
            );
        }
    }
    if optional(&root, "user").is_some_and(|user| !user.is_string()) {
        return bad_request("'user' must be a string", "invalid_parameter_type");
    }

    let cancelled = || request.is_cancelled.as_ref().is_some_and(|check| check());
    let mut batch = EmbeddingBatch::default();
    let mut message = String::new();
    match service.embed(&texts, &cancelled, &mut batch, &mut message) {
        EmbeddingStatus::Ok => {}
        EmbeddingStatus::Cancelled => {
            return error(499, "embedding request cancelled", "cancelled");
        }
        EmbeddingStatus::InputTooLong => {
            return bad_request(
                &format!(
                    "input at index {} exceeds the {MAX_INPUT_TOKENS}-token context window",
                    batch.failed_index
                ),
                "context_length_exceeded",
            );
        }
        EmbeddingStatus::Failed => {
            return error(
                500,
                if message.is_empty() {
                    "embedding failed"
                } else {
                    &message
                },
                "embedding_failed",
            );
        }
        EmbeddingStatus::DeviceLost => {
            return error(503, "GPU device lost; restart the server", "device_lost");
        }
    }
    if cancelled() {
        return error(499, "embedding request cancelled", "cancelled");
    }

    if batch.embeddings.len() != texts.len() || batch.token_counts.len() != texts.len() {
        return error(
            500,
            "embedding runtime returned an inconsistent batch",
            "embedding_failed",
        );
    }
    let prompt_tokens: usize = batch.token_counts.iter().sum();

    let mut body = String::from(r#"{"object":"list","data":["#);
    for (index, full) in batch.embeddings.iter().enumerate() {
        if full.is_empty() {
            return error(
                500,
                "embedding runtime returned an empty embedding",
                "embedding_failed",
            );
        }
        if full.iter().any(|value| !value.is_finite()) {
            return error(
                500,
                "embedding runtime returned a non-finite value",
                "embedding_failed",
            );
        }
        let truncated;
        let mut values: &[f32] = full;
        if dimensions != 0 && dimensions < values.len() {
            match truncate_and_normalize(values, dimensions) {
                Some(prefix) => truncated = prefix,
                None => {
                    return error(
                        500,
                        "embedding cannot be truncated to the requested size",
                        "embedding_failed",
                    );
                }
            }
            values = &truncated;
        }
        if index != 0 {
            body.push(',');
        }
        body.push_str(&format!(
            r#"{{"object":"embedding","index":{index},"embedding":"#
        ));
        if base64 {
            body.push('"');
            body.push_str(&encode_base64_floats(values));
            body.push('"');
        } else {
            append_floats(&mut body, values);
        }
        body.push('}');
    }
    let model = Value::from(service.model_id()).dump().unwrap_or_default();
    body.push_str(&format!(
        r#"],"model":{model},"usage":{{"prompt_tokens":{prompt_tokens},"total_tokens":{prompt_tokens}}}}}"#
    ));

    let mut response = HttpResponse::new(200, "OK", body)
        .with_header("Content-Type", "application/json")
        .with_header("X-Gufo-Schema", EMBEDDINGS_API_SCHEMA)
        .with_header("X-Gufo-Embedding-Backend", service.backend_name());
    response.log_details = format!(
        "model={} inputs={} prompt_tokens={prompt_tokens} dimensions={} encoding_format={}",
        service.model_id(),
        texts.len(),
        if dimensions != 0 {
            dimensions
        } else {
            batch.embeddings[0].len()
        },
        if base64 { "base64" } else { "float" }
    );
    response
}

pub fn is_embeddings_api_path(path: &str) -> bool {
    path == EMBEDDINGS_PATH
}

/// `POST /v1/embeddings`; other methods answer 405.
pub fn handle_embeddings_api_request(
    request: &HttpRequest,
    service: &Arc<EmbeddingService>,
) -> Result<HttpResponse> {
    if request.path == EMBEDDINGS_PATH && request.method == "POST" {
        return Ok(embed(request, service));
    }
    Ok(error(
        405,
        "HTTP method is not supported for this embeddings endpoint",
        "method_not_allowed",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_format_like_to_chars() {
        for (value, expected) in [
            (0.0f32, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (-1.0, "-1"),
            (0.5, "0.5"),
            (0.1, "0.1"),
            (0.012_345_67, "0.01234567"),
            (1.2e-5, "1.2e-05"),
            (0.0001, "1e-04"),
            (0.001, "0.001"),
            (123_456_790.0, "123456792"),
            (1e10, "1e+10"),
            (3.402_823_5e38, "3.4028235e+38"),
            (-0.333_333_34, "-0.33333334"),
        ] {
            assert_eq!(format_f32(value), expected, "{value}");
        }
        for value in [0.1f32, -0.023_456_78, 1.0e-7, 0.999_999_9, 7.0e-3] {
            assert_eq!(format_f32(value).parse::<f32>().unwrap_or(f32::NAN), value);
        }
    }

    #[test]
    fn base64_pads_like_rfc_4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(encode_base64_floats(&[1.0]), "AACAPw==");
    }
}
