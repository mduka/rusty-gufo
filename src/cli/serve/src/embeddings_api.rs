//! OpenAI-compatible `POST /v1/embeddings` and llama.cpp's `POST /embeddings`
//! over EmbeddingGemma 2 (`src/cli/serve/embeddings_api.cpp`): request
//! validation of text, image, video and audio inputs, named prompts, Matryoshka
//! truncation, float and base64 encodings.

use std::sync::Arc;

use gufo_core::json::{self, Value};
use gufo_core::text::Result;

use crate::embedding_service::{
    EmbeddingBatch, EmbeddingInput, EmbeddingPart, EmbeddingPartType, EmbeddingRequestOptions,
    EmbeddingService, EmbeddingStatus, MAX_AUDIO, MAX_IMAGE_SOFT_TOKENS, MAX_IMAGES,
    MAX_INPUT_TOKENS, MAX_VIDEOS, supported_image_soft_tokens,
};
use crate::http::{HttpRequest, HttpResponse};

pub const EMBEDDINGS_API_SCHEMA: &str = "gufo.embeddings-api.v1";

const EMBEDDINGS_PATH: &str = "/v1/embeddings";
// llama.cpp's non-OpenAI routes. Both share one handler there.
const NATIVE_EMBEDDINGS_PATH: &str = "/embeddings";
const LEGACY_EMBEDDING_PATH: &str = "/embedding";
// llama.cpp's default multimodal marker in a `prompt_string`. Its server can
// randomize the marker and report it in /props; clients that cannot read
// /props, as here, fall back to this one.
const MEDIA_MARKER: &str = "<__media__>";
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

/// Whether a base64 `multimodal_data` file looks like audio, as llama.cpp's
/// mtmd helper decides (`is_audio_file`): WAV, MP3 (an ID3 tag or an MPEG
/// sync word) or FLAC. Ogg, which the audio decoder also reads, is sniffed
/// too. Anything else, including undecodable base64, goes to the image
/// reader.
fn looks_like_audio(base64: &str) -> bool {
    const DIGITS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let base64 = base64.as_bytes();
    if base64.len() < 16 {
        return false;
    }
    let mut head = [0u8; 12];
    for i in (0..16).step_by(4) {
        let mut bits = 0u32;
        for &digit in &base64[i..i + 4] {
            let Some(value) = DIGITS.iter().position(|&d| d == digit) else {
                return false;
            };
            bits = (bits << 6) | value as u32;
        }
        head[i / 4 * 3] = (bits >> 16) as u8;
        head[i / 4 * 3 + 1] = (bits >> 8) as u8;
        head[i / 4 * 3 + 2] = bits as u8;
    }
    let starts = |offset: usize, magic: &[u8]| head[offset..].starts_with(magic);
    (starts(0, b"RIFF") && starts(8, b"WAVE"))
        || starts(0, b"ID3")
        || (head[0] == 0xFF && head[1] & 0xE0 == 0xE0)
        || starts(0, b"fLaC")
        || starts(0, b"OggS")
}

fn is_known_field(name: &str, native: bool) -> bool {
    matches!(
        name,
        "input"
            | "model"
            | "encoding_format"
            | "dimensions"
            | "user"
            | "truncate"
            | "prompt_name"
            | "video_options"
            | "mm_processor_kwargs"
    ) || (native && matches!(name, "content" | "embd_normalize"))
}

/// Counts of the media parts parsed so far in one request.
#[derive(Default)]
struct MediaCounts {
    images: usize,
    audio: usize,
    videos: usize,
}

/// A boxed refusal: the error of a parser that returns early.
fn refuse(message: &str, code: &str) -> Box<HttpResponse> {
    Box::new(bad_request(message, code))
}

/// `parse_item`: one `input` item as an embedding input, or the response
/// that refuses it.
fn parse_item(
    item: &Value,
    native: bool,
    counts: &mut MediaCounts,
) -> std::result::Result<EmbeddingInput, Box<HttpResponse>> {
    let mut parsed = EmbeddingInput::default();
    let invalid = |message: &str| refuse(message, "invalid_parameter_type");
    let too_many_images = || {
        refuse(
            "embedding requests accept at most 32 images",
            "too_many_images",
        )
    };
    let too_many_audio = || {
        refuse(
            "embedding requests accept at most 32 audio clips",
            "too_many_audio",
        )
    };
    if let Value::String(text) = item {
        if text.is_empty() {
            return Err(refuse(
                "'input' must not contain empty strings",
                "empty_input",
            ));
        }
        parsed.content.push(EmbeddingPart::text(text.clone()));
    } else if native && item.is_object() && item.contains("prompt_string") {
        // llama.cpp's multimodal prompt: each marker in the string stands for
        // the next base64 file in `multimodal_data`.
        let prompt = item.find("prompt_string").expect("checked member");
        let data = item.find("multimodal_data");
        if !prompt.is_string()
            || item.len() != if data.is_some() { 2 } else { 1 }
            || data.is_some_and(|d| !d.is_array())
        {
            return Err(invalid(
                "prompt objects take a 'prompt_string' string and an optional 'multimodal_data' array",
            ));
        }
        let text = prompt.str();
        let files = data.map_or(0, Value::len);
        let markers = text.matches(MEDIA_MARKER).count();
        if markers != files {
            return Err(refuse(
                &format!(
                    "'prompt_string' has {markers} media markers ({MEDIA_MARKER}) but 'multimodal_data' has {files} items"
                ),
                "invalid_multimodal_data",
            ));
        }
        let mut begin = 0;
        for i in 0..=files {
            let end = if i == files {
                text.len()
            } else {
                begin + text[begin..].find(MEDIA_MARKER).expect("counted marker")
            };
            if end > begin {
                parsed.content.push(EmbeddingPart::text(&text[begin..end]));
            }
            if i == files {
                break;
            }
            let file = &data.expect("files present").items()[i];
            if !file.is_string() || file.str().is_empty() {
                return Err(invalid("'multimodal_data' items must be base64 strings"));
            }
            if looks_like_audio(file.str()) {
                counts.audio += 1;
                if counts.audio > MAX_AUDIO {
                    return Err(too_many_audio());
                }
                // The audio reader takes raw base64; the decoder checks the
                // bytes.
                parsed.content.push(EmbeddingPart {
                    kind: EmbeddingPartType::Audio,
                    value: file.str().to_owned(),
                });
            } else {
                counts.images += 1;
                if counts.images > MAX_IMAGES {
                    return Err(too_many_images());
                }
                // The image reader identifies PNG, JPEG and WebP by their
                // bytes.
                parsed.content.push(EmbeddingPart {
                    kind: EmbeddingPartType::Image,
                    value: format!("data:image/jpeg;base64,{}", file.str()),
                });
            }
            begin = end + MEDIA_MARKER.len();
        }
        if parsed.content.is_empty() {
            return Err(refuse("'prompt_string' must not be empty", "empty_input"));
        }
    } else if item.is_object() {
        let content = item.find("content");
        let Some(content) = content.filter(|c| item.len() == 1 && c.is_array()) else {
            return Err(invalid("typed input requires a 'content' array"));
        };
        if content.items().is_empty() {
            return Err(refuse("'content' must not be empty", "empty_input"));
        }
        for part in content.items() {
            if !part.is_object() {
                return Err(invalid("content parts must be objects"));
            }
            let kind = part.member_str("type", "");
            match kind.as_str() {
                "text" => {
                    let text = part.find("text");
                    let Some(text) = text.filter(|t| part.len() == 2 && t.is_string()) else {
                        return Err(invalid("text parts require a 'text' string"));
                    };
                    parsed.content.push(EmbeddingPart::text(text.str()));
                }
                "image_url" => {
                    let image = part.find("image_url");
                    let url = image.filter(|i| i.is_object()).and_then(|i| i.find("url"));
                    let valid = part.len() == 2
                        && image.is_some_and(|i| i.len() == 1)
                        && url.is_some_and(|u| u.is_string() && !u.str().is_empty());
                    if !valid {
                        return Err(invalid("image parts require 'image_url': {'url': '...'}"));
                    }
                    counts.images += 1;
                    if counts.images > MAX_IMAGES {
                        return Err(too_many_images());
                    }
                    parsed.content.push(EmbeddingPart {
                        kind: EmbeddingPartType::Image,
                        value: url.expect("checked url").str().to_owned(),
                    });
                }
                "input_video" => {
                    let video = part.find("input_video");
                    let Some(video) = video.filter(|v| part.len() == 2 && v.is_object()) else {
                        return Err(invalid("video parts require an 'input_video' object"));
                    };
                    if video.members().iter().any(|(key, value)| {
                        !matches!(key.as_str(), "data" | "url") || !value.is_string()
                    }) {
                        return Err(invalid("input_video accepts string data and url fields"));
                    }
                    let source = video.find("data").or_else(|| video.find("url"));
                    let Some(source) = source.filter(|s| !s.str().is_empty()) else {
                        return Err(invalid("input_video requires nonempty data or url"));
                    };
                    counts.videos += 1;
                    if counts.videos > MAX_VIDEOS {
                        return Err(refuse(
                            "embedding requests accept at most 32 videos",
                            "too_many_videos",
                        ));
                    }
                    parsed.content.push(EmbeddingPart {
                        kind: EmbeddingPartType::Video,
                        value: source.str().to_owned(),
                    });
                }
                "input_audio" => {
                    let audio = part.find("input_audio");
                    let Some(audio) = audio.filter(|a| part.len() == 2 && a.is_object()) else {
                        return Err(invalid("audio parts require an 'input_audio' object"));
                    };
                    if audio.members().iter().any(|(key, value)| {
                        !matches!(key.as_str(), "data" | "url" | "format") || !value.is_string()
                    }) {
                        return Err(invalid(
                            "input_audio accepts string data, url and format fields",
                        ));
                    }
                    // Match llama.cpp: data takes precedence; format is
                    // ignored and the actual container is detected from its
                    // bytes.
                    let source = audio.find("data").or_else(|| audio.find("url"));
                    let Some(source) = source.filter(|s| !s.str().is_empty()) else {
                        return Err(invalid("input_audio requires nonempty data or url"));
                    };
                    counts.audio += 1;
                    if counts.audio > MAX_AUDIO {
                        return Err(too_many_audio());
                    }
                    parsed.content.push(EmbeddingPart {
                        kind: EmbeddingPartType::Audio,
                        value: source.str().to_owned(),
                    });
                }
                _ => {
                    return Err(refuse(
                        &format!("unsupported embedding content type: {kind}"),
                        "unsupported_input",
                    ));
                }
            }
        }
        if parsed.content.iter().all(|part| part.value.is_empty()) {
            return Err(refuse(
                "'content' must contain text or an image",
                "empty_input",
            ));
        }
    } else if matches!(item, Value::Number(_) | Value::Array(_)) {
        return Err(refuse(
            "token-array input is not supported; send strings",
            "unsupported_input",
        ));
    } else {
        return Err(invalid("'input' items must be strings or content objects"));
    }
    Ok(parsed)
}

/// The member's value, or `None` when it is absent or null.
fn optional<'a>(root: &'a Value, name: &str) -> Option<&'a Value> {
    root.find(name).filter(|value| !value.is_null())
}

/// `native` selects llama.cpp's `POST /embeddings` request and response
/// form.
fn embed(request: &HttpRequest, service: &EmbeddingService, native: bool) -> HttpResponse {
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
        .find(|(name, _)| !is_known_field(name, native))
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

    // As in llama.cpp, `input` takes precedence over the legacy `content`.
    let input =
        optional(&root, "input").or_else(|| native.then(|| optional(&root, "content")).flatten());
    let Some(input) = input else {
        return bad_request(
            if native {
                "'input' or 'content' is required"
            } else {
                "'input' is required"
            },
            "missing_input",
        );
    };
    let mut inputs: Vec<EmbeddingInput> = Vec::new();
    let mut counts = MediaCounts::default();
    // llama.cpp also takes one bare prompt or content object.
    if input.is_string() || (native && input.is_object()) {
        match parse_item(input, native, &mut counts) {
            Ok(parsed) => inputs.push(parsed),
            Err(response) => return *response,
        }
    } else if let Value::Array(items) = input {
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
            match parse_item(item, native, &mut counts) {
                Ok(parsed) => inputs.push(parsed),
                Err(response) => return *response,
            }
        }
    } else {
        return bad_request(
            "'input' must be a string or an array of strings or content objects",
            "invalid_parameter_type",
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
        // As in llama.cpp, the native form validates it but returns floats.
        base64 = !native && format.str() == "base64";
    }
    // llama.cpp's per-request normalization. EmbeddingGemma 2 vectors are
    // always L2-normalized, so only its default, 2 (Euclidean), is accepted.
    if let Some(normalize) = optional(&root, "embd_normalize")
        && (!normalize.is_number() || normalize.as_double(0.0) != 2.0)
    {
        return bad_request(
            "'embd_normalize' supports only 2 (Euclidean)",
            "unsupported_embd_normalize",
        );
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

    let mut prompt_name = service.default_prompt_name().to_owned();
    let mut requested_prompt = false;
    if let Some(value) = optional(&root, "prompt_name") {
        if !value.is_string() {
            return bad_request("'prompt_name' must be a string", "invalid_parameter_type");
        }
        prompt_name = value.str().to_owned();
        requested_prompt = true;
    }
    if !prompt_name.is_empty() {
        let Some(prefix) = service.prompt(&prompt_name) else {
            return bad_request(
                &format!("unknown prompt_name '{prompt_name}'"),
                "invalid_prompt_name",
            );
        };
        // A leading text part: the service joins adjacent text before
        // tokenizing, so this equals a client-side prefix. A requested
        // prompt applies to every input, as in sentence-transformers. The
        // model card passes images and audio without a prefix, so the server
        // default skips inputs without text.
        for input in &mut inputs {
            let has_text = input
                .content
                .iter()
                .any(|part| part.kind == EmbeddingPartType::Text && !part.value.is_empty());
            if requested_prompt || has_text {
                input.content.insert(0, EmbeddingPart::text(prefix));
            }
        }
    }

    let mut options = EmbeddingRequestOptions::default();
    if let Some(video) = optional(&root, "video_options") {
        if !video.is_object() {
            return bad_request("video_options must be an object", "invalid_parameter_type");
        }
        let refused = |message: &str| bad_request(message, "invalid_video_options");
        for (key, value) in video.members() {
            match key.as_str() {
                "fps" => {
                    let fps = value.as_double(0.0);
                    if !value.is_number() || !fps.is_finite() || fps <= 0.0 || fps > 120.0 {
                        return refused("video fps must be in (0, 120]");
                    }
                    options.video.fps = fps;
                }
                "max_frames" => {
                    let n = value.as_size(0);
                    if !value.is_number()
                        || !(1..=32).contains(&n)
                        || value.as_double(0.0) != n as f64
                    {
                        return refused("video max_frames must be an integer in [1, 32]");
                    }
                    options.video.max_frames = n;
                }
                "add_timestamps" => {
                    if !value.is_bool() {
                        return refused("add_timestamps must be boolean");
                    }
                    options.video.timestamps = value.as_bool(false);
                }
                "overflow_strategy" => {
                    if !value.is_string() || !matches!(value.str(), "uniform" | "truncate") {
                        return refused("video overflow_strategy must be uniform or truncate");
                    }
                    options.video.truncate_overflow = value.str() == "truncate";
                }
                _ => return refused(&format!("unsupported video option: {key}")),
            }
        }
    }

    if let Some(truncate) = optional(&root, "truncate") {
        if !truncate.is_bool() {
            return bad_request("'truncate' must be a boolean", "invalid_parameter_type");
        }
        options.truncate = truncate.as_bool(false);
    }

    // vLLM's per-request processor options. Only the image budget is
    // supported.
    if let Some(kwargs) = optional(&root, "mm_processor_kwargs") {
        if !kwargs.is_object() {
            return bad_request(
                "'mm_processor_kwargs' must be an object",
                "invalid_parameter_type",
            );
        }
        for (key, value) in kwargs.members() {
            if key != "max_soft_tokens" {
                return bad_request(
                    &format!(
                        "unsupported mm_processor_kwargs key '{key}'; only 'max_soft_tokens' is supported"
                    ),
                    "invalid_mm_processor_kwargs",
                );
            }
            let budget = value.as_size(0);
            if budget > MAX_IMAGE_SOFT_TOKENS as usize
                || !supported_image_soft_tokens(budget as u32)
            {
                return bad_request(
                    "'max_soft_tokens' must be one of 70, 140, 280, 560, or 1120",
                    "invalid_mm_processor_kwargs",
                );
            }
            options.image_soft_tokens = budget as u32;
        }
    }

    let cancelled = || request.is_cancelled.as_ref().is_some_and(|check| check());
    let mut batch = EmbeddingBatch::default();
    let mut message = String::new();
    match service.embed_inputs(
        &inputs,
        request.is_cancelled.clone(),
        &mut batch,
        &mut message,
        &options,
    ) {
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
        EmbeddingStatus::InvalidVideoInput => {
            return bad_request(&message, "invalid_video_input");
        }
        EmbeddingStatus::InvalidAudioInput => {
            return bad_request(&message, "invalid_audio_input");
        }
        EmbeddingStatus::InvalidInput => {
            return bad_request(
                if message.is_empty() {
                    "invalid embedding input"
                } else {
                    &message
                },
                "invalid_image_input",
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

    if batch.embeddings.len() != inputs.len() || batch.token_counts.len() != inputs.len() {
        return error(
            500,
            "embedding runtime returned an inconsistent batch",
            "embedding_failed",
        );
    }
    let prompt_tokens: usize = batch.token_counts.iter().sum();

    // llama.cpp's native form is a bare array whose items nest the pooled
    // vector in a one-element list, without model or usage.
    let mut body = String::from(if native {
        "["
    } else {
        r#"{"object":"list","data":["#
    });
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
        if native {
            body.push_str(&format!(r#"{{"index":{index},"embedding":["#));
            append_floats(&mut body, values);
            body.push_str("]}");
            continue;
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
    if native {
        body.push(']');
    } else {
        let model = Value::from(service.model_id()).dump().unwrap_or_default();
        body.push_str(&format!(
            r#"],"model":{model},"usage":{{"prompt_tokens":{prompt_tokens},"total_tokens":{prompt_tokens}}}}}"#
        ));
    }

    let mut response = HttpResponse::new(200, "OK", body)
        .with_header("Content-Type", "application/json")
        .with_header("X-Gufo-Schema", EMBEDDINGS_API_SCHEMA)
        .with_header("X-Gufo-Embedding-Backend", service.backend_name());
    response.log_details = format!(
        "model={} inputs={} prompt_tokens={prompt_tokens} dimensions={} encoding_format={}{}",
        service.model_id(),
        inputs.len(),
        if dimensions != 0 {
            dimensions
        } else {
            batch.embeddings[0].len()
        },
        if base64 { "base64" } else { "float" },
        if native { " shape=llama.cpp" } else { "" }
    );
    response
}

pub fn is_embeddings_api_path(path: &str) -> bool {
    matches!(
        path,
        EMBEDDINGS_PATH | NATIVE_EMBEDDINGS_PATH | LEGACY_EMBEDDING_PATH
    )
}

/// `POST /v1/embeddings`, and llama.cpp's `POST /embeddings` and
/// `POST /embedding`; other methods answer 405.
pub fn handle_embeddings_api_request(
    request: &HttpRequest,
    service: &Arc<EmbeddingService>,
) -> Result<HttpResponse> {
    if is_embeddings_api_path(&request.path) && request.method == "POST" {
        return Ok(embed(request, service, request.path != EMBEDDINGS_PATH));
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
