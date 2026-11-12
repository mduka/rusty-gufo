//! OpenAI-style `/v1/images/generations` and `/v1/images/edits` over
//! Qwen-Image-2.1 (`src/cli/serve/image_api.cpp`).
//!
//! The service serialises generations behind a gate that polls
//! cancellation every 20 ms. Generation ([`ImageRunner`]) and the PNG/JPEG
//! codec ([`ImageCodec`]: `qwen_image_21::DecodeImage` and `EncodePng`) are
//! injected; `gufo-qwen-image-21` implements them. `ValidateRequest` is
//! pure and ported here.

use std::fmt::Write as _;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use gufo_core::json::Value;
use gufo_core::text::{CancellationCheck, Error, ErrorKind, Result};

use crate::audio_stream::encode_audio_base64;
use crate::http::{HttpRequest, HttpResponse};
use crate::logging;
use crate::multipart;

/// `core::kMaxEncodedImageBytes`.
pub const MAX_ENCODED_IMAGE_BYTES: usize = 20 * 1024 * 1024;
/// `core::kMaxImagePixels`.
pub const MAX_IMAGE_PIXELS: usize = 32 * 1024 * 1024;
const GATE_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Unpremultiplied RGBA8 (`qwen_image_21::Image`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Image {
    pub width: i32,
    pub height: i32,
    pub rgba: Vec<u8>,
}

/// `qwen_image_21::Request`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRequest {
    pub prompt: String,
    pub images: Vec<Image>,
    pub width: i32,
    pub height: i32,
    pub steps: i32,
    pub seed: u64,
}

impl Default for ImageRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            images: Vec::new(),
            width: 1024,
            height: 1024,
            steps: 40,
            seed: 0,
        }
    }
}

/// `qwen_image_21::Metrics`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ImageMetrics {
    pub prompt_ms: f64,
    pub denoise_ms: f64,
    pub vae_ms: f64,
    pub total_ms: f64,
    pub resident_weight_bytes: usize,
}

/// `qwen_image_21::Result`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageResult {
    pub image: Image,
    pub metrics: ImageMetrics,
}

/// `qwen_image_21::Model::Generate`. Invalid-argument errors become 400
/// responses, every other error a 500.
pub type ImageRunner =
    Box<dyn Fn(&ImageRequest, Option<&CancellationCheck>) -> Result<ImageResult> + Send + Sync>;

/// PNG/JPEG decoding to RGBA and PNG encoding (`qwen_image_21::DecodeImage`
/// and `EncodePng`); invalid input is an invalid-argument error.
pub trait ImageCodec: Send + Sync {
    fn decode(&self, bytes: &[u8]) -> Result<Image>;
    fn encode_png(&self, image: &Image) -> Result<Vec<u8>>;
}

/// `CheckImage`.
fn check_image(image: &Image) -> Result<()> {
    let pixels = i64::from(image.width) * i64::from(image.height);
    if image.width <= 0
        || image.height <= 0
        || pixels as u64 > MAX_IMAGE_PIXELS as u64
        || image.rgba.len() as u64 != pixels as u64 * 4
    {
        return Err(Error::invalid("invalid RGBA image dimensions"));
    }
    Ok(())
}

/// `qwen_image_21::ValidateRequest`.
pub fn validate_request(request: &ImageRequest) -> Result<()> {
    if request.prompt.len() > 64 << 10 {
        return Err(Error::invalid("image prompt exceeds 64 KiB"));
    }
    if request.width < 32
        || request.height < 32
        || request.width % 32 != 0
        || request.height % 32 != 0
        || request.width > 4096
        || request.height > 4096
        || i64::from(request.width) * i64::from(request.height) > 4_194_304
    {
        return Err(Error::invalid(
            "image size must be multiples of 32, at most 4096 per side and 4 megapixels",
        ));
    }
    if !(2..=100).contains(&request.steps) {
        return Err(Error::invalid("image steps must be between 2 and 100"));
    }
    if request.images.len() > 10 {
        return Err(Error::invalid(
            "Qwen-Image supports at most 10 reference images",
        ));
    }
    let mut pixels = 0usize;
    for image in &request.images {
        check_image(image)?;
        pixels += image.width as usize * image.height as usize;
    }
    if pixels > MAX_IMAGE_PIXELS {
        return Err(Error::invalid(
            "reference images exceed 32 megapixels combined",
        ));
    }
    Ok(())
}

pub struct ImageService {
    model_id: String,
    runner: ImageRunner,
    codec: Arc<dyn ImageCodec>,
    busy: Mutex<bool>,
    released: Condvar,
}

struct GateGuard<'a>(&'a ImageService);

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        *self.0.busy.lock().unwrap_or_else(|e| e.into_inner()) = false;
        self.0.released.notify_one();
    }
}

impl ImageService {
    pub fn new(
        model_id: impl Into<String>,
        runner: ImageRunner,
        codec: Arc<dyn ImageCodec>,
    ) -> Result<Self> {
        let model_id = model_id.into();
        if model_id.is_empty() {
            return Err(Error::invalid("image model ID must not be empty"));
        }
        Ok(Self {
            model_id,
            runner,
            codec,
            busy: Mutex::new(false),
            released: Condvar::new(),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn codec(&self) -> &Arc<dyn ImageCodec> {
        &self.codec
    }

    /// One generation at a time; waiting polls `cancelled` every 20 ms.
    pub fn generate(
        &self,
        request: &ImageRequest,
        cancelled: Option<&CancellationCheck>,
    ) -> Result<ImageResult> {
        let is_cancelled = || cancelled.is_some_and(|c| c());
        let cancellation = || Error::runtime("image request cancelled");
        let _guard = {
            let mut busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
            while *busy {
                let (guard, timeout) = self
                    .released
                    .wait_timeout(busy, GATE_POLL_INTERVAL)
                    .unwrap_or_else(|e| e.into_inner());
                busy = guard;
                if *busy && timeout.timed_out() && is_cancelled() {
                    return Err(cancellation());
                }
            }
            *busy = true;
            GateGuard(self)
        };
        if is_cancelled() {
            return Err(cancellation());
        }
        (self.runner)(request, cancelled)
    }
}

fn error(status: u16, message: &str, code: &str) -> HttpResponse {
    let body = Value::object().with(
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
    let reason = match status {
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    HttpResponse::new(status, reason, body.dump().unwrap_or_default())
        .with_header("Content-Type", "application/json")
}

fn parse_json(text: &[u8]) -> Result<Value> {
    gufo_core::json::parse(text).map_err(|e| Error::invalid(e.message()))
}

struct Parsed {
    fields: Value,
    images: Vec<Image>,
}

/// `Multipart`: form fields, with reference images decoded as they arrive.
fn parse_multipart(request: &HttpRequest, codec: &dyn ImageCodec) -> Result<Parsed> {
    let mut parsed = Parsed {
        fields: Value::object(),
        images: Vec::new(),
    };
    let (mut image_bytes, mut image_pixels) = (0usize, 0usize);
    // Keeps the kind of a field error (codec failures need not be 400s).
    let mut failure: Option<Error> = None;
    let framing = multipart::image::parse(
        &request.header("content-type"),
        &request.body,
        &mut |field| {
            let result = (|| -> Result<()> {
                let data = field.content;
                if field.name == "image" || field.name == "image[]" {
                    if parsed.images.len() == 10
                        || data.len() > MAX_ENCODED_IMAGE_BYTES - image_bytes
                    {
                        return Err(Error::invalid("image inputs exceed 10 images or 20 MiB"));
                    }
                    image_bytes += data.len();
                    let decoded = codec.decode(data)?;
                    image_pixels += decoded.width.max(0) as usize * decoded.height.max(0) as usize;
                    parsed.images.push(decoded);
                    if image_pixels > MAX_IMAGE_PIXELS {
                        return Err(Error::invalid(
                            "reference images exceed 32 megapixels combined",
                        ));
                    }
                    return Ok(());
                }
                if field.name == "mask" {
                    return Err(Error::invalid(
                        "mask uploads are unsupported; provide annotated reference images",
                    ));
                }
                if parsed.fields.contains(&field.name) {
                    return Err(Error::invalid(format!("duplicate field {}", field.name)));
                }
                if data.len() > 64 << 10 {
                    return Err(Error::invalid("multipart field too large"));
                }
                let value = if matches!(field.name.as_str(), "n" | "seed" | "steps" | "stream") {
                    parse_json(data)?
                } else {
                    Value::from(String::from_utf8_lossy(data).into_owned())
                };
                parsed.fields.set(&field.name, value);
                Ok(())
            })();
            result.map_err(|error| {
                let message = error.message().to_owned();
                failure = Some(error);
                message
            })
        },
    );
    if let Err(message) = framing {
        return Err(failure.unwrap_or_else(|| Error::invalid(message)));
    }
    if parsed.images.is_empty() {
        return Err(Error::invalid("image edit requires an image"));
    }
    Ok(parsed)
}

fn number(fields: &Value, name: &str, fallback: u64, low: u64, high: u64) -> Result<u64> {
    let Some(value) = fields.find(name) else {
        return Ok(fallback);
    };
    let v = value.as_double(0.0);
    if !value.is_number() || !v.is_finite() || v < low as f64 || v > high as f64 || v.floor() != v {
        return Err(Error::invalid(format!("invalid {name}")));
    }
    Ok(v as u64)
}

fn string(fields: &Value, name: &str, fallback: &str) -> Result<String> {
    match fields.find(name) {
        None => Ok(fallback.to_owned()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(Error::invalid(format!("{name} must be a string"))),
    }
}

/// A decimal `int` as `std::from_chars` reads it (optional `-`, digits).
fn parse_dimension(text: &str) -> Result<i32> {
    let digits = text.strip_prefix('-').unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return Err(Error::invalid("invalid image size"));
    }
    text.parse()
        .map_err(|_| Error::invalid("invalid image size"))
}

/// A double as `std::ostream` prints it by default (`%g`, six significant
/// digits).
fn format_g(value: f64) -> String {
    if value == 0.0 || !value.is_finite() {
        return format!("{value}");
    }
    let exponent = value.abs().log10().floor() as i32;
    let rounded = format!("{value:.5e}");
    let (mantissa, exp) = rounded.split_once('e').unwrap_or((&rounded, "0"));
    let exp: i32 = exp.parse().unwrap_or(exponent);
    if (-4..6).contains(&exp) {
        let decimals = (5 - exp).max(0) as usize;
        let fixed = format!("{value:.decimals$}");
        if fixed.contains('.') {
            fixed.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            fixed
        }
    } else {
        let mantissa = if mantissa.contains('.') {
            mantissa.trim_end_matches('0').trim_end_matches('.')
        } else {
            mantissa
        };
        format!(
            "{mantissa}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.unsigned_abs()
        )
    }
}

/// 32 random bits (`std::random_device`).
fn entropy() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
    );
    hasher.finish() & 0xffff_ffff
}

fn generate(request: &HttpRequest, service: &ImageService) -> Result<HttpResponse> {
    let edit = request.path == "/v1/images/edits";
    let mut parsed = if edit {
        parse_multipart(request, service.codec().as_ref())?
    } else {
        if !request
            .header("content-type")
            .starts_with("application/json")
        {
            return Err(Error::invalid("image generations require application/json"));
        }
        Parsed {
            fields: parse_json(&request.body)?,
            images: Vec::new(),
        }
    };
    let fields = &parsed.fields;
    if !fields.is_object() {
        return Err(Error::invalid("request must be a JSON object"));
    }
    const ALLOWED: [&str; 12] = [
        "prompt",
        "model",
        "n",
        "size",
        "response_format",
        "output_format",
        "quality",
        "background",
        "user",
        "seed",
        "steps",
        "stream",
    ];
    if let Some((name, _)) = fields
        .members()
        .iter()
        .find(|(name, _)| !ALLOWED.contains(&name.as_str()))
    {
        return Err(Error::invalid(format!(
            "unsupported image parameter: {name}"
        )));
    }
    if !fields.contains("prompt") {
        return Err(Error::invalid("prompt is required"));
    }
    let model = string(fields, "model", service.model_id())?;
    if model != service.model_id() {
        return Ok(error(
            404,
            "requested image model is not loaded",
            "model_not_found",
        ));
    }
    if string(fields, "response_format", "b64_json")? != "b64_json"
        || string(fields, "output_format", "png")? != "png"
    {
        return Err(Error::invalid(
            "image output supports PNG with response_format b64_json",
        ));
    }
    let quality = string(fields, "quality", "auto")?;
    if !matches!(quality.as_str(), "auto" | "standard" | "high") {
        return Err(Error::invalid(
            "quality must be auto, standard or high (40 steps)",
        ));
    }
    if fields
        .find("stream")
        .is_some_and(|s| !s.is_bool() || s.as_bool(false))
    {
        return Err(Error::invalid("streaming image responses are unsupported"));
    }
    string(fields, "user", "")?;
    let background = string(fields, "background", "auto")?;
    if !matches!(background.as_str(), "auto" | "transparent" | "opaque") {
        return Err(Error::invalid("invalid image background"));
    }
    let mut generation = ImageRequest {
        prompt: string(fields, "prompt", "")?,
        ..ImageRequest::default()
    };
    if background == "transparent" {
        generation.prompt = format!(
            "This is an RGBA image with transparency. {} The image has alpha channel and the \
             background is transparent.",
            generation.prompt
        );
    }
    generation.images = std::mem::take(&mut parsed.images);
    let fields = &parsed.fields;
    let size = string(fields, "size", "auto")?;
    if size == "auto" {
        if let Some(last) = generation.images.last() {
            let ratio = f64::from(last.width) / f64::from(last.height);
            let w = (1024.0 * 1024.0 * ratio).sqrt();
            generation.width = (w / 32.0).round_ties_even() as i32 * 32;
            generation.height = (w / ratio / 32.0).round_ties_even() as i32 * 32;
        }
    } else {
        let Some((width, height)) = size.split_once('x') else {
            return Err(Error::invalid("size must be WIDTHxHEIGHT or auto"));
        };
        generation.width = parse_dimension(width)?;
        generation.height = parse_dimension(height)?;
    }
    generation.steps = number(fields, "steps", 40, 2, 100)? as i32;
    let fallback_seed = (entropy() << 21) ^ entropy();
    generation.seed = number(fields, "seed", fallback_seed, 0, (1 << 53) - 1)?;
    let count = number(fields, "n", 1, 1, 10)?;
    validate_request(&generation)?;
    let mut data = Value::array();
    for _ in 0..count {
        let mut result = service.generate(&generation, request.is_cancelled.as_ref())?;
        if background == "opaque" {
            for pixel in result.image.rgba.as_chunks_mut::<4>().0 {
                let a = u32::from(pixel[3]);
                for channel in &mut pixel[..3] {
                    *channel = ((u32::from(*channel) * a + 255 * (255 - a) + 127) / 255) as u8;
                }
                pixel[3] = 255;
            }
        }
        let png = service.codec().encode_png(&result.image)?;
        data.push(Value::object().with("b64_json", encode_audio_base64(&png)));
        let m = &result.metrics;
        let mut details = format!(
            "request={} event=generated mode={} seed={} size={}x{} references={} steps={}",
            request.request_id,
            if edit { "edit" } else { "generate" },
            generation.seed,
            generation.width,
            generation.height,
            generation.images.len(),
            generation.steps
        );
        let _ = write!(
            details,
            " prompt_ms={} denoise_ms={} vae_ms={} total_ms={} resident_weight_mib={}",
            format_g(m.prompt_ms),
            format_g(m.denoise_ms),
            format_g(m.vae_ms),
            format_g(m.total_ms),
            m.resident_weight_bytes / (1024 * 1024)
        );
        logging::info("image", &details);
        generation.seed = (generation.seed + 1) & ((1 << 53) - 1);
    }
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let body = Value::object()
        .with("created", created as f64)
        .with("data", data);
    Ok(
        HttpResponse::new(200, "OK", body.dump().unwrap_or_default())
            .with_header("Content-Type", "application/json"),
    )
}

pub fn handle_image_api_request(
    request: &HttpRequest,
    service: &Arc<ImageService>,
) -> Result<HttpResponse> {
    if request.method != "POST" {
        return Ok(error(400, "image endpoints require POST", "invalid_method"));
    }
    Ok(match generate(request, service) {
        Ok(response) => response,
        Err(failure) if failure.kind() == ErrorKind::InvalidArgument => {
            error(400, failure.message(), "invalid_request")
        }
        Err(failure) => error(500, failure.message(), "image_generation_failed"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_double_formatting() {
        for (value, expected) in [
            (0.0, "0"),
            (1.0, "1"),
            (12.3456789, "12.3457"),
            (0.0001234, "0.0001234"),
            (0.00001234, "1.234e-05"),
            (1234567.0, "1.23457e+06"),
            (100000.0, "100000"),
            (999999.5, "1e+06"),
        ] {
            assert_eq!(format_g(value), expected, "{value}");
        }
    }

    #[test]
    fn validate_request_matches_the_model() {
        assert!(
            validate_request(&ImageRequest {
                prompt: "a red cube".into(),
                ..ImageRequest::default()
            })
            .is_ok()
        );
        for invalid in [
            ImageRequest {
                width: 31,
                ..ImageRequest::default()
            },
            ImageRequest {
                steps: 1,
                ..ImageRequest::default()
            },
            ImageRequest {
                height: 8192,
                ..ImageRequest::default()
            },
        ] {
            assert_eq!(
                validate_request(&invalid).map_err(|e| e.kind()),
                Err(ErrorKind::InvalidArgument)
            );
        }
    }
}
