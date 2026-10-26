//! OpenAI-style `/v1/videos` over the MiniMax H3 job queue
//! (`src/cli/serve/video_api.cpp`): create from JSON or multipart forms,
//! status lookup, content with single byte ranges, and deletion.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use gufo_core::json::Value;
use gufo_core::text::Result;

use crate::http::{HttpRequest, HttpResponse};
use crate::multipart;
use crate::video_jobs::{VideoJobRequest, VideoJobResult, VideoJobService, VideoJobSnapshot};
use crate::video_model::{
    resolve_duration_frames, resolve_generation_preset, validate_generation_parameters,
};

pub const VIDEO_API_SCHEMA: &str = "gufo.video-api.v1";
const VIDEOS_PATH: &str = "/v1/videos";
const MAXIMUM_EXACT_JSON_INTEGER: f64 = 9_007_199_254_740_991.0;

fn dump(value: &Value) -> Vec<u8> {
    // Every value built here holds finite numbers only.
    value.dump().unwrap_or_default().into_bytes()
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

fn snapshot_value(snapshot: &VideoJobSnapshot) -> Value {
    let optional_time = |t: i64| if t > 0 { Value::from(t) } else { Value::Null };
    let error = if snapshot.error_code.is_empty() {
        Value::Null
    } else {
        Value::object()
            .with("code", snapshot.error_code.as_str())
            .with("message", snapshot.error_message.as_str())
    };
    Value::object()
        .with("id", snapshot.id.as_str())
        .with("object", "video")
        .with("schema", VIDEO_API_SCHEMA)
        .with("created_at", snapshot.created_at)
        .with("status", snapshot.status.name())
        .with("progress", snapshot.progress)
        .with("model", snapshot.model.as_str())
        .with("size", snapshot.size.as_str())
        .with("seconds", snapshot.seconds.as_str())
        .with("output_format", snapshot.output_format.as_str())
        .with("completed_at", optional_time(snapshot.completed_at))
        .with("expires_at", optional_time(snapshot.expires_at))
        .with("error", error)
}

fn snapshot_response(status: u16, reason: &str, snapshot: &VideoJobSnapshot) -> HttpResponse {
    let mut response = HttpResponse::new(status, reason, dump(&snapshot_value(snapshot)));
    response.log_details = format!(
        "job={} state={} model={} size={}",
        snapshot.id,
        snapshot.status.name(),
        snapshot.model,
        snapshot.size
    );
    response
}

fn has_only_members(value: &Value, allowed: &[&str]) -> bool {
    value
        .members()
        .iter()
        .all(|(key, _)| allowed.contains(&key.as_str()))
}

fn media_type(content_type: &str) -> &str {
    let media = match content_type.find(';') {
        Some(semicolon) => &content_type[..semicolon],
        None => content_type,
    };
    media.trim_matches(|c: char| c.is_ascii() && (c.is_ascii_whitespace() || c == '\x0b'))
}

/// `ParseMultipartBody`: text fields become strings, file fields empty
/// strings, and a text `gufo` field its JSON object.
fn parse_multipart_body(request: &HttpRequest) -> std::result::Result<Value, String> {
    let mut result = Value::object();
    multipart::video::parse(
        &request.header("content-type"),
        &request.body,
        &mut |field| {
            if field.name == "gufo" && !field.has_filename {
                let extension = gufo_core::json::parse(field.content)
                    .map_err(|_| "multipart 'gufo' field must contain valid JSON".to_owned())?;
                if !extension.is_object() {
                    return Err("multipart 'gufo' field must contain a JSON object".into());
                }
                result.set(&field.name, extension);
            } else if field.has_filename {
                result.set(&field.name, "");
            } else {
                result.set(
                    &field.name,
                    String::from_utf8_lossy(field.content).into_owned(),
                );
            }
            Ok(())
        },
    )?;
    Ok(result)
}

fn json_unsigned(value: Option<&Value>) -> Option<u64> {
    let value = value?;
    if !value.is_number() {
        return None;
    }
    let number = value.as_double(-1.0);
    if !number.is_finite()
        || number < 0.0
        || number > MAXIMUM_EXACT_JSON_INTEGER
        || number.floor() != number
    {
        return None;
    }
    Some(number as u64)
}

fn json_frame(value: Option<&Value>) -> Option<i32> {
    json_unsigned(value).and_then(|n| i32::try_from(n).ok())
}

fn is_unsupported_conditioning(body: &Value) -> bool {
    ["input_reference", "first_frame", "last_frame", "references"]
        .iter()
        .any(|key| body.contains(key))
}

fn create_video(request: &HttpRequest, service: &VideoJobService) -> HttpResponse {
    let content_type = request.header("content-type");
    let body = if multipart::video::is_multipart_form_data(&content_type) {
        match parse_multipart_body(request) {
            Ok(body) => body,
            Err(message) => return error(400, "Bad Request", &message, "invalid_multipart_form"),
        }
    } else {
        if !content_type.is_empty()
            && !media_type(&content_type).eq_ignore_ascii_case("application/json")
        {
            return error(
                415,
                "Unsupported Media Type",
                "video requests must use application/json or multipart/form-data",
                "unsupported_media_type",
            );
        }
        match gufo_core::json::parse(&request.body) {
            Ok(body) => body,
            Err(_) => {
                return error(
                    400,
                    "Bad Request",
                    "request body must be valid JSON",
                    "parse_error",
                );
            }
        }
    };
    if !body.is_object() {
        return error(
            400,
            "Bad Request",
            "request body must be a JSON object",
            "invalid_body",
        );
    }
    if is_unsupported_conditioning(&body) {
        return error(
            400,
            "Bad Request",
            "image and ordered-reference conditioning are not supported",
            "unsupported_input_reference",
        );
    }
    if !has_only_members(&body, &["model", "prompt", "size", "seconds", "gufo"]) {
        return error(
            400,
            "Bad Request",
            "request contains an unsupported video field",
            "unsupported_field",
        );
    }
    let (model_value, prompt_value, size_value) =
        (body.find("model"), body.find("prompt"), body.find("size"));
    if !model_value.is_some_and(Value::is_string)
        || !prompt_value.is_some_and(Value::is_string)
        || size_value.is_some_and(|s| !s.is_string())
    {
        return error(
            400,
            "Bad Request",
            "'model', 'prompt', and optional 'size' must be strings",
            "invalid_parameter_type",
        );
    }
    let model = body.member_str("model", "");
    let prompt = body.member_str("prompt", "");
    let size = body.member_str("size", "512x512");
    let seconds = match body.find("seconds") {
        Some(Value::String(s)) => s.clone(),
        Some(value @ Value::Number(_)) => value.dump().unwrap_or_default(),
        _ => String::new(),
    };
    if model.is_empty() || prompt.is_empty() || seconds.is_empty() {
        return error(
            400,
            "Bad Request",
            "'model', 'prompt', and 'seconds' are required",
            "missing_required_parameter",
        );
    }
    let duration_frames = match resolve_duration_frames(&seconds) {
        Ok(frames) => frames,
        Err(message) => return error(400, "Bad Request", &message, "invalid_seconds"),
    };
    let mut preset = if size == "1344x768" {
        "exact-1344x768"
    } else {
        "exact"
    }
    .to_owned();
    let model_preset = match model.as_str() {
        "minimax-h3-exact" => "exact",
        "minimax-h3-fast" => "fast",
        "minimax-h3-aggressive" => "aggressive",
        "minimax-h3-dev" => "dev",
        "minimax-h3-fullres" => "exact-1344x768",
        "minimax-h3" => "",
        _ => {
            return error(
                400,
                "Bad Request",
                "unsupported video model",
                "invalid_model",
            );
        }
    };
    if !model_preset.is_empty() {
        preset = model_preset.to_owned();
    }
    let mut output_format = "mp4".to_owned();
    let mut seed = 42;
    let mut frames = None;
    let mut selected_frame = None;
    let mut preset_explicit = false;
    if let Some(extension) = body.find("gufo") {
        if !extension.is_object()
            || !has_only_members(
                extension,
                &[
                    "preset",
                    "seed",
                    "frames",
                    "output_format",
                    "selected_frame",
                ],
            )
        {
            return error(
                400,
                "Bad Request",
                "'gufo' must contain only supported video options",
                "invalid_strix_options",
            );
        }
        if let Some(value) = extension.find("preset") {
            if !value.is_string() {
                return error(
                    400,
                    "Bad Request",
                    "'gufo.preset' must be a string",
                    "invalid_preset",
                );
            }
            preset = value.str().to_owned();
            preset_explicit = true;
        }
        if extension
            .find("output_format")
            .is_some_and(|v| !v.is_string())
        {
            return error(
                400,
                "Bad Request",
                "'gufo.output_format' must be a string",
                "invalid_output_format",
            );
        }
        output_format = extension.member_str("output_format", &output_format);
        if extension.contains("seed") {
            match json_unsigned(extension.find("seed")) {
                Some(parsed) => seed = parsed,
                None => {
                    return error(
                        400,
                        "Bad Request",
                        "'gufo.seed' must be a non-negative exact integer",
                        "invalid_seed",
                    );
                }
            }
        }
        if extension.contains("frames") {
            frames = json_frame(extension.find("frames"));
            if frames.is_none() {
                return error(
                    400,
                    "Bad Request",
                    "'gufo.frames' must be a positive legal frame count",
                    "invalid_frames",
                );
            }
        }
        if extension.contains("selected_frame") {
            selected_frame = json_frame(extension.find("selected_frame"));
            if selected_frame.is_none() {
                return error(
                    400,
                    "Bad Request",
                    "'gufo.selected_frame' must be a non-negative integer",
                    "invalid_selected_frame",
                );
            }
        }
    }
    if preset_explicit && !model_preset.is_empty() && preset != model_preset {
        return error(
            400,
            "Bad Request",
            "model alias and 'gufo.preset' must select the same preset",
            "preset_model_mismatch",
        );
    }
    let mut parameters = match resolve_generation_preset(&preset) {
        Ok(parameters) => parameters,
        Err(message) => return error(400, "Bad Request", &message, "invalid_preset"),
    };
    parameters.frames = frames.unwrap_or(duration_frames);
    match output_format.as_str() {
        "ppm" => {
            if parameters.preset != "development-256" || size != "256x256" {
                return error(
                    400,
                    "Bad Request",
                    "PPM output requires preset 'dev' and size '256x256'",
                    "invalid_output_format",
                );
            }
            parameters.selected_frames = vec![selected_frame.unwrap_or(parameters.frames / 2)];
        }
        "mp4" => {
            if !parameters.mux || selected_frame.is_some() {
                return error(
                    400,
                    "Bad Request",
                    "MP4 output requires a supported preset, duration, and resolution combination",
                    "invalid_output_format",
                );
            }
        }
        _ => {
            return error(
                400,
                "Bad Request",
                "'gufo.output_format' must be 'mp4' or 'ppm'",
                "invalid_output_format",
            );
        }
    }
    if let Err(message) = validate_generation_parameters(&parameters) {
        return error(400, "Bad Request", &message, "invalid_frames");
    }
    if parameters.frames != duration_frames {
        return error(
            400,
            "Bad Request",
            "'gufo.frames' must match the aligned 'seconds' duration",
            "invalid_frames",
        );
    }
    let created = service.create(&VideoJobRequest {
        model,
        prompt,
        size,
        seconds,
        output_format,
        parameters,
        seed,
    });
    if created.result == VideoJobResult::QueueFull {
        return error(429, "Too Many Requests", &created.error, "video_queue_full")
            .with_header("Retry-After", "1");
    }
    let job = match (&created.result, &created.job) {
        (VideoJobResult::Ok, Some(job)) => job,
        _ => {
            let unavailable = created.result == VideoJobResult::IoError;
            return error(
                if unavailable { 503 } else { 400 },
                if unavailable {
                    "Service Unavailable"
                } else {
                    "Bad Request"
                },
                &created.error,
                if unavailable {
                    "video_service_unavailable"
                } else {
                    "invalid_video_request"
                },
            );
        }
    };
    snapshot_response(202, "Accepted", job)
        .with_header("Location", format!("/v1/videos/{}", job.id))
}

fn lookup_video(id: &str, service: &VideoJobService) -> HttpResponse {
    let lookup = service.get(id);
    match (&lookup.result, &lookup.job) {
        (VideoJobResult::Ok, Some(job)) => snapshot_response(200, "OK", job),
        _ => error(404, "Not Found", "video job not found", "video_not_found"),
    }
}

fn parse_unsigned(text: &str) -> Option<u64> {
    if text.is_empty() || !text.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// `ParseRange`: one `bytes=first-last` or `bytes=-suffix` range.
fn parse_range(header: &str, bytes: u64) -> Option<(u64, u64)> {
    let value = header.strip_prefix("bytes=")?;
    if bytes == 0 || value.contains(',') {
        return None;
    }
    let (first_text, last_text) = value.split_once('-')?;
    if first_text.is_empty() {
        let suffix = parse_unsigned(last_text).filter(|&s| s != 0)?.min(bytes);
        return Some((bytes - suffix, bytes - 1));
    }
    let first = parse_unsigned(first_text).filter(|&f| f < bytes)?;
    let last = if last_text.is_empty() {
        bytes - 1
    } else {
        parse_unsigned(last_text).filter(|&l| l >= first)?
    };
    Some((first, last.min(bytes - 1)))
}

fn read_range(path: &std::path::Path, first: u64, count: u64) -> Option<Vec<u8>> {
    let mut file = File::open(path).ok()?;
    file.seek(SeekFrom::Start(first)).ok()?;
    let mut body = vec![0; usize::try_from(count).ok()?];
    file.read_exact(&mut body).ok()?;
    Some(body)
}

fn video_content(id: &str, request: &HttpRequest, service: &VideoJobService) -> HttpResponse {
    let result = service.content(id);
    let content = match (result.result, result.content) {
        (VideoJobResult::NotFound, _) => {
            return error(404, "Not Found", "video job not found", "video_not_found");
        }
        (VideoJobResult::NotReady, _) => {
            return error(
                409,
                "Conflict",
                "video content is not ready",
                "video_not_ready",
            );
        }
        (VideoJobResult::Ok, Some(content)) => content,
        _ => {
            return error(
                503,
                "Service Unavailable",
                "video content is unavailable",
                "video_content_unavailable",
            );
        }
    };
    let range_header = request.header("range");
    let mut range = (0, content.bytes.saturating_sub(1));
    let mut partial = false;
    if !range_header.is_empty() {
        match parse_range(&range_header, content.bytes) {
            Some(parsed) => {
                range = parsed;
                partial = true;
            }
            None => {
                return error(
                    416,
                    "Range Not Satisfiable",
                    "invalid byte range",
                    "invalid_range",
                )
                .with_header("Content-Range", format!("bytes */{}", content.bytes))
                .with_header("Accept-Ranges", "bytes");
            }
        }
    }
    let count = if content.bytes == 0 {
        0
    } else {
        range.1 - range.0 + 1
    };
    let Some(body) = read_range(&content.path, range.0, count) else {
        return error(
            503,
            "Service Unavailable",
            "video content is unavailable",
            "video_content_unavailable",
        );
    };
    let extension = if content.media_type == "video/mp4" {
        ".mp4\""
    } else {
        ".ppm\""
    };
    let mut response = HttpResponse::new(
        if partial { 206 } else { 200 },
        if partial { "Partial Content" } else { "OK" },
        body,
    )
    .with_header("Content-Type", content.media_type.as_str())
    .with_header("Accept-Ranges", "bytes")
    .with_header(
        "Content-Disposition",
        format!("attachment; filename=\"{id}{extension}"),
    );
    if partial {
        response = response.with_header(
            "Content-Range",
            format!("bytes {}-{}/{}", range.0, range.1, content.bytes),
        );
    }
    response
}

fn delete_video(id: &str, service: &VideoJobService) -> HttpResponse {
    let deleted = service.delete(id);
    match (&deleted.result, &deleted.job) {
        (VideoJobResult::Ok, Some(job)) => HttpResponse::new(
            200,
            "OK",
            dump(
                &Value::object()
                    .with("id", job.id.as_str())
                    .with("object", "video.deleted")
                    .with("deleted", true),
            ),
        ),
        _ => error(404, "Not Found", "video job not found", "video_not_found"),
    }
}

/// `(id, content)` of `/v1/videos/<id>[/content]`.
fn parse_video_path(path: &str) -> Option<(&str, bool)> {
    let mut remainder = path.strip_prefix("/v1/videos/")?;
    let mut content = false;
    if let Some(stripped) = remainder.strip_suffix("/content") {
        remainder = stripped;
        content = true;
    }
    if remainder.is_empty() || remainder.contains('/') {
        return None;
    }
    Some((remainder, content))
}

pub fn is_video_api_path(path: &str) -> bool {
    path == VIDEOS_PATH || path.starts_with("/v1/videos/")
}

pub fn handle_video_api_request(
    request: &HttpRequest,
    service: &Arc<VideoJobService>,
) -> Result<HttpResponse> {
    let method_not_allowed = || {
        error(
            405,
            "Method Not Allowed",
            "method is not allowed",
            "method_not_allowed",
        )
    };
    if request.path == VIDEOS_PATH {
        return Ok(if request.method == "POST" {
            create_video(request, service)
        } else {
            method_not_allowed()
        });
    }
    let Some((id, content)) = parse_video_path(&request.path) else {
        return Ok(error(
            404,
            "Not Found",
            "video job not found",
            "video_not_found",
        ));
    };
    Ok(match (content, request.method.as_str()) {
        (true, "GET") => video_content(id, request, service),
        (false, "GET") => lookup_video(id, service),
        (false, "DELETE") => delete_video(id, service),
        _ => method_not_allowed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges() {
        assert_eq!(parse_range("bytes=0-4", 14), Some((0, 4)));
        assert_eq!(parse_range("bytes=-3", 14), Some((11, 13)));
        assert_eq!(parse_range("bytes=10-", 14), Some((10, 13)));
        assert_eq!(parse_range("bytes=5-100", 14), Some((5, 13)));
        for invalid in [
            "bytes=99-100",
            "bytes=4-2",
            "bytes=0-1,3-4",
            "bytes=-0",
            "items=0-1",
            "bytes=+1-2",
        ] {
            assert_eq!(parse_range(invalid, 14), None, "{invalid}");
        }
    }
}
