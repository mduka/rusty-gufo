//! `multipart/form-data` parsing for the ASR, video and image APIs.
//!
//! The C++ server carries three independent parsers
//! (`audio_asr_api.cpp`, `video_api.cpp`, `image_api.cpp`) whose parameter
//! syntax, limits, boundary search and error messages differ; the messages
//! reach clients in 400 responses. Each dialect is reproduced here as its
//! own entry point sharing only byte helpers:
//!
//! - [`asr`]: `ExtractBoundary` + `ParseMultipart`: unique field names, at
//!   most 12 fields, part headers up to 16 KiB, the last
//!   `Content-Disposition` wins, lines without a colon are ignored, and the
//!   final `--` ends parsing without checking what follows.
//! - [`video`]: quote- and escape-aware parameters, a boundary up to 200
//!   bytes, at most 16 fields, strict headers, nothing but an optional CRLF
//!   after the final boundary. Fields are handed to a callback in order so
//!   the caller's per-field errors keep their C++ precedence.
//! - [`image`]: RFC 7578 quoted-string parameters with escapes, a
//!   case-sensitive media type prefix, boundaries only where the delimiter
//!   is followed by CRLF or `--`, at most 32 fields; fields also go to a
//!   callback in order.
//!
//! Errors are the C++ messages as strings.

/// The first index of `needle` in `haystack` at or after `from`.
pub fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > haystack.len() {
        return None;
    }
    if needle.is_empty() {
        return Some(from);
    }
    haystack[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|position| position + from)
}

/// C `isspace` in the C locale.
fn is_c_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// Trims C `isspace` bytes on both sides.
fn trim_c_space(mut value: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = value {
        if !is_c_space(*first) {
            break;
        }
        value = rest;
    }
    while let [rest @ .., last] = value {
        if !is_c_space(*last) {
            break;
        }
        value = rest;
    }
    value
}

fn trim_str(value: &str) -> &str {
    // Only ASCII bytes are trimmed, so UTF-8 boundaries are kept.
    value.trim_matches(|c: char| c.is_ascii() && is_c_space(c as u8))
}

fn lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The ASR dialect (`src/cli/serve/audio_asr_api.cpp`).
pub mod asr {
    use std::collections::HashMap;

    use super::{find, lossy, trim_str};

    /// Part headers larger than this are rejected (`kMaximumTextFieldBytes`).
    pub const MAXIMUM_HEADER_BYTES: usize = 16 << 10;
    pub const MAXIMUM_FIELDS: usize = 12;

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct Part {
        /// The `filename` parameter, empty when absent.
        pub filename: String,
        pub body: Vec<u8>,
    }

    /// `Parameter`: the first `;`-separated item named `name`
    /// (case-insensitive), without surrounding quotes; `None` when absent or
    /// when that value contains CR or LF.
    pub fn parameter(value: &str, name: &str) -> Option<String> {
        let target = name.to_ascii_lowercase();
        for item in value.split(';') {
            let item = trim_str(item);
            let Some(equals) = item.find('=') else {
                continue;
            };
            if trim_str(&item[..equals]).to_ascii_lowercase() != target {
                continue;
            }
            let mut parameter = trim_str(&item[equals + 1..]);
            if parameter.len() >= 2 && parameter.starts_with('"') && parameter.ends_with('"') {
                parameter = &parameter[1..parameter.len() - 1];
            }
            if parameter.contains(['\r', '\n']) {
                return None;
            }
            return Some(parameter.to_owned());
        }
        None
    }

    /// `ExtractBoundary`: the boundary of a `multipart/form-data` content
    /// type (1..=70 bytes, no CR/LF).
    pub fn boundary(content_type: &str) -> Option<String> {
        let media_type = match content_type.find(';') {
            Some(semicolon) => &content_type[..semicolon],
            None => content_type,
        };
        if !trim_str(media_type).eq_ignore_ascii_case("multipart/form-data") {
            return None;
        }
        let parsed = parameter(content_type, "boundary")?;
        if parsed.is_empty() || parsed.len() > 70 || parsed.contains(['\r', '\n']) {
            return None;
        }
        Some(parsed)
    }

    /// `ParseMultipart`: the fields by name.
    pub fn parse(body: &[u8], boundary: &str) -> Result<HashMap<String, Part>, String> {
        let delimiter = format!("--{boundary}");
        let delimiter = delimiter.as_bytes();
        if !body.starts_with(delimiter) {
            return Err("multipart body does not start with its boundary".into());
        }
        let next_marker = [b"\r\n".as_slice(), delimiter].concat();
        let mut output = HashMap::new();
        let mut cursor = delimiter.len();
        loop {
            let rest = body.get(cursor..).unwrap_or_default();
            if rest.starts_with(b"--") {
                return Ok(output);
            }
            if !rest.starts_with(b"\r\n") {
                return Err("multipart boundary is malformed".into());
            }
            cursor += 2;
            let headers_end = match find(body, b"\r\n\r\n", cursor) {
                Some(end) if end - cursor <= MAXIMUM_HEADER_BYTES => end,
                _ => return Err("multipart part headers are malformed".into()),
            };
            let mut disposition = String::new();
            let mut line_begin = cursor;
            while line_begin < headers_end {
                let bounded_end = match find(body, b"\r\n", line_begin) {
                    Some(end) if end <= headers_end => end,
                    _ => headers_end,
                };
                let line = lossy(&body[line_begin..bounded_end]);
                if let Some(colon) = line.find(':')
                    && trim_str(&line[..colon]).eq_ignore_ascii_case("content-disposition")
                {
                    disposition = trim_str(&line[colon + 1..]).to_owned();
                }
                if bounded_end == headers_end {
                    break;
                }
                line_begin = bounded_end + 2;
            }
            let name = match parameter(&disposition, "name") {
                Some(name) if !name.is_empty() => name,
                _ => return Err("multipart part is missing a field name".into()),
            };
            let payload_begin = headers_end + 4;
            let Some(next) = find(body, &next_marker, payload_begin) else {
                return Err("multipart body is missing a closing boundary".into());
            };
            let part = Part {
                filename: parameter(&disposition, "filename").unwrap_or_default(),
                body: body[payload_begin..next].to_vec(),
            };
            if output.contains_key(&name) {
                return Err(format!("multipart field is duplicated: {name}"));
            }
            output.insert(name, part);
            if output.len() > MAXIMUM_FIELDS {
                return Err("multipart request contains too many fields".into());
            }
            cursor = next + 2 + delimiter.len();
        }
    }
}

/// The video dialect (`src/cli/serve/video_api.cpp`).
pub mod video {
    use super::{find, lossy, trim_c_space};

    pub const MAXIMUM_HEADER_BYTES: usize = 16 << 10;
    pub const MAXIMUM_FIELDS: usize = 16;

    /// One field of a video form.
    pub struct Field<'a> {
        pub name: String,
        /// Whether the disposition carries a `filename` parameter.
        pub has_filename: bool,
        pub content: &'a [u8],
    }

    fn equal_case_insensitive(left: &[u8], right: &[u8]) -> bool {
        left.eq_ignore_ascii_case(right)
    }

    /// `HeaderSegments`: `;`-separated, trimmed segments outside quotes;
    /// `None` for an unterminated quote or escape.
    fn header_segments(value: &[u8]) -> Option<Vec<&[u8]>> {
        let mut result = Vec::new();
        let mut begin = 0;
        let mut quoted = false;
        let mut escaped = false;
        for (index, &character) in value.iter().enumerate() {
            if escaped {
                escaped = false;
                continue;
            }
            if quoted && character == b'\\' {
                escaped = true;
                continue;
            }
            if character == b'"' {
                quoted = !quoted;
            } else if character == b';' && !quoted {
                result.push(trim_c_space(&value[begin..index]));
                begin = index + 1;
            }
        }
        if quoted || escaped {
            return None;
        }
        result.push(trim_c_space(&value[begin..]));
        Some(result)
    }

    /// `DecodeHeaderParameter`.
    fn decode_header_parameter(value: &[u8]) -> Option<Vec<u8>> {
        let value = trim_c_space(value);
        if value.is_empty() {
            return Some(Vec::new());
        }
        if value[0] != b'"' {
            if value.iter().any(|c| matches!(c, b'"' | b'\r' | b'\n')) {
                return None;
            }
            return Some(value.to_vec());
        }
        if value.len() < 2 || value[value.len() - 1] != b'"' {
            return None;
        }
        let mut output = Vec::new();
        let mut escaped = false;
        for &character in &value[1..value.len() - 1] {
            if escaped {
                output.push(character);
                escaped = false;
            } else if character == b'\\' {
                escaped = true;
            } else if matches!(character, b'"' | b'\r' | b'\n') {
                return None;
            } else {
                output.push(character);
            }
        }
        (!escaped).then_some(output)
    }

    /// `HeaderParameter`: `Err(())` for a malformed header, else the value
    /// of `parameter` if present (at most once).
    #[allow(clippy::result_unit_err)]
    pub fn header_parameter(value: &[u8], parameter: &[u8]) -> Result<Option<Vec<u8>>, ()> {
        let segments = header_segments(value).ok_or(())?;
        if segments.is_empty() {
            return Err(());
        }
        let mut found = None;
        for segment in &segments[1..] {
            let equals = segment.iter().position(|&c| c == b'=').ok_or(())?;
            let name = trim_c_space(&segment[..equals]);
            if !equal_case_insensitive(name, parameter) {
                continue;
            }
            if found.is_some() {
                return Err(());
            }
            found = Some(decode_header_parameter(&segment[equals + 1..]).ok_or(())?);
        }
        Ok(found)
    }

    /// `IsMultipartFormData`.
    pub fn is_multipart_form_data(content_type: &str) -> bool {
        let media_type = match content_type.find(';') {
            Some(semicolon) => &content_type[..semicolon],
            None => content_type,
        };
        equal_case_insensitive(trim_c_space(media_type.as_bytes()), b"multipart/form-data")
    }

    /// `MultipartBoundary`: 1..=200 bytes without CR/LF.
    pub fn boundary(content_type: &str) -> Option<Vec<u8>> {
        match header_parameter(content_type.as_bytes(), b"boundary") {
            Ok(Some(boundary))
                if !boundary.is_empty()
                    && boundary.len() <= 200
                    && !boundary.iter().any(|c| matches!(c, b'\r' | b'\n')) =>
            {
                Some(boundary)
            }
            _ => None,
        }
    }

    /// `ParseMultipartBody`'s framing: calls `on_field` for every field in
    /// order; its error aborts parsing and is returned.
    pub fn parse(
        content_type: &str,
        body: &[u8],
        on_field: &mut dyn FnMut(Field<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        let Some(boundary) = boundary(content_type) else {
            return Err("multipart video request has an invalid boundary".into());
        };
        let delimiter = [b"--".as_slice(), &boundary].concat();
        if !body.starts_with(&delimiter) {
            return Err("multipart video request does not start with its boundary".into());
        }
        let next_boundary = [b"\r\n".as_slice(), &delimiter].concat();
        let mut names: Vec<String> = Vec::new();
        let mut cursor = delimiter.len();
        let mut fields = 0;
        loop {
            let rest = &body[cursor..];
            if rest.starts_with(b"--") {
                cursor += 2;
                if body[cursor..].starts_with(b"\r\n") {
                    cursor += 2;
                }
                if cursor != body.len() {
                    return Err("multipart video request has data after its final boundary".into());
                }
                return Ok(());
            }
            if !rest.starts_with(b"\r\n") {
                return Err("multipart video request has an invalid boundary separator".into());
            }
            cursor += 2;
            let headers_end = match find(body, b"\r\n\r\n", cursor) {
                Some(end) if end - cursor <= MAXIMUM_HEADER_BYTES => end,
                _ => return Err("multipart video request has invalid part headers".into()),
            };
            let mut content_disposition: Vec<u8> = Vec::new();
            let mut headers = &body[cursor..headers_end];
            while !headers.is_empty() {
                let newline = find(headers, b"\r\n", 0);
                let line = &headers[..newline.unwrap_or(headers.len())];
                let Some(colon) = line.iter().position(|&c| c == b':') else {
                    return Err("multipart video request has a malformed part header".into());
                };
                let name = trim_c_space(&line[..colon]);
                let value = trim_c_space(&line[colon + 1..]);
                if equal_case_insensitive(name, b"content-disposition") {
                    if !content_disposition.is_empty() {
                        return Err("multipart video request repeats Content-Disposition".into());
                    }
                    content_disposition = value.to_vec();
                }
                match newline {
                    Some(newline) => headers = &headers[newline + 2..],
                    None => break,
                }
            }
            let invalid = || "multipart video request has invalid Content-Disposition".to_owned();
            let segments = header_segments(&content_disposition).ok_or_else(invalid)?;
            if segments.is_empty() || !equal_case_insensitive(segments[0], b"form-data") {
                return Err(invalid());
            }
            let field_name = match header_parameter(&content_disposition, b"name") {
                Ok(Some(name)) if !name.is_empty() => name,
                _ => return Err(invalid()),
            };
            let has_filename = header_parameter(&content_disposition, b"filename")
                .map_err(|()| invalid())?
                .is_some();
            let content_begin = headers_end + 4;
            let Some(content_end) = find(body, &next_boundary, content_begin) else {
                return Err("multipart video request is missing a closing boundary".into());
            };
            fields += 1;
            let name = lossy(&field_name);
            if fields > MAXIMUM_FIELDS {
                return Err("multipart video request contains too many fields".into());
            }
            if names.contains(&name) {
                return Err("multipart video request repeats a field".into());
            }
            names.push(name.clone());
            on_field(Field {
                name,
                has_filename,
                content: &body[content_begin..content_end],
            })?;
            cursor = content_end + next_boundary.len();
        }
    }
}

/// The image dialect (`src/cli/serve/image_api.cpp`).
pub mod image {
    use super::{find, lossy};

    pub const MAXIMUM_HEADER_BYTES: usize = 16384;
    pub const MAXIMUM_FIELDS: usize = 32;

    fn trim_blank(mut value: &[u8]) -> &[u8] {
        while let [b' ' | b'\t', rest @ ..] = value {
            value = rest;
        }
        while let [rest @ .., b' ' | b'\t'] = value {
            value = rest;
        }
        value
    }

    /// `Parameter`: RFC 7578 quoted-string parameters after the first `;`
    /// (the first item is the media or disposition type). Empty when
    /// absent; escape errors anywhere in the list are errors.
    pub fn parameter(value: &[u8], key: &[u8]) -> Result<Vec<u8>, String> {
        let mut value = value;
        while !value.is_empty() {
            let Some(separator) = value.iter().position(|&c| c == b';') else {
                break;
            };
            value = trim_blank(&value[separator + 1..]);
            let Some(equal) = value.iter().position(|&c| c == b'=') else {
                break;
            };
            let name = trim_blank(&value[..equal]).to_vec();
            value = &value[equal + 1..];
            let mut result = Vec::new();
            if value.first() == Some(&b'"') {
                value = &value[1..];
                let mut closed = false;
                while let [first, rest @ ..] = value {
                    let mut c = *first;
                    value = rest;
                    if c == b'"' {
                        closed = true;
                        break;
                    }
                    if c == b'\\' {
                        let [escaped, rest @ ..] = value else {
                            return Err("truncated multipart escape".into());
                        };
                        c = *escaped;
                        value = rest;
                    }
                    if c < 32 {
                        return Err("control character in multipart parameter".into());
                    }
                    result.push(c);
                }
                if !closed {
                    return Err("unclosed multipart parameter".into());
                }
            } else {
                let end = value.iter().position(|&c| c == b';');
                result = trim_blank(&value[..end.unwrap_or(value.len())]).to_vec();
                value = match end {
                    Some(end) => &value[end..],
                    None => &[],
                };
            }
            if name == key {
                return Ok(result);
            }
        }
        Ok(Vec::new())
    }

    /// One field of an image form.
    pub struct Field<'a> {
        pub name: String,
        pub content: &'a [u8],
    }

    /// `Multipart`'s framing: calls `on_field` for each field in order; its
    /// error aborts parsing and is returned.
    pub fn parse(
        content_type: &str,
        body: &[u8],
        on_field: &mut dyn FnMut(Field<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        if !content_type.starts_with("multipart/form-data") {
            return Err("image edits require multipart/form-data".into());
        }
        let boundary = parameter(content_type.as_bytes(), b"boundary")?;
        if boundary.is_empty()
            || boundary.len() > 70
            || boundary.iter().any(|c| matches!(c, b'\r' | b'\n'))
        {
            return Err("invalid multipart boundary".into());
        }
        let marker = [b"--".as_slice(), &boundary].concat();
        let mut body = body;
        if !body.starts_with(&marker) {
            return Err("multipart body does not start with boundary".into());
        }
        body = &body[marker.len()..];
        let delimiter = [b"\r\n".as_slice(), &marker].concat();
        let mut fields = 0;
        loop {
            if body.starts_with(b"--") {
                body = &body[2..];
                if !body.is_empty() && body != b"\r\n" {
                    return Err("data after final multipart boundary".into());
                }
                return Ok(());
            }
            if !body.starts_with(b"\r\n") {
                return Err("invalid multipart separator".into());
            }
            body = &body[2..];
            let header_end = match find(body, b"\r\n\r\n", 0) {
                Some(end) if end <= MAXIMUM_HEADER_BYTES => end,
                _ => return Err("invalid multipart headers".into()),
            };
            let mut headers = &body[..header_end];
            let mut name = Vec::new();
            let mut disposition_seen = false;
            while !headers.is_empty() {
                let end = find(headers, b"\r\n", 0);
                let line = &headers[..end.unwrap_or(headers.len())];
                let Some(colon) = line.iter().position(|&c| c == b':') else {
                    return Err("malformed multipart header".into());
                };
                if line[..colon].eq_ignore_ascii_case(b"content-disposition") {
                    if disposition_seen {
                        return Err("duplicate multipart disposition".into());
                    }
                    disposition_seen = true;
                    let value = trim_blank(&line[colon + 1..]);
                    if !value.starts_with(b"form-data") {
                        return Err("expected form-data disposition".into());
                    }
                    name = parameter(value, b"name")?;
                }
                match end {
                    Some(end) => headers = &headers[end + 2..],
                    None => break,
                }
            }
            if name.is_empty() {
                return Err("multipart field has no name".into());
            }
            body = &body[header_end + 4..];
            let mut end = 0;
            loop {
                let Some(found) = find(body, &delimiter, end) else {
                    return Err("missing final multipart boundary".into());
                };
                let suffix = &body[found + delimiter.len()..];
                if suffix.starts_with(b"\r\n") || suffix.starts_with(b"--") {
                    end = found;
                    break;
                }
                end = found + 1;
            }
            let data = &body[..end];
            body = &body[end + delimiter.len()..];
            fields += 1;
            if fields > MAXIMUM_FIELDS {
                return Err("too many multipart fields".into());
            }
            on_field(Field {
                name: lossy(&name),
                content: data,
            })?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(boundary: &str, fields: &[(&str, &str)]) -> Vec<u8> {
        let mut body = String::new();
        for (name, value) in fields {
            body.push_str(&format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ));
        }
        body.push_str(&format!("--{boundary}--\r\n"));
        body.into_bytes()
    }

    #[test]
    fn asr_dialect() {
        assert_eq!(
            asr::boundary("Multipart/Form-Data; boundary=\"abc\"").as_deref(),
            Some("abc")
        );
        assert!(asr::boundary("text/plain; boundary=abc").is_none());
        let parts =
            asr::parse(&form("abc", &[("file", "RIFF"), ("model", "m")]), "abc").expect("parse");
        assert_eq!(parts["file"].body, b"RIFF");
        let duplicate = asr::parse(&form("abc", &[("a", "1"), ("a", "2")]), "abc");
        assert_eq!(duplicate, Err("multipart field is duplicated: a".into()));
        let many: Vec<(String, &str)> = (0..13).map(|i| (format!("f{i}"), "x")).collect();
        let many: Vec<(&str, &str)> = many.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        assert_eq!(
            asr::parse(&form("abc", &many), "abc"),
            Err("multipart request contains too many fields".into())
        );
    }

    #[test]
    fn video_dialect() {
        let body = form("b", &[("model", "m"), ("prompt", "p")]);
        let mut seen = Vec::new();
        video::parse("multipart/form-data; boundary=\"b\"", &body, &mut |field| {
            seen.push((field.name, field.content.to_vec()));
            Ok(())
        })
        .expect("parse");
        assert_eq!(
            seen,
            vec![
                ("model".into(), b"m".to_vec()),
                ("prompt".into(), b"p".to_vec())
            ]
        );
        let repeated = form("b", &[("a", "1"), ("a", "2")]);
        assert_eq!(
            video::parse(
                "multipart/form-data; boundary=b",
                &repeated,
                &mut |_| Ok(())
            ),
            Err("multipart video request repeats a field".into())
        );
        assert!(video::boundary("multipart/form-data; boundary=\"a;b\"").is_some());
        assert!(video::boundary("multipart/form-data; boundary=\"unterminated").is_none());
    }

    #[test]
    fn image_dialect() {
        assert_eq!(
            image::parameter(b"form-data; name=\"a\\\"b\"; filename=x", b"name"),
            Ok(b"a\"b".to_vec())
        );
        assert_eq!(
            image::parameter(b"form-data; name=\"a", b"name"),
            Err("unclosed multipart parameter".into())
        );
        let body = form("b", &[("prompt", "x")]);
        let mut names = Vec::new();
        image::parse("multipart/form-data; boundary=b", &body, &mut |field| {
            names.push(field.name);
            Ok(())
        })
        .expect("parse");
        assert_eq!(names, vec!["prompt".to_owned()]);
        assert_eq!(
            image::parse("Multipart/form-data; boundary=b", &body, &mut |_| Ok(())),
            Err("image edits require multipart/form-data".into())
        );
    }
}
