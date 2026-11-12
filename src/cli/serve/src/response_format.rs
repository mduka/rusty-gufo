//! `response_format` and Responses `text.format` as a grammar
//! (`src/cli/serve/response_format.hpp`), per the Chat Completions wire
//! contract: <https://developers.openai.com/api/docs/guides/structured-outputs>.

use std::sync::Arc;

use gufo_core::json::Value;
use gufo_core::json_constraint::JsonConstraint;

/// The grammar of `format`: `None` for absent, null or `text`. `responses`
/// accepts the flat Responses form, whose `json_schema` members sit beside
/// `type`.
pub fn parse_response_format(
    format: Option<&Value>,
    responses: bool,
) -> Result<Option<Arc<JsonConstraint>>, String> {
    let Some(format) = format.filter(|format| !format.is_null()) else {
        return Ok(None);
    };
    if responses && format.is_object() && format.member_str("type", "") == "json_schema" {
        let mut chat = Value::object();
        chat.set("type", "json_schema");
        let mut specification = Value::object();
        for (key, value) in format.members() {
            if key != "type" {
                specification.set(key, value.clone());
            }
        }
        chat.set("json_schema", specification);
        return parse_response_format(Some(&chat), false);
    }
    let invalid = |message: &str| Err(message.to_owned());
    if !format.is_object() {
        return invalid("'response_format' must be an object");
    }
    let kind = format.member_str("type", "");
    if kind == "text" || kind == "json_object" {
        if format.len() != 1 {
            return invalid("unexpected response_format member");
        }
        return if kind == "text" {
            Ok(None)
        } else {
            JsonConstraint::object()
                .map(Some)
                .map_err(|e| e.message().to_owned())
        };
    }
    if kind != "json_schema" || format.len() != 2 {
        return invalid("response_format type must be text, json_object or json_schema");
    }
    let Some(specification) = format.find("json_schema").filter(|s| s.is_object()) else {
        return invalid("'json_schema' must be an object");
    };
    if specification
        .members()
        .iter()
        .any(|(key, _)| !matches!(key.as_str(), "name" | "description" | "schema" | "strict"))
    {
        return invalid("unexpected json_schema member");
    }
    let name = specification.member_str("name", "");
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
    {
        return invalid("json_schema.name must contain 1–64 letters, digits, '_' or '-'");
    }
    if specification
        .find("description")
        .is_some_and(|d| !d.is_string() || d.str().len() > 8192)
    {
        return invalid("json_schema.description must be a string of at most 8192 bytes");
    }
    let strict = specification.find("strict");
    if strict.is_some_and(|s| !s.is_null() && !matches!(s, Value::Bool(_))) {
        return invalid("json_schema.strict must be a boolean or null");
    }
    let Some(schema) = specification.find("schema") else {
        return invalid("json_schema.schema is required");
    };
    JsonConstraint::compile_response(schema, strict.is_some_and(|s| s.as_bool(false)))
        .map(Some)
        .map_err(|e| e.message().to_owned())
}

/// The `json_schema.description` of a Chat Completions `response_format`,
/// or of a Responses `text.format`, empty without one.
pub fn response_format_description(format: Option<&Value>, responses: bool) -> String {
    let Some(format) = format else {
        return String::new();
    };
    if responses {
        return format.member_str("description", "");
    }
    format
        .find("json_schema")
        .map_or_else(String::new, |s| s.member_str("description", ""))
}
