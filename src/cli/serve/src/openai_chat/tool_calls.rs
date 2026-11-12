//! Tool calls parsed from generated text: Qwen XML and JSON calls, DeepSeek
//! DSML calls, and the schema checks of their arguments.

use super::*;

pub(super) fn arguments_json(arguments: &[ToolArgument]) -> String {
    let mut object = Value::object();
    for argument in arguments {
        if argument.is_string {
            object.set(&argument.name, argument.value.as_str());
            continue;
        }
        match json::parse_str(&argument.value) {
            Ok(value) => object.set(&argument.name, value),
            Err(_) => object.set(&argument.name, argument.value.as_str()),
        }
    }
    object.dump().unwrap_or_default()
}

/// Qwen's XML-like arguments carry no type marker; the advertised schema
/// tells a string such as 42 from the JSON number 42.
/// The target of a local JSON pointer `reference` (`#` or `#/...`) in
/// `root`, or `None` for anything else.
pub(super) fn resolve_json_pointer<'a>(root: &'a Value, reference: &Value) -> Option<&'a Value> {
    let Value::String(reference) = reference else {
        return None;
    };
    let pointer = reference.strip_prefix('#')?;
    if pointer.is_empty() {
        return Some(root);
    }
    let mut target = root;
    for segment in pointer.strip_prefix('/')?.split('/') {
        let mut decoded = String::with_capacity(segment.len());
        let mut chars = segment.chars();
        while let Some(c) = chars.next() {
            decoded.push(match (c, c == '~') {
                (_, true) => match chars.next()? {
                    '0' => '~',
                    '1' => '/',
                    _ => return None,
                },
                (c, false) => c,
            });
        }
        target = match target {
            Value::Array(items) => items.get(decoded.parse::<usize>().ok()?)?,
            _ => target.find(&decoded)?,
        };
    }
    Some(target)
}

/// `schema` with its `$ref` chain followed within `root`; `None` for a
/// reference cycle or a dangling reference, which non-strict schemas may
/// carry.
pub(super) fn resolve_tool_schema<'a>(root: &'a Value, schema: &'a Value) -> Option<&'a Value> {
    let mut target = schema;
    let mut seen: Vec<*const Value> = Vec::new();
    while let Some(reference) = target.find("$ref") {
        if seen.contains(&std::ptr::from_ref(target)) {
            return None;
        }
        seen.push(target);
        target = resolve_json_pointer(root, reference)?;
        // A parameter pointing to its containing tool schema is recursive:
        // native best-effort grammar cannot enforce that, so no scalar type
        // is inferred from the outer container.
        if std::ptr::eq(target, root) {
            return None;
        }
    }
    Some(target)
}

/// Whether `value` fits the property `schema` of the tool schema `root`;
/// an unknown or unresolvable type accepts, so text is retained.
pub(super) fn schema_accepts(root: &Value, schema: &Value, value: &Value) -> bool {
    schema_accepts_at(root, schema, value, 0)
}

pub(super) fn schema_accepts_at(root: &Value, schema: &Value, value: &Value, depth: usize) -> bool {
    let Some(schema) = resolve_tool_schema(root, schema).filter(|_| depth <= 16) else {
        return true;
    };
    // Value kinds, as llama.cpp's value_types(), rather than guesses from
    // the lexeme: {"const":"42"} is text even without an explicit type.
    let same_kind = |candidate: &Value| {
        (candidate.is_string() && value.is_string())
            || (candidate.is_number() && value.is_number())
            || (candidate.is_bool() && value.is_bool())
            || (candidate.is_null() && value.is_null())
            || (candidate.is_array() && value.is_array())
            || (candidate.is_object() && value.is_object())
    };
    if let Some(constant) = schema.find("const") {
        return same_kind(constant);
    }
    if let Some(Value::Array(values)) = schema.find("enum") {
        return values.iter().any(same_kind);
    }
    let matches = |kind: &str| match kind {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_number() && value.as_double(0.0).floor() == value.as_double(0.0),
        "boolean" => value.is_bool(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    };
    if let Some(kind) = schema.find("type") {
        return match kind {
            Value::String(kind) => matches(kind),
            Value::Array(kinds) => kinds.iter().any(|k| k.is_string() && matches(k.str())),
            _ => false,
        };
    }
    for name in ["anyOf", "oneOf"] {
        if let Some(Value::Array(choices)) = schema.find(name) {
            return choices
                .iter()
                .any(|choice| schema_accepts_at(root, choice, value, depth + 1));
        }
    }
    if schema.contains("properties")
        || schema
            .find("additionalProperties")
            .is_some_and(|extra| !matches!(extra, Value::Bool(true)))
    {
        return value.is_object();
    }
    if let Some(Value::Array(parts)) = schema.find("allOf") {
        return parts
            .iter()
            .all(|part| schema_accepts_at(root, part, value, depth + 1));
    }
    if schema.contains("items") || schema.contains("prefixItems") {
        return value.is_array();
    }
    if schema.contains("pattern") || schema.contains("minLength") || schema.contains("maxLength") {
        return value.is_string();
    }
    true
}

/// Whether the property `schema` of the tool schema `root` admits only
/// strings, so an unfinished value can stream as a JSON string prefix.
pub(super) fn string_only(root: &Value, schema: &Value, depth: usize) -> bool {
    if depth > 32 {
        return false;
    }
    let Some(resolved) = resolve_tool_schema(root, schema) else {
        return false;
    };
    if !std::ptr::eq(resolved, schema) {
        return string_only(root, resolved, depth + 1);
    }
    if let Some(Value::String(kind)) = schema.find("type") {
        return kind == "string";
    }
    if let Some(value) = schema.find("const") {
        return value.is_string();
    }
    if let Some(Value::Array(values)) = schema.find("enum") {
        return !values.is_empty() && values.iter().all(Value::is_string);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(Value::Array(values)) = schema.find(key) {
            return !values.is_empty()
                && values
                    .iter()
                    .all(|value| string_only(root, value, depth + 1));
        }
    }
    false
}

/// The arguments of `call` as JSON without the closing brace.
pub(super) fn arguments_prefix(call: &ParsedToolCall) -> String {
    let mut prefix = arguments_json(&call.arguments);
    prefix.pop();
    prefix
}

/// `call`'s arguments followed by the unfinished string `value` of `name`,
/// unquoted at its end.
pub(super) fn partial_string_argument(call: &mut ParsedToolCall, name: &str, value: &str) {
    // Identical repeated parameters are discarded by the full parser.
    if call.arguments.iter().any(|argument| argument.name == name) {
        return;
    }
    let mut prefix = arguments_prefix(call);
    if !call.arguments.is_empty() {
        prefix.push(',');
    }
    prefix.push_str(&Value::from(name).dump().unwrap_or_default());
    prefix.push(':');
    let mut quoted = Value::from(value).dump().unwrap_or_default();
    quoted.pop();
    prefix.push_str(&quoted);
    call.partial_arguments = Some(prefix);
}

/// `value` without a suffix that may begin `delimiter`, held until its next
/// byte disambiguates it.
pub(super) fn before_partial_delimiter<'v>(value: &'v str, delimiter: &str) -> &'v str {
    for size in (1..=value.len().min(delimiter.len())).rev() {
        if value.is_char_boundary(value.len() - size)
            && delimiter
                .as_bytes()
                .starts_with(&value.as_bytes()[value.len() - size..])
        {
            return &value[..value.len() - size];
        }
    }
    value
}

/// Where `part`, a slice of `text`, begins in it. `trim` cuts both ends, so
/// a trimmed rest is no suffix of `text`.
pub(super) fn offset_in(text: &str, part: &str) -> usize {
    let offset = part.as_ptr() as usize - text.as_ptr() as usize;
    debug_assert!(offset + part.len() <= text.len(), "a slice of the text");
    offset
}

/// Byte-oriented cursor over `&str` (all delimiters are ASCII or complete
/// UTF-8 markers, so slicing at found positions stays on boundaries).
/// `spans` receives each complete call's byte range, markup included.
/// `partial` (streaming) also returns the first unfinished call of a
/// declared tool under the canonical header, with its arguments so far as
/// `partial_arguments`, and stops there.
pub(super) fn parse_qwen_calls(
    text: &str,
    tools: &[ChatTool],
    calls: &mut Vec<ParsedToolCall>,
    mut spans: Option<&mut Vec<(usize, usize)>>,
    require_schema: bool,
    partial: bool,
) {
    const START: &str = "<tool_call>";
    const END: &str = "</tool_call>";
    let mut quotes = QuoteTracker::default();
    quotes.reset(text, 0);
    let mut cursor = 0;
    while let Some(found) = text[cursor..].find(START) {
        let marker_begin = cursor + found;
        if quotes.quoted_at(marker_begin) || (partial && quotes.unclosed_at(marker_begin)) {
            cursor = marker_begin + START.len();
            continue;
        }
        let canonical_header = text[marker_begin..].starts_with("<tool_call>\n<function=");
        let begin = marker_begin + START.len();
        cursor = begin; // A malformed call may be followed by a valid call.
        let mut body = trim(&text[begin..]);
        let consume = |body: &mut &str, tag: &str| {
            *body = trim(body);
            match body.strip_prefix(tag) {
                Some(rest) => {
                    *body = rest;
                    true
                }
                None => false,
            }
        };
        let mut call = ParsedToolCall {
            id: String::new(),
            name: String::new(),
            arguments: Vec::new(),
            partial_arguments: None,
        };
        let mut complete = false;
        // An unfinished call is streamed only for a declared tool under the
        // native grammar's header, outside quotation.
        let mut tentative = false;
        if consume(&mut body, "<function=") {
            let Some(name_end) = body.find('>') else {
                continue;
            };
            call.name = trim(&body[..name_end]).to_owned();
            body = &body[name_end + 1..];
            let mut schema: Option<Option<Value>> = None;
            for tool in tools {
                if tool.name == call.name {
                    schema = Some(try_parse_json(&tool.parameters_json));
                    break;
                }
            }
            let schema = schema.flatten();
            if call.name.is_empty() {
                continue;
            }
            tentative = partial
                && canonical_header
                && schema.is_some()
                && !require_schema
                && !quotes.unclosed_at(marker_begin);
            let root = schema.as_ref();
            let properties = root
                .and_then(|root| resolve_tool_schema(root, root))
                .and_then(|object| object.find("properties"));
            let accepts = |property: &Value, value: &Value| {
                root.is_none_or(|root| schema_accepts(root, property, value))
            };
            let mut valid = true;
            let mut canonical_parameters = canonical_header;
            while consume(&mut body, "<parameter=") {
                let Some(name_end) = body.find('>') else {
                    valid = false;
                    break;
                };
                // The schema's exact key is kept: llama.cpp's PEG matches
                // this spelling too, but its JSON mapper trims it, which
                // would corrupt a declared key with surrounding spaces.
                let spelled = &body[..name_end];
                let name = if properties.is_some_and(|p| p.is_object() && p.contains(spelled)) {
                    spelled
                } else {
                    trim(spelled)
                }
                .to_owned();
                body = &body[name_end + 1..];
                // A framed value ends at a closing tag on its own line;
                // other closing tags inside it are data.
                let framed = body.starts_with('\n') || body.starts_with("\r\n");
                canonical_parameters &= framed;
                let mut close = body.find("</parameter>");
                if framed {
                    // The canonical delimiter includes the following newline:
                    // a line such as "</parameter> is literal" is data.
                    // Compact legacy headers also admit compact closers; a
                    // canonical header needs the complete delimiter, as in
                    // llama.cpp.
                    const FRAMED_CLOSE: &str = "\n</parameter>";
                    let mut framed_close = body.find(FRAMED_CLOSE);
                    while let Some(at) = framed_close {
                        let tail = &body[at + FRAMED_CLOSE.len()..];
                        if tail.starts_with('\n')
                            || tail.starts_with("\r\n")
                            || (!canonical_parameters
                                && (trim(tail).starts_with("</function>")
                                    || trim(tail).starts_with("<parameter=")))
                        {
                            break;
                        }
                        framed_close = body[at + 1..].find(FRAMED_CLOSE).map(|next| at + 1 + next);
                    }
                    close = framed_close.map(|framed| framed + 1);
                    // Never recover a nested example from an unfinished value.
                    cursor = close.map_or(text.len(), |close| {
                        offset_in(text, body) + close + "</parameter>".len()
                    });
                }
                let Some(close) = close.filter(|_| !name.is_empty()) else {
                    // Arguments after a rejected parameter are never
                    // streamed; an unfinished string value is.
                    let property = properties.and_then(|p| p.find(&name));
                    if tentative
                        && framed
                        && !name.is_empty()
                        && property.is_none_or(|p| root.is_some_and(|r| string_only(r, p, 0)))
                    {
                        let frame = if body.starts_with("\r\n") {
                            "\r\n"
                        } else {
                            "\n"
                        };
                        let mut value = &body[frame.len()..];
                        value = before_partial_delimiter(value, "\n</parameter>\n");
                        value = before_partial_delimiter(value, "\n</parameter>\r\n");
                        if frame == "\r\n" {
                            value = value.strip_suffix('\r').unwrap_or(value);
                        }
                        partial_string_argument(&mut call, &name, value);
                    }
                    valid = false;
                    break;
                };
                // An unframed, unterminated call is recovered at the next
                // complete opener; a literal <tool_call> alone is data.
                if !framed
                    && body.find(START).is_some_and(|nested| {
                        nested < close
                            && trim(&body[nested + START.len()..]).starts_with("<function=")
                    })
                {
                    valid = false;
                    break;
                }
                let value = strip_framing_newlines(&body[..close]);
                let property = properties.and_then(|p| p.find(&name));
                let string_allowed = property.is_none_or(|p| accepts(p, &Value::from(value)));
                // Prefer text if the schema permits it: parsing ambiguous
                // scalars as JSON would change a declared string type. A
                // union that also admits other types tries those first, as
                // llama.cpp's qwen3-coder parser does.
                let typed = property.filter(|_| string_allowed).and_then(|p| {
                    try_parse_json(trim(value))
                        .filter(|typed| !typed.is_string() && accepts(p, typed))
                });
                let is_string = string_allowed && typed.is_none();
                let mut raw = if is_string { value } else { trim(value) }.to_owned();
                if let Some(property) = property.filter(|_| !is_string && typed.is_none()) {
                    let parses = |raw: &str| {
                        try_parse_json(raw).is_some_and(|parsed| accepts(property, &parsed))
                    };
                    if !parses(&raw) {
                        raw = python_literals_to_json(&raw);
                    }
                    if !parses(&raw) {
                        valid = false;
                        break;
                    }
                }
                // Models sometimes repeat a parameter: an identical copy is
                // harmless, conflicting copies leave no safe choice.
                match call.arguments.iter().find(|a| a.name == name) {
                    None => call.arguments.push(ToolArgument {
                        name,
                        value: raw,
                        is_string,
                    }),
                    Some(previous) if previous.value == raw && previous.is_string == is_string => {}
                    Some(_) => {
                        valid = false;
                        break;
                    }
                }
                body = &body[close + "</parameter>".len()..];
            }
            complete = valid && consume(&mut body, "</function>") && consume(&mut body, END);
        } else {
            // JSON calls encode their argument types. The envelope ends at
            // the first closing (or a new opening) marker outside a JSON
            // string: an unfinished string owns the rest of the text, tool
            // tags included, so recovery never runs a call quoted in an
            // argument. A rejected envelope is skipped as a whole.
            let close = json_envelope_end(body, START, END);
            cursor = offset_in(text, body) + close;
            if body[close..].starts_with(END) {
                cursor += END.len();
                if let Some(parsed) = try_parse_json(trim(&body[..close])).filter(|p| p.is_object())
                {
                    call.name = parsed.member_str("name", "");
                    if let Some(Value::Object(arguments)) = parsed.find("arguments")
                        && !call.name.is_empty()
                    {
                        for (name, value) in arguments {
                            call.arguments.push(ToolArgument {
                                name: name.clone(),
                                value: match value {
                                    Value::String(s) => s.clone(),
                                    other => other.dump().unwrap_or_default(),
                                },
                                is_string: value.is_string(),
                            });
                        }
                        complete = true;
                        body = &body[close + END.len()..];
                    }
                }
            }
        }
        if complete && !tools.is_empty() && !tools.iter().any(|t| t.name == call.name) {
            complete = false;
        }
        // A call in an unfinished quotation is ambiguous: it counts only for a
        // declared tool whose complete arguments satisfy its schema.
        if complete && (require_schema || quotes.unclosed_at(marker_begin)) {
            complete = tools
                .iter()
                .find(|tool| tool.name == call.name)
                .is_some_and(|tool| arguments_satisfy(&tool.parameters_json, &call.arguments));
        }
        if complete || tentative {
            call.id = random_id("call_", 20);
            if !complete && call.partial_arguments.is_none() {
                call.partial_arguments = Some(arguments_prefix(&call));
            }
            calls.push(call);
            if !complete {
                break;
            }
            cursor = offset_in(text, body);
            // Parameter bytes belong to the call, not to prose quotation
            // state: an unmatched backtick in a file write must not quote
            // the next call.
            quotes.reset(text, cursor);
            if let Some(spans) = spans.as_deref_mut() {
                spans.push((marker_begin, cursor));
            }
        }
    }
}

/// Whether `arguments` encode a document the (non-strict) parameters schema
/// accepts; an uncompilable schema accepts nothing.
pub(super) fn arguments_satisfy(parameters_json: &str, arguments: &[ToolArgument]) -> bool {
    let Ok(schema) = json::parse_str(parameters_json) else {
        return false;
    };
    let Ok(grammar) = JsonConstraint::compile(&schema, false) else {
        return false;
    };
    let Ok(mut state) = grammar.start() else {
        return false;
    };
    for byte in arguments_json(arguments).bytes() {
        match grammar.advance(&state, byte) {
            Ok(next) => state = next,
            Err(_) => return false,
        }
    }
    grammar.complete(&state)
}

/// The offset of the first `end` or `start` marker in `body` outside a JSON
/// string, or `body.len()` when an unfinished string (or no marker) runs to
/// the end.
pub(super) fn json_envelope_end(body: &str, start: &str, end: &str) -> usize {
    let bytes = body.as_bytes();
    let (mut in_string, mut escaped) = (false, false);
    for (at, &byte) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
        } else if byte == b'"' {
            in_string = true;
        } else if bytes[at..].starts_with(end.as_bytes())
            || bytes[at..].starts_with(start.as_bytes())
        {
            return at;
        }
    }
    body.len()
}

pub(super) fn attribute(tag: &str, name: &str) -> Option<String> {
    let prefix = format!("{name}=\"");
    let start = tag.find(&prefix)? + prefix.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_owned())
}

/// How many of `arguments` the advertised schema of `name` accepts, in
/// order: a `string="false"` value that fails its declared type rejects the
/// call, as for Qwen. Strings and undeclared parameters
/// stay as the dialect wrote them.
pub(super) fn deepseek_accepted_argument_count(
    tools: &[ChatTool],
    name: &str,
    arguments: &[ToolArgument],
) -> usize {
    let Some(schema) = tools
        .iter()
        .find(|tool| tool.name == name)
        .and_then(|tool| try_parse_json(&tool.parameters_json))
    else {
        return arguments.len();
    };
    let Some(properties) = resolve_tool_schema(&schema, &schema)
        .and_then(|object| object.find("properties"))
        .filter(|properties| properties.is_object())
    else {
        return arguments.len();
    };
    arguments
        .iter()
        .position(|argument| {
            !argument.is_string
                && properties.find(&argument.name).is_some_and(|property| {
                    !try_parse_json(trim(&argument.value))
                        .is_some_and(|parsed| schema_accepts(&schema, property, &parsed))
                })
        })
        .unwrap_or(arguments.len())
}

/// DSML invocations in `text`. With `spans`, invocations count only inside
/// an outer call block opened by one of `markers` (bare invocation examples
/// are prose), and each block's byte range is recorded. `partial`
/// (streaming) also returns the first unfinished invocation, with the
/// arguments its schema accepts so far, and stops there.
pub(super) fn parse_dsml_calls(
    text: &str,
    markers: &[&str],
    tools: &[ChatTool],
    calls: &mut Vec<ParsedToolCall>,
    mut spans: Option<&mut Vec<(usize, usize)>>,
    partial: bool,
) {
    const INVOKE_STARTS: [&str; 4] = [
        "<｜DSML｜invoke",
        "<DSML｜invoke",
        "<｜DS｜invoke",
        "<DS｜invoke",
    ];
    const INVOKE_ENDS: [&str; 4] = [
        "</｜DSML｜invoke>",
        "</DSML｜invoke>",
        "</｜DS｜invoke>",
        "</DS｜invoke>",
    ];
    const PARAMETER_STARTS: [&str; 4] = [
        "<｜DSML｜parameter",
        "<DSML｜parameter",
        "<｜DS｜parameter",
        "<DS｜parameter",
    ];
    const PARAMETER_ENDS: [&str; 4] = [
        "</｜DSML｜parameter>",
        "</DSML｜parameter>",
        "</｜DS｜parameter>",
        "</DS｜parameter>",
    ];
    let find_from = |needle: &str, from: usize| {
        text.get(from..)
            .and_then(|t| t.find(needle))
            .map(|p| p + from)
    };
    let mut cursor = 0;
    let mut envelope_begin: Option<usize> = None;
    let mut envelope_end = String::new();
    while cursor < text.len() {
        let mut invoke_start: Option<usize> = None;
        let mut syntax = 0;
        for (index, start) in INVOKE_STARTS.iter().enumerate() {
            if let Some(position) = find_from(start, cursor)
                && invoke_start.is_none_or(|s| position < s)
            {
                invoke_start = Some(position);
                syntax = index;
            }
        }
        if let Some(spans) = spans.as_deref_mut() {
            // Envelope boundaries count only outside the parameter and
            // invocation ranges consumed below: an outer closing tag can be
            // argument data.
            match envelope_begin {
                None => {
                    let mut opening = "";
                    let mut begin: Option<usize> = None;
                    for &marker in markers {
                        if marker == "<tool_call>" {
                            continue;
                        }
                        if let Some(position) = find_from(marker, cursor)
                            && begin.is_none_or(|begin| position < begin)
                        {
                            begin = Some(position);
                            opening = marker;
                        }
                    }
                    // Bare invocation examples outside an envelope are prose.
                    let Some(begin) = begin else {
                        break;
                    };
                    envelope_begin = Some(begin);
                    let tag = opening.find('<').map_or(opening, |at| &opening[at + 1..]);
                    envelope_end = format!("</{tag}");
                    cursor = begin + opening.len();
                    continue;
                }
                Some(begin) => {
                    if let Some(end) = find_from(&envelope_end, cursor)
                        && invoke_start.is_none_or(|start| end < start)
                    {
                        cursor = end + envelope_end.len();
                        spans.push((begin, cursor));
                        envelope_begin = None;
                        continue;
                    }
                }
            }
        }
        let Some(invoke_start) = invoke_start else {
            break;
        };
        let Some(tag_end) = find_from(">", invoke_start) else {
            break;
        };
        let mut call = ParsedToolCall {
            id: random_id("call_", 20),
            name: attribute(&text[invoke_start..=tag_end], "name").unwrap_or_default(),
            arguments: Vec::new(),
            partial_arguments: None,
        };
        let mut valid = !call.name.is_empty();
        // An unterminated `string="true"` parameter while streaming, held
        // back from a possible closing delimiter.
        let mut unfinished: Option<(String, &str)> = None;
        // Where scanning resumes after an invalid call: past its last
        // complete parameter, or the whole text inside an unfinished one.
        let mut protected_end = tag_end + 1;
        let mut parameter_cursor = tag_end + 1;
        while parameter_cursor < text.len() {
            let remainder = trim(&text[parameter_cursor..]);
            parameter_cursor = offset_in(text, remainder);
            if !remainder.starts_with(PARAMETER_STARTS[syntax]) {
                break;
            }
            let parameter_start = parameter_cursor;
            let parameter_tag_end = find_from(">", parameter_start);
            let (Some(parameter_tag_end), Some(mut parameter_end)) = (
                parameter_tag_end,
                parameter_tag_end.and_then(|tag_end| find_from(PARAMETER_ENDS[syntax], tag_end)),
            ) else {
                if partial && let Some(tag_end) = parameter_tag_end {
                    let tag = &text[parameter_start..=tag_end];
                    if let Some(name) = attribute(tag, "name").filter(|name| !name.is_empty())
                        && attribute(tag, "string").is_none_or(|value| value == "true")
                    {
                        unfinished = Some((
                            name,
                            before_partial_delimiter(&text[tag_end + 1..], PARAMETER_ENDS[syntax]),
                        ));
                    }
                }
                protected_end = text.len();
                valid = false;
                break;
            };
            let tag = &text[parameter_start..=parameter_tag_end];
            let name = attribute(tag, "name").filter(|name| !name.is_empty());
            let string_value = attribute(tag, "string");
            match (name, string_value.as_deref()) {
                (Some(name), None | Some("true" | "false")) => {
                    let is_string = string_value.as_deref() != Some("false");
                    let mut unterminated = false;
                    // JSON values may contain DSML delimiters: the value ends
                    // where it parses.
                    while !is_string
                        && try_parse_json(trim(&text[parameter_tag_end + 1..parameter_end]))
                            .is_none()
                    {
                        let Some(next) = find_from(
                            PARAMETER_ENDS[syntax],
                            parameter_end + PARAMETER_ENDS[syntax].len(),
                        ) else {
                            unterminated = true;
                            break;
                        };
                        parameter_end = next;
                    }
                    if unterminated {
                        protected_end = text.len();
                        valid = false;
                        break;
                    }
                    let raw = &text[parameter_tag_end + 1..parameter_end];
                    let argument = ToolArgument {
                        name,
                        // String values keep their whitespace, as edits need.
                        value: if is_string { raw } else { trim(raw) }.to_owned(),
                        is_string,
                    };
                    // As for Qwen calls: an identical repeat is dropped, a
                    // conflicting one rejects the call.
                    match call.arguments.iter().find(|a| a.name == argument.name) {
                        None => call.arguments.push(argument),
                        Some(previous) if *previous == argument => {}
                        Some(_) => valid = false,
                    }
                }
                _ => valid = false,
            }
            parameter_cursor = parameter_end + PARAMETER_ENDS[syntax].len();
            protected_end = parameter_cursor;
        }
        let complete = text[parameter_cursor.min(text.len())..].starts_with(INVOKE_ENDS[syntax]);
        let accepted = deepseek_accepted_argument_count(tools, &call.name, &call.arguments);
        let schema_valid = accepted == call.arguments.len();
        if valid && complete && schema_valid {
            calls.push(call);
        } else if partial && !call.name.is_empty() {
            call.arguments.truncate(accepted);
            if let Some((name, value)) = unfinished.filter(|_| schema_valid) {
                partial_string_argument(&mut call, &name, value);
            }
            if call.partial_arguments.is_none() {
                call.partial_arguments = Some(arguments_prefix(&call));
            }
            calls.push(call);
            break;
        }
        cursor = if complete {
            parameter_cursor + INVOKE_ENDS[syntax].len()
        } else {
            protected_end
        };
    }
}

pub(super) fn tool_choice_unsatisfied() -> Error {
    Error::generation(
        GenerationErrorCode::ToolChoiceUnsatisfied,
        "model did not produce a declared tool call",
    )
}
