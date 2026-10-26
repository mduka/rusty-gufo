//! Tool-call markers in generated text: the reasoning separator, Qwen and
//! DeepSeek markers, the XML envelope and the framing held back while a
//! marker may still start.

use super::*;

/// Whether `c` is ASCII whitespace (`std::isspace` in the C locale).
pub(super) fn is_space(c: char) -> bool {
    c.is_ascii_whitespace() || c == '\u{b}'
}

/// Trims ASCII whitespace (`std::isspace` in the C locale).
pub(super) fn trim(value: &str) -> &str {
    value.trim_matches(is_space)
}

pub(super) fn trim_trailing(value: &str) -> &str {
    value.trim_end_matches(is_space)
}

/// Qwen writes a parameter as `<parameter=name>\nVALUE\n</parameter>`: one
/// newline on each side is framing, everything else (a file's final newline,
/// indentation, blank lines) belongs to the value.
pub(super) fn strip_framing_newlines(value: &str) -> &str {
    // The closing frame matches the opening one: a literal final CR before
    // an LF frame is data.
    let frame = if value.starts_with("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    match value.strip_prefix(frame) {
        Some(inner) => inner.strip_suffix(frame).unwrap_or(inner),
        None => value,
    }
}

/// Qwen's calls look like Python keyword arguments, and the models sometimes
/// write `True`, `False` and `None` where the schema asks for a JSON boolean
/// or null: rewrites those words outside JSON string literals.
pub(super) fn python_literals_to_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.char_indices().peekable();
    let mut in_string = false;
    while let Some((start, c)) = chars.next() {
        if in_string {
            out.push(c);
            if c == '\\' {
                if let Some((_, escaped)) = chars.next() {
                    out.push(escaped);
                }
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if !c.is_ascii_alphabetic() {
            out.push(c);
            in_string = c == '"';
            continue;
        }
        let mut end = start + c.len_utf8();
        while let Some(&(at, next)) = chars.peek() {
            if !(next.is_ascii_alphanumeric() || next == '_') {
                break;
            }
            end = at + next.len_utf8();
            chars.next();
        }
        out.push_str(match &value[start..end] {
            "True" => "true",
            "False" => "false",
            "None" => "null",
            word => word,
        });
    }
    out
}

pub(super) fn try_parse_json(value: &str) -> Option<Value> {
    json::parse_str(value).ok()
}

pub(super) const QWEN_TOOL_MARKERS: [&str; 1] = ["<tool_call>"];

/// DeepSeek's canonical call block. The template writes "\n\n" before it:
/// as in llama.cpp (`common/parsers/deepseek.cpp`, `TC_SEPARATOR` +
/// `FC_START`) that separator is framing, and returning it as content would
/// double it when the history is replayed.
pub(super) const DEEPSEEK_TOOL_MARKERS: [&str; 2] =
    ["\n\n<｜DSML｜tool_calls>", "<｜DSML｜tool_calls>"];

/// Closers removed directly after an accepted Qwen or JSON call. Standalone
/// closers and other XML stay literal; client envelopes follow their own
/// rule.
pub(super) const QWEN_TOOL_CLOSERS: [&str; 4] = [
    "</parameter>",
    "</function>",
    "</tool_call>",
    // The client's envelope, whose block rules already strip it whole.
    "</invoke>",
];

/// A native function header, rather than a bare marker mentioned in prose,
/// permits the implicit reasoning boundary of llama.cpp's native parsers
/// (`ReasoningToolMarkers`).
pub(super) fn reasoning_tool_markers(markers: &[&str]) -> &'static [&'static str] {
    const QWEN: [&str; 1] = ["<tool_call>\n<function="];
    const DEEPSEEK: [&str; 2] = [
        "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"",
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"",
    ];
    if markers == QWEN_TOOL_MARKERS {
        &QWEN
    } else if markers == DEEPSEEK_TOOL_MARKERS {
        &DEEPSEEK
    } else {
        &[]
    }
}

/// Every known closer, for backends that do not report the admitted format.
pub(super) const TOOL_CLOSERS: [&str; 9] = [
    // Qwen family.
    "</parameter>",
    "</function>",
    "</tool_call>",
    // Envelope dialects clients and harness prompts render; Qwen models echo
    // them back verbatim, so they are framing too.
    "</invoke>",
    "</function_results>",
    "</tool_calls>",
    // Model-native dialects, admitted like any other opener.
    "</｜DSML｜parameter>",
    "</｜DSML｜invoke>",
    "</｜DSML｜tool_calls>",
];

/// The markup that delimits tool calls in a request's output.
#[derive(Clone, Copy)]
pub(super) struct ToolSyntax {
    /// Openers of a call block.
    pub(super) markers: &'static [&'static str],
    /// Framing removed when the model echoes it after a call.
    pub(super) closers: &'static [&'static str],
    /// Qwen's answer follows `</think>` after any whitespace, with or
    /// without tools.
    pub(super) qwen: bool,
}

/// Output without tools: markup is ordinary text.
pub(super) const NO_TOOL_SYNTAX: ToolSyntax = ToolSyntax {
    markers: &[],
    closers: &[],
    qwen: false,
};

/// The tool syntax of a request admitted with `format` (`ToolMarkers`,
/// `ToolClosers`): the canonical opener of a native grammar, under which
/// other dialects are prose; every dialect for the JSON fallback, whose free
/// text may carry native calls, and without format metadata. DeepSeek
/// content is everything outside its call block, which ends the output, so
/// nothing is stripped there, as in llama.cpp. Without tools (or with
/// `tool_choice: none`) markup is ordinary text; the model's dialect still
/// decides the reasoning separator.
pub(super) fn tool_syntax(chat: &ChatRequest, format: Option<ToolFormat>) -> ToolSyntax {
    let qwen = format == Some(ToolFormat::Qwen);
    if chat.tools.is_empty() || chat.tool_choice == ToolChoice::None {
        return ToolSyntax {
            qwen,
            ..NO_TOOL_SYNTAX
        };
    }
    match format {
        None => ToolSyntax {
            markers: &TOOL_MARKERS,
            closers: &TOOL_CLOSERS,
            qwen,
        },
        Some(ToolFormat::Json) => ToolSyntax {
            markers: &TOOL_MARKERS,
            closers: &QWEN_TOOL_CLOSERS,
            qwen,
        },
        Some(ToolFormat::Qwen) => ToolSyntax {
            markers: &QWEN_TOOL_MARKERS,
            closers: &QWEN_TOOL_CLOSERS,
            qwen,
        },
        Some(ToolFormat::DeepSeek) => ToolSyntax {
            markers: &DEEPSEEK_TOOL_MARKERS,
            closers: &[],
            qwen,
        },
    }
}

/// The answer after an explicit `</think>` (`AfterReasoningSeparator`):
/// Qwen's llama.cpp PEG reads `reasoning << content`, where `<<` consumes
/// `space()`; other dialects drop line breaks. Answer and argument interiors
/// are untouched.
pub(super) fn after_reasoning_separator(value: &str, qwen: bool) -> &str {
    if qwen {
        value.trim_start_matches([' ', '\t', '\r', '\n', '\x0c', '\x0b'])
    } else {
        value.trim_start_matches(['\r', '\n'])
    }
}

/// The earliest of `markers` at or after `from` in `full` that is not quoted
/// prose, with the marker. A marker inside a backtick span or a code fence
/// is the model explaining the format; an answer naming its tags must not be
/// parsed or cut at them. `quotes` tracks `full`.
pub(super) fn earliest_marker(
    full: &str,
    markers: &[&'static str],
    quotes: &QuoteTracker,
    from: usize,
) -> Option<(usize, &'static str)> {
    let bytes = full.as_bytes();
    let mut position = from;
    // Only a position that carries a marker is worth asking about quoting,
    // which keeps a line full of '<' linear.
    while let Some(found) = full.get(position..).and_then(|rest| rest.find('<')) {
        let next = position + found;
        for &marker in markers {
            let Some(tag_offset) = marker.find('<') else {
                continue;
            };
            if next - from >= tag_offset
                && bytes[next - tag_offset..].starts_with(marker.as_bytes())
                && !quotes.quoted_at(next)
            {
                return Some((next - tag_offset, marker));
            }
        }
        position = next + 1;
    }
    None
}

/// The earliest native call header in reasoning outside any quotation,
/// unfinished ones included (`ReasoningToolMarker`).
pub(super) fn reasoning_tool_marker(
    full: &str,
    markers: &[&'static str],
    quotes: &QuoteTracker,
    from: usize,
) -> Option<usize> {
    let mut found = earliest_marker(full, markers, quotes, from).map(|(found, _)| found);
    while let Some(position) = found.filter(|&position| quotes.unclosed_at(position)) {
        found = earliest_marker(full, markers, quotes, position + 1).map(|(found, _)| found);
    }
    found
}

/// [`earliest_marker`] in a text of its own.
pub(super) fn earliest_marker_in(
    text: &str,
    markers: &[&'static str],
) -> Option<(usize, &'static str)> {
    let mut quotes = QuoteTracker::default();
    quotes.reset(text, 0);
    earliest_marker(text, markers, &quotes, 0)
}

/// Whether `text` begins one of `markers`.
pub(super) fn tool_marker_prefix(text: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| marker.starts_with(text))
}

/// The length of the longest suffix of `full[..end]` that begins one of
/// `markers` outside quoted prose: those bytes may still open a call.
pub(super) fn held_marker_prefix(
    full: &str,
    end: usize,
    markers: &[&str],
    quotes: &QuoteTracker,
) -> usize {
    let maximum_marker = markers.iter().map(|m| m.len()).max().unwrap_or(0);
    let maximum = end.min(maximum_marker.saturating_sub(1));
    for length in (1..=maximum).rev() {
        let suffix = &full.as_bytes()[end - length..end];
        if markers.iter().any(|m| m.as_bytes().starts_with(suffix))
            && !quotes.quoted_at(end - length)
        {
            return length;
        }
    }
    0
}

/// A client envelope (`<invoke name=…>…</invoke>`) is removed only when it
/// names a declared tool or directly echoes an accepted call; other XML
/// stays content.
pub(super) const ENVELOPE_HEAD: &str = "<invoke name=";

pub(super) const ENVELOPE_PARAMETER: &str = "<parameter name=";

pub(super) const ENVELOPE_CLOSERS: [&str; 2] = ["</invoke>", "</parameter>"];

/// The start of the last `tag` in `full[floor..stop]`, unless it sits in
/// quoted prose. Only what the caller can use is searched: a tag before
/// `floor` is discarded there, and the tag holds no `<` past its first byte,
/// so none straddles a marker or a closer.
pub(super) fn last_unquoted_tag(
    full: &str,
    stop: usize,
    floor: usize,
    tag: &str,
    quotes: &QuoteTracker,
) -> Option<usize> {
    if stop <= floor {
        return None;
    }
    let found = floor + full[floor..stop].rfind(tag)?;
    (!quotes.quoted_at(found)).then_some(found)
}

/// Whether an envelope head names a declared tool.
pub(super) fn declared_envelope(tag: &str, tools: &[ChatTool]) -> bool {
    let value = &tag[ENVELOPE_HEAD.len()..];
    let Some(quote) = value.chars().next().filter(|c| matches!(c, '"' | '\'')) else {
        return false;
    };
    value[1..]
        .find(quote)
        .is_some_and(|end| tools.iter().any(|tool| tool.name == value[1..1 + end]))
}

/// The start of a closed or truncated client envelope ending `full[..stop]`
/// whose request context identifies a tool attempt: a declared tool's name,
/// or `after_call` (directly after an accepted call). A bare parameter block
/// is not enough evidence.
pub(super) fn envelope_block_start(
    full: &str,
    stop: usize,
    floor: usize,
    closers: &[&str],
    tools: &[ChatTool],
    after_call: bool,
    quotes: &QuoteTracker,
) -> Option<usize> {
    let opening = last_unquoted_tag(full, stop, floor, ENVELOPE_HEAD, quotes)?;
    if !after_call {
        let tag_end = full[opening..]
            .find('>')
            .map(|end| opening + end)
            .filter(|&end| end < stop)?;
        if !declared_envelope(&full[opening..tag_end], tools) {
            return None;
        }
    }
    let region = trim_trailing(&full[opening..stop]);
    if ENVELOPE_CLOSERS
        .iter()
        .any(|closer| region.ends_with(closer))
    {
        return Some(opening);
    }
    if !region.contains(ENVELOPE_PARAMETER) || closers.iter().any(|closer| region.contains(closer))
    {
        return None;
    }
    Some(opening)
}

/// `full[begin..end]` as content: framing goes only with request context,
/// a closer echoed directly after an accepted call (`after_call`) or a
/// client envelope naming a declared tool; other XML is data
/// (`ContentBefore`). A syntax without closers removes nothing.
pub(super) fn content_before<'t>(
    full: &'t str,
    begin: usize,
    end: usize,
    closers: &[&str],
    quotes: &QuoteTracker,
    tools: &[ChatTool],
    after_call: bool,
) -> &'t str {
    let stop = end.min(full.len());
    if stop <= begin {
        return "";
    }
    if closers.is_empty() {
        return &full[begin..stop];
    }
    let mut begin = begin;
    if after_call {
        let mut cursor = begin;
        let mut removed = false;
        while cursor < stop {
            cursor +=
                full[cursor..stop].len() - full[cursor..stop].trim_start_matches(is_space).len();
            let Some(closer) = closers
                .iter()
                .find(|closer| full[cursor..stop].starts_with(**closer))
            else {
                break;
            };
            if quotes.quoted_at(cursor) {
                break;
            }
            cursor += closer.len();
            removed = true;
        }
        if removed {
            begin = cursor;
        }
    }
    let slice = &full[begin..stop];
    // The newline a model writes between or after its calls is layout, not
    // a reply: a turn of parallel calls would otherwise carry content "\n"
    //.
    if after_call && trim(slice).is_empty() {
        return "";
    }
    let adjacent_envelope = after_call && trim(slice).starts_with(ENVELOPE_HEAD);
    match envelope_block_start(full, stop, begin, closers, tools, adjacent_envelope, quotes) {
        Some(block) => trim_trailing(&full[begin..block]),
        None => slice,
    }
}

/// The bytes at the end of `full[floor..end]` to hold back as a possible
/// client envelope (`FramingHold`). A declared tool's head stays pending
/// until the final parse decides whether it is quoted documentation or
/// framing.
pub(super) fn framing_hold(
    full: &str,
    end: usize,
    floor: usize,
    closers: &[&str],
    quotes: &QuoteTracker,
    tools: &[ChatTool],
) -> usize {
    if closers.is_empty() {
        return 0;
    }
    let mut hold = 0;
    // Only a suffix that can still become admitted framing is held: a bare
    // comparison such as "3 < 5" is prose once the space arrives.
    if let Some(open) = full[..end].rfind('<')
        && open >= floor
        && !quotes.quoted_at(open)
        && ENVELOPE_HEAD.starts_with(&full[open..end])
    {
        hold = end - open;
    }
    if let Some(opening) = last_unquoted_tag(full, end, floor, ENVELOPE_HEAD, quotes) {
        let tag_end = full[opening..].find('>').map(|tag_end| opening + tag_end);
        if tag_end.is_none_or(|tag_end| {
            tag_end >= end || declared_envelope(&full[opening..tag_end], tools)
        }) {
            hold = hold.max(end - opening);
        }
    }
    // The separator before framing goes with it: trailing whitespace stays
    // undecided until the next piece shows whether it introduces prose or a
    // held tag, rather than streaming a separator that cannot be retracted.
    let ready = &full[floor..end - hold];
    hold + ready.len() - trim_trailing(ready).len()
}
