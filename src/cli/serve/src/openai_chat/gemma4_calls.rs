//! Gemma 4 tool calls, `<|tool_call>call:NAME{ARGUMENTS}<tool_call|>`: the
//! argument parser of the Gemma 4 extension's `tool_syntax.cpp`
//! (`ParseToolCall`, `ParsePartialToolCall`) and the call scan of its
//! `text_contract.cpp` (`ScanToolCalls`), read by the host's scanned-call
//! rules (`ParseScannedCalls`).

use super::*;

/// The string delimiter of Gemma 4 arguments.
const QUOTE: &str = "<|\"|>";
/// The call block markers of the chat template.
const CALL_START: &str = "<|tool_call>";
const CALL_END: &str = "<tool_call|>";
/// Containers deeper than this reject the call.
const MAX_DEPTH: usize = 64;

/// One complete call: its name and its arguments, always a JSON object.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Gemma4Call {
    pub(super) name: String,
    pub(super) arguments: Value,
}

/// The part of an unfinished call body that later bytes cannot change
/// (`PartialToolCall`): the top-level members whose value has ended (a
/// delimiter follows it), in order, then the text so far of a string value
/// still open. A trailing prefix of the string delimiter is held back.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PartialGemma4Call {
    pub(super) name: String,
    pub(super) arguments: Value,
    /// Key and text so far of an open top-level string value.
    pub(super) open_string: Option<(String, String)>,
}

/// `std::isspace` in the C locale.
fn space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

struct Parser<'t> {
    text: &'t str,
    pos: usize,
    error: String,
}

impl<'t> Parser<'t> {
    fn new(text: &'t str) -> Self {
        Self {
            text,
            pos: 0,
            error: String::new(),
        }
    }

    fn fail(&mut self, message: &str) -> bool {
        if self.error.is_empty() {
            self.error = format!("{message} at offset {}", self.pos);
        }
        false
    }

    fn rest(&self) -> &'t str {
        &self.text[self.pos..]
    }

    fn skip_space(&mut self) {
        let bytes = self.text.as_bytes();
        while self.pos < bytes.len() && space(bytes[self.pos]) {
            self.pos += 1;
        }
    }

    fn consume(&mut self, token: &str) -> bool {
        if self.rest().starts_with(token) {
            self.pos += token.len();
            true
        } else {
            false
        }
    }

    fn quoted_string(&mut self) -> Option<String> {
        if !self.consume(QUOTE) {
            self.fail("expected string delimiter");
            return None;
        }
        let Some(end) = self.rest().find(QUOTE) else {
            self.fail("unterminated string");
            return None;
        };
        let text = self.rest()[..end].to_owned();
        self.pos += end + QUOTE.len();
        Some(text)
    }

    fn key(&mut self) -> Option<String> {
        self.skip_space();
        if self.rest().starts_with(QUOTE) {
            return self.quoted_string();
        }
        let Some(colon) = self.rest().find(':') else {
            self.fail("expected ':' after key");
            return None;
        };
        let key = self.rest()[..colon].trim_end_matches(|c: char| c.is_ascii() && space(c as u8));
        if key.is_empty() || key.contains(['{', '}', '[', ']', ',']) {
            self.fail("invalid key");
            return None;
        }
        let key = key.to_owned();
        self.pos += colon;
        Some(key)
    }

    /// A bare scalar ends at the next structural delimiter: a JSON literal or
    /// number, otherwise kept as a string.
    fn bare(&mut self) -> Value {
        let start = self.pos;
        let bytes = self.text.as_bytes();
        while self.pos < bytes.len() && !matches!(bytes[self.pos], b',' | b'}' | b']') {
            self.pos += 1;
        }
        let raw =
            self.text[start..self.pos].trim_end_matches(|c: char| c.is_ascii() && space(c as u8));
        match raw {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" => Value::Null,
            _ => number(raw).map_or_else(|| Value::from(raw), Value::from),
        }
    }

    fn value(&mut self, depth: usize) -> Option<Value> {
        if depth > MAX_DEPTH {
            self.fail("arguments nest too deeply");
            return None;
        }
        self.skip_space();
        if self.pos >= self.text.len() {
            self.fail("expected a value");
            return None;
        }
        if self.rest().starts_with(QUOTE) {
            return self.quoted_string().map(Value::from);
        }
        match self.text.as_bytes()[self.pos] {
            b'{' => self.object(depth + 1),
            b'[' => {
                self.pos += 1;
                let mut array = Value::array();
                self.skip_space();
                if self.consume("]") {
                    return Some(array);
                }
                loop {
                    array.push(self.value(depth + 1)?);
                    self.skip_space();
                    if self.consume("]") {
                        return Some(array);
                    }
                    if !self.consume(",") {
                        self.fail("expected ',' or ']'");
                        return None;
                    }
                }
            }
            _ => Some(self.bare()),
        }
    }

    fn object(&mut self, depth: usize) -> Option<Value> {
        if !self.consume("{") {
            self.fail("expected '{'");
            return None;
        }
        let mut object = Value::object();
        self.skip_space();
        if self.consume("}") {
            return Some(object);
        }
        loop {
            let key = self.key()?;
            self.skip_space();
            if !self.consume(":") {
                self.fail("expected ':'");
                return None;
            }
            let value = self.value(depth)?;
            object.set(&key, value);
            self.skip_space();
            if self.consume("}") {
                return Some(object);
            }
            if !self.consume(",") {
                self.fail("expected ',' or '}'");
                return None;
            }
        }
    }
}

/// `raw` as a finite number where `std::from_chars` reads all of it: no
/// leading '+', and no result outside the normal double range.
fn number(raw: &str) -> Option<f64> {
    if raw.is_empty() || raw.starts_with('+') {
        return None;
    }
    let value: f64 = raw.parse().ok()?;
    if !value.is_finite() {
        return None;
    }
    // An underflow is out of range for from_chars, as for strtod.
    let mantissa = raw.split(['e', 'E']).next().unwrap_or(raw);
    let nonzero = mantissa.bytes().any(|b| matches!(b, b'1'..=b'9'));
    if (value == 0.0 && nonzero) || (value != 0.0 && value.abs() < f64::MIN_POSITIVE) {
        return None;
    }
    Some(value)
}

/// Parses the body of one `<|tool_call>…<tool_call|>` block,
/// `call:NAME{key:value,…}` (`ParseToolCall`). Values use the Gemma 4
/// argument syntax the chat template renders: `<|"|>`-delimited strings,
/// numbers, true/false/null, `{…}` objects with bare or `<|"|>`-quoted keys,
/// and `[…]` arrays. A bare value that is not a JSON literal is kept as a
/// string.
pub(super) fn parse_gemma4_call(body: &str) -> std::result::Result<Gemma4Call, String> {
    let mut parser = Parser::new(body);
    parser.skip_space();
    if !parser.consume("call:") {
        parser.fail("expected 'call:'");
        return Err(parser.error);
    }
    let Some(brace) = parser.rest().find('{').filter(|&brace| brace != 0) else {
        parser.fail("expected a function name followed by '{'");
        return Err(parser.error);
    };
    let name = parser.rest()[..brace].to_owned();
    parser.pos += brace;
    let Some(arguments) = parser.object(0) else {
        return Err(parser.error);
    };
    parser.skip_space();
    if parser.pos != body.len() {
        parser.fail("unexpected text after arguments");
        return Err(parser.error);
    }
    Ok(Gemma4Call { name, arguments })
}

/// Parses an unfinished call body, such as one still being generated
/// (`ParsePartialToolCall`). `None` until `call:NAME{` has arrived. A
/// malformed or repeated member ends the result before it: the complete
/// parse decides that call.
pub(super) fn parse_partial_gemma4_call(body: &str) -> Option<PartialGemma4Call> {
    let mut parser = Parser::new(body);
    parser.skip_space();
    if !parser.consume("call:") {
        return None;
    }
    let brace = parser.rest().find('{').filter(|&brace| brace != 0)?;
    let mut call = PartialGemma4Call {
        name: parser.rest()[..brace].to_owned(),
        arguments: Value::object(),
        open_string: None,
    };
    parser.pos += brace + 1;
    loop {
        parser.skip_space();
        if parser.pos >= body.len() || parser.rest().starts_with('}') {
            return Some(call);
        }
        let Some(key) = parser.key() else {
            return Some(call);
        };
        parser.skip_space();
        if !parser.consume(":") || call.arguments.find(&key).is_some() {
            return Some(call);
        }
        parser.skip_space();
        let rest = parser.rest();
        if rest.starts_with(QUOTE) && !rest[QUOTE.len()..].contains(QUOTE) {
            let text = before_partial_delimiter(&rest[QUOTE.len()..], QUOTE);
            call.open_string = Some((key, text.to_owned()));
            return Some(call);
        }
        let Some(value) = parser.value(0) else {
            return Some(call);
        };
        // A bare scalar may still grow; any value ends at its delimiter.
        parser.skip_space();
        if !parser.rest().starts_with([',', '}']) {
            return Some(call);
        }
        call.arguments.set(&key, value);
        if !parser.consume(",") {
            return Some(call);
        }
    }
}

/// A scanned argument as the API reports it: strings verbatim, other values
/// as JSON.
fn arguments(object: &Value) -> Vec<ToolArgument> {
    object
        .members()
        .iter()
        .map(|(name, value)| ToolArgument {
            name: name.clone(),
            value: if value.is_string() {
                value.str().to_owned()
            } else {
                value.dump().unwrap_or_default()
            },
            is_string: value.is_string(),
        })
        .collect()
}

/// Gemma 4 calls in `text`, which starts at the first call marker
/// (`ScanToolCalls`, then `ParseScannedCalls`). Each complete block's byte
/// range goes to `spans`, also when its body does not parse; its call counts
/// when the declared schema accepts its arguments. `partial` (streaming)
/// also returns an unfinished last call with its settled arguments and an
/// open string argument, and stops there.
pub(super) fn parse_gemma4_calls(
    text: &str,
    tools: &[ChatTool],
    calls: &mut Vec<ParsedToolCall>,
    mut spans: Option<&mut Vec<(usize, usize)>>,
    partial: bool,
) {
    let mut cursor = 0;
    while let Some(open) = text[cursor..].find(CALL_START).map(|at| at + cursor) {
        let body = open + CALL_START.len();
        let Some(close) = text[body..].find(CALL_END).map(|at| at + body) else {
            // Without a partial parse an interrupted call is never reported.
            if partial && let Some(call) = parse_partial_gemma4_call(&text[body..]) {
                let arguments = arguments(&call.arguments);
                let accepted = deepseek_accepted_argument_count(tools, &call.name, &arguments);
                let schema_valid = accepted == arguments.len();
                let mut parsed = ParsedToolCall {
                    id: random_id("call_", 20),
                    name: call.name,
                    arguments,
                    partial_arguments: None,
                };
                parsed.arguments.truncate(accepted);
                if let Some((name, value)) = call.open_string.filter(|_| schema_valid) {
                    partial_string_argument(&mut parsed, &name, &value);
                }
                if parsed.partial_arguments.is_none() {
                    parsed.partial_arguments = Some(arguments_prefix(&parsed));
                }
                if !parsed.name.is_empty() {
                    calls.push(parsed);
                }
            }
            return;
        };
        let end = close + CALL_END.len();
        if let Some(spans) = spans.as_deref_mut() {
            spans.push((open, end));
        }
        if let Ok(call) = parse_gemma4_call(&text[body..close]) {
            let arguments = arguments(&call.arguments);
            let accepted = deepseek_accepted_argument_count(tools, &call.name, &arguments);
            let mut parsed = ParsedToolCall {
                id: random_id("call_", 20),
                name: call.name,
                arguments,
                partial_arguments: None,
            };
            if accepted == parsed.arguments.len() {
                calls.push(parsed);
            } else if partial && !parsed.name.is_empty() {
                // A complete call its schema rejects ends a streamed prefix.
                parsed.arguments.truncate(accepted);
                parsed.partial_arguments = Some(arguments_prefix(&parsed));
                calls.push(parsed);
                return;
            }
        }
        cursor = end;
    }
}

/// The Gemma 4 extension's `tests/tool_syntax_test.cpp` (values, rejections
/// and partial calls) and the scan cases of `tests/text_contract_test.cpp`.
#[cfg(test)]
mod tests {
    use super::*;

    /// The arguments of `body` as JSON, with the call's name.
    fn parsed(body: &str) -> (String, String) {
        let call = parse_gemma4_call(body).unwrap_or_else(|error| panic!("{body}: {error}"));
        (call.name, call.arguments.dump().unwrap_or_default())
    }

    fn arguments_of(body: &str) -> String {
        parsed(body).1
    }

    #[test]
    fn values() {
        assert_eq!(
            parsed("call:get_weather{location:<|\"|>Paris, FR<|\"|>}"),
            ("get_weather".into(), r#"{"location":"Paris, FR"}"#.into())
        );
        assert_eq!(
            parsed("call:mcp/github.create_issue:v2{n:1}"),
            ("mcp/github.create_issue:v2".into(), r#"{"n":1}"#.into()),
            "namespaced function name"
        );
        assert_eq!(arguments_of("call:f{}"), "{}");
        assert_eq!(
            arguments_of("call:f{n:42,x:-1.5,e:1e-05,t:true,f:false,z:null}"),
            r#"{"n":42,"x":-1.5,"e":1e-05,"t":true,"f":false,"z":null}"#
        );
        assert_eq!(
            arguments_of("call:f{a:[1,<|\"|>two<|\"|>,[]],o:{k:{deep:1}}}"),
            r#"{"a":[1,"two",[]],"o":{"k":{"deep":1}}}"#
        );
        assert_eq!(
            arguments_of("call:f{<|\"|>quoted key<|\"|>:1}"),
            r#"{"quoted key":1}"#
        );
        // Strings are raw between delimiters: quotes, braces, commas,
        // newlines.
        assert_eq!(
            arguments_of("call:f{s:<|\"|>a \"b\" {c}, d\n<|\"|>}"),
            r#"{"s":"a \"b\" {c}, d\n"}"#
        );
        assert_eq!(arguments_of("call:f{mode:fast}"), r#"{"mode":"fast"}"#);
        assert_eq!(
            arguments_of("call:f{ a : 1 , b : <|\"|>x<|\"|> }"),
            r#"{"a":1,"b":"x"}"#
        );
        // What std::from_chars does not read whole stays a string.
        assert_eq!(
            arguments_of("call:f{a:+1,b:inf,c:1e999,d:0x10,e:.5}"),
            r#"{"a":"+1","b":"inf","c":"1e999","d":"0x10","e":0.5}"#
        );
    }

    #[test]
    fn rejections() {
        for body in [
            "",
            "call:",
            "call:{}",
            "f{a:1}",
            "call:f{a:1",
            "call:f{a:<|\"|>x}",
            "call:f{a:1}}",
            "call:f{a}",
            "call:f{a:[1,2}",
            "call:f{:1}",
        ] {
            let error = parse_gemma4_call(body).expect_err(body);
            assert!(!error.is_empty(), "{body}");
        }
        let deep = format!("call:f{{a:{}", "[".repeat(100));
        assert!(parse_gemma4_call(&deep).is_err(), "unbounded nesting");
    }

    /// Settled arguments of `body`, then the open string (if any) after `|`.
    fn partial(body: &str) -> String {
        let Some(call) = parse_partial_gemma4_call(body) else {
            return "none".into();
        };
        let mut out = format!("{}{}", call.name, call.arguments.dump().unwrap_or_default());
        if let Some((key, text)) = call.open_string {
            out += &format!("|{key}={text}");
        }
        out
    }

    #[test]
    fn partial_calls() {
        assert_eq!(partial(""), "none");
        assert_eq!(partial("call:"), "none");
        assert_eq!(partial("call:get_we"), "none");
        assert_eq!(partial("call:f{"), "f{}", "name settles at its brace");
        assert_eq!(partial("call:f{a:<|\"|>Par"), "f{}|a=Par", "open string");
        // A delimiter prefix may close the string; hold it.
        assert_eq!(partial("call:f{a:<|\"|>x<|"), "f{}|a=x");
        assert_eq!(partial("call:f{a:<|\"|>x<|\""), "f{}|a=x");
        assert_eq!(partial("call:f{a:<|\"|>x<|\"|>"), "f{}");
        assert_eq!(partial("call:f{a:<|\"|>x<|\"|>,b:1"), r#"f{"a":"x"}"#);
        assert_eq!(
            partial("call:f{n:12"),
            "f{}",
            "a bare scalar may still grow"
        );
        assert_eq!(partial("call:f{n:12,"), r#"f{"n":12}"#);
        assert_eq!(partial("call:f{o:{k:[1,2],s:<|\"|>}"), "f{}");
        assert_eq!(partial("call:f{o:{k:[1,2]}}"), r#"f{"o":{"k":[1,2]}}"#);
        assert_eq!(
            partial("call:f{a:1,a:2}"),
            r#"f{"a":1}"#,
            "a repeated key ends the settled members"
        );
        assert_eq!(
            partial("call:f{<|\"|>quoted key<|\"|>:<|\"|>v"),
            "f{}|quoted key=v"
        );
        // Every byte prefix of a full call settles a prefix of its
        // arguments.
        let body = "call:edit{count:3,edits:[{newText:<|\"|><|\"|>,oldText:<|\"|>a, \
                    {b}\n<|\"|>}],mode:safe,path:<|\"|>calc.py<|\"|>}";
        let full = arguments_of(body);
        for size in 0..=body.len() {
            let Some(call) = parse_partial_gemma4_call(&body[..size]) else {
                continue;
            };
            let mut settled = call.arguments.dump().unwrap_or_default();
            settled.pop();
            assert!(full.starts_with(&settled), "prefix at {size}");
        }
        assert_eq!(
            partial(body),
            format!("edit{full}"),
            "the full body settles"
        );
    }

    fn scan(text: &str, partial: bool) -> (Vec<ParsedToolCall>, Vec<(usize, usize)>) {
        let (mut calls, mut spans) = (Vec::new(), Vec::new());
        parse_gemma4_calls(text, &[], &mut calls, Some(&mut spans), partial);
        (calls, spans)
    }

    fn argument<'c>(call: &'c ParsedToolCall, name: &str) -> &'c ToolArgument {
        call.arguments
            .iter()
            .find(|argument| argument.name == name)
            .unwrap_or_else(|| panic!("argument {name}"))
    }

    #[test]
    fn buffered_scan() {
        let first = "<|tool_call>call:get_weather{days:3,location:<|\"|>Paris, \
                     FR<|\"|>,units:{c:true}}<tool_call|>";
        let second = "<|tool_call>call:noop{}<tool_call|>";
        let text = format!("{first}{second} trailing");
        let (calls, spans) = scan(&text, false);
        assert_eq!(calls.len(), 2);
        let call = &calls[0];
        assert_eq!(call.name, "get_weather");
        assert!(call.partial_arguments.is_none(), "complete call");
        let location = argument(call, "location");
        assert!(location.is_string && location.value == "Paris, FR");
        let days = argument(call, "days");
        assert!(!days.is_string && days.value == "3");
        let units = argument(call, "units");
        assert!(!units.is_string && units.value == r#"{"c":true}"#);
        assert!(calls[1].name == "noop" && calls[1].arguments.is_empty());
        assert_eq!(
            spans,
            [(0, first.len()), (first.len(), first.len() + second.len())],
            "spans cover each call block"
        );
        // A malformed block keeps its span and reports no call; later calls
        // stay.
        let broken = "<|tool_call>call:f{x:<tool_call|>";
        let (calls, spans) = scan(&format!("{broken}{second}"), false);
        assert!(calls.len() == 1 && calls[0].name == "noop");
        assert!(spans.len() == 2 && spans[0].1 == broken.len());
        // Without a partial parse an unfinished call is never reported.
        let (calls, spans) = scan(&format!("{second}<|tool_call>call:f{{x:1"), false);
        assert!(calls.len() == 1 && spans.len() == 1);
    }

    #[test]
    fn streamed_scan() {
        let text = "<|tool_call>call:write{path:<|\"|>a.txt<|\"|>,mode:1,\
                    body:<|\"|>hello, world<|\"|>}<tool_call|>";
        let (complete, _) = scan(text, false);
        assert_eq!(complete.len(), 1);
        let full = &complete[0].arguments;
        let (mut settled, mut named) = (0, false);
        let mut open_text = String::new();
        for size in 1..text.len() {
            let (calls, spans) = scan(&text[..size], true);
            assert!(calls.len() <= 1, "one call at a time");
            let Some(call) = calls.first() else {
                assert!(!named, "a named call stays reported");
                continue;
            };
            named = true;
            assert_eq!(call.name, "write", "partial call name");
            assert!(
                call.arguments.len() >= settled,
                "settled arguments only grow"
            );
            for (argument, last) in call.arguments.iter().zip(full) {
                assert_eq!(argument, last, "a settled argument never changes");
            }
            if call.arguments.len() != settled {
                open_text.clear();
            }
            settled = call.arguments.len();
            if call.partial_arguments.is_some() {
                assert!(spans.is_empty(), "an unfinished call has no span");
                let body = &text[CALL_START.len()..size];
                if let Some((_, open)) =
                    parse_partial_gemma4_call(body).and_then(|call| call.open_string)
                {
                    assert!(open.starts_with(&open_text), "an open string only grows");
                    assert!(
                        "hello, world".starts_with(&open) || "a.txt".starts_with(&open),
                        "an open string never shows its delimiter: {open}"
                    );
                    open_text = open;
                }
            }
        }
        assert!(
            named && settled == 3,
            "every argument settled while streaming"
        );
        assert_eq!(
            arguments_json(full),
            r#"{"path":"a.txt","mode":1,"body":"hello, world"}"#
        );
    }

    #[test]
    fn schemas_check_the_arguments() {
        let tool = ChatTool {
            name: "f".into(),
            parameters_json: r#"{"type":"object","properties":{"n":{"type":"integer"}}}"#.into(),
            ..ChatTool::default()
        };
        let tools = [tool];
        let mut calls = Vec::new();
        parse_gemma4_calls(
            "<|tool_call>call:f{n:<|\"|>x<|\"|>}<tool_call|><|tool_call>call:f{n:true}<tool_call|>\
             <|tool_call>call:f{n:2}<tool_call|>",
            &tools,
            &mut calls,
            None,
            false,
        );
        // A string argument stays as written; a typed one must match.
        assert_eq!(calls.len(), 2);
        assert_eq!(arguments_json(&calls[0].arguments), r#"{"n":"x"}"#);
        assert_eq!(arguments_json(&calls[1].arguments), r#"{"n":2}"#);
    }
}
