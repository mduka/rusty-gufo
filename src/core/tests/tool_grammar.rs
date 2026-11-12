//! Tool-call grammars: native Qwen and DeepSeek parameter tags, open and
//! best-effort non-strict agent schemas, and the JSON fallback, as the C++
//! `tests/core/json_constraint_test.cpp` checks them (native models keep
//! their native syntax for every schema):
//! cargo test -p gufo-core --test tool_grammar

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use gufo_core::json::{self, Value};
use gufo_core::json_constraint::{
    ConstraintVocabulary, JsonConstraint, Piece, SchemaError, TokenConstraint, ToolFormat,
};

fn parse(text: &str) -> Value {
    json::parse_str(text).unwrap()
}

fn accepts(grammar: &JsonConstraint, text: impl AsRef<[u8]>) -> bool {
    let mut state = grammar.start().unwrap();
    for &byte in text.as_ref() {
        state = grammar.advance(&state, byte).unwrap();
        if state.is_empty() {
            return false;
        }
    }
    grammar.complete(&state)
}

fn tool(schema: &Value, strict: bool, format: ToolFormat) -> Option<Arc<JsonConstraint>> {
    JsonConstraint::tool_parameters(schema, strict, format).unwrap()
}

fn with_tools(
    answer: Option<Arc<JsonConstraint>>,
    name: &str,
    parameters: Arc<JsonConstraint>,
    required: bool,
    parallel: bool,
    format: ToolFormat,
) -> Arc<JsonConstraint> {
    JsonConstraint::with_tools(
        answer,
        vec![(name.into(), parameters)],
        required,
        parallel,
        format,
    )
    .unwrap()
    .unwrap()
}

const NATIVE: [ToolFormat; 2] = [ToolFormat::Qwen, ToolFormat::DeepSeek];
const ALL: [ToolFormat; 3] = [ToolFormat::Qwen, ToolFormat::DeepSeek, ToolFormat::Json];

#[test]
fn native_tools() {
    let schema = parse(
        r#"{"type":"object","properties":{
            "text":{"type":"string","minLength":1,"maxLength":256},
            "n":{"type":"integer","minimum":1,"maximum":5}},
          "required":["text","n"],"additionalProperties":false}"#,
    );
    for format in NATIVE {
        let parameters = tool(&schema, true, format).unwrap();
        let qwen = format == ToolFormat::Qwen;
        let wrap = |text: &str, n: i32, reverse: bool| {
            let argument = |name: &str, value: &str, string: bool| {
                if qwen {
                    format!("<parameter={name}>\n{value}\n</parameter>\n")
                } else {
                    format!(
                        "<｜DSML｜parameter name=\"{name}\" string=\"{}\">{value}</｜DSML｜parameter>\n",
                        if string { "true" } else { "false" }
                    )
                }
            };
            let a = argument("text", text, true);
            let b = argument("n", &n.to_string(), false);
            format!(
                "{}{}{}",
                if qwen {
                    "<tool_call>\n<function=f>\n"
                } else {
                    "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n"
                },
                if reverse { b + &a } else { a + &b },
                if qwen {
                    "</function>\n</tool_call>"
                } else {
                    "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
                }
            )
        };
        let grammar = with_tools(None, "f", parameters.clone(), true, false, format);
        let call = wrap(" <think>literal</think> \"42\" \\\nπ🦉\n ", 3, false);
        assert!(accepts(&grammar, &call));
        assert!(accepts(&grammar, wrap("</tool_call><tool_call>", 3, true)));
        assert!(!accepts(&grammar, wrap("", 3, false)));
        assert!(!accepts(&grammar, wrap("value", 6, false)));
        assert!(!accepts(&grammar, "ordinary text"));
        assert!(!accepts(&grammar, format!("{call}{call}")));
        let parallel = with_tools(None, "f", parameters, true, true, format);
        if qwen {
            // A raw value must not absorb a delimiter prefix and let the
            // actual closer complete it across the boundary: two closers
            // are rejected by the native parser.
            assert!(!accepts(
                &grammar,
                wrap("archive.txt\n</parameter>", 3, false)
            ));
            assert!(!accepts(
                &grammar,
                wrap("archive.txt\n</parameter>\n", 3, false)
            ));
            assert!(accepts(
                &grammar,
                wrap("archive.txt\n</parameter> literal", 3, false)
            ));
            assert!(accepts(&grammar, wrap("archive.txt\n", 3, false)));
            assert!(accepts(&parallel, format!("{call}{call}")));
            // Only llama.cpp's SPACE_RULE may follow a call.
            for gap in ["", " ", "\n", "\n\n", "\n\t \t", "\n\n   "] {
                assert!(accepts(&parallel, format!("{call}{gap}")), "{gap:?}");
                assert!(accepts(&parallel, format!("{call}{gap}{call}")), "{gap:?}");
            }
            assert!(accepts(&parallel, format!("{call}\n{}", "\t".repeat(20))));
            for gap in ["  ", "\t", "\r\n", "\n\n\n"] {
                assert!(!accepts(&parallel, format!("{call}{gap}")), "{gap:?}");
            }
            assert!(!accepts(&parallel, format!("{call}\n{}", " ".repeat(21))));
        } else {
            // Parallel DeepSeek calls share one block, which ends the output.
            const OPEN: &str = "<｜DSML｜tool_calls>";
            const CLOSE: &str = "\n</｜DSML｜tool_calls>";
            assert!(accepts(
                &parallel,
                format!(
                    "{}{}",
                    &call[..call.len() - CLOSE.len()],
                    &call[OPEN.len()..]
                )
            ));
            assert!(!accepts(&parallel, format!("{call}{call}")));
        }
        assert!(accepts(
            &JsonConstraint::with_reasoning(grammar.clone()),
            format!("Thinking.</think>{call}")
        ));
        // Masks and speculative copies agree with byte matching, also at
        // UTF-8, literal-markup and raw-string escape boundaries.
        let mut pieces: Vec<Vec<u8>> = [
            "\"42\"",
            "<think>",
            "🦉",
            "\n",
            "\\",
            "</parameter>",
            "",
            "abc",
        ]
        .iter()
        .map(|p| p.as_bytes().to_vec())
        .collect();
        pieces.extend((0..=255u8).map(|byte| vec![byte]));
        pieces.extend((0..=255u8).map(|byte| [b"shared-prefix".as_slice(), &[byte]].concat()));
        let vocabulary = Arc::new(
            ConstraintVocabulary::new(pieces.len() as u32, |i| {
                Ok(Piece {
                    text: pieces[i as usize].clone(),
                    stop: i == 6,
                })
            })
            .unwrap(),
        );
        let binding = TokenConstraint::new(grammar.clone(), vocabulary.clone());
        let mut state = grammar.start().unwrap();
        for &byte in call.as_bytes() {
            let mask = vocabulary.allowed(&grammar, &state).unwrap();
            assert_eq!(*binding.allowed(&state).unwrap(), mask);
            for token in 0..pieces.len() as u32 {
                assert_eq!(
                    vocabulary.allows(&grammar, &state, token).unwrap(),
                    mask[token as usize]
                );
            }
            state = grammar.advance(&state, byte).unwrap();
        }
        assert!(grammar.complete(&state));
    }
    let finite = parse(
        r#"{"type":"object",
          "properties":{"text":{"type":"string","enum":["<think>literal</think>"]}},
          "required":["text"],"additionalProperties":false}"#,
    );
    assert!(tool(&finite, true, ToolFormat::Qwen).is_some());
    let ambiguous = parse(
        r#"{"type":"object","properties":{"text":{"type":["string","null"]}},
          "required":["text"],"additionalProperties":false}"#,
    );
    // As llama.cpp, a strict union admitting strings keeps native raw text
    // instead of moving every tool to a JSON envelope.
    for format in NATIVE {
        let grammar = with_tools(
            None,
            "f",
            tool(&ambiguous, true, format).unwrap(),
            true,
            false,
            format,
        );
        assert!(accepts(
            &grammar,
            if format == ToolFormat::Qwen {
                "<tool_call>\n<function=f>\n<parameter=text>\nnull\n</parameter>\n</function>\n</tool_call>"
            } else {
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"text\" \
                 string=\"false\">null</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
            }
        ));
    }
    let delimiter_pattern = parse(
        r#"{"type":"object","properties":{"text":{"type":"string","pattern":"^\\n</parameter>$"}},
          "required":["text"],"additionalProperties":false}"#,
    );
    assert!(accepts(
        &JsonConstraint::compile(&delimiter_pattern, true).unwrap(),
        r#"{"text":"\n</parameter>"}"#
    ));
    // The pattern needs the native delimiter, so it cannot be enforced; as
    // in llama.cpp the parameter stays raw native text.
    let delimited = with_tools(
        None,
        "f",
        tool(&delimiter_pattern, true, ToolFormat::Qwen).unwrap(),
        true,
        false,
        ToolFormat::Qwen,
    );
    assert!(accepts(
        &delimited,
        "<tool_call>\n<function=f>\n<parameter=text>\nanything\n</parameter>\n</function>\n</tool_call>"
    ));
    for name in [" text", "text ", "\ttext", "text\x0c"] {
        let mut unusual = Value::object();
        unusual.set("type", "object");
        unusual
            .member_mut("properties")
            .member_mut(name)
            .set("type", "string");
        let mut required = Value::array();
        required.push(name);
        unusual.set("required", required);
        unusual.set("additionalProperties", false);
        let value = Value::object().with(name, "literal");
        assert!(accepts(
            &JsonConstraint::compile(&unusual, true).unwrap(),
            value.dump().unwrap()
        ));
        // As llama.cpp, the name is written literally in native syntax (the
        // parser matches declared names exactly) rather than a JSON envelope.
        let qwen = with_tools(
            None,
            "f",
            tool(&unusual, true, ToolFormat::Qwen).unwrap(),
            true,
            false,
            ToolFormat::Qwen,
        );
        assert!(accepts(
            &qwen,
            format!(
                "<tool_call>\n<function=f>\n<parameter={name}>\nliteral\n</parameter>\n\
                 </function>\n</tool_call>"
            )
        ));
        assert!(!accepts(
            &qwen,
            "<tool_call>\n<function=f>\n</function>\n</tool_call>"
        ));
        assert!(tool(&unusual, true, ToolFormat::DeepSeek).is_some());
    }
}

#[test]
fn open_native_tools() {
    for format in NATIVE {
        let parameters = JsonConstraint::open_tool_parameters(format).unwrap();
        assert!(Arc::ptr_eq(
            &parameters,
            &JsonConstraint::open_tool_parameters(format).unwrap()
        ));
        let qwen = format == ToolFormat::Qwen;
        let begin = if qwen {
            "<tool_call>\n<function=f>\n"
        } else {
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n"
        };
        let end = if qwen {
            "</function>\n</tool_call>"
        } else {
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        };
        let parameter = |name: &str, value: &str| {
            if qwen {
                format!("<parameter={name}>\n{value}\n</parameter>\n")
            } else {
                format!(
                    "<｜DSML｜parameter name=\"{name}\" string=\"true\">{value}</｜DSML｜parameter>\n"
                )
            }
        };
        let grammar = with_tools(None, "f", parameters, false, true, format);
        assert!(accepts(&grammar, "No tool is necessary."));
        assert!(accepts(&grammar, format!("{begin}{end}")));
        let call = format!(
            "{begin}{}{}{end}",
            parameter("value", "42"),
            parameter("city name", " é🦉\n\\path\n</tool_call> ")
        );
        assert!(accepts(&grammar, &call));
        // DeepSeek parallel calls share one block. As in llama.cpp, calls
        // end the output in both formats.
        assert_eq!(accepts(&grammar, format!("{call}\n{call}")), qwen);
        assert!(!accepts(&grammar, format!("{call} Done.")));
        if !qwen {
            assert!(accepts(
                &grammar,
                format!(
                    "{begin}{}</｜DSML｜invoke>{}{end}",
                    parameter("value", "42"),
                    &begin[begin.find('\n').unwrap()..]
                )
            ));
        }
        for name in [" value", "value ", "", "v>evil"] {
            assert!(!accepts(
                &grammar,
                format!("{begin}{}{end}", parameter(name, "x"))
            ));
        }
        let mut invalid_utf8 = begin.as_bytes().to_vec();
        invalid_utf8.extend_from_slice(
            parameter("value", "\u{1}")
                .as_bytes()
                .iter()
                .map(|&b| if b == 1 { 0xff } else { b })
                .collect::<Vec<_>>()
                .as_slice(),
        );
        invalid_utf8.extend_from_slice(end.as_bytes());
        assert!(!accepts(&grammar, invalid_utf8));
        if !qwen {
            let typed = "<｜DSML｜parameter name=\"value\" string=\"false\">";
            assert!(accepts(
                &grammar,
                format!(
                    "{begin}{typed}{{\"count\":7,\"ok\":true,\"values\":[null]}}</｜DSML｜parameter>\n{end}"
                )
            ));
            assert!(!accepts(
                &grammar,
                format!("{begin}{typed}not JSON</｜DSML｜parameter>\n{end}")
            ));
        }
        let mut state = grammar.start().unwrap();
        for &byte in format!("{begin}{}", parameter("value", "alpha")).as_bytes() {
            state = grammar.advance(&state, byte).unwrap();
        }
        let mut replay = state.clone();
        for &byte in end.as_bytes() {
            state = grammar.advance(&state, byte).unwrap();
            replay = grammar.advance(&replay, byte).unwrap();
        }
        assert!(state == replay && grammar.complete(&state));
    }
}

#[test]
fn non_strict_agent_tools() {
    // An ordinary agent edit schema leaves nested objects open: neither the
    // tool set nor the model's chat template switches to JSON.
    let schema = parse(
        r#"{"type":"object","properties":{
            "path":{"type":"string"},
            "edits":{"type":"array","items":{"type":"object","properties":{
              "oldText":{"type":"string"},"newText":{"type":"string"}},
              "required":["oldText","newText"]}}},
          "required":["path","edits"],"additionalProperties":false}"#,
    );
    let edits = r#"[{"oldText":"return a - b","newText":"return a + b"}]"#;
    for format in NATIVE {
        let qwen = format == ToolFormat::Qwen;
        let parameters = tool(&schema, false, format).unwrap();
        let call = if qwen {
            format!(
                "<tool_call>\n<function=edit>\n<parameter=path>\ncalc.py\n</parameter>\n\
                 <parameter=edits>\n{edits}\n</parameter>\n</function>\n</tool_call>"
            )
        } else {
            format!(
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"edit\">\n\
                 <｜DSML｜parameter name=\"path\" string=\"true\">calc.py</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"edits\" string=\"false\">{edits}</｜DSML｜parameter>\n\
                 </｜DSML｜invoke>\n</｜DSML｜tool_calls>"
            )
        };
        let replaced = |value: &str| call.replacen(edits, value, 1);
        for required in [false, true] {
            let grammar = with_tools(None, "edit", parameters.clone(), required, false, format);
            assert!(accepts(&grammar, &call));
            assert!(!accepts(&grammar, &call[..call.len() - 1]));
            assert_eq!(accepts(&grammar, "No edit is needed."), !required);
            let empty = if qwen {
                "<tool_call>\n<function=edit>\n</function>\n</tool_call>"
            } else {
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"edit\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
            };
            assert!(!accepts(&grammar, empty));
            assert!(!accepts(&grammar, replaced("false")));
            for invalid in [
                r#"[{}]"#,
                r#"[{"oldText":"a"}]"#,
                r#"[{"oldText":"a","newText":42}]"#,
                r#"[{"oldText":"a","newText":"b","oldText":42}]"#,
                r#"[{"oldText":"a","newText":"b","oldText":42}]"#,
            ] {
                assert!(!accepts(&grammar, replaced(invalid)), "{invalid}");
            }
            // Multiline edits keep their JSON escapes; raw controls in an
            // array's string are excluded while generating.
            assert!(accepts(
                &grammar,
                replaced(r#"[{"oldText":"a,\n  b","newText":"a,\n\tJSX,\n  b"}]"#)
            ));
            assert!(!accepts(
                &grammar,
                replaced("[{\"oldText\":\"a,\n  b\",\"newText\":\"a,\n\tJSX,\n  b\"}]")
            ));
            assert!(accepts(
                &grammar,
                replaced(r#"[{"oldText":"a","newText":"b","note":{"line":3}}]"#)
            ));
            let mut missing = call.clone();
            let begin = missing
                .find(if qwen {
                    "<parameter=edits>"
                } else {
                    "<｜DSML｜parameter name=\"edits\""
                })
                .unwrap();
            let close = if qwen {
                "</parameter>\n"
            } else {
                "</｜DSML｜parameter>\n"
            };
            let stop = begin + missing[begin..].find(close).unwrap() + close.len();
            missing.replace_range(begin..stop, "");
            assert!(!accepts(&grammar, &missing));
            assert!(!accepts(&grammar, call.replacen("edit", "undeclared", 1)));
        }
        assert!(JsonConstraint::tool_parameters(&schema, true, format).is_err());
        // A literal holding the native closing delimiter cannot be written
        // in native tags. As llama.cpp, the call stays native and the string
        // raw, while the neighbor keeps its nested requirements.
        let mut ambiguous = schema.clone();
        ambiguous.member_mut("properties").member_mut("path").set(
            "const",
            if qwen {
                "a\n</parameter>\nb"
            } else {
                "a</｜DSML｜parameter>b"
            },
        );
        let delimited = with_tools(
            None,
            "edit",
            tool(&ambiguous, false, format).unwrap(),
            true,
            false,
            format,
        );
        assert!(accepts(&delimited, &call));
        assert!(!accepts(&delimited, replaced(r#"[{"oldText":"a"}]"#)));
        // Unsupported root annotations keep supported nested requirements.
        let mut annotated = schema.clone();
        annotated.set("unevaluatedProperties", true);
        let edits_schema = annotated.member_mut("properties").member_mut("edits");
        edits_schema.set("x-client-extension", true);
        edits_schema
            .member_mut("items")
            .set("unevaluatedProperties", true);
        // Root wildcards cannot keep arbitrary JSON types in Qwen's tags. As
        // llama.cpp, Qwen then generates only the declared parameters.
        let annotated_parameters = tool(&annotated, false, format).unwrap();
        let annotated_call = with_tools(None, "edit", annotated_parameters, true, false, format);
        assert!(accepts(&annotated_call, &call));
        if qwen {
            let mut wildcard = call.clone();
            wildcard.insert_str(
                wildcard.find("</function>").unwrap(),
                "<parameter=extra>\n1\n</parameter>\n",
            );
            assert!(!accepts(&annotated_call, wildcard));
        }
        let annotated_json = tool(&annotated, false, ToolFormat::Json).unwrap();
        assert!(accepts(
            &annotated_json,
            r#"{"path":"a","edits":[{"oldText":"a","newText":"b"}]}"#
        ));
        let mut referenced = parse(r##"{"$ref":"#/$defs/edit","$defs":{}}"##);
        referenced.member_mut("$defs").set("edit", schema.clone());
        let referenced_call = with_tools(
            None,
            "edit",
            tool(&referenced, false, format).unwrap(),
            true,
            false,
            format,
        );
        assert!(accepts(&referenced_call, &call));
        for invalid in [
            r#"[{}]"#,
            r#"[{"oldText":"a"}]"#,
            r#"[{"oldText":"a","newText":42}]"#,
        ] {
            assert!(!accepts(&annotated_call, replaced(invalid)));
            assert!(!accepts(
                &annotated_json,
                format!(r#"{{"path":"a","edits":{invalid}}}"#)
            ));
            assert!(!accepts(&referenced_call, replaced(invalid)));
        }
    }
    // Best-effort reference handling stays bounded.
    let cyclic = parse(r##"{"$ref":"#","type":"object"}"##);
    assert!(tool(&cyclic, false, ToolFormat::Qwen).is_some());
    assert!(tool(&cyclic, false, ToolFormat::Json).is_some());

    // Open objects belong to non-strict tools, never to strict response
    // schemas; native and JSON fallback arguments keep nested schemas.
    let json = tool(&schema, false, ToolFormat::Json).unwrap();
    assert!(accepts(
        &json,
        r#"{"path":"a","edits":[{"oldText":"a","newText":"b"}]}"#
    ));
    assert!(!accepts(&json, r#"{"path":"a","edits":[{}]}"#));
    assert!(JsonConstraint::compile(&schema, false).is_err());

    // Additional-property schemas and escaped or metacharacter keys keep the
    // declared property rules; the key matcher sees decoded Unicode.
    let extras = parse(
        r#"{"type":"object","properties":{"value":{"type":"object",
            "properties":{"a.b":{"type":"integer"},"😀":{"type":"string"}},
            "required":["a.b","😀"],"additionalProperties":{"type":"boolean"}}},
          "required":["value"],"additionalProperties":false}"#,
    );
    let extra_json = tool(&extras, false, ToolFormat::Json).unwrap();
    assert!(accepts(
        &extra_json,
        r#"{"value":{"a.b":3,"😀":"x","other":true}}"#
    ));
    assert!(accepts(
        &extra_json,
        r#"{"value":{"a.b":3,"😀":"x","aXb":false}}"#
    ));
    for invalid in [
        r#"{"value":{"a.b":3,"😀":"x","other":1}}"#,
        r#"{"value":{"a.b":3,"😀":"x","a.b":false}}"#,
        r#"{"value":{"a.b":3,"😀":"x","a.b":false}}"#,
        r#"{"value":{"a.b":3,"😀":"x","😀":false}}"#,
    ] {
        assert!(!accepts(&extra_json, invalid), "{invalid}");
    }
    let mut finite = extras.clone();
    finite.member_mut("properties").member_mut("value").set(
        "enum",
        parse(r#"[{"a.b":3,"😀":"x","other":true},{"a.b":3,"😀":"x","other":42}]"#),
    );
    let finite_json = tool(&finite, false, ToolFormat::Json).unwrap();
    assert!(accepts(
        &finite_json,
        r#"{"value":{"a.b":3,"😀":"x","other":true}}"#
    ));
    assert!(!accepts(
        &finite_json,
        r#"{"value":{"a.b":3,"😀":"x","other":42}}"#
    ));
}

#[test]
fn untyped_non_strict_tools() {
    for format in NATIVE {
        let qwen = format == ToolFormat::Qwen;
        let begin = if qwen {
            "<tool_call>\n<function=record>\n"
        } else {
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n"
        };
        let end = if qwen {
            "</function>\n</tool_call>"
        } else {
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        };
        let value = if qwen {
            "<parameter=value>\nalpha\n</parameter>\n"
        } else {
            "<｜DSML｜parameter name=\"value\" string=\"true\">alpha</｜DSML｜parameter>\n"
        };
        for text in [
            r#"{"type":"object","additionalProperties":true}"#,
            r#"{"type":"object","unevaluatedProperties":true}"#,
            r#"{"type":"object","additionalProperties":true,"dependentRequired":{"a":["b"]}}"#,
            r#"{"type":"object","properties":{},"additionalProperties":true}"#,
            r#"{"type":"object","properties":{},"additionalProperties":false,"unevaluatedProperties":true}"#,
        ] {
            let parameters = tool(&parse(text), false, format).unwrap();
            let grammar = with_tools(None, "record", parameters.clone(), false, false, format);
            assert!(accepts(&grammar, format!("{begin}{value}{end}")), "{text}");
            assert!(accepts(&grammar, format!("{begin}{end}")), "{text}");
            let final_answer = JsonConstraint::compile(
                &parse(
                    r#"{"type":"object","properties":{"done":{"type":"boolean","const":true}},
                      "required":["done"],"additionalProperties":false}"#,
                ),
                false,
            )
            .unwrap();
            let combined = with_tools(
                Some(final_answer),
                "record",
                parameters,
                false,
                false,
                format,
            );
            assert!(accepts(&combined, r#"{"done":true}"#));
            assert!(!accepts(&combined, r#"{"done":false}"#));
            assert!(!accepts(&combined, r#"{"other":true}"#));
            assert!(accepts(&combined, format!("{begin}{value}{end}")));
        }
        // A genuinely closed empty schema still describes a zero-argument
        // tool.
        for text in [
            r#"{"type":"object","properties":{},"additionalProperties":false}"#,
            r#"{"type":"object","unevaluatedProperties":false}"#,
            r#"{}"#,
        ] {
            let parameters = tool(&parse(text), false, format).unwrap();
            let grammar = with_tools(None, "record", parameters, true, false, format);
            assert!(accepts(&grammar, format!("{begin}{end}")), "{text}");
            assert!(!accepts(&grammar, format!("{begin}{value}{end}")), "{text}");
            let json = tool(&parse(text), false, ToolFormat::Json).unwrap();
            assert!(accepts(&json, "{}"));
            assert!(!accepts(&json, r#"{"value":"alpha"}"#));
        }
        // Open properties do not erase exact finite values or typed extras.
        for text in [
            r#"{"type":"object","additionalProperties":true,"const":{"value":"alpha"}}"#,
            r#"{"type":"object","const":{"value":"alpha"}}"#,
            r#"{"type":"object","enum":[{"value":"alpha"}],"unevaluatedProperties":true}"#,
            r#"{"type":"object","additionalProperties":true,"const":{"value":"alpha"},"unevaluatedProperties":true}"#,
            r#"{"type":"object","additionalProperties":true,"enum":[{"value":"alpha"}]}"#,
            r#"{"type":"object","additionalProperties":{"type":"string"}}"#,
        ] {
            let schema = parse(text);
            // No declared properties: as in llama.cpp the call stays native
            // and takes no arguments.
            let native = with_tools(
                None,
                "record",
                tool(&schema, false, format).unwrap(),
                true,
                false,
                format,
            );
            assert!(accepts(&native, format!("{begin}{end}")), "{text}");
            assert!(!accepts(&native, format!("{begin}{value}{end}")), "{text}");
            let json = tool(&schema, false, ToolFormat::Json).unwrap();
            assert!(accepts(&json, r#"{"value":"alpha"}"#), "{text}");
            assert!(!accepts(&json, r#"{"value":42}"#), "{text}");
        }
    }
}

fn record_call(format: ToolFormat, name: &str, value: &str, string: bool) -> String {
    if format == ToolFormat::Qwen {
        format!(
            "<tool_call>\n<function=record>\n<parameter={name}>\n{value}\n</parameter>\n\
             </function>\n</tool_call>"
        )
    } else {
        format!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n\
             <｜DSML｜parameter name=\"{name}\" string=\"{}\">{value}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            if string { "true" } else { "false" }
        )
    }
}

#[test]
fn mixed_best_effort_tool_routes() {
    let call = record_call;
    for text in [
        r#"{"type":"object","unevaluatedProperties":true}"#,
        r#"{"type":"object","oneOf":[{"properties":{"a":{"type":"string"}},"required":["a"]}]}"#,
        r#"{"type":"object","allOf":[{"properties":{"a":{"type":"string"}},"required":["a"]}]}"#,
    ] {
        let json = tool(&parse(text), false, ToolFormat::Json).unwrap();
        assert!(accepts(&json, r#"{"a":"x"}"#), "{text}");
    }
    for text in [
        r##"{"$schema":"http://json-schema.org/draft-07/schema#","$id":"urn:record","$comment":"annotation",
            "$ref":"#/definitions/A","definitions":{"A":{"type":"object",
            "properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}}}"##,
        r##"{"type":"object","$ref":"#/$defs/A","$defs":{"A":{"type":"object",
            "properties":{"a":{"type":"string"}},"required":["a"]}}}"##,
        r##"{"$ref":"#/$defs/A","default":{"a":"x"},"examples":[{"a":"x"}],
            "$defs":{"A":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}}"##,
        r#"{"type":"object","properties":{"a":{"type":"string","format":"uri","minLength":1}},
            "required":["a"],"additionalProperties":false}"#,
    ] {
        for format in ALL {
            let parameters = tool(&parse(text), false, format).unwrap();
            if format == ToolFormat::Json {
                assert!(accepts(&parameters, r#"{"a":"x"}"#), "{text}");
                assert!(!accepts(&parameters, "{}"), "{text}");
            } else {
                let grammar = with_tools(None, "record", parameters, true, false, format);
                assert!(accepts(&grammar, call(format, "a", "x", true)), "{text}");
                assert!(
                    !accepts(&grammar, call(format, "other", "x", true)),
                    "{text}"
                );
            }
        }
    }
    let nested = parse(
        r#"{"type":"object","properties":{"m":{"type":"object",
            "patternProperties":{"^k":{"type":"integer"}},"additionalProperties":false}},
          "required":["m"]}"#,
    );
    for format in ALL {
        let parameters = tool(&nested, false, format).unwrap();
        if format == ToolFormat::Json {
            assert!(accepts(&parameters, r#"{"m":{"k1":1}}"#));
            assert!(!accepts(&parameters, "{}"));
        } else {
            let grammar = with_tools(None, "record", parameters, true, false, format);
            assert!(accepts(&grammar, call(format, "m", r#"{"k1":1}"#, false)));
        }
    }
    let mut finite = nested.clone();
    finite.set("const", parse(r#"{"m":{"k1":1}}"#));
    let finite_json = tool(&finite, false, ToolFormat::Json).unwrap();
    assert!(accepts(&finite_json, r#"{"m":{"k1":1}}"#));
    assert!(!accepts(&finite_json, r#"{"m":{}}"#));
    for text in [
        r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],
            "patternProperties":{"^x_":{"type":"string"}}}"#,
        r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],
            "dependentSchemas":{"a":{"properties":{"x_b":{"type":"string"}}}}}"#,
        r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],
            "additionalProperties":true}"#,
    ] {
        for format in ALL {
            let parameters = tool(&parse(text), false, format).unwrap();
            if format == ToolFormat::Qwen {
                // Qwen tags cannot type wildcard values. As llama.cpp, only
                // declared parameters are generated and the call stays native.
                let grammar = with_tools(None, "record", parameters, true, false, format);
                let mut with_extra = call(format, "a", "1", true);
                with_extra.insert_str(
                    with_extra.find("</function>").unwrap(),
                    "<parameter=x_b>\n2\n</parameter>\n",
                );
                assert!(accepts(&grammar, call(format, "a", "1", true)), "{text}");
                assert!(!accepts(&grammar, with_extra), "{text}");
                assert!(!accepts(&grammar, call(format, "x_b", "2", true)), "{text}");
                continue;
            }
            if format == ToolFormat::Json {
                assert!(accepts(&parameters, r#"{"a":"1","x_b":"2"}"#), "{text}");
                assert!(!accepts(&parameters, r#"{"a":1,"x_b":"2"}"#), "{text}");
                assert!(!accepts(&parameters, r#"{"x_b":"2"}"#), "{text}");
                assert!(!accepts(&parameters, r#"{"a":"1","a":2}"#), "{text}");
            } else {
                let grammar = with_tools(None, "record", parameters, true, false, format);
                let mut both = call(format, "a", "1", true);
                let offset = both.find("</｜DSML｜invoke>").unwrap();
                both.insert_str(
                    offset,
                    "<｜DSML｜parameter name=\"x_b\" string=\"true\">2</｜DSML｜parameter>\n",
                );
                assert!(accepts(&grammar, &both), "{text}");
                assert!(!accepts(&grammar, call(format, "x_b", "2", true)), "{text}");
                assert!(!accepts(&grammar, both.replacen("x_b", "a", 1)), "{text}");
            }
        }
    }
    // Non-strict unions keep native framing, as in llama.cpp: a JSON
    // envelope would contradict the template and the native calls in
    // history. Qwen takes raw text for a union with strings; DeepSeek's
    // string flag selects raw text or the typed JSON alternatives.
    for format in NATIVE {
        let nullable = parse(
            r#"{"type":"object","properties":{"a":{
                "anyOf":[{"type":"string"},{"type":"null"}]}},"required":["a"]}"#,
        );
        let parameters = tool(&nullable, false, format).unwrap();
        let grammar = with_tools(None, "record", parameters, true, false, format);
        assert!(accepts(&grammar, call(format, "a", "null", false)));
        assert!(accepts(
            &grammar,
            call(format, "a", "#include \"a.h\"\nx", true)
        ));
        assert!(!accepts(&grammar, call(format, "b", "x", true)));
        if format == ToolFormat::DeepSeek {
            assert!(!accepts(&grammar, call(format, "a", "x", false)));
        }
        // Strict unions keep native framing too; llama.cpp has no strict
        // mode and the parser types the value against the schema.
        let mut closed = nullable.clone();
        closed.set("additionalProperties", false);
        let strict_grammar = with_tools(
            None,
            "record",
            tool(&closed, true, format).unwrap(),
            true,
            false,
            format,
        );
        assert!(accepts(&strict_grammar, call(format, "a", "null", false)));
        assert!(accepts(&strict_grammar, call(format, "a", "text", true)));
        assert!(!accepts(&strict_grammar, call(format, "b", "x", true)));
        let json = tool(&nullable, false, ToolFormat::Json).unwrap();
        assert!(accepts(&json, r#"{"a":null}"#));
        assert!(accepts(&json, r#"{"a":"null"}"#));
        // Finite string alternatives and untyped values are text; a union
        // without strings keeps its JSON value constraint inside the tag.
        let agent = parse(
            r#"{"type":"object","properties":{
                "query":{"type":"string"},
                "provider":{"anyOf":[{"type":"string","const":"brave"},
                                     {"type":"string","const":"exa"}]},
                "args":{"anyOf":[{"type":"string"},
                                 {"type":"object","additionalProperties":true}]},
                "extra":{"description":"anything"},
                "limit":{"anyOf":[{"type":"integer"},{"type":"null"}]}}}"#,
        );
        let tools = tool(&agent, false, format).unwrap();
        let agent_grammar = with_tools(None, "record", tools, true, false, format);
        assert!(accepts(
            &agent_grammar,
            call(format, "query", "a \"quoted\" b", true)
        ));
        assert!(accepts(
            &agent_grammar,
            call(format, "provider", "exa", true)
        ));
        assert!(accepts(&agent_grammar, call(format, "extra", "x", true)));
        assert!(accepts(&agent_grammar, call(format, "limit", "3", false)));
        assert!(accepts(
            &agent_grammar,
            call(format, "limit", "null", false)
        ));
        assert!(!accepts(&agent_grammar, call(format, "limit", "x", false)));
    }
    // Ignoring a numeric format annotation keeps supported bounds.
    let integer = parse(
        r#"{"type":"object","properties":{"a":{
            "type":"integer","format":"int32","minimum":5,"maximum":10}},
          "required":["a"]}"#,
    );
    for format in ALL {
        let parameters = tool(&integer, false, format).unwrap();
        if format == ToolFormat::Json {
            assert!(accepts(&parameters, r#"{"a":5}"#));
            assert!(!accepts(&parameters, r#"{"a":4}"#));
        } else {
            let grammar = with_tools(None, "record", parameters, true, false, format);
            assert!(accepts(&grammar, call(format, "a", "5", false)));
            assert!(!accepts(&grammar, call(format, "a", "4", false)));
        }
    }
}

#[test]
fn tool_schema_safety() {
    let ordinary = parse(
        r#"{"type":"object","properties":{
            "value":{"type":"string","const":"alpha"}},"required":["value"],
          "additionalProperties":false}"#,
    );
    let uri = parse(
        r#"{"type":"object","properties":{
            "url":{"type":"string","format":"uri"}},"required":["url"]}"#,
    );
    for format in NATIVE {
        // As in llama.cpp, a required call uses the same native arguments as
        // an automatic one, whatever annotations the schema carries.
        assert!(tool(&ordinary, false, format).is_some());
        let required = with_tools(
            None,
            "record",
            tool(&uri, false, format).unwrap(),
            true,
            false,
            format,
        );
        assert!(accepts(
            &required,
            record_call(format, "url", "https://a.b", true)
        ));
    }
    for text in [
        r#"{"type":"object","properties":{"x":{"type":"integer","minimum":5,"maximum":2}},"required":["x"]}"#,
        r##"{"type":"object","properties":{"x":{"$ref":"#"}},"required":["x"]}"##,
        r##"{"$ref":"#"}"##,
    ] {
        let schema = parse(text);
        // Unsatisfiable or recursive guidance cannot be enforced; the value
        // stays a generic native argument instead of switching to a JSON
        // envelope. A cyclic root declares no properties, so as llama.cpp it
        // takes none.
        let declared = schema.contains("properties");
        for format in NATIVE {
            let grammar = with_tools(
                None,
                "record",
                tool(&schema, false, format).unwrap(),
                true,
                false,
                format,
            );
            let empty = if format == ToolFormat::Qwen {
                "<tool_call>\n<function=record>\n</function>\n</tool_call>"
            } else {
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
            };
            assert_eq!(accepts(&grammar, empty), !declared, "{text}");
            assert_eq!(
                accepts(&grammar, record_call(format, "x", "5", false)),
                declared,
                "{text}"
            );
        }
        let json = tool(&schema, false, ToolFormat::Json).unwrap();
        assert!(accepts(&json, r#"{"x":5}"#), "{text}");
    }
    // Optional impossible branches are pruned rather than admitting dead
    // prefixes; the remaining native call keeps a finite completion.
    let optional = parse(
        r#"{"type":"object","properties":{
            "x":{"type":"integer","minimum":5,"maximum":2},
            "y":{"type":"string","const":"ok"}},"required":["y"],
          "additionalProperties":false}"#,
    );
    let parameters = tool(&optional, false, ToolFormat::Qwen).unwrap();
    let native = with_tools(None, "record", parameters, true, false, ToolFormat::Qwen);
    assert!(accepts(
        &native,
        "<tool_call>\n<function=record>\n<parameter=y>\nok\n</parameter>\n</function>\n</tool_call>"
    ));
    let mut state = native.start().unwrap();
    for &byte in b"<tool_call>\n<function=record>\n<parameter=x>\n" {
        state = native.advance(&state, byte).unwrap();
    }
    assert!(state.is_empty());

    for extension in [
        r#"{"patternProperties":{"^payload$":{"type":"integer"}}}"#,
        r#"{"if":{"properties":{"kind":{"const":"x"}}},"then":{"properties":{"payload":{"type":"integer"}},"required":["payload"]}}"#,
        r#"{"if":{"properties":{"kind":{"const":"y"}}},"else":{"properties":{"payload":{"type":"integer"}},"required":["payload"]}}"#,
        r#"{"dependencies":{"kind":{"properties":{"payload":{"type":"integer"}},"required":["payload"]}}}"#,
    ] {
        let mut schema = parse(extension);
        schema.set("type", "object");
        schema.set("properties", parse(r#"{"kind":{"type":"string"}}"#));
        schema.set("required", parse(r#"["kind"]"#));
        // Qwen stays native and, as llama.cpp, generates only the root
        // object's declared names; branch keywords declare none.
        let qwen = with_tools(
            None,
            "record",
            tool(&schema, false, ToolFormat::Qwen).unwrap(),
            true,
            false,
            ToolFormat::Qwen,
        );
        assert!(
            accepts(&qwen, record_call(ToolFormat::Qwen, "kind", "x", true)),
            "{extension}"
        );
        assert!(
            !accepts(
                &qwen,
                "<tool_call>\n<function=record>\n<parameter=kind>\nx\n</parameter>\n\
                 <parameter=payload>\n1\n</parameter>\n</function>\n</tool_call>"
            ),
            "{extension}"
        );
        let json = tool(&schema, false, ToolFormat::Json).unwrap();
        for value in ["1", "true", "null", "[1]", r#"{"x":1}"#] {
            // Unsupported branches stay guidance: the types the model chooses
            // reach the HTTP parser intact instead of becoming raw strings.
            assert!(
                accepts(&json, format!(r#"{{"kind":"x","payload":{value}}}"#)),
                "{extension} {value}"
            );
        }
        assert!(!accepts(&json, r#"{"kind":1,"payload":1}"#));
        assert!(!accepts(&json, r#"{"payload":1}"#));
        assert!(tool(&schema, false, ToolFormat::DeepSeek).is_some());
    }
    let metadata = parse(
        r#"{"type":"object","properties":{
            "edits":{"type":"array","items":{"type":"object","properties":{
              "oldText":{"type":"string"},"newText":{"type":"string"},"metadata":{}},
              "required":["oldText","newText"]}}},"required":["edits"]}"#,
    );
    let json = tool(&metadata, false, ToolFormat::Json).unwrap();
    assert!(accepts(
        &json,
        r#"{"edits":[{"oldText":"a","newText":"b","metadata":{"x":1}}]}"#
    ));
    assert!(!accepts(&json, r#"{"edits":[{}]}"#));
    assert!(!accepts(
        &json,
        r#"{"edits":[{"oldText":1,"newText":"b"}]}"#
    ));
    for format in NATIVE {
        assert!(tool(&metadata, false, format).is_some());
    }
    for format in NATIVE {
        let required = with_tools(
            None,
            "record",
            tool(&metadata, false, format).unwrap(),
            true,
            false,
            format,
        );
        let edits = r#"[{"oldText":"a","newText":"b","metadata":{"x":1}}]"#;
        let call = record_call(format, "edits", edits, false);
        assert!(accepts(&required, &call));
        assert!(!accepts(&required, call.replacen(edits, "[{}]", 1)));
    }
}

/// Optional calls leave sampling unconstrained until a call marker:
/// identical draws and RNG to the unconstrained sampler, a legal argmax
/// without the mask, then a binding mask inside the call; calls end the
/// output.
#[test]
fn automatic_tools() {
    use gufo_core::sampling::{SamplerState, SamplingConfig};
    let schema = parse(
        r#"{"type":"object","properties":{
            "text":{"type":"string"},"optional":{"type":"integer"}},
          "required":["text"],"additionalProperties":false}"#,
    );
    for format in NATIVE {
        let qwen = format == ToolFormat::Qwen;
        let opener = if qwen {
            "<tool_call>"
        } else {
            "<｜DSML｜tool_calls>"
        };
        let call = if qwen {
            "\n<function=f>\n<parameter=text>\nalpha\n</parameter>\n</function>\n</tool_call>"
        } else {
            "\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"text\" string=\"true\">alpha\
             </｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        };
        let automatic = with_tools(
            None,
            "f",
            tool(&schema, false, format).unwrap(),
            false,
            true,
            format,
        );
        let vocabulary = Arc::new(
            ConstraintVocabulary::new(259, |i| {
                Ok(Piece {
                    text: match i {
                        0..256 => vec![i as u8],
                        258 => opener.as_bytes().to_vec(),
                        _ => Vec::new(),
                    },
                    stop: i == 256,
                })
            })
            .unwrap(),
        );
        for thinking in [false, true] {
            let grammar = if thinking {
                JsonConstraint::with_reasoning(automatic.clone())
            } else {
                automatic.clone()
            };
            let binding = Arc::new(TokenConstraint::new(grammar, vocabulary.clone()));
            for temperature in [0.0, 0.7, 1.4] {
                let config = SamplingConfig {
                    temperature,
                    top_k: 20,
                    top_p: 0.8,
                    min_p: 0.05,
                    seed: 41,
                    frequency_penalty: 0.2,
                    presence_penalty: 0.3,
                    constraint: Some(binding.clone()),
                    ..SamplingConfig::default()
                };
                let mut sampler = SamplerState::new(config, &[]).unwrap();
                let mut plain = sampler.without_constraint();
                assert!(!sampler.needs_constraint_mask().unwrap());
                assert!(!sampler.can_select_argmax(b'a'.into(), false).unwrap());
                assert_eq!(
                    sampler.can_select_argmax(b'a'.into(), true).unwrap(),
                    temperature == 0.0
                );
                let mut logits = vec![f32::NEG_INFINITY; 259];
                logits[usize::from(b'a')] = 1.0;
                logits[usize::from(b'b')] = 0.8;
                for _ in 0..8 {
                    let expected = plain.sample(&logits).unwrap();
                    let actual = sampler.sample(&logits).unwrap();
                    assert!(actual == expected && sampler.rng_state() == plain.rng_state());
                    sampler.accept(actual).unwrap();
                    plain.accept(expected).unwrap();
                    assert!(!sampler.needs_constraint_mask().unwrap());
                }
                let accept = |sampler: &mut SamplerState, text: &str| {
                    for &byte in text.as_bytes() {
                        sampler.accept(byte.into()).unwrap();
                    }
                };
                if thinking {
                    accept(&mut sampler, "</thi");
                    let saved = sampler.clone();
                    accept(&mut sampler, "nk>");
                    assert!(!sampler.needs_constraint_mask().unwrap());
                    sampler = saved;
                    accept(&mut sampler, "nk>");
                }
                let before_call = sampler.clone();
                sampler.accept(258).unwrap();
                assert!(sampler.needs_constraint_mask().unwrap());
                assert!(!sampler.can_select_argmax(b'!'.into(), true).unwrap());
                assert!(!sampler.can_select_argmax(256, true).unwrap());
                accept(&mut sampler, call);
                // As in llama.cpp, calls end the output: no text may follow
                // them. Parallel Qwen calls may follow each other; DeepSeek's
                // share a block.
                assert!(sampler.needs_constraint_mask().unwrap());
                assert!(!sampler.can_select_argmax(b'a'.into(), true).unwrap());
                if !qwen {
                    assert!(!sampler.can_select_argmax(258, true).unwrap());
                }
                let mut after_call = vec![f32::NEG_INFINITY; 259];
                after_call[usize::from(b'a')] = 1.0;
                after_call[256] = 0.0;
                assert_eq!(sampler.clone().sample(&after_call).unwrap(), 256);
                sampler.accept(257).unwrap(); // Empty pieces keep the path.
                sampler.accept(256).unwrap(); // A natural stop needs no call.
                sampler = before_call;
                assert!(!sampler.needs_constraint_mask().unwrap());
                sampler.reset_history(&[]);
                assert!(!sampler.needs_constraint_mask().unwrap());
            }
        }
        // A token can span the trigger and an invalid name: such a
        // vocabulary never takes the unrestricted fast path.
        let crossing_vocabulary = Arc::new(
            ConstraintVocabulary::new(2, |i| {
                Ok(Piece {
                    text: if i == 0 {
                        b"text".to_vec()
                    } else {
                        format!("{opener}\n<unknown>").into_bytes()
                    },
                    stop: false,
                })
            })
            .unwrap(),
        );
        let crossing = Arc::new(TokenConstraint::new(automatic.clone(), crossing_vocabulary));
        let mut sampler = SamplerState::new(
            SamplingConfig {
                constraint: Some(crossing),
                ..SamplingConfig::default()
            },
            &[],
        )
        .unwrap();
        assert!(sampler.needs_constraint_mask().unwrap());
        assert_eq!(sampler.sample(&[0.0, 100.0]).unwrap(), 0);
    }
}

/// llama.cpp `common/parsers/qwen3-coder.cpp` and `deepseek.cpp` keep native
/// call syntax for every schema: strings are raw text and other values
/// follow their schema as far as it is supported. No schema, tool choice or
/// strict flag moves a native model to a JSON envelope.
#[test]
fn native_tool_schema_matrix() {
    struct Property {
        schema: &'static str,
        value: &'static str,
        string: bool,
        /// A value the native grammar must still refuse.
        rejected: Option<&'static str>,
    }
    let property = |schema, value, string, rejected| Property {
        schema,
        value,
        string,
        rejected,
    };
    let properties = [
        property(r#"{"type":"string"}"#, "plain \"text\"\n<b>", true, None),
        property(
            r#"{"type":"string","pattern":"^[0-9]{4}$"}"#,
            "not a year",
            true,
            None,
        ),
        property(
            r#"{"type":"string","format":"uri"}"#,
            "https://example.com",
            true,
            None,
        ),
        property(
            r#"{"type":"string","default":"x","examples":["y"],"deprecated":true,"nullable":true,"title":"T"}"#,
            "x",
            true,
            None,
        ),
        property(
            r#"{"type":"string","enum":["red","blue"]}"#,
            "red",
            true,
            None,
        ),
        property(r#"{"type":"string","minLength":2}"#, "ab", true, None),
        property(
            r#"{"type":"integer","exclusiveMinimum":0,"maximum":9007199254740991}"#,
            "240000",
            false,
            Some("x"),
        ),
        property(
            r#"{"type":"integer","minimum":5,"maximum":2}"#,
            "5",
            false,
            None,
        ),
        property(r#"{"type":"boolean"}"#, "true", false, Some("yes")),
        property(
            r#"{"oneOf":[{"type":"string"},{"type":"integer"}]}"#,
            "free text",
            true,
            None,
        ),
        property(
            r#"{"anyOf":[{"type":"integer"},{"type":"null"}]}"#,
            "3",
            false,
            Some("x"),
        ),
        property(
            r#"{"allOf":[{"type":"string"},{"minLength":1}]}"#,
            "text",
            true,
            None,
        ),
        property(
            r#"{"allOf":[{"type":"integer"},{"minimum":1}]}"#,
            "2",
            false,
            Some("x"),
        ),
        property(r#"{"not":{"type":"null"}}"#, "anything", true, None),
        property(r#"{"type":["string","null"]}"#, "text", true, None),
        property(
            r#"{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}"#,
            r#"{"x": 1}"#,
            false,
            Some("x"),
        ),
        property(
            r#"{"type":"array","items":{"type":"object","properties":{"k":{"type":"string"}},"required":["k"]}}"#,
            r#"[{"k": "v"}]"#,
            false,
            Some("[{}]"),
        ),
        property(r##"{"$ref":"#/$defs/S"}"##, "referenced", true, None),
        property("{}", "untyped", true, None),
        property(
            r#"{"if":{"type":"string"},"then":{"minLength":1}}"#,
            "conditional",
            true,
            None,
        ),
    ];
    // Roots marked open admit extra names, which only DeepSeek's typed
    // string flag can carry; Qwen generates declared names only, as
    // llama.cpp.
    let roots = [
        (r#"{"type":"object"}"#, false),
        (
            r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false}"#,
            false,
        ),
        (r#"{"type":"object","additionalProperties":true}"#, true),
        (
            r#"{"type":"object","patternProperties":{"^x_":{"type":"integer"}}}"#,
            true,
        ),
        (
            r#"{"type":"object","if":{"required":["v"]},"then":{"required":["v"]}}"#,
            true,
        ),
        (
            r#"{"type":"object","unevaluatedProperties":true,"title":"Args","description":"d"}"#,
            true,
        ),
    ];
    let mut checked = 0;
    for (root_text, open) in roots {
        for property in &properties {
            let mut schema = parse(root_text);
            schema.set("properties", Value::object());
            schema
                .member_mut("properties")
                .set("v", parse(property.schema));
            schema
                .member_mut("properties")
                .set("other", parse(r#"{"type":"string"}"#));
            schema.set("required", parse(r#"["v"]"#));
            schema.set("$defs", parse(r#"{"S":{"type":"string"}}"#));
            for format in NATIVE {
                let qwen = format == ToolFormat::Qwen;
                let argument = |name: &str, value: &str, string: bool| {
                    if qwen {
                        format!("<parameter={name}>\n{value}\n</parameter>\n")
                    } else {
                        format!(
                            "<｜DSML｜parameter name=\"{name}\" string=\"{}\">{value}</｜DSML｜parameter>\n",
                            if string { "true" } else { "false" }
                        )
                    }
                };
                let call = |arguments: &str| {
                    if qwen {
                        format!("<tool_call>\n<function=t>\n{arguments}</function>\n</tool_call>")
                    } else {
                        format!(
                            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"t\">\n{arguments}\
                             </｜DSML｜invoke>\n</｜DSML｜tool_calls>"
                        )
                    }
                };
                for strict in [false, true] {
                    for required in [false, true] {
                        let case = format!(
                            "root={root_text} v={} qwen={qwen} strict={strict} required={required}",
                            property.schema
                        );
                        let parameters =
                            match JsonConstraint::tool_parameters(&schema, strict, format) {
                                Ok(parameters) => parameters,
                                // Only an impossible or unsupported strict schema
                                // is rejected.
                                Err(error) => {
                                    assert!(
                                        strict
                                            && matches!(
                                                error,
                                                SchemaError::Empty(_) | SchemaError::Invalid(_)
                                            ),
                                        "{case}: {}",
                                        error.message()
                                    );
                                    continue;
                                }
                            };
                        let parameters = parameters
                            .unwrap_or_else(|| panic!("{case}: a native model never uses JSON"));
                        let grammar = with_tools(None, "t", parameters, required, false, format);
                        let value = argument("v", property.value, property.string);
                        assert!(accepts(&grammar, call(&value)), "value refused: {case}");
                        assert!(
                            accepts(&grammar, call(&(argument("other", "o", true) + &value))),
                            "{case}"
                        );
                        assert!(
                            !accepts(&grammar, call(&argument("other", "o", true))),
                            "missing required accepted: {case}"
                        );
                        // Undeclared names are never generated for Qwen,
                        // strict tools or closed roots; DeepSeek's typed
                        // string flag may carry them for an open root, as its
                        // native tool parameters already admit.
                        if qwen || strict || !open {
                            assert!(
                                !accepts(
                                    &grammar,
                                    call(&(value.clone() + &argument("zz", "1", true)))
                                ),
                                "undeclared accepted: {case}"
                            );
                        }
                        if let Some(rejected) = property.rejected {
                            assert!(
                                !accepts(&grammar, call(&argument("v", rejected, property.string))),
                                "rejected value accepted: {case}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked >= roots.len() * properties.len() * 2 * 2);
}

/// Verdicts of llama.cpp's own Qwen3-Coder and DeepSeek tool grammars (GBNF
/// engine, tool_choice required) for 86 schemas x 4 outputs, every
/// difference recorded with its reason (upstream's `tools/llama_parity`
/// regenerates `fixtures/llama_cpp_tool_grammar.json`).
#[test]
fn llama_cpp_tool_grammar_goldens() {
    let fixture = parse(include_str!("fixtures/llama_cpp_tool_grammar.json"));
    let (mut rows, mut differences) = (0, 0);
    for row in fixture.find("rows").unwrap().items() {
        let format = if row.member_str("format", "") == "qwen" {
            ToolFormat::Qwen
        } else {
            ToolFormat::DeepSeek
        };
        let grammar = with_tools(
            None,
            "record",
            tool(row.find("parameters").unwrap(), false, format).unwrap(),
            true,
            false,
            format,
        );
        let accepted = accepts(&grammar, row.member_str("output", ""));
        let llama = row.find("llama_cpp").unwrap().as_bool(false);
        let expected = row.find("gufo").unwrap().as_bool(false);
        let class = row.member_str("class", "");
        assert_eq!(
            accepted,
            expected,
            "llama.cpp golden mismatch: {} {} {class}",
            row.member_str("format", ""),
            row.member_str("case", "")
        );
        let explained = row.contains("difference");
        assert_eq!(explained, accepted != llama);
        // gufo is never stricter on a valid output, except a const it
        // enforces.
        if llama && class == "valid" && !accepted {
            assert!(row.member_str("difference", "").contains("const"));
        }
        rows += 1;
        differences += usize::from(explained);
    }
    assert!(rows == 688 && differences < rows / 5);
}
