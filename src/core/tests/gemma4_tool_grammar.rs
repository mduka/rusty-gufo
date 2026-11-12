//! Gemma 4's native call grammar (`ToolFormat::Gemma4`) and model-specific
//! reasoning delimiters, as the Gemma 4 extension's `tests/tool_syntax_test.cpp`
//! (`CheckGrammar`) and `tests/core/json_constraint_test.cpp` of
//! franzmoca/gufo's `feat/gemma` branch
//! check them: cargo test -p gufo-core --test gemma4_tool_grammar

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use gufo_core::json::{self, Value};
use gufo_core::json_constraint::{JsonConstraint, ToolFormat};

const CALL_START: &str = "<|tool_call>";
const CALL_END: &str = "<tool_call|>";
const THOUGHT_START: &str = "<|channel>thought\n";
const THOUGHT_END: &str = "<channel|>";
const QUOTE: &str = "<|\"|>";

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

fn parameters(schema: &Value, strict: bool) -> Arc<JsonConstraint> {
    JsonConstraint::tool_parameters(schema, strict, ToolFormat::Gemma4)
        .unwrap()
        .unwrap()
}

fn calls(
    name: &str,
    parameters: Arc<JsonConstraint>,
    required: bool,
    parallel: bool,
) -> Arc<JsonConstraint> {
    JsonConstraint::with_tools(
        None,
        vec![(name.into(), parameters)],
        required,
        parallel,
        ToolFormat::Gemma4,
    )
    .unwrap()
    .unwrap()
}

fn call(name: &str, arguments: &str) -> String {
    format!("{CALL_START}call:{name}{{{arguments}}}{CALL_END}")
}

/// `<|"|>`-delimited text.
fn quoted(text: &str) -> String {
    format!("{QUOTE}{text}{QUOTE}")
}

#[test]
fn gemma4_calls_follow_the_template_spelling() {
    let schema = parse(
        r#"{
        "type":"object",
        "properties":{
          "path":{"type":"string","minLength":1},
          "edits":{"type":"array","minItems":1,"items":{
            "type":"object",
            "properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},
            "required":["oldText","newText"]}},
          "mode":{"type":"string","enum":["fast","safe"]},
          "count":{"type":"integer","minimum":1,"maximum":9},
          "dry":{"type":["boolean","null"]}},
        "required":["path","edits"]}"#,
    );
    let parameters = parameters(&schema, false);
    let required = calls("edit", parameters.clone(), true, false);
    // The chat template renders {"path":"calc.py","mode":"safe","count":3,
    // "dry":null,"edits":[{"oldText":"a, {b}: \"c\"\n","newText":""}]} with
    // sorted, compact keys and raw strings.
    let rendered = call(
        "edit",
        &format!(
            "count:3,dry:null,edits:[{{newText:{},oldText:{}}}],mode:{},path:{}",
            quoted(""),
            quoted("a, {b}: \"c\"\n"),
            quoted("safe"),
            quoted("calc.py")
        ),
    );
    assert!(accepts(&required, &rendered), "template call: {rendered}");
    let empty = quoted("");
    for invalid in [
        // Missing a required argument, wrong types, enum and bounds.
        call("edit", &format!("path:{}", quoted("x"))),
        call("edit", &format!("edits:[],path:{}", quoted("x"))),
        call(
            "edit",
            &format!(
                "count:0,edits:[{{newText:{empty},oldText:{empty}}}],path:{}",
                quoted("x")
            ),
        ),
        call(
            "edit",
            &format!(
                "edits:[{{newText:{empty},oldText:{empty}}}],mode:{},path:{}",
                quoted("slow"),
                quoted("x")
            ),
        ),
        // Unsorted keys, JSON spelling, an undeclared tool and prose.
        call(
            "edit",
            &format!(
                "path:{},edits:[{{newText:{empty},oldText:{empty}}}]",
                quoted("x")
            ),
        ),
        call("edit", r#""edits":[],"path":"x""#),
        call("other", ""),
        "ordinary text".to_owned(),
    ] {
        assert!(!accepts(&required, &invalid), "rejected: {invalid}");
    }
    assert!(
        !accepts(&required, format!("{rendered}{rendered}")),
        "one call without parallel calls"
    );
    // A required call admits llama.cpp's bounded space before its marker,
    // never a stream of whitespace in place of the call.
    assert!(accepts(&required, format!("\n{rendered}")));
    assert!(accepts(
        &required,
        format!("\n\n{}{rendered}", " ".repeat(20))
    ));
    assert!(!accepts(
        &required,
        format!("{}{rendered}", "\n".repeat(21))
    ));
    assert!(!accepts(
        &required,
        format!("{}{rendered}", " ".repeat(100))
    ));
    let automatic = calls("edit", parameters, false, true);
    assert!(
        accepts(&automatic, format!("Let me fix it. {rendered}{rendered}")),
        "automatic calls keep prose and parallel calls"
    );
    assert!(accepts(&automatic, "No call needed."));
    // With thinking on, the output may open the thought channel first.
    let thinking =
        JsonConstraint::with_reasoning_markup(required, THOUGHT_END, THOUGHT_START).unwrap();
    assert!(accepts(&thinking, &rendered));
    assert!(accepts(
        &thinking,
        format!("{THOUGHT_START}Plan <|tool_call>.{THOUGHT_END}{rendered}")
    ));
    assert!(!accepts(&thinking, format!("Plan{rendered}")));
}

#[test]
fn gemma4_keys_and_names() {
    // Keys the parser cannot read bare are quoted.
    for key in ["a:b", " value ", "<x>"] {
        let mut unusual = Value::object();
        unusual.set("type", "object");
        let mut properties = Value::object();
        properties.set(key, Value::object().with("type", "string"));
        unusual.set("properties", properties);
        unusual.set("required", vec![Value::from(key)]);
        unusual.set("additionalProperties", false);
        let grammar = parameters(&unusual, true);
        let spelled = format!("{{{}:{}}}", quoted(key), quoted("v"));
        assert!(accepts(&grammar, &spelled), "quoted key: {key:?}");
    }
    // A key that must be quoted but holds the quote has no spelling: strict
    // fails, best effort keeps native open arguments.
    let quote = parse(
        r#"{"type":"object","properties":{"<|\"|>b":{"type":"string"}},
        "required":["<|\"|>b"],"additionalProperties":false}"#,
    );
    assert!(JsonConstraint::tool_parameters(&quote, true, ToolFormat::Gemma4).is_err());
    let open = parameters(&quote, false);
    assert!(accepts(&open, format!("{{x:1,y:{}}}", quoted("z"))));
    // Gemma 4 ends a function name at its first brace.
    let any = parameters(&parse(r#"{"type":"object","properties":{}}"#), false);
    assert!(
        JsonConstraint::with_tools(
            None,
            vec![("x{y".into(), any)],
            true,
            false,
            ToolFormat::Gemma4
        )
        .is_err()
    );
}

#[test]
fn gemma4_untyped_arrays_keep_their_siblings() {
    // An array without `items` holds any values in best effort; it must not
    // open the whole call, so its siblings' const, required keys and closed
    // object still hold (functional tool-streaming's record tool).
    let untyped = parse(
        r#"{"type":"object",
        "properties":{"text":{"type":"string","const":"a\\, b"},
          "values":{"type":"array"}},
        "required":["text","values"],"additionalProperties":false}"#,
    );
    let record = calls("record", parameters(&untyped, false), true, false);
    let valid = call(
        "record",
        &format!(
            "text:{},values:[true,null,{{key:{}}},1]",
            quoted("a\\, b"),
            quoted("value")
        ),
    );
    assert!(accepts(&record, &valid), "untyped array holds any values");
    for invalid in [
        format!("text:{},values:[]", quoted("a\\ b")),
        "values:[]".to_owned(),
        format!("text:{},values:[],other:1", quoted("a\\, b")),
    ] {
        assert!(
            !accepts(&record, call("record", &invalid)),
            "untyped array keeps its siblings' schema: {invalid}"
        );
    }
    assert!(
        JsonConstraint::tool_parameters(&untyped, true, ToolFormat::Gemma4).is_err(),
        "strict arrays still need an items schema"
    );
}

#[test]
fn reasoning_with_model_delimiters() {
    let plain = JsonConstraint::compile(
        &parse(
            r#"{"type":"object","properties":{"x":{"type":"integer"}},
            "required":["x"],"additionalProperties":false}"#,
        ),
        true,
    )
    .unwrap();
    let think = JsonConstraint::with_reasoning(plain.clone());
    // Another model's close marker (Gemma 4's thought channel).
    let channel = JsonConstraint::with_reasoning_markup(plain.clone(), THOUGHT_END, "").unwrap();
    assert!(!Arc::ptr_eq(&channel, &think));
    assert!(Arc::ptr_eq(
        &channel,
        &JsonConstraint::with_reasoning_markup(plain.clone(), THOUGHT_END, "").unwrap()
    ));
    assert!(Arc::ptr_eq(
        &think,
        &JsonConstraint::with_reasoning_markup(plain.clone(), "</think>", "").unwrap()
    ));
    assert!(accepts(&channel, "thought </think> <channel|>{\"x\":1}"));
    assert!(!accepts(&channel, "</think>{\"x\":1}"));
    // An opener makes the reasoning optional.
    let optional =
        JsonConstraint::with_reasoning_markup(plain.clone(), THOUGHT_END, THOUGHT_START).unwrap();
    assert!(accepts(&optional, "{\"x\":1}"));
    assert!(accepts(
        &optional,
        format!("{THOUGHT_START}plan{THOUGHT_END}{{\"x\":1}}")
    ));
    assert!(!accepts(&optional, format!("plan{THOUGHT_END}{{\"x\":1}}")));
    assert!(JsonConstraint::with_reasoning_markup(plain, "", "").is_err());
}
