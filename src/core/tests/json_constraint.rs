//! Port of `tests/core/json_constraint_test.cpp`.

#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Barrier, Mutex, MutexGuard, PoisonError};

use gufo_core::json::{Value, parse_str};
use gufo_core::json_constraint::{
    ConstraintVocabulary, JsonConstraint, Piece, SchemaError, State, TokenConstraint, ToolFormat,
};
use gufo_core::sampling::{SamplerState, SamplingConfig, TokenId};

/// The grammar caches are process-wide and bounded; identity checks must
/// not race another test's compilations.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

fn parse(text: &str) -> Value {
    parse_str(text).unwrap()
}

fn compile(schema: &str, strict: bool) -> Arc<JsonConstraint> {
    JsonConstraint::compile(&parse(schema), strict).unwrap()
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

fn prefix_allowed(grammar: &JsonConstraint, text: impl AsRef<[u8]>) -> bool {
    let mut state = grammar.start().unwrap();
    for &byte in text.as_ref() {
        state = grammar.advance(&state, byte).unwrap();
    }
    !state.is_empty()
}

fn vocabulary(pieces: &[&[u8]], stop: usize) -> Arc<ConstraintVocabulary> {
    Arc::new(
        ConstraintVocabulary::new(pieces.len() as u32, |i| {
            Ok(Piece {
                text: pieces[i as usize].to_vec(),
                stop: i as usize == stop,
            })
        })
        .unwrap(),
    )
}

/// A schema with the single required property `x`.
fn property(schema: &str) -> Arc<JsonConstraint> {
    compile(
        &format!(
            r#"{{"type":"object","properties":{{"x":{schema}}},"required":["x"],"additionalProperties":false}}"#
        ),
        true,
    )
}

fn x(value: &str) -> String {
    format!(r#"{{"x":{value}}}"#)
}

#[test]
fn json_language() {
    let _serial = serial();
    let grammar = JsonConstraint::object().unwrap();
    for valid in [
        "{}",
        r#"{"a":[null,true,false,0,-2,1.25,2e-3]}"#,
        r#"{"x":"\"\\\/\b\f\n\r\té😀"}"#,
        "{\"utf8\":\"é 中 😀\"}",
        " \n { \"nested\" : {\"a\":1} } \t",
    ] {
        assert!(accepts(&grammar, valid), "{valid}");
    }
    for invalid in [
        &b"[]"[..],
        b"null",
        b"{",
        b"{}{}",
        br#"{"a":01}"#,
        br#"{"a":1.}"#,
        br#"{"a":+1}"#,
        br#"{"a":NaN}"#,
        br#"{"a":[1,]}"#,
        br#"{"a":"\uDE00"}"#,
        br#"{"a":"\uD83Dx"}"#,
        br#"{"a":"\uD83D\u0000"}"#,
        b"{\"x\":\"\n\"}",
        b"{\"x\":\"\xc0\x80\"}",
        b"{\"x\":\"\xed\xa0\x80\"}",
    ] {
        assert!(
            !accepts(&grammar, invalid),
            "{}",
            String::from_utf8_lossy(invalid)
        );
    }
    assert!(accepts(&grammar, "\n".repeat(32) + "{}"));
    assert!(!accepts(&grammar, "\n".repeat(33) + "{}"));
    assert!(accepts(
        &grammar,
        format!("{{\"text\":\"{}\"}}", " ".repeat(100))
    ));
}

#[test]
fn schema_language() {
    let _serial = serial();
    let schema = parse(
        r#"{
    "type":"object","properties":{
      "name":{"type":"string","enum":["red","blue"]},
      "values":{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2},
      "optional":{"anyOf":[{"type":"boolean"},{"type":"null"}]}
    },"required":["name","values","optional"],"additionalProperties":false
  }"#,
    );
    let grammar = JsonConstraint::compile(&schema, true).unwrap();
    assert!(Arc::ptr_eq(
        &grammar,
        &JsonConstraint::compile(&schema, true).unwrap()
    ));
    for valid in [
        r#"{"name":"red","values":[1],"optional":null}"#,
        r#"{"name":"blue","values":[-1,2],"optional":true}"#,
    ] {
        assert!(accepts(&grammar, valid), "{valid}");
    }
    for invalid in [
        r#"{"name":"green","values":[1],"optional":null}"#,
        r#"{"name":"red","values":[],"optional":null}"#,
        r#"{"name":"red","values":[1,2,3],"optional":null}"#,
        r#"{"name":"red","values":[1.2],"optional":null}"#,
        r#"{"name":"red","values":[1]}"#,
        r#"{"name":"red","values":[1],"optional":null,"extra":1}"#,
    ] {
        assert!(!accepts(&grammar, invalid), "{invalid}");
    }
    let optional = compile(
        r#"{
    "type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},
    "additionalProperties":false}"#,
        false,
    );
    for text in ["{}", "{\"a\":1}", "{\"b\":true}", "{\"a\":1,\"b\":true}"] {
        assert!(accepts(&optional, text), "{text}");
    }
    assert!(!accepts(&optional, "{\"b\":true,}"));
    let unbounded = compile(
        r#"{
    "type":"object","properties":{"x":{"type":"array","items":{"type":"null"},"minItems":2}},
    "required":["x"],"additionalProperties":false}"#,
        true,
    );
    assert!(!accepts(&unbounded, "{\"x\":[null]}"));
    let many = format!("{{\"x\":[null{}]}}", ",null".repeat(299));
    assert!(accepts(&unbounded, many));
    let referenced = compile(
        r##"{
    "type":"object","properties":{"x":{"$ref":"#/$defs/Label"}},
    "required":["x"],"additionalProperties":false,
    "$defs":{"Label":{"type":["string","null"],"enum":["a",null]}}
  }"##,
        true,
    );
    assert!(accepts(&referenced, "{\"x\":null}"));
    assert!(accepts(&referenced, "{\"x\":\"a\"}"));
    assert!(!accepts(&referenced, "{\"x\":\"b\"}"));
}

#[test]
fn rejected_schemas() {
    let _serial = serial();
    for input in [
        r#"{"type":"array","items":{"type":"integer"}}"#,
        r#"{"type":"object","properties":{},"additionalProperties":true}"#,
        r#"{"type":"object","properties":{},"additionalProperties":false,"oneOf":[]}"#,
        r#"{"type":"object","properties":{"x":{"type":"string","pattern":"["}},"additionalProperties":false,"required":["x"]}"#,
        r#"{"type":"object","properties":{"x":{"type":"integer"}},"additionalProperties":false}"#,
        r#"{"type":"object","properties":{"x":{"type":"string","anyOf":[true,{"type":"string"}]}},"required":["x"],"additionalProperties":false}"#,
        r#"{"type":"object","properties":{"x":{"type":"string","anyOf":[{"anyOf":"invalid"},{"type":"string"}]}},"required":["x"],"additionalProperties":false}"#,
        r#"{"type":"object","properties":{"x":{"type":"array","items":{"type":"null"},"minItems":2,"maxItems":1}},"required":["x"],"additionalProperties":false}"#,
        r#"{"type":"object","properties":{"x":{"$ref":"https://example.invalid/schema"}},"required":["x"],"additionalProperties":false}"#,
        r##"{"type":"object","properties":{},"additionalProperties":false,"$defs":{"x":{"$ref":"#/$defs/x"}}}"##,
        r#"{"type":"object","properties":{},"additionalProperties":false,"$defs":{"x":{"type":"string","bad":1}}}"#,
    ] {
        let error = JsonConstraint::compile(&parse(input), true).unwrap_err();
        assert!(
            matches!(error, SchemaError::Empty(_) | SchemaError::Invalid(_)),
            "{input}: {error:?}"
        );
    }
    // The server keys its HTTP errors on the message.
    let empty = JsonConstraint::compile(
        &parse(
            r#"{"type":"object","properties":{"x":{"type":"array","items":{"type":"null"},"minItems":2,"maxItems":1}},"required":["x"],"additionalProperties":false}"#,
        ),
        true,
    )
    .unwrap_err();
    assert_eq!(
        empty,
        SchemaError::Empty("JSON Schema: schema has no finite value".into())
    );
    let unsupported = JsonConstraint::compile(
        &parse(r#"{"type":"object","properties":{},"additionalProperties":false,"oneOf":[]}"#),
        true,
    )
    .unwrap_err();
    assert_eq!(
        unsupported,
        SchemaError::Invalid("JSON Schema: unsupported keyword: oneOf".into())
    );
}

#[test]
fn schema_intersections() {
    let _serial = serial();
    let grammar = compile(
        r##"{
    "type":"object","properties":{
      "tag":{"$ref":"#/$defs/tag","pattern":"b","minLength":2,"maxLength":3},
      "n":{"$ref":"#/$defs/n","maximum":7},
      "record":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},
        "required":["a","b"],"additionalProperties":false,
        "enum":[{"b":true,"a":1},{"a":2,"b":false}]},
      "list":{"type":"array","items":{"type":"integer","minimum":0},
        "enum":[[1,2],[-1],[2]],"minItems":2}
    },"required":["tag","n","record","list"],"additionalProperties":false,
    "$defs":{"tag":{"type":"string","pattern":"^a","maxLength":8},
             "n":{"type":"integer","minimum":2,"maximum":10}}
  }"##,
        true,
    );
    assert!(accepts(
        &grammar,
        r#"{"tag":"ab","n":7,"record":{"a":1,"b":true},"list":[1, 2]}"#
    ));
    for text in [
        r#"{"tag":"ac","n":7,"record":{"a":1,"b":true},"list":[1,2]}"#,
        r#"{"tag":"ab","n":8,"record":{"a":1,"b":true},"list":[1,2]}"#,
        r#"{"tag":"ab","n":7,"record":{"a":1,"b":false},"list":[1,2]}"#,
        r#"{"tag":"ab","n":7,"record":{"a":1,"b":true},"list":[2]}"#,
    ] {
        assert!(!accepts(&grammar, text), "{text}");
    }
    let boundaries = compile(
        r#"{
    "type":"object","properties":{"x":{"type":"string","pattern":"\\bcat\\b"},
    "y":{"type":"string","pattern":"^\\u{1f600}\\cJ\\0$"}},
    "required":["x","y"],"additionalProperties":false}"#,
        true,
    );
    assert!(accepts(&boundaries, r#"{"x":"cat!","y":"😀\n\u0000"}"#));
    assert!(!accepts(&boundaries, r#"{"x":"cats","y":"😀\n\u0000"}"#));
    let any = compile(
        r#"{
    "type":"object","properties":{"x":{"anyOf":[
      {"type":"integer","minimum":1},{"type":"integer","minimum":4}],
      "maximum":5}},"required":["x"],"additionalProperties":false}"#,
        true,
    );
    assert!(accepts(&any, "{\"x\":5}"));
    assert!(!accepts(&any, "{\"x\":6}"));
    let partial = compile(
        r#"{
    "type":"object","properties":{"x":{"anyOf":[
      {"type":"integer","minimum":6},{"type":"integer","minimum":1}],
      "maximum":5}},"required":["x"],"additionalProperties":false}"#,
        true,
    );
    assert!(accepts(&partial, "{\"x\":2}"));
    assert!(!accepts(&partial, "{\"x\":6}"));
    let dead = compile(
        r#"{
    "type":"object","properties":{"x":{"anyOf":[
      {"type":"string","pattern":"^a$","minLength":2},
      {"type":"string","const":"b"}]}},
    "required":["x"],"additionalProperties":false}"#,
        true,
    );
    assert!(!prefix_allowed(&dead, "{\"x\":\"a"));
    assert!(accepts(&dead, "{\"x\":\"b\"}"));
    let multiples = compile(
        r##"{
    "type":"object","properties":{"x":{"$ref":"#/$defs/N","multipleOf":0.3}},
    "required":["x"],"additionalProperties":false,
    "$defs":{"N":{"type":"number","multipleOf":0.2}}}"##,
        true,
    );
    assert!(accepts(&multiples, r#"{"x":0.6}"#));
    assert!(accepts(&multiples, r#"{"x":-1.20}"#));
    assert!(!accepts(&multiples, r#"{"x":0.3}"#));
    assert!(!accepts(&multiples, r#"{"x":0.4}"#));
    let formats = compile(
        r##"{
    "type":"object","properties":{"x":{"$ref":"#/$defs/S","format":"hostname",
      "pattern":"^127"}},
    "required":["x"],"additionalProperties":false,
    "$defs":{"S":{"type":"string","format":"ipv4"}}}"##,
        true,
    );
    assert!(accepts(&formats, r#"{"x":"127.0.0.1"}"#));
    assert!(!accepts(&formats, r#"{"x":"192.168.1.1"}"#));
    assert!(!accepts(&formats, r#"{"x":"127.example"}"#));
}

#[test]
fn mixed_enum_values() {
    // An untyped member can hold a string in one enum value and a number in
    // another; each must stay reachable whatever the enum order.
    for values in [r#"[{"x":"a"},{"x":1}]"#, r#"[{"x":1},{"x":"a"}]"#] {
        let mut schema = parse(
            r#"{"type":"object","properties":{"x":{"anyOf":[
              {"type":"string"},{"type":"integer"}]}},"required":["x"],
              "additionalProperties":false}"#,
        );
        schema.set("enum", parse(values));
        let grammar = JsonConstraint::compile(&schema, true).unwrap();
        assert!(accepts(&grammar, r#"{"x":"a"}"#));
        assert!(accepts(&grammar, r#"{"x":1}"#));
        assert!(!accepts(&grammar, r#"{"x":2}"#));
    }
    let list = compile(
        r#"{"type":"object","properties":{"l":{"type":"array","items":{
          "anyOf":[{"type":"string"},{"type":"number"}]},"enum":[["a",1]]}},
          "required":["l"],"additionalProperties":false}"#,
        true,
    );
    assert!(accepts(&list, r#"{"l":["a",1]}"#));
}

/// Response formats follow llama.cpp: objects that declare properties stay
/// closed unless `additionalProperties` is `true` or a schema, and objects
/// without properties admit any keys. Tool schemas keep the closed contract.
#[test]
fn open_response_objects() {
    let _serial = serial();
    let response = |schema: &Value, strict: bool| JsonConstraint::compile_response(schema, strict);
    let person = parse(
        r#"{"type":"object","properties":{
          "name":{"type":"string"},
          "born":{"type":"integer","description":"birth year"}
        },"required":["name","born"]}"#,
    );
    // Compiling the open form first must not leak into the closed cache entry.
    let implicit = response(&person, true).unwrap();
    assert!(Arc::ptr_eq(&implicit, &response(&person, true).unwrap()));
    assert!(JsonConstraint::compile(&person, true).is_err());
    assert!(JsonConstraint::compile(&person, false).is_err());
    assert!(accepts(&implicit, r#"{"name":"Nostradamus","born":1503}"#));
    for invalid in [
        r#"{"name":"a","born":1,"occupation":"seer"}"#,
        r#"{"name":"a"}"#,
    ] {
        assert!(!accepts(&implicit, invalid), "{invalid}");
    }

    let mut extensible = person.clone();
    extensible.set("additionalProperties", Value::Bool(true));
    let open = response(&extensible, true).unwrap();
    for valid in [
        r#"{"name":"Nostradamus","born":1503}"#,
        r#"{"name":"a","born":1,"occupation":"seer"}"#,
        r#"{ "name" : "a" , "born" : 1 , "x" : [1,{"y":null}], "z" : true }"#,
    ] {
        assert!(accepts(&open, valid), "{valid}");
    }
    for invalid in [
        r#"{"name":"a"}"#,
        r#"{"name":"a","born":"1503"}"#,
        r#"{"name":"a","born":1,"name":"b"}"#,
        r#"{"name":"a","born":1,"born":2}"#,
        r#"{"name":"a","born":1,}"#,
        r#"{"name":"a","born":1,"x":}"#,
    ] {
        assert!(!accepts(&open, invalid), "{invalid}");
    }

    let nested = response(
        &parse(
            r#"{"type":"object","properties":{
              "items":{"type":"array","minItems":1,"items":{
                "type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}},
              "closed":{"type":"object","properties":{"a":{"type":"boolean"}},
                "required":["a"],"additionalProperties":false},
              "typed":{"type":"object","properties":{},"additionalProperties":{"type":"integer"}},
              "explicit":{"type":"object","properties":{},"additionalProperties":true},
              "any":{"type":"object"}
            },"required":["items","closed","typed","explicit","any"]}"#,
        ),
        true,
    )
    .unwrap();
    assert!(accepts(
        &nested,
        r#"{"items":[{"id":1}],"closed":{"a":true},"typed":{"n":2},"explicit":{"k":[]},"any":{"m":{}}}"#
    ));
    for invalid in [
        r#"{"items":[{"id":1,"note":"x"}],"closed":{"a":true},"typed":{},"explicit":{},"any":{}}"#,
        r#"{"items":[{"id":1}],"closed":{"a":true,"b":1},"typed":{},"explicit":{},"any":{}}"#,
        r#"{"items":[{"id":1}],"closed":{"a":true},"typed":{"n":"2"},"explicit":{},"any":{}}"#,
        r#"{"items":[{"note":"x"}],"closed":{"a":true},"typed":{},"explicit":{},"any":{}}"#,
        r#"{"items":[{"id":1}],"closed":{"a":true},"typed":{},"explicit":{},"any":{},"more":1}"#,
    ] {
        assert!(!accepts(&nested, invalid), "{invalid}");
    }

    // Closed schemas compile to the same language with or without the rule.
    let closed = parse(
        r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},
          "required":["a"],"additionalProperties":false}"#,
    );
    for text in [
        r#"{"a":1}"#,
        r#"{"a":1,"b":false}"#,
        r#"{"a":1,"c":0}"#,
        r#"{"b":true}"#,
    ] {
        assert_eq!(
            accepts(&JsonConstraint::compile(&closed, false).unwrap(), text),
            accepts(&response(&closed, false).unwrap(), text),
            "{text}"
        );
    }

    // Only undeclared keys open up: strict requirements, untyped leaves,
    // unsupported keywords and a non-object root remain errors.
    for (input, strict) in [
        (
            r#"{"type":"object","properties":{"a":{"type":"integer"}}}"#,
            true,
        ),
        (
            r#"{"type":"object","properties":{"a":{}},"required":["a"]}"#,
            false,
        ),
        (
            r#"{"type":"object","properties":{},"patternProperties":{"x":{"type":"integer"}}}"#,
            false,
        ),
        (r#"{"type":"array","items":{"type":"integer"}}"#, false),
    ] {
        assert!(response(&parse(input), strict).is_err(), "{input}");
    }
}

#[test]
fn primitive_constraints() {
    let _serial = serial();
    let pattern = property(r#"{"type":"string","pattern":"^@[a-zA-Z0-9_]+$","maxLength":6}"#);
    let positive = property(r#"{"type":"number","exclusiveMinimum":0}"#);
    let mut number_state = positive.start().unwrap();
    for byte in format!("{{\"x\":0.{}", "0".repeat(4093)).bytes() {
        number_state = positive.advance(&number_state, byte).unwrap();
    }
    assert!(!number_state.is_empty());
    assert!(positive.advance(&number_state, b'0').unwrap().is_empty());
    let last_digit = positive.advance(&number_state, b'1').unwrap();
    assert!(positive.complete(&positive.advance(&last_digit, b'}').unwrap()));
    for value in [r#""@abc""#, r#""@abc""#, r#""@x_42""#] {
        assert!(accepts(&pattern, x(value)), "{value}");
    }
    for value in [
        r#""abc""#,
        r#""@""#,
        r#""@abcde!""#,
        r#""@abcdef""#,
        r#""@a-b""#,
    ] {
        assert!(!accepts(&pattern, x(value)), "{value}");
    }
    let unicode = property(r#"{"type":"string","pattern":"^é😀$","minLength":2,"maxLength":2}"#);
    assert!(accepts(&unicode, r#"{"x":"é😀"}"#));
    assert!(!accepts(&unicode, r#"{"x":"é😀"}"#));
    assert!(accepts(&unicode, r#"{"x":"é😀"}"#));
    for prefix in [
        &br#"{"x":"\u"#[..],
        br#"{"x":"\u00e"#,
        "{\"x\":\"é\\uD83D\\uDE0".as_bytes(),
        b"{\"x\":\"\xc3\xa9\xf0\x9f",
    ] {
        assert!(
            prefix_allowed(&unicode, prefix),
            "{}",
            String::from_utf8_lossy(prefix)
        );
    }
    for prefix in [
        &br#"{"x":"\uD800"#[..],
        "{\"x\":\"é\\uD800".as_bytes(),
        "{\"x\":\"é\\uD83D\\uDF".as_bytes(),
        b"{\"x\":\"\xc3\xa9\xf0\x90",
    ] {
        assert!(
            !prefix_allowed(&unicode, prefix),
            "{}",
            String::from_utf8_lossy(prefix)
        );
    }
    assert!(!prefix_allowed(&pattern, r#"{"x":"@\uD800"#));
    let unanchored = property(r#"{"type":"string","pattern":"cat"}"#);
    assert!(accepts(&unanchored, r#"{"x":"😀 cat"}"#));
    let alternatives = property(r#"{"type":"string","pattern":"^a|😀$"}"#);
    assert!(accepts(&alternatives, r#"{"x":"😀"}"#));
    let word = property(r#"{"type":"string","pattern":"^[\\w]+$"}"#);
    assert!(accepts(&word, r#"{"x":"a_Z09"}"#));
    assert!(!accepts(&word, r#"{"x":"é"}"#));
    assert!(!accepts(&word, r#"{"x":"é"}"#));
    let digit = property(r#"{"type":"string","pattern":"^\\d+$"}"#);
    assert!(accepts(&digit, r#"{"x":"123"}"#));
    assert!(!accepts(&digit, r#"{"x":"١"}"#));
    let whitespace = property(r#"{"type":"string","pattern":"^\\s+$"}"#);
    assert!(accepts(&whitespace, r#"{"x":"﻿ \t"}"#));
    assert!(!accepts(&whitespace, r#"{"x":"\u0085"}"#));
    let dot = property(r#"{"type":"string","pattern":"^.$"}"#);
    assert!(accepts(&dot, r#"{"x":"\u000b"}"#));
    assert!(accepts(&dot, r#"{"x":"\u0085"}"#));
    assert!(!accepts(&dot, r#"{"x":"a\n"}"#));
    assert!(!accepts(&dot, r#"{"x":" "}"#));
    let any = property(r#"{"type":"string","pattern":"^[^]$"}"#);
    assert!(accepts(&any, r#"{"x":"\n"}"#));
    let empty = property(r#"{"type":"string","pattern":"^[]*$"}"#);
    assert!(accepts(&empty, r#"{"x":""}"#));
    assert!(!accepts(&empty, r#"{"x":"a"}"#));
    let spaces = property(r#"{"type":"string","pattern":"^[a b]+$"}"#);
    assert!(accepts(&spaces, r#"{"x":"a b"}"#));
    assert!(accepts(&spaces, r#"{"x":"a b"}"#));
    let lengths = property(r#"{"type":"string","minLength":2,"maxLength":2}"#);
    for text in [
        r#"{"x":"ab"}"#,
        r#"{"x":"é😀"}"#,
        r#"{"x":"é😀"}"#,
        r#"{"x":"\\\""}"#,
    ] {
        assert!(accepts(&lengths, text), "{text}");
    }
    for text in [
        r#"{"x":""}"#,
        r#"{"x":"a"}"#,
        r#"{"x":"abc"}"#,
        r#"{"x":"😀"}"#,
        r#"{"x":"é😀"}"#,
    ] {
        assert!(!accepts(&lengths, text), "{text}");
    }
    for text in [
        &br#"{"x":"\uDC00a"}"#[..],
        br#"{"x":"\uD800\u0000a"}"#,
        b"{\"x\":\"\xe0\x80\x80a\"}",
        b"{\"x\":\"\xed\xa0\x80a\"}",
        b"{\"x\":\"\xf4\x90\x80\x80a\"}",
    ] {
        assert!(
            !accepts(&lengths, text),
            "{}",
            String::from_utf8_lossy(text)
        );
    }
    assert!(accepts(&lengths, r#"{"x":"𐏿a"}"#));
    let bounded_alternatives =
        property(r#"{"type":"string","pattern":"^(?:(?:ab){2}|c)$","maxLength":3}"#);
    assert!(accepts(&bounded_alternatives, r#"{"x":"c"}"#));
    assert!(!prefix_allowed(&bounded_alternatives, r#"{"x":"a"#));
    assert!(!prefix_allowed(&bounded_alternatives, r#"{"x":"a"#));
    let lookahead =
        property(r#"{"type":"string","pattern":"^(?=a|c)(?:(?:ab){2}|c)$","maxLength":3}"#);
    assert!(accepts(&lookahead, r#"{"x":"c"}"#));
    assert!(!prefix_allowed(&lookahead, r#"{"x":"a"#));
    assert!(!prefix_allowed(&lookahead, r#"{"x":"a"#));
    let intersection =
        property(r#"{"type":"string","pattern":"^(?:999|127)\\.0\\.0\\.1$","format":"ipv4"}"#);
    assert!(accepts(&intersection, r#"{"x":"127.0.0.1"}"#));
    assert!(!prefix_allowed(&intersection, r#"{"x":"9"#));
    assert!(!prefix_allowed(&intersection, r#"{"x":"9"#));
    let negative = property(r#"{"type":"string","pattern":"^(?!.*bb)[ab]{1,4}$"}"#);
    assert!(accepts(&negative, r#"{"x":"aba"}"#));
    assert!(!prefix_allowed(&negative, r#"{"x":"abb"#));
    let letters = property(r#"{"type":"string","pattern":"^\\p{L}+$","maxLength":3}"#);
    assert!(accepts(&letters, r#"{"x":"é中a"}"#));
    assert!(accepts(&letters, r#"{"x":"é中a"}"#));
    assert!(!accepts(&letters, r#"{"x":"123"}"#));
    let large_repeat =
        property(r#"{"type":"string","pattern":"^(?:a{500}|b{1,500})$","maxLength":3}"#);
    assert!(accepts(&large_repeat, r#"{"x":"bbb"}"#));
    assert!(!prefix_allowed(&large_repeat, r#"{"x":"a"#));
    let bounded_repeat =
        property(r#"{"type":"string","pattern":"^(?:ab)+$","minLength":3,"maxLength":4}"#);
    assert!(accepts(&bounded_repeat, r#"{"x":"abab"}"#));
    assert!(!prefix_allowed(&bounded_repeat, r#"{"x":"ababa"#));
    let bounded_class =
        property(r#"{"type":"string","pattern":"^(?:[^a-c]{4}|x)$","maxLength":3}"#);
    assert!(accepts(&bounded_class, r#"{"x":"x"}"#));
    assert!(!prefix_allowed(&bounded_class, r#"{"x":"y"#));
    let repeated_unicode =
        property(r#"{"type":"string","pattern":"^(?:é😀)+$","minLength":3,"maxLength":4}"#);
    assert!(accepts(&repeated_unicode, r#"{"x":"é😀é😀"}"#));
    let number =
        property(r#"{"type":"number","minimum":-0.4,"exclusiveMaximum":0.5,"multipleOf":0.1}"#);
    for value in ["-0.4", "-0.3", "0", "0.3", "0.40"] {
        assert!(accepts(&number, x(value)), "{value}");
    }
    for value in ["-0.5", "0.31", "0.5", "0.40000000000000001"] {
        assert!(!accepts(&number, x(value)), "{value}");
    }
    let integer = property(r#"{"type":"integer","multipleOf":1.5,"minimum":-12,"maximum":12}"#);
    for value in -14..=14 {
        assert_eq!(
            accepts(&integer, x(&value.to_string())),
            (-12..=12).contains(&value) && value % 3 == 0,
            "{value}"
        );
    }
    for (format, valid, invalid) in [
        ("date", "2024-02-29", "2023-02-29"),
        (
            "date-time",
            "2026-09-26T23:30:00+02:00",
            "2026-09-31T23:30:00Z",
        ),
        ("time", "23:59:01.25Z", "24:00:00Z"),
        ("uuid", "12345678-1234-1234-1234-123456789abc", "1234"),
        ("ipv4", "192.168.1.89", "256.168.1.89"),
        ("ipv6", "2001:db8::1", "1:::2"),
        ("hostname", "example.com", "-example.com"),
        ("email", "name@example.com", "name@bad domain"),
        ("duration", "P1DT2H30M", "PT"),
    ] {
        let grammar = property(&format!(r#"{{"type":"string","format":"{format}"}}"#));
        assert!(accepts(&grammar, x(&format!("\"{valid}\""))), "{format}");
        assert!(!accepts(&grammar, x(&format!("\"{invalid}\""))), "{format}");
    }
}

#[test]
fn integer_bounds_and_recursion() {
    let _serial = serial();
    let scalar = |constraints: &str| property(&format!(r#"{{"type":"integer",{constraints}}}"#));
    for low in -12..=12 {
        for high in low..=12 {
            let grammar = scalar(&format!(r#""minimum":{low},"maximum":{high}"#));
            for number in -15..=15 {
                assert_eq!(
                    accepts(&grammar, x(&number.to_string())),
                    low <= number && number <= high,
                    "{low}..={high}: {number}"
                );
            }
        }
    }
    let exclusive = scalar(r#""exclusiveMinimum":-2,"exclusiveMaximum":3"#);
    for number in -4..=4 {
        assert_eq!(
            accepts(&exclusive, x(&number.to_string())),
            -2 < number && number < 3
        );
    }
    let fractional = scalar(r#""minimum":-1.2,"maximum":2.8"#);
    assert!(accepts(&fractional, "{\"x\":-1}"));
    assert!(accepts(&fractional, "{\"x\":2}"));
    assert!(!accepts(&fractional, "{\"x\":-2}"));
    assert!(!accepts(&fractional, "{\"x\":3}"));
    let large = scalar(r#""exclusiveMinimum":9007199254740992"#);
    assert!(!accepts(&large, "{\"x\":9007199254740992}"));
    assert!(accepts(&large, "{\"x\":9007199254740993}"));
    assert!(accepts(
        &large,
        "{\"x\":1000000000000000000000000000000000}"
    ));
    let negative = scalar(r#""exclusiveMaximum":-9007199254740992"#);
    assert!(!accepts(&negative, "{\"x\":-9007199254740992}"));
    assert!(accepts(&negative, "{\"x\":-9007199254740993}"));
    let tree = compile(
        r##"{
    "type":"object","properties":{
      "value":{"type":"integer","minimum":1,"maximum":5},
      "children":{"type":"array","items":{"$ref":"#"}}
    },"required":["value","children"],"additionalProperties":false
  }"##,
        true,
    );
    assert!(accepts(
        &tree,
        r#"{"value":1,"children":[{"value":5,"children":[]}]}"#
    ));
    assert!(!accepts(
        &tree,
        r#"{"value":1,"children":[{"value":6,"children":[]}]}"#
    ));
    let list = compile(
        r##"{
    "type":"object","properties":{"head":{"$ref":"#/$defs/node"}},
    "required":["head"],"additionalProperties":false,"$defs":{
      "node":{"anyOf":[{"type":"null"},{"type":"object",
        "properties":{"next":{"$ref":"#/$defs/node"}},
        "required":["next"],"additionalProperties":false}]}
    }
  }"##,
        true,
    );
    assert!(accepts(&list, r#"{"head":{"next":{"next":null}}}"#));
}

#[test]
fn token_masks() {
    let _serial = serial();
    let pieces: &[&[u8]] = &[
        b"{\"x\":",
        b"true",
        b"false",
        b"}",
        b"INVALID",
        b"",
        b"{\"x\":\"",
        b"\xe2\x94",
        b"\x8c",
        b"\"}",
        b"false}BAD",
        b"false}",
        b"<think>",
        b"{\"x\":[",
        b",",
        b"]}",
    ];
    let vocabulary = vocabulary(pieces, 5);
    let constraint = TokenConstraint::new(JsonConstraint::object().unwrap(), vocabulary.clone());
    let grammar = constraint.grammar().clone();
    let mut state = grammar.start().unwrap();
    let mask = constraint.allowed(&state).unwrap();
    assert!(mask[0] && mask[6] && !mask[4] && !mask[5] && !mask[12]);
    state = vocabulary.accept(&grammar, &state, 6).unwrap();
    assert!(constraint.allowed(&state).unwrap()[12]);
    assert!(constraint.allowed(&state).unwrap()[7]);
    state = vocabulary.accept(&grammar, &state, 7).unwrap();
    assert!(constraint.allowed(&state).unwrap()[8]);
    assert!(!constraint.allowed(&state).unwrap()[9]);
    state = vocabulary.accept(&grammar, &state, 8).unwrap();
    state = vocabulary.accept(&grammar, &state, 9).unwrap();
    let mask = constraint.allowed(&state).unwrap();
    for (i, &allowed) in mask.iter().enumerate() {
        assert_eq!(allowed, i == 5, "token {i}");
    }
    // A stop token keeps a complete state; other tokens need admitted bytes.
    assert_eq!(vocabulary.accept(&grammar, &state, 5).unwrap(), state);
    assert!(vocabulary.accept(&grammar, &state, 3).is_err());
    let start = grammar.start().unwrap();
    assert!(vocabulary.accept(&grammar, &start, 4).is_err());
}

#[test]
fn string_mask_cache() {
    let _serial = serial();
    let pieces: &[&[u8]] = &[
        b"{\"x\":\"",
        b"a",
        b"abcdef",
        b"\"}",
        b"\\u0061",
        b"\\uD83D\\uDE00",
        b"\xc2",
        b"\x80",
        b"a\"}",
        b"",
    ];
    let vocabulary = vocabulary(pieces, pieces.len() - 1);
    for pattern in [false, true] {
        let mut schema = parse(
            r#"{
      "type":"object","properties":{"x":{
        "type":"string","minLength":1,"maxLength":48}},
      "required":["x"],"additionalProperties":false}"#,
        );
        if pattern {
            schema
                .member_mut("properties")
                .member_mut("x")
                .set("pattern", "^[a-z]+$");
        }
        let constraint = TokenConstraint::new(
            JsonConstraint::compile(&schema, true).unwrap(),
            vocabulary.clone(),
        );
        let grammar = constraint.grammar().clone();
        let at = |count: usize| -> State {
            let mut state = vocabulary
                .accept(&grammar, &grammar.start().unwrap(), 0)
                .unwrap();
            for _ in 0..count {
                state = vocabulary.accept(&grammar, &state, 1).unwrap();
            }
            state
        };
        let (one, five) = (at(1), at(5));
        assert_ne!(one, five);
        assert!(Arc::ptr_eq(
            &constraint.allowed(&one).unwrap(),
            &constraint.allowed(&five).unwrap()
        ));
        let near = at(47);
        let mask = constraint.allowed(&near).unwrap();
        assert!(
            mask[1] && !mask[2] && mask[3] && mask[4],
            "pattern {pattern}"
        );
        assert_eq!(mask[5], !pattern);
        assert!(!mask[9]);
        let last = vocabulary.accept(&grammar, &near, 4).unwrap();
        let full = constraint.allowed(&last).unwrap();
        assert!(!full[1] && !full[2] && full[3] && !full[4]);
        if !pattern {
            let partial = vocabulary.accept(&grammar, &near, 6).unwrap();
            assert!(constraint.allowed(&partial).unwrap()[7]);
            assert!(!constraint.allowed(&partial).unwrap()[3]);
            let partial = vocabulary.accept(&grammar, &partial, 7).unwrap();
            assert_eq!(*constraint.allowed(&partial).unwrap(), *full);
        }
    }
    {
        let constraint = TokenConstraint::new(
            compile(
                r#"{
      "type":"object","properties":{"x":{"type":"string","minLength":48,
      "maxLength":96}},"required":["x"],"additionalProperties":false}"#,
                true,
            ),
            vocabulary.clone(),
        );
        let grammar = constraint.grammar().clone();
        let mut state = vocabulary
            .accept(&grammar, &grammar.start().unwrap(), 0)
            .unwrap();
        let empty = constraint.allowed(&state).unwrap();
        for _ in 0..16 {
            state = vocabulary.accept(&grammar, &state, 1).unwrap();
        }
        assert!(Arc::ptr_eq(&constraint.allowed(&state).unwrap(), &empty));
        assert!(!empty[3] && !empty[8]);
        for _ in 16..47 {
            state = vocabulary.accept(&grammar, &state, 1).unwrap();
        }
        let near = constraint.allowed(&state).unwrap();
        assert!(!Arc::ptr_eq(&near, &empty) && !near[3] && near[8]);
        state = vocabulary.accept(&grammar, &state, 1).unwrap();
        assert!(constraint.allowed(&state).unwrap()[3]);
    }
    // Character limits count Unicode scalars, not escaped bytes, and do not
    // impose the old 64 KiB serialized-prefix ceiling.
    let mut schema = parse(
        r#"{
    "type":"object","properties":{"x":{"type":"string","minLength":70000,
    "maxLength":70000}},"required":["x"],"additionalProperties":false}"#,
    );
    let grammar = JsonConstraint::compile(&schema, true).unwrap();
    let content = "a".repeat(69999) + "\\uD83D\\uDE00";
    assert!(accepts(&grammar, format!("{{\"x\":\"{content}\"}}")));
    assert!(!accepts(&grammar, format!("{{\"x\":\"{content}a\"}}")));
    // Large minLength must not require one allocated DFA frontier per
    // character.
    let x = schema.member_mut("properties").member_mut("x");
    x.set("pattern", "^(?:ab)+$");
    x.set("minLength", 1_048_575);
    x.set("maxLength", 1_048_576);
    let grammar = JsonConstraint::compile(&schema, true).unwrap();
    assert!(!grammar.start().unwrap().is_empty());
}

#[test]
fn unsupported_patterns() {
    let _serial = serial();
    for pattern in [r"(a)\1", "(?<=a)b", "(?i)a", "(?:(?=a)a)*", r"\_", r"\01"] {
        let mut schema = parse(
            r#"{
      "type":"object","properties":{"x":{"type":"string"}},
      "required":["x"],"additionalProperties":false}"#,
        );
        schema
            .member_mut("properties")
            .member_mut("x")
            .set("pattern", pattern);
        assert!(
            matches!(
                JsonConstraint::compile(&schema, true),
                Err(SchemaError::Invalid(_))
            ),
            "{pattern}"
        );
    }
}

/// Every request starts at the initial state: argument masks of a native
/// call must not evict its mask and force another full-vocabulary
/// traversal on a warm cache hit.
#[test]
fn native_request_mask_reuse() {
    let _serial = serial();
    let arguments = JsonConstraint::tool_parameters(
        &parse(
            r#"{
    "type":"object","properties":{"value":{"type":"string"}},
    "required":["value"],"additionalProperties":false}"#,
        ),
        false,
        ToolFormat::Qwen,
    )
    .unwrap()
    .unwrap();
    let grammar = JsonConstraint::with_tools(
        None,
        vec![("record".into(), arguments)],
        false,
        true,
        ToolFormat::Qwen,
    )
    .unwrap()
    .unwrap();
    let vocabulary = Arc::new(
        ConstraintVocabulary::new(129, |id| {
            Ok(Piece {
                text: if id == 128 {
                    Vec::new()
                } else {
                    vec![id as u8]
                },
                stop: id == 128,
            })
        })
        .unwrap(),
    );
    let binding = TokenConstraint::new(grammar.clone(), vocabulary);
    let initial = grammar.start().unwrap();
    let first = binding.allowed(&initial).unwrap();
    let mut state = initial.clone();
    for &byte in b"<tool_call>\n<function=record>\n<parameter=value>\nalpha\n\
                   </parameter>\n</function>\n</tool_call>"
    {
        state = grammar.advance(&state, byte).unwrap();
        assert!(!state.is_empty());
        binding.allowed(&state).unwrap();
    }
    assert!(grammar.complete(&state));
    assert!(Arc::ptr_eq(&binding.allowed(&initial).unwrap(), &first));
}

#[test]
fn reasoning_and_tool_constraints() {
    let _serial = serial();
    let plain = JsonConstraint::object().unwrap();
    let grammar = JsonConstraint::with_reasoning(plain.clone());
    assert!(Arc::ptr_eq(
        &grammar,
        &JsonConstraint::with_reasoning(plain.clone())
    ));
    assert!(accepts(
        &grammar,
        "Think freely <tool_call> and <think>.</think>{\"x\":1}"
    ));
    assert!(accepts(&grammar, "</think>{\"x\":\"</think>\"}"));
    assert!(!accepts(&grammar, "thinking only"));
    assert!(!accepts(&grammar, "</think>not JSON"));
    let binding = TokenConstraint::new(
        grammar.clone(),
        vocabulary(
            &[
                b"thinking",
                b"</thi",
                b"nk>{\"x\":",
                b"true}",
                b"nk>INVALID",
                b"",
            ],
            5,
        ),
    );
    let mask = binding.allowed(&grammar.start().unwrap()).unwrap();
    assert!(!mask[5]);
    let mut state = grammar.start().unwrap();
    for token in [0, 1] {
        state = binding
            .vocabulary()
            .accept(&grammar, &state, token)
            .unwrap();
    }
    let mask = binding.allowed(&state).unwrap();
    assert!(mask[2] && !mask[4] && !mask[5]);
    for token in [2, 3] {
        state = binding
            .vocabulary()
            .accept(&grammar, &state, token)
            .unwrap();
    }
    assert!(binding.allowed(&state).unwrap()[5]);
    let tool = compile(
        r#"{
    "type":"object","properties":{"value":{"type":"integer","minimum":1,"maximum":5}},
    "required":["value"],"additionalProperties":false}"#,
        true,
    );
    let mixed = JsonConstraint::with_tools(
        Some(plain.clone()),
        vec![
            ("score".into(), tool.clone()),
            ("rating".into(), tool.clone()),
            ("metadata".into(), plain.clone()),
        ],
        false,
        false,
        ToolFormat::Json,
    )
    .unwrap()
    .unwrap();
    assert!(accepts(&mixed, "{\"final\":true}"));
    assert!(accepts(
        &mixed,
        r#"<tool_call>{"name":"score","arguments":{"value":3}}</tool_call>"#
    ));
    assert!(!accepts(
        &mixed,
        r#"<tool_call>{"name":"score","arguments":{"value":6}}</tool_call>"#
    ));
    assert!(!accepts(
        &mixed,
        r#"<tool_call>{"name":"unknown","arguments":{"value":3}}</tool_call>"#
    ));
    assert!(accepts(
        &mixed,
        r#"<tool_call>{"name":"rating","arguments":{"value":4}}</tool_call>"#
    ));
    assert!(!accepts(
        &mixed,
        r#"<tool_call>{"name":"rating","arguments":{"value":0}}</tool_call>"#
    ));
    assert!(accepts(
        &mixed,
        r#"<tool_call>{"name":"metadata","arguments":{"extra":true}}</tool_call>"#
    ));
    let required = JsonConstraint::with_tools(
        Some(plain.clone()),
        vec![("score".into(), tool.clone())],
        true,
        false,
        ToolFormat::Json,
    )
    .unwrap()
    .unwrap();
    assert!(!accepts(&required, "{\"final\":true}"));
    assert!(accepts(
        &JsonConstraint::with_reasoning(required),
        r#"Use the scoring tool.</think><tool_call>{"name":"score","arguments":{"value":3}}</tool_call>"#
    ));
    let automatic = JsonConstraint::with_tools(
        None,
        vec![("score".into(), tool.clone())],
        false,
        true,
        ToolFormat::Json,
    )
    .unwrap()
    .unwrap();
    let call = r#"<tool_call>{"name":"score","arguments":{"value":3}}</tool_call>"#;
    assert!(accepts(&automatic, "Ordinary prose."));
    assert!(accepts(&automatic, format!("First: {call}\n{call}")));
    assert!(!accepts(
        &automatic,
        r#"First: <tool_call>{"name":"score","arguments":{"value":6}}</tool_call>"#
    ));
    assert!(!accepts(
        &automatic,
        r#"<tool_call>{"name":"other","arguments":{}}</tool_call>"#
    ));
    let single = JsonConstraint::with_tools(
        None,
        vec![("score".into(), tool.clone())],
        true,
        false,
        ToolFormat::Json,
    )
    .unwrap()
    .unwrap();
    assert!(accepts(&single, call));
    assert!(!accepts(&single, "Ordinary prose."));
    assert!(!accepts(&single, format!("{call}{call}")));
    // A native call header also ends the reasoning; quoted and fenced
    // mentions of it stay reasoning.
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let qwen = format == ToolFormat::Qwen;
        let arguments = JsonConstraint::tool_parameters(
            &parse(
                r#"{
      "type":"object","properties":{"value":{"type":"string","const":"42"}},
      "required":["value"],"additionalProperties":false}"#,
            ),
            true,
            format,
        )
        .unwrap()
        .unwrap();
        let native = if qwen {
            "<tool_call>\n<function=record>\n<parameter=value>\n42\n\
             </parameter>\n</function>\n</tool_call>"
        } else {
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n\
             <｜DSML｜parameter name=\"value\" string=\"true\">42\
             </｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        };
        for required in [false, true] {
            let native_grammar = JsonConstraint::with_reasoning(
                JsonConstraint::with_tools(
                    None,
                    vec![("record".into(), arguments.clone())],
                    required,
                    false,
                    format,
                )
                .unwrap()
                .unwrap(),
            );
            assert!(accepts(
                &native_grammar,
                format!("Read the file.\n\n{native}")
            ));
            assert!(accepts(
                &native_grammar,
                format!("A literal <tool_call> is data.</think>{native}")
            ));
            assert!(accepts(
                &native_grammar,
                format!("Example:\n```xml\n{native}\n```\nNow call it.</think>{native}")
            ));
            assert!(accepts(
                &native_grammar,
                format!("Example: ``{native}``.</think>{native}")
            ));
            let invalid = native.replacen("42", "43", 1);
            assert!(!accepts(
                &native_grammar,
                format!("Read the file.\n\n{invalid}")
            ));
            // The fast reasoning mask agrees with scalar grammar evaluation,
            // including partial and overlapping markers, quoted headers, and
            // tokens that cross from reasoning into invalid tool output.
            let (closed, closed_invalid, quoted) = (
                format!("</think>{native}"),
                format!("</think>{invalid}"),
                format!("``{native}``"),
            );
            let texts: [&str; 15] = [
                "word",
                "\n",
                "`",
                "~~~\n",
                "<",
                "</think>",
                "think>",
                "tool_call>\n<function=record>",
                native,
                &invalid,
                &closed,
                &closed_invalid,
                &quoted,
                "<tool_call>\n<tool_call>\n<function=",
                "",
            ];
            let pieces: Vec<(String, bool)> = texts
                .into_iter()
                .map(|text| (text.to_owned(), false))
                .chain([(String::new(), true)])
                .collect();
            let vocabulary = ConstraintVocabulary::new(pieces.len() as u32, |id| {
                let (text, stop) = &pieces[id as usize];
                Ok(Piece {
                    text: text.as_bytes().to_vec(),
                    stop: *stop,
                })
            })
            .unwrap();
            let (open_quote, after_call) =
                (format!("``{native}"), format!("Read the file.\n\n{native}"));
            let prefixes: [&str; 11] = [
                "",
                "Read the file.\n\n",
                "Example: `literal ",
                "```xml\n",
                "<",
                "</",
                "<tool_call>\n<",
                "~~~\n",
                &open_quote,
                "</think>",
                &after_call,
            ];
            for prefix in prefixes {
                let mut checked = native_grammar.start().unwrap();
                for &byte in prefix.as_bytes() {
                    checked = native_grammar.advance(&checked, byte).unwrap();
                }
                assert!(!checked.is_empty(), "{prefix:?}");
                let mask = vocabulary.allowed(&native_grammar, &checked).unwrap();
                for id in 0..pieces.len() as u32 {
                    assert_eq!(
                        mask[id as usize],
                        vocabulary.allows(&native_grammar, &checked, id).unwrap(),
                        "{format:?} required={required} {prefix:?} {:?}",
                        pieces[id as usize]
                    );
                }
            }
            assert_eq!(accepts(&native_grammar, "Reasoning only."), !required);
            let mut state = native_grammar.start().unwrap();
            for &byte in b"Read the file.\n\n" {
                state = native_grammar.advance(&state, byte).unwrap();
            }
            let checkpoint = state.clone();
            for &byte in native.as_bytes() {
                state = native_grammar.advance(&state, byte).unwrap();
            }
            assert!(native_grammar.complete(&state));
            state = checkpoint;
            for &byte in format!("</think>{native}").as_bytes() {
                state = native_grammar.advance(&state, byte).unwrap();
            }
            assert!(native_grammar.complete(&state));
        }
    }
    assert!(
        JsonConstraint::with_tools(
            Some(plain.clone()),
            Vec::new(),
            true,
            false,
            ToolFormat::Json
        )
        .unwrap()
        .is_some_and(|same| Arc::ptr_eq(&same, &plain)),
        "no tools leave the answer grammar"
    );
    let tool_vocabulary = vocabulary(
        &[
            b"Hello ",
            b"<tool_",
            b"call>{\"name\":\"score\",\"arguments\":{\"value\":",
            b"3}}</tool_call>",
            b"6}}</tool_call>",
            b"",
        ],
        5,
    );
    let mut state = automatic.start().unwrap();
    let allowed = tool_vocabulary.allowed(&automatic, &state).unwrap();
    assert!(allowed[0] && allowed[1] && allowed[5]);
    for token in [0, 1, 2] {
        state = tool_vocabulary.accept(&automatic, &state, token).unwrap();
    }
    let allowed = tool_vocabulary.allowed(&automatic, &state).unwrap();
    assert!(allowed[3] && !allowed[4] && !allowed[5]);
    let abandoned = state.clone();
    state = tool_vocabulary.accept(&automatic, &state, 3).unwrap();
    assert!(tool_vocabulary.allowed(&automatic, &state).unwrap()[5]);
    assert_eq!(
        tool_vocabulary.allowed(&automatic, &abandoned).unwrap(),
        allowed
    );

    let ready = Barrier::new(8);
    let concurrent: Vec<Arc<JsonConstraint>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    ready.wait();
                    let tools = JsonConstraint::with_tools(
                        Some(plain.clone()),
                        vec![("concurrent".into(), tool.clone())],
                        true,
                        false,
                        ToolFormat::Json,
                    )
                    .unwrap()
                    .unwrap();
                    JsonConstraint::with_reasoning(tools)
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    for grammar in &concurrent {
        assert!(Arc::ptr_eq(grammar, &concurrent[0]));
        assert!(accepts(
            grammar,
            r#"Think.</think><tool_call>{"name":"concurrent","arguments":{"value":3}}</tool_call>"#
        ));
    }
}

#[test]
fn prompts_describe_the_constraint() {
    let _serial = serial();
    assert_eq!(
        JsonConstraint::object().unwrap().prompt(),
        "Respond with a single valid JSON object."
    );
    let schema = r#"{"type":"object","properties":{"a":{"type":"number","minimum":1e2}},"additionalProperties":false}"#;
    assert_eq!(
        compile(schema, false).prompt(),
        "Respond with a single JSON object matching this JSON Schema:\n\
         {\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\",\"minimum\":100}},\"additionalProperties\":false}"
    );
}

/// The sampler halves of `TestTokensAndSampling` and
/// `TestReasoningConstraint`: masks apply before every filter, tentative
/// copies keep their own grammar state, and greedy constrained sampling
/// matches a pre-masked control with every penalty.
#[test]
fn constrained_sampling() {
    let _serial = serial();
    // The C++ incomplete constraint (no grammar or vocabulary) cannot be
    // built: both are required, and an empty vocabulary is refused.
    assert!(
        ConstraintVocabulary::new(0, |_| Ok(Piece {
            text: Vec::new(),
            stop: false,
        }))
        .is_err()
    );
    let pieces: &[&[u8]] = &[
        b"{\"x\":",
        b"true",
        b"false",
        b"}",
        b"INVALID",
        b"",
        b"{\"x\":\"",
        b"\xe2\x94",
        b"\x8c",
        b"\"}",
        b"false}BAD",
        b"false}",
        b"<think>",
        b"{\"x\":[",
        b",",
        b"]}",
    ];
    let vocabulary = vocabulary(pieces, 5);
    let constraint = Arc::new(TokenConstraint::new(
        JsonConstraint::object().unwrap(),
        vocabulary.clone(),
    ));
    let grammar = constraint.grammar().clone();
    let constrained = |config: SamplingConfig| SamplingConfig {
        constraint: Some(constraint.clone()),
        ..config
    };
    let mut sampler = SamplerState::new(
        constrained(SamplingConfig {
            temperature: 0.5,
            top_p: 0.8,
            seed: 42,
            ..SamplingConfig::default()
        }),
        &[],
    )
    .unwrap();
    sampler.accept(0).unwrap();
    let mut logits = vec![f32::NEG_INFINITY; pieces.len()];
    logits[1] = 0.0;
    logits[2] = -1.0;
    // An invalid high-logit token must not affect the top-p support, nor
    // a valid prefix with an invalid suffix in the same token.
    logits[4] = 100.0;
    logits[10] = 101.0;
    let distribution = sampler.distribution(&logits).unwrap();
    assert_eq!(distribution.entries().len(), 1);
    assert_eq!(distribution.probability(1), 1.0);
    assert_eq!(distribution.probability(4), 0.0);
    let before = sampler.clone();
    let mut tentative = sampler.clone();
    tentative.accept(1).unwrap();
    tentative.accept(3).unwrap();
    logits[5] = 0.0;
    assert_eq!(tentative.sample(&logits).unwrap(), 5);
    assert_eq!(before.distribution(&logits).unwrap().probability(1), 1.0);
    assert_eq!(sampler.sample_residual(&logits, &[4], &[1.0]).unwrap(), 1);
    assert!(!sampler.config().can_use_unmodified_argmax());
    let mut greedy = SamplerState::new(constrained(SamplingConfig::default()), &[]).unwrap();
    greedy.accept(0).unwrap();
    assert_eq!(greedy.sample(&logits).unwrap(), 1);
    assert!(!greedy.config().can_use_unmodified_argmax());
    assert!(
        greedy
            .without_constraint()
            .config()
            .can_use_unmodified_argmax()
    );

    // An independent pre-masked control for the allocation-free constrained
    // argmax: negative logits, ties, excluded nonfinite values, penalties.
    for penalties in [false, true] {
        let base = SamplingConfig {
            repeat_penalty: if penalties { 1.2 } else { 1.0 },
            frequency_penalty: if penalties { 0.3 } else { 0.0 },
            presence_penalty: if penalties { 0.5 } else { 0.0 },
            ..SamplingConfig::default()
        };
        let generated: [TokenId; 3] = [13, 1, 14];
        let mut subject = SamplerState::new(constrained(base.clone()), &[1, 1]).unwrap();
        subject.accept_tokens(&generated).unwrap();
        let mut control = SamplerState::new(base, &[1, 1]).unwrap();
        control.accept_tokens(&generated).unwrap();
        let mut state: State = grammar.start().unwrap();
        for token in generated {
            state = vocabulary.accept(&grammar, &state, token).unwrap();
        }
        let allowed = constraint.allowed(&state).unwrap();
        for score in [-3.0f32, 0.0, 1.0] {
            let mut logits = vec![f32::INFINITY; pieces.len()];
            logits[1] = score;
            logits[2] = score;
            logits[4] = f32::NAN;
            let masked: Vec<f32> = logits
                .iter()
                .zip(allowed.iter())
                .map(|(&logit, &allowed)| if allowed { logit } else { f32::NEG_INFINITY })
                .collect();
            let expected = control.sample(&masked).unwrap();
            assert_eq!(subject.sample(&logits).unwrap(), expected);
            assert_eq!(
                subject.distribution(&logits).unwrap().probability(expected),
                1.0
            );
        }
    }
    let nothing = vec![f32::NEG_INFINITY; pieces.len()];
    assert!(sampler.sample(&nothing).is_err());
    assert!(greedy.sample(&nothing).is_err());

    // Reasoning precedes the JSON answer and admits no stop until it ends.
    let pieces: &[&[u8]] = &[
        b"thinking",
        b"</thi",
        b"nk>{\"x\":",
        b"true}",
        b"nk>INVALID",
        b"",
    ];
    let reasoning = JsonConstraint::with_reasoning(JsonConstraint::object().unwrap());
    let binding = Arc::new(TokenConstraint::new(
        reasoning.clone(),
        self::vocabulary(pieces, 5),
    ));
    let mut sampler = SamplerState::new(
        SamplingConfig {
            constraint: Some(binding.clone()),
            ..SamplingConfig::default()
        },
        &[],
    )
    .unwrap();
    sampler.accept(0).unwrap();
    sampler.accept(1).unwrap();
    let mut tentative = sampler.clone();
    tentative.accept(2).unwrap();
    tentative.accept(3).unwrap();
    assert!(!binding.allowed(&reasoning.start().unwrap()).unwrap()[5]);
    let mut logits = vec![f32::NEG_INFINITY; pieces.len()];
    logits[4] = 100.0;
    logits[2] = 0.0;
    assert_eq!(sampler.sample(&logits).unwrap(), 2);
    logits[5] = 101.0;
    assert_eq!(tentative.sample(&logits).unwrap(), 5);
}

/// Enum literals spell integers as decimals, so the model is never forced
/// into 1e+05.
#[test]
fn integer_enum_literals_are_decimal() {
    let mut large = parse(
        r#"{"type":"object","properties":{"x":{"type":"integer"}},
          "required":["x"],"additionalProperties":false}"#,
    );
    large.set("enum", parse(r#"[{"x":100000}]"#));
    let grammar = JsonConstraint::compile(&large, true).unwrap();
    assert!(accepts(&grammar, r#"{"x":100000}"#));
}

/// Wide native argument lists keep any order, one use per name and the
/// required members, across the old ten-property cutoff and the bitset word
/// boundary (`TestWideNativeArgumentOrder`).
#[test]
fn wide_native_argument_order() {
    let _serial = serial();
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let parameter = |i: usize, value: &str| match format {
            ToolFormat::Qwen => format!("<parameter=p{i}>\n{value}\n</parameter>\n"),
            _ => format!(
                "<｜DSML｜parameter name=\"p{i}\" string=\"false\">{value}</｜DSML｜parameter>\n"
            ),
        };
        let p = |i: usize| parameter(i, "1");
        for count in [10usize, 11, 12, 16, 24, 65] {
            let properties: Vec<String> = (0..count)
                .map(|i| format!(r#""p{i}":{{"type":"integer","minimum":1,"maximum":9}}"#))
                .collect();
            let schema = |required: &str| {
                parse(&format!(
                    r#"{{"type":"object","properties":{{{}}},"required":[{required}],"additionalProperties":false}}"#,
                    properties.join(",")
                ))
            };
            let compile = |schema: &Value, strict: bool| {
                JsonConstraint::tool_parameters(schema, strict, format)
                    .unwrap()
                    .unwrap()
            };
            let grammar = compile(&schema(r#""p0""#), false);
            let case = format!("{format:?} {count}");
            assert!(accepts(&grammar, p(0) + &p(count - 1) + &p(4)), "{case}");
            assert!(accepts(&grammar, p(count - 1) + &p(0)), "{case}");
            assert!(!accepts(&grammar, p(count - 1)), "{case}: required p0");
            assert!(!accepts(&grammar, p(0) + &p(0)), "{case}: repeated name");
            assert!(!accepts(&grammar, p(0) + &p(count)), "{case}: unknown name");
            assert!(
                !accepts(&grammar, p(0) + &parameter(4, "10")),
                "{case}: bound"
            );
            let optional = compile(&schema(""), false);
            assert!(accepts(&optional, ""), "{case}");
            assert!(accepts(&optional, p(count - 1) + &p(4)), "{case}");
            assert!(!accepts(&optional, p(4) + &p(4)), "{case}");
            // Many required fields stay unordered without 2^N rules.
            let names: Vec<String> = (0..count).rev().map(|i| format!(r#""p{i}""#)).collect();
            let reverse: String = (0..count).rev().map(p).collect();
            let required = compile(&schema(&names.join(",")), true);
            assert!(
                accepts(&required, &reverse),
                "{case}: all required, reversed"
            );
            assert!(
                !accepts(&required, &reverse[..reverse.len() - p(0).len()]),
                "{case}: a required member missing"
            );
        }
    }
}
