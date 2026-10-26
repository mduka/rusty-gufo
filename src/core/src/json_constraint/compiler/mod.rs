//! The JSON Schema to byte-grammar compiler (`JsonConstraintCompiler`).
//!
//! Schemas are visited by address, so a recursive `$ref` becomes a
//! recursive rule; schemas derived by intersecting a reference with its
//! siblings live in reference-counted storage whose addresses stay fixed
//! until compilation ends.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use super::lexeme::{self, JsonSchemaLexeme, dump};
use super::{
    ARGUMENTS, ArgumentSet, ByteClass, JsonConstraint, LEAF, LEXEME, MAX_DEPTH, MAX_RULES, Rule,
    SchemaError, Sequence, TERMINAL, ToolFormat,
};
use crate::json::Value;

mod containers;
mod rules;
mod schema;
mod tools;
mod values;

use values::*;

static EMPTY_OBJECT: Value = Value::Object(Vec::new());

fn invalid(reason: &str) -> SchemaError {
    SchemaError::invalid(reason)
}

/// The identity of a schema value while compiling.
fn address(value: &Value) -> usize {
    std::ptr::from_ref(value).addr()
}

/// A literal byte: the first 256 classes are single bytes.
fn byte(value: u8) -> u32 {
    TERMINAL | u32::from(value)
}

/// Whether an unsupported applicator admits fields outside `properties`
/// (`HasPropertySchemas`): ignoring it must not close a best-effort tool.
pub(super) fn has_property_schemas(schema: &Value) -> bool {
    [
        "patternProperties",
        "dependentSchemas",
        "dependencies",
        "if",
        "then",
        "else",
        "oneOf",
        "allOf",
    ]
    .iter()
    .any(|name| schema.find(name).is_some_and(|value| !value.is_empty()))
}

/// `AdmitsExtraProperties`.
fn admits_extra_properties(schema: &Value) -> bool {
    has_property_schemas(schema)
        || schema
            .find("unevaluatedProperties")
            .is_some_and(|unevaluated| !unevaluated.is_bool() || unevaluated.as_bool(false))
}

/// Whether an object admits no undeclared field (`ClosedProperties`).
pub(super) fn closed_properties(schema: &Value, best_effort: bool) -> bool {
    if best_effort && admits_extra_properties(schema) {
        return false;
    }
    if let Some(extra) = schema.find("additionalProperties") {
        return extra.is_bool() && !extra.as_bool(true);
    }
    best_effort
        && schema
            .find("unevaluatedProperties")
            .is_some_and(|unevaluated| unevaluated.is_bool() && !unevaluated.as_bool(true))
}

/// `text` as a regex matching it literally (`RegexLiteral`).
fn regex_literal(text: &str) -> String {
    let mut result = String::new();
    for c in text.chars() {
        if u32::from(c) < 0x20 {
            result.push_str(&format!("\\x{:02x}", u32::from(c)));
        } else {
            if "^$\\.*+?()[]{}|/".contains(c) {
                result.push('\\');
            }
            result.push(c);
        }
    }
    result
}

/// Every name of `properties` excluded, ahead of `pattern` without its `^`.
fn exclude_names(properties: &Value, pattern: &str) -> String {
    let names: Vec<String> = properties
        .members()
        .iter()
        .map(|(name, _)| regex_literal(name))
        .collect();
    format!("^(?!(?:{})$){}", names.join("|"), pattern)
}

/// Resolves a local JSON pointer against `root` (`ResolveReference`).
pub(crate) fn resolve_reference<'v>(
    root: &'v Value,
    reference: &Value,
) -> Result<&'v Value, SchemaError> {
    let Value::String(pointer) = reference else {
        return Err(invalid("only local JSON pointer references are supported"));
    };
    if pointer != "#" && !pointer.starts_with("#/") {
        return Err(invalid("only local JSON pointer references are supported"));
    }
    let mut target = root;
    let mut rest = &pointer[1..];
    while !rest.is_empty() {
        rest = &rest[1..];
        let slash = rest.find('/');
        let segment = &rest[..slash.unwrap_or(rest.len())];
        let mut decoded = String::new();
        let mut chars = segment.chars();
        while let Some(c) = chars.next() {
            if c == '~' {
                match chars.next() {
                    Some('0') => decoded.push('~'),
                    Some('1') => decoded.push('/'),
                    _ => return Err(invalid("invalid JSON pointer escape")),
                }
            } else {
                decoded.push(c);
            }
        }
        let next = if let Value::Array(items) = target {
            let digits = !decoded.is_empty() && decoded.bytes().all(|b| b.is_ascii_digit());
            digits
                .then(|| decoded.parse::<usize>().ok())
                .flatten()
                .and_then(|index| items.get(index))
        } else {
            target.find(&decoded)
        };
        target = next.ok_or_else(|| invalid("local reference does not exist"))?;
        let Some(slash) = slash else {
            break;
        };
        rest = &rest[slash..];
    }
    Ok(target)
}

/// The JSON value kinds a parameter admits, one bit each in the order of
/// [`VALUE_TYPE_NAMES`].
type ValueTypes = u8;
const VALUE_TYPE_NAMES: [&str; 6] = ["string", "number", "boolean", "null", "array", "object"];
const ALL_VALUE_TYPES: ValueTypes = 0b11_1111;
const STRING_TYPE: ValueTypes = 1;

fn matches_type(value: &Value, ty: &str) -> bool {
    match ty {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_bool(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "integer" => value.is_number() && value.as_double(0.0).floor() == value.as_double(0.0),
        _ => false,
    }
}

/// Structural equality with numbers compared by value (`EqualValue`).
fn equal_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x == y,
        (Value::Object(x), Value::Object(_)) => {
            a.len() == b.len()
                && x.iter()
                    .all(|(key, value)| b.find(key).is_some_and(|other| equal_value(value, other)))
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| equal_value(p, q))
        }
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        _ => false,
    }
}

/// `schema` without `keys`, in order (`Without`).
fn without(schema: &Value, keys: &[&str]) -> Value {
    let mut result = Value::object();
    for (key, value) in schema.members() {
        if !keys.contains(&key.as_str()) {
            result.append_member(key.clone(), value.clone());
        }
    }
    result
}

/// An integer bound as sign and decimal magnitude.
#[derive(Debug, Clone)]
struct IntegerBound {
    negative: bool,
    digits: Vec<u8>,
}

pub(super) struct Compiler<'a> {
    schema: &'a Value,
    strict: bool,
    grammar: JsonConstraint,
    ws: u32,
    string: u32,
    integer: u32,
    number: u32,
    boolean: u32,
    null: u32,
    generic_values: BTreeMap<usize, u32>,
    compiled: HashMap<usize, u32>,
    derived: Vec<Rc<Value>>,
    /// Per schema and kind (string or number): an untyped schema can admit
    /// both.
    value_checks: HashMap<(usize, bool), Arc<JsonSchemaLexeme>>,
    properties: usize,
    enum_values: usize,
    string_characters: usize,
    /// Non-strict tool parameters: objects without
    /// `additionalProperties: false` stay open and untyped leaves admit any
    /// value.
    open_objects: bool,
    /// Response formats, as in llama.cpp: an object admits undeclared keys
    /// when `additionalProperties` is `true` or a schema, or when it declares
    /// no properties. Untyped leaves and unsupported keywords stay errors.
    open_response_objects: bool,
    /// Best-effort tool parameters: unsupported keywords are guidance.
    ignore_unknown_keys: bool,
}

impl<'a> Compiler<'a> {
    /// With `tool`, a non-strict schema compiles as tool parameters, whose
    /// nested objects may stay open.
    pub(super) fn new(schema: &'a Value, strict: bool, tool: bool) -> Result<Self, SchemaError> {
        let mut grammar = JsonConstraint::empty();
        // Literal bytes occupy the first 256 terminal classes.
        for i in 0..=255u8 {
            let mut bits = ByteClass::default();
            bits.set(i);
            grammar.classes.push(bits);
        }
        let mut c = Self {
            schema,
            strict,
            grammar,
            ws: 0,
            string: 0,
            integer: 0,
            number: 0,
            boolean: 0,
            null: 0,
            generic_values: BTreeMap::new(),
            compiled: HashMap::new(),
            derived: Vec::new(),
            value_checks: HashMap::new(),
            properties: 0,
            enum_values: 0,
            string_characters: 0,
            open_objects: tool && !strict,
            open_response_objects: false,
            ignore_unknown_keys: false,
        };
        let whitespace = c.lexeme(JsonSchemaLexeme::whitespace());
        c.ws = c.optional(whitespace)?;
        let digits = c.class(b"0123456789");
        let hex = c.class(b"0123456789abcdefABCDEF");
        let hex_tail = c.seq(vec![hex, hex])?;
        let unicode = {
            let first = c.class(b"0123456789abcefABCEF");
            let plain = c.seq(vec![first, hex, hex, hex])?;
            let d = c.class(b"dD");
            let low = c.class(b"01234567");
            let below_surrogates = c.seq(vec![d, low, hex_tail])?;
            let d = c.class(b"dD");
            let high = c.class(b"89abAB");
            let escape = c.literal(b"\\u")?;
            let d2 = c.class(b"dD");
            let trail = c.class(b"cdefCDEF");
            let pair = c.seq(vec![d, high, hex_tail, escape, d2, trail, hex_tail])?;
            c.alt(vec![plain, below_surrogates, pair])?
        };
        let continuation = c.range(0x80, 0xbf);
        let character = {
            let mut choices = vec![
                c.range(0x20, 0x21),
                c.range(0x23, 0x5b),
                c.range(0x5d, 0x7f),
            ];
            let lead = c.range(0xc2, 0xdf);
            choices.push(c.seq(vec![lead, continuation])?);
            let second = c.range(0xa0, 0xbf);
            choices.push(c.seq(vec![byte(0xe0), second, continuation])?);
            let lead = c.range(0xe1, 0xec);
            choices.push(c.seq(vec![lead, continuation, continuation])?);
            let second = c.range(0x80, 0x9f);
            choices.push(c.seq(vec![byte(0xed), second, continuation])?);
            let lead = c.range(0xee, 0xef);
            choices.push(c.seq(vec![lead, continuation, continuation])?);
            let second = c.range(0x90, 0xbf);
            choices.push(c.seq(vec![byte(0xf0), second, continuation, continuation])?);
            let lead = c.range(0xf1, 0xf3);
            choices.push(c.seq(vec![lead, continuation, continuation, continuation])?);
            let second = c.range(0x80, 0x8f);
            choices.push(c.seq(vec![byte(0xf4), second, continuation, continuation])?);
            let escaped = c.class(b"\"\\/bfnrt");
            choices.push(c.seq(vec![byte(b'\\'), escaped])?);
            let escape = c.literal(b"\\u")?;
            choices.push(c.seq(vec![escape, unicode])?);
            c.alt(choices)?
        };
        let characters = c.repeat(character)?;
        c.string = c.seq(vec![byte(b'"'), characters, byte(b'"')])?;
        let positive = {
            let first = c.range(b'1', b'9');
            let rest = c.repeat(digits)?;
            let many = c.seq(vec![first, rest])?;
            c.alt(vec![byte(b'0'), many])?
        };
        let minus = c.optional(byte(b'-'))?;
        c.integer = c.seq(vec![minus, positive])?;
        let fraction = {
            let rest = c.repeat(digits)?;
            c.seq(vec![byte(b'.'), digits, rest])?
        };
        let exponent = {
            let e = c.class(b"eE");
            let sign = c.class(b"+-");
            let sign = c.optional(sign)?;
            let rest = c.repeat(digits)?;
            c.seq(vec![e, sign, digits, rest])?
        };
        let fraction = c.optional(fraction)?;
        let exponent = c.optional(exponent)?;
        c.number = c.seq(vec![c.integer, fraction, exponent])?;
        let truth = c.literal(b"true")?;
        let falsity = c.literal(b"false")?;
        c.boolean = c.alt(vec![truth, falsity])?;
        c.null = c.literal(b"null")?;
        Ok(c)
    }

    /// Compiles a response format: objects follow llama.cpp's open rules
    /// (`open_response_objects`).
    pub(super) fn with_open_response_objects(mut self) -> Self {
        self.open_response_objects = true;
        self
    }

    /// With `ignore_unknown_keys`, open tool parameters treat unsupported
    /// keywords as guidance.
    pub(super) fn compile(
        mut self,
        object_only: bool,
        ignore_unknown_keys: bool,
    ) -> Result<JsonConstraint, SchemaError> {
        self.ignore_unknown_keys = ignore_unknown_keys && self.open_objects;
        if object_only {
            let object = self.generic_object(MAX_DEPTH)?;
            self.grammar.root = self.seq(vec![self.ws, object, self.ws])?;
            self.grammar.prompt = "Respond with a single valid JSON object.".to_owned();
        } else {
            let mut root: &'a Value = self.schema;
            let mut roots: HashSet<usize> = HashSet::new();
            while let Some(reference) = root.find("$ref") {
                if !roots.insert(address(root)) {
                    return Err(invalid("root reference cycle"));
                }
                root = self.reference(reference)?;
            }
            if !root.is_object()
                || root.member_str("type", "") != "object"
                || root.contains("anyOf")
            {
                return Err(invalid("the root must have type object"));
            }
            let body = self.visit(self.schema, 0)?;
            self.grammar.root = self.seq(vec![self.ws, body, self.ws])?;
            self.grammar.prompt = format!(
                "Respond with a single JSON object matching this JSON Schema:\n{}",
                dump(self.schema)?
            );
        }
        self.finish()
    }

    /// Checks productivity and left recursion, prunes impossible
    /// alternatives and starts the grammar (`Finish`). Native parameters need
    /// the same check: an impossible required field could admit its tag and
    /// then dead-end.
    fn finish(mut self) -> Result<JsonConstraint, SchemaError> {
        let (rules, sets) = (&mut self.grammar.rules, &self.grammar.argument_sets);
        // A recursive schema must have a finite witness. References that
        // recurse without first consuming input are invalid for a predictive
        // grammar.
        let mut productive = vec![false; rules.len()];
        let mut nullable = vec![false; rules.len()];
        // A wide argument set is productive when its required members are,
        // and nullable when it requires none.
        let is_productive = |productive: &[bool], symbol: u32| {
            if symbol & LEAF != 0 {
                return true;
            }
            if symbol & ARGUMENTS != 0 {
                let set = &sets[(symbol & !ARGUMENTS) as usize];
                return set
                    .members
                    .iter()
                    .enumerate()
                    .all(|(i, &member)| !set.is_required(i) || productive[member as usize]);
            }
            productive[symbol as usize]
        };
        let is_nullable = |nullable: &[bool], symbol: u32| {
            if symbol & LEAF != 0 {
                return false;
            }
            if symbol & ARGUMENTS != 0 {
                return sets[(symbol & !ARGUMENTS) as usize].nullable();
            }
            nullable[symbol as usize]
        };
        let mut changed = true;
        while changed {
            changed = false;
            for id in 0..rules.len() {
                for sequence in &rules[id] {
                    let p = sequence
                        .iter()
                        .all(|&symbol| is_productive(&productive, symbol));
                    let n = sequence
                        .iter()
                        .all(|&symbol| is_nullable(&nullable, symbol));
                    changed |= (p && !productive[id]) || (n && !nullable[id]);
                    productive[id] |= p;
                    nullable[id] |= n;
                }
            }
        }
        check_left_recursion(rules, sets, &nullable)?;
        if !productive[self.grammar.root as usize] {
            return Err(SchemaError::empty(
                "JSON Schema: schema has no finite value",
            ));
        }
        // Discard impossible alternatives before they can admit dead
        // prefixes.
        for rule in rules.iter_mut() {
            rule.retain(|sequence| {
                sequence
                    .iter()
                    .all(|&symbol| is_productive(&productive, symbol))
            });
        }
        self.grammar
            .start()
            .map_err(|error| SchemaError::runtime(error.message()))?;
        Ok(self.grammar)
    }
}

/// Rejects references that recurse without consuming input: a rule may not
/// reach itself through nullable leading symbols (the depth-first `check`
/// of `Compile`, without recursion).
fn check_left_recursion(
    rules: &[Rule],
    sets: &[ArgumentSet],
    nullable: &[bool],
) -> Result<(), SchemaError> {
    // 0: unvisited, 1: on the path, 2: done.
    let mut visited = vec![0u8; rules.len()];
    for start in 0..rules.len() {
        if visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        // (rule, sequence, symbol) frames.
        let mut stack: Vec<(usize, usize, usize)> = vec![(start, 0, 0)];
        while let Some(frame) = stack.last_mut() {
            let (rule, sequence, symbol) = *frame;
            let Some(symbols) = rules[rule].get(sequence) else {
                visited[rule] = 2;
                stack.pop();
                continue;
            };
            let Some(&next) = symbols.get(symbol) else {
                *frame = (rule, sequence + 1, 0);
                continue;
            };
            if next & LEAF != 0 {
                *frame = (rule, sequence + 1, 0);
                continue;
            }
            // Every member of an argument set starts with a nonempty native
            // parameter tag, so no cycle passes through one; the sequence
            // continues after it when the set may stay empty.
            if next & ARGUMENTS != 0 {
                *frame = if sets[(next & !ARGUMENTS) as usize].nullable() {
                    (rule, sequence, symbol + 1)
                } else {
                    (rule, sequence + 1, 0)
                };
                continue;
            }
            // After this symbol the sequence continues only when it can be
            // empty.
            *frame = if nullable[next as usize] {
                (rule, sequence, symbol + 1)
            } else {
                (rule, sequence + 1, 0)
            };
            match visited[next as usize] {
                1 => return Err(invalid("reference cycle does not consume input")),
                2 => {}
                _ => {
                    visited[next as usize] = 1;
                    stack.push((next as usize, 0, 0));
                }
            }
        }
    }
    Ok(())
}
