//! Values of one schema: literals, primitives and lexemes, and integer
//! bounds as digit grammars. The Gemma 4 forms follow franzmoca/gufo's
//! `feat/gemma` branch (`src/core/json_constraint.cpp`), whose names the
//! comments cite.

use super::*;

/// Gemma 4's string delimiter in call arguments.
pub(super) const GEMMA_QUOTE: &str = "<|\"|>";

/// Whether Gemma 4's parser reads `name` as a bare key (`BareGemmaKey`): it
/// ends at the first ':' and is trimmed, and `<|"|>` starts a quoted key.
pub(super) fn bare_gemma_key(name: &str) -> bool {
    const SPACE: &[char] = &[' ', '\t', '\r', '\n', '\x0c', '\x0b'];
    !name.is_empty()
        && !name.contains([':', '{', '}', '[', ']', ','])
        && !name.starts_with(SPACE)
        && !name.ends_with(SPACE)
        && !name.starts_with('<')
}

pub(super) fn count_characters(value: &str) -> usize {
    value.bytes().filter(|&b| b & 0xc0 != 0x80).count()
}

pub(super) fn increment_digits(mut value: Vec<u8>) -> Vec<u8> {
    for i in (0..value.len()).rev() {
        if value[i] != b'9' {
            value[i] += 1;
            return value;
        }
        value[i] = b'0';
    }
    value.insert(0, b'1');
    value
}

pub(super) fn decrement_digits(mut value: Vec<u8>) -> Vec<u8> {
    for i in (0..value.len()).rev() {
        if value[i] != b'0' {
            value[i] -= 1;
            break;
        }
        value[i] = b'9';
    }
    if value.len() > 1 && value[0] == b'0' {
        value.remove(0);
    }
    value
}

impl<'a> Compiler<'a> {
    /// A finite enum/const value that satisfies every sibling constraint,
    /// its objects in schema property order (`ValueFor`); `None` when it
    /// does not.
    pub(super) fn value_for(
        &mut self,
        schema: &Value,
        value: &Value,
        depth: usize,
    ) -> Result<Option<Value>, SchemaError> {
        if depth > 256 {
            return Err(invalid(
                "enum/const reference expansion exceeds its resource budget",
            ));
        }
        let mut result = value.clone();
        if let Some(reference) = schema.find("$ref") {
            let target = self.reference(reference)?;
            let Some(referenced) = self.value_for(target, value, depth + 1)? else {
                return Ok(None);
            };
            result = referenced;
        }
        if let Some(any) = schema.find("anyOf") {
            let mut matched = false;
            for branch in any.items() {
                match self.value_for(branch, value, depth + 1) {
                    Ok(Some(candidate)) => {
                        result = candidate;
                        matched = true;
                        break;
                    }
                    Ok(None) | Err(SchemaError::Empty(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            if !matched {
                return Ok(None);
            }
        }
        if let Some(ty) = schema.find("type") {
            let matches = if ty.is_string() {
                matches_type(value, ty.str())
            } else {
                ty.items().iter().any(|t| matches_type(value, t.str()))
            };
            if !matches {
                return Ok(None);
            }
        }
        if let Some(enumeration) = schema.find("enum")
            && !enumeration
                .items()
                .iter()
                .any(|item| equal_value(value, item))
        {
            return Ok(None);
        }
        if let Some(constant) = schema.find("const")
            && !equal_value(value, constant)
        {
            return Ok(None);
        }
        if value.is_string() || value.is_number() {
            let key = (address(schema), value.is_string());
            let check = match self.value_checks.get(&key) {
                Some(check) => check.clone(),
                None => {
                    let check = if value.is_string() {
                        JsonSchemaLexeme::string(&self.string_predicate(schema))?
                    } else {
                        JsonSchemaLexeme::number(schema, false)?
                    };
                    self.value_checks.insert(key, check.clone());
                    check
                }
            };
            if !check.accept_value(value)? {
                return Ok(None);
            }
        } else if value.is_array() {
            let too_short = schema
                .find("minItems")
                .is_some_and(|minimum| value.len() < minimum.as_size(0));
            let too_long = schema
                .find("maxItems")
                .is_some_and(|maximum| value.len() > maximum.as_size(0));
            if too_short || too_long {
                return Ok(None);
            }
            if let Some(items) = schema.find("items") {
                let mut array = Value::array();
                for item in value.items() {
                    let Some(candidate) = self.value_for(items, item, depth + 1)? else {
                        return Ok(None);
                    };
                    array.push(candidate);
                }
                result = array;
            }
        } else if value.is_object() {
            if let Some(required) = schema.find("required")
                && required
                    .items()
                    .iter()
                    .any(|key| !value.contains(key.str()))
            {
                return Ok(None);
            }
            let additional = schema.find("additionalProperties");
            let properties = match schema.find("properties") {
                Some(properties) => Some(properties),
                None if additional.is_some() => Some(&EMPTY_OBJECT),
                None => None,
            };
            if let Some(properties) = properties {
                let mut object = Value::object();
                for (name, child) in properties.members() {
                    if let Some(item) = value.find(name) {
                        let Some(candidate) = self.value_for(child, item, depth + 1)? else {
                            return Ok(None);
                        };
                        object.append_member(name.clone(), candidate);
                    }
                }
                for (name, item) in value.members() {
                    if properties.contains(name) {
                        continue;
                    }
                    if closed_properties(schema, self.ignore_unknown_keys) {
                        return Ok(None);
                    }
                    match additional {
                        Some(extra)
                            if extra.is_object()
                                && !(self.ignore_unknown_keys
                                    && admits_extra_properties(schema)) =>
                        {
                            let Some(candidate) = self.value_for(extra, item, depth + 1)? else {
                                return Ok(None);
                            };
                            object.append_member(name.clone(), candidate);
                        }
                        _ => object.append_member(name.clone(), item.clone()),
                    }
                }
                result = object;
            }
        }
        Ok(Some(result))
    }

    /// The exact serialization of a finite value, with optional
    /// whitespace between tokens (`ValueLiteral`).
    pub(super) fn value_literal(&mut self, value: &Value) -> Result<u32, SchemaError> {
        let (open, close) = match value {
            Value::Object(_) => (b'{', b'}'),
            Value::Array(_) => (b'[', b']'),
            Value::String(text) if self.gemma => {
                if text.contains(GEMMA_QUOTE) {
                    return Err(invalid("string value cannot be a Gemma 4 string"));
                }
                return self.literal(format!("{GEMMA_QUOTE}{text}{GEMMA_QUOTE}").as_bytes());
            }
            _ => return self.literal(dump(value)?.as_bytes()),
        };
        let mut parts = vec![byte(open), self.ws];
        let mut comma = false;
        let separator = |parts: &mut Sequence, comma: &mut bool, ws: u32| {
            if *comma {
                parts.extend([ws, byte(b','), ws]);
            }
            *comma = true;
        };
        if let Value::Object(_) = value {
            for (name, item) in self.members(value) {
                separator(&mut parts, &mut comma, self.ws);
                let name = self.key_literal(name)?;
                let item = self.value_literal(item)?;
                parts.extend([name, self.ws, byte(b':'), self.ws, item]);
            }
        } else {
            for item in value.items() {
                separator(&mut parts, &mut comma, self.ws);
                parts.push(self.value_literal(item)?);
            }
        }
        parts.extend([self.ws, byte(close)]);
        self.seq(parts)
    }

    /// A Gemma 4 string satisfying the JSON string `predicate`
    /// (`GemmaString`). Its raw text cannot hold the closing delimiter, which
    /// the parser takes at its first occurrence.
    pub(super) fn gemma_string(&mut self, predicate: &Value) -> Result<u32, SchemaError> {
        let raw = JsonSchemaLexeme::raw_string(predicate, GEMMA_QUOTE)?;
        let nullable = raw.check(b"").complete;
        let mut text = self.lexeme(raw);
        if nullable {
            text = self.optional(text)?;
        }
        let quote = self.literal(GEMMA_QUOTE.as_bytes())?;
        self.seq(vec![quote, text, quote])
    }

    /// A dynamic bare Gemma 4 key other than the `excluded` names
    /// (`GemmaKey`): it ends at its first ':', holds no structure bytes and
    /// neither starts nor ends with whitespace, nor starts like a string.
    pub(super) fn gemma_key(&mut self, excluded: &BTreeSet<String>) -> Result<u32, SchemaError> {
        let mut pattern = "^".to_owned();
        if !excluded.is_empty() {
            let names: Vec<String> = excluded.iter().map(|name| regex_literal(name)).collect();
            pattern.push_str(&format!("(?!(?:{})$)", names.join("|")));
        }
        pattern.push_str(r"[^:{}\[\],<\s](?:[^:{}\[\],]*[^:{}\[\],\s])?$");
        let predicate = Value::object().with("pattern", pattern);
        Ok(self.lexeme(JsonSchemaLexeme::raw_string(&predicate, ":")?))
    }

    /// A declared property name as the syntax spells it (`KeyLiteral`): JSON
    /// quoted, or for Gemma 4 bare where its parser reads it so and
    /// `<|"|>`-quoted otherwise, as declarations render it.
    pub(super) fn key_literal(&mut self, name: &str) -> Result<u32, SchemaError> {
        if !self.gemma {
            return self.literal(dump(&Value::from(name))?.as_bytes());
        }
        if bare_gemma_key(name) {
            return self.literal(name.as_bytes());
        }
        if name.contains(GEMMA_QUOTE) {
            return Err(invalid("property name cannot be a Gemma 4 key"));
        }
        self.literal(format!("{GEMMA_QUOTE}{name}{GEMMA_QUOTE}").as_bytes())
    }

    /// The members of `object` in the order the syntax writes them: Gemma 4
    /// renders keys sorted at every depth (`Members`).
    pub(super) fn members<'v>(&self, object: &'v Value) -> Vec<(&'v str, &'v Value)> {
        let mut members: Vec<(&str, &Value)> = object
            .members()
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();
        if self.gemma {
            members.sort_by(|a, b| a.0.cmp(b.0));
        }
        members
    }

    pub(super) fn primitive(&self, ty: &str) -> Result<u32, SchemaError> {
        match ty {
            "string" => Ok(self.string),
            "integer" => Ok(self.integer),
            "number" => Ok(self.number),
            "boolean" => Ok(self.boolean),
            "null" => Ok(self.null),
            _ => Err(invalid("unsupported or missing type")),
        }
    }

    pub(super) fn lexeme(&mut self, lexeme: Arc<JsonSchemaLexeme>) -> u32 {
        let symbol = LEXEME | self.grammar.lexemes.len() as u32;
        self.grammar.lexemes.push(lexeme);
        symbol
    }

    /// An unsigned interval of equally long digit strings through shared
    /// digit prefixes (`DigitsBetween`): work grows with the number of
    /// digits, not with the integers between the bounds.
    pub(super) fn digits_between(&mut self, low: &[u8], high: &[u8]) -> Result<u32, SchemaError> {
        if low.is_empty() {
            return self.seq(vec![]);
        }
        if low.iter().all(|&d| d == b'0') && high.iter().all(|&d| d == b'9') {
            let digit = self.range(b'0', b'9');
            return self.seq(vec![digit; low.len()]);
        }
        if low[0] == high[0] {
            let rest = self.digits_between(&low[1..], &high[1..])?;
            return self.seq(vec![byte(low[0]), rest]);
        }
        let zeros = vec![b'0'; low.len() - 1];
        let nines = vec![b'9'; low.len() - 1];
        let mut choices = Vec::new();
        let rest = self.digits_between(&low[1..], &nines)?;
        choices.push(self.seq(vec![byte(low[0]), rest])?);
        if low[0] + 1 < high[0] {
            let mut middle = vec![self.range(low[0] + 1, high[0] - 1)];
            let digit = self.range(b'0', b'9');
            middle.extend(std::iter::repeat_n(digit, low.len() - 1));
            choices.push(self.seq(middle)?);
        }
        let rest = self.digits_between(&zeros, &high[1..])?;
        choices.push(self.seq(vec![byte(high[0]), rest])?);
        self.alt(choices)
    }

    pub(super) fn unsigned_interval(
        &mut self,
        low: &[u8],
        high: Option<&[u8]>,
    ) -> Result<u32, SchemaError> {
        let mut choices = Vec::new();
        let last = high.map_or(low.len(), <[u8]>::len);
        for length in low.len()..=last {
            let first = if length == low.len() {
                low.to_vec()
            } else {
                let mut first = vec![b'0'; length];
                first[0] = b'1';
                first
            };
            let end = match high {
                Some(high) if length == high.len() => high.to_vec(),
                _ => vec![b'9'; length],
            };
            if first <= end {
                choices.push(self.digits_between(&first, &end)?);
            }
        }
        if high.is_none() {
            let mut longer = vec![self.range(b'1', b'9')];
            let digit = self.range(b'0', b'9');
            longer.extend(std::iter::repeat_n(digit, low.len()));
            let digit = self.range(b'0', b'9');
            longer.push(self.repeat(digit)?);
            choices.push(self.seq(longer)?);
        }
        if choices.is_empty() {
            return Err(SchemaError::empty(
                "JSON Schema: numeric bounds describe an empty interval",
            ));
        }
        self.alt(choices)
    }

    /// An integer under bounds, as a digit grammar (`Integer`).
    pub(super) fn integer_bounds(&mut self, schema: &Value) -> Result<u32, SchemaError> {
        let (mut minimum, mut maximum): (Option<f64>, Option<f64>) = (None, None);
        for key in ["minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"] {
            let Some(value) = schema.find(key) else {
                continue;
            };
            if !value.is_number() || !value.as_double(0.0).is_finite() {
                return Err(invalid(&format!("{key} must be finite")));
            }
            let number = value.as_double(0.0);
            let lower = key.ends_with("Minimum") || key == "minimum";
            let rounded = if lower { number.ceil() } else { number.floor() };
            let bound = if lower { &mut minimum } else { &mut maximum };
            if bound.is_none_or(|bound| {
                if lower {
                    rounded > bound
                } else {
                    rounded < bound
                }
            }) {
                *bound = Some(rounded);
            }
        }
        if minimum.is_none() && maximum.is_none() {
            return Ok(self.integer);
        }
        // JSON stores finite doubles. Fixed notation gives the exact
        // represented integer even for bounds outside int64.
        let decimal = |value: f64| format!("{:.0}", value.abs()).into_bytes();
        let bound = |value: Option<f64>, exclusive: &str, lower: bool| {
            let value = value?;
            let mut result = IntegerBound {
                negative: value < 0.0,
                digits: decimal(value),
            };
            let excluded = schema.find(exclusive);
            if excluded.is_some_and(|excluded| excluded.as_double(0.0) == value) {
                if result.digits == b"0" {
                    result = IntegerBound {
                        negative: !lower,
                        digits: b"1".to_vec(),
                    };
                } else if lower == result.negative {
                    result.digits = decrement_digits(result.digits);
                    if result.digits == b"0" {
                        result.negative = false;
                    }
                } else {
                    result.digits = increment_digits(result.digits);
                }
            }
            Some(result)
        };
        let low = bound(minimum, "exclusiveMinimum", true);
        let high = bound(maximum, "exclusiveMaximum", false);
        let compare = |a: &[u8], b: &[u8]| a.len().cmp(&b.len()).then_with(|| a.cmp(b));
        if let (Some(low), Some(high)) = (&low, &high) {
            let empty = if low.negative != high.negative {
                !low.negative
            } else if low.negative {
                compare(&low.digits, &high.digits).is_lt()
            } else {
                compare(&low.digits, &high.digits).is_gt()
            };
            if empty {
                return Err(SchemaError::empty(
                    "JSON Schema: numeric bounds describe an empty interval",
                ));
            }
        }
        let mut choices = Vec::new();
        if high.as_ref().is_none_or(|high| !high.negative) {
            let start: &[u8] = match &low {
                Some(low) if !low.negative => &low.digits,
                _ => b"0",
            };
            choices
                .push(self.unsigned_interval(start, high.as_ref().map(|h| h.digits.as_slice()))?);
        }
        if low.as_ref().is_none_or(|low| low.negative) {
            let start: &[u8] = match &high {
                Some(high) if high.negative => &high.digits,
                _ => b"1",
            };
            let magnitude =
                self.unsigned_interval(start, low.as_ref().map(|l| l.digits.as_slice()))?;
            choices.push(self.seq(vec![byte(b'-'), magnitude])?);
        }
        if low
            .as_ref()
            .is_none_or(|low| low.negative || low.digits == b"0")
            && high.as_ref().is_none_or(|high| !high.negative)
        {
            choices.push(self.literal(b"-0")?);
        }
        self.alt(choices)
    }
}
