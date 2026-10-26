//! Objects and arrays: property keys, the visit of a schema's body, finite
//! value sets and references.

use super::*;

impl<'a> Compiler<'a> {
    /// Rejects keywords outside the supported subset (`Keys`).
    pub(super) fn keys(&self, schema: &Value) -> Result<(), SchemaError> {
        if !schema.is_object() {
            return Err(invalid("each schema must be an object"));
        }
        const ALLOWED: &[&str] = &[
            "type",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "minItems",
            "maxItems",
            "enum",
            "const",
            "anyOf",
            "$defs",
            "$ref",
            "title",
            "description",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "pattern",
            "format",
            "minLength",
            "maxLength",
        ];
        for (key, value) in schema.members() {
            if !ALLOWED.contains(&key.as_str()) && !self.ignore_unknown_keys {
                return Err(invalid(&format!("unsupported keyword: {key}")));
            }
            if (key == "title" || key == "description") && !value.is_string() {
                return Err(invalid(&format!("{key} must be a string")));
            }
            if key == "anyOf" && (!value.is_array() || value.is_empty()) {
                return Err(invalid("anyOf needs at least one branch"));
            }
        }
        Ok(())
    }

    /// The rule of a schema, compiled once per address; an empty schema
    /// becomes a rule without alternatives (`Visit`).
    pub(super) fn visit(&mut self, schema: &Value, depth: usize) -> Result<u32, SchemaError> {
        let key = address(schema);
        if let Some(&id) = self.compiled.get(&key) {
            return Ok(id);
        }
        let id = self.new_rule(vec![])?;
        self.compiled.insert(key, id);
        match self.visit_body(schema, depth) {
            Ok(body) => self.grammar.rules[id as usize] = vec![vec![body]],
            Err(SchemaError::Empty(_)) => self.grammar.rules[id as usize] = vec![],
            Err(error) => return Err(error),
        }
        Ok(id)
    }

    pub(super) fn count_strings(&mut self, value: &str) -> Result<(), SchemaError> {
        self.string_characters += count_characters(value);
        if self.string_characters > 120_000 {
            return Err(invalid(
                "property/definition names and enum/const strings exceed 120000 characters",
            ));
        }
        Ok(())
    }

    pub(super) fn count_value_strings(&mut self, value: &Value) -> Result<(), SchemaError> {
        match value {
            Value::String(text) => self.count_strings(text),
            Value::Array(items) => items
                .iter()
                .try_for_each(|item| self.count_value_strings(item)),
            Value::Object(members) => members.iter().try_for_each(|(name, item)| {
                self.count_strings(name)?;
                self.count_value_strings(item)
            }),
            _ => Ok(()),
        }
    }

    pub(super) fn visit_body(&mut self, schema: &Value, depth: usize) -> Result<u32, SchemaError> {
        if !schema.is_object() {
            return Err(invalid("each schema must be an object"));
        }
        if depth > MAX_DEPTH {
            return Err(invalid("maximum schema depth is 16"));
        }
        self.keys(schema)?;
        if let Some(defs) = schema.find("$defs") {
            if !defs.is_object() {
                return Err(invalid("$defs must be an object"));
            }
            for (name, value) in defs.members() {
                self.count_strings(name)?;
                self.visit(value, depth)?;
            }
        }
        if let Some(reference) = schema.find("$ref") {
            let siblings = without(schema, &["$ref", "$defs", "title", "description"]);
            let target = self.reference(reference)?;
            if siblings.is_empty() {
                return self.visit(target, depth);
            }
            let conjoined = self.conjoin(target, &siblings, 0)?;
            let stored = self.store(conjoined)?;
            return self.visit(&stored, depth);
        }
        if let Some(any) = schema.find("anyOf") {
            if !any.is_array() || any.is_empty() {
                return Err(invalid("anyOf needs at least one branch"));
            }
            let siblings = without(schema, &["anyOf", "$defs", "title", "description"]);
            let mut branches = Vec::new();
            for value in any.items() {
                let branch = if siblings.is_empty() {
                    self.visit(value, depth + 1)
                } else {
                    match self.conjoin(value, &siblings, 0) {
                        Ok(conjoined) => {
                            let stored = self.store(conjoined)?;
                            self.visit(&stored, depth + 1)
                        }
                        Err(error) => Err(error),
                    }
                };
                match branch {
                    Ok(branch) => branches.push(branch),
                    Err(SchemaError::Empty(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            return self.alt(branches);
        }
        let ty = schema.find("type");
        // An unconstrained leaf is a valid schema, not a reason to discard
        // the requirements on its containing array or object.
        if ty.is_none()
            && self.open_objects
            && without(schema, &["title", "description", "default", "examples"]).is_empty()
        {
            return self.generic_value(MAX_DEPTH - depth);
        }
        let mut types: Vec<String> = Vec::new();
        match ty {
            Some(Value::String(name)) => types.push(name.clone()),
            Some(Value::Array(items)) if (1..=2).contains(&items.len()) => {
                for value in items {
                    let Value::String(name) = value else {
                        return Err(invalid("type members must be strings"));
                    };
                    types.push(name.clone());
                }
                if types.len() == 2 && types.iter().filter(|t| *t == "null").count() != 1 {
                    return Err(invalid("type unions must be nullable; use anyOf otherwise"));
                }
            }
            _ => {
                return Err(invalid("type must be a string or nullable type array"));
            }
        }
        let has = |name: &str| types.iter().any(|t| t == name);
        let (object, array) = (has("object"), has("array"));
        for (key, _) in schema.members() {
            let key = key.as_str();
            if matches!(key, "properties" | "required" | "additionalProperties") && !object {
                return Err(invalid("object keyword on a non-object schema"));
            }
            if matches!(key, "items" | "minItems" | "maxItems") && !array {
                return Err(invalid("array keyword on a non-array schema"));
            }
            if matches!(
                key,
                "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" | "multipleOf"
            ) && !has("integer")
                && !has("number")
            {
                return Err(invalid("numeric constraint on a non-numeric schema"));
            }
            if matches!(key, "pattern" | "format" | "minLength" | "maxLength")
                && !has("string")
                && !(self.ignore_unknown_keys && key == "format")
            {
                return Err(invalid("string constraint on a non-string schema"));
            }
        }
        if schema.contains("enum") || schema.contains("const") {
            return self.finite_values(schema, depth, &types);
        }
        let mut alternatives = Vec::new();
        for name in &types {
            let alternative = match name.as_str() {
                "object" => self.object(schema, depth)?,
                "array" => self.array(schema, depth)?,
                "integer" if schema.contains("multipleOf") => {
                    self.lexeme(JsonSchemaLexeme::number(schema, true)?)
                }
                "number"
                    if [
                        "minimum",
                        "maximum",
                        "exclusiveMinimum",
                        "exclusiveMaximum",
                        "multipleOf",
                    ]
                    .iter()
                    .any(|key| schema.contains(key)) =>
                {
                    self.lexeme(JsonSchemaLexeme::number(schema, false)?)
                }
                "string"
                    if ["pattern", "format", "minLength", "maxLength"]
                        .iter()
                        .any(|key| schema.contains(key)) =>
                {
                    self.lexeme(JsonSchemaLexeme::string(&self.string_predicate(schema))?)
                }
                "integer" => self.integer_bounds(schema)?,
                other => self.primitive(other)?,
            };
            alternatives.push(alternative);
        }
        self.alt(alternatives)
    }

    /// The choices of an `enum` or `const` that satisfy the rest of the
    /// schema. The whole underlying schema is still validated, even where
    /// the finite choices would hide malformed properties or predicates.
    pub(super) fn finite_values(
        &mut self,
        schema: &Value,
        depth: usize,
        types: &[String],
    ) -> Result<u32, SchemaError> {
        let base = self.store(without(schema, &["enum", "const", "$defs"]))?;
        self.visit(&base, depth)?;
        let values: Vec<Value> = match schema.find("enum") {
            Some(enumeration) => {
                if !enumeration.is_array() || enumeration.is_empty() {
                    return Err(invalid("enum must be a nonempty array"));
                }
                enumeration.items().to_vec()
            }
            None => schema.find("const").cloned().into_iter().collect(),
        };
        self.enum_values += values.len();
        if self.enum_values > 1000 {
            return Err(invalid("maximum enum/const value count is 1000"));
        }
        let mut choices = Vec::new();
        let mut enum_characters = 0;
        for value in &values {
            self.count_value_strings(value)?;
            if !types.iter().any(|t| matches_type(value, t)) {
                return Err(invalid("enum/const value does not match its type"));
            }
            if let Value::String(text) = value {
                enum_characters += count_characters(text);
                if values.len() > 250 && enum_characters > 15000 {
                    return Err(invalid(
                        "an enum with more than 250 entries is limited to 15000 string characters",
                    ));
                }
            }
            if let Some(constant) = schema.find("const")
                && !equal_value(value, constant)
            {
                continue;
            }
            match self.value_for(&base, value, 0) {
                Ok(Some(canonical)) => choices.push(self.value_literal(&canonical)?),
                Ok(None) | Err(SchemaError::Empty(_)) => {}
                Err(error) => return Err(error),
            }
        }
        if choices.is_empty() {
            return Err(SchemaError::empty(
                "JSON Schema: enum/const has no value satisfying its constraints",
            ));
        }
        self.alt(choices)
    }

    /// Resolves a local JSON pointer (`Reference`).
    pub(super) fn reference(&self, reference: &Value) -> Result<&'a Value, SchemaError> {
        resolve_reference(self.schema, reference)
    }

    pub(super) fn object(&mut self, schema: &Value, depth: usize) -> Result<u32, SchemaError> {
        let properties = schema.find("properties").unwrap_or(&EMPTY_OBJECT);
        let additional = schema.find("additionalProperties");
        // As in llama.cpp `common/json-schema.cpp` at 23b0202a1, a response
        // object that declares properties without additionalProperties
        // stays closed.
        let closed = closed_properties(schema, self.ignore_unknown_keys)
            || (self.open_response_objects
                && additional.is_none()
                && schema.find("properties").is_some());
        if !properties.is_object()
            || additional.is_some_and(|additional| !additional.is_bool() && !additional.is_object())
        {
            return Err(invalid("invalid object properties or additionalProperties"));
        }
        if !closed && !self.open_objects && !self.open_response_objects {
            return Err(invalid(
                "objects require properties and additionalProperties: false",
            ));
        }
        self.properties += properties.len();
        if self.properties > 5000 {
            return Err(invalid("maximum property count is 5000"));
        }
        let mut required: BTreeSet<&str> = BTreeSet::new();
        if let Some(fields) = schema.find("required") {
            if !fields.is_array() {
                return Err(invalid("required must be an array"));
            }
            for field in fields.items() {
                if !field.is_string()
                    || !properties.contains(field.str())
                    || !required.insert(field.str())
                {
                    return Err(invalid(
                        "required contains an unknown or duplicate property",
                    ));
                }
            }
        }
        if self.strict && required.len() != properties.len() {
            return Err(invalid(
                "strict schemas require every property (use null for optional values)",
            ));
        }
        // Two suffix states represent whether a comma is needed. Optional
        // fields keep schema order without enumerating every subset.
        let mut suffix = [self.seq(vec![])?, self.seq(vec![])?];
        if !closed {
            // Open objects as llama.cpp `common/json-schema-to-grammar.cpp`
            // at e358d591: declared fields keep their schemas and extra keys
            // may follow them. The wildcard excludes declared names, so an
            // extra, wrongly typed duplicate cannot overwrite a validated
            // field; it matches decoded Unicode, so escapes cannot bypass it.
            let key = if properties.is_empty() {
                self.string
            } else {
                let predicate = Value::object().with("pattern", exclude_names(properties, ""));
                self.lexeme(JsonSchemaLexeme::string(&predicate)?)
            };
            let value = match additional {
                Some(extra)
                    if extra.is_object()
                        && !(self.ignore_unknown_keys && admits_extra_properties(schema)) =>
                {
                    self.visit(extra, depth + 1)?
                }
                _ => self.generic_value(MAX_DEPTH - depth)?,
            };
            let member = self.seq(vec![key, self.ws, byte(b':'), self.ws, value])?;
            let more = self.seq(vec![self.ws, byte(b','), self.ws, member])?;
            let tail = self.repeat(more)?;
            let first = self.seq(vec![member, tail])?;
            suffix = [self.optional(first)?, tail];
        }
        for (key, value) in properties.members().iter().rev() {
            self.count_strings(key)?;
            let name = self.literal(dump(&Value::from(key.as_str()))?.as_bytes())?;
            let body = self.visit(value, depth + 1)?;
            let member = self.seq(vec![name, self.ws, byte(b':'), self.ws, body])?;
            let mut next = [0u32; 2];
            for (comma, slot) in next.iter_mut().enumerate() {
                let mut sequence = if comma != 0 {
                    vec![self.ws, byte(b','), self.ws]
                } else {
                    Vec::new()
                };
                sequence.extend([member, suffix[1]]);
                let mut rule: Rule = vec![sequence];
                if !required.contains(key.as_str()) {
                    rule.push(vec![suffix[comma]]);
                }
                *slot = self.new_rule(rule)?;
            }
            suffix = next;
        }
        self.seq(vec![byte(b'{'), self.ws, suffix[0], self.ws, byte(b'}')])
    }

    pub(super) fn array(&mut self, schema: &Value, depth: usize) -> Result<u32, SchemaError> {
        let Some(items) = schema.find("items") else {
            return Err(invalid("arrays require an items schema"));
        };
        let bound = |name: &str, fallback: usize| -> Result<usize, SchemaError> {
            let Some(value) = schema.find(name) else {
                return Ok(fallback);
            };
            let number = value.as_double(0.0);
            if !value.is_number()
                || number < 0.0
                || number > f64::from(u32::MAX)
                || number.floor() != number
            {
                return Err(invalid("array bounds must be nonnegative 32-bit integers"));
            }
            Ok(value.as_size(0))
        };
        let minimum = bound("minItems", 0)?;
        let maximum = bound("maxItems", u32::MAX as usize)?;
        if minimum > maximum {
            return Err(SchemaError::empty("JSON Schema: minItems exceeds maxItems"));
        }
        let item = self.visit(items, depth + 1)?;
        if maximum == 0 {
            return self.seq(vec![byte(b'['), self.ws, byte(b']')]);
        }
        let additional = self.seq(vec![self.ws, byte(b','), self.ws, item])?;
        let required = minimum.saturating_sub(1);
        let tail = if schema.contains("maxItems") {
            self.at_most(additional, maximum - required - 1)?
        } else {
            self.repeat(additional)?
        };
        let repeated = self.exact(additional, required)?;
        let mut members = self.seq(vec![item, repeated, tail])?;
        if minimum == 0 {
            members = self.optional(members)?;
        }
        self.seq(vec![byte(b'['), self.ws, members, self.ws, byte(b']')])
    }
}
