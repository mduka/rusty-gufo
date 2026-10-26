//! A schema's value types, its native form, and the intersection of
//! `allOf`, references and siblings (`conjoin`).

use super::*;

impl<'a> Compiler<'a> {
    /// The JSON value kinds a parameter admits (`ValueTypes`), as llama.cpp
    /// `common/json-schema.cpp` `value_types()`: unions add, untyped schemas
    /// admit all.
    pub(super) fn value_types(&self, schema: &Value, depth: usize) -> ValueTypes {
        let of = |value: &Value| -> ValueTypes {
            1 << match value {
                Value::String(_) => 0,
                Value::Number(_) => 1,
                Value::Bool(_) => 2,
                Value::Null => 3,
                Value::Array(_) => 4,
                Value::Object(_) => 5,
            }
        };
        if let Value::Bool(admits) = schema {
            return if *admits { ALL_VALUE_TYPES } else { 0 };
        }
        if !schema.is_object() || depth > MAX_DEPTH {
            return ALL_VALUE_TYPES;
        }
        if let Some(reference) = schema.find("$ref") {
            return match self.reference(reference) {
                Ok(target) => self.value_types(target, depth + 1),
                Err(_) => ALL_VALUE_TYPES,
            };
        }
        if let Some(value) = schema.find("const") {
            return of(value);
        }
        if let Some(Value::Array(values)) = schema.find("enum") {
            return values.iter().fold(0, |types, value| types | of(value));
        }
        if let Some(ty) = schema.find("type") {
            let mut types = 0;
            let mut add = |name: &Value| {
                if let Value::String(name) = name {
                    let spelling = if name == "integer" { "number" } else { name };
                    for (i, known) in VALUE_TYPE_NAMES.iter().enumerate() {
                        if *known == spelling {
                            types |= 1 << i;
                        }
                    }
                }
            };
            match ty {
                Value::Array(names) => names.iter().for_each(&mut add),
                name => add(name),
            }
            return types;
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(choices)) = schema.find(key) {
                return choices.iter().fold(0, |types, choice| {
                    types | self.value_types(choice, depth + 1)
                });
            }
        }
        // Untyped schemas imply a kind through their keywords.
        if schema.contains("properties")
            || schema
                .find("additionalProperties")
                .is_some_and(|extra| !matches!(extra, Value::Bool(true)))
        {
            return 1 << 5;
        }
        if let Some(Value::Array(parts)) = schema.find("allOf") {
            return parts.iter().fold(ALL_VALUE_TYPES, |types, part| {
                types & self.value_types(part, depth + 1)
            });
        }
        if schema.contains("items") || schema.contains("prefixItems") {
            return 1 << 4;
        }
        if schema.contains("pattern")
            || schema.contains("minLength")
            || schema.contains("maxLength")
        {
            return STRING_TYPE;
        }
        ALL_VALUE_TYPES
    }

    /// The schema a native parameter follows through `$ref`s whose siblings
    /// only annotate it, or `None` (`NativeSchema`).
    pub(super) fn native_schema(
        &self,
        original: &'a Value,
    ) -> Result<Option<&'a Value>, SchemaError> {
        let mut schema = original;
        let mut ty: Option<&'a Value> = None;
        let mut seen: HashSet<usize> = HashSet::new();
        while let Some(reference) = schema.find("$ref") {
            if !seen.insert(address(schema)) {
                return Ok(None);
            }
            for (key, value) in schema.members() {
                if matches!(key.as_str(), "$ref" | "$defs" | "title" | "description") {
                    continue;
                }
                if self.strict {
                    return Ok(None);
                }
                if matches!(
                    key.as_str(),
                    "$schema"
                        | "$id"
                        | "$comment"
                        | "definitions"
                        | "examples"
                        | "default"
                        | "deprecated"
                        | "readOnly"
                        | "writeOnly"
                ) {
                    continue;
                }
                if key != "type" || ty.is_some_and(|ty| !equal_value(ty, value)) {
                    return Ok(None);
                }
                ty = Some(value);
            }
            schema = self.reference(reference)?;
        }
        if let Some(ty) = ty
            && schema.find("type").is_none_or(|own| !equal_value(ty, own))
        {
            return Ok(None);
        }
        Ok(Some(schema))
    }

    /// The intersection of two schemas (`Conjoin`).
    pub(super) fn conjoin(
        &self,
        left: &Value,
        right: &Value,
        depth: u32,
    ) -> Result<Value, SchemaError> {
        if depth > 64 {
            return Err(invalid("schema intersection exceeds its reference budget"));
        }
        self.keys(left)?;
        self.keys(right)?;
        if let Some(reference) = left.find("$ref") {
            let target = self.reference(reference)?;
            let referenced = self.conjoin(target, &without(left, &["$ref", "$defs"]), depth + 1)?;
            return self.conjoin(&referenced, right, depth + 1);
        }
        if let Some(reference) = right.find("$ref") {
            let target = self.reference(reference)?;
            let referenced =
                self.conjoin(target, &without(right, &["$ref", "$defs"]), depth + 1)?;
            return self.conjoin(left, &referenced, depth + 1);
        }
        if let Some(any) = left.find("anyOf") {
            let mut branches = Value::array();
            let siblings = self.conjoin(&without(left, &["anyOf"]), right, depth + 1)?;
            for branch in any.items() {
                match self.conjoin(branch, &siblings, depth + 1) {
                    Ok(branch) => branches.push(branch),
                    Err(SchemaError::Empty(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            if branches.is_empty() {
                return Err(SchemaError::empty(
                    "JSON Schema: schema intersection is empty",
                ));
            }
            let mut result = Value::object();
            result.set("anyOf", branches);
            return Ok(result);
        }
        if right.contains("anyOf") {
            return self.conjoin(right, left, depth + 1);
        }
        if let (Some(format), Some(previous)) = (right.find("format"), left.find("format"))
            && !equal_value(previous, format)
        {
            if !format.is_string() {
                return Err(invalid("format must be a string"));
            }
            let pattern = self.conjoin(
                &without(right, &["format"]),
                &lexeme::format(format.str())?,
                depth + 1,
            )?;
            return self.conjoin(left, &pattern, depth + 1);
        }
        let mut result = left.clone();
        for (key, value) in right.members() {
            let key = key.as_str();
            let Some(previous) = left
                .find(key)
                .filter(|_| key != "title" && key != "description")
            else {
                result.set(key, value.clone());
                continue;
            };
            if equal_value(previous, value) {
                continue;
            }
            match key {
                "minimum" | "exclusiveMinimum" | "minLength" | "minItems" | "maximum"
                | "exclusiveMaximum" | "maxLength" | "maxItems" => {
                    if !previous.is_number() || !value.is_number() {
                        return Err(invalid("bounds must be numbers"));
                    }
                    let lower = key.starts_with("min") || key == "exclusiveMinimum";
                    let (a, b) = (previous.as_double(0.0), value.as_double(0.0));
                    result.set(key, if lower { a.max(b) } else { a.min(b) });
                }
                "multipleOf" => result.set(key, lexeme::intersect_multiple_of(previous, value)?),
                "pattern" => {
                    if !previous.is_string() || !value.is_string() {
                        return Err(invalid("pattern must be a string"));
                    }
                    result.set(
                        key,
                        format!(
                            "(?=[\\s\\S]*(?:{}))(?=[\\s\\S]*(?:{}))",
                            previous.str(),
                            value.str()
                        ),
                    );
                }
                "required" => {
                    if !previous.is_array() || !value.is_array() {
                        return Err(invalid("required must be an array"));
                    }
                    for field in value.items() {
                        if !previous.items().iter().any(|old| equal_value(old, field)) {
                            result.member_mut(key).push(field.clone());
                        }
                    }
                }
                "enum" => {
                    if !previous.is_array() || !value.is_array() {
                        return Err(invalid("enum must be an array"));
                    }
                    let mut common = Value::array();
                    for item in previous.items() {
                        if value.items().iter().any(|other| equal_value(item, other)) {
                            common.push(item.clone());
                        }
                    }
                    if common.is_empty() {
                        return Err(SchemaError::empty(
                            "JSON Schema: enum intersection is empty",
                        ));
                    }
                    result.set(key, common);
                }
                "type" => {
                    let types = |t: &Value| {
                        if t.is_array() {
                            t.items().to_vec()
                        } else {
                            vec![t.clone()]
                        }
                    };
                    let mut common = Value::array();
                    for a in types(previous) {
                        for b in types(value) {
                            if equal_value(&a, &b) {
                                common.push(a.clone());
                            } else if (a.str() == "number" && b.str() == "integer")
                                || (a.str() == "integer" && b.str() == "number")
                            {
                                common.push("integer");
                            }
                        }
                    }
                    if common.is_empty() {
                        return Err(SchemaError::empty(
                            "JSON Schema: schema constraints have no common type",
                        ));
                    }
                    let common = if common.len() == 1 {
                        common.items()[0].clone()
                    } else {
                        common
                    };
                    result.set(key, common);
                }
                "items" => result.set(key, self.conjoin(previous, value, depth + 1)?),
                "properties" => {
                    if !previous.is_object() || !value.is_object() {
                        return Err(invalid("properties must be an object"));
                    }
                    let closed = |schema: &Value| {
                        schema
                            .find("additionalProperties")
                            .is_some_and(|additional| {
                                additional.is_bool() && !additional.as_bool(true)
                            })
                    };
                    let (left_closed, right_closed) = (closed(left), closed(right));
                    let mut properties = Value::object();
                    for (name, child) in previous.members() {
                        if let Some(other) = value.find(name) {
                            properties.append_member(
                                name.clone(),
                                self.conjoin(child, other, depth + 1)?,
                            );
                        } else if !right_closed {
                            properties.append_member(name.clone(), child.clone());
                        }
                    }
                    if !left_closed {
                        for (name, child) in value.members() {
                            if !previous.contains(name) {
                                properties.append_member(name.clone(), child.clone());
                            }
                        }
                    }
                    result.set(key, properties);
                }
                _ => {
                    if key == "const" {
                        return Err(SchemaError::empty(
                            "JSON Schema: const intersection is empty",
                        ));
                    }
                    return Err(invalid(&format!(
                        "incompatible schema constraints for {key}"
                    )));
                }
            }
        }
        if let Some(properties) = result.find("properties") {
            let permits = |schema: &Value, name: &str| {
                let closed = schema.find("additionalProperties");
                let fields = schema.find("properties");
                closed.is_none_or(|closed| !closed.is_bool() || closed.as_bool(false))
                    || fields.is_some_and(|fields| fields.contains(name))
            };
            let mut allowed = Value::object();
            for (name, child) in properties.members() {
                if permits(left, name) && permits(right, name) {
                    allowed.append_member(name.clone(), child.clone());
                }
            }
            result.set("properties", allowed);
        }
        Ok(result)
    }
}
