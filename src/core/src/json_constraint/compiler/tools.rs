//! Tool-call grammars: strict and loose tool parameters, plain-text
//! arguments, native suffixes and open parameters, and the finished probes.

use super::*;

impl<'a> Compiler<'a> {
    /// Native parameter tags for a tool's arguments (`ToolParameters`), or
    /// `None` where they cannot represent the schema exactly and the loose
    /// native route takes it (`loose_tool_parameters`). The native envelope and typed parameters
    /// follow llama.cpp `common/parsers/qwen3-coder.cpp` and `deepseek.cpp`
    /// at 6a2743f0: unlike an injected JSON envelope, they keep the syntax
    /// each model's chat template teaches.
    pub(in crate::json_constraint) fn tool_parameters(
        mut self,
        format: ToolFormat,
        best_effort: bool,
    ) -> Result<Option<JsonConstraint>, SchemaError> {
        self.ignore_unknown_keys = best_effort && !self.strict;
        let Some(root) = self.native_schema(self.schema)? else {
            return Ok(None);
        };
        // Object-level unions and finite values take the loose native route.
        if root.member_str("type", "") != "object"
            || root.contains("anyOf")
            || root.contains("const")
            || root.contains("enum")
        {
            return Ok(None);
        }
        let Some(properties) = root.find("properties").filter(|p| p.is_object()) else {
            return Ok(None);
        };
        let open = !self.strict
            && !closed_properties(root, best_effort)
            && (root.contains("additionalProperties")
                || (best_effort && admits_extra_properties(root)));
        let qwen = format == ToolFormat::Qwen;
        // Qwen has no type flag for wildcard parameters: its parser reads
        // unknown names as text, which would turn 1/true/null/[]/{} into
        // strings. JSON keeps their types instead of guessing.
        if open && qwen {
            return Ok(None);
        }
        if root
            .find("additionalProperties")
            .is_some_and(|extra| !extra.is_bool())
            && !(best_effort && admits_extra_properties(root))
        {
            return Ok(None);
        }
        let mut required = BTreeSet::new();
        if let Some(fields) = root.find("required") {
            if !fields.is_array() {
                return Ok(None);
            }
            for field in fields.items() {
                if !field.is_string() || !properties.contains(field.str()) {
                    return Ok(None);
                }
                required.insert(field.str().to_owned());
            }
        }
        let close = if qwen {
            "\n</parameter>\n"
        } else {
            "</｜DSML｜parameter>"
        };
        let end_tag = if qwen {
            close.to_owned()
        } else {
            format!("{close}\n")
        };
        let mut members: Vec<(String, u32)> = Vec::new();
        for (name, original) in properties.members() {
            if name.is_empty() || name.contains(['<', '>', '"', '=', '\r', '\n']) {
                return Ok(None);
            }
            // The loose route handles literal names outside this subset.
            const TRIMMED: [char; 4] = [' ', '\t', '\x0c', '\x0b'];
            if qwen && (name.starts_with(TRIMMED) || name.ends_with(TRIMMED)) {
                return Ok(None);
            }
            let schema = self.native_schema(original)?;
            let typed = schema
                .filter(|schema| !schema.contains("anyOf"))
                .and_then(|schema| match schema.find("type") {
                    Some(Value::String(ty)) => Some((schema, ty.as_str())),
                    _ => None,
                });
            let Some((schema, ty)) = typed else {
                // Strict unions take the loose native route. Agent
                // harnesses routinely declare unions: switching
                // the request to a JSON envelope contradicts the chat
                // template and the native calls in history, so long
                // conversations mix both syntaxes. As llama.cpp's
                // qwen3-coder parser at 46ca246de does, a union admitting
                // strings is raw text, and the HTTP parser tries its typed
                // alternatives first.
                if self.strict {
                    return Ok(None);
                }
                let types = self.value_types(original, 0);
                if types == 0 {
                    return Ok(None);
                }
                let text = types & STRING_TYPE != 0;
                let mut typed = 0;
                if !text || !qwen {
                    typed = if self.probe(original)? {
                        self.visit(original, 1)?
                    } else {
                        self.generic_value(MAX_DEPTH)?
                    };
                }
                let raw = self.lexeme(JsonSchemaLexeme::raw_string(&Value::object(), close)?);
                let raw = self.optional(raw)?;
                let value = if qwen {
                    let tag = self.literal(format!("<parameter={name}>\n").as_bytes())?;
                    self.seq(vec![tag, if text { raw } else { typed }])?
                } else {
                    let tag = format!("<｜DSML｜parameter name=\"{name}\" string=\"");
                    let mut choices = Vec::new();
                    if text {
                        let string = self.literal(format!("{tag}true\">").as_bytes())?;
                        choices.push(self.seq(vec![string, raw])?);
                    }
                    if types.count_ones() > u32::from(text) {
                        let other = self.literal(format!("{tag}false\">").as_bytes())?;
                        choices.push(self.seq(vec![other, typed])?);
                    }
                    self.alt(choices)?
                };
                let end = self.literal(end_tag.as_bytes())?;
                members.push((name.clone(), self.seq(vec![value, end])?));
                continue;
            };
            let string = ty == "string";
            let value = if string {
                if schema.contains("const") || schema.contains("enum") {
                    let values: Vec<Value> = match schema.find("enum") {
                        Some(values) => values.items().to_vec(),
                        None => schema.find("const").cloned().into_iter().collect(),
                    };
                    let mut choices = Vec::new();
                    for item in &values {
                        // A value ending in a delimiter prefix would admit a
                        // duplicated closer.
                        if format!("{}{close}", item.str()).find(close) != Some(item.str().len()) {
                            return Ok(None);
                        }
                        if self.value_for(schema, item, 0)?.is_some() {
                            choices.push(self.literal(item.str().as_bytes())?);
                        }
                    }
                    self.alt(choices)?
                } else {
                    // A regex may require the raw delimiter itself: its
                    // intersection with native framing could then be empty
                    // for a satisfiable schema; the loose route takes it.
                    if schema.contains("pattern") {
                        return Ok(None);
                    }
                    let matcher =
                        JsonSchemaLexeme::raw_string(&self.string_predicate(schema), close)?;
                    let nullable = matcher.check(b"").complete;
                    let value = self.lexeme(matcher);
                    if nullable {
                        self.optional(value)?
                    } else {
                        value
                    }
                }
            } else if best_effort {
                // An unsupported keyword elsewhere must not erase this
                // parameter's supported nested constraints.
                if self.probe(original)? {
                    self.visit(original, 1)?
                } else {
                    match ty {
                        "array" => self.generic_array(MAX_DEPTH)?,
                        "object" => self.generic_object(MAX_DEPTH)?,
                        other => self.primitive(other)?,
                    }
                }
            } else {
                self.visit(original, 1)?
            };
            let tag = if qwen {
                format!("<parameter={name}>\n")
            } else {
                format!(
                    "<｜DSML｜parameter name=\"{name}\" string=\"{}\">",
                    if string { "true" } else { "false" }
                )
            };
            let tag = self.literal(tag.as_bytes())?;
            let end = self.literal(end_tag.as_bytes())?;
            members.push((name.clone(), self.seq(vec![tag, value, end])?));
        }
        let body = self.arguments(&members, &required)?;
        self.grammar.root = if open {
            let wildcard = self.open_tool_parameter(format, properties)?;
            let wildcards = self.repeat(wildcard)?;
            self.seq(vec![body, wildcards])?
        } else {
            body
        };
        self.finish().map(Some)
    }

    /// Native parameters for schemas `tool_parameters` cannot enforce exactly
    /// (`LooseToolParameters`), as llama.cpp `common/parsers/qwen3-coder.cpp`
    /// and `deepseek.cpp` build them: only the root object's declared
    /// properties are parameters, a parameter admitting strings is raw text,
    /// and other values follow the supported parts of their schema. The
    /// request keeps the template's native syntax instead of a JSON envelope
    ///.
    pub(in crate::json_constraint) fn loose_tool_parameters(
        mut self,
        format: ToolFormat,
    ) -> Result<JsonConstraint, SchemaError> {
        self.ignore_unknown_keys = true;
        let qwen = format == ToolFormat::Qwen;
        let close = if qwen {
            "\n</parameter>\n"
        } else {
            "</｜DSML｜parameter>"
        };
        // A root reference is followed as `tool_parameters` does; other
        // roots that are not objects declare no parameters, as in llama.cpp.
        let mut root = self.schema;
        let mut seen: HashSet<usize> = HashSet::new();
        while let Some(reference) = root.find("$ref") {
            if !seen.insert(address(root)) {
                break;
            }
            match self.reference(reference) {
                Ok(target) => root = target,
                Err(_) => break,
            }
        }
        let properties = root.find("properties").filter(|properties| {
            properties.is_object()
                && !["anyOf", "oneOf", "const", "enum"]
                    .iter()
                    .any(|key| root.contains(key))
        });
        let mut required = BTreeSet::new();
        if let (Some(properties), Some(Value::Array(fields))) = (properties, root.find("required"))
        {
            for field in fields {
                if field.is_string() && properties.contains(field.str()) {
                    required.insert(field.str().to_owned());
                }
            }
        }
        let mut members: Vec<(String, u32)> = Vec::new();
        for (name, original) in properties.unwrap_or(&EMPTY_OBJECT).members() {
            // A name holding the tag's own delimiter cannot be read back.
            // Other names, surrounding spaces included, are written literally
            // as llama.cpp does; the parser matches declared names exactly.
            let delimiters: &[char] = if qwen {
                &['>', '\r', '\n']
            } else {
                &['"', '\r', '\n']
            };
            if name.is_empty() || name.contains(delimiters) {
                required.remove(name);
                continue;
            }
            let mut types = self.value_types(original, 0);
            if types == 0 {
                types = ALL_VALUE_TYPES;
            }
            let text = types & STRING_TYPE != 0;
            let mut typed = 0;
            if !text || !qwen {
                // Probed separately, productivity included: an unsatisfiable
                // value is guidance here and must not make the tool
                // uncallable.
                typed = if self.probe_finished(original)? {
                    self.visit(original, 1)?
                } else {
                    self.generic_value(MAX_DEPTH)?
                };
            }
            let any_text = self.lexeme(JsonSchemaLexeme::raw_string(&Value::object(), close)?);
            let mut raw = self.optional(any_text)?;
            // A plain string keeps the finite values, lengths and formats
            // `tool_parameters` enforces, so another parameter needing this
            // route never relaxes it. Patterns stay raw, as on both routes.
            if types.count_ones() == 1 && text {
                match self.plain_text(original, close) {
                    Ok(Some(value)) => raw = value,
                    Ok(None) | Err(SchemaError::Empty(_) | SchemaError::Invalid(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            let value = if qwen {
                let tag = self.literal(format!("<parameter={name}>\n").as_bytes())?;
                self.seq(vec![tag, if text { raw } else { typed }])?
            } else {
                let tag = format!("<｜DSML｜parameter name=\"{name}\" string=\"");
                let mut choices = Vec::new();
                if text {
                    let string = self.literal(format!("{tag}true\">").as_bytes())?;
                    choices.push(self.seq(vec![string, raw])?);
                }
                if types.count_ones() > u32::from(text) {
                    let other = self.literal(format!("{tag}false\">").as_bytes())?;
                    choices.push(self.seq(vec![other, typed])?);
                }
                self.alt(choices)?
            };
            let end = self.literal(
                if qwen {
                    close.to_owned()
                } else {
                    format!("{close}\n")
                }
                .as_bytes(),
            )?;
            members.push((name.clone(), self.seq(vec![value, end])?));
        }
        self.grammar.root = self.arguments(&members, &required)?;
        self.finish()
    }

    /// The raw value of a plain string parameter with its finite values,
    /// lengths and formats, or `None` for any other schema.
    pub(super) fn plain_text(
        &mut self,
        original: &'a Value,
        close: &str,
    ) -> Result<Option<u32>, SchemaError> {
        let Some(schema) = self.native_schema(original)? else {
            return Ok(None);
        };
        if schema.member_str("type", "") != "string"
            || !schema.find("type").is_some_and(Value::is_string)
            || schema.contains("anyOf")
            || schema.contains("pattern")
        {
            return Ok(None);
        }
        if schema.contains("const") || schema.contains("enum") {
            let values: Vec<Value> = match schema.find("enum") {
                Some(values) => values.items().to_vec(),
                None => schema.find("const").cloned().into_iter().collect(),
            };
            let mut choices = Vec::new();
            let mut literal = true;
            for item in &values {
                literal = literal
                    && item.is_string()
                    && format!("{}{close}", item.str()).find(close) == Some(item.str().len());
                if literal && self.value_for(schema, item, 0)?.is_some() {
                    choices.push(self.literal(item.str().as_bytes())?);
                }
            }
            return if literal && !choices.is_empty() {
                self.alt(choices).map(Some)
            } else {
                Ok(None)
            };
        }
        let matcher = JsonSchemaLexeme::raw_string(&self.string_predicate(schema), close)?;
        let nullable = matcher.check(b"").complete;
        let value = self.lexeme(matcher);
        if nullable {
            self.optional(value).map(Some)
        } else {
            Ok(Some(value))
        }
    }

    /// Native arguments in any order (`Arguments`): small objects share
    /// subset suffixes; wide ones are a request-local subset instead of all
    /// 2^N subsets or a forced declaration order.
    /// Required fields and one use per parameter name hold in both forms.
    pub(super) fn arguments(
        &mut self,
        members: &[(String, u32)],
        required: &BTreeSet<String>,
    ) -> Result<u32, SchemaError> {
        if members.len() > 10 {
            let mut set = ArgumentSet {
                members: Vec::with_capacity(members.len()),
                required: vec![0; members.len().div_ceil(64)],
            };
            for (i, (name, rule)) in members.iter().enumerate() {
                set.members.push(*rule);
                if required.contains(name) {
                    set.required[i / 64] |= 1 << (i % 64);
                }
            }
            let id = self.grammar.argument_sets.len() as u32;
            self.grammar.argument_sets.push(set);
            return self.seq(vec![ARGUMENTS | id]);
        }
        let mut suffixes = BTreeMap::new();
        self.native_suffix(members, required, 0, &mut suffixes)
    }

    /// The members not yet in `used`, in any order, each at most once.
    pub(super) fn native_suffix(
        &mut self,
        members: &[(String, u32)],
        required: &BTreeSet<String>,
        used: u32,
        suffixes: &mut BTreeMap<u32, u32>,
    ) -> Result<u32, SchemaError> {
        if let Some(&id) = suffixes.get(&used) {
            return Ok(id);
        }
        let mut alternatives: Rule = Vec::new();
        let mut complete = true;
        for (i, (name, rule)) in members.iter().enumerate() {
            if used & (1 << i) != 0 {
                continue;
            }
            complete &= !required.contains(name);
            let rest = self.native_suffix(members, required, used | (1 << i), suffixes)?;
            alternatives.push(vec![*rule, rest]);
        }
        if complete {
            alternatives.push(Vec::new());
        }
        let id = self.new_rule(alternatives)?;
        suffixes.insert(used, id);
        Ok(id)
    }

    /// Best-effort native framing of a non-strict tool without declared
    /// argument types (`OpenToolParameters`): the model's own string or DSML
    /// typed values.
    pub(in crate::json_constraint) fn open_tool_parameters(
        mut self,
        format: ToolFormat,
    ) -> Result<JsonConstraint, SchemaError> {
        let parameter = self.open_tool_parameter(format, &EMPTY_OBJECT)?;
        self.grammar.root = self.repeat(parameter)?;
        self.finish()
    }

    /// One parameter of any name outside `properties` (`OpenToolParameter`).
    pub(super) fn open_tool_parameter(
        &mut self,
        format: ToolFormat,
        properties: &Value,
    ) -> Result<u32, SchemaError> {
        let qwen = format == ToolFormat::Qwen;
        // Dynamic names must survive the native parser's trimming and contain
        // no tag or attribute delimiter; values keep literal UTF-8.
        const NAME: &str = r#"^[^<>"=\s](?:[^<>"=\r\n]*[^<>"=\s])?$"#;
        let pattern = if properties.is_empty() {
            NAME.to_owned()
        } else {
            exclude_names(properties, &NAME[1..])
        };
        let name_schema = Value::object()
            .with("type", "string")
            .with("pattern", pattern);
        let name = self.lexeme(JsonSchemaLexeme::raw_string(
            &name_schema,
            if qwen { ">" } else { "\"" },
        )?);
        let close = if qwen {
            "\n</parameter>\n"
        } else {
            "</｜DSML｜parameter>"
        };
        let raw = self.lexeme(JsonSchemaLexeme::raw_string(&Value::object(), close)?);
        let raw = self.optional(raw)?;
        let value = if qwen {
            let tag = self.literal(b">\n")?;
            self.seq(vec![tag, raw])?
        } else {
            let string = self.literal(b"\" string=\"true\">")?;
            let string = self.seq(vec![string, raw])?;
            let other = self.literal(b"\" string=\"false\">")?;
            let any = self.generic_value(MAX_DEPTH)?;
            let other = self.seq(vec![other, any])?;
            self.alt(vec![string, other])?
        };
        let open = self.literal(if qwen {
            b"<parameter=".as_slice()
        } else {
            "<｜DSML｜parameter name=\"".as_bytes()
        })?;
        let end = self.literal(
            if qwen {
                close.to_owned()
            } else {
                format!("{close}\n")
            }
            .as_bytes(),
        )?;
        self.seq(vec![open, name, value, end])
    }

    /// Whether `schema` compiles on its own into a productive grammar, with
    /// unknown keys ignored (the loose route's probe).
    pub(super) fn probe_finished(&self, schema: &Value) -> Result<bool, SchemaError> {
        let mut probe = Compiler::new(self.schema, self.strict, true)?;
        probe.ignore_unknown_keys = true;
        match probe.visit(schema, 1).and_then(|root| {
            probe.grammar.root = root;
            probe.finish()
        }) {
            Ok(_) => Ok(true),
            Err(SchemaError::Empty(_) | SchemaError::Invalid(_)) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Whether `schema` compiles on its own; a separate compiler, since a
    /// failed visit can leave unfinished recursive rules behind.
    pub(super) fn probe(&self, schema: &Value) -> Result<bool, SchemaError> {
        let mut probe = Compiler::new(self.schema, self.strict, true)?;
        probe.ignore_unknown_keys = self.ignore_unknown_keys;
        match probe.visit(schema, 1) {
            Ok(_) => Ok(true),
            Err(SchemaError::Empty(_) | SchemaError::Invalid(_)) => Ok(false),
            Err(error) => Err(error),
        }
    }
}
