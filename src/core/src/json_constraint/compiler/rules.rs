//! Rule constructors of the byte grammar and the generic JSON values.

use super::*;

impl<'a> Compiler<'a> {
    /// The string predicate of `schema`; best-effort tools ignore an unknown
    /// format (`StringPredicate`).
    pub(super) fn string_predicate(&self, schema: &Value) -> Value {
        if self.ignore_unknown_keys
            && let Some(Value::String(format)) = schema.find("format")
            && lexeme::format(format).is_err()
        {
            return without(schema, &["format"]);
        }
        schema.clone()
    }

    pub(super) fn new_rule(&mut self, rule: Rule) -> Result<u32, SchemaError> {
        if self.grammar.rules.len() >= MAX_RULES {
            return Err(invalid("compiled grammar exceeds the rule limit"));
        }
        self.grammar.rules.push(rule);
        Ok(self.grammar.rules.len() as u32 - 1)
    }

    pub(super) fn seq(&mut self, sequence: Sequence) -> Result<u32, SchemaError> {
        self.new_rule(vec![sequence])
    }

    pub(super) fn alt(&mut self, alternatives: Sequence) -> Result<u32, SchemaError> {
        self.new_rule(alternatives.into_iter().map(|id| vec![id]).collect())
    }

    pub(super) fn class(&mut self, characters: &[u8]) -> u32 {
        let mut bits = ByteClass::default();
        for &b in characters {
            bits.set(b);
        }
        self.grammar.classes.push(bits);
        TERMINAL | (self.grammar.classes.len() as u32 - 1)
    }

    pub(super) fn range(&mut self, first: u8, last: u8) -> u32 {
        let bytes: Vec<u8> = (first..=last).collect();
        self.class(&bytes)
    }

    pub(super) fn literal(&mut self, value: &[u8]) -> Result<u32, SchemaError> {
        let mut chunks = Vec::new();
        for chunk in value.chunks(64) {
            chunks.push(self.seq(chunk.iter().copied().map(byte).collect())?);
        }
        self.seq(chunks)
    }

    pub(super) fn optional(&mut self, rule: u32) -> Result<u32, SchemaError> {
        self.new_rule(vec![vec![], vec![rule]])
    }

    pub(super) fn repeat(&mut self, rule: u32) -> Result<u32, SchemaError> {
        let id = self.new_rule(vec![])?;
        self.grammar.rules[id as usize] = vec![vec![], vec![rule, id]];
        Ok(id)
    }

    pub(super) fn exact(&mut self, mut item: u32, mut count: usize) -> Result<u32, SchemaError> {
        let mut parts = Vec::new();
        while count != 0 {
            if count & 1 != 0 {
                parts.push(item);
            }
            count >>= 1;
            if count != 0 {
                item = self.seq(vec![item, item])?;
            }
        }
        self.seq(parts)
    }

    pub(super) fn at_most(&mut self, item: u32, count: usize) -> Result<u32, SchemaError> {
        if count == 0 {
            return self.seq(vec![]);
        }
        if count == 1 {
            return self.optional(item);
        }
        // All counts 0..N: pairs cover the even counts; a final item covers
        // the odd counts. For even N the longest pair sequence must have no
        // extra item.
        if count & 1 != 0 {
            let pair = self.seq(vec![item, item])?;
            let pairs = self.at_most(pair, count / 2)?;
            let last = self.optional(item)?;
            return self.seq(vec![pairs, last]);
        }
        let fewer = self.at_most(item, count - 1)?;
        let all = self.exact(item, count)?;
        self.alt(vec![fewer, all])
    }

    pub(super) fn generic_value(&mut self, depth: usize) -> Result<u32, SchemaError> {
        if let Some(&id) = self.generic_values.get(&depth) {
            return Ok(id);
        }
        let mut alternatives = vec![self.string, self.number, self.boolean, self.null];
        if depth != 0 {
            alternatives.push(self.generic_array(depth)?);
            alternatives.push(self.generic_object(depth)?);
        }
        let id = self.alt(alternatives)?;
        self.generic_values.insert(depth, id);
        Ok(id)
    }

    pub(super) fn generic_array(&mut self, depth: usize) -> Result<u32, SchemaError> {
        let value = self.generic_value(depth - 1)?;
        let item = self.seq(vec![self.ws, byte(b','), self.ws, value])?;
        let tail = self.repeat(item)?;
        let items = self.seq(vec![value, tail])?;
        let items = self.optional(items)?;
        self.seq(vec![byte(b'['), self.ws, items, self.ws, byte(b']')])
    }

    pub(super) fn generic_object(&mut self, depth: usize) -> Result<u32, SchemaError> {
        let value = self.generic_value(depth - 1)?;
        let member = self.seq(vec![self.string, self.ws, byte(b':'), self.ws, value])?;
        let more = self.seq(vec![self.ws, byte(b','), self.ws, member])?;
        let more = self.repeat(more)?;
        let members = self.seq(vec![member, more])?;
        let members = self.optional(members)?;
        self.seq(vec![byte(b'{'), self.ws, members, self.ws, byte(b'}')])
    }

    /// Keeps a derived schema at a fixed address (`Store`).
    pub(super) fn store(&mut self, schema: Value) -> Result<Rc<Value>, SchemaError> {
        if self.derived.len() >= MAX_RULES {
            return Err(invalid("schema expansion exceeds its resource budget"));
        }
        let stored = Rc::new(schema);
        self.derived.push(stored.clone());
        Ok(stored)
    }
}
