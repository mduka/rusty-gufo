//! An immutable Unicode DFA for the intersection of string predicates
//! (`src/core/json_schema_regex.cpp`). Regex derivatives preserve
//! intersections and lookaheads, unlike testing predicates separately
//! against the already-generated prefix. Every admitted prefix has an
//! accepting continuation within the length bounds.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;

use super::SchemaError;
use super::unicode_set::{self, CodepointSet};

type Id = u32;

/// The dead DFA state (`JsonSchemaRegex::kDead`).
pub(crate) const DEAD: Id = u32::MAX;
const INFINITE: Id = u32::MAX;
const MAX_EXPRESSIONS: usize = 32768;
const MAX_STATES: usize = 4096;
const MAX_TRANSITIONS: usize = 262_144;
const MAX_DERIVATIVES: usize = 1_048_576;

fn invalid(reason: &str) -> SchemaError {
    SchemaError::invalid(reason)
}

fn word(code_point: u32) -> bool {
    matches!(code_point, 0x61..=0x7a | 0x41..=0x5a | 0x30..=0x39 | 0x5f)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Kind {
    Empty,
    Epsilon,
    Chars,
    Start,
    Boundary,
    Or,
    And,
    Not,
    Concat,
    Repeat,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Expression {
    kind: Kind,
    children: Vec<Id>,
    low: Id,
    high: Id,
}

impl Expression {
    fn leaf(kind: Kind) -> Self {
        Self::with(kind, Vec::new(), 0, 0)
    }

    fn with(kind: Kind, children: Vec<Id>, low: Id, high: Id) -> Self {
        Self {
            kind,
            children,
            low,
            high,
        }
    }
}

/// Interned regex expressions and their derivatives (`Expressions`).
struct Expressions {
    maximum: Id,
    nodes: Vec<Expression>,
    minimums: Vec<u64>,
    ids: HashMap<Expression, Id>,
    derivatives: HashMap<(Id, u32, bool, bool), Id>,
    classes: Vec<CodepointSet>,
    word_boundaries: bool,
    empty: Id,
    epsilon: Id,
    start: Id,
    all: Id,
}

impl Expressions {
    fn new(maximum: Id) -> Result<Self, SchemaError> {
        let mut e = Self {
            maximum,
            nodes: Vec::new(),
            minimums: Vec::new(),
            ids: HashMap::new(),
            derivatives: HashMap::new(),
            classes: Vec::new(),
            word_boundaries: false,
            empty: 0,
            epsilon: 0,
            start: 0,
            all: DEAD,
        };
        e.empty = e.add(Expression::leaf(Kind::Empty))?;
        e.epsilon = e.add(Expression::leaf(Kind::Epsilon))?;
        e.start = e.add(Expression::leaf(Kind::Start))?;
        let any = e.chars(CodepointSet::scalars())?;
        e.all = e.repeat(any, 0, INFINITE)?;
        Ok(e)
    }

    fn boundary(&mut self, positive: bool) -> Result<Id, SchemaError> {
        self.word_boundaries = true;
        let mut words = CodepointSet::range(0x61, 0x7a);
        words.add(0x41, 0x5a);
        words.add(0x30, 0x39);
        words.add(0x5f, 0x5f);
        // Include word membership in the DFA alphabet.
        self.chars(words)?;
        self.add(Expression::with(
            Kind::Boundary,
            Vec::new(),
            u32::from(positive),
            0,
        ))
    }

    fn chars(&mut self, mut set: CodepointSet) -> Result<Id, SchemaError> {
        set.remove(0xd800, 0xdfff);
        if set.is_empty() {
            return Ok(self.empty);
        }
        let index = match self.classes.iter().position(|class| *class == set) {
            Some(index) => index,
            None => {
                self.classes.push(set);
                self.classes.len() - 1
            }
        };
        self.add(Expression::with(Kind::Chars, Vec::new(), index as Id, 0))
    }

    fn not(&mut self, child: Id) -> Result<Id, SchemaError> {
        if child == self.empty {
            return Ok(self.all);
        }
        if child == self.all {
            return Ok(self.empty);
        }
        if self.nodes[child as usize].kind == Kind::Not {
            return Ok(self.nodes[child as usize].children[0]);
        }
        self.add(Expression::with(Kind::Not, vec![child], 0, 0))
    }

    fn combine(&mut self, kind: Kind, parts: Vec<Id>) -> Result<Id, SchemaError> {
        let mut flat: Vec<Id> = Vec::new();
        for part in parts {
            if kind == Kind::Concat {
                if part == self.empty {
                    return Ok(self.empty);
                }
                if part == self.epsilon {
                    continue;
                }
            } else {
                let (absorbing, neutral) = if kind == Kind::Or {
                    (self.all, self.empty)
                } else {
                    (self.empty, self.all)
                };
                if part == absorbing {
                    return Ok(part);
                }
                if part == neutral {
                    continue;
                }
            }
            let node = &self.nodes[part as usize];
            if node.kind == kind {
                flat.extend_from_slice(&node.children);
            } else if kind != Kind::Concat
                || part != self.all
                || flat.last().is_none_or(|&last| last != self.all)
            {
                flat.push(part);
            }
        }
        if kind != Kind::Concat {
            flat.sort_unstable();
            flat.dedup();
            for &part in &flat {
                let node = &self.nodes[part as usize];
                if node.kind == Kind::Not && flat.binary_search(&node.children[0]).is_ok() {
                    return Ok(if kind == Kind::Or {
                        self.all
                    } else {
                        self.empty
                    });
                }
            }
        }
        match flat.len() {
            0 => Ok(match kind {
                Kind::Or => self.empty,
                Kind::And => self.all,
                _ => self.epsilon,
            }),
            1 => Ok(flat[0]),
            _ => self.add(Expression::with(kind, flat, 0, 0)),
        }
    }

    fn repeat(&mut self, mut child: Id, mut low: Id, mut high: Id) -> Result<Id, SchemaError> {
        if high < low {
            return Err(invalid("invalid regex repetition"));
        }
        if high == 0 || child == self.epsilon {
            return Ok(self.epsilon);
        }
        if child == self.empty {
            return Ok(if low != 0 { self.empty } else { self.epsilon });
        }
        if low == 1 && high == 1 {
            return Ok(child);
        }
        // Repetitions of a nullable language need only its nonempty members.
        // Missing mandatory copies can be filled with epsilon.
        if self.nullable(child, false, false, false) {
            let nonempty = self.not(self.epsilon)?;
            child = self.combine(Kind::And, vec![child, nonempty])?;
            low = 0;
        }
        let width = self.minimums[child as usize];
        if width != 0 && high != INFINITE {
            high = u64::from(high).min(u64::from(self.maximum) / width) as Id;
        }
        if high < low {
            return Ok(self.empty);
        }
        self.add(Expression::with(Kind::Repeat, vec![child], low, high))
    }

    fn nullable(&self, id: Id, at_start: bool, previous_word: bool, next_word: bool) -> bool {
        let node = &self.nodes[id as usize];
        match node.kind {
            Kind::Empty | Kind::Chars => false,
            Kind::Epsilon => true,
            Kind::Start => at_start,
            Kind::Boundary => (previous_word != next_word) == (node.low != 0),
            Kind::Not => !self.nullable(node.children[0], at_start, previous_word, next_word),
            Kind::Or => node
                .children
                .iter()
                .any(|&c| self.nullable(c, at_start, previous_word, next_word)),
            Kind::And | Kind::Concat => node
                .children
                .iter()
                .all(|&c| self.nullable(c, at_start, previous_word, next_word)),
            Kind::Repeat => node.low == 0,
        }
    }

    fn derive(
        &mut self,
        id: Id,
        code_point: u32,
        at_start: bool,
        previous_word: bool,
    ) -> Result<Id, SchemaError> {
        let key = (id, code_point, at_start, previous_word);
        if let Some(&found) = self.derivatives.get(&key) {
            return Ok(found);
        }
        // Adding nodes may move the node vector while deriving children.
        let node = self.nodes[id as usize].clone();
        let result = match node.kind {
            Kind::Empty | Kind::Epsilon | Kind::Start | Kind::Boundary => self.empty,
            Kind::Chars => {
                if self.classes[node.low as usize].contains(code_point) {
                    self.epsilon
                } else {
                    self.empty
                }
            }
            Kind::Not => {
                let derived = self.derive(node.children[0], code_point, at_start, previous_word)?;
                self.not(derived)?
            }
            Kind::Or | Kind::And => {
                let mut parts = Vec::with_capacity(node.children.len());
                for &child in &node.children {
                    parts.push(self.derive(child, code_point, at_start, previous_word)?);
                }
                self.combine(node.kind, parts)?
            }
            Kind::Concat => {
                let mut choices = Vec::new();
                for (i, &child) in node.children.iter().enumerate() {
                    let mut suffix =
                        vec![self.derive(child, code_point, at_start, previous_word)?];
                    suffix.extend_from_slice(&node.children[i + 1..]);
                    choices.push(self.combine(Kind::Concat, suffix)?);
                    if !self.nullable(child, at_start, previous_word, word(code_point)) {
                        break;
                    }
                }
                self.combine(Kind::Or, choices)?
            }
            Kind::Repeat => {
                let first = self.derive(node.children[0], code_point, at_start, previous_word)?;
                let rest = self.repeat(
                    node.children[0],
                    node.low.saturating_sub(1),
                    if node.high == INFINITE {
                        INFINITE
                    } else {
                        node.high - 1
                    },
                )?;
                self.combine(Kind::Concat, vec![first, rest])?
            }
        };
        if self.derivatives.len() >= MAX_DERIVATIVES {
            return Err(invalid("regex derivative budget exceeded"));
        }
        self.derivatives.insert(key, result);
        Ok(result)
    }

    fn add(&mut self, node: Expression) -> Result<Id, SchemaError> {
        if let Some(&found) = self.ids.get(&node) {
            return Ok(found);
        }
        if self.nodes.len() >= MAX_EXPRESSIONS {
            return Err(invalid("regex expression budget exceeded"));
        }
        let infinite = u64::from(INFINITE);
        let child_minimum = |child: &Id| self.minimums[*child as usize];
        let minimum = match node.kind {
            Kind::Empty => infinite,
            Kind::Chars => 1,
            Kind::Or => node
                .children
                .iter()
                .map(child_minimum)
                .fold(infinite, u64::min),
            Kind::Concat => node
                .children
                .iter()
                .map(child_minimum)
                .fold(0, |sum, child| infinite.min(sum + child)),
            Kind::And => node.children.iter().map(child_minimum).fold(0, u64::max),
            Kind::Repeat => infinite.min(child_minimum(&node.children[0]) * u64::from(node.low)),
            _ => 0,
        };
        if !self.nodes.is_empty() && minimum > u64::from(self.maximum) {
            return Ok(self.empty);
        }
        let id = self.nodes.len() as Id;
        self.ids.insert(node.clone(), id);
        self.nodes.push(node);
        self.minimums.push(minimum);
        Ok(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyntaxKind {
    Atom,
    Sequence,
    Alternative,
    Repeat,
    Positive,
    Negative,
    End,
}

#[derive(Debug, Clone)]
struct Syntax {
    kind: SyntaxKind,
    atom: Id,
    low: Id,
    high: Id,
    assertion: bool,
    children: Vec<Syntax>,
}

impl Syntax {
    fn new(kind: SyntaxKind) -> Self {
        Self {
            kind,
            atom: 0,
            low: 0,
            high: 0,
            assertion: false,
            children: Vec::new(),
        }
    }
}

/// The pattern parser (`Parser`), over code points. Past the end `peek`
/// returns U+FFFF, as `UnicodeString::char32At` does.
struct Parser<'e> {
    e: &'e mut Expressions,
    text: Vec<u32>,
    position: usize,
}

const END: u32 = 0xffff;

impl<'e> Parser<'e> {
    fn new(expressions: &'e mut Expressions, pattern: &str) -> Result<Self, SchemaError> {
        if pattern.len() > 16384 {
            return Err(invalid("regex exceeds 16384 bytes"));
        }
        Ok(Self {
            e: expressions,
            text: pattern.chars().map(u32::from).collect(),
            position: 0,
        })
    }

    fn parse(&mut self) -> Result<Id, SchemaError> {
        let syntax = self.alternatives(0)?;
        if self.position != self.text.len() {
            return Err(invalid("unbalanced regex parentheses"));
        }
        let all = self.e.all;
        let body = self.expand(&syntax, all)?;
        self.e.combine(Kind::Concat, vec![all, body])
    }

    fn peek(&self) -> u32 {
        self.at(self.position)
    }

    fn at(&self, index: usize) -> u32 {
        self.text.get(index).copied().unwrap_or(END)
    }

    fn take(&mut self) -> Result<u32, SchemaError> {
        if self.position >= self.text.len() {
            return Err(invalid("truncated regex"));
        }
        let code_point = self.peek();
        self.position += 1;
        Ok(code_point)
    }

    fn count(&mut self) -> Result<Id, SchemaError> {
        if !(0x30..=0x39).contains(&self.peek()) {
            return Err(invalid("invalid regex repetition"));
        }
        let mut value = 0u64;
        while (0x30..=0x39).contains(&self.peek()) {
            value = value * 10 + u64::from(self.take()? - 0x30);
            if value >= u64::from(INFINITE) {
                return Err(invalid("regex repetition is too large"));
            }
        }
        Ok(value as Id)
    }

    fn hex(&mut self, count: usize) -> Result<u32, SchemaError> {
        let mut value = 0u32;
        for _ in 0..count {
            let digit = char::from_u32(self.take()?)
                .and_then(|c| c.to_digit(16))
                .ok_or_else(|| invalid("invalid regex Unicode escape"))?;
            value = (value << 4) | digit;
        }
        Ok(value)
    }

    fn escape(&mut self, in_class: bool) -> Result<CodepointSet, SchemaError> {
        let mut code_point = self.take()?;
        let letter = char::from_u32(code_point).unwrap_or('\0');
        let property = match letter {
            'd' | 'D' => Some(CodepointSet::range(0x30, 0x39)),
            's' | 'S' => {
                // ECMA-262 WhiteSpace + LineTerminator, not ICU White_Space.
                let mut set = unicode_set::property("Zs")
                    .ok_or_else(|| invalid("invalid regex Unicode property"))?;
                set.add(0x09, 0x0d);
                set.add(0x2028, 0x2029);
                set.add(0xfeff, 0xfeff);
                Some(set)
            }
            'w' | 'W' => {
                let mut set = CodepointSet::range(0x61, 0x7a);
                set.add(0x41, 0x5a);
                set.add(0x30, 0x39);
                set.add(0x5f, 0x5f);
                Some(set)
            }
            'p' | 'P' => {
                if self.take()? != u32::from('{') {
                    return Err(invalid("Unicode property escape requires braces"));
                }
                let mut name = String::new();
                while self.peek() != u32::from('}') {
                    let c = self.take()?;
                    if c > 127 || name.len() >= 128 {
                        return Err(invalid("invalid Unicode property name"));
                    }
                    name.push(char::from(c as u8));
                }
                self.position += 1;
                let mut set = unicode_set::property(&name)
                    .ok_or_else(|| invalid("invalid regex Unicode property"))?;
                if letter == 'P' {
                    set.complement();
                }
                Some(set)
            }
            'n' => {
                code_point = 0x0a;
                None
            }
            'r' => {
                code_point = 0x0d;
                None
            }
            't' => {
                code_point = 0x09;
                None
            }
            'f' => {
                code_point = 0x0c;
                None
            }
            'v' => {
                code_point = 0x0b;
                None
            }
            'b' => {
                if !in_class {
                    return Err(invalid("word-boundary regex assertions are unsupported"));
                }
                code_point = 0x08;
                None
            }
            '0' => {
                if (0x30..=0x39).contains(&self.peek()) {
                    return Err(invalid("legacy octal regex escapes are unsupported"));
                }
                code_point = 0;
                None
            }
            'c' => {
                code_point = self.take()?;
                if !char::from_u32(code_point).is_some_and(|c| c.is_ascii_alphabetic()) {
                    return Err(invalid("regex control escape requires an ASCII letter"));
                }
                code_point %= 32;
                None
            }
            'u' => {
                if self.peek() == u32::from('{') {
                    self.position += 1;
                    code_point = 0;
                    let mut digits = 0;
                    while self.peek() != u32::from('}') {
                        digits += 1;
                        code_point = (code_point << 4) | self.hex(1)?;
                        if code_point > unicode_set::MAX_CODE_POINT {
                            return Err(invalid("invalid regex Unicode code point"));
                        }
                    }
                    self.position += 1;
                    if digits == 0 {
                        return Err(invalid("empty regex Unicode code point"));
                    }
                } else {
                    code_point = self.hex(4)?;
                    if (0xd800..=0xdbff).contains(&code_point) {
                        if self.take()? != u32::from('\\') || self.take()? != u32::from('u') {
                            return Err(invalid("unpaired surrogate in regex"));
                        }
                        let low = self.hex(4)?;
                        if !(0xdc00..=0xdfff).contains(&low) {
                            return Err(invalid("unpaired surrogate in regex"));
                        }
                        code_point = 0x10000 + ((code_point - 0xd800) << 10) + low - 0xdc00;
                    }
                }
                None
            }
            'x' => {
                code_point = self.hex(2)?;
                None
            }
            _ => {
                let syntax = code_point <= 127 && "^$\\.*+?()[]{}|/".contains(letter);
                if !syntax && !(in_class && letter == '-') {
                    return Err(invalid("unsupported regex escape or backreference"));
                }
                None
            }
        };
        let Some(mut set) = property else {
            return Ok(CodepointSet::single(code_point));
        };
        if matches!(letter, 'D' | 'S' | 'W') {
            set.complement();
        }
        Ok(set)
    }

    fn class(&mut self) -> Result<CodepointSet, SchemaError> {
        // Parse range endpoints explicitly: whitespace is a literal here, and
        // regex shorthands differ from ICU UnicodeSet shorthand syntax.
        let mut result = CodepointSet::default();
        let negate = self.peek() == u32::from('^');
        if negate {
            self.position += 1;
        }
        while self.peek() != u32::from(']') {
            let code_point = self.take()?;
            let mut chars = if code_point == u32::from('\\') {
                self.escape(true)?
            } else {
                CodepointSet::single(code_point)
            };
            if self.peek() == u32::from('-') && self.at(self.position + 1) != u32::from(']') {
                self.position += 1;
                let code_point = self.take()?;
                let end = if code_point == u32::from('\\') {
                    self.escape(true)?
                } else {
                    CodepointSet::single(code_point)
                };
                match (chars.first(), end.first()) {
                    (Some(first), Some(last))
                        if chars.size() == 1 && end.size() == 1 && first <= last =>
                    {
                        chars.add(first, last);
                    }
                    _ => return Err(invalid("invalid regex character range")),
                }
            }
            result.add_all(&chars);
        }
        self.position += 1;
        if negate {
            result.complement();
        }
        Ok(result)
    }

    fn alternatives(&mut self, depth: u32) -> Result<Syntax, SchemaError> {
        if depth > 32 {
            return Err(invalid("regex nesting exceeds 32 levels"));
        }
        let mut alternatives = Syntax::new(SyntaxKind::Alternative);
        loop {
            let mut sequence = Syntax::new(SyntaxKind::Sequence);
            while self.position < self.text.len()
                && self.peek() != u32::from(')')
                && self.peek() != u32::from('|')
            {
                let code_point = self.take()?;
                let mut atom = Syntax::new(SyntaxKind::Atom);
                match char::from_u32(code_point).unwrap_or('\0') {
                    '(' => {
                        let (mut positive, mut negative) = (false, false);
                        if self.peek() == u32::from('?') {
                            self.position += 1;
                            let modifier = self.take()?;
                            positive = modifier == u32::from('=');
                            negative = modifier == u32::from('!');
                            if !positive && !negative && modifier != u32::from(':') {
                                return Err(invalid(
                                    "unsupported regex group, lookbehind or inline flag",
                                ));
                            }
                        }
                        atom = self.alternatives(depth + 1)?;
                        if self.take()? != u32::from(')') {
                            return Err(invalid("unbalanced regex parentheses"));
                        }
                        if positive || negative {
                            let mut assertion = Syntax::new(if positive {
                                SyntaxKind::Positive
                            } else {
                                SyntaxKind::Negative
                            });
                            assertion.assertion = true;
                            assertion.children.push(atom);
                            atom = assertion;
                        }
                    }
                    '[' => {
                        let class = self.class()?;
                        atom.atom = self.e.chars(class)?;
                    }
                    '^' => {
                        atom.atom = self.e.start;
                        atom.assertion = true;
                    }
                    '$' => {
                        atom.kind = SyntaxKind::End;
                        atom.assertion = true;
                        atom.atom = self.e.epsilon;
                    }
                    '\\' => {
                        if self.peek() == u32::from('b') || self.peek() == u32::from('B') {
                            let positive = self.take()? == u32::from('b');
                            atom.atom = self.e.boundary(positive)?;
                            atom.assertion = true;
                        } else {
                            let set = self.escape(false)?;
                            atom.atom = self.e.chars(set)?;
                        }
                    }
                    '.' => {
                        let mut chars = CodepointSet::scalars();
                        chars.remove(0x0a, 0x0a);
                        chars.remove(0x0d, 0x0d);
                        chars.remove(0x2028, 0x2029);
                        atom.atom = self.e.chars(chars)?;
                    }
                    other => {
                        if code_point < 128 && "*+?{}".contains(other) {
                            return Err(invalid("regex repetition has no preceding atom"));
                        }
                        atom.atom = self.e.chars(CodepointSet::single(code_point))?;
                    }
                }
                let quantifier = self.peek();
                if [
                    u32::from('*'),
                    u32::from('+'),
                    u32::from('?'),
                    u32::from('{'),
                ]
                .contains(&quantifier)
                {
                    self.position += 1;
                    let mut repeated = Syntax::new(SyntaxKind::Repeat);
                    repeated.low = Id::from(quantifier == u32::from('+'));
                    repeated.high = if quantifier == u32::from('?') {
                        1
                    } else {
                        INFINITE
                    };
                    if quantifier == u32::from('{') {
                        repeated.low = self.count()?;
                        repeated.high = repeated.low;
                        if self.peek() == u32::from(',') {
                            self.position += 1;
                            repeated.high = if self.peek() == u32::from('}') {
                                INFINITE
                            } else {
                                self.count()?
                            };
                        }
                        if self.take()? != u32::from('}') || repeated.high < repeated.low {
                            return Err(invalid("invalid regex repetition"));
                        }
                    }
                    if atom.assertion && (repeated.high == INFINITE || repeated.high > 32) {
                        return Err(invalid(
                            "regex assertions in unbounded or large repetitions are unsupported",
                        ));
                    }
                    repeated.assertion = atom.assertion;
                    repeated.children.push(atom);
                    atom = repeated;
                    if self.peek() == u32::from('?') {
                        self.position += 1;
                    }
                }
                sequence.assertion |= atom.assertion;
                sequence.children.push(atom);
            }
            alternatives.assertion |= sequence.assertion;
            alternatives.children.push(sequence);
            if self.peek() != u32::from('|') {
                break;
            }
            self.position += 1;
        }
        Ok(alternatives)
    }

    fn expand(&mut self, node: &Syntax, mut tail: Id) -> Result<Id, SchemaError> {
        match node.kind {
            SyntaxKind::Atom => self.e.combine(Kind::Concat, vec![node.atom, tail]),
            SyntaxKind::Sequence => {
                for child in node.children.iter().rev() {
                    tail = self.expand(child, tail)?;
                }
                Ok(tail)
            }
            SyntaxKind::Alternative => {
                let mut choices = Vec::with_capacity(node.children.len());
                for child in &node.children {
                    choices.push(self.expand(child, tail)?);
                }
                self.e.combine(Kind::Or, choices)
            }
            SyntaxKind::Repeat => {
                if node.assertion {
                    let mut choices = Vec::new();
                    for count in 0..=node.high {
                        if count >= node.low {
                            choices.push(tail);
                        }
                        if count < node.high {
                            tail = self.expand(&node.children[0], tail)?;
                        }
                    }
                    return self.e.combine(Kind::Or, choices);
                }
                let epsilon = self.e.epsilon;
                let body = self.expand(&node.children[0], epsilon)?;
                let repeated = self.e.repeat(body, node.low, node.high)?;
                self.e.combine(Kind::Concat, vec![repeated, tail])
            }
            SyntaxKind::Positive | SyntaxKind::Negative => {
                let all = self.e.all;
                let mut condition = self.expand(&node.children[0], all)?;
                if node.kind == SyntaxKind::Negative {
                    condition = self.e.not(condition)?;
                }
                self.e.combine(Kind::And, vec![condition, tail])
            }
            SyntaxKind::End => self.e.combine(Kind::And, vec![node.atom, tail]),
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    accepting: bool,
    next: Vec<Id>,
    successors: Vec<Id>,
    distance: Id,
}

/// The compiled DFA (`JsonSchemaRegex`). The alphabet partitions Unicode by
/// membership in every character class of the patterns.
#[derive(Debug)]
pub(crate) struct JsonSchemaRegex {
    alphabet: Vec<CodepointSet>,
    /// Every partition range, sorted, with its partition index.
    lookup: Vec<(u32, u32, u32)>,
    states: Vec<State>,
    maximum_suffix: Id,
}

impl JsonSchemaRegex {
    /// The intersection of `patterns`, admitting at most `maximum` code
    /// points.
    pub(crate) fn compile(patterns: &[String], maximum: Id) -> Result<Arc<Self>, SchemaError> {
        let mut expressions = Expressions::new(maximum)?;
        let mut conditions = Vec::with_capacity(patterns.len());
        for pattern in patterns {
            conditions.push(Parser::new(&mut expressions, pattern)?.parse()?);
        }
        let root = expressions.combine(Kind::And, conditions)?;

        // Partition Unicode once by membership in all character classes.
        // Every code point in a partition has the same derivative for every
        // expression.
        let mut endpoints: Vec<u32> = vec![0, 0xd800, 0xe000, 0x110000];
        for chars in &expressions.classes {
            for &(first, last) in chars.ranges() {
                endpoints.push(first);
                endpoints.push(last + 1);
            }
        }
        endpoints.sort_unstable();
        endpoints.dedup();
        let mut alphabet: Vec<CodepointSet> = Vec::new();
        let mut partitions: HashMap<Vec<Id>, Id> = HashMap::new();
        for pair in endpoints.windows(2) {
            let (first, last) = (pair[0], pair[1] - 1);
            if (0xd800..=0xdfff).contains(&first) {
                continue;
            }
            let membership: Vec<Id> = expressions
                .classes
                .iter()
                .enumerate()
                .filter(|(_, class)| class.contains(first))
                .map(|(c, _)| c as Id)
                .collect();
            let next = alphabet.len() as Id;
            let index = *partitions.entry(membership).or_insert(next);
            if index == next {
                alphabet.push(CodepointSet::default());
            }
            alphabet[index as usize].add(first, last);
        }

        type Key = (Id, bool, bool);
        let mut ids: HashMap<Key, Id> = HashMap::new();
        let mut pending: Vec<Key> = Vec::new();
        let mut depths: Vec<Id> = Vec::new();
        let mut states: Vec<State> = Vec::new();
        let mut intern = |key: Key,
                          depth: Id,
                          pending: &mut Vec<Key>,
                          depths: &mut Vec<Id>,
                          states: &mut Vec<State>|
         -> Result<Id, SchemaError> {
            if let Some(&found) = ids.get(&key) {
                return Ok(found);
            }
            let id = ids.len() as Id;
            ids.insert(key, id);
            if ids.len() > MAX_STATES || ids.len() * alphabet.len() > MAX_TRANSITIONS {
                return Err(invalid("compiled regex exceeds the state budget"));
            }
            pending.push(key);
            depths.push(depth);
            states.push(State {
                accepting: false,
                next: Vec::new(),
                successors: Vec::new(),
                distance: DEAD,
            });
            Ok(id)
        };
        intern(
            (root, true, false),
            0,
            &mut pending,
            &mut depths,
            &mut states,
        )?;
        let mut i = 0;
        while i < pending.len() {
            let (expression, at_start, previous_word) = pending[i];
            states[i].accepting = expressions.nullable(expression, at_start, previous_word, false);
            // BFS discovers each state at its minimum reachable length.
            // Beyond maxLength no transition can contribute to an accepted
            // value.
            if depths[i] >= maximum {
                states[i].next = vec![DEAD; alphabet.len()];
                i += 1;
                continue;
            }
            let mut next_states = Vec::with_capacity(alphabet.len());
            for chars in &alphabet {
                let representative = chars.first().unwrap_or(0);
                let derivative =
                    expressions.derive(expression, representative, at_start, previous_word)?;
                let next = if derivative == expressions.empty {
                    DEAD
                } else {
                    let key = (
                        derivative,
                        false,
                        expressions.word_boundaries && word(representative),
                    );
                    intern(key, depths[i] + 1, &mut pending, &mut depths, &mut states)?
                };
                next_states.push(next);
            }
            states[i].next = next_states;
            i += 1;
        }

        let mut predecessors: Vec<Vec<Id>> = vec![Vec::new(); states.len()];
        let mut queue: VecDeque<Id> = VecDeque::new();
        for (i, state) in states.iter_mut().enumerate() {
            if state.accepting {
                state.distance = 0;
                queue.push_back(i as Id);
            }
            let unique: BTreeSet<Id> = state.next.iter().copied().filter(|&n| n != DEAD).collect();
            state.successors = unique.iter().copied().collect();
            for &next in &unique {
                predecessors[next as usize].push(i as Id);
            }
        }
        while let Some(state) = queue.pop_front() {
            let distance = states[state as usize].distance;
            for &previous in &predecessors[state as usize] {
                if states[previous as usize].distance == DEAD {
                    states[previous as usize].distance = distance + 1;
                    queue.push_back(previous);
                }
            }
        }
        let distances: Vec<Id> = states.iter().map(|state| state.distance).collect();
        for state in &mut states {
            for next in &mut state.next {
                if *next != DEAD && distances[*next as usize] == DEAD {
                    *next = DEAD;
                }
            }
        }
        let maximum_suffix = distances
            .iter()
            .copied()
            .filter(|&d| d != DEAD)
            .max()
            .unwrap_or(0);
        let mut lookup: Vec<(u32, u32, u32)> = alphabet
            .iter()
            .enumerate()
            .flat_map(|(index, chars)| {
                chars
                    .ranges()
                    .iter()
                    .map(move |&(first, last)| (first, last, index as u32))
            })
            .collect();
        lookup.sort_unstable();
        Ok(Arc::new(Self {
            alphabet,
            lookup,
            states,
            maximum_suffix,
        }))
    }

    pub(crate) fn start(&self) -> Id {
        0
    }

    pub(crate) fn accepting(&self, state: Id) -> bool {
        state != DEAD && self.states[state as usize].accepting
    }

    pub(crate) fn maximum_suffix(&self) -> Id {
        self.maximum_suffix
    }

    pub(crate) fn advance(&self, state: Id, code_point: u32) -> Id {
        if state == DEAD {
            return DEAD;
        }
        let index = self
            .lookup
            .partition_point(|&(_, last, _)| last < code_point);
        match self.lookup.get(index) {
            Some(&(first, _, partition)) if first <= code_point => {
                self.states[state as usize].next[partition as usize]
            }
            _ => DEAD,
        }
    }

    /// Whether some accepted value continues `state` with between `minimum`
    /// and `maximum` more code points.
    pub(crate) fn can_finish(&self, state: Id, minimum: Id, maximum: Id) -> bool {
        if state == DEAD || minimum > maximum {
            return false;
        }
        let distance = self.states[state as usize].distance;
        if distance == DEAD || distance > maximum {
            return false;
        }
        if distance >= minimum {
            return true;
        }
        let mut active: Vec<Id> = vec![state];
        let mut anchor = active.clone();
        let mut reachable = vec![false; self.states.len()];
        let (mut steps, mut block, mut period) = (0u32, 1u32, 0u32);
        let mut cycle = false;
        while steps < minimum {
            reachable.fill(false);
            for &current in &active {
                for &successor in &self.states[current as usize].successors {
                    if self.states[successor as usize].distance != DEAD {
                        reachable[successor as usize] = true;
                    }
                }
            }
            active.clear();
            active.extend(
                reachable
                    .iter()
                    .enumerate()
                    .filter(|&(_, &r)| r)
                    .map(|(n, _)| n as Id),
            );
            if active.is_empty() {
                return false;
            }
            steps += 1;
            if !cycle {
                period += 1;
                if active == anchor {
                    // Brent cycle detection bounds scratch memory even for
                    // very large minLength. Whole periods preserve the set of
                    // possible DFA states.
                    steps += ((minimum - steps) / period) * period;
                    cycle = true;
                } else if period == block {
                    anchor.clone_from(&active);
                    block = block.wrapping_mul(2);
                    period = 0;
                }
            }
        }
        active
            .iter()
            .any(|&current| self.states[current as usize].distance <= maximum - minimum)
    }

    /// Whether a code point of `first..=last` continues `state` towards an
    /// accepted value within the bounds.
    pub(crate) fn can_advance(
        &self,
        state: Id,
        first: u32,
        last: u32,
        minimum: Id,
        maximum: Id,
    ) -> bool {
        if state == DEAD || first > last {
            return false;
        }
        let row = &self.states[state as usize];
        self.alphabet.iter().enumerate().any(|(i, chars)| {
            row.next[i] != DEAD
                && chars.contains_some(first, last)
                && self.can_finish(row.next[i], minimum, maximum)
        })
    }
}
