//! The minimal ordered JSON value of the server and tools
//! (`src/core/json.hpp`): object members keep their order, numbers are
//! `f64`, parsing rejects duplicate keys, invalid UTF-8, unpaired surrogates
//! and nesting deeper than 128, and serialisation writes numbers like
//! `std::to_chars` and escapes only quotes, backslashes and control bytes.
//! Its output and error messages are part of the HTTP contract.

use crate::text::{Error, Result};

const MAX_DEPTH: usize = 128;

#[derive(Debug, Clone, Default, PartialEq)]
pub enum Value {
    #[default]
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Self::Number(v)
    }
}

impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Self::Number(f64::from(v))
    }
}

macro_rules! from_integer {
    ($($t:ty),*) => {$(
        impl From<$t> for Value {
            fn from(v: $t) -> Self {
                Self::Number(v as f64)
            }
        }
    )*};
}
from_integer!(i32, i64, u32, u64, usize);

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&String> for Value {
    fn from(v: &String) -> Self {
        Self::String(v.clone())
    }
}

impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Self::Array(v)
    }
}

impl Value {
    pub fn object() -> Self {
        Self::Object(Vec::new())
    }

    pub fn array() -> Self {
        Self::Array(Vec::new())
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool(_))
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Self::Number(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    pub fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    /// The member for `key`, inserting null if absent; a non-object becomes
    /// an empty object first (`operator[]`).
    pub fn member_mut(&mut self, key: &str) -> &mut Value {
        if !self.is_object() {
            *self = Self::object();
        }
        let Self::Object(members) = self else {
            unreachable!("value was just made an object")
        };
        let index = match members.iter().position(|(k, _)| k == key) {
            Some(index) => index,
            None => {
                members.push((key.to_owned(), Value::Null));
                members.len() - 1
            }
        };
        &mut members[index].1
    }

    /// Sets member `key` (`value[key] = v`), keeping its position if present.
    pub fn set(&mut self, key: &str, value: impl Into<Value>) {
        *self.member_mut(key) = value.into();
    }

    /// Builder form of `set`.
    pub fn with(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.set(key, value);
        self
    }

    pub fn find(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.find(key).is_some()
    }

    /// The members of an object, empty for other values.
    pub fn members(&self) -> &[(String, Value)] {
        match self {
            Self::Object(members) => members,
            _ => &[],
        }
    }

    /// Appends a member without checking for an existing key; a non-object
    /// becomes an empty object first.
    pub fn append_member(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        if !self.is_object() {
            *self = Self::object();
        }
        if let Self::Object(members) = self {
            members.push((key.into(), value.into()));
        }
    }

    /// The string member `key`, or `default` when missing or not a string.
    pub fn member_str(&self, key: &str, default: &str) -> String {
        match self.find(key) {
            Some(Self::String(s)) => s.clone(),
            _ => default.to_owned(),
        }
    }

    pub fn member_size(&self, key: &str, default: usize) -> usize {
        match self.find(key) {
            Some(v @ Self::Number(_)) => v.as_size(default),
            _ => default,
        }
    }

    pub fn member_double(&self, key: &str, default: f64) -> f64 {
        match self.find(key) {
            Some(Self::Number(n)) => *n,
            _ => default,
        }
    }

    /// Appends to an array; a non-array becomes an empty array first.
    pub fn push(&mut self, value: impl Into<Value>) {
        if !self.is_array() {
            *self = Self::array();
        }
        if let Self::Array(items) = self {
            items.push(value.into());
        }
    }

    /// The items of an array, empty for other values.
    pub fn items(&self) -> &[Value] {
        match self {
            Self::Array(items) => items,
            _ => &[],
        }
    }

    /// Items of an array, members of an object, zero otherwise.
    pub fn len(&self) -> usize {
        match self {
            Self::Array(items) => items.len(),
            Self::Object(members) => members.len(),
            _ => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_bool(&self, default: bool) -> bool {
        match self {
            Self::Bool(b) => *b,
            _ => default,
        }
    }

    pub fn as_double(&self, default: f64) -> f64 {
        match self {
            Self::Number(n) => *n,
            _ => default,
        }
    }

    /// A nonnegative integral number below 2^64, else `default`.
    pub fn as_size(&self, default: usize) -> usize {
        match self {
            Self::Number(n)
                if n.is_finite()
                    && *n >= 0.0
                    && n.floor() == *n
                    && *n < 18_446_744_073_709_551_616.0 =>
            {
                *n as usize
            }
            _ => default,
        }
    }

    /// The string payload, empty for other values.
    pub fn str(&self) -> &str {
        match self {
            Self::String(s) => s,
            _ => "",
        }
    }

    pub fn get_str(&self, default: &str) -> String {
        match self {
            Self::String(s) => s.clone(),
            _ => default.to_owned(),
        }
    }

    /// Serialises compactly; fails on non-finite numbers.
    pub fn dump(&self) -> Result<String> {
        let mut out = String::new();
        self.dump_to(&mut out, false)?;
        Ok(out)
    }

    /// Jinja's `tojson` spelling, as Python's `json.dumps(ensure_ascii=False)`
    /// writes it: ", " between items and ": " after keys. Chat templates
    /// render non-string tool arguments this way, so models generate it.
    pub fn tojson(&self) -> Result<String> {
        let mut out = String::new();
        self.dump_to(&mut out, true)?;
        Ok(out)
    }

    fn dump_to(&self, out: &mut String, spaced: bool) -> Result<()> {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Number(n) => {
                if !n.is_finite() {
                    return Err(Error::invalid("JSON numbers must be finite"));
                }
                // Integers keep their decimal spelling, as Python's
                // json.dumps and chat templates write them. The shortest
                // round-trip form would spell 100000 as 1e+05: clients that
                // declared an integer get a float, and replayed tool calls
                // no longer match the tokens the model generated.
                const EXACT_INTEGERS: f64 = 9_007_199_254_740_992.0; // 2^53
                if n.trunc() == *n
                    && n.abs() <= EXACT_INTEGERS
                    && !(*n == 0.0 && n.is_sign_negative())
                {
                    out.push_str(&(*n as i64).to_string());
                } else {
                    out.push_str(&format_number(*n));
                }
            }
            Self::String(s) => {
                out.push('"');
                escape_into(s, out);
                out.push('"');
            }
            Self::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(if spaced { ", " } else { "," });
                    }
                    item.dump_to(out, spaced)?;
                }
                out.push(']');
            }
            Self::Object(members) => {
                out.push('{');
                for (i, (key, value)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push_str(if spaced { ", " } else { "," });
                    }
                    out.push('"');
                    escape_into(key, out);
                    out.push_str(if spaced { "\": " } else { "\":" });
                    value.dump_to(out, spaced)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
}

fn escape_into(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
}

/// `std::to_chars(double)` without a format: the shortest round-trip digits
/// in fixed or scientific notation, whichever is shorter (fixed on a tie);
/// fixed integers print their exact value, exponents have a sign and at
/// least two digits.
pub fn format_number(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    let sign = if value < 0.0 { "-" } else { "" };
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .unwrap_or((scientific.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };

    let mut sci = String::from(sign);
    sci.push_str(&digits[..1]);
    if digits.len() > 1 {
        sci.push('.');
        sci.push_str(&digits[1..]);
    }
    sci.push('e');
    sci.push(if exponent < 0 { '-' } else { '+' });
    sci.push_str(&format!("{:02}", exponent.unsigned_abs()));

    let count = digits.len() as i32;
    let mut fixed = String::from(sign);
    if exponent >= count - 1 {
        // An integer: print its exact value (it fits u128 wherever fixed can
        // win, i.e. below about 1e22).
        if exponent < 38 {
            fixed.push_str(&exact_integer(value.abs()));
        } else {
            return sci;
        }
    } else if exponent >= 0 {
        let split = (exponent + 1) as usize;
        fixed.push_str(&digits[..split]);
        fixed.push('.');
        fixed.push_str(&digits[split..]);
    } else {
        fixed.push_str("0.");
        for _ in 0..(-exponent - 1) {
            fixed.push('0');
        }
        fixed.push_str(digits);
    }
    if fixed.len() <= sci.len() { fixed } else { sci }
}

/// The exact decimal value of an integral double below 2^128.
fn exact_integer(value: f64) -> String {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1 << 52) - 1);
    let (mantissa, shift) = if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), exponent - 1075)
    };
    let mantissa = u128::from(mantissa);
    let integer = if shift >= 0 {
        mantissa << shift
    } else {
        mantissa >> (-shift)
    };
    integer.to_string()
}

/// Parses `text` (bytes, since strings are validated as UTF-8 here).
/// Errors are `runtime` with a `JSON parse error: ...` message.
pub fn parse(text: &[u8]) -> Result<Value> {
    let mut parser = Parser { s: text, i: 0 };
    let value = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.i != text.len() {
        return Err(parse_error("trailing input"));
    }
    Ok(value)
}

/// `parse` over text.
pub fn parse_str(text: &str) -> Result<Value> {
    parse(text.as_bytes())
}

fn parse_error(message: &str) -> Error {
    Error::runtime(format!("JSON parse error: {message}"))
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.s.get(self.i) {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value> {
        self.skip_ws();
        let Some(c) = self.peek() else {
            return Err(parse_error("unexpected end of input"));
        };
        if (c == b'{' || c == b'[') && depth >= MAX_DEPTH {
            return Err(parse_error("maximum nesting depth exceeded"));
        }
        match c {
            b'{' => self.parse_object(depth + 1),
            b'[' => self.parse_array(depth + 1),
            b'"' => Ok(Value::String(self.parse_string()?)),
            b't' | b'f' => self.parse_bool(),
            b'n' => self.parse_null(),
            _ => self.parse_number(),
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<Value> {
        let mut members: Vec<(String, Value)> = Vec::new();
        let mut keys = std::collections::HashSet::new();
        self.i += 1;
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(parse_error("expected object key string"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(parse_error("expected ':' after key"));
            }
            self.i += 1;
            if !keys.insert(key.clone()) {
                return Err(parse_error("duplicate object key"));
            }
            let value = self.parse_value(depth)?;
            members.push((key, value));
            self.skip_ws();
            match self.peek() {
                None => return Err(parse_error("unterminated object")),
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(members));
                }
                Some(_) => return Err(parse_error("expected ',' or '}' in object")),
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<Value> {
        let mut items = Vec::new();
        self.i += 1;
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.parse_value(depth)?);
            self.skip_ws();
            match self.peek() {
                None => return Err(parse_error("unterminated array")),
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                Some(_) => return Err(parse_error("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_hex_quad(&mut self) -> Result<u32> {
        if self.i + 4 > self.s.len() {
            return Err(parse_error("bad \\u"));
        }
        let mut code = 0;
        for _ in 0..4 {
            let h = self.s[self.i];
            self.i += 1;
            let digit = match h {
                b'0'..=b'9' => h - b'0',
                b'a'..=b'f' => h - b'a' + 10,
                b'A'..=b'F' => h - b'A' + 10,
                _ => return Err(parse_error("bad hex")),
            };
            code = (code << 4) | u32::from(digit);
        }
        Ok(code)
    }

    fn parse_string(&mut self) -> Result<String> {
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        while self.i < self.s.len() {
            let c = self.s[self.i];
            self.i += 1;
            if c == b'"' {
                // Every byte appended was validated as UTF-8 or encoded
                // from a scalar value.
                return String::from_utf8(out).map_err(|_| parse_error("invalid UTF-8"));
            }
            if c == b'\\' {
                let Some(e) = self.peek() else {
                    return Err(parse_error("bad escape"));
                };
                self.i += 1;
                match e {
                    b'"' => out.push(b'"'),
                    b'\\' => out.push(b'\\'),
                    b'/' => out.push(b'/'),
                    b'b' => out.push(8),
                    b'f' => out.push(12),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'u' => {
                        let mut code = self.parse_hex_quad()?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            if self.i + 2 > self.s.len()
                                || self.s[self.i] != b'\\'
                                || self.s[self.i + 1] != b'u'
                            {
                                return Err(parse_error("unpaired high surrogate"));
                            }
                            self.i += 2;
                            let low = self.parse_hex_quad()?;
                            if !(0xDC00..=0xDFFF).contains(&low) {
                                return Err(parse_error("invalid low surrogate"));
                            }
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                        } else if (0xDC00..=0xDFFF).contains(&code) {
                            return Err(parse_error("unpaired low surrogate"));
                        }
                        let scalar = char::from_u32(code).ok_or_else(|| parse_error("bad \\u"))?;
                        let mut buffer = [0; 4];
                        out.extend_from_slice(scalar.encode_utf8(&mut buffer).as_bytes());
                    }
                    _ => return Err(parse_error("bad escape")),
                }
                continue;
            }
            if c < 0x20 {
                return Err(parse_error("unescaped control character"));
            }
            if c < 0x80 {
                out.push(c);
                continue;
            }
            let length = match c {
                0xC2..=0xDF => 2,
                0xE0..=0xEF => 3,
                0xF0..=0xF4 => 4,
                _ => 0,
            };
            if length == 0 || length - 1 > self.s.len() - self.i {
                return Err(parse_error("invalid UTF-8"));
            }
            for j in 1..length {
                let byte = self.s[self.i + j - 1];
                let bad_first = j == 1
                    && ((c == 0xE0 && byte < 0xA0)
                        || (c == 0xED && byte > 0x9F)
                        || (c == 0xF0 && byte < 0x90)
                        || (c == 0xF4 && byte > 0x8F));
                if !(0x80..=0xBF).contains(&byte) || bad_first {
                    return Err(parse_error("invalid UTF-8"));
                }
            }
            out.extend_from_slice(&self.s[self.i - 1..self.i - 1 + length]);
            self.i += length - 1;
        }
        Err(parse_error("unterminated string"))
    }

    fn parse_bool(&mut self) -> Result<Value> {
        if self.s[self.i..].starts_with(b"true") {
            self.i += 4;
            return Ok(Value::Bool(true));
        }
        if self.s[self.i..].starts_with(b"false") {
            self.i += 5;
            return Ok(Value::Bool(false));
        }
        Err(parse_error("bad literal"))
    }

    fn parse_null(&mut self) -> Result<Value> {
        if self.s[self.i..].starts_with(b"null") {
            self.i += 4;
            return Ok(Value::Null);
        }
        Err(parse_error("bad literal"))
    }

    fn skip_digits(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
    }

    fn parse_number(&mut self) -> Result<Value> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            None => return Err(parse_error("bad number")),
            Some(b'0') => {
                self.i += 1;
                if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    return Err(parse_error("bad number"));
                }
            }
            Some(b'1'..=b'9') => self.skip_digits(),
            Some(_) => return Err(parse_error("bad number")),
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(parse_error("bad number"));
            }
            self.skip_digits();
        }
        let mantissa_end = self.i;
        if let Some(b'e' | b'E') = self.peek() {
            self.i += 1;
            if let Some(b'-' | b'+') = self.peek() {
                self.i += 1;
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(parse_error("bad number"));
            }
            self.skip_digits();
        }
        let token =
            std::str::from_utf8(&self.s[start..self.i]).map_err(|_| parse_error("bad number"))?;
        let value: f64 = token.parse().map_err(|_| parse_error("bad number"))?;
        // `std::from_chars` reports overflow and underflow to zero as out
        // of range.
        let nonzero_mantissa = self.s[start..mantissa_end]
            .iter()
            .any(|c| (b'1'..=b'9').contains(c));
        if !value.is_finite() || (value == 0.0 && nonzero_mantissa) {
            return Err(parse_error("bad number"));
        }
        Ok(Value::Number(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::ErrorKind;

    fn expect_invalid(input: &[u8]) {
        let error = parse(input).expect_err("malformed JSON is rejected");
        assert_eq!(error.kind(), ErrorKind::Runtime, "{input:?}");
    }

    #[test]
    fn numbers() {
        for number in [
            0.0,
            -0.0,
            0.123_456_789_012_345_68,
            9_007_199_254_740_991.0,
            1e-100,
            1e100,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
        ] {
            let serialized = Value::from(number).dump().expect("dump");
            let parsed = parse(serialized.as_bytes()).expect("parse").as_double(1.0);
            assert_eq!(parsed.to_bits(), number.to_bits(), "{serialized}");
        }
        for number in [-1.0, 1.5, 1e100, 2f64.powi(64), f64::INFINITY, f64::NAN] {
            assert_eq!(Value::from(number).as_size(777), 777);
        }
        assert_eq!(Value::from(0).as_size(777), 0);
        assert_eq!(Value::from(4096).as_size(777), 4096);
        let largest = f64::from_bits(2f64.powi(64).to_bits() - 1);
        assert_eq!(Value::from(largest).as_size(777), largest as usize);
        for number in [f64::INFINITY, f64::NAN] {
            let error = Value::from(number).dump().expect_err("non-finite");
            assert_eq!(error.kind(), ErrorKind::InvalidArgument);
        }
        assert_eq!(Value::from(1.25).dump().expect("dump"), "1.25");
    }

    #[test]
    fn numbers_format_like_to_chars() {
        // Pairs from libstdc++ `std::to_chars` (GCC 15).
        let cases = [
            (0.0, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (0.1, "0.1"),
            (1e21, "1e+21"),
            (1e-7, "1e-07"),
            (0.0001, "1e-04"),
            (0.001, "0.001"),
            (123_456_789.0, "123456789"),
            (1e15, "1e+15"),
            (1e16, "1e+16"),
            (1e17, "1e+17"),
            (1.5e300, "1.5e+300"),
            (5e-324, "5e-324"),
            (2.225_073_858_507_201_4e-308, "2.2250738585072014e-308"),
            (1234.5678, "1234.5678"),
            (100.0, "100"),
            (1e22, "1e+22"),
            (12_345_678_901_234_567_890.0, "12345678901234567168"),
            (0.300_000_000_000_000_04, "0.30000000000000004"),
            (1e-5, "1e-05"),
            (3.14e-10, "3.14e-10"),
            (65536.0, "65536"),
            (1e5, "1e+05"),
            (1e6, "1e+06"),
            (1e20, "1e+20"),
            (-2.5, "-2.5"),
        ];
        for (value, expected) in cases {
            assert_eq!(format_number(value), expected, "{value:e}");
        }
    }

    #[test]
    fn integers_keep_their_decimal_spelling() {
        // Python's json.dumps and chat templates spell integers in decimal,
        // not as the shortest round-trip form 1e+05.
        for (input, expected) in [
            ("100000", "100000"),
            ("-1000000", "-1000000"),
            ("1e+05", "100000"),
            ("1E5", "100000"),
            ("1.0", "1"),
            ("1e15", "1000000000000000"),
            ("9007199254740992", "9007199254740992"),
            ("-0", "-0"),
            ("0.5", "0.5"),
            ("12345.678", "12345.678"),
            ("1e16", "1e+16"),
            ("1e-7", "1e-07"),
        ] {
            let value = parse_str(input).unwrap();
            assert_eq!(value.dump().unwrap(), expected, "{input}");
            assert_eq!(value.tojson().unwrap(), expected, "{input}");
        }
        assert_eq!(Value::from(100_000).dump().unwrap(), "100000");
        assert_eq!(
            parse_str(r#"{"limit":100000}"#).unwrap().tojson().unwrap(),
            r#"{"limit": 100000}"#
        );
    }

    #[test]
    fn structures() {
        for c in 0u8..0x20 {
            expect_invalid(&[b'"', c, b'"']);
        }
        let bad: [&[u8]; 9] = [
            b"\x80",
            b"\xC0\xAF",
            b"\xE0\x80\xAF",
            b"\xED\xA0\x80",
            b"\xF4\x90\x80\x80",
            b"\xF5\x80\x80\x80",
            b"\xC2",
            b"\xE2\x82",
            b"\xC2x",
        ];
        for bytes in bad {
            expect_invalid(&[b"\"", bytes, b"\""].concat());
            expect_invalid(&[b"{\"", bytes, b"\":0}"].concat());
        }
        let utf8 = "\u{80}\u{800}\u{10FFFF}";
        assert_eq!(
            parse(format!("\"{utf8}\"").as_bytes())
                .expect("parse")
                .str(),
            utf8
        );
        for input in [
            "",
            "01",
            "1.",
            "+1",
            "1e999",
            "1e-400",
            "[1,]",
            "{\"x\":1,}",
            r#"{"x":1,"x":2}"#,
            r#"{"x":1,"x":2}"#,
            r#""\uD800""#,
            r#""\uDC00""#,
            r#""\uD800A""#,
            r#""\uD800\uD800""#,
            "true false",
        ] {
            expect_invalid(input.as_bytes());
        }
        let unicode = parse_str(r#"{"text":"😀\n\t\\\""}"#).expect("parse");
        assert_eq!(unicode.member_str("text", ""), "\u{1F600}\n\t\\\"");
        assert_eq!(
            parse_str(&unicode.dump().expect("dump"))
                .expect("parse")
                .member_str("text", ""),
            unicode.member_str("text", "")
        );
        let ordered = parse_str(r#"{"z":1,"a":[true,null,{"n":2}]}"#).expect("parse");
        assert_eq!(ordered.members()[0].0, "z");
        assert_eq!(ordered.members()[1].0, "a");
        assert_eq!(
            ordered.dump().expect("dump"),
            r#"{"z":1,"a":[true,null,{"n":2}]}"#
        );
        let bounded = format!("{}null{}", "[".repeat(128), "]".repeat(128));
        assert!(parse_str(&bounded).expect("parse").is_array());
        expect_invalid(format!("[{bounded}]").as_bytes());
        let wide = format!(
            "{{{}}}",
            (0..4096)
                .map(|i| format!("\"{i}\":{i}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let metadata = parse_str(&wide).expect("parse");
        assert_eq!(metadata.len(), 4096);
        assert_eq!(metadata.member_size("4095", 0), 4095);
    }

    /// Jinja `tojson`, as Python `json.dumps(ensure_ascii=False)` and
    /// llama.cpp's Jinja runtime write it: spaced separators, raw UTF-8,
    /// ordered keys.
    #[test]
    fn tojson_spaces_separators() {
        let ordered = parse_str(r#"{"z":1,"a":[true,null,{"n":2}]}"#).unwrap();
        assert_eq!(
            ordered.tojson().unwrap(),
            r#"{"z": 1, "a": [true, null, {"n": 2}]}"#
        );
        assert_eq!(
            parse_str(r#"{"e":[],"o":{},"t":"\u00e9\n"}"#)
                .unwrap()
                .tojson()
                .unwrap(),
            "{\"e\": [], \"o\": {}, \"t\": \"\u{e9}\\n\"}"
        );
    }

    #[test]
    fn errors_carry_the_cpp_messages() {
        let error = parse(b"{\"a\":}").expect_err("invalid");
        assert_eq!(error.message(), "JSON parse error: bad number");
        let error = parse(b"[1").expect_err("invalid");
        assert_eq!(error.message(), "JSON parse error: unterminated array");
    }

    #[test]
    fn escaping_keeps_non_ascii_and_escapes_controls() {
        let value = Value::object().with("t", "é\u{1}/\"").with("n", 3);
        assert_eq!(
            value.dump().expect("dump"),
            "{\"t\":\"é\\u0001/\\\"\",\"n\":3}"
        );
    }
}
