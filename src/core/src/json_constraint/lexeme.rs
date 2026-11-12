//! Incremental constraints on a single primitive
//! (`src/core/json_schema_lexeme.cpp`): bounded whitespace, numbers under
//! bounds and `multipleOf`, and strings under lengths, patterns and formats.
//! A complete value can also be a prefix (e.g. `1` of `1.5`); the grammar
//! keeps both continuations. The state is request-local bytes, compared and
//! ordered like the C++ `std::string`.

use std::sync::{Arc, OnceLock};

use super::SchemaError;
use super::regex::JsonSchemaRegex;
use crate::json::{self, Value};

/// Bytes a lexeme state keeps inline.
const INLINE: usize = 22;

/// A lexeme's request-local bytes, compared, ordered and hashed as bytes.
/// Up to 22 bytes live inline, like a `std::string` in its small buffer,
/// so copying a whitespace count, a short number or a string's DFA state
/// allocates nothing.
#[derive(Clone)]
pub(crate) enum LexemeState {
    Inline { len: u8, bytes: [u8; INLINE] },
    Heap(Vec<u8>),
}

impl Default for LexemeState {
    fn default() -> Self {
        Self::Inline {
            len: 0,
            bytes: [0; INLINE],
        }
    }
}

impl From<&[u8]> for LexemeState {
    fn from(value: &[u8]) -> Self {
        if value.len() > INLINE {
            return Self::Heap(value.to_vec());
        }
        let mut bytes = [0; INLINE];
        bytes[..value.len()].copy_from_slice(value);
        Self::Inline {
            len: value.len() as u8,
            bytes,
        }
    }
}

impl LexemeState {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Inline { len, bytes } => &bytes[..usize::from(*len)],
            Self::Heap(bytes) => bytes,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    fn set(&mut self, value: &[u8]) {
        *self = Self::from(value);
    }

    fn push(&mut self, byte: u8) {
        match self {
            Self::Inline { len, bytes } if usize::from(*len) < INLINE => {
                bytes[usize::from(*len)] = byte;
                *len += 1;
            }
            Self::Inline { len, bytes } => {
                let mut heap = Vec::with_capacity(2 * INLINE);
                heap.extend_from_slice(&bytes[..usize::from(*len)]);
                heap.push(byte);
                *self = Self::Heap(heap);
            }
            Self::Heap(bytes) => bytes.push(byte),
        }
    }
}

impl PartialEq for LexemeState {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for LexemeState {}

impl PartialOrd for LexemeState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LexemeState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl std::hash::Hash for LexemeState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl std::fmt::Debug for LexemeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// A byte's effect on a lexeme (`JsonSchemaLexeme::Match`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Match {
    pub(crate) prefix: bool,
    pub(crate) complete: bool,
}

impl Match {
    const REJECT: Self = Self {
        prefix: false,
        complete: false,
    };

    fn new(prefix: bool, complete: bool) -> Self {
        Self { prefix, complete }
    }
}

fn invalid(reason: &str) -> SchemaError {
    SchemaError::invalid(reason)
}

/// Exact finite-decimal arithmetic (`Decimal`). JSON Schema `multipleOf`
/// must not rely on a floating-point remainder (0.3 is a multiple of 0.1).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Decimal {
    /// ASCII digits without leading zeros; `0` for zero.
    digits: Vec<u8>,
    scale: i32,
    negative: bool,
}

impl Decimal {
    fn new(digits: &[u8], scale: i32) -> Self {
        Self {
            digits: digits.to_vec(),
            scale,
            negative: false,
        }
    }

    fn is_zero(&self) -> bool {
        self.digits == b"0"
    }

    fn parse(text: &[u8]) -> Result<Self, SchemaError> {
        let mut value = Self::new(b"", 0);
        let mut text = text;
        if let [b'-', rest @ ..] = text {
            value.negative = true;
            text = rest;
        }
        let mut fraction = false;
        let mut i = 0;
        while i < text.len() && text[i] != b'e' && text[i] != b'E' {
            if text[i] == b'.' {
                fraction = true;
            } else {
                value.digits.push(text[i]);
                value.scale += i32::from(fraction);
            }
            i += 1;
        }
        if i < text.len() {
            i += 1;
            if text.get(i) == Some(&b'+') {
                i += 1;
            }
            value.scale = parse_exponent(&text[i..])
                .and_then(|exponent| value.scale.checked_sub(exponent))
                .ok_or_else(|| invalid("numeric exponent is outside the supported range"))?;
        }
        value.normalize();
        Ok(value)
    }

    fn normalize(&mut self) {
        let Some(first) = self.digits.iter().position(|&d| d != b'0') else {
            self.digits = b"0".to_vec();
            self.scale = 0;
            self.negative = false;
            return;
        };
        self.digits.drain(..first);
        while self.digits.len() > 1 && self.digits.last() == Some(&b'0') {
            self.digits.pop();
            self.scale -= 1;
        }
    }
}

/// The leading `[-]digits` of `text` as an `int` (`std::from_chars`).
fn parse_exponent(text: &[u8]) -> Option<i32> {
    let (negative, digits) = match text {
        [b'-', rest @ ..] => (true, rest),
        _ => (false, text),
    };
    let length = digits.iter().take_while(|d| d.is_ascii_digit()).count();
    if length == 0 {
        return None;
    }
    let mut value: i64 = 0;
    for &digit in &digits[..length] {
        value = value * 10 + i64::from(digit - b'0');
        if value > i64::from(i32::MAX) + 1 {
            return None;
        }
    }
    i32::try_from(if negative { -value } else { value }).ok()
}

fn magnitude(a: &Decimal, b: &Decimal) -> i32 {
    if a.is_zero() || b.is_zero() {
        return i32::from(!a.is_zero()) - i32::from(!b.is_zero());
    }
    let ae = a.digits.len() as i64 - i64::from(a.scale);
    let be = b.digits.len() as i64 - i64::from(b.scale);
    if ae != be {
        return if ae < be { -1 } else { 1 };
    }
    for i in 0..a.digits.len().max(b.digits.len()) {
        let x = a.digits.get(i).copied().unwrap_or(b'0');
        let y = b.digits.get(i).copied().unwrap_or(b'0');
        if x != y {
            return if x < y { -1 } else { 1 };
        }
    }
    0
}

fn compare(a: &Decimal, b: &Decimal) -> i32 {
    if a.negative != b.negative {
        return if a.negative { -1 } else { 1 };
    }
    (if a.negative { -1 } else { 1 }) * magnitude(a, b)
}

/// Compares nonnegative integers without leading zeros.
fn natural_compare(a: &[u8], b: &[u8]) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

fn subtract(a: &mut Vec<u8>, b: &[u8]) {
    let mut borrow = 0i32;
    for i in 0..a.len() {
        let index = a.len() - 1 - i;
        let mut digit = i32::from(a[index] - b'0') - borrow;
        if i < b.len() {
            digit -= i32::from(b[b.len() - 1 - i] - b'0');
        }
        borrow = i32::from(digit < 0);
        a[index] = b'0' + (digit + 10 * borrow) as u8;
    }
    match a.iter().position(|&d| d != b'0') {
        Some(first) => {
            a.drain(..first);
        }
        None => *a = b"0".to_vec(),
    }
}

/// The integer quotient and remainder of `value / divisor` on their common
/// scale (`Divide`).
fn divide(value: &Decimal, divisor: &Decimal) -> (Vec<u8>, Vec<u8>) {
    let scale = value.scale.max(divisor.scale);
    let mut numerator = value.digits.clone();
    numerator.resize(numerator.len() + (scale - value.scale) as usize, b'0');
    let mut denominator = divisor.digits.clone();
    denominator.resize(denominator.len() + (scale - divisor.scale) as usize, b'0');
    let mut remainder = b"0".to_vec();
    let mut quotient = Vec::new();
    for digit in numerator {
        if remainder == b"0" {
            remainder.clear();
        }
        remainder.push(digit);
        let mut count = 0u8;
        while natural_compare(&remainder, &denominator).is_ge() {
            subtract(&mut remainder, &denominator);
            count += 1;
        }
        if count != 0 || !quotient.is_empty() {
            quotient.push(b'0' + count);
        }
    }
    if quotient.is_empty() {
        quotient = b"0".to_vec();
    }
    (quotient, remainder)
}

fn multiple(value: &Decimal, divisor: &Decimal) -> bool {
    value.is_zero() || divide(value, divisor).1 == b"0"
}

fn increment(mut value: Vec<u8>) -> Vec<u8> {
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

fn product(value: &Decimal, factor: &[u8]) -> Decimal {
    let mut digits = vec![0u32; value.digits.len() + factor.len()];
    for (i, &a) in value.digits.iter().enumerate() {
        for (j, &b) in factor.iter().enumerate() {
            digits[i + j + 1] += u32::from(a - b'0') * u32::from(b - b'0');
        }
    }
    for i in (1..digits.len()).rev() {
        digits[i - 1] += digits[i] / 10;
        digits[i] %= 10;
    }
    let mut result = Decimal::new(b"", value.scale);
    result.digits = digits.iter().map(|&d| b'0' + d as u8).collect();
    result.normalize();
    result
}

fn add_unit(value: &Decimal, scale: i32) -> Decimal {
    let mut next = value.clone();
    next.negative = false;
    if next.is_zero() {
        return Decimal::new(b"1", scale);
    }
    next.digits
        .resize(next.digits.len() + (scale - next.scale) as usize, b'0');
    next.scale = scale;
    let mut carry = true;
    for digit in next.digits.iter_mut().rev() {
        if !carry {
            break;
        }
        carry = *digit == b'9';
        *digit = if carry { b'0' } else { *digit + 1 };
    }
    if carry {
        next.digits.insert(0, b'1');
    }
    next.normalize();
    next
}

/// Bounded whitespace between tokens (`WhitespaceLexeme`): at most 32 bytes
/// of formatting, never inside a JSON string. Every JSON value still has an
/// admitted (compact) serialization.
fn whitespace_advance(state: &mut LexemeState, byte: u8) -> Match {
    let count = state.as_slice().first().copied().unwrap_or(0);
    if count >= 32 || !matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
        return Match::REJECT;
    }
    state.set(&[count + 1]);
    Match::new(count + 1 < 32, true)
}

#[derive(Debug, Clone)]
struct Bound {
    value: Decimal,
    exclusive: bool,
}

/// A number under bounds and `multipleOf` (`NumberLexeme`); the state is the
/// number's text.
#[derive(Debug)]
pub(crate) struct NumberLexeme {
    integer: bool,
    lower: Option<Bound>,
    upper: Option<Bound>,
    multiple: Option<Decimal>,
}

impl NumberLexeme {
    fn new(schema: &Value, integer: bool) -> Result<Self, SchemaError> {
        let mut lexeme = Self {
            integer,
            lower: None,
            upper: None,
            multiple: None,
        };
        for key in [
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        ] {
            let Some(entry) = schema.find(key) else {
                continue;
            };
            if !entry.is_number() || !entry.as_double(0.0).is_finite() {
                return Err(invalid(&format!("{key} must be a finite number")));
            }
            let value = Decimal::parse(dump(entry)?.as_bytes())?;
            if key == "multipleOf" {
                if value.negative || value.is_zero() {
                    return Err(invalid("multipleOf must be positive"));
                }
                lexeme.multiple = Some(value);
                continue;
            }
            let lower = key == "minimum" || key == "exclusiveMinimum";
            let bound = if lower {
                &mut lexeme.lower
            } else {
                &mut lexeme.upper
            };
            let exclusive = key.starts_with("exclusive");
            let comparison = bound.as_ref().map_or(0, |b| compare(&value, &b.value));
            if bound.is_none()
                || (if lower {
                    comparison > 0
                } else {
                    comparison < 0
                })
                || (comparison == 0 && exclusive)
            {
                *bound = Some(Bound { value, exclusive });
            }
        }
        if let (Some(lower), Some(upper)) = (&lexeme.lower, &lexeme.upper) {
            let comparison = compare(&lower.value, &upper.value);
            if comparison > 0
                || (comparison == 0
                    && (lower.exclusive || upper.exclusive || !lexeme.valid(&lower.value)))
            {
                return Err(SchemaError::empty(
                    "JSON Schema: numeric constraints describe an empty interval",
                ));
            }
        }
        if lexeme.integer {
            let multiple = lexeme.multiple.get_or_insert_with(|| Decimal::new(b"1", 0));
            if multiple.scale > 0 {
                let scale = multiple.scale;
                multiple.scale = 0;
                for factor in [b"2", b"5"] {
                    for _ in 0..scale {
                        let (quotient, remainder) = divide(multiple, &Decimal::new(factor, 0));
                        if remainder != b"0" {
                            break;
                        }
                        multiple.digits = quotient;
                    }
                }
                multiple.normalize();
            }
        }
        if let (Some(lower), Some(upper), Some(_)) =
            (&lexeme.lower, &lexeme.upper, &lexeme.multiple)
            && !lexeme.grid_point(lower.clone(), upper.clone())
        {
            return Err(SchemaError::empty(
                "JSON Schema: numeric constraints contain no multipleOf value",
            ));
        }
        Ok(lexeme)
    }

    fn accept_value(&self, value: &Value) -> Result<bool, SchemaError> {
        Ok(value.is_number() && self.valid(&Decimal::parse(dump(value)?.as_bytes())?))
    }

    fn check(&self, bytes: &[u8]) -> Match {
        if bytes.is_empty() {
            return Match::new(true, false);
        }
        if bytes.len() > 4096 {
            return Match::REJECT;
        }
        let negative = bytes[0] == b'-';
        let text = if negative { &bytes[1..] } else { bytes };
        if text.is_empty() {
            return Match::new(
                self.lower.as_ref().is_none_or(|lower| {
                    lower.value.negative || (lower.value.is_zero() && !lower.exclusive)
                }),
                false,
            );
        }
        let dot = text.iter().position(|&b| b == b'.');
        if dot == Some(0) || (self.integer && dot.is_some()) {
            return Match::REJECT;
        }
        if text[0] == b'0' && text.len() > 1 && text[1] != b'.' {
            return Match::REJECT;
        }
        if text
            .iter()
            .enumerate()
            .any(|(i, &b)| !b.is_ascii_digit() && (b != b'.' || Some(i) != dot))
        {
            return Match::REJECT;
        }
        // Digits, one inner dot and a sign parse without an exponent.
        let Ok(value) = Decimal::parse(bytes) else {
            return Match::REJECT;
        };
        let complete = text.last() != Some(&b'.') && self.valid(&value);
        // At the scalar budget there is no room for a completing digit. Do
        // not admit an unfinished prefix which would fail on the next step.
        if bytes.len() == 4096 {
            return Match::new(complete, complete);
        }
        // Decimal prefixes cover an interval. Before the decimal point,
        // appending integer digits also shifts that interval by powers of
        // ten.
        let mut low = value;
        low.negative = false;
        let places = dot.map_or(0, |dot| (text.len() - dot - 1) as i32);
        let mut high = add_unit(&low, places);
        let extend_integer = dot.is_none() && text[0] != b'0';
        for _ in 0..=1024 {
            let mut first = if negative { high.clone() } else { low.clone() };
            let mut last = if negative { low.clone() } else { high.clone() };
            first.negative = negative && !first.is_zero();
            last.negative = negative && !last.is_zero();
            let intersects_lower = self.lower.as_ref().is_none_or(|lower| {
                compare(&last, &lower.value) > 0
                    || (negative && !lower.exclusive && compare(&last, &lower.value) == 0)
            });
            let intersects_upper = self.upper.as_ref().is_none_or(|upper| {
                compare(&first, &upper.value) < 0
                    || (!negative && !upper.exclusive && compare(&first, &upper.value) == 0)
            });
            if intersects_lower && intersects_upper {
                let mut lower = Bound {
                    value: first,
                    exclusive: negative,
                };
                let mut upper = Bound {
                    value: last,
                    exclusive: !negative,
                };
                if let Some(bound) = &self.lower {
                    let comparison = compare(&bound.value, &lower.value);
                    if comparison > 0 || (comparison == 0 && bound.exclusive) {
                        lower = bound.clone();
                    }
                }
                if let Some(bound) = &self.upper {
                    let comparison = compare(&bound.value, &upper.value);
                    if comparison < 0 || (comparison == 0 && bound.exclusive) {
                        upper = bound.clone();
                    }
                }
                if self.multiple.is_none() || self.grid_point(lower, upper) {
                    return Match::new(true, complete);
                }
            }
            if !extend_integer
                || (negative && !intersects_lower)
                || (!negative && !intersects_upper)
            {
                break;
            }
            low.scale -= 1;
            high.scale -= 1;
        }
        Match::new(complete, complete)
    }

    /// Whether a multiple lies between the bounds (`GridPoint`).
    fn grid_point(&self, mut low: Bound, mut high: Bound) -> bool {
        let Some(multiple) = &self.multiple else {
            return true;
        };
        if low.value.negative && !high.value.negative && !high.value.is_zero() {
            return true; // Zero is inside.
        }
        if low.value.negative {
            std::mem::swap(&mut low, &mut high);
            low.value.negative = false;
            high.value.negative = false;
        }
        let (mut quotient, remainder) = divide(&low.value, multiple);
        if remainder != b"0" || low.exclusive {
            quotient = increment(quotient);
        }
        let candidate = product(multiple, &quotient);
        let comparison = compare(&candidate, &high.value);
        comparison < 0 || (comparison == 0 && !high.exclusive)
    }

    fn valid(&self, value: &Decimal) -> bool {
        if self.integer && value.scale > 0 {
            return false;
        }
        if let Some(lower) = &self.lower
            && compare(value, &lower.value) < i32::from(lower.exclusive)
        {
            return false;
        }
        if let Some(upper) = &self.upper
            && compare(value, &upper.value) > -i32::from(upper.exclusive)
        {
            return false;
        }
        self.multiple
            .as_ref()
            .is_none_or(|multiple| self::multiple(value, multiple))
    }
}

/// The pattern of a JSON Schema string format (`FormatPattern`). Formats
/// follow the RFC-backed definitions; the grammar validates decoded strings,
/// so an escaped character means the same as its UTF-8 spelling.
fn format_pattern(format: &str) -> Result<String, SchemaError> {
    let leap = r"(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)";
    let date = format!(
        r"(?:[0-9]{{4}}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))|{leap}-02-29)"
    );
    let time = r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])";
    Ok(match format {
        "date" => format!("^{date}$"),
        "time" => format!("^{time}$"),
        "date-time" => format!("^{date}[tT]{time}$"),
        "uuid" => r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$".to_owned(),
        "ipv4" => r"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$".to_owned(),
        "hostname" => r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.?$".to_owned(),
        "email" => r"^[^@\s]+@[^@\s]+$".to_owned(),
        "duration" => r"^P(?:[0-9]+W|(?=[0-9]|T[0-9])(?:[0-9]+Y)?(?:[0-9]+M)?(?:[0-9]+D)?(?:T(?=[0-9])(?:[0-9]+H)?(?:[0-9]+M)?(?:[0-9]+(?:\.[0-9]+)?S)?)?)$".to_owned(),
        "ipv6" => {
            let group = "[0-9a-fA-F]{1,4}";
            let groups = |count: usize| vec![group; count].join(":");
            let mut result = format!("^(?:{}", groups(8));
            for left in 0..8 {
                for right in 0..8 - left {
                    result += &format!("|{}::{}", groups(left), groups(right));
                }
            }
            let ipv4 = format_pattern("ipv4")?;
            let ipv4 = &ipv4[1..ipv4.len() - 1];
            result += &format!("|{}:{ipv4}", groups(6));
            for left in 0..6 {
                for right in 0..6 - left {
                    let separator = if right != 0 { ":" } else { "" };
                    result += &format!("|{}::{}{separator}{ipv4}", groups(left), groups(right));
                }
            }
            result + ")$"
        }
        _ => return Err(invalid("unsupported string format")),
    })
}

/// A string lexeme's decoding phase.
mod phase {
    pub(super) const BODY: u32 = 0;
    pub(super) const ESCAPE: u32 = 1;
    pub(super) const HEX: u32 = 2;
    pub(super) const LOW_SLASH: u32 = 3;
    pub(super) const LOW_U: u32 = 4;
    pub(super) const LOW_HEX: u32 = 5;
    pub(super) const UTF8: u32 = 6;
}

// Fields of the canonical request-local string state. Completed characters
// do not keep their spelling, so literal and escaped Unicode share cached
// vocabulary transitions.
const DFA: usize = 0;
const COUNT: usize = 1;
const VALUE: usize = 2;
const EXTRA: usize = 3;
const MODE: usize = 4;
const STATE_BYTES: usize = 20;

type StringState = [u32; 5];

fn decode_state(encoded: &[u8]) -> Option<StringState> {
    if encoded.len() != STATE_BYTES {
        return None;
    }
    let mut state = [0u32; 5];
    for (field, bytes) in state.iter_mut().zip(encoded.as_chunks::<4>().0) {
        *field = u32::from_le_bytes(*bytes);
    }
    Some(state)
}

fn encode_state(state: &StringState, encoded: &mut LexemeState) {
    let mut bytes = [0u8; STATE_BYTES];
    for (field, out) in state.iter().zip(bytes.as_chunks_mut::<4>().0) {
        *out = field.to_le_bytes();
    }
    encoded.set(&bytes);
}

/// A string under lengths, patterns and formats (`StringLexeme`), counting
/// Unicode scalars.
#[derive(Debug)]
pub(crate) struct StringLexeme {
    minimum: u32,
    maximum: u32,
    scalar_only: bool,
    regex: Arc<JsonSchemaRegex>,
}

impl StringLexeme {
    fn new(schema: &Value) -> Result<Self, SchemaError> {
        let (mut minimum, mut maximum) = (0u32, u32::MAX);
        for key in ["minLength", "maxLength"] {
            if let Some(value) = schema.find(key) {
                let number = value.as_double(0.0);
                if !value.is_number()
                    || number < 0.0
                    || number > 1_048_576.0
                    || number.floor() != number
                {
                    return Err(invalid(
                        "string lengths must be nonnegative integers up to 1048576",
                    ));
                }
                let length = value.as_size(0) as u32;
                if key == "minLength" {
                    minimum = length;
                } else {
                    maximum = length;
                }
            }
        }
        if minimum > maximum {
            return Err(SchemaError::empty(
                "JSON Schema: minLength exceeds maxLength",
            ));
        }
        let mut patterns = Vec::new();
        for key in ["pattern", "format"] {
            let Some(value) = schema.find(key) else {
                continue;
            };
            if !value.is_string() {
                return Err(invalid(&format!("{key} must be a string")));
            }
            let expression = if key == "format" {
                if value.str() == "hostname" {
                    maximum = maximum.min(253);
                }
                format_pattern(value.str())?
            } else {
                value.str().to_owned()
            };
            patterns.push(expression);
        }
        let scalar_only = patterns.is_empty();
        let regex = if scalar_only {
            static UNCONSTRAINED: OnceLock<Result<Arc<JsonSchemaRegex>, SchemaError>> =
                OnceLock::new();
            UNCONSTRAINED
                .get_or_init(|| JsonSchemaRegex::compile(&[], u32::MAX))
                .clone()?
        } else {
            JsonSchemaRegex::compile(&patterns, maximum)?
        };
        if !regex.can_finish(regex.start(), minimum, maximum) {
            return Err(SchemaError::empty(
                "JSON Schema: string predicates and lengths have no matching value",
            ));
        }
        Ok(Self {
            minimum,
            maximum,
            scalar_only,
            regex,
        })
    }

    /// Makes the count part of a mask cache key canonical
    /// (`CanonicalMaskState`); the request keeps its true count.
    fn canonical_mask_state(&self, encoded: &mut LexemeState, token_bytes: usize) {
        let Some(mut state) = decode_state(encoded.as_slice()) else {
            return;
        };
        // A token can complete at most one Unicode character per byte. If
        // even its longest accepting suffix fits, the upper bound cannot
        // affect its mask.
        let window = token_bytes as u64 + u64::from(self.regex.maximum_suffix());
        let count = state[COUNT];
        if self.scalar_only && count < self.minimum && (self.minimum - count) as usize > token_bytes
        {
            // No vocabulary token can reach the minimum or close this string
            // yet. With no pattern, every valid Unicode continuation can
            // still finish.
            state[COUNT] = 0;
        } else if count >= self.minimum && u64::from(self.maximum - count) >= window {
            state[COUNT] = self.minimum;
        }
        encode_state(&state, encoded);
    }

    fn check(&self, bytes: &[u8]) -> Match {
        let mut state = LexemeState::default();
        let mut result = Match::new(true, false);
        for &byte in bytes {
            if !result.prefix {
                return Match::REJECT;
            }
            result = self.advance(&mut state, byte);
        }
        result
    }

    fn advance(&self, encoded: &mut LexemeState, byte: u8) -> Match {
        let mut state: StringState = [0; 5];
        if encoded.is_empty() {
            if byte != b'"' {
                return Match::REJECT;
            }
            state[DFA] = self.regex.start();
        } else {
            let Some(decoded) = decode_state(encoded.as_slice()) else {
                return Match::REJECT;
            };
            state = decoded;
            match state[MODE] {
                phase::BODY => {
                    if byte == b'"' {
                        return Match::new(
                            false,
                            state[COUNT] >= self.minimum && self.regex.accepting(state[DFA]),
                        );
                    }
                    if byte == b'\\' {
                        state[MODE] = phase::ESCAPE;
                    } else if byte < 0x20 {
                        return Match::REJECT;
                    } else if byte < 0x80 {
                        if !self.character(&mut state, u32::from(byte)) {
                            return Match::REJECT;
                        }
                    } else {
                        let width: u32 = match byte {
                            0xc2..=0xdf => 2,
                            0xe0..=0xef => 3,
                            0xf0..=0xf4 => 4,
                            _ => return Match::REJECT,
                        };
                        state[MODE] = phase::UTF8;
                        if self.scalar_only {
                            let low: u32 = match byte {
                                0xe0 => 0xa0,
                                0xf0 => 0x90,
                                _ => 0x80,
                            };
                            let high: u32 = match byte {
                                0xed => 0x9f,
                                0xf4 => 0x8f,
                                _ => 0xbf,
                            };
                            state[VALUE] = (low << 8) | high;
                            state[EXTRA] = width - 1;
                        } else {
                            state[VALUE] = u32::from(byte) & ((1 << (7 - width)) - 1);
                            state[EXTRA] = (width << 8) | (width - 1);
                        }
                    }
                }
                phase::ESCAPE => {
                    if byte == b'u' {
                        state[MODE] = phase::HEX;
                        state[EXTRA] = 4;
                    } else {
                        let Some(index) = b"\"\\/bfnrt".iter().position(|&b| b == byte) else {
                            return Match::REJECT;
                        };
                        const VALUES: [u8; 8] =
                            [b'"', b'\\', b'/', 0x08, 0x0c, b'\n', b'\r', b'\t'];
                        if !self.character(&mut state, u32::from(VALUES[index])) {
                            return Match::REJECT;
                        }
                    }
                }
                phase::LOW_SLASH => {
                    if byte != b'\\' {
                        return Match::REJECT;
                    }
                    state[MODE] = phase::LOW_U;
                }
                phase::LOW_U => {
                    if byte != b'u' {
                        return Match::REJECT;
                    }
                    state[MODE] = phase::LOW_HEX;
                    state[EXTRA] = (state[VALUE] << 8) | 4;
                    state[VALUE] = 0;
                }
                phase::HEX | phase::LOW_HEX => {
                    let Some(digit) = char::from(byte).to_digit(16) else {
                        return Match::REJECT;
                    };
                    if self.scalar_only {
                        let remaining = state[EXTRA] & 0xff;
                        if state[MODE] == phase::LOW_HEX {
                            if (remaining == 4 && digit != 13) || (remaining == 3 && digit < 12) {
                                return Match::REJECT;
                            }
                        } else if remaining == 4 {
                            state[VALUE] = u32::from(digit == 13);
                        } else if remaining == 3 && state[VALUE] == 1 {
                            if digit >= 12 {
                                return Match::REJECT;
                            }
                            state[VALUE] = if digit >= 8 { 2 } else { 0 };
                        }
                        state[EXTRA] -= 1;
                        if state[EXTRA] & 0xff == 0 {
                            if state[MODE] == phase::HEX && state[VALUE] == 2 {
                                state[VALUE] = 0xd800;
                                state[MODE] = phase::LOW_SLASH;
                            } else if !self.character(&mut state, 0) {
                                return Match::REJECT;
                            }
                        }
                    } else {
                        state[VALUE] = (state[VALUE] << 4) | digit;
                        state[EXTRA] -= 1;
                        if state[EXTRA] & 0xff == 0 {
                            let mut code_point = state[VALUE];
                            if state[MODE] == phase::LOW_HEX {
                                if !(0xdc00..=0xdfff).contains(&code_point) {
                                    return Match::REJECT;
                                }
                                code_point =
                                    0x10000 + ((state[EXTRA] >> 8) - 0xd800) * 1024 + code_point
                                        - 0xdc00;
                            } else if (0xd800..=0xdbff).contains(&code_point) {
                                state[MODE] = phase::LOW_SLASH;
                                state[EXTRA] = 0;
                                return self.finish(state, encoded);
                            }
                            if !self.character(&mut state, code_point) {
                                return Match::REJECT;
                            }
                        }
                    }
                }
                phase::UTF8 => {
                    if self.scalar_only {
                        if u32::from(byte) < (state[VALUE] >> 8)
                            || u32::from(byte) > (state[VALUE] & 0xff)
                        {
                            return Match::REJECT;
                        }
                        state[EXTRA] -= 1;
                        if state[EXTRA] == 0 {
                            if !self.character(&mut state, 0) {
                                return Match::REJECT;
                            }
                        } else {
                            state[VALUE] = (0x80 << 8) | 0xbf;
                        }
                    } else {
                        if !(0x80..=0xbf).contains(&byte) {
                            return Match::REJECT;
                        }
                        state[VALUE] = (state[VALUE] << 6) | u32::from(byte & 0x3f);
                        state[EXTRA] -= 1;
                        if state[EXTRA] & 0xff == 0 {
                            let width = state[EXTRA] >> 8;
                            let minimum = match width {
                                2 => 0x80,
                                3 => 0x800,
                                _ => 0x10000,
                            };
                            let value = state[VALUE];
                            if value < minimum || !self.character(&mut state, value) {
                                return Match::REJECT;
                            }
                        }
                    }
                }
                _ => return Match::REJECT,
            }
        }
        self.finish(state, encoded)
    }

    /// Keeps a state that still has a completing continuation.
    fn finish(&self, state: StringState, encoded: &mut LexemeState) -> Match {
        if state[MODE] != phase::BODY && !self.pending(&state) {
            return Match::REJECT;
        }
        encode_state(&state, encoded);
        Match::new(true, false)
    }

    fn remaining_minimum(&self, count: u32) -> u32 {
        self.minimum.saturating_sub(count)
    }

    /// Appends a completed character (`Character`).
    fn character(&self, state: &mut StringState, code_point: u32) -> bool {
        if state[COUNT] == self.maximum
            || code_point > 0x10ffff
            || (0xd800..=0xdfff).contains(&code_point)
        {
            return false;
        }
        let count = state[COUNT] + 1;
        if !self.scalar_only {
            let next = self.regex.advance(state[DFA], code_point);
            if !self
                .regex
                .can_finish(next, self.remaining_minimum(count), self.maximum - count)
            {
                return false;
            }
            state[DFA] = next;
        }
        state[COUNT] = if self.maximum == u32::MAX {
            count.min(self.minimum)
        } else {
            count
        };
        state[VALUE] = 0;
        state[EXTRA] = 0;
        state[MODE] = phase::BODY;
        true
    }

    fn range(&self, state: &StringState, first: u32, last: u32) -> bool {
        if first > last || state[COUNT] == self.maximum {
            return false;
        }
        let count = state[COUNT] + 1;
        self.regex.can_advance(
            state[DFA],
            first,
            last,
            self.remaining_minimum(count),
            self.maximum - count,
        )
    }

    /// Whether an unfinished character can still complete (`Pending`).
    fn pending(&self, state: &StringState) -> bool {
        if self.scalar_only {
            return state[COUNT] < self.maximum;
        }
        if state[MODE] == phase::ESCAPE {
            return self.range(state, 0, 0x10ffff);
        }
        if state[MODE] == phase::UTF8 {
            let shift = 6 * (state[EXTRA] & 0xff);
            let width = state[EXTRA] >> 8;
            let minimum = match width {
                2 => 0x80,
                3 => 0x800,
                _ => 0x10000,
            };
            return self.range(
                state,
                minimum.max(state[VALUE] << shift),
                0x10ffff.min(((state[VALUE] + 1) << shift) - 1),
            );
        }
        if state[MODE] == phase::LOW_SLASH || state[MODE] == phase::LOW_U {
            let first = 0x10000 + (state[VALUE] - 0xd800) * 1024;
            return self.range(state, first, first + 1023);
        }
        let shift = 4 * (state[EXTRA] & 0xff);
        let first = state[VALUE] << shift;
        let last = ((state[VALUE] + 1) << shift) - 1;
        if state[MODE] == phase::LOW_HEX {
            let (low, high) = (first.max(0xdc00), last.min(0xdfff));
            if low > high {
                return false;
            }
            let base = 0x10000 + ((state[EXTRA] >> 8) - 0xd800) * 1024;
            return self.range(state, base + low - 0xdc00, base + high - 0xdc00);
        }
        if first <= 0xd7ff && self.range(state, first, last.min(0xd7ff)) {
            return true;
        }
        if last >= 0xe000 && self.range(state, first.max(0xe000), last) {
            return true;
        }
        if first <= 0xdbff && last >= 0xd800 {
            return self.range(
                state,
                0x10000 + (first.max(0xd800) - 0xd800) * 1024,
                0x10000 + (last.min(0xdbff) - 0xd800 + 1) * 1024 - 1,
            );
        }
        false
    }
}

/// A native tool argument (`RawStringLexeme`): raw UTF-8, not a JSON string
/// literal. The string predicate checks it with its escaped bytes, quotes
/// and backslashes are literal data, and the enclosing grammar supplies the
/// parameter delimiter, which the text never contains. The state is the
/// delimiter prefix matched so far, then the inner string state.
#[derive(Debug)]
pub(crate) struct RawStringLexeme {
    string: StringLexeme,
    delimiter: Vec<u8>,
    /// Whether the value may end with `i` delimiter bytes already matched:
    /// the first delimiter must begin after the value, not inside a trailing
    /// prefix of it.
    can_close: Vec<bool>,
}

impl RawStringLexeme {
    fn new(string: StringLexeme, delimiter: &str) -> Self {
        let delimiter = delimiter.as_bytes().to_vec();
        let mut can_close = vec![true; delimiter.len()];
        for (matched, close) in can_close.iter_mut().enumerate().skip(1) {
            let mut suffix = delimiter[..matched].to_vec();
            suffix.extend_from_slice(&delimiter);
            *close = suffix
                .windows(delimiter.len())
                .position(|window| window == delimiter.as_slice())
                == Some(matched);
        }
        Self {
            string,
            delimiter,
            can_close,
        }
    }

    fn canonical_mask_state(&self, state: &mut LexemeState, token_bytes: usize) {
        let Some((&matched, inner)) = state.as_slice().split_first() else {
            return;
        };
        let mut inner = LexemeState::from(inner);
        self.string.canonical_mask_state(&mut inner, token_bytes);
        let mut encoded = vec![matched];
        encoded.extend_from_slice(inner.as_slice());
        *state = LexemeState::from(encoded.as_slice());
    }

    fn check(&self, bytes: &[u8]) -> Match {
        let mut state = LexemeState::default();
        let mut result = Match::new(true, self.string.check(b"\"\"").complete);
        for &byte in bytes {
            result = self.advance(&mut state, byte);
            if !result.prefix {
                break;
            }
        }
        result
    }

    fn advance(&self, encoded: &mut LexemeState, byte: u8) -> Match {
        let (matched, mut state) = match encoded.as_slice().split_first() {
            None => (0, LexemeState::default()),
            Some((&matched, inner)) => (usize::from(matched), LexemeState::from(inner)),
        };
        let mut suffix = self.delimiter[..matched].to_vec();
        suffix.push(byte);
        let mut matched = suffix.len().min(self.delimiter.len());
        while matched != 0 && !suffix.ends_with(&self.delimiter[..matched]) {
            matched -= 1;
        }
        if matched == self.delimiter.len() {
            return Match::REJECT;
        }
        if state.is_empty() {
            self.string.advance(&mut state, b'"');
        }
        let result = if byte == b'"' || byte == b'\\' {
            let escape = self.string.advance(&mut state, b'\\');
            if escape.prefix {
                self.string.advance(&mut state, byte)
            } else {
                escape
            }
        } else if byte < 0x20 {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut result = Match::REJECT;
            for part in [
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX[usize::from(byte >> 4)],
                HEX[usize::from(byte & 15)],
            ] {
                result = self.string.advance(&mut state, part);
                if !result.prefix {
                    return Match::REJECT;
                }
            }
            result
        } else {
            self.string.advance(&mut state, byte)
        };
        if !result.prefix {
            return Match::REJECT;
        }
        let mut closing = state.clone();
        let mut next = vec![matched as u8];
        next.extend_from_slice(state.as_slice());
        *encoded = LexemeState::from(next.as_slice());
        // For "\n</parameter>\n", a value ending in "\n</parameter>" would
        // otherwise admit duplicated closers and disagree with the parser.
        Match::new(
            true,
            self.can_close[matched] && self.string.advance(&mut closing, b'"').complete,
        )
    }
}

/// Where native reasoning may end (`JsonSchemaLexeme::ReasoningEnd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReasoningEnd {
    /// At `</think>`.
    Explicit,
    /// At a complete native call header.
    Tool,
    /// Not at all: a natural stop may end it.
    Unfinished,
}

/// The scan state of native reasoning: marker prefixes and Markdown quoting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Scan {
    explicit_prefix: u32,
    tool_prefix: u32,
    inline_quote: u32,
    fence: u32,
    run: u32,
    run_char: u32,
    run_at_start: u32,
    fence_char: u32,
    nonblank: u32,
    closing_fence: u32,
}

impl Scan {
    const BYTES: usize = 40;

    fn decode(state: &LexemeState) -> Self {
        let bytes = state.as_slice();
        if bytes.len() != Self::BYTES {
            return Self::default();
        }
        let word = |i: usize| {
            let mut word = [0; 4];
            word.copy_from_slice(&bytes[4 * i..4 * i + 4]);
            u32::from_le_bytes(word)
        };
        Self {
            explicit_prefix: word(0),
            tool_prefix: word(1),
            inline_quote: word(2),
            fence: word(3),
            run: word(4),
            run_char: word(5),
            run_at_start: word(6),
            fence_char: word(7),
            nonblank: word(8),
            closing_fence: word(9),
        }
    }

    fn encode(&self, state: &mut LexemeState) {
        if *self == Self::default() {
            state.clear();
            return;
        }
        let mut bytes = [0; Self::BYTES];
        for (i, word) in [
            self.explicit_prefix,
            self.tool_prefix,
            self.inline_quote,
            self.fence,
            self.run,
            self.run_char,
            self.run_at_start,
            self.fence_char,
            self.nonblank,
            self.closing_fence,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        state.set(&bytes);
    }
}

/// Native reasoning before a call (`ReasoningLexeme`):
/// a native call can start before `</think>`. Literal marker mentions and
/// Markdown code stay quoted; a complete native function header starts
/// constrained tool decoding. Only a small scan state belongs to each
/// speculative branch.
#[derive(Debug)]
pub(crate) struct ReasoningLexeme {
    tool_prefix: Vec<u8>,
    end: ReasoningEnd,
    /// The KMP failure function of `tool_prefix`.
    tool_failure: Vec<u32>,
}

impl ReasoningLexeme {
    fn new(tool_prefix: &str, end: ReasoningEnd) -> Self {
        let tool_prefix = tool_prefix.as_bytes().to_vec();
        let mut tool_failure = vec![0u32; tool_prefix.len()];
        let mut matched = 0usize;
        for i in 1..tool_prefix.len() {
            while matched != 0 && tool_prefix[i] != tool_prefix[matched] {
                matched = tool_failure[matched - 1] as usize;
            }
            if tool_prefix[i] == tool_prefix[matched] {
                matched += 1;
            }
            tool_failure[i] = matched as u32;
        }
        Self {
            tool_prefix,
            end,
            tool_failure,
        }
    }

    /// Whether neither terminator has a pending prefix: any nonempty token
    /// without '<' is then admissible (`PlainReasoning`).
    fn plain(&self, encoded: &LexemeState) -> bool {
        let state = Scan::decode(encoded);
        state.explicit_prefix == 0 && state.tool_prefix == 0
    }

    fn check(&self, bytes: &[u8]) -> Match {
        let mut state = LexemeState::default();
        let mut result = Match::new(true, self.end == ReasoningEnd::Unfinished);
        for &byte in bytes {
            result = self.advance(&mut state, byte);
            if !result.prefix {
                break;
            }
        }
        result
    }

    fn advance(&self, encoded: &mut LexemeState, byte: u8) -> Match {
        let mut state = Scan::decode(encoded);
        let line_start = state.nonblank == 0;
        if !matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
            state.nonblank = 1;
        }
        if state.run != 0 && u32::from(byte) != state.run_char {
            if state.fence != 0 {
                if state.run_char == state.fence_char
                    && state.run >= state.fence
                    && state.run_at_start != 0
                {
                    state.closing_fence = 1;
                }
            } else if state.inline_quote == 0 && state.run >= 3 && state.run_at_start != 0 {
                state.fence = state.run;
                state.fence_char = state.run_char;
            } else if state.run_char == u32::from(b'`') {
                if state.inline_quote == 0 {
                    state.inline_quote = state.run;
                } else if state.run == state.inline_quote {
                    state.inline_quote = 0;
                }
            }
            state.run = 0;
            state.run_char = 0;
            state.run_at_start = 0;
        }
        if byte == b'`' || byte == b'~' {
            if state.run == 0 {
                state.run_char = u32::from(byte);
                state.run_at_start = u32::from(line_start);
            }
            state.run = state.run.saturating_add(1);
        } else if byte == b'\n' {
            if state.fence != 0 && state.closing_fence != 0 {
                state.fence = 0;
                state.fence_char = 0;
            }
            if state.inline_quote != 0 && state.nonblank == 0 {
                state.inline_quote = 0;
            }
            state.closing_fence = 0;
            state.nonblank = 0;
        } else if !matches!(byte, b' ' | b'\t' | b'\r') {
            state.closing_fence = 0;
        }
        // Overlapping marker prefixes, without allocating per byte.
        let step = |prefix: u32, marker: &[u8], failure: &[u32]| -> u32 {
            let mut prefix = prefix as usize;
            while prefix != 0 && byte != marker[prefix] {
                prefix = failure[prefix - 1] as usize;
            }
            if byte == marker[prefix] {
                prefix as u32 + 1
            } else {
                0
            }
        };
        const EXPLICIT_END: &[u8] = b"</think>";
        const EXPLICIT_FAILURE: [u32; 8] = [0; 8];
        state.explicit_prefix = step(state.explicit_prefix, EXPLICIT_END, &EXPLICIT_FAILURE);
        state.tool_prefix = if state.inline_quote != 0 || state.fence != 0 {
            0
        } else {
            step(state.tool_prefix, &self.tool_prefix, &self.tool_failure)
        };
        if state.explicit_prefix as usize == EXPLICIT_END.len() {
            return Match::new(false, self.end == ReasoningEnd::Explicit);
        }
        if state.tool_prefix as usize == self.tool_prefix.len() {
            return Match::new(false, self.end == ReasoningEnd::Tool);
        }
        state.encode(encoded);
        Match::new(true, self.end == ReasoningEnd::Unfinished)
    }
}

/// A primitive's incremental constraint (`JsonSchemaLexeme`).
#[derive(Debug)]
pub(crate) enum JsonSchemaLexeme {
    Whitespace,
    Number(NumberLexeme),
    String(StringLexeme),
    RawString(RawStringLexeme),
    Reasoning(ReasoningLexeme),
}

impl JsonSchemaLexeme {
    /// The shared whitespace lexeme (`JsonSchemaLexeme::Whitespace`).
    pub(crate) fn whitespace() -> Arc<Self> {
        static WHITESPACE: OnceLock<Arc<JsonSchemaLexeme>> = OnceLock::new();
        WHITESPACE
            .get_or_init(|| Arc::new(Self::Whitespace))
            .clone()
    }

    pub(crate) fn string(schema: &Value) -> Result<Arc<Self>, SchemaError> {
        Ok(Arc::new(Self::String(StringLexeme::new(schema)?)))
    }

    pub(crate) fn number(schema: &Value, integer: bool) -> Result<Arc<Self>, SchemaError> {
        Ok(Arc::new(Self::Number(NumberLexeme::new(schema, integer)?)))
    }

    /// A native tool argument checked by the string predicate of `schema`,
    /// ending before `delimiter` (`JsonSchemaLexeme::RawString`).
    pub(crate) fn raw_string(schema: &Value, delimiter: &str) -> Result<Arc<Self>, SchemaError> {
        Ok(Arc::new(Self::RawString(RawStringLexeme::new(
            StringLexeme::new(schema)?,
            delimiter,
        ))))
    }

    /// Native reasoning ending at `end` (`JsonSchemaLexeme::Reasoning`).
    pub(crate) fn reasoning(tool_prefix: &str, end: ReasoningEnd) -> Arc<Self> {
        Arc::new(Self::Reasoning(ReasoningLexeme::new(tool_prefix, end)))
    }

    /// Whether native reasoning may take any nonempty token without '<'
    /// here (`PlainReasoning`); other lexemes opt out.
    pub(crate) fn plain_reasoning(&self, state: &LexemeState) -> bool {
        match self {
            Self::Reasoning(reasoning) => reasoning.plain(state),
            _ => false,
        }
    }

    /// Rejects impossible bytes before copying request state
    /// (`AllowsByte`): a superset of every valid transition.
    pub(crate) fn allows_byte(&self, byte: u8) -> bool {
        match self {
            Self::Whitespace => matches!(byte, b' ' | b'\t' | b'\r' | b'\n'),
            Self::Number(_) => matches!(byte, b'0'..=b'9' | b'-' | b'.'),
            Self::String(_) | Self::RawString(_) | Self::Reasoning(_) => true,
        }
    }

    pub(crate) fn check(&self, bytes: &[u8]) -> Match {
        match self {
            Self::Whitespace => {
                let mut state = LexemeState::default();
                let mut result = Match::new(true, false);
                for &byte in bytes {
                    result = whitespace_advance(&mut state, byte);
                    if !result.prefix && !result.complete {
                        return Match::REJECT;
                    }
                }
                result
            }
            Self::Number(number) => number.check(bytes),
            Self::String(string) => string.check(bytes),
            Self::RawString(raw) => raw.check(bytes),
            Self::Reasoning(reasoning) => reasoning.check(bytes),
        }
    }

    pub(crate) fn advance(&self, state: &mut LexemeState, byte: u8) -> Match {
        match self {
            Self::Whitespace => whitespace_advance(state, byte),
            Self::Number(number) => {
                state.push(byte);
                number.check(state.as_slice())
            }
            Self::String(string) => string.advance(state, byte),
            Self::RawString(raw) => raw.advance(state, byte),
            Self::Reasoning(reasoning) => reasoning.advance(state, byte),
        }
    }

    /// Whether token-trie walks may intern the states this lexeme reaches
    /// (`CacheTransitions`); numbers keep their text.
    pub(crate) fn cache_transitions(&self) -> bool {
        !matches!(self, Self::Number(_))
    }

    pub(crate) fn canonical_mask_state(&self, state: &mut LexemeState, token_bytes: usize) {
        match self {
            Self::String(string) => string.canonical_mask_state(state, token_bytes),
            Self::RawString(raw) => raw.canonical_mask_state(state, token_bytes),
            _ => {}
        }
    }

    /// Whether a finite enum/const value satisfies the predicate
    /// (`AcceptValue`).
    pub(crate) fn accept_value(&self, value: &Value) -> Result<bool, SchemaError> {
        match self {
            Self::Number(number) => number.accept_value(value),
            _ => Ok(self.check(dump(value)?.as_bytes()).complete),
        }
    }
}

/// `json::Value::dump`, whose only failure (a non-finite number) is an
/// `std::invalid_argument` in C++.
pub(crate) fn dump(value: &Value) -> Result<String, SchemaError> {
    value
        .dump()
        .map_err(|error| SchemaError::Invalid(error.message().to_owned()))
}

/// The least common multiple of two `multipleOf` steps
/// (`IntersectMultipleOf`).
pub(crate) fn intersect_multiple_of(a: &Value, b: &Value) -> Result<Value, SchemaError> {
    for value in [a, b] {
        if !value.is_number() || !value.as_double(0.0).is_finite() || value.as_double(0.0) <= 0.0 {
            return Err(invalid("multipleOf must be a positive finite number"));
        }
    }
    let mut left = Decimal::parse(dump(a)?.as_bytes())?;
    let mut right = Decimal::parse(dump(b)?.as_bytes())?;
    // Put both steps on the same integer grid, then compute the exact LCM.
    let scale = left.scale.max(right.scale);
    left.digits
        .resize(left.digits.len() + (scale - left.scale) as usize, b'0');
    right
        .digits
        .resize(right.digits.len() + (scale - right.scale) as usize, b'0');
    left.scale = 0;
    right.scale = 0;
    let (mut x, mut y) = (left.clone(), right.clone());
    while !y.is_zero() {
        let remainder = divide(&x, &y).1;
        x = y;
        y = Decimal::new(&remainder, 0);
    }
    let mut common = product(&Decimal::new(&divide(&left, &x).0, 0), &right.digits);
    common.scale += scale;
    common.normalize();
    let text = format!(
        "{}e{}",
        String::from_utf8_lossy(&common.digits),
        -common.scale
    );
    // A parse failure is a `std::runtime_error` in C++.
    let result = json::parse_str(&text).map_err(|error| SchemaError::runtime(error.message()))?;
    if !result.is_number()
        || !result.as_double(0.0).is_finite()
        || compare(&Decimal::parse(dump(&result)?.as_bytes())?, &common) != 0
    {
        return Err(invalid(
            "combined multipleOf exceeds the exact schema-number range",
        ));
    }
    Ok(result)
}

/// The schema a string format denotes (`JsonSchemaLexeme::Format`).
pub(crate) fn format(format: &str) -> Result<Value, SchemaError> {
    let mut schema = Value::object();
    schema.set("pattern", format_pattern(format)?);
    if format == "hostname" {
        schema.set("maxLength", 253);
    }
    Ok(schema)
}
