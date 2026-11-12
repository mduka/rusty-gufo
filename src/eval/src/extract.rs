//! Final-answer extraction and grading (`extract.cpp`). Like the C++ code it
//! works on bytes with the C locale's character classes, so non-ASCII bytes
//! are neither letters, digits nor spaces.

use std::collections::BTreeSet;

use crate::dataset::{AnswerKind, EvalCase};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExtractionStatus {
    Found,
    #[default]
    NoAnswer,
    Ambiguous,
}

impl ExtractionStatus {
    /// `ExtractionStatusName`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Found => "found",
            Self::NoAnswer => "no_answer",
            Self::Ambiguous => "ambiguous",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExtractionResult {
    pub status: ExtractionStatus,
    pub answer: String,
}

impl ExtractionResult {
    fn found(answer: String) -> Self {
        Self {
            status: ExtractionStatus::Found,
            answer,
        }
    }

    fn ambiguous() -> Self {
        Self {
            status: ExtractionStatus::Ambiguous,
            answer: String::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GradeResult {
    pub extraction: ExtractionResult,
    pub passed: bool,
}

/// `std::isspace` in the C locale.
fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn is_word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn lower(bytes: &[u8]) -> Vec<u8> {
    bytes.to_ascii_lowercase()
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn rfind(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .rposition(|window| window == needle)
}

/// The text after the last `</think>`.
fn visible_text(text: &[u8]) -> &[u8] {
    match rfind(text, b"</think>") {
        Some(marker) => &text[marker + 8..],
        None => text,
    }
}

/// The last whole word "answer" followed by a colon on its line, else the
/// last whole word "answer".
fn last_answer_marker(text: &[u8]) -> Option<usize> {
    let lower = lower(text);
    let (mut fallback, mut with_colon) = (None, None);
    let mut position = 0;
    while let Some(found) = lower[position..]
        .windows(6)
        .position(|window| window == b"answer")
    {
        let at = position + found;
        let after = at + 6;
        let left = at == 0 || !is_word(lower[at - 1]);
        let right = after == lower.len() || !is_word(lower[after]);
        if left && right {
            fallback = Some(at);
            let mut cursor = after;
            while cursor < lower.len() && is_space(lower[cursor]) && lower[cursor] != b'\n' {
                cursor += 1;
            }
            if cursor < lower.len() && lower[cursor] == b':' {
                with_colon = Some(at);
            }
        }
        position = after;
    }
    with_colon.or(fallback)
}

/// The line of the answer marker, at most `limit` bytes, or the whole
/// visible text without a marker.
fn answer_window(text: &[u8], limit: usize) -> &[u8] {
    let visible = visible_text(text);
    let Some(marker) = last_answer_marker(visible) else {
        return visible;
    };
    let mut window = &visible[marker..];
    if let Some(newline) = window.iter().position(|&byte| byte == b'\n') {
        window = &window[..newline];
    }
    &window[..window.len().min(limit)]
}

fn is_letter_boundary(byte: u8) -> bool {
    byte == 0 || !byte.is_ascii_alphanumeric()
}

/// Whether the word before the letter negates it ("not B", "rules out C").
fn has_negation_cue(text: &[u8], letter_position: usize) -> bool {
    let mut end = letter_position;
    while end > 0 && matches!(text[end - 1], b' ' | b'\t' | b',' | b';') {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && (text[start - 1].is_ascii_alphabetic() || text[start - 1] == b'\'') {
        start -= 1;
    }
    let word = lower(&text[start..end]);
    if word.ends_with(b"n't") {
        return true;
    }
    const CUES: [&[u8]; 12] = [
        b"not",
        b"except",
        b"excluding",
        b"exclude",
        b"excludes",
        b"eliminate",
        b"eliminates",
        b"eliminated",
        b"reject",
        b"rejects",
        b"rejected",
        b"rejecting",
    ];
    if CUES.contains(&word.as_slice()) {
        return true;
    }
    if word != b"out" {
        return false;
    }
    end = start;
    while end > 0 && matches!(text[end - 1], b' ' | b'\t') {
        end -= 1;
    }
    start = end;
    while start > 0 && text[start - 1].is_ascii_alphabetic() {
        start -= 1;
    }
    matches!(
        lower(&text[start..end]).as_slice(),
        b"rule" | b"rules" | b"ruled"
    )
}

/// "A" or "I" starting prose ("I think", "A cat", "I'll").
fn looks_like_leading_prose(text: &[u8], position: usize, candidate: u8) -> bool {
    if candidate != b'A' && candidate != b'I' {
        return false;
    }
    if position + 1 >= text.len() {
        return false;
    }
    if text[position + 1] == b'\'' {
        return true;
    }
    let mut cursor = position + 1;
    if text[cursor] != b' ' && text[cursor] != b'\t' {
        return false;
    }
    while cursor < text.len() && matches!(text[cursor], b' ' | b'\t') {
        cursor += 1;
    }
    cursor < text.len() && text[cursor].is_ascii_lowercase()
}

fn extract_multiple_choice(eval_case: &EvalCase, text: &[u8]) -> ExtractionResult {
    let visible = visible_text(text);
    let marker = last_answer_marker(visible);
    let window = if marker.is_some() {
        answer_window(visible, 160)
    } else {
        visible
    };
    let maximum = b'A'.wrapping_add(eval_case.choices.len().wrapping_sub(1) as u8);
    let mut candidates: Vec<(u8, usize)> = Vec::new();
    for (index, &byte) in window.iter().enumerate() {
        let candidate = byte.to_ascii_uppercase();
        if !(b'A'..=maximum).contains(&candidate) {
            continue;
        }
        let before = if index == 0 { 0 } else { window[index - 1] };
        let after = window.get(index + 1).copied().unwrap_or(0);
        if !is_letter_boundary(before)
            || !is_letter_boundary(after)
            || looks_like_leading_prose(window, index, candidate)
            || has_negation_cue(window, index)
        {
            continue;
        }
        candidates.push((candidate, index));
    }
    let (Some(&first), Some(&last)) = (candidates.first(), candidates.last()) else {
        return ExtractionResult::default();
    };
    let selected = if marker.is_some() { first.0 } else { last.0 };
    for &(candidate, position) in &candidates {
        if candidate == selected {
            continue;
        }
        // Without a marker C++ compares each candidate with itself, so only
        // an answer line can be ambiguous.
        let selected_position = if marker.is_some() { first.1 } else { position };
        let (begin, end) = (
            selected_position.min(position),
            selected_position.max(position),
        );
        let between = lower(&window[begin..end]);
        if contains(&between, b" or ") || between.contains(&b'/') || contains(&between, b" and ") {
            return ExtractionResult::ambiguous();
        }
    }
    ExtractionResult::found(char::from(selected).to_string())
}

/// `NormalizeInteger`: no thousands separators or dollar signs, no leading
/// zeros, and no sign on zero.
fn normalize_integer(value: &[u8]) -> Vec<u8> {
    let value: Vec<u8> = value
        .iter()
        .copied()
        .filter(|&byte| byte != b',' && byte != b'$')
        .collect();
    let mut negative = false;
    let mut start = 0;
    if let Some(&sign @ (b'-' | b'+')) = value.first() {
        negative = sign == b'-';
        start = 1;
    }
    while start + 1 < value.len() && value[start] == b'0' {
        start += 1;
    }
    let mut normalized = value.get(start..).unwrap_or_default().to_vec();
    if negative && normalized != b"0" {
        normalized.insert(0, b'-');
    }
    normalized
}

/// Every signed integer of `text` (digits and commas), normalized, with its
/// position.
fn integers(text: &[u8]) -> Vec<(Vec<u8>, usize)> {
    let mut values = Vec::new();
    let mut index = 0;
    while index < text.len() {
        let start = index;
        if matches!(text[index], b'-' | b'+')
            && index + 1 < text.len()
            && text[index + 1].is_ascii_digit()
        {
            index += 1;
        }
        if !text[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        while index < text.len() && (text[index].is_ascii_digit() || text[index] == b',') {
            index += 1;
        }
        values.push((normalize_integer(&text[start..index]), start));
    }
    values
}

fn extract_integer(text: &[u8]) -> ExtractionResult {
    let visible = visible_text(text);
    let marker = last_answer_marker(visible);
    let mut window = if marker.is_some() {
        answer_window(visible, 200)
    } else {
        visible
    };
    let equals = window.iter().rposition(|&byte| byte == b'=');
    if let Some(equals) = equals {
        window = &window[equals + 1..];
    }
    let values = integers(window);
    let (Some(first), Some(last)) = (values.first(), values.last()) else {
        return ExtractionResult::default();
    };
    if values.len() > 1 {
        let lower = lower(window);
        if contains(&lower, b" or ") || lower.contains(&b'/') {
            return ExtractionResult::ambiguous();
        }
    }
    let answer = if marker.is_some() || equals.is_some() {
        &first.0
    } else {
        &last.0
    };
    ExtractionResult::found(String::from_utf8_lossy(answer).into_owned())
}

/// `std::from_chars` of an `int` at the start of `text`: an optional `-`,
/// decimal digits; `Err(true)` on overflow, `Err(false)` without digits.
fn from_chars_int(text: &[u8]) -> Result<(i32, usize), bool> {
    let sign = usize::from(text.first() == Some(&b'-'));
    let digits = text[sign..]
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digits == 0 {
        return Err(false);
    }
    let end = sign + digits;
    std::str::from_utf8(&text[..end])
        .ok()
        .and_then(|number| number.parse::<i32>().ok())
        .map(|value| (value, end))
        .ok_or(true)
}

/// The set of line numbers and inclusive ranges (`9-10`) in `text`.
fn parse_line_set(text: &[u8]) -> Vec<i32> {
    let mut lines = BTreeSet::new();
    let mut index = 0;
    while index < text.len() {
        if !text[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        let Ok((mut first, length)) = from_chars_int(&text[index..]) else {
            // C++ retries from the next byte, so an overflowing run yields
            // its longest representable suffix.
            index += 1;
            continue;
        };
        index += length;
        let mut last = first;
        let mut cursor = index;
        while cursor < text.len() && is_space(text[cursor]) {
            cursor += 1;
        }
        if cursor < text.len() && text[cursor] == b'-' {
            cursor += 1;
            while cursor < text.len() && is_space(text[cursor]) {
                cursor += 1;
            }
            match from_chars_int(&text[cursor..]) {
                Ok((value, length)) => {
                    last = value;
                    index = cursor + length;
                }
                Err(_) => last = first,
            }
        }
        if first > last {
            std::mem::swap(&mut first, &mut last);
        }
        let mut line = first;
        while line <= last && line < 100_000 {
            lines.insert(line);
            line += 1;
        }
    }
    lines.into_iter().collect()
}

fn format_line_set(lines: &[i32]) -> String {
    lines
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn extract_line_spec(text: &[u8]) -> ExtractionResult {
    let lines = parse_line_set(answer_window(text, 240));
    if lines.is_empty() {
        return ExtractionResult::default();
    }
    ExtractionResult::found(format_line_set(&lines))
}

/// `ExtractAnswer`: the final answer of a response.
pub fn extract_answer(eval_case: &EvalCase, text: &str) -> ExtractionResult {
    let text = text.as_bytes();
    match eval_case.kind {
        AnswerKind::MultipleChoice => extract_multiple_choice(eval_case, text),
        AnswerKind::Integer => extract_integer(text),
        AnswerKind::LineSpec => extract_line_spec(text),
    }
}

/// `AnswerMatches`: a line answer may name a subset of the expected lines;
/// integers compare normalized.
pub fn answer_matches(eval_case: &EvalCase, extracted: &str) -> bool {
    match eval_case.kind {
        AnswerKind::LineSpec => {
            let expected = parse_line_set(eval_case.answer.as_bytes());
            let actual = parse_line_set(extracted.as_bytes());
            !actual.is_empty() && actual.iter().all(|line| expected.contains(line))
        }
        AnswerKind::Integer => {
            normalize_integer(extracted.as_bytes())
                == normalize_integer(eval_case.answer.as_bytes())
        }
        AnswerKind::MultipleChoice => extracted == eval_case.answer,
    }
}

/// `GradeAnswer`.
pub fn grade_answer(eval_case: &EvalCase, text: &str) -> GradeResult {
    let extraction = extract_answer(eval_case, text);
    let passed = extraction.status == ExtractionStatus::Found
        && answer_matches(eval_case, &extraction.answer);
    GradeResult { extraction, passed }
}
