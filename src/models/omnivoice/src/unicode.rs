//! UTF-8 and Unicode helpers (`unicode.cpp`), with Python's string
//! semantics where OmniVoice's reference relies on them. ICU's answers come
//! from `gufo_core::unicode` and the standard library (Unicode 17 on both
//! sides).

/// One code point of a UTF-8 string and its byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Codepoint {
    pub value: u32,
    pub begin: usize,
    pub end: usize,
}

/// Decodes `text`; `None` on malformed UTF-8 (overlong forms, surrogates)
/// (`DecodeUtf8`).
pub fn decode_utf8(text: &[u8]) -> Option<Vec<Codepoint>> {
    let mut output = Vec::with_capacity(text.len());
    let mut index = 0;
    while index < text.len() {
        let begin = index;
        let first = text[index];
        index += 1;
        let (mut value, continuation) = match first {
            0x00..=0x7F => (u32::from(first), 0),
            0xC2..=0xDF => (u32::from(first & 0x1F), 1),
            0xE0..=0xEF => (u32::from(first & 0x0F), 2),
            0xF0..=0xF4 => (u32::from(first & 0x07), 3),
            _ => return None,
        };
        if continuation > text.len() - index {
            return None;
        }
        for _ in 0..continuation {
            let next = text[index];
            index += 1;
            if next & 0xC0 != 0x80 {
                return None;
            }
            value = (value << 6) | u32::from(next & 0x3F);
        }
        if (continuation == 1 && value < 0x80)
            || (continuation == 2 && (value < 0x800 || (0xD800..=0xDFFF).contains(&value)))
            || (continuation == 3 && !(0x10000..=0x10FFFF).contains(&value))
        {
            return None;
        }
        output.push(Codepoint {
            value,
            begin,
            end: index,
        });
    }
    Some(output)
}

/// `AppendUtf8`: `value` must be a Unicode scalar value or a code point the
/// encoder accepts the way the C++ does (no validation).
pub fn append_utf8(value: u32, output: &mut Vec<u8>) {
    if value <= 0x7F {
        output.push(value as u8);
    } else if value <= 0x7FF {
        output.push((0xC0 | (value >> 6)) as u8);
        output.push((0x80 | (value & 0x3F)) as u8);
    } else if value <= 0xFFFF {
        output.push((0xE0 | (value >> 12)) as u8);
        output.push((0x80 | ((value >> 6) & 0x3F)) as u8);
        output.push((0x80 | (value & 0x3F)) as u8);
    } else {
        output.push((0xF0 | (value >> 18)) as u8);
        output.push((0x80 | ((value >> 12) & 0x3F)) as u8);
        output.push((0x80 | ((value >> 6) & 0x3F)) as u8);
        output.push((0x80 | (value & 0x3F)) as u8);
    }
}

/// UTF-8 text of `points` (the C++ `Encode`). Code points come from decoded
/// UTF-8 or from literals, so the result is valid.
pub fn encode(points: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(points.len());
    for &point in points {
        append_utf8(point, &mut bytes);
    }
    String::from_utf8(bytes).unwrap_or_default()
}

/// Python's `str.isspace()`: Unicode White_Space plus the ASCII information
/// separators U+001C..U+001F. Oniguruma's and Python's `\s` agree with it
/// (`IsSpace`).
pub fn is_space(value: u32) -> bool {
    char::from_u32(value).is_some_and(gufo_core::unicode::is_space)
        || (0x1C..=0x1F).contains(&value)
}

/// The CJK Unified Ideographs block U+4E00..U+9FFF that OmniVoice uses to
/// detect Chinese text (`IsCjkIdeograph`).
pub const fn is_cjk_ideograph(value: u32) -> bool {
    value >= 0x4E00 && value <= 0x9FFF
}

/// `ContainsCjkIdeograph`.
pub fn contains_cjk_ideograph(text: &str) -> bool {
    text.chars().any(|c| is_cjk_ideograph(c as u32))
}

/// Python's `str.strip()` without arguments (`StripSpace`).
pub fn strip_space(text: &str) -> String {
    text.trim_matches(|c: char| is_space(c as u32)).to_owned()
}

/// Python's `str.lower()`: full case mapping in the root locale, as ICU's
/// `toLower(Locale::getRoot())` maps it (`ToLower`).
pub fn to_lower(text: &str) -> String {
    text.to_lowercase()
}
