//! Text cleanup of the reference (`synthesizer.cpp`'s `CollapseWhitespace`
//! and `WhitespaceAt`).

/// Unicode whitespace as Python's str regex `\s` matches it, at `text[i]`;
/// its byte length or 0.
fn whitespace_at(text: &[u8], i: usize) -> usize {
    let byte = text[i];
    if byte == b' ' || (0x09..=0x0D).contains(&byte) || (0x1C..=0x1F).contains(&byte) {
        return 1;
    }
    let starts = |sequence: &[u8]| text[i..].starts_with(sequence);
    if starts(b"\xC2\x85") || starts(b"\xC2\xA0") {
        return 2;
    }
    if starts(b"\xE1\x9A\x80")
        || starts(b"\xE2\x80\xA8")
        || starts(b"\xE2\x80\xA9")
        || starts(b"\xE2\x80\xAF")
        || starts(b"\xE2\x81\x9F")
        || starts(b"\xE3\x80\x80")
    {
        return 3;
    }
    if byte == 0xE2 && i + 2 < text.len() && text[i + 1] == 0x80 && text[i + 2] <= 0x8A {
        return 3; // U+2000..U+200A
    }
    0
}

/// The reference's text cleanup: newlines become spaces and whitespace runs
/// one space (`CollapseWhitespace`).
pub fn collapse_whitespace(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut space = false;
    let mut i = 0;
    while i < bytes.len() {
        let length = whitespace_at(bytes, i);
        if length > 0 {
            space = true;
            i += length;
            continue;
        }
        if space {
            result.push(b' ');
        }
        space = false;
        result.push(bytes[i]);
        i += 1;
    }
    if space {
        result.push(b' ');
    }
    // Only whole whitespace sequences are replaced, so the bytes stay UTF-8.
    String::from_utf8_lossy(&result).into_owned()
}

#[cfg(test)]
mod tests {
    use super::collapse_whitespace;

    #[test]
    fn runs_of_whitespace_become_one_space() {
        assert_eq!(collapse_whitespace("a\n\tb  c"), "a b c");
        assert_eq!(collapse_whitespace("  a "), " a ");
        assert_eq!(collapse_whitespace("a\u{3000}\u{2009}b\u{00A0}"), "a b ");
        assert_eq!(collapse_whitespace("\u{200B}"), "\u{200B}");
        assert_eq!(collapse_whitespace(""), "");
    }
}
