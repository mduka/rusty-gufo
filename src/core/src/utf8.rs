//! Streaming UTF-8 repair, a port of `core::Utf8Decoder`: token pieces can
//! split a scalar value or carry malformed byte-fallback output. Complete
//! sequences pass, a valid but incomplete tail waits for more bytes (unless
//! `last`), and every maximal malformed subsequence becomes U+FFFD.

#[derive(Default)]
pub struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    pub fn push(&mut self, bytes: &[u8], last: bool) -> String {
        self.pending.extend_from_slice(bytes);
        let pending = &self.pending;
        let mut output = String::with_capacity(pending.len());
        let mut offset = 0;
        while offset < pending.len() {
            let lead = pending[offset];
            if lead < 0x80 {
                output.push(char::from(lead));
                offset += 1;
                continue;
            }
            let length = match lead {
                0xC2..=0xDF => 2,
                0xE0..=0xEF => 3,
                0xF0..=0xF4 => 4,
                _ => 0,
            };
            if length == 0 {
                output.push('\u{FFFD}');
                offset += 1;
                continue;
            }
            let mut prefix = 1;
            while prefix < length && offset + prefix < pending.len() {
                let byte = pending[offset + prefix];
                let first_ok = prefix != 1
                    || !((lead == 0xE0 && byte < 0xA0)
                        || (lead == 0xED && byte > 0x9F)
                        || (lead == 0xF0 && byte < 0x90)
                        || (lead == 0xF4 && byte > 0x8F));
                if !(0x80..=0xBF).contains(&byte) || !first_ok {
                    break;
                }
                prefix += 1;
            }
            if prefix == length {
                match std::str::from_utf8(&pending[offset..offset + length]) {
                    Ok(text) => output.push_str(text),
                    Err(_) => output.push('\u{FFFD}'),
                }
                offset += length;
            } else if offset + prefix == pending.len() && !last {
                break;
            } else {
                output.push('\u{FFFD}');
                offset += prefix;
            }
        }
        self.pending.drain(..offset);
        output
    }
}

/// Repairs a complete byte string.
pub fn lossy(bytes: &[u8]) -> String {
    Utf8Decoder::default().push(bytes, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_scalars_wait_and_malformed_bytes_are_replaced() {
        let mut decoder = Utf8Decoder::default();
        let euro = "€".as_bytes();
        assert_eq!(decoder.push(&euro[..2], false), "");
        assert_eq!(decoder.push(&euro[2..], false), "€");
        assert_eq!(decoder.push(b"a\xFFb\xE2\x82", false), "a\u{FFFD}b");
        assert_eq!(decoder.push(b"", true), "\u{FFFD}");
        assert_eq!(lossy(b"\xED\xA0\x80x"), "\u{FFFD}\u{FFFD}\u{FFFD}x");
        for sample in [
            &b"\xF0\x9F\x98"[..],
            b"\xC3\x28",
            b"\xE0\x80\xAF",
            b"ok \xF4\x90\x80\x80",
        ] {
            assert_eq!(lossy(sample), String::from_utf8_lossy(sample));
        }
    }
}
