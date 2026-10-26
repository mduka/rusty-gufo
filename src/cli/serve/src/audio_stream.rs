//! PCM16 and strict base64 for audio APIs and the WebSocket handshake
//! (`src/cli/serve/audio_stream.hpp`).

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Little-endian PCM16; empty for nonempty input with a non-finite sample.
pub fn encode_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        if !sample.is_finite() {
            return Vec::new();
        }
        let value = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn encode_audio_base64(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let a = chunk[0];
        let b = chunk.get(1).copied().unwrap_or(0);
        let c = chunk.get(2).copied().unwrap_or(0);
        encoded.push(char::from(ALPHABET[usize::from(a >> 2)]));
        encoded.push(char::from(ALPHABET[usize::from(((a & 3) << 4) | (b >> 4))]));
        encoded.push(if chunk.len() > 1 {
            char::from(ALPHABET[usize::from(((b & 15) << 2) | (c >> 6))])
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            char::from(ALPHABET[usize::from(c & 63)])
        } else {
            '='
        });
    }
    encoded
}

/// Strict canonical base64 (padding only at the end, zero pad bits) of at
/// most `limit` decoded bytes.
pub fn decode_audio_base64(text: &[u8], limit: usize) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(4) || text.len() / 4 > limit.saturating_add(2) / 3 {
        return None;
    }
    let mut bytes = Vec::with_capacity(text.len() / 4 * 3);
    for (index, quad) in text.chunks(4).enumerate() {
        let last = (index + 1) * 4 == text.len();
        let mut word: u32 = 0;
        let mut padding = 0;
        for (j, &c) in quad.iter().enumerate() {
            if c == b'=' {
                if j < 2 || !last {
                    return None;
                }
                padding += 1;
                word <<= 6;
            } else {
                let value = ALPHABET.iter().position(|&a| a == c)?;
                if padding != 0 {
                    return None;
                }
                word = (word << 6) | value as u32;
            }
        }
        if (padding == 1 && (word & 255) != 0) || (padding == 2 && (word & 65535) != 0) {
            return None;
        }
        let count = 3 - padding;
        if count > limit - bytes.len() {
            return None;
        }
        for j in 0..count {
            bytes.push(((word >> (16 - j * 8)) & 255) as u8);
        }
    }
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip_and_strictness() {
        for input in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"] {
            let encoded = encode_audio_base64(input);
            assert_eq!(
                decode_audio_base64(encoded.as_bytes(), 16).as_deref(),
                Some(input)
            );
        }
        assert_eq!(encode_audio_base64(b"foobar"), "Zm9vYmFy");
        assert!(
            decode_audio_base64(b"Zm9=", 16).is_none(),
            "nonzero pad bits"
        );
        assert!(
            decode_audio_base64(b"Zg==Zg==", 16).is_none(),
            "inner padding"
        );
        assert!(decode_audio_base64(b"Zm9v", 2).is_none(), "limit");
        assert!(decode_audio_base64(b"Zm9", 16).is_none(), "length");
    }

    #[test]
    fn pcm16_clamps_and_rejects_non_finite() {
        assert_eq!(
            encode_pcm16(&[0.0, 1.0, -2.0]),
            [0, 0, 0xff, 0x7f, 0x01, 0x80]
        );
        assert!(encode_pcm16(&[0.0, f32::NAN]).is_empty());
    }
}
