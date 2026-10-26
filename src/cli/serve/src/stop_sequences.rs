//! Request stop sequences (`src/cli/serve/stop_sequences.hpp`): validation,
//! the OpenAI and Anthropic request spellings, and the filter that matches
//! accepted output bytes independently of how token pieces are split.

use gufo_core::json::Value;

pub const MAX_STOP_SEQUENCES: usize = 64;
pub const MAX_STOP_SEQUENCE_BYTES: usize = 4096;
pub const MAX_STOP_TOTAL_BYTES: usize = 16384;
/// OpenAI accepts at most four `stop` strings.
const MAX_OPENAI_STOP_SEQUENCES: usize = 4;

pub fn validate_stop_sequences(sequences: &[String], max_count: usize) -> Result<(), String> {
    if sequences.len() > max_count {
        return Err(format!("too many stop sequences (maximum {max_count})"));
    }
    let mut bytes = 0usize;
    for sequence in sequences {
        if sequence.is_empty() || sequence.len() > MAX_STOP_SEQUENCE_BYTES {
            return Err(format!(
                "stop sequences must contain 1 to {MAX_STOP_SEQUENCE_BYTES} UTF-8 bytes"
            ));
        }
        bytes += sequence.len();
        if bytes > MAX_STOP_TOTAL_BYTES {
            return Err("stop sequences exceed the total byte limit".into());
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopSequenceFormat {
    /// `stop`: null, one string or an array of up to four strings.
    OpenAi,
    /// `stop_sequences`: an array of strings.
    Anthropic,
}

/// The stop sequences of a request field; absent means none.
pub fn parse_stop_sequences(
    value: Option<&Value>,
    format: StopSequenceFormat,
) -> Result<Vec<String>, String> {
    let openai = format == StopSequenceFormat::OpenAi;
    let Some(value) = value.filter(|v| !(openai && v.is_null())) else {
        return Ok(Vec::new());
    };
    let sequences = if openai && value.is_string() {
        vec![value.str().to_owned()]
    } else if value.is_array() {
        value
            .items()
            .iter()
            .map(|item| {
                item.is_string()
                    .then(|| item.str().to_owned())
                    .ok_or_else(|| "stop sequences must be strings".to_owned())
            })
            .collect::<Result<_, _>>()?
    } else {
        return Err(if openai {
            "'stop' must be null, a string, or an array of strings".into()
        } else {
            "'stop_sequences' must be an array of strings".into()
        });
    };
    validate_stop_sequences(
        &sequences,
        if openai {
            MAX_OPENAI_STOP_SEQUENCES
        } else {
            MAX_STOP_SEQUENCES
        },
    )?;
    Ok(sequences)
}

/// Matches raw accepted output, retaining only a possible stop prefix. The
/// first completed match wins; ties at the same end prefer the longer
/// sequence.
#[derive(Debug, Default)]
pub struct StopSequenceFilter {
    sequences: Vec<String>,
    pending: Vec<u8>,
    matched: Option<usize>,
}

impl StopSequenceFilter {
    pub fn new(sequences: Vec<String>) -> Result<Self, String> {
        validate_stop_sequences(&sequences, MAX_STOP_SEQUENCES)?;
        Ok(Self {
            sequences,
            ..Self::default()
        })
    }

    pub fn enabled(&self) -> bool {
        !self.sequences.is_empty()
    }

    pub fn stopped(&self) -> bool {
        self.matched.is_some()
    }

    pub fn matched_sequence(&self) -> Option<&str> {
        self.matched.map(|index| self.sequences[index].as_str())
    }

    /// The bytes of `piece` and earlier withheld ones that can no longer
    /// start a stop sequence; nothing once stopped.
    pub fn push(&mut self, piece: &[u8]) -> Vec<u8> {
        if self.stopped() {
            return Vec::new();
        }
        self.pending.extend_from_slice(piece);
        let mut best: Option<(usize, usize)> = None;
        for (index, sequence) in self.sequences.iter().enumerate() {
            let Some(start) = find(&self.pending, sequence.as_bytes()) else {
                continue;
            };
            let end = start + sequence.len();
            if best.is_none_or(|(best_end, best_start)| {
                end < best_end || (end == best_end && start < best_start)
            }) {
                best = Some((end, start));
                self.matched = Some(index);
            }
        }
        if let Some((_, start)) = best {
            let mut safe = std::mem::take(&mut self.pending);
            safe.truncate(start);
            return safe;
        }
        let mut keep = 0;
        for sequence in &self.sequences {
            let sequence = sequence.as_bytes();
            let longest = self.pending.len().min(sequence.len() - 1);
            if let Some(length) = (keep + 1..=longest)
                .rev()
                .find(|&n| self.pending.ends_with(&sequence[..n]))
            {
                keep = length;
            }
        }
        let tail = self.pending.split_off(self.pending.len() - keep);
        std::mem::replace(&mut self.pending, tail)
    }

    /// The withheld prefix once generation ends without a match.
    pub fn finish(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pending)
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(sequences: &[&str]) -> StopSequenceFilter {
        StopSequenceFilter::new(sequences.iter().map(|s| (*s).to_owned()).collect()).unwrap()
    }

    #[test]
    fn a_match_split_across_pieces_withholds_only_its_prefix() {
        let mut f = filter(&["END"]);
        assert_eq!(f.push(b"abcE"), b"abc");
        assert_eq!(f.push(b"N"), b"");
        assert!(!f.stopped());
        assert_eq!(f.push(b"Dtail"), b"");
        assert!(f.stopped());
        assert_eq!(f.matched_sequence(), Some("END"));
        assert_eq!(f.push(b"more"), b"");
    }

    #[test]
    fn an_unmatched_prefix_is_released_or_flushed() {
        let mut f = filter(&["END"]);
        assert_eq!(f.push(b"xE"), b"x");
        assert_eq!(f.push(b"x"), b"Ex");
        assert_eq!(f.push(b"EN"), b"");
        assert_eq!(f.finish(), b"EN");
    }

    #[test]
    fn the_earliest_end_wins_and_ties_prefer_the_longer_sequence() {
        let mut f = filter(&["bcd", "cd", "abcdef"]);
        assert_eq!(f.push(b"xabcdef"), b"xa");
        assert_eq!(f.matched_sequence(), Some("bcd"));
        let mut f = filter(&["c", "bc"]);
        assert_eq!(f.push(b"abc"), b"a");
        assert_eq!(f.matched_sequence(), Some("bc"));
    }

    #[test]
    fn matching_is_independent_of_token_boundaries() {
        let check = |stops: &[&str], text: &str, expected: &str, matched: &str| {
            let text = text.as_bytes();
            // Single-byte pieces and every two-piece boundary.
            for split in 0..=text.len() + 1 {
                let mut f = filter(stops);
                let mut output = Vec::new();
                if split > text.len() {
                    for byte in text {
                        output.extend(f.push(std::slice::from_ref(byte)));
                    }
                } else {
                    output.extend(f.push(&text[..split]));
                    output.extend(f.push(&text[split..]));
                }
                output.extend(f.finish());
                assert_eq!(output, expected.as_bytes(), "split {split}");
                assert_eq!(f.stopped(), !matched.is_empty());
                if f.stopped() {
                    assert_eq!(f.matched_sequence(), Some(matched));
                }
            }
        };
        check(&["<STOP>"], "before<STOP>after", "before", "<STOP>");
        check(&["END", "STOP"], "stENDlaterSTOP", "st", "END");
        check(&["abc", "b"], "abc", "a", "b");
        check(&["aba", "ba"], "aba", "", "aba");
        check(&["abab"], "aaababtail", "aa", "abab");
        check(&["┌終"], "hi┌終later", "hi", "┌終");
        check(&["STOP"], "SSTOSTxST", "SSTOSTxST", "");
        check(&["STOP"], "STOP", "", "STOP");
    }

    #[test]
    fn requests_spell_stops_per_protocol() {
        let parse = |text: &str, format| {
            let body = gufo_core::json::parse_str(text).unwrap();
            parse_stop_sequences(body.find("stop"), format)
        };
        let openai = StopSequenceFormat::OpenAi;
        assert_eq!(parse(r#"{"stop":null}"#, openai), Ok(vec![]));
        assert_eq!(parse(r#"{"stop":"x"}"#, openai), Ok(vec!["x".to_owned()]));
        assert_eq!(parse(r#"{"stop":["a","b"]}"#, openai).unwrap().len(), 2);
        assert!(parse(r#"{"stop":["a","b","c","d","e"]}"#, openai).is_err());
        assert!(parse(r#"{"stop":[""]}"#, openai).is_err());
        assert!(parse(r#"{"stop":[1]}"#, openai).is_err());
        let anthropic = StopSequenceFormat::Anthropic;
        assert!(parse(r#"{"stop":"x"}"#, anthropic).is_err());
        assert!(parse(r#"{"stop":null}"#, anthropic).is_err());
        assert_eq!(
            parse(r#"{"stop":["a","b","c","d","e"]}"#, anthropic)
                .unwrap()
                .len(),
            5
        );
        let long = format!(
            r#"{{"stop":["{}"]}}"#,
            "x".repeat(MAX_STOP_SEQUENCE_BYTES + 1)
        );
        assert!(parse(&long, anthropic).is_err());
    }
}
