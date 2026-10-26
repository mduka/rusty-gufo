//! The quoting state of a model response (`quote_tracker.hpp`). Completed
//! inline spans, also across nonblank lines, and completed fences protect
//! documentation: a tool marker inside them is prose. Markers in unfinished
//! spans stay undecided while streaming, until a closing delimiter or the
//! final parse decides.

#[derive(Debug, Clone, Copy)]
struct Span {
    begin: usize,
    end: usize,
}

/// Owns the scanned bytes, so positions never refer to a reused buffer.
#[derive(Debug, Default, Clone)]
pub struct QuoteTracker {
    text: Vec<u8>,
    spans: Vec<Span>,
    unfinished_inline: Vec<Span>,
    bytes_scanned: usize,
    first_nonblank: Option<usize>,
    run_begin: usize,
    run_size: usize,
    run_char: u8,
    inline_begin: usize,
    inline_size: usize,
    fence_begin: usize,
    fence_size: usize,
    fence_char: u8,
    close_begin: usize,
    closing_fence: bool,
}

impl QuoteTracker {
    /// Restarts on `text`, scanning from `origin` (a phase boundary such as
    /// the end of `</think>` starts a fresh quote scope).
    pub fn reset(&mut self, text: &str, origin: usize) {
        let origin = origin.min(text.len());
        *self = Self {
            text: vec![b' '; origin],
            bytes_scanned: self.bytes_scanned,
            ..Self::default()
        };
        self.append(&text[origin..]);
    }

    pub fn append(&mut self, piece: &str) {
        let begin = self.text.len();
        self.text.extend_from_slice(piece.as_bytes());
        for cursor in begin..self.text.len() {
            self.read(cursor);
        }
        self.bytes_scanned += piece.len();
    }

    /// Whether `position` lies inside completed quoting.
    pub fn quoted_at(&self, position: usize) -> bool {
        let next = self.spans.partition_point(|span| span.begin <= position);
        if next > 0 && position < self.spans[next - 1].end {
            return true;
        }
        // A closing delimiter can be the final bytes, without a newline or
        // ordinary character to flush the pending run. This is provisional:
        // more text can make the last line cease to be a closing fence.
        if self.fence_size > 0 {
            if self.closing_fence && position >= self.fence_begin && position < self.close_begin {
                return true;
            }
            if self.run_size >= self.fence_size
                && self.run_char == self.fence_char
                && self.at_line_start(self.run_begin)
                && position >= self.fence_begin
                && position < self.run_begin
            {
                return true;
            }
        } else if self.inline_size > 0
            && self.run_char == b'`'
            && self.run_size == self.inline_size
            && position >= self.inline_begin
            && position < self.run_begin
        {
            return true;
        }
        false
    }

    /// Whether `position` lies inside quoting that has not closed, including
    /// an inline span a blank line ended: a call there needs the schema.
    pub fn unclosed_at(&self, position: usize) -> bool {
        if self.quoted_at(position) {
            return false;
        }
        if (self.fence_size > 0 && position >= self.fence_begin)
            || (self.inline_size > 0 && position >= self.inline_begin)
        {
            return true;
        }
        let next = self
            .unfinished_inline
            .partition_point(|span| span.begin <= position);
        next > 0 && position < self.unfinished_inline[next - 1].end
    }

    pub fn bytes_scanned(&self) -> usize {
        self.bytes_scanned
    }

    fn at_line_start(&self, position: usize) -> bool {
        self.first_nonblank == Some(position)
    }

    fn flush_run(&mut self) {
        if self.run_size == 0 {
            return;
        }
        if self.fence_size > 0 {
            if self.run_char == self.fence_char
                && self.run_size >= self.fence_size
                && self.at_line_start(self.run_begin)
            {
                self.closing_fence = true;
                self.close_begin = self.run_begin;
            }
        } else if self.inline_size == 0 && self.run_size >= 3 && self.at_line_start(self.run_begin)
        {
            self.fence_char = self.run_char;
            self.fence_size = self.run_size;
            self.fence_begin = self.run_begin + self.run_size;
        } else if self.run_char == b'`' {
            if self.inline_size == 0 {
                self.inline_size = self.run_size;
                self.inline_begin = self.run_begin + self.run_size;
            } else if self.run_size == self.inline_size {
                self.spans.push(Span {
                    begin: self.inline_begin,
                    end: self.run_begin,
                });
                self.inline_size = 0;
            }
        }
        self.run_size = 0;
    }

    fn read(&mut self, cursor: usize) {
        let byte = self.text[cursor];
        if self.first_nonblank.is_none() && !matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
            self.first_nonblank = Some(cursor);
        }
        if byte == b'`' || byte == b'~' {
            if self.run_size != 0 && self.run_char != byte {
                self.flush_run();
            }
            if self.run_size == 0 {
                self.run_begin = cursor;
                self.run_char = byte;
            }
            self.run_size += 1;
            return;
        }
        self.flush_run();
        if byte == b'\n' {
            if self.fence_size > 0 && self.closing_fence {
                self.spans.push(Span {
                    begin: self.fence_begin,
                    end: self.close_begin,
                });
                self.fence_size = 0;
                self.fence_char = 0;
            }
            if self.inline_size > 0 && self.first_nonblank.is_none() {
                // A blank line ends an unfinished inline span. Its range is
                // kept, so a possible call still needs the schema fallback.
                self.unfinished_inline.push(Span {
                    begin: self.inline_begin,
                    end: cursor,
                });
                self.inline_size = 0;
            }
            self.closing_fence = false;
            self.first_nonblank = None;
        } else if !matches!(byte, b' ' | b'\t' | b'\r') {
            self.closing_fence = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QuoteTracker;

    fn snapshot(text: &str) -> QuoteTracker {
        let mut tracker = QuoteTracker::default();
        tracker.reset(text, 0);
        tracker
    }

    #[test]
    fn completed_spans() {
        for (text, quoted) in [
            ("plain <marker>", false),
            ("`<marker>`", true),
            ("`<marker>", false),
            ("`<marker>\nnext", false),
            ("`before\n<marker>\nafter`", true),
            ("``before\n<marker>\nafter``", true),
            ("`before\n\n<marker>", false),
            ("`<marker>\n\nlater`", false),
            ("`before\n \t\r\n<marker>", false),
            ("`before <think>\n<marker>`", true),
            ("``<marker>``", true),
            ("``<marker>`", false),
            ("```python\n<marker>\n```", true),
            ("```python\n<marker>\n```\n", true),
            ("```python\n<marker>", false),
            ("~~~xml\n<marker>\n~~~", true),
            ("~~~xml\n<marker>\n```", false),
            ("````\n<marker>\n```", false),
            ("```\n<marker>\n```more text", false),
            ("```\n</think>\n<marker>", false),
            ("```\n</think>\n`<marker>`", false),
            ("`literal` and <marker>", false),
            ("```\nexample\n```\n<marker>", false),
        ] {
            let marker = text.find("<marker>").expect("marker");
            assert_eq!(snapshot(text).quoted_at(marker), quoted, "{text:?}");
            let mut incremental = QuoteTracker::default();
            for (index, c) in text.char_indices() {
                incremental.append(&text[index..index + c.len_utf8()]);
            }
            assert_eq!(
                incremental.quoted_at(marker),
                quoted,
                "bytewise reading agrees with the final snapshot: {text:?}"
            );
        }
    }

    #[test]
    fn unfinished_fence_decision() {
        let mut tracker = QuoteTracker::default();
        tracker.append("```xml\n<marker>\n");
        assert!(
            !tracker.quoted_at(7),
            "an unfinished fence does not hide a possible call"
        );
        tracker.append("```\n");
        assert!(
            tracker.quoted_at(7),
            "a completed fence turns the held marker into documentation"
        );
        tracker.reset("`<marker>\n\n", 0);
        assert!(
            !tracker.quoted_at(1) && tracker.unclosed_at(1),
            "a blank-line-terminated inline example still needs the schema"
        );
        tracker.reset("`<marker>`", 0);
        assert!(
            tracker.quoted_at(1) && !tracker.unclosed_at(1),
            "a complete inline span never uses the fallback"
        );
        tracker.reset("plain <marker>", 0);
        assert!(
            !tracker.quoted_at(6),
            "reset discards the preceding response's quote state"
        );
    }

    #[test]
    fn explicit_phase_boundary() {
        let text = "```reasoning\n</think>\n<marker>";
        let mut tracker = QuoteTracker::default();
        tracker.reset(text, text.find("</think>").expect("end") + "</think>".len());
        assert!(
            !tracker.quoted_at(text.find("<marker>").expect("marker")),
            "the actual phase transition starts a fresh quote scope"
        );
        tracker.reset("`literal </think>\n<marker>`", 0);
        assert!(
            tracker.quoted_at(18),
            "spelled phase tags inside inline text do not reset quotes"
        );
    }

    #[test]
    fn ownership_and_linearity() {
        let mut tracker = QuoteTracker::default();
        let mut reused = "`<marker>`".to_owned();
        tracker.reset(&reused, 0);
        reused = "no quotes".into();
        assert!(tracker.quoted_at(1), "the tracker owns its text");
        tracker.reset(&reused, 0);
        assert!(
            !tracker.quoted_at(1),
            "an explicit reset installs the new response"
        );
        tracker.reset("", 0);
        let before = tracker.bytes_scanned();
        const UNITS: usize = 4000;
        for i in 0..UNITS {
            tracker.append("<a>");
            assert!(
                !tracker.quoted_at(i * 3),
                "bracket-dense text remains prose"
            );
        }
        for i in (0..UNITS).rev() {
            assert!(!tracker.quoted_at(i * 3), "backward lookups remain prose");
        }
        assert_eq!(
            tracker.bytes_scanned() - before,
            UNITS * 3,
            "append and backward probes read each byte once"
        );
    }
}
