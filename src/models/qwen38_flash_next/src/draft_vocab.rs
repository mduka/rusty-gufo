//! The request vocabulary the MTP draft head scores, a port of
//! `draft_vocab.hpp`.
//!
//! Each draft token projects through the output head; over all 248,320 rows
//! that is a 675 MB read, about 3 ms on gfx1151 and 12 to 18% of decode
//! time. Byte-level BPE numbers tokens in merge order, and in this vocabulary
//! the early merges are the frequent tokens: the first 98,304 cover about
//! 99.8% of the tokens in English code and prose (measured, not guaranteed;
//! other text falls back to the full head). The draft scores those rows in
//! place, plus a small per-request table: the special tokens (chat, thinking
//! and tool-call markers at the end of the vocabulary) and every other token
//! the request has already used.
//!
//! Only proposals change. The target verifies every drafted token over the
//! full vocabulary, so greedy output is unchanged; sampled proposals are
//! corrected against the distribution they were drawn from.

/// Leading output rows the small head scores (`kMtpDraftVocab`).
pub const DRAFT_VOCAB: u32 = 98_304;

/// Rows of the per-request table: the special tokens, then the request's own
/// tokens at or above [`DRAFT_VOCAB`] in first-use order
/// (`kMtpDraftExtraRows`).
pub const DRAFT_EXTRA_ROWS: u32 = 1024;

/// Committed tokens the head choice looks back over
/// (`kMtpDraftVocabWindow`).
pub const DRAFT_VOCAB_WINDOW: usize = 32;

/// Whether a request's cycle may draft through the small head; the table
/// then decides per cycle ([`DraftVocab::fits`]). Penalties and constraints
/// never matter: drafts ignore both, and verification applies them over the
/// full target vocabulary. Sampled requests may at any session count:
/// verification corrects their proposals against the full target
/// distribution, and the choice reads only the request's own tokens, so
/// seeded output does not depend on overlapping requests. Greedy requests use
/// the small head on one session, when they decode alone (`occupancy`, the
/// requests in this decode step, is 1, so no peer shares the step), or in a
/// jointly planned batch; greedy requests in an unplanned multi-session batch
/// keep the full head (`MtpDraftSubsetAllowed`).
pub const fn draft_subset_allowed(
    sampled: bool,
    sessions: u32,
    occupancy: u32,
    planned_batch: bool,
) -> bool {
    sampled || sessions == 1 || occupancy == 1 || planned_batch
}

/// The request's draft vocabulary and when to use it (`MtpDraftVocab`). A
/// token is a miss when it is outside the prefix, not special and not yet in
/// the request's table (or the table is full). A draft cycle uses the small
/// head when its anchor is not a miss and at most one of the last
/// [`DRAFT_VOCAB_WINDOW`] committed tokens was: text that keeps bringing new
/// tokens (a short reply in another script) drafts over the full vocabulary,
/// a lone new word does not leave the small head, and a request that repeats
/// its own vocabulary gets the small head. Everything is a function of the
/// request's own token history, so batched, isolated and restored sessions
/// draft the same way.
#[derive(Debug, Clone, Default)]
pub struct DraftVocab {
    capacity: usize,
    rare: Vec<i32>,
    known: Vec<bool>,
    scanned: usize,
    last_miss: Option<usize>,
    /// The miss before `last_miss`.
    prior_miss: Option<usize>,
}

impl DraftVocab {
    /// `rare_rows`: table rows left after the special tokens.
    pub fn new(rare_rows: u32) -> Self {
        Self {
            capacity: rare_rows as usize,
            ..Self::default()
        }
    }

    pub fn reset(&mut self) {
        self.rare.clear();
        self.known.clear();
        self.scanned = 0;
        self.last_miss = None;
        self.prior_miss = None;
    }

    /// Takes in `committed[scanned()..]` and appends the newly tabled tokens
    /// to `added` in table order. `common(id)` is true for prefix and special
    /// IDs.
    pub fn catch_up(
        &mut self,
        committed: &[i32],
        common: impl Fn(i32) -> bool,
        mut added: Option<&mut Vec<i32>>,
    ) {
        while self.scanned < committed.len() {
            let id = committed[self.scanned];
            if !common(id) && !self.knows(id) {
                self.prior_miss = self.last_miss;
                self.last_miss = Some(self.scanned);
                if id >= 0 && self.rare.len() < self.capacity {
                    let at = id as usize;
                    if self.known.len() <= at {
                        self.known.resize(at + 1, false);
                    }
                    self.known[at] = true;
                    self.rare.push(id);
                    if let Some(added) = added.as_deref_mut() {
                        added.push(id);
                    }
                }
            }
            self.scanned += 1;
        }
    }

    /// Whether a cycle anchored on `anchor`, after the scanned tokens, drafts
    /// through the small head.
    pub fn fits(&self, anchor: i32, common: impl Fn(i32) -> bool) -> bool {
        (common(anchor) || self.knows(anchor)) && self.quiet()
    }

    /// Whether the window allows the small head; [`fits`](Self::fits) also
    /// needs the anchor.
    pub fn quiet(&self) -> bool {
        self.prior_miss
            .is_none_or(|miss| miss + DRAFT_VOCAB_WINDOW < self.scanned)
    }

    pub fn scanned(&self) -> usize {
        self.scanned
    }

    /// Tabled rare tokens, in table order.
    pub fn rare(&self) -> &[i32] {
        &self.rare
    }

    fn knows(&self, id: i32) -> bool {
        id >= 0 && self.known.get(id as usize).copied().unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// IDs below 100 stand for the prefix and the special tokens.
    fn common(id: i32) -> bool {
        (0..100).contains(&id)
    }

    /// Sampled requests may use the small head at any session count and
    /// occupancy, in or out of a batch, so a seeded request drafts the same
    /// way whatever overlaps it. Greedy requests, with or without penalties
    /// and constraints, need one session, to decode alone or a jointly
    /// planned batch.
    #[test]
    fn head_eligibility() {
        for sessions in [1, 2, 8] {
            for occupancy in [1, 2, 8] {
                for planned in [false, true] {
                    assert!(draft_subset_allowed(true, sessions, occupancy, planned));
                    assert_eq!(
                        draft_subset_allowed(false, sessions, occupancy, planned),
                        sessions == 1 || occupancy == 1 || planned,
                        "greedy eligibility changed"
                    );
                }
            }
        }
    }

    /// Text inside the prefix needs no table rows and keeps the small head.
    #[test]
    fn common_text_keeps_the_small_head() {
        let mut vocab = DraftVocab::new(8);
        let tokens = vec![1; DRAFT_VOCAB_WINDOW + 10];
        let mut added = Vec::new();
        vocab.catch_up(&tokens, common, Some(&mut added));
        assert!(
            added.is_empty() && vocab.rare().is_empty(),
            "common tokens were tabled"
        );
        assert!(vocab.fits(2, common), "common text left the small head");
        assert!(DraftVocab::new(8).fits(2, common), "empty history");
    }

    /// A new rare token is tabled in first-use order. One miss in the window
    /// keeps the small head; a second one leaves it until the earlier of the
    /// two has left the window.
    #[test]
    fn a_second_miss_leaves_the_small_head_for_a_window() {
        let mut vocab = DraftVocab::new(8);
        let mut tokens = vec![1, 500, 1];
        let mut added = Vec::new();
        vocab.catch_up(&tokens, common, Some(&mut added));
        assert!(
            vocab.fits(1, common) && vocab.fits(500, common),
            "a lone miss left the small head"
        );
        tokens.extend([700, 500]);
        vocab.catch_up(&tokens, common, Some(&mut added));
        assert_eq!(added, [500, 700], "tabled order");
        assert!(
            !vocab.fits(1, common),
            "small head right after a second miss"
        );
        // The earlier miss (500, index 1) stays in the window for W more
        // tokens.
        tokens.resize(1 + DRAFT_VOCAB_WINDOW, 500);
        added.clear();
        vocab.catch_up(&tokens, common, Some(&mut added));
        assert!(added.is_empty(), "a known token was tabled again");
        assert!(!vocab.fits(1, common), "miss left the window too early");
        tokens.push(1);
        vocab.catch_up(&tokens, common, Some(&mut added));
        assert!(
            vocab.fits(1, common) && vocab.fits(700, common),
            "known text did not return to the small head"
        );
        assert!(!vocab.fits(900, common), "an untabled anchor drafted small");
    }

    /// A full table turns every further rare token into a miss.
    #[test]
    fn a_full_table_turns_rare_tokens_into_misses() {
        let mut vocab = DraftVocab::new(2);
        let mut tokens = vec![500, 600, 700];
        tokens.resize(3 + DRAFT_VOCAB_WINDOW, 1);
        let mut added = Vec::new();
        vocab.catch_up(&tokens, common, Some(&mut added));
        assert_eq!(added, [500, 600], "table overfilled");
        assert!(
            vocab.fits(600, common) && !vocab.fits(700, common),
            "untabled token proposed"
        );
        tokens.extend([700, 700]);
        vocab.catch_up(&tokens, common, Some(&mut added));
        assert!(!vocab.fits(1, common), "overflow token is not a miss");
    }

    /// Reset forgets the table and the history.
    #[test]
    fn reset_forgets_table_and_history() {
        let mut vocab = DraftVocab::new(8);
        vocab.catch_up(&[500], common, None);
        vocab.reset();
        assert!(
            vocab.scanned() == 0 && vocab.rare().is_empty() && vocab.fits(1, common),
            "reset kept state"
        );
        assert!(!vocab.fits(500, common), "reset kept a tabled token");
    }
}
