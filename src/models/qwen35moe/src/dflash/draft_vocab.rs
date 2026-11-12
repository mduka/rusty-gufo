//! The request vocabulary the DFlash2 block head scores, a port of the
//! family's `draft_vocab.hpp`.
//!
//! Over all 248,320 rows each draft row reads the whole output head. Byte-level
//! BPE numbers tokens in merge order, and in this vocabulary the early merges
//! are the frequent tokens: the first 98,304 cover about 99.8% of the tokens
//! in English code and prose (measured, not guaranteed; other text falls back
//! to the full head). The draft scores those rows in place, plus a small
//! per-request table: the special tokens (chat, thinking and tool-call
//! markers at the end of the vocabulary) and every other token the request
//! has already used.
//!
//! Only proposals change. The target verifies every drafted token over the
//! full vocabulary, so greedy output is unchanged; sampled proposals are
//! corrected against the distribution they were drawn from.

/// Leading output rows the small head scores (`kDraftVocabPrefix`).
pub const DRAFT_VOCAB_PREFIX: u32 = 98_304;

/// Rows of the per-request table: the special tokens, then the request's own
/// tokens at or above [`DRAFT_VOCAB_PREFIX`] in first-use order
/// (`kDraftVocabExtraRows`).
pub const DRAFT_EXTRA_ROWS: u32 = 1024;

/// Committed tokens the head choice looks back over (`kDraftVocabWindow`).
pub const DRAFT_VOCAB_WINDOW: usize = 32;

/// Whether a request's cycle may draft through the small head; the table
/// then decides per cycle ([`DraftVocab::fits`]). Penalties and constraints
/// never matter: drafts ignore both, and verification applies them over the
/// full target vocabulary. Sampled requests may at any session count:
/// verification corrects their proposals against the full target
/// distribution, and the choice reads only the request's own tokens, so
/// seeded output does not depend on overlapping requests. Greedy requests use
/// the small head on one session, when they decode alone (`occupancy`, the
/// requests in this decode step, is 1), or in a jointly planned batch;
/// greedy requests in an unplanned multi-session batch keep the full head
/// (`DraftSubsetAllowed`).
pub const fn draft_subset_allowed(
    sampled: bool,
    sessions: u32,
    occupancy: u32,
    planned_batch: bool,
) -> bool {
    sampled || sessions == 1 || occupancy == 1 || planned_batch
}

/// The request's draft vocabulary and when to use it (`DraftVocab`). A token
/// is a miss when it is outside the prefix, not special and not yet in the
/// request's table (or the table is full). A draft cycle uses the small head
/// when its anchor is not a miss and at most one of the last
/// [`DRAFT_VOCAB_WINDOW`] committed tokens was: text that keeps bringing new
/// tokens drafts over the full vocabulary, a lone new word does not leave
/// the small head, and a request that repeats its own vocabulary gets the
/// small head. Everything is a function of the request's own token history,
/// so batched, isolated and restored sessions draft the same way.
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
    /// to `added` in table order (`CatchUp`). `common(id)` is true for prefix
    /// and special IDs.
    pub fn catch_up(
        &mut self,
        committed: &[u32],
        common: impl Fn(i32) -> bool,
        added: &mut Vec<i32>,
    ) {
        while self.scanned < committed.len() {
            // Token IDs travel as `int32` in C++ (`reinterpret_cast`).
            let id = committed[self.scanned] as i32;
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
                    added.push(id);
                }
            }
            self.scanned += 1;
        }
    }

    /// Whether a cycle anchored on `anchor`, after the scanned tokens, drafts
    /// through the small head (`Fits`).
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

    #[test]
    fn head_eligibility() {
        for sessions in [1, 2, 8] {
            for occupancy in [1, 2, 8] {
                for planned in [false, true] {
                    assert!(draft_subset_allowed(true, sessions, occupancy, planned));
                    assert_eq!(
                        draft_subset_allowed(false, sessions, occupancy, planned),
                        sessions == 1 || occupancy == 1 || planned
                    );
                }
            }
        }
    }

    #[test]
    fn misses_table_and_window() {
        let mut vocab = DraftVocab::new(2);
        let mut added = Vec::new();
        // One miss leaves the window quiet; a second one within the window
        // does not.
        vocab.catch_up(&[1, 500, 2], common, &mut added);
        assert_eq!(added, [500]);
        assert!(vocab.fits(500, common) && vocab.fits(3, common));
        assert!(!vocab.fits(600, common), "an unknown anchor misses");
        vocab.catch_up(&[1, 500, 2, 600], common, &mut added);
        assert_eq!(vocab.rare(), [500, 600]);
        assert!(!vocab.quiet());
        // A full table still counts misses but tables nothing.
        let mut tokens = vec![1, 500, 2, 600, 700];
        added.clear();
        vocab.catch_up(&tokens, common, &mut added);
        assert!(added.is_empty() && !vocab.fits(700, common));
        tokens.resize(tokens.len() + DRAFT_VOCAB_WINDOW, 1);
        vocab.catch_up(&tokens, common, &mut added);
        assert!(vocab.quiet() && vocab.fits(600, common) && !vocab.fits(700, common));
        vocab.reset();
        assert_eq!((vocab.scanned(), vocab.rare().len()), (0, 0));
    }
}
