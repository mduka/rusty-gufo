//! The small draft head's vocabulary, a port of `draft_vocab.hpp` and
//! `mtp_draft_vocab.hpp`.
//!
//! Each draft token projects through the output head; over all 248,320 rows
//! that reads the whole head. Byte-level BPE numbers tokens in merge order,
//! and in this vocabulary the early merges are the frequent tokens: the first
//! 98,304 cover about 99.8% of the tokens in English code and prose
//! (measured, not guaranteed; other text falls back to the full head). The
//! draft scores those rows in place, plus a small per-request table: the
//! special tokens (chat, thinking and tool-call markers at the end of the
//! vocabulary) and every other token the request has already used.
//!
//! Only proposals change. The target verifies every drafted token over the
//! full vocabulary, so greedy output is unchanged; sampled proposals are
//! corrected against the distribution they were drawn from.

/// Leading output rows the small head scores (`kDraftVocabPrefix`).
pub const DRAFT_VOCAB: u32 = 98_304;

/// Rows of the per-request table: the special tokens, then the request's own
/// tokens at or above [`DRAFT_VOCAB`] in first-use order
/// (`kDraftVocabExtraRows`).
pub const DRAFT_EXTRA_ROWS: u32 = 1024;

/// Committed tokens the head choice looks back over (`kDraftVocabWindow`).
pub const DRAFT_VOCAB_WINDOW: usize = 32;

/// Rows of the Qwen3.5/3.6/3.8 vocabulary the small head assumes
/// (`kQwenMtpDraftVocabRows`).
pub const DRAFT_VOCAB_ROWS: u32 = 248_320;

/// The architecture whose MTP drafts through the small head, where it is
/// measured (`kQwenMtpDraftArchitecture`); other targets keep the full head.
pub const DRAFT_ARCHITECTURE: &str = "qwen35moe";

/// Whether a request's cycle may draft through the small head
/// (`DraftSubsetAllowed`); the table then decides per cycle. Sampled
/// requests may at any session count; greedy requests on one session, when
/// they decode alone (`occupancy` 1) or in a jointly planned batch.
pub const fn draft_subset_allowed(
    sampled: bool,
    sessions: u32,
    occupancy: u32,
    planned_batch: bool,
) -> bool {
    sampled || sessions == 1 || occupancy == 1 || planned_batch
}

/// The special tokens that open every draft table, ascending: those at or
/// above the prefix (`QwenMtpDraftSpecials`). `None` when the draft keeps
/// the full head: another architecture or vocabulary, a special token
/// outside the vocabulary, or more special tokens than table rows.
pub fn draft_specials(
    architecture: &str,
    vocab_size: u32,
    specials: impl IntoIterator<Item = u32>,
) -> Option<Vec<i32>> {
    if architecture != DRAFT_ARCHITECTURE || vocab_size != DRAFT_VOCAB_ROWS {
        return None;
    }
    let mut ids = Vec::new();
    for id in specials {
        if id >= vocab_size {
            return None;
        }
        if id >= DRAFT_VOCAB {
            ids.push(id as i32);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    (ids.len() < DRAFT_EXTRA_ROWS as usize).then_some(ids)
}

/// The request's draft vocabulary and when to use it (`DraftVocab`). A
/// token is a miss when it is outside the prefix, not special and not yet in
/// the request's table (or the table is full). A cycle uses the small head
/// when its anchor is not a miss and at most one of the last
/// [`DRAFT_VOCAB_WINDOW`] committed tokens was. Everything is a function of
/// the request's own token history, so batched, isolated and restored
/// sessions draft the same way.
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
        committed: &[i32],
        common: impl Fn(i32) -> bool,
        added: &mut Vec<i32>,
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

    /// Whether the window allows the small head (`Quiet`).
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

/// What [`DraftTable::sync`] changed: with `restart`, the rows above the
/// special tokens are dropped before `added` is appended in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Update<'a> {
    pub restart: bool,
    pub added: &'a [i32],
}

/// One session's draft table and head choice (`QwenMtpDraftTable`), kept in
/// step with the token sequence each proposal receives: committed history,
/// then the anchor. The anchor is decided against the history before it and
/// enters the table with the next proposal. The table is a function of that
/// sequence alone, so serial, batched and restored sessions draft the same
/// way.
#[derive(Debug, Clone)]
pub struct DraftTable {
    specials: Vec<i32>,
    vocab: DraftVocab,
    /// Scanned tokens, as table IDs.
    history: Vec<i32>,
    added: Vec<i32>,
    stale: bool,
}

impl DraftTable {
    pub fn new(specials: Vec<i32>) -> Self {
        let rare_rows = DRAFT_EXTRA_ROWS - specials.len() as u32;
        Self {
            specials,
            vocab: DraftVocab::new(rare_rows),
            history: Vec::new(),
            added: Vec::new(),
            stale: true,
        }
    }

    /// The session's history was replaced (reset, restore, new request): the
    /// next sync rescans it from the first token (`Forget`).
    pub fn forget(&mut self) {
        self.stale = true;
    }

    /// Brings the table up to `tokens` and returns the change (`Sync`).
    pub fn sync(&mut self, tokens: &[u32]) -> Update<'_> {
        let size = tokens.len().saturating_sub(1);
        let mut restart = false;
        if self.stale || self.history.len() > size {
            self.vocab.reset();
            self.history.clear();
            self.stale = false;
            restart = true;
        }
        let scanned = self.history.len();
        self.history
            .extend(tokens[scanned..size].iter().map(|&t| Self::id(t)));
        self.added.clear();
        let specials = &self.specials;
        self.vocab.catch_up(
            &self.history,
            |id| Self::common(specials, id),
            &mut self.added,
        );
        Update {
            restart,
            added: &self.added,
        }
    }

    /// Whether a proposal anchored on `anchor`, after the synced history,
    /// scores the small head (`Fits`).
    pub fn fits(&self, anchor: u32) -> bool {
        let specials = &self.specials;
        self.vocab
            .fits(Self::id(anchor), |id| Self::common(specials, id))
    }

    /// Table rows in use: the special tokens, then the tabled tokens.
    pub fn rows(&self) -> u32 {
        (self.specials.len() + self.vocab.rare().len()) as u32
    }

    pub fn specials(&self) -> &[i32] {
        &self.specials
    }

    /// IDs outside the vocabulary read as -1: misses that never take a row.
    fn id(token: u32) -> i32 {
        if token < DRAFT_VOCAB_ROWS {
            token as i32
        } else {
            -1
        }
    }

    fn common(specials: &[i32], id: i32) -> bool {
        id >= 0 && ((id as u32) < DRAFT_VOCAB || specials.binary_search(&id).is_ok())
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
    fn a_second_miss_leaves_the_small_head_for_a_window() {
        let mut vocab = DraftVocab::new(8);
        let mut tokens = vec![1, 500, 1];
        let mut added = Vec::new();
        vocab.catch_up(&tokens, common, &mut added);
        assert!(vocab.fits(1, common) && vocab.fits(500, common));
        tokens.extend([700, 500]);
        vocab.catch_up(&tokens, common, &mut added);
        assert_eq!(added, [500, 700]);
        assert!(!vocab.fits(1, common));
        tokens.resize(1 + DRAFT_VOCAB_WINDOW, 500);
        added.clear();
        vocab.catch_up(&tokens, common, &mut added);
        assert!(added.is_empty() && !vocab.fits(1, common));
        tokens.push(1);
        vocab.catch_up(&tokens, common, &mut added);
        assert!(vocab.fits(1, common) && vocab.fits(700, common));
        assert!(!vocab.fits(900, common));
    }

    #[test]
    fn a_full_table_turns_rare_tokens_into_misses() {
        let mut vocab = DraftVocab::new(2);
        let mut tokens = vec![500, 600, 700];
        tokens.resize(3 + DRAFT_VOCAB_WINDOW, 1);
        let mut added = Vec::new();
        vocab.catch_up(&tokens, common, &mut added);
        assert_eq!(added, [500, 600]);
        assert!(vocab.fits(600, common) && !vocab.fits(700, common));
        tokens.extend([700, 700]);
        vocab.catch_up(&tokens, common, &mut added);
        assert!(!vocab.fits(1, common));
    }

    /// Specials sort, deduplicate and keep only those above the prefix; other
    /// architectures, vocabularies and out-of-range IDs keep the full head.
    #[test]
    fn specials_open_the_table() {
        let ids = [248_046, 5, 248_045, 248_046, DRAFT_VOCAB];
        assert_eq!(
            draft_specials("qwen35moe", DRAFT_VOCAB_ROWS, ids),
            Some(vec![DRAFT_VOCAB as i32, 248_045, 248_046])
        );
        assert_eq!(draft_specials("qwen35", DRAFT_VOCAB_ROWS, ids), None);
        assert_eq!(draft_specials("qwen35moe", 151_936, ids), None);
        assert_eq!(
            draft_specials("qwen35moe", DRAFT_VOCAB_ROWS, [DRAFT_VOCAB_ROWS]),
            None
        );
        let many = (0..DRAFT_EXTRA_ROWS).map(|i| DRAFT_VOCAB + i);
        assert_eq!(draft_specials("qwen35moe", DRAFT_VOCAB_ROWS, many), None);
    }

    /// The table follows the sequence through the anchor: the anchor is
    /// tabled with the next proposal, a shorter sequence restarts it, and a
    /// forgotten table rescans from the first token.
    #[test]
    fn the_table_follows_the_sequence() {
        let rare = DRAFT_VOCAB + 7;
        let mut table = DraftTable::new(vec![248_045]);
        let update = table.sync(&[1, 2, rare]);
        assert!(update.restart && update.added.is_empty());
        assert!(!table.fits(rare + 1) && table.rows() == 1);
        let update = table.sync(&[1, 2, rare, 3]);
        assert!(!update.restart);
        assert_eq!(update.added, [rare as i32]);
        assert!(table.fits(rare) && table.fits(248_045) && table.rows() == 2);
        let update = table.sync(&[1, 2]);
        assert!(update.restart && update.added.is_empty() && table.rows() == 1);
        table.forget();
        let update = table.sync(&[rare, 1]);
        assert!(update.restart);
        assert_eq!(update.added, [rare as i32]);
        // Out-of-vocabulary IDs are misses that never take a row.
        let update = table.sync(&[rare, 1, u32::MAX, 2]);
        assert!(update.added.is_empty() && !table.fits(1));
    }
}
