//! The drafter's small head over frequent tokens (`draft_vocab.cpp`,
//! `draft_vocab.hpp`).
//!
//! The `gemma4-assistant` head projects onto all 262,144 tokens: as Q4_K a
//! 151 MB read, 0.65 ms on gfx1151 and about a third of a drafter step.
//! Gemma's token ids are not ordered by frequency, but the merge that
//! creates a token is a usable proxy, so the device keeps the head rows in
//! [`draft_head_order`]. On 26B outputs the first 98,304 rows (plus the
//! request's own tokens, see [`DraftVocab`]) serve 92-100% of code draft
//! cycles and 55-72% of English prose cycles; other scripts fall back to the
//! full head.
//!
//! Only proposals change. The target verifies every drafted token over the
//! full vocabulary, so greedy output is unchanged, and sampled proposals are
//! corrected against the top candidates they were drawn from.
//!
//! Only the 26B-A4B uses it: its ~30 ms cycle gains 4-6% on code and up to
//! 3.5% on English prose. The dense 31B's ~130 ms cycle gains nothing
//! measurable and loses up to 5% to missed proposals in other languages.

use crate::tokenizer::{TokenId, TokenType, Tokenizer};

/// Drafter head rows an eligible draft step scores (`kDraftHeadRows`).
pub const DRAFT_HEAD_ROWS: u32 = 98_304;

/// Per-session head rows holding the request's tokens beyond
/// [`DRAFT_HEAD_ROWS`], in first-use order (`kDraftTableRows`).
pub const DRAFT_TABLE_ROWS: u32 = 1024;

/// Committed tokens the head choice looks back over (`kDraftVocabWindow`).
pub const DRAFT_VOCAB_WINDOW: usize = 32;

/// The request's draft vocabulary and when to use it (`DraftVocab`, copied
/// in the extension from Gufo's `src/families/qwen/draft_vocab` at upstream
/// 1c75887b). A token is a miss when it is outside the prefix, not special
/// and not yet in the request's table (or the table is full). A draft cycle
/// uses the small head when its anchor is not a miss and at most one of the
/// last [`DRAFT_VOCAB_WINDOW`] committed tokens was, so text that keeps
/// bringing new tokens (a short reply in another script) drafts over the
/// full vocabulary as before, a lone new word does not leave the small head,
/// and a request that repeats its own vocabulary (code and comments in the
/// user's language) gets the small head. Everything is a function of the
/// request's own token history, so batched, isolated and restored sessions
/// draft the same way.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
        usize::try_from(id)
            .ok()
            .and_then(|at| self.known.get(at).copied())
            .unwrap_or(false)
    }
}

/// Whether `piece` is one Latin (below U+0250), general punctuation or space
/// (`▁`) symbol: no merge creates those, but text needs them as often as the
/// first merges (`CommonSymbol`).
fn common_symbol(piece: &[u8]) -> bool {
    let Some(&lead) = piece.first() else {
        return false;
    };
    let length = if lead < 0x80 {
        1
    } else if lead < 0xE0 {
        2
    } else if lead < 0xF0 {
        3
    } else {
        4
    };
    if piece.len() != length {
        return false;
    }
    let mut code = u32::from(match length {
        1 => lead,
        2 => lead & 0x1F,
        3 => lead & 0x0F,
        _ => lead & 0x07,
    });
    for &b in &piece[1..] {
        code = (code << 6) | (u32::from(b) & 0x3F);
    }
    code < 0x250 || (0x2000..0x2070).contains(&code) || code == 0x2581
}

/// The token of each drafter head row in device order: control and
/// user-defined tokens, then tokens by the rank of the earliest merge that
/// creates them, with single Latin, punctuation and space symbols ranked with
/// the first merges; every other token follows by id (`DraftHeadOrder`).
pub fn draft_head_order(tokenizer: &Tokenizer) -> Vec<TokenId> {
    let ranks = tokenizer.creation_ranks();
    let keys: Vec<i64> = ranks
        .iter()
        .enumerate()
        .map(|(i, &rank)| {
            let token = i as TokenId;
            let ty = tokenizer.token_type(token);
            if matches!(ty, TokenType::Control | TokenType::UserDefined) {
                -1
            } else if rank != u32::MAX {
                i64::from(rank)
            } else if common_symbol(tokenizer.piece(token)) {
                0
            } else {
                (1i64 << 40) + i64::from(token)
            }
        })
        .collect();
    let mut order: Vec<TokenId> = (0..ranks.len() as TokenId).collect();
    order.sort_by_key(|&token| keys[token as usize]);
    order
}
