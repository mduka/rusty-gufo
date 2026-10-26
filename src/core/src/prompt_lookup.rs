//! Prompt lookup (upstream `src/core/speculative/prompt_lookup.hpp`):
//! draft proposals copied from the session's own committed tokens. Qwen3.8
//! Flash-Next proposes them beside its MTP head, the dense Qwen DFlash2
//! draft instead of a neural block.
//!
//! The drafter's context (committed tokens plus the cycle's chain so far)
//! ends in a key of [`KEY_TOKENS`]; among the most recent earlier
//! occurrences of that key, the one whose preceding tokens match the context
//! furthest back wins, and the tokens that followed it are the proposals.
//! Greedy requests require [`MIN_MATCH`] tokens; sampled requests require the
//! longer [`SAMPLED_MIN_MATCH`]: common code syntax and field names alone
//! poorly predict their continuation at temperature and can displace useful
//! neural drafts. Sampled requests index only their starting context: earlier
//! parts of a freshly generated answer are especially poor sources for new
//! prose or code. Greedy requests can also copy repetition within their
//! current answer.
//!
//! Where the last verified copy left off, [`RESUME_MIN_MATCH`] tokens are
//! enough: output that edits a line of the text it is copying rejoins it a
//! few tokens later. A copy that keeps fewer than half of its tokens cannot
//! enable this shorter gate. A new request starts with nothing to resume.
//!
//! A request's first speculative step copies nothing. Its position also
//! bounds later query matches: a shared chat wrapper or tool header must not
//! count as a long matching answer before that answer has been generated.
//!
//! Proposals depend only on the prefix, so rejection sampling (with the
//! proposal as a point mass) keeps the target distribution exact.

pub const KEY_TOKENS: usize = 3;
pub const MIN_MATCH: usize = 12;
/// Sampled code and field extraction often share short syntax without
/// sharing the continuation: a longer match comes first.
pub const SAMPLED_MIN_MATCH: usize = 24;
pub const MAX_MATCH: usize = 64;
pub const MAX_CANDIDATES: usize = 16;
pub const RESUME_MIN_MATCH: usize = 5;
/// A resumed copy continues at most this far past the last copy's end, or
/// [`RESUME_BACK`] tokens before it (the output repeated a few tokens).
pub const RESUME_WINDOW: usize = 512;
pub const RESUME_BACK: usize = 8;

/// A token ID the index keys on: the low 21 bits of its two's-complement
/// value, as the C++ `PromptLookup<TokenId>` packs `std::int32_t` and
/// `std::uint32_t` IDs alike.
pub trait LookupToken: Copy + PartialEq + Default {
    fn bits(self) -> u32;
}

impl LookupToken for i32 {
    fn bits(self) -> u32 {
        self as u32
    }
}

impl LookupToken for u32 {
    fn bits(self) -> u32 {
        self
    }
}

/// A match: committed index of the first proposal and the matched context
/// tokens (at least [`KEY_TOKENS`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Match {
    pub start: usize,
    pub length: usize,
}

/// Copy state the index cannot rebuild from the tokens (`State`). Session
/// snapshots carry it, so a restored session copies as the original would
/// have and seeded sampled output replays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookupState {
    pub resume: u32,
    pub request_start: u32,
}

/// No resume point or request boundary (`kNoState`).
pub const NO_STATE: u32 = u32::MAX;

impl Default for LookupState {
    fn default() -> Self {
        Self {
            resume: NO_STATE,
            request_start: NO_STATE,
        }
    }
}

impl LookupState {
    pub const BYTES: usize = 8;

    pub fn write(&self, out: &mut [u8]) {
        out[..4].copy_from_slice(&self.resume.to_le_bytes());
        out[4..8].copy_from_slice(&self.request_start.to_le_bytes());
    }

    pub fn read(bytes: &[u8]) -> Self {
        let word = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        Self {
            resume: word(0),
            request_start: word(4),
        }
    }

    /// Whether the state can belong to a history of `tokens` committed
    /// tokens (`Valid`).
    pub fn valid(&self, tokens: usize) -> bool {
        (self.request_start == NO_STATE || self.request_start as usize <= tokens)
            && (self.resume == NO_STATE || self.resume as usize <= tokens)
    }
}

const NONE: u32 = u32::MAX;

/// The index of one session (`PromptLookup<TokenId>`).
#[derive(Debug, Default)]
pub struct PromptLookup<T = i32> {
    head: Heads,
    /// `prev[end]` chains the occurrences of the key ending just before
    /// `end`.
    prev: Vec<u32>,
    /// `committed[prev.len() - 1]` when indexed.
    last: T,
    /// Where the last copy's source continues.
    resume: Option<usize>,
    request_start: Option<usize>,
}

impl<T: LookupToken> PromptLookup<T> {
    pub fn clear(&mut self) {
        self.clear_index();
        self.request_start = None;
    }

    /// A request starts: nothing to resume yet (`BeginRequest`).
    pub fn begin_request(&mut self) {
        self.resume = None;
        self.request_start = None;
    }

    /// Called once per decode step; false on a request's first step.
    pub fn step(&mut self, position: usize) -> bool {
        if self.request_start.is_some() {
            return true;
        }
        self.request_start = Some(position);
        false
    }

    pub fn save(&self) -> LookupState {
        let state = |value: Option<usize>| value.map_or(NO_STATE, |v| v as u32);
        LookupState {
            resume: state(self.resume),
            request_start: state(self.request_start),
        }
    }

    /// Clears the index and adopts `state` (checked with
    /// [`LookupState::valid`]).
    pub fn restore(&mut self, state: LookupState) {
        self.clear();
        let value = |v: u32| (v != NO_STATE).then_some(v as usize);
        self.resume = value(state.resume);
        self.request_start = value(state.request_start);
    }

    /// Records a verified copy of `proposed` tokens from committed index
    /// `start` of which `accepted` were kept; the copied text continues at
    /// `start + accepted`. Only a mostly accepted copy enables short
    /// matches.
    pub fn observe(&mut self, start: usize, proposed: usize, accepted: usize) {
        self.resume = (2 * accepted >= proposed).then_some(start + accepted);
    }

    /// Indexes committed tokens appended since the last call. `committed`
    /// must extend what was indexed before ([`Self::clear`] after anything
    /// else); a shorter history, or one whose last indexed token changed,
    /// restarts the index, so a missed clear never proposes stale copies.
    pub fn extend(&mut self, committed: &[T], prompt_only: bool) {
        let committed = match self.request_start {
            Some(start) if prompt_only => &committed[..committed.len().min(start)],
            _ => committed,
        };
        if committed.len() < self.prev.len()
            || (!self.prev.is_empty() && committed[self.prev.len() - 1] != self.last)
        {
            // begin_request/step may already have set the new request's
            // boundary before its shorter restored history is indexed.
            self.clear_index();
        }
        for end in self.prev.len()..committed.len() {
            let previous = if end >= KEY_TOKENS {
                self.head
                    .exchange(key(&committed[end - KEY_TOKENS..end]), end as u32)
            } else {
                NONE
            };
            self.prev.push(previous);
        }
        if let Some(&last) = committed.get(self.prev.len().wrapping_sub(1)) {
            self.last = last;
        }
        // The newest key (ending at committed.len()) has no continuation yet;
        // it is indexed once the next token commits.
    }

    /// Copies must end inside the indexed source, even when the query's
    /// committed history has grown beyond a sampled request's source.
    pub fn indexed_tokens(&self) -> usize {
        self.prev.len()
    }

    /// The best match for the context `committed` + `tail` among the indexed
    /// committed tokens, or a zero-length match. `tail` holds the
    /// uncommitted chain (anchor and drafts). With `require_resume`, only a
    /// full match at the last copy's resume point counts: skipping a sampled
    /// neural proposal needs an established copy.
    pub fn find(&self, committed: &[T], tail: &[T], minimum: usize, require_resume: bool) -> Match {
        self.find_accepted(committed, tail, minimum, |_| true, require_resume)
    }

    /// [`Self::find`] among the sources `accept_source` admits, by the
    /// committed index of their first proposal. A backend may reject a
    /// source whose continuation crosses message framing: the search keeps
    /// looking at earlier matches instead of proposing the same rejected
    /// boundary again. Flash-Next admits every source.
    pub fn find_accepted(
        &self,
        committed: &[T],
        tail: &[T],
        minimum: usize,
        accept_source: impl Fn(usize) -> bool,
        require_resume: bool,
    ) -> Match {
        let context = committed.len() + tail.len();
        let available = match self.request_start {
            Some(start) => context - context.min(start),
            None => context,
        };
        let max_match = MAX_MATCH.min(available);
        if max_match < KEY_TOKENS
            || (require_resume && (self.resume.is_none() || max_match < minimum))
        {
            return Match::default();
        }
        let at = |i: usize| {
            if i < committed.len() {
                committed[i]
            } else {
                tail[i - committed.len()]
            }
        };
        let mut key_tokens = [T::default(); KEY_TOKENS];
        for (i, token) in key_tokens.iter_mut().enumerate() {
            *token = at(context - KEY_TOKENS + i);
        }
        let mut best = Match::default();
        let mut end = self.head.find(key(&key_tokens));
        for _ in 0..MAX_CANDIDATES {
            if end == NONE {
                break;
            }
            let end_at = end as usize;
            // An occurrence ending at `end` proposes committed[end..]. The
            // key matched; extend backwards against the context.
            let mut length = KEY_TOKENS;
            while length < max_match
                && end_at > length
                && committed[end_at - length - 1] == at(context - length - 1)
            {
                length += 1;
            }
            let resumes = self.resume.is_some_and(|resume| {
                end_at + RESUME_BACK >= resume && end_at <= resume + RESUME_WINDOW
            });
            let allowed = if require_resume {
                resumes && length >= minimum
            } else {
                length >= minimum || (resumes && length >= RESUME_MIN_MATCH)
            };
            if end_at < self.prev.len() && length > best.length && allowed && accept_source(end_at)
            {
                best = Match {
                    start: end_at,
                    length,
                };
            }
            end = self.prev[end_at];
        }
        best
    }

    fn clear_index(&mut self) {
        self.head.clear();
        self.prev.clear();
        self.resume = None;
    }
}

/// Token ids fit in 21 bits (vocabularies up to 2M), so a key packs exactly:
/// equal keys are equal token triples.
fn key<T: LookupToken>(tokens: &[T]) -> u64 {
    tokens.iter().fold(0u64, |key, &t| {
        (key << 21) | (u64::from(t.bits()) & 0x1F_FFFF)
    })
}

/// Key -> end of its most recent occurrence: open addressing with linear
/// probing, at most half full. Clearing keeps the capacity, so a session
/// re-indexing after a restore allocates nothing.
#[derive(Debug)]
struct Heads {
    keys: Vec<u64>,
    values: Vec<u32>,
    size: usize,
    shift: u32,
}

/// Keys use 63 bits (three 21-bit ids), so all ones is never a key.
const EMPTY: u64 = u64::MAX;

impl Default for Heads {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            size: 0,
            shift: 64,
        }
    }
}

impl Heads {
    /// Sets `key` to `value` and returns its previous value, or `NONE`.
    fn exchange(&mut self, key: u64, value: u32) -> u32 {
        if 2 * (self.size + 1) > self.keys.len() {
            self.grow();
        }
        let mask = self.keys.len() - 1;
        let mut i = self.slot(key);
        while self.keys[i] != EMPTY && self.keys[i] != key {
            i = (i + 1) & mask;
        }
        if self.keys[i] == EMPTY {
            self.keys[i] = key;
            self.size += 1;
            self.values[i] = value;
            return NONE;
        }
        std::mem::replace(&mut self.values[i], value)
    }

    fn find(&self, key: u64) -> u32 {
        if self.size == 0 {
            return NONE;
        }
        let mask = self.keys.len() - 1;
        let mut i = self.slot(key);
        loop {
            if self.keys[i] == key {
                return self.values[i];
            }
            if self.keys[i] == EMPTY {
                return NONE;
            }
            i = (i + 1) & mask;
        }
    }

    fn clear(&mut self) {
        if self.size != 0 {
            self.keys.fill(EMPTY);
            self.size = 0;
        }
    }

    /// Fibonacci hashing: the multiply's top bits index the table.
    fn slot(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    fn grow(&mut self) {
        let capacity = if self.keys.is_empty() {
            1024
        } else {
            2 * self.keys.len()
        };
        let keys = std::mem::replace(&mut self.keys, vec![EMPTY; capacity]);
        let values = std::mem::replace(&mut self.values, vec![0; capacity]);
        self.shift = 64 - capacity.trailing_zeros();
        self.size = 0;
        let mask = capacity - 1;
        for (key, value) in keys.into_iter().zip(values) {
            if key != EMPTY {
                let mut j = self.slot(key);
                while self.keys[j] != EMPTY {
                    j = (j + 1) & mask;
                }
                self.keys[j] = key;
                self.values[j] = value;
                self.size += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(first: i32, count: usize) -> Vec<i32> {
        (0..count as i32).map(|i| first + i).collect()
    }

    /// A copied passage is proposed once the context repeats MIN_MATCH of
    /// its tokens, continuing where the earlier occurrence continued
    /// (upstream `CheckCopy`).
    #[test]
    fn copies_a_repeated_passage() {
        let mut committed = range(1000, 40);
        committed.extend(range(5000, 7));
        let quote = committed[5..15].to_vec();
        committed.extend(quote);
        let mut lookup = PromptLookup::default();
        lookup.extend(&committed, false);
        let found = lookup.find(&committed, &[1015, 1016], MIN_MATCH, false);
        assert_eq!((found.length, found.start), (12, 17));
        assert_eq!(committed[found.start], 1017);
        assert_eq!(
            lookup.find(&committed, &[1015], MIN_MATCH, false).length,
            0,
            "11 tokens"
        );
        assert_eq!(
            lookup
                .find(&committed, &[1015, 1016], SAMPLED_MIN_MATCH, false)
                .length,
            0,
            "a short match passed the sampled gate"
        );
        let sampled = lookup.find(&committed, &range(1015, 14), SAMPLED_MIN_MATCH, false);
        assert_eq!((sampled.length, sampled.start), (24, 29));
    }

    /// Unsigned IDs, as the dense Qwen draft keeps them, key exactly like
    /// the same signed IDs.
    #[test]
    fn unsigned_ids_match_like_signed_ones() {
        let signed = {
            let mut committed = range(1000, 40);
            committed.extend(range(5000, 7));
            let quote = committed[5..15].to_vec();
            committed.extend(quote);
            committed
        };
        let unsigned: Vec<u32> = signed.iter().map(|&t| t as u32).collect();
        let mut lookup = PromptLookup::<u32>::default();
        lookup.extend(&unsigned, false);
        let found = lookup.find(&unsigned, &[1015, 1016], MIN_MATCH, false);
        assert_eq!((found.length, found.start), (12, 17));
    }

    /// The occurrence matching furthest back wins, not the most recent one
    /// (`CheckLongestWins`).
    #[test]
    fn the_longest_match_wins() {
        let a = range(100, 20);
        let mut committed = a.clone();
        committed.push(7);
        committed.extend([900, 901, 117, 118, 119, 8]);
        committed.extend(&a[..17]);
        let mut lookup = PromptLookup::default();
        lookup.extend(&committed, false);
        let found = lookup.find(&committed, &[117, 118, 119], MIN_MATCH, false);
        assert!(found.length >= 12 && committed[found.start] == 7);
    }

    /// The nearest repeated passage ends at a message boundary: rejecting
    /// that source finds an earlier complete continuation, while the default
    /// search Flash-Next uses still takes the nearest one, and a filtered
    /// search still needs an established copy when one is required
    /// (`CheckFilteredContinuation`).
    #[test]
    fn a_rejected_source_selects_an_earlier_continuation() {
        let passage = range(1000, 40);
        let mut committed = passage.clone();
        committed.extend(&passage[..24]);
        committed.extend([9000, 9001, 9002]);
        committed.extend(&passage[..23]);
        let mut lookup = PromptLookup::default();
        lookup.extend(&committed, false);
        let tail = [1023];
        let nearest = lookup.find(&committed, &tail, 24, false);
        assert!(
            nearest.start == 64 && committed[nearest.start] == 9000,
            "unfiltered lookup no longer chooses the nearest equal match"
        );
        let valid_source = |start: usize| {
            let count = 7.min(committed.len() - start);
            !committed[start..start + count].contains(&9000)
        };
        let filtered = lookup.find_accepted(&committed, &tail, 24, valid_source, false);
        assert!(
            filtered.start == 24 && filtered.length == 24 && committed[filtered.start] == 1024,
            "rejected boundary did not select an earlier valid continuation"
        );
        assert_eq!(
            lookup
                .find_accepted(&committed, &tail, 24, |_| false, false)
                .length,
            0,
            "lookup ignored a source rejection"
        );
        assert_eq!(
            lookup
                .find_accepted(&committed, &tail, 24, valid_source, true)
                .length,
            0,
            "filtering bypassed the required accepted copy"
        );
        lookup.observe(23, 1, 1);
        let resumed = lookup.find_accepted(&committed, &tail, 24, valid_source, true);
        assert!(
            resumed.start == 24 && resumed.length == 24,
            "a valid filtered continuation did not resume"
        );
        lookup.observe(63, 1, 1);
        assert_eq!(
            lookup
                .find_accepted(&committed, &tail, 24, valid_source, true)
                .length,
            0,
            "filtering resumed a different source outside the resume window"
        );
    }

    /// Incremental indexing equals a full build; a shorter or rewritten
    /// history restarts the index (`CheckIncremental`, `CheckRewrite`).
    #[test]
    fn incremental_and_rewritten_histories() {
        let mut committed = range(1, 30);
        committed.extend(range(1, 20));
        let mut whole = PromptLookup::default();
        whole.extend(&committed, false);
        let mut steps = PromptLookup::default();
        for n in (1..=committed.len()).step_by(3) {
            steps.extend(&committed[..n], false);
        }
        steps.extend(&committed, false);
        let (a, b) = (
            whole.find(&committed, &[21], MIN_MATCH, false),
            steps.find(&committed, &[21], MIN_MATCH, false),
        );
        assert!(a == b && a.length >= 12);
        let fresh = range(500, 10);
        steps.extend(&fresh, false);
        assert_eq!(steps.find(&fresh, &[21], MIN_MATCH, false).length, 0);

        let rewritten = range(700, committed.len());
        whole.extend(&rewritten, false);
        assert_eq!(
            whole.find(&rewritten, &[1, 2, 3], MIN_MATCH, false).length,
            0
        );
        let mut longer = range(300, 25);
        longer.extend(range(300, 15));
        whole.extend(&longer, false);
        let found = whole.find(&longer, &[315], MIN_MATCH, false);
        assert!(found.length >= 12 && longer[found.start] == 316);
    }

    /// A mostly accepted copy lets five tokens resume it, only near where it
    /// left off; snapshots carry that state; a new request forgets it
    /// (`CheckResume`).
    #[test]
    fn resumes_a_verified_copy() {
        let mut committed = range(1000, 40);
        committed.extend(range(5000, 3));
        committed.extend(&range(1000, 40)[20..25]);
        let mut lookup = PromptLookup::default();
        lookup.extend(&committed, false);
        let tail = [1025];
        assert_eq!(
            lookup.find(&committed, &tail, MIN_MATCH, false).length,
            0,
            "no resume yet"
        );
        lookup.observe(20, 4, 1);
        assert_eq!(
            lookup.find(&committed, &tail, MIN_MATCH, false).length,
            0,
            "a rejected copy"
        );
        lookup.observe(20, 4, 4);
        let found = lookup.find(&committed, &tail, MIN_MATCH, false);
        assert_eq!((found.length, found.start), (6, 26));
        let state = lookup.save();
        assert!(state.valid(committed.len()) && !state.valid(19));
        let mut bytes = [0u8; LookupState::BYTES];
        state.write(&mut bytes);
        let mut restored = PromptLookup::default();
        restored.restore(LookupState::read(&bytes));
        restored.extend(&committed, false);
        assert_eq!(restored.find(&committed, &tail, MIN_MATCH, false), found);
        restored.begin_request();
        assert_eq!(restored.find(&committed, &tail, MIN_MATCH, false).length, 0);
        // A sampled copy skips the neural head only from a resume point and
        // with a full sampled match.
        assert_eq!(
            lookup
                .find(&committed, &tail, SAMPLED_MIN_MATCH, true)
                .length,
            0
        );
    }

    /// A request's first step copies nothing and bounds later matches to
    /// its own output (`Step`).
    #[test]
    fn the_first_step_bounds_matches() {
        let mut committed = range(1, 30);
        committed.extend(range(1, 20));
        let mut lookup = PromptLookup::default();
        assert!(!lookup.step(committed.len()));
        assert!(lookup.step(committed.len()));
        lookup.extend(&committed, false);
        assert_eq!(lookup.find(&committed, &[21], MIN_MATCH, false).length, 0);
    }
}
