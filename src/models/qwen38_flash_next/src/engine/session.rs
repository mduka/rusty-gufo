//! Sessions: creating them, feeding prompts, syncing the MTP draft and
//! evaluating.

use super::*;

impl Session {
    pub fn logits(&self) -> &[f32] {
        if self.valid { &self.logits } else { &[] }
    }

    pub fn tokens(&self) -> &[i32] {
        if self.valid { &self.tokens } else { &[] }
    }

    pub fn position(&self) -> u32 {
        self.native.position()
    }

    pub fn context_size(&self) -> u32 {
        self.native.max_context()
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Names the storage the session's borrowed snapshots point into, once
    /// one exists.
    pub fn snapshot_owner(&self) -> Option<u64> {
        self.native.snapshots.get().map(|registry| registry.id())
    }

    pub fn allocated_bytes(&self) -> usize {
        self.native.allocated_bytes()
    }

    pub fn statistics(&self) -> SpeculativeStats {
        self.stats
    }

    /// A new request reusing cached context starts its own acceptance
    /// history.
    pub fn reset_draft_policy(&mut self) {
        self.draft_length.reset();
        // A fully reused prompt can start decoding without a sync. It still
        // starts a new request: no copy cursor, and no wider first step.
        self.lookup.begin_request();
    }

    /// Cancellation is checked at forward, layer and snapshot-copy
    /// boundaries.
    pub fn set_cancellation_check(&mut self, check: Option<Arc<dyn Fn() -> bool + Send + Sync>>) {
        self.native.set_cancellation_check(check);
    }

    pub(super) fn mtp_enabled(&self) -> bool {
        self.native.mtp_enabled()
    }

    /// The image identity of the first `token_count` tokens (`ImageIdentity`).
    pub(super) fn image_identity(&self, token_count: usize) -> &[u8] {
        self.image_prompt
            .as_ref()
            .map_or(&[], |prompt| prompt.identity_for_prefix(token_count))
    }

    /// Trunk rows the draft block may still read: `hidden_base..len`.
    pub(super) fn kept_hidden_rows(&self) -> u32 {
        if self.mtp_enabled() {
            self.tokens.len() as u32 - self.hidden_base
        } else {
            0
        }
    }
}

impl Model {
    pub fn create_session(&self, mode: SessionMode, max_context: u32) -> Result<Session, String> {
        if max_context == 0 || max_context > self.options.max_context {
            return Err("session context is outside the model limits".into());
        }
        let native = self.executor.create_session(mode, max_context)?;
        Ok(Session {
            native,
            tokens: Vec::new(),
            logits: vec![0.0; self.vocab_size() as usize],
            draft_token: 0,
            verify_logits: Vec::new(),
            hidden_base: 0,
            draft_length: LengthController::new(
                self.options.max_draft_tokens,
                self.options.decode_concurrency,
            ),
            stats: SpeculativeStats::default(),
            image_prompt: None,
            anchor_hint: None,
            valid: true,
            draft_vocab: DraftVocab::new(
                DRAFT_EXTRA_ROWS - self.draft_specials.as_ref().map_or(0, |s| s.len() as u32),
            ),
            draft_vocab_stale: false,
            lookup: PromptLookup::default(),
        })
    }

    /// Prefix and special tokens: always in the small head
    /// (`InDraftSubset`).
    pub(super) fn in_draft_subset(&self, token: i32) -> bool {
        token >= 0
            && ((token as u32) < DRAFT_VOCAB
                || self
                    .draft_specials
                    .as_ref()
                    .is_some_and(|s| s.binary_search(&token).is_ok()))
    }

    /// Brings the session's draft table up to its committed tokens
    /// (`SyncDraftVocab`).
    pub(super) fn sync_draft_vocab(&mut self, s: &mut Session) -> Result<(), String> {
        let Some(specials) = &self.draft_specials else {
            return Ok(());
        };
        if !s.mtp_enabled() {
            return Ok(());
        }
        let first = specials.len() as u32;
        let stale = s.draft_vocab_stale || s.draft_vocab.scanned() > s.tokens.len();
        if stale {
            s.draft_vocab.reset();
            s.draft_vocab_stale = false;
        }
        let tabled = s.draft_vocab.rare().len() as u32;
        let mut added = Vec::new();
        s.draft_vocab
            .catch_up(&s.tokens, |t| self.in_draft_subset(t), Some(&mut added));
        if added.is_empty() && !stale {
            return Ok(());
        }
        self.executor
            .write_draft_rows(&mut s.native, first + tabled, &added, stale)
    }

    /// Drops every token; the next sync starts at position 0.
    pub fn reset(&mut self, s: &mut Session) {
        self.reset_session(s, false);
    }

    /// [`reset`](Self::reset) of a healthy session the continuation cache
    /// reassigns: an MTP session keeps its verification scratch and graphs
    /// (`ResetImpl(true)`).
    pub fn reset_reusing_mtp(&mut self, s: &mut Session) {
        self.reset_session(s, true);
    }

    fn reset_session(&mut self, s: &mut Session, reuse_mtp: bool) {
        s.valid = false;
        s.anchor_hint = None;
        let reset = s.native.reset(self.executor.stream(), reuse_mtp);
        s.tokens.clear();
        s.draft_vocab_stale = true;
        s.lookup.clear();
        s.hidden_base = 0;
        s.draft_length.reset();
        self.executor.mtp_rewind(&mut s.native, 0);
        s.valid = reset.is_ok();
    }

    /// The draft block trails the trunk: MTP position i consumes token i+1
    /// and the trunk's hidden row of position i, so positions up to the
    /// current one replay once their successor is known (`DraftReplay`).
    pub(super) fn draft_replay(
        &self,
        s: &Session,
        next_token: i32,
    ) -> Result<Option<(Vec<i32>, i32)>, String> {
        let size = s.tokens.len() as u32;
        let mp = self.executor.mtp_position(&s.native);
        if mp >= size {
            return Ok(None);
        }
        if mp < s.hidden_base {
            return Err("draft block fell behind the kept hidden rows".into());
        }
        let mut replay = s.tokens[mp as usize + 1..].to_vec();
        replay.push(next_token);
        Ok(Some((replay, (mp - s.hidden_base) as i32)))
    }

    /// `DraftCatchUp`: replays the draft block up to `next_token`, proposing
    /// the next draft (a greedy token into `draft_token`, or candidates),
    /// through the small head with `draft_subset`.
    pub(super) fn draft_catch_up(
        &mut self,
        s: &mut Session,
        next_token: i32,
        propose: bool,
        candidates: Option<&mut MtpCandidates>,
        draft_subset: bool,
    ) -> Result<(), String> {
        let Some((replay, hidden_row)) = self.draft_replay(s, next_token)? else {
            return Ok(());
        };
        let mut token = s.draft_token;
        let greedy = propose && candidates.is_none();
        // A headless step needs only the persistent K/V.
        let output = MtpOutput {
            token: greedy.then_some(&mut token),
            candidates,
            kv_only: !propose,
            draft_subset,
        };
        self.executor
            .mtp_forward(&mut s.native, &replay, hidden_row, output)?;
        s.draft_token = token;
        Ok(())
    }

    /// Appends `tokens` in passes of at most the prefill capacity, or one
    /// pass when the rest fits the through capacity. The pass crossing a
    /// nonzero `boundary` (a token count) also returns its checkpoint's
    /// payload and the milliseconds its copy took (`Feed`). Each prefill
    /// pass names the one after it, the last one the start of `after`, so
    /// its n-gram rows are read on the way.
    pub(super) fn feed(
        &mut self,
        s: &mut Session,
        tokens: &[i32],
        prefill: bool,
        boundary: usize,
        after: &[i32],
    ) -> Result<Option<(SessionSnapshot, f64)>, String> {
        let max_batch = self.executor.options().max_batch as usize;
        let max_speculative = self.executor.options().max_speculative as usize;
        let mut payload = None;
        let mut off = 0;
        while off < tokens.len() {
            let remaining = tokens.len() - off;
            let capacity = if remaining <= max_batch {
                max_batch
            } else {
                self.prefill_capacity() as usize
            };
            let mut n = capacity.min(remaining);
            // Past the fused WMMA kernel's extent (only with YaRN) attention
            // falls back to the per-token kernel, which an in-pass
            // checkpoint cannot use: the pass ends at the boundary instead,
            // keeping split-pass arithmetic on both sides.
            if boundary > s.tokens.len()
                && boundary < s.tokens.len() + n
                && s.tokens.len() + n > wmma_context_limit(self.config()) as usize
            {
                n = boundary - s.tokens.len();
            }
            let chunk = &tokens[off..off + n];
            if s.mtp_enabled() && !s.tokens.is_empty() {
                self.draft_catch_up(s, chunk[0], false, None, false)?;
            }
            let mode = if prefill {
                ForwardMode::Prefill
            } else {
                ForwardMode::Decode
            };
            let mut checkpoint = if boundary > s.tokens.len() && boundary <= s.tokens.len() + n {
                Some(PrefillCheckpoint::new(
                    self.config(),
                    &s.native,
                    chunk,
                    (boundary - s.tokens.len()) as u32,
                    max_speculative as u32,
                )?)
            } else {
                None
            };
            // The chunk this loop, or the following sync, takes next.
            let rest = if off + n < tokens.len() {
                &tokens[off + n..]
            } else {
                after
            };
            let next_len = if rest.len() <= max_batch {
                max_batch
            } else {
                self.prefill_capacity() as usize
            };
            let next = if prefill {
                &rest[..next_len.min(rest.len())]
            } else {
                &[]
            };
            self.executor.forward_checkpoint(
                &mut s.native,
                chunk,
                1,
                Some(&mut s.logits),
                mode,
                checkpoint.as_mut(),
                next,
            )?;
            // Past a checkpoint the session keeps only the rows after it.
            let kept = match &checkpoint {
                Some(c) if (c.tokens as usize) < n => n - c.tokens as usize,
                _ => n,
            };
            s.hidden_base = (s.tokens.len() + n - kept.min(max_speculative)) as u32;
            s.tokens.extend_from_slice(chunk);
            if let Some(checkpoint) = &checkpoint {
                let storage = self
                    .executor
                    .core
                    .checkpoint
                    .as_ref()
                    .ok_or("checkpoint storage is missing")?;
                let started = std::time::Instant::now();
                let captured = self.snapshots.capture_checkpoint(s, checkpoint, storage)?;
                payload = Some((captured, started.elapsed().as_secs_f64() * 1e3));
            }
            off += n;
        }
        Ok(payload)
    }

    /// Makes the session's state equal to `prompt`, keeping the longest
    /// common prefix; recurrent state cannot rewind, so a divergence
    /// restarts the session (`Sync`).
    pub fn sync(&mut self, s: &mut Session, prompt: &[i32]) -> Result<(), String> {
        self.sync_before(s, prompt, &[])
    }

    /// [`sync`](Self::sync) that may name `next`, the tokens the following
    /// sync appends, so their n-gram rows are read during this one; it never
    /// changes the result.
    pub fn sync_before(
        &mut self,
        s: &mut Session,
        prompt: &[i32],
        next: &[i32],
    ) -> Result<(), String> {
        if prompt.is_empty() {
            return Err("prompt is empty".into());
        }
        if prompt.len() > s.context_size() as usize {
            return Err("prompt exceeds the session context".into());
        }
        s.lookup.begin_request();
        if !s.valid {
            self.reset(s);
        }
        let common = s
            .tokens
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        if common == prompt.len() && common == s.tokens.len() {
            return Ok(());
        }
        let common = if common != s.tokens.len() {
            self.reset(s);
            0
        } else {
            common
        };
        s.valid = false;
        self.feed(s, &prompt[common..], true, 0, next)?;
        s.valid = true;
        Ok(())
    }

    /// [`sync`](Self::sync) through a stable text boundary in the same
    /// pass (`SyncThrough`): returns the boundary's checkpoint, which
    /// restores like a snapshot of a session synced to `prompt[..boundary]`,
    /// and the milliseconds its copy took. The boundary must extend the
    /// frontier and lie at most [`state::VEC_BATCH`] tokens before the end.
    pub fn sync_through(
        &mut self,
        s: &mut Session,
        prompt: &[i32],
        boundary: usize,
    ) -> Result<(SessionSnapshot, f64), String> {
        if boundary == 0
            || boundary > prompt.len()
            || prompt.len() - boundary > state::VEC_BATCH as usize
            || boundary <= s.tokens.len()
        {
            return Err("checkpoint must extend the frontier in the text tail".into());
        }
        if prompt.len() > s.context_size() as usize {
            return Err("prompt exceeds the session context".into());
        }
        s.lookup.begin_request();
        if !s.valid {
            self.reset(s);
        }
        let common = s
            .tokens
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        let common = if common != s.tokens.len() {
            self.reset(s);
            0
        } else {
            common
        };
        s.valid = false;
        let payload = self.feed(s, &prompt[common..], true, boundary, &[])?;
        s.valid = true;
        payload.ok_or_else(|| "the prefill did not cross its checkpoint".into())
    }

    /// [`Snapshotter::checkpoint_bytes`].
    pub fn prefill_checkpoint_bytes(&self, s: &Session, position: u32) -> u64 {
        self.snapshots.checkpoint_bytes(s, position)
    }

    /// Appends one token (`Evaluate`).
    pub fn evaluate(&mut self, s: &mut Session, token: i32) -> Result<(), String> {
        if !s.valid {
            return Err("session needs a successful Sync after a failed operation".into());
        }
        if s.tokens.len() >= s.context_size() as usize {
            return Err("session context is full".into());
        }
        s.valid = false;
        self.feed(s, &[token], false, 0, &[])?;
        s.valid = true;
        Ok(())
    }
}
