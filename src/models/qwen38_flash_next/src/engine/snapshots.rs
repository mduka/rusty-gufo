//! Session snapshots: the snapshotter that captures and writes them, their
//! parts, and restoring a session.

use super::*;

impl Snapshotter {
    /// Bytes a snapshot of the session's current context occupies.
    pub fn bytes(&self, s: &Session) -> u64 {
        if !s.valid {
            return 0;
        }
        session_host_bytes(
            s.tokens.len() as u32,
            self.vocab,
            s.image_identity(s.tokens.len()).len(),
        ) + self.saver.bytes(&s.native, s.kept_hidden_rows())
    }

    /// The whole context (tokens, device caches and recurrent state, the
    /// draft block's state, the last logits and the length policy) in host
    /// memory; the same bytes persist to disk (`SaveSnapshot`).
    pub fn save(&self, s: &Session) -> Result<Vec<u8>, String> {
        if !s.valid
            || s.tokens.is_empty()
            || s.tokens.len() != s.native.position() as usize
            || s.tokens.len() > u32::MAX as usize
        {
            return Err("snapshot needs a synced, non-empty context".into());
        }
        // A cancelled request copies nothing (`CheckCancellation`).
        if s.native.cancellation_requested() {
            return Err("generation cancelled".into());
        }
        let identity = s.image_identity(s.tokens.len());
        let hidden_rows = s.kept_hidden_rows();
        let executor_bytes = self.saver.bytes(&s.native, hidden_rows);
        self.write(
            s,
            &s.tokens,
            &s.logits,
            identity,
            hidden_rows,
            executor_bytes,
            |out| self.saver.save(&s.native, hidden_rows, out),
        )
    }

    /// Payload bytes of the checkpoint a pass from the session's frontier
    /// would capture at `position` tokens; 0 when it cannot (an image
    /// prompt; `PrefillCheckpointBytes`).
    pub fn checkpoint_bytes(&self, s: &Session, position: u32) -> u64 {
        if !s.native.vision.layout().images.is_empty() {
            return 0;
        }
        session_host_bytes(position, self.vocab, 0)
            + self.saver.checkpoint_bytes(&s.native, position)
    }

    /// [`save`](Self::save) that borrows the session's append-only rows
    /// instead of copying them (`SaveSnapshot`, `kBorrowed`); without device
    /// headroom it saves host bytes.
    pub fn capture(&self, s: &Session) -> Result<SessionSnapshot, String> {
        self.capture_parts(s, true)
    }

    /// [`save`](Self::save) as a [`SessionSnapshot`] in host bytes.
    pub fn capture_host(&self, s: &Session) -> Result<SessionSnapshot, String> {
        self.capture_parts(s, false)
    }

    /// The session part and the executor state in buffers of their own
    /// size, the state borrowed when `borrow` allows it and the device has
    /// headroom. Splitting one [`save`](Self::save) buffer instead kept its
    /// whole capacity behind the head, twice the payload in host memory.
    pub(super) fn capture_parts(
        &self,
        s: &Session,
        borrow: bool,
    ) -> Result<SessionSnapshot, String> {
        if !s.valid
            || s.tokens.is_empty()
            || s.tokens.len() != s.native.position() as usize
            || s.tokens.len() > u32::MAX as usize
        {
            return Err("snapshot needs a synced, non-empty context".into());
        }
        if s.native.cancellation_requested() {
            return Err("generation cancelled".into());
        }
        let identity = s.image_identity(s.tokens.len());
        let hidden_rows = s.kept_hidden_rows();
        let executor_bytes = self.saver.bytes(&s.native, hidden_rows);
        let head = self.head(
            s,
            &s.tokens,
            &s.logits,
            identity,
            hidden_rows,
            executor_bytes,
        );
        let borrowed = if borrow {
            self.saver.borrow(&s.native, hidden_rows)?
        } else {
            None
        };
        let executor = match borrowed {
            Some(handle) => Part::Borrowed(handle),
            None => {
                let mut bytes = gufo_hip::huge_page_buffer(executor_bytes as usize);
                self.saver.save(&s.native, hidden_rows, &mut bytes)?;
                Part::Host(bytes)
            }
        };
        Ok(SessionSnapshot { head, executor })
    }

    /// The checkpoint a pass from `s` captured at `checkpoint` in executor
    /// `storage`, over the first `boundary` tokens, as [`capture`](Self::capture)
    /// takes a pass ending there; frozen before the next forward reuses
    /// `storage`.
    pub(crate) fn capture_checkpoint(
        &self,
        s: &Session,
        checkpoint: &PrefillCheckpoint,
        storage: &CheckpointStorage,
    ) -> Result<SessionSnapshot, String> {
        let boundary = checkpoint.frontier.position as usize;
        let hidden_rows = checkpoint.frontier.hidden_rows;
        let executor_bytes = self.saver.frontier_bytes(&s.native, &checkpoint.frontier);
        let head = self.head(
            s,
            &s.tokens[..boundary],
            &checkpoint.logits,
            &[],
            hidden_rows,
            executor_bytes,
        );
        let executor = match self
            .saver
            .borrow_checkpoint(&s.native, checkpoint, storage)?
        {
            Some(handle) => Part::Borrowed(handle),
            None => {
                let mut bytes = gufo_hip::huge_page_buffer(executor_bytes as usize);
                self.saver
                    .save_checkpoint(&s.native, checkpoint, storage, &mut bytes)?;
                Part::Host(bytes)
            }
        };
        Ok(SessionSnapshot { head, executor })
    }

    /// The session part of a payload: header, image identity, `tokens` and
    /// `logits`.
    pub(super) fn head(
        &self,
        s: &Session,
        tokens: &[i32],
        logits: &[f32],
        identity: &[u8],
        hidden_rows: u32,
        executor_bytes: u64,
    ) -> Vec<u8> {
        let host_bytes = session_host_bytes(tokens.len() as u32, self.vocab, identity.len());
        let mut out = vec![0u8; host_bytes as usize];
        self.write_head(
            &mut out,
            s,
            tokens,
            logits,
            identity,
            hidden_rows,
            executor_bytes,
        );
        out
    }

    /// The session header, `tokens` and `logits`, then the executor's
    /// `executor_bytes` from `executor`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write(
        &self,
        s: &Session,
        tokens: &[i32],
        logits: &[f32],
        identity: &[u8],
        hidden_rows: u32,
        executor_bytes: u64,
        executor: impl FnOnce(&mut [u8]) -> Result<(), String>,
    ) -> Result<Vec<u8>, String> {
        let token_count = tokens.len() as u32;
        let host_bytes = session_host_bytes(token_count, self.vocab, identity.len());
        // Huge pages, as in the C++ `SessionSnapshot`: a 4 KiB fault per page
        // of a fresh buffer made captures several times slower while other
        // sessions decode.
        let mut out = gufo_hip::huge_page_buffer((host_bytes + executor_bytes) as usize);
        let at = host_bytes as usize;
        self.write_head(
            &mut out[..at],
            s,
            tokens,
            logits,
            identity,
            hidden_rows,
            executor_bytes,
        );
        executor(&mut out[at..])?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_head(
        &self,
        out: &mut [u8],
        s: &Session,
        tokens: &[i32],
        logits: &[f32],
        identity: &[u8],
        hidden_rows: u32,
        executor_bytes: u64,
    ) {
        let token_count = tokens.len() as u32;
        let h = &mut out[..SESSION_HEADER_BYTES];
        h[..8].copy_from_slice(&SESSION_MAGIC);
        h[8..12].copy_from_slice(&SNAPSHOT_PAYLOAD_VERSION.to_le_bytes());
        h[12..16].copy_from_slice(&self.vocab.to_le_bytes());
        h[16..20].copy_from_slice(&token_count.to_le_bytes());
        h[20..24].copy_from_slice(&hidden_rows.to_le_bytes());
        h[24..32].copy_from_slice(&executor_bytes.to_le_bytes());
        s.draft_length
            .state()
            .write(&mut h[32..32 + LengthState::BYTES]);
        h[108..112].copy_from_slice(&(identity.len() as u32).to_le_bytes());
        h[112..116].copy_from_slice(&self.concurrency.to_le_bytes());
        s.lookup.save().write(&mut h[120..120 + LookupState::BYTES]);
        let mut at = SESSION_HEADER_BYTES;
        out[at..at + identity.len()].copy_from_slice(identity);
        at += identity.len();
        for &t in tokens {
            out[at..at + 4].copy_from_slice(&t.to_le_bytes());
            at += 4;
        }
        for &v in logits {
            out[at..at + 4].copy_from_slice(&v.to_le_bytes());
            at += 4;
        }
    }
}

impl SessionSnapshot {
    /// The payload's size, borrowed rows included.
    pub fn len(&self) -> usize {
        self.head.len() + self.executor_len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn executor_len(&self) -> usize {
        match &self.executor {
            Part::Host(bytes) => bytes.len(),
            Part::Borrowed(handle) => handle.len(),
        }
    }

    /// Whether the executor state borrows a session's rows.
    pub fn is_borrowed(&self) -> bool {
        matches!(self.executor, Part::Borrowed(_))
    }

    /// The storage its borrowed rows live in ([`Session::snapshot_owner`]).
    pub fn source(&self) -> Option<u64> {
        match &self.executor {
            Part::Borrowed(handle) => Some(handle.registry().id()),
            Part::Host(_) => None,
        }
    }

    /// Whether `s` holds this snapshot's borrowed rows (`OwnsSnapshot`).
    pub fn borrows_from(&self, s: &Session) -> bool {
        match (&self.executor, s.native.snapshots.get()) {
            (Part::Borrowed(handle), Some(registry)) => Arc::ptr_eq(handle.registry(), registry),
            _ => false,
        }
    }

    /// Device bytes the snapshot holds, shared blocks counted once
    /// (`DeviceBytes`).
    pub fn device_bytes(&self) -> usize {
        match &self.executor {
            Part::Host(_) => 0,
            Part::Borrowed(handle) => handle.device_bytes(),
        }
    }

    /// Host memory the session part and a host executor part allocate,
    /// spare capacity included; borrowed rows count in their storage.
    pub fn host_capacity(&self) -> usize {
        self.head.capacity()
            + match &self.executor {
                Part::Host(bytes) => bytes.capacity(),
                Part::Borrowed(_) => 0,
            }
    }

    /// Writes the payload into `out`, which must have [`len`](Self::len)
    /// bytes; a borrowed snapshot materializes once (`CopyTo`).
    pub fn copy_to(&self, out: &mut [u8]) -> Result<(), String> {
        if out.len() != self.len() {
            return Err("snapshot destination size does not match the payload".into());
        }
        let (head, rest) = out.split_at_mut(self.head.len());
        head.copy_from_slice(&self.head);
        match &self.executor {
            Part::Host(bytes) => rest.copy_from_slice(bytes),
            Part::Borrowed(handle) => rest.copy_from_slice(&handle.payload()?),
        }
        Ok(())
    }

    /// The payload in a new buffer.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut out = gufo_hip::huge_page_buffer(self.len());
        self.copy_to(&mut out)?;
        Ok(out)
    }
}

pub(super) fn session_host_bytes(token_count: u32, vocab: u32, image_bytes: usize) -> u64 {
    SESSION_HEADER_BYTES as u64
        + image_bytes as u64
        + 4 * u64::from(token_count)
        + 4 * u64::from(vocab)
}

/// The executor part's length a session payload names in its header.
pub(super) fn executor_part_len(payload: &[u8]) -> usize {
    payload
        .get(24..32)
        .map_or(0, |b| {
            u64::from_le_bytes(b.try_into().unwrap_or_default()) as usize
        })
        .min(payload.len())
}

impl Model {
    /// Bytes a snapshot of the current context occupies.
    pub fn snapshot_bytes(&self, s: &Session) -> u64 {
        self.snapshots.bytes(s)
    }

    /// The whole context in host memory (`SaveSnapshot`, `kMaterialized`).
    pub fn save_snapshot(&self, s: &Session) -> Result<Vec<u8>, String> {
        self.snapshots.save(s)
    }

    /// The context as a snapshot that borrows the session's append-only rows
    /// ([`Snapshotter::capture`]).
    pub fn snapshot(&self, s: &Session) -> Result<SessionSnapshot, String> {
        self.snapshots.capture(s)
    }

    /// [`restore_snapshot`](Self::restore_snapshot) of a [`SessionSnapshot`]:
    /// device copies for a borrowed one, and into the session it borrows
    /// from, the rows it still holds stay.
    pub fn restore(&mut self, s: &mut Session, snapshot: &SessionSnapshot) -> Result<(), String> {
        let at = snapshot.head.len();
        let total = snapshot.len();
        match &snapshot.executor {
            Part::Host(bytes) => {
                self.restore_with(s, &snapshot.head, total, at, |executor, native, next| {
                    executor.restore_snapshot(native, bytes, next)
                })
            }
            Part::Borrowed(handle) => {
                self.restore_with(s, &snapshot.head, total, at, |executor, native, next| {
                    executor.restore_borrowed(native, handle, next)
                })
            }
        }
    }

    /// A capture context for another thread.
    pub fn snapshotter(&self) -> Result<Snapshotter, String> {
        Ok(Snapshotter {
            saver: self.executor.saver()?,
            vocab: self.vocab_size(),
            concurrency: self.decode_concurrency(),
        })
    }

    /// Device scratch the first batched decode allocates
    /// (`DeferredScratchBytes`).
    pub fn deferred_scratch_bytes(&self) -> usize {
        self.executor.deferred_scratch_bytes()
    }

    /// Replaces the session's context with a snapshot of the same model
    /// (`RestoreSnapshot`).
    pub fn restore_snapshot(&mut self, s: &mut Session, payload: &[u8]) -> Result<(), String> {
        self.restore_with(s, payload, payload.len(), 0, |executor, native, next| {
            let at = payload.len() - executor_part_len(payload);
            executor.restore_snapshot(native, &payload[at..], next)
        })
    }

    /// Restores the session part `payload` (at least the head of a
    /// `total`-byte payload whose executor part starts at `at`, or at its
    /// computed offset when `at` is 0) and the executor part through
    /// `executor`.
    pub(super) fn restore_with(
        &mut self,
        s: &mut Session,
        payload: &[u8],
        total: usize,
        at: usize,
        executor: impl FnOnce(&mut Executor, &mut state::Session, u32) -> Result<SnapshotInfo, String>,
    ) -> Result<(), String> {
        if payload.len() < SESSION_HEADER_BYTES {
            return Err("session snapshot is truncated".into());
        }
        let word =
            |at: usize| u32::from_le_bytes(payload[at..at + 4].try_into().unwrap_or_default());
        if payload[..8] != SESSION_MAGIC || word(8) != SNAPSHOT_PAYLOAD_VERSION {
            return Err("session snapshot format is not supported".into());
        }
        let (vocab, token_count, hidden_rows) = (word(12), word(16), word(20));
        let executor_bytes = u64::from_le_bytes(payload[24..32].try_into().unwrap_or_default());
        let policy = LengthState::read(&payload[32..32 + LengthState::BYTES]);
        let (identity_bytes, concurrency) = (word(108), word(112));
        let lookup = LookupState::read(&payload[120..120 + LookupState::BYTES]);
        if !lookup.valid(token_count as usize) {
            return Err("session snapshot prompt-lookup state is invalid".into());
        }
        if vocab != self.vocab_size()
            || token_count == 0
            || concurrency != self.decode_concurrency()
            || token_count > s.context_size()
            || (identity_bytes != 0 && identity_bytes != 32)
            || total as u64
                != session_host_bytes(token_count, vocab, identity_bytes as usize)
                    .saturating_add(executor_bytes)
            || payload.len()
                < session_host_bytes(token_count, vocab, identity_bytes as usize) as usize
            || (at != 0
                && at as u64 != session_host_bytes(token_count, vocab, identity_bytes as usize))
        {
            return Err("session snapshot does not fit this session".into());
        }
        let mut restored_policy = s.draft_length;
        if !restored_policy.restore(&policy) {
            return Err("session snapshot draft policy is invalid".into());
        }
        let mut at = SESSION_HEADER_BYTES;
        let identity = payload[at..at + identity_bytes as usize].to_vec();
        if !identity.is_empty() && identity != s.image_identity(token_count as usize) {
            return Err("image snapshot requires its matching prompt attachment".into());
        }
        at += identity.len();
        let tokens: Vec<i32> = payload[at..at + 4 * token_count as usize]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        at += 4 * token_count as usize;
        let logits: Vec<f32> = payload[at..at + 4 * vocab as usize]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if identity.is_empty() {
            self.configure_vision(s, None)?;
        }
        s.valid = false;
        let remaining = s.context_size() - token_count;
        let budget = remaining.saturating_sub(1);
        let mut next_drafts = if s.mtp_enabled() {
            restored_policy.choose(budget, token_count, 0, false)
        } else {
            0
        };
        if s.mtp_enabled() && self.draft_specials.is_some() {
            // The next request's sampler and head are not known yet: keep the
            // scratch either eligible policy can use after the restore.
            next_drafts = next_drafts.max(restored_policy.choose(budget, token_count, 0, true));
        }
        let info = match executor(&mut self.executor, &mut s.native, next_drafts) {
            Ok(info) => info,
            Err(e) => {
                self.reset(s);
                return Err(e);
            }
        };
        if info.position != token_count || info.hidden_rows != hidden_rows {
            self.reset(s);
            return Err("session snapshot positions are inconsistent".into());
        }
        s.tokens = tokens;
        s.draft_vocab_stale = true;
        s.lookup.restore(lookup);
        s.logits = logits;
        s.anchor_hint = None;
        s.hidden_base = info.position - info.hidden_rows;
        s.draft_token = 0;
        s.draft_length = restored_policy;
        s.stats = SpeculativeStats::default();
        s.valid = true;
        Ok(())
    }
}
