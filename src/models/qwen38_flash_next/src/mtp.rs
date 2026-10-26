//! The draft block (MTP) on the executor, a port of `MtpForward`, `MtpBody`,
//! `MtpHead`, `GreedyMtpPredictions` and `CopyTrunkHidden`
//! (`kernels/rocm/executor.cpp`): a one-layer predictor that embeds the
//! shifted tokens, reads the trunk's residual rows (kept on the session, or
//! still in scratch right after a prefill chunk) or its own carried residual,
//! and proposes a greedy token or the 64 best logits. Prefill keeps only
//! the draft K/V; the carried residual is valid only after a full forward
//! from a known trunk row.

use std::sync::Arc;

use gufo_core::sampling::{LogitCandidate, SamplerState, TokenId};
use gufo_hip::Stream;
use gufo_qwen38_flash_next_kernels as k;

use crate::device::Block;
use crate::draft_vocab::DRAFT_EXTRA_ROWS;
use crate::executor::{
    Core, Executor, PenaltyRows, ROPE_KEY_BIT, VerificationScratch, verification_logits,
};
use crate::forward::{AttentionControl, at, at_mut};
use crate::routes::{Error, launch_error};
use crate::state::{DraftTable, PENALTY_ROWS, Session, VEC_BATCH};

/// Partial candidates per row of the argmax kernels (`kArgmaxParts`).
const ARGMAX_PARTS: usize = 64;

/// Candidates of a draft proposal (`kMtpCandidates`).
pub const MTP_CANDIDATES: usize = 64;

/// The most verification rows one compact selection covers.
pub const COMPACT_VERIFICATION_ROWS: usize = 7;

/// Penalized tokens a compact verification row may carry
/// (`kCompactPenaltyLimit`).
pub const COMPACT_PENALTY_LIMIT: usize = 4096;

/// Whether sampled verification rows can stay on the device and reach the
/// host as compact candidate sets (`CanCompactMtpVerification`): a bounded
/// top-k within the 64 exact candidates, a bounded penalty history and no
/// active constraint or forced token.
pub fn can_compact_mtp_verification(sampler: &SamplerState) -> bool {
    let config = sampler.config();
    config.uses_random_sampling()
        && config.top_k > 0
        && config.top_k as usize <= MTP_CANDIDATES
        && config.min_keep <= MTP_CANDIDATES
        && sampler.penalties().len() + 8 <= COMPACT_PENALTY_LIMIT
        && !sampler.needs_constraint_mask().unwrap_or(true)
}

/// The best draft logits, descending with the lowest ID first on ties
/// (`MtpCandidateLogits`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpCandidates {
    pub ids: [u32; MTP_CANDIDATES],
    pub logits: [f32; MTP_CANDIDATES],
    pub size: usize,
}

impl Default for MtpCandidates {
    fn default() -> Self {
        Self {
            ids: [0; MTP_CANDIDATES],
            logits: [0.0; MTP_CANDIDATES],
            size: 0,
        }
    }
}

/// What a draft step returns; neither skips the output head (catch-up).
#[derive(Default)]
pub struct MtpOutput<'a> {
    pub token: Option<&'a mut i32>,
    pub candidates: Option<&'a mut MtpCandidates>,
    /// Prefill needs only the persistent draft K/V: a full forward from
    /// known trunk rows must follow before a head or recursive proposal.
    pub kv_only: bool,
    /// Draft through the small head (the output prefix plus the session's
    /// draft table) when the model has one.
    pub draft_subset: bool,
}

/// A row's greedy token and its logit (`ArgmaxCandidate`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ArgmaxCandidate {
    pub value: f32,
    pub index: i32,
}

/// Where the draft block's hidden input rows come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HiddenSource {
    /// The session's kept trunk rows (`max_speculative` of them).
    Kept,
    /// The residual of the trunk batch that just ran, still in scratch
    /// (`max_batch` rows): prefill catches up at once.
    Trunk,
}

/// What one draft body runs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MtpBody {
    pub n: u32,
    pub pos: u32,
    pub hidden_row: i32,
    pub source: HiddenSource,
    pub token: bool,
    pub candidates: bool,
    pub kv_only: bool,
    pub pool_grid: u32,
    /// The head scores the small vocabulary.
    pub subset: bool,
}

impl Executor {
    /// Runs the draft block over `tokens` (at most `max_batch`) at the
    /// session's MTP position. The hidden input of token i is kept trunk row
    /// `hidden_row + i`, or with `hidden_row < 0` (one token) the block's own
    /// residual from the previous call (`MtpForward`).
    pub fn mtp_forward(
        &mut self,
        session: &mut Session,
        tokens: &[i32],
        hidden_row: i32,
        output: MtpOutput<'_>,
    ) -> Result<(), String> {
        self.mtp_forward_from(session, tokens, hidden_row, output, HiddenSource::Kept)
    }

    pub(crate) fn mtp_forward_from(
        &mut self,
        session: &mut Session,
        tokens: &[i32],
        hidden_row: i32,
        output: MtpOutput<'_>,
        source: HiddenSource,
    ) -> Result<(), String> {
        let c = self.config().clone();
        let options = self.options();
        let n = tokens.len() as u32;
        if session.owner != self.id || tokens.iter().any(|&t| t < 0 || t as u32 >= c.vocab_size) {
            return Err("invalid MTP session or token".into());
        }
        if session.check_cancelled() {
            return Err("generation cancelled".into());
        }
        if !session.mtp_enabled || self.core.model.mtp.is_none() {
            return Err("no MTP block loaded".into());
        }
        let source_rows = match source {
            HiddenSource::Kept => options.max_speculative,
            HiddenSource::Trunk => options.max_batch,
        };
        if n == 0
            || n > options.max_batch
            || (hidden_row < 0 && n != 1)
            || (hidden_row >= 0 && hidden_row as u32 + n > source_rows)
        {
            return Err("MTP batch outside the kept hidden rows".into());
        }
        // Snapshots borrowing the draft rows it writes copy them out first.
        let mtp_position = session.mtp.as_ref().map_or(0, |m| m.position);
        session.preserve(session.position, mtp_position)?;
        let (pos, blocks, residual_valid) = {
            let mtp = session.mtp.as_ref().ok_or("no MTP block loaded")?;
            (mtp.position, mtp.blocks, mtp.residual_valid)
        };
        let (token, candidates, kv_only) = (
            output.token.is_some(),
            output.candidates.is_some(),
            output.kv_only,
        );
        if (hidden_row < 0 && !residual_valid)
            || (kv_only && (hidden_row < 0 || token || candidates))
        {
            return Err("MTP residual requires a full known-hidden forward".into());
        }
        if pos + n > session.max_context {
            return Err("MTP context is full".into());
        }
        session.mutation_epoch += 1;
        if let Some(mtp) = session.mtp.as_mut() {
            mtp.residual_valid = false;
        }
        let sparse = pos + n > c.indexer_top_k;
        let complete = (pos + n) / c.compress_ratio;
        let pool = if sparse {
            complete.saturating_sub(blocks)
        } else {
            0
        };
        let graph_pool = n / c.compress_ratio + 1;
        let graph = source == HiddenSource::Kept && n <= VEC_BATCH && pool <= graph_pool;
        // Captured graphs bake in the head: key on the one this draft reads.
        let subset =
            (token || candidates) && output.draft_subset && self.core.uses_draft_subset(session);
        let key = u64::from(n)
            | (u64::from(hidden_row < 0) << 32)
            | (1 << 40)
            | (u64::from(token) << 41)
            | (u64::from(candidates) << 43)
            | (u64::from(sparse) << 44)
            | (u64::from(kv_only) << 45)
            | (u64::from(subset) << 46)
            | (u64::from(session.vision.rope().is_some()) << ROPE_KEY_BIT);
        self.core
            .upload_mtp_inputs(&self.stream, session, tokens, pos, hidden_row, source_rows)
            .map_err(|e| e.to_string())?;
        let body = MtpBody {
            n,
            pos,
            hidden_row,
            source,
            token,
            candidates,
            kv_only,
            pool_grid: if graph { graph_pool } else { pool },
            subset,
        };
        self.run(session, key, graph, |core, stream, session| {
            core.mtp_body(stream, session, body)
        })
        .map_err(|e| e.to_string())?;
        self.core
            .read_mtp_outputs(&self.stream, output)
            .map_err(|e| e.to_string())?;
        let mtp = session.mtp.as_mut().ok_or("no MTP block loaded")?;
        mtp.position = pos + n;
        mtp.residual_valid = !kv_only;
        if sparse {
            mtp.blocks = complete;
        }
        Ok(())
    }

    /// Greedy tokens and their logits of the first `predictions.len()`
    /// verification rows, which stay on the device (`GreedyMtpPredictions`).
    /// Under penalties row r selects with the sampler's history after
    /// `drafts[..r]`, in the sampler's FP64 arithmetic.
    pub fn greedy_mtp_predictions(
        &mut self,
        predictions: &mut [ArgmaxCandidate],
        sampler: &SamplerState,
        drafts: &[i32],
    ) -> Result<(), String> {
        let rows = predictions.len();
        let config = sampler.config();
        if rows == 0
            || rows > self.core.options.max_logit_rows as usize
            || (config.penalties_enabled() && rows > PENALTY_ROWS)
            || drafts.len() + 1 < rows
            || self.core.model.mtp.is_none()
        {
            return Err("invalid greedy MTP verification request".into());
        }
        let vocab = self.config().vocab_size as usize;
        if config.penalties_enabled() {
            // Drafts may break the target grammar; only their penalty history
            // matters here.
            let mut tentative = sampler.without_constraint();
            let (mut triples, mut offsets) = (Vec::new(), vec![0u32]);
            // Row r sees drafts[..r]; the last row's draft is never needed.
            let mut accepted = drafts.iter().take(rows - 1);
            loop {
                for p in tentative.penalties() {
                    triples.extend([p.token, p.generated_count, p.repeated]);
                }
                offsets.push(u32::try_from(triples.len() / 3).map_err(|e| e.to_string())?);
                let Some(&draft) = accepted.next() else {
                    break;
                };
                let draft = TokenId::try_from(draft).map_err(|e| e.to_string())?;
                tentative.accept(draft).map_err(|e| e.to_string())?;
            }
            self.upload_penalty_rows(&triples)?;
            let [e0, e1, e2, e3, e4, e5, e6] = row_ends(&offsets);
            let core = &mut self.core;
            let penalty_rows = core.penalty_rows.as_ref().ok_or("penalty rows missing")?;
            let logits = verification_logits(
                core.selected,
                &core.batch_logits,
                &core.s.logits,
                vocab as u32,
                rows * vocab,
            );
            let partial = core
                .s
                .mtp_scratch_ids
                .slice_mut(0..rows * ARGMAX_PARTS * 4)
                .cast::<u8>()
                .ok_or("misaligned penalty argmax scratch")?;
            let launched = k::penalized_argmax(
                &self.stream,
                logits,
                penalty_rows.triples.as_slice(),
                e0,
                e1,
                e2,
                e3,
                e4,
                e5,
                e6,
                config.repeat_penalty,
                config.frequency_penalty,
                config.presence_penalty,
                partial,
                core.s.mtp_argmax.slice_mut(0..rows * 8),
                rows as u32,
                vocab as u32,
                penalty_rows.capacity as u32,
            )
            .map_err(|e| format!("greedy MTP predictions: {e}"))?;
            if launched != gufo_hip::Launch::Launched {
                return Err("greedy MTP predictions: penalized argmax refused".into());
            }
        } else {
            let core = &mut self.core;
            let logits = verification_logits(
                core.selected,
                &core.batch_logits,
                &core.s.logits,
                vocab as u32,
                rows * vocab,
            );
            let workspace = k::mtp_candidate_workspace(vocab as u32) as usize;
            let scratch = core
                .s
                .mtp_scratch_ids
                .slice_mut(0..workspace)
                .cast::<f32>()
                .ok_or("misaligned argmax scratch")?;
            gufo_qwen_kernels::batched_argmax(
                &self.stream,
                logits,
                core.s.mtp_ids.slice_mut(0..rows),
                rows,
                vocab,
                scratch,
                workspace,
            )
            .map_err(|e| format!("greedy MTP predictions: {e}"))?;
            let logits = verification_logits(
                core.selected,
                &core.batch_logits,
                &core.s.logits,
                vocab as u32,
                rows * vocab,
            );
            k::gather_argmax_candidates(
                &self.stream,
                logits,
                core.s.mtp_ids.slice(0..rows),
                core.s.mtp_argmax.slice_mut(0..rows * 8),
                rows as u32,
                vocab as u32,
            )
            .map_err(|e| format!("greedy MTP predictions: {e}"))?;
        }
        let mut bytes = vec![0u8; rows * 8];
        self.stream
            .download(self.core.s.mtp_argmax.slice(0..rows * 8), &mut bytes)
            .map_err(|e| format!("greedy MTP predictions download: {e}"))?;
        for (p, b) in predictions.iter_mut().zip(bytes.as_chunks::<8>().0) {
            *p = ArgmaxCandidate {
                value: f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                index: i32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            };
        }
        Ok(())
    }

    /// The compact target rows of the first `rows` verification rows, which
    /// stay on the device (`SampledMtpCandidates`): per row the exact 64
    /// best unpenalized tokens and every token the sampler's history after
    /// `drafts[..r]` penalizes, with their original logits. One transfer
    /// serves every row; no full row reaches the host.
    pub fn sampled_mtp_candidates(
        &mut self,
        rows: usize,
        sampler: &SamplerState,
        drafts: &[i32],
    ) -> Result<Vec<Vec<LogitCandidate>>, String> {
        if rows == 0
            || rows > COMPACT_VERIFICATION_ROWS
            || rows >= self.core.options.max_speculative as usize
            || rows != drafts.len()
            || self.core.model.mtp.is_none()
            || !can_compact_mtp_verification(sampler)
        {
            return Err("invalid compact MTP verification request".into());
        }
        let vocab = self.config().vocab_size;
        let mut tentative = sampler.without_constraint();
        let (mut triples, mut offsets) = (Vec::new(), vec![0u32]);
        // Row r sees drafts[..r]; the last row's draft is never needed.
        for (row, &draft) in drafts.iter().enumerate() {
            for p in tentative.penalties() {
                triples.extend([p.token, p.generated_count, p.repeated]);
            }
            offsets.push(u32::try_from(triples.len() / 3).map_err(|e| e.to_string())?);
            if row + 1 < rows {
                let draft = TokenId::try_from(draft).map_err(|e| e.to_string())?;
                tentative.accept(draft).map_err(|e| e.to_string())?;
            }
        }
        self.upload_penalty_rows(&triples)?;
        let [e0, e1, e2, e3, e4, e5, e6] = row_ends(&offsets);
        let error = |e: gufo_hip::HipError| format!("compact MTP verification: {e}");
        let core = &mut self.core;
        let capacity = core.penalty_rows.as_ref().map_or(0, |p| p.capacity);
        let workspace = k::mtp_candidate_workspace(vocab) as usize;
        let entries = COMPACT_VERIFICATION_ROWS * MTP_CANDIDATES + capacity;
        if core
            .verification
            .as_ref()
            .is_none_or(|v| v.candidate_capacity < entries)
        {
            let ids = COMPACT_VERIFICATION_ROWS * workspace;
            // Exactly sized, as C++'s `AllocateDevice`: kernels run slower on
            // pool memory.
            let s = &self.stream;
            core.verification = Some(VerificationScratch {
                ids: s.alloc_persistent::<u32>(ids).map_err(error)?,
                scratch_ids: s.alloc_persistent::<u32>(ids).map_err(error)?,
                candidates: s.alloc_persistent::<u8>(8 * entries).map_err(error)?,
                candidate_capacity: entries,
            });
        }
        let penalty_rows = core.penalty_rows.as_ref().ok_or("penalty rows missing")?;
        let scratch = core
            .verification
            .as_mut()
            .ok_or("verification scratch missing")?;
        let logits = verification_logits(
            core.selected,
            &core.batch_logits,
            &core.s.logits,
            vocab,
            rows * vocab as usize,
        );
        let count = rows * MTP_CANDIDATES + triples.len() / 3;
        let launched = k::verification_candidates(
            &self.stream,
            logits,
            penalty_rows.triples.as_slice(),
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            scratch.ids.slice_mut(0..rows * workspace),
            scratch.scratch_ids.slice_mut(0..rows * workspace),
            scratch
                .candidates
                .slice_mut(0..8 * (rows * MTP_CANDIDATES + capacity)),
            rows as u32,
            vocab,
            capacity as u32,
            workspace as u32,
        )
        .map_err(|e| format!("compact MTP verification: {e}"))?;
        if launched != gufo_hip::Launch::Launched {
            return Err("compact MTP verification refused".into());
        }
        let mut bytes = vec![0u8; 8 * count];
        self.stream
            .download(scratch.candidates.slice(0..8 * count), &mut bytes)
            .map_err(|e| format!("compact MTP verification download: {e}"))?;
        let candidates: Vec<LogitCandidate> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| LogitCandidate {
                token: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                value: f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            })
            .collect();
        Ok((0..rows)
            .map(|row| {
                let begin = row * MTP_CANDIDATES + offsets[row] as usize;
                let end = (row + 1) * MTP_CANDIDATES + offsets[row + 1] as usize;
                candidates[begin..end].to_vec()
            })
            .collect())
    }

    /// Rows `row..` of the last verification's logits on the host
    /// (`ReadVerificationRows`): the exact masked selection once a greedy
    /// argmax breaks the grammar, downloaded once rather than per rejected
    /// row.
    pub fn read_verification_rows(&self, row: usize, logits: &mut [f32]) -> Result<(), String> {
        let vocab = self.config().vocab_size as usize;
        let max = self.core.options.max_speculative as usize;
        if row >= max
            || logits.is_empty()
            || !logits.len().is_multiple_of(vocab)
            || logits.len() / vocab > max - row
        {
            return Err("invalid verification row".into());
        }
        let core = &self.core;
        let rows = verification_logits(
            core.selected,
            &core.batch_logits,
            &core.s.logits,
            vocab as u32,
            row * vocab + logits.len(),
        );
        if rows.len() != row * vocab + logits.len() {
            return Err("invalid verification row".into());
        }
        self.stream
            .download(rows.range(row * vocab..rows.len()), logits)
            .map_err(|e| format!("constrained verification row download: {e}"))
    }

    /// Uploads the verification rows' penalty histories, growing the shared
    /// buffers when they no longer fit.
    fn upload_penalty_rows(&mut self, triples: &[u32]) -> Result<(), String> {
        let needed = (triples.len() / 3).max(1);
        let error = |e: gufo_hip::HipError| format!("verification penalty upload: {e}");
        let core = &mut self.core;
        if core
            .penalty_rows
            .as_ref()
            .is_none_or(|p| p.capacity < needed)
        {
            let capacity = needed.max(2 * core.penalty_rows.as_ref().map_or(0, |p| p.capacity));
            core.penalty_rows = Some(PenaltyRows {
                triples: self
                    .stream
                    .alloc_persistent::<u32>(3 * capacity)
                    .map_err(error)?,
                capacity,
            });
        }
        let rows = core.penalty_rows.as_mut().ok_or("penalty rows missing")?;
        if !triples.is_empty() {
            self.stream
                .upload(triples, rows.triples.slice_mut(0..triples.len()))
                .map_err(error)?;
        }
        Ok(())
    }

    /// The leading kept trunk rows, for predictor diagnostics
    /// (`CopyTrunkHidden`).
    pub fn copy_trunk_hidden(&self, session: &Session, hidden: &mut [f32]) -> Result<(), String> {
        let hc = self.config().hc_dim() as usize;
        let mtp = session
            .mtp
            .as_ref()
            .filter(|_| session.owner == self.id && session.position != 0)
            .ok_or("invalid trunk hidden diagnostic input")?;
        if hidden.is_empty()
            || !hidden.len().is_multiple_of(hc)
            || hidden.len() / hc > session.position.min(self.core.options.max_speculative) as usize
        {
            return Err("invalid trunk hidden diagnostic input".into());
        }
        self.stream
            .download(mtp.target_hidden.slice(0..hidden.len()), hidden)
            .map_err(|e| format!("trunk hidden download: {e}"))
    }

    /// Rewinds the draft block's own context (`MtpRewind`).
    pub fn mtp_rewind(&self, session: &mut Session, position: u32) {
        let ratio = self.config().compress_ratio;
        if let Some(mtp) = &mut session.mtp {
            if mtp.position != position {
                mtp.residual_valid = false;
            }
            mtp.position = position;
            mtp.blocks = mtp.blocks.min(position / ratio);
        }
    }

    pub fn mtp_position(&self, session: &Session) -> u32 {
        session.mtp.as_ref().map_or(0, |m| m.position)
    }
}

impl Core {
    /// Uploads the draft tokens and control values.
    fn upload_mtp_inputs(
        &mut self,
        stream: &Stream,
        session: &mut Session,
        tokens: &[i32],
        pos: u32,
        hidden_row: i32,
        source_rows: u32,
    ) -> Result<(), Error> {
        let n = tokens.len();
        let host = self.host.tokens.host_mut()?;
        for (slot, &t) in host.iter_mut().zip(tokens) {
            *slot = t as u32;
        }
        stream.upload_indices_async(&mut self.host.tokens, n, &mut self.s.tokens)?;
        let n = n as u32;
        let blocks = session.mtp.as_ref().map_or(0, |m| m.blocks);
        let control = &mut session.control;
        control.set_mtp_position(stream, n, pos)?;
        control.set_hidden_row(stream, n, source_rows, hidden_row)?;
        control.set_mtp_blocks(stream, blocks)?;
        Ok(())
    }

    /// One draft batch (`MtpBody`), control and token uploads excluded.
    pub(crate) fn mtp_body(
        &mut self,
        stream: &Stream,
        session: &mut Session,
        b: MtpBody,
    ) -> Result<(), Error> {
        let c = self.model.config.clone();
        let model = Arc::clone(&self.model);
        let l = model.mtp.as_ref().ok_or("no MTP block loaded")?;
        let nextn = l
            .nextn
            .as_ref()
            .ok_or("MTP block without its projections")?;
        let Block::Attention(a) = &l.block else {
            return Err("the MTP block has no attention".into());
        };
        self.reset_route();
        let n = b.n;
        let hc = c.hc_dim() as usize;
        let (hidden, eps) = (c.hidden_size, c.rms_eps);
        let token_embd = &model.token_embd;
        // MTP embeds the shifted token IDs; visual information arrives
        // through the trunk's hidden rows.
        self.route.mixed_q8 = None;
        k::embed_tokens(
            stream,
            token_embd.view(),
            self.s.tokens.slice(0..n as usize),
            at_mut(&mut self.s.mixed, 0),
            n,
            hidden,
            1,
            token_embd.rows,
        )
        .map_err(launch_error("draft embedding"))?;
        k::rms_norm_rows_in_place(
            stream,
            nextn.enorm.view(),
            at_mut(&mut self.s.mixed, 0),
            n,
            hidden,
            1,
            eps,
        )
        .map_err(launch_error("draft embedding norm"))?;
        let cancelled = session.check_cancelled();
        let Session {
            mtp,
            control,
            max_context,
            index_capacity,
            vision,
            ..
        } = session;
        let (max_context, index_capacity) = (*max_context, *index_capacity);
        let mtp = mtp.as_mut().ok_or("no MTP block loaded")?;
        let source_rows = match b.source {
            HiddenSource::Kept => self.options.max_speculative,
            HiddenSource::Trunk => self.options.max_batch,
        };
        let row = control
            .hidden_row(n, source_rows)
            .ok_or("hidden row not uploaded")?;
        if b.source == HiddenSource::Trunk && b.hidden_row >= 0 {
            // Eager catch-up normalizes the trunk rows in place of a copy;
            // captured blocks (kept rows) select theirs on the device, since
            // a replay can change the row. The trunk residual stays as it is.
            let from = b.hidden_row as usize * hc;
            k::rms_norm_rows(
                stream,
                at(&self.s.res, from),
                nextn.hnorm.view(),
                at_mut(&mut self.s.xn, 0),
                n,
                c.hc_dim(),
                1,
                eps,
            )
            .map_err(launch_error("draft hidden norm"))?;
        } else {
            let base = match b.source {
                HiddenSource::Kept => mtp.target_hidden.slice(0..source_rows as usize * hc),
                HiddenSource::Trunk => self.s.res.slice(0..source_rows as usize * hc),
            };
            // The carried residual is read only for one token at row -1; a
            // kept row reads the source instead.
            let alt = if b.hidden_row < 0 {
                mtp.h.slice(0..hc)
            } else {
                base.range(0..n as usize * hc)
            };
            k::mtp_hidden(
                stream,
                base,
                alt,
                row.view(),
                at_mut(&mut self.s.xn, 0),
                n,
                c.hc_dim(),
                source_rows,
            )
            .map_err(launch_error("draft hidden input"))?;
            // Unlike the mixers, this norm spans the whole HC * H row.
            k::rms_norm_rows_in_place(
                stream,
                nextn.hnorm.view(),
                at_mut(&mut self.s.xn, 0),
                n,
                c.hc_dim(),
                1,
                eps,
            )
            .map_err(launch_error("draft hidden norm"))?;
        }
        let skipped;
        {
            let mut pass = self.pass(stream, 0);
            // These projections determine the persistent draft K/V: prompt
            // chunks and decode share the row-wise activation quantization
            // and reduction, so sampled replay survives cache reuse.
            pass.ctx.dense_batch(
                &nextn.fc_embedding,
                at(&pass.s.mixed, 0),
                pass.s.phase.slice_mut(pass.s.qkvz.at(0)),
                n,
            )?;
            pass.ctx.dense_batch(
                &nextn.fc_hidden,
                at(&pass.s.xn, 0),
                at_mut(&mut pass.s.res, 0),
                n * c.hc_count,
            )?;
            k::mtp_add_embedding(
                stream,
                pass.s.phase.slice(pass.s.qkvz.at(0)),
                at_mut(&mut pass.s.res, 0),
                n,
                hidden,
                c.hc_count,
            )
            .map_err(launch_error("draft embedding sum"))?;
            // Catch-up exports KV for every row but carries only its final
            // residual: past 32 rows only the final query tile attends.
            let last_only = !b.token && !b.candidates && n > 32 && b.hidden_row >= 0;
            // Keep the final 128-row tile (and its predecessor for short
            // tails): at least 96 rows keep the wide arithmetic, and
            // attention writes every KV row before the view narrows. Short
            // catch-up carries only its final row too: two rows keep the
            // router's and projections' multi-row arithmetic, which a single
            // row reduces differently.
            skipped = if b.hidden_row >= 0 && n > 2 && n <= VEC_BATCH {
                n - 2
            } else if last_only && n >= 224 {
                (n - 96) / 128 * 128
            } else {
                0
            };
            let cache_only = b.kv_only && a.qkv.is_some();
            pass.hc_mix_batch(&l.hc_attn, false, !cache_only, n)?;
            let position = control
                .mtp_position(n)
                .ok_or("draft position not uploaded")?;
            if cache_only {
                // Known trunk rows replace the draft residual on the next
                // forward: only K/V and the pooled indexer keys survive.
                // The packed weight's K/V rows keep its arithmetic.
                let kv = a
                    .qkv
                    .as_ref()
                    .and_then(|qkv| qkv.last_rows(2 * c.attention_kv_dim()))
                    .ok_or("MTP cache projection failed")?;
                pass.ctx.dense_batch(
                    &kv,
                    at(&pass.s.mixed, 0),
                    pass.s.phase.slice_mut(pass.s.qg.at(0)),
                    n,
                )?;
                // Only K/V: the query and gate views are empty.
                let ([qg], [q, gate]) = pass.s.phase.views(
                    [pass.s.qg.at(0)],
                    [pass.s.q.part(0..0), pass.s.attn_gate.part(0..0)],
                );
                let prepared = k::prepare_attention(
                    stream,
                    qg,
                    kv.rows,
                    a.k_norm.view(),
                    a.k_norm.view(),
                    q,
                    gate,
                    mtp.attention.k_cache.as_mut_slice(),
                    mtp.attention.v_cache.as_mut_slice(),
                    n,
                    0,
                    c.num_kv_heads,
                    c.head_dim,
                    c.rotary_dim,
                    position.view(),
                    c.rope_theta,
                    eps,
                    vision.rope(),
                    false,
                    pass.ctx.yarn.enabled,
                    pass.ctx.yarn.low,
                    pass.ctx.yarn.high,
                    pass.ctx.yarn.inv_factor,
                    pass.ctx.yarn.mscale,
                    max_context,
                )
                .map_err(launch_error("MTP cache projection"))?;
                if prepared != gufo_hip::Launch::Launched {
                    return Err("MTP cache projection failed".into());
                }
                pass.ctx.dense_batch(
                    &a.indexer_k,
                    at(&pass.s.mixed, 0),
                    pass.s.phase.slice_mut(pass.s.ik.at(0)),
                    n,
                )?;
                k::store_rows(
                    stream,
                    pass.s.phase.slice(pass.s.ik.at(0)),
                    mtp.attention.index_k.as_mut_slice(),
                    n,
                    c.indexer_head_dim,
                    position.view(),
                    index_capacity,
                    max_context,
                )
                .map_err(launch_error("indexer key store"))?;
                if b.pos + n > c.indexer_top_k {
                    k::pool_indexer_blocks(
                        stream,
                        mtp.attention.index_k.as_slice(),
                        a.indexer_k_norm.view(),
                        mtp.attention.block_k.as_mut_slice(),
                        control.mtp_blocks().view(),
                        position.view(),
                        n,
                        b.pool_grid,
                        c.compress_ratio,
                        c.indexer_head_dim,
                        c.rotary_dim,
                        c.rope_theta,
                        eps,
                        index_capacity,
                        vision.rope(),
                        pass.ctx.yarn.enabled,
                        pass.ctx.yarn.low,
                        pass.ctx.yarn.high,
                        pass.ctx.yarn.inv_factor,
                        pass.ctx.yarn.mscale,
                        max_context,
                    )
                    .map_err(launch_error("indexer pooling"))?;
                }
                return Ok(());
            }
            match &a.qkv {
                Some(qkv) => pass.ctx.dense_batch(
                    qkv,
                    at(&pass.s.mixed, 0),
                    pass.s.phase.slice_mut(pass.s.qg.at(0)),
                    n,
                )?,
                None => {
                    let (q, kk, v) = (
                        a.q.as_ref().ok_or("missing query projection")?,
                        a.k.as_ref().ok_or("missing key projection")?,
                        a.v.as_ref().ok_or("missing value projection")?,
                    );
                    let x = at(&pass.s.mixed, 0);
                    pass.ctx
                        .dense_batch(q, x, pass.s.phase.slice_mut(pass.s.qg.at(0)), n)?;
                    let x = at(&pass.s.mixed, 0);
                    pass.ctx
                        .dense_batch(kk, x, pass.s.phase.slice_mut(pass.s.k.at(0)), n)?;
                    let x = at(&pass.s.mixed, 0);
                    pass.ctx
                        .dense_batch(v, x, pass.s.phase.slice_mut(pass.s.v.at(0)), n)?;
                }
            }
            let attention_control = AttentionControl {
                position: position.view(),
                first_block: control.mtp_blocks().view(),
                start_pos: b.pos,
                pool_grid: b.pool_grid,
                max_context,
                index_capacity,
                sparse: b.pos + n > c.indexer_top_k,
                rope: vision.rope(),
            };
            pass.attention(
                a,
                &mut mtp.attention,
                &attention_control,
                n,
                last_only,
                true,
                skipped == 0,
            )?;
            let inject_parts = pass.ctx.route.inject_parts as usize;
            let tail_rows = n - skipped;
            if skipped != 0 {
                // The tail keeps the producer's inject stride.
                pass.row = skipped as usize;
                pass.inject_at = Some(skipped as usize * c.hc_count as usize * inject_parts);
                pass.ctx.stage.q8t_src = None;
                pass.ctx.stage.half_src = None;
                pass.ctx.route.xn_half = false;
                pass.ctx.route.moe_pending = false;
                let r = pass.row;
                let (q, h) = (pass.w.q, pass.w.hidden);
                pass.ctx.dense(
                    &a.out,
                    pass.s.phase.slice(pass.s.ctx.at(r * q)),
                    at_mut(&mut pass.s.block_out, r * h),
                    tail_rows,
                )?;
            }
            pass.combine(Some(&l.hc_ffn.norm), Some(&l.hc_ffn.down), tail_rows, None)?;
            if cancelled {
                return Err("generation cancelled".into());
            }
            let res_at = pass.row * hc;
            pass.hc_mix(&l.hc_ffn, res_at, true, true, tail_rows, false, false)?;
            if skipped != 0 && tail_rows == 2 {
                // The discarded padding row shares the final row's routed
                // experts, so the multi-row arithmetic reads no other set.
                let h = pass.w.hidden;
                let r = pass.row;
                crate::batch::move_row(stream, at_mut(&mut pass.s.mixed, r * h), h, 1, 0)?;
                pass.ctx.stage.q8t_src = None;
                pass.ctx.stage.half_src = None;
                pass.ctx.route.mixed_q8 = None;
            }
            pass.moe(&l.moe, tail_rows, last_only)?;
            pass.combine(None, None, tail_rows, None)?;
        }
        let last = (n - 1) as usize * hc;
        stream.graph_copy(self.s.res.slice(last..last + hc), mtp.h.as_mut_slice())?;
        if b.token || b.candidates {
            let table = mtp.draft.as_ref().filter(|_| b.subset);
            self.mtp_head(stream, &nextn.head, last, b.token, b.candidates, table)?;
        }
        let _ = skipped;
        Ok(())
    }

    /// Whether a draft of `session` that asks for the small head gets it
    /// (`UseDraftSubset`).
    pub(crate) fn uses_draft_subset(&self, session: &Session) -> bool {
        self.model.draft.is_some() && session.mtp.as_ref().is_some_and(|m| m.draft.is_some())
    }

    /// Greedy token or compact candidates of the residual row at element
    /// `res_at` of the scratch (`MtpHead`); with `table`, over that
    /// session's small head, the selections remapped to token IDs.
    pub(crate) fn mtp_head(
        &mut self,
        stream: &Stream,
        head: &crate::device::Mixer,
        res_at: usize,
        token: bool,
        candidates: bool,
        table: Option<&DraftTable>,
    ) -> Result<(), Error> {
        let model = Arc::clone(&self.model);
        let small = table.zip(model.draft.as_ref());
        let rows = {
            let mut pass = self.pass(stream, 0);
            pass.hc_mix(head, res_at, false, false, 1, false, false)?;
            match small {
                Some((table, draft)) => {
                    pass.draft_logits(draft, table, 0)?;
                    draft.prefix.rows + DRAFT_EXTRA_ROWS
                }
                None => {
                    pass.ctx.dense(
                        &model.output,
                        at(&pass.s.mixed, 0),
                        at_mut(&mut pass.s.logits, 0),
                        1,
                    )?;
                    model.config.vocab_size
                }
            }
        };
        self.select_draft(stream, 0, token, candidates, rows)?;
        match small {
            Some((table, draft)) => {
                self.remap_draft(stream, table, draft.prefix.rows, token, candidates)
            }
            None => Ok(()),
        }
    }

    /// The small head's row numbers in the draft selections back to token
    /// IDs (`RemapDraftIds`).
    pub(crate) fn remap_draft(
        &mut self,
        stream: &Stream,
        table: &DraftTable,
        prefix: u32,
        token: bool,
        candidates: bool,
    ) -> Result<(), Error> {
        let vocab = self.model.config.vocab_size;
        let remap = |ids: gufo_hip::DeviceSliceMut<'_, u32>, n: u32| {
            k::remap_draft_ids(
                stream,
                ids,
                n,
                prefix,
                table.ids.view(),
                DRAFT_EXTRA_ROWS,
                vocab,
            )
            .map_err(launch_error("draft ID remap"))
        };
        if candidates {
            let count = MTP_CANDIDATES.min((prefix + DRAFT_EXTRA_ROWS) as usize);
            remap(self.s.mtp_ids.slice_mut(0..count), count as u32)?;
        }
        if token {
            let id = self
                .s
                .mtp_token
                .slice_mut(0..1)
                .cast::<u32>()
                .ok_or("misaligned draft token")?;
            remap(id, 1)?;
        }
        Ok(())
    }

    /// The head's selections over logits row `row` of `logits`.
    pub(crate) fn select_draft(
        &mut self,
        stream: &Stream,
        at_row: usize,
        token: bool,
        candidates: bool,
        vocab: u32,
    ) -> Result<(), Error> {
        let v = vocab as usize;
        let logits = self.s.logits.slice(at_row * v..(at_row + 1) * v);
        let workspace = k::mtp_candidate_workspace(vocab);
        if candidates {
            k::mtp_top_candidates(
                stream,
                logits,
                self.s.mtp_ids.as_mut_slice(),
                self.s.mtp_scratch_ids.as_mut_slice(),
                vocab,
                workspace,
            )
            .map_err(launch_error("draft candidates"))?;
        }
        if token {
            k::argmax(
                stream,
                logits,
                self.s.mtp_argmax.slice_mut(0..64 * 8),
                self.s.mtp_token.as_mut_slice(),
                1,
                vocab,
            )
            .map_err(launch_error("draft argmax"))?;
        }
        Ok(())
    }

    /// Downloads the head's selections into pinned staging, waits for the
    /// stream and reads them.
    fn read_mtp_outputs(&mut self, stream: &Stream, output: MtpOutput<'_>) -> Result<(), Error> {
        let count = (self.model.config.vocab_size as usize).min(MTP_CANDIDATES);
        if output.token.is_some() {
            stream.download_async(self.s.mtp_token.slice(0..1), &mut self.host.draft_tokens[0])?;
        }
        if output.candidates.is_some() {
            stream.download_async(
                self.s.mtp_ids.slice(0..MTP_CANDIDATES + count),
                &mut self.host.draft_candidates[0],
            )?;
        }
        stream.synchronize()?;
        if let Some(token) = output.token {
            *token = self.host.draft_tokens[0].host()?[0];
        }
        if let Some(candidates) = output.candidates {
            let words = self.host.draft_candidates[0].host()?;
            candidates.size = count;
            candidates.ids.copy_from_slice(&words[..MTP_CANDIDATES]);
            for (logit, &bits) in candidates.logits.iter_mut().zip(&words[MTP_CANDIDATES..]) {
                *logit = f32::from_bits(bits);
            }
        }
        Ok(())
    }
}

/// The penalty rows' ends as the kernels take them (`GreedyPenaltyRows`):
/// row r ends at `offsets[r + 1]` (row 0 starts at 0), and the ends past the
/// last row repeat it.
fn row_ends(offsets: &[u32]) -> [u32; PENALTY_ROWS] {
    let last = offsets.last().copied().unwrap_or(0);
    std::array::from_fn(|r| offsets.get(r + 1).copied().unwrap_or(last))
}
