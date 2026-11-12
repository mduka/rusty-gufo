//! The DFlash2 draft forward passes, a port of `QwenDFlashGpuExecutor`
//! (`q35/model/hip/dflash/executor.cpp`, `batch.cpp` of the qwen35moe
//! extension): target-feature injection into the per-layer K/V history ring,
//! and the block pass that turns an anchor and mask tokens into proposals,
//! greedy or drawn at a temperature. A [`DraftScratch`] holds the activations
//! one pass uses; each [`DraftSession`] holds one sequence's history ring and
//! draft table, on the same stream, so a batched pass reads several
//! sessions. Every call waits for its work, as in C++.
//!
//! The batched passes differ from C++ in two launch groupings that compute
//! the same values: the selector runs once per sequence (its sequences are
//! independent blocks of one kernel), and the scratch is separate buffers
//! instead of one packed allocation. The sessions of a pass share one
//! injection input, so their numeric penalties are packed into it and
//! uploaded once before the pass, where each C++ executor copies its own
//! asynchronously before its selector. A single block is the batched pass of
//! one session: its GEMMs and selector take the same routes as C++'s
//! `ForwardBlock` (no shape of this draft has a `verify_gemm_27b` tile).
//!
//! Each session keeps its draft table ([`draft_vocab`](crate::dflash::draft_vocab)):
//! head rows, their IDs (-1 empty) on the device and a host mirror of the
//! IDs. A block drafted through it scores the head's leading rows in place
//! and the table's rows; empty slots score -inf.

use std::ops::Range;

use gufo_core::sampling::SamplerState;
use gufo_hip::{
    DeviceBuffer, DeviceIndices, DeviceSlice, DeviceSliceMut, DeviceTensor, IndexSpec, Launch,
    LaunchError, QuantSlice, Stream,
};
use gufo_qwen35moe_kernels as k;

use crate::arena::alloc;
use crate::config::NORM_EPS;
use crate::dflash::draft_vocab::DRAFT_EXTRA_ROWS;
use crate::dflash::model::{DFlashModel, Matrix};

/// Rows the single-session block buffers hold (`kDFlashMaxSharedBatch`).
const MAX_SHARED_BATCH: usize = 8;
/// Rows a batched block pass may hold: eight sessions of eight rows.
const BATCH_ROWS: usize = 64;
/// Rows per injection chunk.
const INJECTION_CAPACITY: u32 = 256;
/// Selector partials per 2048-token vocabulary slice.
const SELECTOR_SLICE: usize = 2048;
/// Prompt-sized context injections for MoE targets take the BF16 WMMA GEMM
/// from this width, the target's own BF16 prefill threshold
/// (`kDFlashInjectWmmaMinRows`).
const INJECT_WMMA_MIN_ROWS: usize = k::sizes::BF16_WMMA_MIN_PREFILL_BATCH;
/// Table rows per session (`kDraftRows`).
const TABLE_ROWS: usize = DRAFT_EXTRA_ROWS as usize;

fn launch(error: LaunchError) -> String {
    format!("DFlash2 forward pass: {error}")
}

/// `y[rows][m] = x[rows][k] w^T` for a block (`RunBlockGemm`,
/// `ProjectBlock`).
fn block_gemm(
    s: &Stream,
    w: &Matrix,
    x: DeviceSlice<'_, f32>,
    mut y: DeviceSliceMut<'_, f32>,
    rows: usize,
    m: usize,
    kk: usize,
) -> Result<(), LaunchError> {
    match w {
        Matrix::Quant(q) => {
            if rows <= MAX_SHARED_BATCH
                && k::verify_gemm_27b(s, q.view(), x, y.reborrow(), rows, m, kk)?
                    == Launch::Launched
            {
                return Ok(());
            }
            k::batched_quant_gemm_f32(s, q.view(), x, y, rows, m, kk)
        }
        Matrix::Bf16(b) => k::exact_bf16_gemm_f32(s, b.view(), x, y, rows, m, kk),
    }
}

/// BF16 injection projections share weight reads over 16 rows, tails over 8
/// (`RunInjectGemm`). Prompt-sized injections for an MoE target take the
/// BF16 WMMA GEMM instead: sixteen-row exact groups re-read the whole matrix
/// once per group, and BF16-rounded activations change only the draft's
/// context K/V, never what the target verifies.
#[allow(clippy::too_many_arguments)]
fn inject_gemm(
    s: &Stream,
    model: &DFlashModel,
    w: &DeviceTensor<gufo_hip::Bf16>,
    x: DeviceSlice<'_, f32>,
    mut y: DeviceSliceMut<'_, f32>,
    n: usize,
    m: usize,
    kk: usize,
) -> Result<(), LaunchError> {
    if n == 0 {
        return Ok(());
    }
    if model.target_is_moe && n >= INJECT_WMMA_MIN_ROWS && k::sizes::grouped_bf16_gemm(m, kk, 1) {
        return k::bf16_wmma_gemm(s, w.view(), x.range(0..n * kk), y.range(0..n * m), n, m, kk);
    }
    let mut offset = 0;
    while offset < n {
        let chunk = if n - offset >= 16 {
            16
        } else {
            MAX_SHARED_BATCH.min(n - offset)
        };
        k::exact_bf16_gemm_f32(
            s,
            w.view(),
            x.range(offset * kk..(offset + chunk) * kk),
            y.reborrow().range(offset * m..(offset + chunk) * m),
            chunk,
            m,
            kk,
        )?;
        offset += chunk;
    }
    Ok(())
}

/// The draft tokens after an anchor, with each row's top-k candidates and
/// their probabilities when they were drawn at a temperature
/// (`DraftProposal`). A copy from the prompt history reports one candidate
/// per row, the copied token at probability one, when sampled.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Proposal {
    pub tokens: Vec<u32>,
    pub candidate_ids: Vec<u32>,
    pub candidate_probabilities: Vec<f32>,
    /// Copied by prompt lookup instead of drafted by the neural block.
    pub from_lookup: bool,
}

/// The numeric penalties of a request's sampler that its draft q follows
/// across the chain (`PrepareDraftSampling`): repeat-window, frequency and
/// presence penalties, with the generated counts and the repetition window
/// they read. Target rejection and residual correction are unchanged.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DraftPenalties {
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    repeat_last_n: usize,
    /// `(token, generated count, repeated)` per penalty, then the history.
    words: Vec<u32>,
    penalty_count: u32,
    history_count: u32,
    /// The request's sampler has no seed (`seed < 0`): its proposals need
    /// not be reproducible across schedules, so a copy may verify wide.
    unseeded: bool,
}

impl DraftPenalties {
    /// Whether the request's sampler is unseeded.
    pub fn unseeded(&self) -> bool {
        self.unseeded
    }

    /// The controls of `sampler`. The selector already restricts q to a
    /// small lattice; filtering it a second time (top-k, top-p, min-p) lost
    /// useful coverage on short sampled completions, so only the penalties
    /// apply.
    pub fn of(sampler: &SamplerState) -> Self {
        let config = sampler.config();
        let penalties = if config.frequency_penalty != 0.0 || config.presence_penalty != 0.0 {
            sampler.penalties()
        } else {
            &[]
        };
        let history = if config.repeat_penalty != 1.0 {
            sampler.history()
        } else {
            &[]
        };
        let mut words = Vec::with_capacity(penalties.len() * 3 + history.len());
        for penalty in penalties {
            words.extend([penalty.token, penalty.generated_count, penalty.repeated]);
        }
        words.extend_from_slice(history);
        Self {
            repeat_penalty: config.repeat_penalty,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            repeat_last_n: config.repeat_last_n,
            words,
            penalty_count: penalties.len() as u32,
            history_count: history.len() as u32,
            unseeded: config.seed < 0,
        }
    }
}

/// One block's selector controls in a pass.
struct SelectorControls<'a> {
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    repeat_last_n: u32,
    words: &'a [u32],
    penalty_count: u32,
    history_count: u32,
    /// Where `words` lands in the scratch's injection input.
    at: Range<usize>,
}

impl<'a> SelectorControls<'a> {
    /// The controls of `penalties` for a session of `max_context`
    /// positions; neutral when the policy would not fit the injection input
    /// of `capacity` words, which keeps the original valid q.
    fn new(penalties: Option<&'a DraftPenalties>, max_context: u32, capacity: usize) -> Self {
        let neutral = Self {
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: 0,
            words: &[],
            penalty_count: 0,
            history_count: 0,
            at: 0..0,
        };
        match penalties {
            Some(p) if p.words.len() <= capacity => Self {
                repeat_penalty: p.repeat_penalty,
                frequency_penalty: p.frequency_penalty,
                presence_penalty: p.presence_penalty,
                repeat_last_n: p.repeat_last_n.min(max_context as usize) as u32,
                words: &p.words,
                penalty_count: p.penalty_count,
                history_count: p.history_count,
                at: 0..0,
            },
            _ => neutral,
        }
    }
}

/// How the selector picks: greedy, or drawn at `temperature` with one
/// uniform per row.
#[derive(Debug, Clone, Copy)]
pub enum Selection<'a> {
    Greedy,
    Sampled {
        temperature: f32,
        uniforms: &'a [f32],
    },
}

/// Activations of one injection or block pass.
pub struct DraftScratch {
    stream_id: u64,
    injection_capacity: u32,
    /// Rows the block buffers hold: one block, grown for batched passes.
    rows: usize,
    target_features: DeviceBuffer<f32>,
    fused_features: DeviceBuffer<f32>,
    block_normed: DeviceBuffer<f32>,
    k_block: DeviceBuffer<f32>,
    v_block: DeviceBuffer<f32>,
    block_hidden: DeviceBuffer<f32>,
    conv_hidden: DeviceBuffer<f32>,
    dynamic_coefficients: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    attn_out: DeviceBuffer<f32>,
    ffn_gate: DeviceBuffer<f32>,
    ffn_up: DeviceBuffer<f32>,
    ffn_down: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    /// One block of table scores per proposal row (`draft_logits`); empty
    /// when the model drafts over the full head.
    draft_logits: DeviceBuffer<f32>,
    selector_hidden: DeviceBuffer<f32>,
    confidences: DeviceBuffer<f32>,
    /// `[anchor, mask...]` per session for the embedding lookup.
    block_tokens: DeviceIndices,
    /// The selector's anchor slots and proposals.
    selector_tokens: DeviceBuffer<u32>,
    uniforms: DeviceBuffer<f32>,
    candidate_ids: DeviceBuffer<u32>,
    candidate_probabilities: DeviceBuffer<f32>,
}

/// One sequence's injected K/V history and its draft table.
pub struct DraftSession {
    stream_id: u64,
    max_context: u32,
    injected: u32,
    /// First position whose K/V is present; positions below it were skipped
    /// (`context_floor_`).
    context_floor: u32,
    history_capacity: u32,
    injected_k: Vec<DeviceBuffer<f32>>,
    injected_v: Vec<DeviceBuffer<f32>>,
    table: Option<DraftTable>,
}

/// A session's draft table (`d_draft_rows_`, `d_draft_ids_`): head rows and
/// their IDs, -1 for an empty slot.
struct DraftTable {
    /// `[TABLE_ROWS][row_bytes]` in the head's format; stale rows stay.
    rows: DeviceBuffer<u8>,
    /// The slots' IDs as the scoring kernels read them, padding empty.
    ids: DeviceIndices,
    /// The appended IDs for the gather kernel.
    gather: DeviceIndices,
    /// Host mirror of `ids`.
    host: Vec<u32>,
}

/// The occupied ring slots of every layer (`QwenDFlashGpuSnapshot`),
/// immutable: K then V, each `[layer][slots]`.
pub struct DFlashSnapshot {
    k: DeviceTensor<f32>,
    v: DeviceTensor<f32>,
    valid_context: u32,
}

impl DFlashSnapshot {
    pub fn valid_context(&self) -> u32 {
        self.valid_context
    }

    /// Device bytes of both planes.
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 4
    }

    pub fn k(&self) -> &DeviceTensor<f32> {
        &self.k
    }

    pub fn v(&self) -> &DeviceTensor<f32> {
        &self.v
    }
}

impl DraftScratch {
    /// The block buffers hold one block (`Allocate`); a batched pass grows
    /// them ([`DraftScratch::reserve`]).
    pub fn new(s: &Stream, model: &DFlashModel, max_context: u32) -> Result<Self, String> {
        let rows = (model.config.block_size as usize).min(BATCH_ROWS);
        Self::with_rows(s, model, max_context.min(INJECTION_CAPACITY), rows)
    }

    /// Room for `rows` block rows, rounded up to a power of two like the C++
    /// batch scratch (`BatchScratchBytes`). Contents do not carry over: call
    /// it before a pass writes its inputs.
    fn reserve(&mut self, s: &Stream, model: &DFlashModel, rows: usize) -> Result<(), String> {
        if rows > self.rows {
            let rows = rows.next_power_of_two().min(BATCH_ROWS);
            *self = Self::with_rows(s, model, self.injection_capacity, rows)?;
        }
        Ok(())
    }

    fn with_rows(
        s: &Stream,
        model: &DFlashModel,
        injection_capacity: u32,
        rows: usize,
    ) -> Result<Self, String> {
        let c = &model.config;
        let (hidden, kv, q) = (c.hidden_size as usize, c.kv_size(), c.attention_size());
        let vocab = c.vocab_size as usize;
        let partials = vocab.div_ceil(SELECTOR_SLICE) * 16;
        let ffn_width = (c.intermediate_size as usize).max(partials);
        let wide_rows = (injection_capacity as usize).max(rows);
        let top_k = c.selector_top_k as usize;
        let block_tokens = alloc(s, rows)?;
        Ok(Self {
            stream_id: s.id(),
            injection_capacity,
            rows,
            target_features: alloc(s, injection_capacity as usize * c.target_features())?,
            fused_features: alloc(s, wide_rows * hidden)?,
            block_normed: alloc(s, wide_rows * hidden)?,
            k_block: alloc(s, wide_rows * kv)?,
            v_block: alloc(s, wide_rows * kv)?,
            block_hidden: alloc(s, rows * hidden)?,
            conv_hidden: alloc(s, rows * hidden)?,
            dynamic_coefficients: alloc(s, rows * c.dynamic_coefficients())?,
            q: alloc(s, rows * q)?,
            attn_out: alloc(s, rows * q)?,
            ffn_gate: alloc(s, rows * ffn_width)?,
            ffn_up: alloc(s, rows * ffn_width)?,
            ffn_down: alloc(s, rows * hidden)?,
            logits: alloc(s, rows * vocab)?,
            draft_logits: alloc(
                s,
                if model.draft_prefix_rows == 0 {
                    0
                } else {
                    rows * TABLE_ROWS
                },
            )?,
            selector_hidden: alloc(s, rows * c.selector_rank as usize)?,
            confidences: alloc(s, rows)?,
            block_tokens: s
                .zeroed_indices(block_tokens, c.vocab_size)
                .map_err(|e| format!("DFlash2 buffer: {e}"))?,
            selector_tokens: alloc(s, rows)?,
            uniforms: alloc(s, rows)?,
            candidate_ids: alloc(s, rows * top_k)?,
            candidate_probabilities: alloc(s, rows * top_k)?,
        })
    }
}

impl DraftSession {
    pub fn new(s: &Stream, model: &DFlashModel, max_context: u32) -> Result<Self, String> {
        let c = &model.config;
        if max_context == 0 || max_context > c.context_length {
            return Err("DFlash2 context exceeds the draft model".into());
        }
        let history_capacity = max_context.min(c.sliding_window);
        let history = history_capacity as usize * c.kv_size();
        let mut injected_k = Vec::new();
        let mut injected_v = Vec::new();
        for _ in 0..c.num_layers {
            injected_k.push(alloc(s, history)?);
            injected_v.push(alloc(s, history)?);
        }
        let table = if model.draft_prefix_rows == 0 {
            None
        } else {
            let error = |e| format!("DFlash2 draft table: {e}");
            let spec = IndexSpec::below(c.vocab_size).signed();
            Some(DraftTable {
                rows: alloc(s, TABLE_ROWS * model.draft_row_bytes)?,
                ids: s
                    .zeroed_indices_with(alloc(s, TABLE_ROWS)?, spec.padded())
                    .map_err(error)?,
                gather: s
                    .zeroed_indices_with(alloc(s, TABLE_ROWS)?, spec)
                    .map_err(error)?,
                host: vec![u32::MAX; TABLE_ROWS],
            })
        };
        let mut session = Self {
            stream_id: s.id(),
            max_context,
            injected: 0,
            context_floor: 0,
            history_capacity,
            injected_k,
            injected_v,
            table,
        };
        if session.table.is_some() {
            // The table starts with the special tokens; every other slot is
            // empty.
            session.clear_draft_rows(s, 0)?;
            session.append_draft_rows(s, model, 0, &model.draft_specials)?;
            s.synchronize()
                .map_err(|e| format!("DFlash2 draft table: {e}"))?;
        }
        Ok(session)
    }

    /// Device bytes of a session of `max_context` positions: the K/V window
    /// rings of every layer and the draft table (`EstimateMemoryUsage`;
    /// the table's scores live in the shared [`DraftScratch`] here).
    pub fn device_bytes(model: &DFlashModel, max_context: u32) -> usize {
        let c = &model.config;
        let rings = 2
            * c.num_layers as usize
            * max_context.min(c.sliding_window) as usize
            * c.kv_size()
            * 4;
        let table = if model.draft_prefix_rows == 0 {
            0
        } else {
            TABLE_ROWS * (model.draft_row_bytes + 2 * 4)
        };
        rings + table
    }

    /// Whether the session has a draft table to draft through.
    pub fn has_draft_table(&self) -> bool {
        self.table.is_some()
    }

    /// Empties the draft table from slot `first` on: the slots score -inf
    /// (`ClearDraftRows`). Stale rows stay.
    pub fn clear_draft_rows(&mut self, s: &Stream, first: u32) -> Result<(), String> {
        let table = self
            .table
            .as_mut()
            .filter(|_| first as usize <= TABLE_ROWS && self.stream_id == s.id())
            .ok_or("invalid DFlash draft table clear")?;
        if first as usize == TABLE_ROWS {
            return Ok(());
        }
        table.host[first as usize..].fill(u32::MAX);
        s.upload_indices(&table.host, &mut table.ids)
            .map_err(|e| format!("DFlash2 draft table: {e}"))
    }

    /// Copies the head rows of `ids` into the draft table from slot `first`
    /// on (`AppendDraftRows`).
    pub fn append_draft_rows(
        &mut self,
        s: &Stream,
        model: &DFlashModel,
        first: u32,
        ids: &[i32],
    ) -> Result<(), String> {
        if ids.is_empty() {
            return Ok(());
        }
        let vocab = model.config.vocab_size;
        let first = first as usize;
        let table = self
            .table
            .as_mut()
            .filter(|_| {
                self.stream_id == s.id()
                    && first <= TABLE_ROWS
                    && ids.len() <= TABLE_ROWS - first
                    && ids.iter().all(|&id| id >= 0 && (id as u32) < vocab)
            })
            .ok_or("invalid DFlash draft table append")?;
        let error = |e| format!("DFlash2 draft table: {e}");
        let host: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        s.upload_indices(&host, &mut table.gather).map_err(error)?;
        table.host[first..first + ids.len()].copy_from_slice(&host);
        s.upload_indices(&table.host, &mut table.ids)
            .map_err(error)?;
        let row = model.draft_row_bytes;
        k::dflash_gather_rows(
            s,
            model.output.view().bytes(),
            row,
            table.gather.slice(0..ids.len()),
            table.rows.slice_mut(first * row..(first + ids.len()) * row),
            ids.len() as u32,
            vocab as usize,
        )
        .map_err(launch)
    }

    /// Advances the injected length to `position` without writing K/V
    /// (`SkipTargetContextTo`). The skipped positions must leave the
    /// attention window before the next block is drafted;
    /// [`context_complete`](Self::context_complete) reports whether they
    /// have.
    pub fn skip_to(&mut self, position: u32) -> Result<(), String> {
        if position < self.injected || position > self.max_context {
            return Err("DFlash context skip position is invalid".into());
        }
        if position != self.injected {
            self.injected = position;
            self.context_floor = position;
        }
        Ok(())
    }

    /// Whether every position in the attention window has its K/V
    /// (`ContextComplete`).
    pub fn context_complete(&self) -> bool {
        self.injected - self.context_floor >= self.injected.min(self.history_capacity)
    }

    pub fn injected(&self) -> u32 {
        self.injected
    }

    pub fn history_capacity(&self) -> u32 {
        self.history_capacity
    }

    pub fn max_context(&self) -> u32 {
        self.max_context
    }

    /// Forgets the injected context; new rows overwrite the ring. The draft
    /// table stays (the backend restarts it from the next history).
    pub fn reset(&mut self) {
        self.injected = 0;
        self.context_floor = 0;
    }

    fn check(&self, s: &Stream, x: &DraftScratch) -> Result<(), String> {
        if self.stream_id != s.id() || x.stream_id != s.id() {
            return Err("DFlash2 session belongs to another stream".into());
        }
        Ok(())
    }

    /// Copies the occupied ring slots of every layer on `transfer`, which
    /// may be another stream than the session's: the session must be idle
    /// (`SaveSnapshot`).
    pub fn capture(
        &self,
        transfer: &Stream,
        model: &DFlashModel,
    ) -> Result<DFlashSnapshot, String> {
        let kv = model.config.kv_size();
        let slots = self.injected.min(self.history_capacity) as usize * kv;
        let layers = self.injected_k.len();
        let across = transfer.id() != self.stream_id;
        let error = |e| format!("DFlash2 snapshot: {e}");
        let mut k_copy = transfer.alloc_persistent(layers * slots).map_err(error)?;
        let mut v_copy = transfer.alloc_persistent(layers * slots).map_err(error)?;
        for (i, (k_ring, v_ring)) in self.injected_k.iter().zip(&self.injected_v).enumerate() {
            let range = i * slots..(i + 1) * slots;
            if across {
                transfer
                    .copy_across(k_ring.slice(0..slots), k_copy.slice_mut(range.clone()))
                    .map_err(error)?;
                transfer
                    .copy_across(v_ring.slice(0..slots), v_copy.slice_mut(range))
                    .map_err(error)?;
            } else {
                transfer
                    .copy(k_ring.slice(0..slots), k_copy.slice_mut(range.clone()))
                    .map_err(error)?;
                transfer
                    .copy(v_ring.slice(0..slots), v_copy.slice_mut(range))
                    .map_err(error)?;
            }
        }
        Ok(DFlashSnapshot {
            k: k_copy.freeze(transfer).map_err(error)?,
            v: v_copy.freeze(transfer).map_err(error)?,
            valid_context: self.injected,
        })
    }

    /// Returns to `snapshot` (`RestoreSnapshot`).
    pub fn restore(
        &mut self,
        s: &Stream,
        model: &DFlashModel,
        snapshot: &DFlashSnapshot,
    ) -> Result<(), String> {
        let kv = model.config.kv_size();
        let slots = snapshot.valid_context.min(self.history_capacity) as usize * kv;
        let layers = self.injected_k.len();
        if self.stream_id != s.id()
            || snapshot.valid_context > self.max_context
            || snapshot.k.len() != layers * slots
        {
            return Err("DFlash2 snapshot is incompatible with the executor".into());
        }
        let error = |e| format!("DFlash2 snapshot restore: {e}");
        for (i, (k_ring, v_ring)) in self
            .injected_k
            .iter_mut()
            .zip(&mut self.injected_v)
            .enumerate()
        {
            s.copy(
                snapshot.k.slice(i * slots..(i + 1) * slots),
                k_ring.slice_mut(0..slots),
            )
            .map_err(error)?;
            s.copy(
                snapshot.v.slice(i * slots..(i + 1) * slots),
                v_ring.slice_mut(0..slots),
            )
            .map_err(error)?;
        }
        s.synchronize().map_err(error)?;
        self.injected = snapshot.valid_context;
        self.context_floor = 0;
        Ok(())
    }

    /// Loads persisted ring slots (`GDFGPU01` K and V sections, f32 bytes)
    /// of `valid_context` positions (`RestorePersistentSnapshot`).
    pub fn restore_bytes(
        &mut self,
        s: &Stream,
        model: &DFlashModel,
        k_bytes: &[u8],
        v_bytes: &[u8],
        valid_context: u32,
    ) -> Result<(), String> {
        let kv = model.config.kv_size();
        let slots = valid_context.min(self.history_capacity) as usize * kv;
        let layers = self.injected_k.len();
        if self.stream_id != s.id()
            || valid_context > self.max_context
            || k_bytes.len() != layers * slots * 4
            || v_bytes.len() != k_bytes.len()
        {
            return Err("DFlash GPU persistent payload size is invalid".into());
        }
        let error = |e| format!("DFlash2 persistent restore: {e}");
        let unaddressable = "DFlash2 history is not byte addressable";
        let row = slots * 4;
        for (i, (k_ring, v_ring)) in self
            .injected_k
            .iter_mut()
            .zip(&mut self.injected_v)
            .enumerate()
        {
            if slots == 0 {
                continue;
            }
            let target = k_ring
                .slice_mut(0..slots)
                .cast::<u8>()
                .ok_or(unaddressable)?;
            s.upload(&k_bytes[i * row..(i + 1) * row], target)
                .map_err(error)?;
            let target = v_ring
                .slice_mut(0..slots)
                .cast::<u8>()
                .ok_or(unaddressable)?;
            s.upload(&v_bytes[i * row..(i + 1) * row], target)
                .map_err(error)?;
        }
        self.injected = valid_context;
        self.context_floor = 0;
        Ok(())
    }
}

/// Projects `rows` target feature rows (`[row][taps * hidden]`) at
/// positions `position..` into every layer's K/V history
/// (`InjectTargetContext`). Rows the ring would overwrite anyway are skipped
/// in whole injection chunks.
pub fn inject(
    s: &Stream,
    model: &DFlashModel,
    x: &mut DraftScratch,
    st: &mut DraftSession,
    features: &[f32],
    position: u32,
    rows: u32,
) -> Result<(), String> {
    st.check(s, x)?;
    let width = model.config.target_features();
    if position != st.injected
        || position > st.max_context
        || rows > st.max_context - position
        || features.len() != rows as usize * width
    {
        return Err("DFlash2 target context does not continue the injected rows".into());
    }
    let expired = rows.saturating_sub(st.history_capacity);
    let mut offset = (expired / x.injection_capacity) * x.injection_capacity;
    while offset < rows {
        let count = x.injection_capacity.min(rows - offset);
        let range = offset as usize * width..(offset + count) as usize * width;
        inject_chunk(s, model, x, st, &features[range], position + offset, count)?;
        offset += count;
    }
    Ok(())
}

fn inject_chunk(
    s: &Stream,
    model: &DFlashModel,
    x: &mut DraftScratch,
    st: &mut DraftSession,
    features: &[f32],
    position: u32,
    rows: u32,
) -> Result<(), String> {
    let c = &model.config;
    let (hidden, kv, width) = (c.hidden_size as usize, c.kv_size(), c.target_features());
    let n = rows as usize;
    s.upload(features, x.target_features.slice_mut(0..n * width))
        .map_err(|e| format!("DFlash2 target features: {e}"))?;
    inject_gemm(
        s,
        model,
        &model.fc,
        x.target_features.slice(0..n * width),
        x.fused_features.slice_mut(0..n * hidden),
        n,
        hidden,
        width,
    )
    .map_err(launch)?;
    k::batched_rms_norm(
        s,
        x.fused_features.slice(0..n * hidden),
        model.fc_norm.view(),
        x.block_normed.slice_mut(0..n * hidden),
        n,
        hidden,
        NORM_EPS,
    )
    .map_err(launch)?;
    for (i, layer) in model.layers.iter().enumerate() {
        let input = x.block_normed.slice(0..n * hidden);
        inject_gemm(
            s,
            model,
            &layer.k,
            input,
            x.k_block.slice_mut(0..n * kv),
            n,
            kv,
            hidden,
        )
        .map_err(launch)?;
        inject_gemm(
            s,
            model,
            &layer.v,
            input,
            x.v_block.slice_mut(0..n * kv),
            n,
            kv,
            hidden,
        )
        .map_err(launch)?;
        k::dflash_k_norm_rope(
            s,
            layer.k_norm.view(),
            x.k_block.slice_mut(0..n * kv),
            position,
            n,
            c.num_key_value_heads,
            c.head_dim,
            c.rotary_dim,
            c.rope_theta,
            NORM_EPS,
        )
        .map_err(launch)?;
        write_ring(s, x, st, i, 0, position, rows, kv)?;
    }
    st.injected = st.injected.max(position + rows);
    s.synchronize()
        .map_err(|e| format!("DFlash2 injection: {e}"))
}

/// Copies `rows` K/V block rows from `block_offset` into layer `layer`'s
/// ring at `position..`, split where the ring wraps.
#[allow(clippy::too_many_arguments)]
fn write_ring(
    s: &Stream,
    x: &DraftScratch,
    st: &mut DraftSession,
    layer: usize,
    block_offset: usize,
    position: u32,
    rows: u32,
    kv: usize,
) -> Result<(), String> {
    let mut offset = 0;
    while offset < rows {
        let slot = (position + offset) % st.history_capacity;
        let count = (rows - offset).min(st.history_capacity - slot);
        let from =
            (block_offset + offset as usize) * kv..(block_offset + (offset + count) as usize) * kv;
        let to = slot as usize * kv..(slot + count) as usize * kv;
        let error = |e| format!("DFlash2 history: {e}");
        s.copy(
            x.k_block.slice(from.clone()),
            st.injected_k[layer].slice_mut(to.clone()),
        )
        .map_err(error)?;
        s.copy(x.v_block.slice(from), st.injected_v[layer].slice_mut(to))
            .map_err(error)?;
        offset += count;
    }
    Ok(())
}

/// Injects the target features of several sessions in one pass
/// (`InjectTargetContextBatch`); each continues its session's injected
/// rows. One session, or more rows than an injection chunk holds, inject one
/// after the other.
pub fn inject_batch(
    s: &Stream,
    model: &DFlashModel,
    x: &mut DraftScratch,
    requests: &mut [(&mut DraftSession, &[f32])],
) -> Result<(), String> {
    let c = &model.config;
    let width = c.target_features();
    if requests.is_empty() || requests.len() > 8 {
        return Err("DFlash2 context batch requires one to eight sessions".into());
    }
    let mut counts = Vec::with_capacity(requests.len());
    for (st, features) in requests.iter() {
        st.check(s, x)?;
        let position = st.injected;
        if !features.len().is_multiple_of(width)
            || features.len() / width > (st.max_context - position) as usize
        {
            return Err("DFlash2 context batch has an invalid request".into());
        }
        counts.push((features.len() / width) as u32);
    }
    let rows: u32 = counts.iter().sum();
    if rows == 0 {
        return Ok(());
    }
    if requests.len() == 1 || rows > x.injection_capacity || c.head_dim > 256 {
        for ((st, features), &count) in requests.iter_mut().zip(&counts) {
            let position = st.injected;
            inject(s, model, x, st, features, position, count)?;
        }
        return Ok(());
    }
    let (hidden, kv) = (c.hidden_size as usize, c.kv_size());
    let n = rows as usize;
    let mut offsets = Vec::with_capacity(requests.len());
    let mut offset = 0usize;
    for ((_, features), &count) in requests.iter().zip(&counts) {
        offsets.push(offset);
        if count != 0 {
            s.upload(
                features,
                x.target_features
                    .slice_mut(offset * width..(offset + count as usize) * width),
            )
            .map_err(|e| format!("DFlash2 target features: {e}"))?;
        }
        offset += count as usize;
    }
    inject_gemm(
        s,
        model,
        &model.fc,
        x.target_features.slice(0..n * width),
        x.fused_features.slice_mut(0..n * hidden),
        n,
        hidden,
        width,
    )
    .map_err(launch)?;
    k::batched_rms_norm(
        s,
        x.fused_features.slice(0..n * hidden),
        model.fc_norm.view(),
        x.block_normed.slice_mut(0..n * hidden),
        n,
        hidden,
        NORM_EPS,
    )
    .map_err(launch)?;
    for (i, layer) in model.layers.iter().enumerate() {
        let input = x.block_normed.slice(0..n * hidden);
        inject_gemm(
            s,
            model,
            &layer.k,
            input,
            x.k_block.slice_mut(0..n * kv),
            n,
            kv,
            hidden,
        )
        .map_err(launch)?;
        inject_gemm(
            s,
            model,
            &layer.v,
            input,
            x.v_block.slice_mut(0..n * kv),
            n,
            kv,
            hidden,
        )
        .map_err(launch)?;
        for ((st, _), (&count, &offset)) in requests.iter_mut().zip(counts.iter().zip(&offsets)) {
            if count == 0 {
                continue;
            }
            let position = st.injected;
            k::dflash_k_norm_rope(
                s,
                layer.k_norm.view(),
                x.k_block
                    .slice_mut(offset * kv..(offset + count as usize) * kv),
                position,
                count as usize,
                c.num_key_value_heads,
                c.head_dim,
                c.rotary_dim,
                c.rope_theta,
                NORM_EPS,
            )
            .map_err(launch)?;
            write_ring(s, x, st, i, offset, position, count, kv)?;
        }
    }
    s.synchronize()
        .map_err(|e| format!("DFlash2 injection: {e}"))?;
    for ((st, _), &count) in requests.iter_mut().zip(&counts) {
        st.injected += count;
    }
    Ok(())
}

/// One session of a batched block pass.
pub struct BlockRequest<'a> {
    pub session: &'a mut DraftSession,
    pub anchor: u32,
    pub position: u32,
    pub count: u32,
    pub selection: Selection<'a>,
    /// The request's numeric penalties, which guide q.
    pub penalties: Option<&'a DraftPenalties>,
    /// Score the session's draft vocabulary instead of the full head.
    pub draft_subset: bool,
}

/// Proposal rows `(offset, count)` of a block of `rows` session rows.
struct Block {
    /// First row of the session (its anchor) in the pass.
    offset: usize,
    count: usize,
    /// Drafted through the session's draft vocabulary.
    subset: bool,
}

/// Up to `count` proposals after `anchor` at `position`, which must be the
/// injected length (`ForwardBlock`); `penalties` evolve across the chain.
/// With `draft_subset` the block scores the session's draft vocabulary.
#[allow(clippy::too_many_arguments)]
pub fn forward_block(
    s: &Stream,
    model: &DFlashModel,
    x: &mut DraftScratch,
    st: &DraftSession,
    anchor: u32,
    position: u32,
    count: u32,
    selection: Selection<'_>,
    penalties: Option<&DraftPenalties>,
    draft_subset: bool,
) -> Result<Proposal, String> {
    st.check(s, x)?;
    let c = &model.config;
    let count = count.min(c.block_size - 1);
    if anchor >= c.vocab_size
        || c.mask_token_id >= c.vocab_size
        || position != st.injected
        || position >= st.max_context
        || (draft_subset && st.table.is_none())
    {
        return Err("DFlash block token or committed position is invalid".into());
    }
    let count = count.min(st.max_context - position - 1);
    if count == 0 {
        return Ok(Proposal::default());
    }
    let blocks = [Block {
        offset: 0,
        count: count as usize,
        subset: draft_subset,
    }];
    let mut tokens = vec![c.mask_token_id; count as usize + 1];
    tokens[0] = anchor;
    run_blocks(
        s,
        model,
        x,
        &[st],
        &blocks,
        &[position],
        &tokens,
        &[selection],
        &[penalties],
    )
    .map(|mut proposals| proposals.remove(0))
}

/// Proposals for several sessions in one pass (`ForwardBlockBatch`). One
/// session, or a session without room for a proposal, runs alone.
pub fn forward_block_batch(
    s: &Stream,
    model: &DFlashModel,
    x: &mut DraftScratch,
    requests: &mut [BlockRequest<'_>],
) -> Result<Vec<Proposal>, String> {
    let c = &model.config;
    if requests.is_empty() || requests.len() > 8 {
        return Err("DFlash2 block batch requires one to eight sessions".into());
    }
    let mut counts = Vec::with_capacity(requests.len());
    for request in requests.iter() {
        request.session.check(s, x)?;
        let temperature = match request.selection {
            Selection::Greedy => 0.0,
            Selection::Sampled { temperature, .. } => temperature,
        };
        if request.anchor >= c.vocab_size
            || c.mask_token_id >= c.vocab_size
            || request.position != request.session.injected
            || request.position >= request.session.max_context
            || !temperature.is_finite()
            || temperature < 0.0
            || (request.draft_subset && request.session.table.is_none())
        {
            return Err("DFlash2 block batch has an invalid request".into());
        }
        let count = request
            .count
            .min(c.block_size - 1)
            .min(request.session.max_context - request.position - 1);
        if let Selection::Sampled { uniforms, .. } = request.selection
            && uniforms.len() < count as usize
        {
            return Err("DFlash2 block batch is missing sampling draws".into());
        }
        counts.push(count);
    }
    if requests.len() == 1 || c.head_dim > 256 || counts.contains(&0) {
        return requests
            .iter()
            .map(|request| {
                forward_block(
                    s,
                    model,
                    x,
                    request.session,
                    request.anchor,
                    request.position,
                    request.count,
                    request.selection,
                    request.penalties,
                    request.draft_subset,
                )
            })
            .collect();
    }
    let mut blocks = Vec::with_capacity(requests.len());
    let mut tokens = Vec::new();
    for (request, &count) in requests.iter().zip(&counts) {
        blocks.push(Block {
            offset: tokens.len(),
            count: count as usize,
            subset: request.draft_subset,
        });
        tokens.push(request.anchor);
        tokens.extend(std::iter::repeat_n(c.mask_token_id, count as usize));
    }
    let sessions: Vec<&DraftSession> = requests.iter().map(|r| &*r.session).collect();
    let positions: Vec<u32> = requests.iter().map(|r| r.position).collect();
    let selections: Vec<Selection<'_>> = requests.iter().map(|r| r.selection).collect();
    let penalties: Vec<Option<&DraftPenalties>> = requests.iter().map(|r| r.penalties).collect();
    run_blocks(
        s,
        model,
        x,
        &sessions,
        &blocks,
        &positions,
        &tokens,
        &selections,
        &penalties,
    )
}

/// The block pass of one or more sessions: `tokens` holds each session's
/// anchor and masks at its block offset.
#[allow(clippy::too_many_arguments)]
fn run_blocks(
    s: &Stream,
    model: &DFlashModel,
    x: &mut DraftScratch,
    sessions: &[&DraftSession],
    blocks: &[Block],
    positions: &[u32],
    tokens: &[u32],
    selections: &[Selection<'_>],
    penalties: &[Option<&DraftPenalties>],
) -> Result<Vec<Proposal>, String> {
    let c = &model.config;
    let vocab = c.vocab_size;
    let rows = tokens.len();
    if rows > BATCH_ROWS {
        return Err("DFlash2 block batch exceeds its scratch".into());
    }
    x.reserve(s, model, rows)?;
    let (hidden, ffn, q_dim, kv) = (
        c.hidden_size as usize,
        c.intermediate_size as usize,
        c.attention_size(),
        c.kv_size(),
    );
    let dynamic = c.dynamic_coefficients();
    let scale = 1.0 / (c.head_dim as f32).sqrt();
    s.upload_indices(tokens, &mut x.block_tokens)
        .map_err(|e| format!("DFlash2 block tokens: {e}"))?;
    let proposal_rows: usize = blocks.iter().map(|b| b.count).sum();
    let mut uniforms = vec![0f32; proposal_rows];
    let mut compact = 0;
    for (block, selection) in blocks.iter().zip(selections) {
        if let Selection::Sampled {
            uniforms: draws, ..
        } = selection
        {
            uniforms[compact..compact + block.count].copy_from_slice(&draws[..block.count]);
        }
        compact += block.count;
    }
    if selections
        .iter()
        .any(|s| matches!(s, Selection::Sampled { .. }))
    {
        s.upload(&uniforms, x.uniforms.slice_mut(0..proposal_rows))
            .map_err(|e| format!("DFlash2 sampling draws: {e}"))?;
    }
    // Each request's penalty words, packed into the injection input, which
    // the context projection no longer needs. The words of a policy that
    // does not fit behind the earlier ones are uploaded again from the
    // front right before its selector, after the earlier selectors in
    // stream order.
    let capacity = x.target_features.len();
    let mut controls: Vec<SelectorControls<'_>> = penalties
        .iter()
        .zip(sessions)
        .map(|(penalties, st)| SelectorControls::new(*penalties, st.max_context, capacity))
        .collect();
    let mut packed = Vec::new();
    let mut late = Vec::new();
    for (index, control) in controls.iter_mut().enumerate() {
        if control.words.is_empty() {
            continue;
        }
        if packed.len() + control.words.len() <= capacity {
            control.at = packed.len()..packed.len() + control.words.len();
            packed.extend_from_slice(control.words);
        } else {
            control.at = 0..control.words.len();
            late.push(index);
        }
    }
    if !packed.is_empty() {
        upload_penalties(s, x, &packed)?;
    }
    let result = (|| -> Result<(), LaunchError> {
        k::batched_embedding_lookup(
            s,
            model.token_embd.view(),
            x.block_tokens.slice(0..rows),
            x.block_hidden.slice_mut(0..rows * hidden),
            rows,
            hidden,
            vocab as usize,
        )?;
        for (i, layer) in model.layers.iter().enumerate() {
            let conv = |x: &mut DraftScratch,
                        input: Which,
                        base: DeviceSlice<'_, f32>,
                        side: u32|
             -> Result<(), LaunchError> {
                for block in blocks {
                    let (from, n) = (block.offset, block.count + 1);
                    let source = match input {
                        Which::Normed => x.block_normed.slice(from * hidden..(from + n) * hidden),
                        Which::Fused => x.fused_features.slice(from * hidden..(from + n) * hidden),
                        Which::Down => x.ffn_down.slice(from * hidden..(from + n) * hidden),
                    };
                    k::dflash_grouped_dynamic_conv(
                        s,
                        source,
                        x.dynamic_coefficients
                            .slice(from * dynamic..(from + n) * dynamic),
                        base,
                        x.conv_hidden.slice_mut(from * hidden..(from + n) * hidden),
                        n as u32,
                        c.hidden_size,
                        c.conv_kernel_size,
                        c.conv_group_size,
                        side,
                    )?;
                }
                Ok(())
            };
            k::batched_rms_norm(
                s,
                x.block_hidden.slice(0..rows * hidden),
                layer.attn_norm.view(),
                x.block_normed.slice_mut(0..rows * hidden),
                rows,
                hidden,
                NORM_EPS,
            )?;
            block_gemm(
                s,
                &layer.attn_conv_proj,
                x.block_normed.slice(0..rows * hidden),
                x.dynamic_coefficients.slice_mut(0..rows * dynamic),
                rows,
                dynamic,
                hidden,
            )?;
            conv(x, Which::Normed, layer.attn_conv_base.view(), 0)?;
            let input = x.conv_hidden.slice(0..rows * hidden);
            block_gemm(
                s,
                &layer.q,
                input,
                x.q.slice_mut(0..rows * q_dim),
                rows,
                q_dim,
                hidden,
            )?;
            k::exact_bf16_gemm_f32(
                s,
                layer.k.view(),
                input,
                x.k_block.slice_mut(0..rows * kv),
                rows,
                kv,
                hidden,
            )?;
            k::exact_bf16_gemm_f32(
                s,
                layer.v.view(),
                input,
                x.v_block.slice_mut(0..rows * kv),
                rows,
                kv,
                hidden,
            )?;
            for (block, &position) in blocks.iter().zip(positions) {
                let (from, n) = (block.offset, block.count + 1);
                k::dflash_qk_norm_rope(
                    s,
                    layer.q_norm.view(),
                    layer.k_norm.view(),
                    x.q.slice_mut(from * q_dim..(from + n) * q_dim),
                    x.k_block.slice_mut(from * kv..(from + n) * kv),
                    position,
                    n,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    c.rotary_dim,
                    c.rope_theta,
                    NORM_EPS,
                )?;
            }
            for ((block, &position), st) in blocks.iter().zip(positions).zip(sessions) {
                let (from, n) = (block.offset, block.count + 1);
                k::dflash_noncausal_attention(
                    s,
                    x.q.slice(from * q_dim..(from + n) * q_dim),
                    st.injected_k[i].as_slice(),
                    st.injected_v[i].as_slice(),
                    x.k_block.slice(from * kv..(from + n) * kv),
                    x.v_block.slice(from * kv..(from + n) * kv),
                    x.attn_out.slice_mut(from * q_dim..(from + n) * q_dim),
                    position,
                    position.min(st.injected),
                    n as u32,
                    c.sliding_window,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    scale,
                    st.history_capacity,
                )?;
            }
            block_gemm(
                s,
                &layer.output,
                x.attn_out.slice(0..rows * q_dim),
                x.fused_features.slice_mut(0..rows * hidden),
                rows,
                hidden,
                q_dim,
            )?;
            conv(x, Which::Fused, layer.attn_conv_base.view(), 1)?;
            k::batched_residual_add(
                s,
                x.conv_hidden.slice(0..rows * hidden),
                x.block_hidden.slice_mut(0..rows * hidden),
                rows,
                hidden,
            )?;
            k::batched_rms_norm(
                s,
                x.block_hidden.slice(0..rows * hidden),
                layer.ffn_norm.view(),
                x.block_normed.slice_mut(0..rows * hidden),
                rows,
                hidden,
                NORM_EPS,
            )?;
            block_gemm(
                s,
                &layer.ffn_conv_proj,
                x.block_normed.slice(0..rows * hidden),
                x.dynamic_coefficients.slice_mut(0..rows * dynamic),
                rows,
                dynamic,
                hidden,
            )?;
            conv(x, Which::Normed, layer.ffn_conv_base.view(), 0)?;
            let input = x.conv_hidden.slice(0..rows * hidden);
            block_gemm(
                s,
                &layer.gate,
                input,
                x.ffn_gate.slice_mut(0..rows * ffn),
                rows,
                ffn,
                hidden,
            )?;
            block_gemm(
                s,
                &layer.up,
                input,
                x.ffn_up.slice_mut(0..rows * ffn),
                rows,
                ffn,
                hidden,
            )?;
            k::dflash_silu_mul(
                s,
                x.ffn_gate.slice_mut(0..rows * ffn),
                x.ffn_up.slice(0..rows * ffn),
                (rows * ffn) as u32,
            )?;
            block_gemm(
                s,
                &layer.down,
                x.ffn_gate.slice(0..rows * ffn),
                x.ffn_down.slice_mut(0..rows * hidden),
                rows,
                hidden,
                ffn,
            )?;
            conv(x, Which::Down, layer.ffn_conv_base.view(), 1)?;
            k::batched_residual_add(
                s,
                x.conv_hidden.slice(0..rows * hidden),
                x.block_hidden.slice_mut(0..rows * hidden),
                rows,
                hidden,
            )?;
        }
        k::batched_rms_norm(
            s,
            x.block_hidden.slice(0..rows * hidden),
            model.output_norm.view(),
            x.block_normed.slice_mut(0..rows * hidden),
            rows,
            hidden,
            NORM_EPS,
        )?;
        // The vocabulary projection reads the proposal rows only; the
        // finished hidden rows hold them compacted.
        let mut compact = 0;
        for block in blocks {
            s.copy(
                x.block_normed
                    .slice((block.offset + 1) * hidden..(block.offset + 1 + block.count) * hidden),
                x.block_hidden
                    .slice_mut(compact * hidden..(compact + block.count) * hidden),
            )?;
            compact += block.count;
        }
        let rank = c.selector_rank as usize;
        let proposals = proposal_rows;
        let input = x.block_hidden.slice(0..proposals * hidden);
        block_gemm(
            s,
            &model.selector_hidden,
            input,
            x.selector_hidden.slice_mut(0..proposals * rank),
            proposals,
            rank,
            hidden,
        )?;
        let mut firsts = Vec::with_capacity(blocks.len());
        let mut compact = 0;
        for block in blocks {
            firsts.push(compact);
            compact += block.count;
        }
        // Each block keeps the head it chose. When every block drafts
        // through its draft vocabulary, one projection reads the shared
        // prefix and each block's table follows. A mixed pass reads the full
        // head anyway, so every row goes through it once and each
        // draft-vocabulary block reads its table scores from its own full
        // rows.
        let prefix = model.draft_prefix_rows as usize;
        let all_subset = blocks.iter().all(|b| b.subset);
        let stride = if all_subset { prefix } else { vocab as usize };
        let head = if all_subset {
            QuantSlice::new(
                model
                    .output
                    .view()
                    .bytes()
                    .range(0..prefix * model.draft_row_bytes),
                model.output.format(),
            )
        } else {
            model.output.view()
        };
        k::batched_quant_gemm_f32(
            s,
            head,
            input,
            x.logits.slice_mut(0..proposals * stride),
            proposals,
            stride,
            hidden,
        )?;
        for ((block, st), &first) in blocks.iter().zip(sessions).zip(&firsts) {
            if !block.subset {
                continue;
            }
            let table = st.table.as_ref().ok_or(LaunchError::InvalidArgument {
                param: "draft_subset",
                reason: "the session has no draft table",
            })?;
            let n = block.count;
            let table_logits = x
                .draft_logits
                .slice_mut(first * TABLE_ROWS..(first + n) * TABLE_ROWS);
            if all_subset {
                k::batched_quant_gemm_f32(
                    s,
                    QuantSlice::new(table.rows.as_slice(), model.output.format()),
                    x.block_hidden.slice(first * hidden..(first + n) * hidden),
                    table_logits,
                    n,
                    TABLE_ROWS,
                    hidden,
                )?;
                k::dflash_draft_table_logits(
                    s,
                    None,
                    0,
                    table.ids.view(),
                    x.draft_logits
                        .slice_mut(first * TABLE_ROWS..(first + n) * TABLE_ROWS),
                    n as u32,
                    TABLE_ROWS as u32,
                    vocab,
                )?;
            } else {
                k::dflash_draft_table_logits(
                    s,
                    Some(x.logits.slice(first * stride..(first + n) * stride)),
                    stride as u32,
                    table.ids.view(),
                    table_logits,
                    n as u32,
                    TABLE_ROWS as u32,
                    vocab,
                )?;
            }
        }
        // The finished FFN rows hold the selector's top-k partials.
        let per_row = (vocab as usize).div_ceil(SELECTOR_SLICE) * 16;
        let top_k = c.selector_top_k as usize;
        // Packed controls stay valid until a late upload overwrites them.
        let order = (0..blocks.len())
            .filter(|index| !late.contains(index))
            .chain(late.iter().copied());
        for index in order {
            let (block, selection) = (&blocks[index], &selections[index]);
            let control = &controls[index];
            if late.contains(&index) {
                upload_penalties(s, x, control.words).map_err(|_| {
                    LaunchError::InvalidArgument {
                        param: "penalties",
                        reason: "DFlash2 draft penalties could not be uploaded",
                    }
                })?;
            }
            let split = control.at.start + control.penalty_count as usize * 3;
            let penalty_words = word_view(&x.target_features, control.at.start..split)?;
            let history_words = word_view(&x.target_features, split..control.at.end)?;
            let (n, first) = (block.count, firsts[index]);
            let partials = first * per_row..(first + n) * per_row;
            let partial_ids = x.ffn_up.slice_mut(partials.clone()).cast::<u32>().ok_or(
                LaunchError::InvalidArgument {
                    param: "partial_ids",
                    reason: "FFN rows are not whole u32 values",
                },
            )?;
            // A draft-vocabulary block reads the first `prefix` columns of
            // its rows.
            let width = if block.subset { prefix } else { stride };
            let logits = x
                .logits
                .slice(first * stride..(first + n - 1) * stride + width);
            let selector = x.selector_hidden.slice(first * rank..(first + n) * rank);
            let anchor = x.block_tokens.slice(block.offset..block.offset + 1);
            let tokens = x
                .selector_tokens
                .slice_mut(block.offset..block.offset + n + 1);
            if block.subset {
                let table = sessions[index]
                    .table
                    .as_ref()
                    .ok_or(LaunchError::InvalidArgument {
                        param: "draft_subset",
                        reason: "the session has no draft table",
                    })?;
                let (temperature, candidates) = match *selection {
                    Selection::Greedy => (0.0, None),
                    Selection::Sampled { temperature, .. } => (
                        temperature,
                        Some((
                            x.candidate_ids
                                .slice_mut(first * top_k..(first + n) * top_k),
                            x.candidate_probabilities
                                .slice_mut(first * top_k..(first + n) * top_k),
                        )),
                    ),
                };
                let (candidate_ids, candidate_probabilities) = candidates.unzip();
                k::dflash_selector_vocab_strided(
                    s,
                    logits,
                    stride as u32,
                    selector,
                    model.selector_predecessor.view(),
                    model.selector_successor.view(),
                    anchor,
                    tokens,
                    x.confidences.slice_mut(first..first + n),
                    x.ffn_gate.slice_mut(partials),
                    partial_ids,
                    x.uniforms.slice(first..first + n),
                    candidate_ids,
                    candidate_probabilities,
                    x.draft_logits
                        .slice(first * TABLE_ROWS..(first + n) * TABLE_ROWS),
                    table.ids.view(),
                    TABLE_ROWS as u32,
                    prefix as u32,
                    n as u32,
                    temperature,
                    vocab,
                    c.selector_rank,
                    c.selector_top_k,
                    control.repeat_penalty,
                    control.frequency_penalty,
                    control.presence_penalty,
                    control.repeat_last_n,
                    penalty_words,
                    control.penalty_count,
                    history_words,
                    control.history_count,
                )?;
                continue;
            }
            match *selection {
                Selection::Greedy => k::dflash_selector(
                    s,
                    logits,
                    selector,
                    model.selector_predecessor.view(),
                    model.selector_successor.view(),
                    anchor,
                    tokens,
                    x.confidences.slice_mut(first..first + n),
                    x.ffn_gate.slice_mut(partials),
                    partial_ids,
                    n as u32,
                    vocab,
                    c.selector_rank,
                    c.selector_top_k,
                    control.repeat_penalty,
                    control.frequency_penalty,
                    control.presence_penalty,
                    control.repeat_last_n,
                    penalty_words,
                    control.penalty_count,
                    history_words,
                    control.history_count,
                )?,
                Selection::Sampled { temperature, .. } => k::dflash_selector_sampled(
                    s,
                    logits,
                    selector,
                    model.selector_predecessor.view(),
                    model.selector_successor.view(),
                    anchor,
                    tokens,
                    x.confidences.slice_mut(first..first + n),
                    x.ffn_gate.slice_mut(partials),
                    partial_ids,
                    x.uniforms.slice(first..first + n),
                    x.candidate_ids
                        .slice_mut(first * top_k..(first + n) * top_k),
                    x.candidate_probabilities
                        .slice_mut(first * top_k..(first + n) * top_k),
                    n as u32,
                    temperature,
                    vocab,
                    c.selector_rank,
                    c.selector_top_k,
                    control.repeat_penalty,
                    control.frequency_penalty,
                    control.presence_penalty,
                    control.repeat_last_n,
                    penalty_words,
                    control.penalty_count,
                    history_words,
                    control.history_count,
                )?,
            }
        }
        Ok(())
    })();
    result.map_err(launch)?;
    let top_k = c.selector_top_k as usize;
    let error = |e| format!("DFlash2 proposals: {e}");
    let mut out = Vec::with_capacity(blocks.len());
    let mut compact = 0;
    for (block, selection) in blocks.iter().zip(selections) {
        let n = block.count;
        let mut proposal = Proposal {
            tokens: vec![0u32; n],
            ..Proposal::default()
        };
        s.download(
            x.selector_tokens
                .slice(block.offset + 1..block.offset + 1 + n),
            &mut proposal.tokens,
        )
        .map_err(error)?;
        if matches!(selection, Selection::Sampled { .. }) {
            proposal.candidate_ids = vec![0u32; n * top_k];
            proposal.candidate_probabilities = vec![0f32; n * top_k];
            s.download(
                x.candidate_ids
                    .slice(compact * top_k..(compact + n) * top_k),
                &mut proposal.candidate_ids,
            )
            .map_err(error)?;
            s.download(
                x.candidate_probabilities
                    .slice(compact * top_k..(compact + n) * top_k),
                &mut proposal.candidate_probabilities,
            )
            .map_err(error)?;
        }
        compact += n;
        out.push(proposal);
    }
    Ok(out)
}

/// Penalty words `range` of the injection input, none when empty.
fn word_view(
    input: &DeviceBuffer<f32>,
    range: Range<usize>,
) -> Result<Option<DeviceSlice<'_, u32>>, LaunchError> {
    if range.is_empty() {
        return Ok(None);
    }
    input
        .slice(range)
        .cast::<u32>()
        .map(Some)
        .ok_or(LaunchError::InvalidArgument {
            param: "penalties",
            reason: "the injection input is not whole u32 values",
        })
}

/// Copies penalty words to the front of the injection input in stream order.
fn upload_penalties(s: &Stream, x: &mut DraftScratch, words: &[u32]) -> Result<(), String> {
    let target = x
        .target_features
        .slice_mut(0..words.len())
        .cast::<u32>()
        .ok_or("DFlash2 injection input is not whole u32 values")?;
    s.upload_blocking(words, target)
        .map_err(|e| format!("DFlash2 draft penalties: {e}"))
}

/// Which activation a grouped convolution reads.
#[derive(Clone, Copy)]
enum Which {
    Normed,
    Fused,
    Down,
}
