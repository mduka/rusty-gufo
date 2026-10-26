//! Device buffers of the executor and of each session: the scratch sized for
//! `max_batch` tokens (`Executor::Create`), the per-session recurrent state,
//! caches, rollback rows and the device control values
//! (`Executor::CreateSession`, `EnsureRollback`).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use gufo_hip::{
    DeviceBuffer, DeviceCopy, DeviceIndices, F16, GraphCache, HipError, IndexSpec, PinnedBuffer,
    Stream,
};
use gufo_qwen::vision::VisionInput;
use gufo_qwen38_flash_next_kernels as k;

use crate::borrowed::{Keep, Registry};
use crate::config::Config;
use crate::device::{DeviceModel, TAIL};
use crate::draft_vocab::DRAFT_EXTRA_ROWS;
use crate::ngram::NgramHistory;

/// Rows the vector kernels take at most; wider batches take the tiled GEMMs.
pub(crate) const VEC_BATCH: u32 = 8;
/// Verification rows the penalized argmax selects at once.
pub(crate) const PENALTY_ROWS: usize = 7;
/// Key-tile splits per row of a narrow attention batch.
pub(crate) const ATTN_SPLITS: u32 = 8;
/// The most queries per block-selection launch; the score scratch (at most
/// 5 Mi floats beyond VEC_BATCH full rows) bounds it at long contexts.
pub(crate) const SELECT_CHUNK: u32 = 512;
/// Most rollback rows a session keeps (`RollbackRows`).
pub(crate) const ROLLBACK_ROWS: u32 = 7;

/// A zeroed buffer of the scratch or of a session, exactly sized like the
/// C++ `Alloc`. Both live as long as the executor or a server slot. The
/// stream-ordered pool would return their blocks at every synchronisation
/// and map them again for the next session, and the kernels run at lower
/// clocks on its memory (pp512 2.2% slower).
pub(crate) fn alloc<T: DeviceCopy>(
    stream: &Stream,
    len: usize,
) -> Result<DeviceBuffer<T>, HipError> {
    let mut buffer = stream.alloc_persistent::<T>(len.max(1))?;
    stream.fill_bytes(buffer.as_mut_slice(), 0)?;
    Ok(buffer)
}

/// Zeroed indices with `spec`, placed like [`alloc`].
fn zeroed_indices(stream: &Stream, len: usize, spec: IndexSpec) -> Result<DeviceIndices, HipError> {
    stream.zeroed_indices_with(stream.alloc_persistent(len.max(1))?, spec)
}

/// Upper bound on launched routed tiles: every 16-padded bucket adds at most
/// one partial tile beyond its rows.
pub(crate) fn routed_tile_capacity(slots: usize, c: &Config) -> usize {
    (slots + c.num_experts as usize * 15) / 16 + c.num_experts as usize + 1
}

/// Selected blocks per row a narrow batch's attention compacts: the
/// selection budget.
pub(crate) fn attention_list_capacity(c: &Config) -> u32 {
    c.indexer_top_k.checked_div(c.compress_ratio).unwrap_or(0)
}

/// Raw indexer rows a session keeps in its power-of-two ring: before sparse
/// attention starts, pooling may lag by `indexer_top_k` rows.
pub(crate) fn indexer_capacity(c: &Config, batch: u32, context: u32) -> u32 {
    (u64::from(context).min(u64::from(c.indexer_top_k) + u64::from(batch)))
        .next_power_of_two()
        .min(u64::from(u32::MAX)) as u32
}

/// Floats of rollback row `row` of one linear layer's recurrent state: the
/// state after the first token, then keys, decays, betas and errors
/// (`GdnRollbackRowFloats`).
pub(crate) fn gdn_rollback_row_floats(row: u32, c: &Config) -> usize {
    let (k, v, d) = (
        c.ssm_num_k_heads as usize,
        c.ssm_num_v_heads as usize,
        c.ssm_head_dim as usize,
    );
    if row == 0 {
        v * d * d
    } else {
        k * d + v * (d + 2)
    }
}

/// Activation staging of the projections, owned by the executor: the F16
/// rows of the F16 and hipBLASLt routes, the Q8_1 rows of the decode GEMVs
/// (two slots, alternating), the tiled Q8 rows of the W8A8 route, and the
/// quantized tier's scratch. The `*_src` fields name what a staging buffer
/// holds, so a projection over the same rows skips its activation pass.
pub(crate) struct Stage {
    pub x_half: DeviceBuffer<F16>,
    pub x_q8: [DeviceBuffer<u8>; 2],
    pub q8_slot: usize,
    /// (device address, rows, cols) of the rows in each `x_q8` slot.
    pub q8_src: [Option<(usize, u32, u32)>; 2],
    pub x_q8t: DeviceBuffer<u8>,
    pub mmq: DeviceBuffer<u8>,
    /// (device address, rows, cols) of the rows in `x_q8t`.
    pub q8t_src: Option<(usize, u32, usize)>,
    /// (device address, rows, cols, bf16) of the rows in `x_half`.
    pub half_src: Option<(usize, u32, usize, bool)>,
}

/// A run of the phase scratch, in floats (one of the views `stage` hands
/// out in `Executor::Create`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Region {
    start: usize,
    end: usize,
}

impl Region {
    /// The region from `offset` floats in to its end.
    pub fn at(self, offset: usize) -> Range<usize> {
        let start = self.start + offset;
        assert!(start <= self.end, "scratch region offset out of range");
        start..self.end
    }

    /// `range` of the region, relative to its start.
    pub fn part(self, range: Range<usize>) -> Range<usize> {
        assert!(
            range.start <= range.end && self.start + range.end <= self.end,
            "scratch region range out of range"
        );
        self.start + range.start..self.start + range.end
    }
}

/// Consecutive regions from a restartable cursor.
struct Layout {
    cursor: usize,
    end: usize,
}

impl Layout {
    /// `len` floats from the cursor, 256-byte aligned like a separate
    /// allocation.
    fn take(&mut self, len: usize) -> Region {
        let start = self.cursor.next_multiple_of(64);
        let region = Region {
            start,
            end: start + len,
        };
        self.cursor = region.end;
        self.end = self.end.max(self.cursor);
        region
    }

    fn restart(&mut self) {
        self.cursor = 0;
    }
}

/// Scratch of one forward, sized for `max_batch` tokens (names follow
/// reference.cpp). Aliases of the C++ executor are the same buffer here:
/// PLE key/norm = `hc_gate`, value = `block_out`, query/conv = `xn`,
/// gated = `qkvz`; MTP h = `xn`, embedding = `mixed`, projected embedding =
/// `qkvz`, residual = `res`.
///
/// Linear attention, full attention and the routed experts run one after
/// another, and their intermediates die before the next stage, including
/// batched rows, MTP catch-up and the deferred MoE epilogue the next combine
/// consumes: they are regions of one `phase` buffer at stable addresses (for
/// graph replay) instead of reserving all three. The
/// mixer gates die before attention and the experts start; PLE also keys
/// with them, so `hc_gate` lies behind its gated output in `qkvz`.
pub(crate) struct Scratch {
    pub tokens: DeviceIndices,
    /// The linear, attention and expert stages' regions.
    pub phase: DeviceBuffer<f32>,
    pub res: DeviceBuffer<f32>,
    pub xn: DeviceBuffer<f32>,
    pub xn_half: DeviceBuffer<F16>,
    pub xn_q8t: DeviceBuffer<u8>,
    pub lo: DeviceBuffer<f32>,
    pub hc_gate: Region,
    pub mixed: DeviceBuffer<f32>,
    pub inject: DeviceBuffer<f32>,
    pub block_out: DeviceBuffer<f32>,
    /// [t][streams] stream weights of a deferred combine.
    pub hc_defer_w: DeviceBuffer<f32>,
    /// [t][qkv | z] of the stacked projection; the separate projections use
    /// qkv at 0 and z at `T * channels`.
    pub qkvz: Region,
    pub alpha_beta: Region,
    pub conv_scratch: Region,
    pub qn: Region,
    pub kn: Region,
    pub gdn_raw: Region,
    pub gdn_out: Region,
    pub qg: Region,
    pub q: Region,
    pub attn_gate: Region,
    pub k: Region,
    pub v: Region,
    pub iq: Region,
    pub ik: Region,
    /// u32 mask words.
    pub mask: Region,
    pub scores: DeviceBuffer<f32>,
    pub ctx: Region,
    pub attn_partials: DeviceBuffer<f32>,
    /// A narrow batch's selected blocks per row, compacted once for every
    /// head and split: [row][selection budget], and their counts.
    pub attn_lists: DeviceBuffer<u32>,
    pub attn_counts: DeviceBuffer<u32>,
    pub ple_emb: DeviceBuffer<f32>,
    pub ple_history_scratch: DeviceBuffer<f32>,
    pub router: DeviceBuffer<f32>,
    pub ids: DeviceIndices,
    pub expert_counts: DeviceIndices,
    pub routed_bounds: DeviceIndices,
    pub routed_cursors: DeviceBuffer<i32>,
    pub rows_token: DeviceIndices,
    pub rows_slot: DeviceIndices,
    pub routed_tiles: DeviceIndices,
    pub weights: DeviceBuffer<f32>,
    pub gate_e: Region,
    pub up_e: Region,
    pub down_e: Region,
    pub shexp_gate: DeviceBuffer<f32>,
    pub shexp_up: DeviceBuffer<f32>,
    pub shexp_out: DeviceBuffer<f32>,
    pub shexp_half: DeviceBuffer<F16>,
    pub logits: DeviceBuffer<f32>,
    /// The draft head's selections: argmax candidates (8 bytes each), the
    /// greedy token, and the top-64 IDs with their scores behind them.
    pub mtp_argmax: DeviceBuffer<u8>,
    pub mtp_token: DeviceBuffer<i32>,
    pub mtp_ids: DeviceBuffer<u32>,
    pub mtp_scratch_ids: DeviceBuffer<u32>,
}

/// Pinned host staging the forward reads and writes.
pub(crate) struct Host {
    pub tokens: PinnedBuffer<u32>,
    pub ple: Option<PinnedBuffer<f32>>,
    pub ple_rows: Vec<u32>,
    pub counts: PinnedBuffer<u32>,
    pub tiles: PinnedBuffer<u32>,
    /// Logit rows on their way to the caller (`max_logit_rows` of them).
    pub logits: PinnedBuffer<f32>,
    /// Per batch session: the draft head's greedy token and its candidates
    /// (IDs, then the scores' bits).
    pub draft_tokens: Vec<PinnedBuffer<i32>>,
    pub draft_candidates: Vec<PinnedBuffer<u32>>,
}

impl Scratch {
    /// `context_capacity` sizes the sparse-mask rows: the native context or
    /// a longer YaRN one.
    pub fn new(
        stream: &Stream,
        model: &DeviceModel,
        max_batch: u32,
        max_logit_rows: u32,
        context_capacity: u32,
    ) -> Result<(Self, Stage, Host, u32), HipError> {
        let c = &model.config;
        let t = max_batch as usize;
        let hc_dim = c.hc_dim() as usize;
        let hidden = c.hidden_size as usize;
        let slots = t * c.num_experts_used as usize;
        let experts = c.num_experts as usize;
        let mask_words = c.sparse_mask_words(context_capacity);
        let channels = c.ssm_conv_channels() as usize;
        let value_dim = c.ssm_value_dim() as usize;
        let key_dim = c.ssm_key_dim() as usize;
        let q_dim = c.attention_q_dim() as usize;
        let kv_dim = c.attention_kv_dim() as usize;
        let compact = k::routed_compact_rows(slots, experts);
        let tile_capacity = routed_tile_capacity(slots, c);
        let ple = c.ple_layer.is_some();
        let f32s = |n: usize| alloc::<f32>(stream, n);
        let signed = |bound: u32| IndexSpec::below(bound);
        let indices = |len: usize, spec: IndexSpec| zeroed_indices(stream, len, spec);
        // The candidate IDs, or the penalized argmax's partial candidates
        // (16 bytes, 64 per verification row).
        let workspace =
            (k::mtp_candidate_workspace(c.vocab_size) as usize).max(PENALTY_ROWS * 64 * 16 / 4);
        // MTP's projected embedding and the PLE gate reuse the stacked
        // projection's space.
        let projection = (channels + value_dim).max(hc_dim);
        let mut layout = Layout { cursor: 0, end: 0 };
        let qkvz = layout.take(t * projection);
        let alpha_beta = layout.take(t * 2 * c.ssm_num_v_heads as usize);
        let conv_scratch = layout.take((t + c.ssm_conv_kernel as usize) * channels);
        let qn = layout.take(t * key_dim);
        let kn = layout.take(t * key_dim);
        let gdn_raw = layout.take(t * value_dim);
        let gdn_out = layout.take(t * value_dim);
        layout.restart();
        layout.take(t * projection);
        let hc_gate = layout.take(t * hc_dim);
        layout.restart();
        let qg = layout.take(t * (2 * q_dim + 2 * kv_dim));
        let q = layout.take(t * q_dim);
        let attn_gate = layout.take(t * q_dim);
        let keys = layout.take(t * kv_dim);
        let values = layout.take(t * kv_dim);
        let iq = layout.take(t * (c.indexer_heads * c.indexer_head_dim) as usize);
        let ik = layout.take(t * c.indexer_head_dim as usize);
        let mask = layout.take(t * mask_words as usize);
        let ctx = layout.take(t * q_dim);
        layout.restart();
        let gate_e = layout.take(slots * c.expert_ff as usize);
        let up_e = layout.take(slots * c.expert_ff as usize);
        let down_e = layout.take(slots * hidden);
        let scratch = Self {
            tokens: indices(t, signed(c.vocab_size))?,
            phase: f32s(layout.end)?,
            res: f32s(t * hc_dim)?,
            xn: f32s(t * hc_dim)?,
            xn_half: alloc::<F16>(stream, t * hc_dim)?,
            xn_q8t: alloc::<u8>(stream, k::q8_tiled_bytes(t, hc_dim))?,
            lo: f32s(t * c.hc_low_rank as usize)?,
            hc_gate,
            mixed: f32s(t * hidden)?,
            inject: f32s(t * c.hc_count as usize * k::inject_parts(c.hidden_size) as usize)?,
            block_out: f32s(t * hidden)?,
            hc_defer_w: f32s(t * c.hc_count as usize)?,
            qkvz,
            alpha_beta,
            conv_scratch,
            qn,
            kn,
            gdn_raw,
            gdn_out,
            qg,
            q,
            attn_gate,
            k: keys,
            v: values,
            iq,
            ik,
            mask,
            // Score traffic near the device cache size; captured
            // verification still needs all its rows at the maximum context
            //.
            scores: f32s({
                let stride = mask_words as usize * 32;
                (VEC_BATCH as usize * stride).max((5 << 20).min(SELECT_CHUNK as usize * stride))
            })?,
            ctx,
            attn_partials: f32s(
                VEC_BATCH as usize
                    * c.num_heads as usize
                    * ATTN_SPLITS as usize
                    * (c.head_dim as usize + 2),
            )?,
            attn_lists: alloc::<u32>(
                stream,
                VEC_BATCH as usize * attention_list_capacity(c) as usize,
            )?,
            attn_counts: alloc::<u32>(stream, VEC_BATCH as usize)?,
            ple_emb: f32s(if ple {
                t * c.ple_embedding_dim() as usize
            } else {
                1
            })?,
            ple_history_scratch: f32s(if ple {
                c.ple_conv_history() as usize * hc_dim
            } else {
                1
            })?,
            router: f32s(t * (experts + 1))?,
            ids: indices(slots, signed(c.num_experts).padded())?,
            expert_counts: indices(experts, IndexSpec::below(slots as u32 + 1))?,
            routed_bounds: indices(experts + 1, signed(compact as u32 + 1))?,
            routed_cursors: alloc::<i32>(stream, experts)?,
            rows_token: indices(compact, IndexSpec::below(max_batch).padded())?,
            rows_slot: indices(compact, IndexSpec::below(slots as u32).padded())?,
            routed_tiles: indices(
                3 * tile_capacity,
                IndexSpec::below(c.num_experts).masked(0xFFFF).signed(),
            )?,
            weights: f32s(slots)?,
            gate_e,
            up_e,
            down_e,
            shexp_gate: f32s(t * c.shared_expert_ff as usize)?,
            shexp_up: f32s(t * c.shared_expert_ff as usize)?,
            shexp_out: f32s(t * hidden)?,
            shexp_half: alloc::<F16>(stream, t * c.shared_expert_ff as usize)?,
            logits: f32s(max_logit_rows as usize * c.vocab_size as usize)?,
            mtp_argmax: alloc::<u8>(stream, 64 * 8)?,
            mtp_token: alloc::<i32>(stream, 1)?,
            mtp_ids: alloc::<u32>(stream, workspace)?,
            mtp_scratch_ids: alloc::<u32>(stream, workspace)?,
        };
        let used = c.num_experts_used as usize;
        let mmq = k::moe_tiled_scratch(t, hidden, experts, used)
            .max(k::moe_tiled_scratch(
                slots,
                c.expert_ff as usize,
                experts,
                1,
            ))
            .max(k::moe_vec_scratch(slots, hidden.max(c.expert_ff as usize)))
            .max(k::moe_gated_vec_scratch(t.min(64), hidden, used));
        let stage = Stage {
            x_half: alloc::<F16>(stream, t * model.max_half_cols)?,
            x_q8: [
                alloc::<u8>(stream, k::q8_1_bytes(VEC_BATCH as usize, model.max_q8_cols))?,
                alloc::<u8>(stream, k::q8_1_bytes(VEC_BATCH as usize, model.max_q8_cols))?,
            ],
            q8_slot: 0,
            q8_src: [None; 2],
            x_q8t: alloc::<u8>(stream, k::q8_tiled_bytes(t, model.max_q8_cols))?,
            mmq: alloc::<u8>(stream, mmq)?,
            q8t_src: None,
            half_src: None,
        };
        let host = Host {
            tokens: PinnedBuffer::zeroed(t)?,
            ple: if ple {
                Some(PinnedBuffer::zeroed(t * c.ple_embedding_dim() as usize)?)
            } else {
                None
            },
            ple_rows: vec![0; if ple { t * c.ple_heads as usize } else { 0 }],
            counts: PinnedBuffer::zeroed(experts)?,
            tiles: PinnedBuffer::zeroed(3 * tile_capacity)?,
            logits: PinnedBuffer::zeroed(max_logit_rows as usize * c.vocab_size as usize)?,
            draft_tokens: Vec::new(),
            draft_candidates: Vec::new(),
        };
        let mut host = host;
        if model.mtp.is_some() {
            let count = (c.vocab_size as usize).min(crate::mtp::MTP_CANDIDATES);
            for _ in 0..crate::batch::BATCH_SESSIONS {
                host.draft_tokens.push(PinnedBuffer::zeroed(1)?);
                host.draft_candidates
                    .push(PinnedBuffer::zeroed(crate::mtp::MTP_CANDIDATES + count)?);
            }
        }
        Ok((scratch, stage, host, mask_words))
    }
}

/// One linear layer's recurrent state.
pub(crate) struct LinearState {
    /// [kernel - 1][channels]
    pub conv_state: DeviceBuffer<f32>,
    /// [v_heads][d][d]
    pub state: DeviceBuffer<f32>,
}

/// One attention layer's caches.
pub(crate) struct AttentionState {
    /// [max_context][kv_heads * d] each.
    pub k_cache: DeviceBuffer<F16>,
    pub v_cache: DeviceBuffer<F16>,
    /// [index_capacity][indexer_dim] raw ring.
    pub index_k: DeviceBuffer<f32>,
    /// [max_context / ratio + 1][indexer_dim]
    pub block_k: DeviceBuffer<F16>,
}

/// The draft block's context.
pub(crate) struct MtpState {
    pub attention: AttentionState,
    pub blocks: u32,
    /// [hc_dim] wide residual handed to the next draft.
    pub h: DeviceBuffer<f32>,
    /// The last `max_speculative` trunk rows.
    pub target_hidden: DeviceBuffer<f32>,
    pub position: u32,
    /// Whether `h` is the residual of the draft at `position`: a K/V-only
    /// catch-up or a rewind leaves none, so the next draft needs a known
    /// trunk row.
    pub residual_valid: bool,
    /// The request's draft table, when the model drafts over a vocabulary
    /// subset.
    pub draft: Option<DraftTable>,
}

/// A session's MTP draft table: [`DRAFT_EXTRA_ROWS`]
/// output rows and their token IDs, padding for an empty slot. Not part of
/// snapshots: it rebuilds from the committed tokens.
pub(crate) struct DraftTable {
    /// The rows, then the GEMV tail.
    pub rows: DeviceBuffer<u8>,
    pub ids: DeviceIndices,
    /// The IDs as uploaded; a change uploads them whole.
    pub host: PinnedBuffer<u32>,
}

/// One device control value and the pinned host word it is uploaded from,
/// so uploads of different values never wait for each other.
struct Slot {
    indices: DeviceIndices,
    host: PinnedBuffer<u32>,
    /// The value the last enqueued upload writes: only these uploads write
    /// the device word, in stream order, so an equal value needs none.
    uploaded: Option<u32>,
}

impl Slot {
    fn new(stream: &Stream, spec: IndexSpec) -> Result<Self, HipError> {
        Ok(Self {
            indices: zeroed_indices(stream, 1, spec)?,
            host: PinnedBuffer::zeroed(1)?,
            uploaded: None,
        })
    }

    /// Checks `value` against the slot's spec and enqueues its upload.
    fn set(&mut self, stream: &Stream, value: u32) -> Result<(), String> {
        if self.uploaded == Some(value) {
            return Ok(());
        }
        self.uploaded = None;
        let upload = |slot: &mut Self| -> Result<(), HipError> {
            slot.host.host_mut()?[0] = value;
            stream.upload_indices_async(&mut slot.host, 1, &mut slot.indices)
        };
        upload(self).map_err(|e| format!("control upload: {e}"))?;
        self.uploaded = Some(value);
        Ok(())
    }
}

/// Positions and block counts the kernels read from device memory, so a
/// captured graph replays at any position. A position read by a launch of
/// `n` rows must lie below `max_context - n + 1`: each row count gets its
/// own buffer with that bound.
pub(crate) struct Control {
    max_context: u32,
    positions: HashMap<u32, Slot>,
    mtp_positions: HashMap<u32, Slot>,
    blocks: Slot,
    mtp_blocks: Slot,
    /// Kept trunk row the draft reads (-1: its own carried residual), by
    /// (rows, rows of the hidden source).
    hidden_rows: HashMap<(u32, u32), Slot>,
}

impl Control {
    fn new(stream: &Stream, c: &Config, max_context: u32) -> Result<Self, HipError> {
        let blocks = IndexSpec::below(max_context / c.compress_ratio + 1);
        Ok(Self {
            max_context,
            positions: HashMap::new(),
            mtp_positions: HashMap::new(),
            blocks: Slot::new(stream, blocks)?,
            mtp_blocks: Slot::new(stream, blocks)?,
            hidden_rows: HashMap::new(),
        })
    }

    fn slot<'m, K: std::hash::Hash + Eq + Copy>(
        stream: &Stream,
        map: &'m mut HashMap<K, Slot>,
        key: K,
        spec: IndexSpec,
    ) -> Result<&'m mut Slot, String> {
        if let std::collections::hash_map::Entry::Vacant(e) = map.entry(key) {
            let slot = Slot::new(stream, spec).map_err(|e| format!("control allocation: {e}"))?;
            e.insert(slot);
        }
        Ok(map.get_mut(&key).expect("inserted above"))
    }

    /// Uploads the first trunk position of an `n`-row launch.
    pub fn set_position(&mut self, stream: &Stream, n: u32, value: u32) -> Result<(), String> {
        let spec = IndexSpec::below(self.position_bound(n)?);
        Self::slot(stream, &mut self.positions, n, spec)?.set(stream, value)
    }

    pub fn set_mtp_position(&mut self, stream: &Stream, n: u32, value: u32) -> Result<(), String> {
        let spec = IndexSpec::below(self.position_bound(n)?);
        Self::slot(stream, &mut self.mtp_positions, n, spec)?.set(stream, value)
    }

    pub fn set_blocks(&mut self, stream: &Stream, value: u32) -> Result<(), String> {
        self.blocks.set(stream, value)
    }

    pub fn set_mtp_blocks(&mut self, stream: &Stream, value: u32) -> Result<(), String> {
        self.mtp_blocks.set(stream, value)
    }

    /// Uploads the kept trunk row of an `n`-row draft launch over a source
    /// of `source_rows` rows; -1 names the carried residual.
    pub fn set_hidden_row(
        &mut self,
        stream: &Stream,
        n: u32,
        source_rows: u32,
        value: i32,
    ) -> Result<(), String> {
        let bound = source_rows
            .checked_sub(n)
            .map(|b| b + 1)
            .ok_or("draft rows exceed the hidden source")?;
        let spec = IndexSpec::below(bound).padded();
        Self::slot(stream, &mut self.hidden_rows, (n, source_rows), spec)?.set(stream, value as u32)
    }

    pub fn position(&self, n: u32) -> Option<&DeviceIndices> {
        self.positions.get(&n).map(|s| &s.indices)
    }

    pub fn mtp_position(&self, n: u32) -> Option<&DeviceIndices> {
        self.mtp_positions.get(&n).map(|s| &s.indices)
    }

    pub fn blocks(&self) -> &DeviceIndices {
        &self.blocks.indices
    }

    pub fn mtp_blocks(&self) -> &DeviceIndices {
        &self.mtp_blocks.indices
    }

    pub fn hidden_row(&self, n: u32, source_rows: u32) -> Option<&DeviceIndices> {
        self.hidden_rows.get(&(n, source_rows)).map(|s| &s.indices)
    }

    fn position_bound(&self, n: u32) -> Result<u32, String> {
        self.max_context
            .checked_sub(n)
            .map(|b| b + 1)
            .ok_or_else(|| "batch exceeds the session context".into())
    }
}

/// One rollback row: every linear layer's conv history and recurrent-state
/// part for one proper prefix, then the PLE history.
pub(crate) struct RollbackRow {
    pub buffer: DeviceBuffer<f32>,
    /// (conv offset, state offset) per linear layer, by layer index.
    pub layers: Vec<Option<(usize, usize)>>,
    pub ple: Option<usize>,
}

/// Per-sequence device state: recurrent state, KV and indexer caches, PLE
/// history, the host n-gram window, and for a speculative batch the
/// rollback rows of every recurrent buffer.
pub struct Session {
    pub(crate) owner: u64,
    pub(crate) mtp_enabled: bool,
    pub(crate) max_context: u32,
    pub(crate) index_capacity: u32,
    pub(crate) position: u32,
    pub(crate) linear: Vec<Option<LinearState>>,
    pub(crate) attention: Vec<Option<AttentionState>>,
    pub(crate) ple_history: Option<DeviceBuffer<f32>>,
    pub(crate) ngram: NgramHistory,
    pub(crate) ngram_snapshots: Vec<NgramHistory>,
    pub(crate) spec_base: u32,
    pub(crate) spec_tokens: u32,
    pub(crate) mtp: Option<MtpState>,
    pub(crate) control: Control,
    /// Host shadow of the pooled indexer blocks.
    pub(crate) blocks: u32,
    pub(crate) graphs: GraphCache<u64>,
    pub(crate) rollback: Vec<RollbackRow>,
    pub(crate) mutation_epoch: u64,
    pub(crate) cancelled: bool,
    cancel: Option<Cancel>,
    pub(crate) allocated_bytes: usize,
    /// Image positions and embeddings of the current request.
    pub(crate) vision: VisionInput,
    pub(crate) compress_ratio: u32,
    /// The snapshots that borrow this session's rows, from the first on.
    pub(crate) snapshots: OnceLock<Arc<Registry>>,
}

/// Whether a session's request was cancelled; asked at forward, layer and
/// snapshot-copy boundaries.
pub type Cancel = Arc<dyn Fn() -> bool + Send + Sync>;

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        stream: &Stream,
        owner: u64,
        c: &Config,
        max_context: u32,
        max_batch: u32,
        max_speculative: u32,
        mtp_enabled: bool,
        draft_row_bytes: Option<usize>,
    ) -> Result<Self, HipError> {
        let index_capacity = indexer_capacity(c, max_batch, max_context);
        let channels = c.ssm_conv_channels() as usize;
        let conv = (c.ssm_conv_kernel as usize - 1) * channels;
        let state = c.ssm_num_v_heads as usize * (c.ssm_head_dim as usize).pow(2);
        let kv_row = c.attention_kv_dim() as usize;
        let indexer_dim = c.indexer_head_dim as usize;
        let blocks = (max_context / c.compress_ratio + 1) as usize;
        let mut bytes = 0usize;
        let mut count = |n: usize, width: usize| {
            bytes += n.max(1) * width;
        };
        let attention_state =
            |count: &mut dyn FnMut(usize, usize)| -> Result<AttentionState, HipError> {
                count(max_context as usize * kv_row, 2);
                count(max_context as usize * kv_row, 2);
                count(index_capacity as usize * indexer_dim, 4);
                count(blocks * indexer_dim, 2);
                Ok(AttentionState {
                    k_cache: alloc::<F16>(stream, max_context as usize * kv_row)?,
                    v_cache: alloc::<F16>(stream, max_context as usize * kv_row)?,
                    index_k: alloc::<f32>(stream, index_capacity as usize * indexer_dim)?,
                    block_k: alloc::<F16>(stream, blocks * indexer_dim)?,
                })
            };
        let mut linear = Vec::with_capacity(c.num_layers as usize);
        let mut attention = Vec::with_capacity(c.num_layers as usize);
        for il in 0..c.num_layers {
            if c.is_linear_layer(il) {
                count(conv, 4);
                count(state, 4);
                linear.push(Some(LinearState {
                    conv_state: alloc::<f32>(stream, conv)?,
                    state: alloc::<f32>(stream, state)?,
                }));
                attention.push(None);
            } else {
                linear.push(None);
                attention.push(Some(attention_state(&mut count)?));
            }
        }
        let ple_history = match c.ple_layer {
            Some(_) => {
                let hist = c.ple_conv_history() as usize * c.hc_dim() as usize;
                count(hist, 4);
                Some(alloc::<f32>(stream, hist)?)
            }
            None => None,
        };
        // The C++ control block.
        count(5, 4);
        let mtp = if mtp_enabled {
            let hc_dim = c.hc_dim() as usize;
            count(max_speculative as usize * hc_dim, 4);
            let target_hidden = alloc::<f32>(stream, max_speculative as usize * hc_dim)?;
            let attention = attention_state(&mut count)?;
            count(hc_dim, 4);
            let draft = match draft_row_bytes {
                Some(row_bytes) => {
                    let rows = DRAFT_EXTRA_ROWS as usize;
                    count(rows * row_bytes + TAIL, 1);
                    count(rows, 4);
                    let ids = zeroed_indices(
                        stream,
                        rows,
                        IndexSpec::below(c.vocab_size).padded().signed(),
                    )?;
                    Some(DraftTable {
                        rows: alloc::<u8>(stream, rows * row_bytes + TAIL)?,
                        ids,
                        host: PinnedBuffer::zeroed(rows)?,
                    })
                }
                None => None,
            };
            Some(MtpState {
                attention,
                blocks: 0,
                h: alloc::<f32>(stream, hc_dim)?,
                target_hidden,
                position: 0,
                residual_valid: false,
                draft,
            })
        } else {
            None
        };
        // The zero fills above ran on this stream; nothing may read the new
        // buffers before they land.
        stream.synchronize()?;
        Ok(Self {
            owner,
            mtp_enabled,
            max_context,
            index_capacity,
            position: 0,
            linear,
            attention,
            ple_history,
            ngram: NgramHistory::default(),
            ngram_snapshots: Vec::new(),
            spec_base: 0,
            spec_tokens: 0,
            mtp,
            control: Control::new(stream, c, max_context)?,
            blocks: 0,
            graphs: GraphCache::new(),
            rollback: Vec::new(),
            mutation_epoch: 0,
            cancelled: false,
            cancel: None,
            allocated_bytes: bytes,
            vision: VisionInput::default(),
            compress_ratio: c.compress_ratio,
            snapshots: OnceLock::new(),
        })
    }

    /// Copies out every borrowed row that a write keeping only `position`
    /// trunk and `mtp_position` draft rows would overwrite
    /// (`PreserveSnapshots`).
    pub(crate) fn preserve(&self, position: u32, mtp_position: u32) -> Result<(), String> {
        let Some(registry) = self.snapshots.get() else {
            return Ok(());
        };
        let ratio = self.compress_ratio.max(1);
        registry.preserve(Keep {
            position,
            mtp_position,
            blocks: self.blocks.min(position / ratio),
            mtp_blocks: self
                .mtp
                .as_ref()
                .map_or(0, |m| m.blocks.min(mtp_position / ratio)),
        })
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn max_context(&self) -> u32 {
        self.max_context
    }

    pub fn mtp_enabled(&self) -> bool {
        self.mtp_enabled
    }

    pub fn mutation_epoch(&self) -> u64 {
        self.mutation_epoch
    }

    /// Replaces the cancellation check and clears an earlier cancellation
    /// (`SetCancellationCheck`).
    pub fn set_cancellation_check(&mut self, check: Option<Cancel>) {
        self.cancelled = false;
        self.cancel = check;
    }

    /// Encodes the images among the next `n` tokens before a forward
    /// injects them (`VisionInput::prepare`), outside any graph capture.
    pub(crate) fn prepare_vision(&mut self, n: u32) -> Result<(), String> {
        let cancel = self.cancel.clone();
        let check = cancel.as_deref().map(|c| c as &dyn Fn() -> bool);
        self.vision
            .prepare(self.position, n, check)
            .map_err(|e| e.message().to_string())
    }

    /// Asks the check once more unless cancelled already
    /// (`CheckCancellation`); a cancelled session stays cancelled.
    pub fn check_cancelled(&mut self) -> bool {
        if !self.cancelled
            && let Some(check) = &self.cancel
        {
            self.cancelled = check();
        }
        self.cancelled
    }

    /// Whether a check reported cancellation (`Cancelled`).
    pub fn cancelled(&self) -> bool {
        self.cancelled
    }

    /// Whether the request is cancelled, asking the check without recording
    /// it (for read-only callers such as a snapshot capture).
    pub fn cancellation_requested(&self) -> bool {
        self.cancelled || self.cancel.as_ref().is_some_and(|check| check())
    }

    /// Asks the check without recording the answer, for readers of an idle
    /// session.
    pub(crate) fn peek_cancelled(&self) -> bool {
        self.cancelled || self.cancel.as_ref().is_some_and(|check| check())
    }

    /// Device bytes of the state, rollback rows included.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
            + self
                .rollback
                .iter()
                .map(|r| r.buffer.len() * 4)
                .sum::<usize>()
    }

    /// Drops every token; the next forward starts at position 0. Borrowed
    /// rows stay until that forward would overwrite them.
    pub fn reset(&mut self, stream: &Stream) -> Result<(), HipError> {
        self.trim_rollback(0);
        self.position = 0;
        self.spec_tokens = 0;
        self.ngram.reset();
        if let Some(mtp) = &mut self.mtp {
            mtp.position = 0;
            mtp.residual_valid = false;
            mtp.blocks = 0;
        }
        for l in self.linear.iter_mut().flatten() {
            stream.fill_bytes(l.conv_state.as_mut_slice(), 0)?;
            stream.fill_bytes(l.state.as_mut_slice(), 0)?;
        }
        self.blocks = 0;
        if let Some(history) = &mut self.ple_history {
            stream.fill_bytes(history.as_mut_slice(), 0)?;
        }
        Ok(())
    }

    /// Releases rollback rows from `depth` on; graphs that captured one are
    /// captured again (the graph cache sees the freed buffers).
    pub(crate) fn trim_rollback(&mut self, depth: u32) {
        self.rollback.truncate(depth as usize);
        self.ngram_snapshots.truncate(depth as usize);
    }

    /// Allocates rollback rows up to `depth` (`EnsureRollback`).
    pub(crate) fn ensure_rollback(
        &mut self,
        stream: &Stream,
        c: &Config,
        depth: u32,
        max_speculative: u32,
    ) -> Result<(), String> {
        if depth as usize <= self.rollback.len() {
            return Ok(());
        }
        if depth >= max_speculative || depth > ROLLBACK_ROWS || self.spec_tokens != 0 {
            return Err("invalid rollback allocation depth".into());
        }
        let conv = (c.ssm_conv_kernel as usize - 1) * c.ssm_conv_channels() as usize;
        let ple = match c.ple_layer {
            Some(_) => c.ple_conv_history() as usize * c.hc_dim() as usize,
            None => 0,
        };
        self.ngram_snapshots
            .resize(depth as usize, NgramHistory::default());
        for row in self.rollback.len() as u32..depth {
            let state = gdn_rollback_row_floats(row, c);
            let mut layers = Vec::with_capacity(c.num_layers as usize);
            let mut next = 0usize;
            for il in 0..c.num_layers {
                if c.is_linear_layer(il) {
                    layers.push(Some((next, next + conv)));
                    next += conv + state;
                } else {
                    layers.push(None);
                }
            }
            let ple_offset = (ple != 0).then_some(next);
            // Exactly sized, as C++'s `AllocateDevice`: every verification
            // writes these rows, and kernels run slower on pool memory.
            let buffer = stream
                .alloc_persistent::<f32>((next + ple).max(1))
                .map_err(|e| format!("rollback allocation: {e}"))?;
            self.rollback.push(RollbackRow {
                buffer,
                layers,
                ple: ple_offset,
            });
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Snapshots may outlive the session: copy out the rows they borrow,
        // so its buffers are freed. A failure is recorded in the registry,
        // whose snapshots then refuse their rows.
        if let Some(registry) = self.snapshots.get() {
            let _ = registry.preserve(Keep::default());
        }
    }
}
