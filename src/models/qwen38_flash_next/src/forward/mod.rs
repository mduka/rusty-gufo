//! The layers of one forward, a port of the per-layer functions of
//! `executor.cpp`: hyper-connection mixers and combines, the PLE injection,
//! Gated DeltaNet, sparse attention and the mixture of experts. A pass
//! reads and writes the executor's scratch at a row offset (`RowScratch`),
//! so batched and draft forwards run the same code on their rows.

use gufo_formats::GgmlType;
use gufo_hip::{
    DeviceBuffer, DeviceCopy, DeviceSlice, DeviceSliceMut, DeviceTensor, F16, IndexSlice,
    QuantSlice, RowTable,
};
use gufo_qwen38_flash_next_kernels as k;

use crate::checkpoint::CheckpointStorage;
use crate::config::Config;
use crate::device::{Attention, DraftHead, Linear, Matrix, Mixer, Moe, Ple};
use crate::draft_vocab::DRAFT_EXTRA_ROWS;
use crate::executor::PlePrefetch;
use crate::ngram::NgramTable;
use crate::routes::{Ctx, Error, Q8Input, Result, launch_error, launched, routed_tile_rows};
use crate::state::{
    ATTN_SPLITS, AttentionState, DraftTable, Host, LinearState, RollbackRow, SELECT_CHUNK, Scratch,
    VEC_BATCH,
};

mod attention;
mod hyper;
mod linear;
mod moe;
mod ple;

pub(crate) use attention::*;
pub(crate) use ple::*;

pub(crate) fn at<T: DeviceCopy>(b: &DeviceBuffer<T>, offset: usize) -> DeviceSlice<'_, T> {
    b.slice(offset..b.len())
}

pub(crate) fn at_mut<T: DeviceCopy>(
    b: &mut DeviceBuffer<T>,
    offset: usize,
) -> DeviceSliceMut<'_, T> {
    let len = b.len();
    b.slice_mut(offset..len)
}

fn half_mut(view: DeviceSliceMut<'_, f32>) -> Result<DeviceSliceMut<'_, F16>> {
    Ok(view.cast::<F16>().ok_or("misaligned F16 view")?)
}

/// The selection mask words of a phase region.
fn words(view: DeviceSlice<'_, f32>) -> Result<DeviceSlice<'_, u32>> {
    Ok(view.cast::<u32>().ok_or("misaligned mask view")?)
}

fn words_mut(view: DeviceSliceMut<'_, f32>) -> Result<DeviceSliceMut<'_, u32>> {
    Ok(view.cast::<u32>().ok_or("misaligned mask view")?)
}

/// Row widths of the scratch buffers, for row offsets.
#[derive(Clone, Copy)]
pub(crate) struct Widths {
    pub hc: usize,
    pub lo: usize,
    pub hidden: usize,
    /// Scalar inject partial sums per row (`RowScratch`).
    pub inject: usize,
    pub channels: usize,
    pub value: usize,
    pub qkvz: usize,
    pub alpha_beta: usize,
    pub key: usize,
    pub qg: usize,
    pub q: usize,
    pub kv: usize,
    pub iq: usize,
    pub ik: usize,
    pub mask: usize,
    pub router: usize,
    pub used: usize,
    pub expert_ff: usize,
    pub shared_ff: usize,
    pub ple: usize,
    pub max_batch: usize,
}

impl Widths {
    pub fn new(c: &Config, mask_words: u32, max_batch: u32) -> Self {
        Self {
            hc: c.hc_dim() as usize,
            lo: c.hc_low_rank as usize,
            hidden: c.hidden_size as usize,
            inject: (c.hc_count * k::inject_parts(c.hidden_size)) as usize,
            channels: c.ssm_conv_channels() as usize,
            value: c.ssm_value_dim() as usize,
            qkvz: (c.ssm_conv_channels() + c.ssm_value_dim()) as usize,
            alpha_beta: 2 * c.ssm_num_v_heads as usize,
            key: c.ssm_key_dim() as usize,
            qg: (2 * c.attention_q_dim() + 2 * c.attention_kv_dim()) as usize,
            q: c.attention_q_dim() as usize,
            kv: c.attention_kv_dim() as usize,
            iq: (c.indexer_heads * c.indexer_head_dim) as usize,
            ik: c.indexer_head_dim as usize,
            mask: mask_words as usize,
            router: c.num_experts as usize + 1,
            used: c.num_experts_used as usize,
            expert_ff: c.expert_ff as usize,
            shared_ff: c.shared_expert_ff as usize,
            ple: c.ple_embedding_dim() as usize,
            max_batch: max_batch as usize,
        }
    }
}

/// One forward's view of the executor.
pub(crate) struct Pass<'e> {
    pub ctx: Ctx<'e>,
    pub s: &'e mut Scratch,
    pub host: &'e mut Host,
    pub ngram: Option<&'e NgramTable>,
    pub ple_pending: &'e mut bool,
    pub prefetch: &'e mut PlePrefetch,
    pub w: Widths,
    /// Row offset into the scratch (`RowScratch`).
    pub row: usize,
    /// Element offset of the inject partial sums when it differs from the
    /// scalar layout (the draft block's tail keeps its producer's stride).
    pub inject_at: Option<usize>,
    pub mask_words: u32,
    pub wide_mixer: bool,
    /// A prefill checkpoint the pass captures on its way.
    pub checkpoint: Option<Checkpoint<'e>>,
}

/// The checkpoint after the first `tokens` rows of a prefill pass and the
/// storage its mutable state goes to (upstream `PrefillCheckpoint`).
pub(crate) struct Checkpoint<'e> {
    pub tokens: u32,
    pub storage: &'e mut CheckpointStorage,
}

impl Pass<'_> {
    fn c(&self) -> &Config {
        self.ctx.config()
    }

    fn inject_offset(&self) -> usize {
        self.inject_at.unwrap_or(self.row * self.w.inject)
    }
}
