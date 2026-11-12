//! The checkpoint a prefill pass captures on its way (upstream
//! `PrefillCheckpoint`): the frontier a pass ending after its
//! first `tokens` rows would leave, taken inside the final pass instead of
//! splitting it there. The pass writes the mutable state (each linear
//! layer's convolution history and recurrent state, the PLE history, the
//! draft block's kept trunk rows and the boundary logits) into executor
//! storage; the append-only caches below the boundary stay in the live
//! session, which the rest of the pass only appends to. A snapshot of both
//! equals one of a pass ending at the boundary.

use gufo_hip::{DeviceBuffer, DeviceSlice, DeviceSliceMut, HipError, Stream};

use crate::config::Config;
use crate::ngram::{NgramHistory, hash_ngram_rows};
use crate::state::{Session, VEC_BATCH};

/// Element offsets of the regions of [`CheckpointStorage`]: the payload
/// order of a snapshot's mutable sections, then the kept trunk rows and the
/// boundary logits.
pub(crate) struct Layout {
    /// Per layer, the convolution history's offset (the recurrent state
    /// follows it) for linear layers.
    linear: Vec<Option<usize>>,
    conv: usize,
    state: usize,
    ple: Option<usize>,
    ple_len: usize,
    hidden: usize,
    hidden_len: usize,
    logits: usize,
    vocab: usize,
    len: usize,
}

impl Layout {
    fn new(c: &Config, max_speculative: u32, has_mtp: bool) -> Self {
        let conv = ((c.ssm_conv_kernel - 1) * c.ssm_conv_channels()) as usize;
        let state = (c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim) as usize;
        let mut at = 0;
        let linear = (0..c.num_layers)
            .map(|il| {
                c.is_linear_layer(il).then(|| {
                    let offset = at;
                    at += conv + state;
                    offset
                })
            })
            .collect();
        let ple_len = if c.ple_layer.is_some() {
            (c.ple_conv_history() * c.hc_dim()) as usize
        } else {
            0
        };
        let ple = c.ple_layer.is_some().then_some(at);
        at += ple_len;
        let hidden = at;
        let hidden_len = if has_mtp {
            (max_speculative * c.hc_dim()) as usize
        } else {
            0
        };
        at += hidden_len;
        let logits = at;
        let vocab = c.vocab_size as usize;
        Self {
            linear,
            conv,
            state,
            ple,
            ple_len,
            hidden,
            hidden_len,
            logits,
            vocab,
            len: logits + vocab,
        }
    }
}

/// Executor storage a checkpointed pass writes; shared scratch, consumed
/// before the next forward.
pub(crate) struct CheckpointStorage {
    buffer: DeviceBuffer<f32>,
    layout: Layout,
}

impl CheckpointStorage {
    pub fn new(
        stream: &Stream,
        c: &Config,
        max_speculative: u32,
        has_mtp: bool,
    ) -> Result<Self, HipError> {
        let layout = Layout::new(c, max_speculative, has_mtp);
        Ok(Self {
            buffer: stream.alloc_persistent::<f32>(layout.len)?,
            layout,
        })
    }

    /// Device bytes the storage occupies once allocated.
    pub fn bytes(c: &Config, max_speculative: u32, has_mtp: bool) -> usize {
        Layout::new(c, max_speculative, has_mtp).len * 4
    }

    /// Bytes of a snapshot's mutable prefix: each linear layer's history and
    /// state, and the PLE history.
    pub fn mutable_bytes(c: &Config) -> usize {
        Layout::new(c, 0, false).hidden * 4
    }

    fn range(&self, at: usize, len: usize) -> std::ops::Range<usize> {
        at..at + len
    }

    /// Layer `layer`'s convolution history and recurrent state.
    pub fn linear_mut(
        &mut self,
        layer: usize,
    ) -> Option<(DeviceSliceMut<'_, f32>, DeviceSliceMut<'_, f32>)> {
        let at = self.layout.linear.get(layer).copied().flatten()?;
        let (conv, state) = (self.layout.conv, self.layout.state);
        let (history, rest) = self.buffer.slice_mut(at..at + conv + state).split_at(conv);
        Some((history, rest))
    }

    pub fn linear(&self, layer: usize) -> Option<(DeviceSlice<'_, f32>, DeviceSlice<'_, f32>)> {
        let at = self.layout.linear.get(layer).copied().flatten()?;
        let (conv, state) = (self.layout.conv, self.layout.state);
        Some((
            self.buffer.slice(self.range(at, conv)),
            self.buffer.slice(self.range(at + conv, state)),
        ))
    }

    pub fn ple_mut(&mut self) -> Option<DeviceSliceMut<'_, f32>> {
        let at = self.layout.ple?;
        Some(self.buffer.slice_mut(at..at + self.layout.ple_len))
    }

    pub fn ple(&self) -> Option<DeviceSlice<'_, f32>> {
        let at = self.layout.ple?;
        Some(self.buffer.slice(self.range(at, self.layout.ple_len)))
    }

    pub fn hidden_mut(&mut self) -> DeviceSliceMut<'_, f32> {
        let at = self.layout.hidden;
        self.buffer.slice_mut(at..at + self.layout.hidden_len)
    }

    pub fn hidden(&self) -> DeviceSlice<'_, f32> {
        self.buffer
            .slice(self.range(self.layout.hidden, self.layout.hidden_len))
    }

    pub fn logits_mut(&mut self) -> DeviceSliceMut<'_, f32> {
        let at = self.layout.logits;
        self.buffer.slice_mut(at..at + self.layout.vocab)
    }

    pub fn logits(&self) -> DeviceSlice<'_, f32> {
        self.buffer
            .slice(self.range(self.layout.logits, self.layout.vocab))
    }
}

/// Positions and host state of a session at `position` (a frontier a pass
/// ending there leaves), from the session before that pass
/// (`MakePrefillCheckpoint`, `PrefillCheckpointBytes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Frontier {
    pub position: u32,
    /// Pooled indexer blocks.
    pub blocks: u32,
    /// The draft block's position and pooled blocks.
    pub mtp: Option<(u32, u32)>,
    /// Kept trunk rows below the frontier.
    pub hidden_rows: u32,
}

impl Frontier {
    pub fn new(c: &Config, session: &Session, position: u32, max_speculative: u32) -> Self {
        let pooled = |at: u32, current: u32| {
            if at > c.indexer_top_k {
                at / c.compress_ratio
            } else {
                current
            }
        };
        let mtp = session
            .mtp
            .as_ref()
            .filter(|_| session.mtp_enabled)
            .map(|m| (position - 1, pooled(position - 1, m.blocks)));
        Self {
            position,
            blocks: pooled(position, session.blocks),
            mtp,
            hidden_rows: if mtp.is_some() {
                (position - session.position).min(max_speculative)
            } else {
                0
            },
        }
    }
}

/// A pass's checkpoint after its first `tokens` rows.
pub struct PrefillCheckpoint {
    pub(crate) tokens: u32,
    pub(crate) frontier: Frontier,
    /// The n-gram window past the boundary token.
    pub(crate) ngram: NgramHistory,
    /// The boundary token's logits, read after the pass.
    pub(crate) logits: Vec<f32>,
}

impl PrefillCheckpoint {
    /// The checkpoint after `prefix` rows of the prefill chunk `chunk`; the
    /// pass may take at most [`VEC_BATCH`] rows past it, from a text-only
    /// session whose draft block has no residual to keep (`prefix >= 2`).
    pub(crate) fn new(
        c: &Config,
        session: &Session,
        chunk: &[i32],
        prefix: u32,
        max_speculative: u32,
    ) -> Result<Self, String> {
        let n = chunk.len() as u32;
        if prefix < 2
            || prefix > n
            || n - prefix > VEC_BATCH
            || !session.vision.layout().images.is_empty()
        {
            return Err("prefill checkpoint must be in the text tail".into());
        }
        let mut ngram = session.ngram;
        if c.ple_layer.is_some() {
            let mut rows = vec![0u32; prefix as usize * c.ple_heads as usize];
            hash_ngram_rows(c, &mut ngram, &chunk[..prefix as usize], &mut rows);
        }
        Ok(Self {
            tokens: prefix,
            frontier: Frontier::new(c, session, session.position + prefix, max_speculative),
            ngram,
            logits: vec![0.0; c.vocab_size as usize],
        })
    }
}
