//! Device buffers of one sequence, a port of `QwenGpuArena` (`hip/arena.cpp`).

use std::time::Duration;

use gufo_core::text::CancellationCheck;
use gufo_hip::{
    DeviceBuffer, DeviceCopy, DeviceIndexPair, DeviceIndices, DeviceSlice, DeviceSliceMut, Event,
    F16, IndexSpec, PinnedBuffer, Stream,
};

use crate::config::ModelConfig;
use crate::prefill::PrefillError;
use crate::routes::quantized_activation_bytes;
use crate::weights::Weights;
use gufo_qwen::vision::VisionInput;

/// Prompt rows per prefill batch (`kMaxPromptBatch`).
pub const MAX_PROMPT_BATCH: u32 = 4096;
/// Splits of the split-K decode attention (`kSplitKDecodeAttentionMaxSplits`).
pub(crate) const SPLIT_K_MAX_SPLITS: usize = 32;

pub(crate) fn alloc<T: DeviceCopy>(stream: &Stream, len: usize) -> Result<DeviceBuffer<T>, String> {
    stream
        .alloc_persistent(len)
        .map_err(|e| format!("Qwen device buffer: {e}"))
}

/// Activations and workspaces every forward pass uses, shared by the
/// sessions of a model: one operation runs at a time. Batch-sized buffers
/// have the C++ arena's sizes; a decode step uses their first row.
pub(crate) struct Scratch {
    pub(crate) hidden: DeviceBuffer<f32>,
    pub(crate) normed: DeviceBuffer<f32>,
    pub(crate) q: DeviceBuffer<f32>,
    pub(crate) k: DeviceBuffer<f32>,
    pub(crate) v: DeviceBuffer<f32>,
    pub(crate) attn_out: DeviceBuffer<f32>,
    pub(crate) ffn_act: DeviceBuffer<f32>,
    pub(crate) ffn_out: DeviceBuffer<f32>,
    pub(crate) ssm_qkv: DeviceBuffer<f32>,
    pub(crate) conv_out: DeviceBuffer<f32>,
    pub(crate) ssm_gate: DeviceBuffer<f32>,
    pub(crate) ssm_out: DeviceBuffer<f32>,
    pub(crate) alpha: DeviceBuffer<f32>,
    pub(crate) beta: DeviceBuffer<f32>,
    /// Token IDs below the vocabulary size.
    pub(crate) tokens: DeviceIndices,
    /// A decode step's token ID (below the vocabulary size) and position
    /// (below the context length), uploaded in one copy as C++ does.
    pub(crate) decode_input: DeviceIndexPair,
    pub(crate) sampled: DeviceBuffer<u32>,
    pub(crate) split_k: DeviceBuffer<f32>,
    /// Prefill: FP16 activations (the C++ BF16 scratch, same size).
    pub(crate) scratch16: DeviceBuffer<F16>,
    /// Prefill: tiled Q8_1 activations, large enough for the macro-tile
    /// reads of the Q8_0 GEMM (`launch.toml`).
    pub(crate) q8: DeviceBuffer<u8>,
    /// Prefill: FFN gate and up rows, also the packed KV workspaces of the
    /// WMMA attention.
    pub(crate) ffn_gate: DeviceBuffer<f32>,
    pub(crate) ffn_up: DeviceBuffer<f32>,
    /// Prefill: scratch of the row-split DeltaNet recurrence.
    pub(crate) ssm_kq_scales: DeviceBuffer<f32>,
    pub(crate) ssm_alpha_beta: DeviceBuffer<f32>,
    /// Verification scratch: the packed gate/up and alpha/beta rows and the
    /// split-K rows (the C++ BF16 weight scratch, same bytes, as floats).
    pub(crate) packed: DeviceBuffer<f32>,
    /// Verification: the position of each row, where the SSM replay capture
    /// reads its ring slot.
    pub(crate) positions: DeviceBuffer<u32>,
    /// Verification logits `[row][vocab]`, also the tapped hidden rows while
    /// the layers run; allocated with the first verification.
    pub(crate) verification_logits: Option<DeviceBuffer<f32>>,
    /// Pinned staging of the token IDs and positions of a decode, batch or
    /// verification step: they go up without a stream synchronisation (a
    /// pageable upload waits for its copy), as the C++ executor enqueues
    /// them.
    pub(crate) host_tokens: PinnedBuffer<u32>,
    pub(crate) host_decode_input: PinnedBuffer<u32>,
    pub(crate) host_positions: PinnedBuffer<u32>,
}

/// The device state of one sequence: caches, recurrent state and what a
/// later call reads back (`QwenGpuArena` per session).
pub(crate) struct SessionBuffers {
    /// K plane then V plane, each `[attention layer][max_context][kv heads *
    /// head_dim]`.
    pub(crate) kv: DeviceBuffer<F16>,
    /// The recurrent state and its rollback twin.
    pub(crate) recurrent: RecurrentState,
    /// Logits of the last row that computed them.
    pub(crate) logits: DeviceBuffer<f32>,
    /// Hidden rows of the tapped layers after the last decode or
    /// verification row, `[tap][hidden]`, for the draft model.
    pub(crate) target_features: DeviceBuffer<f32>,
    /// The inputs of verified rows, to replay the recurrence after a
    /// rollback; allocated with the first saved state.
    pub(crate) replay: Option<ReplayRing>,
    /// Logits `[row][vocab]` of the last verification of this session;
    /// allocated with its first verification.
    pub(crate) verification_logits: Option<DeviceBuffer<f32>>,
    /// Image positions and embeddings of the current request.
    pub(crate) vision: VisionInput,
    /// The request's cancellation check, asked before prefill chunks and
    /// layers and while images encode (`SetCancellationCheck`).
    pub(crate) cancel: Option<CancellationCheck>,
    /// Completion events of the last two four-layer prefill groups, created
    /// with the first prefill that has a cancellation check.
    prefill_events: Option<[Event; 2]>,
}

/// One recurrent state: `[recurrent layer][qkv][conv kernel]` conv windows
/// and FP32 `[recurrent layer][heads][value][state]` DeltaNet states.
pub(crate) struct RecurrentBuffers {
    pub(crate) conv: DeviceBuffer<f32>,
    pub(crate) deltanet: DeviceBuffer<f32>,
}

/// The recurrent state in one of two buffers (`QwenGpuArena`'s rollback
/// pair). Saving pins the live buffer and restoring makes
/// the pinned one live again; neither copies. The first pass after either
/// reads the pinned buffer and writes the other one, so the rollback point
/// survives until the next save; every other pass updates in place, which
/// every kernel computes bit for bit like the forked form.
pub(crate) struct RecurrentState {
    /// Buffer 0, from creation.
    first: RecurrentBuffers,
    /// Buffer 1, from the first save; it exists whenever an index is 1.
    second: Option<RecurrentBuffers>,
    /// The buffer that holds the current state.
    live: usize,
    /// The pinned buffer, meaningful while a state is saved.
    saved: usize,
    /// The buffer the most recent pass read.
    source: usize,
    has_saved: bool,
}

/// The buffers of one recurrent-state pass: `*_in` is `None` when the pass
/// updates in place.
pub(crate) struct RecurrentIo<'a> {
    pub(crate) conv_in: Option<DeviceSlice<'a, f32>>,
    pub(crate) conv: DeviceSliceMut<'a, f32>,
    pub(crate) deltanet_in: Option<DeviceSlice<'a, f32>>,
    pub(crate) deltanet: DeviceSliceMut<'a, f32>,
}

const SECOND_MISSING: &str = "buffer 1 exists once a recurrent state is saved";

impl RecurrentState {
    fn new(stream: &Stream, conv: usize, deltanet: usize) -> Result<Self, String> {
        Ok(Self {
            first: RecurrentBuffers {
                conv: alloc(stream, conv)?,
                deltanet: alloc(stream, deltanet)?,
            },
            second: None,
            live: 0,
            saved: 0,
            source: 0,
            has_saved: false,
        })
    }

    /// The current state.
    pub(crate) fn live(&self) -> &RecurrentBuffers {
        match self.live {
            0 => &self.first,
            _ => self.second.as_ref().expect(SECOND_MISSING),
        }
    }

    pub(crate) fn live_mut(&mut self) -> &mut RecurrentBuffers {
        match self.live {
            0 => &mut self.first,
            _ => self.second.as_mut().expect(SECOND_MISSING),
        }
    }

    /// Back to buffer 0 without a saved state (`QwenGpuArena::Reset`):
    /// whatever buffer 1 holds is overwritten before it is read.
    pub(crate) fn reset(&mut self) {
        (self.live, self.saved, self.source, self.has_saved) = (0, 0, 0, false);
    }

    /// Allocates the second buffer unless it exists
    /// (`AllocateRecurrentSnapshot`).
    pub(crate) fn allocate_second(&mut self, stream: &Stream) -> Result<(), String> {
        if self.second.is_none() {
            self.second = Some(RecurrentBuffers {
                conv: alloc(stream, self.first.conv.len())?,
                deltanet: alloc(stream, self.first.deltanet.len())?,
            });
        }
        Ok(())
    }

    /// Pins the live buffer for rollback (`SaveState`), allocating the
    /// second buffer with the first save.
    pub(crate) fn save(&mut self, stream: &Stream) -> Result<(), String> {
        self.allocate_second(stream)?;
        self.saved = self.live;
        self.has_saved = true;
        Ok(())
    }

    /// Makes the pinned buffer live again (`RestoreState`).
    pub(crate) fn restore(&mut self) {
        self.live = self.saved;
    }

    /// Starts one forward, replay or prefill pass (`BeginRecurrentWrite`):
    /// called once before the pass's first launch. Writing into the pinned
    /// buffer would destroy the rollback point, so the first pass after a
    /// save or restore writes the other one.
    pub(crate) fn begin_write(&mut self) {
        self.source = self.live;
        if self.has_saved && self.live == self.saved {
            self.live ^= 1;
        }
    }

    /// The buffers of the current pass for its first launch per layer; later
    /// launches of the same layer in the pass continue in place
    /// ([`Self::continued`]).
    pub(crate) fn io(&mut self) -> RecurrentIo<'_> {
        if self.source == self.live {
            return self.continued();
        }
        let second = self.second.as_mut().expect(SECOND_MISSING);
        let (source, live) = if self.live == 1 {
            (&self.first, second)
        } else {
            (&*second, &mut self.first)
        };
        RecurrentIo {
            conv_in: Some(source.conv.as_slice()),
            conv: live.conv.as_mut_slice(),
            deltanet_in: Some(source.deltanet.as_slice()),
            deltanet: live.deltanet.as_mut_slice(),
        }
    }

    /// The live buffers updated in place.
    pub(crate) fn continued(&mut self) -> RecurrentIo<'_> {
        let live = self.live_mut();
        RecurrentIo {
            conv_in: None,
            conv: live.conv.as_mut_slice(),
            deltanet_in: None,
            deltanet: live.deltanet.as_mut_slice(),
        }
    }

    /// The source and live buffer of the most recent pass, for graph keys
    /// (`RecurrentIoIdentity`).
    pub(crate) fn identity(&self) -> u8 {
        ((self.source << 1) | self.live) as u8
    }
}

/// Recurrence inputs of the last 16 positions per recurrent layer
/// (`[layer][16][width]`, slot `position % 16`) and the device flag that
/// switches capture on (`QwenGpuArena` replay log).
pub(crate) struct ReplayRing {
    pub(crate) qkv: DeviceBuffer<f32>,
    pub(crate) alpha: DeviceBuffer<f32>,
    pub(crate) beta: DeviceBuffer<f32>,
    pub(crate) enabled: DeviceBuffer<u32>,
}

/// What every forward pass reads: the stream, the weights and the context
/// length the arena was sized for.
#[derive(Clone, Copy)]
pub(crate) struct Forward<'a> {
    pub(crate) stream: &'a Stream,
    pub(crate) weights: &'a Weights,
    pub(crate) max_context: u32,
}

/// Rows of one verification or decode batch (`kMaxDecodeBatch`).
pub const MAX_VERIFY_ROWS: usize = 8;
/// Positions each recurrent layer's replay ring holds (`kSsmReplayCapacity`).
pub(crate) const REPLAY_CAPACITY: usize = 16;
/// Tapped target layers the draft model may read (`kMaxTargetLayerTaps`).
pub const MAX_TARGET_TAPS: usize = 5;

impl ReplayRing {
    pub(crate) fn new(stream: &Stream, c: &ModelConfig) -> Result<Self, String> {
        let slots = c.ssm_layer_count() as usize * REPLAY_CAPACITY;
        let rank = c.ssm_time_step_rank as usize;
        Ok(Self {
            qkv: alloc(stream, slots * c.ssm_qkv_size() as usize)?,
            alpha: alloc(stream, slots * rank)?,
            beta: alloc(stream, slots * rank)?,
            enabled: alloc(stream, 1)?,
        })
    }
}

impl Scratch {
    pub(crate) fn new(stream: &Stream, c: &ModelConfig, max_context: u32) -> Result<Self, String> {
        let batch = max_context.min(MAX_PROMPT_BATCH) as usize;
        let hidden = c.hidden_size as usize;
        let attention = c.attention_size() as usize;
        let kv = c.kv_size() as usize;
        let qkv = c.ssm_qkv_size() as usize;
        let inner = c.ssm_inner_size as usize;
        let rank = c.ssm_time_step_rank as usize;
        let recurrent = attention.max(inner);
        let projection = (2 * attention).max(qkv);
        let ffn = c.intermediate_size as usize;
        let scratch_row = ffn.max(hidden).max(projection).max(qkv + inner + 2 * rank);
        // The C++ arena's Q8_1 size, or the macro-tile envelope of the widest
        // GEMM input if that is larger (a batch that is no multiple of 128).
        let q8_rows = batch.next_multiple_of(16);
        let q8_bytes = ((q8_rows * scratch_row).div_ceil(32) * 40 + 4096).max(
            quantized_activation_bytes(batch.next_multiple_of(128), scratch_row),
        );
        // The BF16 weight scratch of the C++ arena: the largest per-layer
        // matrix in BF16, here as floats of the same bytes.
        let widest = (2 * attention)
            .max(kv)
            .max(attention)
            .max(ffn)
            .max(qkv)
            .max(inner)
            .max(rank);
        let packed_floats = hidden * widest / 2;
        let tokens = alloc(stream, batch.max(2))?;
        let decode_input = alloc(stream, 2)?;
        let indices = |buffer, bound| {
            stream
                .zeroed_indices(buffer, bound)
                .map_err(|e| format!("Qwen device buffer: {e}"))
        };
        Ok(Self {
            hidden: alloc(stream, batch * hidden)?,
            normed: alloc(stream, batch * hidden)?,
            q: alloc(stream, batch * attention)?,
            k: alloc(stream, batch * kv)?,
            v: alloc(stream, batch * kv)?,
            attn_out: alloc(stream, batch * hidden)?,
            ffn_act: alloc(stream, batch * ffn)?,
            ffn_out: alloc(stream, batch * hidden)?,
            ssm_qkv: alloc(stream, batch * projection)?,
            conv_out: alloc(stream, batch * qkv)?,
            ssm_gate: alloc(stream, batch * recurrent)?,
            ssm_out: alloc(stream, batch * recurrent)?,
            alpha: alloc(stream, batch * rank)?,
            beta: alloc(stream, batch * rank)?,
            tokens: indices(tokens, c.vocab_size)?,
            decode_input: stream
                .zeroed_index_pair(
                    decode_input,
                    1,
                    [
                        IndexSpec::below(c.vocab_size),
                        IndexSpec::below(max_context),
                    ],
                )
                .map_err(|e| format!("Qwen device buffer: {e}"))?,
            sampled: alloc(stream, MAX_VERIFY_ROWS * MAX_VERIFY_ROWS)?,
            split_k: alloc(
                stream,
                c.num_attention_heads as usize * SPLIT_K_MAX_SPLITS * (c.head_dim as usize + 2),
            )?,
            scratch16: alloc(stream, batch * scratch_row)?,
            q8: alloc(stream, q8_bytes)?,
            ffn_gate: alloc(stream, batch * ffn)?,
            ffn_up: alloc(stream, batch * ffn)?,
            ssm_kq_scales: alloc(stream, batch * c.ssm_group_count as usize * 3)?,
            ssm_alpha_beta: alloc(stream, batch * rank * 2)?,
            packed: alloc(stream, packed_floats)?,
            positions: alloc(stream, MAX_VERIFY_ROWS * MAX_VERIFY_ROWS)?,
            verification_logits: None,
            host_tokens: pinned(MAX_VERIFY_ROWS * MAX_VERIFY_ROWS)?,
            host_decode_input: pinned(2)?,
            host_positions: pinned(MAX_VERIFY_ROWS * MAX_VERIFY_ROWS)?,
        })
    }

    /// Enqueues the token IDs of a step (at most `MAX_VERIFY_ROWS^2`)
    /// through the pinned staging.
    pub(crate) fn stage_tokens(&mut self, stream: &Stream, tokens: &[u32]) -> Result<(), String> {
        stage(&mut self.host_tokens, tokens)?;
        stream
            .upload_indices_async(&mut self.host_tokens, tokens.len(), &mut self.tokens)
            .map_err(|e| e.to_string())
    }

    /// Enqueues a decode step's token ID and position through the pinned
    /// staging, in one copy.
    pub(crate) fn stage_decode_input(
        &mut self,
        stream: &Stream,
        token: u32,
        position: u32,
    ) -> Result<(), String> {
        stage(&mut self.host_decode_input, &[token, position])?;
        stream
            .upload_index_pair_async(&mut self.host_decode_input, &mut self.decode_input)
            .map_err(|e| e.to_string())
    }

    /// Enqueues the row positions of a batch or verification step through
    /// the pinned staging.
    pub(crate) fn stage_positions(
        &mut self,
        stream: &Stream,
        positions: &[u32],
    ) -> Result<(), String> {
        stage(&mut self.host_positions, positions)?;
        let len = positions.len();
        stream
            .upload_async_front(
                &mut self.host_positions,
                len,
                self.positions.slice_mut(0..len),
            )
            .map_err(|e| e.to_string())
    }
}

fn pinned(len: usize) -> Result<PinnedBuffer<u32>, String> {
    PinnedBuffer::zeroed(len).map_err(|e| format!("Qwen pinned staging: {e}"))
}

/// Writes `values` to the front of `host` once its last copy completed.
fn stage(host: &mut PinnedBuffer<u32>, values: &[u32]) -> Result<(), String> {
    host.host_mut()
        .map_err(|e| e.to_string())?
        .get_mut(..values.len())
        .ok_or("more values than the pinned staging holds")?
        .copy_from_slice(values);
    Ok(())
}

impl SessionBuffers {
    pub(crate) fn new(stream: &Stream, c: &ModelConfig, max_context: u32) -> Result<Self, String> {
        let kv = c.kv_size() as usize;
        let qkv = c.ssm_qkv_size() as usize;
        let rank = c.ssm_time_step_rank as usize;
        let ssm_layers = c.ssm_layer_count() as usize;
        let kv_plane = c.full_attention_layer_count() as usize * kv * max_context as usize;
        Ok(Self {
            kv: alloc(stream, 2 * kv_plane)?,
            recurrent: RecurrentState::new(
                stream,
                ssm_layers * qkv * c.ssm_conv_kernel as usize,
                ssm_layers * rank * c.ssm_state_size as usize * c.ssm_value_size() as usize,
            )?,
            logits: alloc(stream, c.vocab_size as usize)?,
            target_features: alloc(stream, MAX_TARGET_TAPS * c.hidden_size as usize)?,
            replay: None,
            verification_logits: None,
            vision: VisionInput::default(),
            cancel: None,
            prefill_events: None,
        })
    }

    /// Whether the request's check reports cancellation.
    pub(crate) fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|cancelled| cancelled())
    }

    /// Before prefill layer `layer` (`PrefillLayerCheckpoint`): stops a
    /// cancelled request, and while a check is set keeps the host one
    /// four-layer group ahead of the device. Without that backpressure a
    /// prefill enqueues the whole prompt before a client disconnects, and
    /// checking on the host cannot stop seconds of abandoned device work.
    pub(crate) fn prefill_layer_checkpoint(
        &mut self,
        stream: &Stream,
        layer: u32,
    ) -> Result<(), PrefillError> {
        if self.cancel.is_none() {
            return Ok(());
        }
        if self.cancelled() {
            return Err(PrefillError::Cancelled);
        }
        if layer == 0 || !layer.is_multiple_of(4) {
            return Ok(());
        }
        if self.prefill_events.is_none() {
            self.prefill_events = Some([Event::new()?, Event::new()?]);
        }
        let Some(events) = &self.prefill_events else {
            return Ok(());
        };
        let slot = (layer / 4 - 1) as usize % events.len();
        stream.record(&events[slot])?;
        if layer < 8 {
            return Ok(());
        }
        while !events[1 - slot].query()? {
            if self.cancelled() {
                return Err(PrefillError::Cancelled);
            }
            std::thread::sleep(Duration::from_micros(200));
        }
        Ok(())
    }
}
