//! The dense Qwen MTP head, a port of `QwenMtpWeights::LoadFromGguf`
//! (`mtp_reference.cpp`), `QwenMtpGpuModel` and `QwenMtpGpuExecutor`
//! (`hip/mtp/model.cpp`, `hip/mtp/executor.cpp`).
//!
//! One full-attention layer (`blk.64` with its `nextn` fusion projection and
//! norms) predicts the token after the next: it fuses the normalised
//! embedding of the input token with the normalised final hidden state of
//! the target (or of its own previous step), runs the layer over its own
//! FP32 KV cache and reads the target's LM head. Matrices are expanded once
//! to BF16 and vectors to F32; the embedding table and LM head are the
//! target's. The head runs on the target's stream. `MtpDraft` drives it as
//! a draft (`QwenMtpGpuDraftBackend`, `hip/mtp/backend.cpp`).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::gguf::{Gguf, Tensor};
use gufo_hip::{
    DeviceBuffer, DeviceSlice, DeviceSliceMut, DeviceTensor, GgmlType, MappedFile, QuantTensor,
    Stream,
};
use gufo_qwen_kernels as k;

use crate::arena::alloc;
use crate::config::{ModelConfig, NORM_EPS};
use crate::pack::Packer;
use crate::weights::Weights;
use gufo_qwen::vision::VisionInput;

/// The `nextn` layer of the supported artifacts.
const LAYER: &str = "blk.64";

fn launch(error: gufo_hip::LaunchError) -> String {
    format!("MTP launch failed: {error}")
}

/// The packed head of one target model (`QwenMtpGpuModel`).
pub struct MtpModel {
    config: ModelConfig,
    embedding_norm: DeviceTensor<f32>,
    hidden_norm: DeviceTensor<f32>,
    shared_head_norm: DeviceTensor<f32>,
    fusion: QuantTensor,
    attn_norm: DeviceTensor<f32>,
    q: QuantTensor,
    k: QuantTensor,
    v: QuantTensor,
    output: QuantTensor,
    q_norm: DeviceTensor<f32>,
    k_norm: DeviceTensor<f32>,
    ffn_norm: DeviceTensor<f32>,
    gate: QuantTensor,
    up: QuantTensor,
    down: QuantTensor,
    token_embd: QuantTensor,
    lm_head: QuantTensor,
    packed_bytes: usize,
    pack_seconds: f64,
}

/// The named tensor after the C++ `TensorRef` and `Validate` checks.
fn tensor<'a>(
    gguf: &'a Gguf<'a>,
    name: &str,
    elements: usize,
    matrix: bool,
) -> Result<&'a Tensor<'a>, String> {
    let tensor = gguf
        .tensor(name)
        .ok_or_else(|| format!("Missing Qwen MTP tensor: {name}"))?;
    let norm = matches!(tensor.ty, GgmlType::F32 | GgmlType::Bf16);
    let supported =
        norm || (matrix && matches!(tensor.ty, GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q6K));
    if !supported {
        return Err(format!(
            "Unsupported Qwen MTP tensor type for {name}: {}",
            tensor.ty.name()
        ));
    }
    if tensor.elements() != elements as u64 {
        return Err(format!("Qwen MTP tensor shape mismatch for {name}"));
    }
    Ok(tensor)
}

/// Maps `path` with the messages of `GgufReader::OpenSingleFile`.
pub fn open(path: &Path) -> Result<Arc<MappedFile>, String> {
    let file = std::fs::File::open(path)
        .map_err(|_| format!("Failed to open file: {}", path.display()))?;
    if file.metadata().map_or(true, |m| m.len() == 0) {
        return Err(format!("Invalid or empty file: {}", path.display()));
    }
    MappedFile::open(path).map_err(|_| format!("Failed to mmap file: {}", path.display()))
}

impl MtpModel {
    /// Loads the head of `target` from `path` (`QwenMtpGpuModel::Create`).
    pub fn load(stream: &Stream, path: &Path, target: &Weights) -> Result<Self, String> {
        let file = open(path)?;
        let gguf = Gguf::parse(file.bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
        let config = ModelConfig::from_gguf(&gguf)?;
        let c = &config;
        let (hidden, ffn) = (c.hidden_size as usize, c.intermediate_size as usize);
        let (attention, kv) = (c.attention_size() as usize, c.kv_size() as usize);
        let (head, vocab) = (c.head_dim as usize, c.vocab_size as usize);
        let n = |suffix: &str| format!("{LAYER}.{suffix}");
        // The C++ validation order.
        tensor(&gguf, "token_embd.weight", vocab * hidden, true)?;
        let output = if gguf.tensor("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        tensor(&gguf, output, vocab * hidden, true)?;
        let enorm = tensor(&gguf, &n("nextn.enorm.weight"), hidden, false)?;
        let hnorm = tensor(&gguf, &n("nextn.hnorm.weight"), hidden, false)?;
        let fusion = tensor(&gguf, &n("nextn.eh_proj.weight"), hidden * hidden * 2, true)?;
        let shared = tensor(&gguf, &n("nextn.shared_head_norm.weight"), hidden, false)?;
        let attn_norm = tensor(&gguf, &n("attn_norm.weight"), hidden, false)?;
        let q = tensor(&gguf, &n("attn_q.weight"), 2 * attention * hidden, true)?;
        let k = tensor(&gguf, &n("attn_k.weight"), kv * hidden, true)?;
        let v = tensor(&gguf, &n("attn_v.weight"), kv * hidden, true)?;
        let o = tensor(&gguf, &n("attn_output.weight"), hidden * attention, true)?;
        let q_norm = tensor(&gguf, &n("attn_q_norm.weight"), head, false)?;
        let k_norm = tensor(&gguf, &n("attn_k_norm.weight"), head, false)?;
        let ffn_norm = tensor(&gguf, &n("post_attention_norm.weight"), hidden, false)?;
        let gate = tensor(&gguf, &n("ffn_gate.weight"), ffn * hidden, true)?;
        let up = tensor(&gguf, &n("ffn_up.weight"), ffn * hidden, true)?;
        let down = tensor(&gguf, &n("ffn_down.weight"), hidden * ffn, true)?;
        let t = &target.config;
        if t.hidden_size != c.hidden_size || t.vocab_size != c.vocab_size {
            return Err("MTP and target GPU model dimensions are incompatible".into());
        }

        let start = Instant::now();
        let mut p = Packer {
            stream,
            file: &file,
            model: "Qwen MTP",
            bytes: 0,
        };
        // `PackPrivateWeights` order.
        let embedding_norm = p.vector(enorm)?;
        let hidden_norm = p.vector(hnorm)?;
        let shared_head_norm = p.vector(shared)?;
        let fusion = p.bf16_matrix(fusion, hidden, 2 * hidden)?;
        let attn_norm = p.vector(attn_norm)?;
        let q = p.bf16_matrix(q, 2 * attention, hidden)?;
        let k = p.bf16_matrix(k, kv, hidden)?;
        let v = p.bf16_matrix(v, kv, hidden)?;
        let output = p.bf16_matrix(o, hidden, attention)?;
        let q_norm = p.vector(q_norm)?;
        let k_norm = p.vector(k_norm)?;
        let ffn_norm = p.vector(ffn_norm)?;
        let gate = p.bf16_matrix(gate, ffn, hidden)?;
        let up = p.bf16_matrix(up, ffn, hidden)?;
        let down = p.bf16_matrix(down, hidden, ffn)?;
        let packed_bytes = p.bytes;
        Ok(Self {
            config,
            embedding_norm,
            hidden_norm,
            shared_head_norm,
            fusion,
            attn_norm,
            q,
            k,
            v,
            output,
            q_norm,
            k_norm,
            ffn_norm,
            gate,
            up,
            down,
            token_embd: target.token_embd.clone(),
            lm_head: target.output.clone(),
            packed_bytes,
            pack_seconds: start.elapsed().as_secs_f64(),
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Device bytes of the head's own tensors (`GetPackedWeightBytes`).
    pub fn packed_bytes(&self) -> usize {
        self.packed_bytes
    }

    /// Seconds the packing took (`GetPackTimeSeconds`).
    pub fn pack_seconds(&self) -> f64 {
        self.pack_seconds
    }
}

/// The head's input hidden state for one step.
#[derive(Clone, Copy)]
enum Input<'a> {
    /// A target row, copied in.
    Target(&'a [f32]),
    /// A target row already on the device.
    Device(DeviceSlice<'a, f32>),
    /// The head's own previous output.
    Feedback,
}

/// The mutable state of one sequence (`QwenMtpGpuExecutor`).
pub struct MtpHead {
    max_context: u32,
    next_position: u32,
    target_hidden: DeviceBuffer<f32>,
    embedding: DeviceBuffer<f32>,
    fusion: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    qg: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    context: DeviceBuffer<f32>,
    attn_out: DeviceBuffer<f32>,
    ffn_act: DeviceBuffer<f32>,
    ffn_out: DeviceBuffer<f32>,
    feedback: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    k_cache: DeviceBuffer<f32>,
    v_cache: DeviceBuffer<f32>,
    split_k: DeviceBuffer<f32>,
    token: DeviceBuffer<u32>,
}

impl MtpHead {
    /// The state for `max_context` positions (`QwenMtpGpuExecutor::Create`).
    pub fn new(stream: &Stream, model: &MtpModel, max_context: u32) -> Result<Self, String> {
        let c = &model.config;
        if max_context == 0 || max_context > c.context_length {
            return Err("MTP GPU executor context is invalid".into());
        }
        let hidden = c.hidden_size as usize;
        let attention = c.attention_size() as usize;
        let kv = c.kv_size() as usize;
        let cache = kv * max_context as usize;
        let heads = c.num_attention_heads as usize;
        Ok(Self {
            max_context,
            next_position: 0,
            target_hidden: alloc(stream, hidden)?,
            embedding: alloc(stream, hidden)?,
            fusion: alloc(stream, 2 * hidden)?,
            hidden: alloc(stream, hidden)?,
            normed: alloc(stream, hidden)?,
            qg: alloc(stream, 2 * attention)?,
            q: alloc(stream, attention)?,
            k: alloc(stream, kv)?,
            v: alloc(stream, kv)?,
            gate: alloc(stream, attention)?,
            context: alloc(stream, attention)?,
            attn_out: alloc(stream, hidden)?,
            ffn_act: alloc(stream, c.intermediate_size as usize)?,
            ffn_out: alloc(stream, hidden)?,
            feedback: alloc(stream, hidden)?,
            logits: alloc(stream, c.vocab_size as usize)?,
            k_cache: alloc(stream, cache)?,
            v_cache: alloc(stream, cache)?,
            // `DecodeAttentionScratchElements`.
            split_k: alloc(stream, heads * 32 * (c.head_dim as usize + 2))?,
            token: alloc(stream, 1)?,
        })
    }

    /// Starts a new sequence; attention reads only the valid prefix, so the
    /// cache stays (`Reset`).
    pub fn reset(&mut self) {
        self.next_position = 0;
    }

    /// Returns to `position`, dropping the rows after it (`Rewind`).
    pub fn rewind(&mut self, position: u32) -> Result<(), String> {
        if position > self.next_position {
            return Err("MTP rewind position exceeds executed context".into());
        }
        self.next_position = position;
        Ok(())
    }

    pub fn next_position(&self) -> u32 {
        self.next_position
    }

    /// One step on a target row (`ForwardTargetHidden`); with
    /// `compute_logits`, returns the head's greedy token, else 0.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_target_hidden(
        &mut self,
        stream: &Stream,
        model: &MtpModel,
        vision: Option<&VisionInput>,
        token: u32,
        target_hidden: &[f32],
        position: u32,
        compute_logits: bool,
    ) -> Result<u32, String> {
        if target_hidden.len() != model.config.hidden_size as usize {
            return Err("MTP target hidden width is invalid".into());
        }
        self.run(
            stream,
            model,
            vision,
            token,
            Input::Target(target_hidden),
            position,
            compute_logits,
        )
    }

    /// [`MtpHead::forward_target_hidden`] on a row already on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_row(
        &mut self,
        stream: &Stream,
        model: &MtpModel,
        vision: Option<&VisionInput>,
        token: u32,
        row: DeviceSlice<'_, f32>,
        position: u32,
        compute_logits: bool,
    ) -> Result<u32, String> {
        if row.len() != model.config.hidden_size as usize {
            return Err("MTP target hidden width is invalid".into());
        }
        self.run(
            stream,
            model,
            vision,
            token,
            Input::Device(row),
            position,
            compute_logits,
        )
    }

    /// One step on the head's previous output (`ForwardFeedback`).
    pub fn forward_feedback(
        &mut self,
        stream: &Stream,
        model: &MtpModel,
        vision: Option<&VisionInput>,
        token: u32,
        position: u32,
        compute_logits: bool,
    ) -> Result<u32, String> {
        self.run(
            stream,
            model,
            vision,
            token,
            Input::Feedback,
            position,
            compute_logits,
        )
    }

    /// The logits of the last step that computed them (`CopyLastLogits`).
    pub fn logits(&self, stream: &Stream) -> Result<Vec<f32>, String> {
        let mut logits = vec![0f32; self.logits.len()];
        stream
            .download(self.logits.as_slice(), &mut logits)
            .map_err(|e| format!("MTP logit synchronization failed: {e}"))?;
        Ok(logits)
    }

    /// The head's last output row (`CopyLastHidden`).
    pub fn last_hidden(&self, stream: &Stream) -> Result<Vec<f32>, String> {
        let mut row = vec![0f32; self.feedback.len()];
        stream
            .download(self.feedback.as_slice(), &mut row)
            .map_err(|e| format!("MTP hidden synchronization failed: {e}"))?;
        Ok(row)
    }

    /// `QwenMtpGpuExecutor::Run`.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        s: &Stream,
        m: &MtpModel,
        vision: Option<&VisionInput>,
        token: u32,
        input: Input<'_>,
        position: u32,
        compute_logits: bool,
    ) -> Result<u32, String> {
        if position != self.next_position || position >= self.max_context {
            return Err("MTP GPU position is not sequential".into());
        }
        let c = &m.config;
        let hidden = c.hidden_size as usize;
        let attention = c.attention_size() as usize;
        let kv = c.kv_size() as usize;
        let ffn = c.intermediate_size as usize;
        let vocab = c.vocab_size as usize;
        if let Input::Target(row) = input {
            s.upload(row, self.target_hidden.as_mut_slice())
                .map_err(|e| format!("MTP hidden copy failed: {e}"))?;
        }

        k::embedding_lookup_token(
            s,
            m.token_embd.view(),
            token,
            self.embedding.as_mut_slice(),
            hidden,
            vocab,
        )
        .map_err(launch)?;
        if let Some(vision) = vision {
            vision
                .inject(
                    s,
                    self.embedding.as_mut_slice(),
                    position + 1,
                    1,
                    hidden as u32,
                    1,
                )
                .map_err(launch)?;
        }
        k::rms_norm(
            s,
            self.embedding.as_slice(),
            m.embedding_norm.view(),
            self.fusion.slice_mut(0..hidden),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        let source = match input {
            Input::Target(_) => self.target_hidden.as_slice(),
            Input::Device(row) => row,
            Input::Feedback => self.feedback.as_slice(),
        };
        k::rms_norm(
            s,
            source,
            m.hidden_norm.view(),
            self.fusion.slice_mut(hidden..2 * hidden),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        k::gemv(
            s,
            m.fusion.view(),
            self.fusion.as_slice(),
            self.hidden.as_mut_slice(),
            hidden,
            2 * hidden,
        )
        .map_err(launch)?;

        k::rms_norm(
            s,
            self.hidden.as_slice(),
            m.attn_norm.view(),
            self.normed.as_mut_slice(),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        k::fused_qkv_projections(
            s,
            m.q.view(),
            m.k.view(),
            m.v.view(),
            self.normed.as_slice(),
            self.qg.as_mut_slice(),
            self.k.as_mut_slice(),
            self.v.as_mut_slice(),
            2 * attention,
            kv,
            hidden,
        )
        .map_err(launch)?;
        k::unpack_qg(
            s,
            self.qg.as_slice(),
            self.q.as_mut_slice(),
            self.gate.as_mut_slice(),
            c.num_attention_heads,
            c.head_dim,
        )
        .map_err(launch)?;
        k::per_head_rms_norm(
            s,
            m.q_norm.view(),
            self.q.as_mut_slice(),
            c.num_attention_heads,
            c.head_dim,
            NORM_EPS,
        )
        .map_err(launch)?;
        k::per_head_rms_norm(
            s,
            m.k_norm.view(),
            self.k.as_mut_slice(),
            c.num_key_value_heads,
            c.head_dim,
            NORM_EPS,
        )
        .map_err(launch)?;
        k::rope(
            s,
            self.q.as_mut_slice(),
            self.k.as_mut_slice(),
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.rotary_dim,
            position,
            c.rope_theta,
            vision.and_then(VisionInput::rope),
        )
        .map_err(launch)?;
        k::attention_write_fp32(
            s,
            self.q.as_slice(),
            self.k.as_slice(),
            self.v.as_slice(),
            self.gate.as_slice(),
            self.k_cache.as_mut_slice(),
            self.v_cache.as_mut_slice(),
            self.context.as_mut_slice(),
            0,
            position,
            self.max_context,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            self.split_k.as_mut_slice(),
        )
        .map_err(launch)?;
        k::gemv(
            s,
            m.output.view(),
            self.context.as_slice(),
            self.attn_out.as_mut_slice(),
            hidden,
            attention,
        )
        .map_err(launch)?;
        k::residual_add(
            s,
            self.attn_out.as_slice(),
            self.hidden.as_mut_slice(),
            hidden,
        )
        .map_err(launch)?;

        k::rms_norm(
            s,
            self.hidden.as_slice(),
            m.ffn_norm.view(),
            self.normed.as_mut_slice(),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        k::fused_swiglu_gemv(
            s,
            m.gate.view(),
            m.up.view(),
            self.normed.as_slice(),
            self.ffn_act.as_mut_slice(),
            ffn,
            hidden,
        )
        .map_err(launch)?;
        k::gemv(
            s,
            m.down.view(),
            self.ffn_act.as_slice(),
            self.ffn_out.as_mut_slice(),
            hidden,
            ffn,
        )
        .map_err(launch)?;
        k::residual_add(
            s,
            self.ffn_out.as_slice(),
            self.hidden.as_mut_slice(),
            hidden,
        )
        .map_err(launch)?;
        k::rms_norm(
            s,
            self.hidden.as_slice(),
            m.shared_head_norm.view(),
            self.feedback.as_mut_slice(),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        self.next_position += 1;

        if !compute_logits {
            return Ok(0);
        }
        k::gemv(
            s,
            m.lm_head.view(),
            self.feedback.as_slice(),
            self.logits.as_mut_slice(),
            vocab,
            hidden,
        )
        .map_err(launch)?;
        k::argmax(s, self.logits.as_slice(), self.token.as_mut_slice(), vocab).map_err(launch)?;
        let mut result = [0u32];
        s.download(self.token.as_slice(), &mut result)
            .map_err(|_| "MTP GPU result synchronization failed".to_string())?;
        if result[0] as usize >= vocab {
            return Err("logit distribution contains no finite values".into());
        }
        Ok(result[0])
    }
}

/// Target rows prime the head from this many at a time.
const PRIME_CHUNK: usize = 16;

/// What a draft step runs on: the target's stream, the packed head and the
/// target session's image input, which the head reads at its own positions.
pub struct MtpContext<'a> {
    pub stream: &'a Stream,
    pub model: &'a MtpModel,
    pub vision: Option<&'a VisionInput>,
}

/// Target hidden rows handed to the draft: host rows (the prompt) or rows
/// still on the device (verified and decoded rows), `[row][hidden]`.
#[derive(Clone, Copy)]
pub enum Rows<'a> {
    Host(&'a [f32]),
    Device(DeviceSlice<'a, f32>),
}

impl Rows<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Host(rows) => rows.len(),
            Self::Device(rows) => rows.len(),
        }
    }

    /// Writes row `index` of `width` elements into `destination`, in the
    /// stream's order.
    fn store(
        &self,
        stream: &Stream,
        index: usize,
        width: usize,
        destination: DeviceSliceMut<'_, f32>,
    ) -> Result<(), String> {
        let error = |e| format!("MTP hidden copy failed: {e}");
        let range = index * width..(index + 1) * width;
        match self {
            Self::Host(rows) => stream.upload(&rows[range], destination).map_err(error),
            Self::Device(rows) => stream.copy(rows.range(range), destination).map_err(error),
        }
    }
}

/// The MTP draft of one sequence (`QwenMtpGpuDraftBackend`): primed with the
/// target's prompt rows, it proposes greedy blocks and, once verified,
/// replays the committed rows on the target's hidden states. It has no
/// sampled proposals and no snapshots. The C++ backend keeps its target rows
/// on the host and copies each one in; this one keeps them on the device
/// and copies verified and decoded rows there in the stream's order.
pub struct MtpDraft {
    head: MtpHead,
    max_draft_tokens: u32,
    width: usize,
    /// The newest committed target row (`target_hidden_`).
    target_row: DeviceBuffer<f32>,
    /// The row a proposal started from (`proposal_target_hidden_`).
    proposal_row: DeviceBuffer<f32>,
    /// Rows committed during a proposal (`committed_target_hidden_`), at most
    /// a block plus its correction.
    committed: DeviceBuffer<f32>,
    committed_rows: usize,
    /// Host prompt rows on their way to the head.
    staging: DeviceBuffer<f32>,
    proposed: Vec<u32>,
    proposal_input: u32,
    proposal_checkpoint: u32,
    primed: bool,
    proposal_active: bool,
}

impl MtpDraft {
    /// `QwenMtpGpuDraftBackend::Create`.
    pub fn new(
        stream: &Stream,
        model: &MtpModel,
        max_context: u32,
        max_draft_tokens: u32,
    ) -> Result<Self, String> {
        if max_draft_tokens == 0 {
            return Err("MTP GPU draft configuration is invalid".into());
        }
        let width = model.config.hidden_size as usize;
        let mut target_row = alloc::<f32>(stream, width)?;
        stream
            .fill_bytes(target_row.as_mut_slice(), 0)
            .map_err(|e| format!("MTP hidden reset failed: {e}"))?;
        Ok(Self {
            head: MtpHead::new(stream, model, max_context)?,
            max_draft_tokens,
            width,
            target_row,
            proposal_row: alloc(stream, width)?,
            committed: alloc(stream, (max_draft_tokens as usize + 1) * width)?,
            committed_rows: 0,
            staging: alloc(stream, PRIME_CHUNK * width)?,
            proposed: Vec::new(),
            proposal_input: 0,
            proposal_checkpoint: 0,
            primed: false,
            proposal_active: false,
        })
    }

    pub fn max_draft_tokens(&self) -> u32 {
        self.max_draft_tokens
    }

    /// `Reset`. The C++ backend zeroes its host row; nothing reads this one
    /// before priming writes it.
    pub fn reset(&mut self) {
        self.head.reset();
        self.proposed.clear();
        self.committed_rows = 0;
        self.proposal_input = 0;
        self.proposal_checkpoint = 0;
        self.primed = false;
        self.proposal_active = false;
    }

    /// Pairs each prompt token after the first with the target row before
    /// it and keeps the last row for the first proposal
    /// (`PrimeTargetContext`).
    pub fn prime(
        &mut self,
        ctx: &MtpContext<'_>,
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<(), String> {
        self.reset();
        let width = self.width;
        if tokens.is_empty() || hidden.len() != tokens.len() * width {
            return Err("MTP prompt hidden-state shape is invalid".into());
        }
        let primed = self.prime_rows(ctx, tokens, hidden);
        if let Err(error) = primed {
            self.reset();
            return Err(error);
        }
        self.primed = true;
        Ok(())
    }

    fn prime_rows(
        &mut self,
        ctx: &MtpContext<'_>,
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<(), String> {
        let width = self.width;
        let error = |e| format!("MTP hidden copy failed: {e}");
        let rows = tokens.len() - 1;
        for start in (0..rows).step_by(PRIME_CHUNK) {
            let count = PRIME_CHUNK.min(rows - start);
            ctx.stream
                .upload(
                    &hidden[start * width..(start + count) * width],
                    self.staging.slice_mut(0..count * width),
                )
                .map_err(error)?;
            for row in 0..count {
                self.head.forward_row(
                    ctx.stream,
                    ctx.model,
                    ctx.vision,
                    tokens[start + row + 1],
                    self.staging.slice(row * width..(row + 1) * width),
                    (start + row) as u32,
                    false,
                )?;
            }
        }
        Rows::Host(hidden).store(ctx.stream, rows, width, self.target_row.as_mut_slice())
    }

    /// Appends committed tokens at `position` with their target rows
    /// (`AppendTargetContext`).
    pub fn append(
        &mut self,
        ctx: &MtpContext<'_>,
        tokens: &[u32],
        rows: Rows<'_>,
        position: u32,
    ) -> Result<(), String> {
        if !self.primed
            || self.proposal_active
            || self.head.next_position() + 1 != position
            || rows.len() != tokens.len() * self.width
        {
            return Err("MTP draft cannot append the target context".into());
        }
        for (row, &token) in tokens.iter().enumerate() {
            self.head.forward_row(
                ctx.stream,
                ctx.model,
                ctx.vision,
                token,
                self.target_row.as_slice(),
                position + row as u32 - 1,
                false,
            )?;
            rows.store(ctx.stream, row, self.width, self.target_row.as_mut_slice())?;
        }
        Ok(())
    }

    /// Up to `max_tokens` greedy drafts after `last`, the token at
    /// `current_position` (`Propose`).
    pub fn propose(
        &mut self,
        ctx: &MtpContext<'_>,
        last: u32,
        current_position: u32,
        max_tokens: u32,
    ) -> Result<Vec<u32>, String> {
        if !self.primed || current_position == 0 {
            return Err("MTP GPU draft backend is not primed".into());
        }
        if self.proposal_active {
            return Err("MTP GPU proposal feedback is pending".into());
        }
        if self.head.next_position() + 1 != current_position {
            return Err("MTP GPU draft position is inconsistent".into());
        }
        let count = max_tokens.min(self.max_draft_tokens);
        if count == 0 {
            return Ok(Vec::new());
        }
        self.proposal_checkpoint = self.head.next_position();
        self.proposal_input = last;
        ctx.stream
            .copy(self.target_row.as_slice(), self.proposal_row.as_mut_slice())
            .map_err(|e| format!("MTP hidden copy failed: {e}"))?;
        self.committed_rows = 0;
        self.proposed.clear();
        let mut token = self.head.forward_row(
            ctx.stream,
            ctx.model,
            ctx.vision,
            last,
            self.target_row.as_slice(),
            self.proposal_checkpoint,
            true,
        )?;
        self.proposed.push(token);
        for index in 1..count {
            token = self.head.forward_feedback(
                ctx.stream,
                ctx.model,
                ctx.vision,
                token,
                self.proposal_checkpoint + index,
                true,
            )?;
            self.proposed.push(token);
        }
        self.proposal_active = true;
        Ok(self.proposed.clone())
    }

    /// The target row paired with the next correction; during a proposal it
    /// also collects the committed rows (`UpdateTargetHidden`).
    pub fn update_hidden(&mut self, ctx: &MtpContext<'_>, row: Rows<'_>) -> Result<(), String> {
        let width = self.width;
        if row.len() != width {
            return Err("MTP target hidden-state shape is invalid".into());
        }
        if self.proposal_active {
            let slot = self.committed_rows;
            if (slot + 1) * width > self.committed.len() {
                return Err("MTP feedback has more committed rows than a block".into());
            }
            row.store(
                ctx.stream,
                0,
                width,
                self.committed.slice_mut(slot * width..(slot + 1) * width),
            )?;
            self.committed_rows += 1;
        }
        row.store(ctx.stream, 0, width, self.target_row.as_mut_slice())
    }

    /// Replays the proposal's anchor and its `accepted` tokens on the
    /// committed target rows instead of the head's own (`AcceptFeedback`).
    pub fn accept(&mut self, ctx: &MtpContext<'_>, accepted: &[u32]) -> Result<(), String> {
        let width = self.width;
        if !self.proposal_active || accepted.len() > self.proposed.len() {
            return Err("MTP GPU proposal feedback is invalid".into());
        }
        if self.committed_rows != accepted.len() + 1 {
            return Err("MTP feedback requires each committed target row".into());
        }
        let checkpoint = self.proposal_checkpoint;
        self.head.rewind(checkpoint)?;
        self.head.forward_row(
            ctx.stream,
            ctx.model,
            ctx.vision,
            self.proposal_input,
            self.proposal_row.as_slice(),
            checkpoint,
            false,
        )?;
        for (index, &token) in accepted.iter().enumerate() {
            self.head.forward_row(
                ctx.stream,
                ctx.model,
                ctx.vision,
                token,
                self.committed.slice(index * width..(index + 1) * width),
                checkpoint + index as u32 + 1,
                false,
            )?;
        }
        self.proposal_active = false;
        self.proposed.clear();
        self.committed_rows = 0;
        Ok(())
    }
}
