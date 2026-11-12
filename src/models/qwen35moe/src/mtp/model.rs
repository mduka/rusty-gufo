//! The packed MTP block of one target model, a port of
//! `QwenMtpWeights::LoadFromGguf` (`mtp_reference.cpp`) and
//! `QwenMtpGpuModel::Create` (`hip/mtp/model.cpp`).
//!
//! The block after the target's layers (`blk.40` of Qwen3.6 35B-A3B) with
//! its `nextn` fusion projection and norms: its attention matrices are
//! expanded once to BF16 and its vectors to F32; its routed experts, router
//! and shared expert keep their stored encoding, so the block runs the
//! target's one-token MoE kernels. The embedding table and the LM head are
//! the target's. The small draft head is enabled for Qwen3.6 35B-A3B
//! (`mtp_draft_vocab.hpp`).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::gguf::{Gguf, Tensor};
use gufo_hip::{DeviceTensor, GgmlType, MappedFile, QuantTensor, Stream};

use crate::arena::alloc;
use crate::config::ModelConfig;
use crate::mtp::vocab::draft_specials;
use crate::pack::Packer;
use crate::weights::{MoeWeights, Weights};

/// Bytes the MMQ expert kernels may read past a stacked expert tensor
/// (`tail = "4096"` in launch.toml); zero here, as the C++ copies end at
/// the tensor.
const EXPERT_TAIL: usize = 4096;

/// The packed MTP block (`QwenMtpGpuModel`).
pub struct MtpModel {
    pub(crate) config: ModelConfig,
    pub(crate) embedding_norm: DeviceTensor<f32>,
    pub(crate) hidden_norm: DeviceTensor<f32>,
    pub(crate) shared_head_norm: DeviceTensor<f32>,
    /// `hidden` rows of `2 * hidden`, BF16.
    pub(crate) fusion: QuantTensor,
    pub(crate) attn_norm: DeviceTensor<f32>,
    /// Q and gate interleaved per head, BF16.
    pub(crate) q: QuantTensor,
    pub(crate) k: QuantTensor,
    pub(crate) v: QuantTensor,
    pub(crate) output: QuantTensor,
    pub(crate) q_norm: DeviceTensor<f32>,
    pub(crate) k_norm: DeviceTensor<f32>,
    pub(crate) ffn_norm: DeviceTensor<f32>,
    pub(crate) moe: MoeWeights,
    pub(crate) token_embd: QuantTensor,
    pub(crate) lm_head: QuantTensor,
    draft_specials: Option<Vec<i32>>,
    output_row_bytes: usize,
    packed_bytes: usize,
    pack_seconds: f64,
}

fn is_norm_type(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::F32 | GgmlType::Bf16)
}

fn is_matrix_type(ty: GgmlType) -> bool {
    is_norm_type(ty)
        || matches!(
            ty,
            GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
        )
}

/// The named tensor after the C++ `TensorRef` and `Validate` checks, with
/// `label` in the messages.
fn tensor<'a>(
    gguf: &'a Gguf<'a>,
    name: &str,
    label: &str,
    elements: usize,
    matrix: bool,
) -> Result<&'a Tensor<'a>, String> {
    let tensor = gguf
        .tensor(name)
        .ok_or_else(|| format!("Missing Qwen MTP tensor: {label}"))?;
    let supported = if matrix {
        is_matrix_type(tensor.ty)
    } else {
        is_norm_type(tensor.ty)
    };
    if !supported {
        return Err(format!(
            "Unsupported Qwen MTP tensor type for {label}: {}",
            tensor.ty.name()
        ));
    }
    if tensor.elements() != elements as u64 {
        return Err(format!("Qwen MTP tensor shape mismatch for {label}"));
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

/// A device copy of `tensor` in its stored encoding (`CopyRawToDevice`),
/// followed by `tail` zero bytes.
fn raw(p: &mut Packer<'_>, tensor: &Tensor<'_>, tail: usize) -> Result<QuantTensor, String> {
    let stream = p.stream;
    let error = |e| format!("Qwen MTP weights: {e}");
    let payload = p.payload(tensor);
    let mut buffer = alloc::<u8>(stream, payload.len() + tail)?;
    stream
        .upload(payload, buffer.slice_mut(0..payload.len()))
        .map_err(error)?;
    if tail != 0 {
        stream
            .fill_bytes(buffer.slice_mut(payload.len()..payload.len() + tail), 0)
            .map_err(error)?;
    }
    p.bytes += payload.len();
    Ok(QuantTensor::new(
        buffer.freeze(stream).map_err(error)?,
        tensor.ty,
    ))
}

impl MtpModel {
    /// Loads the MTP block of `target` from `path`
    /// (`QwenMtpGpuModel::Create` over `GgufReader::OpenFile`).
    pub fn load(stream: &Stream, path: &Path, target: &Weights) -> Result<Self, String> {
        let file = open(path)?;
        let gguf = Gguf::parse(file.bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
        let config = ModelConfig::from_gguf(&gguf)?;
        let c = &config;
        let (hidden, vocab) = (c.hidden_size as usize, c.vocab_size as usize);
        let (attention, kv) = (c.attention_size() as usize, c.kv_size() as usize);
        let head = c.head_dim as usize;
        // The block follows the target's layers: blk.40 in Qwen3.6 35B-A3B.
        let prefix = format!("blk.{}.", c.num_layers);
        let n = |suffix: &str| format!("{prefix}{suffix}");
        let output_name = if gguf.tensor("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        // The C++ validation order.
        let check =
            |name: &str, elements: usize, matrix: bool| tensor(&gguf, name, name, elements, matrix);
        check("token_embd.weight", vocab * hidden, true)?;
        tensor(&gguf, output_name, "output.weight", vocab * hidden, true)?;
        let enorm = check(&n("nextn.enorm.weight"), hidden, false)?;
        let hnorm = check(&n("nextn.hnorm.weight"), hidden, false)?;
        let fusion = check(&n("nextn.eh_proj.weight"), hidden * 2 * hidden, true)?;
        let shared = check(&n("nextn.shared_head_norm.weight"), hidden, false)?;
        let attn_norm = check(&n("attn_norm.weight"), hidden, false)?;
        let q = check(&n("attn_q.weight"), 2 * attention * hidden, true)?;
        let k = check(&n("attn_k.weight"), kv * hidden, true)?;
        let v = check(&n("attn_v.weight"), kv * hidden, true)?;
        let o = check(&n("attn_output.weight"), hidden * attention, true)?;
        let q_norm = check(&n("attn_q_norm.weight"), head, false)?;
        let k_norm = check(&n("attn_k_norm.weight"), head, false)?;
        let ffn_norm = check(&n("post_attention_norm.weight"), hidden, false)?;
        if !c.is_moe() {
            // The dense FFN of the 27B block runs in gufo-qwen35.
            return Err("the qwen35moe MTP head runs routed-expert blocks".into());
        }
        let (experts, expert_ff, shared_ff) = (
            c.expert_count as usize,
            c.expert_ff_length as usize,
            c.expert_shared_ff_length as usize,
        );
        let router = check(&n("ffn_gate_inp.weight"), experts * hidden, true)?;
        let shared_gate = check(&n("ffn_gate_inp_shexp.weight"), hidden, true)?;
        let gate_exps = check(
            &n("ffn_gate_exps.weight"),
            experts * expert_ff * hidden,
            true,
        )?;
        let up_exps = check(&n("ffn_up_exps.weight"), experts * expert_ff * hidden, true)?;
        let down_exps = check(
            &n("ffn_down_exps.weight"),
            experts * hidden * expert_ff,
            true,
        )?;
        let gate_shexp = check(&n("ffn_gate_shexp.weight"), shared_ff * hidden, true)?;
        let up_shexp = check(&n("ffn_up_shexp.weight"), shared_ff * hidden, true)?;
        let down_shexp = check(&n("ffn_down_shexp.weight"), hidden * shared_ff, true)?;
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
        let moe = MoeWeights {
            router: raw(&mut p, router, 0)?,
            shared_gate: raw(&mut p, shared_gate, 0)?,
            gate_exps: raw(&mut p, gate_exps, EXPERT_TAIL)?,
            up_exps: raw(&mut p, up_exps, EXPERT_TAIL)?,
            down_exps: raw(&mut p, down_exps, EXPERT_TAIL)?,
            gate_shexp: raw(&mut p, gate_shexp, 0)?,
            up_shexp: raw(&mut p, up_shexp, 0)?,
            down_shexp: raw(&mut p, down_shexp, 0)?,
        };
        let packed_bytes = p.bytes;
        let pack_seconds = start.elapsed().as_secs_f64();

        // The draft table copies whole output rows, so the head must be
        // row-addressable; the table's IDs come from the target's tokenizer.
        let lm_head = target.output.clone();
        let mut specials = draft_specials(
            &t.architecture,
            c.vocab_size,
            target.tokenizer.special_ids(),
        );
        let row_bytes = lm_head
            .format()
            .encoded_bytes(c.hidden_size as u64)
            .and_then(|b| usize::try_from(b).ok())
            .filter(|&b| b != 0 && lm_head.bytes().len() >= b * vocab);
        let output_row_bytes = match (&specials, row_bytes) {
            (Some(_), Some(bytes)) => bytes,
            _ => {
                specials = None;
                0
            }
        };
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
            moe,
            token_embd: target.token_embd.clone(),
            lm_head,
            draft_specials: specials,
            output_row_bytes,
            packed_bytes,
            pack_seconds,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Device bytes of the block's own tensors (`GetPackedWeightBytes`).
    pub fn packed_bytes(&self) -> usize {
        self.packed_bytes
    }

    /// Seconds the packing took (`GetPackTimeSeconds`).
    pub fn pack_seconds(&self) -> f64 {
        self.pack_seconds
    }

    /// Whether proposals may score the small draft head (`HasDraftVocab`).
    pub fn has_draft_vocab(&self) -> bool {
        self.draft_specials.is_some()
    }

    /// The special tokens that open every session's draft table
    /// (`GetDraftSpecials`).
    pub fn draft_specials(&self) -> &[i32] {
        self.draft_specials.as_deref().unwrap_or(&[])
    }

    /// Bytes of one output-head row; nonzero with a draft vocabulary
    /// (`GetOutputRowBytes`).
    pub fn output_row_bytes(&self) -> usize {
        self.output_row_bytes
    }
}
