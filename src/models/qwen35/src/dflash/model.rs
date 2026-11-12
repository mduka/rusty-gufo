//! The DFlash2 draft model: configuration and weights from its GGUF file, a
//! port of `QwenDFlashWeights::LoadFromGguf` (`dflash_weights.cpp`) and
//! `QwenDFlashGpuModel::Create` (`hip/dflash/model.cpp`).
//!
//! Vectors become F32 device tensors. The feature projection, the selector
//! codebooks and the K/V projections become BF16 (dequantised on the host,
//! rounded to nearest even); the other matrices stay in their Q8_0, Q4_K or
//! Q6_K blocks. Each tensor gets its own exactly sized allocation, as in C++.
//! The draft shares the target's embedding table and LM head.

use std::path::Path;

use gufo_core::gguf::{Gguf, Tensor, Value};
use gufo_hip::{Bf16, DeviceTensor, GgmlType, MappedFile, QuantTensor, Stream};

use crate::arena::alloc;
use crate::weights::Weights;
use gufo_qwen_kernels as k;

#[derive(Debug, Clone, PartialEq)]
pub struct DFlashConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub rope_theta: f32,
    pub context_length: u32,
    pub num_layers: u32,
    pub vocab_size: u32,
    /// Zero-based target layers whose outputs feed the draft.
    pub target_layer_ids: Vec<u32>,
    /// Block size including the anchor token.
    pub block_size: u32,
    pub mask_token_id: u32,
    pub conv_kernel_size: u32,
    pub conv_group_size: u32,
    pub selector_rank: u32,
    pub selector_top_k: u32,
    pub sliding_window: u32,
}

impl DFlashConfig {
    pub fn attention_size(&self) -> usize {
        self.num_attention_heads as usize * self.head_dim as usize
    }

    pub fn kv_size(&self) -> usize {
        self.num_key_value_heads as usize * self.head_dim as usize
    }

    /// Dynamic convolution coefficients per token: both sides, every tap,
    /// every channel group.
    pub fn dynamic_coefficients(&self) -> usize {
        2 * self.conv_kernel_size as usize * (self.hidden_size / self.conv_group_size) as usize
    }

    /// Width of one row of tapped target features.
    pub fn target_features(&self) -> usize {
        self.target_layer_ids.len() * self.hidden_size as usize
    }
}

/// A matrix the block GEMMs read: packed quant blocks or BF16.
pub enum Matrix {
    Quant(QuantTensor),
    Bf16(DeviceTensor<Bf16>),
}

pub struct DFlashLayer {
    pub attn_norm: DeviceTensor<f32>,
    pub q: Matrix,
    pub k: DeviceTensor<Bf16>,
    pub v: DeviceTensor<Bf16>,
    pub output: Matrix,
    pub q_norm: DeviceTensor<f32>,
    pub k_norm: DeviceTensor<f32>,
    pub ffn_norm: DeviceTensor<f32>,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
    /// `[2][kernel][hidden]`, before and after the mixer.
    pub attn_conv_base: DeviceTensor<f32>,
    pub ffn_conv_base: DeviceTensor<f32>,
    pub attn_conv_proj: Matrix,
    pub ffn_conv_proj: Matrix,
}

pub struct DFlashModel {
    pub config: DFlashConfig,
    pub fc_norm: DeviceTensor<f32>,
    /// `[hidden][taps * hidden]`.
    pub fc: DeviceTensor<Bf16>,
    pub output_norm: DeviceTensor<f32>,
    pub selector_predecessor: DeviceTensor<Bf16>,
    pub selector_successor: DeviceTensor<Bf16>,
    pub selector_hidden: Matrix,
    pub layers: Vec<DFlashLayer>,
    /// The target's embedding table and LM head.
    pub token_embd: QuantTensor,
    pub output: QuantTensor,
    /// Device bytes of the draft's own tensors (`GetPackedWeightBytes`).
    pub packed_bytes: usize,
    /// The target tokenizer's message framing, `<|im_start|>`, `<|im_end|>`
    /// and `<|endoftext|>`, as far as it has them: prompt-history copies
    /// stop before these tokens (`lookup_boundaries_`).
    pub message_boundaries: Vec<u32>,
}

/// The Q8_0 copy of a BF16 LM head of `elements` weights
/// (`QuantizeDraftHead`); its bytes count as the draft's.
fn draft_head(
    stream: &Stream,
    head: &QuantTensor,
    elements: usize,
    packed_bytes: &mut usize,
) -> Result<QuantTensor, String> {
    let error = |e: String| format!("DFlash-2 draft head: {e}");
    if !elements.is_multiple_of(32) {
        return Err(error(
            "the BF16 target head is no whole number of Q8_0 blocks".into(),
        ));
    }
    let bytes = elements / 32 * 34;
    let mut copy = alloc::<u8>(stream, bytes)?;
    let source = crate::routes::bf16_weights(head).map_err(|e| error(e.to_string()))?;
    k::quantize_bf16_to_q8_0(stream, source, copy.as_mut_slice(), elements)
        .map_err(|e| error(e.to_string()))?;
    stream.synchronize().map_err(|e| error(e.to_string()))?;
    *packed_bytes += bytes;
    let copy = copy.freeze(stream).map_err(|e| error(e.to_string()))?;
    Ok(QuantTensor::new(copy, GgmlType::Q8_0))
}

fn metadata_u32(gguf: &Gguf<'_>, key: &str) -> Result<u32, String> {
    gguf.u32(key)
        .ok_or_else(|| format!("DFlash-2 GGUF is missing required metadata '{key}'"))
}

/// A non-negative integer array (`GetNonnegativeIntegerArray`).
fn integer_array(gguf: &Gguf<'_>, key: &str) -> Option<Vec<u64>> {
    match gguf.metadata(key)? {
        Value::UIntArray(values) => Some(values.clone()),
        Value::IntArray(values) => values.iter().map(|&v| u64::try_from(v).ok()).collect(),
        _ => None,
    }
}

impl DFlashConfig {
    pub fn from_gguf(gguf: &Gguf<'_>) -> Result<Self, String> {
        if gguf.string("general.architecture") != Some(b"dflash") {
            return Err("DFlash-2 GGUF requires general.architecture='dflash'".into());
        }
        let key = |name: &str| format!("dflash.{name}");
        let rotary_dim = gguf.u32(&key("rope.dimension_count")).or_else(|| {
            let sections = integer_array(gguf, &key("rope.dimension_sections"))?;
            (sections.len() == 4
                && sections[0] > 0
                && sections[1..] == [0, 0, 0]
                && sections[0] <= u64::from(u32::MAX / 2))
            .then(|| (sections[0] * 2) as u32)
        });
        let c = Self {
            hidden_size: metadata_u32(gguf, &key("embedding_length"))?,
            intermediate_size: metadata_u32(gguf, &key("feed_forward_length"))?,
            num_attention_heads: metadata_u32(gguf, &key("attention.head_count"))?,
            num_key_value_heads: metadata_u32(gguf, &key("attention.head_count_kv"))?,
            head_dim: metadata_u32(gguf, &key("attention.key_length"))?,
            rotary_dim: rotary_dim.ok_or(
                "DFlash-2 GGUF requires a valid rotary dimension or degenerate four-section M-RoPE layout",
            )?,
            context_length: metadata_u32(gguf, &key("context_length"))?,
            num_layers: metadata_u32(gguf, &key("block_count"))?,
            rope_theta: gguf
                .f32(&key("rope.freq_base"))
                .ok_or("DFlash-2 GGUF is missing required metadata 'dflash.rope.freq_base'")?,
            vocab_size: 0,
            target_layer_ids: Vec::new(),
            block_size: metadata_u32(gguf, &key("block_size"))?,
            conv_kernel_size: metadata_u32(gguf, &key("conv_kernel_size"))?,
            conv_group_size: metadata_u32(gguf, &key("conv_group_size"))?,
            selector_rank: metadata_u32(gguf, &key("selector_rank"))?,
            selector_top_k: metadata_u32(gguf, &key("selector_top_k"))?,
            sliding_window: metadata_u32(gguf, &key("attention.sliding_window"))?,
            mask_token_id: metadata_u32(gguf, "tokenizer.ggml.mask_token_id")?,
        };
        if c.hidden_size == 0
            || c.intermediate_size == 0
            || c.num_attention_heads == 0
            || c.num_key_value_heads == 0
            || c.head_dim == 0
            || c.num_layers == 0
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
            || c.rotary_dim == 0
            || c.rotary_dim > c.head_dim
            || !c.rotary_dim.is_multiple_of(2)
        {
            return Err("DFlash-2 GGUF has an invalid transformer topology".into());
        }
        let causal = gguf
            .bool(&key("attention.causal"))
            .ok_or("DFlash-2 GGUF is missing required metadata 'dflash.attention.causal'")?;
        if c.block_size < 2
            || c.conv_kernel_size == 0
            || c.conv_group_size == 0
            || !c.hidden_size.is_multiple_of(c.conv_group_size)
            || c.selector_rank == 0
            || c.selector_top_k == 0
            || c.selector_top_k > 16
            || c.sliding_window < c.block_size
            || causal
        {
            return Err("DFlash-2 GGUF has an invalid diffusion topology".into());
        }
        // GGUF numbers layer inputs; the taps are zero-based layer outputs.
        let encoded = integer_array(gguf, &key("target_layers"))
            .ok_or("DFlash-2 GGUF is missing required integer array 'dflash.target_layers'")?;
        if encoded.is_empty() {
            return Err("DFlash-2 target layer list must not be empty".into());
        }
        let mut c = c;
        for layer in encoded {
            let normalized = layer
                .checked_sub(1)
                .and_then(|l| u32::try_from(l).ok())
                .ok_or(
                    "DFlash-2 target layer indices must use positive GGUF layer-input numbering",
                )?;
            if c.target_layer_ids
                .last()
                .is_some_and(|&last| normalized <= last)
            {
                return Err("DFlash-2 target layer indices must be strictly increasing".into());
            }
            c.target_layer_ids.push(normalized);
        }
        Ok(c)
    }
}

/// Matrix formats the loader accepts (`IsSupportedMatrixType`).
const MATRIX: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::Bf16,
    GgmlType::Q8_0,
    GgmlType::Q4K,
    GgmlType::Q6K,
];
const VECTOR: &[GgmlType] = &[GgmlType::F32, GgmlType::Bf16];

struct Packer<'a> {
    pack: crate::pack::Packer<'a>,
    gguf: &'a Gguf<'a>,
}

impl<'a> Packer<'a> {
    /// The named tensor after the C++ `BindRequiredTensor` checks.
    fn tensor(&self, name: &str, elements: usize, matrix: bool) -> Result<Tensor<'a>, String> {
        let tensor = self
            .gguf
            .tensor(name)
            .ok_or_else(|| format!("DFlash-2 GGUF is missing required tensor '{name}'"))?
            .clone();
        if tensor.elements() != elements as u64 {
            return Err(format!(
                "DFlash-2 tensor '{name}' has {} elements; expected {elements}",
                tensor.elements()
            ));
        }
        if !(if matrix { MATRIX } else { VECTOR }).contains(&tensor.ty) {
            return Err(format!(
                "DFlash-2 tensor '{name}' has unsupported type {}",
                tensor.ty.name()
            ));
        }
        Ok(tensor)
    }

    /// An F32 copy of a vector (`CopyVectorF32`).
    fn vector(&mut self, name: &str, elements: usize) -> Result<DeviceTensor<f32>, String> {
        let tensor = self.tensor(name, elements, false)?;
        self.pack.vector(&tensor)
    }

    /// BF16 rows (`PackMatrixBf16`).
    fn bf16(&mut self, name: &str, rows: usize, cols: usize) -> Result<DeviceTensor<Bf16>, String> {
        let tensor = self.tensor(name, rows * cols, true)?;
        self.pack.bf16(&tensor, rows, cols)
    }

    /// Packed blocks kept as they are, other formats as BF16
    /// (`PackBlockMatrix`).
    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<Matrix, String> {
        let tensor = self.tensor(name, rows * cols, true)?;
        if !matches!(tensor.ty, GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K) {
            return self.bf16(name, rows, cols).map(Matrix::Bf16);
        }
        let format = tensor.ty;
        let bytes = self.pack.payload(&tensor).to_vec();
        Ok(Matrix::Quant(QuantTensor::new(
            self.pack.upload(&bytes)?,
            format,
        )))
    }
}

impl DFlashModel {
    /// Loads the draft of `target` from `path`.
    pub fn load(stream: &Stream, path: &Path, target: &Weights) -> Result<Self, String> {
        let file = MappedFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let gguf = Gguf::parse(file.bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut config = DFlashConfig::from_gguf(&gguf)?;
        if gguf.tensor("d2t").is_some() {
            return Err("DFlash-2 reduced-vocabulary d2t exports are not supported yet".into());
        }
        let rank = config.selector_rank as usize;
        let predecessor = gguf
            .tensor("selector_predecessor.weight")
            .filter(|t| t.elements().is_multiple_of(rank as u64))
            .ok_or("DFlash-2 selector_predecessor.weight has invalid dimensions")?;
        config.vocab_size = u32::try_from(predecessor.elements() / rank as u64)
            .ok()
            .filter(|&v| v > 0)
            .ok_or("DFlash-2 selector vocabulary size is invalid")?;
        let t = &target.config;
        if t.hidden_size != config.hidden_size || t.vocab_size != config.vocab_size {
            return Err(
                "DFlash-2 target hidden/vocabulary dimensions do not match the draft".into(),
            );
        }
        if config
            .target_layer_ids
            .last()
            .is_none_or(|&l| l >= t.num_layers)
        {
            return Err("DFlash-2 target layer taps exceed the target model layer count".into());
        }
        if let Some(scale) = gguf.tensor("fc.scale") {
            let unit = scale.elements() == 1
                && match scale.ty {
                    GgmlType::F32 => {
                        file.bytes()[scale.offset..scale.offset + 4] == 1f32.to_le_bytes()
                    }
                    GgmlType::Bf16 => file.bytes()[scale.offset..scale.offset + 2] == [0x80, 0x3F],
                    _ => false,
                };
            if !unit {
                return Err("DFlash-2 fc.scale must be absent or a unit F32/BF16 scalar".into());
            }
        }
        file.prefault(0..file.len())
            .map_err(|e| format!("{}: {e}", path.display()))?;

        let c = config.clone();
        let (hidden, ffn) = (c.hidden_size as usize, c.intermediate_size as usize);
        let (attention, kv, head) = (c.attention_size(), c.kv_size(), c.head_dim as usize);
        let vocab = c.vocab_size as usize;
        let conv_base = hidden * c.conv_kernel_size as usize * 2;
        let dynamic = c.dynamic_coefficients();
        let mut p = Packer {
            pack: crate::pack::Packer {
                stream,
                file: &file,
                model: "DFlash-2",
                bytes: 0,
            },
            gguf: &gguf,
        };
        // The C++ packing order, one allocation per tensor.
        let fc_norm = p.vector("enc.output_norm.weight", hidden)?;
        let fc = p.bf16("fc.weight", hidden, c.target_features())?;
        let output_norm = p.vector("output_norm.weight", hidden)?;
        let selector_predecessor = p.bf16("selector_predecessor.weight", vocab, rank)?;
        let selector_successor = p.bf16("selector_successor.weight", vocab, rank)?;
        let selector_hidden = p.matrix("selector_hidden.weight", rank, hidden)?;
        let mut layers = Vec::with_capacity(c.num_layers as usize);
        for i in 0..c.num_layers {
            let n = |suffix: &str| format!("blk.{i}.{suffix}");
            layers.push(DFlashLayer {
                attn_norm: p.vector(&n("attn_norm.weight"), hidden)?,
                q: p.matrix(&n("attn_q.weight"), attention, hidden)?,
                k: p.bf16(&n("attn_k.weight"), kv, hidden)?,
                v: p.bf16(&n("attn_v.weight"), kv, hidden)?,
                output: p.matrix(&n("attn_output.weight"), hidden, attention)?,
                q_norm: p.vector(&n("attn_q_norm.weight"), head)?,
                k_norm: p.vector(&n("attn_k_norm.weight"), head)?,
                ffn_norm: p.vector(&n("ffn_norm.weight"), hidden)?,
                gate: p.matrix(&n("ffn_gate.weight"), ffn, hidden)?,
                up: p.matrix(&n("ffn_up.weight"), ffn, hidden)?,
                down: p.matrix(&n("ffn_down.weight"), hidden, ffn)?,
                attn_conv_base: p.vector(&n("attn_conv_base"), conv_base)?,
                ffn_conv_base: p.vector(&n("ffn_conv_base"), conv_base)?,
                attn_conv_proj: p.matrix(&n("attn_conv_proj.weight"), dynamic, hidden)?,
                ffn_conv_proj: p.matrix(&n("ffn_conv_proj.weight"), dynamic, hidden)?,
            });
        }
        // A BF16 target head is read by the draft from a Q8_0 copy, allocated
        // after the draft's own tensors (`QuantizeDraftHead`).
        let output = if target.output.format() == GgmlType::Bf16 {
            draft_head(stream, &target.output, vocab * hidden, &mut p.pack.bytes)?
        } else {
            target.output.clone()
        };
        let packed_bytes = p.pack.bytes;
        let message_boundaries = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|name| target.tokenizer.special(name))
            .collect();
        Ok(Self {
            config,
            fc_norm,
            fc,
            output_norm,
            selector_predecessor,
            selector_successor,
            selector_hidden,
            layers,
            token_embd: target.token_embd.clone(),
            output,
            packed_bytes,
            message_boundaries,
        })
    }
}
