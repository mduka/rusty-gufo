//! Device weights of the text model, a port of `QwenModelWeights::LoadFromGguf`
//! (`weights.cpp`) and `QwenGpuModel::CreateFromGguf` (`hip/model_loader.cpp`).
//!
//! Like the C++ loader, the whole GGUF file is copied into device memory when
//! the heap has room above its dispatch reserve, else into anonymous huge
//! pages registered with HIP (`Placement::DeviceImage`); tensors are offsets
//! into that copy.

use std::path::Path;
use std::sync::Arc;

use gufo_core::gguf::{Gguf, Tensor};
use gufo_core::qwen_tokenizer::QwenTokenizer;
use gufo_hip::{
    DeviceTensor, GgmlType, LoadedWeights, MappedFile, Pending, Placement, QuantTensor, Stream,
    WeightLoader,
};

use crate::config::ModelConfig;

/// Formats the embedding lookup decodes (`SupportsTensorType` kEmbedding).
pub(crate) const EMBEDDING: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::Bf16,
    GgmlType::Q8_0,
    GgmlType::Q4K,
    GgmlType::Q3K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Iq4Nl,
    GgmlType::Iq4Xs,
    GgmlType::Iq3S,
];

/// Projection formats: the quantized ones, whose prefill reads tiled Q8_1
/// activations (`reads_q8_activation`), and BF16 (the UD-Q8_K_XL attention
/// and LM head, BF16 conversions), which every path computes from the FP32
/// rows. F32 and Q8_K projections, which the C++ engine reads through its
/// hipBLAS/hipBLASLt routes, are not supported (no published Qwen3.5
/// artifact uses them): [`formats_supported`] declines such a model and the
/// loader rejects its tensors as an unsupported tensor type.
pub(crate) const PROJECTION: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::Bf16,
    GgmlType::Q8K,
    GgmlType::Q8_0,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Q4K,
    GgmlType::Q3K,
    GgmlType::Iq4Nl,
    GgmlType::Iq4Xs,
    GgmlType::Iq3S,
];

/// Bytes the MMQ expert kernels may read past a stacked expert tensor
/// (`tail = "4096"` in launch.toml). The tensors keep their file offsets in
/// the image of the file, so the tail is the bytes that follow them there,
/// as in the C++ engine's image.
const EXPERT_TAIL: usize = 4096;

/// Projection tensors of a Qwen3.5 GGUF, by their name suffix.
const PROJECTION_SUFFIXES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_out.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
];

/// Routed gate/up formats every expert route takes: the prefill's grouped
/// MMQ GEMM and the decode's gated per-slot GEMV (`IsMoeMmqGatedVecType`).
const EXPERT_GATE_UP: &[GgmlType] = &[GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q8_0];
/// Routed down formats every expert route takes: the grouped GEMM and the
/// per-slot GEMV (`IsMoeMmqVecType`, whose Q5_1 the grouped GEMM lacks).
const EXPERT_DOWN: &[GgmlType] = &[GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K, GgmlType::Q8_0];

/// Whether the routed experts of the `num_layers` text layers have formats
/// every route of the ported kernels takes, with an error naming the first
/// tensor and its format that one does not: gate and up in one of
/// [`EXPERT_GATE_UP`], down in [`EXPERT_DOWN`], rows of `hidden` and
/// `expert_ff` values that fill whole blocks, and an F32 router and
/// shared-expert gate. Files such as UD-Q6_K_XL (Q6_K gate/up) or UD-Q8_K_XL
/// (BF16 experts) would otherwise fail in the middle of a request: the C++
/// fallback routes for them (per-slot GEMVs of other formats, the grouped
/// BF16 GEMM, a BF16 router) are not ported. The MTP block after the text
/// layers loads apart. Reads the header only.
pub fn check_expert_formats(gguf: &Gguf<'_>, config: &ModelConfig) -> Result<(), String> {
    let refuse = |name: &str, ty: GgmlType, takes: &str| {
        Err(format!(
            "{name} is {}: the qwen35moe expert routes take {takes}",
            ty.name()
        ))
    };
    let whole_rows = |ty: GgmlType, k: u32| match ty {
        GgmlType::Q8_0 => k.is_multiple_of(64),
        _ => k.is_multiple_of(256),
    };
    for layer in 0..config.num_layers {
        let tensor = |role: &str| {
            let name = format!("blk.{layer}.{role}");
            gguf.tensor(&name).map(|t| (name, t.ty))
        };
        let gate = tensor("ffn_gate_exps.weight");
        let up = tensor("ffn_up_exps.weight");
        for (name, ty) in gate.iter().chain(&up) {
            if !EXPERT_GATE_UP.contains(ty) {
                return refuse(name, *ty, "gate/up Q4_K, Q5_K or Q8_0");
            }
            if !whole_rows(*ty, config.hidden_size) {
                return refuse(name, *ty, "rows that fill whole blocks");
            }
        }
        if let (Some((name, gate)), Some((_, up))) = (&gate, &up)
            && gate != up
        {
            return Err(format!(
                "{name} is {} and its up projection {}: the qwen35moe expert routes take one \
                 format for gate and up",
                gate.name(),
                up.name()
            ));
        }
        if let Some((name, ty)) = tensor("ffn_down_exps.weight") {
            if !EXPERT_DOWN.contains(&ty) {
                return refuse(&name, ty, "down Q4_K, Q5_K, Q6_K or Q8_0");
            }
            if !whole_rows(ty, config.expert_ff_length) {
                return refuse(&name, ty, "rows that fill whole blocks");
            }
        }
        for role in ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"] {
            if let Some((name, ty)) = tensor(role)
                && ty != GgmlType::F32
            {
                return refuse(&name, ty, "an F32 router and shared-expert gate");
            }
        }
    }
    Ok(())
}

/// Whether the loader takes every tensor format it reads from the GGUF: the
/// projections of the `num_layers` text layers (not the MTP layers after
/// them) and the LM head (or the embedding it is tied to) in [`PROJECTION`],
/// the embedding in [`EMBEDDING`], and the routed experts
/// ([`check_expert_formats`]). Reads the header only.
pub fn formats_supported(gguf: &gufo_core::gguf::Gguf<'_>, num_layers: u32) -> bool {
    if ModelConfig::from_gguf(gguf).is_ok_and(|config| check_expert_formats(gguf, &config).is_err())
    {
        return false;
    }
    let tied = gguf.tensor("output.weight").is_none();
    gguf.tensors().iter().all(|t| {
        let layer = t
            .name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(index, role)| Some((index.parse::<u32>().ok()?, role)));
        let projection = t.name == "output.weight"
            || layer.is_some_and(|(index, role)| {
                index < num_layers && PROJECTION_SUFFIXES.contains(&role)
            });
        if projection {
            PROJECTION.contains(&t.ty)
        } else if t.name == "token_embd.weight" {
            EMBEDDING.contains(&t.ty) && (!tied || PROJECTION.contains(&t.ty))
        } else {
            true
        }
    })
}

pub struct AttentionWeights {
    /// Q and gate interleaved per head: `2 * attention_size` rows.
    pub q: QuantTensor,
    pub k: QuantTensor,
    pub v: QuantTensor,
    pub output: QuantTensor,
    pub q_norm: DeviceTensor<f32>,
    pub k_norm: DeviceTensor<f32>,
}

pub struct RecurrentWeights {
    pub qkv: QuantTensor,
    pub gate: QuantTensor,
    pub a: DeviceTensor<f32>,
    pub conv1d: DeviceTensor<f32>,
    pub dt: DeviceTensor<f32>,
    pub alpha: QuantTensor,
    pub beta: QuantTensor,
    pub norm: DeviceTensor<f32>,
    pub out: QuantTensor,
}

pub enum Mixer {
    Attention(AttentionWeights),
    Recurrent(RecurrentWeights),
}

/// The routed-expert feed-forward of a `qwen35moe` layer: a softmax router
/// over `expert_count` experts whose top `expert_used_count` run as SwiGLU
/// FFNs of width `expert_ff_length`, plus a shared expert of width
/// `expert_shared_ff_length` scaled by a sigmoid gate.
pub struct MoeWeights {
    /// `expert_count` rows of `hidden` (`ffn_gate_inp`).
    pub router: QuantTensor,
    /// The shared expert's gate logit row, `hidden` values (`ffn_gate_inp_shexp`).
    pub shared_gate: QuantTensor,
    /// Stacked experts: `expert_count` matrices of `expert_ff_length` rows,
    /// the view extended by the 4096 bytes the MMQ kernels may read past it.
    pub gate_exps: QuantTensor,
    pub up_exps: QuantTensor,
    /// Stacked experts: `expert_count` matrices of `hidden` rows, extended
    /// like the gate and up experts.
    pub down_exps: QuantTensor,
    pub gate_shexp: QuantTensor,
    pub up_shexp: QuantTensor,
    pub down_shexp: QuantTensor,
}

pub struct LayerWeights {
    pub attn_norm: DeviceTensor<f32>,
    pub ffn_norm: DeviceTensor<f32>,
    pub mixer: Mixer,
    pub moe: MoeWeights,
}

pub struct Weights {
    pub config: ModelConfig,
    /// The tokenizer of the same file.
    pub tokenizer: QwenTokenizer,
    pub token_embd: QuantTensor,
    pub output_norm: DeviceTensor<f32>,
    /// The LM head; the embedding table when the file ties them.
    pub output: QuantTensor,
    pub layers: Vec<LayerWeights>,
}

/// A tensor queued with the loader.
enum Queued {
    Quant(Pending<u8>, GgmlType),
    Float(Pending<f32>),
}

impl Queued {
    fn quant(self, loaded: &LoadedWeights) -> QuantTensor {
        match self {
            Self::Quant(pending, format) => QuantTensor::new(loaded.get(pending), format),
            Self::Float(_) => unreachable!("queued as f32"),
        }
    }

    fn float(self, loaded: &LoadedWeights) -> DeviceTensor<f32> {
        match self {
            Self::Float(pending) => loaded.get(pending),
            Self::Quant(..) => unreachable!("queued as quantized"),
        }
    }
}

struct Binder<'g, 'l> {
    gguf: &'g Gguf<'g>,
    file: &'g Arc<MappedFile>,
    loader: WeightLoader<'l>,
}

impl Binder<'_, '_> {
    /// The first of `names` present in the file.
    fn find(&self, names: &[String]) -> Option<&Tensor<'_>> {
        names.iter().find_map(|name| self.gguf.tensor(name))
    }

    /// Queues the first of `names` after the C++ `ValidateTensor` checks:
    /// present, a supported format, `elements` values, whole rows of
    /// `row` elements (0: no row check). The GGUF parser has checked that the
    /// payload lies within the file.
    fn queue(
        &mut self,
        names: &[String],
        formats: &[GgmlType],
        elements: u64,
        row: u64,
    ) -> Result<Queued, String> {
        let float = formats == [GgmlType::F32];
        self.queue_as(names, formats, elements, row, float, 0)
    }

    /// [`Self::queue`] for stacked experts, whose view extends
    /// [`EXPERT_TAIL`] bytes into what follows them in the file.
    fn experts(&mut self, names: &[String], elements: u64, row: u64) -> Result<Queued, String> {
        self.queue_as(names, PROJECTION, elements, row, false, EXPERT_TAIL)
    }

    /// [`Self::queue`] as a format-tagged tensor even when it is F32: the
    /// router launchers take the router's format.
    fn tagged(
        &mut self,
        names: &[String],
        formats: &[GgmlType],
        elements: u64,
    ) -> Result<Queued, String> {
        self.queue_as(names, formats, elements, 0, false, 0)
    }

    fn queue_as(
        &mut self,
        names: &[String],
        formats: &[GgmlType],
        elements: u64,
        row: u64,
        float: bool,
        tail: usize,
    ) -> Result<Queued, String> {
        let name = &names[0];
        let tensor = self
            .find(names)
            .ok_or_else(|| format!("Missing required Qwen tensor: {name}"))?;
        let (ty, count, offset, bytes) =
            (tensor.ty, tensor.elements(), tensor.offset, tensor.bytes);
        if !formats.contains(&ty) {
            return Err(format!("Unsupported tensor type for {name}: {}", ty.name()));
        }
        if count != elements {
            return Err(format!(
                "Tensor shape mismatch for {name}: expected {elements} elements, found {count}"
            ));
        }
        let (block, _) = ty.block();
        if row != 0 && !row.is_multiple_of(block as u64) {
            return Err(format!(
                "Quantized tensor row is not block aligned: {name} (row elements {row})"
            ));
        }
        let error = |e| format!("{name}: {e}");
        if offset + bytes + tail > self.file.len() {
            return Err(format!(
                "{name}: the expert kernels read {tail} bytes past the tensor, beyond the file"
            ));
        }
        Ok(if float && ty == GgmlType::F32 {
            Queued::Float(
                self.loader
                    .add::<f32>(self.file, offset, bytes / 4)
                    .map_err(error)?,
            )
        } else {
            Queued::Quant(
                self.loader
                    .add::<u8>(self.file, offset, bytes + tail)
                    .map_err(error)?,
                ty,
            )
        })
    }

    fn norm(&mut self, names: &[String], elements: u32) -> Result<Queued, String> {
        self.queue(names, &[GgmlType::F32], u64::from(elements), 0)
    }

    fn projection(&mut self, names: &[String], rows: u32, cols: u32) -> Result<Queued, String> {
        self.queue(
            names,
            PROJECTION,
            u64::from(rows) * u64::from(cols),
            u64::from(cols),
        )
    }
}

/// `name`, then its alternatives.
fn named(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| (*n).to_string()).collect()
}

/// Tensor names of layer `layer`, preferred first.
fn blk(layer: u32, suffixes: &[&str]) -> Vec<String> {
    suffixes
        .iter()
        .map(|s| format!("blk.{layer}.{s}"))
        .collect()
}

enum QueuedMixer {
    Attention([Queued; 6]),
    Recurrent([Queued; 9]),
}

struct QueuedLayer {
    attn_norm: Queued,
    ffn_norm: Queued,
    mixer: QueuedMixer,
    moe: [Queued; 8],
}

impl Weights {
    /// Loads a single-file GGUF (split files are not supported yet).
    pub fn load(stream: &Stream, path: &Path) -> Result<Self, String> {
        let file = MappedFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let gguf = Gguf::parse(file.bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
        if gguf.u32("split.count").is_some_and(|n| n > 1) {
            return Err("split GGUF files are not supported by the Rust Qwen loader yet".into());
        }
        let config = ModelConfig::from_gguf(&gguf)?;
        // Before anything is copied: a request would fail on these routes.
        check_expert_formats(&gguf, &config)?;
        let tokenizer = QwenTokenizer::from_gguf(&gguf)?;
        let mut b = Binder {
            gguf: &gguf,
            file: &file,
            loader: WeightLoader::new(stream, Placement::DeviceImage),
        };
        let c = &config;
        let (hidden, vocab) = (c.hidden_size, c.vocab_size);
        let embd = b.queue(
            &named(&["token_embd.weight"]),
            EMBEDDING,
            u64::from(vocab) * u64::from(hidden),
            u64::from(hidden),
        )?;
        let output_norm = b.norm(&named(&["output_norm.weight"]), hidden)?;
        // A missing LM head shares the embedding table (tied weights).
        let output = if b.find(&named(&["output.weight"])).is_some() {
            Some(b.projection(&named(&["output.weight"]), vocab, hidden)?)
        } else if PROJECTION.contains(
            &b.find(&named(&["token_embd.weight"]))
                .map_or(GgmlType::F32, |t| t.ty),
        ) {
            None
        } else {
            return Err("Unsupported projection tensor type for the tied output.weight".into());
        };

        let mut layers = Vec::with_capacity(c.num_layers as usize);
        for i in 0..c.num_layers {
            let attn_norm = blk(i, &["attn_norm.weight", "input_norm.weight"]);
            let ffn_norm = blk(
                i,
                &[
                    "ffn_norm.weight",
                    "post_attention_norm.weight",
                    "attn_post_norm.weight",
                ],
            );
            let q_names = blk(i, &["attn_q.weight", "wq.weight"]);
            let full = b.find(&q_names).is_some();
            if full != c.is_full_attention(i) {
                return Err(format!(
                    "Unexpected Qwen layer kind at blk.{i}: expected {}",
                    if c.is_full_attention(i) {
                        "full attention"
                    } else {
                        "Gated DeltaNet"
                    }
                ));
            }
            let attn_norm = b.norm(&attn_norm, hidden)?;
            let ffn_norm = b.norm(&ffn_norm, hidden)?;
            let (experts, expert_ff, shared_ff) = (
                c.expert_count,
                c.expert_ff_length,
                c.expert_shared_ff_length,
            );
            // The router and the shared-expert gate are F32 ("normalization"
            // role in C++); the stacked experts are projections with rows of
            // `hidden` (gate, up) and `expert_ff` (down) values.
            let moe = [
                b.tagged(
                    &blk(i, &["ffn_gate_inp.weight"]),
                    &[GgmlType::F32],
                    u64::from(hidden) * u64::from(experts),
                )?,
                b.tagged(
                    &blk(i, &["ffn_gate_inp_shexp.weight"]),
                    &[GgmlType::F32],
                    u64::from(hidden),
                )?,
                b.experts(
                    &blk(i, &["ffn_gate_exps.weight"]),
                    u64::from(hidden) * u64::from(expert_ff) * u64::from(experts),
                    u64::from(hidden),
                )?,
                b.experts(
                    &blk(i, &["ffn_up_exps.weight"]),
                    u64::from(hidden) * u64::from(expert_ff) * u64::from(experts),
                    u64::from(hidden),
                )?,
                b.experts(
                    &blk(i, &["ffn_down_exps.weight"]),
                    u64::from(expert_ff) * u64::from(hidden) * u64::from(experts),
                    u64::from(expert_ff),
                )?,
                b.projection(&blk(i, &["ffn_gate_shexp.weight"]), shared_ff, hidden)?,
                b.projection(&blk(i, &["ffn_up_shexp.weight"]), shared_ff, hidden)?,
                b.projection(&blk(i, &["ffn_down_shexp.weight"]), hidden, shared_ff)?,
            ];
            let mixer = if full {
                let (attention, kv) = (c.attention_size(), c.kv_size());
                QueuedMixer::Attention([
                    b.projection(&q_names, 2 * attention, hidden)?,
                    b.projection(&blk(i, &["attn_k.weight", "wk.weight"]), kv, hidden)?,
                    b.projection(&blk(i, &["attn_v.weight", "wv.weight"]), kv, hidden)?,
                    b.projection(
                        &blk(i, &["attn_output.weight", "attn_out.weight", "wo.weight"]),
                        hidden,
                        attention,
                    )?,
                    b.norm(&blk(i, &["attn_q_norm.weight"]), c.head_dim)?,
                    b.norm(&blk(i, &["attn_k_norm.weight"]), c.head_dim)?,
                ])
            } else {
                let (qkv, inner, rank) = (c.ssm_qkv_size(), c.ssm_inner_size, c.ssm_time_step_rank);
                QueuedMixer::Recurrent([
                    b.projection(&blk(i, &["attn_qkv.weight", "wqkv.weight"]), qkv, hidden)?,
                    b.projection(
                        &blk(i, &["attn_gate.weight", "wqkv_gate.weight"]),
                        inner,
                        hidden,
                    )?,
                    b.norm(&blk(i, &["ssm_a", "ssm_a.weight"]), rank)?,
                    b.norm(
                        &blk(i, &["ssm_conv1d.weight", "ssm_conv1d"]),
                        qkv * c.ssm_conv_kernel,
                    )?,
                    b.norm(&blk(i, &["ssm_dt.bias", "ssm_dt.weight", "ssm_dt"]), rank)?,
                    b.projection(&blk(i, &["ssm_alpha.weight"]), rank, hidden)?,
                    b.projection(&blk(i, &["ssm_beta.weight"]), rank, hidden)?,
                    b.norm(&blk(i, &["ssm_norm.weight"]), c.ssm_value_size())?,
                    b.projection(&blk(i, &["ssm_out.weight"]), hidden, inner)?,
                ])
            };
            layers.push(QueuedLayer {
                attn_norm,
                ffn_norm,
                mixer,
                moe,
            });
        }

        let loaded = b
            .loader
            .finish()
            .map_err(|e| format!("Qwen weights: {e}"))?;
        let token_embd = embd.quant(&loaded);
        let output = match output {
            Some(queued) => queued.quant(&loaded),
            None => token_embd.clone(),
        };
        let layers = layers
            .into_iter()
            .map(|l| {
                let [
                    router,
                    shared_gate,
                    gate_exps,
                    up_exps,
                    down_exps,
                    gate_shexp,
                    up_shexp,
                    down_shexp,
                ] = l.moe;
                let mixer = match l.mixer {
                    QueuedMixer::Attention([q, k, v, o, qn, kn]) => {
                        Mixer::Attention(AttentionWeights {
                            q: q.quant(&loaded),
                            k: k.quant(&loaded),
                            v: v.quant(&loaded),
                            output: o.quant(&loaded),
                            q_norm: qn.float(&loaded),
                            k_norm: kn.float(&loaded),
                        })
                    }
                    QueuedMixer::Recurrent([qkv, gate, a, conv, dt, alpha, beta, norm, out]) => {
                        Mixer::Recurrent(RecurrentWeights {
                            qkv: qkv.quant(&loaded),
                            gate: gate.quant(&loaded),
                            a: a.float(&loaded),
                            conv1d: conv.float(&loaded),
                            dt: dt.float(&loaded),
                            alpha: alpha.quant(&loaded),
                            beta: beta.quant(&loaded),
                            norm: norm.float(&loaded),
                            out: out.quant(&loaded),
                        })
                    }
                };
                LayerWeights {
                    attn_norm: l.attn_norm.float(&loaded),
                    ffn_norm: l.ffn_norm.float(&loaded),
                    mixer,
                    moe: MoeWeights {
                        router: router.quant(&loaded),
                        shared_gate: shared_gate.quant(&loaded),
                        gate_exps: gate_exps.quant(&loaded),
                        up_exps: up_exps.quant(&loaded),
                        down_exps: down_exps.quant(&loaded),
                        gate_shexp: gate_shexp.quant(&loaded),
                        up_shexp: up_shexp.quant(&loaded),
                        down_shexp: down_shexp.quant(&loaded),
                    },
                }
            })
            .collect();
        Ok(Self {
            config,
            tokenizer,
            token_embd,
            output_norm: output_norm.float(&loaded),
            output,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// A GGUF header of a four-layer `qwen35moe` model (2048 wide, eight
    /// experts of 512) with an MTP block after it: its router and expert
    /// tensors, the formats `overrides` give by name, no data.
    fn header(overrides: &[(&str, GgmlType)]) -> Vec<u8> {
        fn string(out: &mut Vec<u8>, value: &str) {
            out.extend_from_slice(&(value.len() as u64).to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        }
        let u32s = [
            ("qwen35moe.block_count", 5),
            ("qwen35moe.nextn_predict_layers", 1),
            ("qwen35moe.embedding_length", 2048),
            ("qwen35moe.attention.head_count", 16),
            ("qwen35moe.attention.head_count_kv", 2),
            ("qwen35moe.attention.key_length", 256),
            ("qwen35moe.expert_count", 8),
            ("qwen35moe.expert_used_count", 2),
            ("qwen35moe.expert_feed_forward_length", 512),
            ("qwen35moe.expert_shared_feed_forward_length", 512),
        ];
        let mut tensors = Vec::new();
        for layer in 0..5 {
            let mut add = |role: &str, ty: GgmlType, dims: &[u64]| {
                let name = format!("blk.{layer}.{role}");
                let ty = overrides
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map_or(ty, |&(_, t)| t);
                tensors.push((name, ty, dims.to_vec()));
            };
            add("ffn_gate_inp.weight", GgmlType::F32, &[2048, 8]);
            add("ffn_gate_inp_shexp.weight", GgmlType::F32, &[2048]);
            add("ffn_gate_exps.weight", GgmlType::Q4K, &[2048, 512, 8]);
            add("ffn_up_exps.weight", GgmlType::Q4K, &[2048, 512, 8]);
            add("ffn_down_exps.weight", GgmlType::Q6K, &[512, 2048, 8]);
        }
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(u32s.len() as u64 + 1).to_le_bytes());
        string(&mut out, "general.architecture");
        out.extend_from_slice(&8u32.to_le_bytes());
        string(&mut out, "qwen35moe");
        for (key, value) in u32s {
            string(&mut out, key);
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&(value as u32).to_le_bytes());
        }
        for (name, ty, dims) in &tensors {
            string(&mut out, name);
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&ty.id().to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
        }
        out
    }

    fn check(overrides: &[(&str, GgmlType)]) -> Result<(), String> {
        let bytes = header(overrides);
        let gguf = Gguf::parse_prefix(&bytes, 1 << 40).unwrap();
        let config = ModelConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.num_layers, 4);
        check_expert_formats(&gguf, &config)
    }

    /// Published UD-Q4_K_M and Q8_0 expert formats load; others are refused
    /// up front with the tensor and its format; the MTP block is not checked.
    #[test]
    fn expert_formats_are_checked_before_loading() {
        assert_eq!(check(&[]), Ok(()));
        let q8 = GgmlType::Q8_0;
        assert_eq!(
            check(&[
                ("blk.1.ffn_gate_exps.weight", q8),
                ("blk.1.ffn_up_exps.weight", q8),
                ("blk.1.ffn_down_exps.weight", q8),
                ("blk.2.ffn_down_exps.weight", GgmlType::Q5K),
                ("blk.4.ffn_gate_inp.weight", GgmlType::Bf16),
                ("blk.4.ffn_gate_exps.weight", GgmlType::Bf16),
            ]),
            Ok(()),
            "mixed supported formats and an MTP block of other formats"
        );
        let error = |overrides: &[(&str, GgmlType)]| check(overrides).unwrap_err();
        let q6 = error(&[
            ("blk.2.ffn_gate_exps.weight", GgmlType::Q6K),
            ("blk.2.ffn_up_exps.weight", GgmlType::Q6K),
        ]);
        assert!(
            q6.starts_with("blk.2.ffn_gate_exps.weight is Q6_K")
                && q6.contains("Q4_K, Q5_K or Q8_0"),
            "{q6}"
        );
        let bf16 = error(&[("blk.3.ffn_down_exps.weight", GgmlType::Bf16)]);
        assert!(
            bf16.starts_with("blk.3.ffn_down_exps.weight is BF16"),
            "{bf16}"
        );
        let router = error(&[("blk.0.ffn_gate_inp.weight", GgmlType::Bf16)]);
        assert!(router.contains("an F32 router"), "{router}");
        let mixed = error(&[("blk.1.ffn_up_exps.weight", GgmlType::Q5K)]);
        assert!(mixed.contains("one format for gate and up"), "{mixed}");
        let q51 = error(&[("blk.0.ffn_down_exps.weight", GgmlType::Q5_1)]);
        assert!(q51.contains("down Q4_K, Q5_K, Q6_K or Q8_0"), "{q51}");
        // `supports` sees the same refusal through the header check.
        let bytes = header(&[("blk.3.ffn_down_exps.weight", GgmlType::Bf16)]);
        assert!(!formats_supported(
            &Gguf::parse_prefix(&bytes, 1 << 40).unwrap(),
            4
        ));
        let bytes = header(&[]);
        assert!(formats_supported(
            &Gguf::parse_prefix(&bytes, 1 << 40).unwrap(),
            4
        ));
    }
}
