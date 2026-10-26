//! Device weights of the text model, a port of `QwenModelWeights::LoadFromGguf`
//! (`weights.cpp`) and `QwenGpuModel::CreateFromGguf` (`hip/model_loader.cpp`).
//!
//! Like the C++ loader, the whole GGUF file is copied into anonymous huge
//! pages and registered with HIP (`Placement::HostCopy`); tensors are offsets
//! into that copy and the kernels read them through the mapping.

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
/// rows. The C++ engine also accepts F32 and Q8_K projections through its
/// hipBLAS/hipBLASLt routes, which the Rust port does not carry (no published
/// Qwen3.5 artifact uses them; `runner::supports` leaves them to C++).
pub(crate) const PROJECTION: &[GgmlType] = &[
    GgmlType::Bf16,
    GgmlType::Q8_0,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Q4K,
    GgmlType::Q3K,
    GgmlType::Iq4Nl,
    GgmlType::Iq4Xs,
    GgmlType::Iq3S,
];

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
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// Whether the loader takes every tensor format it reads from the GGUF: the
/// projections of the `num_layers` text layers (not the MTP layers after
/// them) and the LM head (or the embedding it is tied to) in [`PROJECTION`],
/// the embedding in [`EMBEDDING`]. Reads the header only.
pub fn formats_supported(gguf: &gufo_core::gguf::Gguf<'_>, num_layers: u32) -> bool {
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

pub struct LayerWeights {
    pub attn_norm: DeviceTensor<f32>,
    pub ffn_norm: DeviceTensor<f32>,
    pub mixer: Mixer,
    pub ffn_gate: QuantTensor,
    pub ffn_up: QuantTensor,
    pub ffn_down: QuantTensor,
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
        Ok(if ty == GgmlType::F32 && formats == [GgmlType::F32] {
            Queued::Float(
                self.loader
                    .add::<f32>(self.file, offset, bytes / 4)
                    .map_err(error)?,
            )
        } else {
            Queued::Quant(
                self.loader
                    .add::<u8>(self.file, offset, bytes)
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
    ffn: [Queued; 3],
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
        let tokenizer = QwenTokenizer::from_gguf(&gguf)?;
        let mut b = Binder {
            gguf: &gguf,
            file: &file,
            loader: WeightLoader::new(stream, Placement::HostCopy),
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
            let ffn = [
                b.projection(&blk(i, &["ffn_gate.weight"]), c.intermediate_size, hidden)?,
                b.projection(&blk(i, &["ffn_up.weight"]), c.intermediate_size, hidden)?,
                b.projection(&blk(i, &["ffn_down.weight"]), hidden, c.intermediate_size)?,
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
                ffn,
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
                let [gate, up, down] = l.ffn;
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
                    ffn_gate: gate.quant(&loaded),
                    ffn_up: up.quant(&loaded),
                    ffn_down: down.quant(&loaded),
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
