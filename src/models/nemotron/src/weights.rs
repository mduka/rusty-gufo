//! Binding of the trunk, drafter and vision projector tensors
//! (`weights.cpp`): every tensor is checked for its exact shape and an
//! accepted format, located in its mapped GGUF shard, and paired with its
//! per-tensor `.scale` companion when the file carries one. Payloads stay
//! mapped and untouched.

use gufo_core::gguf::{Gguf, Tensor};
use gufo_formats::GgmlType;

use crate::config::{BlockType, Config, DrafterConfig, VisionConfig};

/// One mapped GGUF file of an artifact: a whole file or one shard of a split
/// set, with the header parsed from its bytes. Metadata is read from the
/// first shard, as the C++ reader of a split set does.
pub struct GgufShard<'a> {
    pub gguf: Gguf<'a>,
    /// The whole mapped file.
    pub bytes: &'a [u8],
}

impl<'a> GgufShard<'a> {
    /// Parses the header of a whole mapped file.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        Ok(Self {
            gguf: Gguf::parse(bytes)?,
            bytes,
        })
    }
}

/// The first shard holding `name`, and its index.
pub fn find_tensor<'s, 'a>(
    shards: &'s [GgufShard<'a>],
    name: &str,
) -> Option<(u32, &'s Tensor<'a>)> {
    shards
        .iter()
        .enumerate()
        .find_map(|(i, shard)| shard.gguf.tensor(name).map(|t| (i as u32, t)))
}

/// Tensors of every shard.
pub fn tensor_count(shards: &[GgufShard<'_>]) -> usize {
    shards.iter().map(|s| s.gguf.tensors().len()).sum()
}

/// Storage geometry of the formats this runtime decodes, as (elements,
/// bytes) per block. Unknown types get a zero block, which every validation
/// rejects (`FormatOf`).
pub fn format_of(ty: GgmlType) -> (u64, u64) {
    match ty {
        GgmlType::F32 => (1, 4),
        GgmlType::F16 | GgmlType::Bf16 => (1, 2),
        GgmlType::Q8_0 => (32, 34),
        GgmlType::Iq2Xxs => (256, 66),
        GgmlType::Nvfp4 => (64, 36),
        _ => (0, 0),
    }
}

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole
/// matrices. The bytes live in the shard's mapping ([`TensorRef::bytes`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorRef {
    pub ty: GgmlType,
    pub cols: u64,
    pub rows: u64,
    pub experts: u64,
    /// Per-tensor output scale from the `<name>.scale` companion (NVFP4).
    pub scale: f32,
    /// Byte offset inside the owning shard.
    pub file_offset: u64,
    /// Index of the owning shard.
    pub shard: u32,
    pub name: String,
    /// False for an unbound tensor (the C++ null `data`).
    pub bound: bool,
}

impl Default for TensorRef {
    fn default() -> Self {
        Self {
            ty: GgmlType::F32,
            cols: 0,
            rows: 1,
            experts: 1,
            scale: 1.0,
            file_offset: 0,
            shard: 0,
            name: String::new(),
            bound: false,
        }
    }
}

impl TensorRef {
    pub fn is_empty(&self) -> bool {
        !self.bound
    }

    /// Encoded bytes of one row (`cols` elements) in this format.
    pub fn row_bytes(&self) -> usize {
        let (block, bytes) = format_of(self.ty);
        if block == 0 || !self.cols.is_multiple_of(block) {
            return 0;
        }
        (self.cols / block * bytes) as usize
    }

    pub fn size_bytes(&self) -> usize {
        self.row_bytes() * self.rows as usize * self.experts as usize
    }

    /// Byte offset of one stacked expert matrix within the tensor.
    pub fn expert_offset(&self, e: u64) -> usize {
        self.row_bytes() * self.rows as usize * e as usize
    }

    /// The tensor's payload in its shard's mapping.
    ///
    /// # Panics
    /// When `shards` is not the set the tensor was bound from.
    pub fn bytes<'a>(&self, shards: &[GgufShard<'a>]) -> &'a [u8] {
        let start = self.file_offset as usize;
        &shards[self.shard as usize].bytes[start..start + self.size_bytes()]
    }

    /// One stacked expert matrix's payload.
    pub fn expert<'a>(&self, shards: &[GgufShard<'a>], e: u64) -> &'a [u8] {
        let matrix = self.row_bytes() * self.rows as usize;
        &self.bytes(shards)[self.expert_offset(e)..][..matrix]
    }

    /// The payload as F32 values (F32 tensors only).
    pub fn f32_values(&self, shards: &[GgufShard<'_>]) -> Vec<f32> {
        self.bytes(shards)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerWeights {
    pub ty: BlockType,
    /// `[hidden]` pre-mixer RMSNorm gamma, on every block.
    pub norm: TensorRef,

    // Mamba-2.
    /// `[hidden -> z|x|B|C|dt]`
    pub ssm_in: TensorRef,
    /// `[conv_kernel, channels]`, oldest tap first
    pub ssm_conv1d: TensorRef,
    /// `[channels]`
    pub ssm_conv1d_b: TensorRef,
    /// `[heads]` softplus bias
    pub ssm_dt_b: TensorRef,
    /// `[1, heads]` = -exp(A_log)
    pub ssm_a: TensorRef,
    /// `[1, heads]` skip connection
    pub ssm_d: TensorRef,
    /// `[inner / groups, groups]` grouped RMSNorm gamma
    pub ssm_norm: TensorRef,
    /// `[inner -> hidden]`
    pub ssm_out: TensorRef,

    // NoPE GQA.
    /// `[hidden -> heads * head_dim]`
    pub attn_q: TensorRef,
    /// `[hidden -> kv_heads * head_dim]`
    pub attn_k: TensorRef,
    /// `[hidden -> kv_heads * head_dim]`
    pub attn_v: TensorRef,
    /// `[heads * head_dim -> hidden]`
    pub attn_out: TensorRef,

    // LatentMoE.
    /// Routed experts of this block.
    pub num_experts: u32,
    /// `[hidden -> experts]` F32 logits
    pub router: TensorRef,
    /// `[experts]` selection-only score bias
    pub router_bias: TensorRef,
    /// `[hidden -> latent]`
    pub latent_down: TensorRef,
    /// `[latent -> hidden]`
    pub latent_up: TensorRef,
    /// `[latent -> expert_ff] x experts`
    pub up_exps: TensorRef,
    /// `[expert_ff -> latent] x experts`
    pub down_exps: TensorRef,
    /// `[hidden -> shared_ff]`
    pub up_shexp: TensorRef,
    /// `[shared_ff -> hidden]`
    pub down_shexp: TensorRef,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelWeights {
    pub config: Config,
    /// `[hidden -> vocab]`
    pub token_embd: TensorRef,
    /// `[hidden]`
    pub output_norm: TensorRef,
    /// `[hidden -> vocab]`
    pub output: TensorRef,
    pub layers: Vec<LayerWeights>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DrafterLayerWeights {
    /// `[hidden]`
    pub attn_norm: TensorRef,
    /// `[hidden, kernel, 2 sides]` static taps
    pub attn_conv_base: TensorRef,
    /// `[hidden -> 2 * kernel * groups]` BF16
    pub attn_conv_proj: TensorRef,
    /// `[hidden -> heads * head_dim]` BF16
    pub attn_q: TensorRef,
    /// `[hidden -> kv_heads * head_dim]` BF16
    pub attn_k: TensorRef,
    /// `[hidden -> kv_heads * head_dim]` BF16
    pub attn_v: TensorRef,
    /// `[head_dim]`
    pub attn_q_norm: TensorRef,
    /// `[head_dim]`
    pub attn_k_norm: TensorRef,
    /// `[heads * head_dim -> hidden]` BF16
    pub attn_out: TensorRef,
    /// `[heads]` softmax sink logits
    pub attn_sinks: TensorRef,
    /// `[hidden]`
    pub ffn_norm: TensorRef,
    /// `[hidden, kernel, 2 sides]`
    pub ffn_conv_base: TensorRef,
    /// `[hidden -> 2 * kernel * groups]` BF16
    pub ffn_conv_proj: TensorRef,
    /// `[hidden -> ff]` NVFP4
    pub ffn_gate: TensorRef,
    /// `[hidden -> ff]` NVFP4
    pub ffn_up: TensorRef,
    /// `[ff -> hidden]` NVFP4
    pub ffn_down: TensorRef,
}

/// One LiLiCorr transformer layer over the candidate nodes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LiliCorrLayerWeights {
    /// `[lc_hidden]`
    pub attn_norm: TensorRef,
    pub mlp_norm: TensorRef,
    /// `[lc_hidden -> 3 lc_hidden]` F32
    pub in_proj: TensorRef,
    pub in_proj_b: TensorRef,
    /// `[lc_hidden -> lc_hidden]` NVFP4
    pub out_proj: TensorRef,
    pub out_proj_b: TensorRef,
    /// `[lc_hidden -> lc_mlp]` NVFP4
    pub mlp_up: TensorRef,
    pub mlp_up_b: TensorRef,
    /// `[lc_mlp -> lc_hidden]` NVFP4
    pub mlp_down: TensorRef,
    pub mlp_down_b: TensorRef,
}

/// The LiLiCorr DFlash drafter. It has no embedding or head of its own and
/// uses the trunk's.
#[derive(Debug, Clone, PartialEq)]
pub struct DrafterWeights {
    pub config: DrafterConfig,
    /// `[features -> hidden]` NVFP4
    pub fc: TensorRef,
    /// `[hidden]` after fc
    pub enc_norm: TensorRef,
    /// `[hidden]` before the trunk's head
    pub output_norm: TensorRef,
    pub layers: Vec<DrafterLayerWeights>,
    // LiLiCorr.
    /// `[hidden -> lc_hidden]` NVFP4
    pub token_proj: TensorRef,
    pub token_proj_b: TensorRef,
    /// `[hidden -> lc_hidden]` NVFP4
    pub pass_proj: TensorRef,
    pub pass_proj_b: TensorRef,
    /// `[hidden -> lc_hidden]` NVFP4
    pub context_proj: TensorRef,
    pub context_proj_b: TensorRef,
    /// `[5]` LayerNorm affine
    pub feature_norm: TensorRef,
    pub feature_norm_b: TensorRef,
    /// `[5 -> lc_hidden]` F32
    pub feature_up: TensorRef,
    pub feature_up_b: TensorRef,
    /// `[lc_hidden -> lc_hidden]` F32
    pub feature_down: TensorRef,
    pub feature_down_b: TensorRef,
    /// `[lc_hidden, slots]`
    pub slot_embd: TensorRef,
    /// `[lc_hidden, top_k]`
    pub rank_embd: TensorRef,
    /// `[nodes, nodes, lc_heads]`, key fastest
    pub attention_bias: TensorRef,
    pub lc_layers: Vec<LiliCorrLayerWeights>,
    /// `[lc_hidden]`
    pub lc_output_norm: TensorRef,
    pub anchor_norm: TensorRef,
    /// `[3 lc_hidden -> lc_hidden]` F32
    pub factor_in: TensorRef,
    pub factor_in_b: TensorRef,
    /// `[lc_hidden -> factor]` F32
    pub out_head: TensorRef,
    pub out_head_b: TensorRef,
    /// `[lc_hidden -> factor]` F32
    pub in_head: TensorRef,
    pub in_head_b: TensorRef,
    /// `[lc_hidden -> factor]` F32
    pub anchor_head: TensorRef,
    pub anchor_head_b: TensorRef,
}

/// One ViT block: pre-norm attention and GELU MLP, each with a residual add.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisionLayerWeights {
    /// `[width]` LayerNorm affine
    pub norm1: TensorRef,
    pub norm1_b: TensorRef,
    /// `[width -> width]` BF16, F32 bias
    pub q: TensorRef,
    pub q_b: TensorRef,
    pub k: TensorRef,
    pub k_b: TensorRef,
    pub v: TensorRef,
    pub v_b: TensorRef,
    pub out: TensorRef,
    pub out_b: TensorRef,
    pub norm2: TensorRef,
    pub norm2_b: TensorRef,
    /// `[width -> ff_size]`
    pub up: TensorRef,
    pub up_b: TensorRef,
    /// `[ff_size -> width]`
    pub down: TensorRef,
    pub down_b: TensorRef,
}

/// The vision projector's tensors. Every value is a BF16 number; the tables
/// and vectors are stored as F32.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionWeights {
    pub config: VisionConfig,
    /// `[PatchValues -> width]` patch projection without bias; a patch is
    /// laid out channel-major, then by row and column.
    pub patch: TensorRef,
    /// `[width] x side²` rows, row-major over the grid
    pub position: TensorRef,
    /// `[width] x prefix_tokens` rows
    pub prefix: TensorRef,
    pub layers: Vec<VisionLayerWeights>,
    /// `[MergedWidth]` RMSNorm gamma
    pub projector_norm: TensorRef,
    /// `[MergedWidth -> projector_hidden]` BF16
    pub projector_up: TensorRef,
    /// `[projector_hidden -> projection]` BF16
    pub projector_down: TensorRef,
}

const F32: &[GgmlType] = &[GgmlType::F32];
const BF16: &[GgmlType] = &[GgmlType::Bf16];
const FP4: &[GgmlType] = &[GgmlType::Nvfp4];

/// Tensor binding that keeps the first error, as the C++ `Binder`.
struct Binder<'s, 'a> {
    shards: &'s [GgufShard<'a>],
    error: Option<String>,
}

impl Binder<'_, '_> {
    fn ok(&self) -> bool {
        self.error.is_none()
    }

    fn fail(&mut self, message: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(message.into());
        }
    }

    /// Binds `name` with the exact shape and one of the accepted formats. A
    /// `.weight` tensor picks up the scalar F32 `.scale` companion when the
    /// file carries one; llama.cpp multiplies the matmul output by it.
    fn get(
        &mut self,
        name: &str,
        cols: u64,
        rows: u64,
        experts: u64,
        types: &[GgmlType],
    ) -> TensorRef {
        let Some((shard, info)) = find_tensor(self.shards, name) else {
            self.fail(format!("missing tensor {name}"));
            return TensorRef::default();
        };
        let d = &info.dims;
        let got_cols = d.first().copied().unwrap_or(1);
        let got_rows = d.get(1).copied().unwrap_or(1);
        let got_experts = d.get(2).copied().unwrap_or(1);
        if got_cols != cols || got_rows != rows || got_experts != experts || d.len() > 3 {
            self.fail(format!(
                "tensor {name} has shape [{got_cols}, {got_rows}, {got_experts}], expected [{cols}, {rows}, {experts}]"
            ));
            return TensorRef::default();
        }
        self.finish(shard, info, name, cols, rows, experts, types)
    }

    /// A tensor of exactly `dims` whose leading dimensions flatten into the
    /// contiguous reduction dimension and whose last one counts the rows.
    fn get_flat(&mut self, name: &str, dims: &[u64], types: &[GgmlType]) -> TensorRef {
        let Some((shard, info)) = find_tensor(self.shards, name) else {
            self.fail(format!("missing tensor {name}"));
            return TensorRef::default();
        };
        if info.dims.as_slice() != dims {
            self.fail(format!("tensor {name} has an unexpected shape"));
            return TensorRef::default();
        }
        let cols: u64 = dims[..dims.len() - 1].iter().product();
        self.finish(shard, info, name, cols, dims[dims.len() - 1], 1, types)
    }

    /// Checks the format and payload of a tensor of the given shape.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &mut self,
        shard: u32,
        info: &Tensor<'_>,
        name: &str,
        cols: u64,
        rows: u64,
        experts: u64,
        types: &[GgmlType],
    ) -> TensorRef {
        let type_ok = types.contains(&info.ty);
        let (block, _) = format_of(info.ty);
        if !type_ok || block == 0 || !cols.is_multiple_of(block) {
            self.fail(format!(
                "tensor {name} has unsupported format {}",
                info.ty.name()
            ));
            return TensorRef::default();
        }
        let mut t = TensorRef {
            ty: info.ty,
            cols,
            rows,
            experts,
            scale: 1.0,
            file_offset: info.offset as u64,
            shard,
            name: info.name.to_string(),
            bound: true,
        };
        // The payload must lie inside its shard, so that a disk reader can
        // address it and a payload running past its shard is caught here
        // rather than as a fault later.
        let region = self.shards[shard as usize].bytes.len();
        let inside = info.offset < region && t.size_bytes() <= region - info.offset;
        if !inside {
            self.fail(format!("tensor {name} is truncated"));
            return TensorRef::default();
        }
        if let Some(base) = name.strip_suffix(".weight")
            && !base.is_empty()
        {
            let scale_name = format!("{base}.scale");
            if let Some((scale_shard, s)) = find_tensor(self.shards, &scale_name) {
                if s.ty != GgmlType::F32 || s.elements() != 1 {
                    self.fail(format!("tensor {scale_name} must be one F32 value"));
                    return TensorRef::default();
                }
                let bytes = &self.shards[scale_shard as usize].bytes[s.offset..s.offset + 4];
                t.scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                if !t.scale.is_finite() || t.scale == 0.0 {
                    self.fail(format!("tensor {scale_name} is not a finite nonzero scale"));
                    return TensorRef::default();
                }
            }
        }
        t
    }

    fn layer(&mut self, c: &Config, il: u32) -> LayerWeights {
        let mut l = LayerWeights {
            ty: c.blocks[il as usize],
            ..LayerWeights::default()
        };
        let p = format!("blk.{il}.");
        let hidden = u64::from(c.hidden_size);
        let dense = &[
            GgmlType::Nvfp4,
            GgmlType::Q8_0,
            GgmlType::Bf16,
            GgmlType::F16,
            GgmlType::F32,
        ];
        let experts = &[GgmlType::Iq2Xxs, GgmlType::Nvfp4, GgmlType::Q8_0];

        l.norm = self.get(&format!("{p}attn_norm.weight"), hidden, 1, 1, F32);
        match l.ty {
            BlockType::Mamba => {
                l.ssm_in = self.get(
                    &format!("{p}ssm_in.weight"),
                    hidden,
                    c.ssm_in_projection().into(),
                    1,
                    dense,
                );
                l.ssm_conv1d = self.get(
                    &format!("{p}ssm_conv1d.weight"),
                    c.ssm_conv_kernel.into(),
                    c.ssm_conv_channels().into(),
                    1,
                    F32,
                );
                l.ssm_conv1d_b = self.get(
                    &format!("{p}ssm_conv1d.bias"),
                    c.ssm_conv_channels().into(),
                    1,
                    1,
                    F32,
                );
                l.ssm_dt_b = self.get(&format!("{p}ssm_dt.bias"), c.ssm_heads.into(), 1, 1, F32);
                l.ssm_a = self.get(&format!("{p}ssm_a"), 1, c.ssm_heads.into(), 1, F32);
                l.ssm_d = self.get(&format!("{p}ssm_d"), 1, c.ssm_heads.into(), 1, F32);
                l.ssm_norm = self.get(
                    &format!("{p}ssm_norm.weight"),
                    (c.ssm_inner_size / c.ssm_groups).into(),
                    c.ssm_groups.into(),
                    1,
                    F32,
                );
                l.ssm_out = self.get(
                    &format!("{p}ssm_out.weight"),
                    c.ssm_inner_size.into(),
                    hidden,
                    1,
                    dense,
                );
            }
            BlockType::Attention => {
                l.attn_q = self.get(
                    &format!("{p}attn_q.weight"),
                    hidden,
                    c.attention_q_dim().into(),
                    1,
                    dense,
                );
                l.attn_k = self.get(
                    &format!("{p}attn_k.weight"),
                    hidden,
                    c.attention_kv_dim().into(),
                    1,
                    dense,
                );
                l.attn_v = self.get(
                    &format!("{p}attn_v.weight"),
                    hidden,
                    c.attention_kv_dim().into(),
                    1,
                    dense,
                );
                l.attn_out = self.get(
                    &format!("{p}attn_output.weight"),
                    c.attention_q_dim().into(),
                    hidden,
                    1,
                    dense,
                );
            }
            BlockType::Moe => {
                // Routed experts per block are the router's row count.
                let n = find_tensor(self.shards, &format!("{p}ffn_gate_inp.weight"))
                    .filter(|(_, t)| t.dims.len() == 2)
                    .map_or(0, |(_, t)| t.dims[1]);
                if n < u64::from(c.experts_used[il as usize]) || n > u64::from(c.max_experts) {
                    self.fail(format!("invalid routed expert count in block {il}"));
                    return l;
                }
                l.num_experts = n as u32;
                let ff = u64::from(c.expert_ff[il as usize]);
                let latent = u64::from(c.latent_size);
                let shared = u64::from(c.shared_expert_ff);
                l.router = self.get(&format!("{p}ffn_gate_inp.weight"), hidden, n, 1, F32);
                l.router_bias = self.get(&format!("{p}exp_probs_b.bias"), n, 1, 1, F32);
                l.latent_down = self.get(
                    &format!("{p}ffn_latent_down.weight"),
                    hidden,
                    latent,
                    1,
                    dense,
                );
                l.latent_up = self.get(
                    &format!("{p}ffn_latent_up.weight"),
                    latent,
                    hidden,
                    1,
                    dense,
                );
                l.up_exps = self.get(&format!("{p}ffn_up_exps.weight"), latent, ff, n, experts);
                l.down_exps = self.get(&format!("{p}ffn_down_exps.weight"), ff, latent, n, experts);
                l.up_shexp = self.get(
                    &format!("{p}ffn_up_shexp.weight"),
                    hidden,
                    shared,
                    1,
                    experts,
                );
                l.down_shexp = self.get(
                    &format!("{p}ffn_down_shexp.weight"),
                    shared,
                    hidden,
                    1,
                    experts,
                );
            }
        }
        l
    }

    fn result<T>(self, value: T) -> Result<T, String> {
        match self.error {
            None => Ok(value),
            Some(error) => Err(error),
        }
    }
}

fn first<'s, 'a>(shards: &'s [GgufShard<'a>]) -> Result<&'s Gguf<'a>, String> {
    shards
        .first()
        .map(|s| &s.gguf)
        .ok_or_else(|| "no GGUF file".to_string())
}

impl ModelWeights {
    /// Binds and validates the trunk artifact (`Bind`). Tensor payloads stay
    /// mapped and untouched; only headers and the scalar scale companions
    /// are read.
    pub fn bind(shards: &[GgufShard<'_>]) -> Result<Self, String> {
        let config = Config::from_gguf(first(shards)?)?;
        let mut b = Binder {
            shards,
            error: None,
        };
        let c = &config;
        let dense = &[GgmlType::Q8_0, GgmlType::Bf16, GgmlType::F16, GgmlType::F32];
        let hidden = u64::from(c.hidden_size);
        let vocab = u64::from(c.vocab_size);
        let token_embd = b.get("token_embd.weight", hidden, vocab, 1, dense);
        let output_norm = b.get("output_norm.weight", hidden, 1, 1, F32);
        let output = b.get("output.weight", hidden, vocab, 1, dense);
        let mut layers = Vec::with_capacity(c.num_layers as usize);
        for il in 0..c.num_layers {
            if !b.ok() {
                break;
            }
            layers.push(b.layer(c, il));
        }
        b.result(ModelWeights {
            token_embd,
            output_norm,
            output,
            layers,
            config: config.clone(),
        })
    }
}

impl DrafterWeights {
    /// Binds the LiLiCorr drafter for `target` (`Bind`).
    pub fn bind(shards: &[GgufShard<'_>], target: &Config) -> Result<Self, String> {
        let config = DrafterConfig::from_gguf(first(shards)?, target)?;
        let mut b = Binder {
            shards,
            error: None,
        };
        let c = &config;
        let hidden = u64::from(c.hidden_size);
        let lc = u64::from(c.lc_hidden);
        let slots = u64::from(c.slots());
        let nodes = slots * u64::from(c.lc_top_k);
        let ff = u64::from(c.ff_size);
        let kernel = u64::from(c.conv_kernel);

        let fc = b.get("fc.weight", c.feature_size().into(), hidden, 1, FP4);
        let enc_norm = b.get("enc.output_norm.weight", hidden, 1, 1, F32);
        let output_norm = b.get("output_norm.weight", hidden, 1, 1, F32);
        let mut layers = Vec::new();
        for il in 0..c.num_layers {
            if !b.ok() {
                break;
            }
            let p = format!("blk.{il}.");
            let n = |suffix: &str| format!("{p}{suffix}");
            let projection = u64::from(c.conv_projection());
            layers.push(DrafterLayerWeights {
                attn_norm: b.get(&n("attn_norm.weight"), hidden, 1, 1, F32),
                attn_conv_base: b.get(&n("attn_conv_base"), hidden, kernel, 2, F32),
                attn_conv_proj: b.get(&n("attn_conv_proj.weight"), hidden, projection, 1, BF16),
                attn_q: b.get(&n("attn_q.weight"), hidden, c.q_dim().into(), 1, BF16),
                attn_k: b.get(&n("attn_k.weight"), hidden, c.kv_dim().into(), 1, BF16),
                attn_v: b.get(&n("attn_v.weight"), hidden, c.kv_dim().into(), 1, BF16),
                attn_q_norm: b.get(&n("attn_q_norm.weight"), c.head_dim.into(), 1, 1, F32),
                attn_k_norm: b.get(&n("attn_k_norm.weight"), c.head_dim.into(), 1, 1, F32),
                attn_out: b.get(&n("attn_output.weight"), c.q_dim().into(), hidden, 1, BF16),
                attn_sinks: b.get(&n("attn_sinks.weight"), c.num_heads.into(), 1, 1, F32),
                ffn_norm: b.get(&n("ffn_norm.weight"), hidden, 1, 1, F32),
                ffn_conv_base: b.get(&n("ffn_conv_base"), hidden, kernel, 2, F32),
                ffn_conv_proj: b.get(&n("ffn_conv_proj.weight"), hidden, projection, 1, BF16),
                ffn_gate: b.get(&n("ffn_gate.weight"), hidden, ff, 1, FP4),
                ffn_up: b.get(&n("ffn_up.weight"), hidden, ff, 1, FP4),
                ffn_down: b.get(&n("ffn_down.weight"), ff, hidden, 1, FP4),
            });
        }

        let token_proj = b.get("lilicorr.token_proj.weight", hidden, lc, 1, FP4);
        let token_proj_b = b.get("lilicorr.token_proj.bias", lc, 1, 1, F32);
        let pass_proj = b.get("lilicorr.pass_hidden_proj.weight", hidden, lc, 1, FP4);
        let pass_proj_b = b.get("lilicorr.pass_hidden_proj.bias", lc, 1, 1, F32);
        let context_proj = b.get("lilicorr.context_proj.weight", hidden, lc, 1, FP4);
        let context_proj_b = b.get("lilicorr.context_proj.bias", lc, 1, 1, F32);
        // llama.cpp stores the candidate-feature MLP as the block after the
        // LiLiCorr layers, which follow the drafter's own blocks.
        let f = format!("blk.{}.", c.num_layers + c.lc_layers);
        let feature_norm = b.get(&format!("{f}ffn_norm.weight"), 5, 1, 1, F32);
        let feature_norm_b = b.get(&format!("{f}ffn_norm.bias"), 5, 1, 1, F32);
        let feature_up = b.get(&format!("{f}ffn_up.weight"), 5, lc, 1, F32);
        let feature_up_b = b.get(&format!("{f}ffn_up.bias"), lc, 1, 1, F32);
        let feature_down = b.get(&format!("{f}ffn_down.weight"), lc, lc, 1, F32);
        let feature_down_b = b.get(&format!("{f}ffn_down.bias"), lc, 1, 1, F32);
        let slot_embd = b.get("lilicorr.slot_embedding.weight", lc, slots, 1, F32);
        let rank_embd = b.get(
            "lilicorr.rank_embedding.weight",
            lc,
            c.lc_top_k.into(),
            1,
            F32,
        );
        let attention_bias = b.get(
            "lilicorr.attention_bias.bias",
            nodes,
            nodes,
            c.lc_heads.into(),
            F32,
        );
        let mut lc_layers = Vec::new();
        for i in 0..c.lc_layers {
            if !b.ok() {
                break;
            }
            let p = format!("blk.{}.", c.num_layers + i);
            let q = format!("lilicorr.layers.{i}.");
            let mlp = u64::from(c.lc_mlp);
            lc_layers.push(LiliCorrLayerWeights {
                attn_norm: b.get(&format!("{p}attn_norm.weight"), lc, 1, 1, F32),
                mlp_norm: b.get(&format!("{p}ffn_norm.weight"), lc, 1, 1, F32),
                in_proj: b.get(&format!("{q}attn.in_proj.weight"), lc, 3 * lc, 1, F32),
                in_proj_b: b.get(&format!("{q}attn.in_proj.bias"), 3 * lc, 1, 1, F32),
                out_proj: b.get(&format!("{p}attn_output.weight"), lc, lc, 1, FP4),
                out_proj_b: b.get(&format!("{p}attn_output.bias"), lc, 1, 1, F32),
                mlp_up: b.get(&format!("{p}ffn_up.weight"), lc, mlp, 1, FP4),
                mlp_up_b: b.get(&format!("{p}ffn_up.bias"), mlp, 1, 1, F32),
                mlp_down: b.get(&format!("{p}ffn_down.weight"), mlp, lc, 1, FP4),
                mlp_down_b: b.get(&format!("{p}ffn_down.bias"), lc, 1, 1, F32),
            });
        }
        let factor = u64::from(c.lc_factor);
        let lc_output_norm = b.get("lilicorr.output_norm.weight", lc, 1, 1, F32);
        let anchor_norm = b.get("lilicorr.anchor_norm.weight", lc, 1, 1, F32);
        let factor_in = b.get("lilicorr.factor_input_proj.weight", 3 * lc, lc, 1, F32);
        let factor_in_b = b.get("lilicorr.factor_input_proj.bias", lc, 1, 1, F32);
        let out_head = b.get("lilicorr.out_head.weight", lc, factor, 1, F32);
        let out_head_b = b.get("lilicorr.out_head.bias", factor, 1, 1, F32);
        let in_head = b.get("lilicorr.in_head.weight", lc, factor, 1, F32);
        let in_head_b = b.get("lilicorr.in_head.bias", factor, 1, 1, F32);
        let anchor_head = b.get("lilicorr.anchor_out_head.weight", lc, factor, 1, F32);
        let anchor_head_b = b.get("lilicorr.anchor_out_head.bias", factor, 1, 1, F32);
        b.result(DrafterWeights {
            config: config.clone(),
            fc,
            enc_norm,
            output_norm,
            layers,
            token_proj,
            token_proj_b,
            pass_proj,
            pass_proj_b,
            context_proj,
            context_proj_b,
            feature_norm,
            feature_norm_b,
            feature_up,
            feature_up_b,
            feature_down,
            feature_down_b,
            slot_embd,
            rank_embd,
            attention_bias,
            lc_layers,
            lc_output_norm,
            anchor_norm,
            factor_in,
            factor_in_b,
            out_head,
            out_head_b,
            in_head,
            in_head_b,
            anchor_head,
            anchor_head_b,
        })
    }
}

impl VisionWeights {
    /// Binds an mmproj GGUF for `trunk` (`Bind`). The tensors stay mapped.
    pub fn bind(shards: &[GgufShard<'_>], trunk: &Config) -> Result<Self, String> {
        let config = VisionConfig::from_gguf(first(shards)?, trunk)?;
        let mut b = Binder {
            shards,
            error: None,
        };
        let c = &config;
        let width = u64::from(c.width);
        let ff = u64::from(c.ff_size);
        let patch_size = u64::from(c.patch_size);
        let patch = b.get_flat(
            "v.patch_embd.weight",
            &[patch_size, patch_size, 3, width],
            F32,
        );
        let side = u64::from(c.position_side);
        let position = b.get("v.position_embd.weight", width, side * side, 1, F32);
        let prefix = b.get("v.class_embd", width, c.prefix_tokens.into(), 1, F32);
        let mut layers = vec![VisionLayerWeights::default(); c.num_layers as usize];
        for (il, l) in layers.iter_mut().enumerate() {
            if !b.ok() {
                break;
            }
            let p = format!("v.blk.{il}.");
            let n = |suffix: &str| format!("{p}{suffix}");
            *l = VisionLayerWeights {
                norm1: b.get(&n("ln1.weight"), width, 1, 1, F32),
                norm1_b: b.get(&n("ln1.bias"), width, 1, 1, F32),
                q: b.get(&n("attn_q.weight"), width, width, 1, BF16),
                q_b: b.get(&n("attn_q.bias"), width, 1, 1, F32),
                k: b.get(&n("attn_k.weight"), width, width, 1, BF16),
                k_b: b.get(&n("attn_k.bias"), width, 1, 1, F32),
                v: b.get(&n("attn_v.weight"), width, width, 1, BF16),
                v_b: b.get(&n("attn_v.bias"), width, 1, 1, F32),
                out: b.get(&n("attn_out.weight"), width, width, 1, BF16),
                out_b: b.get(&n("attn_out.bias"), width, 1, 1, F32),
                norm2: b.get(&n("ln2.weight"), width, 1, 1, F32),
                norm2_b: b.get(&n("ln2.bias"), width, 1, 1, F32),
                up: b.get(&n("ffn_up.weight"), width, ff, 1, BF16),
                up_b: b.get(&n("ffn_up.bias"), ff, 1, 1, F32),
                down: b.get(&n("ffn_down.weight"), ff, width, 1, BF16),
                down_b: b.get(&n("ffn_down.bias"), width, 1, 1, F32),
            };
        }
        let merged = u64::from(c.merged_width());
        let hidden = u64::from(c.projector_hidden);
        let projector_norm = b.get("mm.model.mlp.0.weight", merged, 1, 1, F32);
        let projector_up = b.get("mm.model.mlp.1.weight", merged, hidden, 1, BF16);
        let projector_down = b.get(
            "mm.model.mlp.3.weight",
            hidden,
            c.projection.into(),
            1,
            BF16,
        );
        if let Some(error) = b.error {
            return Err(error);
        }
        // Tensors this runtime does not apply (layer scales, pre or post
        // norms, patch bias) would change the result; the video patch
        // projection is the only extra one allowed.
        let expected = 3 + 16 * c.num_layers as usize + 3;
        let extra = shards
            .iter()
            .flat_map(|s| s.gguf.tensors())
            .filter(|t| t.name == "v.patch_embd.weight.1")
            .count();
        if tensor_count(shards) != expected + extra {
            return Err("the vision projector has tensors this runtime does not apply".into());
        }
        Ok(VisionWeights {
            config: config.clone(),
            patch,
            position,
            prefix,
            layers,
            projector_norm,
            projector_up,
            projector_down,
        })
    }
}
