//! Binding of the target and MTP drafter tensors (`weights.cpp`): every
//! tensor is checked for its exact shape and an accepted format and located
//! in its mapped GGUF shard. Payloads stay mapped and untouched; only the
//! tiny F32 scalars and the rope factors are read.

use gufo_core::gguf::{Gguf, Tensor};
use gufo_formats::GgmlType;

use crate::config::Config;

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

/// Storage geometry of the formats this runtime decodes, as (elements,
/// bytes) per block. Unknown types get a zero block, which every validation
/// rejects (`FormatOf`).
pub fn format_of(ty: GgmlType) -> (u64, u64) {
    match ty {
        GgmlType::F32 => (1, 4),
        GgmlType::F16 | GgmlType::Bf16 => (1, 2),
        GgmlType::Q8_0 => (32, 34),
        GgmlType::Q4_0 => (32, 18),
        GgmlType::Q5_0 => (32, 22),
        GgmlType::Q5_1 => (32, 24),
        GgmlType::Q4K => (256, 144),
        GgmlType::Q5K => (256, 176),
        GgmlType::Q6K => (256, 210),
        _ => (0, 0),
    }
}

/// Projection formats with decode GEMV, batched verification and prefill
/// GEMM kernels. BF16 projections run as binary16 (converted at upload).
pub const PROJECTION: &[GgmlType] = &[
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Q5_0,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::F16,
    GgmlType::Bf16,
];
/// Embedding rows are decoded one token at a time; the tied LM head also
/// needs a GEMV, which every listed format has.
pub const EMBEDDING: &[GgmlType] = &[
    GgmlType::Bf16,
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
];
const VECTOR: &[GgmlType] = &[GgmlType::F32];
/// Routed expert formats with decode, verification and prefill kernels: the
/// fused gate/up projection reduces over the hidden width, the down
/// projection over the (not 256-aligned) expert width. BF16 experts run as
/// binary16, like BF16 projections.
pub const EXPERT_GATE_UP: &[GgmlType] = &[
    GgmlType::Q4_0,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Q8_0,
    GgmlType::F16,
    GgmlType::Bf16,
];
pub const EXPERT_DOWN: &[GgmlType] = &[
    GgmlType::Q4_0,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::F16,
    GgmlType::Bf16,
];

/// Non-owning view of one GGUF tensor. `cols` (ne[0]) is the contiguous
/// reduction dimension, `rows` (ne[1]) the output dimension and `experts`
/// (ne[2]) the number of stacked expert matrices. The bytes live in the
/// shard's mapping ([`TensorRef::bytes`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorRef {
    pub ty: GgmlType,
    pub cols: u64,
    pub rows: u64,
    pub experts: u64,
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
        if block == 0 {
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

    /// The view of one stacked expert matrix (`experts` 1).
    pub fn expert_view(&self, e: u64) -> TensorRef {
        TensorRef {
            experts: 1,
            file_offset: self.file_offset + self.expert_offset(e) as u64,
            ..self.clone()
        }
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
    /// `[hidden]`
    pub attn_norm: TensorRef,
    /// `[hidden -> heads * head_dim]`
    pub attn_q: TensorRef,
    /// `[hidden -> kv_heads * head_dim]`; target only.
    pub attn_k: TensorRef,
    /// Absent on global layers: V is the K projection.
    pub attn_v: TensorRef,
    /// `[head_dim]`
    pub attn_q_norm: TensorRef,
    /// `[head_dim]`; target only.
    pub attn_k_norm: TensorRef,
    /// `[heads * head_dim -> hidden]`
    pub attn_output: TensorRef,
    /// `[hidden]`
    pub post_attn_norm: TensorRef,
    /// `[hidden]`
    pub ffn_norm: TensorRef,
    /// `[hidden -> ffn]`
    pub ffn_gate: TensorRef,
    /// `[hidden -> ffn]`
    pub ffn_up: TensorRef,
    /// `[ffn -> hidden]`
    pub ffn_down: TensorRef,
    /// `[hidden]`
    pub post_ffn_norm: TensorRef,
    /// layer_output_scale, applied to the residual.
    pub output_scale: f32,

    // Routed experts (26B-A4B), run beside the dense MLP above; empty on
    // dense targets and drafts.
    /// ffn_gate_inp, F32 `[hidden -> experts]`
    pub router: TensorRef,
    /// ffn_gate_inp.scale, F32 `[hidden]`
    pub router_scale: TensorRef,
    /// Expert input norm, `[hidden]`
    pub pre_ffn_norm_2: TensorRef,
    /// Dense MLP output norm, `[hidden]`
    pub post_ffn_norm_1: TensorRef,
    /// Expert mixture output norm, `[hidden]`
    pub post_ffn_norm_2: TensorRef,
    /// `[experts][2 * expert_ffn][hidden]`: the gate rows, then the up rows.
    pub gate_up_exps: TensorRef,
    /// `[experts][hidden][expert_ffn]`
    pub down_exps: TensorRef,
    /// F32 `[experts]`, scales each expert output.
    pub down_exps_scale: TensorRef,
}

/// Host copy of the rope frequency divisors used by global layers. A value
/// of 1 rotates the pair; the GGUF marks non-rotating pairs with 1e30.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RopeFactors {
    /// `[head_dim_global / 2]`
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelWeights {
    pub config: Config,
    /// `[hidden -> vocab]`; scaled by sqrt(hidden).
    pub token_embd: TensorRef,
    /// LM head; the token embedding when tied.
    pub output: TensorRef,
    /// `[hidden]`
    pub output_norm: TensorRef,
    pub rope_factors: RopeFactors,
    pub vocab_size: u32,
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    /// Whether the LM head is the token embedding.
    pub fn tied_output(&self) -> bool {
        self.output.shard == self.token_embd.shard
            && self.output.file_offset == self.token_embd.file_offset
    }

    /// Binds and validates the target. Payloads stay mapped; only headers and
    /// the tiny F32 scalars are read.
    pub fn bind(shards: &[GgufShard<'_>]) -> Result<Self, String> {
        let first = shards.first().ok_or("gemma4 GGUF has no shards")?;
        let config = Config::from_gguf(&first.gguf)?;
        let mut b = Binder::new(shards);
        let mut w = ModelWeights {
            config,
            ..ModelWeights::default()
        };
        let vocab = vocab_rows(shards);
        if vocab == 0 || vocab > 0x7fff_ffff {
            return Err("token_embd.weight must be a [hidden, vocab] matrix".into());
        }
        let c = &w.config;
        w.vocab_size = vocab as u32;
        w.token_embd = b.get(
            "token_embd.weight",
            u64::from(c.hidden_size),
            vocab,
            EMBEDDING,
            true,
            1,
        );
        w.output = b.get(
            "output.weight",
            u64::from(c.hidden_size),
            vocab,
            EMBEDDING,
            false,
            1,
        );
        if w.output.is_empty() {
            w.output = w.token_embd.clone();
        }
        w.output_norm = b.get(
            "output_norm.weight",
            u64::from(c.hidden_size),
            1,
            VECTOR,
            true,
            1,
        );
        w.rope_factors = b.factors(c);
        let mut layers = Vec::with_capacity(c.num_layers as usize);
        for l in 0..c.num_layers {
            if !b.ok() {
                break;
            }
            let layer = b.layer(c, l, true);
            // A missing V projection means V is the unnormalized K
            // projection; that is only defined when K and V share a width.
            if b.ok() && layer.attn_v.is_empty() && c.is_sliding(l) {
                b.fail(format!("sliding layer {l} lacks attn_v"));
            }
            layers.push(layer);
        }
        b.check()?;
        w.layers = layers;
        Ok(w)
    }
}

/// The `gemma4-assistant` drafter: Q-only attention over the target's KV,
/// its own vocabulary head, and projections to and from the target width.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DraftWeights {
    pub config: Config,
    /// `[2 * target_hidden -> hidden]`
    pub pre_projection: TensorRef,
    /// `[hidden -> target_hidden]`
    pub post_projection: TensorRef,
    /// `[hidden -> vocab]`, the draft LM head.
    pub token_embd: TensorRef,
    /// `[hidden]`
    pub output_norm: TensorRef,
    pub rope_factors: RopeFactors,
    pub layers: Vec<LayerWeights>,
}

impl DraftWeights {
    pub fn bind(shards: &[GgufShard<'_>], target: &ModelWeights) -> Result<Self, String> {
        let first = shards
            .first()
            .ok_or("gemma4-assistant GGUF has no shards")?;
        let config = Config::draft_from_gguf(&first.gguf, &target.config)?;
        if vocab_rows(shards) != u64::from(target.vocab_size) {
            return Err("gemma4-assistant vocabulary differs from the target".into());
        }
        let mut b = Binder::new(shards);
        let c = &config;
        let target_hidden = u64::from(c.target_hidden_size);
        let hidden = u64::from(c.hidden_size);
        let pre_projection = b.get(
            "nextn.pre_projection.weight",
            2 * target_hidden,
            hidden,
            PROJECTION,
            true,
            1,
        );
        let post_projection = b.get(
            "nextn.post_projection.weight",
            hidden,
            target_hidden,
            PROJECTION,
            true,
            1,
        );
        let token_embd = b.get(
            "token_embd.weight",
            hidden,
            u64::from(target.vocab_size),
            EMBEDDING,
            true,
            1,
        );
        let output_norm = b.get("output_norm.weight", hidden, 1, VECTOR, true, 1);
        let rope_factors = b.factors(c);
        let mut layers = Vec::with_capacity(c.num_layers as usize);
        for l in 0..c.num_layers {
            if !b.ok() {
                break;
            }
            layers.push(b.layer(c, l, false));
        }
        b.check()?;
        Ok(Self {
            config,
            pre_projection,
            post_projection,
            token_embd,
            output_norm,
            rope_factors,
            layers,
        })
    }
}

/// Vocabulary size from the embedding tensor, which every other vocabulary
/// sized tensor must match (`VocabRows`).
fn vocab_rows(shards: &[GgufShard<'_>]) -> u64 {
    match find_tensor(shards, "token_embd.weight") {
        Some((_, t)) if t.dims.len() == 2 => t.dims[1],
        _ => 0,
    }
}

struct Binder<'s, 'a> {
    shards: &'s [GgufShard<'a>],
    error: Option<String>,
}

impl<'s, 'a> Binder<'s, 'a> {
    fn new(shards: &'s [GgufShard<'a>]) -> Self {
        Self {
            shards,
            error: None,
        }
    }

    fn ok(&self) -> bool {
        self.error.is_none()
    }

    fn fail(&mut self, message: String) {
        if self.error.is_none() {
            self.error = Some(message);
        }
    }

    fn check(&self) -> Result<(), String> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn has(&self, name: &str) -> bool {
        find_tensor(self.shards, name).is_some()
    }

    /// Binds `name` with the exact shape and one of the accepted formats;
    /// `experts` > 1 expects a stack of that many `[cols, rows]` matrices.
    fn get(
        &mut self,
        name: &str,
        cols: u64,
        rows: u64,
        types: &[GgmlType],
        required: bool,
        experts: u64,
    ) -> TensorRef {
        let Some((shard, info)) = find_tensor(self.shards, name) else {
            if required {
                self.fail(format!("missing tensor {name}"));
            }
            return TensorRef::default();
        };
        let d = &info.dims;
        let got_cols = d.first().copied().unwrap_or(1);
        let got_rows = d.get(1).copied().unwrap_or(1);
        let got_experts = d.get(2).copied().unwrap_or(1);
        if got_cols != cols || got_rows != rows || got_experts != experts || d.len() > 3 {
            self.fail(format!(
                "tensor {name} has shape [{got_cols}, {got_rows}, {got_experts}], expected \
                 [{cols}, {rows}, {experts}]"
            ));
            return TensorRef::default();
        }
        let (block, _) = format_of(info.ty);
        if !types.contains(&info.ty) || block == 0 || !cols.is_multiple_of(block) {
            self.fail(format!(
                "tensor {name} has unsupported format {}",
                info.ty.name()
            ));
            return TensorRef::default();
        }
        let t = TensorRef {
            ty: info.ty,
            cols,
            rows,
            experts,
            file_offset: info.offset as u64,
            shard,
            name: info.name.to_string(),
            bound: true,
        };
        // A payload running past its shard is caught here, not as a fault
        // later.
        let size = self.shards[shard as usize].bytes.len();
        if t.size_bytes() > size.saturating_sub(info.offset) {
            self.fail(format!("tensor {name} is truncated"));
            return TensorRef::default();
        }
        t
    }

    /// Reads a one-element F32 tensor such as layer_output_scale.
    fn scalar(&mut self, name: &str) -> f32 {
        let t = self.get(name, 1, 1, VECTOR, true, 1);
        if !self.ok() {
            return 0.0;
        }
        let value = t.f32_values(self.shards)[0];
        if !value.is_finite() {
            self.fail(format!("tensor {name} is not finite"));
        }
        value
    }

    fn factors(&mut self, c: &Config) -> RopeFactors {
        let t = self.get(
            "rope_freqs.weight",
            u64::from(c.head_dim_global / 2),
            1,
            VECTOR,
            true,
            1,
        );
        if !self.ok() {
            return RopeFactors::default();
        }
        let values = t.f32_values(self.shards);
        if values
            .iter()
            .any(|&v| !v.is_finite() || v.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater))
        {
            self.fail("rope_freqs.weight must contain positive finite divisors".into());
        }
        RopeFactors { values }
    }

    /// Binds one block. `with_kv` is false for Q-only draft layers.
    fn layer(&mut self, c: &Config, l: u32, with_kv: bool) -> LayerWeights {
        let p = format!("blk.{l}.");
        let hidden = u64::from(c.hidden_size);
        let head_dim = u64::from(c.head_dim(l));
        let mut w = LayerWeights {
            output_scale: 1.0,
            ..LayerWeights::default()
        };
        w.attn_norm = self.get(&format!("{p}attn_norm.weight"), hidden, 1, VECTOR, true, 1);
        w.attn_q = self.get(
            &format!("{p}attn_q.weight"),
            hidden,
            u64::from(c.q_dim(l)),
            PROJECTION,
            true,
            1,
        );
        w.attn_q_norm = self.get(
            &format!("{p}attn_q_norm.weight"),
            head_dim,
            1,
            VECTOR,
            true,
            1,
        );
        if with_kv {
            w.attn_k = self.get(
                &format!("{p}attn_k.weight"),
                hidden,
                u64::from(c.kv_dim(l)),
                PROJECTION,
                true,
                1,
            );
            w.attn_v = self.get(
                &format!("{p}attn_v.weight"),
                hidden,
                u64::from(c.kv_dim(l)),
                PROJECTION,
                false,
                1,
            );
            w.attn_k_norm = self.get(
                &format!("{p}attn_k_norm.weight"),
                head_dim,
                1,
                VECTOR,
                true,
                1,
            );
        } else if self.has(&format!("{p}attn_k.weight")) || self.has(&format!("{p}attn_v.weight")) {
            self.fail(format!(
                "shared-KV draft layer {l} must not carry K/V projections"
            ));
        }
        w.attn_output = self.get(
            &format!("{p}attn_output.weight"),
            u64::from(c.q_dim(l)),
            hidden,
            PROJECTION,
            true,
            1,
        );
        w.post_attn_norm = self.get(
            &format!("{p}post_attention_norm.weight"),
            hidden,
            1,
            VECTOR,
            true,
            1,
        );
        w.ffn_norm = self.get(&format!("{p}ffn_norm.weight"), hidden, 1, VECTOR, true, 1);
        let ffn = u64::from(c.ffn_size);
        w.ffn_gate = self.get(
            &format!("{p}ffn_gate.weight"),
            hidden,
            ffn,
            PROJECTION,
            true,
            1,
        );
        w.ffn_up = self.get(
            &format!("{p}ffn_up.weight"),
            hidden,
            ffn,
            PROJECTION,
            true,
            1,
        );
        w.ffn_down = self.get(
            &format!("{p}ffn_down.weight"),
            ffn,
            hidden,
            PROJECTION,
            true,
            1,
        );
        w.post_ffn_norm = self.get(
            &format!("{p}post_ffw_norm.weight"),
            hidden,
            1,
            VECTOR,
            true,
            1,
        );
        let scale = format!("{p}layer_output_scale.weight");
        if self.has(&scale) {
            w.output_scale = self.scalar(&scale);
        }
        if c.has_experts() {
            self.experts(c, &p, &mut w);
        } else if self.has(&format!("{p}ffn_gate_inp.weight")) {
            self.fail(format!("layer {l} carries experts in a dense model"));
        }
        w
    }

    /// The router and the routed experts of an expert layer. Only the fused
    /// gate/up layout is supported.
    fn experts(&mut self, c: &Config, p: &str, w: &mut LayerWeights) {
        let hidden = u64::from(c.hidden_size);
        let experts = u64::from(c.num_experts);
        let width = u64::from(c.expert_ffn_size);
        w.router = self.get(
            &format!("{p}ffn_gate_inp.weight"),
            hidden,
            experts,
            VECTOR,
            true,
            1,
        );
        w.router_scale = self.get(
            &format!("{p}ffn_gate_inp.scale"),
            hidden,
            1,
            VECTOR,
            true,
            1,
        );
        w.pre_ffn_norm_2 = self.get(
            &format!("{p}pre_ffw_norm_2.weight"),
            hidden,
            1,
            VECTOR,
            true,
            1,
        );
        w.post_ffn_norm_1 = self.get(
            &format!("{p}post_ffw_norm_1.weight"),
            hidden,
            1,
            VECTOR,
            true,
            1,
        );
        w.post_ffn_norm_2 = self.get(
            &format!("{p}post_ffw_norm_2.weight"),
            hidden,
            1,
            VECTOR,
            true,
            1,
        );
        if self.ok() && !self.has(&format!("{p}ffn_gate_up_exps.weight")) {
            self.fail(format!(
                "tensor {p}ffn_gate_up_exps.weight is missing; separate gate and up experts \
                 are not supported"
            ));
            return;
        }
        w.gate_up_exps = self.get(
            &format!("{p}ffn_gate_up_exps.weight"),
            hidden,
            2 * width,
            EXPERT_GATE_UP,
            true,
            experts,
        );
        w.down_exps = self.get(
            &format!("{p}ffn_down_exps.weight"),
            width,
            hidden,
            EXPERT_DOWN,
            true,
            experts,
        );
        w.down_exps_scale = self.get(
            &format!("{p}ffn_down_exps.scale"),
            experts,
            1,
            VECTOR,
            true,
            1,
        );
    }
}
