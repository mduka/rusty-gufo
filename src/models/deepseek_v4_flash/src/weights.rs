//! Tensor binding and layout checks (`weights_bind`,
//! `weights_validate_layout`, `dspark_bind`): the GGUF directory becomes
//! the fixed per-layer weight table, every type and dimension checked, so
//! the graph can rely on the shape constants.

use crate::config::INVALID;
use crate::gguf::{Model, TensorInfo, type_name};
use crate::shape::tensor_type::*;
use crate::shape::*;

/// The weights of one layer (`ds4_layer_weights`); the compressor and
/// indexer tensors exist for compressed layers only, the hash table in the
/// first three layers, the probability bias optionally.
#[derive(Debug, Clone, Default)]
pub struct LayerWeights {
    pub hc_attn_fn: TensorInfo,
    pub hc_attn_scale: TensorInfo,
    pub hc_attn_base: TensorInfo,
    pub attn_norm: TensorInfo,
    pub attn_q_a: TensorInfo,
    pub attn_q_a_norm: TensorInfo,
    pub attn_q_b: TensorInfo,
    pub attn_kv: TensorInfo,
    pub attn_kv_a_norm: TensorInfo,
    pub attn_sinks: TensorInfo,
    pub attn_output_a: TensorInfo,
    pub attn_output_b: TensorInfo,
    pub attn_compressor_ape: Option<TensorInfo>,
    pub attn_compressor_kv: Option<TensorInfo>,
    pub attn_compressor_gate: Option<TensorInfo>,
    pub attn_compressor_norm: Option<TensorInfo>,
    pub indexer_attn_q_b: Option<TensorInfo>,
    pub indexer_proj: Option<TensorInfo>,
    pub indexer_compressor_ape: Option<TensorInfo>,
    pub indexer_compressor_kv: Option<TensorInfo>,
    pub indexer_compressor_gate: Option<TensorInfo>,
    pub indexer_compressor_norm: Option<TensorInfo>,
    pub hc_ffn_fn: TensorInfo,
    pub hc_ffn_scale: TensorInfo,
    pub hc_ffn_base: TensorInfo,
    pub ffn_norm: TensorInfo,
    pub ffn_gate_tid2eid: Option<TensorInfo>,
    pub ffn_gate_inp: TensorInfo,
    pub ffn_exp_probs_b: Option<TensorInfo>,
    pub ffn_gate_exps: TensorInfo,
    pub ffn_up_exps: TensorInfo,
    pub ffn_down_exps: TensorInfo,
    pub ffn_gate_shexp: TensorInfo,
    pub ffn_up_shexp: TensorInfo,
    pub ffn_down_shexp: TensorInfo,
}

/// The weights of the target model (`ds4_weights`).
#[derive(Debug, Clone)]
pub struct Weights {
    pub token_embd: TensorInfo,
    pub output_hc_base: TensorInfo,
    pub output_hc_fn: TensorInfo,
    pub output_hc_scale: TensorInfo,
    pub output_norm: TensorInfo,
    pub output: TensorInfo,
    pub layers: Vec<LayerWeights>,
}

fn fail(message: String) -> String {
    eprintln!("{message}");
    INVALID.to_owned()
}

fn name_of(m: &Model, t: &TensorInfo) -> String {
    String::from_utf8_lossy(m.tensor_name(t)).into_owned()
}

fn required(m: &Model, name: &str) -> Result<TensorInfo, String> {
    m.find_tensor(name)
        .cloned()
        .ok_or_else(|| fail(format!("ds4: required tensor is missing: {name}")))
}

fn optional(m: &Model, name: &str) -> Option<TensorInfo> {
    m.find_tensor(name).cloned()
}

/// `tensor_expect_layout`: the type, rank and leading dimensions.
pub(crate) fn expect(
    m: &Model,
    t: &TensorInfo,
    ty: u32,
    ndim: u32,
    dims: [u64; 3],
) -> Result<(), String> {
    if t.ty != ty {
        return Err(fail(format!(
            "ds4: tensor {} has type {}, expected {}",
            name_of(m, t),
            type_name(t.ty),
            type_name(ty)
        )));
    }
    expect_shape(m, t, ndim, dims)
}

fn expect_shape(m: &Model, t: &TensorInfo, ndim: u32, dims: [u64; 3]) -> Result<(), String> {
    if t.ndim != ndim {
        return Err(fail(format!(
            "ds4: tensor {} has {} dimensions, expected {ndim}",
            name_of(m, t),
            t.ndim
        )));
    }
    for (i, want) in dims.iter().enumerate().take(ndim as usize) {
        if t.dim[i] != *want {
            return Err(fail(format!(
                "ds4: tensor {} has dim[{i}]={}, expected {want}",
                name_of(m, t),
                t.dim[i]
            )));
        }
    }
    Ok(())
}

/// `tensor_expect_routed_expert`: IQ2_XXS, Q2_K or Q4_K of the given shape.
pub(crate) fn expect_routed(m: &Model, t: &TensorInfo, dims: [u64; 3]) -> Result<(), String> {
    if !matches!(t.ty, IQ2_XXS | Q2_K | Q4_K) {
        return Err(fail(format!(
            "ds4: tensor {} has type {} ({}), expected a routed expert quant type",
            name_of(m, t),
            t.ty,
            type_name(t.ty)
        )));
    }
    expect_shape(m, t, 3, dims)
}

const HC_DIM: u64 = N_EMBD as u64 * N_HC as u64;
const HC_MIX_DIM: u64 = 2 * N_HC as u64 + N_HC as u64 * N_HC as u64;
const Q_DIM: u64 = N_HEAD as u64 * N_HEAD_DIM as u64;
const OUT_LOW_DIM: u64 = N_OUT_GROUP as u64 * N_LORA_O as u64;
const OUT_A_IN: u64 = N_HEAD_DIM as u64 * (N_HEAD / N_OUT_GROUP) as u64;

/// The block of a DSpark stage (`dspark_validate_stage_block`): the
/// target's layout with a Q8_0 router projection.
pub(crate) fn expect_stage_block(m: &Model, l: &LayerWeights, stage: u32) -> Result<(), String> {
    expect_attention_and_compressors(m, l, 0)?;
    expect_ffn(m, l, Q8_0)?;
    if l.ffn_gate_exps.ty != l.ffn_up_exps.ty {
        return Err(fail(format!(
            "ds4: DSpark stage {stage} routed gate/up experts use different quant types"
        )));
    }
    expect_shared(m, l)
}

fn expect_shared(m: &Model, l: &LayerWeights) -> Result<(), String> {
    let e = u64::from(N_EMBD);
    expect(m, &l.ffn_gate_shexp, Q8_0, 2, [e, u64::from(N_FF_EXP), 0])?;
    expect(m, &l.ffn_up_shexp, Q8_0, 2, [e, u64::from(N_FF_EXP), 0])?;
    expect(m, &l.ffn_down_shexp, Q8_0, 2, [u64::from(N_FF_EXP), e, 0])
}

impl Weights {
    /// `weights_bind`: binds and checks every tensor of the target model.
    pub fn bind(m: &Model) -> Result<Self, String> {
        let token_embd = required(m, "token_embd.weight")?;
        let output_hc_base = required(m, "output_hc_base.weight")?;
        let output_hc_fn = required(m, "output_hc_fn.weight")?;
        let output_hc_scale = required(m, "output_hc_scale.weight")?;
        let output_norm = required(m, "output_norm.weight")?;
        let output = required(m, "output.weight")?;
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for il in 0..N_LAYER {
            let t = |suffix: &str| required(m, &format!("blk.{il}.{suffix}"));
            let ratio = expected_compress_ratio(il);
            let compressed = |suffix: &str| -> Result<Option<TensorInfo>, String> {
                if ratio != 0 {
                    t(suffix).map(Some)
                } else {
                    Ok(None)
                }
            };
            let indexed = |suffix: &str| -> Result<Option<TensorInfo>, String> {
                if ratio == 4 {
                    t(suffix).map(Some)
                } else {
                    Ok(None)
                }
            };
            layers.push(LayerWeights {
                hc_attn_fn: t("hc_attn_fn.weight")?,
                hc_attn_scale: t("hc_attn_scale.weight")?,
                hc_attn_base: t("hc_attn_base.weight")?,
                attn_norm: t("attn_norm.weight")?,
                attn_q_a: t("attn_q_a.weight")?,
                attn_q_a_norm: t("attn_q_a_norm.weight")?,
                attn_q_b: t("attn_q_b.weight")?,
                attn_kv: t("attn_kv.weight")?,
                attn_kv_a_norm: t("attn_kv_a_norm.weight")?,
                attn_sinks: t("attn_sinks.weight")?,
                attn_output_a: t("attn_output_a.weight")?,
                attn_output_b: t("attn_output_b.weight")?,
                attn_compressor_ape: compressed("attn_compressor_ape.weight")?,
                attn_compressor_kv: compressed("attn_compressor_kv.weight")?,
                attn_compressor_gate: compressed("attn_compressor_gate.weight")?,
                attn_compressor_norm: compressed("attn_compressor_norm.weight")?,
                indexer_attn_q_b: indexed("indexer.attn_q_b.weight")?,
                indexer_proj: indexed("indexer.proj.weight")?,
                indexer_compressor_ape: indexed("indexer_compressor_ape.weight")?,
                indexer_compressor_kv: indexed("indexer_compressor_kv.weight")?,
                indexer_compressor_gate: indexed("indexer_compressor_gate.weight")?,
                indexer_compressor_norm: indexed("indexer_compressor_norm.weight")?,
                hc_ffn_fn: t("hc_ffn_fn.weight")?,
                hc_ffn_scale: t("hc_ffn_scale.weight")?,
                hc_ffn_base: t("hc_ffn_base.weight")?,
                ffn_norm: t("ffn_norm.weight")?,
                ffn_gate_inp: t("ffn_gate_inp.weight")?,
                ffn_exp_probs_b: optional(m, &format!("blk.{il}.exp_probs_b.bias")),
                ffn_gate_exps: t("ffn_gate_exps.weight")?,
                ffn_up_exps: t("ffn_up_exps.weight")?,
                ffn_down_exps: t("ffn_down_exps.weight")?,
                ffn_gate_shexp: t("ffn_gate_shexp.weight")?,
                ffn_up_shexp: t("ffn_up_shexp.weight")?,
                ffn_down_shexp: t("ffn_down_shexp.weight")?,
                ffn_gate_tid2eid: if il < N_HASH_LAYER {
                    Some(t("ffn_gate_tid2eid.weight")?)
                } else {
                    None
                },
            });
        }
        let w = Self {
            token_embd,
            output_hc_base,
            output_hc_fn,
            output_hc_scale,
            output_norm,
            output,
            layers,
        };
        w.validate(m)?;
        Ok(w)
    }

    /// `weights_validate_layout`.
    fn validate(&self, m: &Model) -> Result<(), String> {
        let e = u64::from(N_EMBD);
        let v = u64::from(N_VOCAB);
        expect(m, &self.token_embd, F16, 2, [e, v, 0])?;
        expect(m, &self.output_hc_base, F32, 1, [u64::from(N_HC), 0, 0])?;
        expect(m, &self.output_hc_fn, F16, 2, [HC_DIM, u64::from(N_HC), 0])?;
        expect(m, &self.output_hc_scale, F32, 1, [1, 0, 0])?;
        expect(m, &self.output_norm, F32, 1, [e, 0, 0])?;
        expect(m, &self.output, Q8_0, 2, [e, v, 0])?;
        for (il, l) in self.layers.iter().enumerate() {
            let ratio = expected_compress_ratio(il as u32);
            // C++ checks the attention part, the compressors, then the FFN.
            expect_attention_and_compressors(m, l, ratio)?;
            expect_ffn(m, l, F16)?;
            if l.ffn_gate_exps.ty != l.ffn_up_exps.ty {
                return Err(fail(format!(
                    "ds4: routed gate/up experts use different quant types in layer {il}"
                )));
            }
            expect_shared(m, l)?;
            if let Some(hash) = &l.ffn_gate_tid2eid {
                expect(m, hash, I32, 2, [u64::from(N_EXPERT_USED), v, 0])?;
            }
        }
        Ok(())
    }
}

fn expect_attention_and_compressors(m: &Model, l: &LayerWeights, ratio: u32) -> Result<(), String> {
    let e = u64::from(N_EMBD);
    expect(m, &l.hc_attn_fn, F16, 2, [HC_DIM, HC_MIX_DIM, 0])?;
    expect(m, &l.hc_attn_scale, F32, 1, [3, 0, 0])?;
    expect(m, &l.hc_attn_base, F32, 1, [HC_MIX_DIM, 0, 0])?;
    expect(m, &l.attn_norm, F32, 1, [e, 0, 0])?;
    expect(m, &l.attn_q_a, Q8_0, 2, [e, u64::from(N_LORA_Q), 0])?;
    expect(m, &l.attn_q_a_norm, F32, 1, [u64::from(N_LORA_Q), 0, 0])?;
    expect(m, &l.attn_q_b, Q8_0, 2, [u64::from(N_LORA_Q), Q_DIM, 0])?;
    expect(m, &l.attn_kv, Q8_0, 2, [e, u64::from(N_HEAD_DIM), 0])?;
    expect(m, &l.attn_kv_a_norm, F32, 1, [u64::from(N_HEAD_DIM), 0, 0])?;
    expect(m, &l.attn_sinks, F32, 1, [u64::from(N_HEAD), 0, 0])?;
    expect(m, &l.attn_output_a, Q8_0, 2, [OUT_A_IN, OUT_LOW_DIM, 0])?;
    expect(m, &l.attn_output_b, Q8_0, 2, [OUT_LOW_DIM, e, 0])?;
    if ratio != 0 {
        let coff: u64 = if ratio == 4 { 2 } else { 1 };
        let width = coff * u64::from(N_HEAD_DIM);
        let get = |t: &Option<TensorInfo>| t.clone().unwrap_or_default();
        expect(
            m,
            &get(&l.attn_compressor_ape),
            F16,
            2,
            [width, u64::from(ratio), 0],
        )?;
        expect(m, &get(&l.attn_compressor_kv), F16, 2, [e, width, 0])?;
        expect(m, &get(&l.attn_compressor_gate), F16, 2, [e, width, 0])?;
        expect(
            m,
            &get(&l.attn_compressor_norm),
            F32,
            1,
            [u64::from(N_HEAD_DIM), 0, 0],
        )?;
        if ratio == 4 {
            let q_dim = u64::from(N_INDEXER_HEAD) * u64::from(N_INDEXER_HEAD_DIM);
            let index_width = 2 * u64::from(N_INDEXER_HEAD_DIM);
            expect(
                m,
                &get(&l.indexer_attn_q_b),
                F16,
                2,
                [u64::from(N_LORA_Q), q_dim, 0],
            )?;
            expect(
                m,
                &get(&l.indexer_proj),
                F16,
                2,
                [e, u64::from(N_INDEXER_HEAD), 0],
            )?;
            expect(
                m,
                &get(&l.indexer_compressor_ape),
                F16,
                2,
                [index_width, u64::from(ratio), 0],
            )?;
            expect(
                m,
                &get(&l.indexer_compressor_kv),
                F16,
                2,
                [e, index_width, 0],
            )?;
            expect(
                m,
                &get(&l.indexer_compressor_gate),
                F16,
                2,
                [e, index_width, 0],
            )?;
            expect(
                m,
                &get(&l.indexer_compressor_norm),
                F32,
                1,
                [u64::from(N_INDEXER_HEAD_DIM), 0, 0],
            )?;
        }
    }
    Ok(())
}

fn expect_ffn(m: &Model, l: &LayerWeights, router_type: u32) -> Result<(), String> {
    let e = u64::from(N_EMBD);
    expect(m, &l.hc_ffn_fn, F16, 2, [HC_DIM, HC_MIX_DIM, 0])?;
    expect(m, &l.hc_ffn_scale, F32, 1, [3, 0, 0])?;
    expect(m, &l.hc_ffn_base, F32, 1, [HC_MIX_DIM, 0, 0])?;
    expect(m, &l.ffn_norm, F32, 1, [e, 0, 0])?;
    expect(
        m,
        &l.ffn_gate_inp,
        router_type,
        2,
        [e, u64::from(N_EXPERT), 0],
    )?;
    if let Some(bias) = &l.ffn_exp_probs_b {
        expect(m, bias, F32, 1, [u64::from(N_EXPERT), 0, 0])?;
    }
    expect_routed(
        m,
        &l.ffn_gate_exps,
        [e, u64::from(N_FF_EXP), u64::from(N_EXPERT)],
    )?;
    expect_routed(
        m,
        &l.ffn_up_exps,
        [e, u64::from(N_FF_EXP), u64::from(N_EXPERT)],
    )?;
    expect_routed(
        m,
        &l.ffn_down_exps,
        [u64::from(N_FF_EXP), e, u64::from(N_EXPERT)],
    )
}

/// `ds4_routed_expert_row_bytes`: bytes of one row of a routed expert.
pub fn routed_row_bytes(t: &TensorInfo) -> u64 {
    let block = match t.ty {
        IQ2_XXS => 66,
        Q2_K => 84,
        _ => 144,
    };
    t.dim[0] / 256 * block
}
