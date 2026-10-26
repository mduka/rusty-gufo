//! The DSpark support model (`dspark_internal.h`, `ds4_dspark_open`): a
//! separate GGUF of `stage_count` DS4 blocks without compressors that drafts
//! a block of tokens from the target's hidden states, plus a Markov path
//! selector. Binding and checks are as strict as the target's.

use crate::config::INVALID;
use crate::gguf::{Model, TensorInfo, VALUE_UINT32};
use crate::shape::tensor_type::*;
use crate::shape::*;
use crate::weights::{LayerWeights, expect, expect_stage_block};

pub const MAX_STAGES: u32 = 8;
pub const MAX_BLOCK: u32 = 16;
pub const MAX_TARGET_LAYERS: u32 = 8;
pub const RETRY_TOKENS: u32 = 64;

/// One stage (`ds4_dspark_stage_weights`): the fusion of the target
/// features on the first, the head on the last.
#[derive(Debug, Clone, Default)]
pub struct StageWeights {
    pub main_proj: Option<TensorInfo>,
    pub main_norm: Option<TensorInfo>,
    pub norm: Option<TensorInfo>,
    pub hc_head_base: Option<TensorInfo>,
    pub hc_head_fn: Option<TensorInfo>,
    pub hc_head_scale: Option<TensorInfo>,
    pub markov_w1: Option<TensorInfo>,
    pub markov_w2: Option<TensorInfo>,
    pub confidence: Option<TensorInfo>,
    pub block: LayerWeights,
}

/// The bound support model (`ds4_dspark_model`).
pub struct DsparkModel {
    pub model: Model,
    pub block_size: u32,
    pub markov_rank: u32,
    pub noise_token_id: u32,
    pub target_layer_ids: Vec<u32>,
    pub stages: Vec<StageWeights>,
}

fn fail(message: String) -> String {
    eprintln!("{message}");
    INVALID.to_owned()
}

fn required_u32(m: &Model, key: &str) -> Result<u32, String> {
    m.get_u32(key).ok_or_else(|| {
        fail(format!(
            "ds4: DSpark support model is missing metadata key {key}"
        ))
    })
}

fn stage_tensor(m: &Model, suffix: &str, stage: u32) -> Result<TensorInfo, String> {
    let name = format!("mtp.{stage}.{suffix}");
    m.find_tensor(&name).cloned().ok_or_else(|| {
        fail(format!(
            "ds4: DSpark support model is missing tensor {name}"
        ))
    })
}

impl DsparkModel {
    /// Total width of the concatenated target features `main_proj` reads.
    pub fn feature_width(&self) -> u32 {
        self.target_layer_ids.len() as u32 * N_EMBD
    }

    pub fn n_stages(&self) -> u32 {
        self.stages.len() as u32
    }

    /// `dspark_validate_metadata` and `dspark_bind`.
    pub fn bind(model: Model) -> Result<Self, String> {
        let m = &model;
        let architecture = m.get_string("general.architecture").unwrap_or_default();
        if architecture != b"deepseek4-dspark" {
            return Err(fail(format!(
                "ds4: expected general.architecture=deepseek4-dspark for the DSpark support model, got '{}'",
                String::from_utf8_lossy(architecture)
            )));
        }
        let n_stages = required_u32(m, "dspark.stage_count")?;
        let block_size = required_u32(m, "dspark.block_size")?;
        let markov_rank = required_u32(m, "dspark.markov_rank")?;
        let noise_token_id = required_u32(m, "dspark.noise_token_id")?;
        if n_stages == 0 || n_stages > MAX_STAGES {
            return Err(fail(format!(
                "ds4: DSpark stage_count={n_stages} is outside 1..{MAX_STAGES}"
            )));
        }
        if !(2..=MAX_BLOCK).contains(&block_size) {
            return Err(fail(format!(
                "ds4: DSpark block_size={block_size} is outside 2..{MAX_BLOCK}"
            )));
        }
        if markov_rank == 0 || markov_rank % 32 != 0 {
            return Err(fail(format!(
                "ds4: DSpark markov_rank={markov_rank} must be a non-zero multiple of the 32-element Q8_0 block"
            )));
        }
        if noise_token_id >= N_VOCAB {
            return Err(fail(format!(
                "ds4: DSpark noise_token_id={noise_token_id} is outside the vocabulary"
            )));
        }
        let layer_count = required_u32(m, "dspark.n_layers")?;
        if layer_count != n_stages {
            return Err(fail(format!(
                "ds4: DSpark n_layers={layer_count} disagrees with stage_count={n_stages}"
            )));
        }
        let layers = m
            .get_array("dspark.target_layer_ids")
            .filter(|a| a.ty == VALUE_UINT32 && a.len != 0 && a.len <= u64::from(MAX_TARGET_LAYERS))
            .ok_or_else(|| {
                fail(format!(
                    "ds4: DSpark target_layer_ids must list 1..{MAX_TARGET_LAYERS} uint32 target layers"
                ))
            })?;
        let mut cursor = m.array_cursor(layers);
        let mut target_layer_ids: Vec<u32> = Vec::new();
        for i in 0..layers.len {
            let value = cursor.u32().ok().filter(|&v| v < N_LAYER).ok_or_else(|| {
                fail(format!(
                    "ds4: DSpark target_layer_ids[{i}] is not a valid target layer"
                ))
            })?;
            if target_layer_ids.last().is_some_and(|&last| value <= last) {
                return Err(fail(
                    "ds4: DSpark target_layer_ids must be strictly increasing".into(),
                ));
            }
            target_layer_ids.push(value);
        }

        let mut stages = Vec::with_capacity(n_stages as usize);
        for stage in 0..n_stages {
            let t = |suffix: &str| stage_tensor(m, suffix, stage);
            let block = LayerWeights {
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
                hc_ffn_fn: t("hc_ffn_fn.weight")?,
                hc_ffn_scale: t("hc_ffn_scale.weight")?,
                hc_ffn_base: t("hc_ffn_base.weight")?,
                ffn_norm: t("ffn_norm.weight")?,
                ffn_gate_inp: t("ffn_gate_inp.weight")?,
                ffn_exp_probs_b: m
                    .find_tensor(&format!("mtp.{stage}.exp_probs_b.bias"))
                    .cloned(),
                ffn_gate_exps: t("ffn_gate_exps.weight")?,
                ffn_up_exps: t("ffn_up_exps.weight")?,
                ffn_down_exps: t("ffn_down_exps.weight")?,
                ffn_gate_shexp: t("ffn_gate_shexp.weight")?,
                ffn_up_shexp: t("ffn_up_shexp.weight")?,
                ffn_down_shexp: t("ffn_down_shexp.weight")?,
                ..LayerWeights::default()
            };
            expect_stage_block(m, &block, stage)?;
            stages.push(StageWeights {
                block,
                ..StageWeights::default()
            });
        }
        let feature_width = u64::from(target_layer_ids.len() as u32 * N_EMBD);
        let e = u64::from(N_EMBD);
        let main_proj = stage_tensor(m, "main_proj.weight", 0)?;
        let main_norm = stage_tensor(m, "main_norm.weight", 0)?;
        expect(m, &main_proj, Q8_0, 2, [feature_width, e, 0])?;
        expect(m, &main_norm, F32, 1, [e, 0, 0])?;
        stages[0].main_proj = Some(main_proj);
        stages[0].main_norm = Some(main_norm);
        let last = n_stages - 1;
        let norm = stage_tensor(m, "norm.weight", last)?;
        let hc_head_base = stage_tensor(m, "hc_head_base.weight", last)?;
        let hc_head_fn = stage_tensor(m, "hc_head_fn.weight", last)?;
        let hc_head_scale = stage_tensor(m, "hc_head_scale.weight", last)?;
        let markov_w1 = stage_tensor(m, "markov_head.markov_w1.weight", last)?;
        let markov_w2 = stage_tensor(m, "markov_head.markov_w2.weight", last)?;
        expect(m, &norm, F32, 1, [e, 0, 0])?;
        expect(m, &hc_head_base, F32, 1, [u64::from(N_HC), 0, 0])?;
        expect(
            m,
            &hc_head_fn,
            F16,
            2,
            [e * u64::from(N_HC), u64::from(N_HC), 0],
        )?;
        expect(m, &hc_head_scale, F32, 1, [1, 0, 0])?;
        expect(
            m,
            &markov_w1,
            Q8_0,
            2,
            [u64::from(markov_rank), u64::from(N_VOCAB), 0],
        )?;
        expect(
            m,
            &markov_w2,
            Q8_0,
            2,
            [u64::from(markov_rank), u64::from(N_VOCAB), 0],
        )?;
        let confidence = stage_tensor(m, "confidence_head.proj.weight", last)?;
        expect(m, &confidence, Q8_0, 2, [e + u64::from(markov_rank), 1, 0])?;
        let tail = &mut stages[last as usize];
        tail.norm = Some(norm);
        tail.hc_head_base = Some(hc_head_base);
        tail.hc_head_fn = Some(hc_head_fn);
        tail.hc_head_scale = Some(hc_head_scale);
        tail.markov_w1 = Some(markov_w1);
        tail.markov_w2 = Some(markov_w2);
        tail.confidence = Some(confidence);
        Ok(Self {
            model,
            block_size,
            markov_rank,
            noise_token_id,
            target_layer_ids,
            stages,
        })
    }
}
