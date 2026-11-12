//! The metadata checks of the loader (`config_validate_model`): the GGUF
//! must describe exactly the fixed shape of [`crate::shape`], its
//! compression ratios, SwiGLU clamp, RoPE scaling and routing constants.
//! Messages are the C++ ones; a refused artifact prints its reason to
//! stderr and fails with "incompatible or invalid DeepSeek artifact".

use crate::gguf::{Model, VALUE_FLOAT32, VALUE_FLOAT64, VALUE_INT32, VALUE_UINT32};
use crate::shape::*;

pub(crate) const INVALID: &str = "incompatible or invalid DeepSeek artifact";

fn invalid(message: String) -> String {
    eprintln!("{message}");
    INVALID.to_owned()
}

fn required_u32(m: &Model, key: &str) -> Result<u32, String> {
    m.get_u32(key)
        .ok_or_else(|| invalid(format!("ds4: required metadata key is missing: {key}")))
}

fn required_f32(m: &Model, key: &str) -> Result<f32, String> {
    m.get_f32(key)
        .ok_or_else(|| invalid(format!("ds4: required metadata key is missing: {key}")))
}

fn required_bool(m: &Model, key: &str) -> Result<bool, String> {
    m.get_bool(key)
        .ok_or_else(|| invalid(format!("ds4: required metadata key is missing: {key}")))
}

fn expect_u32(name: &str, got: u32, expected: u32) -> Result<(), String> {
    if got == expected {
        return Ok(());
    }
    Err(invalid(format!(
        "ds4: expected {name}={expected} for {MODEL_SHAPE_NAME}, got {got}"
    )))
}

/// `config_expect_f32`: within a relative 1e-6.
fn expect_f32(name: &str, got: f32, expected: f32) -> Result<(), String> {
    let scale = if expected.abs() > 1.0 {
        expected.abs()
    } else {
        1.0
    };
    if (got - expected).abs() <= scale * 1.0e-6 {
        return Ok(());
    }
    Err(invalid(format!(
        "ds4: expected {name}={} for {MODEL_SHAPE_NAME}, got {}",
        g9(expected),
        g9(got)
    )))
}

/// `%.9g`.
fn g9(value: f32) -> String {
    let value = f64::from(value);
    if value == 0.0 {
        return "0".into();
    }
    let exponent = value.abs().log10().floor() as i32;
    if !(-5..9).contains(&exponent) {
        let s = format!("{value:.8e}");
        let (mantissa, exp) = s.split_once('e').unwrap_or((&s, "0"));
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        let exp: i32 = exp.parse().unwrap_or(0);
        return format!(
            "{mantissa}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        );
    }
    let decimals = (8 - exponent).max(0) as usize;
    let s = format!("{value:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        s
    }
}

/// `validate_compress_ratio_metadata`: every layer's ratio is the fixed
/// layout's ([`expected_compress_ratio`]).
fn validate_compress_ratios(m: &Model) -> Result<(), String> {
    let key = "deepseek4.attention.compress_ratios";
    let array = m
        .get_array(key)
        .filter(|a| a.ty == VALUE_UINT32 || a.ty == VALUE_INT32)
        .ok_or_else(|| {
            invalid(format!(
                "ds4: required int32/uint32 array metadata key is missing: {key}"
            ))
        })?;
    if array.len < u64::from(N_LAYER) {
        return Err("deepseek4.attention.compress_ratios is shorter than the layer count".into());
    }
    let mut c = m.array_cursor(array);
    for layer in 0..N_LAYER {
        let got = if array.ty == VALUE_UINT32 {
            c.u32()?
        } else {
            let v = c.i32()?;
            if v < 0 {
                return Err("metadata array contains a negative value".into());
            }
            v as u32
        };
        let expected = expected_compress_ratio(layer);
        if got != expected {
            return Err(invalid(format!(
                "ds4: unexpected DeepSeek4 compression ratio at layer {layer} for {MODEL_SHAPE_NAME}: got {got}, expected {expected}"
            )));
        }
    }
    Ok(())
}

/// `validate_swiglu_clamp_metadata`.
fn validate_swiglu_clamp(m: &Model) -> Result<(), String> {
    let key = "deepseek4.swiglu_clamp_exp";
    let array = m
        .get_array(key)
        .filter(|a| a.ty == VALUE_FLOAT32 || a.ty == VALUE_FLOAT64)
        .ok_or_else(|| {
            invalid(format!(
                "ds4: required float array metadata key is missing: {key}"
            ))
        })?;
    if array.len < u64::from(N_LAYER) {
        return Err("deepseek4.swiglu_clamp_exp is shorter than the layer count".into());
    }
    let mut c = m.array_cursor(array);
    for _ in 0..N_LAYER {
        let got = if array.ty == VALUE_FLOAT32 {
            c.f32()?
        } else {
            c.f64()? as f32
        };
        expect_f32("swiglu_clamp_exp", got, SWIGLU_CLAMP_EXP)?;
    }
    Ok(())
}

/// `config_validate_model`.
pub fn validate(m: &Model) -> Result<(), String> {
    let n_layer = required_u32(m, "deepseek4.block_count")?;
    let n_embd = required_u32(m, "deepseek4.embedding_length")?;
    let n_vocab = required_u32(m, "deepseek4.vocab_size")?;
    let n_head = required_u32(m, "deepseek4.attention.head_count")?;
    let n_head_kv = required_u32(m, "deepseek4.attention.head_count_kv")?;
    let n_head_dim = required_u32(m, "deepseek4.attention.key_length")?;
    let n_value_dim = required_u32(m, "deepseek4.attention.value_length")?;
    let n_rot = required_u32(m, "deepseek4.rope.dimension_count")?;
    let n_lora_q = required_u32(m, "deepseek4.attention.q_lora_rank")?;
    let n_lora_o = required_u32(m, "deepseek4.attention.output_lora_rank")?;
    let n_out_group = required_u32(m, "deepseek4.attention.output_group_count")?;
    let n_expert = required_u32(m, "deepseek4.expert_count")?;
    let n_expert_used = required_u32(m, "deepseek4.expert_used_count")?;
    let n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length")?;
    let n_expert_shared = required_u32(m, "deepseek4.expert_shared_count")?;
    let n_hash_layer = required_u32(m, "deepseek4.hash_layer_count")?;
    let n_expert_groups = m.get_u32("deepseek4.expert_group_count").unwrap_or(0);
    let n_group_used = m.get_u32("deepseek4.expert_group_used_count").unwrap_or(0);
    let n_swa = required_u32(m, "deepseek4.attention.sliding_window")?;
    let n_indexer_head = required_u32(m, "deepseek4.attention.indexer.head_count")?;
    let n_indexer_head_dim = required_u32(m, "deepseek4.attention.indexer.key_length")?;
    let n_indexer_top_k = required_u32(m, "deepseek4.attention.indexer.top_k")?;
    let n_hc = required_u32(m, "deepseek4.hyper_connection.count")?;
    let n_hc_sinkhorn_iter = required_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations")?;

    let matches = [
        (n_layer, N_LAYER),
        (n_embd, N_EMBD),
        (n_vocab, N_VOCAB),
        (n_head, N_HEAD),
        (n_head_kv, N_HEAD_KV),
        (n_head_dim, N_HEAD_DIM),
        (n_value_dim, N_VALUE_DIM),
        (n_rot, N_ROT),
        (n_lora_q, N_LORA_Q),
        (n_lora_o, N_LORA_O),
        (n_out_group, N_OUT_GROUP),
        (n_expert, N_EXPERT),
        (n_expert_used, N_EXPERT_USED),
        (n_ff_exp, N_FF_EXP),
        (n_expert_shared, N_EXPERT_SHARED),
        (n_hash_layer, N_HASH_LAYER),
        (n_swa, N_SWA),
        (n_indexer_head, N_INDEXER_HEAD),
        (n_indexer_head_dim, N_INDEXER_HEAD_DIM),
        (n_indexer_top_k, N_INDEXER_TOP_K),
        (n_hc, N_HC),
        (n_hc_sinkhorn_iter, N_HC_SINKHORN_ITER),
    ]
    .iter()
    .all(|(got, want)| got == want);
    if !matches {
        return Err(invalid(format!(
            "ds4: unsupported DeepSeek4 shape: layers={n_layer} embd={n_embd} heads={n_head} \
             q_lora={n_lora_q} out_groups={n_out_group} experts={n_expert} ff_exp={n_ff_exp} \
             indexer_top_k={n_indexer_top_k}"
        )));
    }
    expect_u32("expert_group_count", n_expert_groups, 0)?;
    expect_u32("expert_group_used_count", n_group_used, 0)?;
    validate_compress_ratios(m)?;
    validate_swiglu_clamp(m)?;

    let rope_orig_ctx = m
        .get_u64("deepseek4.rope.scaling.original_context_length")
        .unwrap_or(ROPE_ORIG_CTX);
    if rope_orig_ctx != ROPE_ORIG_CTX {
        return Err(invalid(format!(
            "ds4: expected rope.scaling.original_context_length={ROPE_ORIG_CTX} for {MODEL_SHAPE_NAME}, got {rope_orig_ctx}"
        )));
    }
    expect_f32(
        "rope.freq_base",
        required_f32(m, "deepseek4.rope.freq_base")?,
        ROPE_FREQ_BASE,
    )?;
    for (key, name, default) in [
        (
            "deepseek4.rope.scaling.factor",
            "rope.scaling.factor",
            ROPE_SCALE_FACTOR,
        ),
        (
            "deepseek4.rope.scaling.yarn_beta_fast",
            "rope.scaling.yarn_beta_fast",
            ROPE_YARN_BETA_FAST,
        ),
        (
            "deepseek4.rope.scaling.yarn_beta_slow",
            "rope.scaling.yarn_beta_slow",
            ROPE_YARN_BETA_SLOW,
        ),
    ] {
        expect_f32(name, m.get_f32(key).unwrap_or(default), default)?;
    }
    expect_f32(
        "attention.compress_rope_freq_base",
        required_f32(m, "deepseek4.attention.compress_rope_freq_base")?,
        COMPRESS_ROPE_FREQ_BASE,
    )?;
    expect_f32(
        "expert_weights_scale",
        required_f32(m, "deepseek4.expert_weights_scale")?,
        EXPERT_WEIGHT_SCALE,
    )?;
    expect_f32(
        "attention.layer_norm_rms_epsilon",
        required_f32(m, "deepseek4.attention.layer_norm_rms_epsilon")?,
        RMS_EPS,
    )?;
    expect_f32(
        "hyper_connection.epsilon",
        required_f32(m, "deepseek4.hyper_connection.epsilon")?,
        HC_EPS,
    )?;
    let norm = required_bool(m, "deepseek4.expert_weights_norm")?;
    if !norm {
        return Err(invalid(format!(
            "ds4: expected expert_weights_norm=true for {MODEL_SHAPE_NAME}, got false"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::g9;

    #[test]
    fn floats_print_like_percent_9g() {
        assert_eq!(g9(10000.0), "10000");
        assert_eq!(g9(1.0e-6), "9.99999997e-07");
        assert_eq!(g9(1.5), "1.5");
        assert_eq!(g9(160000.0), "160000");
        assert_eq!(g9(0.0625), "0.0625");
    }
}
