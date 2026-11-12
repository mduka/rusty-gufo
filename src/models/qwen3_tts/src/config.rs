//! `config.json` and `speech_tokenizer/config.json` of a Qwen3-TTS
//! checkpoint (`config.cpp`). Only the quality-qualified 12Hz 1.7B family is
//! supported; the 0.6B and the experimental 25Hz checkpoints are rejected.

use std::collections::BTreeMap;
use std::path::Path;

use gufo_core::json::{self, Value};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ModelVariant {
    Base,
    VoiceDesign,
    CustomVoice,
    #[default]
    Unsupported,
}

/// `ParseModelVariant`: the `tts_model_type` of the checkpoint.
pub fn parse_model_variant(model_type: &str) -> ModelVariant {
    match model_type {
        "base" => ModelVariant::Base,
        "voice_design" => ModelVariant::VoiceDesign,
        "custom_voice" => ModelVariant::CustomVoice,
        _ => ModelVariant::Unsupported,
    }
}

/// `talker_config`: the main decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct TalkerConfig {
    pub vocab_size: u32,
    pub text_vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub num_code_groups: u32,
    pub text_hidden_size: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: Vec<u32>,
    pub mrope_interleaved: bool,
    pub codec_eos_token_id: u32,
    pub codec_bos_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub spk_id: BTreeMap<String, u32>,
    pub codec_language_id: BTreeMap<String, u32>,
    /// `spk_is_dialect`: a dialect name, or `None` for `false`.
    pub spk_dialect: BTreeMap<String, Option<String>>,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 3072,
            text_vocab_size: 151_936,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            num_code_groups: 16,
            text_hidden_size: 2048,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            mrope_section: vec![24, 20, 20],
            mrope_interleaved: true,
            codec_eos_token_id: 2150,
            codec_bos_id: 2149,
            codec_pad_id: 2148,
            codec_think_id: 2154,
            codec_nothink_id: 2155,
            codec_think_bos_id: 2156,
            codec_think_eos_id: 2157,
            spk_id: BTreeMap::new(),
            codec_language_id: BTreeMap::new(),
            spk_dialect: BTreeMap::new(),
        }
    }
}

/// `talker_config.code_predictor_config`: the sub-codebook predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct CodePredictorConfig {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub num_code_groups: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Default for CodePredictorConfig {
    fn default() -> Self {
        Self {
            vocab_size: 2048,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 5,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            num_code_groups: 16,
            max_position_embeddings: 65536,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

/// `speech_tokenizer/config.json`: the codec decoder and (Base) encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechTokenizerConfig {
    pub model_type: String,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    pub decode_upsample_rate: u32,
    pub latent_dim: u32,
    pub codebook_dim: u32,
    pub codebook_size: u32,
    pub decoder_dim: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub head_dim: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: u32,
    pub num_quantizers: u32,
    pub num_semantic_quantizers: u32,
    pub sliding_window: u32,
    pub layer_scale_initial_scale: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub upsample_rates: Vec<u32>,
    pub upsampling_ratios: Vec<u32>,

    pub encode_downsample_rate: u32,
    pub encoder_valid_num_quantizers: u32,
    pub encoder_audio_channels: u32,
    pub encoder_codebook_dim: u32,
    pub encoder_codebook_size: u32,
    pub encoder_hidden_size: u32,
    pub encoder_intermediate_size: u32,
    pub encoder_head_dim: u32,
    pub encoder_num_attention_heads: u32,
    pub encoder_num_hidden_layers: u32,
    pub encoder_num_key_value_heads: u32,
    pub encoder_num_quantizers: u32,
    pub encoder_num_semantic_quantizers: u32,
    pub encoder_num_filters: u32,
    pub encoder_num_residual_layers: u32,
    pub encoder_kernel_size: u32,
    pub encoder_last_kernel_size: u32,
    pub encoder_residual_kernel_size: u32,
    pub encoder_compress: u32,
    pub encoder_dilation_growth_rate: u32,
    pub encoder_sliding_window: u32,
    pub encoder_layer_scale_initial_scale: f32,
    pub encoder_norm_eps: f32,
    pub encoder_rope_theta: f32,
    pub encoder_upsampling_ratios: Vec<u32>,
}

impl Default for SpeechTokenizerConfig {
    fn default() -> Self {
        Self {
            model_type: String::new(),
            input_sample_rate: 24000,
            output_sample_rate: 24000,
            decode_upsample_rate: 1920,
            latent_dim: 1024,
            codebook_dim: 512,
            codebook_size: 2048,
            decoder_dim: 1536,
            hidden_size: 512,
            intermediate_size: 1024,
            head_dim: 64,
            num_attention_heads: 16,
            num_hidden_layers: 8,
            num_key_value_heads: 16,
            num_quantizers: 16,
            num_semantic_quantizers: 1,
            sliding_window: 72,
            layer_scale_initial_scale: 0.01,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            upsample_rates: vec![8, 5, 4, 3],
            upsampling_ratios: vec![2, 2],
            encode_downsample_rate: 1920,
            encoder_valid_num_quantizers: 16,
            encoder_audio_channels: 1,
            encoder_codebook_dim: 256,
            encoder_codebook_size: 2048,
            encoder_hidden_size: 512,
            encoder_intermediate_size: 2048,
            encoder_head_dim: 64,
            encoder_num_attention_heads: 8,
            encoder_num_hidden_layers: 8,
            encoder_num_key_value_heads: 8,
            encoder_num_quantizers: 32,
            encoder_num_semantic_quantizers: 1,
            encoder_num_filters: 64,
            encoder_num_residual_layers: 1,
            encoder_kernel_size: 7,
            encoder_last_kernel_size: 3,
            encoder_residual_kernel_size: 3,
            encoder_compress: 2,
            encoder_dilation_growth_rate: 2,
            encoder_sliding_window: 250,
            encoder_layer_scale_initial_scale: 0.01,
            encoder_norm_eps: 1e-5,
            encoder_rope_theta: 10000.0,
            encoder_upsampling_ratios: vec![8, 6, 5, 4],
        }
    }
}

/// `speaker_encoder_config` (Base checkpoints).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerEncoderConfig {
    pub embedding_dim: u32,
    pub sample_rate: u32,
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 2048,
            sample_rate: 24000,
        }
    }
}

/// The whole checkpoint configuration (`ModelConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub model_type: String,
    pub tokenizer_type: String,
    pub tts_model_size: String,
    pub tts_model_type: String,
    pub variant: ModelVariant,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    pub tts_pad_token_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub talker: TalkerConfig,
    pub code_predictor: CodePredictorConfig,
    pub speech_tokenizer: SpeechTokenizerConfig,
    pub speaker_encoder: Option<SpeakerEncoderConfig>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: String::new(),
            tokenizer_type: String::new(),
            tts_model_size: String::new(),
            tts_model_type: String::new(),
            variant: ModelVariant::Unsupported,
            im_start_token_id: 151_644,
            im_end_token_id: 151_645,
            tts_pad_token_id: 151_671,
            tts_bos_token_id: 151_672,
            tts_eos_token_id: 151_673,
            talker: TalkerConfig::default(),
            code_predictor: CodePredictorConfig::default(),
            speech_tokenizer: SpeechTokenizerConfig::default(),
            speaker_encoder: None,
        }
    }
}

impl ModelConfig {
    /// `IsSupportedModelConfig`.
    pub fn is_supported(&self) -> bool {
        if self.model_type != "qwen3_tts"
            || self.tokenizer_type != "qwen3_tts_tokenizer_12hz"
            || self.tts_model_size != "1b7"
            || self.variant == ModelVariant::Unsupported
        {
            return false;
        }
        self.variant != ModelVariant::Base || self.speaker_encoder.is_some()
    }

    /// `ParseModelConfig`: `None` for invalid JSON or a non-object root.
    pub fn parse(contents: &str) -> Option<Self> {
        let root = json::parse_str(contents).ok()?;
        if !root.is_object() {
            return None;
        }
        let mut config = Self {
            model_type: root.member_str("model_type", ""),
            tokenizer_type: root.member_str("tokenizer_type", ""),
            tts_model_size: root.member_str("tts_model_size", ""),
            tts_model_type: root.member_str("tts_model_type", ""),
            ..Self::default()
        };
        config.variant = parse_model_variant(&config.tts_model_type);
        parse_u32(&root, "im_start_token_id", &mut config.im_start_token_id);
        parse_u32(&root, "im_end_token_id", &mut config.im_end_token_id);
        parse_u32(&root, "tts_pad_token_id", &mut config.tts_pad_token_id);
        parse_u32(&root, "tts_bos_token_id", &mut config.tts_bos_token_id);
        parse_u32(&root, "tts_eos_token_id", &mut config.tts_eos_token_id);
        if let Some(talker) = root.find("talker_config") {
            parse_talker(&mut config.talker, talker);
            if let Some(predictor) = talker
                .is_object()
                .then(|| talker.find("code_predictor_config"))
                .flatten()
            {
                parse_code_predictor(&mut config.code_predictor, predictor);
            }
        }
        if let Some(speaker) = root.find("speaker_encoder_config")
            && speaker.is_object()
        {
            let mut parsed = SpeakerEncoderConfig::default();
            parse_u32(speaker, "enc_dim", &mut parsed.embedding_dim);
            parse_u32(speaker, "sample_rate", &mut parsed.sample_rate);
            config.speaker_encoder = Some(parsed);
        }
        Some(config)
    }

    /// `LoadModelConfigFromPath`: `config.json`, plus the speech tokenizer's
    /// configuration when it is readable.
    pub fn load(model_dir: &Path) -> Option<Self> {
        let contents = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
        let mut config = Self::parse(&contents)?;
        let _ = load_speech_tokenizer_config(model_dir, &mut config.speech_tokenizer);
        Some(config)
    }
}

/// `LoadSpeechTokenizerConfigFromPath`.
pub fn load_speech_tokenizer_config(model_dir: &Path, config: &mut SpeechTokenizerConfig) -> bool {
    let path = model_dir.join("speech_tokenizer").join("config.json");
    let Ok(contents) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(root) = json::parse_str(&contents) else {
        return false;
    };
    if !root.is_object() {
        return false;
    }
    config.model_type = root.member_str("model_type", "");
    parse_u32(&root, "input_sample_rate", &mut config.input_sample_rate);
    parse_u32(&root, "output_sample_rate", &mut config.output_sample_rate);
    parse_u32(
        &root,
        "decode_upsample_rate",
        &mut config.decode_upsample_rate,
    );
    parse_u32(
        &root,
        "encode_downsample_rate",
        &mut config.encode_downsample_rate,
    );
    parse_u32(
        &root,
        "encoder_valid_num_quantizers",
        &mut config.encoder_valid_num_quantizers,
    );
    if let Some(decoder) = root.find("decoder_config")
        && decoder.is_object()
    {
        parse_u32(decoder, "latent_dim", &mut config.latent_dim);
        parse_u32(decoder, "codebook_dim", &mut config.codebook_dim);
        parse_u32(decoder, "codebook_size", &mut config.codebook_size);
        parse_u32(decoder, "decoder_dim", &mut config.decoder_dim);
        parse_u32(decoder, "hidden_size", &mut config.hidden_size);
        parse_u32(decoder, "intermediate_size", &mut config.intermediate_size);
        parse_u32(decoder, "head_dim", &mut config.head_dim);
        parse_u32(
            decoder,
            "num_attention_heads",
            &mut config.num_attention_heads,
        );
        parse_u32(decoder, "num_hidden_layers", &mut config.num_hidden_layers);
        parse_u32(
            decoder,
            "num_key_value_heads",
            &mut config.num_key_value_heads,
        );
        parse_u32(decoder, "num_quantizers", &mut config.num_quantizers);
        parse_u32(
            decoder,
            "num_semantic_quantizers",
            &mut config.num_semantic_quantizers,
        );
        parse_u32(decoder, "sliding_window", &mut config.sliding_window);
        parse_f32(
            decoder,
            "layer_scale_initial_scale",
            &mut config.layer_scale_initial_scale,
        );
        parse_f32(decoder, "rms_norm_eps", &mut config.rms_norm_eps);
        parse_f32(decoder, "rope_theta", &mut config.rope_theta);
        parse_u32_array(decoder, "upsample_rates", &mut config.upsample_rates);
        parse_u32_array(decoder, "upsampling_ratios", &mut config.upsampling_ratios);
    }
    if let Some(encoder) = root.find("encoder_config")
        && encoder.is_object()
    {
        parse_speech_encoder(config, encoder);
    }
    true
}

fn parse_u32(object: &Value, key: &str, out: &mut u32) {
    if !object.is_object() {
        return;
    }
    if let Some(value) = object.find(key)
        && value.is_number()
    {
        *out = value.as_size(*out as usize) as u32;
    }
}

fn parse_f32(object: &Value, key: &str, out: &mut f32) {
    if let Some(value) = object.find(key)
        && value.is_number()
    {
        *out = value.as_double(f64::from(*out)) as f32;
    }
}

/// The numeric items of an array (`ParseU32Array`).
fn u32_array(value: &Value) -> Vec<u32> {
    if !value.is_array() {
        return Vec::new();
    }
    value
        .items()
        .iter()
        .filter(|item| item.is_number())
        .map(|item| item.as_size(0) as u32)
        .collect()
}

/// Replaces `out` with a non-empty parsed array.
fn parse_u32_array(object: &Value, key: &str, out: &mut Vec<u32>) {
    if let Some(value) = object.find(key) {
        let parsed = u32_array(value);
        if !parsed.is_empty() {
            *out = parsed;
        }
    }
}

fn string_u32_map(value: &Value) -> BTreeMap<String, u32> {
    let mut out = BTreeMap::new();
    if !value.is_object() {
        return out;
    }
    for (key, member) in value.members() {
        if member.is_number() {
            out.insert(key.clone(), member.as_size(0) as u32);
        }
    }
    out
}

/// `ParseSpeakerDialects`: `false` maps to `None`, a string to its value;
/// the first occurrence of a key wins.
fn speaker_dialects(value: &Value) -> BTreeMap<String, Option<String>> {
    let mut out = BTreeMap::new();
    if !value.is_object() {
        return out;
    }
    for (key, member) in value.members() {
        if member.is_bool() {
            if !member.as_bool(true) {
                out.entry(key.clone()).or_insert(None);
            }
        } else if member.is_string() {
            out.entry(key.clone())
                .or_insert_with(|| Some(member.str().to_owned()));
        }
    }
    out
}

fn parse_talker(out: &mut TalkerConfig, talker: &Value) {
    parse_u32(talker, "vocab_size", &mut out.vocab_size);
    parse_u32(talker, "text_vocab_size", &mut out.text_vocab_size);
    parse_u32(talker, "hidden_size", &mut out.hidden_size);
    parse_u32(talker, "intermediate_size", &mut out.intermediate_size);
    parse_u32(talker, "num_hidden_layers", &mut out.num_hidden_layers);
    parse_u32(talker, "num_attention_heads", &mut out.num_attention_heads);
    parse_u32(talker, "num_key_value_heads", &mut out.num_key_value_heads);
    parse_u32(talker, "head_dim", &mut out.head_dim);
    parse_u32(talker, "num_code_groups", &mut out.num_code_groups);
    parse_u32(talker, "text_hidden_size", &mut out.text_hidden_size);
    parse_u32(
        talker,
        "max_position_embeddings",
        &mut out.max_position_embeddings,
    );
    parse_u32(talker, "codec_eos_token_id", &mut out.codec_eos_token_id);
    parse_u32(talker, "codec_bos_id", &mut out.codec_bos_id);
    parse_u32(talker, "codec_pad_id", &mut out.codec_pad_id);
    parse_u32(talker, "codec_think_id", &mut out.codec_think_id);
    parse_u32(talker, "codec_nothink_id", &mut out.codec_nothink_id);
    parse_u32(talker, "codec_think_bos_id", &mut out.codec_think_bos_id);
    parse_u32(talker, "codec_think_eos_id", &mut out.codec_think_eos_id);
    if !talker.is_object() {
        return;
    }
    parse_f32(talker, "rms_norm_eps", &mut out.rms_norm_eps);
    parse_f32(talker, "rope_theta", &mut out.rope_theta);
    if let Some(speaker) = talker.find("spk_id") {
        out.spk_id = string_u32_map(speaker);
    }
    if let Some(language) = talker.find("codec_language_id") {
        out.codec_language_id = string_u32_map(language);
    }
    if let Some(dialect) = talker.find("spk_is_dialect") {
        out.spk_dialect = speaker_dialects(dialect);
    }
    if let Some(rope) = talker.find("rope_scaling")
        && rope.is_object()
    {
        parse_u32_array(rope, "mrope_section", &mut out.mrope_section);
        if let Some(interleaved) = rope.find("interleaved")
            && interleaved.is_bool()
        {
            out.mrope_interleaved = interleaved.as_bool(true);
        }
    }
}

fn parse_code_predictor(out: &mut CodePredictorConfig, config: &Value) {
    parse_u32(config, "vocab_size", &mut out.vocab_size);
    parse_u32(config, "hidden_size", &mut out.hidden_size);
    parse_u32(config, "intermediate_size", &mut out.intermediate_size);
    parse_u32(config, "num_hidden_layers", &mut out.num_hidden_layers);
    parse_u32(config, "num_attention_heads", &mut out.num_attention_heads);
    parse_u32(config, "num_key_value_heads", &mut out.num_key_value_heads);
    parse_u32(config, "head_dim", &mut out.head_dim);
    parse_u32(config, "num_code_groups", &mut out.num_code_groups);
    parse_u32(
        config,
        "max_position_embeddings",
        &mut out.max_position_embeddings,
    );
    if !config.is_object() {
        return;
    }
    parse_f32(config, "rms_norm_eps", &mut out.rms_norm_eps);
    parse_f32(config, "rope_theta", &mut out.rope_theta);
}

fn parse_speech_encoder(out: &mut SpeechTokenizerConfig, encoder: &Value) {
    parse_u32(encoder, "audio_channels", &mut out.encoder_audio_channels);
    parse_u32(encoder, "codebook_dim", &mut out.encoder_codebook_dim);
    parse_u32(encoder, "codebook_size", &mut out.encoder_codebook_size);
    parse_u32(encoder, "hidden_size", &mut out.encoder_hidden_size);
    parse_u32(
        encoder,
        "intermediate_size",
        &mut out.encoder_intermediate_size,
    );
    parse_u32(encoder, "head_dim", &mut out.encoder_head_dim);
    parse_u32(
        encoder,
        "num_attention_heads",
        &mut out.encoder_num_attention_heads,
    );
    parse_u32(
        encoder,
        "num_hidden_layers",
        &mut out.encoder_num_hidden_layers,
    );
    parse_u32(
        encoder,
        "num_key_value_heads",
        &mut out.encoder_num_key_value_heads,
    );
    parse_u32(encoder, "num_quantizers", &mut out.encoder_num_quantizers);
    parse_u32(
        encoder,
        "num_semantic_quantizers",
        &mut out.encoder_num_semantic_quantizers,
    );
    parse_u32(encoder, "num_filters", &mut out.encoder_num_filters);
    parse_u32(
        encoder,
        "num_residual_layers",
        &mut out.encoder_num_residual_layers,
    );
    parse_u32(encoder, "kernel_size", &mut out.encoder_kernel_size);
    parse_u32(
        encoder,
        "last_kernel_size",
        &mut out.encoder_last_kernel_size,
    );
    parse_u32(
        encoder,
        "residual_kernel_size",
        &mut out.encoder_residual_kernel_size,
    );
    parse_u32(encoder, "compress", &mut out.encoder_compress);
    parse_u32(
        encoder,
        "dilation_growth_rate",
        &mut out.encoder_dilation_growth_rate,
    );
    parse_u32(encoder, "sliding_window", &mut out.encoder_sliding_window);
    parse_f32(
        encoder,
        "layer_scale_initial_scale",
        &mut out.encoder_layer_scale_initial_scale,
    );
    parse_f32(encoder, "norm_eps", &mut out.encoder_norm_eps);
    parse_f32(encoder, "rope_theta", &mut out.encoder_rope_theta);
    parse_u32_array(
        encoder,
        "upsampling_ratios",
        &mut out.encoder_upsampling_ratios,
    );
}
