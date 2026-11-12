//! The checkpoint configuration (`config.cpp`): `config.json`,
//! `audio_tokenizer/config.json` and `audio_tokenizer/preprocessor_config.json`
//! parsed and checked for a consistent OmniVoice checkpoint.

use std::path::Path;

use gufo_core::json::{self, Value};

/// The Qwen3 backbone (`llm_config`). OmniVoice runs it bidirectionally over
/// text and audio-code positions; its text LM head is unused.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BackboneConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_positions: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

/// HuBERT encoder of the audio tokenizer's semantic branch, at
/// `sample_rate` with one frame per `frame_samples` input samples.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SemanticEncoderConfig {
    pub sample_rate: u32,
    pub frame_samples: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub heads: usize,
    pub layers: usize,
    pub position_kernel: usize,
    pub position_groups: usize,
    pub layer_norm_eps: f32,
    pub conv_channels: Vec<usize>,
    pub conv_kernels: Vec<usize>,
    pub conv_strides: Vec<usize>,
}

/// DAC encoder and decoder of the audio tokenizer's acoustic branch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AcousticConfig {
    pub encoder_channels: usize,
    pub decoder_channels: usize,
    pub latent_channels: usize,
    pub downsampling_ratios: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
}

/// Higgs Audio v2 tokenizer: waveform at `sample_rate` <-> residual vector
/// quantized codes, one frame per `frame_samples` samples.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CodecConfig {
    pub sample_rate: u32,
    pub frame_samples: usize,
    pub codebooks: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    /// Semantic encoder frames per codec frame.
    pub semantic_stride: usize,
    pub semantic: SemanticEncoderConfig,
    pub acoustic: AcousticConfig,
}

impl CodecConfig {
    pub fn frame_rate(&self) -> u32 {
        if self.frame_samples == 0 {
            0
        } else {
            self.sample_rate / self.frame_samples as u32
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Config {
    pub backbone: BackboneConfig,
    /// Codebooks the backbone embeds and predicts per audio frame.
    pub audio_codebooks: usize,
    /// Entries per codebook, including the mask.
    pub audio_vocab_size: usize,
    pub audio_mask_id: u32,
    pub codec: CodecConfig,
}

type ConfigResult<T> = Result<T, String>;

/// Typed access to one JSON object; failures name the file and the key.
struct Fields<'a> {
    object: &'a Value,
    place: String,
}

impl<'a> Fields<'a> {
    fn new(object: &'a Value, place: String) -> ConfigResult<Self> {
        if !object.is_object() {
            return Err(format!("{place} is not a JSON object"));
        }
        Ok(Self { object, place })
    }

    fn get(&self, key: &str) -> ConfigResult<&'a Value> {
        self.object
            .find(key)
            .ok_or_else(|| format!("{} lacks \"{key}\"", self.place))
    }

    fn size(&self, key: &str) -> ConfigResult<usize> {
        let size = self.get(key)?.as_size(0);
        if size == 0 {
            return Err(format!("{}.{key} is not a positive integer", self.place));
        }
        Ok(size)
    }

    fn number(&self, key: &str) -> ConfigResult<f32> {
        let value = self.get(key)?;
        if !value.is_number() || !value.as_double(0.0).is_finite() {
            return Err(format!("{}.{key} is not a number", self.place));
        }
        Ok(value.as_double(0.0) as f32)
    }

    fn string(&self, key: &str) -> ConfigResult<String> {
        let value = self.get(key)?;
        if !value.is_string() {
            return Err(format!("{}.{key} is not a string", self.place));
        }
        Ok(value.str().to_owned())
    }

    fn boolean(&self, key: &str) -> ConfigResult<bool> {
        let value = self.get(key)?;
        if !value.is_bool() {
            return Err(format!("{}.{key} is not a boolean", self.place));
        }
        Ok(value.as_bool(false))
    }

    fn sizes(&self, key: &str) -> ConfigResult<Vec<usize>> {
        let value = self.get(key)?;
        if !value.is_array() || value.is_empty() {
            return Err(format!("{}.{key} is not a non-empty array", self.place));
        }
        value
            .items()
            .iter()
            .map(|item| match item.as_size(0) {
                0 => Err(format!("{}.{key} holds a non-positive entry", self.place)),
                size => Ok(size),
            })
            .collect()
    }

    fn numbers(&self, key: &str) -> ConfigResult<Vec<f64>> {
        let value = self.get(key)?;
        if !value.is_array() || value.is_empty() {
            return Err(format!("{}.{key} is not a non-empty array", self.place));
        }
        value
            .items()
            .iter()
            .map(|item| {
                if !item.is_number() || !item.as_double(0.0).is_finite() {
                    Err(format!("{}.{key} holds a non-number", self.place))
                } else {
                    Ok(item.as_double(0.0))
                }
            })
            .collect()
    }

    fn object(&self, key: &str) -> ConfigResult<Fields<'a>> {
        Fields::new(self.get(key)?, format!("{}.{key}", self.place))
    }

    fn expect(&self, key: &str, expected: &str) -> ConfigResult<()> {
        let actual = self.string(key)?;
        if actual != expected {
            return Err(format!(
                "{}.{key} is \"{actual}\", expected \"{expected}\"",
                self.place
            ));
        }
        Ok(())
    }

    fn expect_bool(&self, key: &str, expected: bool) -> ConfigResult<()> {
        if self.boolean(key)? != expected {
            return Err(format!("{}.{key} must be {expected}", self.place));
        }
        Ok(())
    }
}

fn parse_json(text: &[u8], place: &str) -> ConfigResult<Value> {
    json::parse(text).map_err(|error| format!("{place}: {error}"))
}

fn product(values: &[usize]) -> usize {
    values.iter().product()
}

fn require(condition: bool, message: &str) -> ConfigResult<()> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

fn parse_backbone(llm: &Fields<'_>) -> ConfigResult<BackboneConfig> {
    llm.expect("model_type", "qwen3")?;
    llm.expect("hidden_act", "silu")?;
    llm.expect_bool("attention_bias", false)?;
    let mut backbone = BackboneConfig {
        vocab_size: llm.size("vocab_size")?,
        hidden_size: llm.size("hidden_size")?,
        intermediate_size: llm.size("intermediate_size")?,
        layers: llm.size("num_hidden_layers")?,
        heads: llm.size("num_attention_heads")?,
        kv_heads: llm.size("num_key_value_heads")?,
        head_dim: llm.size("head_dim")?,
        max_positions: llm.size("max_position_embeddings")?,
        rms_norm_eps: llm.number("rms_norm_eps")?,
        rope_theta: 0.0,
    };
    // Transformers 5 moved rope_theta into rope_parameters.
    let rope = llm.object("rope_parameters")?;
    rope.expect("rope_type", "default")?;
    backbone.rope_theta = rope.number("rope_theta")?;
    require(
        backbone.heads.is_multiple_of(backbone.kv_heads),
        "llm_config: attention heads are not a multiple of the KV heads",
    )?;
    require(
        backbone.rms_norm_eps > 0.0 && backbone.rope_theta > 0.0,
        "llm_config: rms_norm_eps and rope_theta must be positive",
    )?;
    Ok(backbone)
}

fn parse_semantic(
    codec: &Fields<'_>,
    semantic: &Fields<'_>,
) -> ConfigResult<SemanticEncoderConfig> {
    semantic.expect("model_type", "hubert")?;
    // The variant the checkpoint uses: group-normalized feature extractor
    // without convolution bias, post-norm encoder layers, GELU throughout.
    semantic.expect("feat_extract_norm", "group")?;
    semantic.expect("feat_extract_activation", "gelu")?;
    semantic.expect("hidden_act", "gelu")?;
    semantic.expect_bool("conv_bias", false)?;
    semantic.expect_bool("do_stable_layer_norm", false)?;
    semantic.expect_bool("feat_proj_layer_norm", true)?;
    let config = SemanticEncoderConfig {
        sample_rate: codec.size("semantic_sample_rate")? as u32,
        frame_samples: codec.size("downsample_factor")?,
        hidden_size: semantic.size("hidden_size")?,
        intermediate_size: semantic.size("intermediate_size")?,
        heads: semantic.size("num_attention_heads")?,
        layers: semantic.size("num_hidden_layers")?,
        position_kernel: semantic.size("num_conv_pos_embeddings")?,
        position_groups: semantic.size("num_conv_pos_embedding_groups")?,
        layer_norm_eps: semantic.number("layer_norm_eps")?,
        conv_channels: semantic.sizes("conv_dim")?,
        conv_kernels: semantic.sizes("conv_kernel")?,
        conv_strides: semantic.sizes("conv_stride")?,
    };
    require(
        config.conv_channels.len() == config.conv_kernels.len()
            && config.conv_kernels.len() == config.conv_strides.len(),
        "semantic_model_config: conv_dim, conv_kernel and conv_stride differ in length",
    )?;
    require(
        product(&config.conv_strides) == config.frame_samples,
        "semantic_model_config: the feature extractor's strides don't multiply to \
         downsample_factor",
    )?;
    require(
        config.hidden_size.is_multiple_of(config.heads)
            && config.hidden_size.is_multiple_of(config.position_groups),
        "semantic_model_config: hidden_size is not a multiple of the heads and the \
         positional convolution groups",
    )?;
    Ok(config)
}

fn parse_acoustic(acoustic: &Fields<'_>, frame_samples: usize) -> ConfigResult<AcousticConfig> {
    acoustic.expect("model_type", "dac")?;
    let config = AcousticConfig {
        encoder_channels: acoustic.size("encoder_hidden_size")?,
        decoder_channels: acoustic.size("decoder_hidden_size")?,
        latent_channels: acoustic.size("hidden_size")?,
        downsampling_ratios: acoustic.sizes("downsampling_ratios")?,
        upsampling_ratios: acoustic.sizes("upsampling_ratios")?,
    };
    require(
        product(&config.downsampling_ratios) == frame_samples
            && product(&config.upsampling_ratios) == frame_samples,
        "acoustic_model_config: the resampling ratios don't multiply to hop_length",
    )?;
    let halvings = 1usize
        .checked_shl(config.upsampling_ratios.len() as u32)
        .unwrap_or(0);
    require(
        halvings != 0 && config.decoder_channels.is_multiple_of(halvings),
        "acoustic_model_config: decoder_hidden_size can't be halved once per upsampling block",
    )?;
    Ok(config)
}

fn parse_codec(codec_root: &Value, preprocessor_root: &Value) -> ConfigResult<CodecConfig> {
    let codec = Fields::new(codec_root, "audio_tokenizer/config.json".into())?;
    let preprocessor = Fields::new(
        preprocessor_root,
        "audio_tokenizer/preprocessor_config.json".into(),
    )?;
    codec.expect("model_type", "higgs_audio_v2_tokenizer")?;
    let acoustic = codec.object("acoustic_model_config")?;
    let mut config = CodecConfig {
        sample_rate: preprocessor.size("sampling_rate")? as u32,
        frame_samples: preprocessor.size("hop_length")?,
        ..CodecConfig::default()
    };
    // acoustic_model_config.sampling_rate (16 kHz) is a leftover of the DAC
    // base config; the codec runs at the preprocessor's rate.
    require(
        acoustic.size("hop_length")? == config.frame_samples,
        "audio_tokenizer: the acoustic hop_length differs from the preprocessor's",
    )?;
    require(
        codec.size("sample_rate")? == config.sample_rate as usize,
        "audio_tokenizer: sample_rate differs from the preprocessor's",
    )?;
    require(
        (config.sample_rate as usize).is_multiple_of(config.frame_samples),
        "audio_tokenizer: hop_length doesn't divide the sample rate",
    )?;
    config.codebook_size = codec.size("codebook_size")?;
    config.codebook_dim = codec.size("codebook_dim")?;
    // Higgs Audio v2 quantizes with as many codebooks as its highest target
    // bandwidth (kbit/s) carries: bandwidth / (frame rate * bits per code).
    let bandwidths = codec.numbers("target_bandwidths")?;
    let bits = (config.codebook_size as f64).log2();
    let last = bandwidths.last().copied().unwrap_or(0.0);
    let codebooks = 1000.0 * last / (f64::from(config.frame_rate()) * bits);
    require(
        bits == bits.floor() && codebooks >= 1.0 && codebooks == codebooks.floor(),
        "audio_tokenizer: target_bandwidths imply a fractional codebook count",
    )?;
    config.codebooks = codebooks as usize;
    config.semantic = parse_semantic(&codec, &codec.object("semantic_model_config")?)?;
    config.acoustic = parse_acoustic(&acoustic, config.frame_samples)?;
    // HuBERT frames (16 kHz / 320) are pooled down to the codec frame rate.
    let semantic_rate = config.semantic.sample_rate as usize / config.semantic.frame_samples;
    require(
        (config.semantic.sample_rate as usize).is_multiple_of(config.semantic.frame_samples)
            && semantic_rate.is_multiple_of(config.frame_rate() as usize),
        "audio_tokenizer: the semantic frame rate is not a multiple of the codec frame rate",
    )?;
    config.semantic_stride = semantic_rate / config.frame_rate() as usize;
    Ok(config)
}

/// Parses `config.json`, `audio_tokenizer/config.json` and
/// `audio_tokenizer/preprocessor_config.json` and checks that they describe a
/// consistent OmniVoice checkpoint (`ParseConfig`).
pub fn parse_config(
    model_json: &[u8],
    codec_json: &[u8],
    preprocessor_json: &[u8],
) -> Result<Config, String> {
    let model_root = parse_json(model_json, "config.json")?;
    let codec_root = parse_json(codec_json, "audio_tokenizer/config.json")?;
    let preprocessor_root = parse_json(
        preprocessor_json,
        "audio_tokenizer/preprocessor_config.json",
    )?;
    let model = Fields::new(&model_root, "config.json".into())?;
    model.expect("model_type", "omnivoice")?;
    let config = Config {
        backbone: parse_backbone(&model.object("llm_config")?)?,
        audio_codebooks: model.size("num_audio_codebook")?,
        audio_vocab_size: model.size("audio_vocab_size")?,
        audio_mask_id: model.size("audio_mask_id")? as u32,
        codec: parse_codec(&codec_root, &preprocessor_root)?,
    };
    require(
        config.audio_codebooks == config.codec.codebooks,
        "config.json: num_audio_codebook differs from the audio tokenizer's codebook count",
    )?;
    require(
        config.audio_vocab_size == config.codec.codebook_size + 1
            && config.audio_mask_id as usize == config.codec.codebook_size,
        "config.json: the audio vocabulary must be the codebook plus one mask entry at its end",
    )?;
    Ok(config)
}

fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    // Configuration files are a few KiB; refuse anything implausibly large.
    const MAX_BYTES: u64 = 1 << 20;
    let metadata = std::fs::metadata(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if metadata.len() > MAX_BYTES {
        return Err(format!(
            "{} is too large for a configuration",
            path.display()
        ));
    }
    std::fs::read(path).map_err(|_| format!("cannot read {}", path.display()))
}

/// Reads and parses the three files of the model directory `model_root`
/// (`LoadConfig`).
pub fn load_config(model_root: &Path) -> Result<Config, String> {
    let model = read_file(&model_root.join("config.json"))?;
    let codec = read_file(&model_root.join("audio_tokenizer").join("config.json"))?;
    let preprocessor = read_file(
        &model_root
            .join("audio_tokenizer")
            .join("preprocessor_config.json"),
    )?;
    parse_config(&model, &codec, &preprocessor)
}

/// True when this crate implements the checkpoint's architecture
/// (`IsSupported`); the error names what is missing.
pub fn is_supported(config: &Config) -> Result<(), String> {
    if config.backbone.head_dim != 128 {
        return Err("the backbone's attention kernels need a head size of 128".into());
    }
    if config.codec.codebooks != 8 {
        return Err("the generator handles exactly 8 audio codebooks".into());
    }
    if config.codec.semantic.conv_channels.first() != config.codec.semantic.conv_channels.last() {
        return Err("the semantic feature extractor must keep its width".into());
    }
    Ok(())
}
