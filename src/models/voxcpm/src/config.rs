//! The checkpoint configuration (`config.hpp`, `config.cpp`): `config.json`
//! of an openbmb VoxCPM2 directory (or the copy an audio.cpp GGUF embeds),
//! parsed with the reference's pydantic defaults and checked against what
//! the kernels support.

use std::path::Path;

use gufo_core::json::{self, Value};

/// One MiniCPM4 transformer stack. VoxCPM2 derives its four stacks (text LM,
/// residual LM, local encoder, local DiT) from `lm_config`, overriding the
/// sizes per stack as the reference model does.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// 0 for the stacks without a token embedding.
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope: bool,
    /// muP scaling: embeddings times `scale_emb`, residual branches times
    /// `scale_depth / sqrt(layers)`. VoxCPM2 disables it.
    pub use_mup: bool,
    pub scale_emb: f32,
    pub scale_depth: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            hidden: 0,
            intermediate: 0,
            layers: 0,
            heads: 0,
            kv_heads: 0,
            head_dim: 0,
            vocab: 0,
            rms_eps: 0.0,
            rope: true,
            use_mup: false,
            scale_emb: 1.0,
            scale_depth: 1.0,
        }
    }
}

/// MiniCPM's LongRoPE. The reference fills its table up to
/// `max_position_embeddings` once, so the long factors apply to every
/// position when that exceeds the original length and the short ones
/// otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    pub theta: f64,
    pub dim: usize,
    pub max_positions: usize,
    pub original_max_positions: usize,
    pub short_factor: Vec<f64>,
    pub long_factor: Vec<f64>,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            theta: 10000.0,
            dim: 0,
            max_positions: 0,
            original_max_positions: 0,
            short_factor: Vec::new(),
            long_factor: Vec::new(),
        }
    }
}

impl RopeConfig {
    /// The factors the reference applies (one per frequency).
    pub fn factors(&self) -> &[f64] {
        if self.max_positions > self.original_max_positions {
            &self.long_factor
        } else {
            &self.short_factor
        }
    }

    /// The magnitude scale of the cos/sin table.
    pub fn scaling(&self) -> f64 {
        let scale = self.max_positions as f64 / self.original_max_positions as f64;
        (1.0 + scale.ln() / (self.original_max_positions as f64).ln()).sqrt()
    }
}

/// AudioVAE V2: causal DAC-style encoder at `sample_rate`, decoder to
/// `out_sample_rate` conditioned on the output rate's bucket.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub sample_rate: usize,
    pub out_sample_rate: usize,
    pub sr_bin_boundaries: Vec<usize>,
}

impl AudioVaeConfig {
    /// Input samples per latent frame.
    pub fn hop(&self) -> usize {
        self.encoder_rates.iter().product()
    }

    /// Output samples per latent frame.
    pub fn decode_hop(&self) -> usize {
        self.decoder_rates.iter().product()
    }

    /// The decoder's sample-rate bucket for `out_sample_rate`
    /// (torch.bucketize, right=False).
    pub fn rate_bucket(&self) -> usize {
        self.sr_bin_boundaries
            .iter()
            .take_while(|&&boundary| boundary < self.out_sample_rate)
            .count()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Config {
    pub lm: TransformerConfig,
    pub residual_lm: TransformerConfig,
    pub encoder: TransformerConfig,
    pub dit: TransformerConfig,
    /// head_dim of every stack is lm.head_dim's.
    pub rope: RopeConfig,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub fsq_latent_dim: usize,
    pub fsq_scale: f32,
    pub vae: AudioVaeConfig,
    /// Positions in each LM's KV cache (prompt plus generated patches).
    pub max_length: usize,
    pub dtype: String,
}

type ConfigResult<T> = Result<T, String>;

/// Typed access to one JSON object; failures name the key. Optional keys
/// take the reference's pydantic defaults; JSON null counts as absent.
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

    fn find(&self, key: &str) -> Option<&'a Value> {
        self.object.find(key).filter(|value| !value.is_null())
    }

    fn get(&self, key: &str) -> ConfigResult<&'a Value> {
        self.find(key)
            .ok_or_else(|| format!("{} lacks \"{key}\"", self.place))
    }

    fn size(&self, key: &str) -> ConfigResult<usize> {
        match self.get(key)?.as_size(0) {
            0 => Err(format!("{}.{key} is not a positive integer", self.place)),
            size => Ok(size),
        }
    }

    fn size_or(&self, key: &str, fallback: usize) -> ConfigResult<usize> {
        if self.find(key).is_none() {
            Ok(fallback)
        } else {
            self.size(key)
        }
    }

    fn number(&self, key: &str) -> ConfigResult<f64> {
        let value = self.get(key)?;
        if !value.is_number() || !value.as_double(0.0).is_finite() {
            return Err(format!("{}.{key} is not a number", self.place));
        }
        Ok(value.as_double(0.0))
    }

    fn number_or(&self, key: &str, fallback: f64) -> ConfigResult<f64> {
        if self.find(key).is_none() {
            Ok(fallback)
        } else {
            self.number(key)
        }
    }

    fn string_or(&self, key: &str, fallback: &str) -> ConfigResult<String> {
        match self.find(key) {
            None => Ok(fallback.to_owned()),
            Some(value) if value.is_string() => Ok(value.str().to_owned()),
            Some(_) => Err(format!("{}.{key} is not a string", self.place)),
        }
    }

    fn bool_or(&self, key: &str, fallback: bool) -> ConfigResult<bool> {
        match self.find(key) {
            None => Ok(fallback),
            Some(value) if value.is_bool() => Ok(value.as_bool(false)),
            Some(_) => Err(format!("{}.{key} is not a boolean", self.place)),
        }
    }

    fn sizes_or(&self, key: &str, fallback: &[usize]) -> ConfigResult<Vec<usize>> {
        let Some(value) = self.find(key) else {
            return Ok(fallback.to_vec());
        };
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
                if item.is_number() && item.as_double(0.0).is_finite() {
                    Ok(item.as_double(0.0))
                } else {
                    Err(format!("{}.{key} holds a non-number", self.place))
                }
            })
            .collect()
    }

    fn object(&self, key: &str) -> ConfigResult<Fields<'a>> {
        Fields::new(self.get(key)?, format!("{}.{key}", self.place))
    }

    fn has(&self, key: &str) -> bool {
        self.find(key).is_some()
    }
}

/// The shape of a derived stack: VoxCPMEncoderConfig / VoxCPMDitConfig
/// override hidden size, FFN size, heads, layers and kv_channels of a copy
/// of lm_config (kv_channels None falls back to hidden / heads).
fn derive_stack(lm: &TransformerConfig, fields: &Fields<'_>) -> ConfigResult<TransformerConfig> {
    let mut stack = lm.clone();
    stack.hidden = fields.size_or("hidden_dim", 1024)?;
    stack.intermediate = fields.size_or("ffn_dim", 4096)?;
    stack.heads = fields.size_or("num_heads", 16)?;
    stack.layers = fields.size_or("num_layers", 4)?;
    stack.head_dim = fields.size_or("kv_channels", stack.hidden / stack.heads)?;
    stack.vocab = 0;
    Ok(stack)
}

fn parse(text: &[u8]) -> ConfigResult<Config> {
    let root = json::parse(text).map_err(|failure| format!("config.json: {failure}"))?;
    let config = Fields::new(&root, "config.json".into())?;
    let architecture = config.string_or("architecture", "")?;
    if architecture != "voxcpm2" {
        return Err(format!(
            "config.json names architecture \"{architecture}\", expected \"voxcpm2\""
        ));
    }

    let lm = config.object("lm_config")?;
    let mut base = TransformerConfig {
        hidden: lm.size("hidden_size")?,
        intermediate: lm.size("intermediate_size")?,
        layers: lm.size("num_hidden_layers")?,
        heads: lm.size("num_attention_heads")?,
        kv_heads: lm.size("num_key_value_heads")?,
        ..TransformerConfig::default()
    };
    base.head_dim = lm.size_or("kv_channels", base.hidden / base.heads)?;
    base.vocab = lm.size("vocab_size")?;
    base.rms_eps = lm.number("rms_norm_eps")? as f32;
    base.rope = !lm.bool_or("no_rope", false)?;
    base.use_mup = lm.bool_or("use_mup", true)?;
    base.scale_emb = lm.number("scale_emb")? as f32;
    base.scale_depth = lm.number("scale_depth")? as f32;

    let scaling = lm.object("rope_scaling")?;
    let rope = RopeConfig {
        theta: lm.number("rope_theta")?,
        dim: base.head_dim,
        max_positions: lm.size("max_position_embeddings")?,
        original_max_positions: scaling.size("original_max_position_embeddings")?,
        short_factor: scaling.numbers("short_factor")?,
        long_factor: scaling.numbers("long_factor")?,
    };

    let mut residual_lm = base.clone();
    residual_lm.layers = config.size_or("residual_lm_num_layers", 8)?;
    residual_lm.vocab = 0;
    residual_lm.rope = !config.bool_or("residual_lm_no_rope", false)?;
    let encoder = derive_stack(&base, &config.object("encoder_config")?)?;
    let dit_fields = config.object("dit_config")?;
    let dit = derive_stack(&base, &dit_fields)?;
    if dit_fields.bool_or("dit_mean_mode", false)? {
        return Err("config.json: dit_mean_mode is not supported".into());
    }

    let patch_size = config.size_or("patch_size", 4)?;
    let feat_dim = config.size_or("feat_dim", 64)?;
    let fsq_latent_dim = config.size_or("scalar_quantization_latent_dim", 512)?;
    let fsq_scale = config.number_or("scalar_quantization_scale", 9.0)? as f32;
    let max_length = config.size_or("max_length", 8192)?;
    let dtype = config.string_or("dtype", "bfloat16")?;

    // AudioVAEConfigV2's defaults apply to absent keys.
    let mut vae = AudioVaeConfig {
        encoder_dim: 128,
        encoder_rates: vec![2, 5, 8, 8],
        latent_dim: 64,
        decoder_dim: 2048,
        decoder_rates: vec![8, 6, 5, 2, 2, 2],
        sample_rate: 16000,
        out_sample_rate: 48000,
        sr_bin_boundaries: vec![20000, 30000, 40000],
    };
    if config.has("audio_vae_config") {
        let fields = config.object("audio_vae_config")?;
        vae.encoder_dim = fields.size_or("encoder_dim", vae.encoder_dim)?;
        vae.encoder_rates = fields.sizes_or("encoder_rates", &vae.encoder_rates)?;
        vae.latent_dim = fields.size_or("latent_dim", vae.latent_dim)?;
        vae.decoder_dim = fields.size_or("decoder_dim", vae.decoder_dim)?;
        vae.decoder_rates = fields.sizes_or("decoder_rates", &vae.decoder_rates)?;
        vae.sample_rate = fields.size_or("sample_rate", vae.sample_rate)?;
        vae.out_sample_rate = fields.size_or("out_sample_rate", vae.out_sample_rate)?;
        vae.sr_bin_boundaries = fields.sizes_or("sr_bin_boundaries", &vae.sr_bin_boundaries)?;
        if !fields.bool_or("depthwise", true)?
            || fields.bool_or("use_noise_block", false)?
            || fields.string_or("cond_type", "scale_bias")? != "scale_bias"
            || fields.bool_or("cond_out_layer", false)?
        {
            return Err(
                "config.json: only the depthwise AudioVAE V2 with scale_bias \
                        rate conditioning and no noise blocks is supported"
                    .into(),
            );
        }
    }

    Ok(Config {
        lm: base,
        residual_lm,
        encoder,
        dit,
        rope,
        patch_size,
        feat_dim,
        fsq_latent_dim,
        fsq_scale,
        vae,
        max_length,
        dtype,
    })
}

/// Parses the text of `config.json`.
pub fn parse_config(text: &str) -> Result<Config, String> {
    parse(text.as_bytes())
}

/// Reads `config.json` of an openbmb VoxCPM2 directory.
pub fn load_config(model_root: &Path) -> Result<Config, String> {
    let path = model_root.join("config.json");
    let text = std::fs::read(&path).map_err(|_| format!("cannot read {}", path.display()))?;
    parse(&text)
}

/// Whether this build's kernels support the configuration.
pub fn is_supported(config: &Config) -> Result<(), String> {
    for stack in [
        &config.lm,
        &config.residual_lm,
        &config.encoder,
        &config.dit,
    ] {
        if stack.head_dim != 128 {
            return Err("the attention kernels need a head size of 128".into());
        }
        if stack.kv_heads == 0 || !stack.heads.is_multiple_of(stack.kv_heads) {
            return Err("attention heads must be a multiple of the KV heads".into());
        }
    }
    if config.lm.use_mup {
        return Err("muP-scaled checkpoints are not supported".into());
    }
    if config.rope.short_factor.len() * 2 != config.rope.dim
        || config.rope.long_factor.len() * 2 != config.rope.dim
    {
        return Err("the LongRoPE factors do not match the head size".into());
    }
    if config.feat_dim != config.vae.latent_dim {
        return Err("feat_dim differs from the AudioVAE latent size".into());
    }
    if config
        .vae
        .decoder_dim
        .checked_shr(config.vae.decoder_rates.len() as u32)
        .unwrap_or(0)
        == 0
    {
        return Err("the AudioVAE decoder halves its channels below one".into());
    }
    if config.dtype != "bfloat16" {
        return Err("only bfloat16 checkpoints are supported".into());
    }
    Ok(())
}
