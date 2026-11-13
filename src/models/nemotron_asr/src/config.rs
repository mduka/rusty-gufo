//! The checkpoint configuration (`config.hpp`, `config.cpp`): `config.json`
//! and `processor_config.json` of a Nemotron 3.5 ASR Hugging Face directory,
//! checked against what the kernels support.

use std::path::Path;

use gufo_core::json::{self, Value};

/// The FastConformer encoder of `config.json`'s `encoder_config`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EncoderConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub conv_kernel: usize,
    pub mel_bins: usize,
    pub subsampling_channels: usize,
    pub subsampling_kernel: usize,
    pub subsampling_stride: usize,
    pub subsampling_factor: usize,
    /// Left attention context plus one, in encoder frames.
    pub sliding_window: usize,
    pub max_positions: usize,
    /// Right attention contexts the checkpoint was trained with, in encoder
    /// frames; the first is the default.
    pub lookaheads: Vec<usize>,
    pub default_lookahead: usize,
}

impl EncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Attention frames left of a chunk.
    pub fn left_context(&self) -> usize {
        self.sliding_window - 1
    }
}

/// The log-mel front end of `processor_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub window: usize,
    pub hop: usize,
    pub preemphasis: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512,
            window: 400,
            hop: 160,
            preemphasis: 0.97,
        }
    }
}

/// Nemotron 3.5 ASR (`Nemotron3_5AsrForRNNT`): the encoder, the language
/// prompt and the RNN-T prediction and joint networks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Config {
    pub encoder: EncoderConfig,
    pub features: FeatureConfig,
    pub decoder_hidden: usize,
    pub decoder_layers: usize,
    pub vocab: usize,
    pub blank: usize,
    pub prompts: usize,
    pub prompt_hidden: usize,
    pub default_prompt: usize,
    pub max_symbols: usize,
    /// Language keys (locales, bare codes, "auto") and their prompt indices.
    pub prompt_dictionary: Vec<(String, usize)>,
}

/// Typed access to one JSON object; failures name the key (`Fields`).
struct Fields<'a> {
    object: &'a Value,
    where_: String,
}

impl<'a> Fields<'a> {
    fn new(object: &'a Value, where_: &str) -> Result<Self, String> {
        if !object.is_object() {
            return Err(format!("{where_} is not a JSON object"));
        }
        Ok(Self {
            object,
            where_: where_.to_owned(),
        })
    }

    fn find(&self, key: &str) -> Option<&'a Value> {
        self.object.find(key).filter(|value| !value.is_null())
    }

    fn get(&self, key: &str) -> Result<&'a Value, String> {
        self.find(key)
            .ok_or_else(|| format!("{} lacks \"{key}\"", self.where_))
    }

    fn index(&self, value: &Value, what: &str) -> Result<usize, String> {
        let number = value.as_double(f64::NAN);
        if !value.is_number() || number < 0.0 || number != number.floor() {
            return Err(format!("{}.{what} is not a natural number", self.where_));
        }
        Ok(number as usize)
    }

    fn size(&self, key: &str) -> Result<usize, String> {
        let size = self.index(self.get(key)?, key)?;
        if size == 0 {
            return Err(format!("{}.{key} is not a positive integer", self.where_));
        }
        Ok(size)
    }

    fn natural(&self, key: &str, fallback: usize) -> Result<usize, String> {
        match self.find(key) {
            None => Ok(fallback),
            Some(value) => self.index(value, key),
        }
    }

    fn number(&self, key: &str, fallback: f64) -> Result<f64, String> {
        let Some(value) = self.find(key) else {
            return Ok(fallback);
        };
        let number = value.as_double(f64::NAN);
        if !value.is_number() || !number.is_finite() {
            return Err(format!("{}.{key} is not a number", self.where_));
        }
        Ok(number)
    }

    fn string(&self, key: &str) -> Result<String, String> {
        let value = self.get(key)?;
        if !value.is_string() {
            return Err(format!("{}.{key} is not a string", self.where_));
        }
        Ok(value.str().to_owned())
    }

    fn boolean(&self, key: &str, fallback: bool) -> Result<bool, String> {
        let Some(value) = self.find(key) else {
            return Ok(fallback);
        };
        if !value.is_bool() {
            return Err(format!("{}.{key} is not a boolean", self.where_));
        }
        Ok(value.as_bool(fallback))
    }

    fn sizes(&self, key: &str) -> Result<Vec<usize>, String> {
        let value = self.get(key)?;
        if !value.is_array() || value.items().is_empty() {
            return Err(format!("{}.{key} is not a non-empty array", self.where_));
        }
        value
            .items()
            .iter()
            .map(|item| self.index(item, key))
            .collect()
    }
}

fn parse_json(text: &str, what: &str) -> Result<Value, String> {
    json::parse_str(text).map_err(|error| format!("{what} is not valid JSON: {error}"))
}

fn read_file(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|_| format!("cannot read {}", path.display()))
}

/// Parses `config.json` and `processor_config.json` texts (`ParseConfig`).
pub fn parse_config(config_json: &str, processor_json: &str) -> Result<Config, String> {
    let root = parse_json(config_json, "config.json")?;
    let fields = Fields::new(&root, "config.json")?;
    if fields.string("model_type")? != "nemotron3_5_asr" {
        return Err("config.json model_type is not nemotron3_5_asr".into());
    }
    let mut config = Config::default();
    let encoder = Fields::new(fields.get("encoder_config")?, "config.json.encoder_config")?;
    let e = &mut config.encoder;
    e.hidden = encoder.size("hidden_size")?;
    e.layers = encoder.size("num_hidden_layers")?;
    e.heads = encoder.size("num_attention_heads")?;
    if encoder.natural("num_key_value_heads", e.heads)? != e.heads {
        return Err("grouped key/value heads are not supported".into());
    }
    e.intermediate = encoder.size("intermediate_size")?;
    e.conv_kernel = encoder.size("conv_kernel_size")?;
    e.mel_bins = encoder.size("num_mel_bins")?;
    e.subsampling_channels = encoder.size("subsampling_conv_channels")?;
    e.subsampling_kernel = encoder.size("subsampling_conv_kernel_size")?;
    e.subsampling_stride = encoder.size("subsampling_conv_stride")?;
    e.subsampling_factor = encoder.size("subsampling_factor")?;
    e.sliding_window = encoder.size("sliding_window")?;
    e.max_positions = encoder.size("max_position_embeddings")?;
    e.lookaheads = encoder.sizes("supported_num_lookahead_tokens")?;
    e.default_lookahead = encoder.natural("default_num_lookahead_tokens", e.lookaheads[0])?;
    if encoder.boolean("attention_bias", false)? || encoder.boolean("convolution_bias", false)? {
        return Err("attention and convolution biases are not supported".into());
    }
    if encoder.boolean("scale_input", false)? {
        return Err("scaled encoder input is not supported".into());
    }
    if encoder.string("hidden_act")? != "silu" {
        return Err("the encoder activation is not silu".into());
    }

    config.decoder_hidden = fields.size("decoder_hidden_size")?;
    config.decoder_layers = fields.size("num_decoder_layers")?;
    config.vocab = fields.size("vocab_size")?;
    config.blank = fields.natural("blank_token_id", config.vocab)?;
    config.prompts = fields.size("num_prompts")?;
    config.prompt_hidden = fields.size("prompt_intermediate_size")?;
    config.default_prompt = fields.natural("default_prompt_id", 0)?;
    config.max_symbols = fields.size("max_symbols_per_step")?;
    if fields.string("hidden_act")? != "relu" {
        return Err("the joint activation is not relu".into());
    }
    if let Some(durations) = fields.find("durations")
        && (!durations.is_array() || !durations.items().is_empty())
    {
        return Err("TDT durations are not supported".into());
    }

    let processor = parse_json(processor_json, "processor_config.json")?;
    let p = Fields::new(&processor, "processor_config.json")?;
    let features = Fields::new(
        p.get("feature_extractor")?,
        "processor_config.json.feature_extractor",
    )?;
    config.features.sample_rate = features.size("sampling_rate")?;
    config.features.n_fft = features.size("n_fft")?;
    config.features.window = features.size("win_length")?;
    config.features.hop = features.size("hop_length")?;
    // transformers' feature extractor defaults to 0.97.
    config.features.preemphasis = features.number("preemphasis", 0.97)?;
    if features.size("feature_size")? != config.encoder.mel_bins {
        return Err("the feature size differs from num_mel_bins".into());
    }
    if p.find("default_num_lookahead_tokens").is_some() {
        config.encoder.default_lookahead = p.natural("default_num_lookahead_tokens", 0)?;
    }
    let dictionary = Fields::new(
        p.get("prompt_dictionary")?,
        "processor_config.json.prompt_dictionary",
    )?;
    for (key, value) in dictionary.object.members() {
        config
            .prompt_dictionary
            .push((key.clone(), dictionary.index(value, key)?));
    }
    Ok(config)
}

/// Reads both files from a model directory (`LoadConfig`).
pub fn load_config(root: &Path) -> Result<Config, String> {
    parse_config(
        &read_file(&root.join("config.json"))?,
        &read_file(&root.join("processor_config.json"))?,
    )
}

/// Frames a query may attend to (left context plus chunk): the attention
/// kernel's limit (`kMaxAttendedFrames`).
pub const MAX_ATTENDED_FRAMES: usize = 256;

/// Whether the kernels support these shapes (`IsSupported`).
pub fn is_supported(config: &Config) -> Result<(), String> {
    let e = &config.encoder;
    let fail = |message: &str| Err(format!("unsupported Nemotron ASR checkpoint: {message}"));
    if !e.hidden.is_multiple_of(e.heads) || e.head_dim() != 128 {
        return fail("attention heads must have 128 dimensions");
    }
    if !e.hidden.is_multiple_of(256) {
        return fail("the hidden size must be a multiple of 256");
    }
    if !e.intermediate.is_multiple_of(32) {
        return fail("the feed-forward size must be a multiple of 32");
    }
    if e.subsampling_kernel != 3 || e.subsampling_stride != 2 || e.subsampling_factor != 8 {
        return fail("subsampling must be three stride-2 convolutions of size 3");
    }
    if e.subsampling_channels == 0
        || !e.subsampling_channels.is_multiple_of(32)
        || e.subsampling_channels > 1024
    {
        return fail("the subsampling channels must be a multiple of 32 up to 1024");
    }
    if e.conv_kernel % 2 != 1 || e.conv_kernel > 31 {
        return fail("the convolution kernel must be odd and at most 31");
    }
    if e.sliding_window < 2 {
        return fail("the attention window is too small");
    }
    for &lookahead in &e.lookaheads {
        if !e.left_context().is_multiple_of(lookahead + 1) {
            return fail("the left context must hold whole chunks");
        }
        if e.left_context() + lookahead + 1 > MAX_ATTENDED_FRAMES {
            return fail(&format!(
                "a query may attend to at most {MAX_ATTENDED_FRAMES} frames"
            ));
        }
    }
    if !e.lookaheads.contains(&e.default_lookahead) {
        return fail("the default lookahead is not a supported one");
    }
    if config.features.sample_rate != 16000 {
        return fail("the feature extractor must expect 16 kHz audio");
    }
    if config.features.n_fft & (config.features.n_fft - 1) != 0 {
        return fail("n_fft must be a power of two");
    }
    if config.features.window > config.features.n_fft {
        return fail("the window is longer than n_fft");
    }
    if config.blank >= config.vocab {
        return fail("the blank token is outside the joint's outputs");
    }
    if !config.vocab.is_multiple_of(16) {
        return fail("the joint's outputs must be a multiple of 16");
    }
    if config.prompt_hidden == 0 || !config.prompt_hidden.is_multiple_of(32) {
        return fail("the prompt projector's width must be a multiple of 32");
    }
    if config.default_prompt >= config.prompts {
        return fail("the default prompt is out of range");
    }
    for (key, index) in &config.prompt_dictionary {
        if *index >= config.prompts {
            return fail(&format!("prompt \"{key}\" is out of range"));
        }
    }
    if !config.decoder_hidden.is_multiple_of(128) || config.decoder_hidden > 1024 {
        return fail("the decoder width must be a multiple of 128 up to 1024");
    }
    if config.max_symbols == 0 {
        return fail("greedy decoding needs at least one symbol per frame");
    }
    Ok(())
}

/// Mel frames of the first streaming chunk (`FirstChunkFrames`).
pub fn first_chunk_frames(config: &Config, lookahead: usize) -> usize {
    1 + config.encoder.subsampling_factor * lookahead
}

/// Mel frames of every later chunk (`ChunkFrames`).
pub fn chunk_frames(config: &Config, lookahead: usize) -> usize {
    config.encoder.subsampling_factor * (lookahead + 1)
}
