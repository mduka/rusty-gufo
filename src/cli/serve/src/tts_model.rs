//! The Qwen3-TTS model contract the serving layer needs
//! (`src/models/qwen3_tts/{synthesis,sampling,audio,config}.hpp`,
//! `text_stream.hpp`): request and result types, sampling validation,
//! reference WAV decoding, the checkpoint config checks and speech segment
//! splitting. Synthesis itself is injected (the `gufo-qwen3-tts` engine).

use std::collections::BTreeMap;
use std::path::Path;

use gufo_core::json::{self, Value};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ModelVariant {
    Base,
    VoiceDesign,
    #[default]
    CustomVoice,
    Unsupported,
}

/// Interleaved float audio.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingOptions {
    pub sample: bool,
    pub seed: u32,
    pub top_k: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub predictor_sample: bool,
    pub predictor_top_k: usize,
    pub predictor_temperature: f32,
    pub predictor_top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            sample: true,
            seed: 42,
            top_k: 50,
            temperature: 0.9,
            top_p: 1.0,
            predictor_sample: true,
            predictor_top_k: 50,
            predictor_temperature: 0.9,
            predictor_top_p: 1.0,
            repetition_penalty: 1.05,
        }
    }
}

/// `ValidSamplingOptions`: finite positive temperatures and repetition
/// penalty, top-p in (0, 1].
pub fn valid_sampling_options(options: &SamplingOptions) -> bool {
    let temperature = |v: f32| v.is_finite() && v > 0.0;
    let top_p = |v: f32| v.is_finite() && v > 0.0 && v <= 1.0;
    temperature(options.temperature)
        && temperature(options.predictor_temperature)
        && temperature(options.repetition_penalty)
        && top_p(options.top_p)
        && top_p(options.predictor_top_p)
}

/// `SynthesisRequest`. The C++ `on_audio` callback is a separate argument
/// of the runner, so the request stays plain data.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesisRequest {
    pub text: String,
    pub speaker: String,
    pub language: String,
    pub instruct: String,
    pub reference_audio: AudioBuffer,
    pub reference_text: String,
    pub speaker_embedding_only: bool,
    pub max_new_tokens: usize,
    pub sampling: SamplingOptions,
}

impl Default for SynthesisRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            speaker: "vivian".into(),
            language: "english".into(),
            instruct: String::new(),
            reference_audio: AudioBuffer::default(),
            reference_text: String::new(),
            speaker_embedding_only: false,
            max_new_tokens: 3000,
            sampling: SamplingOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SynthesisResult {
    pub sample_rate: u32,
    pub code_groups: u32,
    /// Empty when streamed.
    pub samples: Vec<f32>,
    /// Frame-major `[frames, code_groups]`.
    pub codes: Vec<i32>,
    pub sample_count: usize,
}

/// Receives newly generated mono audio at 24 kHz; returning false cancels.
pub type AudioSink<'a> = &'a mut dyn FnMut(&[f32]) -> bool;

/// The synthesis entry point (`TtsServiceOptions::Runner`): the request, an
/// optional streaming sink and a cancellation check polled during
/// generation. The error is a human-readable message.
pub type TtsRunner = Box<
    dyn Fn(
            &SynthesisRequest,
            Option<AudioSink<'_>>,
            &dyn Fn() -> bool,
        ) -> Result<SynthesisResult, String>
        + Send
        + Sync,
>;

/// The parts of `config.json` the service checks (`ModelConfig`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelConfig {
    pub model_type: String,
    pub tokenizer_type: String,
    pub tts_model_size: String,
    pub tts_model_type: String,
    pub variant: ModelVariant,
    /// `talker_config.spk_id`: speaker names of a CustomVoice checkpoint.
    pub speakers: BTreeMap<String, u32>,
    pub has_speaker_encoder: bool,
}

pub fn parse_model_variant(model_type: &str) -> ModelVariant {
    match model_type {
        "base" => ModelVariant::Base,
        "voice_design" => ModelVariant::VoiceDesign,
        "custom_voice" => ModelVariant::CustomVoice,
        _ => ModelVariant::Unsupported,
    }
}

/// Only the quality-qualified 1.7B 12 Hz family; Base also needs its
/// speaker encoder.
pub fn is_supported_model_config(config: &ModelConfig) -> bool {
    if config.model_type != "qwen3_tts"
        || config.tokenizer_type != "qwen3_tts_tokenizer_12hz"
        || config.tts_model_size != "1b7"
        || config.variant == ModelVariant::Unsupported
    {
        return false;
    }
    config.variant != ModelVariant::Base || config.has_speaker_encoder
}

pub fn parse_model_config(contents: &[u8]) -> Option<ModelConfig> {
    let root = json::parse(contents).ok()?;
    if !root.is_object() {
        return None;
    }
    let tts_model_type = root.member_str("tts_model_type", "");
    let mut config = ModelConfig {
        model_type: root.member_str("model_type", ""),
        tokenizer_type: root.member_str("tokenizer_type", ""),
        tts_model_size: root.member_str("tts_model_size", ""),
        variant: parse_model_variant(&tts_model_type),
        tts_model_type,
        ..ModelConfig::default()
    };
    if let Some(Value::Object(members)) = root
        .find("talker_config")
        .and_then(|talker| talker.find("spk_id"))
    {
        for (name, value) in members {
            if value.is_number() {
                config
                    .speakers
                    .insert(name.clone(), value.as_size(0) as u32);
            }
        }
    }
    config.has_speaker_encoder = root
        .find("speaker_encoder_config")
        .is_some_and(Value::is_object);
    Some(config)
}

/// Reads `<model_dir>/config.json`.
pub fn load_model_config(model_dir: &Path) -> Option<ModelConfig> {
    parse_model_config(&std::fs::read(model_dir.join("config.json")).ok()?)
}

const MAXIMUM_ENCODED_BYTES: usize = 12 << 20;
const MAXIMUM_CHANNELS: u32 = 8;
const MAXIMUM_SAMPLE_RATE: u32 = 192_000;
const MAXIMUM_FRAMES: u64 = 30 * MAXIMUM_SAMPLE_RATE as u64;

fn base64_value(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some(u32::from(c - b'A')),
        b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Base64 with an optional `data:audio/wav;base64,` prefix; padding only
/// at the end.
fn decode_base64(encoded: &str) -> Result<Vec<u8>, String> {
    let encoded = encoded
        .strip_prefix("data:audio/wav;base64,")
        .unwrap_or(encoded)
        .as_bytes();
    let invalid = || "Qwen3-TTS reference audio base64 is invalid".to_owned();
    let bad_padding = || "Qwen3-TTS reference audio base64 padding is invalid".to_owned();
    if encoded.is_empty()
        || encoded.len() > MAXIMUM_ENCODED_BYTES
        || !encoded.len().is_multiple_of(4)
    {
        return Err(invalid());
    }
    let mut output = Vec::with_capacity(encoded.len() / 4 * 3);
    for (index, quad) in encoded.chunks(4).enumerate() {
        let mut block: u32 = 0;
        let mut padding = 0;
        for (position, &c) in quad.iter().enumerate() {
            if c == b'=' {
                if position < 2 || (index + 1) * 4 != encoded.len() {
                    return Err(bad_padding());
                }
                padding += 1;
                block <<= 6;
            } else {
                if padding != 0 {
                    return Err(bad_padding());
                }
                block = (block << 6) | base64_value(c).ok_or_else(invalid)?;
            }
        }
        output.push(((block >> 16) & 0xff) as u8);
        if padding < 2 {
            output.push(((block >> 8) & 0xff) as u8);
        }
        if padding == 0 {
            output.push((block & 0xff) as u8);
        }
    }
    Ok(output)
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// `DecodeWav`: PCM16 or float32 interleaved RIFF WAV of at most 30 s.
pub fn decode_wav(bytes: &[u8]) -> Result<AudioBuffer, String> {
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Qwen3-TTS reference audio must be a RIFF WAV".into());
    }
    let (mut format, mut channels, mut sample_rate, mut block_align, mut bits) = (0, 0, 0, 0, 0);
    let mut data: &[u8] = &[];
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let chunk_size = read_u32(bytes, offset + 4) as usize;
        let data_offset = offset + 8;
        if chunk_size > bytes.len() - data_offset {
            return Err("Qwen3-TTS reference WAV is truncated".into());
        }
        match &bytes[offset..offset + 4] {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err("Qwen3-TTS reference WAV fmt is invalid".into());
                }
                format = read_u16(bytes, data_offset);
                channels = u32::from(read_u16(bytes, data_offset + 2));
                sample_rate = read_u32(bytes, data_offset + 4);
                block_align = usize::from(read_u16(bytes, data_offset + 12));
                bits = read_u16(bytes, data_offset + 14);
            }
            b"data" => data = &bytes[data_offset..data_offset + chunk_size],
            _ => {}
        }
        let padded = chunk_size + (chunk_size & 1);
        if padded > bytes.len() - data_offset {
            break;
        }
        offset = data_offset + padded;
    }
    if (format != 1 && format != 3)
        || channels == 0
        || channels > MAXIMUM_CHANNELS
        || sample_rate == 0
        || sample_rate > MAXIMUM_SAMPLE_RATE
        || data.is_empty()
    {
        return Err("Qwen3-TTS reference WAV format is unsupported".into());
    }
    let bytes_per_sample = usize::from(bits / 8);
    if (format == 1 && bits != 16)
        || (format == 3 && bits != 32)
        || block_align != channels as usize * bytes_per_sample
        || !data.len().is_multiple_of(block_align)
    {
        return Err("Qwen3-TTS supports PCM16 or float32 interleaved WAV".into());
    }
    let frames = (data.len() / block_align) as u64;
    if frames == 0 || frames > MAXIMUM_FRAMES || frames > 30 * u64::from(sample_rate) {
        return Err("Qwen3-TTS reference WAV must be at most 30 seconds".into());
    }
    let count = frames as usize * channels as usize;
    let mut samples = Vec::with_capacity(count);
    for index in 0..count {
        let offset = index * bytes_per_sample;
        if format == 1 {
            samples.push(f32::from(read_u16(data, offset) as i16) / 32768.0);
        } else {
            let value = f32::from_bits(read_u32(data, offset));
            if !value.is_finite() {
                return Err("Qwen3-TTS reference WAV contains non-finite samples".into());
            }
            samples.push(value.clamp(-1.0, 1.0));
        }
    }
    Ok(AudioBuffer {
        sample_rate,
        channels,
        samples,
    })
}

/// `DecodeBase64Wav`.
pub fn decode_base64_wav(encoded: &str) -> Result<AudioBuffer, String> {
    decode_wav(&decode_base64(encoded)?)
}

/// `TakeSpeechSegments`: splits complete sentences (or clauses) off
/// `pending`; with `flush` the rest too. Boundaries depend on the text, not
/// on packet sizes: Latin punctuation waits for whitespace so a decimal
/// point does not split. Opt-in, since splitting changes the model's
/// future context.
pub fn take_speech_segments(pending: &mut String, granularity: &str, flush: bool) -> Vec<String> {
    let mut result = Vec::new();
    if granularity != "none" {
        let bytes = pending.as_bytes();
        let clause = granularity == "clause";
        let mut start = 0;
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i];
            let latin =
                matches!(c, b'.' | b'!' | b'?') || (clause && matches!(c, b',' | b';' | b':'));
            let mut end = 0;
            if c == b'\n' {
                end = i + 1;
            }
            if latin && i + 1 < bytes.len() && matches!(bytes[i + 1], b' ' | b'\n' | b'\t') {
                end = i + 1;
            }
            let tail = &bytes[i..];
            for mark in ["。", "！", "？", "।"] {
                if tail.starts_with(mark.as_bytes()) {
                    end = i + mark.len();
                }
            }
            if clause {
                for mark in ["，", "；", "：", "،"] {
                    if tail.starts_with(mark.as_bytes()) {
                        end = i + mark.len();
                    }
                }
            }
            if end != 0 {
                result.push(pending[start..end].to_owned());
                start = end;
                i = end;
                continue;
            }
            i += 1;
        }
        pending.drain(..start);
    }
    if flush && !pending.is_empty() {
        result.push(std::mem::take(pending));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_checks_follow_the_checkpoint_family() {
        let base = br#"{"model_type":"qwen3_tts","tokenizer_type":"qwen3_tts_tokenizer_12hz",
            "tts_model_size":"1b7","tts_model_type":"base","speaker_encoder_config":{}}"#;
        let config = parse_model_config(base).expect("config");
        assert_eq!(config.variant, ModelVariant::Base);
        assert!(is_supported_model_config(&config));
        let custom = br#"{"model_type":"qwen3_tts","tokenizer_type":"qwen3_tts_tokenizer_12hz",
            "tts_model_size":"1b7","tts_model_type":"custom_voice",
            "talker_config":{"spk_id":{"vivian":3,"ryan":"x","aiden":1}}}"#;
        let config = parse_model_config(custom).expect("config");
        assert!(is_supported_model_config(&config));
        assert_eq!(
            config.speakers.keys().collect::<Vec<_>>(),
            ["aiden", "vivian"]
        );
        let small = br#"{"model_type":"qwen3_tts","tokenizer_type":"qwen3_tts_tokenizer_12hz",
            "tts_model_size":"0b6","tts_model_type":"base","speaker_encoder_config":{}}"#;
        assert!(!is_supported_model_config(
            &parse_model_config(small).expect("config")
        ));
        assert!(parse_model_config(b"[]").is_none());
    }

    #[test]
    fn reference_wav_decodes_pcm16() {
        let wav = decode_base64_wav(
            "UklGRiwAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQgAAAAAAOgDGPwAAA==",
        )
        .expect("wav");
        assert_eq!((wav.sample_rate, wav.channels), (24000, 1));
        assert_eq!(wav.samples, [0.0, 1000.0 / 32768.0, -1000.0 / 32768.0, 0.0]);
        assert!(decode_base64_wav("UklG").is_err());
        assert!(decode_base64_wav("A===").is_err());
    }
}
