//! End-to-end transcription, a port of `hip/transcription_runtime.cpp`: WAV
//! decoding, chunking by the context budget, log-mel features, the audio
//! encoder and greedy text decoding per chunk, admitted per chunk so that
//! several requests share the device.

use std::path::Path;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use gufo_core::cancellable_gate::CancellableGate;
use gufo_hip::{Placement, Stream};

use crate::audio::{self, SAMPLE_RATE};
use crate::config::ModelConfig;
use crate::decoder::{DecoderTrace, TextDecoder, is_stop_token};
use crate::encoder::{AudioEncoder, EncoderTrace};
use crate::prompt::{build_prompt, parse_transcription};
use crate::safetensors::Checkpoint;
use crate::tokenizer::Tokenizer;
use crate::weights::{SHARDS, Weights};
use gufo_core::utf8::{Utf8Decoder, lossy};

/// The audio of a request: a WAV file, or mono samples at 16 kHz.
pub enum Audio<'a> {
    Wav(&'a [u8]),
    Pcm16k(&'a [f32]),
}

pub struct Request<'a> {
    pub audio: Audio<'a>,
    /// Optional transcription context (names, terms), part of the system turn.
    pub context: &'a str,
    /// Language name or ISO code; `None`, empty or `auto` detects it.
    pub language: Option<&'a str>,
    /// Generated-token limit per audio chunk.
    pub max_new_tokens: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Timings {
    pub decode_audio_ms: f64,
    pub feature_extraction_ms: f64,
    pub audio_encoder_ms: f64,
    pub text_decoder_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Transcription {
    /// Detected (or forced) languages, comma-separated when chunks differ.
    pub language: String,
    pub text: String,
    /// Raw decoded answers of all chunks, including language metadata.
    pub decoded: Vec<u8>,
    pub generated_ids: Vec<u32>,
    pub sample_rate: u32,
    pub audio_samples: usize,
    pub mel_frames: usize,
    pub audio_tokens: usize,
    pub prompt_tokens: usize,
    pub chunks: usize,
    pub timings: Timings,
}

/// Device weights, encoder and decoder state on one stream.
struct Device {
    weights: Weights,
    encoder: AudioEncoder,
    decoder: TextDecoder,
    // Declared last: dropped after every buffer of the stream.
    stream: Stream,
}

/// Loaded model: tokenizer and the device. Requests may come from several
/// threads and are admitted as in C++: at most two hold decoded audio and
/// features at a time, and each audio chunk takes the device in FIFO order,
/// so a long recording yields to shorter requests between its chunks.
pub struct Transcriber {
    config: ModelConfig,
    tokenizer: Tokenizer,
    capacity: usize,
    frontend_gate: CancellableGate,
    device_gate: CancellableGate,
    device: Mutex<Device>,
    /// The mapped shards stay mapped like the C++ runtime's: unmapping the
    /// prefaulted checkpoint would add ~50 ms to every load.
    _checkpoint: Checkpoint,
}

fn milliseconds(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

const ISO_NAMES: [(&str, &str); 30] = [
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("fil", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
];

/// The model's name for a requested language; `None` detects it.
pub fn normalize_language(
    requested: Option<&str>,
    supported: &[String],
) -> Result<Option<String>, String> {
    let Some(requested) = requested.filter(|r| !r.is_empty()) else {
        return Ok(None);
    };
    let lower = requested.to_ascii_lowercase();
    if lower == "auto" {
        return Ok(None);
    }
    if let Some((_, name)) = ISO_NAMES.iter().find(|(code, _)| *code == lower) {
        return Ok(Some((*name).to_string()));
    }
    supported
        .iter()
        .find(|language| language.to_ascii_lowercase() == lower)
        .map(|language| Some(language.clone()))
        .ok_or_else(|| format!("Qwen3-ASR language is unsupported: {requested}"))
}

impl Transcriber {
    /// Loads the checkpoint in `model_root` for prompts plus generated tokens
    /// of up to `capacity` tokens per chunk.
    pub fn load(model_root: &Path, capacity: usize) -> Result<Self, String> {
        Self::load_with(model_root, capacity, Placement::CachedCopy)
    }

    pub fn load_with(
        model_root: &Path,
        capacity: usize,
        placement: Placement,
    ) -> Result<Self, String> {
        if capacity < 32 {
            return Err("Qwen3-ASR context capacity must be at least 32 tokens".into());
        }
        // Any unreadable or other checkpoint gets the C++ message.
        let config = ModelConfig::load(model_root)
            .ok()
            .filter(ModelConfig::is_supported)
            .ok_or("Qwen3-ASR runtime requires the supported 1.7B checkpoint")?;
        std::thread::scope(|scope| {
            let tokenizer = scope.spawn(|| Tokenizer::load(model_root));
            let stream = Stream::new().map_err(|e| format!("Qwen3-ASR stream: {e}"))?;
            let checkpoint = Checkpoint::open(model_root, &SHARDS)?;
            let weights = Weights::load(&stream, &checkpoint, &config, placement)?;
            let encoder = AudioEncoder::new(&stream)?;
            let decoder = TextDecoder::new(&stream, &config.text, capacity)?;
            let tokenizer = tokenizer
                .join()
                .map_err(|_| "Qwen3-ASR tokenizer thread panicked".to_string())??;
            let frontend_gate = CancellableGate::new(2).map_err(|e| e.message().to_string())?;
            Ok(Self {
                config,
                tokenizer,
                capacity,
                frontend_gate,
                device_gate: CancellableGate::default(),
                device: Mutex::new(Device {
                    weights,
                    encoder,
                    decoder,
                    stream,
                }),
                _checkpoint: checkpoint,
            })
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn device(&self) -> std::sync::MutexGuard<'_, Device> {
        self.device.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Transcribes `request`. `cancelled` is polled between phases and layers
    /// and while queued; `on_text` receives the cumulative transcript while
    /// tokens are generated and stops the request by returning `false`.
    pub fn transcribe(
        &self,
        request: &Request<'_>,
        cancelled: &dyn Fn() -> bool,
        mut on_text: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<Transcription, String> {
        let total = Instant::now();
        let _admission = self.frontend_gate.acquire(cancelled)?;
        let check = || {
            if cancelled() {
                Err("Qwen3-ASR transcription cancelled".to_string())
            } else {
                Ok(())
            }
        };
        if request.max_new_tokens == 0 {
            return Err("Qwen3-ASR max_new_tokens must be positive".into());
        }
        let language = normalize_language(request.language, &self.config.supported_languages)?;
        check()?;

        let mut result = Transcription::default();
        let begin = Instant::now();
        let decoded_waveform;
        let waveform: &[f32] = match request.audio {
            Audio::Wav(bytes) => {
                decoded_waveform = audio::resample_mono_16k(&audio::decode_wav(bytes)?)?;
                &decoded_waveform
            }
            Audio::Pcm16k(samples) => samples,
        };
        if !waveform.iter().all(|s| s.is_finite()) {
            return Err("Qwen3-ASR PCM must be finite".into());
        }
        result.timings.decode_audio_ms = milliseconds(begin);
        result.audio_samples = waveform.len();
        check()?;

        let capacity = self.capacity;
        let make_prompt = |tokens: usize| {
            build_prompt(
                &self.tokenizer,
                request.context,
                language.as_deref(),
                tokens,
            )
        };
        let overhead = make_prompt(1)?.len() - 1;
        if overhead >= capacity || request.max_new_tokens >= capacity - overhead {
            return Err("Qwen3-ASR context and output exceed capacity".into());
        }
        let budget = capacity - overhead - request.max_new_tokens;
        let chunks = audio::split_audio(waveform, audio::audio_samples_for_token_budget(budget))?;
        result.chunks = chunks.len();

        let mut previous_language = Vec::new();
        let mut text = Vec::new();
        let mut published = String::new();
        let mut languages: Vec<u8> = Vec::new();
        for chunk in &chunks {
            check()?;
            let mut chunk_waveform = &waveform[chunk.offset..chunk.offset + chunk.samples];
            // Short tails of a split recording are padded like upstream does.
            let padded;
            if chunks.len() > 1 && chunk.samples < SAMPLE_RATE as usize / 2 {
                let mut samples = chunk_waveform.to_vec();
                samples.resize(SAMPLE_RATE as usize / 2, 0.0);
                padded = samples;
                chunk_waveform = &padded;
            }
            let begin = Instant::now();
            let features = audio::log_mel_features(chunk_waveform)?;
            result.timings.feature_extraction_ms += milliseconds(begin);
            result.mel_frames += features.frames;
            let expected = audio::audio_embedding_token_count(features.frames)?;
            let prompt = make_prompt(expected)?;
            if prompt.len() > capacity || request.max_new_tokens > capacity - prompt.len() {
                return Err("Qwen3-ASR audio chunk exceeds context capacity".into());
            }
            result.prompt_tokens += prompt.len();
            check()?;

            let _lease = self.device_gate.acquire(cancelled)?;
            let mut device = self.device();
            let Device {
                weights,
                encoder,
                decoder,
                stream,
            } = &mut *device;
            let begin = Instant::now();
            let layer_check = || check();
            let (audio, tokens) = encoder.encode(
                stream,
                &weights.encoder,
                &features.values,
                features.frames,
                &layer_check,
                None,
            )?;
            result.timings.audio_encoder_ms += milliseconds(begin);
            result.audio_tokens += tokens;
            if tokens != expected {
                return Err("Qwen3-ASR audio frontend token count is inconsistent".into());
            }
            check()?;

            let begin = Instant::now();
            let tokenizer = &self.tokenizer;
            let mut progress = |ids: &[u32]| -> Result<bool, String> {
                check()?;
                let Some(callback) = on_text.as_deref_mut() else {
                    return Ok(true);
                };
                let raw = tokenizer.decode(ids, true);
                // Automatic language metadata is not transcript text.
                if language.is_none() && find(&raw, b"<asr_text>").is_none() {
                    return Ok(true);
                }
                let partial = parse_transcription(&raw, language.as_deref());
                let mut cumulative = text.clone();
                cumulative.extend_from_slice(&partial.text);
                let current = Utf8Decoder::default().push(&cumulative, false);
                if current == published {
                    return Ok(true);
                }
                published = current;
                Ok(callback(&published))
            };
            let generated = generate(
                decoder,
                stream,
                weights,
                &prompt,
                audio,
                tokens,
                request.max_new_tokens,
                &layer_check,
                &mut progress,
            )?;
            result.timings.text_decoder_ms += milliseconds(begin);
            let raw = self.tokenizer.decode(&generated, true);
            let parsed = parse_transcription(&raw, language.as_deref());
            text.extend_from_slice(&parsed.text);
            result.decoded.extend_from_slice(&raw);
            result.generated_ids.extend_from_slice(&generated);
            if !parsed.language.is_empty() && parsed.language != previous_language {
                if !languages.is_empty() {
                    languages.push(b',');
                }
                languages.extend_from_slice(&parsed.language);
                previous_language = parsed.language;
            }
        }
        result.language = lossy(&languages);
        result.text = lossy(&text);
        if let Some(callback) = on_text
            && result.text != published
            && !callback(&result.text)
        {
            return Err("Qwen3-ASR transcription cancelled".into());
        }
        result.sample_rate = SAMPLE_RATE;
        result.timings.total_ms = milliseconds(total);
        Ok(result)
    }

    /// Encoder and prefill parity boundaries of one unsplit recording, for
    /// comparisons with the C++ engine's traces.
    pub fn trace(
        &self,
        features: &audio::LogMelFeatures,
        prompt: &[u32],
    ) -> Result<(EncoderTrace, DecoderTrace), String> {
        let mut encoder_trace = EncoderTrace::default();
        let mut decoder_trace = DecoderTrace::default();
        let never = || Ok(());
        let mut device = self.device();
        let Device {
            weights,
            encoder,
            decoder,
            stream,
        } = &mut *device;
        let (audio, tokens) = encoder.encode(
            stream,
            &weights.encoder,
            &features.values,
            features.frames,
            &never,
            Some(&mut encoder_trace),
        )?;
        decoder.prefill(
            stream,
            &weights.decoder,
            prompt,
            audio,
            tokens,
            &never,
            Some(&mut decoder_trace),
        )?;
        Ok((encoder_trace, decoder_trace))
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Greedy generation after the prompt; the output ends with the stop token
/// when one is produced (as Transformers returns it).
#[allow(clippy::too_many_arguments)]
fn generate(
    decoder: &mut TextDecoder,
    stream: &Stream,
    weights: &Weights,
    prompt: &[u32],
    audio: gufo_hip::DeviceSlice<'_, f32>,
    audio_tokens: usize,
    max_new_tokens: usize,
    cancelled: &dyn Fn() -> Result<(), String>,
    on_tokens: &mut dyn FnMut(&[u32]) -> Result<bool, String>,
) -> Result<Vec<u32>, String> {
    let capacity = decoder.capacity();
    if prompt.len() > capacity || max_new_tokens > capacity - prompt.len() {
        return Err("Qwen3-ASR requested generation exceeds KV capacity".into());
    }
    let w = &weights.decoder;
    decoder.prefill(stream, w, prompt, audio, audio_tokens, cancelled, None)?;
    let mut token = decoder
        .select_token(stream)?
        .ok_or("Qwen3-ASR decoder produced non-finite prefill logits")?;
    let mut generated = vec![token];
    if !on_tokens(&generated)? {
        return Err("Qwen3-ASR generation cancelled".into());
    }
    while !is_stop_token(token) && generated.len() < max_new_tokens {
        decoder.decode(stream, w, token, cancelled)?;
        token = decoder.select_token(stream)?.ok_or_else(|| {
            format!(
                "Qwen3-ASR decoder produced non-finite logits at generation step {}",
                generated.len()
            )
        })?;
        generated.push(token);
        if !on_tokens(&generated)? {
            return Err("Qwen3-ASR generation cancelled".into());
        }
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_resolve_by_code_or_name() {
        let supported = vec!["English".to_string(), "German".to_string()];
        assert_eq!(normalize_language(None, &supported), Ok(None));
        assert_eq!(normalize_language(Some("AUTO"), &supported), Ok(None));
        assert_eq!(normalize_language(Some(""), &supported), Ok(None));
        assert_eq!(
            normalize_language(Some("de"), &supported),
            Ok(Some("German".into()))
        );
        assert_eq!(
            normalize_language(Some("english"), &supported),
            Ok(Some("English".into()))
        );
        assert!(normalize_language(Some("klingon"), &supported).is_err());
    }
}
