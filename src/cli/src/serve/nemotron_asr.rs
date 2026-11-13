//! Nemotron 3.5 ASR Streaming 0.6B behind `gufo serve asr` (`runner.cpp` of
//! the Nemotron ASR extension): uploads and PCM requests transcribed in
//! cache-aware streaming passes on the Rust engine.
//!
//! WAV uploads go through Gufo 0.34.0's strict WAV reader (integer PCM of
//! 16, 24 or 32 bits and 32-bit float, any rate and up to 8 channels, mixed
//! to mono as Gufo does) and Gufo 0.34.0's resampler: the samples the
//! extension decodes from such files. FLAC, MP3 and Ogg, which the
//! extension reads through Gufo's symphonia-based decoder, are refused.
//! Realtime sessions are transcribed after the commit, as Gufo does for the
//! languages it does not transcribe live; this server has no live
//! transcription.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use gufo_nemotron_asr::config::Config;
use gufo_nemotron_asr::language::{language_name, resolve_language};
use gufo_nemotron_asr::model::{Error, MAX_PASS_FRAMES, Model, Utterance};
use gufo_nemotron_asr::tokenizer::TextDecoder;
use gufo_serve::asr_model::{
    AsrRunner, TextSink, TranscriptionRequest, TranscriptionResult, TranscriptionTimings,
};

/// The model ID requests may name (`ModelId`).
pub const MODEL_ID: &str = "nemotron-3.5-asr-streaming-0.6b";

const RATE: u32 = 16_000;
/// Uploads longer than this are refused; the streaming encoder itself has
/// no length limit (`kMaxSeconds`).
const MAX_SECONDS: u64 = 4 * 3600;
/// Encoder frames (80 ms each) per pass of uploads that stream their text:
/// about 4.5 s of audio (`kShortPassFrames`).
const SHORT_PASS_FRAMES: usize = 56;
/// Samples of an upload fed to the utterance at a time (30 s), so that the
/// features never hold the whole recording (`kUploadBlock`).
const UPLOAD_BLOCK: usize = 30 * RATE as usize;

/// Whether `model_root` holds the Nemotron 3.5 ASR checkpoint
/// (`ProbeNemotronAsr`).
pub fn probe(model_root: &Path) -> bool {
    gufo_nemotron_asr::probe::probe(model_root)
}

fn since(begin: Instant) -> f64 {
    begin.elapsed().as_secs_f64() * 1000.0
}

/// The language a transcription reports: the model's last language tag,
/// else the requested locale (`Name`).
fn reported_language(text: &TextDecoder, config: &Config, prompt: usize) -> String {
    if let Some(locale) = text.language() {
        return language_name(locale);
    }
    config
        .prompt_dictionary
        .iter()
        .find(|(key, index)| *index == prompt && key != "auto" && key.contains('-'))
        .map(|(key, _)| language_name(key))
        .unwrap_or_default()
}

/// What `gufo::core::DecodeAudio` names a rejected WAV (`Describe`).
fn wav_error(error: gufo_wav_decode::Error) -> String {
    use gufo_wav_decode::Error as Wav;
    match error {
        Wav::NotRiff => {
            "Nemotron ASR reads WAV uploads; FLAC, MP3 and Ogg are not supported in this build"
        }
        Wav::Truncated => "audio is truncated or fails to decode",
        Wav::InvalidFmt => "audio header is malformed",
        Wav::UnsupportedFormat => "audio codec is not supported",
        Wav::OutOfBounds => "audio channels or sample rate are out of bounds",
        Wav::NoData => "audio has no samples",
        Wav::UnsupportedLayout => "audio sample layout is not supported",
        Wav::TooLong => "audio is too long",
        Wav::NonFinite => "audio has non-finite samples",
    }
    .into()
}

/// Frames of WAV decoded between two cancellation checks.
const WAV_BLOCK_FRAMES: usize = 4096;

/// Mono samples and their rate: each frame's channels summed in order from
/// zero and divided by the channel count in float32, as `DecodeAudio` with
/// `downmix_mono` mixes them (`Mix`).
fn decode_wav(bytes: &[u8], cancelled: &dyn Fn() -> bool) -> Result<(Vec<f32>, u32), String> {
    let limits = gufo_wav_decode::Limits {
        max_channels: 8,
        min_sample_rate: 1,
        max_sample_rate: 192_000,
        max_seconds: MAX_SECONDS,
        max_pcm_bits: 32,
    };
    let wav = gufo_wav_decode::parse(bytes, &limits).map_err(wav_error)?;
    let channels = wav.channels as usize;
    let total = wav.samples();
    let mut mono = Vec::with_capacity(total / channels);
    let mut block = vec![0.0f32; total.min(WAV_BLOCK_FRAMES * channels)];
    let mut first = 0;
    while first < total {
        if cancelled() {
            return Err("audio decoding was stopped".into());
        }
        let len = block.len().min(total - first);
        wav.decode(first, &mut block[..len]).map_err(wav_error)?;
        for frame in block[..len].chunks_exact(channels) {
            let mut sum = 0.0f32;
            for &sample in frame {
                sum += sample;
            }
            mono.push(sum / channels as f32);
        }
        first += len;
    }
    Ok((mono, wav.sample_rate))
}

fn message(error: Error) -> String {
    error.to_string()
}

/// One request (`Runner::Transcribe`).
fn transcribe(
    model: &Model,
    request: &TranscriptionRequest,
    on_text: Option<TextSink<'_>>,
    is_cancelled: &dyn Fn() -> bool,
) -> Result<TranscriptionResult, String> {
    let total = Instant::now();
    let config = model.config();
    let prompt = resolve_language(config, request.language.as_deref())?;
    if request.wav.is_empty() == request.pcm16k.is_empty() {
        return Err("Nemotron ASR needs exactly one audio source".into());
    }

    let decode = Instant::now();
    let (mut samples, rate) = if request.wav.is_empty() {
        (request.pcm16k.clone(), RATE)
    } else {
        decode_wav(&request.wav, is_cancelled)?
    };
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("audio samples must be finite".into());
    }
    if rate != RATE {
        samples = gufo_resample::resample(&samples, rate, RATE)
            .map_err(|_| "Resample: a rate is 0 or more than 64 times the other".to_string())?;
    }
    let decode_audio_ms = since(decode);

    // Whole uploads take the widest lookahead, the most accurate. Text that
    // streams arrives pass by pass, in short passes; otherwise full passes,
    // the rest split evenly at the end, are fastest.
    let streamed = on_text.is_some();
    let pass_frames = if streamed {
        SHORT_PASS_FRAMES
    } else {
        MAX_PASS_FRAMES
    };
    let mut utterance =
        Utterance::new(model, model.max_lookahead(), prompt, pass_frames).map_err(message)?;
    utterance.set_cancellation(Box::new(is_cancelled));
    if let Some(on_text) = on_text {
        utterance.set_text_callback(Box::new(move |text: &TextDecoder| on_text(&text.text())));
    }
    for block in samples.chunks(UPLOAD_BLOCK) {
        utterance.push(block, streamed).map_err(message)?;
    }
    utterance.finish().map_err(message)?;

    let text = utterance.text();
    Ok(TranscriptionResult {
        language: reported_language(text, config, prompt),
        text: text.text(),
        decoded: text.raw().to_string(),
        generated_ids: text.ids().to_vec(),
        sample_rate: RATE,
        audio_samples: samples.len(),
        mel_frames: utterance.mel_frames(),
        audio_tokens: utterance.encoder_frames(),
        prompt_tokens: 0,
        chunks: utterance.chunks(),
        timings: TranscriptionTimings {
            decode_audio_ms,
            feature_extraction_ms: utterance.feature_ms(),
            audio_encoder_ms: utterance.encoder_ms(),
            text_decoder_ms: utterance.decoder_ms(),
            total_ms: since(total),
        },
    })
}

/// Loads the model (`CreateNemotronAsrRunner`); requests share it, and
/// their encoder passes take turns on the GPU in arrival order.
pub fn runtime(model_root: &Path) -> Result<AsrRunner, String> {
    let model = Arc::new(Model::load(model_root)?);
    Ok(Box::new(move |request, on_text, is_cancelled| {
        transcribe(&model, request, on_text, is_cancelled)
    }))
}
