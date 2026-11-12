//! `gufo serve tts` with an OmniVoice model directory (the C++ extension's
//! `runner.cpp`): an automatic voice, voice design from `instructions` and
//! cloning from a reference or a preset, on the Rust synthesizer.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use gufo_omnivoice::synthesis::{ReferenceVoice, SynthesisOptions, Synthesizer};
use gufo_serve::tts_model::{
    AudioBuffer, ConditioningCapabilities, ConditioningMode, SynthesisRequest, SynthesisResult,
    TtsRunner,
};

pub const MODEL_ID: &str = "omnivoice";
pub const AUTO_VOICE: &str = "auto";
pub const DESIGN_VOICE: &str = "voice-design";
/// Prepared reference voices kept for repeated requests (presets reuse
/// theirs).
const CACHED_VOICES: usize = 8;
/// Streamed audio arrives in pieces of this many seconds.
const STREAM_PIECE_SECONDS: f64 = 0.5;

/// Whether `model_root` is an OmniVoice directory this build implements.
pub fn probe(model_root: &Path) -> bool {
    gufo_omnivoice::probe::probe_omnivoice(model_root)
}

/// One checkpoint serves an automatic voice, voice design from
/// instructions and cloning from a reference or a preset.
pub fn conditioning() -> ConditioningCapabilities {
    ConditioningCapabilities {
        unconditioned: true,
        require_voice: false,
        default_voice: AUTO_VOICE.into(),
        reference_audio: ConditioningMode::Optional,
        reference_presets: true,
        instructions: ConditioningMode::Optional,
    }
}

/// The output sample rate the directory's audio tokenizer declares.
pub fn sample_rate(model_root: &Path) -> Result<u32, String> {
    gufo_omnivoice::probe::resolve_omnivoice_artifacts(model_root)?;
    Ok(gufo_omnivoice::config::load_config(model_root)?
        .codec
        .sample_rate)
}

/// FNV-1a over a reference recording and its transcript.
fn fingerprint(audio: &AudioBuffer, text: &str) -> u64 {
    let mut hash = 1_469_598_103_934_665_603u64;
    let mut mix = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
    };
    mix(&audio.sample_rate.to_ne_bytes());
    mix(&audio.channels.to_ne_bytes());
    for sample in &audio.samples {
        mix(&sample.to_ne_bytes());
    }
    mix(text.as_bytes());
    hash
}

struct State {
    synthesizer: Synthesizer,
    cache: VecDeque<(u64, Arc<ReferenceVoice>)>,
}

impl State {
    /// The request's reference voice, encoded once per recording and
    /// transcript.
    fn reference(&mut self, request: &SynthesisRequest) -> Result<Arc<ReferenceVoice>, String> {
        let key = fingerprint(&request.reference_audio, &request.reference_text);
        if let Some(index) = self.cache.iter().position(|(k, _)| *k == key) {
            let hit = self.cache.remove(index).ok_or("cache")?;
            self.cache.push_front(hit);
            return Ok(self.cache[0].1.clone());
        }
        let audio = &request.reference_audio;
        let voice = Arc::new(self.synthesizer.clone_voice(
            &audio.samples,
            audio.channels,
            audio.sample_rate,
            &request.reference_text,
        )?);
        self.cache.push_front((key, voice.clone()));
        self.cache.truncate(CACHED_VOICES);
        Ok(voice)
    }
}

/// The OmniVoice runtime, created once the service has validated its
/// options; requests run one at a time.
pub fn runtime(model_root: &Path) -> Result<TtsRunner, String> {
    let state = Mutex::new(State {
        synthesizer: Synthesizer::load(model_root)?,
        cache: VecDeque::new(),
    });
    Ok(Box::new(move |request, on_audio, is_cancelled| {
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        if request.speaker_embedding_only {
            return Err(
                "OmniVoice clones a voice from its recording and transcript; \
                        send reference_text instead of speaker_embedding_only"
                    .into(),
            );
        }
        let reference = if request.reference_audio.samples.is_empty() {
            None
        } else {
            Some(state.reference(request)?)
        };
        let mut options = SynthesisOptions {
            text: request.text.clone(),
            language: request.language.clone(),
            instruct: request.instruct.clone(),
            reference: reference.as_deref(),
            max_frames: request.max_new_tokens,
            seed: u64::from(request.sampling.seed),
            ..SynthesisOptions::default()
        };
        if !request.sampling.sample {
            options.decoding.position_temperature = 0.0;
        }
        let rate = state.synthesizer.sample_rate();
        // The post-processed audio always opens with 0.1 s of padding.
        // Streams send it before generating, so audio starts at once and a
        // client that leaves is noticed while the model still works.
        let padding = (0.1 * f64::from(rate)) as usize;
        let mut on_audio = on_audio;
        if let Some(sink) = on_audio.as_mut()
            && (!sink(&vec![0.0f32; padding]) || is_cancelled())
        {
            return Err("cancelled".into());
        }
        let synthesis = state.synthesizer.synthesize(&options, is_cancelled)?;
        let mut audio = synthesis.audio;
        // An utterance that is silence throughout still keeps its padding.
        if audio.len() < padding {
            audio = vec![0.0; padding];
        }
        let mut result = SynthesisResult {
            sample_rate: rate,
            code_groups: state.synthesizer.config().audio_codebooks as u32,
            samples: Vec::new(),
            codes: synthesis.codes,
            sample_count: audio.len(),
        };
        let Some(sink) = on_audio else {
            result.samples = audio;
            return Ok(result);
        };
        let piece = (STREAM_PIECE_SECONDS * f64::from(rate)) as usize;
        for chunk in audio[padding..].chunks(piece.max(1)) {
            if is_cancelled() || !sink(chunk) {
                return Err("cancelled".into());
            }
        }
        Ok(result)
    }))
}
