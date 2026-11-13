//! `gufo serve tts` with VoxCPM2 (the C++ extension's `runner.cpp`): an
//! automatic voice, voice design and style from `instructions`, and cloning
//! from a reference or a preset, with or without its transcript, on the Rust
//! synthesizer. The model is the Hugging Face directory or one audio.cpp
//! GGUF file.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use gufo_serve::tts_model::{
    AudioBuffer, ConditioningCapabilities, ConditioningMode, SynthesisRequest, SynthesisResult,
    TtsRunner,
};
use gufo_voxcpm::config::{Config, load_config, parse_config};
use gufo_voxcpm::model_files::ModelFiles;
use gufo_voxcpm::synthesis::{ReferenceVoice, SynthesisOptions, Synthesizer};

pub const MODEL_ID: &str = "voxcpm2";
pub const AUTO_VOICE: &str = "auto";
pub const DESIGN_VOICE: &str = "voice-design";
/// Encoded reference voices kept for repeated requests (presets reuse
/// theirs).
const CACHED_VOICES: usize = 8;

/// Whether `model_root` is a VoxCPM2 directory or GGUF this build
/// implements.
pub fn probe(model_root: &Path) -> bool {
    gufo_voxcpm::probe::probe_voxcpm2(model_root)
}

/// One checkpoint serves an automatic voice, voice design from
/// instructions and cloning from a recording or a preset, with or without
/// its transcript.
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

/// The configuration of either layout: the directory's `config.json` or
/// the one a GGUF embeds (`ProbeVoxCpm2`).
fn config_of(model_root: &Path) -> Result<Config, String> {
    if ModelFiles::is_voxcpm2_gguf(model_root) {
        parse_config(ModelFiles::open(model_root)?.config())
    } else {
        load_config(model_root)
    }
}

/// The output sample rate AudioVAE decodes to (48 kHz for VoxCPM2).
pub fn sample_rate(model_root: &Path) -> Result<u32, String> {
    gufo_voxcpm::probe::resolve_voxcpm2_artifacts(model_root)?;
    let rate = config_of(model_root)?.vae.out_sample_rate;
    u32::try_from(rate).map_err(|_| format!("VoxCPM2 output sample rate {rate} is out of range"))
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
    /// transcript. `voice_clone_mode: speaker_embedding_only` (and a preset
    /// without a transcript) conditions on the recording in isolation, as
    /// the extension's driver does without `--reference-text`; the
    /// extension's runner rejects it, since Gufo's API asks for a transcript
    /// otherwise.
    fn reference(&mut self, request: &SynthesisRequest) -> Result<Arc<ReferenceVoice>, String> {
        let transcript = if request.speaker_embedding_only {
            ""
        } else {
            request.reference_text.as_str()
        };
        let key = fingerprint(&request.reference_audio, transcript);
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
            transcript,
        )?);
        self.cache.push_front((key, voice.clone()));
        self.cache.truncate(CACHED_VOICES);
        Ok(voice)
    }
}

/// The VoxCPM2 runtime, created once the service has validated its
/// options; requests run one at a time. Streams receive every 160 ms patch
/// as soon as AudioVAE has decoded it; a buffered result that ran into the
/// text-length cap is generated again with the next seed (up to three
/// attempts), as the reference retries it.
pub fn runtime(model_root: &Path) -> Result<TtsRunner, String> {
    let state = Mutex::new(State {
        synthesizer: Synthesizer::load(model_root)?,
        cache: VecDeque::new(),
    });
    Ok(Box::new(move |request, on_audio, is_cancelled| {
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        let voice = if request.reference_audio.samples.is_empty() {
            None
        } else {
            Some(state.reference(request)?)
        };
        // `language`, `greedy` and the sampling fields don't apply: VoxCPM2
        // reads the language from the text, and its decoding parameters are
        // the reference's defaults.
        let options = SynthesisOptions {
            text: request.text.clone(),
            instruct: request.instruct.clone(),
            voice: voice.as_deref(),
            max_patches: request.max_new_tokens,
            seed: u64::from(request.sampling.seed),
            ..SynthesisOptions::default()
        };
        let rate = state.synthesizer.sample_rate();
        let mut result = SynthesisResult {
            sample_rate: rate,
            code_groups: 0,
            ..SynthesisResult::default()
        };
        match on_audio {
            Some(sink) => {
                let mut streamed = 0usize;
                let mut forward = |samples: &[f32]| {
                    streamed += samples.len();
                    sink(samples)
                };
                state.synthesizer.synthesize(
                    &options,
                    Some(is_cancelled),
                    Some(&mut forward),
                    None,
                )?;
                result.sample_count = streamed;
            }
            None => {
                let mut audio = Vec::new();
                state.synthesizer.synthesize(
                    &options,
                    Some(is_cancelled),
                    None,
                    Some(&mut audio),
                )?;
                result.sample_count = audio.len();
                result.samples = audio;
            }
        }
        Ok(result)
    }))
}
