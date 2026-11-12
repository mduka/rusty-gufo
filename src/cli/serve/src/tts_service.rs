//! The bounded TTS service (`src/cli/serve/tts_service.cpp`): model
//! identity, voices and voice presets, the runtime's conditioning
//! capabilities and output sample rate (Gufo #984, #983), and FIFO admission
//! of synthesis to the injected runtime.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use gufo_core::cancellable_gate::CancellableGate;

use crate::arbiter::PlaybackBuffer;

use crate::tts_model::{
    self, AudioBuffer, AudioSink, ConditioningCapabilities, ConditioningMode, ModelVariant,
    SynthesisRequest, SynthesisResult, TtsRunner,
};

/// An operator-registered named voice: reference audio the server holds for
/// clients, so a Base checkpoint can be driven by voice name alone.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TtsVoicePreset {
    pub reference_audio: AudioBuffer,
    pub reference_text: String,
    pub speaker_embedding_only: bool,
    /// The language this voice speaks; the request default when set, an
    /// explicit request `language` still wins. Empty means the global
    /// default.
    pub language: String,
}

/// Creates the synthesis runtime once the options have validated, where C++
/// creates its native HIP runtime: from the model root, the native context
/// tokens and the resolved variant. The error becomes the service's
/// initialization error.
pub type TtsRuntimeFactory =
    Box<dyn FnOnce(&Path, usize, ModelVariant) -> Result<TtsRunner, String> + Send>;

pub struct TtsServiceOptions {
    pub model_root: PathBuf,
    pub native_context_tokens: usize,
    /// Reads `config.json` for the variant and CustomVoice speakers.
    pub validate_model: bool,
    /// The runtime's fixed output sample rate, in [1, 192000].
    pub sample_rate: u32,
    pub variant: ModelVariant,
    /// What the runtime supports; `None` takes the variant's behavior.
    pub conditioning: Option<ConditioningCapabilities>,
    pub model_id: String,
    pub voices: Vec<String>,
    pub voice_presets: BTreeMap<String, TtsVoicePreset>,
    /// The synthesis runtime. Without one, `runtime_factory` creates it
    /// after validation, as C++ creates its native HIP runtime; this crate
    /// has no GPU code, so one of the two is required.
    pub runner: Option<TtsRunner>,
    pub runtime_factory: Option<TtsRuntimeFactory>,
    /// Streamed syntheses report to it (`--arbiter`).
    pub playback: Option<Arc<PlaybackBuffer>>,
}

impl Default for TtsServiceOptions {
    fn default() -> Self {
        Self {
            model_root: PathBuf::new(),
            native_context_tokens: 4096,
            validate_model: true,
            sample_rate: 24_000,
            variant: ModelVariant::CustomVoice,
            conditioning: None,
            model_id: String::new(),
            voices: Vec::new(),
            voice_presets: BTreeMap::new(),
            runner: None,
            runtime_factory: None,
            playback: None,
        }
    }
}

fn default_model_id(variant: ModelVariant) -> String {
    match variant {
        ModelVariant::Base => "qwen3-tts-12hz-1.7b-base",
        ModelVariant::VoiceDesign => "qwen3-tts-12hz-1.7b-voice-design",
        ModelVariant::CustomVoice => "qwen3-tts-12hz-1.7b-customvoice",
        ModelVariant::Unsupported => "",
    }
    .into()
}

pub struct TtsService {
    variant: ModelVariant,
    conditioning: ConditioningCapabilities,
    sample_rate: u32,
    model_id: String,
    voices: Vec<String>,
    voice_presets: BTreeMap<String, TtsVoicePreset>,
    runner: Option<TtsRunner>,
    initialization_error: String,
    generation_gate: CancellableGate,
    ready: bool,
    playback: Option<Arc<PlaybackBuffer>>,
}

impl TtsService {
    pub fn new(options: TtsServiceOptions) -> Self {
        let TtsServiceOptions {
            model_root,
            native_context_tokens,
            validate_model,
            sample_rate,
            mut variant,
            conditioning,
            mut model_id,
            mut voices,
            voice_presets,
            runner,
            runtime_factory,
            playback,
        } = options;
        let unready = |variant, model_id: String, voices, error: &str| Self {
            variant,
            conditioning: ConditioningCapabilities::none(),
            sample_rate,
            model_id,
            voices,
            voice_presets: BTreeMap::new(),
            runner: None,
            initialization_error: error.into(),
            generation_gate: CancellableGate::default(),
            ready: false,
            playback: None,
        };
        if validate_model {
            let config = tts_model::load_model_config(&model_root)
                .filter(tts_model::is_supported_model_config);
            let Some(config) = config else {
                return unready(
                    variant,
                    model_id,
                    voices,
                    "Qwen3-TTS service requires a supported 1.7B model",
                );
            };
            variant = config.variant;
            if voices.is_empty() && variant == ModelVariant::CustomVoice {
                voices.extend(config.speakers.into_keys());
            }
        }
        if !(1..=192_000).contains(&sample_rate) {
            return unready(variant, model_id, voices, "invalid TTS output sample rate");
        }
        let conditioning =
            conditioning.unwrap_or_else(|| ConditioningCapabilities::for_variant(variant));
        if !conditioning.unconditioned
            && conditioning.reference_audio == ConditioningMode::Unsupported
            && !conditioning.reference_presets
        {
            return unready(
                variant,
                model_id,
                voices,
                "TTS provider does not support any synthesis conditioning mode",
            );
        }
        if conditioning.unconditioned && conditioning.reference_audio == ConditioningMode::Required
        {
            return unready(
                variant,
                model_id,
                voices,
                "TTS provider cannot require reference audio when unconditioned synthesis is enabled",
            );
        }
        if model_id.is_empty() {
            model_id = default_model_id(variant);
        }
        if voices.is_empty() {
            if !conditioning.default_voice.is_empty() {
                voices.push(conditioning.default_voice.clone());
            } else if conditioning.instructions == ConditioningMode::Required {
                voices.push("voice-design".into());
            } else if conditioning.reference_audio != ConditioningMode::Unsupported {
                voices.push("voice-clone".into());
            }
        }
        if !voice_presets.is_empty() && !conditioning.reference_presets {
            return unready(
                variant,
                model_id,
                voices,
                "voice presets require a TTS model that supports reference presets",
            );
        }
        for name in voice_presets.keys() {
            if !voices.contains(name) {
                voices.push(name.clone());
            }
        }
        let runner = match (runner, runtime_factory) {
            (Some(runner), _) => runner,
            (None, Some(factory)) => match factory(&model_root, native_context_tokens, variant) {
                Ok(runner) => runner,
                Err(error) => return unready(variant, model_id, voices, &error),
            },
            (None, None) => {
                return unready(
                    variant,
                    model_id,
                    voices,
                    "Qwen3-TTS requires an injected synthesis runtime in this build",
                );
            }
        };
        voices.sort();
        Self {
            variant,
            conditioning,
            sample_rate,
            model_id,
            voices,
            voice_presets,
            runner: Some(runner),
            initialization_error: String::new(),
            generation_gate: CancellableGate::default(),
            ready: true,
            playback,
        }
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    pub fn initialization_error(&self) -> String {
        self.initialization_error.clone()
    }

    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    pub fn backend_name(&self) -> String {
        "native-hip".into()
    }

    pub fn voices(&self) -> Vec<String> {
        self.voices.clone()
    }

    /// The registered preset for `name`, if `name` is one.
    pub fn voice_preset(&self, name: &str) -> Option<&TtsVoicePreset> {
        self.voice_presets.get(name)
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    /// What the runtime supports (nothing for a service that failed to
    /// start).
    pub fn conditioning(&self) -> &ConditioningCapabilities {
        &self.conditioning
    }

    /// The runtime's output sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Waits for the runtime in FIFO order (cancellable while queued), then
    /// synthesizes. `on_audio` receives streamed mono audio at
    /// [`TtsService::sample_rate`].
    pub fn synthesize(
        &self,
        request: &SynthesisRequest,
        on_audio: Option<AudioSink<'_>>,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<SynthesisResult, String> {
        let Some(runner) = self.runner.as_ref().filter(|_| self.ready) else {
            return Err(self.initialization_error.clone());
        };
        let _lease = self.generation_gate.acquire(is_cancelled)?;
        match (&self.playback, on_audio) {
            (Some(playback), Some(on_audio)) => {
                playback.begin();
                let mut report = |chunk: &[f32]| {
                    let sent = on_audio(chunk);
                    if sent {
                        playback.add(
                            chunk.len() as f64 / f64::from(self.sample_rate),
                            Instant::now(),
                        );
                    }
                    sent
                };
                let result = runner(request, Some(&mut report), is_cancelled);
                playback.end();
                result
            }
            (_, on_audio) => runner(request, on_audio, is_cancelled),
        }
    }
}
