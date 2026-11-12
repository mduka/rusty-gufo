//! End-to-end synthesis for one checkpoint variant
//! (`hip/synthesis_runtime.cpp`): tokenize, build the variant's prompt,
//! generate codec frames and decode them, buffered or streamed. Base keeps
//! the reference audio's speaker embedding and codes for repeated requests.

use std::path::Path;

use crate::audio::AudioBuffer;
use crate::checkpoint::Checkpoint;
use crate::config::ModelVariant;
use crate::decoder::{SpeechDecoder, SpeechDecoderOutput};
use crate::encoders::{SpeakerEncoder, SpeakerEncoderOutput, SpeechEncoder, SpeechEncoderOutput};
use crate::sampling::SamplingOptions;
use crate::talker::{Talker, TalkerGenerationOutput, TalkerPromptOutput, VoiceClonePromptInput};
use crate::tokenizer::Tokenizer;

const CANCELLED: &str = "Qwen3-TTS native generation cancelled";

/// Receives newly decoded 24 kHz mono audio; returning false cancels.
pub type AudioSink<'a> = &'a mut dyn FnMut(&[f32]) -> bool;

/// `SynthesisRequest`; the C++ `on_audio` callback is an argument of
/// [`Synthesizer::generate`].
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

struct PromptTokens {
    input: Vec<u32>,
    instruction: Vec<u32>,
}

struct NativePrompt {
    talker: TalkerPromptOutput,
    reference_codes: SpeechEncoderOutput,
    prepend_reference_audio: bool,
}

/// The encoded reference of the last Base request.
struct BaseAudioFeatures {
    audio: AudioBuffer,
    speaker: SpeakerEncoderOutput,
    speech: Option<SpeechEncoderOutput>,
}

/// `SynthesisHipRuntime`.
pub struct Synthesizer {
    maximum_context_tokens: usize,
    variant: ModelVariant,
    tokenizer: Tokenizer,
    talker: Talker,
    decoder: SpeechDecoder,
    speech_encoder: Option<SpeechEncoder>,
    speaker_encoder: Option<SpeakerEncoder>,
    base_audio_features: Option<BaseAudioFeatures>,
}

impl Synthesizer {
    /// `SynthesisHipRuntime::Create`.
    pub fn new(
        model_root: &Path,
        maximum_context_tokens: usize,
        variant: ModelVariant,
    ) -> Result<Self, String> {
        if maximum_context_tokens == 0 {
            return Err("Qwen3-TTS native context capacity must be positive".into());
        }
        if variant == ModelVariant::Unsupported {
            return Err("unsupported Qwen3-TTS variant".into());
        }
        let tokenizer = Tokenizer::load(model_root)?;
        let checkpoint = Checkpoint::load(model_root)?;
        let talker = Talker::new(&checkpoint, maximum_context_tokens)?;
        let decoder = SpeechDecoder::new(&checkpoint)?;
        let (speech_encoder, speaker_encoder) = if variant == ModelVariant::Base {
            (
                Some(SpeechEncoder::new(&checkpoint)?),
                Some(SpeakerEncoder::new(&checkpoint)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            maximum_context_tokens,
            variant,
            tokenizer,
            talker,
            decoder,
            speech_encoder,
            speaker_encoder,
            base_audio_features: None,
        })
    }

    pub fn talker(&mut self) -> &mut Talker {
        &mut self.talker
    }

    pub fn decoder(&mut self) -> &mut SpeechDecoder {
        &mut self.decoder
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// `SynthesisHipRuntime::Generate`: buffered audio, or with `on_audio`
    /// the audio as it is decoded (the result then holds no samples).
    pub fn generate(
        &mut self,
        request: &SynthesisRequest,
        is_cancelled: &dyn Fn() -> bool,
        on_audio: Option<AudioSink<'_>>,
    ) -> Result<SynthesisResult, String> {
        if is_cancelled() {
            return Err(CANCELLED.into());
        }
        let tokens = self.tokenize(request)?;
        let prompt = match self.variant {
            ModelVariant::CustomVoice => NativePrompt {
                talker: self.talker.build_custom_voice_prompt(
                    &tokens.input,
                    &tokens.instruction,
                    &request.speaker,
                    &request.language,
                )?,
                reference_codes: SpeechEncoderOutput::default(),
                prepend_reference_audio: false,
            },
            ModelVariant::VoiceDesign => {
                if request.instruct.is_empty() {
                    return Err("Qwen3-TTS VoiceDesign requires a non-empty instruction".into());
                }
                NativePrompt {
                    talker: self.talker.build_voice_design_prompt(
                        &tokens.input,
                        &tokens.instruction,
                        &request.language,
                    )?,
                    reference_codes: SpeechEncoderOutput::default(),
                    prepend_reference_audio: false,
                }
            }
            ModelVariant::Base => self.build_base_prompt(request, &tokens)?,
            ModelVariant::Unsupported => return Err("unsupported Qwen3-TTS variant".into()),
        };
        self.fits_context(&prompt.talker, request.max_new_tokens)?;
        if let Some(on_audio) = on_audio {
            return self.generate_streaming(request, &prompt, is_cancelled, on_audio);
        }
        let generated = self.talker.generate(
            &prompt.talker,
            request.max_new_tokens,
            &request.sampling,
            is_cancelled,
            None,
        )?;
        if generated.frames == 0
            || generated.code_groups != 16
            || generated.codes.len() != generated.frames * generated.code_groups
        {
            return Err("Qwen3-TTS native talker produced invalid codec frames".into());
        }
        if is_cancelled() {
            return Err(CANCELLED.into());
        }
        self.decode(&prompt, &generated)
    }

    fn tokenize(&self, request: &SynthesisRequest) -> Result<PromptTokens, String> {
        let input = self.tokenizer.encode_assistant_prompt(&request.text)?;
        let instruction = if request.instruct.is_empty() {
            Vec::new()
        } else {
            self.tokenizer
                .encode_instruction_prompt(&request.instruct)?
        };
        Ok(PromptTokens { input, instruction })
    }

    /// `FitsContext`.
    fn fits_context(
        &self,
        prompt: &TalkerPromptOutput,
        maximum_new_tokens: usize,
    ) -> Result<(), String> {
        let available =
            self.maximum_context_tokens - prompt.tokens.min(self.maximum_context_tokens);
        if maximum_new_tokens <= available {
            return Ok(());
        }
        Err("Qwen3-TTS prompt plus generated tokens exceeds native context capacity".into())
    }

    /// `BuildBasePrompt`: the speaker embedding of the reference, and for
    /// ICL its transcript and codes.
    fn build_base_prompt(
        &mut self,
        request: &SynthesisRequest,
        tokens: &PromptTokens,
    ) -> Result<NativePrompt, String> {
        self.prepare_base_audio(&request.reference_audio)?;
        let mut reference_ids = Vec::new();
        let mut reference_codes = SpeechEncoderOutput::default();
        if !request.speaker_embedding_only {
            (reference_ids, reference_codes) = self.prepare_base_reference(request)?;
        }
        let features = self
            .base_audio_features
            .as_ref()
            .ok_or("Qwen3-TTS Base encoders are not initialized")?;
        let speaker_embedding = features.speaker.embedding.clone();
        let talker = self
            .talker
            .build_voice_clone_prompt(&VoiceClonePromptInput {
                input_ids: &tokens.input,
                reference_ids: &reference_ids,
                reference_codes: &reference_codes.codes,
                speaker_embedding: &speaker_embedding,
                reference_frames: reference_codes.frames,
                language: &request.language,
                icl_mode: !request.speaker_embedding_only,
            })?;
        Ok(NativePrompt {
            talker,
            reference_codes,
            prepend_reference_audio: !request.speaker_embedding_only,
        })
    }

    /// `PrepareBaseAudio`: the speaker embedding of the reference, reused
    /// while the same audio comes back.
    fn prepare_base_audio(&mut self, audio: &AudioBuffer) -> Result<(), String> {
        if audio.samples.is_empty() {
            return Err("Qwen3-TTS Base requires reference audio".into());
        }
        let (Some(speaker_encoder), Some(_)) =
            (self.speaker_encoder.as_mut(), self.speech_encoder.as_ref())
        else {
            return Err("Qwen3-TTS Base encoders are not initialized".into());
        };
        if self
            .base_audio_features
            .as_ref()
            .is_some_and(|features| features.audio == *audio)
        {
            return Ok(());
        }
        let speaker = speaker_encoder.encode(audio, None)?;
        self.base_audio_features = Some(BaseAudioFeatures {
            audio: audio.clone(),
            speaker,
            speech: None,
        });
        Ok(())
    }

    /// `PrepareBaseReference`: the transcript's tokens and the reference
    /// codes, encoded once per reference.
    fn prepare_base_reference(
        &mut self,
        request: &SynthesisRequest,
    ) -> Result<(Vec<u32>, SpeechEncoderOutput), String> {
        if request.reference_text.is_empty() {
            return Err("Qwen3-TTS Base ICL requires reference text".into());
        }
        let reference_ids = self
            .tokenizer
            .encode_reference_prompt(&request.reference_text)?;
        let features = self
            .base_audio_features
            .as_mut()
            .ok_or("Qwen3-TTS Base encoders are not initialized")?;
        if let Some(speech) = &features.speech {
            return Ok((reference_ids, speech.clone()));
        }
        let encoder = self
            .speech_encoder
            .as_mut()
            .ok_or("Qwen3-TTS Base encoders are not initialized")?;
        let encoded = encoder.encode(&request.reference_audio, None)?;
        features.speech = Some(encoded.clone());
        Ok((reference_ids, encoded))
    }

    /// `GenerateStreaming`: decodes after the first 4 frames, then every 16,
    /// and flushes the rest at the end.
    fn generate_streaming(
        &mut self,
        request: &SynthesisRequest,
        prompt: &NativePrompt,
        is_cancelled: &dyn Fn() -> bool,
        on_audio: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<SynthesisResult, String> {
        let reference_frames = if prompt.prepend_reference_audio {
            prompt.reference_codes.frames
        } else {
            0
        };
        let mut codes: Vec<u32> = if reference_frames != 0 {
            prompt.reference_codes.codes.clone()
        } else {
            Vec::new()
        };
        let mut decoded_frames = 0usize;
        let mut emitted_frames = 0usize;
        let mut stream_error = String::new();
        let mut sample_count = 0usize;
        let decoder = &mut self.decoder;
        let mut emit = |generated: &TalkerGenerationOutput, flush: bool| -> bool {
            let pending = generated.frames - emitted_frames;
            let threshold = if emitted_frames == 0 { 4 } else { 16 };
            if pending == 0 || (!flush && pending < threshold) {
                return true;
            }
            if is_cancelled() {
                stream_error = CANCELLED.into();
                return false;
            }
            codes.truncate(reference_frames * 16);
            codes.extend_from_slice(&generated.codes);
            let frames = reference_frames + generated.frames;
            let first_reference = decoded_frames == 0 && reference_frames != 0;
            let audio = if first_reference {
                decoder.decode_after_reference(&codes, frames, reference_frames)
            } else {
                decoder.decode_incremental(&codes, frames, decoded_frames)
            };
            let audio: SpeechDecoderOutput = match audio {
                Ok(audio) => audio,
                Err(error) => {
                    stream_error = error;
                    return false;
                }
            };
            decoded_frames = frames;
            emitted_frames = generated.frames;
            if !on_audio(&audio.samples) {
                stream_error = "Qwen3-TTS audio stream cancelled".into();
                return false;
            }
            sample_count += audio.samples.len();
            true
        };
        let generated = self.talker.generate(
            &prompt.talker,
            request.max_new_tokens,
            &request.sampling,
            is_cancelled,
            Some(&mut |progress: &TalkerGenerationOutput| emit(progress, false)),
        );
        let outcome = match generated {
            Ok(generated) if emit(&generated, true) && generated.frames != 0 => Ok(generated),
            Ok(_) => Err(String::new()),
            Err(error) => Err(error),
        };
        match outcome {
            Ok(generated) => Ok(SynthesisResult {
                sample_rate: 24000,
                code_groups: generated.code_groups as u32,
                samples: Vec::new(),
                codes: generated.codes.iter().map(|&code| code as i32).collect(),
                sample_count,
            }),
            Err(error) => Err(if stream_error.is_empty() {
                error
            } else {
                stream_error
            }),
        }
    }

    /// `Decode`: the generated frames, after the reference codes for ICL.
    fn decode(
        &mut self,
        prompt: &NativePrompt,
        generated: &TalkerGenerationOutput,
    ) -> Result<SynthesisResult, String> {
        let audio = if prompt.prepend_reference_audio {
            let mut codes = prompt.reference_codes.codes.clone();
            codes.extend_from_slice(&generated.codes);
            self.decoder.decode_after_reference(
                &codes,
                generated.frames + prompt.reference_codes.frames,
                prompt.reference_codes.frames,
            )?
        } else {
            self.decoder
                .decode(&generated.codes, generated.frames, None)?
        };
        Ok(SynthesisResult {
            sample_rate: audio.sample_rate,
            code_groups: generated.code_groups as u32,
            sample_count: audio.samples.len(),
            samples: audio.samples,
            codes: generated.codes.iter().map(|&code| code as i32).collect(),
        })
    }
}
