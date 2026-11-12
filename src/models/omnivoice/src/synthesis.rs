//! OmniVoice's `generate()` (`synthesizer.cpp`): text and voice to 24 kHz
//! mono audio, with the reference's chunking of long texts and its
//! post-processing.

use std::path::Path;

use crate::audio::{cross_fade_chunks, post_process, prepare_reference};
use crate::codec::Codec;
use crate::config::{Config, is_supported, load_config};
use crate::decoding::{ConditionalInput, DecodingConfig, GenerationResult};
use crate::generator::Generator;
use crate::prompt::{PromptText, encode_prompt_text};
use crate::text::{
    add_punctuation, chunk_text_punctuation, estimate_target_frames, resolve_instruct,
    resolve_language,
};
use crate::tokenizer::Tokenizer;
use crate::unicode::{contains_cjk_ideograph, decode_utf8, strip_space};

/// OmniVoiceGenerationConfig: texts estimated longer than 30 s are split
/// into chunks of about 15 s.
const CHUNK_THRESHOLD_SECONDS: f64 = 30.0;
const CHUNK_SECONDS: f64 = 15.0;
/// `CrossFadeChunks`' silence between chunks.
const CHUNK_GAP_SECONDS: f64 = 0.3;

/// A cloned voice: its codes and transcript as OmniVoice's voice clone
/// prompt holds them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReferenceVoice {
    /// `[codebook][frame]`.
    pub codes: Vec<i32>,
    pub frames: usize,
    /// The transcript, ending in punctuation.
    pub text: String,
    /// RMS of the recording before level correction.
    pub rms: f64,
}

#[derive(Debug, Clone)]
pub struct SynthesisOptions<'a> {
    pub text: String,
    /// A language name or ID; empty or unknown selects no language.
    pub language: String,
    /// Voice-design items; empty for none.
    pub instruct: String,
    /// Clone this voice; `None` for voice design or an automatic voice.
    pub reference: Option<&'a ReferenceVoice>,
    /// Upper bound of generated frames per chunk; 0 for none.
    pub max_frames: usize,
    pub seed: u64,
    pub decoding: DecodingConfig,
    pub denoise: bool,
}

impl Default for SynthesisOptions<'_> {
    fn default() -> Self {
        Self {
            text: String::new(),
            language: String::new(),
            instruct: String::new(),
            reference: None,
            max_frames: 0,
            seed: 0,
            decoding: DecodingConfig::default(),
            denoise: true,
        }
    }
}

/// A synthesized utterance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Synthesis {
    pub audio: Vec<f32>,
    /// Every chunk's codes frame by frame (`[frame][codebook]`).
    pub codes: Vec<i32>,
}

fn codepoint_count(text: &str) -> usize {
    decode_utf8(text.as_bytes()).map_or(text.len(), |points| points.len())
}

/// OmniVoice's generate(). Not thread-safe.
pub struct Synthesizer {
    config: Config,
    tokenizer: Tokenizer,
    generator: Generator,
    codec: Codec,
}

impl Synthesizer {
    /// Loads the model directory: configuration, tokenizer, generator and
    /// audio tokenizer.
    pub fn load(model_root: &Path) -> Result<Self, String> {
        let config = load_config(model_root)?;
        is_supported(&config)?;
        let tokenizer = Tokenizer::load(&model_root.join("tokenizer.json"))?;
        if tokenizer.vocabulary_size() > config.backbone.vocab_size {
            return Err("OmniVoice tokenizer is larger than the backbone's vocabulary".into());
        }
        let generator = Generator::load(&config, model_root)?;
        let codec = Codec::load(&config, model_root)?;
        Ok(Self {
            config,
            tokenizer,
            generator,
            codec,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.codec.sample_rate
    }

    /// The generator, for tests that compare its passes with the reference.
    pub fn generator(&mut self) -> &mut Generator {
        &mut self.generator
    }

    /// The audio tokenizer, for tests.
    pub fn codec(&mut self) -> &mut Codec {
        &mut self.codec
    }

    /// `create_voice_clone_prompt`: prepares and encodes a recording
    /// (interleaved, any rate) and punctuates its transcript.
    pub fn clone_voice(
        &mut self,
        interleaved: &[f32],
        channels: u32,
        rate: u32,
        transcript: &str,
    ) -> Result<ReferenceVoice, String> {
        if strip_space(transcript).is_empty() {
            return Err("OmniVoice: voice cloning needs the reference transcript".into());
        }
        let prepared = prepare_reference(
            interleaved,
            channels,
            rate,
            self.sample_rate(),
            self.config.codec.frame_samples,
        )?;
        let (codes, frames) = self.codec.encode(&prepared.samples)?;
        Ok(ReferenceVoice {
            codes,
            frames,
            text: add_punctuation(transcript),
            rms: prepared.rms,
        })
    }

    /// One generation of `text` against an optional reference; returns the
    /// result and its frame count.
    #[allow(clippy::too_many_arguments)]
    fn generate_chunk(
        &mut self,
        text: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        reference_text: Option<&str>,
        reference_codes: &[i32],
        reference_frames: usize,
        options: &SynthesisOptions<'_>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(GenerationResult, usize), String> {
        let mut frames = estimate_target_frames(text, reference_text, reference_frames, 1.0);
        if options.max_frames > 0 {
            frames = frames.min(options.max_frames);
        }
        let prompt = PromptText {
            text: text.to_string(),
            reference_text: reference_text.map(str::to_string),
            language: language.map(str::to_string),
            instruct: instruct.map(str::to_string),
            denoise: options.denoise,
            has_reference: reference_frames > 0,
        };
        let ids = encode_prompt_text(&self.tokenizer, &prompt)?;
        let input = ConditionalInput::build(
            &ids,
            reference_codes,
            reference_frames,
            frames,
            self.config.audio_codebooks,
            self.config.audio_mask_id,
        );
        let result = self
            .generator
            .generate(&input, &options.decoding, options.seed, cancelled)?;
        Ok((result, frames))
    }

    /// Generates, decodes and post-processes `options.text`.
    pub fn synthesize(
        &mut self,
        options: &SynthesisOptions<'_>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Synthesis, String> {
        let codebooks = self.config.audio_codebooks;
        // The reference estimates and chunks the raw text; only the prompt
        // strips it.
        let text = &options.text;
        if strip_space(text).is_empty() {
            return Err("OmniVoice: the text is empty".into());
        }
        let language = resolve_language(&options.language);
        let instruct = resolve_instruct(&options.instruct, contains_cjk_ideograph(text))
            .map_err(|e| format!("OmniVoice: {e}"))?;
        let reference = options.reference;
        let mut reference_text: Option<String> = reference.map(|r| r.text.clone());
        let mut reference_codes: Vec<i32> = reference.map(|r| r.codes.clone()).unwrap_or_default();
        let mut reference_frames = reference.map_or(0, |r| r.frames);
        // Short texts are one generation; longer ones are chunked at
        // punctuation.
        let estimate =
            estimate_target_frames(text, reference_text.as_deref(), reference_frames, 1.0);
        let frame_rate = f64::from(self.config.codec.frame_rate());
        let mut chunks = vec![text.clone()];
        if estimate as f64 > CHUNK_THRESHOLD_SECONDS * frame_rate {
            let frames_per_character = estimate as f64 / codepoint_count(text) as f64;
            let chunk_length = (CHUNK_SECONDS * frame_rate / frames_per_character) as usize;
            chunks = chunk_text_punctuation(text, chunk_length, 3);
        }
        let mut pieces = Vec::with_capacity(chunks.len());
        let mut codes = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let (result, frames) = self.generate_chunk(
                chunk,
                language.as_deref(),
                instruct.as_deref(),
                reference_text.as_deref(),
                &reference_codes,
                reference_frames,
                options,
                cancelled,
            )?;
            pieces.push(self.codec.decode(&result.codes, frames)?);
            for f in 0..frames {
                for c in 0..codebooks {
                    codes.push(result.codes[c * frames + f]);
                }
            }
            // Without a reference, the first chunk's voice carries the rest.
            if reference.is_none() && index == 0 && chunks.len() > 1 {
                reference_text = Some(chunks[0].clone());
                reference_codes = result.codes;
                reference_frames = frames;
            }
        }
        let level = reference.map(|r| r.rms);
        let rate = self.sample_rate();
        let audio = post_process(
            &cross_fade_chunks(&pieces, rate, CHUNK_GAP_SECONDS),
            rate,
            level,
        );
        Ok(Synthesis { audio, codes })
    }
}
