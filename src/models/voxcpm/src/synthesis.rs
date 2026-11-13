//! Text to 48 kHz speech with VoxCPM2 (`synthesizer.hpp`,
//! `synthesizer.cpp`): prompt construction for the zero-shot, reference,
//! continuation and combined modes, the patch loop with its stop predictor,
//! and AudioVAE decoding patch by patch.
//!
//! Follows OpenBMB/VoxCPM's `VoxCPM2Model._generate`
//! (src/voxcpm/model/voxcpm2.py; Apache-2.0, Copyright OpenBMB): the prompt
//! of each mode, the stop rules and the retry of overlong results.

use std::path::Path;

use crate::config::{Config, is_supported, parse_config};
use crate::generator::{Generator, Prompt};
use crate::model_files::ModelFiles;
use crate::text::collapse_whitespace;
use crate::tokenizer::Tokenizer;
use crate::vae::AudioVae;

// VoxCPM2Model's control tokens.
const AUDIO_START: i32 = 101;
const REFERENCE_START: i32 = 103;
const REFERENCE_END: i32 = 104;
/// `_inference`: no stop before the fourth patch.
const MIN_PATCHES: usize = 2;
/// `retry_badcase_ratio_threshold` and the cap `max_len = 6 * tokens + 10`.
const PATCHES_PER_TOKEN: f64 = 6.0;
const RETRIES: usize = 3;
/// Continuations decode the last (`streaming_prefix_len - 1`) prompt
/// patches first, as the reference does before trimming them.
const CONTEXT_PATCHES: usize = 3;

/// The noise of each patch: `std::mt19937_64` seeded with the request's
/// seed and `std::normal_distribution<float>(0, 1)` as libstdc++ computes
/// them, so a seed replays the C++ extension's noise bit for bit (the same
/// port as Qwen-Image-2.1's `random.rs`).
mod random {
    /// `std::mt19937_64`.
    pub struct Mt19937_64 {
        state: [u64; 312],
        index: usize,
    }

    impl Mt19937_64 {
        const N: usize = 312;
        const M: usize = 156;
        const UPPER: u64 = 0xffff_ffff_8000_0000;
        const LOWER: u64 = 0x7fff_ffff;

        pub fn new(seed: u64) -> Self {
            let mut state = [0u64; 312];
            state[0] = seed;
            for i in 1..Self::N {
                let previous = state[i - 1];
                state[i] = 6_364_136_223_846_793_005u64
                    .wrapping_mul(previous ^ (previous >> 62))
                    .wrapping_add(i as u64);
            }
            Self {
                state,
                index: Self::N,
            }
        }

        fn twist(&mut self) {
            for i in 0..Self::N {
                let y =
                    (self.state[i] & Self::UPPER) | (self.state[(i + 1) % Self::N] & Self::LOWER);
                let mut next = self.state[(i + Self::M) % Self::N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= 0xb502_6f5a_a966_19e9;
                }
                self.state[i] = next;
            }
            self.index = 0;
        }

        pub fn next_u64(&mut self) -> u64 {
            if self.index >= Self::N {
                self.twist();
            }
            let mut z = self.state[self.index];
            self.index += 1;
            z ^= (z >> 29) & 0x5555_5555_5555_5555;
            z ^= (z << 17) & 0x71d6_7fff_eda6_0000;
            z ^= (z << 37) & 0xfff7_eee0_0000_0000;
            z ^ (z >> 43)
        }

        /// `std::generate_canonical<float, 24>`: one draw scaled by 2^-64,
        /// with a result that rounds up to 1 replaced by the float below 1.
        fn canonical(&mut self) -> f32 {
            let value = self.next_u64() as f32 / 18_446_744_073_709_551_616.0f32;
            if value >= 1.0 {
                1.0 - f32::EPSILON / 2.0
            } else {
                value
            }
        }
    }

    /// `std::normal_distribution<float>` with mean 0 and deviation 1: the
    /// Marsaglia polar method with its saved second value.
    #[derive(Default)]
    pub struct Normal {
        saved: Option<f32>,
    }

    impl Normal {
        pub fn sample(&mut self, rng: &mut Mt19937_64) -> f32 {
            let value = match self.saved.take() {
                Some(saved) => saved,
                None => loop {
                    let x = 2.0f32 * rng.canonical() - 1.0;
                    let y = 2.0f32 * rng.canonical() - 1.0;
                    let r2 = x * x + y * y;
                    if r2 > 1.0 || r2 == 0.0 {
                        continue;
                    }
                    let mult = (-2.0f32 * r2.ln() / r2).sqrt();
                    self.saved = Some(x * mult);
                    break y * mult;
                },
            };
            value * 1.0 + 0.0
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mt19937_64_matches_the_standard() {
            // The 10000th output of a default-constructed std::mt19937_64
            // (seed 5489), which the standard specifies.
            let mut rng = Mt19937_64::new(5489);
            for _ in 0..9999 {
                rng.next_u64();
            }
            assert_eq!(rng.next_u64(), 9_981_545_732_273_789_042);
        }

        #[test]
        fn normal_draws_pair_up() {
            let mut rng = Mt19937_64::new(42);
            let mut normal = Normal::default();
            let values: Vec<f32> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
            let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / 1000.0;
            assert!(mean.abs() < 0.1, "mean {mean}");
            assert!(values.iter().all(|v| v.is_finite()));
        }
    }
}

use random::{Mt19937_64, Normal};

/// A voice to clone, encoded once (`VoxCPM2Model.build_prompt_cache`): the
/// recording as isolated reference latents and, with its transcript, also
/// as the prompt that generation continues (`ReferenceVoice`).
#[derive(Debug, Clone, Default)]
pub struct ReferenceVoice {
    /// `[patches][patch_size][feat_dim]`; the recording padded on the right.
    pub reference: Vec<f32>,
    /// `[patches][patch_size][feat_dim]`; padded on the left. Empty without
    /// a transcript.
    pub prompt: Vec<f32>,
    pub transcript: String,
}

/// `SynthesisOptions`.
#[derive(Debug, Clone)]
pub struct SynthesisOptions<'a> {
    pub text: String,
    /// Voice design or style hint; prepended in parentheses.
    pub instruct: String,
    pub voice: Option<&'a ReferenceVoice>,
    /// Upper bound of generated patches; the reference also stops at 6
    /// patches per text token plus 10.
    pub max_patches: usize,
    pub seed: u64,
    pub cfg: f32,
    pub timesteps: usize,
    /// Retries (seed + 1, at most three attempts) a buffered result that ran
    /// into the text-length cap, as the reference's `retry_badcase` does.
    pub retry: bool,
}

impl Default for SynthesisOptions<'_> {
    fn default() -> Self {
        Self {
            text: String::new(),
            instruct: String::new(),
            voice: None,
            max_patches: 4096,
            seed: 0,
            cfg: 2.0,
            timesteps: 10,
            retry: true,
        }
    }
}

/// Receives each decoded patch; returning false cancels (`AudioSink`).
pub type AudioSink<'s> = &'s mut dyn FnMut(&[f32]) -> bool;

const CANCELLED: &str = "cancelled";

/// Text to 48 kHz speech with VoxCPM2 (`Synthesizer`). `Send`, not `Sync`.
pub struct Synthesizer {
    config: Config,
    tokenizer: Tokenizer,
    generator: Generator,
    vae: AudioVae,
}

/// The prompt of a request (`Built`).
struct Built {
    prompt: Prompt,
    target_tokens: usize,
    /// Prompt patches the decoder sees before the generated ones.
    context: Vec<f32>,
}

impl Synthesizer {
    /// Loads a Hugging Face directory or an audio.cpp GGUF and warms the
    /// GPU paths up (`Load`).
    pub fn load(model_root: &Path) -> Result<Self, String> {
        // The files stay mapped until the weights are on the device.
        let files = ModelFiles::open(model_root)?;
        let config = parse_config(files.config())?;
        is_supported(&config)?;
        let tokenizer = Tokenizer::parse(files.tokenizer())?;
        let generator = Generator::create(&config, &files)?;
        let vae = AudioVae::create(&config, &files)?;
        let mut synthesizer = Self {
            config,
            tokenizer,
            generator,
            vae,
        };
        synthesizer
            .warmup()
            .map_err(|e| format!("VoxCPM2 warm-up failed: {e}"))?;
        Ok(synthesizer)
    }

    pub fn sample_rate(&self) -> u32 {
        self.vae.output_rate() as u32
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// A reference prompt of 8 patches and 64 text rows, one patch and its
    /// decoding: hipBLASLt times its candidates for the prompt products (up
    /// to 128 rows) and the DiT's time embeddings are computed, so that the
    /// first request doesn't wait for either (`Warmup`).
    fn warmup(&mut self) -> Result<(), String> {
        let patch_values = self.config.patch_size * self.config.feat_dim;
        let mut prompt = Prompt::default();
        let mut add = |token: i32, text_row: bool| {
            prompt.tokens.push(token);
            prompt.text_mask.push(i32::from(text_row));
            prompt.audio_mask.push(i32::from(!text_row));
            prompt.feats.resize(prompt.feats.len() + patch_values, 0.0);
        };
        add(REFERENCE_START, true);
        for _ in 0..8 {
            add(0, false);
        }
        add(REFERENCE_END, true);
        for row in 0..64 {
            add(1000 + row, true);
        }
        add(AUDIO_START, true);
        self.generator.prefill(&prompt)?;
        let noise = vec![0.0f32; patch_values];
        let mut stop = [0.0f32; 2];
        let defaults = SynthesisOptions::default();
        self.generator
            .step(&noise, defaults.timesteps, defaults.cfg, None, &mut stop)?;
        let mut audio = Vec::new();
        self.vae.reset()?;
        self.vae.enqueue(
            self.generator.latent(),
            self.config.patch_size,
            Some(self.generator.latent_event()),
        )?;
        self.vae.collect(&mut audio)?;
        self.vae.reset()
    }

    /// Encodes interleaved audio at any rate and its optional transcript
    /// (`CloneVoice`).
    pub fn clone_voice(
        &mut self,
        samples: &[f32],
        channels: u32,
        sample_rate: u32,
        transcript: &str,
    ) -> Result<ReferenceVoice, String> {
        self.encode_voice(samples, channels, sample_rate, transcript)
            .map_err(|failure| match failure {
                Failure::Error(error) => format!("cannot encode the reference: {error}"),
                Failure::Plain(error) => error,
                Failure::Cancelled => CANCELLED.to_string(),
            })
    }

    fn encode_voice(
        &mut self,
        samples: &[f32],
        channels: u32,
        sample_rate: u32,
        transcript: &str,
    ) -> Result<ReferenceVoice, Failure> {
        let mono = downmix_mono(samples, channels)?;
        let rate = self.config.vae.sample_rate as u32;
        let audio = gufo_resample::resample(&mono, sample_rate, rate)
            .map_err(|_| "Resample: a rate is 0 or more than 64 times the other".to_string())?;
        if audio.is_empty() {
            return Err(Failure::Plain("the reference recording is empty".into()));
        }
        // _encode_wav: pad to whole patches, on the right for the isolated
        // reference and on the left for a continued prompt.
        let patch = self.config.patch_size * self.config.vae.hop();
        let padding = (patch - audio.len() % patch) % patch;
        let mut right = audio.clone();
        right.resize(audio.len() + padding, 0.0);
        let mut voice = ReferenceVoice {
            reference: self.vae.encode(&right)?,
            prompt: Vec::new(),
            // The transcript is used as given, like the reference's
            // prompt_text.
            transcript: transcript.to_string(),
        };
        let cleaned = collapse_whitespace(transcript);
        if !cleaned.is_empty() && cleaned != " " {
            let mut left = vec![0.0f32; padding];
            left.extend_from_slice(&audio);
            voice.prompt = self.vae.encode(&left)?;
        } else {
            voice.transcript.clear();
        }
        Ok(voice)
    }

    /// Synthesizes `options.text`. With `sink`, audio is delivered patch by
    /// patch and not kept; otherwise it is appended to `audio`
    /// (`Synthesize`). Cancellation (by `cancelled` or a sink that returns
    /// false) is the error "cancelled".
    pub fn synthesize(
        &mut self,
        options: &SynthesisOptions<'_>,
        cancelled: Option<&dyn Fn() -> bool>,
        mut sink: Option<AudioSink<'_>>,
        mut audio: Option<&mut Vec<f32>>,
    ) -> Result<(), String> {
        let text = collapse_whitespace(&options.text);
        if text.is_empty() || text == " " {
            return Err("the text to synthesize is empty".into());
        }
        let target = if options.instruct.is_empty() {
            text
        } else {
            format!("({}){text}", options.instruct)
        };
        let voice = options.voice;
        let continuation = voice.is_some_and(|voice| !voice.prompt.is_empty());
        let patch_values = self.config.patch_size * self.config.feat_dim;

        let mut ids = Vec::new();
        let mut target_ids = Vec::new();
        match voice {
            Some(voice) if continuation => self
                .tokenizer
                .encode(&format!("{}{target}", voice.transcript), &mut ids)?,
            _ => self.tokenizer.encode(&target, &mut ids)?,
        }
        self.tokenizer.encode(&target, &mut target_ids)?;
        let mut built = Built {
            prompt: Prompt::default(),
            target_tokens: target_ids.len(),
            context: Vec::new(),
        };
        let prompt = &mut built.prompt;
        let mut add = |token: i32, text_row: bool, feat: Option<&[f32]>| {
            prompt.tokens.push(token);
            prompt.text_mask.push(i32::from(text_row));
            prompt.audio_mask.push(i32::from(!text_row));
            match feat {
                None => prompt.feats.resize(prompt.feats.len() + patch_values, 0.0),
                Some(feat) => prompt.feats.extend_from_slice(feat),
            }
        };
        if let Some(voice) = voice {
            add(REFERENCE_START, true, None);
            for feat in voice.reference.chunks(patch_values) {
                add(0, false, Some(feat));
            }
            add(REFERENCE_END, true, None);
        }
        for &id in &ids {
            add(id as i32, true, None);
        }
        add(AUDIO_START, true, None);
        if let Some(voice) = voice.filter(|_| continuation) {
            for feat in voice.prompt.chunks(patch_values) {
                add(0, false, Some(feat));
            }
            let patches = voice.prompt.len() / patch_values;
            let context = CONTEXT_PATCHES.min(patches);
            built.context = voice.prompt[voice.prompt.len() - context * patch_values..].to_vec();
        }
        let rows = built.prompt.tokens.len();
        let capacity = self.generator.capacity();
        if rows + 1 >= capacity {
            return Err(format!(
                "the text and reference exceed VoxCPM2's {capacity} positions"
            ));
        }
        let mut limit = options
            .max_patches
            .min((built.target_tokens as f64 * PATCHES_PER_TOKEN + 10.0) as usize);
        limit = limit.min(capacity - rows);

        self.generate(options, &built, limit, cancelled, &mut sink, &mut audio)
            .map_err(|failure| match failure {
                Failure::Cancelled => CANCELLED.to_string(),
                Failure::Error(error) => format!("VoxCPM2 synthesis failed: {error}"),
                Failure::Plain(error) => error,
            })
    }

    /// The attempts of [`Synthesizer::synthesize`] (its `try` block).
    fn generate(
        &mut self,
        options: &SynthesisOptions<'_>,
        built: &Built,
        limit: usize,
        cancelled: Option<&dyn Fn() -> bool>,
        sink: &mut Option<AudioSink<'_>>,
        audio: &mut Option<&mut Vec<f32>>,
    ) -> Result<(), Failure> {
        let patch_values = self.config.patch_size * self.config.feat_dim;
        let mut seed = options.seed;
        let mut attempt = 0;
        loop {
            let mut random = Mt19937_64::new(seed);
            let mut normal = Normal::default();
            let mut noise = vec![0.0f32; patch_values];
            let mut decoded = Vec::new();
            let mut stop = [0.0f32; 2];
            if let Some(audio) = audio.as_deref_mut() {
                audio.clear();
            }
            self.generator.prefill(&built.prompt)?;
            self.vae.reset()?;
            if !built.context.is_empty() {
                self.vae.decode(&built.context, &mut decoded)?;
            }
            let mut patches = 0;
            // A patch decodes on the AudioVAE's stream while the next one is
            // generated; `deliver` hands the decoded patch on.
            let mut pending = false;
            for index in 0..limit {
                if cancelled.is_some_and(|cancelled| cancelled()) {
                    return Err(Failure::Cancelled);
                }
                for value in &mut noise {
                    *value = normal.sample(&mut random);
                }
                self.generator
                    .step(&noise, options.timesteps, options.cfg, None, &mut stop)?;
                patches += 1;
                if !self.deliver(&mut pending, &mut decoded, sink, audio)? {
                    return Err(Failure::Cancelled);
                }
                self.vae.enqueue(
                    self.generator.latent(),
                    self.config.patch_size,
                    Some(self.generator.latent_event()),
                )?;
                pending = true;
                // The first patch is sent at once, for the first audio's
                // latency.
                if index == 0 && !self.deliver(&mut pending, &mut decoded, sink, audio)? {
                    return Err(Failure::Cancelled);
                }
                if index > MIN_PATCHES && stop[1] > stop[0] {
                    break;
                }
            }
            if !self.deliver(&mut pending, &mut decoded, sink, audio)? {
                return Err(Failure::Cancelled);
            }
            let runaway = patches as f64 >= built.target_tokens as f64 * PATCHES_PER_TOKEN;
            if !runaway || sink.is_some() || !options.retry || attempt + 1 >= RETRIES {
                return Ok(());
            }
            seed += 1;
            attempt += 1;
        }
    }

    /// Hands the pending decoded patch on: to the sink, or into `audio`
    /// (`deliver`). False when the sink cancels.
    fn deliver(
        &mut self,
        pending: &mut bool,
        decoded: &mut Vec<f32>,
        sink: &mut Option<AudioSink<'_>>,
        audio: &mut Option<&mut Vec<f32>>,
    ) -> Result<bool, Failure> {
        if !*pending {
            return Ok(true);
        }
        *pending = false;
        decoded.clear();
        self.vae.collect(decoded)?;
        if let Some(sink) = sink.as_mut() {
            return Ok(sink(decoded));
        }
        if let Some(audio) = audio.as_deref_mut() {
            audio.extend_from_slice(decoded);
        }
        Ok(true)
    }
}

/// Why the patch loop or a voice's encoding ended early.
enum Failure {
    Cancelled,
    /// An error the caller reports with its context.
    Error(String),
    /// An error reported as it is.
    Plain(String),
}

impl From<String> for Failure {
    fn from(error: String) -> Self {
        Self::Error(error)
    }
}

/// The mean of each frame's channels: a float sum in channel order divided
/// by the channel count (Gufo 0.34.0's `core::DownmixMono`).
fn downmix_mono(interleaved: &[f32], channels: u32) -> Result<Vec<f32>, String> {
    if channels == 0 || !interleaved.len().is_multiple_of(channels as usize) {
        return Err("DownmixMono: channels must divide the sample count".into());
    }
    Ok(interleaved
        .chunks_exact(channels as usize)
        .map(|frame| {
            let mut sum = 0.0f32;
            for &sample in frame {
                sum += sample;
            }
            sum / channels as f32
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_averages_in_channel_order() {
        assert_eq!(
            downmix_mono(&[1.0, 3.0, -2.0, 0.5], 2).unwrap(),
            vec![2.0, -0.75]
        );
        // A float sum from zero: -0 becomes +0 as in the C++ loop.
        assert_eq!(downmix_mono(&[-0.0], 1).unwrap()[0].to_bits(), 0);
        assert!(downmix_mono(&[1.0, 2.0, 3.0], 2).is_err());
        assert!(downmix_mono(&[1.0], 0).is_err());
    }
}
