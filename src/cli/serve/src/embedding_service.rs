//! The EmbeddingGemma 2 text, image, video and audio embedding service
//! (`src/cli/serve/embedding_service.cpp`): model identity, readiness, the
//! input preparation and length checks, the checkpoint's named prompts and
//! the serialised injected runner. The native engine (tokenizer, media
//! preprocessing and GPU encoder) is injected by the binary, since this
//! crate has no GPU dependency.

use std::any::Any;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use gufo_core::json;
use gufo_core::text::CancellationCheck;

use crate::image_url::{ImageReadBudget, read_audio_url, read_image_url, read_video_url};
use crate::logging;

/// Longest accepted input in tokens, including the surrounding markers: the
/// model's context window.
pub const MAX_INPUT_TOKENS: usize = 8192;
/// Most images of one request.
pub const MAX_IMAGES: usize = 32;
/// Most audio clips of one request.
pub const MAX_AUDIO: usize = 32;
/// Most videos of one request.
pub const MAX_VIDEOS: usize = 32;
/// Most sampled frames over every video of one request.
pub const MAX_VIDEO_FRAMES: usize = 64;

/// `Gemma4ImageProcessor`'s supported `max_soft_tokens` values and the
/// checkpoint's default (`image_processor.hpp`); the binary checks them
/// against the engine crate.
pub const IMAGE_SOFT_TOKEN_BUDGETS: [u32; 5] = [70, 140, 280, 560, 1120];
pub const DEFAULT_IMAGE_SOFT_TOKENS: u32 = 280;
pub const MAX_IMAGE_SOFT_TOKENS: u32 = 1120;

/// `SupportedImageSoftTokens`.
pub fn supported_image_soft_tokens(max_soft_tokens: u32) -> bool {
    IMAGE_SOFT_TOKEN_BUDGETS.contains(&max_soft_tokens)
}

/// The model's placeholder tokens and their framing (`config.hpp`); the
/// binary checks them against the engine crate.
pub const IMAGE_TOKEN_ID: u32 = 258880;
pub const VIDEO_TOKEN_ID: u32 = 258884;
pub const AUDIO_TOKEN_ID: u32 = 258881;
pub const IMAGE_TOKEN: &str = "<|image|>";
pub const BEGIN_IMAGE_TOKEN: &str = "<|image>";
pub const END_IMAGE_TOKEN: &str = "<image|>";
pub const AUDIO_TOKEN: &str = "<|audio|>";
pub const BEGIN_AUDIO_TOKEN: &str = "<|audio>";
pub const END_AUDIO_TOKEN: &str = "<audio|>";

const DEFAULT_MODEL_ID: &str = "embeddinggemma-2";
const INVALID_MODEL: &str = "embedding service requires an EmbeddingGemma 2 model directory";

/// Embeddings for one request, in input order. Each vector is the model's
/// full-width, L2-normalised output. The caller applies any Matryoshka
/// truncation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbeddingBatch {
    pub embeddings: Vec<Vec<f32>>,
    /// Tokens per input including the beginning and end markers, after any
    /// truncation.
    pub token_counts: Vec<usize>,
    /// Index of the offending input, set only with
    /// [`EmbeddingStatus::InputTooLong`].
    pub failed_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingPartType {
    Text,
    /// An image URL: a data URL or HTTPS.
    Image,
    /// An audio URL, or raw base64.
    Audio,
    /// A video URL, or raw base64.
    Video,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingPart {
    pub kind: EmbeddingPartType,
    pub value: String,
}

impl EmbeddingPart {
    pub fn text(value: impl Into<String>) -> Self {
        Self {
            kind: EmbeddingPartType::Text,
            value: value.into(),
        }
    }
}

/// One input: text, image, video and audio parts in order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbeddingInput {
    pub content: Vec<EmbeddingPart>,
}

/// The video processor controls of one request (`VideoOptions`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingVideoOptions {
    pub fps: f64,
    pub max_frames: usize,
    pub timestamps: bool,
    pub truncate_overflow: bool,
}

impl Default for EmbeddingVideoOptions {
    fn default() -> Self {
        Self {
            fps: 1.0,
            max_frames: 32,
            timestamps: false,
            truncate_overflow: false,
        }
    }
}

/// Per-request choices that change how inputs are encoded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingRequestOptions {
    /// Cut an input longer than [`MAX_INPUT_TOKENS`] to its first tokens,
    /// keeping the end marker, instead of refusing it. An input whose images
    /// would not fit whole is still refused.
    pub truncate: bool,
    /// `Gemma4ImageProcessor` `max_soft_tokens` for every image of the
    /// request.
    pub image_soft_tokens: u32,
    pub video: EmbeddingVideoOptions,
}

impl Default for EmbeddingRequestOptions {
    fn default() -> Self {
        Self {
            truncate: false,
            image_soft_tokens: DEFAULT_IMAGE_SOFT_TOKENS,
            video: EmbeddingVideoOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingStatus {
    Ok,
    Cancelled,
    InputTooLong,
    InvalidInput,
    InvalidAudioInput,
    InvalidVideoInput,
    Failed,
    /// The device failed a probe after an error. The service stays lost.
    DeviceLost,
}

/// Replaces the native engine: embeds the inputs into the batch, or reports
/// a status with a message. The service serialises calls to it.
pub type EmbeddingRunner = Box<
    dyn Fn(
            &[EmbeddingInput],
            &EmbeddingRequestOptions,
            &mut EmbeddingBatch,
            &mut String,
        ) -> EmbeddingStatus
        + Send
        + Sync,
>;

/// A preprocessed image, video frame or audio clip of the engine, opaque to
/// this crate, with the placeholder tokens it fills.
pub struct EngineMedia {
    pub soft_tokens: usize,
    pub data: Box<dyn Any + Send + Sync>,
}

/// An image or video frame of a sequence: the offset of its first
/// placeholder token and the placeholder ([`IMAGE_TOKEN_ID`] or
/// [`VIDEO_TOKEN_ID`]); video frames share the vision tower.
pub struct EngineImage {
    pub offset: usize,
    pub token_id: u32,
    pub media: EngineMedia,
}

/// One sequence for the engine's encoder: its tokens and the media with the
/// offsets of their first placeholder tokens.
#[derive(Default)]
pub struct EngineSequence {
    pub tokens: Vec<u32>,
    pub images: Vec<EngineImage>,
    pub audio: Vec<(usize, EngineMedia)>,
}

/// A decoded video: its sampled frames and the text that frames them
/// (`FrameVideo`).
pub struct EngineVideo {
    pub text: String,
    pub frames: Vec<EngineMedia>,
}

/// Why a video could not be preprocessed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineVideoError {
    /// The video libraries could not be loaded: the request fails as a
    /// server error.
    Library(String),
    /// The input was refused or failed to decode.
    Invalid(String),
}

/// The native model: its tokenizer, media preprocessing and GPU encoder,
/// which shares device batches between concurrent requests itself.
pub trait EmbeddingEngine: Send + Sync {
    /// Longest vocabulary entry in bytes of normalised text.
    fn max_token_bytes(&self) -> usize;
    /// Token IDs of `text` with the `<bos>` and `<eos>` markers.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// Like `encode`, cut to `max_tokens` before the `<eos>`.
    fn encode_truncated(&self, text: &str, max_tokens: usize) -> Vec<u32>;
    fn supports_vision(&self) -> bool;
    fn supports_video(&self) -> bool;
    fn supports_audio(&self) -> bool;
    /// Why the checkpoint's image processor settings are unsupported; empty
    /// when they are supported.
    fn image_processor_error(&self) -> String;
    /// Why the checkpoint's audio feature extractor is unsupported; empty
    /// when it is supported.
    fn audio_processor_error(&self) -> String;
    /// Why the checkpoint's video processor settings are unsupported; empty
    /// when they are supported.
    fn video_processor_error(&self) -> String;
    /// Decodes and preprocesses an encoded image at a soft-token budget of
    /// [`IMAGE_SOFT_TOKEN_BUDGETS`].
    fn process_image(&self, bytes: &[u8], max_soft_tokens: u32) -> Result<EngineMedia, String>;
    /// Decodes an encoded video and preprocesses at most `frame_budget`
    /// sampled frames.
    fn process_video(
        &self,
        bytes: &[u8],
        options: &EmbeddingVideoOptions,
        is_cancelled: Option<&CancellationCheck>,
        frame_budget: usize,
    ) -> Result<EngineVideo, EngineVideoError>;
    /// Decodes and preprocesses an encoded audio clip; `Ok` with no soft
    /// tokens for a clip too short to embed.
    fn process_audio(
        &self,
        bytes: &[u8],
        is_cancelled: Option<&CancellationCheck>,
    ) -> Result<EngineMedia, String>;
    /// One embedding per sequence; `is_cancelled` is polled before each
    /// device batch.
    fn embed(
        &self,
        sequences: Vec<EngineSequence>,
        is_cancelled: Option<CancellationCheck>,
    ) -> Result<Vec<Vec<f32>>, String>;
    /// True once a device failure was confirmed by a probe.
    fn device_lost(&self) -> bool;
}

/// Whether the model root holds an EmbeddingGemma 2 checkpoint
/// (`LooksLikeEmbeddingGemma2`). The config parser lives with the engine
/// crate, which this crate cannot depend on, so the binary supplies it.
pub type ModelValidator = Box<dyn Fn(&Path) -> bool + Send + Sync>;

/// Loads the tokenizer and creates the encoder from the model root; the
/// error is the service's initialization error.
pub type EmbeddingEngineFactory =
    Box<dyn FnOnce(&Path) -> Result<Box<dyn EmbeddingEngine>, String> + Send>;

pub struct EmbeddingServiceOptions {
    pub model_root: PathBuf,
    /// Rejects other directories through `model_validator` (none supplied
    /// counts as invalid) when a runner is injected; the native engine is
    /// always validated.
    pub validate_model: bool,
    pub model_validator: Option<ModelValidator>,
    /// Empty: `embeddinggemma-2`.
    pub model_id: String,
    /// Named input prefixes, as in a checkpoint's
    /// config_sentence_transformers.json `prompts`.
    pub prompts: BTreeMap<String, String>,
    /// Prompt applied when a request names none; empty applies none. Must
    /// name an entry of `prompts`.
    pub default_prompt_name: String,
    /// Replaces the native engine. The service serialises calls to it.
    pub runner: Option<EmbeddingRunner>,
    /// Creates the native engine after validation; this crate has no GPU
    /// code, so without a runner it is required.
    pub engine_factory: Option<EmbeddingEngineFactory>,
}

impl Default for EmbeddingServiceOptions {
    fn default() -> Self {
        Self {
            model_root: PathBuf::new(),
            validate_model: true,
            model_validator: None,
            model_id: String::new(),
            prompts: BTreeMap::new(),
            default_prompt_name: String::new(),
            runner: None,
            engine_factory: None,
        }
    }
}

/// `TruncateKeepingImages`: cuts `tokens` (an encoding that ends in one
/// `<eos>`) to `max_tokens` by dropping content tokens before the `<eos>`.
/// Returns false, leaving `tokens` unchanged, when the cut would drop an
/// `image_token` or the end-of-image marker right after the last one.
pub fn truncate_keeping_images(tokens: &mut Vec<u32>, max_tokens: usize, image_token: u32) -> bool {
    if tokens.len() <= max_tokens {
        return true;
    }
    if max_tokens < 2 {
        return false;
    }
    // Keep the first tokens and the final <eos>. The last image placeholder
    // and the end-of-image marker after it must both stay.
    let keep = max_tokens - 1;
    let image_end = tokens
        .iter()
        .rposition(|&token| token == image_token)
        .map_or(0, |last| last + 2);
    if image_end > keep {
        return false;
    }
    let end = tokens.len() - 1;
    tokens.drain(keep..end);
    true
}

/// `LoadSentenceTransformerPrompts`: the `prompts` of
/// `model_root`/config_sentence_transformers.json. A missing file yields no
/// prompts; a malformed one fails.
pub fn load_sentence_transformer_prompts(
    model_root: &Path,
) -> Result<BTreeMap<String, String>, String> {
    let path = model_root.join("config_sentence_transformers.json");
    let mut prompts = BTreeMap::new();
    let Ok(text) = std::fs::read(&path) else {
        return Ok(prompts);
    };
    let root = json::parse(&text).map_err(|e| format!("{}: {}", path.display(), e.message()))?;
    let table = if root.is_object() {
        root.find("prompts")
    } else {
        None
    };
    let Some(table) = table.filter(|table| !table.is_null()) else {
        return Ok(prompts);
    };
    if !table.is_object() {
        return Err(format!("{}: 'prompts' must be an object", path.display()));
    }
    for (name, value) in table.members() {
        if !value.is_string() {
            return Err(format!(
                "{}: prompt '{name}' must be a string",
                path.display()
            ));
        }
        prompts
            .entry(name.clone())
            .or_insert_with(|| value.str().to_owned());
    }
    Ok(prompts)
}

/// Native HIP EmbeddingGemma 2 text, image, video and audio embedding
/// service.
pub struct EmbeddingService {
    model_id: String,
    prompts: BTreeMap<String, String>,
    default_prompt_name: String,
    runner: Option<EmbeddingRunner>,
    engine: Option<Box<dyn EmbeddingEngine>>,
    image_processor_error: String,
    audio_processor_error: String,
    video_processor_error: String,
    /// Serialises the injected runner, which has no device turns of its own.
    runner_lock: Mutex<()>,
    device_lost: AtomicBool,
    initialization_error: String,
    ready: bool,
}

impl EmbeddingService {
    pub fn new(options: EmbeddingServiceOptions) -> Self {
        let EmbeddingServiceOptions {
            model_root,
            validate_model,
            model_validator,
            mut model_id,
            prompts,
            default_prompt_name,
            runner,
            engine_factory,
        } = options;
        if model_id.is_empty() {
            model_id = DEFAULT_MODEL_ID.into();
        }
        let mut service = Self {
            model_id,
            prompts,
            default_prompt_name,
            runner: None,
            engine: None,
            image_processor_error: String::new(),
            audio_processor_error: String::new(),
            video_processor_error: String::new(),
            runner_lock: Mutex::new(()),
            device_lost: AtomicBool::new(false),
            initialization_error: String::new(),
            ready: false,
        };
        service.start(
            model_root,
            validate_model,
            model_validator,
            runner,
            engine_factory,
        );
        let name = &service.default_prompt_name;
        if service.ready && !name.is_empty() && !service.prompts.contains_key(name) {
            service.initialization_error =
                format!("--default-prompt-name '{name}' is not a prompt of this model");
            service.ready = false;
        }
        service
    }

    fn start(
        &mut self,
        model_root: PathBuf,
        validate_model: bool,
        model_validator: Option<ModelValidator>,
        runner: Option<EmbeddingRunner>,
        engine_factory: Option<EmbeddingEngineFactory>,
    ) {
        let service = self;
        let valid =
            |validator: Option<ModelValidator>| validator.is_some_and(|valid| valid(&model_root));
        if let Some(runner) = runner {
            // An injected runner needs no checkpoint, so the model is not
            // validated unless the caller asks for it.
            if validate_model && !valid(model_validator) {
                service.initialization_error = INVALID_MODEL.into();
                return;
            }
            service.runner = Some(runner);
            service.ready = true;
            return;
        }
        let Some(factory) = engine_factory else {
            service.initialization_error = "embedding service requires the HIP backend".into();
            return;
        };
        if !valid(model_validator) {
            service.initialization_error = INVALID_MODEL.into();
            return;
        }
        match factory(&model_root) {
            Ok(engine) => {
                service.image_processor_error = engine.image_processor_error();
                service.audio_processor_error = engine.audio_processor_error();
                service.video_processor_error = engine.video_processor_error();
                service.engine = Some(engine);
                service.ready = true;
            }
            Err(error) => service.initialization_error = error,
        }
    }

    /// The prefix registered under `name`.
    pub fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts.get(name).map(String::as_str)
    }

    pub fn default_prompt_name(&self) -> &str {
        &self.default_prompt_name
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    /// True once any request reported [`EmbeddingStatus::DeviceLost`].
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Acquire)
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

    /// Embeds `texts` as one request (see [`Self::embed_inputs`]).
    pub fn embed(
        &self,
        texts: &[String],
        is_cancelled: Option<CancellationCheck>,
        batch: &mut EmbeddingBatch,
        error: &mut String,
    ) -> EmbeddingStatus {
        let inputs: Vec<EmbeddingInput> = texts
            .iter()
            .map(|text| EmbeddingInput {
                content: vec![EmbeddingPart::text(text.clone())],
            })
            .collect();
        self.embed_inputs(
            &inputs,
            is_cancelled,
            batch,
            error,
            &EmbeddingRequestOptions::default(),
        )
    }

    /// Embeds `inputs` as one request. Concurrent requests share device
    /// batches in arrival order, and a long request does not block shorter
    /// ones until it finishes. Returns `Cancelled` for a request cancelled
    /// before one of its batches reaches the device. `batch` is reset first.
    pub fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
        is_cancelled: Option<CancellationCheck>,
        batch: &mut EmbeddingBatch,
        error: &mut String,
        request: &EmbeddingRequestOptions,
    ) -> EmbeddingStatus {
        *batch = EmbeddingBatch::default();
        if !self.ready {
            *error = self.initialization_error.clone();
            return EmbeddingStatus::Failed;
        }
        let status = if let Some(runner) = &self.runner {
            let _lock = self.runner_lock.lock().unwrap_or_else(|e| e.into_inner());
            if is_cancelled.as_ref().is_some_and(|cancelled| cancelled()) {
                *error = "embedding request cancelled".into();
                return EmbeddingStatus::Cancelled;
            }
            runner(inputs, request, batch, error)
        } else if let Some(engine) = &self.engine {
            self.run_native(engine.as_ref(), inputs, request, is_cancelled, batch, error)
        } else {
            EmbeddingStatus::Failed
        };
        if status == EmbeddingStatus::DeviceLost && !self.device_lost.swap(true, Ordering::AcqRel) {
            logging::error(
                "embedding",
                &format!("event=device_lost remedy=restart reason={error}"),
            );
        }
        status
    }

    /// Tokenizes before joining the encoder's queue, so a large request does
    /// not block other requests that are ready to run (`RunNative`).
    fn run_native(
        &self,
        engine: &dyn EmbeddingEngine,
        inputs: &[EmbeddingInput],
        request: &EmbeddingRequestOptions,
        is_cancelled: Option<CancellationCheck>,
        batch: &mut EmbeddingBatch,
        error: &mut String,
    ) -> EmbeddingStatus {
        let cancelled = || is_cancelled.as_ref().is_some_and(|cancelled| cancelled());
        let mut sequences = Vec::with_capacity(inputs.len());
        let mut budget = ImageReadBudget::default();
        let byte_limit = MAX_INPUT_TOKENS.saturating_mul(engine.max_token_bytes());
        let mut image_count = 0;
        let mut audio_count = 0;
        let mut video_count = 0;
        let mut video_frames = 0;
        for (index, input) in inputs.iter().enumerate() {
            if cancelled() {
                return EmbeddingStatus::Cancelled;
            }
            let mut sequence = EngineSequence::default();
            let mut text = String::new();
            // Use the tokenizer once for the complete input, preserving BPE
            // merges across adjacent text parts and the reference BOS/EOS
            // framing.
            for part in &input.content {
                if cancelled() {
                    return EmbeddingStatus::Cancelled;
                }
                match part.kind {
                    EmbeddingPartType::Text => text.push_str(&part.value),
                    EmbeddingPartType::Video => {
                        if !engine.supports_video() || !self.video_processor_error.is_empty() {
                            *error = if self.video_processor_error.is_empty() {
                                "unsupported embedding vision configuration".into()
                            } else {
                                self.video_processor_error.clone()
                            };
                            return EmbeddingStatus::InvalidVideoInput;
                        }
                        let processed = (|| {
                            video_count += 1;
                            if video_count > MAX_VIDEOS {
                                return Err(EngineVideoError::Invalid(
                                    "embedding requests accept at most 32 videos".into(),
                                ));
                            }
                            let bytes = read_video_url(&part.value, &mut budget)
                                .map_err(|e| EngineVideoError::Invalid(e.message().to_owned()))?;
                            let video = engine.process_video(
                                &bytes,
                                &request.video,
                                is_cancelled.as_ref(),
                                MAX_VIDEO_FRAMES - video_frames,
                            )?;
                            video_frames += video.frames.len();
                            if video_frames > MAX_VIDEO_FRAMES {
                                return Err(EngineVideoError::Invalid(
                                    "embedding requests accept at most 64 sampled video frames"
                                        .into(),
                                ));
                            }
                            Ok(video)
                        })();
                        match processed {
                            Ok(video) => {
                                text.push_str(&video.text);
                                sequence
                                    .images
                                    .extend(video.frames.into_iter().map(|media| EngineImage {
                                        offset: 0,
                                        token_id: VIDEO_TOKEN_ID,
                                        media,
                                    }));
                            }
                            Err(EngineVideoError::Library(failure)) => {
                                *error = failure;
                                return EmbeddingStatus::Failed;
                            }
                            Err(EngineVideoError::Invalid(failure)) => {
                                if cancelled() {
                                    return EmbeddingStatus::Cancelled;
                                }
                                *error = failure;
                                return EmbeddingStatus::InvalidVideoInput;
                            }
                        }
                    }
                    EmbeddingPartType::Audio => {
                        if !engine.supports_audio() || !self.audio_processor_error.is_empty() {
                            *error = if self.audio_processor_error.is_empty() {
                                "unsupported embedding audio configuration".into()
                            } else {
                                self.audio_processor_error.clone()
                            };
                            return EmbeddingStatus::InvalidAudioInput;
                        }
                        audio_count += 1;
                        if audio_count > MAX_AUDIO {
                            *error = "embedding requests accept at most 32 audio clips".into();
                            return EmbeddingStatus::InvalidAudioInput;
                        }
                        let processed = read_audio_url(&part.value, &mut budget)
                            .map_err(|e| e.message().to_owned())
                            .and_then(|bytes| engine.process_audio(&bytes, is_cancelled.as_ref()));
                        let features = match processed {
                            Ok(features) => features,
                            Err(failure) => {
                                if cancelled() {
                                    return EmbeddingStatus::Cancelled;
                                }
                                *error = failure;
                                return EmbeddingStatus::InvalidAudioInput;
                            }
                        };
                        if features.soft_tokens == 0 {
                            *error = "embedding audio needs at least 161 samples at 16 kHz".into();
                            return EmbeddingStatus::InvalidAudioInput;
                        }
                        text.push_str(BEGIN_AUDIO_TOKEN);
                        for _ in 0..features.soft_tokens {
                            text.push_str(AUDIO_TOKEN);
                        }
                        text.push_str(END_AUDIO_TOKEN);
                        sequence.audio.push((0, features));
                    }
                    EmbeddingPartType::Image => {
                        if !engine.supports_vision() {
                            *error = "unsupported embedding vision configuration".into();
                            return EmbeddingStatus::InvalidInput;
                        }
                        if !self.image_processor_error.is_empty() {
                            *error = self.image_processor_error.clone();
                            return EmbeddingStatus::InvalidInput;
                        }
                        image_count += 1;
                        if image_count > MAX_IMAGES {
                            *error = "embedding requests accept at most 32 images".into();
                            return EmbeddingStatus::InvalidInput;
                        }
                        let processed = read_image_url(&part.value, &mut budget)
                            .map_err(|e| e.message().to_owned())
                            .and_then(|bytes| {
                                engine.process_image(&bytes, request.image_soft_tokens)
                            });
                        let patches = match processed {
                            Ok(patches) => patches,
                            Err(failure) => {
                                *error = failure;
                                return EmbeddingStatus::InvalidInput;
                            }
                        };
                        text.push_str(BEGIN_IMAGE_TOKEN);
                        for _ in 0..patches.soft_tokens {
                            text.push_str(IMAGE_TOKEN);
                        }
                        text.push_str(END_IMAGE_TOKEN);
                        sequence.images.push(EngineImage {
                            offset: 0,
                            token_id: IMAGE_TOKEN_ID,
                            media: patches,
                        });
                    }
                }
                if !request.truncate && text.len() > byte_limit {
                    batch.failed_index = index;
                    return EmbeddingStatus::InputTooLong;
                }
            }
            let media = !sequence.images.is_empty() || !sequence.audio.is_empty();
            // A text-only input reads just the bytes its kept tokens can span.
            // An input with images keeps every placeholder: one ending past
            // the bytes the kept tokens can span cannot fit, and text after
            // the last image is bounded like a text-only input.
            if request.truncate && media && text.len() / 2 > byte_limit {
                let end = [text.rfind(END_IMAGE_TOKEN), text.rfind(END_AUDIO_TOKEN)]
                    .into_iter()
                    .flatten()
                    .max();
                if end.is_none_or(|end| end + END_IMAGE_TOKEN.len() > byte_limit) {
                    batch.failed_index = index;
                    return EmbeddingStatus::InputTooLong;
                }
                truncate_bytes(&mut text, 2 * byte_limit);
            }
            sequence.tokens = if request.truncate && !media {
                engine.encode_truncated(&text, MAX_INPUT_TOKENS)
            } else {
                engine.encode(&text)
            };
            // Validate after joining and tokenizing: a literal placeholder may
            // straddle adjacent text parts. It must never capture another
            // image's features or leave an unfilled image token in the text
            // backbone.
            let count = |token: u32| sequence.tokens.iter().filter(|&&t| t == token).count();
            if !sequence.images.is_empty() {
                for token in [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID] {
                    let expected: usize = sequence
                        .images
                        .iter()
                        .filter(|image| image.token_id == token)
                        .map(|image| image.media.soft_tokens)
                        .sum();
                    if count(token) != expected {
                        *error = "text must not contain image/video placeholders when vision \
                                  inputs are supplied"
                            .into();
                        return if token == VIDEO_TOKEN_ID {
                            EmbeddingStatus::InvalidVideoInput
                        } else {
                            EmbeddingStatus::InvalidInput
                        };
                    }
                }
            }
            if !sequence.audio.is_empty()
                && count(AUDIO_TOKEN_ID)
                    != sequence
                        .audio
                        .iter()
                        .map(|(_, m)| m.soft_tokens)
                        .sum::<usize>()
            {
                *error = "text must not contain audio placeholders when audio is supplied".into();
                return EmbeddingStatus::InvalidAudioInput;
            }
            if request.truncate && sequence.tokens.len() > MAX_INPUT_TOKENS {
                let last_video = sequence.tokens.iter().rposition(|&t| t == VIDEO_TOKEN_ID);
                if last_video.is_some_and(|last| last + 2 > MAX_INPUT_TOKENS - 1) {
                    batch.failed_index = index;
                    return EmbeddingStatus::InputTooLong;
                }
                let last_audio = sequence.tokens.iter().rposition(|&t| t == AUDIO_TOKEN_ID);
                if !sequence.audio.is_empty()
                    && last_audio.is_some_and(|last| last + 2 > MAX_INPUT_TOKENS - 1)
                {
                    batch.failed_index = index;
                    return EmbeddingStatus::InputTooLong;
                }
                let placeholder = if sequence.images.is_empty() {
                    AUDIO_TOKEN_ID
                } else {
                    IMAGE_TOKEN_ID
                };
                let _ =
                    truncate_keeping_images(&mut sequence.tokens, MAX_INPUT_TOKENS, placeholder);
            }
            if sequence.tokens.len() > MAX_INPUT_TOKENS {
                batch.failed_index = index;
                return EmbeddingStatus::InputTooLong;
            }
            let mut cursor = 0;
            for image in &mut sequence.images {
                while cursor < sequence.tokens.len() && sequence.tokens[cursor] != image.token_id {
                    cursor += 1;
                }
                image.offset = cursor;
                cursor += image.media.soft_tokens;
            }
            cursor = 0;
            for (offset, clip) in &mut sequence.audio {
                while cursor < sequence.tokens.len() && sequence.tokens[cursor] != AUDIO_TOKEN_ID {
                    cursor += 1;
                }
                *offset = cursor;
                cursor += clip.soft_tokens;
            }
            batch.token_counts.push(sequence.tokens.len());
            sequences.push(sequence);
        }
        match engine.embed(sequences, is_cancelled.clone()) {
            Ok(embeddings) => {
                batch.embeddings = embeddings;
                EmbeddingStatus::Ok
            }
            Err(message) => {
                *error = message;
                if engine.device_lost() {
                    EmbeddingStatus::DeviceLost
                } else if cancelled() {
                    EmbeddingStatus::Cancelled
                } else {
                    EmbeddingStatus::Failed
                }
            }
        }
    }
}

/// `std::string::resize` to fewer bytes, which may cut inside a character;
/// the cut stays far from the kept tokens, so its replacement characters are
/// never encoded into them.
fn truncate_bytes(text: &mut String, len: usize) {
    let mut bytes = std::mem::take(text).into_bytes();
    bytes.truncate(len);
    *text = String::from_utf8_lossy(&bytes).into_owned();
}
