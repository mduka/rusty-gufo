//! The EmbeddingGemma 2 text embedding service
//! (`src/cli/serve/embedding_service.cpp`): model identity, readiness, the
//! input-length checks and the serialised injected runner. The native engine
//! (tokenizer and GPU encoder) is injected by the binary, since this crate
//! has no GPU dependency.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::logging;

/// Longest accepted input in tokens, including the surrounding markers: the
/// model's context window.
pub const MAX_INPUT_TOKENS: usize = 8192;

const DEFAULT_MODEL_ID: &str = "embeddinggemma-2";
const INVALID_MODEL: &str = "embedding service requires an EmbeddingGemma 2 model directory";

/// Embeddings for one request, in input order. Each vector is the model's
/// full-width, L2-normalised output. The caller applies any Matryoshka
/// truncation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbeddingBatch {
    pub embeddings: Vec<Vec<f32>>,
    /// Tokens per input including the beginning and end markers.
    pub token_counts: Vec<usize>,
    /// Index of the offending input, set only with
    /// [`EmbeddingStatus::InputTooLong`].
    pub failed_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingStatus {
    Ok,
    Cancelled,
    InputTooLong,
    Failed,
    /// The device failed a probe after an error. The service stays lost.
    DeviceLost,
}

/// Replaces the native engine: embeds the texts into the batch, or reports
/// a status with a message. The service serialises calls to it.
pub type EmbeddingRunner =
    Box<dyn Fn(&[String], &mut EmbeddingBatch, &mut String) -> EmbeddingStatus + Send + Sync>;

/// The native model: its tokenizer and its GPU encoder, which hands out
/// device turns one batch at a time itself.
pub trait EmbeddingEngine: Send + Sync {
    /// Longest vocabulary entry in bytes of normalised text.
    fn max_token_bytes(&self) -> usize;
    /// Token IDs of `text` with the `<bos>` and `<eos>` markers.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// One embedding per sequence; `is_cancelled` is polled before each
    /// device batch.
    fn embed(
        &self,
        sequences: &[Vec<u32>],
        is_cancelled: &dyn Fn() -> bool,
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
            runner: None,
            engine_factory: None,
        }
    }
}

/// Native HIP EmbeddingGemma 2 text embedding service.
pub struct EmbeddingService {
    model_id: String,
    runner: Option<EmbeddingRunner>,
    engine: Option<Box<dyn EmbeddingEngine>>,
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
            runner,
            engine_factory,
        } = options;
        if model_id.is_empty() {
            model_id = DEFAULT_MODEL_ID.into();
        }
        let mut service = Self {
            model_id,
            runner: None,
            engine: None,
            runner_lock: Mutex::new(()),
            device_lost: AtomicBool::new(false),
            initialization_error: String::new(),
            ready: false,
        };
        let valid =
            |validator: Option<ModelValidator>| validator.is_some_and(|valid| valid(&model_root));
        if let Some(runner) = runner {
            // An injected runner needs no checkpoint, so the model is not
            // validated unless the caller asks for it.
            if validate_model && !valid(model_validator) {
                service.initialization_error = INVALID_MODEL.into();
                return service;
            }
            service.runner = Some(runner);
            service.ready = true;
            return service;
        }
        let Some(factory) = engine_factory else {
            service.initialization_error = "embedding service requires the HIP backend".into();
            return service;
        };
        if !valid(model_validator) {
            service.initialization_error = INVALID_MODEL.into();
            return service;
        }
        match factory(&model_root) {
            Ok(engine) => {
                service.engine = Some(engine);
                service.ready = true;
            }
            Err(error) => service.initialization_error = error,
        }
        service
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

    /// Embeds `texts` as one request. Concurrent requests take device turns
    /// one batch at a time, so a long request does not block shorter ones
    /// until it finishes. Returns `Cancelled` for a request cancelled before
    /// one of its batches reaches the device. `batch` is reset first.
    pub fn embed(
        &self,
        texts: &[String],
        is_cancelled: &dyn Fn() -> bool,
        batch: &mut EmbeddingBatch,
        error: &mut String,
    ) -> EmbeddingStatus {
        *batch = EmbeddingBatch::default();
        if !self.ready {
            *error = self.initialization_error.clone();
            return EmbeddingStatus::Failed;
        }
        let status = if let Some(runner) = &self.runner {
            let _lock = self.runner_lock.lock().unwrap_or_else(|e| e.into_inner());
            if is_cancelled() {
                *error = "embedding request cancelled".into();
                return EmbeddingStatus::Cancelled;
            }
            runner(texts, batch, error)
        } else if let Some(engine) = &self.engine {
            run_native(engine.as_ref(), texts, is_cancelled, batch, error)
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
}

/// Tokenizes before taking any device turn, so a large request does not
/// block other requests that are ready to run. The encoder hands out device
/// turns one batch at a time.
fn run_native(
    engine: &dyn EmbeddingEngine,
    texts: &[String],
    is_cancelled: &dyn Fn() -> bool,
    batch: &mut EmbeddingBatch,
    error: &mut String,
) -> EmbeddingStatus {
    let mut sequences = Vec::with_capacity(texts.len());
    batch.token_counts.clear();
    // Normalisation never shortens the text and no token covers more than
    // max_token_bytes() of it, so a longer input cannot fit. Rejecting it
    // here bounds the tokenizing work an oversized input can cause.
    let byte_limit = MAX_INPUT_TOKENS.saturating_mul(engine.max_token_bytes());
    for (index, text) in texts.iter().enumerate() {
        if text.len() > byte_limit {
            batch.failed_index = index;
            return EmbeddingStatus::InputTooLong;
        }
        let tokens = engine.encode(text);
        if tokens.len() > MAX_INPUT_TOKENS {
            batch.failed_index = index;
            return EmbeddingStatus::InputTooLong;
        }
        batch.token_counts.push(tokens.len());
        sequences.push(tokens);
    }
    match engine.embed(&sequences, is_cancelled) {
        Ok(embeddings) => {
            batch.embeddings = embeddings;
            EmbeddingStatus::Ok
        }
        Err(message) => {
            *error = message;
            if engine.device_lost() {
                EmbeddingStatus::DeviceLost
            } else if is_cancelled() {
                EmbeddingStatus::Cancelled
            } else {
                EmbeddingStatus::Failed
            }
        }
    }
}
