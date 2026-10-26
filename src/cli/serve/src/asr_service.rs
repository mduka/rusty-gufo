//! The bounded Qwen3-ASR service (`src/cli/serve/asr_service.cpp`).

use std::path::{Path, PathBuf};

use gufo_core::cancellable_gate::CancellableGate;

use crate::asr_model::{AsrRunner, TextSink, TranscriptionRequest, TranscriptionResult};

/// Whether the checkpoint at the model root is the supported 1.7B model
/// (`LoadModelConfigFromPath` + `IsSupportedModelConfig`). The config parser
/// lives with the engine crate, which this crate cannot depend on, so the
/// binary supplies it.
pub type ModelValidator = Box<dyn Fn(&Path) -> bool + Send + Sync>;

/// Creates the native runtime from the model root and the context capacity
/// (`TranscriptionHipRuntime::Create`); the error is the load message.
pub type AsrRuntimeFactory = Box<dyn FnOnce(&Path, usize) -> Result<AsrRunner, String> + Send>;

pub struct AsrServiceOptions {
    pub model_root: PathBuf,
    pub native_context_tokens: usize,
    /// Rejects unsupported checkpoints through `model_validator` (none
    /// supplied counts as unsupported).
    pub validate_model: bool,
    pub model_validator: Option<ModelValidator>,
    pub model_id: String,
    /// The transcription runtime. Without one, `runtime_factory` creates the
    /// native runtime after validation, as C++ does; this crate has no GPU
    /// code, so one of the two is required. The native runtime admits work
    /// per audio chunk itself, so other requests' frontend work proceeds
    /// while the GPU is busy; an injected runner goes through the service's
    /// FIFO gate.
    pub runner: Option<AsrRunner>,
    pub runtime_factory: Option<AsrRuntimeFactory>,
}

impl Default for AsrServiceOptions {
    fn default() -> Self {
        Self {
            model_root: PathBuf::new(),
            native_context_tokens: 1024,
            validate_model: true,
            model_validator: None,
            model_id: String::new(),
            runner: None,
            runtime_factory: None,
        }
    }
}

pub struct AsrService {
    model_id: String,
    runner: Option<AsrRunner>,
    /// The runner is the native runtime, which admits per chunk.
    native: bool,
    initialization_error: String,
    transcription_gate: CancellableGate,
    ready: bool,
}

impl AsrService {
    pub fn new(options: AsrServiceOptions) -> Self {
        let AsrServiceOptions {
            model_root,
            native_context_tokens,
            validate_model,
            model_validator,
            mut model_id,
            runner,
            runtime_factory,
        } = options;
        let unready = |model_id: String, error: &str| Self {
            model_id,
            runner: None,
            native: false,
            initialization_error: error.into(),
            transcription_gate: CancellableGate::default(),
            ready: false,
        };
        if validate_model && !model_validator.is_some_and(|valid| valid(&model_root)) {
            return unready(
                model_id,
                "Qwen3-ASR service requires the supported 1.7B model",
            );
        }
        if model_id.is_empty() {
            model_id = "qwen3-asr-1.7b".into();
        }
        let native = runner.is_none();
        let runner = match (runner, runtime_factory) {
            (Some(runner), _) => runner,
            (None, Some(factory)) => match factory(&model_root, native_context_tokens) {
                Ok(runner) => runner,
                Err(error) => return unready(model_id, &error),
            },
            (None, None) => {
                return unready(
                    model_id,
                    "Qwen3-ASR requires an injected transcription runtime in this build",
                );
            }
        };
        Self {
            model_id,
            runner: Some(runner),
            native,
            initialization_error: String::new(),
            transcription_gate: CancellableGate::default(),
            ready: true,
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

    /// Transcribes after FIFO admission (cancellable while queued), which
    /// the native runtime does per chunk itself. `on_text` receives the
    /// cumulative transcript.
    pub fn transcribe(
        &self,
        request: &TranscriptionRequest,
        on_text: Option<TextSink<'_>>,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<TranscriptionResult, String> {
        let Some(runner) = self.runner.as_ref().filter(|_| self.ready) else {
            return Err(self.initialization_error.clone());
        };
        if self.native {
            return runner(request, on_text, is_cancelled);
        }
        let _lease = self.transcription_gate.acquire(is_cancelled)?;
        runner(request, on_text, is_cancelled)
    }
}
