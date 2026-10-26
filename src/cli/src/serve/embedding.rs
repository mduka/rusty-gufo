//! `gufo serve embedding`: EmbeddingGemma 2 on the Rust engine behind the
//! embedding service.

use std::path::Path;

use gufo_embedding_gemma2::{EncoderRuntime, MAX_SEQUENCE_TOKENS, Tokenizer};
use gufo_serve::embedding_service::{
    EmbeddingEngine, EmbeddingService, EmbeddingServiceOptions, MAX_INPUT_TOKENS,
};

use super::ModelLoadLog;

// The service's input limit is the model's context window.
const _: () = assert!(MAX_INPUT_TOKENS == MAX_SEQUENCE_TOKENS);

/// The tokenizer and the GPU encoder, which hands out device turns one
/// batch at a time itself.
struct Engine {
    tokenizer: Tokenizer,
    encoder: EncoderRuntime,
}

impl EmbeddingEngine for Engine {
    fn max_token_bytes(&self) -> usize {
        self.tokenizer.max_token_bytes()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text.as_bytes(), true)
    }

    fn embed(
        &self,
        sequences: &[Vec<u32>],
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Vec<f32>>, String> {
        self.encoder.embed(sequences, Some(is_cancelled))
    }

    fn device_lost(&self) -> bool {
        self.encoder.device_lost()
    }
}

/// Loads the tokenizer, then uploads and tunes the encoder.
fn engine(model_root: &Path) -> Result<Box<dyn EmbeddingEngine>, String> {
    let tokenizer = Tokenizer::load(model_root)
        .map_err(|error| format!("embedding tokenizer failed to load: {error}"))?;
    let encoder = EncoderRuntime::create(model_root)
        .map_err(|error| format!("embedding encoder failed to load: {error}"))?;
    Ok(Box::new(Engine { tokenizer, encoder }))
}

/// The embedding service of `--model`, or the exit status after printing
/// why it cannot serve.
pub(super) fn load(model: &Path, name: &str) -> Result<EmbeddingService, u8> {
    let mut load_log = ModelLoadLog::new("embedding_gemma2", model);
    let service = EmbeddingService::new(EmbeddingServiceOptions {
        model_root: model.to_path_buf(),
        validate_model: true,
        model_validator: Some(Box::new(gufo_embedding_gemma2::looks_like_embedding_gemma2)),
        model_id: name.to_owned(),
        runner: None,
        engine_factory: Some(Box::new(engine)),
    });
    if !service.ready() {
        eprintln!(
            "Error enabling EmbeddingGemma 2 service: {}",
            service.initialization_error()
        );
        return Err(1);
    }
    load_log.complete(
        &format!(
            "model={} context_tokens={MAX_INPUT_TOKENS}",
            service.model_id()
        ),
        true,
    );
    Ok(service)
}
