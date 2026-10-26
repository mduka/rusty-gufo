//! EmbeddingGemma 2 (`google/embeddinggemma-2`): BF16 safetensors text
//! embeddings on gfx1151, mean-pooled and L2-normalised.
//!
//! | Module | Content |
//! | --- | --- |
//! | `config` | `config.json`, the supported geometry |
//! | `loader` | `model.safetensors` header, the text-backbone tensors |
//! | `tokenizer` | the Gemma BPE tokenizer of `tokenizer.json` |
//! | `gemm` | BF16 projections, hipBLASLt plans timed per shape and row bucket |
//! | `encoder` | the GPU encoder with arrival-order device turns |

#![forbid(unsafe_code)]

pub mod config;
pub mod encoder;
pub mod gemm;
pub mod loader;
pub mod tokenizer;

pub use config::ModelConfig;
pub use encoder::{EncoderRuntime, MAX_SEQUENCE_TOKENS};
pub use loader::looks_like_embedding_gemma2;
pub use tokenizer::Tokenizer;
