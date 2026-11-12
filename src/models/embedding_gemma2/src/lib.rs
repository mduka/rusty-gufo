//! EmbeddingGemma 2 (`google/embeddinggemma-2`): BF16 safetensors text,
//! image, video and audio embeddings on gfx1151, mean-pooled and
//! L2-normalised.
//!
//! | Module | Content |
//! | --- | --- |
//! | `config` | `config.json`, the supported geometry |
//! | `loader` | `model.safetensors` header, the backbone and tower tensors |
//! | `tokenizer` | the Gemma BPE tokenizer of `tokenizer.json` |
//! | `image_processor` | `Gemma4ImageProcessor`: resize and 16x16 patches |
//! | `video_processor` | sampled video frames as image patches (FFmpeg) |
//! | `audio_decoder` | encoded audio to mono 16-kHz samples |
//! | `audio_processor` | the Gemma4 log-mel audio features |
//! | `gemm` | BF16 projections, hipBLASLt plans timed per shape and row bucket |
//! | `batch_scheduler` | shared device batches of concurrent callers |
//! | `vision`, `audio` | the vision and audio towers |
//! | `encoder` | the GPU encoder |

#![forbid(unsafe_code)]

pub mod audio;
pub mod audio_decoder;
pub mod audio_processor;
pub mod batch_scheduler;
pub mod config;
pub mod encoder;
pub mod gemm;
pub mod image_processor;
pub mod loader;
pub mod tokenizer;
pub mod video_processor;
pub mod vision;

pub use batch_scheduler::{CancelCheck, EncoderSequence, SequenceImage};
pub use config::ModelConfig;
pub use encoder::{EncoderRuntime, MAX_SEQUENCE_TOKENS};
pub use loader::looks_like_embedding_gemma2;
pub use tokenizer::Tokenizer;
