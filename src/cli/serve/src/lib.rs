//! Text serving (`src/cli/serve` in C++): the exact-prefix continuation cache
//! over opaque model states, the restart-safe disk tier, the runner pool that
//! leases states to requests and the single-owner scheduler that drives them.
//! On top: the HTTP transport, the OpenAI-compatible and llama.cpp routes,
//! and the audio, image, video and embeddings APIs over injectable model
//! services.

#![forbid(unsafe_code)]

pub mod arbiter;
pub mod asr_model;
pub mod asr_service;
pub mod audio_asr_api;
pub mod audio_stream;
pub mod audio_tts_api;
pub mod audio_websocket;
pub mod backend;
pub mod continuation_cache;
pub mod disk_store;
pub mod embedding_service;
pub mod embeddings_api;
pub mod generation_metrics;
pub mod gpu_share;
pub mod http;
pub mod image_api;
pub mod image_url;
pub mod logging;
pub mod metrics;
pub mod multipart;
pub mod openai_chat;
pub mod pool;
pub mod publication_wait;
pub mod quote_tracker;
pub mod response_format;
pub mod sampling_request;
pub mod scheduled_backend;
pub mod scheduler;
pub mod stop_sequences;
pub mod tts_model;
pub mod tts_service;
pub mod util;
pub mod video_api;
pub mod video_jobs;
pub mod video_model;
