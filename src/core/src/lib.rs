//! Model-independent host code: GGUF files, BPE tokenizers, text utilities,
//! sampling controls and the text-generation interface that serving drives.
//! No GPU dependency.
//!
//! Unicode data comes from `unicode-normalization` and `unicode-properties`
//! (Unicode 17, as ICU 78, which the C++ tokenizers use).

#![forbid(unsafe_code)]

pub mod bpe;
pub mod cancellable_gate;
pub mod gguf;
pub mod gguf_identity;
pub mod image;
pub mod json;
pub mod json_constraint;
pub mod prompt_lookup;
pub mod qwen_tokenizer;
pub mod sampling;
pub mod text;
pub mod unicode;
pub mod utf8;

pub use gufo_formats::GgmlType;
