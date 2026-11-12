//! Google's Gemma 4 (`gguf:gemma4`): the 31B dense model, its QAT variant and
//! the 26B-A4B mixture of experts, with `gemma4-assistant` MTP drafting,
//! prompt lookup and the BF16 vision sidecar. Ported from Francesco Moca's
//! [Gemma 4 extension for Gufo](https://github.com/franzmoca/gemma4-gufo-extension)
//! (`a31ce42`); comments that cite a C++ file mean that tree.

#![forbid(unsafe_code)]

pub mod chat_template;
pub mod config;
pub mod cycle_trace;
pub mod decode;
pub mod dequant;
pub mod device_model;
pub mod draft_policy;
pub mod draft_vocab;
pub mod engine;
pub mod executor;
pub mod mtp_sampling;
pub mod probe;
pub mod proposal_sampler;
pub mod provider_options;
pub mod reference;
pub mod runner;
pub mod sibling_verification;
pub mod snapshot;
pub mod text_contract;
pub mod tokenizer;
pub mod vision_encoder;
pub mod vision_prompt;
pub mod vision_reference;
pub mod vision_resize;
pub mod vision_weights;
pub mod weights;
