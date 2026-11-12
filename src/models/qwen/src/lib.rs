//! What the Qwen language models share across their architectures (`qwen35`
//! in `gufo-qwen35`, Flash-Next's `qwen4exp`): the chat template, the vision
//! encoder with its image prompts, and through `gufo-qwen-kernels` the HIP
//! kernels of upstream's `src/models/qwen` (sampling, vision and the `qwen35`
//! layers). The audio and image models named Qwen keep their own crates.

#![forbid(unsafe_code)]

pub mod chat_template;
pub mod vision;
