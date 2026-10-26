//! Prompt lookup: MTP proposals copied from the session's own committed
//! tokens. The index is shared with the dense Qwen DFlash2 draft
//! ([`gufo_core::prompt_lookup`]); Flash-Next keys on `i32` token IDs and
//! admits every source.

pub use gufo_core::prompt_lookup::*;
