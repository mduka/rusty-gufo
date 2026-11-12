//! Model detection (`probe.cpp`): a `gemma4` target GGUF (31B, 31B QAT or
//! 26B-A4B). The `gemma4-assistant` MTP drafter and the vision sidecar are
//! artifacts of their target, not models of their own, so they do not match.

use std::path::Path;

use gufo_core::gguf::{Gguf, read_header};

/// `ProbeGemma4Gguf`.
pub fn probe_gemma4_gguf(gguf: &Gguf<'_>) -> bool {
    gguf.string("general.architecture") == Some(b"gemma4".as_slice())
}

/// `model_root` is a target GGUF file (`ProbeGemma4`).
pub fn probe_gemma4(model_root: &Path) -> bool {
    if model_root.extension().is_none_or(|e| e != "gguf") || !model_root.is_file() {
        return false;
    }
    read_header(model_root)
        .ok()
        .and_then(|(header, len)| {
            Gguf::parse_prefix(&header, len)
                .ok()
                .map(|gguf| probe_gemma4_gguf(&gguf))
        })
        .unwrap_or(false)
}
