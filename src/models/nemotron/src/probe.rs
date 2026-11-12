//! Whether a file is a Nemotron 3.5 Bolt trunk (`probe.cpp`).

use std::path::Path;

use gufo_core::gguf::{Gguf, read_header};

use crate::config::Config;

/// A Nemotron 3.5 Bolt trunk: `nemotron_h_moe` metadata that [`Config`]
/// accepts (`ProbeBoltGguf`).
pub fn probe_bolt_gguf(gguf: &Gguf<'_>) -> bool {
    gguf.string("general.architecture") == Some(b"nemotron_h_moe".as_slice())
        && Config::from_gguf(gguf).is_ok()
}

/// `model_root` is a Nemotron 3.5 Bolt trunk GGUF (`ProbeBolt`); only its
/// header is read.
pub fn probe_bolt(model_root: &Path) -> bool {
    if !model_root.is_file() {
        return false;
    }
    let Ok((header, len)) = read_header(model_root) else {
        return false;
    };
    Gguf::parse_prefix(&header, len).is_ok_and(|gguf| probe_bolt_gguf(&gguf))
}
