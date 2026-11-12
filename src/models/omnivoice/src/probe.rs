//! Recognizing an OmniVoice model directory (`probe.cpp`).

use std::path::Path;

use crate::config::{is_supported, load_config};

/// `model_root` is a directory holding a supported OmniVoice checkpoint's
/// configuration (Hugging Face layout: `config.json` with model_type
/// "omnivoice" plus `audio_tokenizer/`) (`ProbeOmniVoice`).
pub fn probe_omnivoice(model_root: &Path) -> bool {
    if !model_root.is_dir() || !model_root.join("config.json").is_file() {
        return false;
    }
    load_config(model_root).is_ok_and(|config| is_supported(&config).is_ok())
}

/// Checks that the configuration is supported and the weight and tokenizer
/// files exist (`ResolveOmniVoiceArtifacts`).
pub fn resolve_omnivoice_artifacts(model_root: &Path) -> Result<(), String> {
    let config = load_config(model_root)?;
    is_supported(&config)?;
    for file in [
        "model.safetensors",
        "tokenizer.json",
        "audio_tokenizer/model.safetensors",
    ] {
        if !model_root.join(file).is_file() {
            return Err(format!("OmniVoice model directory lacks {file}"));
        }
    }
    Ok(())
}
