//! Which paths hold a VoxCPM2 checkpoint this build implements
//! (`probe.hpp`, `probe.cpp`).

use std::path::Path;

use crate::config::{is_supported, load_config, parse_config};
use crate::model_files::ModelFiles;

/// `model_root` is a directory holding a supported VoxCPM2 checkpoint in the
/// openbmb Hugging Face layout (`config.json` with architecture "voxcpm2"),
/// or an audio.cpp VoxCPM2 GGUF (`ProbeVoxCpm2`).
pub fn probe_voxcpm2(model_root: &Path) -> bool {
    if ModelFiles::is_voxcpm2_gguf(model_root) {
        let Ok(files) = ModelFiles::open(model_root) else {
            return false;
        };
        return parse_config(files.config()).is_ok_and(|config| is_supported(&config).is_ok());
    }
    if !model_root.is_dir() || !model_root.join("config.json").is_file() {
        return false;
    }
    load_config(model_root).is_ok_and(|config| is_supported(&config).is_ok())
}

/// Checks that the configuration is supported and the weight and tokenizer
/// files exist (`ResolveVoxCpm2Artifacts`).
pub fn resolve_voxcpm2_artifacts(model_root: &Path) -> Result<(), String> {
    if ModelFiles::is_voxcpm2_gguf(model_root) {
        let files = ModelFiles::open(model_root)?;
        let config = parse_config(files.config())?;
        is_supported(&config)?;
        if files.model("base_lm.embed_tokens.weight").is_none()
            || files.vae("decoder.model.0.weight_v").is_none()
        {
            return Err("the GGUF lacks VoxCPM2's generator or AudioVAE tensors".into());
        }
        return Ok(());
    }
    let config = load_config(model_root)?;
    is_supported(&config)?;
    for file in ["model.safetensors", "tokenizer.json", "audiovae.pth"] {
        if !model_root.join(file).is_file() {
            return Err(format!("VoxCPM2 model directory lacks {file}"));
        }
    }
    Ok(())
}
