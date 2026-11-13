//! Recognizing a model directory (`probe.hpp`, `probe.cpp`).

use std::path::Path;

use gufo_core::json;

use crate::config::{is_supported, load_config};

fn has_architecture(root: &Path) -> bool {
    let Ok(text) = std::fs::read(root.join("config.json")) else {
        return false;
    };
    let Ok(config) = json::parse(&text) else {
        return false;
    };
    config
        .find("architectures")
        .filter(|architectures| architectures.is_array())
        .is_some_and(|architectures| {
            architectures
                .items()
                .iter()
                .any(|item| item.is_string() && item.str() == "Nemotron3_5AsrForRNNT")
        })
}

/// `root` is a Hugging Face directory of Nemotron 3.5 ASR: config.json with
/// model_type "nemotron3_5_asr" (architecture `Nemotron3_5AsrForRNNT`),
/// processor_config.json and a configuration the kernels support
/// (`ProbeNemotronAsr`).
pub fn probe(root: &Path) -> bool {
    if !root.is_dir()
        || !root.join("config.json").is_file()
        || !root.join("processor_config.json").is_file()
    {
        return false;
    }
    if !has_architecture(root) {
        return false;
    }
    load_config(root).is_ok_and(|config| is_supported(&config).is_ok())
}

/// Checks that the configuration is supported and the weight and tokenizer
/// files exist (`ResolveNemotronAsrArtifacts`).
pub fn resolve_artifacts(root: &Path) -> Result<(), String> {
    let config = load_config(root)?;
    is_supported(&config)?;
    for file in ["model.safetensors", "tokenizer.json"] {
        if !root.join(file).is_file() {
            return Err(format!("Nemotron ASR model directory lacks {file}"));
        }
    }
    Ok(())
}
