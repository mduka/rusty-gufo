//! The cases of the C++ extension's `probe_test.cpp`: sample directories,
//! and the released checkpoint (and an audio.cpp GGUF, when
//! `GUFO_VOXCPM_GGUF` names one) with `--features models`.

#![allow(clippy::unwrap_used)]

mod config_samples;

use std::path::{Path, PathBuf};

use config_samples::{MODEL_CONFIG, temp_path};
use gufo_voxcpm::probe::{probe_voxcpm2, resolve_voxcpm2_artifacts};

/// A fresh directory with a VoxCPM2 configuration.
struct SampleDirectory(PathBuf);

impl SampleDirectory {
    fn new(name: &str, config: &str) -> Self {
        let root = temp_path("probe_test", name);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), config).unwrap();
        Self(root)
    }

    fn root(&self) -> &Path {
        &self.0
    }

    fn add(&self, file: &str) {
        std::fs::write(self.0.join(file), "").unwrap();
    }
}

impl Drop for SampleDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn samples() {
    {
        let directory = SampleDirectory::new("complete", MODEL_CONFIG);
        assert!(probe_voxcpm2(directory.root()), "probe accepts VoxCPM2");
        let error = resolve_voxcpm2_artifacts(directory.root()).unwrap_err();
        assert!(
            error.contains("model.safetensors"),
            "resolve names the missing weights: {error}"
        );
        directory.add("model.safetensors");
        directory.add("tokenizer.json");
        let error = resolve_voxcpm2_artifacts(directory.root()).unwrap_err();
        assert!(
            error.contains("audiovae.pth"),
            "resolve names the missing AudioVAE: {error}"
        );
        directory.add("audiovae.pth");
        resolve_voxcpm2_artifacts(directory.root()).unwrap();
    }
    {
        let config = MODEL_CONFIG.replacen("\"voxcpm2\"", "\"voxcpm\"", 1);
        let directory = SampleDirectory::new("voxcpm1", &config);
        assert!(!probe_voxcpm2(directory.root()), "probe rejects VoxCPM 1");
    }
    assert!(
        !probe_voxcpm2(Path::new("/nonexistent/voxcpm")),
        "probe rejects a missing directory"
    );
}

#[cfg(feature = "models")]
#[test]
fn released_model() {
    let root = config_samples::model_root();
    assert!(probe_voxcpm2(&root), "probe accepts the model");
    resolve_voxcpm2_artifacts(&root).unwrap();
    // audio.cpp's GGUFs, when given (bfloat16 and Q8_0), are accepted as
    // well.
    for gguf in config_samples::gguf_paths() {
        assert!(probe_voxcpm2(&gguf), "probe accepts {}", gguf.display());
        resolve_voxcpm2_artifacts(&gguf).unwrap();
    }
}
