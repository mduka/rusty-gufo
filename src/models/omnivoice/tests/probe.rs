//! The cases of the C++ extension's `probe_test.cpp`: sample directories,
//! and the released checkpoint with `--features models`.

#![allow(clippy::unwrap_used)]

mod config_samples;

use std::path::{Path, PathBuf};

use config_samples::{CODEC_CONFIG, MODEL_CONFIG, PREPROCESSOR_CONFIG, replace, temp_path};
use gufo_omnivoice::probe::{probe_omnivoice, resolve_omnivoice_artifacts};

/// A fresh directory with OmniVoice's three configuration files.
struct SampleDirectory(PathBuf);

impl SampleDirectory {
    fn new(name: &str, model_config: &str) -> Self {
        let root = temp_path("probe_test", name);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("audio_tokenizer")).unwrap();
        std::fs::write(root.join("config.json"), model_config).unwrap();
        std::fs::write(root.join("audio_tokenizer/config.json"), CODEC_CONFIG).unwrap();
        std::fs::write(
            root.join("audio_tokenizer/preprocessor_config.json"),
            PREPROCESSOR_CONFIG,
        )
        .unwrap();
        Self(root)
    }

    fn root(&self) -> &Path {
        &self.0
    }
}

impl Drop for SampleDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn sample_directories() {
    {
        let directory = SampleDirectory::new("released", MODEL_CONFIG);
        assert!(
            probe_omnivoice(directory.root()),
            "probe rejects the released configuration"
        );
        let error = resolve_omnivoice_artifacts(directory.root()).unwrap_err();
        assert!(
            error.contains("model.safetensors"),
            "resolve accepts a directory without weights: {error}"
        );
        assert!(
            !probe_omnivoice(&directory.root().join("config.json")),
            "probe accepts a file"
        );
    }
    {
        let directory = SampleDirectory::new(
            "other",
            &replace(
                MODEL_CONFIG,
                r#""model_type": "omnivoice""#,
                r#""model_type": "qwen3_tts""#,
            ),
        );
        assert!(
            !probe_omnivoice(directory.root()),
            "probe accepts another model type"
        );
    }
    assert!(
        !probe_omnivoice(Path::new("/nonexistent/omnivoice")),
        "probe accepts a missing directory"
    );
}

/// The released checkpoint named by `GUFO_OMNIVOICE_MODEL`.
#[cfg(feature = "models")]
#[test]
fn released_directory() {
    let root = config_samples::model_root();
    assert!(probe_omnivoice(&root), "probe rejects GUFO_OMNIVOICE_MODEL");
    resolve_omnivoice_artifacts(&root).unwrap();
}
