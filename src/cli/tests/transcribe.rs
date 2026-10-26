//! `gufo transcribe` as a process: cargo test -p gufo [--features gpu]

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::{Command, Output};

fn gufo(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gufo"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn usage_errors_exit_with_two() {
    for args in [
        &["transcribe"][..],
        &["transcribe", "--model", "m"],
        &["transcribe", "--model", "m", "a.wav", "b.wav"],
        &["transcribe", "--model", "m", "a.wav", "--max-tokens", "0"],
        &["transcribe", "--model", "m", "a.wav", "--format", "yaml"],
        &["transcribe", "--model", "m", "a.wav", "--bogus", "1"],
    ] {
        let output = gufo(args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).starts_with("Error:"),
            "{args:?}"
        );
    }
    let help = gufo(&["transcribe", "--help"]);
    assert!(help.status.success());
    // The C++ help text (ArgParser appends its own usage suffix).
    let text = String::from_utf8_lossy(&help.stdout).into_owned();
    assert!(text.starts_with("Usage: gufo transcribe [OPTIONS] [WAV] [OPTIONS] [ARGS...]"));
    assert!(text.contains("-n, --max-tokens <N>"), "{text}");
    let missing = gufo(&["transcribe", "--model", "m", "/nonexistent.wav"]);
    assert_eq!(missing.status.code(), Some(1));
}

/// `gufo` alone prints help, and `serve` reports a model it cannot open.
#[test]
fn top_level_help_and_serve_without_a_model() {
    let help = gufo(&[]);
    assert_eq!(help.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&help.stdout).starts_with("Usage: "));
    let serve = gufo(&["serve", "--model", "/nonexistent.gguf"]);
    assert_eq!(serve.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&serve.stderr);
    assert!(
        stderr.contains("Error loading model '/nonexistent.gguf': /nonexistent.gguf: No such file"),
        "{stderr}"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn json_report_carries_the_reference_token_ids() {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap());
    let model = std::env::var_os("QWEN3_ASR_MODEL_ROOT").map_or_else(
        || {
            home.join(
                ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/\
                 7278e1e70fe206f11671096ffdd38061171dd6e5",
            )
        },
        PathBuf::from,
    );
    let audio = home.join(".cache/gufo/fixtures/qwen3-asr-en.wav");
    let output = gufo(&[
        "transcribe",
        "--model",
        model.to_str().unwrap(),
        "--audio",
        audio.to_str().unwrap(),
        "--repeat",
        "2",
        "--format",
        "json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../models/qwen3_asr/tests/fixtures/qwen3-asr-en.json"
    ))
    .unwrap();
    assert_eq!(report["schema"], "gufo.qwen3-asr.transcription.v1");
    assert_eq!(report["generated_ids"], fixture["generated_ids"]);
    assert_eq!(report["text"], fixture["text"]);
    assert_eq!(report["generated_tokens"], 49);
    assert_eq!(report["runs"].as_array().unwrap().len(), 2);
    assert!(report["timings"]["load_ms"].as_f64().unwrap() > 0.0);
    assert_eq!(report["runs"][0]["load_ms"], 0.0);
    let keys: Vec<&str> = report
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        &keys[..4],
        ["schema", "backend", "language", "text"],
        "C++ key order"
    );
}
