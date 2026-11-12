//! Shared helpers of the Nemotron tests: fixtures and the released files.

#![allow(dead_code)]

use std::path::PathBuf;

/// `tests/fixtures/<name>`.
pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// A golden fixture's JSON.
pub fn golden(name: &str) -> gufo_core::json::Value {
    let text = std::fs::read(fixture(name)).unwrap_or_else(|error| panic!("{name}: {error}"));
    gufo_core::json::parse(&text).unwrap_or_else(|error| panic!("{name}: {error}"))
}

/// `GUFO_NEMOTRON_BOLT_GGUF`: the released trunk. The model tests fail
/// rather than skip without it, since a missing model is not a pass.
pub fn trunk_path() -> PathBuf {
    std::env::var_os("GUFO_NEMOTRON_BOLT_GGUF")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("GUFO_NEMOTRON_BOLT_GGUF must name a Nemotron 3.5 Bolt trunk GGUF")
        })
}
