//! The cases of the C++ extension's `safetensors_test.cpp`: sample files,
//! and the released checkpoint's two weight files with `--features models`.

#![allow(clippy::unwrap_used)]

mod config_samples;

use std::path::{Path, PathBuf};

use config_samples::temp_path;
use gufo_omnivoice::safetensors::{DType, SafetensorsFile};

/// A safetensors file with this header and payload; removed when dropped.
struct SampleFile(PathBuf);

impl SampleFile {
    fn new(name: &str, header: &str, payload: &[u8]) -> Self {
        let path = temp_path("safetensors_test", name);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for SampleFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn well_formed_file() {
    // A 2x3 F32 tensor (24 bytes) and a 4-element BF16 tensor (8 bytes).
    let mut payload: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    payload.extend(0u8..8);
    let file = SampleFile::new(
        "valid.safetensors",
        concat!(
            r#"{"__metadata__":{"format":"pt"},"#,
            r#""b":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]},"#,
            r#""a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#
        ),
        &payload,
    );
    let opened = SafetensorsFile::open(file.path()).unwrap();
    assert_eq!(opened.tensors().len(), 2, "tensor count");
    let a = opened.expect("a", DType::F32, &[2, 3]).unwrap();
    let values = opened.f32_values(a).unwrap();
    assert!(
        values[0] == 1.0 && values[5] == 6.0 && a.elements() == 6,
        "a's payload"
    );
    let b = opened.find("b").unwrap();
    assert!(
        b.dtype == DType::Bf16 && b.bytes == 8 && opened.bytes(b)[7] == 7,
        "b's payload"
    );
    let error = opened.expect("a", DType::F32, &[3, 2]).unwrap_err();
    assert!(
        error.contains("expected F32 [3, 2]"),
        "a shape mismatch reported as: {error}"
    );
    let error = opened.expect("c", DType::F32, &[1]).unwrap_err();
    assert!(
        error.contains("lacks c"),
        "missing tensor reported as: {error}"
    );
    assert_eq!(b.offset - a.offset, 24, "offsets");
}

fn expect_rejected(name: &str, header: &str, payload_bytes: usize, reason: &str) {
    let file = SampleFile::new(
        &format!("{name}.safetensors"),
        header,
        &vec![0u8; payload_bytes],
    );
    let error = SafetensorsFile::open(file.path())
        .err()
        .unwrap_or_else(|| panic!("{name} accepted"));
    assert!(
        error.contains(reason),
        "{name} rejected for another reason: {error}"
    );
}

#[test]
fn malformed_files() {
    expect_rejected(
        "unknown_dtype",
        r#"{"a":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}"#,
        1,
        "unsupported dtype",
    );
    expect_rejected(
        "short_payload",
        r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
        8,
        "outside the payload",
    );
    expect_rejected(
        "size_mismatch",
        r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}}"#,
        16,
        "don't hold shape [3]",
    );
    expect_rejected(
        "overlap",
        concat!(
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"#,
            r#""b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#
        ),
        12,
        "overlaps",
    );
    expect_rejected("not_json", "{\"a\":", 0, "header");
    expect_rejected("empty", "{}", 0, "no tensors");
}

/// The released checkpoint's two weight files.
#[cfg(feature = "models")]
#[test]
fn released_files() {
    let root = config_samples::model_root();
    let expect = |file: &SafetensorsFile, name: &str, shape: &[usize]| {
        file.expect(name, DType::F32, shape).unwrap();
    };
    let model = SafetensorsFile::open(&root.join("model.safetensors")).unwrap();
    assert_eq!(model.tensors().len(), 313, "generator tensor count");
    expect(&model, "llm.embed_tokens.weight", &[151676, 1024]);
    expect(&model, "audio_embeddings.weight", &[8200, 1024]);
    expect(&model, "audio_heads.weight", &[8200, 1024]);
    expect(
        &model,
        "llm.layers.27.self_attn.q_proj.weight",
        &[2048, 1024],
    );
    expect(&model, "llm.layers.0.self_attn.k_norm.weight", &[128]);
    let codec = SafetensorsFile::open(&root.join("audio_tokenizer/model.safetensors")).unwrap();
    assert_eq!(codec.tensors().len(), 527, "audio tokenizer tensor count");
    expect(&codec, "acoustic_decoder.conv1.weight", &[1024, 256, 7]);
    expect(&codec, "quantizer.quantizers.7.codebook.embed", &[1024, 64]);
    expect(
        &codec,
        "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
        &[768, 48, 128],
    );
}
