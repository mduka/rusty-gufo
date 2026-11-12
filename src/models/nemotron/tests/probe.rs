//! The C++ extension's `probe_test.cpp`: the released trunk, drafter and
//! projector bind, are claimed only by their own probes, and the trunk's
//! NVFP4 tensors are sized. Reads file headers and a few payloads only.
//! cargo test -p gufo-nemotron --features models --test probe

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]
// NVIDIA's pixel statistics, digit for digit as the processor states them.
#![allow(clippy::excessive_precision)]

mod common;

use std::path::Path;
use std::sync::Arc;

use gufo_formats::GgmlType;
use gufo_hip::MappedFile;
use gufo_nemotron::config::Config;
use gufo_nemotron::probe::{probe_bolt, probe_bolt_gguf};
use gufo_nemotron::weights::{DrafterWeights, GgufShard, ModelWeights, VisionWeights};

fn open(path: &Path) -> Arc<MappedFile> {
    MappedFile::open(path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()))
}

/// The reader sizes NVFP4 itself (whole 64-value blocks within the file's
/// bytes); returns how many such tensors there are.
fn count_nvfp4(shard: &GgufShard<'_>) -> usize {
    shard
        .gguf
        .tensors()
        .iter()
        .filter(|t| t.ty == GgmlType::Nvfp4)
        .inspect(|t| {
            assert_eq!(t.elements() % 64, 0, "{} is not whole NVFP4 blocks", t.name);
            assert_eq!(
                t.bytes as u64,
                t.elements() / 64 * 36,
                "{} NVFP4 size",
                t.name
            );
        })
        .count()
}

fn check_trunk(path: &Path) -> Config {
    let file = open(path);
    let shard = GgufShard::parse(file.bytes()).unwrap();
    assert!(
        probe_bolt_gguf(&shard.gguf),
        "trunk not claimed by metadata"
    );
    assert!(probe_bolt(path), "trunk not claimed by path");
    assert!(count_nvfp4(&shard) > 0, "trunk has no NVFP4 tensors");
    let shards = [shard];
    let weights = ModelWeights::bind(&shards).unwrap_or_else(|e| panic!("trunk weights: {e}"));
    assert!(
        VisionWeights::bind(&shards, &weights.config).is_err(),
        "trunk bound as a vision projector"
    );
    weights.config
}

fn check_drafter(path: &Path, target: &Config) {
    if !path.is_file() {
        println!("absent, not checked: {}", path.display());
        return;
    }
    let file = open(path);
    let shard = GgufShard::parse(file.bytes()).unwrap();
    count_nvfp4(&shard);
    DrafterWeights::bind(&[shard], target).unwrap_or_else(|e| panic!("drafter weights: {e}"));
}

fn check_projector(path: &Path, trunk: &Config) {
    if !path.is_file() {
        println!("absent, not checked: {}", path.display());
        return;
    }
    let file = open(path);
    let shards = [GgufShard::parse(file.bytes()).unwrap()];
    let weights =
        VisionWeights::bind(&shards, trunk).unwrap_or_else(|e| panic!("projector weights: {e}"));
    let c = &weights.config;
    assert!(
        c.head_dim() == 80
            && c.min_patches == 1024
            && c.max_patches == 13312
            && c.projection == trunk.hidden_size,
        "unexpected projector geometry"
    );
    assert!(
        (c.mean[0] - 0.48145466).abs() < 1e-7 && (c.deviation[2] - 0.27577711).abs() < 1e-7,
        "unexpected pixel statistics"
    );
    // The device keeps the F32 patch projection as BF16, exactly.
    for raw in weights.patch.bytes(&shards).as_chunks::<4>().0 {
        assert_eq!(
            u32::from_le_bytes(*raw) & 0xFFFF,
            0,
            "patch projection is not BF16-exact"
        );
    }
    // The projector must match its trunk's width.
    let mut narrow = trunk.clone();
    narrow.hidden_size /= 2;
    let error = VisionWeights::bind(&shards, &narrow).unwrap_err();
    assert!(
        error.contains("width"),
        "projector bound to a narrower trunk: {error}"
    );
}

fn check_rejected(path: &Path) {
    if !path.is_file() {
        println!("absent, not checked: {}", path.display());
        return;
    }
    let file = open(path);
    let shard = GgufShard::parse(file.bytes()).unwrap();
    assert!(
        !probe_bolt_gguf(&shard.gguf),
        "{} claimed by metadata",
        path.display()
    );
    assert!(!probe_bolt(path), "{} claimed by path", path.display());
}

#[test]
fn released_files_bind_and_probe() {
    let trunk = common::trunk_path();
    let config = check_trunk(&trunk);
    let dir = trunk.parent().unwrap();
    let stem = trunk.file_stem().unwrap().to_string_lossy();
    let name = trunk.file_name().unwrap().to_string_lossy();
    let drafter = dir.join(format!("{stem}-LiLiCorr.gguf"));
    check_drafter(&drafter, &config);
    check_rejected(&drafter);
    let projector = dir.join(format!("mmproj-{name}"));
    check_projector(&projector, &config);
    check_rejected(&projector);
    assert!(!probe_bolt(dir), "directory claimed");
}
