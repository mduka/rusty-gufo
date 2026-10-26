//! Binds the local artifacts' tensors (headers only):
//! cargo test -p gufo-qwen38-flash-next --features models

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

mod common;

use std::path::PathBuf;

use common::{mtp_path, trunk_path};
use gufo_formats::GgmlType;
use gufo_qwen38_flash_next::weights::{GgufFiles, MixerKind, ModelTensors, MtpTensors};

#[test]
fn the_split_trunk_and_the_mtp_sidecar_bind() {
    // A later shard opens the whole set.
    let trunk = PathBuf::from(
        trunk_path()
            .to_string_lossy()
            .replace("-00001-of-", "-00002-of-"),
    );
    if !trunk.exists() {
        eprintln!("skipped: {} not found", trunk.display());
        return;
    }
    let files = GgufFiles::open(&trunk).unwrap();
    assert_eq!(files.paths.len(), 4);
    assert!(
        files.paths[0]
            .to_str()
            .unwrap()
            .ends_with("-00001-of-00004.gguf")
    );
    let shards = files.parse().unwrap();
    let model = ModelTensors::bind(&shards).unwrap();
    let c = &model.config;
    assert_eq!(
        (c.num_layers, c.vocab_size, c.hc_dim()),
        (48, 248_320, 10_240)
    );
    assert_eq!(c.ple_layer, Some(1));
    assert_eq!(model.layers.len(), 48);
    let linear = model
        .layers
        .iter()
        .filter(|l| matches!(l.mixer, MixerKind::Linear(_)))
        .count();
    assert_eq!(linear, 36);
    assert_eq!(model.token_embd.ty, GgmlType::Q8_0);
    assert_eq!(model.ple_table.as_ref().unwrap().ty, GgmlType::Iq4Nl);
    assert!(model.layers[1].ple.is_some() && model.layers[0].ple.is_none());
    // Tensors live in the three data shards.
    assert!(model.layers.iter().any(|l| l.moe.gate_exps.shard == 3));

    let mtp_files = GgufFiles::open(&mtp_path()).unwrap();
    let mtp = MtpTensors::bind(&mtp_files.parse().unwrap(), c).unwrap();
    assert!(mtp.block.nextn.is_some());
    assert_eq!(mtp.block.moe.gate_exps.ty, GgmlType::Q8_0);
}
