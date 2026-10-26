//! Batched decoding against isolated execution: two sessions at different
//! positions decode and verify chains together, and every logit row equals
//! the session's own forward bit for bit (shared projections, private
//! state, the batched Gated DeltaNet):
//! cargo test -p gufo-qwen38-flash-next --features gpu --test batch

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::trunk_path;
use gufo_hip::{MappedFile, Stream};
use gufo_qwen38_flash_next::batch::BatchItem;
use gufo_qwen38_flash_next::device::DeviceModel;
use gufo_qwen38_flash_next::executor::{Executor, ForwardMode, Options, SessionMode};
use gufo_qwen38_flash_next::ngram::NgramTable;
use gufo_qwen38_flash_next::weights::{GgufFiles, ModelTensors};

fn prompt() -> Vec<i32> {
    let path = std::env::var_os("GUFO_FLASH_NEXT_FIXTURES")
        .map_or_else(
            || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/flash-next"),
            PathBuf::from,
        )
        .join("prompt50.bin");
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b) as i32)
        .collect()
}

#[test]
fn batched_forwards_match_isolated_ones() {
    let files = GgufFiles::open(&trunk_path()).unwrap();
    let model = ModelTensors::bind(&files.parse().unwrap()).unwrap();
    let mapped: Vec<Arc<MappedFile>> = files
        .paths
        .iter()
        .map(|p| MappedFile::open(p).unwrap())
        .collect();
    let stream = Stream::new().unwrap();
    let device = DeviceModel::upload(&stream, &model, &mapped, None).unwrap();
    let table = model.ple_table.as_ref().unwrap();
    let shard = std::fs::File::open(&files.paths[table.shard]).unwrap();
    let ngram = NgramTable::open(
        &shard,
        table.offset as u64,
        table.rows,
        model.config.ple_head_dim,
        table.ty,
    )
    .unwrap();
    let mut ex = Executor::new(
        Arc::new(device),
        Some(Arc::new(ngram)),
        Options {
            max_batch: 2048,
            max_logit_rows: 8,
            max_speculative: 8,
            max_context: 0,
        },
    )
    .unwrap();
    let vocab = ex.config().vocab_size as usize;
    let prompt = prompt();
    let mut a = ex
        .create_session(SessionMode::Autoregressive, 4096)
        .unwrap();
    let mut b = ex
        .create_session(SessionMode::Autoregressive, 4096)
        .unwrap();
    let mut last = vec![0f32; vocab];
    ex.forward(&mut a, &prompt, 1, Some(&mut last), ForwardMode::Prefill)
        .unwrap();
    ex.forward(
        &mut b,
        &prompt[..43],
        1,
        Some(&mut last),
        ForwardMode::Prefill,
    )
    .unwrap();
    // Each case runs the chains alone, restores both contexts, runs them
    // together and restores again.
    for (label, na, nb, spec) in [
        ("1+1 decode", 1usize, 1usize, false),
        ("3+2 verify", 3, 2, true),
    ] {
        let ta: Vec<i32> = (0..na as i32).map(|i| 1000 + i).collect();
        let tb: Vec<i32> = (0..nb as i32).map(|i| 2000 + i).collect();
        let snap_a = ex.snapshot_bytes(&a, 0);
        let mut pa = vec![0u8; snap_a as usize];
        ex.save_snapshot(&a, 0, &mut pa).unwrap();
        let snap_b = ex.snapshot_bytes(&b, 0);
        let mut pb = vec![0u8; snap_b as usize];
        ex.save_snapshot(&b, 0, &mut pb).unwrap();
        let mode = if spec {
            ForwardMode::Verify
        } else {
            ForwardMode::Decode
        };
        let mut single_a = vec![0f32; na * vocab];
        let mut single_b = vec![0f32; nb * vocab];
        ex.forward(&mut a, &ta, na as u32, Some(&mut single_a), mode)
            .unwrap();
        ex.forward(&mut b, &tb, nb as u32, Some(&mut single_b), mode)
            .unwrap();
        if spec {
            ex.rollback(&mut a, na as u32, None).unwrap();
            ex.rollback(&mut b, nb as u32, None).unwrap();
        }
        ex.restore_snapshot(&mut a, &pa, 7).unwrap();
        ex.restore_snapshot(&mut b, &pb, 7).unwrap();
        {
            let mut items = vec![
                BatchItem {
                    session: &mut a,
                    tokens: &ta,
                    speculative: spec,
                },
                BatchItem {
                    session: &mut b,
                    tokens: &tb,
                    speculative: spec,
                },
            ];
            ex.forward_batch(&mut items).unwrap();
        }
        let mut batch_a = vec![0f32; na * vocab];
        let mut batch_b = vec![0f32; nb * vocab];
        ex.select_batch_logits(0, na as u32, Some(&mut batch_a))
            .unwrap();
        ex.select_batch_logits(na as u32, nb as u32, Some(&mut batch_b))
            .unwrap();
        if spec {
            ex.rollback(&mut a, na as u32, None).unwrap();
            ex.rollback(&mut b, nb as u32, None).unwrap();
        }
        for (name, s, bt) in [("a", &single_a, &batch_a), ("b", &single_b, &batch_b)] {
            let differ = s
                .iter()
                .zip(bt)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(differ, 0, "{label}, session {name}: logits differ");
        }
        ex.restore_snapshot(&mut a, &pa, 7).unwrap();
        ex.restore_snapshot(&mut b, &pb, 7).unwrap();
    }
}
