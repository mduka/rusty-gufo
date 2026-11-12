//! The executor against the C++ executor's logits of every step, bit for
//! bit, in the layout of `tests/reference/cpp_forward_dump.cpp` (prefill in
//! chunks, token-by-token decode, then greedy steps). The 2,048-token
//! chunks of the fixture also hold for one pass across the indexer budget,
//! whose earlier queries take the dense tiles, and for chunks that read the
//! next chunk's n-gram rows on the way:
//! cargo test -p gufo-qwen38-flash-next --features gpu --test forward
//!
//! `GUFO_FLASH_NEXT_FIXTURES` names the directory with `prompt*.bin` and
//! the tool's output directories (`cpp-<mode><prompt length>`); the
//! default is `target/flash-next`.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::trunk_path;
use gufo_hip::{MappedFile, Stream};
use gufo_qwen38_flash_next::device::DeviceModel;
use gufo_qwen38_flash_next::executor::{Executor, ForwardMode, Options, SessionMode};
use gufo_qwen38_flash_next::ngram::NgramTable;
use gufo_qwen38_flash_next::weights::{GgufFiles, ModelTensors};

fn fixtures() -> PathBuf {
    std::env::var_os("GUFO_FLASH_NEXT_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/flash-next"),
        PathBuf::from,
    )
}

fn read_u32(path: &Path) -> Vec<u32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    read_u32(path).into_iter().map(f32::from_bits).collect()
}

/// The first maximum, as `std::max_element`.
fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as i32
}

fn compare(what: &str, rust: &[f32], cpp: &[f32]) {
    let mismatches = rust
        .iter()
        .zip(cpp)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if mismatches != 0 {
        let first = rust
            .iter()
            .zip(cpp)
            .position(|(a, b)| a.to_bits() != b.to_bits())
            .unwrap();
        let worst = rust
            .iter()
            .zip(cpp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        panic!(
            "{what}: {mismatches} logits differ, first at {first} ({} vs {}), largest difference {worst}",
            rust[first], cpp[first]
        );
    }
}

struct Loaded {
    executor: Executor,
}

fn load(max_batch: u32) -> Loaded {
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
    let executor = Executor::new(
        Arc::new(device),
        Some(Arc::new(ngram)),
        Options {
            max_batch,
            max_logit_rows: 1,
            max_speculative: 1,
            max_context: 0,
        },
    )
    .unwrap();
    Loaded { executor }
}

/// Feeds the prompt as `mode` does (prefill in passes of `chunk` tokens,
/// each naming the next with `prefetch`), then the fixture's greedy steps,
/// comparing every step's logits.
fn run(
    executor: &mut Executor,
    name: &str,
    prompt: &str,
    mode: &str,
    chunk: usize,
    prefetch: bool,
) {
    let dir = fixtures();
    let expected = dir.join(format!("cpp-{name}"));
    if !expected.join("ids.bin").exists() {
        panic!(
            "missing fixture {}; run tests/reference/cpp_forward_dump.cpp",
            expected.display()
        );
    }
    let prompt: Vec<i32> = read_u32(&dir.join(prompt))
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let ids = read_u32(&expected.join("ids.bin"));
    let vocab = executor.config().vocab_size as usize;
    let mut session = executor
        .create_session(SessionMode::Autoregressive, 4096)
        .unwrap();
    let mut logits = vec![0f32; vocab];
    match mode {
        "prefill" => {
            let chunks: Vec<&[i32]> = prompt.chunks(chunk).collect();
            for (i, chunk) in chunks.iter().enumerate() {
                let next = chunks.get(i + 1).filter(|_| prefetch).copied();
                executor
                    .forward_checkpoint(
                        &mut session,
                        chunk,
                        1,
                        Some(&mut logits),
                        ForwardMode::Prefill,
                        None,
                        next.unwrap_or(&[]),
                    )
                    .unwrap();
            }
        }
        _ => {
            for token in &prompt {
                executor
                    .forward(
                        &mut session,
                        std::slice::from_ref(token),
                        1,
                        Some(&mut logits),
                        ForwardMode::Decode,
                    )
                    .unwrap();
            }
        }
    }
    for (step, &id) in ids.iter().enumerate() {
        let path = expected.join(format!("logits_{step}.bin"));
        compare(&format!("{name} step {step}"), &logits, &read_f32(&path));
        let next = argmax(&logits);
        assert_eq!(next as u32, id, "{name} step {step}");
        if step + 1 < ids.len() {
            executor
                .forward(
                    &mut session,
                    &[next],
                    1,
                    Some(&mut logits),
                    ForwardMode::Decode,
                )
                .unwrap();
        }
    }
}

#[test]
fn logits_match_the_cpp_executor() {
    if !trunk_path().exists() {
        panic!(
            "the Flash-Next artifact is missing: {}",
            trunk_path().display()
        );
    }
    let mut loaded = load(4096);
    let executor = &mut loaded.executor;
    run(executor, "decode12", "prompt12.bin", "decode", 1, false);
    run(
        executor,
        "prefill50",
        "prompt50.bin",
        "prefill",
        2048,
        false,
    );
    // Two chunks: the fused and wide routes, then sparse attention with a
    // pooling backlog, then sparse decode graphs; once more with the second
    // chunk's n-gram rows read during the first.
    run(
        executor,
        "prefill2600",
        "prompt2600.bin",
        "prefill",
        2048,
        false,
    );
    run(
        executor,
        "prefill2600",
        "prompt2600.bin",
        "prefill",
        2048,
        true,
    );
    // One pass: dense tiles below the budget, sparse ones past it.
    run(
        executor,
        "prefill2600",
        "prompt2600.bin",
        "prefill",
        4096,
        false,
    );
}
