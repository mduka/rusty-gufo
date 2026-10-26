//! Sync and decode times of the scenarios `tests/reference/cpp_session_dump.cpp` prints
//! them for, to compare the Rust and C++ engines on an idle device (not a
//! gate: `ab.py` measures the served paths):
//! cargo test -p gufo-qwen38-flash-next --features gpu --test speed -- \
//!   --ignored --nocapture
//!
//! `GUFO_FLASH_NEXT_SPEED` = "PROMPT K STEPS TEMP SEED [REPEAT FREQUENCY
//! PRESENCE]" (default "prompt2600.bin 1 64 0 0"), with the prompt in the
//! fixture directory and optional penalties over the last 64 tokens.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{mtp_path, trunk_path};
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen38_flash_next::engine::{DecodeRequest, Model, ModelOptions};
use gufo_qwen38_flash_next::executor::SessionMode;

#[test]
#[ignore = "timing on an idle device with the artifact"]
fn session_speed() {
    let spec =
        std::env::var("GUFO_FLASH_NEXT_SPEED").unwrap_or_else(|_| "prompt2600.bin 1 64 0 0".into());
    let f: Vec<&str> = spec.split_whitespace().collect();
    let (prompt, k, steps): (&str, usize, usize) =
        (f[0], f[1].parse().unwrap(), f[2].parse().unwrap());
    let (temperature, seed): (f32, i64) = (f[3].parse().unwrap(), f[4].parse().unwrap());
    let penalty = |i: usize, default: f32| f.get(i).map_or(default, |v| v.parse().unwrap());
    let (repeat, frequency, presence) = (penalty(5, 1.0), penalty(6, 0.0), penalty(7, 0.0));
    let dir = std::env::var_os("GUFO_FLASH_NEXT_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/flash-next"),
        PathBuf::from,
    );
    let prompt: Vec<i32> = std::fs::read(dir.join(prompt))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b) as i32)
        .collect();
    let mut model = Model::load(
        &trunk_path(),
        ModelOptions {
            max_context: 4096,
            mtp_model_path: Some(mtp_path()),
            decode_concurrency: k as u32,
            ..ModelOptions::default()
        },
    )
    .unwrap();
    let mut sessions = Vec::new();
    let mut samplers = Vec::new();
    for i in 0..k {
        let mut session = model
            .create_session(SessionMode::Speculative, 4096)
            .unwrap();
        let length = prompt.len() - 7 * i;
        let start = Instant::now();
        model.sync(&mut session, &prompt[..length]).unwrap();
        println!(
            "session {i}: sync of {length} tokens in {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
        sessions.push(session);
        let base = SamplingConfig {
            repeat_penalty: repeat,
            frequency_penalty: frequency,
            presence_penalty: presence,
            ..SamplingConfig::default()
        };
        let config = if temperature > 0.0 {
            SamplingConfig {
                temperature,
                top_k: 40,
                seed: seed + i as i64,
                ..base
            }
        } else {
            base
        };
        samplers.push(SamplerState::new(config, &[]).unwrap());
    }
    let start = Instant::now();
    let mut warm_start = start;
    let (mut generated, mut warm) = (0, 0);
    for step in 0..steps {
        if step == steps / 2 {
            warm_start = Instant::now();
            warm = generated;
        }
        if k == 1 {
            let result = model
                .decode_step(&mut sessions[0], 8, &mut samplers[0], false)
                .unwrap();
            generated += result.tokens.len();
        } else {
            let mut requests: Vec<DecodeRequest<'_>> = sessions
                .iter_mut()
                .zip(samplers.iter_mut())
                .map(|(session, sampler)| DecodeRequest {
                    session,
                    max_tokens: 8,
                    sampler,
                    stop_at_eos: false,
                })
                .collect();
            for result in model.decode_batch(&mut requests) {
                generated += result.unwrap().tokens.len();
            }
        }
    }
    println!(
        "decode: {generated} tokens in {:.1} ms; second half {} tokens in {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3,
        generated - warm,
        warm_start.elapsed().as_secs_f64() * 1e3
    );
}

/// The upload and the executor's decode forwards as
/// `tests/reference/cpp_batch_timing.cpp` times them: four prefilled sessions, batched
/// forwards of 2, 2, 1 and 1 tokens with every chain's logits, then
/// single-token forwards of one session.
#[test]
#[ignore = "timing on an idle device with the artifact"]
fn executor_speed() {
    use std::sync::Arc;

    use gufo_hip::{MappedFile, Stream};
    use gufo_qwen38_flash_next::batch::BatchItem;
    use gufo_qwen38_flash_next::device::DeviceModel;
    use gufo_qwen38_flash_next::executor::{Executor, ForwardMode, Options};
    use gufo_qwen38_flash_next::ngram::NgramTable;
    use gufo_qwen38_flash_next::weights::{GgufFiles, ModelTensors};

    let rounds: usize = std::env::var("GUFO_FLASH_NEXT_ROUNDS").map_or(40, |r| r.parse().unwrap());
    let dir = std::env::var_os("GUFO_FLASH_NEXT_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/flash-next"),
        PathBuf::from,
    );
    let prompt: Vec<i32> = std::fs::read(dir.join("prompt50.bin"))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b) as i32)
        .collect();
    let files = GgufFiles::open(&trunk_path()).unwrap();
    let model = ModelTensors::bind(&files.parse().unwrap()).unwrap();
    let mapped: Vec<Arc<MappedFile>> = files
        .paths
        .iter()
        .map(|p| MappedFile::open(p).unwrap())
        .collect();
    let stream = Stream::new().unwrap();
    let upload = Instant::now();
    let device = DeviceModel::upload(&stream, &model, &mapped, None).unwrap();
    println!("upload {:.2} s", upload.elapsed().as_secs_f64());
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
    let mut logits = vec![0f32; 8 * vocab];
    let mut sessions: Vec<_> = (0..4)
        .map(|i| {
            let mut s = ex
                .create_session(SessionMode::Autoregressive, 4096)
                .unwrap();
            ex.forward(
                &mut s,
                &prompt[..prompt.len() - 7 * i],
                1,
                Some(&mut logits),
                ForwardMode::Prefill,
            )
            .unwrap();
            s
        })
        .collect();
    let widths = [2usize, 2, 1, 1];
    let tokens = [1000i32, 1001];
    let warmup = rounds / 4;
    let mut batch = std::time::Duration::ZERO;
    for round in 0..rounds + warmup {
        let start = Instant::now();
        {
            let mut items: Vec<BatchItem<'_>> = sessions
                .iter_mut()
                .zip(widths)
                .map(|(session, w)| BatchItem {
                    session,
                    tokens: &tokens[..w],
                    speculative: false,
                })
                .collect();
            ex.forward_batch(&mut items).unwrap();
        }
        let mut offset = 0;
        for w in widths {
            ex.select_batch_logits(offset, w as u32, Some(&mut logits))
                .unwrap();
            offset += w as u32;
        }
        if round >= warmup {
            batch += start.elapsed();
        }
    }
    let mut single = std::time::Duration::ZERO;
    for round in 0..rounds + warmup {
        let start = Instant::now();
        ex.forward(
            &mut sessions[0],
            &tokens[..1],
            1,
            Some(&mut logits),
            ForwardMode::Decode,
        )
        .unwrap();
        if round >= warmup {
            single += start.elapsed();
        }
    }
    println!(
        "batch forward {:.3} ms, single forward {:.3} ms",
        batch.as_secs_f64() * 1e3 / rounds as f64,
        single.as_secs_f64() * 1e3 / rounds as f64
    );
}
