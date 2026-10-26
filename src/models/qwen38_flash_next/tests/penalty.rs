//! Greedy verification under penalties on the device against the CPU
//! sampler: the same winner for every MTP width, ragged vocabularies ending
//! at their allocation, FP64-close candidates, rows without a finite logit
//! and untouched frontier logits (`CheckPenaltyArgmax` of the C++
//! `tests/models/qwen38_flash_next/mtp_ops_test.cpp`):
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next --features gpu --test penalty

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_core::sampling::{SamplerState, SamplingConfig, TokenId};
use gufo_hip::Stream;
use gufo_qwen38_flash_next_kernels as k;

fn check_penalty_argmax(vocab: u32) {
    let stream = Stream::new().unwrap();
    let v = vocab as usize;
    // Up to seven verification rows, as GreedyPenaltyRows holds.
    for rows in 1..=7usize {
        for mode in 0..5 {
            let mut config = SamplingConfig {
                seed: 73,
                repeat_penalty: if mode == 1 { 0.7 } else { 1.3 },
                repeat_last_n: if mode == 2 { 0 } else { 3 },
                frequency_penalty: if mode == 1 { -0.4 } else { 0.2 },
                presence_penalty: if mode == 1 { -1.5 } else { 1.5 },
                ..SamplingConfig::default()
            };
            if mode >= 3 {
                config.repeat_penalty = 1.0;
                config.frequency_penalty = 0.0;
                config.presence_penalty = 1.0e-8;
            }
            let mut sampler = SamplerState::new(config.clone(), &[0, 1, 0]).unwrap();
            sampler.accept(0).unwrap();
            sampler.accept(vocab - 1).unwrap();
            let mut logits = vec![0f32; rows * v];
            let (mut triples, mut offsets) = (Vec::new(), vec![0u32]);
            let mut expected = vec![-1i32; rows];
            for row in 0..rows {
                let values = &mut logits[row * v..(row + 1) * v];
                for (i, value) in values.iter_mut().enumerate() {
                    let mixed = (i as u32).wrapping_mul(7919).wrapping_add(row as u32 * 31) % 65521;
                    *value = (mixed as i32 - 32768) as f32 / 32768.0;
                }
                let edge = if mode == 2 { -1.0e35 } else { 1.0 };
                values[0] = edge;
                values[v - 1] = edge;
                if mode == 2 {
                    values.fill(-1.0e35);
                }
                if mode >= 3 {
                    values.fill(-100.0);
                    // FP32 penalty arithmetic would pick token zero here.
                    values[0] = 1.0;
                    values[1] = 1.0;
                    values[v - 1] = f32::INFINITY;
                }
                if mode == 4 {
                    values.fill(f32::NAN);
                }
                for p in sampler.penalties() {
                    triples.extend([p.token, p.generated_count, p.repeated]);
                }
                offsets.push((triples.len() / 3) as u32);
                match sampler.sample(values) {
                    Ok(token) => expected[row] = token as i32,
                    Err(e) => assert_eq!(mode, 4, "{e}"),
                }
                sampler.accept((row as TokenId + 2) % vocab).unwrap();
            }
            let capacity = (triples.len() / 3).max(1);
            triples.resize(3 * capacity, 0);
            let mut device_logits = stream.alloc::<f32>(logits.len()).unwrap();
            stream
                .upload(&logits, device_logits.as_mut_slice())
                .unwrap();
            let mut penalties = stream.alloc::<u32>(triples.len()).unwrap();
            stream.upload(&triples, penalties.as_mut_slice()).unwrap();
            // Row r ends at offsets[r + 1]; the ends past the last row repeat it.
            let last = *offsets.last().unwrap();
            let ends: [u32; 7] =
                std::array::from_fn(|r| offsets.get(r + 1).copied().unwrap_or(last));
            let mut partial = stream.alloc::<u8>(rows * 64 * 16).unwrap();
            let mut out = stream.alloc::<u8>(rows * 8).unwrap();
            let launched = k::penalized_argmax(
                &stream,
                device_logits.as_slice(),
                penalties.as_slice(),
                ends[0],
                ends[1],
                ends[2],
                ends[3],
                ends[4],
                ends[5],
                ends[6],
                config.repeat_penalty,
                config.frequency_penalty,
                config.presence_penalty,
                partial.as_mut_slice(),
                out.as_mut_slice(),
                rows as u32,
                vocab,
                capacity as u32,
            )
            .unwrap();
            assert_eq!(launched, gufo_hip::Launch::Launched);
            let mut bytes = vec![0u8; rows * 8];
            stream.download(out.as_slice(), &mut bytes).unwrap();
            for (row, b) in bytes.as_chunks::<8>().0.iter().enumerate() {
                let value = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let index = i32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                assert!(
                    index == expected[row] && (expected[row] >= 0 || !value.is_finite()),
                    "penalty argmax differs from CPU: vocab={vocab} rows={rows} row={row} \
                     mode={mode}: {index} vs {}",
                    expected[row]
                );
            }
            let mut retained = vec![0f32; logits.len()];
            stream
                .download(device_logits.as_slice(), &mut retained)
                .unwrap();
            assert!(
                retained
                    .iter()
                    .zip(&logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "penalty selection modified frontier logits"
            );
        }
    }
}

#[test]
fn penalized_argmax_matches_the_cpu_sampler() {
    check_penalty_argmax(257);
    check_penalty_argmax(248_320);
}
