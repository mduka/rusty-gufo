//! GPU sampling kernels against independent CPU orderings and
//! distributions (upstream `tests/models/qwen27b/sampling_test.hip`):
//! cargo test --profile gpu-test -p gufo-qwen35 --features gpu --test sampling
//!
//! Needs a gfx1151 device but no model.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_hip::{Bf16, DeviceBuffer, DeviceCopy, Stream};
use gufo_qwen_kernels as k;

fn upload<T: DeviceCopy>(stream: &Stream, values: &[T]) -> DeviceBuffer<T> {
    let mut buffer = stream.alloc::<T>(values.len().max(1)).unwrap();
    if !values.is_empty() {
        stream
            .upload(values, buffer.slice_mut(0..values.len()))
            .unwrap();
    }
    buffer
}

fn download<T: DeviceCopy + Default + Clone>(
    stream: &Stream,
    buffer: &DeviceBuffer<T>,
    len: usize,
) -> Vec<T> {
    let mut host = vec![T::default(); len];
    stream.download(buffer.slice(0..len), &mut host).unwrap();
    host
}

/// Small top-k reduces exact top-32/64 winners of 1024-token tiles instead of
/// sorting the whole row: the prefix keeps the full sort's scores and ID
/// order (lowest ID on ties, signed zeros equal, nonfinite values last), which
/// an independent stable CPU ordering also gives (`TestExactCandidatePrefix`).
#[test]
fn bounded_candidates_match_the_full_sort() {
    let stream = Stream::new().unwrap();
    for size in [1023usize, 1024, 1025, 32769, 248320] {
        let storage_bytes = 16 << 20;
        let mut adjusted = stream.alloc::<f32>(size).unwrap();
        let mut sorted = stream.alloc::<f32>(size).unwrap();
        let mut ids = stream.alloc::<u32>(size).unwrap();
        let mut sorted_ids = stream.alloc::<u32>(size).unwrap();
        let mut storage = stream.alloc::<u8>(storage_bytes).unwrap();
        for fixture in 0..4 {
            let mut logits: Vec<f32> = (0..size)
                .map(|i| match fixture {
                    0 => ((i * 131 % 997) as i32 - 498) as f32 / 97.0,
                    1 => {
                        if i % 2 == 1 {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    2 => {
                        if i % 3 != 0 {
                            f32::NEG_INFINITY
                        } else {
                            f32::NAN
                        }
                    }
                    _ => {
                        if i % 1024 == 1023 {
                            10.0
                        } else {
                            -1.0
                        }
                    }
                })
                .collect();
            if fixture != 1 {
                logits[size / 2] = f32::INFINITY;
                logits[size - 1] = f32::NEG_INFINITY;
            }
            let finite = |id: usize| {
                if logits[id].is_finite() {
                    logits[id]
                } else {
                    f32::NEG_INFINITY
                }
            };
            let mut expected: Vec<usize> = (0..size).collect();
            // A stable sort keeps ascending IDs among equal (and signed-zero)
            // scores.
            expected.sort_by(|&a, &b| finite(b).partial_cmp(&finite(a)).unwrap());
            let device = upload(&stream, &logits);
            let mut sort = |keep: usize,
                            adjusted: &mut DeviceBuffer<f32>,
                            sorted: &mut DeviceBuffer<f32>,
                            ids: &mut DeviceBuffer<u32>,
                            sorted_ids: &mut DeviceBuffer<u32>| {
                k::sort_logits(
                    &stream,
                    device.slice(0..size),
                    adjusted.as_mut_slice(),
                    sorted.as_mut_slice(),
                    ids.as_mut_slice(),
                    sorted_ids.as_mut_slice(),
                    storage.as_mut_slice(),
                    storage_bytes,
                    size,
                    keep,
                )
                .unwrap();
            };
            sort(0, &mut adjusted, &mut sorted, &mut ids, &mut sorted_ids);
            let reference = download(&stream, &sorted, 64);
            let reference_ids = download(&stream, &sorted_ids, 64);
            assert!(
                reference_ids
                    .iter()
                    .zip(&expected)
                    .all(|(&a, &b)| a as usize == b),
                "vocabulary {size}, fixture {fixture}: full sort differs from the CPU order"
            );
            for keep in [1usize, 20, 32, 33, 64] {
                sort(keep, &mut adjusted, &mut sorted, &mut ids, &mut sorted_ids);
                let actual = download(&stream, &sorted, keep);
                let actual_ids = download(&stream, &sorted_ids, keep);
                assert!(
                    actual
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                        && actual_ids[..] == reference_ids[..keep],
                    "vocabulary {size}, fixture {fixture}, keep {keep}: bounded candidates \
                     lost exact scores or tie order"
                );
            }
        }
    }
}

/// Numeric penalties evolve across the draft chain: each row's reported q
/// equals an independent CPU distribution over the same lattice after the
/// earlier drafted tokens were accepted, under positive and negative repeat,
/// frequency and presence penalties with repeat-window expiry, and each
/// draft lies inside its row's reported CDF (`TestDraftSamplingControls`).
#[test]
fn draft_penalties_match_the_conditional_cpu_distribution() {
    const VOCAB: usize = 2051;
    const ROWS: usize = 4;
    const COUNT: usize = 8;
    const RANK: usize = 16;
    let stream = Stream::new().unwrap();
    let ids: [u32; COUNT] = [3, 4, 5, 6, 1025, 1026, 1027, 2048];
    let mut logits = vec![-20.0f32; ROWS * VOCAB];
    for row in 0..ROWS {
        for (i, &id) in ids.iter().enumerate() {
            logits[row * VOCAB + id as usize] = 4.0 - (i / 2) as f32 * 0.2;
        }
    }
    let partials = VOCAB.div_ceil(2048) * 16;
    let uniforms: [f32; ROWS] = [0.0, 0.23, 0.91, 0.999_999_94];
    for policy in 0..8 {
        let config = SamplingConfig {
            temperature: if policy == 7 { 0.0001 } else { 0.7 },
            repeat_penalty: match policy {
                1 | 7 => 1.2,
                2 => 0.8,
                _ => 1.0,
            },
            repeat_last_n: 2,
            frequency_penalty: match policy {
                3 => 0.2,
                4 => -0.2,
                _ => 0.0,
            },
            presence_penalty: match policy {
                5 | 7 => 1.5,
                6 => -1.5,
                _ => 0.0,
            },
            ..SamplingConfig::default()
        };
        let mut expected = SamplerState::new(config.clone(), &ids[..4]).unwrap();
        expected.accept(3).unwrap();
        expected.accept(5).unwrap();
        let d_logits = upload(&stream, &logits);
        let hidden = upload(&stream, &vec![0f32; ROWS * RANK]);
        let codebook = upload(&stream, &vec![Bf16(0); VOCAB * RANK]);
        let anchor = stream
            .zeroed_indices(stream.alloc::<u32>(1).unwrap(), VOCAB as u32)
            .unwrap();
        let mut tokens = stream.alloc::<u32>(ROWS + 1).unwrap();
        let mut confidences = stream.alloc::<f32>(ROWS).unwrap();
        let mut partial_scores = stream.alloc::<f32>(ROWS * partials).unwrap();
        let mut partial_ids = stream.alloc::<u32>(ROWS * partials).unwrap();
        let draws = upload(&stream, &uniforms);
        let mut candidates = stream.alloc::<u32>(ROWS * COUNT).unwrap();
        let mut probabilities = stream.alloc::<f32>(ROWS * COUNT).unwrap();
        let penalty_words: Vec<u32> = expected
            .penalties()
            .iter()
            .flat_map(|p| [p.token, p.generated_count, p.repeated])
            .collect();
        let penalties = upload(&stream, &penalty_words);
        let history = upload(&stream, expected.history());
        k::dflash_selector_sampled(
            &stream,
            d_logits.as_slice(),
            hidden.as_slice(),
            codebook.as_slice(),
            codebook.as_slice(),
            anchor.slice(0..1),
            tokens.as_mut_slice(),
            confidences.as_mut_slice(),
            partial_scores.as_mut_slice(),
            partial_ids.as_mut_slice(),
            draws.as_slice(),
            candidates.as_mut_slice(),
            probabilities.as_mut_slice(),
            ROWS as u32,
            config.temperature,
            VOCAB as u32,
            RANK as u32,
            COUNT as u32,
            config.repeat_penalty,
            config.frequency_penalty,
            config.presence_penalty,
            config.repeat_last_n as u32,
            (!penalty_words.is_empty()).then(|| penalties.slice(0..penalty_words.len())),
            expected.penalties().len() as u32,
            (!expected.history().is_empty()).then(|| history.slice(0..expected.history().len())),
            expected.history().len() as u32,
        )
        .unwrap();
        let actual = download(&stream, &tokens, ROWS + 1);
        let actual_ids = download(&stream, &candidates, ROWS * COUNT);
        let actual_probabilities = download(&stream, &probabilities, ROWS * COUNT);
        for row in 0..ROWS {
            // The base lattice keeps the selector's lane-order tie breaks; its
            // set is known independently. A full-vocabulary CPU row gives the
            // numeric ties by actual token IDs, not compact row indices.
            let selected = &actual_ids[row * COUNT..(row + 1) * COUNT];
            let mut sorted = selected.to_vec();
            sorted.sort_unstable();
            assert_eq!(sorted, ids, "policy {policy} row {row}: lattice changed");
            let mut scores = vec![f32::NEG_INFINITY; VOCAB];
            for &id in &ids {
                scores[id as usize] = logits[row * VOCAB + id as usize];
            }
            let distribution = expected.distribution(&scores).unwrap();
            let mut cumulative = 0.0f64;
            let mut checked = false;
            for (i, &token) in selected.iter().enumerate() {
                let probability = distribution.probability(token);
                let gpu = f64::from(actual_probabilities[row * COUNT + i]);
                assert!(
                    (gpu - probability).abs() < 2e-5,
                    "policy {policy} row {row} token {token}: gpu {gpu} cpu {probability}"
                );
                if actual[row + 1] == token {
                    let draw = f64::from(uniforms[row]);
                    assert!(
                        probability > 0.0
                            && cumulative <= draw + 2e-5
                            && cumulative + probability >= draw - 2e-5,
                        "policy {policy} row {row}: draft outside its reported q CDF"
                    );
                    checked = true;
                }
                cumulative += probability;
            }
            assert!(
                checked,
                "policy {policy} row {row}: draft is not one of its candidates"
            );
            expected.accept(actual[row + 1]).unwrap();
        }
    }
}
