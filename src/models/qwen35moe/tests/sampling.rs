//! Seeded sampled generation on the GPU against the C++ reference (`sampled`
//! fixtures), and the bounded top-k candidates of the sampling kernels
//! against an independent CPU ordering:
//! cargo test --profile gpu-test -p gufo-qwen35moe --features gpu --test sampling
//!
//! The generation tests need `GUFO_QWEN35MOE_MODEL` (UD-Q4_K_M) and
//! `GUFO_QWEN35MOE_Q8_MODEL` (Q8_0) and fail without them: a fresh sequence,
//! the prompt prefilled, then every token drawn by the GPU sampler with one
//! host uniform per token (`Generate` with a `SamplingConfig`), until a stop
//! token or 64 tokens.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{Q8_0, Quant, SERIAL, UD_Q4_K_M, case_prompt, u32s};
use gufo_core::sampling::SamplingConfig;
use gufo_hip::{DeviceBuffer, DeviceCopy, Stream};
use gufo_qwen35moe::Engine;
use gufo_qwen35moe_kernels as k;

fn sampled_generation_matches(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = quant.fixture("sampled");
    let prefill = quant.fixture("prefill");
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let mut engine = Engine::load(&quant.model(), max_context).unwrap();
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let prompt = case_prompt(&prefill, case);
        let s = &case["sampling"];
        let float = |name: &str| s[name].as_f64().unwrap() as f32;
        let config = SamplingConfig {
            temperature: float("temperature"),
            top_k: i32::try_from(s["top_k"].as_i64().unwrap()).unwrap(),
            top_p: float("top_p"),
            min_p: float("min_p"),
            seed: s["seed"].as_i64().unwrap(),
            ..SamplingConfig::default()
        };
        let max_new = case["max_new_tokens"].as_u64().unwrap() as usize;
        let (model, session) = engine.parts_mut();
        let generated = model
            .generate(session, &prompt, max_new, &config, |_| true)
            .unwrap();
        let expected = u32s(&case["ids"]);
        if generated != expected {
            let first = generated
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap_or(generated.len().min(expected.len()));
            failures.push(format!(
                "{}: {} with {s}: first differing token {first} of {}",
                quant.set,
                case["prompt"],
                expected.len()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn qwen36_35b_a3b_ud_q4_k_m() {
    sampled_generation_matches(UD_Q4_K_M);
}

#[test]
fn qwen36_35b_a3b_q8_0() {
    sampled_generation_matches(Q8_0);
}

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
