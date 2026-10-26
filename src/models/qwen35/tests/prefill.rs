//! The Rust prefill and greedy generation against the C++ engine, bit for
//! bit, per checkpoint:
//! cargo test --release -p gufo-qwen35 --features gpu --test prefill
//!
//! Needs each GGUF of [`MODELS`] at its pinned revision in the Hugging Face
//! cache (the Qwen3.8 27B UD-Q4_K_XL also from `GUFO_QWEN27B_MODEL`) and
//! fails without it. The fixtures come from `tests/reference/cpp_decode_dump.cpp
//! prefill|generate` (`tests/reference/record_prefill.py` writes one) on
//! synthetic chat prompts of 19 to 5089 tokens: the Q8_1
//! WMMA prefill (FP16 from 1024 rows on the 27B), two chunks (4096 + 993
//! rows) with the second one attending to the first, and per checkpoint the
//! head shapes (24/4, 16/4 and 8/2 heads of 256) and projection formats it
//! carries (Q8_0, the K-quants and IQ formats, BF16, tied LM heads).

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use gufo_qwen35::{Engine, Model};
use serde_json::Value;

/// One model on the device at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// A checkpoint and its fixture: Hugging Face repository, revision, file.
struct Checkpoint {
    fixture: &'static str,
    repository: &'static str,
    revision: &'static str,
    file: &'static str,
}

const QWEN38_27B_Q4_K_XL: Checkpoint = Checkpoint {
    fixture: include_str!("fixtures/qwen3.8-27b-prefill.json"),
    repository: "unsloth/Qwen3.8-27B-GGUF",
    revision: "4ca720788d1e01f1bff70c033e0d0028fd02e502",
    file: "Qwen3.8-27B-UD-Q4_K_XL.gguf",
};

fn model(checkpoint: &Checkpoint) -> PathBuf {
    if checkpoint.file == QWEN38_27B_Q4_K_XL.file
        && let Some(path) = std::env::var_os("GUFO_QWEN27B_MODEL")
    {
        return PathBuf::from(path);
    }
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub")
        .join(format!(
            "models--{}",
            checkpoint.repository.replace('/', "--")
        ))
        .join("snapshots")
        .join(checkpoint.revision)
        .join(checkpoint.file)
}

fn fnv1a64(values: &[f32]) -> String {
    let hash = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
    format!("{hash:016x}")
}

fn u32s(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

/// The fixture's chat prompt around `body` synthetic tokens.
fn prompt(fixture: &Value, body: usize) -> Vec<u32> {
    let body = (0..body as u32).map(|i| 1000 + (i * 7919 + 13) % 50_000);
    u32s(&fixture["head"])
        .into_iter()
        .chain(body)
        .chain(u32s(&fixture["tail"]))
        .collect()
}

fn prefill_and_generation_match(checkpoint: &Checkpoint) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture: Value = serde_json::from_str(checkpoint.fixture).unwrap();
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let mut engine = Engine::load(&model(checkpoint), max_context).unwrap();
    for case in fixture["prefill"].as_array().unwrap() {
        let prompt = prompt(&fixture, case["body"].as_u64().unwrap() as usize);
        let len = prompt.len();
        assert_eq!(len as u64, case["prompt_len"].as_u64().unwrap());
        let ids = u32s(&case["ids"]);
        let digests = case["logits_fnv1a64"].as_array().unwrap();
        engine.reset().unwrap();
        let mut next = engine.forward_prompt(&prompt, 0, true).unwrap();
        for (step, pos) in (0..ids.len()).zip(len as u32..) {
            let token = next.unwrap();
            assert_eq!(token, ids[step], "prompt of {len}: token of step {step}");
            assert_eq!(
                fnv1a64(&engine.logits().unwrap()),
                digests[step].as_str().unwrap(),
                "prompt of {len}: logits of step {step}"
            );
            next = engine.forward_token(token, pos, true).unwrap();
        }
    }
    for case in fixture["generate"].as_array().unwrap() {
        let prompt = prompt(&fixture, case["body"].as_u64().unwrap() as usize);
        let max_new = case["max_new_tokens"].as_u64().unwrap() as usize;
        let mut streamed = Vec::new();
        let generated = engine
            .generate(&prompt, max_new, |token| {
                streamed.push(token);
                true
            })
            .unwrap();
        assert_eq!(
            generated,
            u32s(&case["ids"]),
            "generation after {}",
            prompt.len()
        );
        assert_eq!(streamed, generated);
    }
    assert_eq!(engine.graph_mismatches(), 0);
}

macro_rules! checkpoints {
    ($($test:ident: $fixture:literal, $repository:literal, $revision:literal, $file:literal;)*) => {
        $(
            #[test]
            fn $test() {
                prefill_and_generation_match(&Checkpoint {
                    fixture: include_str!(concat!("fixtures/", $fixture)),
                    repository: $repository,
                    revision: $revision,
                    file: $file,
                });
            }
        )*
    };
}

#[test]
fn qwen38_27b_q4_k_xl() {
    prefill_and_generation_match(&QWEN38_27B_Q4_K_XL);
}

checkpoints! {
    qwen38_27b_q8_k_xl: "qwen3.8-27b-q8_k_xl-prefill.json", "unsloth/Qwen3.8-27B-GGUF",
        "4ca720788d1e01f1bff70c033e0d0028fd02e502", "Qwen3.8-27B-UD-Q8_K_XL.gguf";
    qwen35_0_8b_q8_0: "qwen3.5-0.8b-q8_0-prefill.json", "unsloth/Qwen3.5-0.8B-GGUF",
        "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "Qwen3.5-0.8B-Q8_0.gguf";
    qwen35_0_8b_bf16: "qwen3.5-0.8b-bf16-prefill.json", "unsloth/Qwen3.5-0.8B-GGUF",
        "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "Qwen3.5-0.8B-BF16.gguf";
    qwen35_0_8b_iq4_xs: "qwen3.5-0.8b-iq4_xs-prefill.json", "unsloth/Qwen3.5-0.8B-GGUF",
        "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "Qwen3.5-0.8B-IQ4_XS.gguf";
    qwen35_2b_q4_k_m: "qwen3.5-2b-q4_k_m-prefill.json", "unsloth/Qwen3.5-2B-GGUF",
        "f6d5376be1edb4d416d56da11e5397a961aca8ae", "Qwen3.5-2B-Q4_K_M.gguf";
    qwen35_4b_q4_k_m: "qwen3.5-4b-q4_k_m-prefill.json", "unsloth/Qwen3.5-4B-GGUF",
        "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q4_K_M.gguf";
    qwen35_9b_q4_k_m: "qwen3.5-9b-q4_k_m-prefill.json", "unsloth/Qwen3.5-9B-GGUF",
        "3885219b6810b007914f3a7950a8d1b469d598a5", "Qwen3.5-9B-Q4_K_M.gguf";
}

/// A cancellation check that never fires leaves the prefill launches alone
/// (logits bit for bit); one that fires inside the layer stack stops the
/// prefill there, and the reset session prefills exactly again.
#[test]
fn prefill_cancellation_stops_between_layers() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture: Value = serde_json::from_str(QWEN38_27B_Q4_K_XL.fixture).unwrap();
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let mut model = Model::load(&model(&QWEN38_27B_Q4_K_XL), max_context).unwrap();
    // One chunk of the Q8_1 WMMA prefill.
    let prompt = prompt(&fixture, 2000);
    let mut plain = model.new_session().unwrap();
    let expected = model.forward_prompt(&mut plain, &prompt, 0, true).unwrap();
    let expected_logits = fnv1a64(&model.logits(&plain).unwrap());

    let mut checked = model.new_session().unwrap();
    let asks = Arc::new(AtomicUsize::new(0));
    let counter = asks.clone();
    checked.set_cancellation_check(Some(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
        false
    })));
    assert_eq!(
        model
            .forward_prompt(&mut checked, &prompt, 0, true)
            .unwrap(),
        expected
    );
    assert_eq!(fnv1a64(&model.logits(&checked).unwrap()), expected_logits);
    let layers = model.config().num_layers as usize;
    assert!(
        asks.load(Ordering::SeqCst) >= layers,
        "every layer asks the check"
    );

    let asks = Arc::new(AtomicUsize::new(0));
    let counter = asks.clone();
    checked.set_cancellation_check(Some(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst) >= layers / 3
    })));
    model.reset(&mut checked).unwrap();
    assert_eq!(
        model.forward_prompt(&mut checked, &prompt, 0, true),
        Err("Qwen prefill cancelled".to_string())
    );
    assert!(
        asks.load(Ordering::SeqCst) < layers,
        "the prefill stopped inside the layer stack"
    );
    checked.set_cancellation_check(None);
    model.reset(&mut checked).unwrap();
    assert_eq!(
        model
            .forward_prompt(&mut checked, &prompt, 0, true)
            .unwrap(),
        expected
    );
    assert_eq!(fnv1a64(&model.logits(&checked).unwrap()), expected_logits);
}

/// The final prefill layer computes its attention outputs and FFN only for
/// the 256-row FP16 tile holding the last token unless that layer is
/// captured; every cache row is still written. Capturing the last layer
/// needs all its rows, so it runs the complete computation as an independent
/// control: the full-vocabulary logits and the continued cache state match
/// it bit for bit, with no capture and with the DFlash2 taps, for a chunk at
/// depth 0 and a ragged tail behind 8K cached tokens (upstream
/// `target_test.cpp`).
#[test]
fn final_output_tile_matches_full_layer_capture() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cases = [(0usize, 2048usize), (8192, 1025)];
    let total = 8192 + 1025 + 1;
    let tokens: Vec<u32> = (0..total as u32)
        .map(|i| 1000 + (i * 7919 + 13) % 50_000)
        .collect();
    let mut engine = Engine::load(&model(&QWEN38_27B_Q4_K_XL), total as u32 + 64).unwrap();
    let hidden = engine.config().hidden_size as usize;
    let last_layer = engine.config().num_layers - 1;
    let bits = |logits: Vec<f32>| logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for (depth, length) in cases {
        let end = depth + length;
        for taps in [&[][..], &[1, 16, 31, 46, 61][..]] {
            let capture = |engine: &mut Engine| {
                engine.set_hidden_capture(!taps.is_empty(), taps).unwrap();
            };
            engine.reset().unwrap();
            capture(&mut engine);
            let prefix = if depth > 0 {
                engine.forward_prompt(&tokens[..depth], 0, false).unwrap();
                Some(engine.save_snapshot(depth as u32).unwrap())
            } else {
                None
            };
            let restore_prefix = |engine: &mut Engine| match &prefix {
                Some(prefix) => engine.restore_snapshot(prefix).unwrap(),
                None => engine.reset().unwrap(),
            };
            let chunk = &tokens[depth..end];
            let next = engine.forward_prompt(chunk, depth as u32, true).unwrap();
            let expected = bits(engine.logits().unwrap());
            let cropped = engine.save_snapshot(end as u32).unwrap();

            restore_prefix(&mut engine);
            engine.set_hidden_capture(true, &[last_layer]).unwrap();
            let full_next = engine.forward_prompt(chunk, depth as u32, true).unwrap();
            assert_eq!(full_next, next, "depth {depth}: prefill token");
            assert!(
                bits(engine.logits().unwrap()) == expected,
                "depth {depth}, taps {taps:?}: the final output tile changed the logits"
            );
            assert_eq!(
                engine.prompt_hidden().unwrap().len(),
                length * hidden,
                "a last-layer capture keeps every prompt row"
            );
            capture(&mut engine);
            engine.forward_token(tokens[end], end as u32, true).unwrap();
            let full_continuation = bits(engine.logits().unwrap());
            engine.restore_snapshot(&cropped).unwrap();
            engine.forward_token(tokens[end], end as u32, true).unwrap();
            assert!(
                bits(engine.logits().unwrap()) == full_continuation,
                "depth {depth}, taps {taps:?}: the final output tile changed the cache state"
            );
        }
    }
    assert_eq!(engine.graph_mismatches(), 0);
}
