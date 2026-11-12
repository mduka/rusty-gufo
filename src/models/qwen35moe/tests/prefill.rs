//! The Rust prefill and greedy generation against the C++ reference, bit for
//! bit (`prefill` fixtures):
//! cargo test --profile gpu-test -p gufo-qwen35moe --features gpu --test prefill
//!
//! Needs `GUFO_QWEN35MOE_MODEL` (UD-Q4_K_M) and `GUFO_QWEN35MOE_Q8_MODEL`
//! (Q8_0) and fails without them. Per case: a fresh sequence, the prompt
//! through `forward_prompt` (whole, or in pieces at their offsets as the
//! serving runner's budgeted prefill), then eight greedy `forward_token`
//! steps; tokens and the FNV-1a digest of every logits row match the C++
//! `ForwardPromptBatch` and `ForwardToken`. The lengths cross every chunk
//! route: the verification forward up to eight rows, hipBLAS routers below
//! 128 rows and the BF16 WMMA ones from there, 48-row routed tiles, the
//! paired gate/up GEMM from 1024 rows, and two chunks (4096 + 4 or + 993).

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{Q8_0, Quant, SERIAL, UD_Q4_K_M, case_prompt, trace, u32s};
use gufo_qwen35moe::Engine;

fn prefill_and_generation_match(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = quant.fixture("prefill");
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let mut engine = Engine::load(&quant.model(), max_context).unwrap();
    assert_eq!(
        u64::from(engine.model().max_prompt_batch()),
        fixture["max_prompt_batch"].as_u64().unwrap()
    );
    let mut failures = Vec::new();
    for case in fixture["prefill"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let prompt = case_prompt(&fixture, case);
        assert_eq!(prompt.len() as u64, case["prompt_len"].as_u64().unwrap());
        engine.reset().unwrap();
        let next = match case["pieces"].as_array() {
            Some(pieces) => {
                let mut offset = 0;
                let mut next = None;
                for (index, piece) in pieces.iter().enumerate() {
                    let len = piece.as_u64().unwrap() as usize;
                    let last = index + 1 == pieces.len();
                    next = engine
                        .forward_prompt(&prompt[offset..offset + len], offset as u32, last)
                        .unwrap();
                    offset += len;
                }
                assert_eq!(offset, prompt.len());
                next
            }
            None => engine.forward_prompt(&prompt, 0, true).unwrap(),
        };
        if let Some(divergence) = trace(&mut engine, case, next, prompt.len(), false) {
            failures.push(format!("{}: {name}: {divergence}", quant.set));
        }
    }
    for case in fixture["generate"].as_array().unwrap() {
        let prompt = case_prompt(&fixture, case);
        let max_new = case["max_new_tokens"].as_u64().unwrap() as usize;
        let mut streamed = Vec::new();
        let generated = engine
            .generate(&prompt, max_new, |token| {
                streamed.push(token);
                true
            })
            .unwrap();
        assert_eq!(streamed, generated);
        let expected = u32s(&case["ids"]);
        if generated != expected {
            let first = generated
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap_or(generated.len().min(expected.len()));
            failures.push(format!(
                "{}: generation after {} tokens: first differing token {first}",
                quant.set,
                prompt.len()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn qwen36_35b_a3b_ud_q4_k_m() {
    prefill_and_generation_match(UD_Q4_K_M);
}

#[test]
fn qwen36_35b_a3b_q8_0() {
    prefill_and_generation_match(Q8_0);
}
