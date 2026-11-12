//! Retained sequences and snapshots against the C++ reference, bit for bit
//! (`session` and `snapshot` fixtures):
//! cargo test --profile gpu-test -p gufo-qwen35moe --features gpu --test session
//!
//! Needs `GUFO_QWEN35MOE_MODEL` (UD-Q4_K_M) and `GUFO_QWEN35MOE_Q8_MODEL`
//! (Q8_0) and fails without them.
//! - `session`: `generate`, then `generate_from_prefix` over the prompt, the
//!   first turn's tokens (all forwarded by `forward_token`) and a suffix
//!   that the second turn prefills at its offset (`GenerateFromPrefix`).
//! - `snapshot`: a snapshot of the haiku prefill restored on the device and
//!   from its compact `GQKVSNP1` serialization continues as the fresh
//!   sequence did; restored and continued with a suffix prefill, it equals
//!   one cold prefill of prompt and suffix.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{Q8_0, Quant, SERIAL, UD_Q4_K_M, case_prompt, trace, u32s};
use gufo_qwen::vision::RopeLayout;
use gufo_qwen35moe::Engine;
use gufo_qwen35moe::persistent::{CompactLayout, KvStorage, RecurrentStorage};

fn retained_sequences(engine: &mut Engine, quant: Quant, failures: &mut Vec<String>) {
    let prefill = quant.fixture("prefill");
    let session = quant.fixture("session");
    let suffix = u32s(&session["suffix"]);
    for case in session["cases"].as_array().unwrap() {
        let prompt = case_prompt(&prefill, case);
        let max_new = case["max_new_tokens"].as_u64().unwrap() as usize;
        let first = engine.generate(&prompt, max_new, |_| true).unwrap();
        if first != u32s(&case["first"]) {
            failures.push(format!(
                "{}: session {}: first turn",
                quant.set, case["prompt"]
            ));
            continue;
        }
        let mut next_prompt = prompt.clone();
        next_prompt.extend(&first);
        let retained = next_prompt.len();
        next_prompt.extend(&suffix);
        let second = engine
            .generate_from_prefix(&next_prompt, retained, max_new, |_| true)
            .unwrap();
        if second != u32s(&case["second"]) {
            failures.push(format!(
                "{}: session {}: second turn",
                quant.set, case["prompt"]
            ));
        }
    }
}

fn snapshots(engine: &mut Engine, quant: Quant, failures: &mut Vec<String>) {
    let fixture = quant.fixture("snapshot");
    let prompt = u32s(&fixture["prompt_ids"]);
    let suffix = u32s(&fixture["suffix"]);
    let len = prompt.len();
    let mut check = |what: &str, divergence: Option<String>| {
        if let Some(divergence) = divergence {
            failures.push(format!("{}: snapshot {what}: {divergence}", quant.set));
        }
    };

    engine.reset().unwrap();
    let frontier = engine.forward_prompt(&prompt, 0, true).unwrap();
    let snapshot = engine.save_snapshot(len as u32).unwrap();
    check(
        "fresh",
        trace(engine, &fixture["fresh"], frontier, len, false),
    );

    // The snapshot holds no logits: the restored trace starts by feeding the
    // frontier token.
    engine.restore_snapshot(&snapshot).unwrap();
    check(
        "restored",
        trace(engine, &fixture["restored"], frontier, len, true),
    );

    // The compact serialization (`SerializeCompact`), restored into a reset
    // sequence (`RestoreCompactSnapshot`).
    let c = engine.config().clone();
    let ssm = c.ssm_layer_count() as usize;
    let layout = CompactLayout::new(
        c.full_attention_layer_count(),
        c.kv_size(),
        engine.max_context(),
        len as u32,
        KvStorage::F16,
        RecurrentStorage::F32,
        ssm * c.ssm_qkv_size() as usize * c.ssm_conv_kernel as usize,
        ssm * c.ssm_time_step_rank as usize
            * c.ssm_state_size as usize
            * c.ssm_value_size() as usize,
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        layout.total_bytes() as u64,
        fixture["compact_payload_bytes"].as_u64().unwrap()
    );
    let mut payload = vec![0u8; layout.total_bytes()];
    layout.write(&mut payload).unwrap();
    let stream = engine.model().stream();
    let kv = snapshot.kv();
    let plane = kv.len() / 2;
    stream
        .download_bytes(kv.slice(0..plane), &mut payload[layout.k()])
        .unwrap();
    stream
        .download_bytes(kv.slice(plane..2 * plane), &mut payload[layout.v()])
        .unwrap();
    stream
        .download_bytes(snapshot.conv().view(), &mut payload[layout.conv()])
        .unwrap();
    stream
        .download_bytes(snapshot.deltanet().view(), &mut payload[layout.deltanet()])
        .unwrap();
    engine.reset().unwrap();
    let parsed = CompactLayout::parse(&payload).unwrap();
    assert_eq!(parsed, layout);
    {
        let (model, session) = engine.parts_mut();
        model
            .restore_compact(
                session,
                &payload[parsed.k()],
                &payload[parsed.v()],
                &payload[parsed.conv()],
                &payload[parsed.deltanet()],
                parsed.valid_context,
                &RopeLayout::default(),
            )
            .unwrap();
    }
    check(
        "compact restored",
        trace(engine, &fixture["compact_restored"], frontier, len, true),
    );

    // Restored and continued with the suffix as one prefill at its offset.
    engine.restore_snapshot(&snapshot).unwrap();
    let next = engine.forward_prompt(&suffix, len as u32, true).unwrap();
    check(
        "continued",
        trace(
            engine,
            &fixture["continued"],
            next,
            len + suffix.len(),
            false,
        ),
    );

    // One cold prefill of prompt and suffix.
    let mut whole = prompt.clone();
    whole.extend(&suffix);
    engine.reset().unwrap();
    let next = engine.forward_prompt(&whole, 0, true).unwrap();
    check(
        "cold",
        trace(engine, &fixture["cold"], next, whole.len(), false),
    );
}

fn sessions_and_snapshots_match(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let max_context =
        u32::try_from(quant.fixture("session")["max_context"].as_u64().unwrap()).unwrap();
    let mut engine = Engine::load(&quant.model(), max_context).unwrap();
    let mut failures = Vec::new();
    retained_sequences(&mut engine, quant, &mut failures);
    snapshots(&mut engine, quant, &mut failures);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn qwen36_35b_a3b_ud_q4_k_m() {
    sessions_and_snapshots_match(UD_Q4_K_M);
}

#[test]
fn qwen36_35b_a3b_q8_0() {
    sessions_and_snapshots_match(Q8_0);
}
