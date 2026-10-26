//! The Q8 routes that replace the Q8 to F16 weight cache once it is disabled
//! (`Backend::disable_f16_weight_cache`, as after an allocation failure):
//! a wide prefill, its decode, and three sessions decoding together. The C++
//! engine has no entry for these routes, so the fixture comes from this
//! runtime, recorded on ROCm 10.1 (on ROCm 10.0 from the C++ runtime it
//! replaced; the device code of both is instruction-identical). The prompts
//! are those of `forward.json`. The switch lasts for the process, hence a
//! test binary of its own.
//! `record_q8_routes` writes `fixtures/q8_routes.json`:
//! cargo test --profile gpu-test -p gufo-deepseek-v4-flash --features gpu --test q8_routes -- --ignored record_q8_routes
//! Needs the model and a gfx1151.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_core::json::{self, Value};
use gufo_deepseek_v4_flash::engine::Engine;
use gufo_deepseek_v4_flash::session::Session;

const CASE: &str = "p1500";
const BATCH: &str = "batch3";
const STEPS: usize = 4;

fn digest(logits: &[f32]) -> String {
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    common::fnv(&bytes)
}

fn encode(engine: &Engine, text: &str) -> Vec<i32> {
    let mut tokens = Vec::new();
    engine.vocab.encode_chat(b"", text.as_bytes(), &mut tokens);
    tokens
}

fn named<'a>(items: &'a [Value], name: &str) -> &'a Value {
    items
        .iter()
        .find(|item| item.member_str("name", "") == name)
        .unwrap()
}

/// The engine with the F16 weight cache disabled, and the `forward.json`
/// case and batch these routes replay.
fn setup() -> (Engine, Value, Value) {
    let forward = json::parse(include_bytes!("fixtures/forward.json")).unwrap();
    let case = named(forward.find("cases").unwrap().items(), CASE).clone();
    let batch = named(forward.find("batches").unwrap().items(), BATCH).clone();
    let engine = Engine::open(&common::model_path(), None).unwrap();
    engine.backend.disable_f16_weight_cache();
    (engine, case, batch)
}

/// Logits digests and greedy IDs of one session, step by step.
fn single(engine: &mut Engine, case: &Value) -> Value {
    let prompt = encode(engine, case.find("text").unwrap().str());
    let mut session = engine
        .create_session(case.member_size("context", 0) as u32, false)
        .unwrap();
    session.sync(engine, &prompt).unwrap();
    let (mut logits, mut ids) = (Value::array(), Value::array());
    for step in 0..=STEPS {
        logits.push(digest(session.logits().unwrap()));
        let next = session.argmax();
        ids.push(next as usize);
        if step < STEPS {
            session.eval(engine, next).unwrap();
        }
    }
    Value::object().with("logits", logits).with("ids", ids)
}

/// Logits digests and greedy IDs of every session of a batch.
fn batched(engine: &mut Engine, batch: &Value) -> Value {
    let context = batch.member_size("context", 0) as u32;
    let mut sessions = Vec::new();
    for text in batch.find("texts").unwrap().items() {
        let prompt = encode(engine, text.str());
        let mut session = engine.create_session(context, false).unwrap();
        session.sync(engine, &prompt).unwrap();
        sessions.push(session);
    }
    let mut out: Vec<(Value, Value)> = vec![(Value::array(), Value::array()); sessions.len()];
    for step in 0..=STEPS {
        let mut tokens = Vec::new();
        for (session, (logits, ids)) in sessions.iter().zip(out.iter_mut()) {
            logits.push(digest(session.logits().unwrap()));
            let next = session.argmax();
            ids.push(next as usize);
            tokens.push(next);
        }
        if step < STEPS {
            let mut items: Vec<(&mut Session, i32)> = sessions.iter_mut().zip(tokens).collect();
            Session::eval_batch(engine, &mut items).unwrap();
        }
    }
    Value::from(
        out.into_iter()
            .map(|(logits, ids)| Value::object().with("logits", logits).with("ids", ids))
            .collect::<Vec<_>>(),
    )
}

#[test]
fn q8_routes_match_the_recorded_runtime() {
    let fixture = json::parse(include_bytes!("fixtures/q8_routes.json")).unwrap();
    let (mut engine, case, batch) = setup();
    assert_eq!(
        single(&mut engine, &case).dump().unwrap(),
        fixture.find("case").unwrap().dump().unwrap(),
        "{CASE} without the F16 weight cache"
    );
    assert_eq!(
        batched(&mut engine, &batch).dump().unwrap(),
        fixture.find("batch").unwrap().dump().unwrap(),
        "{BATCH} without the F16 weight cache"
    );
    eprintln!("{CASE} and {BATCH}: {STEPS} steps equal without the F16 weight cache");
}

/// Writes `fixtures/q8_routes.json` from the current runtime.
#[test]
#[ignore = "records the fixture; run alone"]
fn record_q8_routes() {
    let (mut engine, case, batch) = setup();
    let fixture = Value::object()
        .with(
            "source",
            "the Rust runtime on ROCm 10.1 (tests/q8_routes.rs)",
        )
        .with("case", single(&mut engine, &case))
        .with("batch", batched(&mut engine, &batch));
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/q8_routes.json");
    std::fs::write(path, fixture.dump().unwrap() + "\n").unwrap();
    eprintln!("recorded {path}");
}
