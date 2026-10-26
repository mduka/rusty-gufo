//! Routes `forward.json` does not reach: a prompt of more than 31,744
//! tokens, past which decode attention reads more than 7,936 compressed rows,
//! decoded further; and KV caches in managed memory, which must reproduce the
//! `p300` case of `forward.json` bit for bit. The C++ engine has no entry for
//! the long route, so its fixture comes from this runtime, recorded on ROCm
//! 10.1 (on ROCm 10.0 from the C++ runtime it replaced; the device code of
//! both is instruction-identical). `record_routes` writes
//! `fixtures/routes.json`:
//! cargo test --profile gpu-test -p gufo-deepseek-v4-flash --features gpu --test routes -- --ignored record_routes
//! Needs the model and a gfx1151.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::fmt::Write as _;

use gufo_core::json::{self, Value};
use gufo_deepseek_v4_flash::engine::Engine;

/// Positions past which decode attention reads more than 7,936 compressed
/// rows (ratio 4), with a margin.
const LONG_PROMPT_TOKENS: usize = 32_100;
const LONG_CONTEXT: u32 = 36_864;
const LONG_STEPS: usize = 4;

fn digest(logits: &[f32]) -> String {
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    common::fnv(&bytes)
}

/// A deterministic prose-like text of `paragraphs` paragraphs.
fn long_text(paragraphs: usize) -> String {
    const WORDS: [&str; 64] = [
        "river", "stone", "engine", "garden", "signal", "winter", "copper", "harbor", "lantern",
        "meadow", "circuit", "planet", "window", "silver", "forest", "number", "kernel", "bridge",
        "canvas", "thunder", "valley", "pocket", "mirror", "rocket", "orange", "pillow", "castle",
        "matrix", "button", "candle", "desert", "falcon", "glacier", "hammer", "island", "jacket",
        "ladder", "magnet", "needle", "oyster", "pepper", "quartz", "ribbon", "saddle", "tunnel",
        "umbrella", "velvet", "wagon", "yellow", "zephyr", "anchor", "basket", "cobalt", "dolphin",
        "ember", "fabric", "granite", "horizon", "indigo", "jungle", "kettle", "lemon", "marble",
        "nectar",
    ];
    let mut seed: u32 = 0x9e37_79b9;
    let mut text = String::new();
    for paragraph in 0..paragraphs {
        let _ = write!(text, "Paragraph {paragraph}: ");
        for word in 0..40 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            text.push_str(WORDS[(seed >> 26) as usize]);
            text.push_str(if word % 12 == 11 { ". " } else { " " });
        }
        text.push('\n');
    }
    text
}

fn long_prompt(engine: &Engine, paragraphs: usize) -> Vec<i32> {
    let mut tokens = Vec::new();
    engine
        .vocab
        .encode_chat(b"", long_text(paragraphs).as_bytes(), &mut tokens);
    tokens
}

/// The logits digests and greedy IDs of `prompt` and `steps` decode steps.
fn run(
    engine: &mut Engine,
    prompt: &[i32],
    context: u32,
    steps: usize,
) -> (Vec<String>, Vec<usize>) {
    let mut session = engine.create_session(context, false).unwrap();
    session.sync(engine, prompt).unwrap();
    let (mut logits, mut ids) = (Vec::new(), Vec::new());
    for step in 0..=steps {
        logits.push(digest(session.logits().unwrap()));
        let next = session.argmax();
        ids.push(next as usize);
        if step < steps {
            session.eval(engine, next).unwrap();
        }
    }
    (logits, ids)
}

#[test]
fn routes_match_the_recorded_runtime() {
    let fixture = json::parse(include_bytes!("fixtures/routes.json")).unwrap();
    let mut engine = Engine::open(&common::model_path(), None).unwrap();

    let long = fixture.find("long").unwrap();
    let prompt = long_prompt(&engine, long.member_size("paragraphs", 0));
    assert_eq!(prompt.len(), long.member_size("prompt_tokens", 0));
    assert_eq!(
        common::fnv(&common::token_bytes(&prompt)),
        long.member_str("prompt", "")
    );
    let (logits, ids) = run(
        &mut engine,
        &prompt,
        long.member_size("context", 0) as u32,
        long.member_size("steps", 0),
    );
    for (step, (digest, id)) in logits.iter().zip(&ids).enumerate() {
        assert_eq!(
            digest,
            &long.find("logits").unwrap().items()[step].str(),
            "long prompt: logits after step {step}"
        );
        assert_eq!(
            *id,
            long.find("ids").unwrap().items()[step].as_size(usize::MAX),
            "long prompt: token {step}"
        );
    }
    eprintln!(
        "long prompt: {} tokens, {} steps equal",
        prompt.len(),
        ids.len() - 1
    );

    // Managed memory holds the same values: the forward fixture's bits.
    let forward = json::parse(include_bytes!("fixtures/forward.json")).unwrap();
    let case = forward
        .find("cases")
        .unwrap()
        .items()
        .iter()
        .find(|case| case.member_str("name", "") == "p300")
        .unwrap()
        .clone();
    engine.backend.force_managed_kv_cache();
    let mut prompt = Vec::new();
    engine.vocab.encode_chat(
        b"",
        case.find("text").unwrap().str().as_bytes(),
        &mut prompt,
    );
    let (logits, ids) = run(
        &mut engine,
        &prompt,
        case.member_size("context", 0) as u32,
        case.member_size("steps", 0),
    );
    for (step, (digest, id)) in logits.iter().zip(&ids).enumerate() {
        assert_eq!(
            digest,
            &case.find("logits").unwrap().items()[step].str(),
            "managed KV: logits after step {step}"
        );
        assert_eq!(
            *id,
            case.find("ids").unwrap().items()[step].as_size(usize::MAX),
            "managed KV: token {step}"
        );
    }
    eprintln!("managed KV: {} steps equal", ids.len() - 1);
}

/// Writes `fixtures/routes.json` from the current runtime.
#[test]
#[ignore = "records the fixture; run alone"]
fn record_routes() {
    let mut engine = Engine::open(&common::model_path(), None).unwrap();
    // The fewest paragraphs that reach the target length.
    let (mut low, mut high) = (1, 2);
    while long_prompt(&engine, high).len() < LONG_PROMPT_TOKENS {
        (low, high) = (high, high * 2);
    }
    while low < high {
        let middle = (low + high) / 2;
        if long_prompt(&engine, middle).len() < LONG_PROMPT_TOKENS {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let paragraphs = low;
    let prompt = long_prompt(&engine, paragraphs);
    assert!(prompt.len() + LONG_STEPS < LONG_CONTEXT as usize);
    let (logits, ids) = run(&mut engine, &prompt, LONG_CONTEXT, LONG_STEPS);
    let long = Value::object()
        .with("paragraphs", paragraphs)
        .with("prompt_tokens", prompt.len())
        .with("prompt", common::fnv(&common::token_bytes(&prompt)))
        .with("context", LONG_CONTEXT)
        .with("steps", LONG_STEPS)
        .with(
            "logits",
            Value::from(logits.into_iter().map(Value::from).collect::<Vec<_>>()),
        )
        .with(
            "ids",
            Value::from(ids.into_iter().map(Value::from).collect::<Vec<_>>()),
        );
    let fixture = Value::object()
        .with("source", "the Rust runtime on ROCm 10.1 (tests/routes.rs)")
        .with("long", long);
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/routes.json");
    std::fs::write(path, fixture.dump().unwrap() + "\n").unwrap();
    eprintln!("recorded {path}: {} prompt tokens", prompt.len());
}
