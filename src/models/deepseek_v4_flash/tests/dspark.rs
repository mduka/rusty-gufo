//! Greedy DSpark speculative decoding against the C++ engine (the `dspark`
//! section of `tests/fixtures/forward.json`, recorded by
//! `tests/reference/record_forward.py` with `tests/reference/ds4_ref_dump.cpp dspark`): one
//! speculative session at 274 and at 3,042 tokens, and two, three and four
//! sessions stepping together; then sampled cohorts (one request, two with
//! stochastic proposals, a greedy and a sampled one, two point-mass ones,
//! four). Every cycle's emitted tokens and logits, the
//! statistics and the snapshot payload must equal C++, and the single
//! session must continue identically after restoring its snapshot. Needs
//! the model, the support model and a gfx1151.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_core::json;
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_deepseek_v4_flash::dspark_sampler::DsparkSampler;
use gufo_deepseek_v4_flash::engine::Engine;
use gufo_deepseek_v4_flash::session::{DsparkItem, Session};

fn digest(logits: &[f32]) -> String {
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    common::fnv(&bytes)
}

fn ids(value: &json::Value) -> Vec<i32> {
    value
        .items()
        .iter()
        .map(|v| v.as_size(usize::MAX) as i32)
        .collect()
}

/// The request samplers of a case's "T,P,K,SEED;greedy;..." spec, with
/// the prompts as history.
fn request_samplers(spec: &str, prompts: &[Vec<i32>]) -> Vec<Option<SamplerState>> {
    let items: Vec<&str> = if spec.is_empty() {
        Vec::new()
    } else {
        spec.split(';').collect()
    };
    prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let item = items
                .get(index)
                .copied()
                .filter(|i| !i.is_empty() && *i != "greedy")?;
            let fields: Vec<&str> = item.split(',').collect();
            let config = SamplingConfig {
                temperature: fields[0].parse().unwrap(),
                top_p: fields[1].parse().unwrap(),
                top_k: fields[2].parse().unwrap(),
                seed: fields[3].parse().unwrap(),
                ..SamplingConfig::default()
            };
            let history: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
            Some(SamplerState::new(config, &history).unwrap())
        })
        .collect()
}

#[test]
fn speculative_cycles_match_the_cpp_engine() {
    let fixture = json::parse(include_bytes!("fixtures/forward.json")).unwrap();
    let dspark = common::dspark_path();
    let mut engine = Engine::open(&common::model_path(), Some(&dspark)).unwrap();
    let greedy = fixture.find("dspark").unwrap().items();
    let sampled = fixture.find("dspark_sampled").unwrap().items();
    for case in greedy.iter().chain(sampled) {
        let name = case.member_str("name", "");
        let context = case.member_size("context", 0) as u32;
        let cycles = case.member_size("cycles", 0);
        let max_draft = case.member_size("max_draft", 0) as u32;
        let expected = case.find("sessions").unwrap().items();
        let mut sessions = Vec::new();
        let mut prompts = Vec::new();
        for (text, want) in case.find("texts").unwrap().items().iter().zip(expected) {
            let mut prompt = Vec::new();
            engine
                .vocab
                .encode_chat(b"", text.str().as_bytes(), &mut prompt);
            assert_eq!(
                common::fnv(&common::token_bytes(&prompt)),
                want.member_str("prompt", ""),
                "{name}: prompt tokens"
            );
            let mut session = engine.create_session(context, true).unwrap();
            session.sync(&mut engine, &prompt).unwrap();
            prompts.push(prompt);
            sessions.push(session);
        }
        let mut samplers = request_samplers(&case.member_str("sampling", ""), &prompts);
        for cycle in 0..cycles {
            let mut bridges: Vec<Option<DsparkSampler>> = samplers
                .iter()
                .map(|s| s.as_ref().map(DsparkSampler::new))
                .collect();
            let mut items: Vec<DsparkItem<'_>> = sessions
                .iter_mut()
                .zip(bridges.iter_mut())
                .map(|(session, bridge)| DsparkItem {
                    session,
                    max_tokens: 32,
                    max_draft_tokens: max_draft,
                    sampler: bridge.as_mut(),
                    stop_at_eos: false,
                    emitted: Vec::new(),
                })
                .collect();
            Session::dspark_step_batch(&mut engine, &mut items).unwrap();
            let emitted: Vec<Vec<i32>> = items.into_iter().map(|item| item.emitted).collect();
            // As the serving runner: the cycle's RNG goes back to the request
            // sampler, which accepts the emitted tokens.
            for ((sampler, bridge), emitted) in samplers.iter_mut().zip(&bridges).zip(&emitted) {
                if let (Some(sampler), Some(bridge)) = (sampler, bridge) {
                    sampler.set_rng_state(bridge.rng_state());
                    for &token in emitted {
                        sampler.accept(token as u32).unwrap();
                    }
                }
            }
            for (row, ((session, want), emitted)) in
                sessions.iter().zip(expected).zip(&emitted).enumerate()
            {
                assert_eq!(
                    emitted,
                    &ids(&want.find("emitted").unwrap().items()[cycle]),
                    "{name}: session {row}, tokens of cycle {cycle}"
                );
                assert_eq!(
                    digest(session.logits().unwrap()),
                    want.find("logits").unwrap().items()[cycle].str(),
                    "{name}: session {row}, logits after cycle {cycle}"
                );
            }
        }
        for (row, (session, want)) in sessions.iter().zip(expected).enumerate() {
            let s = session.dspark_stats();
            let stats = [
                s.verifier_rows,
                s.verifier_accepted,
                s.support_drafted,
                s.support_accepted,
                s.positional_accepted,
                s.anchors,
                s.full_blocks,
                s.steps,
                s.skipped,
                u64::from(s.context_tokens),
            ];
            let want_stats: Vec<u64> = want
                .find("stats")
                .unwrap()
                .items()
                .iter()
                .map(|v| v.as_size(usize::MAX) as u64)
                .collect();
            assert_eq!(
                stats.to_vec(),
                want_stats,
                "{name}: session {row} statistics"
            );
            let payload = session.save_snapshot(&engine).unwrap();
            assert_eq!(
                payload.len(),
                want.member_size("payload_bytes", 0),
                "{name}: session {row} payload size"
            );
            assert_eq!(
                common::fnv(&payload),
                want.member_str("payload", ""),
                "{name}: session {row} payload"
            );
        }
        if let Some(restored) = case.find("restored") {
            let payload = sessions[0].save_snapshot(&engine).unwrap();
            sessions.clear();
            let mut session = engine.create_session(context, true).unwrap();
            session.restore_snapshot(&engine, &payload).unwrap();
            let logits = restored.find("logits").unwrap().items();
            for (cycle, want) in restored.find("emitted").unwrap().items().iter().enumerate() {
                let emitted = session
                    .dspark_step(&mut engine, 32, max_draft, None, false)
                    .unwrap();
                assert_eq!(emitted, ids(want), "{name}: restored cycle {cycle}");
                assert_eq!(
                    digest(session.logits().unwrap()),
                    logits[cycle].str(),
                    "{name}: logits after restored cycle {cycle}"
                );
            }
        }
        eprintln!("{name}: {} sessions, {cycles} cycles equal", expected.len());
    }
    // A session that drafted, re-prefilled from scratch for a changed
    // prefix, keeps no support context: it snapshots like a fresh session.
    let mut prompt = Vec::new();
    let text = greedy[0].find("texts").unwrap().items()[0].str();
    engine.vocab.encode_chat(b"", text.as_bytes(), &mut prompt);
    let mut other = Vec::new();
    engine.vocab.encode_chat(b"", b"Count to ten.", &mut other);
    let mut reused = engine.create_session(4096, true).unwrap();
    reused.sync(&mut engine, &prompt).unwrap();
    for _ in 0..3 {
        reused.dspark_step(&mut engine, 32, 5, None, false).unwrap();
    }
    reused.sync(&mut engine, &other).unwrap();
    assert_eq!(
        reused.dspark_stats().context_tokens as usize,
        reused.position(),
        "support context after a changed prefix"
    );
    let mut fresh = engine.create_session(4096, true).unwrap();
    fresh.sync(&mut engine, &other).unwrap();
    assert!(
        reused.save_snapshot(&engine).unwrap() == fresh.save_snapshot(&engine).unwrap(),
        "a re-prefilled session snapshots like a fresh one"
    );
}
