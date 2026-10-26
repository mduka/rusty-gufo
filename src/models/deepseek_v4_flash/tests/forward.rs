//! Prefill and decode against the C++ engine (`tests/fixtures/forward.json`,
//! recorded by `tests/reference/record_forward.py`): per case the chat prompt's token
//! IDs, the FNV-1a digest of every logit row and the greedy token IDs, for a
//! short prompt, prompts of about 300, 1,500 and 2,600 tokens (the last
//! past the indexer's 512 compressed rows), one of about 5,000 tokens in two
//! chunks, and a cached prefix extended token by token and by a resumed
//! prefill; then three and eight sessions decoding together
//! (`Session::eval_batch`); then snapshots after two greedy steps, which must
//! be byte-identical, restored into the same and a larger context and
//! decoded further. Needs the model and a gfx1151.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::HashMap;

use gufo_core::json;
use gufo_deepseek_v4_flash::engine::Engine;
use gufo_deepseek_v4_flash::session::Session;

fn digest(logits: &[f32]) -> String {
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    common::fnv(&bytes)
}

#[test]
fn prefill_and_decode_match_the_cpp_engine() {
    let fixture = json::parse(include_bytes!("fixtures/forward.json")).unwrap();
    let mut engine = Engine::open(&common::model_path(), None).unwrap();
    let mut prompts: HashMap<String, Vec<i32>> = HashMap::new();
    for case in fixture.find("cases").unwrap().items() {
        let name = case.member_str("name", "");
        let prompt = match case.find("text") {
            Some(text) => {
                let mut tokens = Vec::new();
                engine
                    .vocab
                    .encode_chat(b"", text.str().as_bytes(), &mut tokens);
                tokens
            }
            None => prompts[&case.member_str("extend_from", "")].clone(),
        };
        assert_eq!(
            prompt.len(),
            case.member_size("prompt_tokens", 0),
            "{name}: prompt length"
        );
        assert_eq!(
            common::fnv(&common::token_bytes(&prompt)),
            case.member_str("prompt", ""),
            "{name}: prompt tokens"
        );
        prompts.insert(name.clone(), prompt.clone());
        let context = case.member_size("context", 0) as u32;
        let mut session = engine.create_session(context, false).unwrap();
        if let Some(cached) = case.find("cached") {
            session
                .sync(&mut engine, &prompt[..cached.as_size(0)])
                .unwrap();
        }
        session.sync(&mut engine, &prompt).unwrap();
        let ids = case.find("ids").unwrap().items();
        let logits = case.find("logits").unwrap().items();
        let steps = case.member_size("steps", 0);
        for step in 0..=steps {
            assert_eq!(
                digest(session.logits().unwrap()),
                logits[step].str(),
                "{name}: logits after step {step}"
            );
            let next = session.argmax();
            assert_eq!(
                next as usize,
                ids[step].as_size(usize::MAX),
                "{name}: token {step}"
            );
            if step < steps {
                session.eval(&mut engine, next).unwrap();
            }
        }
        eprintln!(
            "{name}: {} prompt tokens, {steps} steps equal",
            prompt.len()
        );
    }
    for batch in fixture.find("batches").unwrap().items() {
        let name = batch.member_str("name", "");
        let context = batch.member_size("context", 0) as u32;
        let steps = batch.member_size("steps", 0);
        let expected = batch.find("sessions").unwrap().items();
        let mut sessions = Vec::new();
        for (text, want) in batch.find("texts").unwrap().items().iter().zip(expected) {
            let mut prompt = Vec::new();
            engine
                .vocab
                .encode_chat(b"", text.str().as_bytes(), &mut prompt);
            assert_eq!(
                common::fnv(&common::token_bytes(&prompt)),
                want.member_str("prompt", ""),
                "{name}: prompt tokens"
            );
            let mut session = engine.create_session(context, false).unwrap();
            session.sync(&mut engine, &prompt).unwrap();
            sessions.push(session);
        }
        for step in 0..=steps {
            let mut tokens = Vec::new();
            for (row, (session, want)) in sessions.iter().zip(expected).enumerate() {
                assert_eq!(
                    digest(session.logits().unwrap()),
                    want.find("logits").unwrap().items()[step].str(),
                    "{name}: session {row}, logits after step {step}"
                );
                let next = session.argmax();
                assert_eq!(
                    next as usize,
                    want.find("ids").unwrap().items()[step].as_size(usize::MAX),
                    "{name}: session {row}, token {step}"
                );
                tokens.push(next);
            }
            if step < steps {
                let mut items: Vec<(&mut Session, i32)> = sessions.iter_mut().zip(tokens).collect();
                Session::eval_batch(&mut engine, &mut items).unwrap();
            }
        }
        eprintln!(
            "{name}: {} sessions, {steps} batched steps equal",
            sessions.len()
        );
    }
    for snap in fixture.find("snapshots").unwrap().items() {
        let name = snap.member_str("name", "");
        let mut prompt = Vec::new();
        engine.vocab.encode_chat(
            b"",
            snap.find("text").unwrap().str().as_bytes(),
            &mut prompt,
        );
        assert_eq!(
            common::fnv(&common::token_bytes(&prompt)),
            snap.member_str("prompt", ""),
            "{name}: prompt tokens"
        );
        let ids = snap.find("ids").unwrap().items();
        let logits = snap.find("logits").unwrap().items();
        let steps = snap.member_size("steps", 0);
        let mut saved = engine
            .create_session(snap.member_size("context", 0) as u32, false)
            .unwrap();
        saved.sync(&mut engine, &prompt).unwrap();
        for id in &ids[..2] {
            let next = saved.argmax();
            assert_eq!(
                next as usize,
                id.as_size(usize::MAX),
                "{name}: token before the snapshot"
            );
            saved.eval(&mut engine, next).unwrap();
        }
        let payload = saved.save_snapshot(&engine).unwrap();
        assert_eq!(
            payload.len() as u64,
            saved.payload_bytes(),
            "{name}: payload size"
        );
        assert_eq!(
            payload.len(),
            snap.member_size("payload_bytes", 0),
            "{name}: payload size"
        );
        assert_eq!(
            common::fnv(&payload),
            snap.member_str("payload", ""),
            "{name}: payload bytes"
        );
        drop(saved);
        let mut restored = engine
            .create_session(snap.member_size("restore_context", 0) as u32, false)
            .unwrap();
        // A damaged payload is refused, and the session restores afterwards.
        let error = restored
            .restore_snapshot(&engine, &payload[..payload.len() - 1])
            .unwrap_err();
        assert_eq!(error.message, "truncated session payload");
        let mut extended = payload.clone();
        extended.push(0);
        let error = restored.restore_snapshot(&engine, &extended).unwrap_err();
        assert_eq!(error.message, "KV checkpoint has trailing payload bytes");
        assert!(restored.logits().is_none());
        restored.restore_snapshot(&engine, &payload).unwrap();
        assert_eq!(restored.position(), prompt.len() + 2);
        for step in 0..=steps {
            assert_eq!(
                digest(restored.logits().unwrap()),
                logits[step].str(),
                "{name}: logits after restored step {step}"
            );
            let next = restored.argmax();
            assert_eq!(
                next as usize,
                ids[2 + step].as_size(usize::MAX),
                "{name}: restored token {step}"
            );
            if step < steps {
                restored.eval(&mut engine, next).unwrap();
            }
        }
        eprintln!(
            "{name}: {} payload bytes equal, {steps} restored steps equal",
            payload.len()
        );
    }
}
