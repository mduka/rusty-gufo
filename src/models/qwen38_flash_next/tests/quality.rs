//! The QUALITY checks of Qwen3.8 Flash-Next on the Rust engine, ported from
//! `tests/models/qwen38_flash_next/session_test.cpp`: a peer snapshot during
//! decode graph capture, independent execution modes, prefill chunk
//! boundaries with sampled MTP replay, seeded MTP replay across cache reuse
//! and snapshot restore, recovery from cancelled mutations, bounded rollback reuse,
//! image snapshot attachment, failure isolation in batches, batched against
//! isolated sessions at C2 to C8 with sampling, MTP against AR at depths 16
//! and 4096, and bitwise replay after a fresh load. One test runs them in
//! order on one model (two 80 GiB loads must not overlap):
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next --features gpu \
//!   --test quality -- --ignored --nocapture
//!
//! `GUFO_FLASH_NEXT_QUALITY` = `batch`, `prefill` or `replay` runs that part
//! with the graph and mode checks (as the C++ `--batch-only`,
//! `--prefill-only`), `cache` only the cache replay (`--cache-only`);
//! `GUFO_FLASH_NEXT_MMPROJ` names the vision sidecar.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use gufo_core::image::Image;
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen::vision::{IMAGE_TOKEN, ImageGrid, PreparedImage, Prompt, RopeLayout};
use gufo_qwen38_flash_next::engine::{
    AdvanceRequest, DecodeRequest, DecodeResult, Model, ModelOptions, Session,
};
use gufo_qwen38_flash_next::executor::SessionMode;

mod common;

use common::{CONTEXT, mtp_path, trunk_path};

fn load() -> Model {
    let vision = common::vision_sidecar();
    Model::load(
        &trunk_path(),
        ModelOptions {
            max_context: CONTEXT,
            mtp_model_path: Some(mtp_path()),
            max_draft_tokens: 7,
            vision_model_path: vision.is_file().then_some(vision),
            ..ModelOptions::default()
        },
    )
    .unwrap()
}

fn speculative(model: &Model) -> SessionMode {
    if model.has_mtp() {
        SessionMode::Speculative
    } else {
        SessionMode::Autoregressive
    }
}

fn exact(expected: &[f32], actual: &[f32], what: &str) {
    assert!(
        expected.len() == actual.len()
            && expected
                .iter()
                .zip(actual)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{what}"
    );
}

/// The first largest logit (`std::max_element`).
fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as i32
}

fn history(tokens: &[i32]) -> Vec<u32> {
    tokens.iter().map(|&t| t as u32).collect()
}

fn repeat(pattern: &[i32], len: usize) -> Vec<i32> {
    (0..len).map(|i| pattern[i % pattern.len()]).collect()
}

/// A cancellation check that cancels from its `at`-th call on.
fn cancel_at(at: u32) -> Arc<dyn Fn() -> bool + Send + Sync> {
    let calls = AtomicU32::new(0);
    Arc::new(move || calls.fetch_add(1, Ordering::Relaxed) + 1 >= at)
}

fn snapshot_during_graph_capture(model: &mut Model) {
    let mode = SessionMode::Autoregressive;
    let mut decoding = model.create_session(mode, 128).unwrap();
    let mut reference = model.create_session(mode, 128).unwrap();
    let frozen = Arc::new(Mutex::new(model.create_session(mode, 128).unwrap()));
    let prompt = model.tokenize("Continue: red, blue, red, blue,");
    model.sync(&mut decoding, &prompt).unwrap();
    model.sync(&mut reference, &prompt).unwrap();
    model.sync(&mut frozen.lock().unwrap(), &prompt).unwrap();
    let expected = model.save_snapshot(&frozen.lock().unwrap()).unwrap();
    // Warm the one-token shape; its next execution records the decode graph.
    model.evaluate(&mut decoding, prompt[0]).unwrap();
    model.evaluate(&mut reference, prompt[0]).unwrap();
    let snapshotter = Arc::new(model.snapshotter().unwrap());
    let concurrent = Arc::new(Mutex::new(None));
    let calls = Arc::new(AtomicU32::new(0));
    {
        let (frozen, snapshotter, concurrent, calls) = (
            Arc::clone(&frozen),
            Arc::clone(&snapshotter),
            Arc::clone(&concurrent),
            Arc::clone(&calls),
        );
        decoding.set_cancellation_check(Some(Arc::new(move || {
            // The forward checks once before capture and again at its first
            // layer.
            if calls.fetch_add(1, Ordering::Relaxed) + 1 == 2 {
                let (frozen, snapshotter) = (Arc::clone(&frozen), Arc::clone(&snapshotter));
                let saved = std::thread::spawn(move || snapshotter.save(&frozen.lock().unwrap()))
                    .join()
                    .unwrap();
                *concurrent.lock().unwrap() = Some(saved);
            }
            false
        })));
    }
    model
        .evaluate(&mut decoding, *prompt.last().unwrap())
        .unwrap();
    decoding.set_cancellation_check(None);
    let saved = concurrent.lock().unwrap().take().unwrap().unwrap();
    assert!(
        saved == expected,
        "decode graph capture changed a peer snapshot"
    );
    model
        .evaluate(&mut reference, *prompt.last().unwrap())
        .unwrap();
    exact(
        reference.logits(),
        decoding.logits(),
        "snapshot capture changed decode logits",
    );
    model.evaluate(&mut decoding, prompt[0]).unwrap();
    model.evaluate(&mut reference, prompt[0]).unwrap();
    exact(
        reference.logits(),
        decoding.logits(),
        "snapshot capture changed graph replay logits",
    );
    println!("peer snapshot bytes, graph capture and replay logits exact");
}

fn execution_modes(model: &mut Model) {
    let mut ar = model
        .create_session(SessionMode::Autoregressive, 128)
        .unwrap();
    let mut mtp = model.create_session(SessionMode::Speculative, 128).unwrap();
    assert!(
        ar.allocated_bytes() < mtp.allocated_bytes(),
        "AR allocated predictor state"
    );
    let prompt = model.tokenize("Continue: red, blue, red, blue,");
    model.sync(&mut ar, &prompt).unwrap();
    model.sync(&mut mtp, &prompt).unwrap();
    exact(
        ar.logits(),
        mtp.logits(),
        "execution mode changes target prefill logits",
    );
    let ar_snapshot = model.save_snapshot(&ar).unwrap();
    let mtp_snapshot = model.save_snapshot(&mtp).unwrap();
    assert!(
        ar_snapshot.len() < mtp_snapshot.len(),
        "AR snapshot includes predictor state"
    );
    let anchor = argmax(ar.logits());
    let results = model.evaluate_batch(&mut [
        AdvanceRequest {
            session: &mut ar,
            token: anchor,
        },
        AdvanceRequest {
            session: &mut mtp,
            token: anchor,
        },
    ]);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    exact(
        ar.logits(),
        mtp.logits(),
        "mixed execution modes contaminate target logits",
    );
    let mut sampler = SamplerState::new(SamplingConfig::default(), &[]).unwrap();
    let step = model.decode_step(&mut ar, 8, &mut sampler, false).unwrap();
    assert!(
        step.tokens.len() == 1 && ar.statistics().drafted == 0,
        "AR session with a resident sidecar executed speculative decoding"
    );
    assert!(
        model.restore_snapshot(&mut mtp, &ar_snapshot).is_err()
            && model.restore_snapshot(&mut ar, &mtp_snapshot).is_err(),
        "snapshots crossed execution modes"
    );
    println!("execution_modes=independent AR_predictor_bytes=0 mixed_batch_exact=1");
}

fn prefill_chunks(model: &mut Model) {
    let pattern = model.tokenize(
        "Virtual memory maps pages to physical storage. \
         A train travels sixty kilometers per hour. \
         Continue red, green, blue, red, green, blue. ",
    );
    let mode = speculative(model);
    for (length, boundary) in [
        (136, 94),
        (136, 103),
        (136, 104),
        (136, 127),
        (136, 128),
        (136, 135),
        (2048, 1025),
        (4096, 2048),
    ] {
        let tokens = repeat(&pattern, length);
        let mut bulk = model.create_session(mode, CONTEXT).unwrap();
        let mut split = model.create_session(mode, CONTEXT).unwrap();
        model.sync(&mut bulk, &tokens).unwrap();
        model.sync(&mut split, &tokens[..boundary]).unwrap();
        model.sync(&mut split, &tokens).unwrap();
        let mut generated = tokens.clone();
        for step in 0..4 {
            exact(
                bulk.logits(),
                split.logits(),
                &format!(
                    "prefill chunking changed logits: length={length} boundary={boundary} step={step}"
                ),
            );
            let token = argmax(bulk.logits());
            model.evaluate(&mut bulk, token).unwrap();
            model.evaluate(&mut split, token).unwrap();
            generated.push(token);
        }
        if model.has_mtp() {
            let config = SamplingConfig {
                temperature: 0.8,
                top_k: 20,
                top_p: 0.95,
                seed: 73,
                ..SamplingConfig::default()
            };
            let mut a = SamplerState::new(config.clone(), &history(&generated)).unwrap();
            let mut b = SamplerState::new(config, &history(&generated)).unwrap();
            let mut count = 0;
            while count < 16 {
                let first = model
                    .decode_step(&mut bulk, 16 - count, &mut a, false)
                    .unwrap();
                let second = model
                    .decode_step(&mut split, 16 - count, &mut b, false)
                    .unwrap();
                assert!(
                    !first.tokens.is_empty()
                        && first.tokens == second.tokens
                        && a.rng_state() == b.rng_state(),
                    "prefill chunking changed sampled MTP replay: length={length} boundary={boundary}"
                );
                exact(
                    bulk.logits(),
                    split.logits(),
                    "prefill chunking changed MTP target logits",
                );
                count += first.tokens.len();
            }
        }
        println!(
            "prefill chunks: length={length} boundary={boundary} four logit rows and sampled \
             MTP replay exact"
        );
    }
}

/// Seeded MTP replay does not depend on how the prompt reached the cache:
/// 200 sampled tokens, their widths and the RNG stay the same whether a
/// previous turn supplied the first 320 tokens or all but the generation
/// suffix, and after restoring the snapshot.
fn mtp_cache_replay(model: &mut Model) {
    use gufo_core::text::{ChatMessage, ChatRole};
    use gufo_qwen::chat_template::{ChatTemplateOptions, render_and_tokenize};
    if !model.has_mtp() {
        println!("MTP cache replay: no MTP sidecar");
        return;
    }
    for seed in [1i64, 2] {
        let mut system = format!("Case {seed}. ");
        for _ in 0..32 {
            system += "Follow the user instruction carefully and answer accurately. ";
        }
        let messages = [
            ChatMessage::new(ChatRole::System, system),
            ChatMessage::new(
                ChatRole::User,
                "Name three rivers in Europe and one fact about each.",
            ),
            ChatMessage::new(ChatRole::Assistant, "The Danube, the Rhine and the Loire."),
            ChatMessage::new(ChatRole::User, "Now write a short story about one of them."),
        ];
        let options = ChatTemplateOptions {
            enable_thinking: false,
            ..ChatTemplateOptions::default()
        };
        let prompt: Vec<i32> = render_and_tokenize(model.tokenizer(), &messages, &[], &options)
            .unwrap()
            .into_iter()
            .map(|t| t as i32)
            .collect();
        assert_eq!(prompt.len(), 355, "MTP cache replay fixture changed");
        let config = SamplingConfig {
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
            seed,
            ..SamplingConfig::default()
        };
        let mut expected: Option<(Vec<i32>, Vec<usize>, _)> = None;
        // The server's immutable frontier excludes the seven-token
        // generation suffix; rebuilding the prompt must not depend on
        // whether a previous turn supplied its first 320 tokens.
        for boundary in [320, 348] {
            let mut session = model
                .create_session(SessionMode::Speculative, CONTEXT)
                .unwrap();
            model.sync(&mut session, &prompt[..boundary]).unwrap();
            model.sync(&mut session, &prompt).unwrap();
            let snapshot = model.save_snapshot(&session).unwrap();
            for restore in [false, true] {
                if restore {
                    model.restore_snapshot(&mut session, &snapshot).unwrap();
                }
                session.reset_draft_policy();
                let mut sampler = SamplerState::new(config.clone(), &history(&prompt)).unwrap();
                let (mut tokens, mut widths) = (Vec::new(), Vec::new());
                while tokens.len() < 200 {
                    let step = model
                        .decode_step(&mut session, 200 - tokens.len(), &mut sampler, false)
                        .unwrap();
                    assert!(!step.tokens.is_empty(), "MTP replay made no progress");
                    widths.push(step.tokens.len());
                    tokens.extend_from_slice(&step.tokens);
                }
                let actual = (tokens, widths, sampler.rng_state());
                match &expected {
                    None => expected = Some(actual),
                    Some(expected) => assert!(
                        *expected == actual,
                        "MTP cache replay changed tokens, acceptance or RNG: seed={seed} \
                         boundary={boundary} restore={restore}"
                    ),
                }
                println!(
                    "MTP cache replay seed={seed} boundary={boundary} restore={restore} \
                     tokens=200 exact=1"
                );
            }
        }
    }
}

fn failure_recovery(model: &mut Model) {
    let mut session = model.create_session(speculative(model), 4096).unwrap();
    let initial = session.allocated_bytes();
    let c = model.config();
    let checkpoint = (c.num_layers - c.num_layers / c.full_attention_interval) as usize
        * c.ssm_num_v_heads as usize
        * (c.ssm_head_dim as usize).pow(2)
        * 4;
    assert!(
        model.session_bytes(SessionMode::Speculative, 4096) > initial + checkpoint,
        "unused deep rollback state was allocated eagerly"
    );
    let pattern = model.tokenize("Explain virtual memory in a short sentence.");
    let prompt = repeat(&pattern, 2048);
    model.sync(&mut session, &prompt).unwrap();
    let expected = session.logits().to_vec();
    let snapshot = model.save_snapshot(&session).unwrap();
    assert!(
        snapshot.len() < 200 << 20,
        "short prompt snapshot retains a full prefill hidden buffer"
    );
    session.set_cancellation_check(Some(cancel_at(6)));
    assert!(
        model.evaluate(&mut session, prompt[0]).is_err() && !session.is_valid(),
        "cancelled mutation left a reusable session"
    );
    assert!(
        session.tokens().is_empty()
            && session.logits().is_empty()
            && model.save_snapshot(&session).is_err(),
        "poisoned state exposed reusable tokens, logits or a snapshot"
    );
    session.set_cancellation_check(None);
    model.sync(&mut session, &prompt).unwrap();
    assert!(session.is_valid());
    exact(
        &expected,
        session.logits(),
        "Sync reused partially mutated state",
    );
    session.set_cancellation_check(Some(cancel_at(3)));
    assert!(
        model.restore_snapshot(&mut session, &snapshot).is_err(),
        "snapshot restoration ignored cancellation"
    );
    session.set_cancellation_check(None);
    model.sync(&mut session, &prompt).unwrap();
    exact(
        &expected,
        session.logits(),
        "cancelled restore leaked device state",
    );
    assert_eq!(
        session.allocated_bytes(),
        initial,
        "reset retained rollback allocations"
    );
    println!("cancelled forward/restore recover exactly; rollback starts empty");
}

fn rollback_reuse(model: &mut Model) {
    let prompt = model.tokenize("Continue: red, blue, red, blue,");
    let context = prompt.len() as u32 + 4;
    let mut session = model.create_session(speculative(model), context).unwrap();
    let empty = session.allocated_bytes();
    model.sync(&mut session, &prompt).unwrap();
    let config = SamplingConfig {
        seed: 73,
        ..SamplingConfig::default()
    };
    let mut sampler = SamplerState::new(config, &history(&prompt)).unwrap();
    model
        .decode_step(&mut session, 2, &mut sampler, false)
        .unwrap();
    let warm_bytes = session.allocated_bytes();
    assert!(
        warm_bytes > empty && warm_bytes <= model.session_bytes(SessionMode::Speculative, context),
        "rollback allocation exceeded its configured bound"
    );
    session.reset_draft_policy();
    let warm = model.save_snapshot(&session).unwrap();
    model.restore_snapshot(&mut session, &warm).unwrap();
    assert_eq!(
        session.allocated_bytes(),
        warm_bytes,
        "restore discarded reusable rollback or allocated deeper rows"
    );
    let mut replay_sampler = sampler.clone();
    let expected = model
        .decode_step(&mut session, 2, &mut sampler, false)
        .unwrap();
    let logits = session.logits().to_vec();
    model.restore_snapshot(&mut session, &warm).unwrap();
    let actual = model
        .decode_step(&mut session, 2, &mut replay_sampler, false)
        .unwrap();
    assert!(
        actual.tokens == expected.tokens && sampler.rng_state() == replay_sampler.rng_state(),
        "reused rollback changed continuation replay"
    );
    exact(&logits, session.logits(), "reused rollback changed logits");
    while session.position() < context {
        model.evaluate(&mut session, prompt[0]).unwrap();
    }
    let full = model.save_snapshot(&session).unwrap();
    model.restore_snapshot(&mut session, &full).unwrap();
    assert_eq!(
        session.allocated_bytes(),
        empty,
        "restore kept rollback that the restored context cannot use"
    );
    model.reset(&mut session);
    assert_eq!(
        session.allocated_bytes(),
        empty,
        "explicit reset retained rollback"
    );
    println!("rollback restore is bounded, reuses warm rows and replays exactly");
}

fn image_snapshot_attachment(model: &mut Model) {
    if model.vision_encoder().is_none() {
        println!("image snapshot attachment: matching vision sidecar unavailable");
        return;
    }
    let grid = ImageGrid {
        offset: 0,
        height: 2,
        width: 2,
    };
    let image = |blue: bool| {
        let mut pixels = vec![0u8; 64 * 64 * 3];
        for value in pixels.iter_mut().skip(if blue { 2 } else { 0 }).step_by(3) {
            *value = 255;
        }
        let identity = [if blue { 2 } else { 1 }; 32];
        Arc::new(Prompt {
            tokens: vec![IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 42],
            stable_prefix_tokens: 5,
            rope: RopeLayout { images: vec![grid] },
            images: vec![Arc::new(PreparedImage {
                pixels: Image::new(64, 64, pixels).unwrap().into(),
                grid,
                prefix_identity: identity,
            })],
            cache_identity: identity.to_vec(),
        })
    };
    let (red, blue) = (image(false), image(true));
    let tokens: Vec<i32> = red.tokens.iter().map(|&t| t as i32).collect();
    let mode = speculative(model);
    let mut origin = model.create_session(mode, 64).unwrap();
    let mut restored = model.create_session(mode, 64).unwrap();
    let mut text = model.create_session(mode, 64).unwrap();
    model
        .configure_vision(&mut origin, Some(Arc::clone(&red)))
        .unwrap();
    model.sync(&mut origin, &tokens[..2]).unwrap();
    let snapshot = model.save_snapshot(&origin).unwrap();
    assert!(
        model.restore_snapshot(&mut restored, &snapshot).is_err(),
        "image snapshot restored without pixels"
    );
    model
        .configure_vision(&mut restored, Some(Arc::clone(&blue)))
        .unwrap();
    model.sync(&mut restored, &tokens[..2]).unwrap();
    assert!(
        model.restore_snapshot(&mut restored, &snapshot).is_err(),
        "image snapshot inherited another image's pixels"
    );
    model
        .configure_vision(&mut restored, Some(Arc::clone(&red)))
        .unwrap();
    model.restore_snapshot(&mut restored, &snapshot).unwrap();
    model.sync(&mut restored, &tokens).unwrap();
    model.sync(&mut origin, &tokens).unwrap();
    exact(
        origin.logits(),
        restored.logits(),
        "attached image snapshot changed continuation logits",
    );
    model.sync(&mut text, &[42, 43]).unwrap();
    let text_snapshot = model.save_snapshot(&text).unwrap();
    model
        .restore_snapshot(&mut restored, &text_snapshot)
        .unwrap();
    model.evaluate(&mut text, 44).unwrap();
    model.evaluate(&mut restored, 44).unwrap();
    exact(
        text.logits(),
        restored.logits(),
        "text snapshot retained stale image input",
    );
    println!("image snapshot attachment, replacement and text restore exact");
}

fn batch_failure_isolation(model: &mut Model) {
    let prompt = model.tokenize("Repeat red, blue, red, blue,");
    let token = prompt[0];
    for mode in [SessionMode::Autoregressive, SessionMode::Speculative] {
        let mut first = model.create_session(mode, 256).unwrap();
        let mut second = model.create_session(mode, 256).unwrap();
        model.sync(&mut first, &prompt).unwrap();
        let snapshot = model.save_snapshot(&first).unwrap();
        model.restore_snapshot(&mut second, &snapshot).unwrap();
        let advance = |model: &mut Model, first: &mut Session, second: &mut Session, a: i32| {
            model.evaluate_batch(&mut [
                AdvanceRequest {
                    session: first,
                    token: a,
                },
                AdvanceRequest {
                    session: second,
                    token,
                },
            ])
        };
        assert!(
            advance(model, &mut first, &mut second, token)
                .iter()
                .all(Result::is_ok)
        );
        let expected = second.logits().to_vec();
        let restore = |model: &mut Model, first: &mut Session, second: &mut Session| {
            model.restore_snapshot(first, &snapshot).unwrap();
            model.restore_snapshot(second, &snapshot).unwrap();
        };
        restore(model, &mut first, &mut second);
        first.set_cancellation_check(Some(cancel_at(12)));
        let outcomes = advance(model, &mut first, &mut second, token);
        assert!(
            outcomes[0].is_err() && outcomes[1].is_ok() && second.is_valid() && !first.is_valid(),
            "cancellation must invalidate only the partially executed session: {outcomes:?}"
        );
        exact(
            &expected,
            second.logits(),
            "cancelled peer changed target logits",
        );
        first.set_cancellation_check(None);
        restore(model, &mut first, &mut second);
        let outcomes = advance(model, &mut first, &mut second, -1);
        assert!(
            outcomes[0].is_err() && first.is_valid() && second.is_valid() && outcomes[1].is_ok(),
            "invalid preparation poisoned a peer or the untouched session: {outcomes:?}"
        );
        exact(
            &expected,
            second.logits(),
            "invalid peer changed target logits",
        );
        restore(model, &mut first, &mut second);
        let config = SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            seed: 73,
            ..SamplingConfig::default()
        };
        let sampler = || SamplerState::new(config.clone(), &[]).unwrap();
        let (mut control, mut a, mut b) = (sampler(), sampler(), sampler());
        let reference = model
            .decode_step(&mut second, 3, &mut control, false)
            .unwrap();
        let speculative_logits = second.logits().to_vec();
        model.restore_snapshot(&mut second, &snapshot).unwrap();
        let outcomes = model.decode_batch(&mut [
            DecodeRequest {
                session: &mut first,
                max_tokens: 0,
                sampler: &mut a,
                stop_at_eos: false,
            },
            DecodeRequest {
                session: &mut second,
                max_tokens: 3,
                sampler: &mut b,
                stop_at_eos: false,
            },
        ]);
        assert!(
            outcomes[0].is_err()
                && first.is_valid()
                && outcomes[1]
                    .as_ref()
                    .is_ok_and(|r| r.tokens == reference.tokens)
                && b.rng_state() == control.rng_state(),
            "invalid decode peer changed sampling replay: {outcomes:?}"
        );
        exact(
            &speculative_logits,
            second.logits(),
            "invalid decode peer changed logits",
        );
        if mode == SessionMode::Speculative {
            restore(model, &mut first, &mut second);
            let (mut a, mut b) = (sampler(), sampler());
            first.set_cancellation_check(Some(cancel_at(6)));
            let outcomes = model.decode_batch(&mut [
                DecodeRequest {
                    session: &mut first,
                    max_tokens: 3,
                    sampler: &mut a,
                    stop_at_eos: false,
                },
                DecodeRequest {
                    session: &mut second,
                    max_tokens: 3,
                    sampler: &mut b,
                    stop_at_eos: false,
                },
            ]);
            assert!(
                outcomes[0].is_err()
                    && second.is_valid()
                    && outcomes[1]
                        .as_ref()
                        .is_ok_and(|r| r.tokens == reference.tokens)
                    && b.rng_state() == control.rng_state(),
                "cancelled MTP peer changed sampling replay: {outcomes:?}"
            );
            exact(
                &speculative_logits,
                second.logits(),
                "cancelled MTP peer changed logits",
            );
            first.set_cancellation_check(None);
        }
    }
    println!("batch preparation and mid-execution cancellation isolate peers and sampling replay");
}

fn batched_sessions(model: &mut Model) {
    let mode = speculative(model);
    let cases: Vec<SamplingConfig> = common::sampling_cases()
        .into_iter()
        .map(|(_, c)| c)
        .collect();
    let (mut serial, mut batched) = (Vec::new(), Vec::new());
    let (mut serial_samplers, mut batch_samplers) = (Vec::new(), Vec::new());
    for i in 0..8 {
        let mut prompt = model.tokenize(&format!(
            "Batch request {i}: Continue red, blue, blue, red,"
        ));
        // One member crosses the sparse-attention boundary while the others
        // remain short; interleaving must not share positions or pooling.
        if i == 1 {
            prompt = repeat(&prompt, 4095);
        }
        let mut s = model.create_session(mode, CONTEXT).unwrap();
        let mut b = model.create_session(mode, CONTEXT).unwrap();
        model.sync(&mut s, &prompt).unwrap();
        let snapshot = model.save_snapshot(&s).unwrap();
        model.restore_snapshot(&mut b, &snapshot).unwrap();
        let mut config = cases[(i * 3) % cases.len()].clone();
        // Greedy members with penalties verify on the device.
        if i % 2 == 0 {
            config.temperature = 0.0;
            config.repeat_penalty = if i == 0 { 0.7 } else { 1.3 };
            config.repeat_last_n = 3;
            config.frequency_penalty = if i == 0 { -0.4 } else { 0.2 };
            config.presence_penalty = if i == 0 { -1.5 } else { 1.5 };
        }
        serial_samplers.push(SamplerState::new(config.clone(), &history(&prompt)).unwrap());
        batch_samplers.push(SamplerState::new(config, &history(&prompt)).unwrap());
        serial.push(s);
        batched.push(b);
    }
    let (mut sampled_acceptances, mut sampled_rejections) = (0u64, 0u64);
    for width in [2usize, 4, 6, 8] {
        // The AR entry point, reusing cohorts at other tokens and positions.
        for _ in 0..2 {
            let mut tokens = Vec::new();
            for i in 0..width {
                let token = argmax(serial[i].logits());
                model.evaluate(&mut serial[i], token).unwrap();
                serial_samplers[i].accept(token as u32).unwrap();
                batch_samplers[i].accept(token as u32).unwrap();
                tokens.push(token);
            }
            let mut advances: Vec<AdvanceRequest<'_>> = batched[..width]
                .iter_mut()
                .zip(&tokens)
                .map(|(session, &token)| AdvanceRequest { session, token })
                .collect();
            assert!(
                model
                    .evaluate_batch(&mut advances)
                    .iter()
                    .all(Result::is_ok)
            );
            drop(advances);
            for i in 0..width {
                exact(
                    serial[i].logits(),
                    batched[i].logits(),
                    &format!("batched AR frontier differs at C{width} row {i}"),
                );
            }
        }
        // Different budgets make ragged chains and mix one-token decoding
        // with speculative verification in one batch.
        for cycle in 0..2usize {
            let budgets: Vec<usize> = (0..width)
                .map(|i| 1 + (i * 3 + cycle * 5 + 7) % 8)
                .collect();
            let mut expected = Vec::new();
            let mut before = Vec::new();
            for i in 0..width {
                before.push(serial[i].statistics());
                expected.push(
                    model
                        .decode_step(&mut serial[i], budgets[i], &mut serial_samplers[i], false)
                        .unwrap(),
                );
            }
            let mut order: Vec<usize> = (0..width).collect();
            if cycle % 2 != 0 {
                order.reverse();
            }
            let mut slots: Vec<Option<(&mut Session, &mut SamplerState)>> = batched[..width]
                .iter_mut()
                .zip(batch_samplers[..width].iter_mut())
                .map(Some)
                .collect();
            let mut requests = Vec::new();
            for &i in &order {
                let (session, sampler) = slots[i].take().unwrap();
                requests.push(DecodeRequest {
                    session,
                    max_tokens: budgets[i],
                    sampler,
                    stop_at_eos: false,
                });
            }
            let results = model.decode_batch(&mut requests);
            drop(requests);
            let mut actual: Vec<DecodeResult> = vec![DecodeResult::default(); width];
            for (&i, result) in order.iter().zip(results) {
                actual[i] = result.unwrap();
            }
            for i in 0..width {
                let (a, b) = (serial[i].statistics(), batched[i].statistics());
                assert!(
                    expected[i] == actual[i]
                        && serial_samplers[i].rng_state() == batch_samplers[i].rng_state()
                        && a == b
                        && serial[i].tokens() == batched[i].tokens(),
                    "batched decode tokens, RNG or acceptance differ at C{width} row {i}"
                );
                exact(
                    serial[i].logits(),
                    batched[i].logits(),
                    &format!("batched MTP frontier differs at C{width} row {i} cycle {cycle}"),
                );
                assert!(
                    serial_samplers[i].history() == batch_samplers[i].history(),
                    "batched sampling history differs"
                );
                // Probe copies: pending residual draws must agree too.
                let (mut serial_next, mut batch_next) =
                    (serial_samplers[i].clone(), batch_samplers[i].clone());
                assert!(
                    serial_next.sample(serial[i].logits()).unwrap()
                        == batch_next.sample(batched[i].logits()).unwrap()
                        && serial_next.rng_state() == batch_next.rng_state(),
                    "batched next draw or deferred residual differs"
                );
                if serial_samplers[i].config().uses_random_sampling() {
                    let drafted = a.drafted - before[i].drafted;
                    let accepted = a.accepted - before[i].accepted;
                    sampled_acceptances += accepted;
                    if accepted < drafted {
                        sampled_rejections += 1;
                        // A sampled rejection defers its correction; taking
                        // it must not draw again.
                        assert_eq!(
                            serial_next.rng_state(),
                            serial_samplers[i].rng_state(),
                            "sampled rejection lost its deferred correction"
                        );
                    }
                }
            }
        }
        println!("batch C{width} AR_MTP_logits_RNG_exact=1");
    }
    assert!(
        sampled_acceptances > 0 && sampled_rejections > 0,
        "batch test must exercise sampled acceptance and residual correction"
    );
    println!(
        "batch sampled_accepted={sampled_acceptances} rejected_cycles={sampled_rejections} history_and_residual_exact=1"
    );
    for i in 0..serial.len() {
        let (a, b) = (
            model.save_snapshot(&serial[i]).unwrap(),
            model.save_snapshot(&batched[i]).unwrap(),
        );
        assert!(a == b, "batched snapshot state differs: {i}");
    }
    println!("batch independent_state_exact=1");
}

/// One greedy or seeded replay kept for the fresh load.
struct Replay {
    depth: usize,
    config: usize,
    prefill_logits: Vec<f32>,
    final_logits: Vec<f32>,
    tokens: Vec<i32>,
    rng: u64,
}

fn replay_configs() -> [SamplingConfig; 5] {
    [
        SamplingConfig {
            seed: 1,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            temperature: 0.7,
            seed: 1,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            temperature: 1.0,
            top_p: 0.95,
            seed: 73,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.01,
            min_keep: 2,
            seed: 1,
            repeat_penalty: 1.1,
            repeat_last_n: 16,
            frequency_penalty: 0.2,
            presence_penalty: 0.1,
            constraint: None,
            reasoning_budget: None,
        },
        // Greedy under penalties: the device selects every verification
        // row with the sampler's FP64 penalty arithmetic.
        SamplingConfig {
            seed: 1,
            repeat_penalty: 1.3,
            repeat_last_n: 3,
            frequency_penalty: 0.2,
            presence_penalty: 1.5,
            ..SamplingConfig::default()
        },
    ]
}

/// MTP against AR at depths 16 and 4096: the same tokens, frontier logits
/// and (greedy) RNG after every MTP step, and the same proposals when the
/// MTP session replays alone.
fn mtp_against_ar(model: &mut Model, prompt: &[i32]) -> Vec<Replay> {
    let configs = replay_configs();
    let mut replays = Vec::new();
    for depth in [16usize, 4096] {
        let prefix = &prompt[..depth];
        for (c, config) in configs.iter().enumerate() {
            let sampled = config.uses_random_sampling();
            let mut ar = model
                .create_session(SessionMode::Autoregressive, CONTEXT)
                .unwrap();
            let mut mtp = model.create_session(speculative(model), CONTEXT).unwrap();
            model.sync(&mut ar, prefix).unwrap();
            // Prefill the other session before decoding either: retained
            // target hidden rows must stay private to their session.
            model.sync(&mut mtp, prefix).unwrap();
            exact(
                ar.logits(),
                mtp.logits(),
                "interleaving changed prefill logits",
            );
            let prefill_logits = ar.logits().to_vec();
            let mut a = SamplerState::new(config.clone(), &history(prefix)).unwrap();
            let mut b = SamplerState::new(config.clone(), &history(prefix)).unwrap();
            let (mut reference, mut candidate) = (Vec::new(), Vec::new());
            let mut widths = Vec::new();
            let count = if c <= 1 || c == 4 { 32 } else { 8 };
            while candidate.len() < count {
                let decoded = model
                    .decode_step(&mut mtp, count - candidate.len(), &mut b, false)
                    .unwrap();
                assert!(!decoded.tokens.is_empty(), "empty MTP step");
                widths.push(decoded.tokens.len());
                candidate.extend_from_slice(&decoded.tokens);
                while reference.len() < candidate.len() {
                    if sampled {
                        // The target replays every committed token, residual
                        // corrections included.
                        let token = candidate[reference.len()];
                        model.evaluate(&mut ar, token).unwrap();
                        reference.push(token);
                        a.accept(token as u32).unwrap();
                        continue;
                    }
                    let single = model.decode_step(&mut ar, 1, &mut a, false).unwrap();
                    reference.extend_from_slice(&single.tokens);
                }
                assert!(
                    ar.logits()
                        .iter()
                        .chain(mtp.logits())
                        .all(|v| v.is_finite()),
                    "non-finite logits"
                );
                assert_eq!(
                    reference, candidate,
                    "AR/MTP token mismatch at depth {depth} config {c}"
                );
                exact(ar.logits(), mtp.logits(), "AR/MTP frontier logits differ");
                if !sampled {
                    assert_eq!(a.rng_state(), b.rng_state(), "greedy AR/MTP RNG mismatch");
                }
                assert_eq!(ar.position(), mtp.position(), "AR/MTP position mismatch");
            }
            let stats = mtp.statistics();
            model.sync(&mut mtp, prefix).unwrap();
            assert_eq!(mtp.position() as usize, depth, "prefix reset failed");
            let mut replay_sampler = SamplerState::new(config.clone(), &history(prefix)).unwrap();
            let (mut replay, mut isolated) = (Vec::new(), Vec::new());
            while replay.len() < count {
                let decoded = model
                    .decode_step(&mut mtp, count - replay.len(), &mut replay_sampler, false)
                    .unwrap();
                assert!(!decoded.tokens.is_empty(), "empty replay step");
                isolated.push(decoded.tokens.len());
                replay.extend_from_slice(&decoded.tokens);
            }
            let after = mtp.statistics();
            assert!(
                replay == candidate
                    && isolated == widths
                    && after.drafted - stats.drafted == stats.drafted
                    && after.accepted - stats.accepted == stats.accepted,
                "interleaving changed MTP proposals or acceptance"
            );
            println!(
                "depth={depth} config={c} tokens={count} exact=1 drafted={} accepted={}",
                stats.drafted, stats.accepted
            );
            // Greedy and seeded sampling replay after a fresh load.
            if c <= 1 {
                replays.push(Replay {
                    depth,
                    config: c,
                    prefill_logits,
                    final_logits: ar.logits().to_vec(),
                    tokens: reference,
                    rng: b.rng_state(),
                });
            }
        }
    }
    replays
}

/// The replays on a freshly loaded model, in reverse order, with and without
/// speculation and varying draft budgets: bitwise the same.
fn fresh_load(model: &mut Model, prompt: &[i32], mut replays: Vec<Replay>) {
    let configs = replay_configs();
    replays.reverse();
    for saved in &replays {
        let prefix = &prompt[..saved.depth];
        let config = &configs[saved.config];
        let sampled = config.uses_random_sampling();
        for speculative_budgets in [false, true] {
            let mut session = model.create_session(speculative(model), CONTEXT).unwrap();
            model.sync(&mut session, prefix).unwrap();
            exact(
                &saved.prefill_logits,
                session.logits(),
                "fresh load changed prefill logits",
            );
            let mut sampler = SamplerState::new(config.clone(), &history(prefix)).unwrap();
            let mut tokens = Vec::new();
            let budgets = [2usize, 8, 1, 4];
            let mut step = 0;
            while tokens.len() < saved.tokens.len() {
                if sampled && !speculative_budgets {
                    let token = saved.tokens[tokens.len()];
                    model.evaluate(&mut session, token).unwrap();
                    tokens.push(token);
                    continue;
                }
                let remaining = saved.tokens.len() - tokens.len();
                let budget = if sampled {
                    remaining
                } else if speculative_budgets {
                    step += 1;
                    budgets[(step - 1) % budgets.len()]
                } else {
                    1
                };
                let decoded = model
                    .decode_step(&mut session, budget.min(remaining), &mut sampler, false)
                    .unwrap();
                assert!(!decoded.tokens.is_empty(), "empty fresh-load replay step");
                tokens.extend_from_slice(&decoded.tokens);
            }
            assert_eq!(
                tokens, saved.tokens,
                "fresh load or draft width changed tokens"
            );
            exact(
                &saved.final_logits,
                session.logits(),
                "fresh load or draft width changed final logits",
            );
            if !sampled || speculative_budgets {
                assert_eq!(
                    sampler.rng_state(),
                    saved.rng,
                    "fresh load or draft width changed RNG"
                );
            }
            println!(
                "fresh_load depth={} config={} mtp={speculative_budgets} tokens={} bitwise_exact=1",
                saved.depth,
                saved.config,
                tokens.len()
            );
        }
    }
}

#[test]
#[ignore = "QUALITY check on the Flash-Next artifact; loads the model twice"]
fn quality_checks_pass() {
    let part = std::env::var("GUFO_FLASH_NEXT_QUALITY").unwrap_or_default();
    let mut model = load();
    if part == "cache" {
        mtp_cache_replay(&mut model);
        return;
    }
    snapshot_during_graph_capture(&mut model);
    execution_modes(&mut model);
    if part != "batch" {
        prefill_chunks(&mut model);
    }
    if part == "prefill" {
        return;
    }
    if part != "replay" {
        if part != "batch" {
            mtp_cache_replay(&mut model);
        }
        failure_recovery(&mut model);
        rollback_reuse(&mut model);
        image_snapshot_attachment(&mut model);
        batch_failure_isolation(&mut model);
        batched_sessions(&mut model);
    }
    if part == "batch" {
        return;
    }
    let pattern = model.tokenize(
        "The quick brown fox jumps over the lazy dog. \
         Strix Halo executes this deterministic benchmark sequence. ",
    );
    let prompt = repeat(&pattern, 4096);
    let replays = mtp_against_ar(&mut model, &prompt);
    // Recreate device weights, executor and plans.
    drop(model);
    let mut model = load();
    fresh_load(&mut model, &prompt, replays);
}
