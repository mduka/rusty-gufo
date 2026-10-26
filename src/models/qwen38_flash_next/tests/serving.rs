//! The serving QUALITY check of Qwen3.8 Flash-Next (`CheckServingSampling`
//! in `tests/models/qwen38_flash_next/session_test.cpp`) on the Rust runner
//! behind the scheduled backend: every sampling strategy completes through
//! AR and MTP serving, replays, runs in concurrent pairs with the same
//! tokens (batched execution reported), keeps the AR and greedy prefix for
//! every budget, and hands the sampler to the session exactly as a direct
//! engine session sees it. The C++ check shares one model between both
//! backends; here the MTP backend, the AR backend and the direct engine load
//! one after another:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next --features gpu \
//!   --test serving -- --ignored --nocapture

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Barrier};

use common::{CONTEXT, mtp_path, sampling_cases, trunk_path};
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen38_flash_next::engine::{Model, ModelOptions};
use gufo_qwen38_flash_next::executor::SessionMode;
use gufo_qwen38_flash_next::runner::{FlashNextRunner, MtpOptions, RunnerOptions};
use gufo_serve::backend::{GenerationResult, TextGenerationBackend};
use gufo_serve::pool::TextRunnerPool;
use gufo_serve::scheduled_backend::ScheduledBackend;
use gufo_serve::scheduler::{TextPrefillPolicy, TextSchedulerPolicy};

const SESSIONS: usize = 2;

fn backend(mtp: bool) -> ScheduledBackend<FlashNextRunner> {
    let runner = FlashNextRunner::load(&RunnerOptions {
        model_path: trunk_path(),
        max_context: CONTEXT,
        persistence: false,
        mtp: mtp.then(|| MtpOptions {
            model_path: mtp_path(),
            max_draft_tokens: 7,
        }),
        sessions: SESSIONS,
        vision_model_path: None,
        enable_yarn: false,
    })
    .unwrap();
    let pool = TextRunnerPool::new(Arc::new(runner), SESSIONS, None).unwrap();
    ScheduledBackend::new(
        Arc::new(pool),
        TextPrefillPolicy::default(),
        TextSchedulerPolicy::default(),
    )
    .unwrap()
}

fn complete(
    backend: &ScheduledBackend<FlashNextRunner>,
    prompt: &str,
    max_tokens: usize,
    config: &SamplingConfig,
) -> GenerationResult {
    backend
        .complete(prompt, max_tokens, config, None, None, "quality", &[], true)
        .unwrap()
}

fn prompt(name: &str) -> String {
    format!("Sampling {name}: Continue red, blue, blue, red,")
}

/// Serial completions, their replays, concurrent pairs and budgets on one
/// backend; returns the serial results.
fn serve(mtp: bool) -> Vec<GenerationResult> {
    let backend = Arc::new(backend(mtp));
    let cases = sampling_cases();
    let mut results = Vec::new();
    for (name, config) in &cases {
        let result = complete(&backend, &prompt(name), 8, config);
        assert_eq!(
            result.physical_execution_width, 1,
            "serial completion reported a batched execution"
        );
        let replay = complete(&backend, &prompt(name), 8, config);
        assert!(
            replay.tokens == result.tokens
                && replay.draft_tokens == result.draft_tokens
                && replay.draft_accepted_tokens == result.draft_accepted_tokens,
            "serving replay changed sampling or draft acceptance: {name} mtp={mtp}"
        );
        println!("serving sampling={name} mtp={mtp} replay_exact=1");
        results.push(result);
    }
    // Every strategy also runs with shared projections in a concurrent pair.
    let mut batched = 0;
    for offset in (0..cases.len()).step_by(2) {
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = (0..2)
            .map(|row| {
                let index = (offset + row) % cases.len();
                let (name, config) = cases[index].clone();
                let (backend, barrier) = (Arc::clone(&backend), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    barrier.wait();
                    (index, complete(&backend, &prompt(name), 8, &config))
                })
            })
            .collect();
        for handle in handles {
            let (index, actual) = handle.join().unwrap();
            let (expected, config) = (&results[index], &cases[index].1);
            assert!(
                actual.tokens == expected.tokens
                    && (!config.uses_random_sampling()
                        || (actual.draft_tokens == expected.draft_tokens
                            && actual.draft_accepted_tokens == expected.draft_accepted_tokens)),
                "interleaving changed serving sampling or acceptance: {} mtp={mtp}",
                cases[index].0
            );
            if actual.physical_execution_width > 1 {
                assert!(
                    actual.physical_execution_width == 2 && actual.execution_plan == "batched-w2",
                    "paired completion reported an incorrect execution plan"
                );
                batched += 1;
            }
        }
    }
    assert!(
        batched > 0,
        "concurrent sampling checks did not report batched execution"
    );
    // AR and greedy prefixes do not depend on the budget; sampled MTP draws
    // for its proposals, so each budget replays its own result.
    for index in [0, cases.len() - 2] {
        let (name, config) = cases[index].clone();
        for budget in [1usize, 2, 3] {
            let actual = complete(&backend, &prompt(name), budget, &config);
            if mtp && config.uses_random_sampling() {
                let replay = complete(&backend, &prompt(name), budget, &config);
                assert!(
                    actual.tokens.len() <= budget
                        && actual.tokens == replay.tokens
                        && actual.draft_tokens == replay.draft_tokens
                        && actual.draft_accepted_tokens == replay.draft_accepted_tokens,
                    "sampled MTP budget or replay differs"
                );
            } else {
                let prefix = &results[index].tokens;
                assert!(
                    actual.tokens.len() == budget.min(prefix.len())
                        && actual.tokens[..] == prefix[..actual.tokens.len()],
                    "output budget changed the AR/greedy prefix"
                );
            }
        }
    }
    println!(
        "serving strategies={} mtp={mtp} batched_requests={batched} C2_and_budgets_exact=1",
        cases.len()
    );
    results
}

#[test]
#[ignore = "QUALITY check on the Flash-Next artifact; loads the model three times"]
fn serving_keeps_sampling_state() {
    let speculative = serve(true);
    let ordinary = serve(false);
    let cases = sampling_cases();
    for ((name, config), (ar, mtp)) in cases.iter().zip(ordinary.iter().zip(&speculative)) {
        assert!(
            !ar.tokens.is_empty() && ar.draft_tokens == 0 && mtp.draft_tokens > 0,
            "sampling strategy did not exercise AR and MTP: {name}"
        );
        if !config.uses_random_sampling() {
            assert_eq!(
                ar.tokens, mtp.tokens,
                "serving greedy AR/MTP mismatch: {name}"
            );
        }
    }
    // The serving handoff against the model session directly: replaying
    // HTTP alone would not detect a consistently dropped residual. The
    // runner's model plans for its session count.
    let mut model = Model::load(
        &trunk_path(),
        ModelOptions {
            max_context: CONTEXT,
            mtp_model_path: Some(mtp_path()),
            max_draft_tokens: 7,
            decode_concurrency: SESSIONS as u32,
            ..ModelOptions::default()
        },
    )
    .unwrap();
    for ((name, config), expected) in cases.iter().zip(&speculative) {
        let mut session = model
            .create_session(SessionMode::Speculative, CONTEXT)
            .unwrap();
        let encoded = model.tokenize(&prompt(name));
        model.sync(&mut session, &encoded).unwrap();
        let history: Vec<u32> = encoded.iter().map(|&t| t as u32).collect();
        let mut sampler = SamplerState::new(config.clone(), &history).unwrap();
        let mut tokens: Vec<u32> = Vec::new();
        while tokens.len() < 8 {
            let step = model
                .decode_step(&mut session, 8 - tokens.len(), &mut sampler, true)
                .unwrap();
            tokens.extend(step.tokens.iter().map(|&t| t as u32));
            if step.stop {
                break;
            }
            assert!(
                !step.tokens.is_empty(),
                "direct serving reference made no progress"
            );
        }
        let stats = session.statistics();
        assert!(
            tokens == expected.tokens
                && stats.drafted as usize == expected.draft_tokens
                && stats.accepted as usize == expected.draft_accepted_tokens,
            "serving lost sampling state: {name}"
        );
        println!("serving direct={name} exact=1");
    }
}
