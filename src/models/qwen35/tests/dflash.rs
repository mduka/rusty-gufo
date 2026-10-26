//! DFlash2 speculative decoding against the C++ engine, step by step, and
//! the serving draft's prompt-history lookup, numeric penalties and batched
//! proposals:
//! cargo test --release -p gufo-qwen35 --features gpu --test dflash
//!
//! Needs the Qwen3.8 27B GGUFs (UD-Q4_K_XL, also from `GUFO_QWEN27B_MODEL`,
//! and UD-Q8_K_XL with BF16 attention projections and LM head) and their
//! DFlash2 draft (`GUFO_QWEN27B_DFLASH`, else the pinned files in the Hugging
//! Face cache) and fails without them. The fixtures come from
//! `tests/reference/cpp_decode_dump.cpp dflash`, which drives the verifier as
//! `gufo bench --speculative dflash2` does: every step's draft length,
//! accepted count and emitted tokens must match.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Mutex;

use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen35::dflash::DFlash;
use gufo_qwen35::dflash::draft::{Draft, DraftConfig, DraftContext, ProposalRequest};
use gufo_qwen35::dflash::executor::{DraftPenalties, DraftScratch, Proposal};
use gufo_qwen35::dflash::model::DFlashModel;
use gufo_qwen35::dflash::policy::DraftPolicy;
use gufo_qwen35::{Engine, Model, Speculative};
use serde_json::Value;

fn cached(variable: &str, relative: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap())
                .join(".cache/huggingface/hub")
                .join(relative)
        })
}

fn u32s(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

/// One target model on the device at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// The pinned file of the 27B repository.
fn target(variable: Option<&str>, file: &str) -> PathBuf {
    let relative = format!(
        "models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/{file}"
    );
    match variable {
        Some(variable) => cached(variable, &relative),
        None => PathBuf::from(std::env::var_os("HOME").unwrap())
            .join(".cache/huggingface/hub")
            .join(relative),
    }
}

fn draft_path() -> PathBuf {
    cached(
        "GUFO_QWEN27B_DFLASH",
        "models--z-lab--Qwen3.8-27B-DFlash2-GGUF/snapshots/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
    )
}

fn speculative_steps_match(fixture: &str, model: PathBuf) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture: Value = serde_json::from_str(fixture).unwrap();
    let decode: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-decode.json")).unwrap();
    let prefill: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-prefill.json")).unwrap();
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let total = fixture["tokens"].as_u64().unwrap() as usize;
    let mut engine = Engine::load(&model, max_context).unwrap();
    let draft = DFlash::load(
        &draft_path(),
        engine.weights(),
        max_context,
        7,
        DraftPolicy::Adaptive,
    )
    .unwrap();
    let mut speculative = Speculative::new(draft);
    for case in fixture["cases"].as_array().unwrap() {
        let prompt = match case["body"].as_u64() {
            Some(body) => {
                let body = (0..body as u32).map(|i| 1000 + (i * 7919 + 13) % 50_000);
                u32s(&prefill["head"])
                    .into_iter()
                    .chain(body)
                    .chain(u32s(&prefill["tail"]))
                    .collect()
            }
            None => u32s(&decode["prompt"]),
        };
        let expected_steps = case["steps"].as_array().unwrap();
        let mut current = speculative.prime(&mut engine, &prompt, true).unwrap();
        let mut position = prompt.len() as u32;
        // The committed history, as `gufo bench` keeps it for the draft's
        // prompt lookup.
        let mut sequence = prompt.clone();
        sequence.push(current);
        let mut ids = Vec::new();
        for (index, expected) in expected_steps.iter().enumerate() {
            let expected = u32s(expected);
            let step = speculative
                .verify_step(
                    &mut engine,
                    &sequence,
                    current,
                    position,
                    999_999,
                    (total - ids.len()) as u32,
                )
                .unwrap();
            let mut got = vec![step.drafted as u32, step.accepted as u32];
            got.extend(&step.emitted);
            assert_eq!(got, expected, "{}: step {index}", case["prompt"]);
            for &token in &step.emitted {
                if ids.len() < total {
                    ids.push(token);
                    sequence.push(token);
                    position += 1;
                }
            }
            current = step.next_token;
        }
        assert_eq!(ids, u32s(&case["ids"]), "{}", case["prompt"]);
    }
    assert_eq!(engine.graph_mismatches(), 0);
}

#[test]
fn speculative_steps_match_the_cpp_verifier() {
    speculative_steps_match(
        include_str!("fixtures/qwen3.8-27b-dflash.json"),
        target(Some("GUFO_QWEN27B_MODEL"), "Qwen3.8-27B-UD-Q4_K_XL.gguf"),
    );
}

/// BF16 q/k/v and LM head: the exact BF16 GEMM of the verification rows.
#[test]
fn speculative_steps_match_the_cpp_verifier_q8_k_xl() {
    speculative_steps_match(
        include_str!("fixtures/qwen3.8-27b-q8_k_xl-dflash.json"),
        target(None, "Qwen3.8-27B-UD-Q8_K_XL.gguf"),
    );
}

/// The Q4 target and its draft model, for the serving draft checks.
struct Serving {
    target: Model,
    draft: DFlashModel,
}

impl Serving {
    fn load(max_context: u32) -> Self {
        let target = Model::load(
            &target(Some("GUFO_QWEN27B_MODEL"), "Qwen3.8-27B-UD-Q4_K_XL.gguf"),
            max_context,
        )
        .unwrap();
        let draft = DFlashModel::load(target.stream(), &draft_path(), target.weights()).unwrap();
        Self { target, draft }
    }

    fn width(&self) -> usize {
        self.draft.config.target_features()
    }

    fn context<'a>(&'a self, scratch: &'a mut DraftScratch) -> DraftContext<'a> {
        DraftContext {
            stream: self.target.stream(),
            model: &self.draft,
            scratch,
        }
    }

    fn special(&self, name: &str) -> u32 {
        self.target
            .weights()
            .tokenizer
            .special(name)
            .unwrap_or_else(|| panic!("lookup fixture control token {name} is missing"))
    }

    /// The persistent `GDFDRF01` bytes of `draft`.
    fn persistent(&self, draft: &Draft) -> Vec<u8> {
        let stream = self.target.stream();
        let snapshot = draft.snapshot(stream, &self.draft).unwrap();
        let capacity = draft.history_capacity();
        let mut bytes = vec![0; snapshot.persistent_bytes(&self.draft, capacity).unwrap()];
        snapshot
            .serialize(stream, &self.draft, capacity, &mut bytes)
            .unwrap();
        bytes
    }
}

fn propose(
    ctx: &mut DraftContext<'_>,
    draft: &mut Draft,
    sequence: &[u32],
    sampling: Option<(f32, &mut u64)>,
) -> Proposal {
    let position = sequence.len() as u32 - 1;
    draft
        .propose(ctx, sequence, position, 7, sampling, None)
        .unwrap()
}

/// Greedy requests may start copying after 12 matching tokens only once the
/// neural acceptance falls below its neutral estimate; a fixed policy keeps
/// that estimate and the 24-token gate. The decision reads only snapshotted
/// state, so both restore forms replay it (`TestGreedyLookupGate`).
#[test]
fn greedy_lookup_gate_follows_the_neural_acceptance() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let serving = Serving::load(256);
    let mut scratch = DraftScratch::new(serving.target.stream(), &serving.draft, 256).unwrap();
    let width = serving.width();
    for policy in [DraftPolicy::Adaptive, DraftPolicy::Fixed] {
        let mut ctx = serving.context(&mut scratch);
        let config = DraftConfig::for_model(&serving.draft, 256, 7, policy).unwrap();
        let mut draft = Draft::new(&ctx, config).unwrap();
        let mut sequence: Vec<u32> = (0..64).map(|i| 1 + i % 16).collect();
        let features = vec![0.01f32; 63 * width];
        draft.prime(&mut ctx, &features, 63).unwrap();
        // Corrective target tokens lower the adaptive neural estimate.
        let first_copy = if policy == DraftPolicy::Adaptive {
            11
        } else {
            23
        };
        let mut copied = false;
        for round in 0..=first_copy {
            let saved =
                (round == first_copy).then(|| draft.snapshot(ctx.stream, ctx.model).unwrap());
            let bytes = (round == first_copy).then(|| serving.persistent(&draft));
            let proposal = propose(&mut ctx, &mut draft, &sequence, None);
            if proposal.from_lookup {
                assert!(
                    round == first_copy && proposal.candidate_ids.is_empty(),
                    "{policy:?}: only weak neural acceptance enables short greedy matches"
                );
                for (i, &token) in proposal.tokens.iter().enumerate() {
                    assert_eq!(
                        token,
                        1 + (sequence.len() + i) as u32 % 16,
                        "greedy lookup must copy the indexed continuation"
                    );
                }
                for persistent in [false, true] {
                    if persistent {
                        draft
                            .restore_persistent(&ctx, bytes.as_deref().unwrap())
                            .unwrap();
                    } else {
                        draft.restore(&ctx, saved.as_ref().unwrap()).unwrap();
                    }
                    let replay = propose(&mut ctx, &mut draft, &sequence, None);
                    assert!(
                        replay.from_lookup && replay.tokens == proposal.tokens,
                        "{policy:?}: restored neural history must choose the same copy gate"
                    );
                }
                copied = true;
                break;
            }
            assert!(
                round < first_copy,
                "{policy:?}: greedy copy gate failed to open"
            );
            let correction = 1 + sequence.len() as u32 % 16;
            draft.accept(0).unwrap();
            draft.update_hidden(&features[..width]).unwrap();
            sequence.push(correction);
        }
        assert!(
            copied,
            "{policy:?}: the copy gate never found the repetition"
        );
    }
}

/// Sampled copies need a 24-token generated match, report point-mass q and
/// draw nothing; a continuation that crosses a message boundary selects an
/// earlier source, while thinking and tool tokens stay copyable. The copy
/// state survives both snapshot forms, a new request cannot resume the
/// previous answer's copy, and a version 3 payload restores without lookup
/// state (`TestHistoryLookup`).
#[test]
fn sampled_history_lookup_skips_message_boundaries_and_restores() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let serving = Serving::load(256);
    let mut scratch = DraftScratch::new(serving.target.stream(), &serving.draft, 256).unwrap();
    let mut ctx = serving.context(&mut scratch);
    let config = DraftConfig::for_model(&serving.draft, 256, 7, DraftPolicy::Adaptive).unwrap();
    let mut draft = Draft::new(&ctx, config).unwrap();
    let width = serving.width();
    let mut pattern: [u32; 16] = std::array::from_fn(|i| 1 + i as u32);
    // Independent reference spellings of real thinking and tool tokens.
    pattern[2] = serving.special("<think>");
    pattern[5] = serving.special("</think>");
    pattern[8] = serving.special("<tool_call>");
    pattern[11] = serving.special("</tool_call>");
    let at = |index: usize| pattern[index % pattern.len()];
    let mut sequence: Vec<u32> = (0..64).map(at).collect();
    // The nearest initial match ends at 55: its seven-token continuation
    // crosses this boundary. The earlier occurrence at 39 remains valid.
    sequence[61] = serving.special("<|im_end|>");
    let features = vec![0.01f32; 63 * width];
    draft.prime(&mut ctx, &features, 63).unwrap();
    let mut rng = 731u64;
    let mut copied = false;
    for round in 0..32 {
        let before = rng;
        let proposal = propose(&mut ctx, &mut draft, &sequence, Some((0.7, &mut rng)));
        copied = proposal.from_lookup;
        if copied {
            assert!(
                round >= 23 && rng == before,
                "sampled lookup requires a long generated match and no draft draw"
            );
            for (i, &token) in proposal.tokens.iter().enumerate() {
                assert!(
                    token == at(sequence.len() + i)
                        && proposal.candidate_ids[i] == token
                        && proposal.candidate_probabilities[i] == 1.0,
                    "lookup must report its exact conditional point mass"
                );
            }
            // The rejected nearest source would stop before the boundary.
            assert_eq!(
                proposal.tokens.len(),
                7,
                "an earlier source without message framing supplies the whole block"
            );
            assert_eq!(proposal.candidate_ids.len(), proposal.tokens.len());
            let correction = at(sequence.len() + proposal.tokens.len());
            draft.accept(proposal.tokens.len()).unwrap();
            for _ in 0..=proposal.tokens.len() {
                draft.update_hidden(&features[..width]).unwrap();
            }
            sequence.extend(&proposal.tokens);
            sequence.push(correction);
            break;
        }
        assert_ne!(
            rng, before,
            "neural drafting draws its private random values"
        );
        let correction = at(sequence.len());
        draft.accept(0).unwrap();
        draft.update_hidden(&features[..width]).unwrap();
        sequence.push(correction);
    }
    assert!(copied, "DFlash must use a matching prompt history");
    let saved = draft.snapshot(ctx.stream, ctx.model).unwrap();
    let mut bytes = serving.persistent(&draft);
    let initial_rng = rng;
    let expected = propose(&mut ctx, &mut draft, &sequence, Some((0.7, &mut rng)));
    assert!(
        expected.from_lookup && rng == initial_rng,
        "verified copies resume without consuming draft RNG"
    );
    for persistent in [false, true] {
        if persistent {
            draft.restore_persistent(&ctx, &bytes).unwrap();
        } else {
            draft.restore(&ctx, &saved).unwrap();
        }
        let actual = propose(&mut ctx, &mut draft, &sequence, Some((0.7, &mut rng)));
        assert!(
            actual == expected && rng == initial_rng,
            "lookup state and point mass must survive both snapshot forms"
        );
    }
    draft.restore(&ctx, &saved).unwrap();
    draft.begin_request();
    let restarted = propose(&mut ctx, &mut draft, &sequence, Some((0.7, &mut rng)));
    assert!(
        !restarted.from_lookup
            && restarted.candidate_ids.len() > restarted.tokens.len()
            && rng != initial_rng,
        "a new request must not resume the previous answer's copy"
    );
    // Version 3 did not carry lookup state and kept these bytes zero.
    bytes[8..12].copy_from_slice(&3u32.to_le_bytes());
    bytes[72..80].fill(0);
    draft.restore_persistent(&ctx, &bytes).unwrap();
    let legacy_rng = rng;
    let legacy = propose(&mut ctx, &mut draft, &sequence, Some((0.7, &mut rng)));
    assert!(
        !legacy.from_lookup && rng != legacy_rng,
        "old snapshots remain readable and start without lookup history"
    );
    // A lookup position beyond the restored history is refused.
    draft.accept(0).unwrap();
    let mut beyond = serving.persistent(&draft);
    let committed = sequence.len() as u32 - 1;
    beyond[72..76].copy_from_slice(&(committed + 1).to_le_bytes());
    assert!(draft.restore_persistent(&ctx, &beyond).is_err());
}

/// Cohorts of two to eight sessions draft as each would alone: greedy and
/// sampled requests at distinct positions and temperatures, under fixed and
/// adaptive policies and repeat, frequency and presence penalties, keep
/// their tokens, selector probabilities, random draws, and after feedback
/// their complete persistent state (the backend half of
/// `TestConcurrentBlocks`).
#[test]
fn batched_penalized_proposals_match_single_sessions() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const SESSIONS: usize = 8;
    let mut serving = Serving::load(128);
    let mut scratch = DraftScratch::new(serving.target.stream(), &serving.draft, 128).unwrap();
    let width = serving.width();
    let taps = serving.draft.config.target_layer_ids.clone();
    let mut prompt = serving.target.weights().tokenizer.encode(
        "Virtual memory gives every process a separate address space. The operating system \
         maps virtual pages to physical memory, handles page faults, and keeps unrelated \
         processes from changing each other's data.",
    );
    assert!(prompt.len() >= 25, "concurrent draft reference prompt");
    prompt.truncate(25);
    let features = {
        let mut session = serving.target.new_session().unwrap();
        session
            .set_hidden_capture(serving.target.config(), true, &taps)
            .unwrap();
        serving
            .target
            .forward_prompt(&mut session, &prompt, 0, false)
            .unwrap();
        serving.target.prompt_hidden().unwrap().to_vec()
    };
    assert_eq!(features.len(), prompt.len() * width);
    let temperatures = [0.0f32, 0.2, 0.8, 1.3, 2.0, 0.0, 0.7, 1.0];
    let mut positions: [u32; SESSIONS] = std::array::from_fn(|i| 3 * (i as u32 + 1));
    let mut drafts = Vec::with_capacity(SESSIONS);
    let mut sequences = Vec::with_capacity(SESSIONS);
    let mut samplers = Vec::with_capacity(SESSIONS);
    let mut rng: [u64; SESSIONS] = std::array::from_fn(|i| 191 + 31 * i as u64);
    {
        let mut ctx = serving.context(&mut scratch);
        for index in 0..SESSIONS {
            let policy = if index % 2 == 0 {
                DraftPolicy::Adaptive
            } else {
                DraftPolicy::Fixed
            };
            let config = DraftConfig::for_model(&serving.draft, 128, 7, policy).unwrap();
            let mut draft = Draft::new(&ctx, config).unwrap();
            let rows = positions[index] as usize;
            draft
                .prime(&mut ctx, &features[..rows * width], rows)
                .unwrap();
            drafts.push(draft);
            let sequence = prompt[..rows + 1].to_vec();
            let config = SamplingConfig {
                temperature: temperatures[index],
                top_k: if index % 2 == 1 { 4 } else { 20 },
                top_p: if index % 3 != 0 { 0.8 } else { 1.0 },
                min_p: if index % 2 == 1 { 0.1 } else { 0.0 },
                repeat_penalty: 1.1,
                repeat_last_n: 8,
                frequency_penalty: if index % 3 != 0 { 0.2 } else { 0.0 },
                presence_penalty: if index % 2 == 1 { 1.5 } else { 0.0 },
                ..SamplingConfig::default()
            };
            let mut sampler = SamplerState::new(config, &sequence).unwrap();
            sampler.accept(*sequence.last().unwrap()).unwrap();
            samplers.push(sampler);
            sequences.push(sequence);
        }
    }
    let limit = |index: usize| {
        if index < 5 { 7 } else { 1 + (index % 7) as u32 }
    };
    for round in 0..2 {
        let penalties: Vec<DraftPenalties> = samplers.iter().map(DraftPenalties::of).collect();
        let mut ctx = serving.context(&mut scratch);
        let mut initial = Vec::with_capacity(SESSIONS);
        let mut expected: Vec<Proposal> = Vec::with_capacity(SESSIONS);
        let mut expected_state = Vec::with_capacity(SESSIONS);
        let mut accepted = [0usize; SESSIONS];
        let initial_rng = rng;
        let mut expected_rng = rng;
        for index in 0..SESSIONS {
            let draft = &mut drafts[index];
            initial.push(draft.snapshot(ctx.stream, ctx.model).unwrap());
            let sampled = temperatures[index] > 0.0;
            let proposal = draft
                .propose(
                    &mut ctx,
                    &sequences[index],
                    positions[index],
                    limit(index),
                    sampled.then_some((temperatures[index], &mut expected_rng[index])),
                    Some(&penalties[index]),
                )
                .unwrap();
            accepted[index] = (index % 4).min(proposal.tokens.len());
            draft.accept(accepted[index]).unwrap();
            expected_state.push(serving.persistent(draft));
            draft.restore(&ctx, &initial[index]).unwrap();
            expected.push(proposal);
        }
        let check = |batch: &[Proposal], first: usize, rng: &[u64; SESSIONS], what: &str| {
            for (row, proposal) in batch.iter().enumerate() {
                let index = first + row;
                let reference = &expected[index];
                assert!(
                    proposal.tokens == reference.tokens
                        && proposal.candidate_ids == reference.candidate_ids
                        && proposal
                            .candidate_probabilities
                            .iter()
                            .map(|p| p.to_bits())
                            .eq(reference
                                .candidate_probabilities
                                .iter()
                                .map(|p| p.to_bits()))
                        && rng[index] == expected_rng[index],
                    "round {round} {what}: session {index} changed its proposal or draws"
                );
            }
        };
        // Sampled and mixed greedy/sampled cohorts keep their private
        // proposal lengths, selector probabilities and random draws.
        for width in 2..=7 {
            for first in [0, 1] {
                let batch = {
                    let mut requests: Vec<ProposalRequest<'_>> = drafts[first..first + width]
                        .iter_mut()
                        .zip(&mut rng[first..first + width])
                        .enumerate()
                        .map(|(row, (draft, rng))| {
                            let index = first + row;
                            ProposalRequest {
                                draft,
                                tokens: &sequences[index],
                                position: positions[index],
                                max_tokens: limit(index),
                                sampling: (temperatures[index] > 0.0)
                                    .then_some((temperatures[index], rng)),
                                penalties: Some(&penalties[index]),
                            }
                        })
                        .collect();
                    Draft::propose_batch(&mut ctx, &mut requests).unwrap()
                };
                assert_eq!(batch.len(), width);
                check(&batch, first, &rng, &format!("width {width} from {first}"));
                for index in first..first + width {
                    drafts[index].restore(&ctx, &initial[index]).unwrap();
                    rng[index] = initial_rng[index];
                }
            }
        }
        let batch = {
            let mut requests: Vec<ProposalRequest<'_>> = drafts
                .iter_mut()
                .zip(rng.iter_mut())
                .enumerate()
                .map(|(index, (draft, rng))| ProposalRequest {
                    draft,
                    tokens: &sequences[index],
                    position: positions[index],
                    max_tokens: limit(index),
                    sampling: (temperatures[index] > 0.0).then_some((temperatures[index], rng)),
                    penalties: Some(&penalties[index]),
                })
                .collect();
            Draft::propose_batch(&mut ctx, &mut requests).unwrap()
        };
        check(&batch, 0, &rng, "all sessions");
        for index in 0..SESSIONS {
            let draft = &mut drafts[index];
            draft.accept(accepted[index]).unwrap();
            assert!(
                serving.persistent(draft) == expected_state[index],
                "round {round}: batched feedback changed session {index}'s state"
            );
            for row in 0..=accepted[index] {
                let feature_row = (positions[index] as usize + row) % prompt.len();
                draft
                    .update_hidden(&features[feature_row * width..(feature_row + 1) * width])
                    .unwrap();
            }
            sequences[index].extend(&batch[index].tokens[..accepted[index]]);
            sequences[index].push(4);
            positions[index] += accepted[index] as u32 + 1;
        }
    }
}
