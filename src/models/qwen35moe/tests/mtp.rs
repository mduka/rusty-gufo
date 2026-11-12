//! The MTP draft against the C++ reference, bit for bit (`mtp-head`, `mtp`
//! and `mtp-batch` fixtures):
//! cargo test --profile gpu-test -p gufo-qwen35moe --features gpu --test mtp
//!
//! Needs `GUFO_QWEN35MOE_MODEL` (UD-Q4_K_M) and `GUFO_QWEN35MOE_Q8_MODEL`
//! (Q8_0), each its own MTP model, and fails without them.
//!
//! - `mtp-head`: the prompt's final hidden rows, the batched prime, then the
//!   full head's token and logits for eight steps (`QwenMtpGpuExecutor`).
//! - `mtp`: `gufo prompt --speculative mtp` (`CreateQwenVerifier`): `Prime`
//!   and `VerifyStep` with EOS ignored for 64 tokens, the drafted, accepted
//!   and emitted tokens of every step. The proposals score the small draft
//!   head (`has_draft_vocab`), so the counts check its table too.
//! - `mtp-batch`: the serve runner's `DecodeBatch` of two sessions: both
//!   through `VerifyBatch` under the concurrent draft cap of 1 (batched
//!   proposals), then the remaining one alone.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{Q8_0, Quant, SERIAL, UD_Q4_K_M, case_prompt, fnv1a64, synthetic, u32s};
use gufo_core::qwen_tokenizer::StopTokens;
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen35moe::Engine;
use gufo_qwen35moe::engine::{Model, Session};
use gufo_qwen35moe::mtp::vocab::DRAFT_VOCAB;
use gufo_qwen35moe::mtp::{
    DEFAULT_PRIME_CHUNK, MtpBackend, MtpContext, MtpDraft, MtpDraftConfig, MtpExec, MtpHead,
    MtpModel, ProposalRequest, Rows,
};
use gufo_qwen35moe::verifier::{StepRequest, StepResult, Verifier};
use serde_json::Value;
use std::sync::Arc;

fn load(quant: Quant, max_context: u32) -> (Engine, Arc<MtpModel>) {
    let engine = Engine::load(&quant.model(), max_context).unwrap();
    let model = MtpModel::load(engine.model().stream(), &quant.model(), engine.weights()).unwrap();
    (engine, Arc::new(model))
}

fn max_context(fixture: &Value) -> u32 {
    u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap()
}

/// A named prompt of the `mtp-head` cases: `haiku` or `body<N>`.
fn named_prompt(prefill: &Value, name: &str) -> Vec<u32> {
    match name.strip_prefix("body") {
        Some(body) => synthetic(prefill, body.parse().unwrap()),
        None => u32s(&prefill["haiku"]),
    }
}

fn head_matches(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = quant.fixture("mtp-head");
    let prefill = quant.fixture("prefill");
    let max_context = max_context(&fixture);
    let (mut engine, model) = load(quant, max_context);
    assert_eq!(
        Value::Bool(model.has_draft_vocab()),
        fixture["has_draft_vocab"]
    );
    let steps = fixture["steps"].as_u64().unwrap() as usize;
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["prompt"].as_str().unwrap();
        let prompt = named_prompt(&prefill, name);
        let n = prompt.len();
        assert_eq!(n as u64, case["prompt_len"].as_u64().unwrap());
        engine.reset().unwrap();
        engine.set_hidden_capture(true, &[]).unwrap();
        let first = engine.forward_prompt(&prompt, 0, true).unwrap().unwrap();
        let hidden = engine.prompt_hidden().unwrap().to_vec();
        engine.set_hidden_capture(false, &[]).unwrap();
        let width = hidden.len() / n;
        assert_eq!(width as u64, case["hidden_width"].as_u64().unwrap());
        assert_eq!(
            fnv1a64(&hidden),
            case["hidden_fnv1a64"],
            "{name}: prompt hidden rows"
        );
        assert_eq!(
            first as u64,
            case["target_first"].as_u64().unwrap(),
            "{name}"
        );

        let ctx = MtpContext {
            stream: engine.model().stream(),
            model: &model,
            vision: None,
        };
        let mut head = MtpHead::new(&ctx, max_context, DEFAULT_PRIME_CHUNK).unwrap();
        // As `PrimeTargetContext` with a non-zero prime chunk.
        head.prime_batch(&ctx, 0, &prompt[1..], &hidden[..(n - 1) * width])
            .unwrap();
        let ids = u32s(&case["ids"]);
        let digests = case["logits_fnv1a64"].as_array().unwrap();
        let mut token = first;
        for (step, position) in (0..steps).zip((n - 1) as u32..) {
            token = if step == 0 {
                head.forward_target_hidden(
                    &ctx,
                    token,
                    &hidden[(n - 1) * width..],
                    position,
                    true,
                    false,
                )
            } else {
                head.forward_feedback(&ctx, token, position, true, false)
            }
            .unwrap();
            assert_eq!(token, ids[step], "{name}: token of step {step}");
            let digest = fnv1a64(&head.logits(ctx.stream).unwrap());
            assert_eq!(digest, digests[step], "{name}: logits of step {step}");
        }
        small_head_scores_the_full_head(&ctx, &mut head, &hidden[(n - 1) * width..], first);
    }
}

/// The small draft head scores rows of the full head: on the same input its
/// prefix and its table rows (the special tokens, then appended rare
/// tokens) carry the full head's logits bit for bit, and its token is their
/// argmax mapped back to a token ID.
fn small_head_scores_the_full_head(
    ctx: &MtpContext<'_>,
    head: &mut MtpHead,
    row: &[f32],
    token: u32,
) {
    let rare = [100_000i32, 200_000, 248_000];
    head.append_draft_rows(ctx, &rare).unwrap();
    let table: Vec<i32> = ctx
        .model
        .draft_specials()
        .iter()
        .chain(&rare)
        .copied()
        .collect();
    assert_eq!(head.draft_rows() as usize, table.len());
    let position = head.next_position();
    head.forward_target_hidden(ctx, token, row, position, true, false)
        .unwrap();
    let full = head.logits(ctx.stream).unwrap();
    head.rewind(position).unwrap();
    let small = head
        .forward_target_hidden(ctx, token, row, position, true, true)
        .unwrap();
    let logits = head.logits(ctx.stream).unwrap();
    let prefix = DRAFT_VOCAB as usize;
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&logits[..prefix]),
        bits(&full[..prefix]),
        "small head prefix"
    );
    for (index, &id) in table.iter().enumerate() {
        assert_eq!(
            logits[prefix + index].to_bits(),
            full[id as usize].to_bits(),
            "small head row of token {id}"
        );
    }
    let scored = &logits[..prefix + table.len()];
    let best = (0..scored.len())
        .reduce(|best, i| if scored[i] > scored[best] { i } else { best })
        .unwrap();
    let expected = if best < prefix {
        best as u32
    } else {
        table[best - prefix] as u32
    };
    assert_eq!(small, expected, "small head token");
    head.reset_draft_rows(ctx);
    assert_eq!(head.draft_rows() as usize, ctx.model.draft_specials().len());
}

/// The target, its MTP block and the context of the fixtures.
struct Rig {
    model: Model,
    mtp: MtpModel,
    max_context: u32,
}

impl Rig {
    fn load(quant: Quant, max_context: u32) -> Self {
        let model = Model::load(&quant.model(), max_context).unwrap();
        let mtp = MtpModel::load(model.stream(), &quant.model(), model.weights()).unwrap();
        Self {
            model,
            mtp,
            max_context,
        }
    }

    fn exec(&mut self) -> MtpExec<'_> {
        MtpExec {
            model: &mut self.model,
            mtp: &self.mtp,
        }
    }

    /// The probe's `MtpBackend` (7 drafts, prime chunk 1024) under a
    /// verifier with the adaptive length from 7 down to 1: `CliOptions`, or
    /// `ServeOptions` with the frontier logits retained.
    fn verifier(&self, serve: bool) -> Verifier<MtpBackend> {
        let config = MtpDraftConfig {
            max_context: self.max_context,
            max_draft_tokens: 7,
            prime_chunk: DEFAULT_PRIME_CHUNK,
        };
        let backend = MtpBackend::new(self.model.stream(), &self.mtp, config).unwrap();
        let mut verifier = Verifier::with_draft(backend, 1, 7, true);
        verifier.set_retain_frontier_logits(serve);
        verifier
    }
}

/// One session's run: the committed sequence and the fixture trace.
struct Lane {
    verifier: Verifier<MtpBackend>,
    session: Session,
    stops: StopTokens,
    sampler: SamplerState,
    sequence: Vec<u32>,
    position: u32,
    current: u32,
    first: u32,
    ids: Vec<u32>,
    steps: Vec<Vec<u32>>,
}

impl Lane {
    /// `Prime` over `prompt` on a session of its own.
    fn prime(rig: &mut Rig, mut verifier: Verifier<MtpBackend>, prompt: &[u32]) -> Self {
        let mut session = rig.model.new_session().unwrap();
        let first = verifier
            .prime(&mut rig.exec(), &mut session, prompt, true)
            .unwrap();
        let mut sequence = prompt.to_vec();
        sequence.push(first);
        Self {
            verifier,
            session,
            stops: rig.model.weights().tokenizer.stop_tokens(),
            sampler: SamplerState::new(SamplingConfig::default(), prompt).unwrap(),
            sequence,
            position: prompt.len() as u32,
            current: first,
            first,
            ids: Vec::new(),
            steps: Vec::new(),
        }
    }

    fn request(&mut self, tokens: usize) -> StepRequest<'_, MtpBackend> {
        StepRequest {
            verifier: &mut self.verifier,
            session: &mut self.session,
            sequence: &self.sequence,
            position: self.position,
            current_token: self.current,
            stops: self.stops,
            max_emitted: (tokens - self.ids.len()) as u32,
            sampler: &mut self.sampler,
        }
    }

    /// Records `step` (`[width,] drafted, accepted, emitted...`) and commits
    /// its tokens up to `tokens` (EOS ignored).
    fn take(&mut self, step: &StepResult, width: Option<u32>, tokens: usize) {
        let mut row: Vec<u32> = width.into_iter().collect();
        row.extend([step.draft_count as u32, step.accepted_count as u32]);
        row.extend(&step.emitted);
        self.steps.push(row);
        for &token in &step.emitted {
            if self.ids.len() < tokens {
                self.ids.push(token);
                self.sequence.push(token);
                self.position += 1;
            }
        }
        self.current = step.next_token;
    }

    /// The first divergence from a fixture entry.
    fn compare(&self, expected: &Value) -> Option<String> {
        if u64::from(self.first) != expected["first"].as_u64().unwrap() {
            return Some(format!(
                "first token {}, C++ {}",
                self.first, expected["first"]
            ));
        }
        let steps = expected["steps"].as_array().unwrap();
        for (index, (got, want)) in self.steps.iter().zip(steps).enumerate() {
            if *got != u32s(want) {
                return Some(format!("step {index}: {got:?}, C++ {want}"));
            }
        }
        if self.steps.len() != steps.len() {
            return Some(format!("{} steps, C++ {}", self.steps.len(), steps.len()));
        }
        (self.ids != u32s(&expected["ids"])).then(|| "tokens differ".to_string())
    }
}

/// The probe's `RunMtp`: per case a fresh backend, `Prime`, then
/// `VerifyStep` until 64 tokens; steps, tokens and statistics match.
fn verifier_matches(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = quant.fixture("mtp");
    let prefill = quant.fixture("prefill");
    let tokens = fixture["tokens"].as_u64().unwrap() as usize;
    let mut rig = Rig::load(quant, max_context(&fixture));
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{} mtp {}", quant.set, case["prompt"].as_str().unwrap());
        let prompt = case_prompt(&prefill, case);
        let verifier = rig.verifier(false);
        let mut lane = Lane::prime(&mut rig, verifier, &prompt);
        while lane.ids.len() < tokens {
            let step = Verifier::verify_step(&mut rig.exec(), &mut lane.request(tokens)).unwrap();
            lane.take(&step, None, tokens);
        }
        if let Some(divergence) = lane.compare(case) {
            failures.push(format!("{name}: {divergence}"));
        }
        let stats = lane.verifier.stats();
        let counted = [
            stats.draft_tokens,
            stats.accepted_tokens,
            stats.verification_steps,
            stats.emitted_tokens,
        ];
        let expected = ["drafted", "accepted", "verification_steps", "emitted"]
            .map(|key| case["stats"][key].as_u64().unwrap());
        if counted != expected {
            failures.push(format!("{name}: stats {counted:?}, C++ {expected:?}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The probe's `RunSpecBatch` for MTP: two sessions in the serve
/// configuration decode through `VerifyBatch` under the concurrent draft cap
/// (`ConcurrentDraftCap(2)` = 1 for MTP), the one left alone through
/// `VerifyStep` without it.
fn batch_matches(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = quant.fixture("mtp-batch");
    let prefill = quant.fixture("prefill");
    let tokens = fixture["tokens"].as_u64().unwrap() as usize;
    let mut rig = Rig::load(quant, max_context(&fixture));
    let mut lanes = Vec::new();
    for entry in fixture["sessions"].as_array().unwrap() {
        let prompt = named_prompt(&prefill, entry["prompt"].as_str().unwrap());
        assert_eq!(prompt.len() as u64, entry["prompt_len"].as_u64().unwrap());
        let verifier = rig.verifier(true);
        lanes.push(Lane::prime(&mut rig, verifier, &prompt));
    }
    loop {
        let mut active: Vec<&mut Lane> =
            lanes.iter_mut().filter(|l| l.ids.len() < tokens).collect();
        match active.len() {
            0 => break,
            1 => {
                let lane = &mut active[0];
                lane.verifier.set_step_draft_cap(u32::MAX);
                let step =
                    Verifier::verify_step(&mut rig.exec(), &mut lane.request(tokens)).unwrap();
                lane.take(&step, Some(1), tokens);
            }
            width => {
                for lane in active.iter_mut() {
                    lane.verifier.set_step_draft_cap(1);
                }
                let mut requests: Vec<StepRequest<'_, MtpBackend>> =
                    active.iter_mut().map(|lane| lane.request(tokens)).collect();
                let results = Verifier::verify_batch(&mut rig.exec(), &mut requests).unwrap();
                drop(requests);
                for (lane, step) in active.iter_mut().zip(&results) {
                    lane.take(step, Some(width as u32), tokens);
                }
            }
        }
    }
    let mut failures = Vec::new();
    for (lane, entry) in lanes.iter().zip(fixture["sessions"].as_array().unwrap()) {
        if let Some(divergence) = lane.compare(entry) {
            failures.push(format!(
                "{} mtp-batch {}: {divergence}",
                quant.set, entry["prompt"]
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The draft paths the fixtures do not reach, against the serial greedy
/// proposal of the same state: a batch of two sessions (`ProposeBatch`), a
/// step cap, a snapshot round trip through the persistent payload, and
/// sampled proposals (same RNG, same draw; each row a distribution over the
/// full head's top candidates).
fn draft_paths(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prefill = quant.fixture("prefill");
    let max_context = 6144;
    let (mut engine, model) = load(quant, max_context);
    let prompt = u32s(&prefill["haiku"]);
    let n = prompt.len();
    engine.reset().unwrap();
    engine.set_hidden_capture(true, &[]).unwrap();
    let first = engine.forward_prompt(&prompt, 0, true).unwrap().unwrap();
    let hidden = engine.prompt_hidden().unwrap().to_vec();
    engine.set_hidden_capture(false, &[]).unwrap();
    let ctx = MtpContext {
        stream: engine.model().stream(),
        model: &model,
        vision: None,
    };
    let primed = || {
        let mut draft = MtpDraft::new(ctx.stream, &model, max_context, 7).unwrap();
        draft.prime(&ctx, &prompt, &hidden).unwrap();
        draft
    };
    let mut sequence = prompt.clone();
    sequence.push(first);
    let position = n as u32;
    let mut serial = primed();
    let reference = serial
        .propose_greedy(&ctx, &sequence, position, 4)
        .unwrap()
        .tokens;
    assert_eq!(reference.len(), 4);

    let (mut a, mut b) = (primed(), primed());
    b.set_step_draft_cap(2);
    let proposals = MtpDraft::propose_batch(
        &ctx,
        &mut [
            ProposalRequest {
                draft: &mut a,
                vision: None,
                tokens: &sequence,
                position,
                max_tokens: 4,
                sampling: None,
            },
            ProposalRequest {
                draft: &mut b,
                vision: None,
                tokens: &sequence,
                position,
                max_tokens: 4,
                sampling: None,
            },
        ],
    )
    .unwrap();
    assert_eq!(proposals[0].tokens, reference, "batched proposal");
    assert_eq!(
        proposals[1].tokens,
        reference[..2],
        "capped batched proposal"
    );

    // Feedback: the anchor's row and one accepted draft's, then the next
    // proposal from the committed state matches across a snapshot.
    let row = &hidden[(n - 1) * hidden.len() / n..];
    for draft in [&mut serial, &mut a] {
        draft.update_hidden(&ctx, Rows::Host(row)).unwrap();
        draft.update_hidden(&ctx, Rows::Host(row)).unwrap();
        draft.accept(&ctx, &reference[..1]).unwrap();
    }
    let snapshot = a.snapshot(ctx.stream, &model).unwrap();
    let mut payload = vec![0u8; snapshot.persistent_bytes()];
    snapshot.serialize(ctx.stream, &mut payload).unwrap();
    let mut restored = MtpDraft::new(ctx.stream, &model, max_context, 7).unwrap();
    restored.restore_persistent(&ctx, &payload).unwrap();
    sequence.extend([reference[0], 5]);
    let next = serial
        .propose_greedy(&ctx, &sequence, position + 2, 3)
        .unwrap()
        .tokens;
    for (name, draft) in [("in place", &mut a), ("restored", &mut restored)] {
        let tokens = draft
            .propose_greedy(&ctx, &sequence, position + 2, 3)
            .unwrap()
            .tokens;
        assert_eq!(tokens, next, "proposal after feedback, {name}");
        draft.discard_pending(position + 2).unwrap();
    }
    a.restore(&ctx, &snapshot).unwrap();
    let tokens = a
        .propose_greedy(&ctx, &sequence, position + 2, 3)
        .unwrap()
        .tokens;
    assert_eq!(tokens, next, "proposal after a snapshot restore");

    let sampled = |draft: &mut MtpDraft, seed: u64| {
        let mut rng = seed;
        let proposal = draft
            .propose_sampled(&ctx, &sequence[..n + 1], position, 3, 0.8, &mut rng)
            .unwrap();
        draft.discard_pending(position).unwrap();
        proposal
    };
    let (mut c, mut d) = (primed(), primed());
    let (p, q) = (sampled(&mut c, 1234), sampled(&mut d, 1234));
    assert_eq!(p, q, "sampled proposals of one seed");
    assert_eq!(p.tokens.len(), 3);
    assert_eq!(p.candidates_per_token, 64);
    for (index, &token) in p.tokens.iter().enumerate() {
        let ids = &p.candidate_ids[index * 64..(index + 1) * 64];
        let mass = &p.candidate_probabilities[index * 64..(index + 1) * 64];
        let sum: f64 = mass.iter().map(|&m| f64::from(m)).sum();
        assert_eq!(sum, 1.0, "sampled row {index} sums to one");
        let at = ids.iter().position(|&id| id == token).unwrap();
        assert!(mass[at] > 0.0, "drawn token without mass");
    }
}

#[test]
fn head_ud_q4_k_m() {
    head_matches(UD_Q4_K_M);
}

#[test]
fn head_q8_0() {
    head_matches(Q8_0);
}

#[test]
fn verifier_ud_q4_k_m() {
    verifier_matches(UD_Q4_K_M);
}

#[test]
fn verifier_q8_0() {
    verifier_matches(Q8_0);
}

#[test]
fn batch_ud_q4_k_m() {
    batch_matches(UD_Q4_K_M);
}

#[test]
fn batch_q8_0() {
    batch_matches(Q8_0);
}

#[test]
fn draft_paths_ud_q4_k_m() {
    draft_paths(UD_Q4_K_M);
}
