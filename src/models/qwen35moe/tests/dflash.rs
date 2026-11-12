//! DFlash2 speculative decoding against the C++ reference, step by step
//! (`dflash` and `dflash-batch` fixtures):
//! cargo test --profile gpu-test -p gufo-qwen35moe --features gpu --test dflash
//!
//! Needs `GUFO_QWEN35MOE_MODEL` (UD-Q4_K_M), `GUFO_QWEN35MOE_Q8_MODEL` (Q8_0)
//! and `GUFO_QWEN35MOE_DFLASH_MODEL` (the DFlash2 drafter) and fails without
//! them. Every verification step's draft length, accepted count and emitted
//! tokens must match the extension's `SpeculativeVerifier` run:
//! - `cases`: `gufo prompt --speculative dflash2` (7 drafts, adaptive draft
//!   policy, fixed verifier length, no lookup copy);
//! - `serve_cases`: the serve runner (copies of a request verified alone grow
//!   to 64 rows, verifier draft limit 63, retained frontier logits);
//! - `dflash-batch`: two serve sessions through `VerifyBatch` under the
//!   concurrent draft cap of two, then the one left alone without it.
//!
//! In a mixed proposal batch, a greedy request over the full head beside a
//! sampled one over its draft vocabulary, the sampled request must propose
//! as it does alone (its table rows are head rows, and every score is per
//! output).
//!
//! EOS is ignored, as in the fixtures, so outputs run past `<|im_end|>`; a
//! stop token still ends a step's accepted run (`IsStopToken`).

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::path::PathBuf;

use common::{Q8_0, Quant, SERIAL, UD_Q4_K_M, case_prompt, u32s};
use gufo_core::qwen_tokenizer::StopTokens;
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_hip::Stream;
use gufo_qwen35moe::dflash::draft::{Draft, DraftConfig, DraftContext, ProposalRequest};
use gufo_qwen35moe::dflash::executor::DraftScratch;
use gufo_qwen35moe::dflash::model::DFlashModel;
use gufo_qwen35moe::dflash::policy::DraftPolicy;
use gufo_qwen35moe::verifier::{SpecExec, StepRequest, StepResult, Verifier};
use gufo_qwen35moe::{Model, Session};
use serde_json::Value;

/// The drafter. Unset, the variable's name stands in for the path, so the
/// failing load says what to set.
fn drafter() -> PathBuf {
    std::env::var_os("GUFO_QWEN35MOE_DFLASH_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("GUFO_QWEN35MOE_DFLASH_MODEL"))
}

/// Verification rows a serve copy may grow to (`kMaxWideVerificationRows`).
const SERVE_COPY_ROWS: u32 = 64;

/// The target, the drafter and the draft scratch on its own stream, as the
/// serve runner holds them.
struct Rig {
    model: Model,
    draft: DFlashModel,
    stream: Stream,
    scratch: DraftScratch,
    max_context: u32,
}

/// The verifier options of a fixture run.
#[derive(Clone, Copy, PartialEq)]
enum Options {
    /// `drivers.cpp` `CreateQwenVerifier`.
    Cli,
    /// `runner.cpp` `CreateQwenTextRunner` and `BindComponentState`.
    Serve,
}

impl Rig {
    fn load(quant: Quant, max_context: u32) -> Self {
        let model = Model::load(&quant.model(), max_context).unwrap();
        let draft = DFlashModel::load(model.stream(), &drafter(), model.weights()).unwrap();
        let stream = Stream::new().unwrap();
        let scratch = DraftScratch::new(&stream, &draft, max_context).unwrap();
        Self {
            model,
            draft,
            stream,
            scratch,
            max_context,
        }
    }

    fn exec(&mut self) -> SpecExec<'_> {
        SpecExec {
            model: &mut self.model,
            draft_model: &self.draft,
            draft_scratch: &mut self.scratch,
            draft_stream: &self.stream,
        }
    }

    fn verifier(&mut self, options: Options) -> Verifier {
        let config =
            DraftConfig::for_model(&self.draft, self.max_context, 7, DraftPolicy::Adaptive)
                .unwrap();
        let (config, max_draft) = match options {
            Options::Cli => (config, 7),
            Options::Serve => (
                config.with_max_copy_rows(SERVE_COPY_ROWS),
                SERVE_COPY_ROWS - 1,
            ),
        };
        let ctx = DraftContext {
            stream: &self.stream,
            model: &self.draft,
            scratch: &mut self.scratch,
        };
        let mut verifier = Verifier::new(&ctx, config, 1, max_draft).unwrap();
        verifier.set_retain_frontier_logits(options == Options::Serve);
        verifier
    }
}

/// One session's run: the committed sequence and the fixture trace.
struct Lane {
    verifier: Verifier,
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
    /// `Prime` over `prompt`.
    fn prime(rig: &mut Rig, mut verifier: Verifier, prompt: &[u32]) -> Result<Self, String> {
        let mut session = rig.model.new_session()?;
        let first = verifier.prime(&mut rig.exec(), &mut session, prompt, true)?;
        let mut sequence = prompt.to_vec();
        sequence.push(first);
        Ok(Self {
            verifier,
            session,
            stops: rig.model.weights().tokenizer.stop_tokens(),
            sampler: SamplerState::new(SamplingConfig::default(), prompt)
                .map_err(|e| e.to_string())?,
            sequence,
            position: prompt.len() as u32,
            current: first,
            first,
            ids: Vec::new(),
            steps: Vec::new(),
        })
    }

    fn request(&mut self, tokens: usize) -> StepRequest<'_> {
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
    /// its tokens up to `tokens`.
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
}

/// The first divergence of a lane from a fixture entry.
fn compare(lane: &Lane, expected: &Value) -> Option<String> {
    if lane.first != expected["first"].as_u64().unwrap() as u32 {
        return Some(format!(
            "first token {}, C++ {}",
            lane.first, expected["first"]
        ));
    }
    let steps = expected["steps"].as_array().unwrap();
    for (index, (got, want)) in lane.steps.iter().zip(steps).enumerate() {
        let want = u32s(want);
        if *got != want {
            return Some(format!("step {index}: {got:?}, C++ {want:?}"));
        }
    }
    if lane.steps.len() != steps.len() {
        return Some(format!("{} steps, C++ {}", lane.steps.len(), steps.len()));
    }
    (lane.ids != u32s(&expected["ids"])).then(|| "ids differ".to_string())
}

/// `RunVerifier` of the probe: `Prime`, then `VerifyStep` until `tokens`
/// tokens after the first.
fn run(rig: &mut Rig, options: Options, prompt: &[u32], tokens: usize) -> Result<Lane, String> {
    let verifier = rig.verifier(options);
    let mut lane = Lane::prime(rig, verifier, prompt)?;
    while lane.ids.len() < tokens {
        let step = Verifier::verify_step(&mut rig.exec(), &mut lane.request(tokens))?;
        lane.take(&step, None, tokens);
    }
    Ok(lane)
}

fn single_sessions(rig: &mut Rig, quant: Quant, failures: &mut Vec<String>) {
    let prefill = quant.fixture("prefill");
    let fixture = quant.fixture("dflash");
    let tokens = fixture["tokens"].as_u64().unwrap() as usize;
    for (key, options) in [("cases", Options::Cli), ("serve_cases", Options::Serve)] {
        for case in fixture[key].as_array().unwrap() {
            let prompt = case_prompt(&prefill, case);
            assert_eq!(prompt.len() as u64, case["prompt_len"].as_u64().unwrap());
            let what = format!("{}: {key} {}", quant.set, case["prompt"]);
            match run(rig, options, &prompt, tokens) {
                Ok(lane) => {
                    eprintln!(
                        "{what}: {} blocks through the draft vocabulary",
                        lane.verifier.draft().draft_subset_blocks()
                    );
                    if let Some(divergence) = compare(&lane, case) {
                        failures.push(format!("{what}: {divergence}"));
                    }
                }
                Err(e) => failures.push(format!("{what}: {e}")),
            }
        }
    }
}

/// The probe's `RunSpecBatch`: both sessions decode through `VerifyBatch`
/// under the concurrent draft cap (`ConcurrentDraftCap(2)` = 2 for DFlash),
/// the one left alone through `VerifyStep` without it.
fn batch(rig: &mut Rig, quant: Quant) -> Result<Vec<Lane>, String> {
    let prefill = quant.fixture("prefill");
    let fixture = quant.fixture("dflash-batch");
    let tokens = fixture["tokens"].as_u64().unwrap() as usize;
    let mut lanes = Vec::new();
    for entry in fixture["sessions"].as_array().unwrap() {
        let prompt = match entry["prompt"].as_str().unwrap() {
            "haiku" => u32s(&prefill["haiku"]),
            name => common::synthetic(&prefill, name["body".len()..].parse().unwrap()),
        };
        assert_eq!(prompt.len() as u64, entry["prompt_len"].as_u64().unwrap());
        let verifier = rig.verifier(Options::Serve);
        lanes.push(Lane::prime(rig, verifier, &prompt)?);
    }
    loop {
        let mut active: Vec<&mut Lane> =
            lanes.iter_mut().filter(|l| l.ids.len() < tokens).collect();
        match active.len() {
            0 => break,
            1 => {
                let lane = &mut active[0];
                lane.verifier.draft_mut().set_step_draft_cap(u32::MAX);
                let step = Verifier::verify_step(&mut rig.exec(), &mut lane.request(tokens))?;
                lane.take(&step, Some(1), tokens);
            }
            width => {
                for lane in active.iter_mut() {
                    lane.verifier.draft_mut().set_step_draft_cap(2);
                }
                let mut requests: Vec<StepRequest<'_>> =
                    active.iter_mut().map(|lane| lane.request(tokens)).collect();
                let results = Verifier::verify_batch(&mut rig.exec(), &mut requests)?;
                drop(requests);
                for (lane, step) in active.iter_mut().zip(&results) {
                    lane.take(step, Some(width as u32), tokens);
                }
            }
        }
    }
    Ok(lanes)
}

fn batched_sessions(rig: &mut Rig, quant: Quant, failures: &mut Vec<String>) {
    let fixture = quant.fixture("dflash-batch");
    match batch(rig, quant) {
        Ok(lanes) => {
            for (lane, entry) in lanes.iter().zip(fixture["sessions"].as_array().unwrap()) {
                eprintln!(
                    "{}: dflash-batch {}: {} blocks through the draft vocabulary",
                    quant.set,
                    entry["prompt"],
                    lane.verifier.draft().draft_subset_blocks()
                );
                if let Some(divergence) = compare(lane, entry) {
                    failures.push(format!(
                        "{}: dflash-batch {}: {divergence}",
                        quant.set, entry["prompt"]
                    ));
                }
            }
        }
        Err(e) => failures.push(format!("{}: dflash-batch: {e}", quant.set)),
    }
}

/// Proposals of a greedy request (the haiku) and a sampled one (body 100)
/// in one unplanned batch: the greedy block reads the full head, the sampled
/// block its draft vocabulary from the same full rows (the strided
/// selector). The sampled proposal must equal the request's proposal alone,
/// drafted over the prefix and its own table; the greedy request alone could
/// take its draft vocabulary, so only its block runs.
fn mixed_batch(rig: &mut Rig, quant: Quant) -> Result<(), String> {
    let prefill = quant.fixture("prefill");
    let prompts = [u32s(&prefill["haiku"]), common::synthetic(&prefill, 100)];
    let mut lanes = Vec::new();
    for prompt in &prompts {
        let verifier = rig.verifier(Options::Serve);
        lanes.push(Lane::prime(rig, verifier, prompt)?);
    }
    let [greedy, sampled] = &mut lanes[..] else {
        unreachable!("two lanes");
    };
    let mut ctx = DraftContext {
        stream: &rig.stream,
        model: &rig.draft,
        scratch: &mut rig.scratch,
    };
    let saved = [
        greedy.verifier.draft().snapshot(ctx.stream, ctx.model)?,
        sampled.verifier.draft().snapshot(ctx.stream, ctx.model)?,
    ];
    let subset_blocks = sampled.verifier.draft().draft_subset_blocks();
    let seed = 0x5eed_u64;
    let mut rng = seed;
    let batched = Draft::propose_batch(
        &mut ctx,
        &mut [
            ProposalRequest {
                draft: greedy.verifier.draft_mut(),
                tokens: &greedy.sequence,
                position: greedy.position,
                max_tokens: 7,
                sampling: None,
                penalties: None,
            },
            ProposalRequest {
                draft: sampled.verifier.draft_mut(),
                tokens: &sampled.sequence,
                position: sampled.position,
                max_tokens: 7,
                sampling: Some((0.8, &mut rng)),
                penalties: None,
            },
        ],
    )?;
    if sampled.verifier.draft().draft_subset_blocks() == subset_blocks {
        return Err("the sampled request did not draft through its vocabulary".into());
    }
    greedy.verifier.draft_mut().restore(&ctx, &saved[0])?;
    sampled.verifier.draft_mut().restore(&ctx, &saved[1])?;
    let mut rng = seed;
    let alone_sampled = sampled.verifier.draft_mut().propose_with_width(
        &mut ctx,
        &sampled.sequence,
        sampled.position,
        7,
        Some((0.8, &mut rng)),
        None,
        false,
    )?;
    if batched[0].tokens.is_empty() || batched[1].tokens.is_empty() {
        return Err("the mixed batch drafted nothing".into());
    }
    if batched[1] != alone_sampled {
        return Err(format!(
            "sampled {:?}, alone {:?}",
            batched[1].tokens, alone_sampled.tokens
        ));
    }
    Ok(())
}

fn speculative(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let max_context = quant.fixture("dflash")["max_context"].as_u64().unwrap() as u32;
    let mut rig = Rig::load(quant, max_context);
    let mut failures = Vec::new();
    single_sessions(&mut rig, quant, &mut failures);
    batched_sessions(&mut rig, quant, &mut failures);
    if let Err(e) = mixed_batch(&mut rig, quant) {
        failures.push(format!("{}: mixed proposal batch: {e}", quant.set));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn speculative_steps_match_the_cpp_verifier_ud_q4_k_m() {
    speculative(UD_Q4_K_M);
}

#[test]
fn speculative_steps_match_the_cpp_verifier_q8_0() {
    speculative(Q8_0);
}
