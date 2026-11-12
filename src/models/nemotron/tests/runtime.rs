//! The C++ extension's `runtime_test.cpp`, plus the serving runner on top:
//! incremental decode (the single-token graph) against one bulk prefill,
//! exact snapshot restore into another session, invalidation, device logit
//! heads, sessions decoded together against the same sessions decoded alone
//! and, with the LiLiCorr drafter (`GUFO_NEMOTRON_BOLT_DRAFTER`, or NVIDIA's
//! drafter file beside the trunk), a verified block committed in part, a
//! discarded verification, drafts from a restored snapshot and
//! draft-and-verify rounds against plain greedy decoding. The runner checks
//! run the same paths through `TextModelRunner`.
//!
//! GUFO_NEMOTRON_BOLT_GGUF=... cargo test --profile gpu-test -p gufo-nemotron \
//!   --features gpu,models --test runtime -- --nocapture
//!
//! One test loads the trunk once: two loads would not fit beside each other.

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_core::text::{TextModelRunner, TextRunnerDecode};
use gufo_nemotron::engine::{Model, ModelOptions, Session};
use gufo_nemotron::logit_head::{LogitHead, LogitHeadFilter, head_filter_for, sample_with_head};
use gufo_nemotron::runner::{NemotronRunner, NemotronState};
use gufo_nemotron::speculative::{DraftHistory, Frontier, speculative_step};

const CONTEXT: u32 = 4096;

/// Failed checks, reported together at the end as the C++ test does.
#[derive(Default)]
struct Checks {
    failures: Vec<String>,
}

impl Checks {
    fn check(&mut self, ok: bool, what: &str) {
        if !ok {
            eprintln!("FAIL {what}");
            self.failures.push(what.to_string());
        }
    }
}

fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    let log_softmax = |v: &[f32]| -> Vec<f64> {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f64 = v.iter().map(|&x| (f64::from(x) - f64::from(m)).exp()).sum();
        v.iter()
            .map(|&x| f64::from(x) - f64::from(m) - z.ln())
            .collect()
    };
    let p = log_softmax(p_logits);
    let q = log_softmax(q_logits);
    p.iter().zip(&q).map(|(p, q)| p.exp() * (p - q)).sum()
}

/// The first maximum, as `std::max_element`.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn session(model: &Arc<Model>) -> Session {
    model.create_session(CONTEXT).unwrap()
}

/// `GUFO_NEMOTRON_BOLT_DRAFTER`, else `<trunk stem>-LiLiCorr.gguf` beside
/// the trunk.
fn drafter_path(trunk: &std::path::Path) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("GUFO_NEMOTRON_BOLT_DRAFTER").filter(|p| !p.is_empty()) {
        return Some(path.into());
    }
    let stem = trunk.file_stem()?.to_string_lossy();
    let beside = trunk.with_file_name(format!("{stem}-LiLiCorr.gguf"));
    beside.exists().then_some(beside)
}

/// Greedy tokens from `logits` onward, one decode step at a time.
fn greedy(session: &mut Session, mut logits: Vec<f32>, count: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let next = argmax(&logits) as i32;
        out.push(next);
        session.append(&[next], &mut logits).unwrap();
    }
    out
}

/// Device heads of real rows, from a verified block and from a decode step,
/// draw the full rows' tokens for top-k sampling with and without top-p.
fn check_heads(checks: &mut Checks, model: &Arc<Model>, prompt: &[i32], block: &[i32]) {
    let vocab = model.config().vocab_size as usize;
    let mut session = session(model);
    let mut logits = vec![0.0f32; vocab];
    session.append(prompt, &mut logits).unwrap();
    let nucleus = SamplingConfig {
        temperature: 1.0,
        top_k: 40,
        top_p: 0.95,
        ..SamplingConfig::default()
    };
    let narrow = SamplingConfig {
        temperature: 0.6,
        top_k: 20,
        ..SamplingConfig::default()
    };
    for config in [nucleus, narrow] {
        let filter = head_filter_for(&config).unwrap();
        let mut rows = vec![0.0f32; block.len() * vocab];
        let mut heads = vec![LogitHead::default(); block.len()];
        session
            .verify_with_heads(block, &mut rows, &filter, &mut heads)
            .unwrap();
        session.commit(0).unwrap();
        let mut step = Frontier {
            logits: vec![0.0; vocab],
            ..Frontier::default()
        };
        session
            .append_with_head(&block[..1], &mut step.logits, &filter, &mut step.head)
            .unwrap();
        heads.push(step.head);
        rows.extend_from_slice(&step.logits);
        let mut summarized = 0;
        let mut same = true;
        for (r, head) in heads.iter().enumerate() {
            summarized += usize::from(!head.candidates.is_empty());
            let row = &rows[r * vocab..(r + 1) * vocab];
            for seed in 0..16 {
                let seeded = SamplingConfig {
                    seed,
                    ..config.clone()
                };
                let mut a = SamplerState::new(seeded.clone(), &[]).unwrap();
                let mut b = SamplerState::new(seeded, &[]).unwrap();
                same = same
                    && a.sample(row).unwrap() == sample_with_head(&mut b, row, head).unwrap()
                    && a.rng_state() == b.rng_state();
            }
        }
        println!(
            "device heads (T {:.1}, top-k {}): {summarized} of {} rows",
            config.temperature,
            config.top_k,
            heads.len()
        );
        checks.check(summarized * 2 > heads.len(), "most rows get a device head");
        checks.check(same, "device heads draw the full rows' tokens");
        session.reset().unwrap();
        session.append(prompt, &mut logits).unwrap();
    }
}

/// `rows` sessions decoded together through `decode_batch` against the
/// same sessions decoded one token at a time: bit-identical rows and the
/// same positions, and with a drafter the same drafts afterwards.
fn check_batch(checks: &mut Checks, model: &Arc<Model>, prompt: &[i32], rows: usize) {
    const STEPS: usize = 8;
    let vocab = model.config().vocab_size as usize;
    let mut batched = Vec::new();
    let mut serial = Vec::new();
    let mut logits = vec![vec![0.0f32; vocab]; rows];
    let mut reference = vec![vec![0.0f32; vocab]; rows];
    for r in 0..rows {
        // Different lengths, so the rows sit at different positions.
        let part = &prompt[..prompt.len() - r];
        let mut b = session(model);
        let mut s = session(model);
        b.append(part, &mut logits[r]).unwrap();
        s.append(part, &mut reference[r]).unwrap();
        batched.push(b);
        serial.push(s);
    }
    let filter = LogitHeadFilter {
        temperature: 1.0,
        keep: 0,
        top_p: 0.95,
        minimum: 1,
    };
    let mut filters = vec![None; rows];
    filters[1] = Some(filter);
    let mut heads = vec![LogitHead::default(); rows];
    let mut worst = 0.0f64;
    let mut same_tokens = true;
    let mut exact = true;
    for _ in 0..STEPS {
        let tokens: Vec<i32> = reference.iter().map(|row| argmax(row) as i32).collect();
        {
            let mut sessions: Vec<&mut Session> = batched.iter_mut().collect();
            let mut outputs: Vec<&mut [f32]> = logits.iter_mut().map(|row| &mut row[..]).collect();
            Session::decode_batch(&mut sessions, &tokens, &mut outputs, &filters, &mut heads)
                .unwrap();
        }
        for r in 0..rows {
            serial[r].append(&[tokens[r]], &mut reference[r]).unwrap();
            worst = worst.max(kl(&reference[r], &logits[r]));
            same_tokens = same_tokens && argmax(&logits[r]) == argmax(&reference[r]);
            exact = exact && logits[r] == reference[r];
        }
        checks.check(
            !heads[1].candidates.is_empty() && heads[0].candidates.is_empty(),
            "batched rows are summarized with their own filters",
        );
    }
    println!(
        "{rows} batched vs serial rows: worst KL {worst:.2e}, {}",
        if exact {
            "bit-identical"
        } else {
            "rows differ"
        }
    );
    checks.check(exact && same_tokens, "batched rows equal serial rows");
    for r in 0..rows {
        checks.check(
            batched[r].position() == serial[r].position(),
            "batched sessions advance one position per step",
        );
    }
    if batched[0].has_draft() {
        let mut drafts = true;
        for r in 0..rows {
            let anchor = argmax(&reference[r]) as i32;
            let position = batched[r].position();
            let a = batched[r].draft_block(anchor, position).unwrap().unwrap();
            let position = serial[r].position();
            let b = serial[r].draft_block(anchor, position).unwrap().unwrap();
            drafts = drafts && a.candidates == b.candidates;
        }
        checks.check(drafts, "batched sessions feed their drafter contexts");
    }
}

fn check_drafter(checks: &mut Checks, model: &Arc<Model>, prompt: &[i32], prompt_logits: &[f32]) {
    let vocab = model.config().vocab_size as usize;
    // A block verified and committed in part leaves the session where
    // appending that part does.
    let mut a = session(model);
    let mut b = session(model);
    assert!(a.has_draft(), "sessions with drafter");
    let mut logits = vec![0.0f32; vocab];
    a.append(prompt, &mut logits).unwrap();
    b.append(prompt, &mut logits).unwrap();
    let greedy6 = greedy(&mut b, prompt_logits.to_vec(), 6);
    b.reset().unwrap();
    b.append(prompt, &mut logits).unwrap();
    let mut rows = vec![0.0f32; 4 * vocab];
    a.verify(&greedy6[..4], &mut rows).unwrap();
    a.commit(2).unwrap();
    checks.check(
        a.position() as usize == prompt.len() + 2,
        "commit advances by the kept part",
    );
    b.append(&greedy6[..2], &mut []).unwrap();
    let mut after_a = vec![0.0f32; vocab];
    let mut after_b = vec![0.0f32; vocab];
    let next = [greedy6[2]];
    a.append(&next, &mut after_a).unwrap();
    b.append(&next, &mut after_b).unwrap();
    println!(
        "commit 2 of 4 vs append 2: KL {:.2e}",
        kl(&after_b, &after_a)
    );
    checks.check(
        after_a == after_b,
        "a partly committed block matches appending its kept part",
    );
    // The verify rows are the next-token rows of the block's prefixes.
    checks.check(
        argmax(&rows[..vocab]) == greedy6[1] as usize,
        "verify row 0 predicts the next greedy token",
    );

    // A discarded verification leaves no trace.
    let mut c = session(model);
    c.append(prompt, &mut logits).unwrap();
    c.verify(&greedy6[..3], &mut rows[..3 * vocab]).unwrap();
    c.commit(0).unwrap();
    let mut fresh = vec![0.0f32; vocab];
    c.append(&greedy6[..2], &mut []).unwrap();
    c.append(&next, &mut fresh).unwrap();
    checks.check(fresh == after_b, "a discarded verify is forgotten");

    // Drafts depend only on the restored state.
    c.reset().unwrap();
    c.append(prompt, &mut logits).unwrap();
    let saved = c.save().unwrap();
    checks.check(
        !saved.draft.is_empty(),
        "snapshots carry the drafter context",
    );
    let mut d = session(model);
    d.restore(&saved).unwrap();
    let anchor = greedy6[0];
    let position = c.position();
    let from_live = c.draft_block(anchor, position).unwrap().unwrap();
    let position = d.position();
    let from_restored = d.draft_block(anchor, position).unwrap().unwrap();
    checks.check(
        from_live.candidates == from_restored.candidates
            && from_live.scores == from_restored.scores,
        "a restored session drafts exactly as the original",
    );

    // Greedy draft-and-verify rounds emit plain greedy decoding's tokens: a
    // verified row equals the row of decoding its token alone.
    const TOKENS: usize = 48;
    b.reset().unwrap();
    b.append(prompt, &mut logits).unwrap();
    let reference = greedy(&mut b, prompt_logits.to_vec(), TOKENS);
    let mut e = session(model);
    let mut frontier = Frontier {
        logits: vec![0.0; vocab],
        ..Frontier::default()
    };
    e.append(prompt, &mut frontier.logits).unwrap();
    let mut sampler = SamplerState::new(SamplingConfig::default(), &[]).unwrap();
    let mut spec = Vec::new();
    let mut accepted = 0usize;
    let mut rounds = 0usize;
    let mut history = DraftHistory::default();
    while spec.len() < TOKENS {
        let round = speculative_step(
            &mut e,
            &mut sampler,
            &mut frontier,
            &mut history,
            (TOKENS - spec.len()) as u32,
            &|_| false,
            false,
        )
        .unwrap();
        spec.extend_from_slice(&round.tokens);
        accepted += round.accepted as usize;
        rounds += 1;
    }
    let same = spec
        .iter()
        .zip(&reference)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "draft rounds: {rounds} for {} tokens, {accepted} accepted; same as plain greedy for {same} tokens",
        spec.len()
    );
    checks.check(
        e.position() as usize == prompt.len() + spec.len(),
        "rounds run every emitted token",
    );
    checks.check(
        same == TOKENS,
        "draft-and-verify follows plain greedy decoding",
    );
    checks.check(
        accepted >= TOKENS / 4,
        "the drafter's proposals are accepted",
    );
}

/// A runner state with `prompt` prefilled.
fn prefilled(runner: &NemotronRunner, prompt: &[u32]) -> NemotronState {
    let mut state = runner.create_state().unwrap();
    let mut offset = 0;
    loop {
        let step = runner
            .prefill(&mut state, prompt, offset, prompt.len())
            .unwrap();
        offset += step.consumed_tokens;
        if step.decode_ready {
            return state;
        }
    }
}

fn greedy_sampler(prompt: &[u32]) -> SamplerState {
    SamplerState::new(SamplingConfig::default(), prompt).unwrap()
}

/// The serving runner without drafts: greedy selection follows the
/// engine's greedy tokens, a snapshot restores into another state, and
/// requests decoding together emit what each emits alone.
fn check_runner(checks: &mut Checks, model: &Arc<Model>, prompt: &[u32], continuation: &[i32]) {
    let runner = NemotronRunner::new(Arc::clone(model), CONTEXT, 0, None).unwrap();
    let mut state = prefilled(&runner, prompt);
    let snapshot = runner.snapshot(&state).unwrap();
    let mut sampler = greedy_sampler(prompt);
    let mut tokens = Vec::new();
    for _ in 0..continuation.len() {
        let selection = runner.select_next(&mut state, &mut sampler).unwrap();
        tokens.push(selection.token as i32);
        runner.advance(&mut state, selection.token).unwrap();
    }
    checks.check(
        tokens == continuation,
        "the runner decodes the engine's greedy tokens",
    );

    let mut restored = runner.create_state().unwrap();
    runner.restore_or_fork(&mut restored, &snapshot).unwrap();
    checks.check(
        runner.checkpoint_position(&restored).unwrap() == prompt.len(),
        "a restored runner state sits at the snapshot",
    );
    let mut sampler = greedy_sampler(prompt);
    let selection = runner.select_next(&mut restored, &mut sampler).unwrap();
    checks.check(
        selection.token as i32 == continuation[0],
        "a restored runner state selects from the snapshot's frontier",
    );

    let mut states: Vec<NemotronState> = (0..3).map(|_| prefilled(&runner, prompt)).collect();
    let mut samplers: Vec<SamplerState> = (0..3).map(|_| greedy_sampler(prompt)).collect();
    let mut emitted = vec![Vec::new(); 3];
    let mut batched = true;
    for _ in 0..continuation.len() {
        let mut decodes: Vec<TextRunnerDecode<'_, NemotronState, SamplerState>> = states
            .iter_mut()
            .zip(samplers.iter_mut())
            .map(|(state, sampler)| TextRunnerDecode {
                state,
                max_tokens: 1,
                sampler,
            })
            .collect();
        let steps = runner.decode_batch(&mut decodes).unwrap();
        for (out, step) in emitted.iter_mut().zip(steps) {
            let step = step.unwrap();
            batched = batched && step.execution_plan.physical_width == 3;
            out.extend(step.selections.iter().map(|s| s.token as i32));
        }
    }
    checks.check(batched, "concurrent requests decode in one pass");
    checks.check(
        emitted.iter().all(|tokens| tokens == continuation),
        "requests decoding together emit what each emits alone",
    );
}

/// The drafting runner's rounds emit plain greedy decoding's tokens.
fn check_drafting_runner(
    checks: &mut Checks,
    model: &Arc<Model>,
    prompt: &[u32],
    reference: &[i32],
) {
    let runner = NemotronRunner::new(Arc::clone(model), CONTEXT, 7, None).unwrap();
    checks.check(
        runner.descriptor().capabilities.multi_token_decode,
        "a runner with a drafter decodes several tokens per step",
    );
    let mut state = prefilled(&runner, prompt);
    let mut sampler = greedy_sampler(prompt);
    let mut tokens = Vec::new();
    let (mut rounds, mut accepted) = (0usize, 0usize);
    while tokens.len() < reference.len() {
        let step = runner
            .decode_step(&mut state, reference.len() - tokens.len(), &mut sampler)
            .unwrap();
        assert!(!step.stop, "greedy decoding stopped early");
        tokens.extend(step.selections.iter().map(|s| s.token as i32));
        rounds += step.draft_rounds;
        accepted += step.draft_accepted_tokens;
    }
    println!("runner draft rounds: {rounds}, {accepted} accepted");
    checks.check(
        tokens == reference,
        "the drafting runner follows plain greedy decoding",
    );

    // Two requests decoding together share passes for the first 16 tokens
    // of their answers, then draft one round each in turn while that is
    // faster; greedy output stays the same either way.
    let mut states: Vec<NemotronState> = (0..2).map(|_| prefilled(&runner, prompt)).collect();
    let mut samplers: Vec<SamplerState> = (0..2).map(|_| greedy_sampler(prompt)).collect();
    let mut emitted: Vec<Vec<i32>> = vec![Vec::new(); 2];
    let (mut shared, mut in_turn) = (0usize, 0usize);
    let mut first_turn_at = None;
    while emitted.iter().any(|out| out.len() < reference.len()) {
        let mut index = Vec::new();
        let mut decodes = Vec::new();
        for (i, (state, sampler)) in states.iter_mut().zip(samplers.iter_mut()).enumerate() {
            if emitted[i].len() < reference.len() {
                index.push(i);
                decodes.push(TextRunnerDecode {
                    state,
                    max_tokens: reference.len() - emitted[i].len(),
                    sampler,
                });
            }
        }
        let steps = runner.decode_batch(&mut decodes).unwrap();
        let together = index.len() == 2;
        for (i, step) in index.into_iter().zip(steps) {
            let step = step.unwrap();
            assert!(!step.stop, "greedy decoding stopped early");
            if together && step.execution_plan.physical_width == 2 {
                shared += 1;
            } else if together {
                in_turn += 1;
                first_turn_at.get_or_insert(emitted[i].len());
            }
            emitted[i].extend(step.selections.iter().map(|s| s.token as i32));
        }
    }
    println!("two requests: {shared} shared steps, {in_turn} rounds in turn");
    checks.check(
        in_turn > 0 && first_turn_at == Some(16),
        "concurrent requests draft in turn after 16 tokens of their answers",
    );
    checks.check(
        emitted.iter().all(|out| out == reference),
        "requests drafting in turn follow plain greedy decoding",
    );
}

#[test]
fn runtime() {
    let trunk = common::trunk_path();
    let mut model = Arc::new(
        Model::load(
            &trunk,
            ModelOptions {
                max_context: CONTEXT,
            },
        )
        .unwrap(),
    );
    let vocab = model.config().vocab_size as usize;
    let mut checks = Checks::default();

    let ids = model.tokenizer().encode(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nName three prime numbers and \
         explain why they are prime.<|im_end|>\n<|im_start|>assistant\n<think>\n",
        true,
    );
    let prompt: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
    assert!(prompt.len() > 8, "prompt tokenizes");

    // Bulk prefill of the whole prompt.
    let mut bulk = session(&model);
    let mut bulk_logits = vec![0.0f32; vocab];
    bulk.append(&prompt, &mut bulk_logits).unwrap();

    // The same prompt one token at a time through the decode graph.
    let mut step = session(&model);
    let mut step_logits = vec![0.0f32; vocab];
    for &token in &prompt {
        step.append(&[token], &mut step_logits).unwrap();
    }
    let divergence = kl(&bulk_logits, &step_logits);
    println!(
        "bulk vs incremental: KL {divergence:.2e}, top-1 {} vs {}",
        argmax(&bulk_logits),
        argmax(&step_logits)
    );
    checks.check(
        argmax(&bulk_logits) == argmax(&step_logits) && divergence < 2e-2,
        "incremental decode matches the bulk prefill",
    );
    checks.check(
        step.position() as usize == prompt.len(),
        "position counts every token",
    );
    drop(step);

    // Snapshot after the prompt, continue greedily, then replay from the
    // snapshot in a fresh session: the same kernels must give identical
    // bits.
    let saved = bulk.save().unwrap();
    let mut continuation = Vec::new();
    let mut first_run = Vec::new();
    let mut logits = bulk_logits.clone();
    for _ in 0..6 {
        let next = argmax(&logits) as i32;
        continuation.push(next);
        bulk.append(&[next], &mut logits).unwrap();
        first_run.push(logits.clone());
    }
    drop(bulk);
    let mut restored = session(&model);
    restored.restore(&saved).unwrap();
    checks.check(
        restored.position() as usize == prompt.len(),
        "restored position",
    );
    let mut identical = true;
    for (token, expected) in continuation.iter().zip(&first_run) {
        restored.append(&[*token], &mut logits).unwrap();
        identical = identical && logits == *expected;
    }
    checks.check(
        identical,
        "restored session reproduces the continuation exactly",
    );

    // Invalidation empties the session before its next append.
    restored.invalidate();
    checks.check(restored.position() == 0, "invalidate resets the position");
    restored.append(&prompt, &mut logits).unwrap();
    checks.check(
        logits == bulk_logits,
        "invalidated session restarts from scratch",
    );
    drop(restored);

    check_heads(&mut checks, &model, &prompt, &prompt[prompt.len() - 8..]);
    for rows in 2..=8 {
        check_batch(&mut checks, &model, &prompt, rows);
    }
    check_runner(&mut checks, &model, &ids, &continuation);

    match drafter_path(&trunk) {
        Some(path) => {
            // Every session and runner of the trunk is gone.
            Arc::get_mut(&mut model)
                .expect("no session holds the model")
                .load_drafter(&path)
                .unwrap();
            check_drafter(&mut checks, &model, &prompt, &bulk_logits);
            check_batch(&mut checks, &model, &prompt, 3);
            let mut reference_session = session(&model);
            let mut reference_logits = vec![0.0f32; vocab];
            reference_session
                .append(&prompt, &mut reference_logits)
                .unwrap();
            let reference = greedy(&mut reference_session, reference_logits, 48);
            drop(reference_session);
            check_drafting_runner(&mut checks, &model, &ids, &reference);
        }
        None => println!("no drafter: set GUFO_NEMOTRON_BOLT_DRAFTER to check it"),
    }

    assert!(checks.failures.is_empty(), "failed: {:?}", checks.failures);
    println!("runtime checks passed");
}
