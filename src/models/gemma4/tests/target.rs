//! The extension's `target_test.cpp` on the GPU runtime, with the snapshot
//! parts of `continuation_test.cpp` that apply to the engine: decode and
//! prefill against the scalar CPU reference, prefix extension and rewind,
//! snapshots past a sliding-ring wrap (typed, serialized and streamed,
//! shared position blocks, a refused capture, held, rewritten and wrapped
//! blocks, image records), sync lookahead with stops, bulk prefill against
//! exact rows, prefill weight copies; then with the MTP drafter:
//! verification rows round like single-token decode, greedy MTP (with
//! siblings, the small draft head and prompt-lookup copies) reproduces
//! autoregressive output and its logits bit for bit, sessions decoded
//! together reproduce their own output, sampled MTP replays its seed, the
//! shared calibration verifies sampled siblings, and MTP cycles after a
//! restore draft and accept exactly as before it.
//!
//! GUFO_GEMMA4_MODEL=... GUFO_GEMMA4_MTP_MODEL=... cargo test --profile \
//!   gpu-test -p gufo-gemma4 --features gpu,models --test target -- --nocapture
//!
//! One model at a time: each load is dropped before the next.

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_core::text::{ChatMessage, ChatRole};
use gufo_gemma4::chat_template::{ChatOptions, render};
use gufo_gemma4::draft_policy::{DraftCalibrationScope, DraftPolicy};
use gufo_gemma4::engine::{
    BatchDecode, DecodeResult, ImageSpan, Model, ModelOptions, Session, SpeculativeStats,
};
use gufo_gemma4::executor::MAX_DRAFT_TOKENS;
use gufo_gemma4::provider_options::PrefillWeights;
use gufo_gemma4::reference::{Reference, Storage};
use gufo_gemma4::tokenizer::TokenId;
use gufo_gemma4::weights::{GgufShard, ModelWeights};

/// Limits recorded when the runtime was introduced; they are never
/// loosened (`Limits`).
struct Limits {
    decode: f64,
    prefill: f64,
    long_prefill: f64,
}
const DENSE: Limits = Limits {
    decode: 1e-5,
    prefill: 0.015,
    long_prefill: 0.15,
};
const EXPERTS: Limits = Limits {
    decode: 1e-4,
    prefill: 1e-3,
    long_prefill: 0.06,
};

#[derive(Debug, Default)]
struct Stats {
    mean_kl: f64,
    max_kl: f64,
    top1: usize,
    rows: usize,
}

/// Mean and largest KL of `got`'s rows from `reference`'s, and top-1
/// agreement (`Compare`).
fn compare(reference: &[f32], got: &[f32], vocab: usize) -> Stats {
    let mut s = Stats {
        rows: reference.len() / vocab,
        ..Stats::default()
    };
    for r in 0..s.rows {
        let a = &reference[r * vocab..(r + 1) * vocab];
        let b = &got[r * vocab..(r + 1) * vocab];
        let (mut ma, mut mb) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let (mut aa, mut ab) = (0, 0);
        for i in 0..vocab {
            assert!(b[i].is_finite(), "non-finite GPU logit");
            if f64::from(a[i]) > ma {
                ma = f64::from(a[i]);
                aa = i;
            }
            if f64::from(b[i]) > mb {
                mb = f64::from(b[i]);
                ab = i;
            }
        }
        let za: f64 = a.iter().map(|&v| (f64::from(v) - ma).exp()).sum();
        let zb: f64 = b.iter().map(|&v| (f64::from(v) - mb).exp()).sum();
        let (la, lb) = (za.ln() + ma, zb.ln() + mb);
        let mut kl = 0.0;
        for i in 0..vocab {
            let pa = f64::from(a[i]) - la;
            kl += pa.exp() * (pa - (f64::from(b[i]) - lb));
        }
        s.mean_kl += kl / s.rows as f64;
        s.max_kl = s.max_kl.max(kl);
        s.top1 += usize::from(aa == ab);
    }
    s
}

fn argmax(row: &[f32]) -> TokenId {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as TokenId
}

fn logits(session: &mut Session) -> Vec<f32> {
    session.logits().unwrap().to_vec()
}

fn payload(snapshot: &gufo_gemma4::engine::SessionSnapshot) -> Vec<u8> {
    let mut out = vec![0u8; snapshot.size_bytes() as usize];
    snapshot.copy_to(&mut out).unwrap();
    out
}

/// The fixture conversation: turns separated by "@@ user" / "@@ model"
/// (`ConversationTokens`).
fn conversation_tokens(model: &Model) -> Vec<TokenId> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/long_conversation.txt"
    ))
    .unwrap();
    let mut messages = Vec::new();
    let mut role = ChatRole::User;
    let mut open = false;
    let mut body = String::new();
    let flush = |role: ChatRole, open: bool, body: &mut String, messages: &mut Vec<ChatMessage>| {
        if open {
            while body.ends_with('\n') {
                body.pop();
            }
            messages.push(ChatMessage::new(role, body.clone()));
        }
        body.clear();
    };
    for line in text.lines() {
        if line == "@@ user" || line == "@@ model" {
            flush(role, open, &mut body, &mut messages);
            role = if line == "@@ user" {
                ChatRole::User
            } else {
                ChatRole::Assistant
            };
            open = true;
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    flush(role, open, &mut body, &mut messages);
    let rendered = render(&messages, &[], &ChatOptions::default(), None).unwrap();
    model.tokenize(&rendered.text)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn prompt_tokens(model: &Model, text: &str) -> Vec<TokenId> {
    let message = ChatMessage::new(ChatRole::User, text);
    let rendered = render(&[message], &[], &ChatOptions::default(), None).unwrap();
    model.tokenize(&rendered.text)
}

/// Logits of every row, evaluated `chunk` tokens per forward (`GpuLogits`).
fn gpu_logits(model: &Arc<Model>, tokens: &[TokenId], chunk: usize) -> Vec<f32> {
    let mut session = model.create_session(0).unwrap();
    let mut all = Vec::new();
    let mut part = Vec::new();
    for piece in tokens.chunks(chunk) {
        session.evaluate_all(piece, &mut part).unwrap();
        all.extend_from_slice(&part);
    }
    all
}

/// Tokens from autoregressive evaluation or from `decode_step` (greedy by
/// default; `Generate`).
fn generate(
    model: &Arc<Model>,
    prompt: &[TokenId],
    count: usize,
    speculative: bool,
    config: &SamplingConfig,
) -> (Vec<TokenId>, SpeculativeStats) {
    let mut session = model.create_session(0).unwrap();
    session.sync(prompt).unwrap();
    let mut sampler = SamplerState::new(config.clone(), &[]).unwrap();
    let mut out = Vec::new();
    if !speculative {
        while out.len() < count {
            let token = sampler.sample(session.logits().unwrap()).unwrap();
            sampler.accept(token).unwrap();
            out.push(token as TokenId);
            session.evaluate(token as TokenId).unwrap();
        }
        return (out, SpeculativeStats::default());
    }
    while out.len() < count {
        let step = session
            .decode_step(count - out.len(), &mut sampler, false)
            .unwrap();
        assert!(!step.tokens.is_empty());
        out.extend_from_slice(&step.tokens);
    }
    let stats = *session.statistics();
    println!(
        "speculative: cycles {}, drafted {}, accepted {}, copied {}, copies accepted {}, siblings {}, siblings accepted {}",
        stats.cycles,
        stats.drafted,
        stats.accepted,
        stats.copied,
        stats.copied_accepted,
        stats.siblings,
        stats.siblings_accepted
    );
    (out, stats)
}

/// Greedy speculation against autoregressive decode, logits included:
/// after every cycle the session's logits equal decode's bit for bit
/// (`CheckSpeculativeLogits`).
fn check_speculative_logits(
    model: &Arc<Model>,
    prompt: &[TokenId],
    count: usize,
    total: &mut SpeculativeStats,
) {
    let mut session = model.create_session(0).unwrap();
    session.sync(prompt).unwrap();
    let mut sampler = SamplerState::new(SamplingConfig::default(), &[]).unwrap();
    let mut ar = Vec::new();
    let mut ar_logits = Vec::new();
    while ar.len() < count {
        let logits = session.logits().unwrap().to_vec();
        let token = sampler.sample(&logits).unwrap();
        sampler.accept(token).unwrap();
        ar_logits.push(logits);
        ar.push(token as TokenId);
        session.evaluate(token as TokenId).unwrap();
    }
    // A fresh session: re-syncing this one would keep its cached prompt
    // keys, written by the forward of a different width.
    let mut session = model.create_session(0).unwrap();
    session.sync(prompt).unwrap();
    let mut sampler = SamplerState::new(SamplingConfig::default(), &[]).unwrap();
    let mut spec = Vec::new();
    while spec.len() < count {
        let step = session
            .decode_step(count - spec.len(), &mut sampler, false)
            .unwrap();
        assert!(!step.tokens.is_empty());
        spec.extend_from_slice(&step.tokens);
        let logits = session.logits().unwrap();
        assert!(
            logits == ar_logits[spec.len() - 1].as_slice(),
            "speculative logits differ from decode at token {}",
            spec.len() - 1
        );
    }
    assert_eq!(spec, ar, "greedy MTP differs from AR");
    let stats = session.statistics();
    total.siblings += stats.siblings;
    total.siblings_accepted += stats.siblings_accepted;
}

fn speculation(path: &Path, draft: &Path) {
    let mut options = ModelOptions {
        max_context: 3072,
        prefill_chunk: 240,
        mtp_model_path: Some(draft.to_path_buf()),
        // The longest chain the engine allows: 16-row verification.
        draft_tokens: MAX_DRAFT_TOKENS,
        // Seeded sampled replay is exact when the calibrated policy restarts
        // its calibration with every request.
        draft_calibration: DraftCalibrationScope::Request,
        ..ModelOptions::default()
    };
    let mtp = Model::load(path, options.clone()).unwrap();
    let greedy = SamplingConfig::default();
    let tokens = prompt_tokens(&mtp, "Write one sentence about the sea near Genoa.");
    // With a drafter, verification rows round exactly like decode...
    let single = gpu_logits(&mtp, &tokens, 1);
    assert!(
        single == gpu_logits(&mtp, &tokens, 5) && single == gpu_logits(&mtp, &tokens, 16),
        "verification rows differ from single-token decode"
    );
    // ...so greedy speculation reproduces autoregressive output.
    let mut greedy_stats = SpeculativeStats::default();
    for text in [
        "Write one sentence about the sea near Genoa.",
        "List the first twelve prime numbers, separated by commas.",
        "Explain in three sentences why the sky is blue.",
    ] {
        let prompt = prompt_tokens(&mtp, text);
        let (ar, _) = generate(&mtp, &prompt, 64, false, &greedy);
        let (spec, stats) = generate(&mtp, &prompt, 64, true, &greedy);
        assert_eq!(ar, spec, "greedy MTP differs from AR: {text}");
        greedy_stats.small_head_cycles += stats.small_head_cycles;
        greedy_stats.small_head_drafted += stats.small_head_drafted;
        greedy_stats.small_head_accepted += stats.small_head_accepted;
    }
    println!(
        "small head: cycles {}, drafted {}, accepted {}",
        greedy_stats.small_head_cycles,
        greedy_stats.small_head_drafted,
        greedy_stats.small_head_accepted
    );
    if mtp.config().has_experts() {
        assert!(
            greedy_stats.small_head_cycles > 0
                && greedy_stats.small_head_accepted * 2 >= greedy_stats.small_head_drafted,
            "small draft head accepts too few drafts"
        );
    } else {
        assert_eq!(
            greedy_stats.small_head_cycles, 0,
            "the dense model drafted on the small head"
        );
    }
    // Siblings: an accepted one's row and the keys it leaves in the cache
    // continue exactly as decode would.
    let mut sibling_stats = SpeculativeStats::default();
    for text in [
        "Write a paragraph about the history of Genoa's harbour.",
        "Describe an autumn morning in a mountain village.",
        "Explain how a lighthouse lens concentrates light.",
    ] {
        check_speculative_logits(&mtp, &prompt_tokens(&mtp, text), 96, &mut sibling_stats);
    }
    println!(
        "siblings: verified {}, accepted {}",
        sibling_stats.siblings, sibling_stats.siblings_accepted
    );
    assert!(
        sibling_stats.siblings_accepted > 0,
        "greedy MTP accepted no siblings"
    );

    // Batched decoding: sessions sharing a forward reproduce their own
    // single-session output (greedy), for plain and speculative steps.
    let texts = [
        "Write one sentence about the sea near Genoa.",
        "List the first twelve prime numbers, separated by commas.",
        "Explain in three sentences why the sky is blue.",
    ];
    let mut want = Vec::new();
    let mut sessions: Vec<Session> = Vec::new();
    for text in texts {
        let prompt = prompt_tokens(&mtp, text);
        want.push(generate(&mtp, &prompt, 48, false, &greedy).0);
        let mut session = mtp.create_session(0).unwrap();
        session.sync(&prompt).unwrap();
        sessions.push(session);
    }
    // Plain decode: one token per session per forward.
    let mut got: Vec<Vec<TokenId>> = vec![Vec::new(); texts.len()];
    let mut samplers: Vec<SamplerState> = (0..texts.len())
        .map(|_| SamplerState::new(greedy.clone(), &[]).unwrap())
        .collect();
    while got[0].len() < 48 {
        let mut next = Vec::new();
        for (i, session) in sessions.iter_mut().enumerate() {
            let token = samplers[i].sample(session.logits().unwrap()).unwrap();
            samplers[i].accept(token).unwrap();
            got[i].push(token as TokenId);
            next.push(token as TokenId);
        }
        let mut refs: Vec<&mut Session> = sessions.iter_mut().collect();
        Session::evaluate_batch(&mut refs, &next).unwrap();
    }
    assert_eq!(got, want, "batched decode differs from single sessions");
    // Speculative decode: every session drafts and all verify together, in
    // fresh sessions as `want`'s.
    let mut spec: Vec<Vec<TokenId>> = vec![Vec::new(); texts.len()];
    for (i, text) in texts.iter().enumerate() {
        sessions[i] = mtp.create_session(0).unwrap();
        sessions[i].sync(&prompt_tokens(&mtp, text)).unwrap();
        samplers[i] = SamplerState::new(greedy.clone(), &[]).unwrap();
    }
    loop {
        let mut batch: Vec<BatchDecode<'_>> = Vec::new();
        let mut which = Vec::new();
        for (i, (session, sampler)) in sessions.iter_mut().zip(samplers.iter_mut()).enumerate() {
            if spec[i].len() < 48 {
                batch.push(BatchDecode {
                    session,
                    max_tokens: 48 - spec[i].len(),
                    sampler,
                    result: DecodeResult::default(),
                    stop_at_eos: false,
                    error: None,
                });
                which.push(i);
            }
        }
        if batch.is_empty() {
            break;
        }
        let ok = Session::decode_batch(&mut batch);
        assert!(
            ok,
            "{:?}",
            batch.iter().map(|b| b.error.clone()).collect::<Vec<_>>()
        );
        for (b, &i) in batch.iter().zip(&which) {
            spec[i].extend_from_slice(&b.result.tokens);
        }
    }
    assert_eq!(spec, want, "batched speculative decode differs from AR");

    // Prompt lookup: repeating a passage from the prompt proposes copies,
    // verified like drafts, so greedy output still equals AR.
    let copy_prompt = prompt_tokens(
        &mtp,
        "Repeat the following paragraph word for word, with nothing else:\n\n\
         A harbour town usually begins with a sheltered cove and a handful of \
         families who fish close to shore. The first boats are small, pulled \
         up on the beach at night, and the village lives by the rhythm of the \
         seasons: anchovies in spring, sardines in summer, storms that keep \
         everyone ashore in winter.",
    );
    let (copy_ar, _) = generate(&mtp, &copy_prompt, 64, false, &greedy);
    let (copy_spec, copy_stats) = generate(&mtp, &copy_prompt, 64, true, &greedy);
    assert_eq!(
        copy_ar, copy_spec,
        "greedy MTP with prompt lookup differs from AR"
    );
    assert!(
        copy_stats.copied_accepted > 0,
        "prompt lookup proposed no accepted copies"
    );
    // Sampled MTP (a chat front end's sampler): drafts are sampled and
    // verified by p/q rejection; a seed replays the same tokens.
    let mut chat = SamplingConfig {
        temperature: 1.0,
        top_k: 64,
        top_p: 0.95,
        repeat_penalty: 1.05,
        seed: 7,
        ..SamplingConfig::default()
    };
    let prompt = prompt_tokens(&mtp, "Write a short story about a lighthouse keeper.");
    let (first, stats) = generate(&mtp, &prompt, 96, true, &chat);
    let (again, _) = generate(&mtp, &prompt, 96, true, &chat);
    assert!(
        first.len() == 96 && first == again,
        "sampled MTP does not replay its seed"
    );
    assert!(stats.accepted > 0, "sampled MTP accepted no drafts");
    // Siblings depend on the rows a batch leaves: exact replay drafts none.
    assert_eq!(
        stats.siblings, 0,
        "exact-replay sampled MTP drafted siblings"
    );
    // Sampled copies are point-mass proposals under the same rule.
    let (sampled_copy, sampled_copy_stats) = generate(&mtp, &copy_prompt, 64, true, &chat);
    assert_eq!(
        sampled_copy,
        generate(&mtp, &copy_prompt, 64, true, &chat).0,
        "sampled MTP with prompt lookup does not replay its seed"
    );
    assert!(
        sampled_copy_stats.copied_accepted > 0,
        "sampled prompt lookup accepted no copies"
    );
    // MTP cycles after a restore draft and accept exactly as before it:
    // with fixed chains every draft depends on the restored hidden state.
    drop(sessions);
    drop(mtp);
    let fixed = Model::load(
        path,
        ModelOptions {
            draft_policy: DraftPolicy::Fixed,
            ..options.clone()
        },
    )
    .unwrap();
    {
        let story = fixed.tokenize(
            "<bos><|turn>user\nWrite a short story about a lighthouse keeper who finds a message in a bottle.<turn|>\n<|turn>model\n",
        );
        let mut c = fixed.create_session(0).unwrap();
        c.sync(&story).unwrap();
        let checkpoint = c.save_snapshot().unwrap();
        let speculate = |s: &mut Session| {
            let mut sampler = SamplerState::new(SamplingConfig::default(), &[]).unwrap();
            let mut cycles = Vec::new();
            for _ in 0..4 {
                let before = *s.statistics();
                let step = s.decode_step(64, &mut sampler, false).unwrap();
                let after = *s.statistics();
                cycles.push((
                    step.tokens,
                    after.drafted - before.drafted,
                    after.accepted - before.accepted,
                ));
            }
            cycles
        };
        let expected = speculate(&mut c);
        assert!(
            expected[0].1 > 0,
            "the first cycle drafted nothing from the restored state"
        );
        let mut d = fixed.create_session(0).unwrap();
        d.restore_payload(&payload(&checkpoint)).unwrap();
        d.begin_request();
        assert!(speculate(&mut d) == expected, "restored MTP cycles diverge");
        let mut e = fixed.create_session(0).unwrap();
        e.restore_snapshot(&checkpoint).unwrap();
        e.begin_request();
        assert!(
            speculate(&mut e) == expected,
            "restored MTP cycles diverge (typed)"
        );
        let summary: Vec<String> = expected
            .iter()
            .map(|c| format!("{}/{}", c.2, c.1))
            .collect();
        println!(
            "mtp cycles (accepted/drafted): {}, identical after restore",
            summary.join(" ")
        );
    }
    drop(fixed);
    // The shared default verifies sampled siblings by recursive rejection
    // sampling: some are drafted and some accepted.
    options.draft_calibration = DraftCalibrationScope::Shared;
    let shared = Model::load(path, options).unwrap();
    let mut shared_stats = SpeculativeStats::default();
    for i in 0..3 {
        chat.seed = 7 + i;
        let (tokens, seed_stats) = generate(&shared, &prompt, 96, true, &chat);
        assert_eq!(tokens.len(), 96, "shared sampled MTP stopped early");
        shared_stats.siblings += seed_stats.siblings;
        shared_stats.siblings_accepted += seed_stats.siblings_accepted;
    }
    println!(
        "sampled siblings: verified {}, accepted {}",
        shared_stats.siblings, shared_stats.siblings_accepted
    );
    assert!(
        shared_stats.siblings > 0 && shared_stats.siblings_accepted > 0,
        "shared sampled MTP verified or accepted no siblings"
    );
}

fn plain(path: &Path) {
    let options = ModelOptions {
        // A small prefill chunk keeps the sliding ring short (window + 256),
        // so the snapshot checks wrap it inside this context.
        max_context: 3072,
        prefill_chunk: 240,
        ..ModelOptions::default()
    };
    let model = Model::load(path, options.clone()).unwrap();
    let vocab = model.vocab_size() as usize;
    let limits = if model.config().has_experts() {
        &EXPERTS
    } else {
        &DENSE
    };
    let tokens = prompt_tokens(&model, "Write one sentence about the sea near Genoa.");
    assert!(tokens.len() > 16, "prompt too short for the prefill path");

    // Independent scalar reference over the same weights.
    let oracle = {
        let shard = GgufShard::parse(model.files()[0].bytes()).unwrap();
        let shards = [shard];
        let weights = ModelWeights::bind(&shards).unwrap();
        let mut reference = Reference::new(&weights, &shards, Storage::HalfKv);
        reference.forward(&tokens, &[], true).unwrap().0
    };
    let decode = gpu_logits(&model, &tokens, 1);
    let chunked = gpu_logits(&model, &tokens, 8);
    let prefill = gpu_logits(&model, &tokens, tokens.len());
    let d = compare(&oracle, &decode, vocab);
    let c = compare(&oracle, &chunked, vocab);
    let p = compare(&oracle, &prefill, vocab);
    println!(
        "decode:  mean KL {}, max {}, top-1 {}/{}",
        d.mean_kl, d.max_kl, d.top1, d.rows
    );
    println!(
        "8-row:   mean KL {}, top-1 {}/{}",
        c.mean_kl, c.top1, c.rows
    );
    println!(
        "prefill: mean KL {}, max {}, top-1 {}/{}",
        p.mean_kl, p.max_kl, p.top1, p.rows
    );
    assert!(
        d.mean_kl < limits.decode && d.top1 == d.rows,
        "decode diverges from the reference"
    );
    assert!(
        c.mean_kl < limits.decode && c.top1 == c.rows,
        "small-batch rows diverge from the reference"
    );
    assert!(
        p.mean_kl < limits.prefill,
        "prefill exceeds its KL envelope"
    );

    // Extending a synced prompt matches evaluating the tokens one by one.
    let split = tokens.len() - 4;
    let head = &tokens[..split];
    let mut a = model.create_session(0).unwrap();
    let mut b = model.create_session(0).unwrap();
    a.sync(head).unwrap();
    b.sync(head).unwrap();
    for &t in &tokens[split..] {
        a.evaluate(t).unwrap();
    }
    b.sync(&tokens).unwrap();
    let ext = compare(&logits(&mut a), &logits(&mut b), vocab);
    assert!(
        ext.top1 == 1 && ext.mean_kl < limits.decode,
        "prefix extension diverges from token-by-token evaluation"
    );
    // Rewinding to a shorter prompt reuses the prefix and matches a fresh
    // sync.
    b.sync(head).unwrap();
    let mut fresh = model.create_session(0).unwrap();
    fresh.sync(head).unwrap();
    let rw = compare(&logits(&mut fresh), &logits(&mut b), vocab);
    assert!(
        rw.top1 == 1 && rw.mean_kl < limits.prefill,
        "rewound session diverges from a fresh one"
    );
    drop((a, b, fresh));

    // A snapshot taken after the ring wrapped restores into a fresh session
    // (typed and serialized) and continues bitwise like the original.
    let mut text = String::new();
    for i in 0..90 {
        text += &format!("Ship {i} left Genoa at dawn carrying salt, wine and letters. ");
    }
    let long_prompt = prompt_tokens(&model, &text);
    assert!(
        long_prompt.len() > model.session_ring_slots() + 64
            && long_prompt.len() + 16 < options.max_context as usize,
        "snapshot prompt ({} tokens) does not wrap the sliding ring ({} slots)",
        long_prompt.len(),
        model.session_ring_slots()
    );
    let mut original = model.create_session(0).unwrap();
    original.sync(&long_prompt).unwrap();
    let snapshot = original.save_snapshot().unwrap();
    assert_eq!(snapshot.size_bytes(), original.snapshot_bytes());
    let bytes = payload(&snapshot);
    let mut streamed = Vec::new();
    snapshot
        .stream(&mut |piece| streamed.extend_from_slice(piece))
        .unwrap();
    assert!(
        streamed == bytes,
        "streamed snapshot payload differs from its copy"
    );
    let mut typed = model.create_session(0).unwrap();
    let mut serialized = model.create_session(0).unwrap();
    typed.restore_snapshot(&snapshot).unwrap();
    serialized.restore_payload(&bytes).unwrap();
    for step in 0..=8 {
        let want = logits(&mut original);
        assert!(
            logits(&mut typed) == want && logits(&mut serialized) == want,
            "restored session diverges at step {step}"
        );
        if step == 8 {
            break;
        }
        let next = argmax(&want);
        original.evaluate(next).unwrap();
        typed.evaluate(next).unwrap();
        serialized.evaluate(next).unwrap();
    }
    // Snapshots of one history share their position blocks: shared and
    // freshly copied snapshots of equal states hold equal payloads.
    let from_bytes = |bytes: &[u8]| {
        let mut session = model.create_session(0).unwrap();
        session.restore_payload(bytes).unwrap();
        session
    };
    {
        let mut plain = from_bytes(&bytes);
        for &t in &original.tokens()[long_prompt.len()..].to_vec() {
            plain.evaluate(t).unwrap();
        }
        let reference = plain.save_snapshot().unwrap();
        let shared = original.save_snapshot().unwrap();
        let adopted = typed.save_snapshot().unwrap();
        assert!(
            payload(&shared) == payload(&reference) && payload(&adopted) == payload(&reference),
            "shared snapshot payload differs from a fresh copy"
        );
    }
    {
        // Without device headroom a capture fails cleanly, and the session
        // and its earlier snapshots are unchanged.
        let before = original.save_snapshot().unwrap();
        gufo_hip::cap_reported_device_free(0);
        let refused = original.save_snapshot();
        gufo_hip::cap_reported_device_free(usize::MAX);
        assert!(
            refused.is_err(),
            "snapshot storage ignored the device reserve"
        );
        let after = original.save_snapshot().unwrap();
        assert!(
            payload(&after) == payload(&before),
            "a refused capture changed the session"
        );
    }
    let mut other = String::new();
    for i in 0..60 {
        other +=
            &format!("Harbour {i} counted barrels of oil, crates of lemons and coils of rope. ");
    }
    let other_tokens = prompt_tokens(&model, &other);
    let n1 = long_prompt.len();
    let window = model.config().sliding_window as usize;
    let ring = model.session_ring_slots();
    // The first whole sliding block of the snapshot's window, and enough
    // tokens past the snapshot to overwrite 16 of its ring slots.
    let sliding_block = (n1 - (window - 1)).div_ceil(64) * 64;
    let wrap = sliding_block + ring + 16 - n1;
    assert!(
        sliding_block + 64 <= n1
            && other_tokens.len() >= wrap
            && n1 + wrap + 16 < options.max_context as usize,
        "snapshot test cannot wrap the ring over a sliding block"
    );
    let mut check_restore = |prompt: &[TokenId], what: &str| {
        original.sync(prompt).unwrap();
        original.restore_snapshot(&snapshot).unwrap();
        let mut plain = from_bytes(&bytes);
        for step in 0..4 {
            let want = logits(&mut plain);
            assert!(
                logits(&mut original) == want,
                "{what}: restore diverges at step {step}"
            );
            let next = argmax(&want);
            original.evaluate(next).unwrap();
            plain.evaluate(next).unwrap();
        }
        let shared = original.save_snapshot().unwrap();
        let reference = plain.save_snapshot().unwrap();
        assert!(
            payload(&shared) == payload(&reference),
            "{what}: shared snapshot payload differs from a fresh copy"
        );
    };
    check_restore(&long_prompt, "held blocks");
    let mut rewritten = long_prompt[..long_prompt.len() - 200].to_vec();
    rewritten.extend_from_slice(&other_tokens[..100]);
    check_restore(&rewritten, "rewritten blocks");
    let mut wrapped = long_prompt.clone();
    wrapped.extend_from_slice(&other_tokens[..wrap]);
    check_restore(&wrapped, "wrapped ring");
    let mut corrupt = bytes.clone();
    corrupt.pop();
    assert!(
        serialized.restore_payload(&corrupt).is_err(),
        "truncated snapshot was accepted"
    );
    drop((original, typed, serialized));

    // A snapshot with image records restores them: the restored session
    // keeps the images when the prompt grows past them, and evaluates from
    // an image again when it changes, as the original does. One-row images
    // with zero embeddings sit at odd positions, each followed by a text
    // token.
    {
        let hidden = model.config().hidden_size as usize;
        let stream = gufo_hip::Stream::new().unwrap();
        let mut zeros = stream.alloc_persistent::<f32>(hidden).unwrap();
        stream.fill_bytes(zeros.as_mut_slice(), 0).unwrap();
        let zeros = Arc::new(zeros.freeze(&stream).unwrap());
        let embed = |_: usize| Ok(Arc::clone(&zeros));
        let mut images = Vec::new();
        for i in 0..5u32 {
            let mut identity = [0u8; 32];
            identity[0] = i as u8;
            images.push(ImageSpan {
                offset: 1 + 2 * i,
                rows: 1,
                identity,
            });
        }
        let base = &long_prompt[..400];
        let mut a = model.create_session(0).unwrap();
        a.sync_with(base, &images[..4], &embed, &[], &[]).unwrap();
        let snapshot = a.save_snapshot().unwrap();
        let mut b = model.create_session(0).unwrap();
        b.restore_payload(&payload(&snapshot)).unwrap();
        assert!(b.images() == &images[..4], "restore lost image records");
        let grown = &long_prompt[..420];
        a.sync_with(grown, &images, &embed, &[], &[]).unwrap();
        b.sync_with(grown, &images, &embed, &[], &[]).unwrap();
        assert!(
            logits(&mut a) == logits(&mut b),
            "the restored session evaluated differently past its images"
        );
        let mut changed = images.clone();
        changed[2].identity[1] = 1;
        a.sync_with(grown, &changed, &embed, &[], &[]).unwrap();
        b.restore_snapshot(&snapshot).unwrap();
        b.sync_with(grown, &changed, &embed, &[], &[]).unwrap();
        assert!(
            logits(&mut a) == logits(&mut b),
            "a changed image evaluates differently after a restore"
        );
        println!("images: records restore with their rows");
    }

    // Sync's lookahead runs in the prompt's last forward while the session
    // ends at the boundary; the adopting sync continues bitwise.
    {
        let tail_len = 7;
        let head = &long_prompt[..long_prompt.len() - tail_len];
        let tail = &long_prompt[head.len()..];
        let none = |_: usize| Err("no images".to_string());
        let mut whole = model.create_session(0).unwrap();
        let mut ahead = model.create_session(0).unwrap();
        whole.sync(&long_prompt).unwrap();
        ahead.sync_with(head, &[], &none, tail, &[]).unwrap();
        assert_eq!(ahead.position() as usize, head.len(), "lookahead sync");
        let boundary = ahead.save_snapshot().unwrap();
        ahead.sync(&long_prompt).unwrap();
        assert_eq!(
            ahead.position() as usize,
            long_prompt.len(),
            "lookahead adoption"
        );
        let first = logits(&mut whole);
        let adopted = compare(&first, &logits(&mut ahead), vocab);
        assert!(
            adopted.top1 == 1 && adopted.mean_kl < limits.decode,
            "adopted lookahead logits diverge"
        );
        for step in 0..8 {
            let next = argmax(&logits(&mut whole));
            whole.evaluate(next).unwrap();
            ahead.evaluate(next).unwrap();
            assert!(
                logits(&mut ahead) == logits(&mut whole),
                "adopted lookahead diverges at step {step}"
            );
        }
        let mut restored = model.create_session(0).unwrap();
        restored.restore_snapshot(&boundary).unwrap();
        assert_eq!(restored.position() as usize, head.len());
        restored.sync(&long_prompt).unwrap();
        let resumed = compare(&first, &logits(&mut restored), vocab);
        assert!(
            resumed.top1 == 1 && resumed.mean_kl < limits.prefill,
            "boundary snapshot diverges after its tail"
        );
    }
    // A longer lookahead with stops rides in the same forward; syncs that
    // stop at each stop adopt in turn.
    {
        let tail_len = 40;
        let stops = [5usize, 23];
        let head = &long_prompt[..long_prompt.len() - tail_len];
        let tail = &long_prompt[head.len()..];
        let none = |_: usize| Err("no images".to_string());
        let mut ahead = model.create_session(0).unwrap();
        ahead.sync_with(head, &[], &none, tail, &stops).unwrap();
        assert!(
            ahead.position() as usize == head.len() && ahead.lookahead_tokens() == tail_len,
            "lookahead with stops"
        );
        for stop in [5usize, 23, tail_len] {
            let prefix = &long_prompt[..head.len() + stop];
            let mut direct = model.create_session(0).unwrap();
            direct.sync(prefix).unwrap();
            ahead.sync(prefix).unwrap();
            assert!(
                ahead.position() as usize == prefix.len()
                    && ahead.lookahead_tokens() == tail_len - stop,
                "lookahead stop {stop} not adopted"
            );
            let adopted = compare(&logits(&mut direct), &logits(&mut ahead), vocab);
            assert!(
                adopted.top1 == 1 && adopted.mean_kl < limits.decode,
                "lookahead stop {stop} diverges: {adopted:?}"
            );
            let next = argmax(&logits(&mut direct));
            if stop == 5 {
                // A sync between stops keeps the rows it shares with the
                // lookahead and evaluates its last token again, as a decode
                // row.
                let mid = &long_prompt[..head.len() + 12];
                let mut between = model.create_session(0).unwrap();
                let mut reference = model.create_session(0).unwrap();
                reference.sync(&mid[..mid.len() - 1]).unwrap();
                reference.evaluate(mid[mid.len() - 1]).unwrap();
                between.sync_with(head, &[], &none, tail, &stops).unwrap();
                between.sync(mid).unwrap();
                assert!(
                    between.position() as usize == mid.len() && between.lookahead_tokens() == 0,
                    "sync between lookahead stops"
                );
                let kept = compare(&logits(&mut reference), &logits(&mut between), vocab);
                println!(
                    "sync between lookahead stops: mean KL {}, top-1 {}/{}",
                    kept.mean_kl, kept.top1, kept.rows
                );
                assert!(
                    kept.top1 == 1 && kept.mean_kl < limits.decode,
                    "sync between lookahead stops diverges"
                );
            }
            if stop == tail_len {
                direct.evaluate(next).unwrap();
                ahead.evaluate(next).unwrap();
                let step = compare(&logits(&mut direct), &logits(&mut ahead), vocab);
                assert!(
                    step.top1 == 1 && step.mean_kl < limits.decode,
                    "decoding after the adopted lookahead diverges"
                );
            }
        }
    }

    // Bulk prefill over many attention tiles and a wrapped ring stays close
    // to the exact small-batch rows on a realistic conversation.
    let conversation = conversation_tokens(&model);
    assert!(
        conversation.len() > model.session_ring_slots() + 64
            && conversation.len() + 16 < options.max_context as usize,
        "conversation ({} tokens) does not wrap the sliding ring",
        conversation.len()
    );
    {
        let exact = gpu_logits(&model, &conversation, 8);
        let bulk = gpu_logits(&model, &conversation, conversation.len());
        let lp = compare(&exact, &bulk, vocab);
        println!(
            "long prefill vs exact rows ({} tokens): mean KL {}, max {}, top-1 {}/{}",
            lp.rows, lp.mean_kl, lp.max_kl, lp.top1, lp.rows
        );
        assert!(
            lp.mean_kl < limits.long_prefill,
            "long prefill exceeds its KL envelope"
        );
    }
    {
        // Reported only: the repeated snapshot prompt is ill-conditioned.
        let exact = gpu_logits(&model, &long_prompt, 8);
        let bulk = gpu_logits(&model, &long_prompt, long_prompt.len());
        let lp = compare(&exact, &bulk, vocab);
        println!(
            "repetitive prefill vs exact rows ({} tokens, reported): mean KL {}, top-1 {}/{}",
            lp.rows, lp.mean_kl, lp.top1, lp.rows
        );
    }
    drop(model);

    // Prefill weight copies over one 2048-row forward of the conversation:
    // K-quant and Q4_0 copies reproduce the GGUF GEMMs bit for bit; a Q8_0
    // copy is held to the prefill envelope when it rounds differently.
    let mut want = Vec::new();
    for mode in [
        PrefillWeights::Off,
        PrefillWeights::Compact,
        PrefillWeights::Decoded,
    ] {
        let copied = Model::load(
            path,
            ModelOptions {
                max_context: options.max_context,
                prefill_weights: mode,
                ..ModelOptions::default()
            },
        )
        .unwrap();
        let conversation = conversation_tokens(&copied);
        let got = gpu_logits(&copied, &conversation, conversation.len());
        if mode == PrefillWeights::Off {
            want = got;
            continue;
        }
        let name = if mode == PrefillWeights::Compact {
            "compact"
        } else {
            "decoded"
        };
        if got == want {
            println!("prefill weights {name}: bit-identical");
            continue;
        }
        let st = compare(&want, &got, vocab);
        println!(
            "prefill weights {name}: mean KL {}, max {}, top-1 {}/{}",
            st.mean_kl, st.max_kl, st.top1, st.rows
        );
        let q8 = copied.weights().layers.iter().any(|l| {
            [
                &l.attn_q,
                &l.attn_k,
                &l.attn_v,
                &l.attn_output,
                &l.ffn_gate,
                &l.ffn_up,
                &l.ffn_down,
            ]
            .iter()
            .any(|t| t.ty == gufo_hip::GgmlType::Q8_0)
        });
        assert!(
            q8 && st.mean_kl < limits.prefill,
            "prefill weights {name} differ from the GGUF GEMMs"
        );
    }
}

#[test]
fn target() {
    let Some(path) = env_path("GUFO_GEMMA4_MODEL") else {
        eprintln!("SKIP: GUFO_GEMMA4_MODEL is not set");
        return;
    };
    plain(&path);
    match env_path("GUFO_GEMMA4_MTP_MODEL") {
        Some(draft) => speculation(&path, &draft),
        None => println!("note: GUFO_GEMMA4_MTP_MODEL not set; speculation unchecked"),
    }
}
