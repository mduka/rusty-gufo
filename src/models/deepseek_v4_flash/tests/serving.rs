//! The serving QUALITY checks of DeepSeek V4 Flash, ported from
//! upstream `tests/models/deepseek_v4_flash/serving_test.cpp`, on the
//! Rust runner behind the scheduled backend. Autoregressive serving: raw,
//! sampled and chat completions against direct sessions, prompt reuse,
//! concurrent branches from the retained root, two requests decoding as one
//! batch with interleaved output, and cancellation. DSpark: execution modes
//! and snapshot isolation, prefix reuse with reproducible draft decisions,
//! C1 seeded sampling against autoregressive and direct DSpark, a sampled
//! batch replayed with stochastic proposals, greedy and sampled requests in
//! one cohort, a late prompt prefilled between decode steps, the context
//! limit, and disk reuse keyed by the support configuration. The EOS
//! boundary is a unit test of `session::dspark`. The C++ test shares one
//! model between backends and direct sessions; here they load one after
//! another, and results are compared once both sides are known:
//! cargo test --profile gpu-test -p gufo-deepseek-v4-flash --features gpu,models \
//!   --test serving -- --ignored --nocapture
//!
//! `GUFO_DS4_SERVING=dspark` runs the DSpark part alone.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_core::text::{
    CancellationCheck, ChatMessage as RequestMessage, ChatRequest, ChatRole, ReasoningOptions,
    Token,
};
use gufo_deepseek_v4_flash::chat_template::{ChatMessage, resolve_chat_options};
use gufo_deepseek_v4_flash::dspark_sampler::DsparkSampler;
use gufo_deepseek_v4_flash::engine::Engine;
use gufo_deepseek_v4_flash::runner::{DeepSeekRunner, RunnerOptions};
use gufo_deepseek_v4_flash::session::{DsparkItem, Session};
use gufo_serve::backend::{GenerationResult, TextGenerationBackend};
use gufo_serve::pool::{TextRunnerDiskCacheOptions, TextRunnerPool};
use gufo_serve::scheduled_backend::ScheduledBackend;
use gufo_serve::scheduler::{TextPrefillPolicy, TextSchedulerPolicy};

const RAW_PROMPT: &str = "The capital of France is";
/// No earlier request of the backend uses it: the cache keeps six snapshots
/// per session and with them `RAW_PROMPT`'s from the first requests, so only
/// a prompt of its own shows whether a cancelled request left state behind.
const CANCEL_PROMPT: &str = "The capital of Italy is";
const PATTERN: &str = "Continue the pattern with twenty terms: red, blue, blue, red, blue, blue,";

type Backend = ScheduledBackend<DeepSeekRunner>;

struct Options {
    context: u32,
    sessions: usize,
    dspark: bool,
    prefill: TextPrefillPolicy,
    disk: Option<std::path::PathBuf>,
    max_draft_tokens: u32,
}

impl Options {
    fn new(context: u32, sessions: usize, dspark: bool) -> Self {
        Self {
            context,
            sessions,
            dspark,
            prefill: TextPrefillPolicy::default(),
            disk: None,
            max_draft_tokens: 7,
        }
    }
}

fn backend(options: Options) -> Backend {
    let runner = DeepSeekRunner::load(&RunnerOptions {
        model_path: common::model_path(),
        max_context: options.context,
        persistence: options.disk.is_some(),
        dspark_model_path: options.dspark.then(common::dspark_path),
        max_draft_tokens: options.max_draft_tokens,
    })
    .unwrap();
    let disk = options.disk.map(|directory| TextRunnerDiskCacheOptions {
        directory,
        capacity_bytes: 4 << 30,
        ..TextRunnerDiskCacheOptions::default()
    });
    let pool = TextRunnerPool::new(Arc::new(runner), options.sessions, disk).unwrap();
    ScheduledBackend::new(
        Arc::new(pool),
        options.prefill,
        TextSchedulerPolicy::default(),
    )
    .unwrap()
}

fn greedy() -> SamplingConfig {
    SamplingConfig::default()
}

fn sampled() -> SamplingConfig {
    SamplingConfig {
        temperature: 0.8,
        top_k: 20,
        top_p: 0.9,
        seed: 1234,
        repeat_penalty: 1.1,
        ..SamplingConfig::default()
    }
}

fn complete(
    backend: &Backend,
    prompt: &str,
    max_tokens: usize,
    sampling: &SamplingConfig,
) -> GenerationResult {
    backend
        .complete(
            prompt,
            max_tokens,
            sampling,
            None,
            None,
            "quality",
            &[],
            true,
        )
        .unwrap()
}

fn chat(
    backend: &Backend,
    request: &ChatRequest,
    max_tokens: usize,
    sampling: &SamplingConfig,
) -> GenerationResult {
    backend
        .chat(request, max_tokens, sampling, None, None)
        .unwrap()
}

fn start(
    backend: &Backend,
    request: &ChatRequest,
    max_tokens: usize,
    sampling: &SamplingConfig,
) -> Box<dyn gufo_serve::backend::GenerationRequest> {
    backend
        .start_chat(request, max_tokens, sampling, None, true)
        .unwrap()
        .expect("asynchronous admission")
}

fn request(messages: Vec<RequestMessage>, client_id: &str) -> ChatRequest {
    ChatRequest {
        client_id: client_id.into(),
        ..ChatRequest::new(messages)
    }
}

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
        ..ChatMessage::default()
    }
}

/// The prompt of `messages` with the request defaults (thinking at high
/// effort), as the runner renders a chat request.
fn chat_prompt(engine: &Engine, messages: &[ChatMessage]) -> Vec<i32> {
    engine.encode_chat_messages(
        messages,
        &[],
        &resolve_chat_options(&ReasoningOptions::default()),
    )
}

fn tokenize(engine: &Engine, text: &str) -> Vec<i32> {
    let mut tokens = Vec::new();
    engine.vocab.tokenize(text.as_bytes(), &mut tokens);
    tokens
}

/// `GenerateDirect`: a session alone, autoregressive (context 512) or
/// DSpark with the request sampler lent to each cycle (context 4096); EOS
/// ends the output.
fn direct(
    engine: &mut Engine,
    prompt: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    dspark: bool,
) -> Vec<Token> {
    scheduled(engine, prompt, prompt.len(), max_tokens, sampling, dspark)
}

/// `direct` with the prompt prefilled to `boundary` first and then to its
/// end, as the server prefills to the cache boundary of a chat prompt.
fn scheduled(
    engine: &mut Engine,
    prompt: &[i32],
    boundary: usize,
    max_tokens: usize,
    sampling: &SamplingConfig,
    dspark: bool,
) -> Vec<Token> {
    let mut session = engine
        .create_session(if dspark { 4096 } else { 512 }, dspark)
        .unwrap();
    if boundary < prompt.len() {
        session.sync(engine, &prompt[..boundary]).unwrap();
    }
    session.sync(engine, prompt).unwrap();
    let history: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
    let mut sampler = SamplerState::new(sampling.clone(), &history).unwrap();
    let mut result = Vec::with_capacity(max_tokens);
    if dspark {
        while result.len() < max_tokens {
            let mut bridge = DsparkSampler::new(&sampler);
            let emitted = session
                .dspark_step(
                    engine,
                    max_tokens - result.len(),
                    7,
                    Some(&mut bridge),
                    false,
                )
                .unwrap();
            assert!(!emitted.is_empty(), "direct DSpark progress");
            sampler.set_rng_state(bridge.rng_state());
            for token in emitted {
                if engine.vocab.is_stop(token) {
                    return result;
                }
                sampler.accept(token as u32).unwrap();
                result.push(token as Token);
            }
        }
        return result;
    }
    for index in 0..max_tokens {
        let token = if sampling.can_use_unmodified_argmax() {
            session.argmax()
        } else {
            sampler.sample(session.logits().unwrap()).unwrap() as i32
        };
        assert!(token >= 0, "direct token selection");
        if engine.vocab.is_stop(token) {
            break;
        }
        sampler.accept(token as u32).unwrap();
        result.push(token as Token);
        if index + 1 < max_tokens {
            session.eval(engine, token).unwrap();
        }
    }
    result
}

/// The results of the autoregressive backend the direct sessions check.
struct Served {
    raw: GenerationResult,
    disabled: GenerationResult,
    sampled: [GenerationResult; 2],
    chat: GenerationResult,
    repeated_chat: GenerationResult,
    continuation: GenerationResult,
    fork: GenerationResult,
    concurrent: [GenerationResult; 2],
    recovered: GenerationResult,
}

fn system_and_user() -> Vec<RequestMessage> {
    vec![
        RequestMessage::new(ChatRole::System, "Answer with one short sentence."),
        RequestMessage::new(ChatRole::User, "Name one primary color."),
    ]
}

fn branch(text: &str, question: &str) -> Vec<RequestMessage> {
    let mut messages = system_and_user();
    messages.push(RequestMessage::new(ChatRole::Assistant, text));
    messages.push(RequestMessage::new(ChatRole::User, question));
    messages
}

const SEQUENCES: [&str; 2] = [
    "Continue this sequence with four short items: one, two, three,",
    "Continue this sequence with four short items: red, green, blue,",
];

fn serve_autoregressive() -> Served {
    let backend = backend(Options::new(512, 2, false));
    assert_eq!(
        backend.model_id(),
        "DeepSeek V4 Flash",
        "HTTP model identifier"
    );
    let raw = complete(&backend, RAW_PROMPT, 2, &greedy());
    assert!(raw.ttft_ms > 0.0, "raw TTFT");
    // A backend without DSpark never drafts (`CheckExecutionModes`).
    let disabled = complete(&backend, RAW_PROMPT, 8, &greedy());
    assert_eq!(disabled.draft_tokens, 0, "disabled backend executed DSpark");
    let sampled = [
        complete(&backend, RAW_PROMPT, 8, &sampled()),
        complete(&backend, RAW_PROMPT, 8, &sampled()),
    ];

    let messages = request(system_and_user(), "anonymous");
    let first = chat(&backend, &messages, 2, &greedy());
    assert!(!first.cache_hit, "first chat request is a cache miss");
    let repeated = chat(&backend, &messages, 2, &greedy());
    assert!(
        repeated.cache_hit
            && repeated.cached_prompt_tokens == repeated.prompt_tokens
            && repeated.prefill_tokens == 0,
        "repeated DeepSeek chat restores the complete prompt boundary"
    );
    assert_eq!(
        repeated.cache_snapshot_bytes, 0,
        "exact DeepSeek reuse avoids another full snapshot copy"
    );

    let text = String::from_utf8_lossy(&first.text).into_owned();
    let mut continuation = start(
        &backend,
        &request(
            branch(&text, "Name a different primary color."),
            "deepseek-snapshot-branch-a",
        ),
        2,
        &greedy(),
    );
    let mut fork = start(
        &backend,
        &request(
            branch(&text, "Name one warm primary color."),
            "deepseek-snapshot-branch-b",
        ),
        2,
        &greedy(),
    );
    let continuation = continuation.wait(None).unwrap();
    let fork = fork.wait(None).unwrap();
    for result in [&continuation, &fork] {
        assert!(
            result.cache_hit,
            "concurrent DeepSeek branches restore the retained root"
        );
        assert!(
            result.cached_prompt_tokens > 0,
            "DeepSeek branches reuse their longest available prefix"
        );
        assert!(
            result.cached_prompt_tokens < result.prompt_tokens,
            "DeepSeek branches prefill only their suffixes"
        );
        assert!(
            result.cache_snapshot_bytes > 0 && result.cache_snapshot_ms > 0.0,
            "DeepSeek branches retain their new prompt snapshot"
        );
        assert_eq!(
            result.cache_restore_bytes == 0,
            result.cache_restore_ms == 0.0,
            "live reuse reports no restore copy; snapshot reuse reports it"
        );
        assert_eq!(
            result.cache_shared_bytes, 0,
            "full-copy DeepSeek snapshots do not claim shared bytes"
        );
    }

    let mut pending: Vec<_> = SEQUENCES
        .iter()
        .zip(["deepseek-a", "deepseek-b"])
        .map(|(text, client)| {
            start(
                &backend,
                &request(vec![RequestMessage::new(ChatRole::User, *text)], client),
                4,
                &greedy(),
            )
        })
        .collect();
    let events = Mutex::new(Vec::new());
    let concurrent: Vec<GenerationResult> = std::thread::scope(|scope| {
        let waiters: Vec<_> = pending
            .iter_mut()
            .zip(['a', 'b'])
            .map(|(request, tag)| {
                let events = &events;
                scope.spawn(move || {
                    let mut on_token = |_: &[u8]| {
                        events.lock().unwrap().push(tag);
                        true
                    };
                    request.wait(Some(&mut on_token)).unwrap()
                })
            })
            .collect();
        waiters.into_iter().map(|w| w.join().unwrap()).collect()
    });
    for result in &concurrent {
        assert_eq!(
            result.requested_logical_concurrency, 2,
            "DeepSeek scheduler reports two logical sessions"
        );
        assert!(
            result.physical_execution_width == 2 && result.execution_plan == "batched-w2",
            "DeepSeek reports native two-session execution"
        );
    }
    let events = events.into_inner().unwrap();
    let first_of = |tag| events.iter().position(|&e| e == tag);
    let last_of = |tag| events.iter().rposition(|&e| e == tag);
    let (Some(first_a), Some(first_b)) = (first_of('a'), first_of('b')) else {
        panic!("both DeepSeek sessions stream output");
    };
    assert!(
        first_b < last_of('a').unwrap() && first_a < last_of('b').unwrap(),
        "both DeepSeek sessions make progress before the other completes"
    );

    let checks = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&checks);
    let cancel: CancellationCheck =
        Arc::new(move || counter.fetch_add(1, Ordering::SeqCst) + 1 > 2);
    let cancelled = backend
        .complete(
            CANCEL_PROMPT,
            4,
            &greedy(),
            Some(&cancel),
            None,
            "quality",
            &[],
            true,
        )
        .unwrap();
    assert!(cancelled.cancelled, "cancellation status");
    assert!(cancelled.completion_tokens <= 1, "cancellation token bound");
    let recovered = complete(&backend, CANCEL_PROMPT, 2, &greedy());
    assert!(
        !recovered.cache_hit,
        "cancelled DeepSeek state must not remain cached"
    );

    let [a, b] = <[GenerationResult; 2]>::try_from(concurrent).ok().unwrap();
    Served {
        raw,
        disabled,
        sampled,
        chat: first,
        repeated_chat: repeated,
        continuation,
        fork,
        concurrent: [a, b],
        recovered,
    }
}

/// The served results against direct autoregressive sessions.
fn check_autoregressive(served: &Served) {
    let mut engine = Engine::open(&common::model_path(), None).unwrap();
    let raw_prompt = tokenize(&engine, RAW_PROMPT);
    let direct_raw = direct(&mut engine, &raw_prompt, 2, &greedy(), false);
    assert_eq!(
        served.raw.tokens, direct_raw,
        "raw direct/HTTP token parity"
    );
    let cancel_prompt = tokenize(&engine, CANCEL_PROMPT);
    assert_eq!(
        served.recovered.tokens,
        direct(&mut engine, &cancel_prompt, 2, &greedy(), false),
        "session reuse after cancellation"
    );
    assert_eq!(
        served.disabled.tokens,
        direct(&mut engine, &raw_prompt, 8, &greedy(), false),
        "disabled backend matches autoregressive execution"
    );
    let direct_sampled = direct(&mut engine, &raw_prompt, 8, &sampled(), false);
    assert!(
        served.sampled.iter().all(|r| r.tokens == direct_sampled),
        "seeded sampling and penalties match direct execution and repeat"
    );

    let system_user = [
        message("system", "Answer with one short sentence."),
        message("user", "Name one primary color."),
    ];
    let chat_tokens = chat_prompt(&engine, &system_user);
    let direct_chat = direct(&mut engine, &chat_tokens, 2, &greedy(), false);
    assert_eq!(
        served.chat.tokens, direct_chat,
        "chat direct/HTTP token parity"
    );
    assert_eq!(
        served.repeated_chat.tokens, direct_chat,
        "repeated DeepSeek chat differs from cold target execution"
    );
    let text = String::from_utf8_lossy(&served.chat.text).into_owned();
    for (result, question, what) in [
        (
            &served.continuation,
            "Name a different primary color.",
            "continuation",
        ),
        (&served.fork, "Name one warm primary color.", "fork"),
    ] {
        let mut messages = system_user.to_vec();
        messages.push(message("assistant", &text));
        messages.push(message("user", question));
        let prompt = chat_prompt(&engine, &messages);
        let expected = direct(&mut engine, &prompt, 2, &greedy(), false);
        assert_eq!(
            result.tokens, expected,
            "cached DeepSeek {what} differs from cold full prefill"
        );
    }
    for (result, text) in served.concurrent.iter().zip(SEQUENCES) {
        let prompt = chat_prompt(&engine, &[message("user", text)]);
        let expected = direct(&mut engine, &prompt, 4, &greedy(), false);
        assert_eq!(
            result.tokens, expected,
            "concurrent DeepSeek request differs from isolated execution"
        );
    }
    println!("DeepSeek V4 Flash autoregressive serving matches direct sessions");
}

/// The direct DSpark checks, and the seeded C1 sequences the DSpark backend
/// must reproduce.
/// A seeded C1 sequence: prefilled at once, and with the server's schedule.
struct Reference {
    sampling: SamplingConfig,
    direct: Vec<Token>,
    scheduled: Vec<Token>,
}

fn direct_dspark() -> Vec<Reference> {
    let mut engine = Engine::open(&common::model_path(), Some(&common::dspark_path())).unwrap();

    // `CheckExecutionModes`.
    let mut ar = engine.create_session(512, false).unwrap();
    let mut spec = engine.create_session(512, true).unwrap();
    assert!(
        !ar.dspark_enabled() && spec.dspark_enabled(),
        "session mode reports model capability instead of enabled execution"
    );
    let prompt = tokenize(&engine, RAW_PROMPT);
    ar.sync(&mut engine, &prompt).unwrap();
    spec.sync(&mut engine, &prompt).unwrap();
    assert!(
        ar.logits() == spec.logits(),
        "DSpark sidecar changes AR logits"
    );
    assert!(
        ar.dspark_stats().context_tokens == 0
            && spec.dspark_stats().context_tokens as usize == prompt.len(),
        "AR session maintains DSpark state"
    );
    let saved_ar = ar.save_snapshot(&engine).unwrap();
    let saved_spec = spec.save_snapshot(&engine).unwrap();
    assert!(
        saved_ar.len() < saved_spec.len(),
        "AR snapshot includes predictor state"
    );
    assert!(
        ar.restore_snapshot(&engine, &saved_spec).is_err()
            && spec.restore_snapshot(&engine, &saved_ar).is_err(),
        "DeepSeek snapshots crossed execution modes"
    );
    drop((ar, spec));
    println!("DSpark execution modes and snapshot isolation passed");

    // C1 keeps the point-mass route and its AR seed-trace contract.
    let direct_prompt = chat_prompt(&engine, &[message("user", PATTERN)]);
    // The cache boundary of the runner: the prefix a further user turn keeps.
    let continued = chat_prompt(&engine, &[message("user", PATTERN), message("user", "")]);
    let boundary = direct_prompt
        .iter()
        .zip(&continued)
        .take_while(|(a, b)| a == b)
        .count();
    let mut references = Vec::new();
    for sampling in [
        SamplingConfig {
            temperature: 0.6,
            seed: 7,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            temperature: 1.0,
            seed: 7,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            temperature: 0.6,
            top_k: 40,
            top_p: 0.95,
            seed: 11,
            repeat_penalty: 1.2,
            frequency_penalty: 0.3,
            ..SamplingConfig::default()
        },
    ] {
        let speculative = direct(&mut engine, &direct_prompt, 32, &sampling, true);
        let autoregressive = direct(&mut engine, &direct_prompt, 32, &sampling, false);
        assert_eq!(speculative, autoregressive, "C1 DSpark and AR agree");
        let served = scheduled(&mut engine, &direct_prompt, boundary, 32, &sampling, true);
        references.push(Reference {
            sampling,
            direct: speculative,
            scheduled: served,
        });
    }
    sampled_batch_replay(&mut engine, &direct_prompt);
    references
}

/// `CheckSampledBatchReplay`: a fixed C2 schedule with independent seeds
/// and filters replays outputs and draft decisions, with stochastic
/// proposals and verifications.
fn sampled_batch_replay(engine: &mut Engine, prompt: &[i32]) {
    let history: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
    let configs = [
        SamplingConfig {
            temperature: 1.0,
            top_p: 0.95,
            seed: 7,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 11,
            repeat_penalty: 1.1,
            ..SamplingConfig::default()
        },
    ];
    let (mut draws, mut verifications) = (0, 0);
    let mut reference: Option<Vec<(Vec<i32>, u64, u64)>> = None;
    for _ in 0..2 {
        let mut sessions: Vec<Session> = (0..2)
            .map(|_| {
                let mut s = engine.create_session(4096, true).unwrap();
                s.sync(engine, prompt).unwrap();
                s
            })
            .collect();
        let mut samplers: Vec<SamplerState> = configs
            .iter()
            .map(|c| SamplerState::new(c.clone(), &history).unwrap())
            .collect();
        let mut output = vec![Vec::new(); 2];
        for _ in 0..16 {
            let mut bridges: Vec<DsparkSampler> = samplers.iter().map(DsparkSampler::new).collect();
            let mut items: Vec<DsparkItem<'_>> = sessions
                .iter_mut()
                .zip(bridges.iter_mut())
                .map(|(session, bridge)| DsparkItem {
                    session,
                    max_tokens: 4,
                    max_draft_tokens: 3,
                    sampler: Some(bridge),
                    stop_at_eos: false,
                    emitted: Vec::new(),
                })
                .collect();
            Session::dspark_step_batch(engine, &mut items).unwrap();
            let emitted: Vec<Vec<i32>> = items.into_iter().map(|item| item.emitted).collect();
            for ((sampler, bridge), (emitted, output)) in samplers
                .iter_mut()
                .zip(&bridges)
                .zip(emitted.iter().zip(output.iter_mut()))
            {
                assert!(!emitted.is_empty(), "sampled batch makes progress");
                sampler.set_rng_state(bridge.rng_state());
                for &token in emitted {
                    sampler.accept(token as u32).unwrap();
                }
                output.extend_from_slice(emitted);
                let (d, v) = bridge.proposal_counts();
                draws += d;
                verifications += v;
            }
        }
        let run: Vec<(Vec<i32>, u64, u64)> = sessions
            .iter()
            .zip(output)
            .map(|(s, output)| {
                let stats = s.dspark_stats();
                (output, stats.support_drafted, stats.support_accepted)
            })
            .collect();
        match &reference {
            None => reference = Some(run),
            Some(reference) => assert!(
                run == *reference,
                "sampled batch replays outputs and decisions independently"
            ),
        }
    }
    let reference = reference.unwrap();
    assert!(
        reference.iter().map(|r| r.1).sum::<u64>() > 0,
        "filtered sampled batch exercises DSpark"
    );
    assert!(
        draws > 0 && verifications > 0,
        "concurrent runtime invokes stochastic proposal and verification"
    );
}

fn serve_dspark(references: &[Reference]) {
    let prompt = request(
        vec![RequestMessage::new(ChatRole::User, PATTERN)],
        "anonymous",
    );
    {
        let backend = backend(Options::new(262_144, 1, true));
        let cold = chat(&backend, &prompt, 32, &greedy());
        let warm = chat(&backend, &prompt, 32, &greedy());
        assert!(
            cold.draft_tokens > 0 && warm.cache_hit && warm.prefill_tokens == 0,
            "DSpark reuses the complete prompt and drafts immediately"
        );
        assert!(
            cold.tokens == warm.tokens
                && cold.draft_tokens == warm.draft_tokens
                && cold.draft_accepted_tokens == warm.draft_accepted_tokens,
            "DSpark prefix reuse resets policy and reproduces output and draft decisions"
        );
    }
    // Greedy and seeded sampled requests share one DSpark cohort.
    for concurrency in [2, 4] {
        let backend = backend(Options::new(4096, concurrency, true));
        let mut pending: Vec<_> = (0..concurrency)
            .map(|i| {
                let sampling = if i % 2 == 0 { greedy() } else { sampled() };
                let mut request = prompt.clone();
                request.client_id = format!("mixed-{i}");
                start(&backend, &request, 32, &sampling)
            })
            .collect();
        for (i, request) in pending.iter_mut().enumerate() {
            let result = request.wait(None).unwrap();
            assert!(
                !result.tokens.is_empty() && result.tokens.len() <= 32,
                "mixed sampling request completes within its budget"
            );
            if i % 2 == 0 {
                assert!(result.draft_tokens > 0, "greedy requests retain DSpark");
            }
            assert!(
                result.physical_execution_width == concurrency
                    && result.execution_plan == format!("batched-w{concurrency}"),
                "mixed sampling reports one DSpark cohort"
            );
        }
    }
    {
        let backend = backend(Options {
            prefill: TextPrefillPolicy {
                decode_active_tokens: 32,
            },
            ..Options::new(4096, 2, true)
        });
        let mut active = start(&backend, &prompt, 64, &greedy());
        let mut long_request = prompt.clone();
        long_request.client_id = "late-long-prompt".into();
        for _ in 0..256 {
            long_request.messages[0]
                .content
                .push_str(" additional context");
        }
        let mut late = None;
        let mut on_token = |_: &[u8]| {
            if late.is_none() {
                late = Some(start(&backend, &long_request, 8, &greedy()));
            }
            true
        };
        let first = active.wait(Some(&mut on_token)).unwrap();
        let mut late = late.expect("late prompt arrives during decoding");
        assert!(
            !first.tokens.is_empty(),
            "late prompt arrives during decoding"
        );
        let second = late.wait(None).unwrap();
        assert!(
            second.active_decode_prefill_chunks > 0
                && second.max_consecutive_active_prefill_chunks == 1,
            "real DSpark serving yields to decoding between long-prompt chunks"
        );
    }
    {
        // Without thinking, as the C++ check was written before thinking
        // became the default: the prompt fits the 64 tokens.
        let short = ChatRequest {
            reasoning: ReasoningOptions {
                enabled: Some(false),
                ..ReasoningOptions::default()
            },
            ..prompt.clone()
        };
        let backend = backend(Options::new(64, 2, true));
        let mut pending = [
            start(&backend, &short, 128, &greedy()),
            start(&backend, &short, 128, &sampled()),
        ];
        for request in &mut pending {
            let result = request.wait(None).unwrap();
            assert_eq!(
                result.prompt_tokens + result.completion_tokens,
                64,
                "greedy and sampled serving stop exactly at context capacity"
            );
        }
    }
    // The C++ test changes the support fingerprint it passes in; the Rust
    // runner fingerprints the files, so another draft limit changes the
    // support configuration of the identity instead.
    let directory = tempfile::Builder::new()
        .prefix("ds4-support-cache-")
        .tempdir()
        .unwrap();
    let mut expected = None;
    for run in 0..3 {
        let backend = backend(Options {
            disk: Some(directory.path().to_path_buf()),
            max_draft_tokens: if run == 2 { 6 } else { 7 },
            ..Options::new(4096, 1, true)
        });
        let result = chat(&backend, &prompt, 8, &sampled());
        assert_eq!(
            result.cache_disk_hit,
            run == 1,
            "DSpark disk reuse requires the matching support configuration"
        );
        match &expected {
            None => expected = Some(result.tokens),
            Some(expected) => assert_eq!(
                &result.tokens, expected,
                "DSpark disk restore retains output quality"
            ),
        }
    }
    println!("DSpark serving: prefix/disk reuse, mixed sampling, and late prefill passed");
    // Last: the C++ server misses it identically. Both servers prefill a
    // prompt to its cache boundary and decode its last tokens, the direct
    // session prefills it at once, and a sampled choice flips on the
    // difference (`docs/models/deepseek-v4-flash/QUALITY.md`).
    {
        let backend = backend(Options::new(4096, 1, true));
        let greedy_result = chat(&backend, &prompt, 32, &greedy());
        let mut sampled_drafts = 0;
        for reference in references {
            let sampling = &reference.sampling;
            let result = chat(&backend, &prompt, 32, sampling);
            let repeat = chat(&backend, &prompt, 32, sampling);
            sampled_drafts += result.draft_tokens;
            assert!(
                result.draft_accepted_tokens <= result.draft_tokens,
                "sampled DSpark accounting includes legitimate AR fallback"
            );
            assert!(
                result.tokens == reference.scheduled && repeat.tokens == result.tokens,
                "C1 DSpark serving and cached seeded replay equal a direct session with the server's prefill schedule: {sampling:?}"
            );
            assert!(
                result.tokens == reference.direct,
                "C1 DSpark, AR, frontend and cached seeded replay agree: {sampling:?}\n\
                 served {:?}\ndirect {:?}",
                result.tokens,
                reference.direct
            );
        }
        assert!(
            sampled_drafts > 0,
            "sampled C1 serving exercises DSpark verification"
        );
        let again = chat(&backend, &prompt, 32, &greedy());
        assert!(
            again.tokens == greedy_result.tokens
                && again.draft_tokens == greedy_result.draft_tokens
                && again.draft_accepted_tokens == greedy_result.draft_accepted_tokens,
            "greedy DSpark output is unchanged after sampled cycles"
        );
    }
}

#[test]
#[ignore = "QUALITY check on the DeepSeek V4 Flash artifact; loads the model thirteen times"]
fn serving_matches_direct_sessions() {
    if std::env::var("GUFO_DS4_SERVING").as_deref() != Ok("dspark") {
        let served = serve_autoregressive();
        check_autoregressive(&served);
    }
    let references = direct_dspark();
    serve_dspark(&references);
    println!("DeepSeek V4 Flash HTTP parity test passed");
}
