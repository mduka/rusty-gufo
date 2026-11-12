//! The process-wide `/metrics` counters.
//! The binary is its own process and its tests run one at a time, so the
//! counters change only by the work of the running test.

mod common;
mod http_fixture;
mod scheduler_fixture;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use gufo_core::json;
use gufo_serve::backend::GenerationResult;
use gufo_serve::http::HttpServerOptions;
use gufo_serve::metrics::{self, MetricsSnapshot, record_server_metrics};
use gufo_serve::scheduler::{SchedulerRequest, TextRequestPhase};

use common::Semaphore;
use http_fixture::{RunningServer, expect_status};
use scheduler_fixture::*;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn expect_gauges_balanced(before: &MetricsSnapshot) {
    let after = metrics::snapshot();
    assert!(
        after.requests_processing == before.requests_processing
            && after.requests_deferred == before.requests_deferred,
        "success, failure, cancellation and shutdown paths balance the gauges: {after:?}"
    );
}

#[test]
fn fallback_backend_metrics() {
    let _serial = serial();
    let server = RunningServer::new(HttpServerOptions::default());
    for (path, body) in [
        (
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        ),
        ("/v1/completions", r#"{"prompt":"hi"}"#),
        ("/v1/responses", r#"{"input":"hi"}"#),
    ] {
        for stream in [false, true] {
            let before = metrics::snapshot();
            let mut request = json::parse_str(body).expect("body");
            request.set("model", "test");
            request.set("stream", stream);
            expect_status(&server.post(path, &request.dump().expect("dump")), 200);
            let response = server.send("GET /metrics HTTP/1.1\r\n\r\n", false);
            expect_status(&response, 200);
            // The backend reports 10 prompt tokens, of which 8 were cached,
            // and 4 of 8 draft tokens accepted.
            for (name, value) in [
                ("prompt_tokens_total", before.prompt_tokens + 2),
                ("tokens_predicted_total", before.generated_tokens + 1),
                (
                    "prompt_tokens_cached_total",
                    before.cached_prompt_tokens + 8,
                ),
                ("spec_decode_num_drafts_total", before.draft_rounds + 2),
                (
                    "spec_decode_num_draft_tokens_total",
                    before.draft_tokens + 8,
                ),
                (
                    "spec_decode_num_accepted_tokens_total",
                    before.draft_accepted_tokens + 4,
                ),
            ] {
                assert!(
                    response.contains(&format!("llamacpp:{name} {value}\n")),
                    "{path} stream={stream}: {name} {value}\n{response}"
                );
            }
            let after = metrics::snapshot();
            assert!(
                after.prompt_seconds - before.prompt_seconds > 0.0039
                    && after.generation_seconds - before.generation_seconds > 0.0019
                    && after.max_sequence_tokens >= 11,
                "request seconds and the longest sequence are recorded"
            );
        }
    }
    for (prefill, cached, cancelled, expected_prompt) in [
        (6, 4, false, 6),
        (0, 4, false, 6),
        (0, 10, false, 0),
        (0, 20, false, 0),
        (0, 0, true, 0),
        (3, 0, true, 3),
    ] {
        let before = metrics::snapshot();
        record_server_metrics(&GenerationResult {
            prompt_tokens: 10,
            cached_prompt_tokens: cached,
            prefill_tokens: prefill,
            completion_tokens: 2,
            cancelled,
            ..GenerationResult::default()
        });
        let after = metrics::snapshot();
        assert_eq!(
            after.prompt_tokens - before.prompt_tokens,
            expected_prompt,
            "prefill={prefill} cached={cached} cancelled={cancelled}"
        );
        assert_eq!(after.generated_tokens - before.generated_tokens, 2);
    }
}

#[test]
fn server_metrics_are_live() {
    let _serial = serial();
    let before = metrics::snapshot();
    {
        let control =
            FakeControl::with_gate(Config::default(), |gate| gate.block_advance_label = Some(1));
        let scheduler = scheduler(&control, 1);
        let mut active = submit(&scheduler, &[1, 10], 3);
        control.wait_for_advance(1);
        let live = metrics::snapshot();
        assert_eq!(
            live.prompt_tokens - before.prompt_tokens,
            2,
            "prompt tokens are counted when prefill executes"
        );
        assert_eq!(
            live.generated_tokens - before.generated_tokens,
            1,
            "generated tokens are counted before the request completes"
        );
        assert_eq!(
            live.requests_processing - before.requests_processing,
            1,
            "the running request is reported"
        );

        let mut queued = submit(&scheduler, &[2, 20], 1);
        assert_eq!(
            metrics::snapshot().requests_deferred - before.requests_deferred,
            1,
            "the waiting request is reported"
        );
        control.release_advance();
        let active_result = wait(&mut active);
        let queued_result = wait(&mut queued);
        assert!(active_result.tokens.len() == 3 && queued_result.tokens.len() == 1);
        assert!(
            active_result.token_metrics_recorded && queued_result.token_metrics_recorded,
            "scheduler results identify live token accounting"
        );
        record_server_metrics(&active_result);
        record_server_metrics(&queued_result);
        let after = metrics::snapshot();
        assert_eq!(
            after.prompt_tokens - before.prompt_tokens,
            4,
            "HTTP completion does not count the prompt again"
        );
        assert_eq!(
            after.generated_tokens - before.generated_tokens,
            4,
            "every generated token is counted once"
        );

        control.gate().throw_advance_label = Some(3);
        assert!(
            submit(&scheduler, &[3, 30], 2).wait(None).is_err(),
            "a runner failure fails the request"
        );
        expect_gauges_balanced(&before);
    }
    expect_gauges_balanced(&before);
}

#[test]
fn metrics_during_cache_admission() {
    let _serial = serial();
    for cancel in [false, true] {
        let before = metrics::snapshot();
        let entered = Arc::new(Semaphore::default());
        let release = Arc::new(Semaphore::default());
        let arm = Arc::new(AtomicBool::new(false));
        let callback = {
            let (entered, release, arm) = (entered.clone(), release.clone(), arm.clone());
            Arc::new(move || {
                if arm.swap(false, Ordering::AcqRel) {
                    entered.release();
                    release.acquire();
                }
            }) as Arc<dyn Fn() + Send + Sync>
        };
        let control = FakeControl::new(Config {
            snapshot_callback: Some(callback),
            ..Config::default()
        });
        let scheduler = scheduler(&control, 2);
        let first = wait(&mut submit(&scheduler, &[1, 10], 1));
        assert_eq!(
            first.tokens,
            expected_tokens(1, 1),
            "first request completes"
        );
        assert_eq!(
            metrics::snapshot().requests_processing,
            before.requests_processing,
            "completed requests retire before wait returns"
        );

        arm.store(true, Ordering::Release);
        let mut next = submit(&scheduler, &[1, 10, 100, 11], 1);
        assert!(
            entered.try_acquire_for(TEST_TIMEOUT),
            "continuation admission captures its live checkpoint"
        );
        let preparing = metrics::snapshot();
        assert!(
            preparing.requests_processing == before.requests_processing + 1
                && preparing.requests_deferred == before.requests_deferred,
            "cache preparation remains visible as processing"
        );
        let mut queued = submit(&scheduler, &[2, 20], 1);
        assert_eq!(
            metrics::snapshot().requests_deferred,
            before.requests_deferred + 1,
            "requests waiting behind cache preparation remain deferred"
        );
        if cancel {
            next.cancel();
            queued.cancel();
        }
        release.release();
        let result = wait(&mut next);
        let queued_result = wait(&mut queued);
        assert!(
            result.cancelled == cancel && queued_result.cancelled == cancel,
            "admission and queued cancellation preserve request outcomes"
        );
        if !cancel {
            assert!(
                result.cache_hit && result.cached_prompt_tokens == 3,
                "continuation still reuses the live frontier"
            );
        }
        expect_gauges_balanced(&before);
    }
}

#[test]
fn request_totals_do_not_need_a_consumer() {
    let _serial = serial();
    let wait_terminal = |request: &SchedulerRequest| {
        let deadline = Instant::now() + Duration::from_secs(10);
        while request.phase() != TextRequestPhase::Terminal {
            assert!(
                Instant::now() < deadline,
                "request reaches terminal without a consumer"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    {
        let control = FakeControl::new(Config::default());
        let scheduler = scheduler(&control, 1);
        record_server_metrics(&wait(&mut submit(&scheduler, &[1, 10], 1)));
        let cached_before = metrics::snapshot().cached_prompt_tokens;
        // Never waited on, like a stream whose first write fails.
        let abandoned = submit(&scheduler, &[1, 10, 100, 11], 1);
        wait_terminal(&abandoned);
        assert_eq!(
            metrics::snapshot().cached_prompt_tokens - cached_before,
            3,
            "a finished request counts before anyone reads it"
        );
        drop(abandoned);
        let consumed = wait(&mut submit(&scheduler, &[1, 10, 100, 11], 1));
        record_server_metrics(&consumed);
        assert_eq!(
            metrics::snapshot().cached_prompt_tokens - cached_before,
            3 + consumed.cached_prompt_tokens as u64,
            "reading the result does not count it again"
        );
    }
    {
        let control = FakeControl::new(Config {
            incremental_prefill: false,
            multi_token_decode: true,
            prefix_reuse: false,
            ..Config::default()
        });
        let scheduler = scheduler(&control, 1);
        let before = metrics::snapshot();
        let request = submit(&scheduler, &[7], 5);
        wait_terminal(&request);
        let after = metrics::snapshot();
        assert!(
            after.draft_rounds - before.draft_rounds == 2
                && after.draft_tokens - before.draft_tokens == 7
                && after.draft_accepted_tokens - before.draft_accepted_tokens == 5,
            "draft totals count at completion: {after:?} {before:?}"
        );
    }
}

/// Parked followers keep their session, so newer arrivals stay queued and
/// admission never exceeds capacity.
#[test]
fn parked_followers_reserve_visible_sessions() {
    let _serial = serial();
    let before = metrics::snapshot();
    let entered = Arc::new(Semaphore::default());
    let release = Arc::new(Semaphore::default());
    let captures = Arc::new(AtomicUsize::new(0));
    let control = FakeControl::with_gate(
        Config {
            max_context: 4096,
            prefill_capacity: 256,
            snapshot_callback: Some({
                let (entered, release, captures) =
                    (entered.clone(), release.clone(), captures.clone());
                Arc::new(move || {
                    if captures.fetch_add(1, Ordering::AcqRel) == 0 {
                        entered.release();
                        assert!(
                            release.try_acquire_for(TEST_TIMEOUT),
                            "shared snapshot gate timed out"
                        );
                    }
                })
            }),
            ..Config::default()
        },
        |gate| gate.block_prefill_label = Some(1),
    );
    let scheduler = scheduler(&control, 2);
    let prompt = shared_prompt(1200, 7000, 100);
    let mut leader = submit(&scheduler, &prompt, 2);
    control.wait_for_prefill(1);
    let mut follower = submit(&scheduler, &prompt, 2);
    control.release_prefill();
    assert!(
        entered.try_acquire_for(TEST_TIMEOUT),
        "shared checkpoint capture starts"
    );
    let mut third = submit(&scheduler, &[2, 20], 2);
    let mut exceeded_capacity = false;
    let deadline = Instant::now() + Duration::from_millis(100);
    while Instant::now() < deadline {
        exceeded_capacity |=
            metrics::snapshot().requests_processing - before.requests_processing > 2;
        std::thread::yield_now();
    }
    let states = scheduler.session_states();
    let reserved = states.len() == 2
        && states[0].processing
        && states[1].processing
        && states[0].request_id == leader.id()
        && states[1].request_id == follower.id();
    let queued = third.phase() == TextRequestPhase::Queued;
    release.release();
    wait(&mut leader);
    let reused = wait(&mut follower);
    let independent = wait(&mut third);
    assert!(
        !exceeded_capacity && reserved && queued,
        "parked followers reserve sessions and keep newer arrivals queued: {states:?}"
    );
    assert!(
        reused.cached_prompt_tokens > 0 && independent.tokens == expected_tokens(2, 2),
        "reserved followers resume and then release admission capacity"
    );
    drop(scheduler);
    expect_gauges_balanced(&before);
}
