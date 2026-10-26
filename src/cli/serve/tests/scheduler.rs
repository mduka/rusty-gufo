//! Port of `tests/cli/text_generation_scheduler_test.cpp`.

mod common;
mod scheduler_fixture;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{ErrorKind, GenerationErrorCode, Token};
use gufo_serve::backend::{FinishReason, GenerationResult, PromptProgress, StreamCallbacks};
use gufo_serve::logging::{self, LogLevel};
use gufo_serve::pool::TextRunnerPool;
use gufo_serve::scheduler::{
    RequestMetadata, SchedulerRequest, TextGenerationScheduler, TextPrefillPolicy,
    TextRequestPhase, TextSchedulerPolicy,
};

use common::Semaphore;
use scheduler_fixture::*;

#[test]
fn idle_prefill_uses_bulk_work_unit() {
    let control = FakeControl::new(Config::default());
    let scheduler = chunked(&control, 1, 2);
    let result = wait(&mut submit(&scheduler, &[7, 70, 71, 72, 73], 2));
    let events = control.events();
    assert!(
        events[0].kind == EventKind::Prefill && events[0].label == 7 && events[0].count == 5,
        "decode-idle prefill consumes the complete prompt"
    );
    assert!(
        result.prefill_chunks == 1 && result.prefill_tokens == 5,
        "decode-idle prefill metrics report one bulk work unit"
    );
    assert_eq!(
        result.active_decode_prefill_chunks, 0,
        "decode-idle prefill is not counted as active-decode work"
    );
    assert!(
        result.requested_logical_concurrency == 1
            && result.physical_execution_width == 1
            && result.execution_plan == "serial-c1",
        "C=1 telemetry reports immediate serial dispatch"
    );
}

#[test]
fn runner_can_skip_unused_final_advance() {
    let control = FakeControl::new(Config {
        final_token_advance_required: false,
        ..Config::default()
    });
    let scheduler = scheduler(&control, 1);
    let result = wait(&mut submit(&scheduler, &[7], 2));
    assert_eq!(
        result.tokens,
        expected_tokens(7, 2),
        "skipped final advance preserves emitted tokens"
    );
    let advances = control
        .events()
        .iter()
        .filter(|e| e.kind == EventKind::Advance && e.label == 7)
        .count();
    assert_eq!(
        advances, 1,
        "runner skips the unused frontier computation after the final token"
    );
}

#[test]
fn runner_can_reuse_exact_incremental_text() {
    let control = FakeControl::new(Config {
        incremental_text_is_exact: true,
        ..Config::default()
    });
    let scheduler = scheduler(&control, 1);
    let result = wait(&mut submit(&scheduler, &[7], 2));
    assert_eq!(
        result.text, b"700701",
        "exact incremental pieces form the final response text"
    );
    assert_eq!(
        control.decode_calls.load(Ordering::Relaxed),
        0,
        "exact incremental text avoids duplicate final decoding"
    );
}

#[test]
fn multi_token_decode_publishes_draft_metrics_and_disables_prefix_reuse() {
    let control = FakeControl::new(Config {
        incremental_prefill: false,
        multi_token_decode: true,
        prefix_reuse: false,
        ..Config::default()
    });
    let scheduler = scheduler(&control, 1);
    let first = wait(&mut submit(&scheduler, &[7], 5));
    assert_eq!(
        first.tokens,
        expected_tokens(7, 5),
        "multi-token decode preserves the generated trajectory"
    );
    assert!(
        first.draft_rounds == 2 && first.draft_tokens == 7 && first.draft_accepted_tokens == 5,
        "multi-token decode reports accumulated draft statistics"
    );
    assert!(
        !first.cache_hit,
        "multi-token state starts without continuation reuse"
    );
    control.wait_for_invalidations(1);
    let second = wait(&mut submit(&scheduler, &[7, 70], 2));
    assert!(
        !second.cache_hit,
        "runner-disabled prefix reuse cannot retain speculative state"
    );
}

#[test]
fn multi_token_runner_can_switch_to_batched_execution() {
    let control = FakeControl::with_gate(
        Config {
            multi_token_decode: true,
            prefix_reuse: false,
            supports_batched_advance: true,
            ..Config::default()
        },
        |g| g.block_prefill_label = Some(1),
    );
    let scheduler = scheduler(&control, 2);
    let mut request_a = submit(&scheduler, &[1, 10], 4);
    control.wait_for_prefill(1);
    let mut request_b = submit(&scheduler, &[2, 20], 4);
    control.release_prefill();
    let result_a = wait(&mut request_a);
    let result_b = wait(&mut request_b);
    assert!(
        result_a.tokens == expected_tokens(1, 4) && result_b.tokens == expected_tokens(2, 4),
        "batched speculative-capable requests preserve both trajectories"
    );
    assert!(
        result_a.execution_plan == "batched-w2" && result_b.execution_plan == "batched-w2",
        "speculative-capable requests can use the physical W2 plan"
    );
    assert!(
        result_a.draft_rounds == 0
            && result_b.draft_rounds == 0
            && result_a.draft_tokens == 0
            && result_b.draft_tokens == 0,
        "target batching bypasses per-request draft steps"
    );
    assert!(
        control.batch_preparations.load(Ordering::Relaxed) >= 2,
        "both resident states are prepared before target batching"
    );
}

#[test]
fn batch_failure_isolation() {
    for speculative in [false, true] {
        let control = FakeControl::with_gate(
            Config {
                multi_token_decode: speculative,
                batched_multi_token_decode: speculative,
                supports_batched_advance: true,
                ..Config::default()
            },
            |g| {
                g.block_prefill_label = Some(1);
                g.throw_advance_label = Some(1);
            },
        );
        let scheduler = scheduler(&control, 2);
        let mut failed = submit(&scheduler, &[1], 12);
        control.wait_for_prefill(1);
        let mut healthy = submit(&scheduler, &[2], 12);
        control.release_prefill();
        assert!(
            failed.wait(None).is_err(),
            "failed batch member reports its error"
        );
        assert_eq!(
            wait(&mut healthy).tokens,
            expected_tokens(2, 12),
            "a failing batch member must not fail or corrupt a healthy peer"
        );
    }
}

#[test]
fn model_owned_batch_metrics() {
    for actual_width in [1, 2] {
        let control = FakeControl::with_gate(
            Config {
                multi_token_decode: true,
                batched_multi_token_decode: true,
                supports_batched_advance: true,
                actual_batch_width: actual_width,
                ..Config::default()
            },
            |g| g.block_prefill_label = Some(1),
        );
        let scheduler = scheduler(&control, 2);
        let mut first = submit(&scheduler, &[1], 12);
        control.wait_for_prefill(1);
        let mut second = submit(&scheduler, &[2], 12);
        control.release_prefill();
        for result in [wait(&mut first), wait(&mut second)] {
            let expected = if actual_width == 1 {
                "serial-fallback"
            } else {
                "batched-w2"
            };
            assert!(
                result.physical_execution_width == actual_width
                    && result.execution_plan == expected,
                "scheduler reports the runner's actual subgroup or serial execution"
            );
            assert_eq!(
                result.draft_rounds, 4,
                "each batched request reports its own verification rounds"
            );
        }
    }
}

#[test]
fn multi_resident_prefill_uses_bounded_work_units() {
    for capacity in [1, 2] {
        let control = FakeControl::new(Config {
            prefill_capacity: 3,
            ..Config::default()
        });
        let scheduler = chunked(&control, capacity, 2);
        let result = wait(&mut submit(&scheduler, &[7, 70, 71, 72, 73, 74, 75], 2));
        let chunks: Vec<usize> = control
            .events()
            .iter()
            .filter(|e| e.kind == EventKind::Prefill && e.label == 7)
            .map(|e| e.count)
            .collect();
        assert_eq!(
            chunks,
            vec![3, 3, 1],
            "idle prefill uses model geometry regardless of spare slots"
        );
        assert!(
            result.prefill_chunks == 3 && result.active_decode_prefill_chunks == 0,
            "model-sized work units still yield for admission and cancellation"
        );
    }
}

#[test]
fn arrival_does_not_wait_for_another_long_chunk() {
    for speculative in [false, true] {
        let control = FakeControl::with_gate(
            Config {
                multi_token_decode: speculative,
                batched_multi_token_decode: speculative,
                supports_batched_advance: speculative,
                prefill_capacity: 4,
                ..Config::default()
            },
            |g| g.block_prefill_label = Some(1),
        );
        let scheduler = chunked(&control, 2, 2);
        let mut long = submit(
            &scheduler,
            &[1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            4,
        );
        control.wait_for_prefill(1);
        let mut short = submit(&scheduler, &[2], 1);
        control.release_prefill();
        assert_eq!(
            wait(&mut short).tokens,
            expected_tokens(2, 1),
            "the arrival produces its independent first token"
        );
        assert_eq!(
            wait(&mut long).tokens,
            expected_tokens(1, 4),
            "the interrupted prefill keeps its trajectory"
        );
        let events = control.events();
        let second_long = event_index(&events, EventKind::Prefill, 1, 1);
        assert!(
            event_index(&events, EventKind::Prefill, 2, 0) < second_long,
            "a newcomer prefills before another chunk of the previous request"
        );
        assert!(
            event_index(&events, EventKind::Advance, 2, 0) < second_long,
            "a ready short request does not wait to batch with a distant peer"
        );
    }
}

#[test]
fn long_prefills_keep_wide_fair_turns() {
    let control = FakeControl::with_gate(
        Config {
            prefill_capacity: 4,
            ..Config::default()
        },
        |g| g.block_prefill_label = Some(1),
    );
    let scheduler = chunked(&control, 2, 2);
    let mut first = submit(
        &scheduler,
        &[1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        2,
    );
    control.wait_for_prefill(1);
    let mut second = submit(
        &scheduler,
        &[2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        2,
    );
    control.release_prefill();
    assert!(
        wait(&mut first).tokens == expected_tokens(1, 2)
            && wait(&mut second).tokens == expected_tokens(2, 2),
        "interleaved prefills preserve both outputs"
    );
    let events = control.events();
    let a = event_index(&events, EventKind::Prefill, 1, 1);
    let b = event_index(&events, EventKind::Prefill, 2, 0);
    let next_b = event_index(&events, EventKind::Prefill, 2, 1);
    assert!(
        b < a && a < next_b,
        "the newcomer gets one turn, then the older prefill progresses"
    );
    assert!(
        events[a].count == 4 && events[b].count == 4 && events[next_b].count == 4,
        "long waiting prefills keep wide chunks before any decoder is ready"
    );
}

#[test]
fn short_arrival_behind_waiting_prefill() {
    let control = FakeControl::with_gate(
        Config {
            prefill_capacity: 4,
            ..Config::default()
        },
        |g| g.block_prefill_label = Some(1),
    );
    let scheduler = chunked(&control, 3, 2);
    let mut first = submit(
        &scheduler,
        &[1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        2,
    );
    control.wait_for_prefill(1);
    let mut second = submit(&scheduler, &[2, 20, 21, 22, 23, 24, 25, 26], 2);
    let mut short = submit(&scheduler, &[3], 1);
    control.release_prefill();
    assert!(
        wait(&mut first).tokens == expected_tokens(1, 2)
            && wait(&mut second).tokens == expected_tokens(2, 2)
            && wait(&mut short).tokens == expected_tokens(3, 1),
        "three independent requests preserve their outputs"
    );
    let events = control.events();
    let second_start = event_index(&events, EventKind::Prefill, 2, 0);
    assert_eq!(
        events[second_start].count, 2,
        "a long peer gets bounded work ahead of a waiting short request"
    );
    assert!(
        second_start < event_index(&events, EventKind::Prefill, 3, 0)
            && event_index(&events, EventKind::Advance, 3, 0)
                < event_index(&events, EventKind::Prefill, 1, 1),
        "the short newcomer gets its first token without repeating the round"
    );
}

fn decode_active_prefill_is_bounded(multi_token: bool, batched: bool) {
    let control = FakeControl::with_gate(
        Config {
            multi_token_decode: multi_token,
            batched_multi_token_decode: batched,
            supports_batched_advance: multi_token,
            ..Config::default()
        },
        |g| g.block_advance_label = Some(1),
    );
    let scheduler = chunked(&control, 2, 2);
    let output_tokens = if multi_token { 30 } else { 10 };
    let mut request_a = submit(&scheduler, &[1], output_tokens);
    control.wait_for_advance(1);
    let mut request_b = submit(&scheduler, &[2, 20, 21, 22, 23, 24, 25], 2);
    control.release_advance();
    let result_a = wait(&mut request_a);
    let result_b = wait(&mut request_b);
    assert_eq!(
        result_a.tokens,
        expected_tokens(1, output_tokens),
        "active decoder preserves its isolated trajectory"
    );
    assert_eq!(
        result_b.tokens,
        expected_tokens(2, 2),
        "chunked prefill preserves the new request trajectory"
    );
    let events = control.events();
    let mut previous_b_prefill: Option<usize> = None;
    let mut b_prefill_chunks = 0;
    for (index, event) in events.iter().enumerate() {
        if event.kind != EventKind::Prefill || event.label != 2 {
            continue;
        }
        assert!(
            event.count <= 2,
            "active-decode prefill respects its token budget"
        );
        if let Some(previous) = previous_b_prefill {
            let a_advanced = events[previous + 1..index]
                .iter()
                .any(|e| e.kind == EventKind::Advance && e.label == 1);
            assert!(
                a_advanced,
                "an active decoder advances between prefill chunks"
            );
        }
        previous_b_prefill = Some(index);
        b_prefill_chunks += 1;
    }
    assert_eq!(
        b_prefill_chunks, 4,
        "long active-decode prompt is split into bounded chunks"
    );
    assert!(
        result_b.prefill_chunks == 4
            && result_b.prefill_tokens == 7
            && result_b.active_decode_prefill_chunks == 4
            && result_b.max_prefill_chunk_tokens == 2
            && result_b.max_consecutive_active_prefill_chunks == 1,
        "chunk metrics capture the selected active-decode policy"
    );
}

fn prefill_yields_to_every_due_decoder(multi_token: bool, batched: bool) {
    let control = FakeControl::with_gate(
        Config {
            multi_token_decode: multi_token,
            batched_multi_token_decode: batched,
            supports_batched_advance: multi_token,
            ..Config::default()
        },
        |g| g.block_advance_label = Some(2),
    );
    let scheduler = chunked(&control, 3, 2);
    let output_tokens = if multi_token { 30 } else { 10 };
    let mut request_a = submit(&scheduler, &[1], output_tokens);
    let mut request_b = submit(&scheduler, &[2], output_tokens);
    control.wait_for_advance(2);
    let mut request_c = submit(&scheduler, &[3, 30, 31, 32, 33, 34, 35], 2);
    control.release_advance();
    let result_a = wait(&mut request_a);
    let result_b = wait(&mut request_b);
    let result_c = wait(&mut request_c);
    assert!(
        result_a.tokens == expected_tokens(1, output_tokens)
            && result_b.tokens == expected_tokens(2, output_tokens)
            && result_c.tokens == expected_tokens(3, 2),
        "all mixed prefill/decode trajectories remain isolated"
    );
    let events = control.events();
    let mut previous_c_prefill: Option<usize> = None;
    for (index, event) in events.iter().enumerate() {
        if event.kind != EventKind::Prefill || event.label != 3 {
            continue;
        }
        if let Some(previous) = previous_c_prefill {
            let between = &events[previous + 1..index];
            let advanced = |label| {
                between
                    .iter()
                    .any(|e| e.kind == EventKind::Advance && e.label == label)
            };
            assert!(
                advanced(1) && advanced(2),
                "every due decoder advances before another prefill chunk"
            );
        }
        previous_c_prefill = Some(index);
    }
    assert_eq!(
        result_c.max_consecutive_active_prefill_chunks, 1,
        "scheduler never runs consecutive chunks while decode is due"
    );
}

#[test]
fn decode_active_prefill_bounds_and_yields() {
    for (multi_token, batched) in [(false, false), (true, false), (true, true)] {
        decode_active_prefill_is_bounded(multi_token, batched);
        prefill_yields_to_every_due_decoder(multi_token, batched);
    }
}

#[test]
fn non_incremental_runner_falls_back_safely() {
    let control = FakeControl::with_gate(
        Config {
            incremental_prefill: false,
            ..Config::default()
        },
        |g| g.block_advance_label = Some(1),
    );
    let scheduler = chunked(&control, 2, 2);
    let mut request_a = submit(&scheduler, &[1], 8);
    control.wait_for_advance(1);
    let mut request_b = submit(&scheduler, &[2, 20, 21, 22, 23], 2);
    control.release_advance();
    assert_eq!(
        wait(&mut request_a).tokens,
        expected_tokens(1, 8),
        "fallback preserves the active decoder trajectory"
    );
    let result_b = wait(&mut request_b);
    assert_eq!(
        result_b.tokens,
        expected_tokens(2, 2),
        "fallback preserves the admitted request trajectory"
    );
    assert!(
        !result_b.incremental_prefill_supported
            && result_b.prefill_chunks == 1
            && result_b.max_prefill_chunk_tokens == 5
            && result_b.prefill_fallback_reason == "incremental_prefill_unavailable",
        "non-incremental runners report their full-prefill fallback"
    );
}

#[test]
fn pending_limits_reject_before_state_admission() {
    let control = FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(1));
    let scheduler = make_scheduler(
        &control,
        1,
        TextPrefillPolicy::default(),
        TextSchedulerPolicy {
            max_pending_requests: 1,
            max_pending_requests_per_client: 1,
            ..TextSchedulerPolicy::default()
        },
    );
    let mut active = submit_for(&scheduler, &[1], 4, "active-client").expect("submit");
    control.wait_for_advance(1);
    let mut queued = submit_for(&scheduler, &[2], 1, "queued-client").expect("submit");
    let rejected = submit_for(&scheduler, &[3], 1, "third-client")
        .err()
        .and_then(|e| generation_code(&e));
    assert_eq!(
        rejected,
        Some(GenerationErrorCode::QueueFull),
        "full pending queue rejects before admission"
    );
    assert_eq!(
        control.states_created.load(Ordering::Relaxed),
        1,
        "rejected request cannot allocate another runner state"
    );
    control.release_advance();
    assert!(
        wait(&mut active).tokens == expected_tokens(1, 4)
            && wait(&mut queued).tokens == expected_tokens(2, 1),
        "accepted work survives a queue rejection"
    );
}

#[test]
fn pending_clients_are_round_robin_and_individually_bounded() {
    let control = FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(1));
    let scheduler = make_scheduler(
        &control,
        1,
        TextPrefillPolicy::default(),
        TextSchedulerPolicy {
            max_pending_requests: 4,
            max_pending_requests_per_client: 2,
            ..TextSchedulerPolicy::default()
        },
    );
    let mut active = submit_for(&scheduler, &[1], 3, "active-client").expect("submit");
    control.wait_for_advance(1);
    let mut a_first = submit_for(&scheduler, &[2], 1, "client-a").expect("submit");
    let mut a_second = submit_for(&scheduler, &[3], 1, "client-a").expect("submit");
    let mut b = submit_for(&scheduler, &[4], 1, "client-b").expect("submit");
    let rejected = submit_for(&scheduler, &[5], 1, "client-a")
        .err()
        .and_then(|e| generation_code(&e));
    assert_eq!(
        rejected,
        Some(GenerationErrorCode::ClientQueueFull),
        "one client cannot monopolize the pending queue"
    );
    control.release_advance();
    for request in [&mut active, &mut a_first, &mut a_second, &mut b] {
        wait(request);
    }
    let events = control.events();
    let a_first = event_index(&events, EventKind::Prefill, 2, 0);
    let a_second = event_index(&events, EventKind::Prefill, 3, 0);
    let b_first = event_index(&events, EventKind::Prefill, 4, 0);
    assert!(
        a_first < b_first && b_first < a_second,
        "pending clients rotate before one client receives another admission"
    );
}

#[test]
fn expired_queued_request_never_consumes_state() {
    let control = FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(1));
    let scheduler = scheduler(&control, 1);
    let mut active = submit(&scheduler, &[1], 3);
    control.wait_for_advance(1);
    let mut expired = scheduler
        .submit(
            vec![2],
            1,
            &SamplingConfig::default(),
            None,
            false,
            RequestMetadata {
                client_id: "expired-client".into(),
                deadline: Some(Instant::now() - Duration::from_millis(1)),
                ..RequestMetadata::default()
            },
        )
        .expect("submit");
    control.release_advance();
    let failure = expired.wait(None).expect_err("deadline");
    assert_eq!(
        generation_code(&failure),
        Some(GenerationErrorCode::DeadlineExceeded),
        "expired queued work reports a stable deadline"
    );
    assert_eq!(
        wait(&mut active).tokens,
        expected_tokens(1, 3),
        "expired queued work does not disturb active generation"
    );
    assert_eq!(
        control.states_created.load(Ordering::Relaxed),
        1,
        "expired queued work never creates or acquires another state"
    );
}

#[test]
fn slow_consumer_output_is_bounded_and_reclaimed() {
    let control = FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(5));
    let scheduler = make_scheduler(
        &control,
        1,
        TextPrefillPolicy::default(),
        TextSchedulerPolicy {
            max_buffered_output_bytes_per_request: 4,
            max_buffered_output_bytes_total: 4,
            ..TextSchedulerPolicy::default()
        },
    );
    let mut request = submit_streaming(&scheduler, &[5], 5);
    let callback_state = Arc::new((Mutex::new((false, false)), Condvar::new()));
    let backpressure_reported = std::thread::scope(|scope| {
        let consumer = {
            let callback_state = callback_state.clone();
            let control = control.clone();
            scope.spawn(move || {
                let mut callback = |_: &[u8]| {
                    let (lock, condition) = &*callback_state;
                    let mut state = lock.lock().expect("callback");
                    state.0 = true;
                    control.release_advance();
                    condition.notify_all();
                    let _state = condition.wait_while(state, |s| !s.1).expect("callback");
                    true
                };
                request
                    .wait(Some(&mut callback))
                    .err()
                    .and_then(|e| generation_code(&e))
                    == Some(GenerationErrorCode::OutputBackpressure)
            })
        };
        {
            let (lock, condition) = &*callback_state;
            let state = lock.lock().expect("callback");
            let (_state, timeout) = condition
                .wait_timeout_while(state, TEST_TIMEOUT, |s| !s.0)
                .expect("callback");
            assert!(
                !timeout.timed_out(),
                "slow consumer receives its first output piece"
            );
        }
        control.wait_for_invalidations(1);
        {
            let (lock, condition) = &*callback_state;
            lock.lock().expect("callback").1 = true;
            condition.notify_all();
        }
        consumer.join().expect("consumer")
    });
    assert!(
        backpressure_reported,
        "bounded output queue fails a persistently slow consumer"
    );
    assert_eq!(
        scheduler.buffered_output_bytes(),
        0,
        "failed slow-consumer output is fully reclaimed"
    );
    assert!(
        scheduler.max_buffered_output_bytes() <= 4,
        "server-wide buffered output never exceeds its declared limit"
    );
    assert_eq!(
        wait(&mut submit(&scheduler, &[6], 2)).tokens,
        expected_tokens(6, 2),
        "state is reusable after output backpressure cancellation"
    );
}

#[test]
fn generated_output_limit_applies_without_streaming() {
    let control = FakeControl::new(Config::default());
    let scheduler = make_scheduler(
        &control,
        1,
        TextPrefillPolicy::default(),
        TextSchedulerPolicy {
            max_output_bytes_per_request: 4,
            ..TextSchedulerPolicy::default()
        },
    );
    let failure = submit(&scheduler, &[7], 3).wait(None).expect_err("limit");
    assert_eq!(
        generation_code(&failure),
        Some(GenerationErrorCode::OutputLimit),
        "non-streaming generation obeys its output byte limit"
    );
}

#[test]
fn mid_generation_admission_and_isolated_trajectories() {
    let control = FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(1));
    let scheduler = scheduler(&control, 2);
    let mut request_a = submit_streaming(&scheduler, &[1, 10], 4);
    control.wait_for_advance(1);
    let mut request_b = submit_streaming(&scheduler, &[2, 20], 2);
    control.release_advance();
    let mut pieces_a = Vec::new();
    let result_a = request_a
        .wait(Some(&mut |piece: &[u8]| {
            pieces_a.push(piece.to_vec());
            true
        }))
        .expect("wait");
    let mut pieces_b = Vec::new();
    let result_b = request_b
        .wait(Some(&mut |piece: &[u8]| {
            pieces_b.push(piece.to_vec());
            true
        }))
        .expect("wait");
    assert_eq!(
        result_a.tokens,
        expected_tokens(1, 4),
        "request A follows its isolated greedy trajectory"
    );
    assert_eq!(
        result_b.tokens,
        expected_tokens(2, 2),
        "request B follows its isolated greedy trajectory"
    );
    assert!(
        pieces_a.len() == 4 && pieces_b.len() == 2,
        "each client receives only its own token pieces"
    );
    let events = control.events();
    assert!(
        event_index(&events, EventKind::Prefill, 2, 0)
            < event_index(&events, EventKind::Advance, 1, 3),
        "request B begins prefill before request A completes"
    );
}

#[test]
fn mid_generation_request_joins_next_decode_batch() {
    let control = FakeControl::with_gate(
        Config {
            supports_batched_advance: true,
            ..Config::default()
        },
        |g| g.block_advance_label = Some(1),
    );
    let scheduler = scheduler(&control, 2);
    let mut request_a = submit(&scheduler, &[1], 8);
    control.wait_for_advance(1);
    let mut request_b = submit(&scheduler, &[2], 4);
    control.release_advance();
    let result_a = wait(&mut request_a);
    let result_b = wait(&mut request_b);
    assert!(
        result_a.tokens == expected_tokens(1, 8) && result_b.tokens == expected_tokens(2, 4),
        "dynamic batching preserves both isolated trajectories"
    );
    let joined = control
        .advance_batches()
        .iter()
        .any(|labels| labels == &[1, 2] || labels == &[2, 1]);
    assert!(
        joined,
        "request B joins request A at a decode boundary after prefill"
    );
    assert!(
        result_a.physical_execution_width == 2
            && result_b.physical_execution_width == 2
            && result_a.execution_plan == "batched-w2"
            && result_b.execution_plan == "batched-w2",
        "joined requests report the real W=2 execution plan"
    );
}

#[test]
fn fifo_replacement_admission_with_one_slot() {
    let control = FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(1));
    let scheduler = scheduler(&control, 1);
    let mut request_a = submit(&scheduler, &[1, 10], 2);
    control.wait_for_advance(1);
    let mut request_b = submit(&scheduler, &[2, 20], 1);
    let mut request_c = submit(&scheduler, &[3, 30], 1);
    control.release_advance();
    let results = [
        wait(&mut request_a),
        wait(&mut request_b),
        wait(&mut request_c),
    ];
    assert!(
        results.iter().all(|r| !r.cancelled),
        "all FIFO requests complete"
    );
    let order: Vec<Token> = control
        .events()
        .iter()
        .filter(|e| e.kind == EventKind::Prefill)
        .map(|e| e.label)
        .collect();
    assert_eq!(
        order,
        vec![1, 2, 3],
        "replacement admission preserves FIFO order"
    );
}

#[test]
fn queued_and_prefill_cancellation() {
    {
        let control =
            FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(1));
        let scheduler = scheduler(&control, 1);
        let mut active = submit(&scheduler, &[1, 10], 2);
        control.wait_for_advance(1);
        let mut queued = submit(&scheduler, &[2, 20], 1);
        queued.cancel();
        control.release_advance();
        assert!(
            !wait(&mut active).cancelled,
            "active request completes normally"
        );
        assert!(
            wait(&mut queued).cancelled,
            "queued request cancellation is reported"
        );
        let events = control.events();
        assert_eq!(
            event_index(&events, EventKind::Prefill, 2, 0),
            events.len(),
            "cancelled queued request never acquires a model state"
        );
    }
    {
        let control = FakeControl::with_gate(Config::default(), |g| {
            g.block_advance_label = Some(1);
            g.block_prefill_label = Some(4);
        });
        let scheduler = chunked(&control, 2, 2);
        let mut active = submit(&scheduler, &[1], 4);
        control.wait_for_advance(1);
        let mut request = submit(&scheduler, &[4, 40, 41, 42], 2);
        control.release_advance();
        control.wait_for_prefill(4);
        request.cancel();
        control.release_prefill();
        let result = wait(&mut request);
        assert!(
            !wait(&mut active).cancelled,
            "active decoder survives another request cancellation"
        );
        assert!(
            result.cancelled,
            "active-decode prefill cancellation is reported"
        );
        let events = control.events();
        assert_eq!(
            event_index(&events, EventKind::Advance, 4, 0),
            events.len(),
            "cancelled prefill never advances decode"
        );
    }
}

#[test]
fn decode_cancellation_and_state_reclamation() {
    let control = FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(5));
    let scheduler = scheduler(&control, 1);
    let mut cancelled = submit_streaming(&scheduler, &[5, 50], 5);
    let mut delivered = 0;
    let result = cancelled
        .wait(Some(&mut |_: &[u8]| {
            delivered += 1;
            control.release_advance();
            false
        }))
        .expect("wait");
    assert!(
        result.cancelled,
        "callback cancellation reaches the scheduler"
    );
    assert_eq!(
        delivered, 1,
        "no token is published after callback cancellation"
    );
    assert_eq!(
        wait(&mut submit(&scheduler, &[6, 60], 2)).tokens,
        expected_tokens(6, 2),
        "cancelled state slot is immediately reusable"
    );
    assert!(
        control.invalidations.load(Ordering::Relaxed) >= 1,
        "decode cancellation invalidates partial model state"
    );
}

#[test]
fn four_resident_requests_make_progress() {
    let control = FakeControl::new(Config::default());
    let scheduler = scheduler(&control, 4);
    let mut requests: Vec<SchedulerRequest> = (1..=4)
        .map(|label| submit(&scheduler, &[label, label + 10], 3))
        .collect();
    for (index, request) in requests.iter_mut().enumerate() {
        let result = wait(request);
        assert_eq!(
            result.tokens,
            expected_tokens(index as Token + 1, 3),
            "C=4 serial fallback preserves isolated output"
        );
        assert!(
            result.requested_logical_concurrency == 4
                && result.physical_execution_width == 1
                && result.execution_plan == "serial-fallback",
            "C=4 telemetry exposes the physical serial fallback"
        );
    }
}

fn submit_with_stops(
    scheduler: &Scheduler,
    prompt: &[Token],
    max_tokens: usize,
    stops: &[&str],
) -> SchedulerRequest {
    scheduler
        .submit(
            prompt.to_vec(),
            max_tokens,
            &SamplingConfig::with_temperature(0.0),
            None,
            true,
            RequestMetadata {
                stop_sequences: stops.iter().map(|s| (*s).to_owned()).collect(),
                ..RequestMetadata::default()
            },
        )
        .expect("submit")
}

/// The result and the pieces streamed to the waiter.
fn wait_streamed(request: &mut SchedulerRequest) -> (GenerationResult, Vec<u8>) {
    let mut streamed = Vec::new();
    let result = request
        .wait(Some(&mut |piece: &[u8]| {
            streamed.extend_from_slice(piece);
            true
        }))
        .expect("wait");
    (result, streamed)
}

#[test]
fn stop_sequences_preserve_executed_state() {
    for multi in [false, true] {
        for preview in [false, true] {
            for first in [false, true] {
                let control = FakeControl::new(Config {
                    incremental_text_is_exact: true,
                    multi_token_decode: multi,
                    preview_first_token: preview,
                    snapshot_callback: Some(Arc::new(|| {})),
                    output_pieces: if first {
                        vec!["<STOP>tail", "later", "last"]
                    } else {
                        vec!["ab<ST", "OP>tail", "later"]
                    },
                    ..Config::default()
                });
                let scheduler = scheduler(&control, 1);
                let (result, streamed) =
                    wait_streamed(&mut submit_with_stops(&scheduler, &[1, 10], 8, &["<STOP>"]));
                assert!(
                    result.finish_reason == FinishReason::StopSequence && !result.cancelled,
                    "stop sequence is a successful completion"
                );
                let text: &[u8] = if first { b"" } else { b"ab" };
                assert!(
                    result.stop_sequence == "<STOP>" && result.text == text && streamed == text,
                    "neither the stop marker nor speculative tail leaks"
                );
                assert!(
                    result.tokens.len() == if first { 1 } else { 2 }
                        && result.completion_tokens == result.tokens.len(),
                    "usage includes the token that completed the stop"
                );
                // Repeating the prompt must not resume from an incorrectly
                // labelled frontier, also after a preview or speculative tail.
                let replay = wait(&mut submit_with_stops(&scheduler, &[1, 10], 8, &["<STOP>"]));
                assert!(
                    replay.tokens == result.tokens && replay.text == result.text,
                    "a stopped request leaves a safe reusable prompt cache"
                );
            }
        }
    }
}

#[test]
fn stop_prefix_flush_and_batch_isolation() {
    for multi in [false, true] {
        let control = FakeControl::with_gate(
            Config {
                incremental_text_is_exact: true,
                multi_token_decode: multi,
                batched_multi_token_decode: multi,
                supports_batched_advance: true,
                output_pieces: vec!["hello", "<ST", "OP>tail", "ST"],
                ..Config::default()
            },
            |g| g.block_prefill_label = Some(1),
        );
        let scheduler = scheduler(&control, 2);
        let mut first = submit_with_stops(&scheduler, &[1], 8, &["<STOP>"]);
        control.wait_for_prefill(1);
        let mut peer = submit(&scheduler, &[2], 4);
        control.release_prefill();
        let stopped = wait(&mut first);
        let full = wait(&mut peer);
        assert!(
            stopped.text == b"hello" && stopped.finish_reason == FinishReason::StopSequence,
            "batched request stops on its own marker"
        );
        assert!(
            full.text == b"hello<STOP>tailST" && full.finish_reason == FinishReason::Length,
            "peer without stop sequences keeps every accepted token"
        );
        for limit in [4, 8] {
            let (result, streamed) =
                wait_streamed(&mut submit_with_stops(&scheduler, &[4], limit, &["STXYZ"]));
            assert!(
                result.text == b"hello<STOP>tailST"
                    && streamed == result.text
                    && result.stop_sequence.is_empty()
                    && result.finish_reason
                        == if limit == 4 {
                            FinishReason::Length
                        } else {
                            FinishReason::Stop
                        },
                "unmatched stop prefix is flushed on both EOS and length"
            );
        }
    }
}

#[test]
fn cancellation_with_buffered_stop_prefix() {
    for multi in [false, true] {
        let control = FakeControl::with_gate(
            Config {
                incremental_text_is_exact: true,
                multi_token_decode: multi,
                output_pieces: vec!["safe<ST", "OP>tail", "end"],
                ..Config::default()
            },
            |g| g.block_advance_label = Some(1),
        );
        let scheduler = scheduler(&control, 1);
        let mut request = submit_with_stops(&scheduler, &[1], 8, &["<STOP>"]);
        control.wait_for_advance(1);
        request.cancel();
        control.release_advance();
        let (result, streamed) = wait_streamed(&mut request);
        assert!(
            result.cancelled
                && result.finish_reason == FinishReason::Cancelled
                && !streamed.windows(3).any(|w| w == b"<ST"),
            "cancellation does not flush a buffered stop prefix"
        );
        let replacement = wait(&mut submit(&scheduler, &[2], 3));
        assert!(
            !replacement.cancelled && replacement.text == b"safe<STOP>tailend",
            "cancellation with a partial stop releases the runner"
        );
    }
}

#[test]
fn limits_follow_the_remaining_context() {
    for speculative in [false, true] {
        for preview in [false, true] {
            let config = Config {
                max_context: 256,
                multi_token_decode: speculative,
                preview_first_token: preview,
                ..Config::default()
            };
            let control = FakeControl::new(config.clone());
            let scheduler = scheduler(&control, 2);
            for (limit, expected) in [(0, 254), (1000, 254), (2, 2)] {
                let mut first = submit(&scheduler, &[1, 10], limit);
                let mut second = submit(&scheduler, &[2, 20, 30], limit);
                let (a, b) = (wait(&mut first), wait(&mut second));
                assert_eq!(
                    a.tokens.len(),
                    expected,
                    "default/explicit limit respects remaining context"
                );
                assert_eq!(
                    b.tokens.len(),
                    if limit == 0 || limit > 253 {
                        253
                    } else {
                        limit
                    },
                    "concurrent requests have independent context budgets"
                );
                assert_eq!(
                    a.finish_reason,
                    FinishReason::Length,
                    "exhausted context is reported as length"
                );
            }
            let control = FakeControl::new(Config {
                stop_after: Some(150),
                ..config
            });
            let stopping = self::scheduler(&control, 2);
            let stopped = wait(&mut submit(&stopping, &[3], 0));
            assert!(
                stopped.tokens.len() == 150 && stopped.finish_reason == FinishReason::Stop,
                "the unlimited default passes 128 tokens and still respects EOS"
            );
            for size in [256, 257] {
                let rejected = stopping.submit_simple(vec![1; size], 0, 0.0);
                assert!(
                    rejected.is_err_and(|e| e.kind() == ErrorKind::Length),
                    "a full context is rejected before generation"
                );
            }
        }
    }
}

#[test]
fn progress_logging_is_opt_in_and_bounded() {
    let run = |enabled: bool| {
        let control = FakeControl::new(Config {
            prefill_capacity: 2,
            ..Config::default()
        });
        let (tokens, log) = gufo_serve::logging::capture_all(|| {
            let scheduler = make_scheduler(
                &control,
                1,
                TextPrefillPolicy::default(),
                TextSchedulerPolicy {
                    log_progress: enabled,
                    ..TextSchedulerPolicy::default()
                },
            );
            wait(&mut submit(&scheduler, &[7, 70, 71, 72, 73], 55))
                .tokens
                .len()
        });
        assert_eq!(tokens, 55, "progress logging preserves generated tokens");
        log
    };
    assert!(
        !run(false).contains("[progress]"),
        "progress logging is disabled by default"
    );
    let log = run(true);
    assert!(
        log.contains("phase=prefill tokens=2/5") && log.contains("phase=prefill tokens=5/5"),
        "progress logging reports model-owned prefill chunks:\n{log}"
    );
    assert!(
        log.contains("phase=decode tokens=50/55") && log.contains("phase=decode tokens=55/55"),
        "progress logging reports decode intervals and the final remainder:\n{log}"
    );
}

#[test]
fn ignore_eos_is_request_scoped() {
    let control = FakeControl::new(Config {
        eos_after: Some(0),
        ..Config::default()
    });
    let scheduler = scheduler(&control, 1);
    let run = |prompt: Token, stop_at_eos: bool| {
        let mut request = scheduler
            .submit(
                vec![prompt],
                4,
                &SamplingConfig::default(),
                None,
                false,
                RequestMetadata {
                    stop_at_eos,
                    ..RequestMetadata::default()
                },
            )
            .expect("submit");
        wait(&mut request)
    };
    let normal = run(1, true);
    assert!(
        normal.tokens.is_empty() && normal.finish_reason == FinishReason::Stop,
        "a normal request stops at EOS"
    );
    let ignored = run(2, false);
    assert!(
        ignored.tokens.len() == 4 && ignored.finish_reason == FinishReason::Length,
        "an ignore-EOS request reaches its exact token limit"
    );
    let restored = run(3, true);
    assert!(
        restored.tokens.is_empty() && restored.finish_reason == FinishReason::Stop,
        "a reused runner restores EOS stopping for the next request"
    );
}

#[test]
fn empty_token_is_published() {
    let control = FakeControl::new(Config {
        output_pieces: vec![""],
        ..Config::default()
    });
    let scheduler = scheduler(&control, 1);
    let mut events = 0;
    let result = submit_streaming(&scheduler, &[1], 1)
        .wait(Some(&mut |piece: &[u8]| {
            assert!(piece.is_empty(), "an empty token has no text bytes");
            events += 1;
            true
        }))
        .expect("wait");
    assert!(
        result.tokens.len() == 1 && events == 1,
        "one empty decoded token still has one streaming event"
    );
    assert!(
        result.max_buffered_output_bytes >= 1,
        "an empty streamed piece still charges the output budget"
    );
}

#[test]
fn runner_failure_invalidates_and_does_not_poison_replacement() {
    let control = FakeControl::with_gate(Config::default(), |g| g.throw_advance_label = Some(9));
    let scheduler = scheduler(&control, 1);
    let failure = submit(&scheduler, &[9, 90], 2)
        .wait(None)
        .expect_err("runner failure");
    assert_eq!(
        failure.message(),
        "injected scheduler runner failure",
        "runner failure reaches the submitting client"
    );
    assert!(
        !scheduler.session_states()[0].processing,
        "a failed request releases its visible session"
    );
    control.gate().throw_advance_label = None;
    assert_eq!(
        wait(&mut submit(&scheduler, &[8, 80], 2)).tokens,
        expected_tokens(8, 2),
        "replacement request succeeds after runner failure"
    );
}

#[test]
fn first_token_precedes_snapshot_and_preserves_budget() {
    for multi in [false, true] {
        let captured = Arc::new(Semaphore::default());
        let release = Arc::new(Semaphore::default());
        let callback = {
            let (captured, release) = (captured.clone(), release.clone());
            Arc::new(move || {
                captured.release();
                release.acquire();
            }) as Arc<dyn Fn() + Send + Sync>
        };
        let control = FakeControl::new(Config {
            multi_token_decode: multi,
            preview_first_token: true,
            snapshot_callback: Some(callback),
            ..Config::default()
        });
        let scheduler = scheduler(&control, 1);
        let mut request = submit_streaming(&scheduler, &[1, 10], 7);
        let first_token = Arc::new(Semaphore::default());
        let tokens = std::thread::scope(|scope| {
            let waiter = {
                let first_token = first_token.clone();
                scope.spawn(move || {
                    request
                        .wait(Some(&mut |_: &[u8]| {
                            first_token.release();
                            true
                        }))
                        .expect("wait")
                        .tokens
                })
            };
            assert!(
                captured.try_acquire_for(TEST_TIMEOUT),
                "snapshot capture reached"
            );
            let published = first_token.try_acquire_for(Duration::from_secs(1));
            release.release();
            assert!(
                published,
                "first token is delivered while snapshot capture blocks"
            );
            waiter.join().expect("waiter")
        });
        assert_eq!(
            tokens,
            expected_tokens(1, 7),
            "preview is emitted once and counts toward the original budget"
        );
    }
    for limit in [1, 7] {
        let control = FakeControl::new(Config {
            multi_token_decode: true,
            preview_first_token: true,
            batched_multi_token_decode: true,
            supports_batched_advance: true,
            ..Config::default()
        });
        let scheduler = scheduler(&control, 4);
        let mut requests: Vec<SchedulerRequest> = (1..=4)
            .map(|label| submit(&scheduler, &[label, 10], limit))
            .collect();
        for (index, request) in requests.iter_mut().enumerate() {
            assert_eq!(
                wait(request).tokens,
                expected_tokens(index as Token + 1, limit),
                "batched previews retain independent token limits"
            );
        }
    }
}

#[test]
fn snapshot_does_not_block_other_requests() {
    for multi in [false, true] {
        let entered = Arc::new(Semaphore::default());
        let release = Arc::new(Semaphore::default());
        let captures = Arc::new(AtomicUsize::new(0));
        let callback = {
            let (entered, release, captures) = (entered.clone(), release.clone(), captures.clone());
            Arc::new(move || {
                if captures.fetch_add(1, Ordering::SeqCst) == 0 {
                    entered.release();
                    release.acquire();
                }
            }) as Arc<dyn Fn() + Send + Sync>
        };
        let control = FakeControl::new(Config {
            multi_token_decode: multi,
            preview_first_token: true,
            supports_batched_advance: true,
            batched_multi_token_decode: multi,
            snapshot_callback: Some(callback),
            ..Config::default()
        });
        let scheduler = chunked(&control, 2, 2);
        let mut first = submit(&scheduler, &[1, 10], 7);
        let started = entered.try_acquire_for(TEST_TIMEOUT);
        let mut second = submit(&scheduler, &[2, 20, 21, 22, 23, 24, 25], 7);
        let (sender, receiver) = mpsc::channel();
        let second_result = std::thread::scope(|scope| {
            scope.spawn(move || {
                let _ = sender.send(second.wait(None));
            });
            let independent = receiver.recv_timeout(TEST_TIMEOUT);
            release.release();
            assert!(
                started && independent.is_ok(),
                "one capture must not block other requests"
            );
            independent
                .ok()
                .and_then(|r| r.ok())
                .expect("second result")
        });
        assert_eq!(
            second_result.tokens,
            expected_tokens(2, 7),
            "other request stays independent"
        );
        assert!(
            second_result.max_prefill_chunk_tokens == 2 && second_result.prefill_chunks == 4,
            "prefill stays bounded while an already-published request captures"
        );
        let first_result = wait(&mut first);
        assert_eq!(
            first_result.tokens,
            expected_tokens(1, 7),
            "captured request resumes exactly"
        );
        let advance_ms = control.first_request_advance_ns.load(Ordering::Relaxed) as f64 / 1e6;
        assert!(
            advance_ms > 0.0 && first_result.decode_ms >= advance_ms,
            "async snapshot time must not be subtracted from timed model work"
        );
        let cached = wait(&mut submit(&scheduler, &[1, 10], 7));
        assert!(
            cached.cache_hit && cached.tokens == expected_tokens(1, 7),
            "asynchronous capture retains the immutable prompt frontier"
        );
    }
}

/// A request leaving its capture decodes before another bounded chunk of a
/// peer's prefill.
#[test]
fn captured_decoder_resumes_before_more_prefill() {
    let entered = Arc::new(Semaphore::default());
    let release = Arc::new(Semaphore::default());
    let finished = Arc::new(Semaphore::default());
    let captures = Arc::new(AtomicUsize::new(0));
    let callback = {
        let (entered, release, finished, captures) = (
            entered.clone(),
            release.clone(),
            finished.clone(),
            captures.clone(),
        );
        Arc::new(move || {
            if captures.fetch_add(1, Ordering::SeqCst) == 0 {
                entered.release();
                release.acquire();
                finished.release();
            }
        }) as Arc<dyn Fn() + Send + Sync>
    };
    let control = FakeControl::with_gate(
        Config {
            max_context: 4096,
            preview_first_token: true,
            snapshot_callback: Some(callback),
            ..Config::default()
        },
        |gate| gate.block_prefill_label = Some(2),
    );
    let scheduler = chunked(&control, 2, 2);
    let mut first = submit(&scheduler, &[1, 10], 3);
    assert!(
        entered.try_acquire_for(TEST_TIMEOUT),
        "first capture starts"
    );
    let mut second = submit(&scheduler, &[2, 20, 21, 22, 23, 24, 25, 26, 27], 1);
    // Finish the capture while the peer's bounded chunk is still running.
    control.wait_for_prefill(2);
    release.release();
    assert!(
        finished.try_acquire_for(TEST_TIMEOUT),
        "first capture finishes"
    );
    std::thread::sleep(Duration::from_millis(50));
    control.release_prefill();
    assert_eq!(
        wait(&mut first).tokens,
        expected_tokens(1, 3),
        "captured request resumes exactly"
    );
    assert_eq!(
        wait(&mut second).tokens,
        expected_tokens(2, 1),
        "peer prefill completes"
    );
    let events = control.events();
    assert!(
        event_index(&events, EventKind::Advance, 1, 0)
            < event_index(&events, EventKind::Prefill, 2, 1),
        "a request leaving capture decodes before another peer chunk: {events:?}"
    );
}

#[test]
fn captures_at_capacity_allow_queued_progress() {
    for multi in [false, true] {
        for capacity in [1usize, 2, 4] {
            let entered = Arc::new(Semaphore::default());
            let release = Arc::new(Semaphore::default());
            let captures = Arc::new(AtomicUsize::new(0));
            let callback = {
                let (entered, release, captures) =
                    (entered.clone(), release.clone(), captures.clone());
                Arc::new(move || {
                    if captures.fetch_add(1, Ordering::SeqCst) < capacity {
                        entered.release();
                        release.acquire();
                    }
                }) as Arc<dyn Fn() + Send + Sync>
            };
            let control = FakeControl::new(Config {
                multi_token_decode: multi,
                preview_first_token: true,
                supports_batched_advance: true,
                batched_multi_token_decode: multi,
                snapshot_callback: Some(callback),
                ..Config::default()
            });
            let scheduler = scheduler(&control, capacity);
            let mut requests = Vec::new();
            for i in 0..capacity {
                // Captures during prefill as well as after first-token
                // publication, including the single-slot admission deadlock.
                requests.push(
                    scheduler
                        .submit(
                            vec![i as Token + 1, 10],
                            7,
                            &SamplingConfig::default(),
                            None,
                            false,
                            RequestMetadata {
                                cache_prefix_tokens: usize::from(multi),
                                ..RequestMetadata::default()
                            },
                        )
                        .expect("submit"),
                );
                assert!(
                    entered.try_acquire_for(TEST_TIMEOUT),
                    "every resident reaches snapshot capture"
                );
            }
            for i in capacity..capacity * 2 {
                requests.push(submit(&scheduler, &[i as Token + 1, 10], 7));
            }
            release.release_n(capacity);
            let (sender, receiver) = mpsc::channel();
            std::thread::scope(|scope| {
                scope.spawn(move || {
                    let tokens: Vec<Vec<Token>> =
                        requests.iter_mut().map(|r| wait(r).tokens).collect();
                    let _ = sender.send(tokens);
                });
                let tokens = receiver
                    .recv_timeout(TEST_TIMEOUT)
                    .expect("all slots capturing must not deadlock queued admission");
                for (i, tokens) in tokens.iter().enumerate() {
                    assert_eq!(
                        tokens,
                        &expected_tokens(i as Token + 1, 7),
                        "capture and queued requests retain independent output"
                    );
                }
            });
        }
    }
}

#[test]
fn shutdown_cancels_runner_acquisition() {
    let control = FakeControl::new(Config::default());
    let pool = Arc::new(
        TextRunnerPool::new(
            Arc::new(FakeRunner {
                control: control.clone(),
            }),
            1,
            None,
        )
        .expect("pool"),
    );
    let lease = pool.acquire(vec![1, 10]).expect("acquire");
    let scheduler = TextGenerationScheduler::new(
        pool.clone(),
        TextPrefillPolicy::default(),
        TextSchedulerPolicy::default(),
    )
    .expect("scheduler");
    let acquiring = Arc::new(Semaphore::default());
    let notified = Arc::new(AtomicBool::new(false));
    let check = {
        let (acquiring, notified) = (acquiring.clone(), notified.clone());
        Arc::new(move || {
            if !notified.swap(true, Ordering::SeqCst) {
                acquiring.release();
            }
            false
        })
    };
    let _request = scheduler
        .submit(
            vec![2, 10],
            7,
            &SamplingConfig::default(),
            Some(check),
            false,
            RequestMetadata::default(),
        )
        .expect("submit");
    assert!(
        acquiring.try_acquire_for(TEST_TIMEOUT),
        "request reaches admission with external runner lease"
    );
    let (sender, receiver) = mpsc::channel();
    let stopper = std::thread::spawn(move || {
        drop(scheduler);
        let _ = sender.send(());
    });
    let completed = receiver.recv_timeout(TEST_TIMEOUT).is_ok();
    drop(lease);
    assert!(completed, "shutdown cancels acquisition of a leased runner");
    stopper.join().expect("stopper");
}

/// Concurrent arrivals sharing a long prefix with a request still prefilling
/// wait for its checkpoint instead of prefilling the same tokens.
#[test]
fn concurrent_shared_prefixes_prefill_once() {
    let control = shared_prefix_control(true);
    let scheduler = scheduler(&control, 4);
    let leader_prompt = shared_prompt(1200, 7000, 100);
    let mut leader = submit(&scheduler, &leader_prompt, 2);
    control.wait_for_prefill(1);
    // While the leader prefills: an identical prompt, one diverging after the
    // shared system prompt, and one sharing too little to wait for.
    let mut identical = submit(&scheduler, &leader_prompt, 2);
    let mut diverging = submit(&scheduler, &shared_prompt(1200, 8000, 100), 2);
    let mut unrelated = submit(&scheduler, &shared_prompt(100, 9000, 900), 2);
    control.release_prefill();
    let a = wait(&mut leader);
    let b = wait(&mut identical);
    let d = wait(&mut diverging);
    let e = wait(&mut unrelated);
    for result in [&a, &b, &d, &e] {
        assert_eq!(
            result.tokens,
            expected_tokens(1, 2),
            "every output is preserved"
        );
    }
    assert!(
        a.prefill_tokens == 1300 && a.shared_prefix_wait_ms == 0.0,
        "the leader prefills its prompt without waiting"
    );
    assert!(
        b.cached_prompt_tokens == 1299 && b.prefill_tokens == 1 && b.shared_prefix_wait_ms > 0.0,
        "an identical arrival restores the leader's last prefill position: {} {} {}",
        b.cached_prompt_tokens,
        b.prefill_tokens,
        b.shared_prefix_wait_ms
    );
    assert!(
        d.cached_prompt_tokens == 1200 && d.prefill_tokens == 100 && d.shared_prefix_wait_ms > 0.0,
        "a diverging arrival restores the shared prefix and prefills its tail: {} {} {}",
        d.cached_prompt_tokens,
        d.prefill_tokens,
        d.shared_prefix_wait_ms
    );
    assert!(
        e.cached_prompt_tokens == 0 && e.prefill_tokens == 1000 && e.shared_prefix_wait_ms == 0.0,
        "a short shared prefix is prefilled without waiting"
    );
}

/// A follower admitted after its leader's cancellation prefills alone
///.
#[test]
fn leader_cancelled_before_follower_is_admitted() {
    let control = shared_prefix_control(true);
    let scheduler = scheduler(&control, 2);
    let prompt = shared_prompt(1200, 7000, 100);
    let mut leader = submit(&scheduler, &prompt, 2);
    control.wait_for_prefill(1);
    let mut follower = submit(&scheduler, &prompt, 2);
    leader.cancel();
    control.release_prefill();
    let a = wait(&mut leader);
    let b = wait(&mut follower);
    assert!(a.cancelled, "the leader is cancelled");
    assert!(
        !b.cancelled
            && b.tokens == expected_tokens(1, 2)
            && b.cached_prompt_tokens + b.prefill_tokens == prompt.len(),
        "a follower admitted after its leader's cancellation prefills alone"
    );
}

/// Sessions follow admitted requests from admission to their end, with live
/// prompt and generation progress.
#[test]
fn session_states_follow_admitted_requests() {
    let control = FakeControl::with_gate(Config::default(), |gate| {
        gate.block_advance_label = Some(1);
        gate.block_prefill_label = Some(2);
    });
    let scheduler = scheduler(&control, 2);
    let all_idle = |scheduler: &Scheduler| {
        let states = scheduler.session_states();
        states.len() == 2
            && states.iter().all(|state| {
                !state.processing
                    && state.request_id == 0
                    && state.prompt_tokens == 0
                    && state.generated_tokens == 0
            })
    };
    assert!(all_idle(&scheduler), "every session starts idle");

    let mut first = submit(&scheduler, &[1, 10], 8);
    control.wait_for_advance(1);
    let states = scheduler.session_states();
    assert!(
        states[0].processing
            && states[0].request_id == first.id()
            && states[0].prompt_tokens == 2
            && states[0].cached_prompt_tokens == 0
            && states[0].processed_prompt_tokens == 2
            && states[0].generated_tokens == 1
            && states[0].remaining_tokens == 7
            && !states[1].processing,
        "a decoding request reports its session progress: {states:?}"
    );
    let mut second = submit(&scheduler, &[2, 20, 21], 1);
    assert!(
        !scheduler.session_states()[1].processing,
        "a queued request does not hold a session"
    );
    control.release_advance();
    control.wait_for_prefill(2);
    let states = scheduler.session_states();
    assert!(
        states[0].processing
            && states[0].request_id == first.id()
            && states[1].processing
            && states[1].request_id == second.id()
            && states[1].prompt_tokens == 3
            && states[1].processed_prompt_tokens == 0
            && states[1].generated_tokens == 0
            && states[1].remaining_tokens == 1,
        "concurrent requests hold distinct sessions: {states:?}"
    );
    control.release_prefill();
    assert!(
        wait(&mut first).tokens.len() == 8 && wait(&mut second).tokens.len() == 1,
        "both requests complete"
    );
    assert!(
        all_idle(&scheduler),
        "completed requests release their sessions"
    );

    let cache_control = FakeControl::new(Config {
        snapshot_callback: Some(Arc::new(|| {})),
        ..Config::default()
    });
    let cache_scheduler = scheduler_fixture::scheduler(&cache_control, 2);
    wait(&mut submit(&cache_scheduler, &[1, 10], 1));
    cache_control.gate().block_prefill_label = Some(1);
    let mut continuation = submit(&cache_scheduler, &[1, 10, 100, 11], 2);
    cache_control.wait_for_prefill(1);
    let resumed = cache_scheduler.session_states()[0].clone();
    assert!(
        resumed.processing
            && resumed.request_id == continuation.id()
            && resumed.prompt_tokens == 4
            && resumed.cached_prompt_tokens == 3
            && resumed.processed_prompt_tokens == 0,
        "a continuation reports its cached prompt tokens: {resumed:?}"
    );
    continuation.cancel();
    cache_control.release_prefill();
    assert!(
        wait(&mut continuation).cancelled,
        "continuation is cancelled"
    );
    assert!(
        !cache_scheduler.session_states()[0].processing,
        "cancelled requests release their sessions"
    );

    let speculative = FakeControl::new(Config {
        multi_token_decode: true,
        ..Config::default()
    });
    assert!(
        scheduler_fixture::scheduler(&speculative, 1).session_states()[0].speculative,
        "multi-token decoding is reported as speculative"
    );
}

fn submit_with_progress(
    scheduler: &Scheduler,
    prompt: &[Token],
    max_tokens: usize,
    stream: bool,
    return_progress: bool,
) -> SchedulerRequest {
    scheduler
        .submit(
            prompt.to_vec(),
            max_tokens,
            &SamplingConfig::default(),
            None,
            stream,
            RequestMetadata {
                return_progress,
                ..RequestMetadata::default()
            },
        )
        .expect("submit")
}

/// Streamed `return_progress` requests report prompt progress before their
/// first token.
#[test]
fn prompt_progress_precedes_streamed_tokens() {
    let control = FakeControl::new(Config {
        prefill_capacity: 2,
        ..Config::default()
    });
    let scheduler = scheduler(&control, 1);
    let run = |prompt: &[Token], stream: bool, report_progress: bool| {
        let mut progress: Vec<PromptProgress> = Vec::new();
        let tokens = std::cell::Cell::new(0);
        let mut request = submit_with_progress(&scheduler, prompt, 2, stream, report_progress);
        request
            .wait_streaming(StreamCallbacks {
                on_token: Some(&mut |_: &[u8]| {
                    tokens.set(tokens.get() + 1);
                    true
                }),
                on_progress: Some(&mut |value: &PromptProgress| {
                    assert_eq!(tokens.get(), 0, "prompt progress precedes generated tokens");
                    assert!(
                        progress
                            .last()
                            .is_none_or(|last| value.processed >= last.processed),
                        "prompt progress never moves backwards"
                    );
                    progress.push(*value);
                    true
                }),
                on_start: None,
            })
            .expect("wait");
        progress
    };

    let cold = run(&[7, 70, 71, 72, 73], true, true);
    assert!(
        cold.last()
            .is_some_and(|last| last.total == 5 && last.cache == 0 && last.processed == 5),
        "cold prompt progress ends with the complete prompt: {cold:?}"
    );
    let cached = run(&[7, 70, 71, 72, 73, 700, 701, 80], true, true);
    assert!(
        cached
            .last()
            .is_some_and(|last| last.total == 8 && last.cache == 7 && last.processed == 8),
        "cached prompt progress counts reused tokens as processed: {cached:?}"
    );
    assert!(
        run(&[9, 90, 91], false, true).is_empty(),
        "buffered requests do not publish prompt progress"
    );
    assert!(
        run(&[9, 90, 91], true, false).is_empty(),
        "streaming alone does not enable progress publication"
    );
}

#[test]
fn prompt_progress_cancellation() {
    let control = FakeControl::with_gate(
        Config {
            prefill_capacity: 2,
            ..Config::default()
        },
        |gate| gate.block_prefill_label = Some(1),
    );
    let scheduler = scheduler(&control, 2);
    let mut request = submit_with_progress(&scheduler, &[1, 10, 11, 12], 8, true, true);
    control.wait_for_prefill(1);
    let received = Arc::new(Semaphore::default());
    let consumer = {
        let received = received.clone();
        std::thread::spawn(move || {
            let result = request
                .wait_streaming(StreamCallbacks {
                    on_progress: Some(&mut |_: &PromptProgress| {
                        received.release();
                        false
                    }),
                    ..StreamCallbacks::default()
                })
                .expect("wait");
            assert!(
                result.cancelled,
                "progress callback can cancel before the first token"
            );
        })
    };
    assert!(
        received.try_acquire_for(TEST_TIMEOUT),
        "initial progress is live"
    );
    control.release_prefill();
    consumer.join().expect("consumer");
    let peer = wait(&mut submit(&scheduler, &[2, 20], 2));
    assert!(
        !peer.cancelled && peer.completion_tokens == 2,
        "progress cancellation leaves peer state usable"
    );
}

/// Admission decisions are logged only at debug.
#[test]
fn admission_logging_is_debug_tier_only() {
    let run = |level: LogLevel| {
        let control =
            FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(1));
        let scheduler = make_scheduler(
            &control,
            1,
            TextPrefillPolicy::default(),
            TextSchedulerPolicy {
                max_pending_requests: 1,
                max_pending_requests_per_client: 1,
                ..TextSchedulerPolicy::default()
            },
        );
        // Submissions log on the submitting thread.
        let ((mut active, mut queued, rejected), log) = logging::capture_at(level, || {
            let active = submit_for(&scheduler, &[1], 4, "active-client").expect("submit");
            control.wait_for_advance(1);
            let queued = submit_for(&scheduler, &[2], 1, "queued-client").expect("submit");
            let rejected = submit_for(&scheduler, &[3], 1, "third-client")
                .err()
                .and_then(|e| generation_code(&e));
            (active, queued, rejected)
        });
        control.release_advance();
        wait(&mut active);
        wait(&mut queued);
        assert_eq!(
            rejected,
            Some(GenerationErrorCode::QueueFull),
            "admission logging still rejects a full pending queue"
        );
        log
    };
    assert!(!run(LogLevel::Info).contains("event=admission"));
    let log = run(LogLevel::Debug);
    assert!(log.contains("[DEBUG] [scheduler] event=admitted"), "{log}");
    assert!(log.contains(" client_id=queued-client queued=1"), "{log}");
    assert!(
        log.contains(
            "event=admission_refused reason=queue_full queued=1 limit=1 client_id=third-client"
        ),
        "{log}"
    );
}

fn is_device_lost(error: &gufo_core::text::Error) -> bool {
    error.generation_code() == Some(GenerationErrorCode::DeviceLost)
        && error.message() == gufo_core::text::DEVICE_LOST_MESSAGE
}

/// A failure on a lost device makes the loss permanent: queued and
/// cancelled work fail without touching the device, no state is reset, and
/// later submissions are refused.
#[test]
fn device_loss_is_sticky_and_reported() {
    let control = FakeControl::with_gate(Config::default(), |g| {
        g.throw_advance_label = Some(9);
        g.block_advance_label = Some(9);
    });
    control.device_lost.store(true, Ordering::Relaxed);
    let scheduler = scheduler(&control, 2);
    let mut failing = submit(&scheduler, &[9, 90], 2);
    control.wait_for_advance(9);
    let invalidations_before_loss = control.invalidations.load(Ordering::SeqCst);
    let mut queued = submit(&scheduler, &[8, 80], 2);
    let mut cancelled = submit(&scheduler, &[7, 70], 2);
    cancelled.cancel();
    control.release_advance();
    let failure = failing.wait(None).expect_err("device lost");
    assert!(
        is_device_lost(&failure),
        "a failure on a lost device reports device_lost"
    );
    assert!(
        scheduler.device_lost() && control.device_probes.load(Ordering::SeqCst) == 1,
        "the failing work unit probes the device once"
    );
    for (request, message) in [
        (
            &mut queued,
            "queued work fails without touching the lost device",
        ),
        (
            &mut cancelled,
            "shutdown never resets a cancelled lost-device request",
        ),
    ] {
        let failure = request.wait(None).expect_err("device lost");
        assert!(is_device_lost(&failure), "{message}");
    }
    assert_eq!(
        control.invalidations.load(Ordering::SeqCst),
        invalidations_before_loss,
        "lost-device failure skips state invalidation"
    );
    assert!(
        control.events().iter().all(|event| event.label == 9),
        "no peer performs model work after loss"
    );
    control.gate().throw_advance_label = None;
    control.device_lost.store(false, Ordering::Relaxed);
    let refused = scheduler
        .submit_simple(vec![8, 80], 2, 0.0)
        .err()
        .expect("refused");
    assert!(
        is_device_lost(&refused),
        "submissions after device loss fail before admission"
    );
    assert_eq!(
        control.device_probes.load(Ordering::SeqCst),
        1,
        "device loss is never probed again"
    );
}

fn is_scheduler_failed(error: &gufo_core::text::Error) -> bool {
    error.generation_code() == Some(GenerationErrorCode::SchedulerFailed)
        && error.message() == gufo_core::text::SCHEDULER_FAILED_MESSAGE
}

/// A panic on the scheduler thread fails the request it ran, the resident
/// and queued ones with it, and every later submission, instead of leaving
/// their waiters blocked for good.
#[test]
fn scheduler_panic_fails_every_request() {
    let control = FakeControl::with_gate(Config::default(), |g| {
        g.block_advance_label = Some(9);
        g.panic_advance_label = Some(9);
    });
    // One session: the second request stays queued behind the first.
    let scheduler = scheduler(&control, 1);
    let mut failing = submit(&scheduler, &[9, 90], 2);
    control.wait_for_advance(9);
    let mut queued = submit(&scheduler, &[8, 80], 2);
    assert_eq!(scheduler.fatal_failure(), None);
    control.release_advance();
    for (request, message) in [
        (&mut failing, "the panicking request fails"),
        (&mut queued, "queued work fails with it"),
    ] {
        let failure = request.wait(None).expect_err("scheduler failed");
        assert!(is_scheduler_failed(&failure), "{message}: {failure:?}");
    }
    assert_eq!(
        scheduler.fatal_failure(),
        Some(GenerationErrorCode::SchedulerFailed)
    );
    control.gate().panic_advance_label = None;
    let refused = scheduler
        .submit_simple(vec![7, 70], 2, 0.0)
        .err()
        .expect("refused");
    assert!(
        is_scheduler_failed(&refused),
        "submissions after the panic fail before admission"
    );
    assert!(
        control.events().iter().all(|event| event.label == 9),
        "no other request ran model work"
    );
}

/// A failed lane is classified before a successful peer commits on the same
/// context.
#[test]
fn batch_device_loss_fails_successful_peers() {
    for speculative in [false, true] {
        let control = FakeControl::with_gate(
            Config {
                multi_token_decode: speculative,
                batched_multi_token_decode: speculative,
                supports_batched_advance: true,
                ..Config::default()
            },
            |g| {
                g.block_prefill_label = Some(1);
                g.throw_advance_label = Some(2);
            },
        );
        control.device_lost.store(true, Ordering::Relaxed);
        let scheduler = scheduler(&control, 2);
        // AR gives the first decoder one advance before admitting the peer;
        // multi-token batching assembles the initial cohort first.
        let max_tokens = if speculative { 1 } else { 2 };
        let mut first = submit(&scheduler, &[1], max_tokens);
        control.wait_for_prefill(1);
        let mut second = submit(&scheduler, &[2], max_tokens);
        control.release_prefill();
        for request in [&mut first, &mut second] {
            let failure = request.wait(None).expect_err("device lost");
            assert!(
                is_device_lost(&failure),
                "device loss fails the entire batch before committing peers ({speculative})"
            );
        }
        assert!(
            control.device_probes.load(Ordering::SeqCst) == 1 && scheduler.device_lost(),
            "batch loss probes once and remains sticky"
        );
    }
}

/// Only an unclassified failure probes the device; a usable one keeps the
/// failure recoverable.
#[test]
fn failures_probe_the_device_only_when_unclassified() {
    let control = FakeControl::new(Config::default());
    control.device_lost.store(true, Ordering::Relaxed);
    let scheduler = make_scheduler(
        &control,
        1,
        TextPrefillPolicy::default(),
        TextSchedulerPolicy {
            max_output_bytes_per_request: 4,
            ..TextSchedulerPolicy::default()
        },
    );
    let limited = submit(&scheduler, &[7], 3).wait(None).expect_err("limit");
    assert_eq!(
        limited.generation_code(),
        Some(GenerationErrorCode::OutputLimit)
    );
    assert!(
        control.device_probes.load(Ordering::SeqCst) == 0 && !scheduler.device_lost(),
        "a scheduler-classified failure never probes the device"
    );

    let control = FakeControl::with_gate(Config::default(), |g| g.throw_advance_label = Some(9));
    let scheduler = scheduler_fixture::scheduler(&control, 1);
    let failure = submit(&scheduler, &[9, 90], 2)
        .wait(None)
        .expect_err("failure");
    assert_eq!(failure.message(), "injected scheduler runner failure");
    assert!(
        control.device_probes.load(Ordering::SeqCst) == 1 && !scheduler.device_lost(),
        "a failure on a usable device is probed once and stays recoverable"
    );
    control.gate().throw_advance_label = None;
    wait(&mut submit(&scheduler, &[8, 80], 2));
    assert_eq!(
        control.device_probes.load(Ordering::SeqCst),
        1,
        "successful work never probes"
    );
}

/// An idle scheduler polls the device without waiting, never during
/// inference, and a confirmed loss is permanent.
#[test]
fn idle_device_probe() {
    let control = FakeControl::new(Config::default());
    control.device_pending.store(true, Ordering::Relaxed);
    let scheduler = make_scheduler(
        &control,
        1,
        TextPrefillPolicy::default(),
        TextSchedulerPolicy {
            device_probe_interval: Duration::from_millis(10),
            ..TextSchedulerPolicy::default()
        },
    );
    {
        let gate = control.gate();
        let (_gate, timeout) = control
            .condition
            .wait_timeout_while(gate, TEST_TIMEOUT, |_| {
                control.device_probes.load(Ordering::Relaxed) < 2
            })
            .expect("gate");
        assert!(
            !timeout.timed_out(),
            "an idle scheduler polls without generation or HTTP traffic"
        );
    }
    assert!(!scheduler.device_lost(), "a pending probe is inconclusive");
    control.gate().block_prefill_label = Some(1);
    let mut request = submit(&scheduler, &[1], 2);
    control.wait_for_prefill(1);
    let probes = control.device_probes.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(
        control.device_probes.load(Ordering::Relaxed),
        probes,
        "an idle probe never polls during active inference"
    );
    control.release_prefill();
    assert_eq!(
        wait(&mut request).tokens,
        expected_tokens(1, 2),
        "arriving inference succeeds with an outstanding probe"
    );
    control.device_lost.store(true, Ordering::Relaxed);
    let deadline = Instant::now() + TEST_TIMEOUT;
    while !scheduler.device_lost() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        scheduler.device_lost(),
        "idle probe loss is fatal and sticky"
    );
}

/// A stream reports its start at admission, before its prefill completes,
/// or after a bounded wait in the queue; buffered and cancelled requests
/// never start.
#[test]
fn stream_start_precedes_prefill() {
    let policy = |delay: Duration| TextSchedulerPolicy {
        stream_start_delay: delay,
        ..TextSchedulerPolicy::default()
    };
    let submit_stream = |scheduler: &Scheduler, prompt: &[Token], stream: bool| {
        scheduler
            .submit(
                prompt.to_vec(),
                2,
                &SamplingConfig::default(),
                None,
                stream,
                RequestMetadata::default(),
            )
            .expect("submit")
    };
    let hour = Duration::from_secs(3600);
    {
        // Admission starts a stream while its prompt is processed.
        let control =
            FakeControl::with_gate(Config::default(), |g| g.block_prefill_label = Some(1));
        let scheduler = make_scheduler(&control, 1, TextPrefillPolicy::default(), policy(hour));
        let mut request = submit_stream(&scheduler, &[1, 10, 11], true);
        let started = Arc::new(Semaphore::default());
        let consumer = {
            let started = started.clone();
            std::thread::spawn(move || {
                let (starts, tokens) = (std::cell::Cell::new(0), std::cell::Cell::new(0));
                let result = request
                    .wait_streaming(StreamCallbacks {
                        on_token: Some(&mut |_: &[u8]| {
                            tokens.set(tokens.get() + 1);
                            true
                        }),
                        on_start: Some(&mut || {
                            assert_eq!(tokens.get(), 0, "stream start precedes tokens");
                            starts.set(starts.get() + 1);
                            started.release();
                            true
                        }),
                        ..StreamCallbacks::default()
                    })
                    .expect("wait");
                (result.cancelled, tokens.get(), starts.get())
            })
        };
        control.wait_for_prefill(1);
        assert!(
            started.try_acquire_for(TEST_TIMEOUT),
            "admission starts a stream before prefill completes"
        );
        control.release_prefill();
        let (cancelled, tokens, starts) = consumer.join().expect("consumer");
        assert!(
            !cancelled && tokens > 0 && starts == 1,
            "an admitted stream starts exactly once"
        );
    }
    {
        // A stream waiting behind another request starts after the bound.
        let control =
            FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(1));
        let scheduler = make_scheduler(
            &control,
            1,
            TextPrefillPolicy::default(),
            policy(Duration::from_millis(20)),
        );
        let mut active = submit_stream(&scheduler, &[1, 10], true);
        control.wait_for_advance(1);
        let mut queued = submit_stream(&scheduler, &[2, 20], true);
        let started = Arc::new(Semaphore::default());
        let consumer = {
            let started = started.clone();
            std::thread::spawn(move || {
                let starts = std::cell::Cell::new(0);
                // The only session is busy, so the start precedes admission.
                let result = queued.wait_streaming(StreamCallbacks {
                    on_start: Some(&mut || {
                        starts.set(starts.get() + 1);
                        started.release();
                        true
                    }),
                    ..StreamCallbacks::default()
                });
                (result.expect("wait").cancelled, starts.get())
            })
        };
        assert!(
            started.try_acquire_for(TEST_TIMEOUT),
            "a queued stream starts after its bound"
        );
        control.release_advance();
        let (cancelled, starts) = consumer.join().expect("consumer");
        assert!(
            !wait(&mut active).cancelled && !cancelled && starts == 1,
            "admission after a queued start does not start again"
        );
    }
    {
        // Buffered requests and failures before admission never start.
        let control =
            FakeControl::with_gate(Config::default(), |g| g.block_advance_label = Some(1));
        let scheduler = make_scheduler(&control, 1, TextPrefillPolicy::default(), policy(hour));
        let mut active = submit_stream(&scheduler, &[1, 10], false);
        control.wait_for_advance(1);
        let mut queued = submit_stream(&scheduler, &[2, 20], true);
        queued.cancel();
        control.release_advance();
        let started = std::cell::Cell::new(false);
        let mut start = || {
            started.set(true);
            true
        };
        let cancelled = queued
            .wait_streaming(StreamCallbacks {
                on_start: Some(&mut start),
                ..StreamCallbacks::default()
            })
            .expect("wait")
            .cancelled;
        assert!(
            cancelled && !started.get(),
            "a request cancelled while queued never starts"
        );
        let mut start = || {
            started.set(true);
            true
        };
        let buffered = active
            .wait_streaming(StreamCallbacks {
                on_start: Some(&mut start),
                ..StreamCallbacks::default()
            })
            .expect("wait");
        assert!(
            !buffered.cancelled && !started.get(),
            "buffered requests never start"
        );
    }
}

/// A completed stream whose output nobody reads returns its charge when its
/// last owner drops it, as after failed HTTP headers.
#[test]
fn abandoned_completed_streams_release_their_output() {
    for drop_early in [false, true] {
        let control =
            FakeControl::with_gate(Config::default(), |g| g.block_prefill_label = Some(2));
        let scheduler = make_scheduler(
            &control,
            1,
            TextPrefillPolicy::default(),
            TextSchedulerPolicy {
                max_buffered_output_bytes_per_request: 3,
                max_buffered_output_bytes_total: 3,
                ..TextSchedulerPolicy::default()
            },
        );
        let mut barrier;
        {
            let abandoned = submit_streaming(&scheduler, &[1], 1);
            barrier = submit(&scheduler, &[2], 1);
            // The single runner admits the barrier only after the stream
            // completed; its token stays unread.
            control.wait_for_prefill(2);
            assert!(
                abandoned.phase() == TextRequestPhase::Terminal
                    && scheduler.buffered_output_bytes() == 3,
                "a completed stream keeps its unread output until it is dropped"
            );
            if drop_early {
                drop(abandoned);
            }
        }
        control.release_prefill();
        assert_eq!(
            wait(&mut barrier).tokens,
            expected_tokens(2, 1),
            "dropping completed output leaves the active request usable"
        );
        let mut output = Vec::new();
        let mut callback = |piece: &[u8]| {
            output.extend_from_slice(piece);
            true
        };
        let replacement = submit_streaming(&scheduler, &[3], 1)
            .wait(Some(&mut callback))
            .expect("the replacement stream reuses the abandoned capacity");
        assert!(
            replacement.tokens == expected_tokens(3, 1)
                && !output.is_empty()
                && !replacement.cancelled
        );
        assert_eq!(
            scheduler.buffered_output_bytes(),
            0,
            "abandoned and consumed output leave no reserved capacity"
        );
    }
}

/// Completed output stays readable after its scheduler is gone, and both
/// unread and consumed requests release their last owner cleanly.
#[test]
fn completed_streams_outlive_their_scheduler() {
    for consume in [false, true] {
        let mut request;
        {
            let control = FakeControl::new(Config::default());
            let scheduler = scheduler(&control, 1);
            request = submit_streaming(&scheduler, &[1], 1);
            // Completion of later work proves the unread stream is done.
            wait(&mut submit(&scheduler, &[2], 1));
            assert_eq!(request.phase(), TextRequestPhase::Terminal);
        }
        if consume {
            let mut output = Vec::new();
            let mut callback = |piece: &[u8]| {
                output.extend_from_slice(piece);
                true
            };
            let result = request.wait(Some(&mut callback)).expect("result");
            assert!(
                result.tokens == expected_tokens(1, 1) && !output.is_empty() && !result.cancelled
            );
        }
    }
}
