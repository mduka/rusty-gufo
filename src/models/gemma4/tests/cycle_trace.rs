//! The extension's `cycle_trace_test.cpp`: one JSON object per line, every
//! field present, appended across writers.

#![allow(clippy::unwrap_used)]

use gufo_core::json::{self, Value};
use gufo_gemma4::cycle_trace::{CycleRecord, CycleTrace, Stop, format_cycle_record};

fn sample() -> CycleRecord {
    CycleRecord {
        forward: 7,
        sessions: 2,
        position: 31463,
        rows: 10,
        siblings: 2,
        drafter_steps: 8,
        small_head: true,
        sibling_accepted: true,
        emitted: 2,
        priced_draft_ms: 22.25,
        priced_verify_ms: 152.5,
        expected: 3.5,
        signals: vec![0.5, 0.75],
        estimates: vec![0.625, 0.875],
        margins: vec![0.0, 1.5],
        sibling_estimates: vec![0.96875],
        proposal_signals: vec![0.5, 0.75, 0.25],
        proposal_top: vec![0.5, 0.75, 0.25],
        proposal_margin: vec![0.25, 0.5, 0.0],
        proposal_q: vec![0.5, 0.75, 0.25],
        pending_top: 0.75,
        pending_entropy: 1.5,
        row_acceptance: vec![1.0, 0.5],
        runner_ups: vec![42, -1],
        runner_up_signals: vec![0.25, 0.0],
        runner_up_acceptance: vec![0.0, 1.0],
        observed: vec![true, false],
        sibling_observed: Some(true),
        stop: Some(Stop {
            signal: 0.25,
            estimate: 0.5,
            gain: 20.5,
            cost: 51.25,
        }),
        begin_ns: 100,
        drafted_ns: 200,
        verified_ns: 300,
        end_ns: 400,
        ..CycleRecord::default()
    }
}

fn len(v: &Value, key: &str) -> usize {
    v.find(key).map_or(usize::MAX, Value::len)
}

#[test]
fn format() {
    let line = format_cycle_record(&sample());
    assert!(!line.contains('\n'), "a record spans lines");
    let v = json::parse_str(&line).unwrap();
    assert!(
        v.member_size("forward", 0) == 7 && v.member_size("sessions", 0) == 2,
        "forward/sessions"
    );
    assert!(
        v.member_size("position", 0) == 31463 && v.member_size("rows", 0) == 10,
        "position/rows"
    );
    assert!(
        v.member_size("drafter_steps", 0) == 8 && v.member_size("emitted", 0) == 2,
        "drafter steps/emitted"
    );
    assert!(
        v.find("small_head").is_some_and(Value::is_bool),
        "small head"
    );
    assert!(
        v.member_double("priced_verify_ms", 0.0) == 152.5
            && v.member_double("expected", 0.0) == 3.5,
        "prices"
    );
    assert!(
        v.find("estimates").is_some_and(Value::is_array) && len(&v, "estimates") == 2,
        "estimates"
    );
    assert!(
        len(&v, "runner_ups") == 2
            && len(&v, "runner_up_acceptance") == 2
            && len(&v, "row_acceptance") == 2
            && len(&v, "proposal_signals") == 3
            && len(&v, "proposal_q") == 3
            && v.member_double("pending_entropy", 0.0) == 1.5,
        "row labels and runner-ups"
    );
    assert!(
        len(&v, "observed") == 2 && v.find("sibling_observed").is_some_and(Value::is_bool),
        "observations"
    );
    let stop = v.find("stop").unwrap();
    assert!(
        stop.member_double("gain", 0.0) == 20.5 && stop.member_double("cost", 0.0) == 51.25,
        "stop decision"
    );
    assert_eq!(v.member_size("end_ns", 0), 400, "times");
    let w = json::parse_str(&format_cycle_record(&CycleRecord::default())).unwrap();
    assert!(
        w.find("stop").is_some_and(Value::is_null),
        "no stop is null"
    );
}

#[test]
fn file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gemma4_cycle_trace.jsonl");
    let path = path.to_str().unwrap();
    CycleTrace::new(path).unwrap().write(&sample()).unwrap();
    CycleTrace::new(path)
        .unwrap()
        .write(&CycleRecord::default())
        .unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "the trace does not append");
    assert_eq!(
        json::parse_str(lines[0]).unwrap().member_size("rows", 0),
        10,
        "first record"
    );
    assert!(
        CycleTrace::new("/nonexistent-dir/trace.jsonl").is_err(),
        "an unwritable trace opened"
    );
}
