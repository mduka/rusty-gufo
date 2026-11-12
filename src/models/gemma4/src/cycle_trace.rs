//! The per-cycle draft trace (`cycle_trace.cpp`): one JSON object per MTP
//! verification cycle, appended to a file (`--draft-trace`).

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use gufo_core::json::Value;

/// A sparse distribution: ids with their values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sparse {
    pub ids: Vec<i32>,
    pub values: Vec<f32>,
}

/// The proposal the calibrated policy declined: its signal, estimate,
/// expected gain and cost (the chain stops when gain < cost).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stop {
    pub signal: f32,
    pub estimate: f32,
    pub gain: f32,
    pub cost: f32,
}

/// One MTP verification cycle of one session: what the draft policy priced
/// and expected, what verification accepted and how long each stage took
/// (`CycleRecord`). Times are steady-clock nanoseconds; a batch's sessions
/// share the drafting and verification stages.
#[derive(Debug, Clone, PartialEq)]
pub struct CycleRecord {
    /// The forward this cycle verified in, counted per model, and the
    /// sessions it verified.
    pub forward: u64,
    pub sessions: u32,
    pub position: u32,
    pub sampled: bool,
    /// Verified rows: the pending token, MTP drafts, copies, then siblings.
    pub rows: u32,
    pub copied: u32,
    pub siblings: u32,
    /// Drafter steps run, including a last proposal the policy declined, and
    /// the most the chain could run.
    pub drafter_steps: u32,
    pub step_cap: u32,
    /// The chain drafted on the small head over frequent tokens.
    pub small_head: bool,
    /// Accepted drafts and copies, whether a sibling was accepted, and the
    /// tokens the cycle emitted.
    pub accepted: u32,
    pub sibling_accepted: bool,
    pub emitted: u32,
    /// Calibrated policy: drafter and verification prices of the cycle and
    /// the tokens it expected (zero under other policies).
    pub priced_draft_ms: f32,
    pub priced_verify_ms: f32,
    pub expected: f32,
    /// Calibrated policy: each verified MTP draft's signal, its acceptance
    /// estimate and its decision's gain minus cost (zero for the first
    /// `min_drafts`), and each sibling's hit estimate.
    pub signals: Vec<f32>,
    pub estimates: Vec<f32>,
    pub margins: Vec<f32>,
    pub sibling_estimates: Vec<f32>,
    /// Every drafter proposal's signal under any policy, a declined last one
    /// included.
    pub proposal_signals: Vec<f32>,
    /// Per proposal, the drafter's top-1 share and top-1 minus top-2 share
    /// among its top-64 logits, and the proposal's own probability (sampled:
    /// q of the drawn token; greedy: the top-1 share).
    pub proposal_top: Vec<f32>,
    pub proposal_margin: Vec<f32>,
    pub proposal_q: Vec<f32>,
    /// Sampled: the top probability and entropy (nats) of the request
    /// sampler's distribution the next pending token was drawn from (-1 when
    /// not recorded).
    pub pending_top: f32,
    pub pending_entropy: f32,
    /// Per verified chain row past the pending one (MTP drafts, then
    /// copies): the probability that verification accepts it given the rows
    /// before it accepted. Greedy: 1 when the target's argmax is the row's
    /// token, else 0; sampled: min(1, p/q) with p the request sampler's
    /// distribution at the cycle's start and q the proposal's probability
    /// (1 for a copy).
    pub row_acceptance: Vec<f32>,
    /// Per MTP draft, the drafter's alternative beside it (-1: none), its
    /// signal, and the probability that verification accepts it as the
    /// sibling once the draft is rejected (a hit adds one token).
    pub runner_ups: Vec<i32>,
    pub runner_up_signals: Vec<f32>,
    pub runner_up_acceptance: Vec<f32>,
    /// Sampled cycles, for offline proposal studies: the request sampler's
    /// temperature, top-k and top-p; per drafter proposal its top-64
    /// candidates (ids and raw logits); per verified MTP draft the proposal q
    /// it was drawn from and the target distribution p of its row.
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub draft_candidates: Vec<Sparse>,
    pub proposal_dist: Vec<Sparse>,
    pub target_dist: Vec<Sparse>,
    /// What the calibrated policy learned: each judged MTP draft's outcome
    /// (`signals[i]` with `observed[i]`) and the sibling's, when one was
    /// judged.
    pub observed: Vec<bool>,
    pub sibling_observed: Option<bool>,
    pub stop: Option<Stop>,
    /// The cycle starts drafting.
    pub begin_ns: i64,
    /// Its rows are complete.
    pub drafted_ns: i64,
    /// The forward's results are on the host.
    pub verified_ns: i64,
    /// Its tokens are emitted.
    pub end_ns: i64,
}

impl Default for CycleRecord {
    fn default() -> Self {
        Self {
            forward: 0,
            sessions: 1,
            position: 0,
            sampled: false,
            rows: 0,
            copied: 0,
            siblings: 0,
            drafter_steps: 0,
            step_cap: 0,
            small_head: false,
            accepted: 0,
            sibling_accepted: false,
            emitted: 0,
            priced_draft_ms: 0.0,
            priced_verify_ms: 0.0,
            expected: 0.0,
            signals: Vec::new(),
            estimates: Vec::new(),
            margins: Vec::new(),
            sibling_estimates: Vec::new(),
            proposal_signals: Vec::new(),
            proposal_top: Vec::new(),
            proposal_margin: Vec::new(),
            proposal_q: Vec::new(),
            pending_top: -1.0,
            pending_entropy: -1.0,
            row_acceptance: Vec::new(),
            runner_ups: Vec::new(),
            runner_up_signals: Vec::new(),
            runner_up_acceptance: Vec::new(),
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            draft_candidates: Vec::new(),
            proposal_dist: Vec::new(),
            target_dist: Vec::new(),
            observed: Vec::new(),
            sibling_observed: None,
            stop: None,
            begin_ns: 0,
            drafted_ns: 0,
            verified_ns: 0,
            end_ns: 0,
        }
    }
}

fn floats(values: &[f32]) -> Value {
    Value::Array(values.iter().map(|&v| Value::from(v)).collect())
}

/// `[{"ids": [...], "v": [...]}, ...]`
fn distributions(rows: &[Sparse]) -> Value {
    Value::Array(
        rows.iter()
            .map(|row| {
                Value::object()
                    .with(
                        "ids",
                        Value::Array(row.ids.iter().map(|&id| Value::from(id)).collect()),
                    )
                    .with("v", floats(&row.values))
            })
            .collect(),
    )
}

/// The record as one line of JSON, without a newline (`FormatCycleRecord`).
pub fn format_cycle_record(r: &CycleRecord) -> String {
    let mut v = Value::object()
        .with("forward", r.forward)
        .with("sessions", r.sessions)
        .with("position", r.position)
        .with("sampled", r.sampled)
        .with("rows", r.rows)
        .with("copied", r.copied)
        .with("siblings", r.siblings)
        .with("drafter_steps", r.drafter_steps)
        .with("step_cap", r.step_cap)
        .with("small_head", r.small_head)
        .with("accepted", r.accepted)
        .with("sibling_accepted", r.sibling_accepted)
        .with("emitted", r.emitted)
        .with("priced_draft_ms", r.priced_draft_ms)
        .with("priced_verify_ms", r.priced_verify_ms)
        .with("expected", r.expected)
        .with("signals", floats(&r.signals))
        .with("estimates", floats(&r.estimates))
        .with("margins", floats(&r.margins))
        .with("sibling_estimates", floats(&r.sibling_estimates))
        .with("proposal_signals", floats(&r.proposal_signals))
        .with("proposal_top", floats(&r.proposal_top))
        .with("proposal_margin", floats(&r.proposal_margin))
        .with("proposal_q", floats(&r.proposal_q))
        .with("pending_top", r.pending_top)
        .with("pending_entropy", r.pending_entropy)
        .with("row_acceptance", floats(&r.row_acceptance))
        .with(
            "runner_ups",
            Value::Array(r.runner_ups.iter().map(|&t| Value::from(t)).collect()),
        )
        .with("runner_up_signals", floats(&r.runner_up_signals))
        .with("runner_up_acceptance", floats(&r.runner_up_acceptance))
        .with(
            "observed",
            Value::Array(r.observed.iter().map(|&o| Value::from(o)).collect()),
        )
        .with(
            "sibling_observed",
            r.sibling_observed.map_or(Value::Null, Value::from),
        );
    let stop = match r.stop {
        Some(stop) => Value::object()
            .with("signal", stop.signal)
            .with("estimate", stop.estimate)
            .with("gain", stop.gain)
            .with("cost", stop.cost),
        None => Value::Null,
    };
    v.set("stop", stop);
    if r.sampled {
        v.set("temperature", r.temperature);
        v.set("top_k", r.top_k);
        v.set("top_p", r.top_p);
        v.set("draft_candidates", distributions(&r.draft_candidates));
        v.set("proposal_dist", distributions(&r.proposal_dist));
        v.set("target_dist", distributions(&r.target_dist));
    }
    v.set("begin_ns", r.begin_ns);
    v.set("drafted_ns", r.drafted_ns);
    v.set("verified_ns", r.verified_ns);
    v.set("end_ns", r.end_ns);
    // Every number is finite unless a caller recorded a non-finite float;
    // that record then reads as null rather than failing the cycle.
    v.dump().unwrap_or_else(|_| "null".into())
}

/// Appends one JSON line per cycle to a file (`CycleTrace`).
#[derive(Debug)]
pub struct CycleTrace {
    out: Mutex<File>,
}

impl CycleTrace {
    /// Opens `path` for appending.
    pub fn new(path: &str) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("cannot open draft trace {path}: {e}"))?;
        Ok(Self {
            out: Mutex::new(file),
        })
    }

    /// Writes and flushes one record; safe from concurrent sessions.
    pub fn write(&self, record: &CycleRecord) -> std::io::Result<()> {
        let mut line = format_cycle_record(record);
        line.push('\n');
        let mut out = self.out.lock().unwrap_or_else(|e| e.into_inner());
        out.write_all(line.as_bytes())?;
        out.flush()
    }

    /// Monotonic nanoseconds since the first call in this process
    /// (`Now`, steady-clock nanoseconds in C++).
    pub fn now() -> i64 {
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_nanos() as i64
    }
}
