//! How the batch scheduler packs concurrent calls into shared batches: FIFO
//! order, whole callers, turns for callers larger than one batch, media
//! callers in batches of their own, cancellation and shared failures, on a
//! fake device (`batch_scheduler_test.cpp`):
//! cargo test -p gufo-embedding-gemma2 --test batch_scheduler

#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use gufo_embedding_gemma2::batch_scheduler::{
    BatchScheduler, CancelCheck, EncoderSequence, Outcome, SequenceImage,
};

const MAX_TOKENS: usize = 100;
// Sequence IDs with a meaning to the fake device.
const HOLD: u32 = 1; // The batch waits for release().
const FAIL: u32 = 2; // The batch fails.
const PANIC: u32 = 3; // The batch panics.

type Batches = Vec<Vec<u32>>;

/// `length` copies of `id`, so the device can name each sequence.
fn tokens(length: usize, id: u32) -> EncoderSequence {
    EncoderSequence::text(vec![id; length])
}

fn embedding_of(sequence: &EncoderSequence) -> Vec<f32> {
    vec![
        sequence.tokens[0] as f32,
        sequence.tokens.len() as f32,
        sequence.images.len() as f32,
    ]
}

/// Records each batch as the IDs of its sequences.
#[derive(Default)]
struct FakeDevice {
    state: Mutex<(Batches, bool, bool)>,
    changed: Condvar,
    running: AtomicBool,
}

impl FakeDevice {
    fn run(&self, sequences: &[&EncoderSequence]) -> Result<Vec<Vec<f32>>, String> {
        let ids: Vec<u32> = sequences.iter().map(|s| s.tokens[0]).collect();
        let total: usize = sequences.iter().map(|s| s.tokens.len()).sum();
        assert!(
            !ids.is_empty() && total <= MAX_TOKENS,
            "a batch is non-empty and fits"
        );
        assert!(
            !self.running.swap(true, Ordering::SeqCst),
            "batches never overlap"
        );
        {
            let mut state = self.state.lock().unwrap();
            state.0.push(ids.clone());
            if ids.contains(&HOLD) {
                state.1 = true;
                self.changed.notify_all();
                let _held = self.changed.wait_while(state, |s| !s.2).unwrap();
            }
        }
        // Long enough for concurrent callers to queue behind the batch.
        thread::sleep(Duration::from_micros(50));
        self.running.store(false, Ordering::SeqCst);
        if ids.contains(&FAIL) {
            return Err("device failed".into());
        }
        if ids.contains(&PANIC) {
            panic!("device threw");
        }
        Ok(sequences.iter().map(|s| embedding_of(s)).collect())
    }

    fn wait_held(&self) {
        let _held = self
            .changed
            .wait_while(self.state.lock().unwrap(), |s| !s.1)
            .unwrap();
    }

    fn release(&self) {
        self.state.lock().unwrap().2 = true;
        self.changed.notify_all();
    }

    fn batches(&self) -> Batches {
        self.state.lock().unwrap().0.clone()
    }
}

/// A fake device, a scheduler on it, and the calls started against it.
struct Rig {
    device: Arc<FakeDevice>,
    scheduler: Arc<BatchScheduler>,
    calls: Vec<(Vec<EncoderSequence>, JoinHandle<Outcome>)>,
}

impl Rig {
    fn new() -> Self {
        Self {
            device: Arc::default(),
            scheduler: Arc::new(BatchScheduler::new(MAX_TOKENS)),
            calls: Vec::new(),
        }
    }

    /// Runs a call on this thread.
    fn run(&self, sequences: Vec<EncoderSequence>, cancelled: Option<CancelCheck>) -> Outcome {
        let device = self.device.clone();
        self.scheduler
            .run(sequences.into(), cancelled, &|batch| device.run(batch))
    }

    fn start(&mut self, sequences: Vec<EncoderSequence>, cancelled: Option<CancelCheck>) {
        let (device, scheduler) = (self.device.clone(), self.scheduler.clone());
        let input = sequences.clone();
        let handle = thread::spawn(move || {
            scheduler.run(input.into(), cancelled, &|batch| device.run(batch))
        });
        self.calls.push((sequences, handle));
    }

    /// Starts a call whose first batch holds the device, and returns once it
    /// does.
    fn hold(&mut self, sequences: Vec<EncoderSequence>, cancelled: Option<CancelCheck>) {
        self.start(sequences, cancelled);
        self.device.wait_held();
    }

    /// Starts a call behind the held one, and returns once it is queued.
    fn queue(&mut self, sequences: Vec<EncoderSequence>, cancelled: Option<CancelCheck>) {
        self.start(sequences, cancelled);
        while self.scheduler.queued() != self.calls.len() - 1 {
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Releases the held batch and returns every call's outcome.
    fn drain(&mut self) -> Vec<(Vec<EncoderSequence>, Outcome)> {
        self.device.release();
        self.calls
            .drain(..)
            .map(|(sequences, handle)| (sequences, handle.join().unwrap()))
            .collect()
    }
}

fn expect_embedded(sequences: &[EncoderSequence], outcome: &Outcome, what: &str) {
    let expected: Vec<Vec<f32>> = sequences.iter().map(embedding_of).collect();
    assert_eq!(outcome, &Outcome::Ok(expected), "{what}");
}

fn always() -> Option<CancelCheck> {
    Some(Arc::new(|| true))
}

fn flag(gone: &Arc<AtomicBool>) -> Option<CancelCheck> {
    let gone = gone.clone();
    Some(Arc::new(move || gone.load(Ordering::SeqCst)))
}

#[test]
fn a_lone_caller_runs_at_once() {
    let rig = Rig::new();
    let fits = vec![tokens(30, 10), tokens(40, 11)];
    expect_embedded(
        &fits,
        &rig.run(fits.clone(), None),
        "a lone caller runs at once",
    );
    let spans = vec![tokens(50, 12), tokens(40, 13), tokens(30, 14)];
    expect_embedded(
        &spans,
        &rig.run(spans.clone(), None),
        "larger than one batch",
    );
    assert_eq!(rig.run(Vec::new(), None), Outcome::Ok(Vec::new()));
    assert_eq!(rig.device.batches(), [vec![10, 11], vec![12, 13], vec![14]]);
}

#[test]
fn queued_callers_share_a_batch() {
    let mut rig = Rig::new();
    rig.hold(vec![tokens(10, HOLD)], None);
    let mut expected = Vec::new();
    for i in 0..8 {
        let first = 100 + 2 * i;
        rig.queue(vec![tokens(5, first), tokens(5, first + 1)], None);
        expected.extend([first, first + 1]);
    }
    for (sequences, outcome) in rig.drain() {
        expect_embedded(&sequences, &outcome, "own vectors in input order");
    }
    assert_eq!(rig.device.batches(), [vec![HOLD], expected]);
}

#[test]
fn a_caller_that_fits_is_not_split() {
    let mut rig = Rig::new();
    rig.hold(vec![tokens(10, HOLD)], None);
    rig.queue(vec![tokens(30, 10), tokens(30, 11)], None);
    // Fits in an empty batch but not in the 40 tokens the first caller
    // leaves.
    rig.queue(vec![tokens(25, 12), tokens(25, 13)], None);
    rig.queue(vec![tokens(5, 14)], None);
    for (sequences, outcome) in rig.drain() {
        expect_embedded(&sequences, &outcome, "own vectors");
    }
    assert_eq!(
        rig.device.batches(),
        [vec![HOLD], vec![10, 11], vec![12, 13, 14]]
    );
}

#[test]
fn a_large_caller_takes_turns() {
    for (long, expected) in [
        (
            vec![tokens(80, HOLD), tokens(80, 10), tokens(80, 11)],
            vec![vec![HOLD], vec![20], vec![10], vec![11]],
        ),
        (
            vec![tokens(60, HOLD), tokens(60, 10), tokens(30, 11)],
            vec![vec![HOLD], vec![20, 10, 11]],
        ),
    ] {
        let mut rig = Rig::new();
        rig.hold(long, None);
        rig.queue(vec![tokens(5, 20)], None);
        for (sequences, outcome) in rig.drain() {
            expect_embedded(&sequences, &outcome, "own vectors");
        }
        assert_eq!(rig.device.batches(), expected);
    }
}

#[test]
fn cancelled_callers_never_reach_the_device() {
    let rig = Rig::new();
    assert_eq!(rig.run(vec![tokens(5, 10)], always()), Outcome::Cancelled);
    assert!(rig.device.batches().is_empty());

    // Cancelled while queued at the front; the caller behind still runs.
    let mut rig = Rig::new();
    let gone = Arc::new(AtomicBool::new(false));
    rig.hold(vec![tokens(10, HOLD)], None);
    rig.queue(vec![tokens(5, 10)], flag(&gone));
    rig.queue(vec![tokens(5, 11)], None);
    gone.store(true, Ordering::SeqCst);
    let outcomes = rig.drain();
    assert_eq!(outcomes[1].1, Outcome::Cancelled);
    expect_embedded(&outcomes[2].0, &outcomes[2].1, "the caller behind runs");
    assert_eq!(rig.device.batches(), [vec![HOLD], vec![11]]);

    // Cancelled in the middle of a forming batch.
    let mut rig = Rig::new();
    rig.hold(vec![tokens(10, HOLD)], None);
    rig.queue(vec![tokens(5, 10)], None);
    rig.queue(vec![tokens(5, 11)], always());
    rig.queue(vec![tokens(5, 12)], None);
    let outcomes = rig.drain();
    assert_eq!(outcomes[2].1, Outcome::Cancelled);
    assert_eq!(rig.device.batches(), [vec![HOLD], vec![10, 12]]);

    // Cancelled during the first of two batches: the second never runs.
    let mut rig = Rig::new();
    let gone = Arc::new(AtomicBool::new(false));
    rig.hold(vec![tokens(60, HOLD), tokens(60, 10)], flag(&gone));
    rig.queue(vec![tokens(5, 11)], None);
    gone.store(true, Ordering::SeqCst);
    let outcomes = rig.drain();
    assert_eq!(outcomes[0].1, Outcome::Cancelled);
    assert_eq!(rig.device.batches(), [vec![HOLD], vec![11]]);

    // Cancelled during its last batch: the batch finishes.
    let mut rig = Rig::new();
    let gone = Arc::new(AtomicBool::new(false));
    rig.hold(vec![tokens(10, HOLD)], flag(&gone));
    gone.store(true, Ordering::SeqCst);
    let outcomes = rig.drain();
    expect_embedded(&outcomes[0].0, &outcomes[0].1, "a running batch finishes");
}

#[test]
fn a_failed_batch_fails_every_caller_in_it() {
    let mut rig = Rig::new();
    rig.hold(vec![tokens(10, HOLD)], None);
    rig.queue(vec![tokens(5, 10)], None);
    rig.queue(vec![tokens(5, FAIL)], None);
    rig.queue(vec![tokens(5, 11)], None);
    let outcomes = rig.drain();
    expect_embedded(
        &outcomes[0].0,
        &outcomes[0].1,
        "the batch before the failure",
    );
    for (_, outcome) in &outcomes[1..] {
        assert_eq!(outcome, &Outcome::Failed("device failed".into()));
    }
    assert_eq!(rig.device.batches(), [vec![HOLD], vec![10, FAIL, 11]]);

    let rig = Rig::new();
    assert_eq!(
        rig.run(vec![tokens(60, FAIL), tokens(60, 10)], None),
        Outcome::Failed("device failed".into())
    );
    assert_eq!(
        rig.run(vec![tokens(5, PANIC)], None),
        Outcome::Failed("device threw".into())
    );
    let after = vec![tokens(5, 11)];
    expect_embedded(
        &after,
        &rig.run(after.clone(), None),
        "a caller after a failure runs",
    );
    assert_eq!(rig.device.batches(), [vec![FAIL], vec![PANIC], vec![11]]);

    let rig = Rig::new();
    assert_eq!(
        rig.run(vec![tokens(60, 10), tokens(60, FAIL)], None),
        Outcome::Failed("device failed".into()),
        "a caller whose second batch fails drops its first batch's vectors"
    );
    assert_eq!(rig.device.batches(), [vec![10], vec![FAIL]]);
}

#[test]
fn a_sequence_longer_than_a_batch_fails_the_call() {
    let rig = Rig::new();
    assert!(matches!(
        rig.run(vec![tokens(5, 10), tokens(MAX_TOKENS + 1, 11)], None),
        Outcome::Failed(_)
    ));
    assert!(rig.device.batches().is_empty());
    let after = vec![tokens(MAX_TOKENS, 12)];
    expect_embedded(
        &after,
        &rig.run(after.clone(), None),
        "exactly one batch runs",
    );
}

#[test]
fn a_media_caller_gets_its_own_batch() {
    let with_image = |length: usize, id: u32| {
        let mut sequence = tokens(length, id);
        sequence.images.push(SequenceImage::default());
        sequence
    };
    let mut rig = Rig::new();
    rig.hold(vec![tokens(10, HOLD)], None);
    rig.queue(vec![tokens(5, 10)], None);
    rig.queue(vec![with_image(5, 11), tokens(5, 12)], None);
    rig.queue(vec![tokens(5, 13)], None);
    for (sequences, outcome) in rig.drain() {
        expect_embedded(&sequences, &outcome, "own vectors");
    }
    assert_eq!(
        rig.device.batches(),
        [vec![HOLD], vec![10], vec![11, 12], vec![13]],
        "a media caller ends the batch behind others and takes the next alone"
    );
}
