//! Shared device batches of concurrent callers (`batch_scheduler.cpp`).

use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::audio_processor::SequenceAudio;
use crate::config::IMAGE_TOKEN_ID;
use crate::image_processor::ImagePatches;

/// A cancellation poll that another caller's thread may run.
pub type CancelCheck = Arc<dyn Fn() -> bool + Send + Sync>;

/// An image and the offset of its placeholder tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceImage {
    pub offset: usize,
    pub patches: ImagePatches,
    /// Video frames share the vision tower but use their own placeholders.
    pub token_id: u32,
}

impl Default for SequenceImage {
    fn default() -> Self {
        Self {
            offset: 0,
            patches: ImagePatches::default(),
            token_id: IMAGE_TOKEN_ID,
        }
    }
}

/// One sequence to embed: its tokens, and the images and audio clips whose
/// features replace their placeholder tokens.
#[derive(Clone, Default)]
pub struct EncoderSequence {
    pub tokens: Vec<u32>,
    pub images: Vec<SequenceImage>,
    pub audio: Vec<SequenceAudio>,
    /// Polled by the audio tower between layers.
    pub is_cancelled: Option<CancelCheck>,
}

impl EncoderSequence {
    pub fn text(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            ..Self::default()
        }
    }

    fn has_media(&self) -> bool {
        !self.images.is_empty() || !self.audio.is_empty()
    }
}

/// How a call ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// One vector per sequence, in input order.
    Ok(Vec<Vec<f32>>),
    Cancelled,
    /// The runner's error, or why the call could not be scheduled.
    Failed(String),
}

/// Embeds one batch into one vector per sequence, in order, or fails with a
/// message. Batches never overlap; a panic fails the batch.
pub type Runner<'r> = &'r (dyn Fn(&[&EncoderSequence]) -> Result<Vec<Vec<f32>>, String> + Sync);

/// One `run` call, touched only under the scheduler lock until it has an
/// outcome.
struct Caller {
    sequences: Arc<[EncoderSequence]>,
    is_cancelled: Option<CancelCheck>,
    /// Some sequence holds an image or audio, so the caller gets a batch of
    /// its own.
    has_media: bool,
    /// First sequence not yet taken into a batch.
    next: usize,
    remaining_tokens: usize,
    embeddings: Vec<Vec<f32>>,
    outcome: Option<Outcome>,
}

#[derive(Default)]
struct State {
    callers: HashMap<u64, Caller>,
    queue: VecDeque<u64>,
    running: bool,
    next_id: u64,
}

/// The sequences `[first, last)` of one caller within a batch.
struct Slice {
    caller: u64,
    first: usize,
    last: usize,
}

/// Packs the token sequences of concurrent callers into shared device
/// batches. Callers queue in arrival order. When the device is free, the
/// caller at the front runs one batch on its own thread, with its runner,
/// for itself and the callers behind it, then hands the device on. A lone
/// caller runs at once. (The C++ scheduler takes one runner at
/// construction; every caller of one encoder passes the same here.)
///
/// A batch takes whole callers, oldest first, while their remaining
/// sequences fit in `max_batch_tokens`. The first caller that does not fit
/// ends the batch, so no caller passes another, and a caller that fits in
/// one batch never spans two. Only a caller too large for one batch is
/// split, and only at the start of a batch. It then rejoins the queue
/// behind the callers that arrived during that batch, so they wait for at
/// most one of its batches. A caller with an image or audio runs in a batch
/// of its own, so a text caller never waits for another caller's modality
/// encoder pass. At the front, it takes the batch alone, and it may still be
/// split by tokens. Behind other callers, it ends the batch like a caller
/// that does not fit.
pub struct BatchScheduler {
    max_batch_tokens: usize,
    state: Mutex<State>,
    changed: Condvar,
}

impl BatchScheduler {
    pub fn new(max_batch_tokens: usize) -> Self {
        Self {
            max_batch_tokens,
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Blocks until every sequence is embedded. A sequence longer than
    /// `max_batch_tokens` fails the call. `is_cancelled` is polled before
    /// each batch that would take this call's sequences, on the thread that
    /// builds the batch, which may belong to another call, with the
    /// scheduler's lock held, so it must not block. A cancelled call sends no
    /// further sequences to the runner. A failed batch fails every call with
    /// sequences in it.
    pub fn run(
        &self,
        sequences: Arc<[EncoderSequence]>,
        is_cancelled: Option<CancelCheck>,
        runner: Runner<'_>,
    ) -> Outcome {
        if sequences.is_empty() {
            return Outcome::Ok(Vec::new());
        }
        let mut tokens = 0;
        let mut has_media = false;
        for sequence in sequences.iter() {
            has_media = has_media || sequence.has_media();
            if sequence.tokens.len() > self.max_batch_tokens {
                return Outcome::Failed("a sequence is longer than the batch token limit".into());
            }
            tokens += sequence.tokens.len();
        }
        let count = sequences.len();
        let mut state = self.lock();
        let id = state.next_id;
        state.next_id += 1;
        state.callers.insert(
            id,
            Caller {
                sequences,
                is_cancelled,
                has_media,
                next: 0,
                remaining_tokens: tokens,
                embeddings: Vec::with_capacity(count),
                outcome: None,
            },
        );
        state.queue.push_back(id);
        loop {
            state = self
                .changed
                .wait_while(state, |s| {
                    s.callers[&id].outcome.is_none() && (s.running || s.queue.front() != Some(&id))
                })
                .unwrap_or_else(|e| e.into_inner());
            if state.callers[&id].outcome.is_some() {
                break;
            }
            state = self.run_next_batch(state, runner);
        }
        let caller = state.callers.remove(&id).expect("registered caller");
        match caller.outcome.expect("finished caller") {
            Outcome::Ok(_) => Outcome::Ok(caller.embeddings),
            other => other,
        }
    }

    /// Calls waiting for their next batch; tests use it to wait for callers
    /// to queue.
    pub fn queued(&self) -> usize {
        self.lock().queue.len()
    }

    /// Runs the next batch with the device free; returns with the lock
    /// held again.
    fn run_next_batch<'s>(
        &'s self,
        mut state: MutexGuard<'s, State>,
        runner: Runner<'_>,
    ) -> MutexGuard<'s, State> {
        state.running = true;
        // The head is the calling thread's own caller.
        let own = *state.queue.front().expect("own caller at the front");
        let mut slices: Vec<Slice> = Vec::new();
        let mut taken: Vec<(Arc<[EncoderSequence]>, usize, usize)> = Vec::new();
        let mut tokens = 0usize;
        let mut batch_sequences = 0usize;
        while let Some(&id) = state.queue.front() {
            let max = self.max_batch_tokens;
            let caller = state.callers.get_mut(&id).expect("queued caller");
            if caller.has_media && batch_sequences != 0 {
                break;
            }
            let mut last = caller.next;
            if caller.remaining_tokens <= max - tokens {
                last = caller.sequences.len();
            } else if tokens == 0 {
                let mut count = 0;
                while count + caller.sequences[last].tokens.len() <= max {
                    count += caller.sequences[last].tokens.len();
                    last += 1;
                }
            } else {
                break;
            }
            state.queue.pop_front();
            let caller = state.callers.get_mut(&id).expect("queued caller");
            if caller
                .is_cancelled
                .as_ref()
                .is_some_and(|cancelled| cancelled())
            {
                caller.outcome = Some(Outcome::Cancelled);
                if id == own {
                    // Leave the batch to the next caller's thread, not this
                    // one.
                    state.running = false;
                    self.changed.notify_all();
                    return state;
                }
                continue;
            }
            slices.push(Slice {
                caller: id,
                first: caller.next,
                last,
            });
            for sequence in &caller.sequences[caller.next..last] {
                tokens += sequence.tokens.len();
                caller.remaining_tokens -= sequence.tokens.len();
            }
            batch_sequences += last - caller.next;
            taken.push((Arc::clone(&caller.sequences), caller.next, last));
            caller.next = last;
            if last < caller.sequences.len() || caller.has_media {
                break;
            }
        }
        // Wakes callers cancelled above.
        self.changed.notify_all();
        drop(state);

        let sequences: Vec<&EncoderSequence> = taken
            .iter()
            .flat_map(|(sequences, first, last)| &sequences[*first..*last])
            .collect();
        // A panic fails the batch with its message, like a thrown exception.
        let result =
            catch_unwind(AssertUnwindSafe(|| runner(&sequences))).unwrap_or_else(|payload| {
                Err(payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| {
                        "embedding device work failed with an unknown exception".into()
                    }))
            });
        let result = result.and_then(|embeddings| {
            if embeddings.len() == sequences.len() {
                Ok(embeddings)
            } else {
                Err("embedding batch returned the wrong number of vectors".into())
            }
        });
        drop(sequences);

        let mut state = self.lock();
        match result {
            Err(error) => {
                for slice in slices {
                    let caller = state
                        .callers
                        .get_mut(&slice.caller)
                        .expect("batched caller");
                    caller.outcome = Some(Outcome::Failed(error.clone()));
                }
            }
            Ok(embeddings) => {
                let mut vectors = embeddings.into_iter();
                for slice in slices {
                    let caller = state
                        .callers
                        .get_mut(&slice.caller)
                        .expect("batched caller");
                    caller
                        .embeddings
                        .extend(vectors.by_ref().take(slice.last - slice.first));
                    if caller.next == caller.sequences.len() {
                        caller.outcome = Some(Outcome::Ok(Vec::new()));
                    } else {
                        state.queue.push_back(slice.caller);
                    }
                }
            }
        }
        state.running = false;
        self.changed.notify_all();
        state
    }
}
