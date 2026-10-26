#![allow(dead_code)]

//! The fake runner and helpers of `tests/cli/text_generation_scheduler_test.cpp`,
//! shared by the scheduler and server metrics tests.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    ChatRequest, DeviceStatus, Error, GenerationErrorCode, NoPromptContext, Result, Sampler,
    TextDecodeSelection, TextDecodeStep, TextExecutionPlan, TextModelRunner, TextPrefillStep,
    TextRunnerAdvance, TextRunnerCapabilities, TextRunnerDecode, TextRunnerDescriptor,
    TextRunnerResourceClaim, TextRunnerSnapshot, TextRunnerState, Token, default_advance_batch,
    default_decode_batch, default_decode_step,
};
use gufo_serve::backend::GenerationResult;
use gufo_serve::pool::TextRunnerPool;
use gufo_serve::scheduler::{
    RequestMetadata, SchedulerRequest, TextGenerationScheduler, TextPrefillPolicy,
    TextSchedulerPolicy,
};

pub const TEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Prefill,
    Advance,
}

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub kind: EventKind,
    pub label: Token,
    pub count: usize,
}

#[derive(Default)]
pub struct Gate {
    pub events: Vec<Event>,
    pub advance_batches: Vec<Vec<Token>>,
    pub block_advance_label: Option<Token>,
    pub block_prefill_label: Option<Token>,
    pub throw_advance_label: Option<Token>,
    /// Advancing this request panics on the scheduler thread.
    pub panic_advance_label: Option<Token>,
    pub advance_gate_label: Token,
    pub prefill_gate_label: Token,
    pub advance_gate_entered: bool,
    pub prefill_gate_entered: bool,
    pub release_advance: bool,
    pub release_prefill: bool,
}

/// Behaviour fixed before the scheduler starts.
#[derive(Clone)]
pub struct Config {
    pub incremental_prefill: bool,
    pub prefill_capacity: usize,
    pub supports_batched_advance: bool,
    pub final_token_advance_required: bool,
    pub incremental_text_is_exact: bool,
    pub multi_token_decode: bool,
    pub batched_multi_token_decode: bool,
    pub actual_batch_width: usize,
    pub prefix_reuse: bool,
    pub preview_first_token: bool,
    /// The pieces of each request's generated tokens in order, with a stop
    /// after the last; empty means the token numbers.
    pub output_pieces: Vec<&'static str>,
    pub max_context: u32,
    /// Generation stops after this many decoded tokens.
    pub stop_after: Option<usize>,
    /// Selects EOS after this many decoded tokens, unless `ignore_eos`.
    pub eos_after: Option<usize>,
    pub snapshot_callback: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            incremental_prefill: true,
            prefill_capacity: usize::MAX,
            supports_batched_advance: false,
            final_token_advance_required: true,
            incremental_text_is_exact: false,
            multi_token_decode: false,
            batched_multi_token_decode: false,
            actual_batch_width: 4,
            prefix_reuse: true,
            preview_first_token: false,
            output_pieces: Vec::new(),
            max_context: 128,
            stop_after: None,
            eos_after: None,
            snapshot_callback: None,
        }
    }
}

#[derive(Default)]
pub struct FakeControl {
    pub config: Config,
    pub gate: Mutex<Gate>,
    pub condition: Condvar,
    pub invalidations: AtomicUsize,
    pub states_created: AtomicUsize,
    pub decode_calls: AtomicUsize,
    pub batch_preparations: AtomicUsize,
    /// The fake device reports itself lost to probes, or else pending.
    pub device_lost: AtomicBool,
    pub device_pending: AtomicBool,
    pub device_probes: AtomicUsize,
    pub first_request_advance_ns: AtomicI64,
}

impl FakeControl {
    pub fn new(config: Config) -> Arc<Self> {
        Arc::new(Self {
            config,
            ..Self::default()
        })
    }

    pub fn with_gate(config: Config, setup: impl FnOnce(&mut Gate)) -> Arc<Self> {
        let control = Self::new(config);
        setup(&mut control.gate());
        control
    }

    pub fn gate(&self) -> MutexGuard<'_, Gate> {
        self.gate.lock().expect("gate")
    }

    pub fn wait_for_advance(&self, label: Token) {
        let gate = self.gate();
        let (_gate, timeout) = self
            .condition
            .wait_timeout_while(gate, TEST_TIMEOUT, |g| {
                !(g.advance_gate_entered && g.advance_gate_label == label)
            })
            .expect("gate");
        assert!(
            !timeout.timed_out(),
            "timed out waiting for blocked decode advance"
        );
    }

    pub fn wait_for_prefill(&self, label: Token) {
        let gate = self.gate();
        let (_gate, timeout) = self
            .condition
            .wait_timeout_while(gate, TEST_TIMEOUT, |g| {
                !(g.prefill_gate_entered && g.prefill_gate_label == label)
            })
            .expect("gate");
        assert!(
            !timeout.timed_out(),
            "timed out waiting for blocked prefill"
        );
    }

    pub fn release_advance(&self) {
        self.gate().release_advance = true;
        self.condition.notify_all();
    }

    pub fn release_prefill(&self) {
        self.gate().release_prefill = true;
        self.condition.notify_all();
    }

    pub fn record_invalidation(&self) {
        let _gate = self.gate();
        self.invalidations.fetch_add(1, Ordering::Relaxed);
        self.condition.notify_all();
    }

    pub fn wait_for_invalidations(&self, count: usize) {
        let gate = self.gate();
        let (_gate, timeout) = self
            .condition
            .wait_timeout_while(gate, TEST_TIMEOUT, |_| {
                self.invalidations.load(Ordering::Relaxed) < count
            })
            .expect("gate");
        assert!(
            !timeout.timed_out(),
            "timed out waiting for state invalidation"
        );
    }

    pub fn events(&self) -> Vec<Event> {
        self.gate().events.clone()
    }

    pub fn advance_batches(&self) -> Vec<Vec<Token>> {
        self.gate().advance_batches.clone()
    }
}

pub struct FakeState {
    pub control: Arc<FakeControl>,
    pub label: Token,
    pub position: usize,
    pub decode_count: usize,
    pub frontier: Option<Token>,
    pub stop_at_eos: bool,
}

impl TextRunnerState for FakeState {
    fn invalidate(&mut self) {
        self.control.record_invalidation();
        self.label = 0;
        self.position = 0;
        self.decode_count = 0;
        self.frontier = None;
    }

    fn set_stop_at_eos(&mut self, stop_at_eos: bool) -> Result<()> {
        self.stop_at_eos = stop_at_eos;
        Ok(())
    }
}

pub struct SnapshotState {
    pub label: Token,
    pub position: usize,
    pub decode_count: usize,
    pub frontier: Option<Token>,
}

impl TextRunnerSnapshot for SnapshotState {
    fn payload_bytes(&self) -> usize {
        40
    }
}

#[derive(Clone)]
pub struct FakeSampler;

impl Sampler for FakeSampler {
    fn accept(&mut self, _token: Token) {}
}

pub struct FakeRunner {
    pub control: Arc<FakeControl>,
}

impl TextModelRunner for FakeRunner {
    type State = FakeState;
    type Snapshot = SnapshotState;
    type Sampler = FakeSampler;
    type PromptContext = NoPromptContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        let c = &self.control.config;
        let snapshots = c.snapshot_callback.is_some();
        TextRunnerDescriptor {
            model_id: "scheduler-fake".into(),
            state_abi: "scheduler-fake-v1".into(),
            max_context: c.max_context,
            capabilities: TextRunnerCapabilities {
                incremental_prefill: c.incremental_prefill,
                snapshot: snapshots,
                fork: snapshots,
                final_token_advance_required: c.final_token_advance_required,
                incremental_text_is_exact: c.incremental_text_is_exact,
                multi_token_decode: c.multi_token_decode,
                batched_multi_token_decode: c.batched_multi_token_decode,
                batched_multi_token_decode_max_width: 4,
                prefix_reuse: c.prefix_reuse,
                in_pass_checkpoint: false,
            },
            persistence: None,
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        TextRunnerResourceClaim {
            resident_weights_bytes: Some(0),
            state_capacity_bytes: Some(8 * 64),
            per_request_state_bytes: Some(64),
            temporary_scratch_bytes: Some(0),
            retained_snapshot_capacity_bytes: Some(
                if self.control.config.snapshot_callback.is_some() {
                    4096
                } else {
                    0
                },
            ),
            retained_snapshot_ceiling_bytes: None,
            requires_device_runtime_lock: true,
        }
    }

    fn supported_plans(&self) -> Vec<TextExecutionPlan> {
        let mut plans = vec![TextExecutionPlan::SERIAL];
        if self.control.config.supports_batched_advance {
            plans.push(TextExecutionPlan::batched(2));
            plans.push(TextExecutionPlan::batched(4));
        }
        plans
    }

    fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        Ok(vec![text.len() as Token])
    }

    fn render_and_tokenize(&self, _request: &ChatRequest) -> Result<Option<Vec<Token>>> {
        Ok(None)
    }

    fn decode(&self, tokens: &[Token]) -> Result<Vec<u8>> {
        self.control.decode_calls.fetch_add(1, Ordering::Relaxed);
        Ok(tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
            .into_bytes())
    }

    fn create_state(&self) -> Result<FakeState> {
        self.control.states_created.fetch_add(1, Ordering::Relaxed);
        Ok(FakeState {
            control: self.control.clone(),
            label: 0,
            position: 0,
            decode_count: 0,
            frontier: None,
            stop_at_eos: true,
        })
    }

    fn create_sampler(&self, _config: &SamplingConfig, _prompt: &[Token]) -> Result<FakeSampler> {
        Ok(FakeSampler)
    }

    fn prepare_batch_execution(&self, _state: &mut FakeState) -> Result<()> {
        self.control
            .batch_preparations
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn prefill(
        &self,
        state: &mut FakeState,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
    ) -> Result<TextPrefillStep> {
        if offset != state.position || prompt.is_empty() {
            return Err(Error::logic("invalid scheduler fake prefill"));
        }
        state.label = prompt[0];
        let consumed = max_input_tokens
            .min(prompt.len() - offset)
            .min(self.control.config.prefill_capacity);
        {
            let mut gate = self.control.gate();
            gate.events.push(Event {
                kind: EventKind::Prefill,
                label: state.label,
                count: consumed,
            });
            if gate.block_prefill_label == Some(state.label) && !gate.prefill_gate_entered {
                gate.prefill_gate_entered = true;
                gate.prefill_gate_label = state.label;
                self.control.condition.notify_all();
                let (_gate, timeout) = self
                    .control
                    .condition
                    .wait_timeout_while(gate, TEST_TIMEOUT, |g| !g.release_prefill)
                    .expect("gate");
                if timeout.timed_out() {
                    return Err(Error::runtime("prefill gate timed out"));
                }
            }
        }
        state.position += consumed;
        let ready = state.position == prompt.len();
        if ready {
            state.frontier = Some(state.label * 100);
        }
        Ok(TextPrefillStep {
            consumed_tokens: consumed,
            decode_ready: ready,
        })
    }

    fn select_next(
        &self,
        state: &mut FakeState,
        _sampler: &mut FakeSampler,
    ) -> Result<TextDecodeSelection> {
        let frontier = state
            .frontier
            .ok_or_else(|| Error::logic("scheduler fake has no frontier"))?;
        let config = &self.control.config;
        if config
            .stop_after
            .is_some_and(|count| state.decode_count >= count)
            || (state.stop_at_eos
                && config
                    .eos_after
                    .is_some_and(|count| state.decode_count >= count))
        {
            return Ok(TextDecodeSelection {
                stop: true,
                token: 0,
                piece: Vec::new(),
            });
        }
        let pieces = &self.control.config.output_pieces;
        if !pieces.is_empty() {
            let index = (frontier - state.label * 100) as usize;
            return Ok(match pieces.get(index) {
                Some(piece) => TextDecodeSelection {
                    stop: false,
                    token: frontier,
                    piece: piece.as_bytes().to_vec(),
                },
                None => TextDecodeSelection {
                    stop: true,
                    token: 0,
                    piece: Vec::new(),
                },
            });
        }
        Ok(TextDecodeSelection {
            stop: false,
            token: frontier,
            piece: frontier.to_string().into_bytes(),
        })
    }

    fn preview_first_token(
        &self,
        state: &mut FakeState,
        sampler: &mut FakeSampler,
    ) -> Result<Option<TextDecodeSelection>> {
        if !self.control.config.preview_first_token {
            return Ok(None);
        }
        self.select_next(state, sampler).map(Some)
    }

    fn poll_device(&self) -> DeviceStatus {
        {
            let _gate = self.control.gate();
            self.control.device_probes.fetch_add(1, Ordering::Relaxed);
        }
        self.control.condition.notify_all();
        if self.control.device_lost.load(Ordering::Relaxed) {
            DeviceStatus::Lost
        } else if self.control.device_pending.load(Ordering::Relaxed) {
            DeviceStatus::Pending
        } else {
            DeviceStatus::Usable
        }
    }

    fn snapshot_payload_bytes(&self, _state: &FakeState) -> Result<usize> {
        Ok(40)
    }

    fn snapshot(&self, state: &FakeState) -> Result<SnapshotState> {
        if let Some(callback) = &self.control.config.snapshot_callback {
            callback();
        }
        Ok(SnapshotState {
            label: state.label,
            position: state.position,
            decode_count: state.decode_count,
            frontier: state.frontier,
        })
    }

    fn restore_or_fork(&self, state: &mut FakeState, snapshot: &SnapshotState) -> Result<()> {
        state.label = snapshot.label;
        state.position = snapshot.position;
        state.decode_count = snapshot.decode_count;
        state.frontier = snapshot.frontier;
        Ok(())
    }

    fn advance(&self, state: &mut FakeState, token: Token) -> Result<()> {
        let started = Instant::now();
        if state.frontier != Some(token) {
            return Err(Error::logic("scheduler fake frontier mismatch"));
        }
        {
            let mut gate = self.control.gate();
            if gate.block_advance_label == Some(state.label)
                && state.decode_count == 0
                && !gate.advance_gate_entered
            {
                gate.advance_gate_entered = true;
                gate.advance_gate_label = state.label;
                self.control.condition.notify_all();
                let (released, timeout) = self
                    .control
                    .condition
                    .wait_timeout_while(gate, TEST_TIMEOUT, |g| !g.release_advance)
                    .expect("gate");
                gate = released;
                if timeout.timed_out() {
                    return Err(Error::runtime("advance gate timed out"));
                }
            }
            if gate.throw_advance_label == Some(state.label) {
                return Err(Error::runtime("injected scheduler runner failure"));
            }
            if gate.panic_advance_label == Some(state.label) {
                // Released first: the gate stays usable after the panic.
                drop(gate);
                panic!("injected scheduler runner panic");
            }
            gate.events.push(Event {
                kind: EventKind::Advance,
                label: state.label,
                count: 1,
            });
        }
        state.position += 1;
        state.decode_count += 1;
        state.frontier = Some(token + 1);
        if state.label == 1 {
            self.control
                .first_request_advance_ns
                .fetch_add(started.elapsed().as_nanos() as i64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn decode_step(
        &self,
        state: &mut FakeState,
        max_tokens: usize,
        sampler: &mut FakeSampler,
    ) -> Result<TextDecodeStep> {
        if !self.control.config.multi_token_decode {
            return default_decode_step(self, state, max_tokens, sampler);
        }
        if state.frontier.is_none() {
            return Err(Error::logic("scheduler fake has no multi-token frontier"));
        }
        let count = max_tokens.min(3);
        let mut step = TextDecodeStep::default();
        for _ in 0..count {
            let selection = self.select_next(state, sampler)?;
            if selection.stop {
                step.stop = true;
                break;
            }
            self.advance(state, selection.token)?;
            step.selections.push(selection);
        }
        step.draft_rounds = 1;
        step.draft_tokens = count + 1;
        step.draft_accepted_tokens = count;
        Ok(step)
    }

    fn decode_batch(
        &self,
        decodes: &mut [TextRunnerDecode<'_, FakeState, FakeSampler>],
    ) -> Result<Vec<Result<TextDecodeStep>>> {
        let width = self.control.config.actual_batch_width.min(decodes.len());
        let mut steps = default_decode_batch(self, decodes);
        for step in steps.iter_mut().flatten() {
            step.execution_plan = if width > 1 {
                TextExecutionPlan::batched(width)
            } else {
                TextExecutionPlan::SERIAL
            };
        }
        Ok(steps)
    }

    fn advance_batch(
        &self,
        advances: &mut [TextRunnerAdvance<'_, FakeState>],
    ) -> Result<Vec<Result<()>>> {
        let labels = advances.iter().map(|a| a.state.label).collect();
        self.control.gate().advance_batches.push(labels);
        Ok(default_advance_batch(self, advances))
    }

    fn checkpoint_position(&self, state: &FakeState) -> Result<usize> {
        Ok(state.position)
    }
}

pub type Scheduler = TextGenerationScheduler<FakeRunner>;

pub fn make_scheduler(
    control: &Arc<FakeControl>,
    capacity: usize,
    prefill_policy: TextPrefillPolicy,
    scheduler_policy: TextSchedulerPolicy,
) -> Scheduler {
    let runner = Arc::new(FakeRunner {
        control: control.clone(),
    });
    let pool = Arc::new(TextRunnerPool::new(runner, capacity, None).expect("pool"));
    TextGenerationScheduler::new(pool, prefill_policy, scheduler_policy).expect("scheduler")
}

pub fn scheduler(control: &Arc<FakeControl>, capacity: usize) -> Scheduler {
    make_scheduler(
        control,
        capacity,
        TextPrefillPolicy::default(),
        TextSchedulerPolicy::default(),
    )
}

pub fn chunked(control: &Arc<FakeControl>, capacity: usize, chunk: usize) -> Scheduler {
    make_scheduler(
        control,
        capacity,
        TextPrefillPolicy {
            decode_active_tokens: chunk,
        },
        TextSchedulerPolicy::default(),
    )
}

pub fn submit(scheduler: &Scheduler, prompt: &[Token], max_tokens: usize) -> SchedulerRequest {
    scheduler
        .submit_simple(prompt.to_vec(), max_tokens, 0.0)
        .expect("submit")
}

pub fn submit_streaming(
    scheduler: &Scheduler,
    prompt: &[Token],
    max_tokens: usize,
) -> SchedulerRequest {
    scheduler
        .submit(
            prompt.to_vec(),
            max_tokens,
            &SamplingConfig::default(),
            None,
            true,
            RequestMetadata::default(),
        )
        .expect("submit")
}

pub fn submit_for(
    scheduler: &Scheduler,
    prompt: &[Token],
    max_tokens: usize,
    client: &str,
) -> Result<SchedulerRequest> {
    scheduler.submit(
        prompt.to_vec(),
        max_tokens,
        &SamplingConfig::default(),
        None,
        false,
        RequestMetadata {
            client_id: client.into(),
            ..RequestMetadata::default()
        },
    )
}

pub fn wait(request: &mut SchedulerRequest) -> GenerationResult {
    request.wait(None).expect("wait")
}

pub fn event_index(events: &[Event], kind: EventKind, label: Token, occurrence: usize) -> usize {
    events
        .iter()
        .enumerate()
        .filter(|(_, e)| e.kind == kind && e.label == label)
        .nth(occurrence)
        .map_or(events.len(), |(i, _)| i)
}

pub fn expected_tokens(label: Token, count: usize) -> Vec<Token> {
    (0..count as Token).map(|i| label * 100 + i).collect()
}

pub fn generation_code(error: &Error) -> Option<GenerationErrorCode> {
    error.generation_code()
}

/// `shared` prompt tokens common to every caller, then `tail` distinct ones.
pub fn shared_prompt(shared: usize, tail_start: Token, tail: usize) -> Vec<Token> {
    std::iter::once(1)
        .chain((1..shared as Token).map(|index| 1000 + index))
        .chain((0..tail as Token).map(|index| tail_start + index))
        .collect()
}

pub fn shared_prefix_control(snapshots: bool) -> Arc<FakeControl> {
    FakeControl::with_gate(
        Config {
            max_context: 4096,
            prefill_capacity: 256,
            snapshot_callback: snapshots.then(|| Arc::new(|| {}) as Arc<dyn Fn() + Send + Sync>),
            ..Config::default()
        },
        |gate| gate.block_prefill_label = Some(1),
    )
}
