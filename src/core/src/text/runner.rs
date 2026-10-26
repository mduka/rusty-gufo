//! The model adapter the serving layer drives (`TextModelRunner`,
//! `src/cli/serve/text_model_runner.hpp`): the runner pool, the continuation
//! caches and the scheduler see only this trait, never a model's internals.
//!
//! Prefill and advance are the model's work units. `select_next` exposes an
//! already computed frontier token separately, so streaming does not wait for
//! the next forward pass. States are model-private and mutable; snapshots are
//! immutable, model-private continuations. Serving code accounts for their
//! bytes but never inspects them.

use std::path::Path;
use std::sync::Arc;

use crate::json_constraint::{ConstraintVocabulary, ToolFormat};
use crate::sampling::SamplingConfig;

use super::chat::ChatRequest;
use super::error::{Error, Result};

pub type Token = u32;

/// A request-scoped cancellation check. It may run on the scheduler thread
/// inside model loops, so it must be cheap and must not block.
pub type CancellationCheck = Arc<dyn Fn() -> bool + Send + Sync>;

/// The input identity of the prefixes ending at or before `token_count`
/// (`ContinuationInputPrefix`). Ordered boundaries let a checkpoint taken
/// before a new image keep the identity it was computed with; the complete
/// identity applies after the last boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContinuationInputPrefix {
    pub token_count: usize,
    pub identity: Vec<u8>,
}

/// The input identity of the first `token_count` tokens
/// (`PrefixInputIdentity`).
pub fn prefix_input_identity<'a>(
    complete: &'a [u8],
    prefixes: &'a [ContinuationInputPrefix],
    token_count: usize,
) -> &'a [u8] {
    prefixes
        .iter()
        .find(|prefix| token_count <= prefix.token_count)
        .map_or(complete, |prefix| &prefix.identity)
}

/// Immutable model-owned input that travels with one request, e.g. image
/// embeddings. Its identity supplements the token IDs as a cache key.
pub trait PromptContext: Send + Sync + 'static {
    fn cache_identity(&self) -> &[u8];

    /// The identities of prefixes that end before later inputs, in token
    /// order; empty when the complete identity covers every prefix.
    fn cache_prefixes(&self) -> &[ContinuationInputPrefix] {
        &[]
    }

    /// The identity of the first `token_count` tokens.
    fn cache_identity_for(&self, token_count: usize) -> &[u8] {
        prefix_input_identity(self.cache_identity(), self.cache_prefixes(), token_count)
    }
}

/// The prompt context of runners without model-owned prompt inputs.
#[derive(Debug, Default)]
pub struct NoPromptContext;

impl PromptContext for NoPromptContext {
    fn cache_identity(&self) -> &[u8] {
        &[]
    }
}

#[derive(Debug)]
pub struct TextPreparedPrompt<C> {
    pub tokens: Vec<Token>,
    pub context: Option<Arc<C>>,
    /// The snapshot boundary before mutable assistant framing; zero uses the
    /// complete prompt.
    pub cache_prefix_tokens: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum TextExecutionPlanKind {
    #[default]
    Serial,
    Batched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextExecutionPlan {
    pub kind: TextExecutionPlanKind,
    pub physical_width: usize,
}

impl Default for TextExecutionPlan {
    fn default() -> Self {
        Self::SERIAL
    }
}

impl TextExecutionPlan {
    pub const SERIAL: Self = Self {
        kind: TextExecutionPlanKind::Serial,
        physical_width: 1,
    };

    pub fn batched(physical_width: usize) -> Self {
        Self {
            kind: TextExecutionPlanKind::Batched,
            physical_width,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextRunnerCapabilities {
    pub incremental_prefill: bool,
    pub snapshot: bool,
    pub fork: bool,
    pub final_token_advance_required: bool,
    pub incremental_text_is_exact: bool,
    pub multi_token_decode: bool,
    pub batched_multi_token_decode: bool,
    /// Zero means no physical width limit.
    pub batched_multi_token_decode_max_width: usize,
    pub prefix_reuse: bool,
    /// A prefill pass can capture a stable boundary near its end and keep
    /// going ([`TextModelRunner::prefill_through`]).
    pub in_pass_checkpoint: bool,
}

impl Default for TextRunnerCapabilities {
    fn default() -> Self {
        Self {
            incremental_prefill: false,
            snapshot: false,
            fork: false,
            final_token_advance_required: true,
            incremental_text_is_exact: false,
            multi_token_decode: false,
            batched_multi_token_decode: false,
            batched_multi_token_decode_max_width: 0,
            prefix_reuse: true,
            in_pass_checkpoint: false,
        }
    }
}

/// Model-owned compatibility identity for restart-safe snapshots. The bytes
/// are canonical and opaque to serving code; they cover every model,
/// tokenizer, template, layout, precision, context-policy and adapter
/// property that can change what a restored continuation means.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextRunnerPersistenceDescriptor {
    pub compatibility_identity: Vec<u8>,
    pub payload_version: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextRunnerDescriptor {
    pub model_id: String,
    pub state_abi: String,
    pub max_context: u32,
    pub capabilities: TextRunnerCapabilities,
    pub persistence: Option<TextRunnerPersistenceDescriptor>,
}

/// Byte claims made before state allocation; `None` means "not yet
/// measurable".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TextRunnerResourceClaim {
    pub resident_weights_bytes: Option<usize>,
    pub state_capacity_bytes: Option<usize>,
    pub per_request_state_bytes: Option<usize>,
    pub temporary_scratch_bytes: Option<usize>,
    /// Bytes currently available for immutable retained snapshots; the pool
    /// queries it again after creating every request state.
    pub retained_snapshot_capacity_bytes: Option<usize>,
    /// The most an explicit RAM-cache limit may claim; `None` makes the
    /// automatic `retained_snapshot_capacity_bytes` the ceiling too.
    pub retained_snapshot_ceiling_bytes: Option<usize>,
    pub requires_device_runtime_lock: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TextRunnerMeasuredResources {
    pub per_request_state_bytes: Option<usize>,
    pub temporary_scratch_bytes: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TextPrefillStep {
    pub consumed_tokens: usize,
    pub decode_ready: bool,
}

/// A [`TextModelRunner::prefill_through`] step and the checkpoint it
/// captured on its way.
pub struct TextPrefillThrough<S> {
    pub step: TextPrefillStep,
    pub checkpoint: S,
    /// The capture's share of the step.
    pub checkpoint_ms: f64,
}

/// A [`TextModelRunner::poll_device`] result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceStatus {
    Usable,
    /// The check may be queued behind long kernels; poll again.
    Pending,
    /// A hard device error: the context is lost for good.
    Lost,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextDecodeSelection {
    pub stop: bool,
    pub token: Token,
    /// The token's text bytes; a piece may end inside a UTF-8 sequence
    /// (byte fallback), which the transport repairs across pieces.
    pub piece: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextDecodeStep {
    pub selections: Vec<TextDecodeSelection>,
    /// Speculative verification rounds actually executed.
    pub draft_rounds: usize,
    pub draft_tokens: usize,
    pub draft_accepted_tokens: usize,
    /// Of `draft_tokens`: proposals copied from the context (prompt
    /// lookup).
    pub lookup_tokens: usize,
    pub lookup_accepted_tokens: usize,
    pub stop: bool,
    /// The execution actually used for this request, including model-owned
    /// subgroup dispatch; an advertised maximum is not evidence of batching.
    pub execution_plan: TextExecutionPlan,
}

/// `TextGenerationBackend::InitialOutputState`: whether generated text starts
/// inside a reasoning block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum InitialOutputState {
    #[default]
    Auto,
    Reasoning,
    Content,
}

/// Model-private mutable continuation state.
pub trait TextRunnerState: Send + 'static {
    /// Discards any partially or fully computed continuation.
    fn invalidate(&mut self);

    /// Installs (or with `None` clears) a request-scoped cancellation check
    /// for model calls that can yield internally.
    fn set_cancellation_check(&mut self, check: Option<CancellationCheck>) {
        let _ = check;
    }

    /// Whether the request stops at the model's stop tokens; false
    /// (`ignore_eos`) generates through them up to the token limit. Runners
    /// that cannot continue past a stop token refuse false.
    fn set_stop_at_eos(&mut self, stop_at_eos: bool) -> Result<()> {
        if stop_at_eos {
            Ok(())
        } else {
            Err(Error::invalid("this model does not support ignore_eos"))
        }
    }

    /// Actual request-owned allocations, when the model can measure them.
    fn measured_resources(&self) -> TextRunnerMeasuredResources {
        TextRunnerMeasuredResources::default()
    }

    /// Names the storage this state's borrowed snapshots point into
    /// ([`TextRunnerSnapshot::source`]); `None` while it lends none.
    fn snapshot_owner(&self) -> Option<u64> {
        None
    }
}

/// Immutable model-private continuation payload.
pub trait TextRunnerSnapshot: Send + Sync + 'static {
    fn payload_bytes(&self) -> usize;

    /// The [`TextRunnerState::snapshot_owner`] of the state whose rows this
    /// snapshot borrows: restoring into that state copies the least. An
    /// allocation hint only; every state must still restore it.
    fn source(&self) -> Option<u64> {
        None
    }
}

/// Request-owned sampling state (penalty history, RNG, pending draw). The
/// pool calls `accept` only for committed tokens and clones it for a
/// first-token preview.
pub trait Sampler: Clone + Send + 'static {
    fn accept(&mut self, token: Token);
}

pub struct TextRunnerDecode<'a, S, P> {
    pub state: &'a mut S,
    pub max_tokens: usize,
    pub sampler: &'a mut P,
}

pub struct TextRunnerAdvance<'a, S> {
    pub state: &'a mut S,
    pub token: Token,
}

/// The coarse text-model adapter used by caching, scheduling and HTTP.
///
/// The reasoning tags of a token budget (`ReasoningTags`): `forced_end` is
/// emitted once the budget is spent, and any of `ends` closes reasoning
/// early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningTags {
    pub start: &'static str,
    pub forced_end: &'static str,
    pub ends: &'static [&'static str],
}

impl Default for ReasoningTags {
    fn default() -> Self {
        Self {
            start: "<think>",
            forced_end: "</think>",
            ends: &["</think>"],
        }
    }
}

/// Methods taking `&self` may run concurrently: tokenization and prompt
/// preparation on HTTP threads, `snapshot` on a capture worker while other
/// states execute, and persistent serialisation on the disk writer. Work on
/// states is serialised by the scheduler.
pub trait TextModelRunner: Send + Sync + 'static {
    type State: TextRunnerState;
    type Snapshot: TextRunnerSnapshot;
    type Sampler: Sampler;
    type PromptContext: PromptContext;

    fn descriptor(&self) -> TextRunnerDescriptor;
    fn resource_claim(&self) -> TextRunnerResourceClaim;
    fn supported_plans(&self) -> Vec<TextExecutionPlan>;

    fn tokenize(&self, text: &str) -> Result<Vec<Token>>;
    /// Control text such as `</think>` as the chat template renders it
    /// (`TokenizeControl`).
    fn tokenize_control(&self, text: &str) -> Result<Vec<Token>> {
        self.tokenize(text)
    }
    /// The reasoning tags of a token budget as llama.cpp's chat format for
    /// the model defines them (`ReasoningBudgetTags`).
    fn reasoning_budget_tags(&self) -> ReasoningTags {
        ReasoningTags::default()
    }
    fn render_and_tokenize(&self, request: &ChatRequest) -> Result<Option<Vec<Token>>>;

    fn prepare_prompt(
        &self,
        request: &ChatRequest,
    ) -> Result<Option<TextPreparedPrompt<Self::PromptContext>>> {
        if request.messages.iter().any(|m| !m.images.is_empty()) {
            return Err(Error::invalid("model does not support image input"));
        }
        Ok(self
            .render_and_tokenize(request)?
            .map(|tokens| TextPreparedPrompt {
                tokens,
                context: None,
                cache_prefix_tokens: 0,
            }))
    }

    /// The prompt tokens `request` would use (`CountPromptTokens`), without
    /// admission to an execution context or inference; `None` when the
    /// template does not render. Image runners override it with their
    /// model's visual token expansion.
    fn count_prompt_tokens(&self, request: &ChatRequest) -> Result<Option<usize>> {
        if request.messages.iter().any(|m| !m.images.is_empty()) {
            return Err(Error::invalid("model does not support image input"));
        }
        Ok(self
            .render_and_tokenize(request)?
            .map(|tokens| tokens.len()))
    }

    /// Called before cache restoration, prefill or selection, including a
    /// complete prefix hit. Replaces, never inherits, the previous request's
    /// context.
    fn set_prompt_context(
        &self,
        state: &mut Self::State,
        context: Option<&Arc<Self::PromptContext>>,
    ) -> Result<()> {
        let _ = state;
        match context {
            Some(_) => Err(Error::invalid("model does not support this prompt context")),
            None => Ok(()),
        }
    }

    fn initial_output_state(&self, request: &ChatRequest) -> InitialOutputState {
        let _ = request;
        InitialOutputState::Auto
    }

    /// The tool-call syntax the model's chat template teaches; constrained
    /// calls use it unless a schema needs the JSON fallback.
    fn tool_format(&self) -> ToolFormat {
        ToolFormat::Json
    }

    /// Whether chat requests may carry images: a vision encoder is loaded.
    fn supports_images(&self) -> bool {
        false
    }

    /// Polls whether the execution device still runs work, without waiting:
    /// a submitted check that has not completed is `Pending`. Called by the
    /// scheduler between work units only.
    fn poll_device(&self) -> DeviceStatus {
        DeviceStatus::Usable
    }

    /// Each token's bytes and whether it stops generation, for structured
    /// output; built only when a request first needs a grammar
    /// (`BuildConstraintVocabulary`).
    fn build_constraint_vocabulary(&self) -> Result<ConstraintVocabulary> {
        Err(Error::invalid("model does not support structured output"))
    }

    /// The bytes of `tokens`, which need not be valid UTF-8.
    fn decode(&self, tokens: &[Token]) -> Result<Vec<u8>>;

    fn create_state(&self) -> Result<Self::State>;

    /// The request's sampler over `config`, with `prompt` as penalty history.
    fn create_sampler(&self, config: &SamplingConfig, prompt: &[Token]) -> Result<Self::Sampler>;

    /// An optional first-token preview that does not advance the state; the
    /// caller passes a copy of the sampler and normal decoding must produce
    /// the same token. Serving publishes it before capturing the prompt
    /// snapshot.
    fn preview_first_token(
        &self,
        state: &mut Self::State,
        sampler: &mut Self::Sampler,
    ) -> Result<Option<TextDecodeSelection>> {
        let _ = (state, sampler);
        Ok(None)
    }

    /// Reconciles model-private metadata after an exact live or restored
    /// prefix is leased.
    fn prepare_prefix_reuse(&self, state: &mut Self::State, prefix: &[Token]) -> Result<()> {
        let _ = (state, prefix);
        Ok(())
    }

    /// Tells a state that its next execution will be part of a multi-request
    /// batch; request-local speculative work the batch cannot use may go.
    fn prepare_batch_execution(&self, state: &mut Self::State) -> Result<()> {
        let _ = state;
        Ok(())
    }

    /// Processes at most `max_input_tokens` of `prompt` from `offset`,
    /// yielding after one model-owned chunk even when granted more.
    fn prefill(
        &self,
        state: &mut Self::State,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
    ) -> Result<TextPrefillStep>;

    /// The payload bytes of the checkpoint at `boundary` that
    /// [`prefill_through`](Self::prefill_through) with the same arguments
    /// would capture, or `None` when that step cannot both capture it and
    /// continue past it. Admission reserves the bytes before the step.
    fn prefill_checkpoint_bytes(
        &self,
        state: &Self::State,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
        boundary: usize,
    ) -> Result<Option<usize>> {
        let _ = (state, prompt, offset, max_input_tokens, boundary);
        Ok(None)
    }

    /// [`prefill`](Self::prefill) that also captures the continuation at
    /// `boundary` (between `offset` and the step's end) in the same pass;
    /// the checkpoint restores like a snapshot of a state prefilled to
    /// exactly `boundary`.
    fn prefill_through(
        &self,
        state: &mut Self::State,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
        boundary: usize,
    ) -> Result<TextPrefillThrough<Self::Snapshot>> {
        let _ = (state, prompt, offset, max_input_tokens, boundary);
        Err(Error::logic(
            "text runner does not support in-pass checkpoints",
        ))
    }

    fn select_next(
        &self,
        state: &mut Self::State,
        sampler: &mut Self::Sampler,
    ) -> Result<TextDecodeSelection>;

    fn advance(&self, state: &mut Self::State, token: Token) -> Result<()>;

    fn decode_step(
        &self,
        state: &mut Self::State,
        max_tokens: usize,
        sampler: &mut Self::Sampler,
    ) -> Result<TextDecodeStep> {
        default_decode_step(self, state, max_tokens, sampler)
    }

    /// One step for each request; the outer error fails the whole batch, the
    /// inner ones single requests.
    fn decode_batch(
        &self,
        decodes: &mut [TextRunnerDecode<'_, Self::State, Self::Sampler>],
    ) -> Result<Vec<Result<TextDecodeStep>>> {
        Ok(default_decode_batch(self, decodes))
    }

    /// Advances each state by its token; the outer error fails the whole
    /// batch, the inner ones single requests.
    fn advance_batch(
        &self,
        advances: &mut [TextRunnerAdvance<'_, Self::State>],
    ) -> Result<Vec<Result<()>>> {
        Ok(default_advance_batch(self, advances))
    }

    fn checkpoint_position(&self, state: &Self::State) -> Result<usize>;

    /// Retains a safe executed frontier when cancellation interrupts a
    /// completed speculative block; runs with cancellation checks cleared.
    fn prepare_cancellation(&self, state: &mut Self::State) -> Result<()> {
        let _ = state;
        Ok(())
    }

    fn snapshot_payload_bytes(&self, state: &Self::State) -> Result<usize> {
        let _ = state;
        Err(Error::logic("text runner does not support snapshot sizing"))
    }

    /// Called before a prefill step that ends where the caller captures a
    /// snapshot of `state`, so the runner can allocate the capture's storage
    /// while the step runs and [`snapshot`](Self::snapshot) only copies.
    /// Storage left unused is the runner's to drop.
    fn prepare_snapshot(&self, state: &mut Self::State) -> Result<()> {
        let _ = state;
        Ok(())
    }

    /// Captures an immutable continuation at `checkpoint_position(state)`.
    /// May run on a capture worker while the state is frozen: it must not
    /// mutate shared execution scratch, other states may execute meanwhile.
    fn snapshot(&self, state: &Self::State) -> Result<Self::Snapshot> {
        let _ = state;
        Err(Error::logic("text runner does not support snapshots"))
    }

    /// [`snapshot`](Self::snapshot) for a disk export: complete before it
    /// returns, so the export adds no device copy while the state resumes.
    fn snapshot_for_persistence(&self, state: &Self::State) -> Result<Self::Snapshot> {
        self.snapshot(state)
    }

    fn restore_or_fork(&self, state: &mut Self::State, snapshot: &Self::Snapshot) -> Result<()> {
        let _ = (state, snapshot);
        Err(Error::logic(
            "text runner does not support snapshot restore/fork",
        ))
    }

    /// The exact serialised size of a persistent snapshot, which may differ
    /// from the in-memory layout (e.g. compact device state).
    fn persistent_snapshot_payload_bytes(&self, snapshot: &Self::Snapshot) -> Result<usize> {
        let _ = snapshot;
        Err(Error::logic(
            "text runner does not support persistent snapshots",
        ))
    }

    /// Serialises into exactly sized staging; returns the bytes written.
    fn serialize_persistent_snapshot(
        &self,
        snapshot: &Self::Snapshot,
        destination: &mut [u8],
    ) -> Result<usize> {
        let _ = (snapshot, destination);
        Err(Error::logic(
            "text runner does not support persistent snapshots",
        ))
    }

    /// Streams the serialised payload; reads only immutable snapshot storage
    /// and may run on the persistence worker.
    fn stream_persistent_snapshot(
        &self,
        snapshot: &Self::Snapshot,
        sink: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let mut payload = vec![0; self.persistent_snapshot_payload_bytes(snapshot)?];
        if self.serialize_persistent_snapshot(snapshot, &mut payload)? != payload.len() {
            return Err(Error::runtime("persistent snapshot byte count mismatch"));
        }
        sink(&payload)
    }

    fn restore_persistent_snapshot(&self, state: &mut Self::State, payload: &[u8]) -> Result<()> {
        let _ = (state, payload);
        Err(Error::logic(
            "text runner does not support persistent snapshots",
        ))
    }
}

/// Select, then advance (`TextModelRunner::DecodeStep`).
pub fn default_decode_step<R: TextModelRunner + ?Sized>(
    runner: &R,
    state: &mut R::State,
    max_tokens: usize,
    sampler: &mut R::Sampler,
) -> Result<TextDecodeStep> {
    if max_tokens == 0 {
        return Err(Error::invalid(
            "text runner decode step budget must be at least one token",
        ));
    }
    let selection = runner.select_next(state, sampler)?;
    if selection.stop {
        return Ok(TextDecodeStep {
            stop: true,
            ..TextDecodeStep::default()
        });
    }
    runner.advance(state, selection.token)?;
    Ok(TextDecodeStep {
        selections: vec![selection],
        ..TextDecodeStep::default()
    })
}

/// One `decode_step` per request (`TextModelRunner::DecodeBatch`).
pub fn default_decode_batch<R: TextModelRunner + ?Sized>(
    runner: &R,
    decodes: &mut [TextRunnerDecode<'_, R::State, R::Sampler>],
) -> Vec<Result<TextDecodeStep>> {
    decodes
        .iter_mut()
        .map(|d| runner.decode_step(d.state, d.max_tokens, d.sampler))
        .collect()
}

/// One `advance` per request (`TextModelRunner::AdvanceBatch`).
pub fn default_advance_batch<R: TextModelRunner + ?Sized>(
    runner: &R,
    advances: &mut [TextRunnerAdvance<'_, R::State>],
) -> Vec<Result<()>> {
    advances
        .iter_mut()
        .map(|a| runner.advance(a.state, a.token))
        .collect()
}

/// The value of a `/proc/meminfo` field in bytes.
fn meminfo_field(meminfo: &str, name: &str) -> Option<u64> {
    meminfo
        .lines()
        .find_map(|l| l.strip_prefix(name))
        .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
        .map(|kib| kib * 1024)
}

/// `MemAvailable` minus `CmaFree` from `/proc/meminfo` text, in bytes: free CMA pages count as available but only hold movable pages,
/// not GPU allocations, and the kernel's KHO scratch area can make that
/// several GiB. `None` without `MemAvailable`.
pub fn meminfo_available_bytes(meminfo: &str) -> Option<u64> {
    let available = meminfo_field(meminfo, "MemAvailable:")?;
    Some(available.saturating_sub(meminfo_field(meminfo, "CmaFree:").unwrap_or(0)))
}

/// `host_available` bounded by every cgroup v2 memory limit from the
/// process's cgroup (named in `membership`, the `/proc/self/cgroup` text) up
/// to `root` (upstream `CgroupAvailableBytes`). A child cgroup may
/// say "max" beneath a limited ancestor. `memory.current` includes page cache
/// the kernel reclaims under pressure: inactive file pages count as free, the
/// active ones may still be in use, and dirty and writeback pages cannot be
/// dropped until written, so both stay charged. A malformed `memory.stat` is
/// ignored.
pub fn cgroup_available_bytes(root: &Path, membership: &str, host_available: u64) -> u64 {
    let mut available = host_available;
    let Some(relative) = membership.lines().find_map(|l| l.strip_prefix("0::/")) else {
        return available;
    };
    if relative.split('/').any(|part| part == "..") {
        return available;
    }
    let mut path = root.join(relative);
    loop {
        let read = |name: &str| {
            std::fs::read_to_string(path.join(name))
                .ok()?
                .trim()
                .parse::<u64>()
                .ok()
        };
        if let (Some(limit), Some(mut used)) = (read("memory.max"), read("memory.current")) {
            if let Some(reclaimable) = std::fs::read_to_string(path.join("memory.stat"))
                .ok()
                .and_then(|stat| reclaimable_file_bytes(&stat))
            {
                used -= used.min(reclaimable);
            }
            available = available.min(limit.saturating_sub(used));
        }
        if path == root || !path.pop() {
            break;
        }
    }
    available
}

/// `inactive_file` minus dirty and writeback pages from a cgroup's
/// `memory.stat`; `None` without `inactive_file` or with a malformed value.
fn reclaimable_file_bytes(stat: &str) -> Option<u64> {
    let mut words = stat.split_whitespace();
    let (mut inactive, mut unreclaimable) = (None, 0u64);
    while let Some(key) = words.next() {
        let value = words.next()?.parse::<u64>().ok()?;
        match key {
            "inactive_file" => inactive = Some(value),
            "file_dirty" | "file_writeback" => unreclaimable = unreclaimable.saturating_add(value),
            _ => {}
        }
    }
    Some(inactive?.saturating_sub(unreclaimable))
}

/// Host RAM still available, after cgroup limits; unified-memory device
/// allocations draw from the same RAM, so HIP's device capacity is not used.
fn host_available_bytes() -> u64 {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let Some(available) =
        meminfo_available_bytes(&meminfo).or_else(|| meminfo_field(&meminfo, "MemFree:"))
    else {
        return 0;
    };
    let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    cgroup_available_bytes(Path::new("/sys/fs/cgroup"), &membership, available)
}

/// Host memory for snapshots: half of what the host and every enclosing
/// cgroup still have (`HostSnapshotBudgetBytes`). Retained snapshots and
/// disk staging draw on it; nothing is reserved.
pub fn host_snapshot_budget_bytes() -> usize {
    usize::try_from(host_available_bytes() / 2).unwrap_or(usize::MAX)
}

/// RAM an explicit retained-snapshot limit always leaves to the OS and other
/// processes.
pub const HOST_SNAPSHOT_HEADROOM_BYTES: u64 = 4 << 30;

/// The most an explicit `--cache-ram-bytes` may claim: the available host
/// RAM minus [`HOST_SNAPSHOT_HEADROOM_BYTES`].
pub fn host_snapshot_ceiling_bytes() -> usize {
    usize::try_from(host_available_bytes().saturating_sub(HOST_SNAPSHOT_HEADROOM_BYTES))
        .unwrap_or(usize::MAX)
}
