//! One resident Gemma 4 model and its sessions (`engine.cpp`): prompt sync
//! with prefix reuse, images and lookahead; one-token evaluation; decoding
//! with MTP drafting, prompt lookup and siblings; batched decoding; and
//! snapshots that share position blocks.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use gufo_core::prompt_lookup::PromptLookup;
use gufo_core::sampling::{SamplerState, TokenLogprobs};
use gufo_hip::{DeviceTensor, MappedFile, Stream};

use crate::chat_template;
use crate::config::Config;
use crate::cycle_trace::CycleTrace;
use crate::device_model::DeviceModel;
use crate::draft_policy::{DraftCalibration, DraftCalibrationScope, DraftPolicy};
use crate::draft_vocab::{DRAFT_TABLE_ROWS, DraftVocab, draft_head_order};
pub use crate::executor::LayerTapSink;
use crate::executor::{Executor, ImageRows, KvCache, MAX_DRAFT_TOKENS, SPLIT_ROWS, hip};
use crate::provider_options::PrefillWeights;
pub use crate::snapshot::SessionSnapshot;
use crate::snapshot::{Held, SnapshotMemory};
use crate::tokenizer::{TokenId, Tokenizer};
use crate::weights::{DraftWeights, GgufShard, ModelWeights};

/// Draft tokens per cycle when the caller sets none (`kDefaultDraftTokens`).
pub const DEFAULT_DRAFT_TOKENS: u32 = 15;
/// Most prompt lookahead rows one prefill forward evaluates past its
/// frontier (`kLookaheadRows`).
pub const LOOKAHEAD_ROWS: usize = 64;

/// `ModelOptions`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelOptions {
    /// Largest context any session may use.
    pub max_context: u32,
    /// Tokens per prefill forward; sets the sliding-window ring size.
    pub prefill_chunk: u32,
    /// Rows of one forward that may request logits.
    pub max_logit_rows: u32,
    /// Optional `gemma4-assistant` MTP drafter; `None` leaves speculation off.
    pub mtp_model_path: Option<PathBuf>,
    /// Draft tokens per cycle (at most the executor's maximum).
    pub draft_tokens: u32,
    /// How many of them a cycle verifies.
    pub draft_policy: DraftPolicy,
    /// Drafts every cycle verifies before a policy may stop the chain.
    pub min_draft_tokens: u32,
    /// Where the calibrated policy learns.
    pub draft_calibration: DraftCalibrationScope,
    /// Appends one JSON line per verification cycle here (`CycleRecord`).
    pub draft_trace_path: Option<PathBuf>,
    /// Prefill copies of the dense projections (memory for prefill speed).
    pub prefill_weights: PrefillWeights,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            max_context: 4096,
            prefill_chunk: 2048,
            max_logit_rows: 64,
            mtp_model_path: None,
            draft_tokens: DEFAULT_DRAFT_TOKENS,
            draft_policy: DraftPolicy::Calibrated,
            min_draft_tokens: 1,
            draft_calibration: DraftCalibrationScope::Shared,
            draft_trace_path: None,
            prefill_weights: PrefillWeights::Off,
        }
    }
}

/// One image of a prompt: its soft-token rows, and an identity that tells it
/// apart from any other image behind the same tokens (`ImageSpan`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageSpan {
    /// First soft token.
    pub offset: u32,
    pub rows: u32,
    pub identity: [u8; 32],
}

/// Device embeddings (`[rows][hidden]` f32) of prompt image `index`
/// (`ImageEmbeddings`). `sync_with` asks only for images it must evaluate.
pub type ImageEmbeddings<'a> = dyn Fn(usize) -> Result<Arc<DeviceTensor<f32>>, String> + 'a;

/// A session's next tokens (`Session::DecodeResult`).
#[derive(Debug, Clone, Default)]
pub struct DecodeResult {
    pub tokens: Vec<TokenId>,
    /// One entry per token when the sampler requests logprobs.
    pub logprobs: Vec<TokenLogprobs>,
    pub stop: bool,
}

/// One session's work in [`Session::decode_batch`] (`Session::BatchDecode`).
pub struct BatchDecode<'a> {
    pub session: &'a mut Session,
    pub max_tokens: usize,
    pub sampler: &'a mut SamplerState,
    pub result: DecodeResult,
    pub stop_at_eos: bool,
    /// Set when this session failed.
    pub error: Option<String>,
}

/// `Session::SpeculativeStats`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub cycles: u64,
    /// Cycles that verified at least one draft.
    pub verified: u64,
    /// MTP drafts and copied tokens.
    pub drafted: u64,
    /// MTP drafts and copied tokens.
    pub accepted: u64,
    /// Prompt-lookup tokens among `drafted`.
    pub copied: u64,
    pub copied_accepted: u64,
    pub siblings: u64,
    pub siblings_accepted: u64,
    pub small_head_cycles: u64,
    pub small_head_drafted: u64,
    pub small_head_accepted: u64,
}

/// One resident model and any number of sessions with independent context
/// state; forwards of different sessions are serialized on the model's
/// executor (`Model`). `Send + Sync`.
pub struct Model {
    pub(crate) options: ModelOptions,
    name: String,
    /// The target's mapped shards, then the drafter's.
    files: Vec<Arc<MappedFile>>,
    weights: ModelWeights,
    draft_weights: Option<DraftWeights>,
    tokenizer: Tokenizer,
    pub(crate) device: DeviceModel,
    pub(crate) exec: Mutex<Exec>,
    /// Why the GGUF's chat template is not the built-in one, if it is not.
    template_warning: Option<String>,
    /// One JSON line per verification cycle (`draft_trace_path`).
    pub(crate) trace: Option<CycleTrace>,
    pub(crate) snapshot_memory: Arc<SnapshotMemory>,
}

/// The executor and the state guarded with it (`Model::mutex_`): the
/// calibrated-policy tables shared by every session (`[greedy, sampled]`
/// drafts, then their siblings) and the count of verification forwards
/// while tracing.
pub(crate) struct Exec {
    pub(crate) executor: Executor,
    pub(crate) calibration: [DraftCalibration; 4],
    pub(crate) forwards: u64,
}

impl std::ops::Deref for Exec {
    type Target = Executor;
    fn deref(&self) -> &Executor {
        &self.executor
    }
}

impl std::ops::DerefMut for Exec {
    fn deref_mut(&mut self) -> &mut Executor {
        &mut self.executor
    }
}

/// Maps a GGUF file.
fn map(path: &Path) -> Result<Arc<MappedFile>, String> {
    MappedFile::open(path).map_err(|e| format!("open {}: {e}", path.display()))
}

impl Model {
    /// Maps, binds and uploads the target and its optional drafter
    /// (`Model::Load`). `max_context` 0 asks for the model's native context.
    pub fn load(path: &Path, options: ModelOptions) -> Result<Arc<Model>, String> {
        if options.prefill_chunk == 0 || options.max_logit_rows == 0 {
            return Err("gemma4 model options must be positive".into());
        }
        let requested_context = options.max_context;
        let mut options = options;
        let mut files = vec![map(path)?];
        let (weights, tokenizer, name, template_warning) = {
            let shard = GgufShard::parse(files[0].bytes())
                .map_err(|e| format!("open {}: {e}", path.display()))?;
            // Prompts render with the built-in template whatever the GGUF
            // carries, so fine-tunes shipping another revision still load.
            let template_warning = chat_template::validate_gguf_template(&shard.gguf)
                .err()
                .map(|e| format!("{e}; rendering prompts with the built-in template"));
            let name = shard.gguf.string("general.name").map_or_else(
                || "gemma4".to_string(),
                |n| String::from_utf8_lossy(n).into_owned(),
            );
            let shards = [shard];
            let weights = ModelWeights::bind(&shards)?;
            let tokenizer = Tokenizer::from_gguf(&shards[0].gguf)?;
            (weights, tokenizer, name, template_warning)
        };
        if options.max_context == 0 {
            options.max_context = weights.config.context_length;
        }
        if options.max_context == 0 {
            return Err("gemma4 GGUF has no native context length; pass --context".into());
        }
        if requested_context > weights.config.context_length {
            return Err(format!(
                "requested context exceeds the model's native {}",
                weights.config.context_length
            ));
        }
        let mut draft_weights = None;
        if let Some(mtp) = &options.mtp_model_path {
            if options.draft_tokens == 0 || options.draft_tokens > MAX_DRAFT_TOKENS {
                return Err(format!("gemma4 draft tokens must be 1..{MAX_DRAFT_TOKENS}"));
            }
            if options.min_draft_tokens == 0 || options.min_draft_tokens > options.draft_tokens {
                return Err("gemma4 minimum draft tokens must be 1..draft tokens".into());
            }
            let file = map(mtp)?;
            let shard = GgufShard::parse(file.bytes())
                .map_err(|e| format!("open {}: {e}", mtp.display()))?;
            let draft =
                DraftWeights::bind(&[shard], &weights).map_err(|e| format!("MTP drafter: {e}"))?;
            files.push(file);
            draft_weights = Some(draft);
        }
        let trace = match &options.draft_trace_path {
            Some(_) if draft_weights.is_none() => {
                return Err("gemma4 draft trace requires an MTP drafter".into());
            }
            Some(path) => Some(CycleTrace::new(&path.to_string_lossy())?),
            None => None,
        };
        // Only the mixture-of-experts cycle is short enough for the smaller
        // draft head to pay (draft_vocab.rs).
        let draft_order = if draft_weights.is_some() && weights.config.has_experts() {
            draft_head_order(&tokenizer)
        } else {
            Vec::new()
        };
        let stream = Stream::new().map_err(hip)?;
        let device = DeviceModel::upload(
            &stream,
            &weights,
            1,
            &files,
            draft_weights.as_ref(),
            &draft_order,
            options.prefill_weights,
        )?;
        // A final remainder of up to SPLIT_ROWS rows joins the last prefill
        // chunk (`Session::extend`).
        let exec = Executor::new(
            &device,
            stream,
            options.prefill_chunk + SPLIT_ROWS,
            options.max_logit_rows,
            options.max_context,
        )?;
        Ok(Arc::new(Model {
            options,
            name,
            files,
            weights,
            draft_weights,
            tokenizer,
            device,
            exec: Mutex::new(Exec {
                executor: exec,
                calibration: [DraftCalibration::default(); 4],
                forwards: 0,
            }),
            template_warning,
            trace,
            snapshot_memory: Arc::new(SnapshotMemory::default()),
        }))
    }

    pub(crate) fn exec(&self) -> MutexGuard<'_, Exec> {
        self.exec.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A session of up to `max_context` tokens (0: the model's maximum).
    pub fn create_session(self: &Arc<Self>, max_context: u32) -> Result<Session, String> {
        let max_context = if max_context == 0 {
            self.options.max_context
        } else {
            max_context
        };
        let cache = self.exec().create_cache(&self.device, max_context)?;
        Ok(Session {
            model: Arc::clone(self),
            cache,
            tokens: Vec::new(),
            images: Vec::new(),
            logits: Vec::new(),
            logits_on_device: false,
            valid: false,
            pending: None,
            ahead: Vec::new(),
            ahead_stops: Vec::new(),
            stats: SpeculativeStats::default(),
            lookup: PromptLookup::default(),
            draft_vocab: DraftVocab::new(DRAFT_TABLE_ROWS),
            draft_added: Vec::new(),
            calibration: [DraftCalibration::default(); 4],
            held: Mutex::new(Held::default()),
        })
    }

    /// Tokenizes text with special tokens parsed and no BOS; rendered chat
    /// prompts carry their own `<bos>`.
    pub fn tokenize(&self, text: &str) -> Vec<TokenId> {
        self.tokenizer.encode(text.as_bytes(), false, true)
    }

    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        self.tokenizer.decode(tokens, false)
    }

    pub fn token_text(&self, token: TokenId) -> Vec<u8> {
        self.tokenizer.token_text(token, false)
    }

    pub fn is_stop_token(&self, token: TokenId) -> bool {
        self.tokenizer.is_end_of_generation(token)
    }

    pub fn vocab_size(&self) -> u32 {
        self.weights.vocab_size
    }

    pub fn max_context(&self) -> u32 {
        self.options.max_context
    }

    pub fn config(&self) -> &Config {
        &self.weights.config
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// The mapped target shards, then the drafter's.
    pub fn files(&self) -> &[Arc<MappedFile>] {
        &self.files
    }

    /// The device weights.
    pub fn device(&self) -> &DeviceModel {
        &self.device
    }

    /// Why the GGUF's chat template is not the built-in one (prompts render
    /// with the built-in one regardless), for the caller to log.
    pub fn template_warning(&self) -> Option<&str> {
        self.template_warning.as_deref()
    }

    pub fn resident_bytes(&self) -> usize {
        self.device.resident_bytes + self.exec().scratch_bytes()
    }

    pub fn session_bytes(&self, context: u32) -> usize {
        let exec = self.exec();
        Executor::cache_bytes(
            self.config(),
            context,
            exec.ring(),
            exec.widths(),
            self.vocab_size(),
            Executor::draft_table_bytes(&self.device),
        )
    }

    /// After each layer of every forward, `sink` receives that layer's
    /// residual rows (`[count][hidden]`, on the host): the target features a
    /// DFlash drafter reads. For tools; unset in serving (`SetTapSink`).
    pub fn set_tap_sink(&self, sink: Option<LayerTapSink>) {
        self.exec().set_tap_sink(sink);
    }

    /// Sliding-window ring slots per session (window + prefill chunk).
    pub fn session_ring_slots(&self) -> usize {
        self.exec().ring() as usize
    }

    pub fn model_name(&self) -> String {
        self.name.clone()
    }

    pub fn has_mtp(&self) -> bool {
        self.draft_weights.is_some()
    }

    pub fn draft_tokens(&self) -> u32 {
        self.options.draft_tokens
    }

    pub fn options(&self) -> &ModelOptions {
        &self.options
    }
}

/// Where a later sync may stop inside the lookahead rows, in order: the
/// lookahead tokens it adopts, the logits of the last of them and that
/// row's hidden state in stash slot `slot` (`AheadStop`).
pub(crate) struct AheadStop {
    count: usize,
    slot: u32,
    logits: Vec<f32>,
}

/// One conversation's context (`Session`): tokens, the KV rows later tokens
/// can still attend, the frontier hidden state and the last logits. The KV
/// rows and hidden state stay in device memory.
pub struct Session {
    pub(crate) model: Arc<Model>,
    pub(crate) cache: KvCache,
    pub(crate) tokens: Vec<TokenId>,
    pub(crate) images: Vec<ImageSpan>,
    /// Host logits of the frontier token; stale while `logits_on_device`,
    /// when the cache holds them.
    pub(crate) logits: Vec<f32>,
    pub(crate) logits_on_device: bool,
    pub(crate) valid: bool,
    /// Emitted but not yet evaluated; its predecessor's hidden state is the
    /// cache's frontier hidden.
    pub(crate) pending: Option<TokenId>,
    /// Tokens evaluated past `tokens` by a sync's lookahead: their KV rows
    /// are in the cache.
    pub(crate) ahead: Vec<TokenId>,
    pub(crate) ahead_stops: Vec<AheadStop>,
    pub(crate) stats: SpeculativeStats,
    /// Prompt-lookup index over `tokens`; cleared whenever `tokens` is
    /// rewritten rather than appended to.
    pub(crate) lookup: PromptLookup<i32>,
    /// The request's draft vocabulary over `tokens`, reset with `lookup`.
    pub(crate) draft_vocab: DraftVocab,
    pub(crate) draft_added: Vec<i32>,
    /// Calibrated-policy tables of a request-scoped session.
    pub(crate) calibration: [DraftCalibration; 4],
    /// Snapshot blocks whose rows the cache still holds (`held_global_`,
    /// `held_sliding_`).
    pub(crate) held: Mutex<Held>,
}

impl Session {
    /// Compatibility version; bump on payload or inference arithmetic changes.
    pub const SNAPSHOT_PAYLOAD_VERSION: u32 = 3;

    /// Feeds a prompt, reusing the longest common prefix the sliding-window
    /// ring can still serve.
    pub fn sync(&mut self, prompt: &[TokenId]) -> Result<(), String> {
        self.sync_with(
            prompt,
            &[],
            &|_| Err("gemma4 prompt has no images".into()),
            &[],
            &[],
        )
    }

    /// Starts a request on the current context (`sync` does this too):
    /// prompt lookup forgets the previous request's copy and skips its first
    /// step.
    pub fn begin_request(&mut self) {
        self.lookup.begin_request();
    }

    /// Forgets rows a sync computed past the frontier (`DropLookahead`).
    pub(crate) fn drop_lookahead(&mut self) {
        self.ahead.clear();
        self.ahead_stops.clear();
    }

    /// `Sync` with images and lookahead (see `engine.hpp`): `images`
    /// (ordered, each followed by a text token) take their rows from
    /// `embed`; reuse stops at the first image whose identity or placement
    /// changed, and an image is always evaluated whole. `lookahead`: text
    /// tokens expected to follow `prompt` in later syncs, evaluated in the
    /// prompt's last forward while the session itself ends at `prompt`.
    pub fn sync_with(
        &mut self,
        prompt: &[TokenId],
        images: &[ImageSpan],
        embed: &ImageEmbeddings<'_>,
        lookahead: &[TokenId],
        lookahead_stops: &[usize],
    ) -> Result<(), String> {
        if prompt.is_empty() {
            self.reset();
            return Err("prompt is empty".into());
        }
        if lookahead_stops.windows(2).any(|w| w[0] > w[1])
            || lookahead_stops
                .last()
                .is_some_and(|&last| last >= lookahead.len())
        {
            return Err("lookahead stops must ascend below the lookahead".into());
        }
        self.begin_request();
        if self.model.options.draft_calibration == DraftCalibrationScope::Request {
            for calibration in &mut self.calibration {
                calibration.reset();
            }
        }
        if prompt.len() > self.cache.max_context as usize {
            return Err("prompt exceeds the session context".into());
        }
        // Rows an earlier sync evaluated past its frontier: adopt them up to
        // a stop if this prompt asks for exactly those tokens next, and keep
        // the rest for the next sync.
        if !self.ahead.is_empty() {
            let n = self.tokens.len();
            let stop = self
                .ahead_stops
                .iter()
                .position(|s| prompt.len() == n + s.count);
            let adopt = stop.filter(|&i| {
                self.valid
                    && self.pending.is_none()
                    && prompt[..n] == self.tokens[..]
                    && prompt[n..] == self.ahead[..self.ahead_stops[i].count]
                    && images == self.images.as_slice()
            });
            if let Some(i) = adopt {
                let slot = self.ahead_stops[i].slot;
                let adopted = {
                    let model = Arc::clone(&self.model);
                    let mut exec = model.exec();
                    exec.adopt_stashed_hidden(&model.device, &mut self.cache, slot)
                };
                if let Err(e) = adopted {
                    self.reset();
                    return Err(e);
                }
                let count = self.ahead_stops[i].count;
                self.tokens.extend_from_slice(&self.ahead[..count]);
                self.logits = std::mem::take(&mut self.ahead_stops[i].logits);
                self.logits_on_device = false;
                self.ahead.drain(..count);
                self.ahead_stops.drain(..=i);
                for later in &mut self.ahead_stops {
                    later.count -= count;
                }
                if self.ahead.is_empty() {
                    self.drop_lookahead();
                }
                return Ok(());
            }
            // Between stops, the rows the prompt shares with the lookahead
            // stay: only the last one is evaluated again, for its logits.
            if self.valid
                && self.pending.is_none()
                && prompt.len() > n + 1
                && prompt[..n] == self.tokens[..]
                && images == self.images.as_slice()
            {
                let mut shared = 0;
                while shared < self.ahead.len()
                    && n + shared < prompt.len()
                    && prompt[n + shared] == self.ahead[shared]
                {
                    shared += 1;
                }
                if shared != 0 {
                    let rows = self.ahead[..shared].to_vec();
                    self.tokens.extend_from_slice(&rows);
                    self.valid = false;
                }
            }
            self.drop_lookahead();
        }
        let max_rows = self.model.exec().max_rows() as usize;
        let mut previous_end = 0usize;
        for image in images {
            // A text token follows every image, so the frontier is never
            // inside one.
            let end = image.offset as usize + image.rows as usize;
            if image.rows == 0
                || (image.offset as usize) < previous_end
                || end >= prompt.len()
                || image.rows as usize > max_rows
            {
                return Err("invalid image placement".into());
            }
            previous_end = end;
        }
        self.pending = None;
        let limit = prompt.len().min(self.tokens.len());
        let mut common = prompt
            .iter()
            .zip(&self.tokens)
            .take(limit)
            .take_while(|(a, b)| a == b)
            .count();
        // The lookup index stays valid while the prompt extends the tokens.
        let extends = common == self.tokens.len();
        // Equal tokens do not mean equal images: stop at the first image that
        // differs, and never resume inside an image of the new prompt.
        for i in 0..images.len().max(self.images.len()) {
            if i < images.len() && i < self.images.len() && images[i] == self.images[i] {
                continue;
            }
            let mut differs = common;
            if let Some(image) = images.get(i) {
                differs = differs.min(image.offset as usize);
            }
            if let Some(image) = self.images.get(i) {
                differs = differs.min(image.offset as usize);
            }
            common = differs;
            break;
        }
        if common == prompt.len() && common == self.tokens.len() && self.valid {
            return Ok(());
        }
        // The last prompt token is re-evaluated to produce its logits.
        common = common.min(prompt.len() - 1);
        for image in images {
            let (first, end) = (
                image.offset as usize,
                image.offset as usize + image.rows as usize,
            );
            if common > first && common < end {
                common = first;
            }
        }
        // Rewinding needs the sliding window before `common` still in the
        // ring.
        let window = self.model.config().sliding_window as usize;
        if self.tokens.len().saturating_sub(common) + window > self.cache.ring as usize {
            common = 0;
        }
        self.tokens = prompt.to_vec();
        self.images = images.to_vec();
        if !extends {
            self.lookup.clear();
        }
        self.draft_vocab.reset();
        let ahead = prompt.len() + lookahead.len() <= self.cache.max_context as usize;
        if ahead {
            self.extend(common, Some(embed), lookahead, lookahead_stops)
        } else {
            self.extend(common, Some(embed), &[], &[])
        }
    }

    /// Evaluates `tokens[begin..]` in prefill chunks that never split an
    /// image; the last row's logits become the frontier's. `lookahead` rows
    /// follow in the last forward: the logits and hidden state of the rows
    /// ending each of `stops` and of the last row are stashed, the session
    /// itself ends at `tokens` (`Extend`).
    fn extend(
        &mut self,
        begin: usize,
        embed: Option<&ImageEmbeddings<'_>>,
        lookahead: &[TokenId],
        stops: &[usize],
    ) -> Result<(), String> {
        let frontier = self.tokens.len();
        self.drop_lookahead();
        self.valid = false;
        let mut ahead = Vec::new();
        let mut ridden = 0;
        let result = self.run_extend(
            begin,
            frontier,
            embed,
            lookahead,
            stops,
            &mut ahead,
            &mut ridden,
        );
        if let Err(e) = result {
            self.tokens.truncate(begin);
            self.images
                .retain(|image| image.offset as usize + image.rows as usize <= begin);
            self.lookup.clear();
            self.draft_vocab.reset();
            self.drop_lookahead();
            return Err(e);
        }
        self.tokens.truncate(frontier);
        self.ahead = lookahead[..ridden].to_vec();
        self.ahead_stops = ahead;
        self.valid = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_extend(
        &mut self,
        begin: usize,
        frontier: usize,
        embed: Option<&ImageEmbeddings<'_>>,
        lookahead: &[TokenId],
        stops: &[usize],
        ahead: &mut Vec<AheadStop>,
        ridden: &mut usize,
    ) -> Result<(), String> {
        let model = Arc::clone(&self.model);
        let chunk = model.options.prefill_chunk as usize;
        let max_logit_rows = model.options.max_logit_rows as usize;
        let window = model.config().sliding_window as usize;
        let vocab = model.vocab_size() as usize;
        let mut exec = model.exec();
        let max_rows = exec.max_rows() as usize;
        let mut rows = Vec::new();
        let mut start = begin;
        while start < frontier {
            let mut end = (start + chunk).min(frontier);
            // A remainder of at most SPLIT_ROWS rows joins this chunk: alone
            // it would read every weight again for a few rows.
            if frontier - end <= SPLIT_ROWS as usize && frontier - start <= max_rows {
                end = frontier;
            }
            rows.clear();
            for (i, image) in self.images.iter().enumerate() {
                let first = image.offset as usize;
                let image_end = first + image.rows as usize;
                if image_end <= start || first >= end {
                    continue;
                }
                if first < start {
                    return Err("gemma4 prefill resumed inside an image".into());
                }
                if image_end > end {
                    // Stop before an image that does not fit; one that starts
                    // the chunk fits because images are at most a chunk long.
                    end = if first > start { first } else { image_end };
                    if first > start {
                        break;
                    }
                }
                let embed = embed.ok_or("gemma4 image has no embeddings")?;
                rows.push(ImageRows {
                    row: (first - start) as u32,
                    count: image.rows,
                    embedding: embed(i)?,
                });
            }
            let mut count = end - start;
            let last = end == frontier;
            if last && !lookahead.is_empty() {
                // The lookahead rides in the frontier's forward where rows are
                // left, whole or up to a later stop; past LOOKAHEAD_ROWS the
                // delay exceeds the separate forward it saves.
                let room = (max_rows - count)
                    .min((self.cache.ring as usize).saturating_sub(window + 64))
                    .min(LOOKAHEAD_ROWS);
                if lookahead.len() <= room {
                    *ridden = lookahead.len();
                } else {
                    for &stop in stops {
                        if stop <= room {
                            *ridden = stop;
                        }
                    }
                }
                for &stop in stops {
                    if stop != 0
                        && stop < *ridden
                        && ahead.len() + 1 < SPLIT_ROWS as usize
                        && ahead.len() + 2 < max_logit_rows
                        && ahead.last().is_none_or(|a: &AheadStop| stop > a.count)
                    {
                        ahead.push(AheadStop {
                            count: stop,
                            slot: ahead.len() as u32,
                            logits: Vec::new(),
                        });
                    }
                }
                if *ridden != 0 && max_logit_rows >= 2 {
                    ahead.push(AheadStop {
                        count: *ridden,
                        slot: ahead.len() as u32,
                        logits: Vec::new(),
                    });
                    self.tokens.extend_from_slice(&lookahead[..*ridden]);
                    count += *ridden;
                } else {
                    ahead.clear();
                    *ridden = 0;
                }
            }
            // Logit rows: the frontier's last token, and the lookahead rows
            // ending each stop.
            let mut logit_rows = Vec::new();
            if last {
                logit_rows.push((frontier - 1 - start) as u32);
                for stop in ahead.iter() {
                    logit_rows.push((frontier + stop.count - 1 - start) as u32);
                }
            }
            exec.forward_one(
                &model.device,
                &mut self.cache,
                &self.tokens[start..start + count],
                start as u32,
                &logit_rows,
                &rows,
                0,
            )?;
            start += count;
            if !last {
                continue;
            }
            exec.commit_hidden(&model.device, &mut self.cache, logit_rows[0])?;
            for (i, stop) in ahead.iter().enumerate() {
                exec.stash_hidden(&model.device, &mut self.cache, logit_rows[1 + i], stop.slot)?;
            }
            let logits = exec.read_logits(logit_rows.len())?;
            self.logits = logits[..vocab].to_vec();
            self.logits_on_device = false;
            for (i, stop) in ahead.iter_mut().enumerate() {
                stop.logits = logits[(1 + i) * vocab..(2 + i) * vocab].to_vec();
            }
        }
        Ok(())
    }

    /// Appends one token (`Evaluate`).
    pub fn evaluate(&mut self, token: TokenId) -> Result<(), String> {
        self.pending = None;
        if self.tokens.len() >= self.cache.max_context as usize {
            return Err("session context is full".into());
        }
        self.tokens.push(token);
        let begin = self.tokens.len() - 1;
        self.extend(begin, None, &[], &[])
    }

    /// Appends `tokens` and returns the logits of every row (`[n][vocab]`;
    /// `EvaluateAll`), in forwards of up to `max_logit_rows` rows.
    pub fn evaluate_all(
        &mut self,
        tokens: &[TokenId],
        logits: &mut Vec<f32>,
    ) -> Result<(), String> {
        self.drop_lookahead();
        if self.tokens.len() + tokens.len() > self.cache.max_context as usize {
            return Err("tokens exceed the session context".into());
        }
        let model = Arc::clone(&self.model);
        let vocab = model.vocab_size() as usize;
        logits.clear();
        self.valid = false;
        self.pending = None;
        let result = (|| -> Result<(), String> {
            let mut exec = model.exec();
            let chunk = (exec.max_rows() as usize).min(model.options.max_logit_rows as usize);
            let mut start = 0;
            while start < tokens.len() {
                let count = chunk.min(tokens.len() - start);
                let rows: Vec<u32> = (0..count as u32).collect();
                let first = self.tokens.len() as u32;
                self.tokens.extend_from_slice(&tokens[start..start + count]);
                exec.forward_one(
                    &model.device,
                    &mut self.cache,
                    &tokens[start..start + count],
                    first,
                    &rows,
                    &[],
                    0,
                )?;
                exec.commit_hidden(&model.device, &mut self.cache, count as u32 - 1)?;
                logits.extend_from_slice(exec.read_logits(count)?);
                start += count;
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.reset();
            return Err(e);
        }
        self.logits = logits[logits.len() - vocab..].to_vec();
        self.logits_on_device = false;
        self.valid = true;
        Ok(())
    }

    pub fn statistics(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// The frontier logits; copies a row decoding kept on the device first.
    /// Empty while the session is not valid.
    pub fn logits(&mut self) -> Result<&[f32], String> {
        if !self.valid {
            return Ok(&[]);
        }
        if self.logits_on_device {
            self.model
                .exec()
                .copy_cache_logits(&self.cache, &mut self.logits)?;
            self.logits_on_device = false;
        }
        Ok(&self.logits)
    }

    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    /// Images inside [`Self::tokens`].
    pub fn images(&self) -> &[ImageSpan] {
        &self.images
    }

    pub fn position(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Tokens a sync evaluated past the frontier that a later sync may adopt.
    pub fn lookahead_tokens(&self) -> usize {
        self.ahead.len()
    }

    pub fn context_size(&self) -> u32 {
        self.cache.max_context
    }

    /// The last emitted token is not yet evaluated.
    pub fn has_pending_token(&self) -> bool {
        self.pending.is_some()
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn allocated_bytes(&self) -> usize {
        self.cache.bytes
    }

    pub fn reset(&mut self) {
        self.drop_lookahead();
        self.pending = None;
        self.tokens.clear();
        self.images.clear();
        self.lookup.clear();
        self.draft_vocab.reset();
        self.logits.clear();
        self.logits_on_device = false;
        self.valid = false;
        // A restored snapshot starts a request without a sync.
        if self.model.options.draft_calibration == DraftCalibrationScope::Request {
            for calibration in &mut self.calibration {
                calibration.reset();
            }
        }
    }
}
