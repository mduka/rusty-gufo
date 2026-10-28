//! The trunk on the GPU for one session at a time, a port of `Executor`
//! (`kernels/rocm/executor.cpp`): scratch sized once for `max_batch`
//! tokens, prefill, decode and verification forwards with every route of
//! the C++ executor, decode graphs per batch shape that read positions from
//! device memory, and rollback of a speculative batch.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use gufo_hip::blas::BlasLt;
use gufo_hip::{DeviceTensor, LaunchError, Stream};
use gufo_qwen::vision::{Encoder, Prompt};
use gufo_qwen38_flash_next_kernels as k;

use crate::borrowed::{Pool, SPARE_BLOCKS};
use crate::checkpoint::{CheckpointStorage, PrefillCheckpoint};
use crate::config::Config;
use crate::device::{Block, DeviceModel, Layer};
use crate::draft_vocab::DRAFT_EXTRA_ROWS;
use crate::forward::{AttentionControl, Checkpoint, Pass, Widths, at, at_mut, inject_weights};
use crate::mtp::{HiddenSource, MtpOutput};
use crate::ngram::{NgramHistory, NgramTable, hash_ngram_rows};
use crate::rope_scaling::{YarnRope, make_yarn_rope};
use crate::routes::{Ctx, Error, Route, launch_error};
use crate::state::{
    Host, ROLLBACK_ROWS, Scratch, Session, Stage, VEC_BATCH, gdn_rollback_row_floats,
    indexer_capacity,
};

static NEXT_EXECUTOR: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub max_batch: u32,
    /// Rows of logits a forward may return.
    pub max_logit_rows: u32,
    /// Longest speculative batch; bounds the rollback storage.
    pub max_speculative: u32,
    /// The longest session; 0 selects the native context, longer ones need
    /// YaRN (`Config::max_context_length`).
    pub max_context: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardMode {
    Decode,
    /// Permits a rollback afterwards (at most `max_speculative` rows).
    Verify,
    /// Prompt arithmetic at every chunk width.
    Prefill,
}

/// The bit of a graph key that says whether the session has an mRoPE
/// table (decode and MTP graphs share the cache; bits 0 to 35 and 40 to 44
/// carry their other launch-shaping inputs).
pub(crate) const ROPE_KEY_BIT: u32 = 36;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionMode {
    Autoregressive,
    Speculative,
}

pub struct Executor {
    pub(crate) id: u64,
    pub(crate) stream: Stream,
    /// Device storage of borrowed snapshots.
    pub(crate) pool: Arc<Pool>,
    /// Snapshot copies of idle sessions, beside the forwards of others.
    pub(crate) saver: crate::snapshot::Saver,
    pub(crate) core: Core,
}

impl Drop for Executor {
    fn drop(&mut self) {
        // The stream goes with the executor (`~Executor`).
        let _ = k::mmq_release_stream(&self.stream);
    }
}

/// Everything but the stream, so a graph body can use it while the graph
/// cache holds the stream.
pub(crate) struct Core {
    pub model: Arc<DeviceModel>,
    pub ngram: Option<Arc<NgramTable>>,
    pub options: Options,
    pub blaslt: BlasLt,
    pub s: Scratch,
    pub stage: Stage,
    pub host: Host,
    pub route: Route,
    pub widths: Widths,
    pub mask_words: u32,
    /// The session capacity sparse-mask rows are sized for: the native
    /// context or a longer YaRN one.
    pub context_capacity: u32,
    /// The YaRN constants every rope launch receives (disabled: the
    /// original instruction sequence).
    pub yarn: YarnRope,
    /// The wide mixer route (F16 norm for the epilogue, tiled Q8 norm for
    /// the W8A8 down projection) fits the model's geometry.
    pub wide_mixer: bool,
    pub ple_pending: bool,
    /// The n-gram rows of the next prefill chunk.
    pub prefetch: PlePrefetch,
    /// Logit rows of the last batched forward, allocated on first use.
    pub batch_logits: Option<gufo_hip::DeviceBuffer<f32>>,
    /// The batched GatedDeltaNet's request descriptors, one slot per layer
    /// in host-mapped page-locked memory (`batch_gdn_host_`), allocated on
    /// first use.
    pub batch_gdn: Option<gufo_hip::DeviceBuffer<u8>>,
    pub batch_rows: u32,
    /// The batch row verification reads instead of the forward's own rows
    /// (`SelectBatchLogits`).
    pub selected: Option<u32>,
    /// Greedy verification's penalty histories, grown on first use.
    pub penalty_rows: Option<PenaltyRows>,
    /// Compact sampled verification's selections, allocated on first use.
    pub verification: Option<VerificationScratch>,
    /// Where a prefill pass leaves its checkpoint, allocated on first use.
    pub checkpoint: Option<CheckpointStorage>,
}

/// The n-gram rows of the next prefill chunk, read while this chunk's layers
/// run. A chunk's rows are a pure function of its
/// tokens and the n-gram history before them, so only a batch with exactly
/// those tokens and that history claims them, whichever session it belongs
/// to.
#[derive(Default)]
pub(crate) struct PlePrefetch {
    /// The read of `tokens` is in flight in the table.
    pub pending: bool,
    /// Its rows are complete in `rows`.
    pub ready: bool,
    /// The current forward collects them at its PLE layer.
    pub claimed: bool,
    pub before: NgramHistory,
    pub after: NgramHistory,
    pub tokens: Vec<i32>,
    pub rows: Vec<f32>,
    /// The successor the current prefill forward names, hashed from the
    /// history after it; its read starts at the PLE layer, once the
    /// forward's own rows are in.
    pub next: Option<NextChunk>,
}

pub(crate) struct NextChunk {
    pub before: NgramHistory,
    pub after: NgramHistory,
    pub tokens: Vec<i32>,
    pub rows: Vec<u32>,
}

impl PlePrefetch {
    /// Waits for a read in flight before the table takes another; its rows
    /// stay claimable (`FinishPrefetch`).
    pub fn finish(&mut self, ngram: &NgramTable) {
        if std::mem::take(&mut self.pending) {
            self.ready = ngram.wait_read_swap(&mut self.rows);
        }
    }

    /// Starts reading the named successor's rows, best effort: a read that
    /// cannot start is skipped (`PrefetchPle`).
    pub fn start(&mut self, ngram: &NgramTable) {
        let Some(next) = self.next.take() else {
            return;
        };
        if !self.pending && ngram.start_read(&next.rows) {
            self.pending = true;
            self.ready = false;
            self.before = next.before;
            self.after = next.after;
            self.tokens = next.tokens;
        }
    }
}

/// The device copy of each verification row's sorted penalty history:
/// shared scratch, never part of a session or its snapshot.
pub(crate) struct PenaltyRows {
    /// (token, generated count, repeated) triples.
    pub triples: gufo_hip::DeviceBuffer<u32>,
    pub capacity: usize,
}

/// The ID workspaces and candidate rows of compact sampled verification
/// (`verification_ids_`, `verification_candidates_`): shared scratch, never
/// part of a session or its snapshot.
pub(crate) struct VerificationScratch {
    pub ids: gufo_hip::DeviceBuffer<u32>,
    pub scratch_ids: gufo_hip::DeviceBuffer<u32>,
    /// 8-byte (id, logit) candidates for `candidate_capacity` entries.
    pub candidates: gufo_hip::DeviceBuffer<u8>,
    pub candidate_capacity: usize,
}

/// What one forward body runs.
#[derive(Clone, Copy)]
pub(crate) struct Body {
    n: u32,
    n_logits: u32,
    speculative: bool,
    sparse: bool,
    start_pos: u32,
    pool_grid: u32,
    first_layer: u32,
    end_layer: u32,
    /// The n-gram rows are on the device before the body runs.
    embeddings_ready: bool,
    /// Rows before a prefill checkpoint the body captures (0: none).
    checkpoint: u32,
}

impl Executor {
    /// Sizes the scratch for `options.max_batch` tokens (`Executor::Create`).
    pub fn new(
        model: Arc<DeviceModel>,
        ngram: Option<Arc<NgramTable>>,
        options: Options,
    ) -> Result<Self, String> {
        let max_batch = options.max_batch.max(1);
        let max_logit_rows = options.max_logit_rows.clamp(1, max_batch);
        let options = Options {
            max_batch,
            max_logit_rows,
            max_speculative: options.max_speculative.clamp(1, max_logit_rows),
            max_context: options.max_context,
        };
        let c = &model.config;
        if options.max_context > c.max_context_length() {
            return Err("executor context exceeds the model context".into());
        }
        let context_capacity = c.context_length.max(options.max_context);
        let yarn = make_yarn_rope(&c.rope_scaling, c.rotary_dim, c.rope_theta);
        let stream = Stream::new().map_err(|e| format!("hipStreamCreate: {e}"))?;
        // The first token would otherwise load these functions on the decode
        // path; preparation launches no work.
        let prepared = k::prepare_decode_kernels(&stream)
            .map_err(|e| format!("decode kernel preparation: {e}"))?;
        if prepared != gufo_hip::Launch::Launched {
            return Err("quantized GEMM tier initialization failed".into());
        }
        if model.mtp.is_some()
            && gufo_qwen_kernels::prepare_batched_argmax(&stream)
                .map_err(|e| format!("MTP argmax preparation: {e}"))?
                != gufo_hip::Launch::Launched
        {
            return Err("MTP argmax preparation failed".into());
        }
        let blaslt = BlasLt::new(&stream, 0).map_err(|e| format!("hipblasLtCreate failed: {e}"))?;
        let (s, stage, host, mask_words) =
            Scratch::new(&stream, &model, max_batch, max_logit_rows, context_capacity)
                .map_err(|e| format!("executor scratch: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| format!("executor scratch: {e}"))?;
        let c = &model.config;
        let wide_mixer = c.hc_count == 4
            && c.hidden_size.is_multiple_of(32)
            && model
                .layers
                .first()
                .is_some_and(|l| l.hc_ffn.down.format() == gufo_formats::GgmlType::Q8_0);
        let widths = Widths::new(c, mask_words, max_batch);
        let id = NEXT_EXECUTOR.fetch_add(1, Ordering::Relaxed);
        // Admission reserved the lazy scratch: snapshots must not take it.
        let pool = Pool::new(lazy_scratch_bytes(
            &model.config,
            options,
            model.mtp.is_some(),
        ))?;
        let saver = crate::snapshot::Saver::new(
            id,
            model.config.clone(),
            options.max_speculative,
            Arc::clone(&pool),
        )?;
        Ok(Self {
            id,
            stream,
            pool,
            saver,
            core: Core {
                model,
                ngram,
                options,
                blaslt,
                s,
                stage,
                host,
                route: Route::default(),
                widths,
                mask_words,
                context_capacity,
                yarn,
                wide_mixer,
                ple_pending: false,
                prefetch: PlePrefetch::default(),
                batch_logits: None,
                batch_gdn: None,
                batch_rows: 0,
                selected: None,
                penalty_rows: None,
                verification: None,
                checkpoint: None,
            },
        })
    }

    pub fn config(&self) -> &Config {
        &self.core.model.config
    }

    pub fn options(&self) -> Options {
        self.core.options
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn has_mtp(&self) -> bool {
        self.core.model.mtp.is_some()
    }

    /// Device bytes of the weights.
    pub fn resident_bytes(&self) -> usize {
        self.core.model.resident_bytes
    }

    /// Device scratch the first batched decode still allocates
    /// (`DeferredScratchBytes`).
    pub fn deferred_scratch_bytes(&self) -> usize {
        let c = self.config();
        let logits = if self.core.batch_logits.is_some() {
            0
        } else {
            batch_logit_bytes(c, self.core.options)
        };
        let checkpoint = if self.core.checkpoint.is_some() {
            0
        } else {
            CheckpointStorage::bytes(c, self.core.options.max_speculative, self.has_mtp())
        };
        // Committed spare blocks for captures, whether or not they exist yet.
        logits + checkpoint + SPARE_BLOCKS * CheckpointStorage::mutable_bytes(c)
    }

    pub fn create_session(&self, mode: SessionMode, max_context: u32) -> Result<Session, String> {
        let speculative = mode == SessionMode::Speculative;
        if speculative && !self.has_mtp() {
            return Err("MTP session requires a loaded predictor".into());
        }
        if max_context == 0 || max_context > self.core.context_capacity {
            return Err("session context exceeds the model context".into());
        }
        let draft = self.core.model.draft.as_ref().filter(|_| speculative);
        let mut session = Session::new(
            &self.stream,
            self.id,
            self.config(),
            max_context,
            self.core.options.max_batch,
            self.core.options.max_speculative,
            speculative,
            draft.map(|d| d.row_bytes),
        )
        .map_err(|e| format!("session allocation: {e}"))?;
        // The table starts with the special tokens; every other slot is
        // empty.
        if let Some(draft) = draft {
            self.write_draft_rows(&mut session, 0, &draft.specials, true)?;
            self.stream
                .synchronize()
                .map_err(|e| format!("draft table init: {e}"))?;
        }
        Ok(session)
    }

    /// Copies output rows `ids` into the session's draft table from slot
    /// `first` on and, with `clear`, empties every later slot
    /// (`AppendDraftRows`, `ClearDraftRows`). Enqueued on the stream, never
    /// during graph capture.
    pub fn write_draft_rows(
        &self,
        session: &mut Session,
        first: u32,
        ids: &[i32],
        clear: bool,
    ) -> Result<(), String> {
        let head = self
            .core
            .model
            .draft
            .as_ref()
            .ok_or("the model drafts over its full vocabulary")?;
        let table = session
            .mtp
            .as_mut()
            .and_then(|m| m.draft.as_mut())
            .filter(|_| session.owner == self.id)
            .ok_or("session without a draft table")?;
        let (first, slots) = (first as usize, DRAFT_EXTRA_ROWS as usize);
        if first > slots || ids.len() > slots - first {
            return Err("invalid draft table update".into());
        }
        if ids.is_empty() && !clear {
            return Ok(());
        }
        let row = head.row_bytes;
        let output = self.core.model.output.view().bytes();
        let vocab = self.config().vocab_size;
        let host = table.host.host_mut().map_err(|e| e.to_string())?;
        for (slot, &id) in (first..).zip(ids) {
            if id < 0 || id as u32 >= vocab {
                return Err("draft table token out of range".into());
            }
            let id = id as usize;
            self.stream
                .copy(
                    output.range(id * row..(id + 1) * row),
                    table.rows.slice_mut(slot * row..(slot + 1) * row),
                )
                .map_err(|e| format!("draft table row: {e}"))?;
            host[slot] = id as u32;
        }
        if clear {
            // Stale rows stay; padding masks the slot.
            host[first + ids.len()..].fill(u32::MAX);
        }
        self.stream
            .upload_indices_async(&mut table.host, slots, &mut table.ids)
            .map_err(|e| format!("draft table ids: {e}"))
    }

    /// Private device bytes of a session with `rollback_depth` rows
    /// (`SessionBytes`).
    pub fn session_bytes(&self, mode: SessionMode, max_context: u32, rollback_depth: u32) -> usize {
        let c = self.config();
        let attention = (c.num_layers / c.full_attention_interval) as usize;
        let linear = c.num_layers as usize - attention;
        let conv = (c.ssm_conv_kernel as usize - 1) * c.ssm_conv_channels() as usize;
        let state = c.ssm_num_v_heads as usize * (c.ssm_head_dim as usize).pow(2);
        let ple = if c.ple_layer.is_some() {
            c.ple_conv_history() as usize * c.hc_dim() as usize
        } else {
            0
        };
        let kv = 2 * max_context as usize * c.attention_kv_dim() as usize * 2;
        let index = indexer_capacity(c, self.core.options.max_batch, max_context) as usize
            * c.indexer_head_dim as usize
            * 4
            + (max_context / c.compress_ratio + 1) as usize * c.indexer_head_dim as usize * 2;
        let rollback_state = if rollback_depth == 0 {
            0
        } else {
            state + (rollback_depth as usize - 1) * gdn_rollback_row_floats(1, c)
        };
        let draft = self.core.model.draft.as_ref().map_or(0, |d| {
            DRAFT_EXTRA_ROWS as usize * (d.row_bytes + 4) + crate::device::TAIL
        });
        let speculative = if mode == SessionMode::Speculative {
            kv + index
                + (self.core.options.max_speculative as usize + 1) * c.hc_dim() as usize * 4
                + draft
        } else {
            0
        };
        ((linear * conv + ple) * (rollback_depth as usize + 1) + linear * (state + rollback_state))
            * 4
            + attention * (kv + index)
            + 5 * 4
            + speculative
    }

    /// Installs a request's image prompt and encoder on the session
    /// (`Session::ConfigureVision`).
    pub fn configure_vision(
        &self,
        session: &mut Session,
        prompt: Option<Arc<Prompt>>,
        encoder: Option<Arc<Encoder>>,
    ) -> Result<(), String> {
        if let Some(prompt) = &prompt {
            prompt
                .rope
                .validate(session.max_context)
                .map_err(|e| e.message().to_string())?;
        }
        session
            .vision
            .configure(prompt, encoder, &self.stream)
            .map_err(|e| e.message().to_string())
    }

    /// Appends `tokens` (at most `max_batch`) to the session and returns the
    /// logits of the last `n_logits` tokens in `logits` (`n_logits * vocab`
    /// floats) when given; the rows stay on the device for verification
    /// otherwise (`Forward`).
    pub fn forward(
        &mut self,
        session: &mut Session,
        tokens: &[i32],
        n_logits: u32,
        logits: Option<&mut [f32]>,
        mode: ForwardMode,
    ) -> Result<(), String> {
        self.forward_checkpoint(session, tokens, n_logits, logits, mode, None, &[])
    }

    /// [`forward`](Self::forward) of a prefill chunk that also captures
    /// `checkpoint` (from [`PrefillCheckpoint::new`] on this session and
    /// chunk) on its way: its logits arrive with the chunk's, its mutable
    /// state stays in the executor until the next forward
    /// (`Forward` with a `PrefillCheckpoint`). A prefill may name the `next`
    /// chunk of the same prompt: once its own n-gram rows are in, it reads
    /// the next chunk's, so that read overlaps its layers; the rows are used
    /// only by a batch of exactly those tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_checkpoint(
        &mut self,
        session: &mut Session,
        tokens: &[i32],
        n_logits: u32,
        logits: Option<&mut [f32]>,
        mode: ForwardMode,
        checkpoint: Option<&mut PrefillCheckpoint>,
        next: &[i32],
    ) -> Result<(), String> {
        let speculative = mode == ForwardMode::Verify;
        self.core.selected = None;
        let c = self.config().clone();
        let n = tokens.len() as u32;
        if n == 0
            || n > self.core.options.max_batch
            || (speculative && n > self.core.options.max_speculative)
        {
            return Err("token batch is empty or exceeds the batch limit".into());
        }
        if n_logits > n || n_logits > self.core.options.max_logit_rows {
            return Err("requested logit rows exceed the batch or limit".into());
        }
        if session.owner != self.id || session.position + n > session.max_context {
            return Err("session context is full".into());
        }
        if tokens.iter().any(|&t| t < 0 || t as u32 >= c.vocab_size) {
            return Err("token out of range".into());
        }
        if let Some(logits) = &logits
            && logits.len() < (n_logits * c.vocab_size) as usize
        {
            return Err("logit buffer is too short".into());
        }
        if session.check_cancelled() {
            return Err("generation cancelled".into());
        }
        if let Some(checkpoint) = &checkpoint
            && (mode != ForwardMode::Prefill
                || checkpoint.tokens > n
                || checkpoint.frontier.position != session.position + checkpoint.tokens)
        {
            return Err("prefill checkpoint does not match this pass".into());
        }
        if checkpoint.is_some() && self.core.checkpoint.is_none() {
            self.core.checkpoint = Some(
                CheckpointStorage::new(
                    &self.stream,
                    &c,
                    self.core.options.max_speculative,
                    self.has_mtp(),
                )
                .map_err(|e| format!("checkpoint storage: {e}"))?,
            );
        }
        let checkpoint_tokens = checkpoint.as_ref().map_or(0, |c| c.tokens);
        session.prepare_vision(n)?;
        let start_pos = session.position;
        let prefill = mode == ForwardMode::Prefill;
        if session.mtp_enabled
            && prefill
            && session
                .mtp
                .as_ref()
                .is_some_and(|m| m.position != start_pos)
        {
            return Err("MTP must consume the preceding frontier before prefill".into());
        }
        if speculative {
            session.ensure_rollback(&self.stream, &c, n - 1, self.core.options.max_speculative)?;
        }
        // Snapshots borrowing rows this pass writes copy them out first.
        let mtp_position = session.mtp.as_ref().map_or(0, |m| m.position);
        session.preserve(start_pos, mtp_position)?;
        session.mutation_epoch += 1;
        session.spec_base = start_pos;
        session.spec_tokens = if speculative { n } else { 0 };
        if c.ple_layer.is_some() {
            if self.core.ngram.is_none() {
                return Err("n-gram table is not open".into());
            }
            self.core
                .ple_fetch(&self.stream, session, tokens, speculative)?;
            if prefill && !next.is_empty() && next.len() <= self.core.options.max_batch as usize {
                let mut after = session.ngram;
                let mut rows = vec![0; next.len() * c.ple_heads as usize];
                hash_ngram_rows(&c, &mut after, next, &mut rows);
                self.core.prefetch.next = Some(NextChunk {
                    before: session.ngram,
                    after,
                    tokens: next.to_vec(),
                    rows,
                });
            }
        }
        // Sparse selection only matters once a query sees more than the
        // token budget; every layer shares one ratio.
        let sparse = start_pos + n > c.indexer_top_k;
        let complete = (start_pos + n) / c.compress_ratio;
        let pool_grid = if sparse && complete > session.blocks {
            complete - session.blocks
        } else {
            0
        };
        // Decode-sized batches replay as graphs; a pooling backlog (the
        // first batch past the budget) needs the wider eager grid.
        let graph_pool_grid = n / c.compress_ratio + 1;
        let graph = !prefill && n <= VEC_BATCH && pool_grid <= graph_pool_grid;
        let download = logits.is_some();
        // A session's first image creates its mRoPE table, which the
        // attention launches then read: graphs from before it do not apply.
        let key = u64::from(n)
            | (u64::from(n_logits) << 16)
            | (u64::from(speculative) << 32)
            | (u64::from(sparse) << 33)
            | (u64::from(download) << 34)
            | (u64::from(session.vision.rope().is_some()) << ROPE_KEY_BIT);
        // Tokens and control values go up before the work that reads them.
        self.core.route.prefill = prefill;
        let result = self
            .core
            .upload_inputs(&self.stream, session, tokens, start_pos)
            .and_then(|()| {
                let pool = if graph { graph_pool_grid } else { pool_grid };
                // PLE first reads the disk rows at its layer: the layers before
                // it are queued (or replayed) before the wait.
                let first_layer = match c.ple_layer {
                    Some(layer) if graph && layer > 0 => layer,
                    _ => 0,
                };
                let body = Body {
                    n,
                    n_logits: 0,
                    speculative,
                    sparse,
                    start_pos,
                    pool_grid: pool,
                    first_layer: 0,
                    end_layer: first_layer,
                    embeddings_ready: false,
                    checkpoint: checkpoint_tokens,
                };
                if first_layer > 0 {
                    self.run(session, key | (1 << 35), graph, |core, stream, session| {
                        core.body(stream, session, body)
                    })?;
                }
                // The wait comes right before the suffix's launch; a replay
                // validates after it, while the suffix runs.
                let embeddings_ready = graph && c.ple_layer.is_some();
                let body = Body {
                    n_logits,
                    first_layer,
                    end_layer: c.num_layers,
                    embeddings_ready,
                    ..body
                };
                self.run_with(
                    session,
                    key,
                    graph,
                    |core, stream, session| core.body(stream, session, body),
                    |core, stream| {
                        if embeddings_ready {
                            let mut pass = core.pass(stream, 0);
                            pass.wait_ple()?;
                            pass.upload_ple(n)?;
                        }
                        Ok(())
                    },
                )
            });
        self.core.route.prefill = false;
        // Only this forward's PLE layer may start the named successor's read.
        self.core.prefetch.next = None;
        result.map_err(|e| e.to_string())?;
        if let Some(checkpoint) = checkpoint {
            let storage = self
                .core
                .checkpoint
                .as_ref()
                .ok_or("checkpoint storage is missing")?;
            self.stream
                .download(storage.logits(), &mut checkpoint.logits)
                .map_err(|e| format!("checkpoint logits: {e}"))?;
        }
        if let Some(logits) = logits {
            let count = (n_logits * c.vocab_size) as usize;
            let Core { s, host, .. } = &mut self.core;
            download_logits(&self.stream, s.logits.slice(0..count), host, logits)
                .map_err(|e| format!("logits download: {e}"))?;
        } else {
            self.stream
                .synchronize()
                .map_err(|e| format!("forward: {e}"))?;
        }
        session.position += n;
        if sparse {
            session.blocks = complete;
        }
        if session.mtp_enabled && prefill && n > 1 {
            // Every successor but the last is known: the draft block catches
            // up while the trunk residual is still in scratch. Prefill needs
            // only the draft K/V; the next draft starts from a known row.
            self.mtp_forward_from(
                session,
                &tokens[1..],
                0,
                MtpOutput {
                    kv_only: true,
                    ..MtpOutput::default()
                },
                HiddenSource::Trunk,
            )?;
        }
        Ok(())
    }

    /// Runs `body` eagerly, or as the session's graph for `key` (`Run`).
    pub(crate) fn run(
        &mut self,
        session: &mut Session,
        key: u64,
        graph: bool,
        body: impl FnMut(&mut Core, &Stream, &mut Session) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.run_graph(session, key, graph, false, body, |_, _| Ok(()))
    }

    /// [`run`](Self::run) of a forward's target pass, with `before_launch`
    /// ahead of the body's work. A capture enqueues the pass first and
    /// records its launches while the device executes it, as `Run` does for
    /// a warmed target shape ([`gufo_hip::GraphCache::run_overlapped`]).
    pub(crate) fn run_with(
        &mut self,
        session: &mut Session,
        key: u64,
        graph: bool,
        body: impl FnMut(&mut Core, &Stream, &mut Session) -> Result<(), Error>,
        before_launch: impl FnOnce(&mut Core, &Stream) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.run_graph(session, key, graph, true, body, before_launch)
    }

    fn run_graph(
        &mut self,
        session: &mut Session,
        key: u64,
        graph: bool,
        overlapped: bool,
        mut body: impl FnMut(&mut Core, &Stream, &mut Session) -> Result<(), Error>,
        before_launch: impl FnOnce(&mut Core, &Stream) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if !graph {
            before_launch(&mut self.core, &self.stream)?;
            return body(&mut self.core, &self.stream, session);
        }
        let message = RefCell::new(None);
        let to_launch = |e: Error| match e {
            Error::Launch(error, _) => error,
            Error::Other(text) => {
                *message.borrow_mut() = Some(text);
                LaunchError::InvalidArgument {
                    param: "forward",
                    reason: "the forward failed",
                }
            }
        };
        let mut graphs = std::mem::take(&mut session.graphs);
        let stream = &self.stream;
        let mut context = (&mut self.core, &mut *session);
        let body = |(core, session): &mut (&mut Core, &mut Session), stream: &Stream| {
            body(core, stream, session).map_err(to_launch)
        };
        let before_launch = |(core, _): &mut (&mut Core, &mut Session)| {
            before_launch(core, stream).map_err(to_launch)
        };
        let result = if overlapped {
            // The producer readiness and the alternating scratch the pass
            // leaves are what the recorded launches start from (`Run`'s
            // dispatch tuple).
            graphs.run_overlapped(
                stream,
                key,
                &mut context,
                body,
                before_launch,
                |(core, _)| core.dispatch(),
                |(core, _), dispatch| core.set_dispatch(dispatch),
            )
        } else {
            graphs.run_with(stream, key, &mut context, body, before_launch)
        };
        session.graphs = graphs;
        result.map(|_| ()).map_err(|e| match message.into_inner() {
            Some(text) => Error::Other(text),
            None => Error::Launch(e, "graph"),
        })
    }

    /// Keeps the first `keep` (1..=n) tokens of the last speculative batch
    /// and copies the kept frontier's logits into `logits` when given
    /// (`Rollback`).
    pub fn rollback(
        &mut self,
        session: &mut Session,
        keep: u32,
        logits: Option<&mut [f32]>,
    ) -> Result<(), String> {
        let c = self.config().clone();
        let n = session.spec_tokens;
        if n == 0 || keep == 0 || keep > n {
            return Err("rollback outside the pending speculative batch".into());
        }
        session.spec_tokens = 0;
        let vocab = c.vocab_size as usize;
        if keep < n {
            let slot = (keep - 1) as usize;
            let conv_len = (c.ssm_conv_kernel as usize - 1) * c.ssm_conv_channels() as usize;
            let state_len = c.ssm_num_v_heads as usize * (c.ssm_head_dim as usize).pow(2);
            let rest_len = gdn_rollback_row_floats(1, &c);
            for (il, layer) in session.linear.iter_mut().enumerate() {
                let Some(layer) = layer else { continue };
                let (first, rest) = session
                    .rollback
                    .split_first()
                    .ok_or("missing rollback rows")?;
                let (_, state_first) = first.layers[il].ok_or("rollback row of a linear layer")?;
                let rest: Vec<_> = rest[..slot]
                    .iter()
                    .map(|row| {
                        let (_, state_at) = row.layers[il].expect("rollback row of a linear layer");
                        row.buffer.slice(state_at..state_at + rest_len)
                    })
                    .collect();
                crate::routes::launched(
                    k::restore_gdn_state(
                        &self.stream,
                        layer.state.as_mut_slice(),
                        first.buffer.slice(state_first..state_first + state_len),
                        &rest,
                        rest.len() as u32,
                        keep,
                        c.ssm_num_k_heads,
                        c.ssm_num_v_heads,
                    ),
                    "state rollback failed",
                )
                .map_err(|e| e.to_string())?;
                let row = &session.rollback[slot];
                let (conv_at, _) = row.layers[il].ok_or("rollback row of a linear layer")?;
                self.stream
                    .copy(
                        row.buffer.slice(conv_at..conv_at + conv_len),
                        layer.conv_state.as_mut_slice(),
                    )
                    .map_err(|e| format!("conv rollback: {e}"))?;
            }
            if let Some(history) = &mut session.ple_history {
                let row = &session.rollback[slot];
                let at = row.ple.ok_or("rollback row without PLE history")?;
                let len = history.len();
                self.stream
                    .copy(row.buffer.slice(at..at + len), history.as_mut_slice())
                    .map_err(|e| format!("PLE rollback: {e}"))?;
                session.ngram = session.ngram_snapshots[slot];
            }
            session.position = session.spec_base + keep;
            // Pooled block keys past the kept prefix are stale; they are
            // rebuilt from the raw keys when needed.
            session.blocks = session.blocks.min(session.position / c.compress_ratio);
        }
        match logits {
            Some(logits) => {
                let at = (keep - 1) as usize * vocab;
                let Core {
                    s,
                    host,
                    selected,
                    batch_logits,
                    ..
                } = &mut self.core;
                let rows = verification_logits(
                    *selected,
                    batch_logits,
                    &s.logits,
                    c.vocab_size,
                    at + vocab,
                );
                download_logits(&self.stream, rows.range(at..at + vocab), host, logits)
                    .map_err(|e| format!("frontier download: {e}"))
            }
            None => self
                .stream
                .synchronize()
                .map_err(|e| format!("rollback: {e}")),
        }
    }

    /// Allocates rollback rows for batches of up to `depth + 1` tokens.
    pub fn ensure_rollback(&self, session: &mut Session, depth: u32) -> Result<(), String> {
        let limit = self.core.options.max_speculative.min(ROLLBACK_ROWS + 1);
        session.ensure_rollback(&self.stream, self.config(), depth, limit)
    }
}

/// The state a pass leaves for the next launches: producer readiness and
/// which rows the staging buffers hold.
#[derive(Clone, Copy)]
pub(crate) struct Dispatch {
    route: Route,
    q8_slot: usize,
    q8_src: [Option<(usize, u32, u32)>; 2],
    q8t_src: Option<(usize, u32, usize)>,
    half_src: Option<(usize, u32, usize, bool)>,
}

impl Core {
    pub(crate) fn dispatch(&self) -> Dispatch {
        Dispatch {
            route: self.route,
            q8_slot: self.stage.q8_slot,
            q8_src: self.stage.q8_src,
            q8t_src: self.stage.q8t_src,
            half_src: self.stage.half_src,
        }
    }

    pub(crate) fn set_dispatch(&mut self, dispatch: Dispatch) {
        self.route = dispatch.route;
        self.stage.q8_slot = dispatch.q8_slot;
        self.stage.q8_src = dispatch.q8_src;
        self.stage.q8t_src = dispatch.q8t_src;
        self.stage.half_src = dispatch.half_src;
    }

    /// Hashes the batch's n-gram rows and starts reading them from disk, so
    /// the read overlaps the layers before the PLE layer (`PleFetch`).
    fn ple_fetch(
        &mut self,
        stream: &Stream,
        session: &mut Session,
        tokens: &[i32],
        speculative: bool,
    ) -> Result<(), String> {
        if self.ple_pending {
            self.pass(stream, 0).wait_ple().map_err(|e| e.to_string())?;
        }
        let ngram = Arc::clone(self.ngram.as_ref().ok_or("n-gram table is not open")?);
        let prefetch = &mut self.prefetch;
        if !speculative
            && (prefetch.pending || prefetch.ready)
            && session.ngram == prefetch.before
            && tokens == prefetch.tokens
        {
            // These are this batch's rows: the PLE layer collects them.
            session.ngram = prefetch.after;
            prefetch.claimed = true;
            self.ple_pending = true;
            return Ok(());
        }
        prefetch.finish(&ngram);
        let c = self.model.config.clone();
        let heads = c.ple_heads as usize;
        let rows = &mut self.host.ple_rows[..tokens.len() * heads];
        // Only proper prefixes need snapshots; the full batch keeps its
        // live state.
        if speculative {
            for (i, token) in tokens.iter().enumerate() {
                hash_ngram_rows(
                    &c,
                    &mut session.ngram,
                    std::slice::from_ref(token),
                    &mut rows[i * heads..(i + 1) * heads],
                );
                if i + 1 < tokens.len() {
                    session.ngram_snapshots[i] = session.ngram;
                }
            }
        } else {
            hash_ngram_rows(&c, &mut session.ngram, tokens, rows);
        }
        self.ple_pending = ngram.start_read(rows);
        if !self.ple_pending {
            return Err("n-gram table read could not start".into());
        }
        Ok(())
    }

    /// Uploads the tokens and the device control values of a forward.
    fn upload_inputs(
        &mut self,
        stream: &Stream,
        session: &mut Session,
        tokens: &[i32],
        start_pos: u32,
    ) -> Result<(), Error> {
        let n = tokens.len();
        let host = self.host.tokens.host_mut()?;
        for (slot, &t) in host.iter_mut().zip(tokens) {
            *slot = t as u32;
        }
        stream.upload_indices_async(&mut self.host.tokens, n, &mut self.s.tokens)?;
        session.control.set_position(stream, n as u32, start_pos)?;
        session.control.set_blocks(stream, session.blocks)?;
        Ok(())
    }

    /// Forgets which rows the staging buffers hold and what the previous
    /// projections left for the next, so every body (and its graph replay)
    /// starts from the same state.
    pub(crate) fn reset_route(&mut self) {
        self.use_scratch();
        self.stage.q8_slot = 0;
        self.stage.q8_src = [None; 2];
        self.route.xn_q8 = None;
        self.route.mixed_q8 = None;
        self.route.shexp_half_ready = false;
        self.route.inject_parts = 1;
        self.route.defer_residual = false;
        self.route.res_deferred = false;
        self.route.inject_fused = None;
    }

    /// Forgets which rows the staging buffers hold when another scratch
    /// view takes over; the inject layout of the last mixer stays
    /// (`UseScratch`).
    pub(crate) fn use_scratch(&mut self) {
        self.stage.q8t_src = None;
        self.stage.half_src = None;
        self.route.xn_half = false;
        self.route.moe_pending = false;
    }

    pub(crate) fn pass<'a>(&'a mut self, stream: &'a Stream, row: usize) -> Pass<'a> {
        Pass {
            ctx: Ctx {
                stream,
                model: &self.model,
                blaslt: &mut self.blaslt,
                stage: &mut self.stage,
                route: &mut self.route,
                max_batch: self.options.max_batch,
                yarn: self.yarn,
            },
            s: &mut self.s,
            host: &mut self.host,
            ngram: self.ngram.as_deref(),
            ple_pending: &mut self.ple_pending,
            prefetch: &mut self.prefetch,
            w: self.widths,
            row,
            inject_at: None,
            mask_words: self.mask_words,
            wide_mixer: self.wide_mixer,
            checkpoint: None,
        }
    }

    /// One trunk batch (`ForwardBody`), control and token uploads excluded.
    pub(crate) fn body(
        &mut self,
        stream: &Stream,
        session: &mut Session,
        b: Body,
    ) -> Result<(), Error> {
        let mut storage = if b.checkpoint != 0 {
            Some(
                self.checkpoint
                    .take()
                    .ok_or("checkpoint storage is missing")?,
            )
        } else {
            None
        };
        let result = self.body_with(stream, session, b, storage.as_mut());
        if storage.is_some() {
            self.checkpoint = storage;
        }
        result
    }

    fn body_with(
        &mut self,
        stream: &Stream,
        session: &mut Session,
        b: Body,
        mut storage: Option<&mut CheckpointStorage>,
    ) -> Result<(), Error> {
        let c = self.model.config.clone();
        let model = Arc::clone(&self.model);
        // Buffer choices restart with every body, so a replay validates
        // against its capture.
        self.reset_route();
        let n = b.n;
        let max_context = session.max_context;
        let index_capacity = session.index_capacity;
        if b.first_layer == 0 {
            let token_embd = &model.token_embd;
            k::embed_tokens(
                stream,
                token_embd.view(),
                self.s.tokens.slice(0..n as usize),
                at_mut(&mut self.s.res, 0),
                n,
                c.hidden_size,
                c.hc_count,
                token_embd.rows,
            )
            .map_err(launch_error("token embedding"))?;
            session
                .vision
                .inject(
                    stream,
                    at_mut(&mut self.s.res, 0),
                    b.start_pos,
                    n,
                    c.hidden_size,
                    c.hc_count,
                )
                .map_err(launch_error("image injection"))?;
        }
        let mut normed = false;
        // Without MTP the last layer serves only the head rows: its attention outputs the final query tile (every K/V
        // and indexer row is still written), and its FFN runs on the trailing
        // rows, at least 96 and aligned to 128 to keep the wide arithmetic.
        let prune_last = self.route.prefill && session.mtp.is_none();
        let hc_dim = c.hc_dim() as usize;
        for il in b.first_layer..b.end_layer {
            let last = il + 1 == c.num_layers;
            if session.check_cancelled() {
                return Err("generation cancelled".into());
            }
            let layer: &Layer = &model.layers[il as usize];
            let Session {
                linear,
                attention,
                ple_history,
                rollback,
                control,
                vision,
                ..
            } = session;
            let mut pass = self.pass(stream, 0);
            pass.checkpoint = storage.as_deref_mut().map(|storage| Checkpoint {
                tokens: b.checkpoint,
                storage,
            });
            if c.is_ple_layer(il) {
                let p = layer.ple.as_ref().ok_or("PLE layer without its weights")?;
                let history = ple_history
                    .as_mut()
                    .ok_or("PLE layer without its history")?;
                pass.ple(p, history, rollback, n, b.speculative, b.embeddings_ready)?;
            }
            // Both stacked SSM projections of a wide linear layer read the
            // same F16 rows: the mixer then writes no F32 or Q8 copy.
            let half_only = match &layer.block {
                Block::Linear(l) => {
                    pass.ctx.wide_prefill(n)
                        && n <= pass.ctx.max_batch
                        && l.ssm_in.as_ref().is_some_and(|ssm_in| {
                            pass.ctx.dense_f16_route(ssm_in, n)
                                && l.alpha_beta.format() == gufo_formats::GgmlType::F16
                                && l.alpha_beta.cols == ssm_in.cols
                        })
                }
                Block::Attention(_) => false,
            };
            pass.hc_mix(&layer.hc_attn, 0, normed, true, n, half_only, false)?;
            match &layer.block {
                Block::Linear(l) => {
                    let state = linear[il as usize]
                        .as_mut()
                        .ok_or("linear layer without state")?;
                    pass.linear(
                        l,
                        state,
                        rollback,
                        il as usize,
                        n,
                        b.speculative,
                        false,
                        true,
                    )?;
                }
                Block::Attention(a) => {
                    let cache = attention[il as usize]
                        .as_mut()
                        .ok_or("attention layer without caches")?;
                    let position = control.position(n).ok_or("position not uploaded")?;
                    let attention_control = AttentionControl {
                        position: position.view(),
                        first_block: control.blocks().view(),
                        start_pos: b.start_pos,
                        pool_grid: b.pool_grid,
                        max_context,
                        index_capacity,
                        sparse: b.sparse,
                        rope: vision.rope(),
                    };
                    let last_only =
                        prune_last && last && n > VEC_BATCH && b.n_logits == 1 && b.checkpoint == 0;
                    pass.attention(a, cache, &attention_control, n, last_only, false, true)?;
                }
            }
            // The checkpoint frontier needs its row too.
            let needed = 96.max(b.n_logits).max(if b.checkpoint != 0 {
                n - b.checkpoint + 1
            } else {
                0
            });
            let skipped = if prune_last && last && n > needed {
                (n - needed) / 128 * 128
            } else {
                0
            };
            if skipped != 0 {
                // The tail keeps the mixer's inject stride.
                let inject_parts = pass.ctx.route.inject_parts as usize;
                pass.row = skipped as usize;
                pass.inject_at = Some(skipped as usize * c.hc_count as usize * inject_parts);
                pass.ctx.stage.q8t_src = None;
                pass.ctx.stage.half_src = None;
            }
            let rows = n - skipped;
            // When the FFN combine fuses the MoE epilogue, this combine may
            // leave its residual update to it.
            let moe = &layer.moe;
            pass.ctx.route.defer_residual = pass.wmma_experts(moe, rows);
            // The wide FFN then reads the mix only through its F16 rows
            // (router, routed experts) and tiled Q8 rows (shared expert).
            let no_f32 = pass.ctx.route.prefill
                && pass.ctx.route.defer_residual
                && rows <= pass.ctx.max_batch
                && moe.router.format() == gufo_formats::GgmlType::F16
                && moe.shexp_gate.format() == gufo_formats::GgmlType::Q8_0
                && moe.shexp_up.format() == gufo_formats::GgmlType::Q8_0;
            // Each combine also norms the residual for the mixer after it,
            // unless PLE rewrites the residual first.
            pass.combine(
                Some(&layer.hc_ffn.norm),
                Some(&layer.hc_ffn.down),
                rows,
                inject_weights(&layer.hc_ffn),
            )?;
            pass.hc_mix(
                &layer.hc_ffn,
                skipped as usize * hc_dim,
                true,
                true,
                rows,
                false,
                no_f32,
            )?;
            pass.moe(&layer.moe, rows, false)?;
            // The head mixer norms its rows itself: the last combine needs no
            // norm.
            let next_norm: Option<&DeviceTensor<f32>> = if last || c.is_ple_layer(il + 1) {
                None
            } else {
                Some(&model.layers[il as usize + 1].hc_attn.norm)
            };
            let next_down = next_norm
                .is_some()
                .then(|| &model.layers[il as usize + 1].hc_attn.down);
            let next_inject =
                next_norm.and_then(|_| inject_weights(&model.layers[il as usize + 1].hc_attn));
            pass.combine(next_norm, next_down, rows, next_inject)?;
            normed = next_norm.is_some();
        }
        if b.end_layer < c.num_layers {
            return Ok(());
        }
        if let Some(storage) = storage {
            // The checkpoint's kept trunk rows and the boundary token's
            // logits, as a pass ending there leaves them.
            let tokens = b.checkpoint as usize;
            if session.mtp_enabled && session.mtp.is_some() {
                let rows = tokens.min(self.options.max_speculative as usize);
                let from = (tokens - rows) * hc_dim;
                stream.graph_copy(
                    self.s.res.slice(from..tokens * hc_dim),
                    storage.hidden_mut().range(0..rows * hc_dim),
                )?;
            }
            let prefill = std::mem::replace(&mut self.route.prefill, false);
            let mut pass = self.pass(stream, 0);
            let result = pass
                .hc_mix(
                    &model.hc_head,
                    (tokens - 1) * hc_dim,
                    false,
                    false,
                    1,
                    false,
                    false,
                )
                .and_then(|()| {
                    pass.ctx
                        .dense(&model.output, at(&pass.s.mixed, 0), storage.logits_mut(), 1)
                });
            self.route.prefill = prefill;
            result?;
        }
        // Prefill catches up from shared scratch at once; only the short tail
        // needed across calls (past a checkpoint) belongs to the session.
        let tail = if b.checkpoint != 0 && b.checkpoint < n {
            n - b.checkpoint
        } else {
            n
        };
        let kept = tail.min(self.options.max_speculative);
        if let Some(mtp) = &mut session.mtp {
            let from = (n - kept) as usize * hc_dim;
            stream.graph_copy(
                self.s.res.slice(from..from + kept as usize * hc_dim),
                mtp.target_hidden.slice_mut(0..kept as usize * hc_dim),
            )?;
        }
        if b.n_logits > 0 {
            // The head mixer norms its tail rows itself: the last combine's
            // norm covers the whole batch (and is tiled on the wide route).
            let prefill = std::mem::replace(&mut self.route.prefill, false);
            let skip = (n - b.n_logits) as usize;
            let mut pass = self.pass(stream, 0);
            let result = pass
                .hc_mix(
                    &model.hc_head,
                    skip * hc_dim,
                    false,
                    false,
                    b.n_logits,
                    false,
                    false,
                )
                .and_then(|()| {
                    pass.ctx.dense(
                        &model.output,
                        at(&pass.s.mixed, 0),
                        at_mut(&mut pass.s.logits, 0),
                        b.n_logits,
                    )
                });
            self.route.prefill = prefill;
            result?;
        }
        Ok(())
    }
}

/// The logit rows verification reads: a selected batch row, or the
/// forward's own rows (`VerificationLogits`); `len` floats.
pub(crate) fn verification_logits<'a>(
    selected: Option<u32>,
    batch: &'a Option<gufo_hip::DeviceBuffer<f32>>,
    logits: &'a gufo_hip::DeviceBuffer<f32>,
    vocab: u32,
    len: usize,
) -> gufo_hip::DeviceSlice<'a, f32> {
    match (selected, batch) {
        (Some(row), Some(rows)) => {
            let at = row as usize * vocab as usize;
            rows.slice(at..(at + len).min(rows.len()))
        }
        _ => logits.slice(0..len.min(logits.len())),
    }
}

/// Copies logit rows to `out` through the pinned staging and waits for them
/// (the stream's queued work comes first), as the C++ executor's
/// `hipMemcpyAsync` of `logits_host_` does.
pub(crate) fn download_logits(
    stream: &Stream,
    rows: gufo_hip::DeviceSlice<'_, f32>,
    host: &mut Host,
    out: &mut [f32],
) -> Result<(), gufo_hip::HipError> {
    let count = rows.len();
    stream.download_async_front(rows, &mut host.logits)?;
    out[..count].copy_from_slice(&host.logits.host()?[..count]);
    Ok(())
}

/// Device bytes of the batched decode's logit rows.
fn batch_logit_bytes(c: &Config, options: Options) -> usize {
    8 * options.max_logit_rows.min(8) as usize * c.vocab_size as usize * 4
}

/// Device scratch a fresh executor may still allocate: what
/// [`Executor::deferred_scratch_bytes`] reports before any of it exists.
fn lazy_scratch_bytes(c: &Config, options: Options, has_mtp: bool) -> usize {
    batch_logit_bytes(c, options)
        + CheckpointStorage::bytes(c, options.max_speculative, has_mtp)
        + SPARE_BLOCKS * CheckpointStorage::mutable_bytes(c)
}
