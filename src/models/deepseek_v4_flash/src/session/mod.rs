//! Sessions (`runtime/session.cpp`): a checkpoint of evaluated tokens, the
//! logits after it, and the graph whose caches hold it. `sync` extends a
//! matching checkpoint (batched from four new tokens, else token by token)
//! or prefills from scratch.

use gufo_deepseek_v4_flash_kernels::{Result as GpuResult, SnapshotReader};

use crate::dspark_policy::RequestState;
use crate::engine::Engine;

mod dspark;

use crate::graph::{
    BatchItem, Graph, eval_sessions_batch, prefill_capacity, resume_prefill_min_tokens,
};
pub use dspark::{DsparkItem, DsparkStats};

/// A failed session call: the C++ message, and whether a cancellation
/// interrupted it (`DS4_SESSION_SYNC_INTERRUPTED`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionError {
    pub message: &'static str,
    pub interrupted: bool,
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for SessionError {}

fn error(message: &'static str) -> SessionError {
    SessionError {
        message,
        interrupted: false,
    }
}

fn interrupted(message: &'static str) -> SessionError {
    SessionError {
        message,
        interrupted: true,
    }
}

/// One session of an engine (`ds4_session`).
pub struct Session {
    pub graph: Graph,
    checkpoint: Vec<i32>,
    logits: Vec<f32>,
    prefill_capacity: u32,
    context_size: u32,
    checkpoint_valid: bool,
    cancel: Option<Box<dyn Fn() -> bool + Send>>,
    /// A support model drafts for this session (`use_dspark`).
    use_dspark: bool,
    dspark_state: RequestState,
    /// Every verified row's logits of a sampled cycle.
    spec_rows: Vec<f32>,
}

/// `ds4_sample_argmax`: the first maximum among the finite logits, or -1.
pub fn argmax(logits: &[f32]) -> i32 {
    let mut best = -1;
    let mut maximum = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > maximum {
            maximum = v;
            best = i as i32;
        }
    }
    best
}

impl Engine {
    /// `ds4_session_create`: a session of `context_size` positions; a
    /// speculative one drafts with the support model.
    pub fn create_session(&mut self, context_size: u32, speculative: bool) -> GpuResult<Session> {
        if speculative && self.dspark.is_none() {
            return Err(gufo_deepseek_v4_flash_kernels::Error {
                op: "speculative session",
            });
        }
        let prefill_capacity = prefill_capacity(context_size);
        let graph = Graph::create(
            &self.backend,
            &self.weights,
            &mut self.workspaces,
            context_size,
            prefill_capacity,
            self.dspark.as_ref().map(|d| d.block_size),
        )?;
        let mut graph = graph;
        if let (true, Some(dspark)) = (speculative, &self.dspark) {
            // Prefill captures target features and seeds the support KV.
            graph.dspark_attach(&self.backend, &self.weights, dspark)?;
        }
        Ok(Session {
            graph,
            checkpoint: Vec::new(),
            logits: vec![0.0; self.vocab.len()],
            prefill_capacity,
            context_size,
            checkpoint_valid: false,
            cancel: None,
            use_dspark: speculative,
            dspark_state: RequestState::default(),
            spec_rows: Vec::new(),
        })
    }
}

impl Session {
    /// The callback a long call polls to stop early
    /// (`ds4_session_set_cancel`).
    pub fn set_cancellation(&mut self, cancel: Option<Box<dyn Fn() -> bool + Send>>) {
        self.cancel = cancel;
    }

    fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c())
    }

    pub fn position(&self) -> usize {
        self.checkpoint.len()
    }

    pub fn context_size(&self) -> u32 {
        self.context_size
    }

    pub fn prefill_capacity(&self) -> u32 {
        self.prefill_capacity
    }

    pub fn checkpoint(&self) -> &[i32] {
        &self.checkpoint
    }

    /// `ds4_session_invalidate`.
    pub fn invalidate(&mut self) {
        self.checkpoint_valid = false;
        self.reset_dspark_request_state(false);
    }

    /// `session_dspark_reset_request_state`: a new request's controller
    /// state; sessions without a support model, and requests forced plain
    /// when `preserve_force_plain`, stay plain.
    fn reset_dspark_request_state(&mut self, preserve_force_plain: bool) {
        let force_plain =
            !self.use_dspark || (preserve_force_plain && self.dspark_state.force_plain_request);
        self.dspark_state = RequestState {
            force_plain_request: force_plain,
            plain_only: force_plain,
            ..RequestState::default()
        };
        self.graph.dspark_set_capture_enabled(!force_plain);
    }

    /// `ds4_session_begin_request`: resets the request statistics and
    /// draft policy, keeping a restored prefix.
    pub fn begin_request(&mut self) {
        let incomplete_support =
            self.use_dspark && (self.graph.dspark_context_len() as usize) < self.checkpoint.len();
        self.reset_dspark_request_state(false);
        if incomplete_support {
            self.prepare_batch_execution();
        }
    }

    /// `ds4_session_prepare_batch_execution`: drops the DSpark-only state
    /// before exact multi-session execution.
    pub fn prepare_batch_execution(&mut self) {
        if self.graph.dspark_block_size() == 0 {
            return;
        }
        self.dspark_state.force_plain_request = true;
        self.dspark_state.plain_only = true;
        self.graph.dspark_set_capture_enabled(false);
    }

    /// `ds4_session_dspark_enabled`.
    pub fn dspark_enabled(&self) -> bool {
        self.use_dspark
    }

    /// The DSpark controller state of the current request.
    pub fn dspark_state(&self) -> &RequestState {
        &self.dspark_state
    }

    /// `ds4_session_payload_bytes`: the snapshot size, 0 without a valid
    /// checkpoint.
    pub fn payload_bytes(&self) -> u64 {
        if !self.checkpoint_valid {
            return 0;
        }
        self.graph.snapshot_bytes(self.checkpoint.len())
    }

    /// `ds4_session_save_snapshot`: the payload of the checkpoint (version
    /// 4, byte-identical to C++).
    pub fn save_snapshot(&self, engine: &Engine) -> Result<Vec<u8>, SessionError> {
        self.save_snapshot_with(&engine.backend.snapshot_reader())
    }

    /// `save_snapshot` through `reader`, on a capture thread while other
    /// sessions run; this session is frozen meanwhile.
    pub fn save_snapshot_with(&self, reader: &SnapshotReader) -> Result<Vec<u8>, SessionError> {
        if !self.checkpoint_valid {
            return Err(error("session has no valid checkpoint"));
        }
        self.graph
            .save_payload(
                reader,
                &self.checkpoint,
                &self.logits,
                self.prefill_capacity,
                self.context_size,
                &self.dspark_state,
            )
            .map_err(error)
    }

    /// `ds4_session_load_snapshot`: restores a payload into this session's
    /// caches; the checkpoint is invalid when it fails.
    pub fn restore_snapshot(
        &mut self,
        engine: &Engine,
        payload: &[u8],
    ) -> Result<(), SessionError> {
        if payload.is_empty() {
            return Err(error("DeepSeek session snapshot payload is empty"));
        }
        self.checkpoint_valid = false;
        let restored = self
            .graph
            .load_payload(&engine.backend, payload, self.context_size)
            .map_err(error)?;
        let n = restored.logits.len().min(self.logits.len());
        self.logits[..n].copy_from_slice(&restored.logits[..n]);
        self.checkpoint = restored.checkpoint;
        self.dspark_state = restored.state;
        self.checkpoint_valid = true;
        Ok(())
    }

    /// The logits after the checkpoint (`ds4_session_copy_logits`).
    pub fn logits(&self) -> Option<&[f32]> {
        self.checkpoint_valid.then_some(&self.logits[..])
    }

    /// `ds4_session_argmax`.
    pub fn argmax(&self) -> i32 {
        if !self.checkpoint_valid {
            return -1;
        }
        argmax(&self.logits)
    }

    /// `session_commit_and_extend`: one decode step at the checkpoint's
    /// end, extending the support ring with the features it captured.
    fn commit(&mut self, engine: &mut Engine, token: i32) -> GpuResult<()> {
        let position = self.checkpoint.len() as u32;
        self.graph
            .eval_token(&engine.target(), token, position, Some(&mut self.logits))?;
        self.checkpoint.push(token);
        if self.graph.dspark_block_size() != 0 && self.graph.dspark_capture_ready() {
            self.dspark_inject(engine, position, 1)?;
        }
        Ok(())
    }

    /// The captured features of `rows` positions from `pos0` into the
    /// support ring, through the workspace the engine lends this session.
    fn dspark_inject(&mut self, engine: &mut Engine, pos0: u32, rows: u32) -> GpuResult<()> {
        let (target, ws) = engine.lend(&self.graph.workspace)?;
        self.graph.dspark_inject(&target, ws, pos0, rows)
    }

    /// `ds4_session_sync`: makes the checkpoint `prompt`.
    pub fn sync(&mut self, engine: &mut Engine, prompt: &[i32]) -> Result<(), SessionError> {
        if prompt.is_empty() || prompt.len() >= self.context_size as usize {
            return Err(error("prompt exceeds context"));
        }
        let vocabulary = engine.vocab.len() as i32;
        if prompt.iter().any(|&t| t < 0 || t >= vocabulary) {
            return Err(error("prompt contains an invalid token ID"));
        }
        if self.cancelled() {
            return Err(interrupted("prefill cancelled"));
        }
        if self.checkpoint_valid
            && prompt.len() >= self.checkpoint.len()
            && prompt.starts_with(&self.checkpoint)
        {
            let start = self.checkpoint.len();
            let suffix = prompt.len() - start;
            if suffix > 0 {
                // A pending capture is materialized before the suffix reuses
                // its storage.
                let _ = self.seed_pending(engine, start as u32);
            }
            if suffix > 0 && suffix as u32 >= resume_prefill_min_tokens() {
                let resumed = engine.lend(&self.graph.workspace).and_then(|(target, ws)| {
                    self.graph.prefill_range(
                        &target,
                        ws,
                        prompt,
                        start as u32,
                        suffix as u32,
                        Some(&mut self.logits),
                    )
                });
                if resumed.is_err() {
                    self.checkpoint_valid = false;
                    return Err(error(
                        "ROCm resumed prefill failed while extending checkpoint",
                    ));
                }
                self.checkpoint = prompt.to_vec();
                self.checkpoint_valid = true;
                return Ok(());
            }
            for &token in &prompt[start..] {
                if self.cancelled() {
                    return Err(interrupted("prefill cancelled"));
                }
                if self.commit(engine, token).is_err() {
                    self.checkpoint_valid = false;
                    return Err(error("ROCm decode failed while extending checkpoint"));
                }
            }
            return Ok(());
        }
        self.checkpoint_valid = false;
        self.reset_dspark_request_state(true);
        if self.graph.reset(&engine.backend).is_err() {
            return Err(error("ROCm prefill state reset failed"));
        }
        let prefilled = engine.lend(&self.graph.workspace).and_then(|(target, ws)| {
            self.graph
                .prefill(&target, ws, prompt, Some(&mut self.logits))
        });
        if prefilled.is_err() {
            return Err(error("ROCm prefill failed"));
        }
        self.checkpoint = prompt.to_vec();
        self.checkpoint_valid = true;
        Ok(())
    }

    /// `ds4_session_eval`: one token after the checkpoint.
    pub fn eval(&mut self, engine: &mut Engine, token: i32) -> Result<(), SessionError> {
        if !self.checkpoint_valid
            || self.checkpoint.len() >= self.context_size as usize
            || token < 0
            || token >= engine.vocab.len() as i32
        {
            return Err(error(
                "decode needs a valid checkpoint, token, and context position",
            ));
        }
        if self.cancelled() {
            return Err(interrupted("decode cancelled"));
        }
        if self.commit(engine, token).is_err() {
            self.checkpoint_valid = false;
            return Err(error("ROCm decode failed"));
        }
        self.checkpoint_valid = true;
        Ok(())
    }

    /// `ds4_sessions_eval_batch`: one token after the checkpoint of each of
    /// two to eight sessions, run as one batch. The borrows make the
    /// sessions distinct (the duplicate check of C++), and the engine is the
    /// one engine of the process.
    pub fn eval_batch(
        engine: &mut Engine,
        items: &mut [(&mut Session, i32)],
    ) -> Result<(), SessionError> {
        if !(2..=8).contains(&items.len()) {
            return Err(error("batch decode requires two to eight sessions"));
        }
        let vocabulary = engine.vocab.len() as i32;
        for (session, token) in items.iter() {
            if !session.checkpoint_valid {
                return Err(error("batch decode contains an invalid session"));
            }
            if session.cancelled() {
                return Err(interrupted("batch decode cancelled"));
            }
            if *token < 0 || *token >= vocabulary {
                return Err(error("batch decode contains an invalid token"));
            }
            if session.checkpoint.len() >= session.context_size as usize {
                return Err(error("batch decode exceeds a session context"));
            }
        }
        let mut batch: Vec<BatchItem<'_>> = items
            .iter_mut()
            .map(|(session, token)| BatchItem {
                position: session.checkpoint.len() as u32,
                token: *token,
                graph: &mut session.graph,
                logits: &mut session.logits,
            })
            .collect();
        let (target, workspaces) = engine.lend_all();
        let result = eval_sessions_batch(&target, workspaces, &mut batch);
        drop(batch);
        for (session, token) in items.iter_mut() {
            match result {
                Ok(()) => {
                    session.checkpoint.push(*token);
                    // The batch does not capture support features; the
                    // request stays on exact target batching.
                    session.dspark_state.plain_only = true;
                    session.dspark_state.force_plain_request = true;
                    session.graph.dspark_set_capture_enabled(false);
                }
                Err(_) => session.checkpoint_valid = false,
            }
        }
        result.map_err(|_| error("ROCm batch decode failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::argmax;

    const EXTREME: [f32; 3] = [-3e35, -2e35, -1e35];
    const MIXED: [f32; 5] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -3e35, -3e35];

    #[test]
    fn argmax_covers_the_finite_float_range() {
        assert_eq!(argmax(&EXTREME), 2);
    }

    #[test]
    fn argmax_skips_nonfinite_values_and_keeps_the_first_tie() {
        assert_eq!(argmax(&MIXED), 3);
    }

    /// A C++ null pointer and a zero length are both the empty slice.
    #[test]
    fn argmax_without_a_finite_logit_fails() {
        assert_eq!(argmax(&MIXED[..3]), -1);
        assert_eq!(argmax(&[]), -1);
        assert_eq!(argmax(&EXTREME[..0]), -1);
    }
}
