//! Loading the engine (`ds4_engine_open`): the process lock, the target
//! GGUF with its checks, the tokenizer, the device upload of every tensor
//! span, and the optional DSpark support model.

use std::fs::File;
use std::io::{Read, Seek, Write};
use std::os::fd::AsFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::Instant;

use gufo_deepseek_v4_flash_kernels::{Backend, HashTable, Result as GpuResult};
use rustix::fs::{FlockOperation, flock};

use crate::config;
use crate::dspark_model::DsparkModel;
use crate::gguf::{Model, align_up};
use crate::graph::{BatchWorkspace, Target, WorkspaceLease};
use crate::shape::N_HASH_LAYER;
use crate::tokenizer::Vocab;
use crate::weights::Weights;

/// Largest upload span (`accelerator_rocm_preload_span_bytes`).
const MAX_SPAN: u64 = 1 << 30;
/// Tensors this close join one span, gap included.
const SPAN_GAP: u64 = 65536;

/// The upload spans of `model` (`accelerator_cache_model_tensor_spans`):
/// the sized tensors sorted by offset, neighbours at most 64 KiB apart
/// joined while the span stays within 1 GiB, and each span cut into 1 GiB
/// pieces.
pub fn upload_spans(model: &Model) -> Vec<(u64, u64)> {
    let mut tensors: Vec<(u64, u64)> = model
        .tensors()
        .iter()
        .filter(|t| t.bytes != 0)
        .map(|t| (t.offset, t.offset + t.bytes))
        .collect();
    tensors.sort_unstable();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < tensors.len() {
        let (mut off, mut end) = tensors[i];
        i += 1;
        while i < tensors.len() && tensors[i].0 <= end + SPAN_GAP && tensors[i].1 - off <= MAX_SPAN
        {
            end = end.max(tensors[i].1);
            i += 1;
        }
        while off < end {
            let chunk_end = end.min(off + MAX_SPAN);
            spans.push((off, chunk_end - off));
            off = chunk_end;
        }
    }
    spans
}

/// The exclusive lock of `$DS4_LOCK_FILE` (default `/tmp/ds4.lock`) holding
/// this process's ID: one DeepSeek engine per machine, as in C++.
struct InstanceLock {
    _file: File,
}

impl InstanceLock {
    fn acquire() -> Result<Self, String> {
        let path = std::env::var_os("DS4_LOCK_FILE")
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| "/tmp/ds4.lock".into());
        let unavailable = || "DeepSeek instance lock unavailable".to_owned();
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&path)
            .map_err(|error| {
                eprintln!(
                    "ds4: failed to open lock file {}: {error}",
                    Path::new(&path).display()
                );
                unavailable()
            })?;
        if let Err(error) = flock(file.as_fd(), FlockOperation::NonBlockingLockExclusive) {
            if error == rustix::io::Errno::WOULDBLOCK {
                let mut owner = String::new();
                let _ = file.read_to_string(&mut owner);
                match owner.trim().parse::<i64>() {
                    Ok(pid) if pid > 0 => eprintln!(
                        "ds4: another ds4 process is already running (pid {pid}); refusing to start"
                    ),
                    _ => {
                        eprintln!("ds4: another ds4 process is already running; refusing to start")
                    }
                }
            } else {
                eprintln!(
                    "ds4: failed to lock {}: {error}",
                    Path::new(&path).display()
                );
            }
            return Err(unavailable());
        }
        let pid = std::process::id();
        if file.set_len(0).is_err() || file.rewind().is_err() || writeln!(file, "{pid}").is_err() {
            eprintln!(
                "ds4: failed to truncate lock file {}",
                Path::new(&path).display()
            );
            return Err(unavailable());
        }
        Ok(Self { _file: file })
    }
}

/// A loaded engine: the backend with the resident weights, the bound
/// target and support models, the tokenizer and the checked hash tables.
pub struct Engine {
    pub backend: Backend,
    pub model: Model,
    pub weights: Weights,
    /// Shared with tokenizing threads.
    pub vocab: std::sync::Arc<Vocab>,
    pub hash_tables: Vec<HashTable>,
    pub dspark: Option<DsparkModel>,
    /// Batch workspaces, the largest first; sessions lease them, and the
    /// engine lends a session's for each call.
    pub workspaces: Vec<BatchWorkspace>,
    _lock: InstanceLock,
}

impl Engine {
    /// What the graph encoders read.
    pub fn target(&self) -> Target<'_> {
        Target {
            b: &self.backend,
            weights: &self.weights,
            hash_tables: &self.hash_tables,
            dspark: self.dspark.as_ref(),
        }
    }

    /// What the graph encoders read, and every batch workspace, lent to one
    /// call.
    pub fn lend_all(&mut self) -> (Target<'_>, &mut [BatchWorkspace]) {
        let target = Target {
            b: &self.backend,
            weights: &self.weights,
            hash_tables: &self.hash_tables,
            dspark: self.dspark.as_ref(),
        };
        (target, &mut self.workspaces)
    }

    /// What the graph encoders read, and the batch workspace `lease` names,
    /// lent to one call.
    pub fn lend(&mut self, lease: &WorkspaceLease) -> GpuResult<(Target<'_>, &mut BatchWorkspace)> {
        let (target, workspaces) = self.lend_all();
        let index = crate::graph::lent(workspaces, lease)?;
        Ok((target, &mut workspaces[index]))
    }

    /// `Model::EncodeChat` with messages: the rendered conversation, its
    /// special tokens parsed.
    pub fn encode_chat_messages(
        &self,
        messages: &[crate::chat_template::ChatMessage],
        tools: &[crate::chat_template::ChatTool],
        options: &crate::chat_template::ChatTemplateOptions,
    ) -> Vec<i32> {
        let rendered = crate::chat_template::render_chat(messages, tools, options);
        let mut tokens = Vec::new();
        self.vocab
            .tokenize_rendered_chat(rendered.as_bytes(), &mut tokens);
        tokens
    }

    /// `ds4_engine_open`: refuses a second engine, parses and checks the
    /// GGUF, uploads every tensor span, then loads the support model.
    pub fn open(model_path: &Path, dspark_path: Option<&Path>) -> Result<Self, String> {
        let result = Self::open_locked(model_path, dspark_path);
        if let Err(error) = &result {
            eprintln!("ds4: model load failed: {error}");
        }
        result
    }

    fn open_locked(model_path: &Path, dspark_path: Option<&Path>) -> Result<Self, String> {
        let lock = InstanceLock::acquire()?;
        let model = Model::open(model_path)?;
        let vocab = std::sync::Arc::new(Vocab::load(&model)?);
        config::validate(&model)?;
        let weights = Weights::bind(&model)?;
        let mut backend = Backend::init().map_err(|error| {
            eprintln!("ds4: ROCm backend unavailable; aborting startup");
            error.to_string()
        })?;
        backend.set_model(model.map().clone()).map_err(|error| {
            eprintln!(
                "ds4: ROCm failed to map model views; aborting startup. This is commonly caused by insufficient memory or accelerator VM budget."
            );
            error.to_string()
        })?;
        let start = Instant::now();
        let cached = backend
            .load_target_spans(&upload_spans(&model))
            .map_err(|error| {
                eprintln!("ds4: ROCm failed to prepare startup model cache");
                error.to_string()
            })?;
        if cached != 0 {
            eprintln!(
                "ds4: ROCm startup model cache prepared {:.2} GiB of tensor spans in {:.3}s",
                cached as f64 / 1073741824.0,
                start.elapsed().as_secs_f64()
            );
        }
        // The routed experts index the hash tables' expert IDs unchecked.
        let mut hash_tables = Vec::new();
        for layer in &weights.layers[..N_HASH_LAYER as usize] {
            let table = layer.ffn_gate_tid2eid.as_ref().ok_or(config::INVALID)?;
            hash_tables.push(
                backend
                    .hash_table(table.offset, table.dim[1] as u32)
                    .map_err(|_| {
                        eprintln!("ds4: hash routing table holds an expert ID outside the model");
                        config::INVALID.to_owned()
                    })?,
            );
        }
        let dspark = match dspark_path {
            Some(path) => Some(load_dspark(&mut backend, path).inspect_err(|_| {
                eprintln!(
                    "ds4: failed to load DSpark support model {}",
                    path.display()
                );
            })?),
            None => None,
        };
        Ok(Self {
            backend,
            model,
            weights,
            vocab,
            hash_tables,
            dspark,
            workspaces: Vec::new(),
            _lock: lock,
        })
    }
}

/// `ds4_dspark_open`: binds the support model and copies every tensor into
/// the backend's managed support arena, 256-byte aligned.
fn load_dspark(backend: &mut Backend, path: &Path) -> Result<DsparkModel, String> {
    let result = (|| {
        let support = DsparkModel::bind(Model::open(path)?)?;
        let start = Instant::now();
        let m = &support.model;
        // Faults from the mapping would otherwise make the copy a long,
        // silent phase.
        let data = m.tensor_data_pos as usize..m.size() as usize;
        let _ = m.map().prefault(data);
        let total = m
            .tensors()
            .iter()
            .fold(0, |total, t| align_up(total, 256) + t.bytes);
        let ranges: Vec<(u64, u64, &str)> = m
            .tensors()
            .iter()
            .map(|t| (t.offset, t.bytes, "dspark_tensor"))
            .collect();
        backend
            .set_support(m.map().clone(), total, &ranges)
            .map_err(|_| {
                eprintln!(
                    "ds4: failed to reserve {:.2} GiB for the DSpark support model",
                    total as f64 / 1073741824.0
                );
                config::INVALID.to_owned()
            })?;
        eprintln!(
            "ds4: DSpark support model cached {:.2} GiB of tensor spans in {:.3}s",
            total as f64 / 1073741824.0,
            start.elapsed().as_secs_f64()
        );
        eprintln!(
            "ds4: DSpark support model loaded stages={} block={} markov_rank={} noise_token={} target_layers={}",
            support.n_stages(),
            support.block_size,
            support.markov_rank,
            support.noise_token_id,
            support.target_layer_ids.len()
        );
        Ok(support)
    })();
    if let Err(error) = &result {
        eprintln!("ds4: DSpark load failed: {error}");
    }
    result
}
