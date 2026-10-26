//! The tensors of a `qwen4exp` artifact, a port of `ModelWeights::Bind` and
//! `MtpWeights::Bind` (`weights.cpp`): names, shapes and formats checked
//! against the configuration, located in their (possibly split) GGUF file.
//! Only headers are read; payloads stay on disk until the device upload.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use gufo_core::gguf::{self, Gguf, Tensor};
use gufo_formats::GgmlType;

use crate::config::Config;

/// The header bytes of every file of one GGUF set, in shard order.
pub struct GgufFiles {
    pub paths: Vec<PathBuf>,
    headers: Vec<(Vec<u8>, usize)>,
}

impl GgufFiles {
    /// Reads the headers of the file at `path` and, when it is one shard of
    /// a split set, of every other shard (`GgufReader::OpenFile`).
    pub fn open(path: &Path) -> Result<Self, String> {
        let first = gguf::read_header(path)?;
        let paths = {
            let header = Gguf::parse_prefix(&first.0, first.1)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            gguf::split_paths(path, &header)?
        };
        let mut first = Some(first);
        let mut headers = Vec::with_capacity(paths.len());
        for shard in &paths {
            // `path` names one of the shards; its header is read already.
            let header = match first.take_if(|_| shard.file_name() == path.file_name()) {
                Some(header) => header,
                None => gguf::read_header(shard)
                    .map_err(|e| format!("Failed to open split GGUF shard: {e}"))?,
            };
            headers.push(header);
        }
        let files = Self { paths, headers };
        gguf::check_split(&files.parse()?)?;
        Ok(files)
    }

    /// The parsed headers, in shard order.
    pub fn parse(&self) -> Result<Vec<Gguf<'_>>, String> {
        self.headers
            .iter()
            .zip(&self.paths)
            .map(|((bytes, len), path)| {
                Gguf::parse_prefix(bytes, *len).map_err(|e| format!("{}: {e}", path.display()))
            })
            .collect()
    }
}

/// One tensor: GGUF shape (`cols` is the contiguous dimension ne[0], `rows`
/// ne[1], `experts` stacks whole matrices along ne[2]), format and place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub ty: GgmlType,
    pub cols: u64,
    pub rows: u64,
    pub experts: u64,
    pub shard: usize,
    /// Absolute byte offset in the shard.
    pub offset: usize,
    pub bytes: usize,
}

impl TensorRef {
    /// Encoded bytes of one row.
    pub fn row_bytes(&self) -> usize {
        self.bytes / (self.rows * self.experts) as usize
    }
}

/// Two hyper-connection mixers per layer and the head mixer share this
/// shape; the head mixer has no `inject`.
#[derive(Debug, Clone)]
pub struct MixerRefs {
    /// [hc_dim] grouped RMSNorm gamma.
    pub norm: TensorRef,
    /// [hc_dim -> low_rank]
    pub down: TensorRef,
    /// [low_rank -> hc_dim]
    pub up: TensorRef,
    /// [hc_dim -> hc_count]
    pub inject: Option<TensorRef>,
}

/// Gated DeltaNet projections of a linear layer.
#[derive(Debug, Clone)]
pub struct LinearRefs {
    /// [hidden -> 2 * key_dim + value_dim]
    pub qkv: TensorRef,
    /// [hidden -> value_dim], the z output gate.
    pub gate: TensorRef,
    /// [conv_kernel, channels]
    pub conv1d: TensorRef,
    /// [hidden -> v_heads] each.
    pub alpha: TensorRef,
    pub beta: TensorRef,
    /// [v_heads] softplus bias and -exp(A_log).
    pub dt: TensorRef,
    pub a: TensorRef,
    /// [ssm_head_dim]
    pub norm: TensorRef,
    /// [value_dim -> hidden]
    pub out: TensorRef,
}

/// Gated GQA and the sparse-attention indexer of an attention layer.
#[derive(Debug, Clone)]
pub struct AttentionRefs {
    /// [hidden -> heads * 2 * head_dim], q|gate per head.
    pub q: TensorRef,
    pub k: TensorRef,
    pub v: TensorRef,
    pub out: TensorRef,
    pub q_norm: TensorRef,
    pub k_norm: TensorRef,
    pub indexer_q: TensorRef,
    pub indexer_k: TensorRef,
    pub indexer_q_norm: TensorRef,
    pub indexer_k_norm: TensorRef,
}

/// The PLE n-gram injection of one linear layer.
#[derive(Debug, Clone)]
pub struct PleRefs {
    /// [ple_dim -> hc_dim]
    pub key: TensorRef,
    /// [ple_dim -> hidden]
    pub value: TensorRef,
    pub norm_key: TensorRef,
    pub norm_query: TensorRef,
    pub norm_conv: TensorRef,
    /// [conv_kernel, hc_dim]
    pub conv1d: TensorRef,
}

#[derive(Debug, Clone)]
pub struct MoeRefs {
    /// [hidden -> num_experts] F32.
    pub router: TensorRef,
    /// [hidden -> expert_ff] x experts, and [expert_ff -> hidden].
    pub gate_exps: TensorRef,
    pub up_exps: TensorRef,
    pub down_exps: TensorRef,
    /// [hidden], the shared expert's scalar sigmoid gate.
    pub shexp_gate_inp: TensorRef,
    pub shexp_gate: TensorRef,
    pub shexp_up: TensorRef,
    pub shexp_down: TensorRef,
}

/// The speculative `nextn` block's own tensors.
#[derive(Debug, Clone)]
pub struct NextnRefs {
    /// [hidden]
    pub enorm: TensorRef,
    /// [hc_dim], one RMS denominator across all streams.
    pub hnorm: TensorRef,
    /// GGUF rows [fc_embedding | fc_hidden].
    pub eh_proj: TensorRef,
    /// The draft block's output mixer.
    pub head: MixerRefs,
}

#[derive(Debug, Clone)]
pub enum MixerKind {
    Linear(LinearRefs),
    Attention(AttentionRefs),
}

#[derive(Debug, Clone)]
pub struct LayerRefs {
    pub hc_attn: MixerRefs,
    pub hc_ffn: MixerRefs,
    pub mixer: MixerKind,
    pub ple: Option<PleRefs>,
    pub moe: MoeRefs,
    pub nextn: Option<NextnRefs>,
}

/// The trunk artifact.
#[derive(Debug, Clone)]
pub struct ModelTensors {
    pub config: Config,
    /// [hidden -> vocab]
    pub token_embd: TensorRef,
    /// [hidden -> vocab]; the embedding when the file ties them.
    pub output: TensorRef,
    /// The final mixer, which doubles as the output norm.
    pub hc_head: MixerRefs,
    /// [ple_head_dim, ple_rows], read from disk per token only.
    pub ple_table: Option<TensorRef>,
    pub layers: Vec<LayerRefs>,
}

/// The MTP draft block of the `shared-*` sidecar: one attention layer with
/// its own mixers and experts; embedding and LM head come from the trunk.
#[derive(Debug, Clone)]
pub struct MtpTensors {
    pub config: Config,
    pub block: LayerRefs,
}

const DENSE: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::Bf16, GgmlType::F16, GgmlType::F32];
const EXPERTS: &[GgmlType] = &[
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
];
const F32: &[GgmlType] = &[GgmlType::F32];

impl ModelTensors {
    /// Binds and validates the trunk artifact.
    pub fn bind(shards: &[Gguf<'_>]) -> Result<Self, String> {
        let first = shards.first().ok_or("empty GGUF shard set")?;
        let mut config = Config::from_gguf(first, true)?;
        let mut b = Binder::new(shards);
        // The vocabulary is the embedding row count; no metadata key has it.
        match b.find("token_embd.weight") {
            Some((_, t)) if t.dims.len() == 2 => {
                config.vocab_size =
                    u32::try_from(t.dims[1]).map_err(|_| "token_embd.weight is too large")?;
            }
            _ => return Err("missing token_embd.weight".into()),
        }
        let c = &config;
        let dense = &[
            GgmlType::Q8_0,
            GgmlType::Q6K,
            GgmlType::Bf16,
            GgmlType::F16,
            GgmlType::F32,
        ];
        let hidden = u64::from(c.hidden_size);
        let vocab = u64::from(c.vocab_size);
        let token_embd = b.get("token_embd.weight", hidden, vocab, 1, dense);
        let output = b
            .get_optional("output.weight", hidden, vocab, 1, dense)
            .or_else(|| token_embd.clone());
        let hc_head = b.mixer("output_hc", c, false);
        let mut ple_table = None;
        if c.ple_layer.is_some() {
            // The converter may pad the table beyond the hashed range: take
            // the row count from the file and check that it covers every head.
            let rows = b
                .find("per_layer_token_embd.weight")
                .filter(|(_, t)| t.dims.len() == 2)
                .map_or(0, |(_, t)| t.dims[1]);
            if rows < c.ple_rows {
                b.fail("per_layer_token_embd.weight is missing or too short".into());
            } else {
                ple_table = b.get(
                    "per_layer_token_embd.weight",
                    u64::from(c.ple_head_dim),
                    rows,
                    1,
                    &[GgmlType::Iq4Nl, GgmlType::Bf16],
                );
            }
        }
        let layers: Vec<_> = (0..c.num_layers)
            .map(|il| b.layer(c, il, c.is_linear_layer(il), c.is_ple_layer(il), false))
            .collect();
        b.finish()?;
        let missing = || "missing tensor".to_string();
        Ok(Self {
            token_embd: token_embd.ok_or_else(missing)?,
            output: output.ok_or_else(missing)?,
            hc_head: hc_head.ok_or_else(missing)?,
            ple_table,
            layers: layers
                .into_iter()
                .collect::<Option<_>>()
                .ok_or_else(missing)?,
            config,
        })
    }
}

impl MtpTensors {
    /// Binds the draft sidecar against the trunk's configuration.
    pub fn bind(shards: &[Gguf<'_>], trunk: &Config) -> Result<Self, String> {
        let first = shards.first().ok_or("empty GGUF shard set")?;
        let mut config = Config::from_gguf(first, false)?;
        if !config.mtp_matches(trunk) {
            return Err(
                "MTP sidecar architecture or inference constants differ from the trunk".into(),
            );
        }
        config.vocab_size = trunk.vocab_size;
        let mut b = Binder::new(shards);
        // The draft block is a full-attention layer at index num_layers.
        let block = b.layer(&config, config.num_layers, false, false, true);
        b.finish()?;
        Ok(Self {
            block: block.ok_or("missing tensor")?,
            config,
        })
    }
}

/// Looks tensors up across the shards; the first failure is kept and
/// reported by [`Binder::finish`].
struct Binder<'s, 'a> {
    index: HashMap<&'a str, (usize, &'s Tensor<'a>)>,
    error: Option<String>,
}

impl<'s, 'a> Binder<'s, 'a> {
    fn new(shards: &'s [Gguf<'a>]) -> Self {
        let mut index = HashMap::new();
        for (shard, g) in shards.iter().enumerate() {
            for t in g.tensors() {
                index.insert(t.name, (shard, t));
            }
        }
        Self { index, error: None }
    }

    fn fail(&mut self, message: String) {
        self.error.get_or_insert(message);
    }

    fn finish(self) -> Result<(), String> {
        self.error.map_or(Ok(()), Err)
    }

    fn find(&self, name: &str) -> Option<(usize, &'s Tensor<'a>)> {
        self.index.get(name).copied()
    }

    /// Binds `name` with the exact shape and one of `types`.
    fn get(
        &mut self,
        name: &str,
        cols: u64,
        rows: u64,
        experts: u64,
        types: &[GgmlType],
    ) -> Option<TensorRef> {
        if self.find(name).is_none() {
            self.fail(format!("missing tensor {name}"));
            return None;
        }
        self.get_optional(name, cols, rows, experts, types)
    }

    fn get_optional(
        &mut self,
        name: &str,
        cols: u64,
        rows: u64,
        experts: u64,
        types: &[GgmlType],
    ) -> Option<TensorRef> {
        let (shard, t) = self.find(name)?;
        let dim = |i: usize| t.dims.get(i).copied().unwrap_or(1);
        if (dim(0), dim(1), dim(2)) != (cols, rows, experts) || t.dims.len() > 3 {
            self.fail(format!(
                "tensor {name} has shape [{}, {}, {}], expected [{cols}, {rows}, {experts}]",
                dim(0),
                dim(1),
                dim(2)
            ));
            return None;
        }
        let row_bytes = t.ty.encoded_bytes(cols);
        let size = row_bytes.and_then(|r| r.checked_mul(rows)?.checked_mul(experts));
        if !types.contains(&t.ty) || size != Some(t.bytes as u64) {
            self.fail(format!(
                "tensor {name} has unsupported format {}",
                t.ty.name()
            ));
            return None;
        }
        Some(TensorRef {
            name: name.to_string(),
            ty: t.ty,
            cols,
            rows,
            experts,
            shard,
            offset: t.offset,
            bytes: t.bytes,
        })
    }

    fn mixer(&mut self, prefix: &str, c: &Config, with_inject: bool) -> Option<MixerRefs> {
        let hc_dim = u64::from(c.hc_dim());
        let low_rank = u64::from(c.hc_low_rank);
        let norm = self.get(&format!("{prefix}_norm.weight"), hc_dim, 1, 1, F32);
        let down = self.get(&format!("{prefix}_down.weight"), hc_dim, low_rank, 1, DENSE);
        let up = self.get(&format!("{prefix}_up.weight"), low_rank, hc_dim, 1, DENSE);
        let inject = if with_inject {
            let inject = self.get(
                &format!("{prefix}_inject.weight"),
                hc_dim,
                u64::from(c.hc_count),
                1,
                &[GgmlType::F32, GgmlType::Q8_0, GgmlType::Bf16],
            );
            Some(inject?)
        } else {
            None
        };
        Some(MixerRefs {
            norm: norm?,
            down: down?,
            up: up?,
            inject,
        })
    }

    fn layer(
        &mut self,
        c: &Config,
        il: u32,
        linear: bool,
        ple: bool,
        nextn: bool,
    ) -> Option<LayerRefs> {
        let p = format!("blk.{il}.");
        let hidden = u64::from(c.hidden_size);
        let hc_dim = u64::from(c.hc_dim());
        let hc_attn = self.mixer(&format!("{p}hc_attn"), c, true);
        let hc_ffn = self.mixer(&format!("{p}hc_ffn"), c, true);
        let mixer = if linear {
            let v_heads = u64::from(c.ssm_num_v_heads);
            let alpha_beta = &[GgmlType::F32, GgmlType::Bf16, GgmlType::Q8_0];
            let refs = (
                self.get(
                    &format!("{p}attn_qkv.weight"),
                    hidden,
                    c.ssm_conv_channels().into(),
                    1,
                    DENSE,
                ),
                self.get(
                    &format!("{p}attn_gate.weight"),
                    hidden,
                    c.ssm_value_dim().into(),
                    1,
                    DENSE,
                ),
                self.get(
                    &format!("{p}ssm_conv1d.weight"),
                    c.ssm_conv_kernel.into(),
                    c.ssm_conv_channels().into(),
                    1,
                    F32,
                ),
                self.get(
                    &format!("{p}ssm_alpha.weight"),
                    hidden,
                    v_heads,
                    1,
                    alpha_beta,
                ),
                self.get(
                    &format!("{p}ssm_beta.weight"),
                    hidden,
                    v_heads,
                    1,
                    alpha_beta,
                ),
                self.get(&format!("{p}ssm_dt.bias"), v_heads, 1, 1, F32),
                self.get(&format!("{p}ssm_a"), v_heads, 1, 1, F32),
                self.get(
                    &format!("{p}ssm_norm.weight"),
                    c.ssm_head_dim.into(),
                    1,
                    1,
                    F32,
                ),
                self.get(
                    &format!("{p}ssm_out.weight"),
                    c.ssm_value_dim().into(),
                    hidden,
                    1,
                    DENSE,
                ),
            );
            match refs {
                (
                    Some(qkv),
                    Some(gate),
                    Some(conv1d),
                    Some(alpha),
                    Some(beta),
                    Some(dt),
                    Some(a),
                    Some(norm),
                    Some(out),
                ) => Some(MixerKind::Linear(LinearRefs {
                    qkv,
                    gate,
                    conv1d,
                    alpha,
                    beta,
                    dt,
                    a,
                    norm,
                    out,
                })),
                _ => None,
            }
        } else {
            let head_dim = u64::from(c.head_dim);
            let indexer_dim = u64::from(c.indexer_head_dim);
            let refs = (
                self.get(
                    &format!("{p}attn_q.weight"),
                    hidden,
                    2 * u64::from(c.attention_q_dim()),
                    1,
                    DENSE,
                ),
                self.get(
                    &format!("{p}attn_k.weight"),
                    hidden,
                    c.attention_kv_dim().into(),
                    1,
                    DENSE,
                ),
                self.get(
                    &format!("{p}attn_v.weight"),
                    hidden,
                    c.attention_kv_dim().into(),
                    1,
                    DENSE,
                ),
                self.get(
                    &format!("{p}attn_output.weight"),
                    c.attention_q_dim().into(),
                    hidden,
                    1,
                    DENSE,
                ),
                self.get(&format!("{p}attn_q_norm.weight"), head_dim, 1, 1, F32),
                self.get(&format!("{p}attn_k_norm.weight"), head_dim, 1, 1, F32),
                self.get(
                    &format!("{p}indexer.q_proj.weight"),
                    hidden,
                    u64::from(c.indexer_heads) * indexer_dim,
                    1,
                    DENSE,
                ),
                self.get(
                    &format!("{p}indexer.k_proj.weight"),
                    hidden,
                    indexer_dim,
                    1,
                    DENSE,
                ),
                self.get(&format!("{p}indexer.q_norm.weight"), indexer_dim, 1, 1, F32),
                self.get(&format!("{p}indexer.k_norm.weight"), indexer_dim, 1, 1, F32),
            );
            match refs {
                (
                    Some(q),
                    Some(k),
                    Some(v),
                    Some(out),
                    Some(q_norm),
                    Some(k_norm),
                    Some(indexer_q),
                    Some(indexer_k),
                    Some(indexer_q_norm),
                    Some(indexer_k_norm),
                ) => Some(MixerKind::Attention(AttentionRefs {
                    q,
                    k,
                    v,
                    out,
                    q_norm,
                    k_norm,
                    indexer_q,
                    indexer_k,
                    indexer_q_norm,
                    indexer_k_norm,
                })),
                _ => None,
            }
        };
        let ple = if ple {
            let ple_dim = u64::from(c.ple_embedding_dim());
            let refs = (
                self.get(&format!("{p}ple_key.weight"), ple_dim, hc_dim, 1, DENSE),
                self.get(&format!("{p}ple_value.weight"), ple_dim, hidden, 1, DENSE),
                self.get(&format!("{p}ple_norm_key.weight"), hc_dim, 1, 1, F32),
                self.get(&format!("{p}ple_norm_query.weight"), hc_dim, 1, 1, F32),
                self.get(&format!("{p}ple_norm_conv.weight"), hc_dim, 1, 1, F32),
                self.get(
                    &format!("{p}ple_conv1d.weight"),
                    c.ple_conv_kernel.into(),
                    hc_dim,
                    1,
                    F32,
                ),
            );
            match refs {
                (
                    Some(key),
                    Some(value),
                    Some(norm_key),
                    Some(norm_query),
                    Some(norm_conv),
                    Some(conv1d),
                ) => Some(Some(PleRefs {
                    key,
                    value,
                    norm_key,
                    norm_query,
                    norm_conv,
                    conv1d,
                })),
                _ => None,
            }
        } else {
            Some(None)
        };
        let experts = u64::from(c.num_experts);
        let expert_ff = u64::from(c.expert_ff);
        let shared_ff = u64::from(c.shared_expert_ff);
        let moe = (
            self.get(&format!("{p}ffn_gate_inp.weight"), hidden, experts, 1, F32),
            self.get(
                &format!("{p}ffn_gate_exps.weight"),
                hidden,
                expert_ff,
                experts,
                EXPERTS,
            ),
            self.get(
                &format!("{p}ffn_up_exps.weight"),
                hidden,
                expert_ff,
                experts,
                EXPERTS,
            ),
            self.get(
                &format!("{p}ffn_down_exps.weight"),
                expert_ff,
                hidden,
                experts,
                EXPERTS,
            ),
            self.get(&format!("{p}ffn_gate_inp_shexp.weight"), hidden, 1, 1, F32),
            self.get(
                &format!("{p}ffn_gate_shexp.weight"),
                hidden,
                shared_ff,
                1,
                DENSE,
            ),
            self.get(
                &format!("{p}ffn_up_shexp.weight"),
                hidden,
                shared_ff,
                1,
                DENSE,
            ),
            self.get(
                &format!("{p}ffn_down_shexp.weight"),
                shared_ff,
                hidden,
                1,
                DENSE,
            ),
        );
        let moe = match moe {
            (
                Some(router),
                Some(gate_exps),
                Some(up_exps),
                Some(down_exps),
                Some(shexp_gate_inp),
                Some(shexp_gate),
                Some(shexp_up),
                Some(shexp_down),
            ) => Some(MoeRefs {
                router,
                gate_exps,
                up_exps,
                down_exps,
                shexp_gate_inp,
                shexp_gate,
                shexp_up,
                shexp_down,
            }),
            _ => None,
        };
        let nextn = if nextn {
            let refs = (
                self.get(&format!("{p}nextn.enorm.weight"), hidden, 1, 1, F32),
                self.get(&format!("{p}nextn.hnorm.weight"), hc_dim, 1, 1, F32),
                self.get(
                    &format!("{p}nextn.eh_proj.weight"),
                    2 * hidden,
                    hidden,
                    1,
                    DENSE,
                ),
                self.mixer(&format!("{p}nextn.hc_head"), c, false),
            );
            match refs {
                (Some(enorm), Some(hnorm), Some(eh_proj), Some(head)) => Some(Some(NextnRefs {
                    enorm,
                    hnorm,
                    eh_proj,
                    head,
                })),
                _ => None,
            }
        } else {
            Some(None)
        };
        Some(LayerRefs {
            hc_attn: hc_attn?,
            hc_ffn: hc_ffn?,
            mixer: mixer?,
            ple: ple?,
            moe: moe?,
            nextn: nextn?,
        })
    }
}
