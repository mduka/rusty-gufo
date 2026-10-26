//! Device weights of the audio encoder and the text decoder.
//!
//! `gufo_hip::WeightLoader` copies the tensors into exactly sized device
//! arenas. Two derived forms are computed on the device once, like the C++
//! engine does: the 3×3 convolution kernels of the second and third
//! downsampler are uploaded to a temporary buffer and repacked for the WMMA
//! kernel (only the packed form stays), and the decoder's RMSNorm scales are
//! widened to f32 (all of them into one tensor).

use std::ops::Range;

use gufo_hip::{
    Bf16, DeviceTensor, LaunchError, LoadedWeights, Pending, Placement, Stream, WeightLoader,
};
use gufo_qwen3_asr_kernels as k;

use crate::config::ModelConfig;
use crate::safetensors::Checkpoint;

pub const SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

pub const AUDIO_HIDDEN: usize = 1024;
pub const AUDIO_FFN: usize = 4096;
pub const AUDIO_OUTPUT: usize = 2048;
pub const AUDIO_LAYERS: usize = 24;
pub const CONV_CHANNELS: usize = 480;
pub const CONV_FLATTENED: usize = 7680;

pub struct EncoderLayer {
    pub q_weight: DeviceTensor<Bf16>,
    pub q_bias: DeviceTensor<Bf16>,
    pub k_weight: DeviceTensor<Bf16>,
    pub k_bias: DeviceTensor<Bf16>,
    pub v_weight: DeviceTensor<Bf16>,
    pub v_bias: DeviceTensor<Bf16>,
    pub out_weight: DeviceTensor<Bf16>,
    pub out_bias: DeviceTensor<Bf16>,
    pub attention_norm_weight: DeviceTensor<Bf16>,
    pub attention_norm_bias: DeviceTensor<Bf16>,
    pub fc1_weight: DeviceTensor<Bf16>,
    pub fc1_bias: DeviceTensor<Bf16>,
    pub fc2_weight: DeviceTensor<Bf16>,
    pub fc2_bias: DeviceTensor<Bf16>,
    pub final_norm_weight: DeviceTensor<Bf16>,
    pub final_norm_bias: DeviceTensor<Bf16>,
}

pub struct EncoderWeights {
    pub conv1_weight: DeviceTensor<Bf16>,
    pub conv1_bias: DeviceTensor<Bf16>,
    /// Repacked by `pack_convolution_weights`.
    pub conv2_weight: DeviceTensor<Bf16>,
    pub conv2_bias: DeviceTensor<Bf16>,
    /// Repacked by `pack_convolution_weights`.
    pub conv3_weight: DeviceTensor<Bf16>,
    pub conv3_bias: DeviceTensor<Bf16>,
    pub conv_out: DeviceTensor<Bf16>,
    pub layers: Vec<EncoderLayer>,
    pub ln_post_weight: DeviceTensor<Bf16>,
    pub ln_post_bias: DeviceTensor<Bf16>,
    pub proj1_weight: DeviceTensor<Bf16>,
    pub proj1_bias: DeviceTensor<Bf16>,
    pub proj2_weight: DeviceTensor<Bf16>,
    pub proj2_bias: DeviceTensor<Bf16>,
}

pub struct DecoderLayer {
    pub q: DeviceTensor<Bf16>,
    pub k: DeviceTensor<Bf16>,
    pub v: DeviceTensor<Bf16>,
    pub o: DeviceTensor<Bf16>,
    pub gate: DeviceTensor<Bf16>,
    pub up: DeviceTensor<Bf16>,
    pub down: DeviceTensor<Bf16>,
    /// Ranges of [`DecoderWeights::norms`].
    pub input_norm: Range<usize>,
    pub q_norm: Range<usize>,
    pub k_norm: Range<usize>,
    pub post_norm: Range<usize>,
}

pub struct DecoderWeights {
    pub layers: Vec<DecoderLayer>,
    /// Every RMSNorm scale widened to f32.
    pub norms: DeviceTensor<f32>,
    pub final_norm: Range<usize>,
    pub embedding: DeviceTensor<Bf16>,
    pub lm_head: DeviceTensor<Bf16>,
}

pub struct Weights {
    pub encoder: EncoderWeights,
    pub decoder: DecoderWeights,
}

/// Queues named tensors with their expected shapes.
struct Queue<'a, 's> {
    checkpoint: &'a Checkpoint,
    loader: WeightLoader<'s>,
}

impl Queue<'_, '_> {
    fn add(&mut self, name: &str, shape: &[u64]) -> Result<Pending<Bf16>, String> {
        let info = self.checkpoint.bf16(name, shape)?;
        self.loader
            .add::<Bf16>(
                self.checkpoint.file(info.file),
                info.offset,
                info.elements(),
            )
            .map_err(|e| format!("{name}: {e}"))
    }
}

struct PendingEncoderLayer([Pending<Bf16>; 16]);

struct PendingDecoderLayer {
    matrices: [Pending<Bf16>; 7],
    norms: [Pending<Bf16>; 4],
}

/// Every queued tensor, in the order `Weights::assemble` redeems them.
struct PendingWeights {
    conv: [Pending<Bf16>; 5],
    encoder_layers: Vec<PendingEncoderLayer>,
    post: [Pending<Bf16>; 6],
    decoder_layers: Vec<PendingDecoderLayer>,
    final_norm: Pending<Bf16>,
    embedding: Pending<Bf16>,
    lm_head: Pending<Bf16>,
}

fn launch(error: LaunchError) -> String {
    format!("Qwen3-ASR weight preparation: {error}")
}

impl Weights {
    /// Loads both components' tensors on `stream` and waits until every
    /// derived tensor is ready.
    pub fn load(
        stream: &Stream,
        checkpoint: &Checkpoint,
        config: &ModelConfig,
        placement: Placement,
    ) -> Result<Self, String> {
        let audio = |name: &str| format!("thinker.audio_tower.{name}");
        // The 3x3 kernels are repacked once; only the packed form is kept.
        let conv2_weight = pack_convolution(stream, checkpoint, &audio("conv2d2.weight"))?;
        let conv3_weight = pack_convolution(stream, checkpoint, &audio("conv2d3.weight"))?;
        let (loaded, pending) = Self::queue_and_load(stream, checkpoint, config, placement)?;
        Self::assemble(stream, loaded, pending, conv2_weight, conv3_weight)
    }

    fn queue_and_load(
        stream: &Stream,
        checkpoint: &Checkpoint,
        config: &ModelConfig,
        placement: Placement,
    ) -> Result<(LoadedWeights, PendingWeights), String> {
        let text = &config.text;
        let (hidden, heads, kv_heads, head_dim, ffn) = (
            u64::from(text.hidden_size),
            u64::from(text.num_attention_heads),
            u64::from(text.num_key_value_heads),
            u64::from(text.head_dim),
            u64::from(text.intermediate_size),
        );
        let vocab = u64::from(text.vocab_size);
        let mut q = Queue {
            checkpoint,
            loader: WeightLoader::new(stream, placement),
        };
        // The two largest tables first: small tensors queued last keep the
        // loader's final, unoverlapped copy short.
        let embedding = q.add("thinker.model.embed_tokens.weight", &[vocab, hidden])?;
        let lm_head = q.add("thinker.lm_head.weight", &[vocab, hidden])?;
        let audio = |name: &str| format!("thinker.audio_tower.{name}");
        let conv = [
            q.add(&audio("conv2d1.weight"), &[480, 1, 3, 3])?,
            q.add(&audio("conv2d1.bias"), &[480])?,
            q.add(&audio("conv2d2.bias"), &[480])?,
            q.add(&audio("conv2d3.bias"), &[480])?,
            q.add(&audio("conv_out.weight"), &[1024, 7680])?,
        ];
        let (h, f) = (AUDIO_HIDDEN as u64, AUDIO_FFN as u64);
        let mut encoder_layers = Vec::with_capacity(AUDIO_LAYERS);
        for layer in 0..AUDIO_LAYERS {
            let name = |suffix: &str| audio(&format!("layers.{layer}.{suffix}"));
            encoder_layers.push(PendingEncoderLayer([
                q.add(&name("self_attn.q_proj.weight"), &[h, h])?,
                q.add(&name("self_attn.q_proj.bias"), &[h])?,
                q.add(&name("self_attn.k_proj.weight"), &[h, h])?,
                q.add(&name("self_attn.k_proj.bias"), &[h])?,
                q.add(&name("self_attn.v_proj.weight"), &[h, h])?,
                q.add(&name("self_attn.v_proj.bias"), &[h])?,
                q.add(&name("self_attn.out_proj.weight"), &[h, h])?,
                q.add(&name("self_attn.out_proj.bias"), &[h])?,
                q.add(&name("self_attn_layer_norm.weight"), &[h])?,
                q.add(&name("self_attn_layer_norm.bias"), &[h])?,
                q.add(&name("fc1.weight"), &[f, h])?,
                q.add(&name("fc1.bias"), &[f])?,
                q.add(&name("fc2.weight"), &[h, f])?,
                q.add(&name("fc2.bias"), &[h])?,
                q.add(&name("final_layer_norm.weight"), &[h])?,
                q.add(&name("final_layer_norm.bias"), &[h])?,
            ]));
        }
        let o = AUDIO_OUTPUT as u64;
        let post = [
            q.add(&audio("ln_post.weight"), &[h])?,
            q.add(&audio("ln_post.bias"), &[h])?,
            q.add(&audio("proj1.weight"), &[h, h])?,
            q.add(&audio("proj1.bias"), &[h])?,
            q.add(&audio("proj2.weight"), &[o, h])?,
            q.add(&audio("proj2.bias"), &[o])?,
        ];

        let mut decoder_layers = Vec::with_capacity(text.num_hidden_layers as usize);
        for layer in 0..text.num_hidden_layers {
            let name = |suffix: &str| format!("thinker.model.layers.{layer}.{suffix}");
            decoder_layers.push(PendingDecoderLayer {
                matrices: [
                    q.add(
                        &name("self_attn.q_proj.weight"),
                        &[heads * head_dim, hidden],
                    )?,
                    q.add(
                        &name("self_attn.k_proj.weight"),
                        &[kv_heads * head_dim, hidden],
                    )?,
                    q.add(
                        &name("self_attn.v_proj.weight"),
                        &[kv_heads * head_dim, hidden],
                    )?,
                    q.add(
                        &name("self_attn.o_proj.weight"),
                        &[hidden, heads * head_dim],
                    )?,
                    q.add(&name("mlp.gate_proj.weight"), &[ffn, hidden])?,
                    q.add(&name("mlp.up_proj.weight"), &[ffn, hidden])?,
                    q.add(&name("mlp.down_proj.weight"), &[hidden, ffn])?,
                ],
                norms: [
                    q.add(&name("input_layernorm.weight"), &[hidden])?,
                    q.add(&name("self_attn.q_norm.weight"), &[head_dim])?,
                    q.add(&name("self_attn.k_norm.weight"), &[head_dim])?,
                    q.add(&name("post_attention_layernorm.weight"), &[hidden])?,
                ],
            });
        }
        let final_norm = q.add("thinker.model.norm.weight", &[hidden])?;

        let loaded = q
            .loader
            .finish()
            .map_err(|e| format!("Qwen3-ASR weight upload: {e}"))?;
        Ok((
            loaded,
            PendingWeights {
                conv,
                encoder_layers,
                post,
                decoder_layers,
                final_norm,
                embedding,
                lm_head,
            },
        ))
    }

    fn assemble(
        stream: &Stream,
        loaded: LoadedWeights,
        pending: PendingWeights,
        conv2_weight: DeviceTensor<Bf16>,
        conv3_weight: DeviceTensor<Bf16>,
    ) -> Result<Self, String> {
        let get = |p: Pending<Bf16>| loaded.get(p);
        let PendingWeights {
            conv,
            encoder_layers,
            post,
            decoder_layers,
            final_norm,
            embedding,
            lm_head,
        } = pending;
        let [c1w, c1b, c2b, c3b, conv_out] = conv;
        let layers = encoder_layers
            .into_iter()
            .map(|PendingEncoderLayer(t)| {
                let [
                    qw,
                    qb,
                    kw,
                    kb,
                    vw,
                    vb,
                    ow,
                    ob,
                    anw,
                    anb,
                    f1w,
                    f1b,
                    f2w,
                    f2b,
                    fnw,
                    fnb,
                ] = t;
                EncoderLayer {
                    q_weight: get(qw),
                    q_bias: get(qb),
                    k_weight: get(kw),
                    k_bias: get(kb),
                    v_weight: get(vw),
                    v_bias: get(vb),
                    out_weight: get(ow),
                    out_bias: get(ob),
                    attention_norm_weight: get(anw),
                    attention_norm_bias: get(anb),
                    fc1_weight: get(f1w),
                    fc1_bias: get(f1b),
                    fc2_weight: get(f2w),
                    fc2_bias: get(f2b),
                    final_norm_weight: get(fnw),
                    final_norm_bias: get(fnb),
                }
            })
            .collect();
        let [lnw, lnb, p1w, p1b, p2w, p2b] = post;
        let encoder = EncoderWeights {
            conv1_weight: get(c1w),
            conv1_bias: get(c1b),
            conv2_weight,
            conv2_bias: get(c2b),
            conv3_weight,
            conv3_bias: get(c3b),
            conv_out: get(conv_out),
            layers,
            ln_post_weight: get(lnw),
            ln_post_bias: get(lnb),
            proj1_weight: get(p1w),
            proj1_bias: get(p1b),
            proj2_weight: get(p2w),
            proj2_bias: get(p2b),
        };

        let (layers, norms, final_norm) = widen_norms(stream, decoder_layers, final_norm, get)?;
        Ok(Self {
            encoder,
            decoder: DecoderWeights {
                layers,
                norms,
                final_norm,
                embedding: get(embedding),
                lm_head: get(lm_head),
            },
        })
    }
}

/// Uploads a `[480, 480, 3, 3]` kernel into a temporary buffer and repacks
/// it for `convolution3x3`; the temporary is freed again.
fn pack_convolution(
    stream: &Stream,
    checkpoint: &Checkpoint,
    name: &str,
) -> Result<DeviceTensor<Bf16>, String> {
    let error = |e: gufo_hip::HipError| format!("Qwen3-ASR convolution packing: {e}");
    let info = checkpoint.bf16(name, &[480, 480, 3, 3])?;
    let bytes = &checkpoint.file(info.file).bytes()[info.offset..info.offset + info.bytes];
    let host: Vec<Bf16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| Bf16(u16::from_le_bytes(pair)))
        .collect();
    let mut original = stream.alloc_persistent::<Bf16>(host.len()).map_err(error)?;
    stream
        .upload(&host, original.as_mut_slice())
        .map_err(error)?;
    let mut packed = stream.alloc_persistent::<Bf16>(host.len()).map_err(error)?;
    k::pack_convolution_weights(stream, original.as_slice(), packed.as_mut_slice())
        .map_err(launch)?;
    packed.freeze(stream).map_err(error)
}

type WidenedNorms = (Vec<DecoderLayer>, DeviceTensor<f32>, Range<usize>);

/// Widens every decoder RMSNorm scale into one f32 tensor.
fn widen_norms(
    stream: &Stream,
    layers: Vec<PendingDecoderLayer>,
    final_norm: Pending<Bf16>,
    get: impl Fn(Pending<Bf16>) -> DeviceTensor<Bf16>,
) -> Result<WidenedNorms, String> {
    let mut sources = Vec::new();
    let mut matrices = Vec::with_capacity(layers.len());
    let mut ranges = Vec::with_capacity(layers.len());
    let mut total = 0usize;
    let mut place = |tensor: DeviceTensor<Bf16>, sources: &mut Vec<(DeviceTensor<Bf16>, usize)>| {
        let range = total..total + tensor.len();
        total = range.end;
        sources.push((tensor, range.start));
        range
    };
    for layer in layers {
        let [input, qn, kn, post] = layer.norms;
        ranges.push([
            place(get(input), &mut sources),
            place(get(qn), &mut sources),
            place(get(kn), &mut sources),
            place(get(post), &mut sources),
        ]);
        matrices.push(layer.matrices);
    }
    let final_range = place(get(final_norm), &mut sources);
    let mut norms = stream
        .alloc_persistent::<f32>(total)
        .map_err(|e| format!("Qwen3-ASR norm widening: {e}"))?;
    for (tensor, start) in &sources {
        k::bf16_to_f32(
            stream,
            tensor.view(),
            norms.slice_mut(*start..start + tensor.len()),
            tensor.len(),
        )
        .map_err(launch)?;
    }
    let norms = norms
        .freeze(stream)
        .map_err(|e| format!("Qwen3-ASR norm widening: {e}"))?;
    let layers = matrices
        .into_iter()
        .zip(ranges)
        .map(
            |([q, k, v, o, gate, up, down], [input, qn, kn, post])| DecoderLayer {
                q: get(q),
                k: get(k),
                v: get(v),
                o: get(o),
                gate: get(gate),
                up: get(up),
                down: get(down),
                input_norm: input,
                q_norm: qn,
                k_norm: kn,
                post_norm: post,
            },
        )
        .collect();
    Ok((layers, norms, final_range))
}
