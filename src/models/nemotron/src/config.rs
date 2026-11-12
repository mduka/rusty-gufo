//! Architecture parameters of the trunk, the LiLiCorr drafter and the vision
//! projector, read and validated from their GGUF metadata (`config.cpp`).

use gufo_core::gguf::{Gguf, Value};

const ARCHITECTURE: &str = "nemotron_h_moe";
/// llama.cpp's gating function id of sigmoid routing.
const SIGMOID_GATING: u32 = 2;

/// Mixer of one trunk block. Every block is pre-norm with a residual add.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum BlockType {
    /// Mamba-2 selective state space mixer.
    #[default]
    Mamba,
    /// GQA without positional encoding (NoPE).
    Attention,
    /// LatentMoE: routed ReLU^2 experts plus one shared expert.
    Moe,
}

/// Architecture parameters of a `nemotron_h_moe` GGUF artifact. Every value
/// is read from the file; the fixed limits only bound what this runtime was
/// written and validated for (Nemotron 3.5 Bolt 74B-A9B, 88 blocks).
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// block_count, 88.
    pub num_layers: u32,
    /// n_embd, 4096.
    pub hidden_size: u32,
    /// 131072.
    pub vocab_size: u32,
    /// 262144.
    pub context_length: u32,
    pub rms_eps: f32,

    // Attention blocks.
    /// 32
    pub num_heads: u32,
    /// 2
    pub num_kv_heads: u32,
    /// 128
    pub head_dim: u32,

    // Mamba-2 blocks.
    /// 4
    pub ssm_conv_kernel: u32,
    /// 128 (d_state)
    pub ssm_state_size: u32,
    /// 8 (B/C groups)
    pub ssm_groups: u32,
    /// 8192 (d_inner)
    pub ssm_inner_size: u32,
    /// 128 (time_step_rank)
    pub ssm_heads: u32,

    // MoE blocks. Routed experts per layer come from the tensor shapes; the
    // metadata expert_count is only their maximum.
    /// 496
    pub max_experts: u32,
    /// 1024, routed experts' model width.
    pub latent_size: u32,
    /// 5376
    pub shared_expert_ff: u32,
    /// 5.0
    pub expert_weights_scale: f32,
    /// true
    pub expert_weights_norm: bool,

    pub blocks: Vec<BlockType>,
    /// Per block; 0 off MoE blocks.
    pub experts_used: Vec<u32>,
    /// Per block; 0 off MoE blocks.
    pub expert_ff: Vec<u32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_layers: 0,
            hidden_size: 0,
            vocab_size: 0,
            context_length: 0,
            rms_eps: 1e-5,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            ssm_conv_kernel: 0,
            ssm_state_size: 0,
            ssm_groups: 0,
            ssm_inner_size: 0,
            ssm_heads: 0,
            max_experts: 0,
            latent_size: 0,
            shared_expert_ff: 0,
            expert_weights_scale: 1.0,
            expert_weights_norm: false,
            blocks: Vec::new(),
            experts_used: Vec::new(),
            expert_ff: Vec::new(),
        }
    }
}

impl Config {
    pub fn ssm_head_dim(&self) -> u32 {
        self.ssm_inner_size / self.ssm_heads
    }

    /// Channels of the x|B|C slice and its causal convolution.
    pub fn ssm_conv_channels(&self) -> u32 {
        self.ssm_inner_size + 2 * self.ssm_groups * self.ssm_state_size
    }

    /// Rows of ssm_in: z | x | B | C | dt.
    pub fn ssm_in_projection(&self) -> u32 {
        self.ssm_inner_size + self.ssm_conv_channels() + self.ssm_heads
    }

    pub fn attention_q_dim(&self) -> u32 {
        self.num_heads * self.head_dim
    }

    pub fn attention_kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
}

/// NVIDIA's LiLiCorr DFlash drafter (`dflash` architecture): a five-layer
/// block-diffusion transformer that reads the trunk's residual stream at
/// `target_layers`, drafts `block_size - 1` tokens in one pass, and a
/// two-layer LiLiCorr head that scores the drafted candidates jointly.
#[derive(Debug, Clone, PartialEq)]
pub struct DrafterConfig {
    /// 5
    pub num_layers: u32,
    /// 4096, equal to the trunk's
    pub hidden_size: u32,
    /// 12288, SiLU-gated FFN width
    pub ff_size: u32,
    /// 32
    pub num_heads: u32,
    /// 2
    pub num_kv_heads: u32,
    /// 128, NeoX RoPE over all of it
    pub head_dim: u32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// 8: the anchor and 7 mask tokens
    pub block_size: u32,
    /// 1024, on every layer
    pub sliding_window: u32,
    /// 2, causal within the block
    pub conv_kernel: u32,
    /// 16 channels per dynamic tap
    pub conv_group: u32,
    /// 990
    pub mask_token: u32,
    /// Trunk blocks whose input residual feeds the drafter.
    pub target_layers: Vec<u32>,
    // LiLiCorr candidate head.
    /// 1024
    pub lc_hidden: u32,
    /// 8 candidates per drafted slot
    pub lc_top_k: u32,
    /// 8
    pub lc_heads: u32,
    /// 2
    pub lc_layers: u32,
    /// 1024
    pub lc_factor: u32,
    /// 2048
    pub lc_mlp: u32,
    pub lc_vector_eps: f32,
    pub lc_logit_scale: f32,
}

impl Default for DrafterConfig {
    fn default() -> Self {
        Self {
            num_layers: 0,
            hidden_size: 0,
            ff_size: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            rms_eps: 1e-6,
            rope_theta: 10000.0,
            block_size: 0,
            sliding_window: 0,
            conv_kernel: 0,
            conv_group: 0,
            mask_token: 0,
            target_layers: Vec::new(),
            lc_hidden: 0,
            lc_top_k: 0,
            lc_heads: 0,
            lc_layers: 0,
            lc_factor: 0,
            lc_mlp: 0,
            lc_vector_eps: 1e-4,
            lc_logit_scale: 8.0,
        }
    }
}

impl DrafterConfig {
    /// Drafted slots per block, also the most tokens one round can draft.
    pub fn slots(&self) -> u32 {
        self.block_size - 1
    }

    pub fn feature_size(&self) -> u32 {
        self.target_layers.len() as u32 * self.hidden_size
    }

    /// Rows of a conv projection: per side and tap, one weight per group.
    pub fn conv_projection(&self) -> u32 {
        2 * self.conv_kernel * (self.hidden_size / self.conv_group)
    }

    pub fn q_dim(&self) -> u32 {
        self.num_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
}

/// NVIDIA's vision projector for the trunk (`clip.projector_type =
/// nemotron_omni`, the mmproj GGUF): a C-RADIO v4-H ViT over 16-pixel
/// patches with prefix tokens and a cropped position table, whose features
/// are merged 2x2 and projected to the trunk's width by an RMSNorm and a
/// ReLU² MLP.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    /// 1280
    pub width: u32,
    /// 32
    pub num_layers: u32,
    /// 16
    pub num_heads: u32,
    /// 5120, exact-GELU MLP
    pub ff_size: u32,
    /// 16 pixels per side
    pub patch_size: u32,
    /// 2: merge² patches form one token
    pub merge: u32,
    /// 10: 3 cls and 7 register tokens
    pub prefix_tokens: u32,
    /// 128: the table is side x side
    pub position_side: u32,
    /// 20480
    pub projector_hidden: u32,
    /// 4096, the trunk's width
    pub projection: u32,
    /// 1024 per image
    pub min_patches: u32,
    /// 13312 per image
    pub max_patches: u32,
    /// ViT LayerNorm
    pub eps: f32,
    /// The projector's RMSNorm epsilon is not stored; NVIDIA's model and
    /// llama.cpp use 1e-5.
    pub projector_eps: f32,
    /// Pixel normalization per RGB channel: (x / 255 - mean) / deviation.
    pub mean: [f32; 3],
    pub deviation: [f32; 3],
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            width: 0,
            num_layers: 0,
            num_heads: 0,
            ff_size: 0,
            patch_size: 0,
            merge: 0,
            prefix_tokens: 0,
            position_side: 0,
            projector_hidden: 0,
            projection: 0,
            min_patches: 0,
            max_patches: 0,
            eps: 1e-6,
            projector_eps: 1e-5,
            mean: [0.0; 3],
            deviation: [0.0; 3],
        }
    }
}

impl VisionConfig {
    pub fn head_dim(&self) -> u32 {
        self.width / self.num_heads
    }

    /// Values of one flattened patch: channel-major RGB rows.
    pub fn patch_values(&self) -> u32 {
        3 * self.patch_size * self.patch_size
    }

    /// Width of a token after the merge, the projector's input.
    pub fn merged_width(&self) -> u32 {
        self.width * self.merge * self.merge
    }
}

/// Metadata reads that keep the first error, as the C++ `Reader`.
struct Reader<'g, 'a> {
    gguf: &'g Gguf<'a>,
    error: Option<String>,
}

impl<'g, 'a> Reader<'g, 'a> {
    fn new(gguf: &'g Gguf<'a>) -> Self {
        Self { gguf, error: None }
    }

    fn ok(&self) -> bool {
        self.error.is_none()
    }

    fn fail(&mut self, message: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(message.into());
        }
    }

    fn present(&self, key: &str) -> bool {
        self.gguf.metadata(key).is_some()
    }

    fn u32(&mut self, key: &str) -> u32 {
        self.u32_or(key, true, 0)
    }

    fn u32_or(&mut self, key: &str, required: bool, fallback: u32) -> u32 {
        if let Some(value) = self.gguf.u32(key) {
            return value;
        }
        if required || self.present(key) {
            self.fail(format!("missing or invalid GGUF integer {key}"));
        }
        fallback
    }

    fn f32(&mut self, key: &str, fallback: f32) -> f32 {
        self.f32_or(key, fallback, true)
    }

    fn f32_or(&mut self, key: &str, fallback: f32, required: bool) -> f32 {
        let value = self.gguf.f32(key);
        if value.is_some_and(|v| !v.is_finite())
            || (value.is_none() && (required || self.present(key)))
        {
            self.fail(format!("missing or invalid GGUF float {key}"));
        }
        value.unwrap_or(fallback)
    }

    fn bool(&mut self, key: &str, fallback: bool) -> bool {
        let value = self.gguf.bool(key);
        if value.is_none() && self.present(key) {
            self.fail(format!("invalid GGUF bool {key}"));
        }
        value.unwrap_or(fallback)
    }

    /// One value per block. Writers store a scalar when every block agrees
    /// and an integer array of any width otherwise.
    fn per_layer(&mut self, key: &str, layers: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let gguf = self.gguf;
        let Some(meta) = gguf.metadata(key) else {
            self.fail(format!("missing GGUF value {key}"));
            return out;
        };
        if let Some(scalar) = gguf.u32(key) {
            return vec![scalar; layers as usize];
        }
        let mut push = |reader: &mut Self, v: u64| match u32::try_from(v) {
            Ok(v) => out.push(v),
            Err(_) => reader.fail(format!("GGUF array value out of range: {key}")),
        };
        match meta {
            Value::UIntArray(values) => {
                for &v in values {
                    push(self, v);
                }
            }
            Value::IntArray(values) => {
                for &v in values {
                    if v < 0 {
                        self.fail(format!("negative GGUF array value {key}"));
                        break;
                    }
                    push(self, v as u64);
                }
            }
            _ => self.fail(format!(
                "GGUF value must be an integer or integer array: {key}"
            )),
        }
        if self.ok() && out.len() != layers as usize {
            self.fail(format!("GGUF array must describe every block: {key}"));
        }
        out
    }

    fn finish<T>(self, value: T) -> Result<T, String> {
        match self.error {
            None => Ok(value),
            Some(error) => Err(error),
        }
    }

    fn error(self) -> String {
        self.error.unwrap_or_default()
    }
}

fn architecture<'a>(gguf: &Gguf<'a>) -> Option<&'a [u8]> {
    gguf.string("general.architecture")
}

impl Config {
    /// Reads and validates the `nemotron_h_moe.*` metadata (`FromGguf`).
    pub fn from_gguf(gguf: &Gguf<'_>) -> Result<Self, String> {
        let mut r = Reader::new(gguf);
        if architecture(gguf) != Some(ARCHITECTURE.as_bytes()) {
            return Err("GGUF architecture is not nemotron_h_moe".into());
        }
        let key = |suffix: &str| format!("{ARCHITECTURE}.{suffix}");
        let mut c = Config {
            num_layers: r.u32(&key("block_count")),
            ..Config::default()
        };
        if r.u32_or(&key("nextn_predict_layers"), false, 0) != 0 {
            r.fail("nemotron_h_moe MTP blocks are not supported");
        }
        c.hidden_size = r.u32(&key("embedding_length"));
        c.vocab_size = r.u32(&key("vocab_size"));
        c.context_length = r.u32(&key("context_length"));
        c.rms_eps = r.f32(&key("attention.layer_norm_rms_epsilon"), 1e-5);

        c.head_dim = r.u32(&key("attention.key_length"));
        if r.u32_or(&key("attention.value_length"), false, c.head_dim) != c.head_dim {
            r.fail("attention value length must match the key length");
        }

        c.ssm_conv_kernel = r.u32(&key("ssm.conv_kernel"));
        c.ssm_state_size = r.u32(&key("ssm.state_size"));
        c.ssm_groups = r.u32(&key("ssm.group_count"));
        c.ssm_inner_size = r.u32(&key("ssm.inner_size"));
        c.ssm_heads = r.u32(&key("ssm.time_step_rank"));

        c.max_experts = r.u32(&key("expert_count"));
        c.latent_size = r.u32(&key("moe_latent_size"));
        c.shared_expert_ff = r.u32(&key("expert_shared_feed_forward_length"));
        c.expert_weights_scale = r.f32_or(&key("expert_weights_scale"), 1.0, false);
        c.expert_weights_norm = r.bool(&key("expert_weights_norm"), false);
        if r.u32(&key("expert_shared_count")) != 1 {
            r.fail("exactly one shared expert is supported");
        }
        if r.u32(&key("expert_gating_func")) != SIGMOID_GATING {
            r.fail("only sigmoid expert gating is supported");
        }
        if r.u32_or(&key("expert_group_count"), false, 1) != 1 {
            r.fail("grouped expert routing is not supported");
        }
        if !r.ok() {
            return Err(r.error());
        }

        let heads = r.per_layer(&key("attention.head_count"), c.num_layers);
        let kv_heads = r.per_layer(&key("attention.head_count_kv"), c.num_layers);
        let ff = r.per_layer(&key("feed_forward_length"), c.num_layers);
        let expert_ff = r.per_layer(&key("expert_feed_forward_length"), c.num_layers);
        let used = r.per_layer(&key("expert_used_count"), c.num_layers);
        if !r.ok() {
            return Err(r.error());
        }

        // As in llama.cpp, a block is recurrent when it declares neither KV
        // heads nor a feed-forward width, attention when it has only KV heads,
        // and MoE when it has a feed-forward width.
        let layers = c.num_layers as usize;
        c.blocks = vec![BlockType::Mamba; layers];
        c.experts_used = vec![0; layers];
        c.expert_ff = vec![0; layers];
        for il in 0..layers {
            if kv_heads[il] == 0 && ff[il] == 0 {
                c.blocks[il] = BlockType::Mamba;
            } else if ff[il] == 0 {
                c.blocks[il] = BlockType::Attention;
                if c.num_heads == 0 {
                    c.num_heads = heads[il];
                    c.num_kv_heads = kv_heads[il];
                } else if heads[il] != c.num_heads || kv_heads[il] != c.num_kv_heads {
                    r.fail("attention blocks must share one head geometry");
                }
            } else {
                c.blocks[il] = BlockType::Moe;
                if kv_heads[il] != 0 {
                    r.fail("MoE blocks cannot declare attention heads");
                }
                c.experts_used[il] = used[il];
                c.expert_ff[il] = if expert_ff[il] != 0 {
                    expert_ff[il]
                } else {
                    ff[il]
                };
            }
        }
        if !r.ok() {
            return Err(r.error());
        }

        // Structural limits of this runtime, not of the format.
        let mut shape_ok = c.num_layers != 0
            && c.context_length != 0
            && c.context_length <= i32::MAX as u32
            && c.rms_eps > 0.0
            && c.hidden_size != 0
            && c.vocab_size != 0
            && c.num_heads != 0
            && c.num_kv_heads != 0
            && c.num_heads.is_multiple_of(c.num_kv_heads)
            && c.head_dim != 0
            && c.ssm_conv_kernel >= 2
            && c.ssm_state_size != 0
            && c.ssm_groups != 0
            && c.ssm_heads != 0
            && c.ssm_heads.is_multiple_of(c.ssm_groups)
            && c.ssm_inner_size.is_multiple_of(c.ssm_heads)
            && c.ssm_inner_size.is_multiple_of(c.ssm_groups)
            && c.max_experts != 0
            && c.latent_size != 0
            && c.shared_expert_ff != 0
            && c.expert_weights_scale > 0.0;
        for il in 0..layers {
            if !shape_ok {
                break;
            }
            if c.blocks[il] == BlockType::Moe {
                shape_ok = c.experts_used[il] != 0
                    && c.experts_used[il] <= c.max_experts
                    && c.expert_ff[il] != 0;
            }
        }
        if !shape_ok {
            return Err("nemotron_h_moe metadata describes an unsupported shape".into());
        }
        // These dimensions select specialized kernels. Reject other profiles
        // before allocating; per-block expert counts and widths stay free.
        if c.num_layers != 88
            || c.hidden_size != 4096
            || c.num_heads != 32
            || c.num_kv_heads != 2
            || c.head_dim != 128
            || c.ssm_conv_kernel != 4
            || c.ssm_state_size != 128
            || c.ssm_groups != 8
            || c.ssm_inner_size != 8192
            || c.ssm_heads != 128
            || c.latent_size != 1024
            || c.shared_expert_ff != 5376
        {
            return Err("unsupported Nemotron 3.5 Bolt geometry".into());
        }
        r.finish(c)
    }
}

impl DrafterConfig {
    /// Reads and validates the `dflash.*` and `lilicorr.*` metadata against
    /// the trunk it drafts for (`FromGguf`).
    pub fn from_gguf(gguf: &Gguf<'_>, target: &Config) -> Result<Self, String> {
        let mut r = Reader::new(gguf);
        if architecture(gguf) != Some(b"dflash".as_slice()) {
            return Err("GGUF architecture is not dflash".into());
        }
        let mut c = DrafterConfig {
            num_layers: r.u32("dflash.block_count"),
            ..DrafterConfig::default()
        };
        c.hidden_size = r.u32("dflash.embedding_length");
        c.ff_size = r.u32("dflash.feed_forward_length");
        c.num_heads = r.u32("dflash.attention.head_count");
        c.num_kv_heads = r.u32("dflash.attention.head_count_kv");
        c.head_dim = r.u32("dflash.attention.key_length");
        if r.u32_or("dflash.attention.value_length", false, c.head_dim) != c.head_dim {
            r.fail("drafter value length must match the key length");
        }
        c.rms_eps = r.f32("dflash.attention.layer_norm_rms_epsilon", 1e-6);
        c.rope_theta = r.f32("dflash.rope.freq_base", 10000.0);
        if r.u32_or("dflash.rope.dimension_count", false, c.head_dim) != c.head_dim {
            r.fail("drafter RoPE must rotate the whole head");
        }
        c.block_size = r.u32("dflash.block_size");
        c.sliding_window = r.u32("dflash.attention.sliding_window");
        c.conv_kernel = r.u32("dflash.conv_kernel_size");
        c.conv_group = r.u32("dflash.conv_group_size");
        if r.u32_or("dflash.selector_rank", false, 0) != 0
            || r.u32_or("dflash.selector_top_k", false, 0) != 0
        {
            r.fail("DFlash2 selector drafters are not supported");
        }
        if r.bool("dflash.attention.causal", true) {
            r.fail("the drafter must attend non-causally within its block");
        }
        for key in [
            "dflash.embedding_scale",
            "dflash.attention.scale",
            "dflash.attention.value_scale",
            "dflash.logit_scale",
            "dflash.final_logit_softcapping",
            "dflash.hyper_connection_count",
        ] {
            if r.present(key) {
                r.fail(format!("unsupported drafter metadata {key}"));
            }
        }
        if let Some(act) = gguf.string("dflash.hidden_act")
            && act != b"silu"
        {
            r.fail("the drafter FFN must use SiLU");
        }
        if r.present("dflash.rope.dimension_sections") {
            r.fail("M-RoPE drafters are not supported");
        }
        let mask = gguf.u32("tokenizer.ggml.mask_token_id");
        if mask.is_none() {
            r.fail("the drafter has no mask token");
        }
        c.mask_token = mask.unwrap_or(0);
        c.lc_hidden = r.u32("lilicorr.hidden_size");
        c.lc_top_k = r.u32("lilicorr.candidate_topk");
        c.lc_heads = r.u32("lilicorr.num_heads");
        c.lc_layers = r.u32("lilicorr.num_layers");
        c.lc_factor = r.u32("lilicorr.factor_dim");
        c.lc_mlp = r.u32("lilicorr.mlp_hidden_size");
        c.lc_vector_eps = r.f32("lilicorr.vector_eps", 1e-4);
        c.lc_logit_scale = r.f32("lilicorr.logit_scale", 8.0);
        if !r.ok() {
            return Err(r.error());
        }
        let window = r.per_layer("dflash.attention.sliding_window_pattern", c.num_layers);
        match gguf.metadata("dflash.target_layers") {
            Some(Value::IntArray(values)) => {
                for &v in values {
                    c.target_layers
                        .push(if v < 0 || v >= i64::from(target.num_layers) {
                            u32::MAX
                        } else {
                            v as u32
                        });
                }
            }
            Some(Value::UIntArray(values)) => {
                for &v in values {
                    c.target_layers.push(if v >= u64::from(target.num_layers) {
                        u32::MAX
                    } else {
                        v as u32
                    });
                }
            }
            _ => {}
        }
        if !r.ok() {
            return Err(r.error());
        }
        let mut ok = !c.target_layers.is_empty() && c.target_layers.is_sorted();
        for &layer in &c.target_layers {
            ok = ok && layer != u32::MAX;
        }
        for &w in &window {
            ok = ok && w == 1;
        }
        if !ok {
            return Err("drafter target layers or sliding-window pattern are invalid".into());
        }
        let ok = c.hidden_size == target.hidden_size
            && c.num_layers != 0
            && c.ff_size != 0
            && c.num_heads != 0
            && c.num_kv_heads != 0
            && c.num_heads.is_multiple_of(c.num_kv_heads)
            && c.head_dim.is_multiple_of(2)
            && c.rms_eps > 0.0
            && c.rope_theta > 0.0
            && c.block_size >= 2
            && c.sliding_window != 0
            && c.conv_kernel != 0
            && c.conv_group != 0
            && c.hidden_size.is_multiple_of(c.conv_group)
            && c.mask_token < target.vocab_size
            && c.lc_hidden != 0
            && c.lc_heads != 0
            && c.lc_hidden.is_multiple_of(c.lc_heads)
            && c.lc_top_k != 0
            && c.lc_layers != 0
            && c.lc_factor != 0
            && c.lc_mlp != 0
            && c.lc_vector_eps > 0.0
            && u64::from(c.lc_top_k) * (u64::from(c.lc_top_k) + 1) <= u64::from(c.hidden_size);
        if !ok {
            return Err("dflash metadata describes an unsupported drafter".into());
        }
        // The kernels specialize on NVIDIA's drafter for Bolt.
        if c.num_layers != 5
            || c.num_heads != 32
            || c.num_kv_heads != 2
            || c.head_dim != 128
            || c.block_size != 8
            || c.conv_kernel != 2
            || c.conv_group != 16
            || c.lc_hidden != 1024
            || c.lc_top_k != 8
            || c.lc_heads != 8
            || c.lc_layers != 2
            || c.lc_factor != 1024
        {
            return Err("unsupported LiLiCorr drafter geometry".into());
        }
        r.finish(c)
    }
}

impl VisionConfig {
    /// Reads and validates the `clip.*` metadata and table shapes for
    /// `trunk` (`FromGguf`).
    pub fn from_gguf(gguf: &Gguf<'_>, trunk: &Config) -> Result<Self, String> {
        let mut r = Reader::new(gguf);
        if architecture(gguf) != Some(b"clip".as_slice())
            || gguf.string("clip.projector_type") != Some(b"nemotron_omni".as_slice())
        {
            return Err("GGUF is not a nemotron_omni vision projector".into());
        }
        if !r.bool("clip.has_vision_encoder", false) {
            r.fail("the projector has no vision encoder");
            return Err(r.error());
        }
        let mut c = VisionConfig {
            width: r.u32("clip.vision.embedding_length"),
            ..VisionConfig::default()
        };
        c.num_layers = r.u32("clip.vision.block_count");
        c.num_heads = r.u32("clip.vision.attention.head_count");
        c.ff_size = r.u32("clip.vision.feed_forward_length");
        c.patch_size = r.u32("clip.vision.patch_size");
        c.merge = r.u32("clip.vision.projector.scale_factor");
        c.projection = r.u32("clip.vision.projection_dim");
        c.eps = r.f32("clip.vision.attention.layer_norm_epsilon", 1e-6);
        let min_pixels = r.u32("clip.vision.image_min_pixels");
        let max_pixels = r.u32("clip.vision.image_max_pixels");
        if !r.bool("clip.use_gelu", false) {
            r.fail("the vision MLP must use GELU");
        }
        let rgb = |r: &mut Reader<'_, '_>, key: &str| {
            let mut out = [0.0f32; 3];
            let Some(Value::FloatArray(values)) = gguf
                .metadata(key)
                .filter(|v| matches!(v, Value::FloatArray(values) if values.len() == 3))
            else {
                r.fail(format!("missing or invalid GGUF array {key}"));
                return out;
            };
            for (slot, &value) in out.iter_mut().zip(values) {
                *slot = value as f32;
                if !slot.is_finite() {
                    r.fail(format!("invalid GGUF array {key}"));
                }
            }
            out
        };
        c.mean = rgb(&mut r, "clip.vision.image_mean");
        c.deviation = rgb(&mut r, "clip.vision.image_std");
        // Sizes the metadata does not state come from the tables' shapes.
        let dimension = |name: &str| -> u64 {
            gguf.tensor(name)
                .filter(|t| t.dims.len() == 2)
                .map_or(0, |t| t.dims[1])
        };
        let prefix = dimension("v.class_embd");
        let positions = dimension("v.position_embd.weight");
        let hidden = dimension("mm.model.mlp.1.weight");
        if !r.ok() {
            return Err(r.error());
        }
        let side = (positions as f64).sqrt().round() as u64;
        let patch_pixels = u64::from(c.patch_size) * u64::from(c.patch_size);
        let (min_pixels, max_pixels) = (u64::from(min_pixels), u64::from(max_pixels));
        let ok = c.num_heads != 0
            && c.width.is_multiple_of(c.num_heads)
            && patch_pixels != 0
            && c.merge != 0
            && side * side == positions
            && side <= u64::from(u32::MAX)
            && prefix <= u64::from(u32::MAX)
            && hidden <= u64::from(u32::MAX)
            && min_pixels.is_multiple_of(patch_pixels)
            && max_pixels.is_multiple_of(patch_pixels)
            && min_pixels <= max_pixels
            && min_pixels >= patch_pixels * u64::from(c.merge) * u64::from(c.merge);
        if !ok {
            return Err("clip metadata describes an unsupported vision projector".into());
        }
        c.prefix_tokens = prefix as u32;
        c.position_side = side as u32;
        c.projector_hidden = hidden as u32;
        c.min_patches = (min_pixels / patch_pixels) as u32;
        c.max_patches = (max_pixels / patch_pixels) as u32;
        if c.deviation.iter().any(|&d| d.is_nan() || d <= 0.0) {
            return Err("the vision projector's pixel deviation must be positive".into());
        }
        if c.projection != trunk.hidden_size {
            return Err(format!(
                "the vision projector's output width {} does not match the trunk's {}",
                c.projection, trunk.hidden_size
            ));
        }
        // The kernels specialize on NVIDIA's projector for Nemotron 3.5.
        if c.width != 1280
            || c.num_layers != 32
            || c.num_heads != 16
            || c.ff_size != 5120
            || c.patch_size != 16
            || c.merge != 2
            || c.prefix_tokens != 10
            || c.position_side != 128
            || c.projector_hidden != 20480
        {
            return Err("unsupported vision projector geometry".into());
        }
        r.finish(c)
    }
}
