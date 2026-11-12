//! Architecture parameters of a `gemma4` target or its `gemma4-assistant`
//! MTP drafter, read and validated from GGUF metadata (`config.cpp`).

use gufo_core::gguf::{Gguf, Value};

const TARGET_ARCHITECTURE: &str = "gemma4";
const DRAFT_ARCHITECTURE: &str = "gemma4-assistant";

/// Routed experts per token the kernels support (`kMaxExpertsUsed`).
pub const MAX_EXPERTS_USED: u32 = 8;
/// Experts per layer the kernels support (`kMaxExperts`).
pub const MAX_EXPERTS: u32 = 256;

/// Architecture parameters of a `gemma4` target or its `gemma4-assistant`
/// MTP drafter. Every value is read from the file; the checks only bound what
/// this runtime implements: text layers with interleaved sliding-window and
/// global attention, either dense or with routed experts beside the dense
/// MLP in every layer (26B-A4B), no per-layer embeddings and no KV sharing
/// inside the target.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// 60 for the 31B target, 4 for its draft.
    pub num_layers: u32,
    /// 5376 (target) or 1024 (draft).
    pub hidden_size: u32,
    /// Dense MLP: 21504 (31B), 2112 (26B-A4B).
    pub ffn_size: u32,
    /// 262144.
    pub context_length: u32,
    pub rms_eps: f32,

    /// Query heads, identical per layer.
    pub num_heads: u32,
    /// Per layer: 16 sliding, 4 global.
    pub kv_heads: Vec<u32>,
    /// Per layer: 1 sliding, 0 global.
    pub sliding: Vec<u8>,

    /// 512
    pub head_dim_global: u32,
    /// 256
    pub head_dim_sliding: u32,
    /// 512; rope_freqs selects the pairs.
    pub rope_dim_global: u32,
    /// 256
    pub rope_dim_sliding: u32,
    /// 1e6
    pub rope_theta_global: f32,
    /// 1e4
    pub rope_theta_sliding: f32,
    /// 1024 keys, including the query.
    pub sliding_window: u32,
    /// 30; zero disables the cap.
    pub final_logit_softcap: f32,

    // Mixture of experts (26B-A4B); zero experts means a dense target.
    /// 128
    pub num_experts: u32,
    /// 8 routed experts per token.
    pub experts_used: u32,
    /// 704 per expert projection.
    pub expert_ffn_size: u32,

    // Draft (`gemma4-assistant`) only.
    /// embedding_length_out, 5376.
    pub target_hidden_size: u32,
    /// Draft layers reading target KV.
    pub shared_kv_layers: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_layers: 0,
            hidden_size: 0,
            ffn_size: 0,
            context_length: 0,
            rms_eps: 1e-6,
            num_heads: 0,
            kv_heads: Vec::new(),
            sliding: Vec::new(),
            head_dim_global: 0,
            head_dim_sliding: 0,
            rope_dim_global: 0,
            rope_dim_sliding: 0,
            rope_theta_global: 0.0,
            rope_theta_sliding: 0.0,
            sliding_window: 0,
            final_logit_softcap: 0.0,
            num_experts: 0,
            experts_used: 0,
            expert_ffn_size: 0,
            target_hidden_size: 0,
            shared_kv_layers: 0,
        }
    }
}

impl Config {
    pub fn has_experts(&self) -> bool {
        self.num_experts != 0
    }

    pub fn is_sliding(&self, layer: u32) -> bool {
        self.sliding[layer as usize] != 0
    }

    pub fn head_dim(&self, layer: u32) -> u32 {
        if self.is_sliding(layer) {
            self.head_dim_sliding
        } else {
            self.head_dim_global
        }
    }

    pub fn rope_dim(&self, layer: u32) -> u32 {
        if self.is_sliding(layer) {
            self.rope_dim_sliding
        } else {
            self.rope_dim_global
        }
    }

    pub fn rope_theta(&self, layer: u32) -> f32 {
        if self.is_sliding(layer) {
            self.rope_theta_sliding
        } else {
            self.rope_theta_global
        }
    }

    pub fn q_dim(&self, layer: u32) -> u32 {
        self.num_heads * self.head_dim(layer)
    }

    pub fn kv_dim(&self, layer: u32) -> u32 {
        self.kv_heads[layer as usize] * self.head_dim(layer)
    }

    pub fn global_layer_count(&self) -> u32 {
        self.sliding.iter().filter(|&&s| s == 0).count() as u32
    }

    /// Target layer whose KV cache a shared-KV draft layer attends: the last
    /// target layer of the same attention kind (`target.num_layers` when
    /// there is none).
    pub fn shared_kv_source(&self, draft_layer: u32, target: &Config) -> u32 {
        let want_sliding = self.is_sliding(draft_layer);
        (0..target.num_layers)
            .rev()
            .find(|&l| target.is_sliding(l) == want_sliding)
            .unwrap_or(target.num_layers)
    }

    /// Reads a `gemma4` target. Rejects variants this runtime does not
    /// implement (per-layer embeddings, KV-shared target layers, more than
    /// [`MAX_EXPERTS_USED`] routed experts per token).
    pub fn from_gguf(gguf: &Gguf<'_>) -> Result<Self, String> {
        let mut r = Reader::new(gguf, TARGET_ARCHITECTURE);
        if gguf.string("general.architecture") != Some(TARGET_ARCHITECTURE.as_bytes()) {
            return Err("GGUF architecture is not gemma4".into());
        }
        let mut c = read_common(&mut r);
        r.check()?;
        c.final_logit_softcap = r.f32("final_logit_softcapping", false, 0.0);
        if r.u32("attention.shared_kv_layers", false, 0) != 0 {
            r.fail("gemma4 targets with shared KV layers are not supported".into());
        }
        c.num_experts = r.u32("expert_count", false, 0);
        if r.ok() && c.num_experts != 0 {
            c.experts_used = r.u32("expert_used_count", true, 0);
            let widths = r.per_layer("expert_feed_forward_length", c.num_layers);
            if r.ok() {
                c.expert_ffn_size = widths[0];
                if widths.iter().any(|&v| v != c.expert_ffn_size) {
                    r.fail("per-layer expert widths are not supported".into());
                }
            }
            if r.ok()
                && (c.experts_used == 0
                    || c.experts_used > c.num_experts
                    || c.experts_used > MAX_EXPERTS_USED
                    || c.num_experts > MAX_EXPERTS)
            {
                r.fail(format!(
                    "gemma4 expert routing must use 1..{MAX_EXPERTS_USED} of at most \
                     {MAX_EXPERTS} experts"
                ));
            }
        } else if r.ok() && gguf.tensor("blk.0.ffn_gate_inp.weight").is_some() {
            r.fail("gemma4 expert tensors without gemma4.expert_count".into());
        }
        if r.ok() && c.final_logit_softcap < 0.0 {
            r.fail("gemma4 final logit softcap must not be negative".into());
        }
        if r.ok() && (c.global_layer_count() == 0 || c.global_layer_count() == c.num_layers) {
            r.fail("gemma4 targets need both sliding and global layers".into());
        }
        r.check()?;
        Ok(c)
    }

    /// Reads a `gemma4-assistant` MTP drafter and checks that every draft
    /// layer can attend its target KV source with identical geometry.
    pub fn draft_from_gguf(gguf: &Gguf<'_>, target: &Config) -> Result<Self, String> {
        let mut r = Reader::new(gguf, DRAFT_ARCHITECTURE);
        if gguf.string("general.architecture") != Some(DRAFT_ARCHITECTURE.as_bytes()) {
            return Err("GGUF architecture is not gemma4-assistant".into());
        }
        let mut c = read_common(&mut r);
        r.check()?;
        c.target_hidden_size = r.u32("embedding_length_out", true, 0);
        c.shared_kv_layers = r.u32("attention.shared_kv_layers", true, 0);
        let nextn = r.u32("nextn_predict_layers", false, c.num_layers);
        r.check()?;
        if c.target_hidden_size != target.hidden_size {
            r.fail("gemma4-assistant embedding_length_out must match the target".into());
        } else if c.shared_kv_layers != c.num_layers || nextn != c.num_layers {
            r.fail("gemma4-assistant drafts must share KV in every layer".into());
        } else if c.head_dim_global != target.head_dim_global
            || c.head_dim_sliding != target.head_dim_sliding
            || c.rope_theta_global != target.rope_theta_global
            || c.rope_theta_sliding != target.rope_theta_sliding
            || c.sliding_window != target.sliding_window
            || c.rms_eps != target.rms_eps
        {
            r.fail("gemma4-assistant attention constants differ from the target".into());
        } else {
            for l in 0..c.num_layers {
                let source = c.shared_kv_source(l, target);
                if source >= target.num_layers
                    || c.kv_heads[l as usize] != target.kv_heads[source as usize]
                {
                    r.fail(format!(
                        "gemma4-assistant layer {l} cannot attend its target KV source"
                    ));
                    break;
                }
            }
        }
        r.check()?;
        Ok(c)
    }
}

/// Metadata access under one architecture prefix that keeps the first
/// failure (`Reader`).
struct Reader<'g, 'a> {
    gguf: &'g Gguf<'a>,
    prefix: String,
    error: Option<String>,
}

impl<'g, 'a> Reader<'g, 'a> {
    fn new(gguf: &'g Gguf<'a>, architecture: &str) -> Self {
        Self {
            gguf,
            prefix: format!("{architecture}."),
            error: None,
        }
    }

    fn ok(&self) -> bool {
        self.error.is_none()
    }

    fn fail(&mut self, message: String) {
        if self.error.is_none() {
            self.error = Some(message);
        }
    }

    fn check(&self) -> Result<(), String> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn key(&self, suffix: &str) -> String {
        format!("{}{suffix}", self.prefix)
    }

    fn u32(&mut self, suffix: &str, required: bool, fallback: u32) -> u32 {
        let key = self.key(suffix);
        if let Some(value) = self.gguf.u32(&key) {
            return value;
        }
        if required || self.gguf.metadata(&key).is_some() {
            self.fail(format!("missing or invalid GGUF integer {key}"));
        }
        fallback
    }

    fn f32(&mut self, suffix: &str, required: bool, fallback: f32) -> f32 {
        let key = self.key(suffix);
        if let Some(value) = self.gguf.f32(&key).filter(|v| v.is_finite()) {
            return value;
        }
        if required || self.gguf.metadata(&key).is_some() {
            self.fail(format!("missing or invalid GGUF float {key}"));
        }
        fallback
    }

    /// Integer or boolean arrays of any width; GGUF writers pick the type.
    fn u64_array(&mut self, suffix: &str) -> Vec<u64> {
        let key = self.key(suffix);
        match self.gguf.metadata(&key) {
            None => {
                self.fail(format!("missing GGUF array {key}"));
                Vec::new()
            }
            Some(Value::UIntArray(values)) => values.clone(),
            Some(Value::IntArray(values)) => {
                let mut out = Vec::with_capacity(values.len());
                for &v in values {
                    if v < 0 {
                        self.fail(format!("negative GGUF array value {key}"));
                        return Vec::new();
                    }
                    out.push(v as u64);
                }
                out
            }
            Some(_) => {
                self.fail(format!("GGUF array must contain integers: {key}"));
                Vec::new()
            }
        }
    }

    /// Per-layer values stored either as one scalar or as one entry per layer.
    fn per_layer(&mut self, suffix: &str, layers: u32) -> Vec<u32> {
        let key = self.key(suffix);
        if self.gguf.metadata(&key).is_none() {
            self.fail(format!("missing GGUF value {key}"));
            return Vec::new();
        }
        if let Some(scalar) = self.gguf.u32(&key) {
            return vec![scalar; layers as usize];
        }
        let values = self.u64_array(suffix);
        if !self.ok() {
            return Vec::new();
        }
        if values.len() != layers as usize {
            self.fail(format!("{key} must describe every layer"));
            return Vec::new();
        }
        let mut out = Vec::with_capacity(values.len());
        for v in values {
            if v == 0 || v > 0xffff_ffff {
                self.fail(format!("{key} has an invalid entry"));
                return Vec::new();
            }
            out.push(v as u32);
        }
        out
    }
}

/// Fields shared by the target and the drafter (`ReadCommon`).
fn read_common(r: &mut Reader<'_, '_>) -> Config {
    let mut c = Config {
        num_layers: r.u32("block_count", true, 0),
        hidden_size: r.u32("embedding_length", true, 0),
        context_length: r.u32("context_length", true, 0),
        rms_eps: r.f32("attention.layer_norm_rms_epsilon", true, 0.0),
        num_heads: r.u32("attention.head_count", true, 0),
        ..Config::default()
    };
    if !r.ok() {
        return c;
    }
    if c.num_layers == 0
        || c.hidden_size == 0
        || c.num_heads == 0
        || c.context_length == 0
        || c.rms_eps.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
    {
        r.fail("gemma4 dimensions must be positive".into());
        return c;
    }

    let ffn = r.per_layer("feed_forward_length", c.num_layers);
    if !r.ok() {
        return c;
    }
    c.ffn_size = ffn[0];
    if ffn.iter().any(|&v| v != c.ffn_size) {
        r.fail("per-layer feed-forward widths are not supported".into());
        return c;
    }
    c.kv_heads = r.per_layer("attention.head_count_kv", c.num_layers);

    let pattern = r.u64_array("attention.sliding_window_pattern");
    if r.ok() && pattern.len() != c.num_layers as usize {
        r.fail("attention.sliding_window_pattern must describe every layer".into());
    }
    if !r.ok() {
        return c;
    }
    c.sliding = pattern.iter().map(|&v| u8::from(v != 0)).collect();

    c.head_dim_global = r.u32("attention.key_length", true, 0);
    c.head_dim_sliding = r.u32("attention.key_length_swa", true, 0);
    if r.u32("attention.value_length", true, 0) != c.head_dim_global
        || r.u32("attention.value_length_swa", true, 0) != c.head_dim_sliding
    {
        r.fail("gemma4 requires equal key and value head dimensions".into());
    }
    c.rope_dim_global = r.u32("rope.dimension_count", true, 0);
    c.rope_dim_sliding = r.u32("rope.dimension_count_swa", true, 0);
    c.rope_theta_global = r.f32("rope.freq_base", true, 0.0);
    c.rope_theta_sliding = r.f32("rope.freq_base_swa", true, 0.0);
    c.sliding_window = r.u32("attention.sliding_window", true, 0);
    if !r.ok() {
        return c;
    }

    for dim in [c.head_dim_global, c.head_dim_sliding] {
        // Kernels reduce heads in 64-wide lane groups and GGUF K-quants need
        // 256-aligned reductions for the output projection.
        if dim == 0 || !dim.is_multiple_of(64) {
            r.fail("gemma4 head dimensions must be positive multiples of 64".into());
            return c;
        }
    }
    if c.rope_dim_global != c.head_dim_global || c.rope_dim_sliding != c.head_dim_sliding {
        r.fail("gemma4 rope must span the full head dimension".into());
        return c;
    }
    let positive = |v: f32| v.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater);
    if !positive(c.rope_theta_global) || !positive(c.rope_theta_sliding) {
        r.fail("gemma4 rope bases must be positive".into());
        return c;
    }
    if c.sliding_window == 0 {
        r.fail("gemma4 sliding window must be positive".into());
        return c;
    }
    for l in 0..c.num_layers {
        if !c.num_heads.is_multiple_of(c.kv_heads[l as usize]) {
            r.fail(format!(
                "gemma4 query heads must be a multiple of KV heads in layer {l}"
            ));
            return c;
        }
    }
    if r.u32("embedding_length_per_layer_input", false, 0) != 0 {
        r.fail("gemma4 per-layer embeddings are not supported".into());
    }
    c
}
