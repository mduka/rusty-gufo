//! The C++ extension's `config_test.cpp`: block classification and the
//! supported profiles of the trunk and the drafter, from synthetic metadata.

#![allow(clippy::unwrap_used)]

use gufo_core::gguf::Gguf;
use gufo_nemotron::config::{BlockType, Config, DrafterConfig};

// GGUF value types.
const UINT32: u32 = 4;
const INT32: u32 = 5;
const FLOAT32: u32 = 6;
const BOOL: u32 = 7;
const STRING: u32 = 8;
const ARRAY: u32 = 9;

/// Metadata-only GGUF image; Config reads no tensors.
#[derive(Default)]
struct MetadataGguf {
    body: Vec<u8>,
    count: u64,
}

impl MetadataGguf {
    fn text(&mut self, s: &str) {
        self.body.extend_from_slice(&(s.len() as u64).to_le_bytes());
        self.body.extend_from_slice(s.as_bytes());
    }

    fn key(&mut self, key: &str, ty: u32) {
        self.text(key);
        self.body.extend_from_slice(&ty.to_le_bytes());
        self.count += 1;
    }

    fn string(&mut self, key: &str, value: &str) {
        self.key(key, STRING);
        self.text(value);
    }

    fn u32(&mut self, key: &str, value: u32) {
        self.key(key, UINT32);
        self.body.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, key: &str, value: f32) {
        self.key(key, FLOAT32);
        self.body.extend_from_slice(&value.to_le_bytes());
    }

    fn bool(&mut self, key: &str, value: bool) {
        self.key(key, BOOL);
        self.body.push(u8::from(value));
    }

    fn i32_array(&mut self, key: &str, values: &[i32]) {
        self.key(key, ARRAY);
        self.body.extend_from_slice(&INT32.to_le_bytes());
        self.body
            .extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            self.body.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn bool_array(&mut self, key: &str, values: &[bool]) {
        self.key(key, ARRAY);
        self.body.extend_from_slice(&BOOL.to_le_bytes());
        self.body
            .extend_from_slice(&(values.len() as u64).to_le_bytes());
        for &v in values {
            self.body.push(u8::from(v));
        }
    }

    fn build(&self) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&self.count.to_le_bytes());
        out.extend_from_slice(&self.body);
        out.resize(out.len().div_ceil(32) * 32, 0);
        out
    }
}

// The pinned artifact's block order: M = Mamba-2, E = MoE, * = attention.
const LAYOUT: &str =
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME";

struct Profile {
    hidden: u32,
    gating: u32,
    kv_entries: usize,
    kv_on_first_moe: bool,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            hidden: 4096,
            gating: 2,
            kv_entries: LAYOUT.len(),
            kv_on_first_moe: false,
        }
    }
}

fn trunk_metadata(p: &Profile) -> Vec<u8> {
    let mut g = MetadataGguf::default();
    let a = |key: &str| format!("nemotron_h_moe.{key}");
    g.string("general.architecture", "nemotron_h_moe");
    g.u32(&a("block_count"), LAYOUT.len() as u32);
    g.u32(&a("context_length"), 262144);
    g.u32(&a("embedding_length"), p.hidden);
    g.u32(&a("vocab_size"), 131072);
    g.u32(&a("attention.head_count"), 32);
    g.f32(&a("attention.layer_norm_rms_epsilon"), 1e-5);
    g.f32(&a("attention.layer_norm_epsilon"), 1e-5);
    g.u32(&a("attention.key_length"), 128);
    g.u32(&a("attention.value_length"), 128);
    g.u32(&a("expert_count"), 496);
    g.u32(&a("expert_group_count"), 1);
    g.u32(&a("expert_group_used_count"), 1);
    g.u32(&a("expert_gating_func"), p.gating);
    g.u32(&a("expert_shared_feed_forward_length"), 5376);
    g.u32(&a("expert_shared_count"), 1);
    g.bool(&a("expert_weights_norm"), true);
    g.f32(&a("expert_weights_scale"), 5.0);
    g.u32(&a("moe_latent_size"), 1024);
    g.u32(&a("rope.dimension_count"), 128);
    g.u32(&a("ssm.conv_kernel"), 4);
    g.u32(&a("ssm.state_size"), 128);
    g.u32(&a("ssm.group_count"), 8);
    g.u32(&a("ssm.inner_size"), 8192);
    g.u32(&a("ssm.time_step_rank"), 128);

    let (mut kv, mut ff, mut used) = (Vec::new(), Vec::new(), Vec::new());
    let mut first_moe = true;
    for (i, block) in LAYOUT.bytes().enumerate() {
        let moe = block == b'E';
        // One narrower block, as in the pinned file, keeps widths per block.
        let width = if i == 3 { 1536 } else { 2048 };
        kv.push(
            if block == b'*' || (moe && first_moe && p.kv_on_first_moe) {
                2
            } else {
                0
            },
        );
        ff.push(if moe { width } else { 0 });
        used.push(if moe { 9 } else { 0 });
        first_moe = first_moe && !moe;
    }
    kv.resize(p.kv_entries, 0);
    g.i32_array(&a("attention.head_count_kv"), &kv);
    g.i32_array(&a("feed_forward_length"), &ff);
    g.i32_array(&a("expert_feed_forward_length"), &ff);
    g.i32_array(&a("expert_used_count"), &used);
    g.build()
}

fn parse(p: &Profile) -> Result<Config, String> {
    let bytes = trunk_metadata(p);
    let gguf = Gguf::parse(&bytes).expect("synthetic metadata GGUF must load");
    Config::from_gguf(&gguf)
}

#[test]
fn pinned_profile() {
    let c = parse(&Profile::default()).unwrap();
    let mut counts = [0usize; 3];
    for &b in &c.blocks {
        counts[b as usize] += 1;
    }
    assert_eq!(
        counts,
        [40, 8, 40],
        "40 Mamba, 8 attention and 40 MoE blocks"
    );
    assert!(
        c.blocks[0] == BlockType::Mamba
            && c.blocks[1] == BlockType::Moe
            && c.blocks[7] == BlockType::Attention,
        "block classification follows head_count_kv and ff widths"
    );
    assert!(
        c.expert_ff[3] == 1536
            && c.expert_ff[1] == 2048
            && c.expert_ff[0] == 0
            && c.experts_used[1] == 9
            && c.experts_used[7] == 0,
        "expert widths and top-k are kept per block"
    );
    assert!(
        c.num_heads == 32 && c.num_kv_heads == 2,
        "attention geometry comes from attention blocks"
    );
    assert!(
        c.ssm_in_projection() == 18560 && c.ssm_conv_channels() == 10240 && c.ssm_head_dim() == 64,
        "Mamba-2 projection layout is z|x|B|C|dt"
    );
    assert!(
        c.expert_weights_norm && c.expert_weights_scale == 5.0,
        "routing weight normalization and scale"
    );
}

#[test]
fn rejections() {
    let p = Profile {
        gating: 1,
        ..Profile::default()
    };
    assert!(parse(&p).is_err(), "softmax gating accepted");
    let p = Profile {
        kv_entries: LAYOUT.len() - 1,
        ..Profile::default()
    };
    assert!(parse(&p).is_err(), "short per-block array accepted");
    let p = Profile {
        hidden: 4224,
        ..Profile::default()
    };
    assert!(parse(&p).is_err(), "unsupported hidden width accepted");
    let p = Profile {
        kv_on_first_moe: true,
        ..Profile::default()
    };
    assert!(
        parse(&p).is_err(),
        "MoE block with attention heads accepted"
    );
}

struct DrafterProfile {
    causal: bool,
    selector_rank: u32,
    last_target_layer: i32,
    mask_token: bool,
    block_size: u32,
}

impl Default for DrafterProfile {
    fn default() -> Self {
        Self {
            causal: false,
            selector_rank: 0,
            last_target_layer: 86,
            mask_token: true,
            block_size: 8,
        }
    }
}

fn parse_drafter(p: &DrafterProfile) -> Result<DrafterConfig, String> {
    let target = parse(&Profile::default()).expect("pinned profile");
    let mut g = MetadataGguf::default();
    g.string("general.architecture", "dflash");
    g.u32("dflash.block_count", 5);
    g.u32("dflash.context_length", 262144);
    g.u32("dflash.embedding_length", 4096);
    g.u32("dflash.feed_forward_length", 12288);
    g.u32("dflash.attention.head_count", 32);
    g.u32("dflash.attention.head_count_kv", 2);
    g.f32("dflash.rope.freq_base", 10000.0);
    g.f32("dflash.attention.layer_norm_rms_epsilon", 1e-6);
    g.u32("dflash.attention.key_length", 128);
    g.u32("dflash.attention.value_length", 128);
    g.u32("dflash.block_size", p.block_size);
    g.u32("dflash.conv_kernel_size", 2);
    g.u32("dflash.conv_group_size", 16);
    g.u32("dflash.selector_rank", p.selector_rank);
    g.u32("dflash.selector_top_k", 0);
    g.i32_array(
        "dflash.target_layers",
        &[2, 19, 36, 52, 69, p.last_target_layer],
    );
    g.u32("dflash.attention.sliding_window", 1024);
    g.bool_array("dflash.attention.sliding_window_pattern", &[true; 5]);
    g.bool("dflash.attention.causal", p.causal);
    g.u32("lilicorr.hidden_size", 1024);
    g.u32("lilicorr.candidate_topk", 8);
    g.u32("lilicorr.num_heads", 8);
    g.u32("lilicorr.num_layers", 2);
    g.u32("lilicorr.factor_dim", 1024);
    g.u32("lilicorr.mlp_hidden_size", 2048);
    g.f32("lilicorr.vector_eps", 1e-4);
    g.f32("lilicorr.logit_scale", 8.0);
    if p.mask_token {
        g.u32("tokenizer.ggml.mask_token_id", 990);
    }
    let bytes = g.build();
    let gguf = Gguf::parse(&bytes).expect("synthetic drafter GGUF must load");
    DrafterConfig::from_gguf(&gguf, &target)
}

#[test]
fn drafter() {
    let c = parse_drafter(&DrafterProfile::default()).unwrap();
    assert_eq!(
        c.target_layers,
        vec![2, 19, 36, 52, 69, 86],
        "drafter target layers"
    );
    assert!(
        c.feature_size() == 6 * 4096
            && c.conv_projection() == 1024
            && c.slots() == 7
            && c.mask_token == 990
            && c.sliding_window == 1024,
        "drafter geometry"
    );
    let p = DrafterProfile {
        causal: true,
        ..DrafterProfile::default()
    };
    assert!(parse_drafter(&p).is_err(), "causal drafter accepted");
    let p = DrafterProfile {
        selector_rank: 64,
        ..DrafterProfile::default()
    };
    assert!(parse_drafter(&p).is_err(), "DFlash2 selector accepted");
    let p = DrafterProfile {
        last_target_layer: 88,
        ..DrafterProfile::default()
    };
    assert!(
        parse_drafter(&p).is_err(),
        "target layer past the trunk accepted"
    );
    let p = DrafterProfile {
        mask_token: false,
        ..DrafterProfile::default()
    };
    assert!(
        parse_drafter(&p).is_err(),
        "drafter without a mask token accepted"
    );
    let p = DrafterProfile {
        block_size: 16,
        ..DrafterProfile::default()
    };
    assert!(parse_drafter(&p).is_err(), "other block size accepted");
}
