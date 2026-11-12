//! The extension's `config_test.cpp`: metadata of the target and the
//! drafter, the supported variants, and tensor binding of a two-layer
//! miniature (dense and with experts).

#![allow(clippy::unwrap_used)]

mod common;

use common::{Meta, Metadata, TensorSpec, Tensors, gguf_image};
use gufo_formats::GgmlType;
use gufo_gemma4::config::Config;
use gufo_gemma4::weights::{DraftWeights, GgufShard, ModelWeights};

fn meta(entries: &[(&str, Meta)]) -> Metadata {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

/// A two-layer miniature of the 31B layout: one sliding and one global
/// layer, with the per-layer KV head array and rope factors of the real
/// file.
fn target_metadata() -> Metadata {
    let p = "gemma4.";
    meta(&[
        ("general.architecture", Meta::Str("gemma4".into())),
        (&format!("{p}block_count"), Meta::U32(2)),
        (&format!("{p}context_length"), Meta::U32(4096)),
        (&format!("{p}embedding_length"), Meta::U32(64)),
        (&format!("{p}feed_forward_length"), Meta::U32(128)),
        (&format!("{p}attention.head_count"), Meta::U32(4)),
        (
            &format!("{p}attention.head_count_kv"),
            Meta::I32Array(vec![2, 1]),
        ),
        (
            &format!("{p}attention.sliding_window_pattern"),
            Meta::BoolArray(vec![true, false]),
        ),
        (&format!("{p}attention.key_length"), Meta::U32(128)),
        (&format!("{p}attention.value_length"), Meta::U32(128)),
        (&format!("{p}attention.key_length_swa"), Meta::U32(64)),
        (&format!("{p}attention.value_length_swa"), Meta::U32(64)),
        (&format!("{p}rope.dimension_count"), Meta::U32(128)),
        (&format!("{p}rope.dimension_count_swa"), Meta::U32(64)),
        (&format!("{p}rope.freq_base"), Meta::F32(1_000_000.0)),
        (&format!("{p}rope.freq_base_swa"), Meta::F32(10_000.0)),
        (&format!("{p}attention.sliding_window"), Meta::U32(16)),
        (
            &format!("{p}attention.layer_norm_rms_epsilon"),
            Meta::F32(1e-6),
        ),
        (&format!("{p}final_logit_softcapping"), Meta::F32(30.0)),
        (&format!("{p}attention.shared_kv_layers"), Meta::U32(0)),
        (
            &format!("{p}embedding_length_per_layer_input"),
            Meta::U32(0),
        ),
    ])
}

fn draft_metadata() -> Metadata {
    let p = "gemma4-assistant.";
    meta(&[
        ("general.architecture", Meta::Str("gemma4-assistant".into())),
        (&format!("{p}block_count"), Meta::U32(2)),
        (&format!("{p}context_length"), Meta::U32(4096)),
        (&format!("{p}embedding_length"), Meta::U32(32)),
        (&format!("{p}feed_forward_length"), Meta::U32(64)),
        (&format!("{p}attention.head_count"), Meta::U32(4)),
        (
            &format!("{p}attention.head_count_kv"),
            Meta::I32Array(vec![2, 1]),
        ),
        (
            &format!("{p}attention.sliding_window_pattern"),
            Meta::BoolArray(vec![true, false]),
        ),
        (&format!("{p}attention.key_length"), Meta::U32(128)),
        (&format!("{p}attention.value_length"), Meta::U32(128)),
        (&format!("{p}attention.key_length_swa"), Meta::U32(64)),
        (&format!("{p}attention.value_length_swa"), Meta::U32(64)),
        (&format!("{p}rope.dimension_count"), Meta::U32(128)),
        (&format!("{p}rope.dimension_count_swa"), Meta::U32(64)),
        (&format!("{p}rope.freq_base"), Meta::F32(1_000_000.0)),
        (&format!("{p}rope.freq_base_swa"), Meta::F32(10_000.0)),
        (&format!("{p}attention.sliding_window"), Meta::U32(16)),
        (
            &format!("{p}attention.layer_norm_rms_epsilon"),
            Meta::F32(1e-6),
        ),
        (&format!("{p}attention.shared_kv_layers"), Meta::U32(2)),
        (&format!("{p}embedding_length_out"), Meta::U32(64)),
        (&format!("{p}nextn_predict_layers"), Meta::U32(2)),
    ])
}

fn parse(metadata: &Metadata) -> Result<Config, String> {
    let image = gguf_image(metadata, &Tensors::new());
    let gguf = gufo_core::gguf::Gguf::parse(&image).expect("in-memory GGUF");
    let config = Config::from_gguf(&gguf);
    if let Err(error) = &config {
        assert!(
            !error.is_empty(),
            "an error message accompanies every rejection"
        );
    }
    config
}

#[test]
fn target() {
    let c = parse(&target_metadata()).expect("valid gemma4 metadata");
    assert!(
        c.num_layers == 2 && c.hidden_size == 64 && c.ffn_size == 128,
        "dimensions"
    );
    assert!(c.is_sliding(0) && !c.is_sliding(1), "sliding pattern");
    assert!(
        c.head_dim(0) == 64 && c.head_dim(1) == 128,
        "per-layer head dims"
    );
    assert!(c.kv_heads == [2, 1], "per-layer KV heads");
    assert!(c.q_dim(1) == 512 && c.kv_dim(1) == 128, "projection widths");
    assert!(
        c.rope_theta(0) == 10_000.0 && c.rope_theta(1) == 1_000_000.0,
        "per-layer rope bases"
    );
    assert!(
        c.sliding_window == 16 && c.final_logit_softcap == 30.0,
        "window and softcap"
    );
    assert_eq!(c.global_layer_count(), 1, "global layer count");
}

#[test]
fn rejections() {
    let valid = target_metadata();
    let reject = |key: &str, value: Meta, what: &str| {
        let mut wrong = valid.clone();
        wrong.insert(key.to_owned(), value);
        assert!(parse(&wrong).is_err(), "{what}");
    };
    reject(
        "general.architecture",
        Meta::Str("qwen35".into()),
        "other architecture",
    );
    reject(
        "gemma4.attention.head_count_kv",
        Meta::I32Array(vec![2]),
        "short KV array",
    );
    reject(
        "gemma4.attention.head_count_kv",
        Meta::I32Array(vec![3, 1]),
        "KV heads not dividing query heads",
    );
    reject(
        "gemma4.attention.head_count_kv",
        Meta::I32Array(vec![-2, 1]),
        "negative KV heads",
    );
    reject(
        "gemma4.attention.sliding_window_pattern",
        Meta::BoolArray(vec![true]),
        "short pattern",
    );
    reject(
        "gemma4.attention.sliding_window_pattern",
        Meta::BoolArray(vec![true, true]),
        "no global layer",
    );
    reject(
        "gemma4.attention.value_length",
        Meta::U32(64),
        "key and value widths differ",
    );
    reject(
        "gemma4.attention.key_length_swa",
        Meta::U32(48),
        "head dimension not a multiple of 64",
    );
    reject("gemma4.rope.dimension_count", Meta::U32(64), "partial rope");
    reject(
        "gemma4.embedding_length_per_layer_input",
        Meta::U32(256),
        "per-layer embeddings",
    );
    reject(
        "gemma4.attention.shared_kv_layers",
        Meta::U32(1),
        "KV-shared target layers",
    );
    reject(
        "gemma4.attention.sliding_window",
        Meta::U32(0),
        "zero window",
    );
    reject(
        "gemma4.rope.freq_base",
        Meta::F32(-1.0),
        "negative rope base",
    );
    for key in [
        "gemma4.block_count",
        "gemma4.attention.head_count_kv",
        "gemma4.attention.sliding_window_pattern",
        "gemma4.attention.key_length_swa",
        "gemma4.rope.freq_base_swa",
        "gemma4.attention.layer_norm_rms_epsilon",
    ] {
        let mut wrong = valid.clone();
        wrong.remove(key);
        assert!(parse(&wrong).is_err(), "missing {key}");
    }
    // A scalar KV head count applies to every layer.
    let mut scalar = valid.clone();
    scalar.insert("gemma4.attention.head_count_kv".into(), Meta::U32(1));
    let c = parse(&scalar).expect("scalar KV head count");
    assert_eq!(c.kv_heads, [1, 1], "scalar KV head count");
}

/// The miniature with routed experts beside the dense MLP (26B-A4B).
fn moe_metadata() -> Metadata {
    let mut m = target_metadata();
    m.insert("gemma4.expert_count".into(), Meta::U32(8));
    m.insert("gemma4.expert_used_count".into(), Meta::U32(2));
    m.insert("gemma4.expert_feed_forward_length".into(), Meta::U32(32));
    m
}

#[test]
fn expert_config() {
    let c = parse(&moe_metadata()).expect("expert metadata");
    assert!(
        c.has_experts() && c.num_experts == 8 && c.experts_used == 2 && c.expert_ffn_size == 32,
        "expert metadata"
    );
    assert!(
        !parse(&target_metadata()).unwrap().has_experts(),
        "dense target has experts"
    );
    let reject = |key: &str, value: Meta, what: &str| {
        let mut wrong = moe_metadata();
        wrong.insert(key.to_owned(), value);
        assert!(parse(&wrong).is_err(), "{what}");
    };
    reject(
        "gemma4.expert_used_count",
        Meta::U32(9),
        "more routed experts than the kernels support",
    );
    reject(
        "gemma4.expert_used_count",
        Meta::U32(0),
        "no routed experts",
    );
    reject("gemma4.expert_count", Meta::U32(512), "too many experts");
    let mut missing = moe_metadata();
    missing.remove("gemma4.expert_feed_forward_length");
    assert!(parse(&missing).is_err(), "missing expert width");
}

/// Q8_0 bytes for a `[cols, rows]` matrix.
fn q8(cols: u64, rows: u64) -> TensorSpec {
    TensorSpec {
        dims: vec![cols, rows],
        ty: GgmlType::Q8_0,
        data: vec![0; (cols / 32 * 34 * rows) as usize],
    }
}

fn f32s(n: u64, value: f32) -> TensorSpec {
    TensorSpec {
        dims: vec![n],
        ty: GgmlType::F32,
        data: (0..n).flat_map(|_| value.to_le_bytes()).collect(),
    }
}

fn target_tensors(c: &Config, vocab: u64) -> Tensors {
    let mut t = Tensors::new();
    let hidden = u64::from(c.hidden_size);
    t.insert("token_embd.weight".into(), q8(hidden, vocab));
    t.insert("output_norm.weight".into(), f32s(hidden, 1.0));
    t.insert(
        "rope_freqs.weight".into(),
        f32s(u64::from(c.head_dim_global / 2), 1.0),
    );
    for l in 0..c.num_layers {
        let p = format!("blk.{l}.");
        let mut add = |name: &str, spec: TensorSpec| {
            t.insert(format!("{p}{name}"), spec);
        };
        let (q, kv, ffn, hd) = (
            u64::from(c.q_dim(l)),
            u64::from(c.kv_dim(l)),
            u64::from(c.ffn_size),
            u64::from(c.head_dim(l)),
        );
        add("attn_norm.weight", f32s(hidden, 1.0));
        add("attn_q.weight", q8(hidden, q));
        add("attn_k.weight", q8(hidden, kv));
        if c.is_sliding(l) {
            add("attn_v.weight", q8(hidden, kv));
        }
        add("attn_q_norm.weight", f32s(hd, 1.0));
        add("attn_k_norm.weight", f32s(hd, 1.0));
        add("attn_output.weight", q8(q, hidden));
        add("post_attention_norm.weight", f32s(hidden, 1.0));
        add("ffn_norm.weight", f32s(hidden, 1.0));
        add("ffn_gate.weight", q8(hidden, ffn));
        add("ffn_up.weight", q8(hidden, ffn));
        add("ffn_down.weight", q8(ffn, hidden));
        add("post_ffw_norm.weight", f32s(hidden, 1.0));
        add("layer_output_scale.weight", f32s(1, 0.5 + l as f32));
    }
    t
}

/// Adds the router, norms and stacked experts of every layer.
fn add_experts(c: &Config, t: &mut Tensors) {
    let (hidden, experts, width) = (
        u64::from(c.hidden_size),
        u64::from(c.num_experts),
        u64::from(c.expert_ffn_size),
    );
    for l in 0..c.num_layers {
        let p = format!("blk.{l}.");
        t.insert(
            format!("{p}ffn_gate_inp.weight"),
            TensorSpec {
                dims: vec![hidden, experts],
                ty: GgmlType::F32,
                data: vec![0; (hidden * experts * 4) as usize],
            },
        );
        t.insert(format!("{p}ffn_gate_inp.scale"), f32s(hidden, 1.0));
        t.insert(format!("{p}pre_ffw_norm_2.weight"), f32s(hidden, 1.0));
        t.insert(format!("{p}post_ffw_norm_1.weight"), f32s(hidden, 1.0));
        t.insert(format!("{p}post_ffw_norm_2.weight"), f32s(hidden, 1.0));
        t.insert(
            format!("{p}ffn_gate_up_exps.weight"),
            TensorSpec {
                dims: vec![hidden, 2 * width, experts],
                ty: GgmlType::Q8_0,
                data: vec![0; (hidden / 32 * 34 * 2 * width * experts) as usize],
            },
        );
        t.insert(
            format!("{p}ffn_down_exps.weight"),
            TensorSpec {
                dims: vec![width, hidden, experts],
                ty: GgmlType::Q5_1,
                data: vec![0; (width / 32 * 24 * hidden * experts) as usize],
            },
        );
        t.insert(format!("{p}ffn_down_exps.scale"), f32s(experts, 1.0));
    }
}

fn bind(metadata: &Metadata, tensors: &Tensors) -> Result<ModelWeights, String> {
    let image = gguf_image(metadata, tensors);
    let shards = [GgufShard::parse(&image).expect("in-memory GGUF")];
    ModelWeights::bind(&shards)
}

#[test]
fn expert_weights() {
    let c = parse(&moe_metadata()).unwrap();
    let mut tensors = target_tensors(&c, 96);
    add_experts(&c, &mut tensors);
    let w = bind(&moe_metadata(), &tensors).expect("valid expert weights");
    let l = &w.layers[1];
    assert!(
        l.gate_up_exps.experts == 8
            && l.gate_up_exps.rows == 64
            && l.down_exps.ty == GgmlType::Q5_1
            // One Q5_1 block of 32 values per row of the 32-wide experts.
            && l.down_exps.size_bytes() == 24 * 64 * 8,
        "stacked expert tensors"
    );
    let rejects = |wrong: &Tensors, meta: &Metadata, what: &str| {
        let error = bind(meta, wrong).expect_err(what);
        assert!(!error.is_empty(), "{what}");
    };
    let mut wrong = tensors.clone();
    wrong.remove("blk.0.ffn_gate_up_exps.weight");
    rejects(&wrong, &moe_metadata(), "separate gate/up experts accepted");
    let mut wrong = tensors.clone();
    wrong.get_mut("blk.0.ffn_down_exps.weight").unwrap().dims[2] = 7;
    rejects(&wrong, &moe_metadata(), "short expert stack accepted");
    let mut wrong = tensors.clone();
    wrong.get_mut("blk.1.ffn_gate_up_exps.weight").unwrap().ty = GgmlType::Q5_1;
    rejects(
        &wrong,
        &moe_metadata(),
        "unsupported expert format accepted",
    );
    rejects(
        &tensors,
        &target_metadata(),
        "experts in a dense model accepted",
    );
}

#[test]
fn weights() {
    let c = parse(&target_metadata()).unwrap();
    let tensors = target_tensors(&c, 96);
    let image = gguf_image(&target_metadata(), &tensors);
    let shards = [GgufShard::parse(&image).unwrap()];
    let w = ModelWeights::bind(&shards).expect("valid weights");
    assert!(
        w.vocab_size == 96 && w.tied_output(),
        "tied vocabulary head"
    );
    assert!(
        w.layers[0].output_scale == 0.5 && w.layers[1].output_scale == 1.5,
        "layer output scales"
    );
    assert!(
        !w.layers[0].attn_v.is_empty() && w.layers[1].attn_v.is_empty(),
        "global layer uses K as V"
    );
    assert_eq!(w.rope_factors.values.len(), 64, "rope factors");

    // The draft reads target KV and embeddings; its own head is separate.
    let mut dt = Tensors::new();
    dt.insert("token_embd.weight".into(), q8(32, 96));
    dt.insert("output_norm.weight".into(), f32s(32, 1.0));
    dt.insert("rope_freqs.weight".into(), f32s(64, 1.0));
    dt.insert("nextn.pre_projection.weight".into(), q8(128, 32));
    dt.insert("nextn.post_projection.weight".into(), q8(32, 64));
    for l in 0..2u64 {
        let p = format!("blk.{l}.");
        let hd = if l == 0 { 64 } else { 128 };
        dt.insert(format!("{p}attn_norm.weight"), f32s(32, 1.0));
        dt.insert(format!("{p}attn_q.weight"), q8(32, 4 * hd));
        dt.insert(format!("{p}attn_q_norm.weight"), f32s(hd, 1.0));
        dt.insert(format!("{p}attn_output.weight"), q8(4 * hd, 32));
        dt.insert(format!("{p}post_attention_norm.weight"), f32s(32, 1.0));
        dt.insert(format!("{p}ffn_norm.weight"), f32s(32, 1.0));
        dt.insert(format!("{p}ffn_gate.weight"), q8(32, 64));
        dt.insert(format!("{p}ffn_up.weight"), q8(32, 64));
        dt.insert(format!("{p}ffn_down.weight"), q8(64, 32));
        dt.insert(format!("{p}post_ffw_norm.weight"), f32s(32, 1.0));
        dt.insert(format!("{p}layer_output_scale.weight"), f32s(1, 0.25));
    }
    let bind_draft = |tensors: &Tensors| {
        let image = gguf_image(&draft_metadata(), tensors);
        let shards = [GgufShard::parse(&image).unwrap()];
        DraftWeights::bind(&shards, &w)
    };
    let d = bind_draft(&dt).expect("valid draft");
    assert!(
        d.config.shared_kv_source(0, &w.config) == 0
            && d.config.shared_kv_source(1, &w.config) == 1,
        "draft KV sources"
    );
    let mut wrong_vocab = dt.clone();
    wrong_vocab.insert("token_embd.weight".into(), q8(32, 95));
    assert!(
        bind_draft(&wrong_vocab).is_err(),
        "draft with another vocabulary accepted"
    );
    let mut with_k = dt.clone();
    with_k.insert("blk.0.attn_k.weight".into(), q8(32, 128));
    assert!(
        bind_draft(&with_k).is_err(),
        "draft layer with its own K accepted"
    );

    let rejects = |wrong: &Tensors, what: &str| {
        let error = bind(&target_metadata(), wrong).expect_err(what);
        assert!(!error.is_empty(), "{what}");
    };
    let mut wrong = tensors.clone();
    wrong.remove("blk.0.attn_v.weight");
    rejects(&wrong, "sliding layer without V accepted");
    let mut wrong = tensors.clone();
    wrong.insert("blk.1.attn_q.weight".into(), q8(64, 256));
    rejects(&wrong, "wrong Q width accepted");
    let mut wrong = tensors.clone();
    wrong.get_mut("blk.0.ffn_up.weight").unwrap().ty = GgmlType::Q4_1;
    rejects(&wrong, "unsupported projection format accepted");
    let mut wrong = tensors.clone();
    wrong.insert("rope_freqs.weight".into(), f32s(64, -1.0));
    rejects(&wrong, "non-positive rope factors accepted");
    let mut wrong = tensors.clone();
    wrong.remove("output_norm.weight");
    rejects(&wrong, "missing output norm accepted");
}
