//! Probes of every launcher wrapper on a gfx1151, at the DeepSeek V4 Flash
//! shapes the graph uses (narrow and wide routes): each call must succeed,
//! and every output gets exactly the extent the wrapper checks followed by
//! a guard region that must stay untouched. Probes catch extents that are
//! too small for what a kernel writes, not for what it reads. Weights are a
//! zero-filled sparse file registered as the target and support models.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::fs::File;
use std::sync::Arc;

use gufo_deepseek_v4_flash_kernels::{
    Backend, BatchMode, CANDIDATES_BYTES, CompressorWeights, DeviceTensor, Input, Output, Rope,
    RoutedExperts, RoutedMoe, Tensor, TensorView, TensorViewMut, Weight,
};
use gufo_hip::MappedFile;

const GUARD: u64 = 16 << 10;
const PATTERN: u8 = 0xa5;
const MODEL_BYTES: u64 = 1280 << 20;

const EMBD: u32 = 4096;
const HC: u32 = 4;
const HEAD: u32 = 64;
const HD: u32 = 512;
const ROT: u32 = 64;
const LORA_Q: u32 = 1024;
const GROUPS: u32 = 8;
const LORA_O: u32 = 1024;
const FF: u32 = 2048;
const VOCAB: u32 = 129280;
const IDX_HEAD: u32 = 64;
const IDX_HD: u32 = 128;
const F32: u64 = 4;
const F16: u64 = 2;

/// An output: `bytes` followed by a guard region.
struct Out {
    base: DeviceTensor,
    bytes: u64,
}

impl Out {
    fn new(b: &Backend, bytes: u64) -> Self {
        let mut base = b.alloc(bytes + GUARD).unwrap();
        b.write(&mut base, 0, &vec![0u8; bytes as usize]).unwrap();
        b.write(&mut base, bytes, &vec![PATTERN; GUARD as usize])
            .unwrap();
        Self { base, bytes }
    }

    fn f32(b: &Backend, count: u64) -> Self {
        Self::new(b, count * F32)
    }

    fn t(&self) -> TensorView<'_> {
        self.base.view(0, self.bytes).unwrap()
    }

    /// The checked extent as the exclusive output of a generated launcher.
    fn t_mut(&mut self) -> TensorViewMut<'_> {
        self.base.view_mut(0, self.bytes).unwrap()
    }

    fn check(&self, b: &Backend, what: &str) {
        b.synchronize().unwrap();
        let mut guard = vec![0u8; GUARD as usize];
        b.read(&self.base, self.bytes, &mut guard).unwrap();
        assert!(
            guard.iter().all(|&g| g == PATTERN),
            "{what} wrote past its checked extent"
        );
    }
}

/// A zeroed input of exactly `bytes`.
fn input(b: &Backend, bytes: u64) -> DeviceTensor {
    let mut t = b.alloc(bytes).unwrap();
    b.write(&mut t, 0, &vec![0u8; bytes as usize]).unwrap();
    t
}

fn input_f32(b: &Backend, count: u64) -> DeviceTensor {
    input(b, count * F32)
}

fn rope(pos0: u32) -> Rope {
    Rope {
        n_rot: ROT,
        pos0,
        n_ctx_orig: 65536,
        freq_base: 160000.0,
        freq_scale: 1.0 / 16.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    }
}

const W: Weight = Weight::target(0);

fn backend() -> (Backend, tempfile::NamedTempFile, tempfile::NamedTempFile) {
    let mut b = Backend::init().unwrap();
    let target = tempfile::NamedTempFile::new().unwrap();
    File::options()
        .write(true)
        .open(target.path())
        .unwrap()
        .set_len(MODEL_BYTES)
        .unwrap();
    let map: Arc<MappedFile> = MappedFile::open(target.path()).unwrap();
    b.set_model(map).unwrap();
    // The sparse file uploads as zeros.
    b.load_target_spans(&[(0, MODEL_BYTES)]).unwrap();
    let support = tempfile::NamedTempFile::new().unwrap();
    let support_bytes: u64 = 64 << 20;
    File::options()
        .write(true)
        .open(support.path())
        .unwrap()
        .set_len(support_bytes)
        .unwrap();
    let map = MappedFile::open(support.path()).unwrap();
    b.set_support(map, support_bytes, &[(0, support_bytes, "probe")])
        .unwrap();
    (b, target, support)
}

fn head_rows(n: u32) -> u64 {
    u64::from(n) * u64::from(HEAD) * u64::from(HD)
}

#[test]
fn every_wrapper_stays_within_its_checked_extents() {
    let (b, _target, _support) = backend();
    attention(&b);
    matmul(&b);
    hc(&b);
    norm(&b);
    indexer(&b);
    compressor(&b);
    moe(&b);
    dspark(&b);
}

fn attention(b: &Backend) {
    let raw_cap = 256u32;
    let raw = input_f32(b, u64::from(raw_cap) * u64::from(HD));
    // Stores.
    let mut ring = Out::f32(b, u64::from(raw_cap) * u64::from(HD));
    let mut kv = Out::f32(b, u64::from(HD));
    b.store_raw_kv(&mut ring.t_mut(), &kv.t(), raw_cap, 300, HD)
        .unwrap();
    b.kv_fp8_store_raw(&mut kv.t_mut(), &mut ring.t_mut(), raw_cap, 7, HD, ROT)
        .unwrap();
    let kv3 = input_f32(b, 3 * u64::from(HD));
    b.store_raw_kv_batch(&mut ring.t_mut(), &kv3, raw_cap, 254, 3, HD)
        .unwrap();
    ring.check(b, "store raw kv");
    kv.check(b, "kv fp8 store");

    for n in [1u32, 3, 128] {
        let q = input_f32(b, head_rows(n));
        let mut heads = Out::f32(b, head_rows(n));
        let comp = input_f32(b, 600 * u64::from(HD));
        if n == 1 {
            let mask = input_f32(b, 600);
            b.attention_decode_heads(
                &mut heads.t_mut(),
                W,
                &q,
                &raw,
                100,
                raw_cap,
                5,
                Some(&comp as &dyn Input),
                600,
                Some(&mask as &dyn Input),
                HEAD,
                HD,
            )
            .unwrap();
            heads.check(b, "attention decode heads");
        }
        b.attention_decode_raw_batch_heads(
            &mut heads.t_mut(),
            W,
            &q,
            &raw,
            n,
            200,
            128,
            raw_cap,
            3,
            128,
            HEAD,
            HD,
        )
        .unwrap();
        heads.check(b, "attention decode raw batch");
        let mask = input_f32(b, u64::from(n) * 50);
        b.attention_decode_mixed_batch_heads(
            &mut heads.t_mut(),
            W,
            &q,
            &raw,
            Some(&comp as &dyn Input),
            Some(&mask as &dyn Input),
            n,
            200,
            128,
            raw_cap,
            3,
            50,
            128,
            4,
            HEAD,
            HD,
        )
        .unwrap();
        heads.check(b, "attention decode mixed batch");
        let topk = input(b, u64::from(n) * 512 * 4);
        b.attention_indexed_mixed_batch_heads(
            &mut heads.t_mut(),
            W,
            &q,
            &raw,
            &comp,
            false,
            &topk,
            n,
            2400,
            128,
            raw_cap,
            3,
            600,
            512,
            128,
            4,
            HEAD,
            HD,
        )
        .unwrap();
        heads.check(b, "attention indexed mixed batch");
        let chunk = input_f32(b, u64::from(n) * u64::from(HD));
        b.attention_prefill_raw_heads(&mut heads.t_mut(), W, &q, &chunk, n, 128, HEAD, HD)
            .unwrap();
        heads.check(b, "attention prefill raw");
        b.attention_prefill_static_mixed_heads(
            &mut heads.t_mut(),
            W,
            &q,
            &chunk,
            Some(&comp as &dyn Input),
            n,
            40,
            128,
            4,
            HEAD,
            HD,
        )
        .unwrap();
        heads.check(b, "attention prefill static mixed");
        b.attention_noncausal_raw_batch_heads(
            &mut heads.t_mut(),
            W,
            &q,
            &raw,
            n,
            64,
            raw_cap,
            10,
            HEAD,
            HD,
        )
        .unwrap();
        heads.check(b, "attention noncausal");

        let group_dim = u64::from(HD) * u64::from(HEAD / GROUPS);
        let mut out = Out::f32(b, u64::from(n) * u64::from(EMBD));
        let mut low = Out::f32(b, u64::from(n) * u64::from(GROUPS) * u64::from(LORA_O));
        let heads_in = input_f32(b, head_rows(n));
        b.attention_output_q8_batch(
            &mut out.t_mut(),
            &mut low.t_mut(),
            W,
            W,
            group_dim,
            u64::from(LORA_O),
            GROUPS,
            u64::from(EMBD),
            &heads_in,
            n,
        )
        .unwrap();
        out.check(b, "attention output q8 batch");
        low.check(b, "attention output q8 batch low");
        if n == 1 {
            b.attention_output_low_q8(
                &mut low.t_mut(),
                W,
                group_dim,
                u64::from(LORA_O),
                GROUPS,
                &heads_in,
            )
            .unwrap();
            low.check(b, "attention output low");
        }
        if n == 128 {
            let applied = b.attention_output_q8_batch_inv_rope(
                &mut out.t_mut(),
                &mut low.t_mut(),
                W,
                W,
                group_dim,
                u64::from(LORA_O),
                GROUPS,
                u64::from(EMBD),
                &heads_in,
                n,
                HD,
                rope(0),
            );
            assert!(applied, "the inverse rotation folds in at 128 rows");
            out.check(b, "attention output inv rope");
        }
    }
    let n = 4u32;
    let mut out = Out::f32(b, u64::from(n) * u64::from(EMBD));
    let mut low = Out::f32(b, u64::from(n) * u64::from(GROUPS) * u64::from(LORA_O));
    let heads_in = input_f32(b, head_rows(n));
    let group_dim = u64::from(HD) * u64::from(HEAD / GROUPS);
    b.attention_output_q8_exact_batch(
        &mut out.t_mut(),
        &mut low.t_mut(),
        W,
        W,
        group_dim,
        u64::from(LORA_O),
        GROUPS,
        u64::from(EMBD),
        &heads_in,
        n,
    )
    .unwrap();
    out.check(b, "attention output exact batch");
    // Verification blocks split into per-row decode.
    b.set_batch_mode(BatchMode::Verifier);
    let q = input_f32(b, head_rows(n));
    let mut heads = Out::f32(b, head_rows(n));
    let comp = input_f32(b, 600 * u64::from(HD));
    let topk = input(b, u64::from(n) * 512 * 4);
    b.attention_indexed_mixed_batch_heads(
        &mut heads.t_mut(),
        W,
        &q,
        &raw,
        &comp,
        false,
        &topk,
        n,
        2400,
        128,
        raw_cap,
        3,
        600,
        512,
        128,
        4,
        HEAD,
        HD,
    )
    .unwrap();
    heads.check(b, "verifier indexed attention");
    assert!(
        b.attention_indexed_mixed_batch_heads(
            &mut heads.t_mut(),
            W,
            &q,
            &raw,
            &comp,
            false,
            &topk,
            n,
            2400,
            2,
            raw_cap,
            3,
            600,
            512,
            128,
            4,
            HEAD,
            HD
        )
        .is_err(),
        "a verification row whose raw count underflows is refused"
    );
    b.set_batch_mode(BatchMode::Off);
}

fn matmul(b: &Backend) {
    for n in [1u32, 3, 128] {
        let x = input_f32(b, u64::from(n) * u64::from(EMBD));
        let mut out = Out::f32(b, u64::from(n) * u64::from(LORA_Q));
        b.matmul_q8_0(
            &mut out.t_mut(),
            W,
            u64::from(EMBD),
            u64::from(LORA_Q),
            &x,
            u64::from(n),
        )
        .unwrap();
        out.check(b, "matmul q8_0");
        b.matmul_f16(
            &mut out.t_mut(),
            W,
            u64::from(EMBD),
            u64::from(LORA_Q),
            &x,
            u64::from(n),
        )
        .unwrap();
        out.check(b, "matmul f16");
        let mut out_b = Out::f32(b, u64::from(n) * u64::from(LORA_Q));
        b.matmul_f16_pair(
            &mut out.t_mut(),
            &mut out_b.t_mut(),
            W,
            W,
            u64::from(EMBD),
            u64::from(LORA_Q),
            &x,
            u64::from(n),
        )
        .unwrap();
        out.check(b, "matmul f16 pair a");
        out_b.check(b, "matmul f16 pair b");
        let mut gate = Out::f32(b, u64::from(n) * u64::from(FF));
        let mut up = Out::f32(b, u64::from(n) * u64::from(FF));
        let mut mid = Out::f32(b, u64::from(n) * u64::from(FF));
        b.shared_gate_up_swiglu_q8_0(
            &mut gate.t_mut(),
            &mut up.t_mut(),
            &mut mid.t_mut(),
            W,
            W,
            u64::from(EMBD),
            u64::from(FF),
            &x,
            10.0,
            n,
        )
        .unwrap();
        for (o, what) in [
            (&gate, "shared gate"),
            (&up, "shared up"),
            (&mid, "shared mid"),
        ] {
            o.check(b, what);
        }
        if n >= 2 {
            let xh = input(b, u64::from(n) * u64::from(EMBD) * F16);
            b.matmul_f16_f16_input(
                &mut out.t_mut(),
                W,
                u64::from(EMBD),
                u64::from(LORA_Q),
                &xh,
                u64::from(n),
            )
            .unwrap();
            out.check(b, "matmul f16 f16 input");
        }
    }
    let n = 3u32;
    let x = input_f32(b, u64::from(n) * u64::from(EMBD));
    let mut out = Out::f32(b, u64::from(n) * 512);
    let mut out_b = Out::f32(b, u64::from(n) * 512);
    b.matmul_f16_pair_narrow(
        &mut out.t_mut(),
        &mut out_b.t_mut(),
        W,
        W,
        u64::from(EMBD),
        512,
        &x,
        u64::from(n),
    )
    .unwrap();
    out.check(b, "matmul f16 pair narrow");
    let x2 = input_f32(b, 2 * u64::from(EMBD));
    let mut out2 = Out::f32(b, 2 * 4);
    b.matmul_f32(&mut out2.t_mut(), W, u64::from(EMBD), 4, &x2, 2)
        .unwrap();
    out2.check(b, "matmul f32");
    let x4 = input_f32(b, 4 * u64::from(EMBD));
    let mut out4 = Out::f32(b, 4 * u64::from(LORA_Q));
    b.matmul_q8_0_group_pairs(
        &mut out4.t_mut(),
        W,
        u64::from(EMBD),
        u64::from(LORA_Q),
        &x4,
        4,
        &[0, 2, 4],
    )
    .unwrap();
    out4.check(b, "matmul q8_0 group pairs");
    for offsets in [&[0, 1, 2, 3, 4][..], &[0, 1, 4]] {
        b.matmul_f16_group_pairs(
            &mut out4.t_mut(),
            W,
            u64::from(EMBD),
            u64::from(LORA_Q),
            &x4,
            4,
            offsets,
        )
        .unwrap();
        out4.check(b, "matmul f16 group pairs");
    }
    assert!(
        b.matmul_q8_0_group_pairs(
            &mut out4.t_mut(),
            W,
            u64::from(EMBD),
            u64::from(LORA_Q),
            &x4,
            4,
            &[0, 2]
        )
        .is_err()
    );
    // Single-token projections expanded into the streams.
    let hc = u64::from(HC) * u64::from(EMBD);
    let mut out_hc = Out::f32(b, hc);
    let mut block = Out::f32(b, u64::from(EMBD));
    let residual = input_f32(b, hc);
    let split = input_f32(b, 24);
    let x1 = input_f32(b, u64::from(FF));
    b.matmul_q8_0_hc_expand(
        &mut out_hc.t_mut(),
        &mut block.t_mut(),
        W,
        u64::from(FF),
        u64::from(EMBD),
        &x1,
        &residual,
        &split,
        EMBD,
        HC,
    )
    .unwrap();
    out_hc.check(b, "matmul q8_0 hc expand");
    block.check(b, "matmul q8_0 hc expand block");
    let routed = input_f32(b, u64::from(EMBD));
    b.shared_down_hc_expand_q8_0(
        &mut out_hc.t_mut(),
        &mut block.t_mut(),
        W,
        u64::from(FF),
        u64::from(EMBD),
        &x1,
        &routed,
        &residual,
        &split,
        EMBD,
        HC,
    )
    .unwrap();
    out_hc.check(b, "shared down hc expand");
}

fn hc(b: &Backend) {
    let embd = u64::from(EMBD);
    for rows in [1u32, 4] {
        let r = u64::from(rows);
        let residual = input_f32(b, r * 4 * embd);
        let mut out = Out::f32(b, r * embd);
        let weights = input_f32(b, r * 4);
        b.hc_weighted_sum(&mut out.t_mut(), &residual, &weights, EMBD, HC)
            .unwrap();
        out.check(b, "hc weighted sum");
        let mix = input_f32(b, r * 24);
        let mut split = Out::f32(b, r * 24);
        b.hc_split_weighted_sum(
            &mut out.t_mut(),
            &mut split.t_mut(),
            &mix,
            &residual,
            W,
            W,
            EMBD,
            HC,
            20,
            1e-6,
        )
        .unwrap();
        out.check(b, "hc split weighted sum");
        split.check(b, "hc split weighted sum split");
        let mut norm = Out::f32(b, r * embd);
        b.hc_split_weighted_sum_norm(
            &mut out.t_mut(),
            &mut norm.t_mut(),
            &mut split.t_mut(),
            &mix,
            &residual,
            W,
            W,
            W,
            EMBD,
            HC,
            20,
            1e-6,
            1e-6,
        )
        .unwrap();
        norm.check(b, "hc split weighted sum norm");
        let mut mix_out = Out::f32(b, r * 24);
        let fused = b
            .hc_norm_mix_split_weighted_sum(
                &mut out.t_mut(),
                &mut mix_out.t_mut(),
                &mut split.t_mut(),
                &residual,
                W,
                W,
                W,
                EMBD,
                HC,
                rows,
                20,
                1e-6,
                1e-6,
            )
            .unwrap();
        assert!(fused, "the fused pre-block chain runs at 4x4096");
        out.check(b, "hc norm mix");
        mix_out.check(b, "hc norm mix mix");
        let pre = input_f32(b, r * 4);
        let mut hw = Out::f32(b, r * 4);
        b.output_hc_weights(&mut hw.t_mut(), &pre, W, W, HC, 1e-6)
            .unwrap();
        hw.check(b, "output hc weights");
        let mut out_hc = Out::f32(b, r * 4 * embd);
        let block = input_f32(b, r * embd);
        b.hc_expand_split(&mut out_hc.t_mut(), &block, &residual, &mix, EMBD, HC)
            .unwrap();
        out_hc.check(b, "hc expand split");
        b.hc_expand_add_split(
            &mut out_hc.t_mut(),
            &block,
            &block,
            &residual,
            &mix,
            EMBD,
            HC,
        )
        .unwrap();
        out_hc.check(b, "hc expand add split");
        let down_h = input(b, r * 6 * embd * F16);
        b.hc_expand_add_split_moesum(
            &mut out_hc.t_mut(),
            &down_h,
            &block,
            &residual,
            &mix,
            EMBD,
            HC,
            6,
            rows,
        )
        .unwrap();
        out_hc.check(b, "hc expand add split moesum");
        let logits = input_f32(b, r * u64::from(VOCAB));
        let mut index = Out::new(b, r * 4);
        b.spec_row_argmax(&mut index.t_mut(), &logits, VOCAB, rows)
            .unwrap();
        index.check(b, "spec row argmax");
        let stride = 3 * EMBD;
        let mut features = Out::f32(b, r * u64::from(stride));
        b.dspark_capture_features(
            &mut features.t_mut(),
            &residual,
            stride,
            2 * EMBD,
            EMBD,
            HC,
            rows,
        )
        .unwrap();
        features.check(b, "dspark capture features");
        assert!(
            b.dspark_capture_features(
                &mut features.t_mut(),
                &residual,
                16,
                u32::MAX - 15,
                32,
                HC,
                rows
            )
            .is_err(),
            "a slot past the stride is refused"
        );
    }
}

fn norm(b: &Backend) {
    let embd = u64::from(EMBD);
    let mut out_hc = Out::f32(b, 4 * embd);
    b.embed_token_hc(&mut out_hc.t_mut(), W, VOCAB, VOCAB - 1, EMBD, HC)
        .unwrap();
    out_hc.check(b, "embed token hc");
    let tokens = input(b, 3 * 4);
    let mut out3 = Out::f32(b, 3 * 4 * embd);
    b.embed_tokens_hc(&mut out3.t_mut(), &tokens, W, VOCAB, 3, EMBD, HC)
        .unwrap();
    out3.check(b, "embed tokens hc");
    for rows in [1u32, 3] {
        let r = u64::from(rows);
        let x = input_f32(b, r * 16384);
        let mut out = Out::f32(b, r * 16384);
        b.rms_norm_plain_rows(&mut out.t_mut(), &x, 16384, rows, 1e-6)
            .unwrap();
        out.check(b, "rms norm plain rows");
        b.rms_norm_weight_rows(&mut out.t_mut(), &x, W, 16384, rows, 1e-6)
            .unwrap();
        out.check(b, "rms norm weight rows");
        b.rms_norm_plain_rows_in_place(&mut out.t_mut(), 16384, rows, 1e-6)
            .unwrap();
        b.rms_norm_weight_rows_in_place(&mut out.t_mut(), W, 16384, rows, 1e-6)
            .unwrap();
        out.check(b, "rms norm in place");
        let q = input_f32(b, r * u64::from(LORA_Q));
        let mut q_out = Out::f32(b, r * u64::from(LORA_Q));
        let kv = input_f32(b, r * u64::from(HD));
        let mut kv_out = Out::f32(b, r * u64::from(HD));
        b.qkv_rms_norm_rows(
            &mut q_out.t_mut(),
            &q,
            W,
            LORA_Q,
            &mut kv_out.t_mut(),
            &kv,
            W,
            HD,
            rows,
            1e-6,
        )
        .unwrap();
        q_out.check(b, "qkv norm q");
        kv_out.check(b, "qkv norm kv");
    }
    let x = input_f32(b, embd);
    let mut out = Out::f32(b, embd);
    b.rms_norm_plain(&mut out.t_mut(), &x, EMBD, 1e-6).unwrap();
    b.rms_norm_weight(&mut out.t_mut(), &x, W, EMBD, 1e-6)
        .unwrap();
    out.check(b, "rms norm");
    for n in [1u32, 128] {
        let mut heads = Out::f32(b, head_rows(n));
        b.head_rms_norm(&mut heads.t_mut(), n, HEAD, HD, 1e-6)
            .unwrap();
        b.rope_tail(&mut heads.t_mut(), n, HEAD, HD, false, rope(7))
            .unwrap();
        let fused = b
            .head_rms_norm_rope_tail(&mut heads.t_mut(), n, HEAD, HD, false, rope(7), 1e-6)
            .unwrap();
        assert_eq!(
            fused,
            n >= 96,
            "the fused norm and rotation start at the wide width"
        );
        heads.check(b, "head norm and rope");
        let mut kv = Out::f32(b, u64::from(n) * u64::from(HD));
        b.fp8_kv_quantize(&mut kv.t_mut(), n, HD, ROT).unwrap();
        let mut mirror = Out::new(b, u64::from(n) * u64::from(HD) * F16);
        b.fp8_kv_quantize_mirror_f16(&mut kv.t_mut(), &mut mirror.t_mut(), n, HD, ROT)
            .unwrap();
        kv.check(b, "fp8 kv quantize");
        mirror.check(b, "fp8 kv mirror");
    }
    let gate = input_f32(b, 2048);
    let mut swiglu = Out::f32(b, 2048);
    b.swiglu(&mut swiglu.t_mut(), &gate, &gate, 2048, 10.0, 1.0)
        .unwrap();
    swiglu.check(b, "swiglu");
    let mut half = Out::new(b, 2048 * F16);
    b.convert_f32_to_f16(&mut half.t_mut(), &gate, 2048)
        .unwrap();
    half.check(b, "convert");
    assert!(b.publish_f16_input(&gate, 2048) || !b.publish_f16_input(&gate, 2048));
    b.clear_f16_input();
}

fn indexer(b: &Backend) {
    let hd = u64::from(IDX_HD);
    for (n, n_comp) in [(1u32, 600u32), (3, 600), (128, 2000)] {
        let t = u64::from(n);
        let q = input_f32(b, t * u64::from(IDX_HEAD) * hd);
        let weights = input_f32(b, t * u64::from(IDX_HEAD));
        let comp = input_f32(b, u64::from(n_comp) * hd);
        let mut scores = Out::f32(b, t * u64::from(n_comp));
        if n == 1 {
            b.indexer_score_one(
                &mut scores.t_mut(),
                &q,
                &weights,
                &comp,
                n_comp,
                IDX_HEAD,
                IDX_HD,
                0.1,
            )
            .unwrap();
            scores.check(b, "indexer score one");
        }
        b.indexer_scores_decode_batch(
            &mut scores.t_mut(),
            &q,
            &weights,
            &comp,
            n_comp,
            n,
            4 * n_comp,
            IDX_HEAD,
            IDX_HD,
            4,
            0.1,
            None::<&mut TensorViewMut<'_>>,
        )
        .unwrap();
        scores.check(b, "indexer scores decode batch");
        let padded = u64::from(n_comp).div_ceil(128) * 128 * 128;
        let queries = t.div_ceil(16) * 16 * 64 * 128;
        let mut scratch = Out::new(b, (padded + queries) * F16);
        b.indexer_scores_prefill(
            &mut scores.t_mut(),
            &q,
            &weights,
            &comp,
            n_comp,
            n,
            IDX_HEAD,
            IDX_HD,
            4,
            0.1,
            Some(&mut scratch.t_mut()),
        )
        .unwrap();
        scores.check(b, "indexer scores prefill");
        scratch.check(b, "indexer score scratch");
        let mut selected = Out::new(b, t * 512 * 4);
        b.indexer_topk(&mut selected.t_mut(), &scores.t(), n_comp, n, 512)
            .unwrap();
        selected.check(b, "indexer topk");
        let mut keys = Out::f32(b, t * hd);
        b.indexer_qat(&mut keys.t_mut(), n, IDX_HD).unwrap();
        keys.check(b, "indexer qat");
    }
}

fn compressor(b: &Backend) {
    let hd = u64::from(HD);
    let w = CompressorWeights { ape: W, norm: W };
    for ratio in [4u32, 128] {
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * hd;
        let rows = coff * u64::from(ratio);
        let mut state_kv = Out::f32(b, rows * width);
        let mut state_score = Out::f32(b, rows * width);
        let kv = input_f32(b, width);
        let mut comp = Out::f32(b, 3 * hd);
        b.compressor_update(
            &kv,
            &kv,
            &mut state_kv.t_mut(),
            &mut state_score.t_mut(),
            &mut comp.t_mut(),
            w,
            HD,
            ratio,
            2 * ratio - 1,
            2,
            rope(0),
            1e-6,
            false,
        )
        .unwrap();
        state_kv.check(b, "compressor update state");
        comp.check(b, "compressor update row");
        let n = 2 * ratio + 3;
        let kv_n = input_f32(b, u64::from(n) * width);
        let mut comp_n = Out::f32(b, 2 * hd);
        let mut mirror = Out::new(b, 2 * hd * F16);
        b.compressor_prefill(
            &mut comp_n.t_mut(),
            &mut state_kv.t_mut(),
            &mut state_score.t_mut(),
            &kv_n,
            &kv_n,
            w,
            HD,
            ratio,
            0,
            n,
            rope(0),
            true,
            1e-6,
            Some(&mut mirror.t_mut() as &mut dyn Output),
        )
        .unwrap();
        comp_n.check(b, "compressor prefill rows");
        mirror.check(b, "compressor prefill mirror");
        state_score.check(b, "compressor prefill state");
    }
    let width = 2 * hd;
    let mut state_kv = Out::f32(b, 8 * width);
    let mut state_score = Out::f32(b, 8 * width);
    let kv = input_f32(b, 16 * width);
    let mut comp = Out::f32(b, 4 * hd);
    b.compressor_prefill_ratio4_replay(
        &mut comp.t_mut(),
        &mut state_kv.t_mut(),
        &mut state_score.t_mut(),
        &kv,
        &kv,
        w,
        HD,
        8,
        16,
        rope(8),
        true,
        1e-6,
        None,
    )
    .unwrap();
    comp.check(b, "compressor replay");
    let tail = input_f32(b, 4 * width);
    b.compressor_prefill_state_ratio4(
        &mut state_kv.t_mut(),
        &mut state_score.t_mut(),
        &tail,
        &tail,
        W,
        HD,
        20,
    )
    .unwrap();
    state_kv.check(b, "compressor state ratio4");
    state_score.check(b, "compressor state ratio4 score");
}

fn moe(b: &Backend) {
    let embd = u64::from(EMBD);
    let table = b.hash_table(0, VOCAB).unwrap();
    let experts = RoutedExperts {
        gate: W,
        up: W,
        down: W,
        gate_type: 16,
        down_type: 10,
        in_dim: EMBD,
        mid_dim: FF,
        out_dim: EMBD,
    };
    for n in [1u32, 3, 128] {
        let t = u64::from(n);
        let mut selection = b.alloc_expert_selection(n).unwrap();
        let mut probs = Out::f32(b, t * 256);
        let logits = input_f32(b, t * 256);
        let tokens = input(b, t * 4);
        if n == 1 {
            b.router_select(
                &mut selection,
                &mut probs.t_mut(),
                &logits,
                Some(W),
                None,
                5,
            )
            .unwrap();
            b.router_select(
                &mut selection,
                &mut probs.t_mut(),
                &logits,
                None,
                Some(&table),
                VOCAB - 1,
            )
            .unwrap();
        }
        b.router_select_batch(
            &mut selection,
            0,
            &mut probs.t_mut(),
            &logits,
            &tokens,
            0,
            Some(W),
            None,
            n,
        )
        .unwrap();
        b.router_select_batch(
            &mut selection,
            0,
            &mut probs.t_mut(),
            &logits,
            &tokens,
            0,
            None,
            Some(&table),
            n,
        )
        .unwrap();
        probs.check(b, "router probs");
        let x = input_f32(b, t * embd);
        let pairs = t * 6;
        let mut out = Out::f32(b, t * embd);
        let mut gate = Out::f32(b, pairs * u64::from(FF));
        let mut up = Out::f32(b, pairs * u64::from(FF));
        let mut mid = Out::f32(b, pairs * u64::from(FF));
        let mut down = Out::f32(b, pairs * embd);
        {
            let (mut out_t, mut gate_t, mut up_t, mut mid_t, mut down_t) = (
                out.t_mut(),
                gate.t_mut(),
                up.t_mut(),
                mid.t_mut(),
                down.t_mut(),
            );
            let scratch = RoutedMoe {
                out: &mut out_t,
                gate: &mut gate_t,
                up: &mut up_t,
                mid: &mut mid_t,
                down: &mut down_t,
            };
            b.set_routed_defer_sum(n >= 128);
            if n == 1 {
                b.routed_moe_one(scratch, &experts, &mut selection, 10.0, &x)
                    .unwrap();
            } else {
                b.routed_moe_batch(scratch, &experts, &mut selection, 10.0, &x, n)
                    .unwrap();
            }
        }
        for (o, what) in [
            (&out, "moe out"),
            (&gate, "moe gate"),
            (&up, "moe up"),
            (&mid, "moe mid"),
            (&down, "moe down"),
        ] {
            o.check(b, what);
        }
        b.set_routed_defer_sum(false);
    }
}

fn dspark(b: &Backend) {
    let s = Weight::support(0);
    let rank = 256u32;
    let requests = [5i32, 7, 9];
    let r = requests.len() as u64;
    let logits_stride = 2 * u64::from(VOCAB);
    let logits = input_f32(b, (r - 1) * logits_stride + u64::from(VOCAB));
    let hidden = input_f32(b, (r - 1) * 2 * u64::from(EMBD) + u64::from(EMBD));
    let mut out = Out::new(b, r * CANDIDATES_BYTES);
    let blocks = u64::from(VOCAB).div_ceil(256);
    let mut scratch = Out::new(b, r * blocks * 8 * 8);
    let mut previous = Out::new(b, 8 * 4);
    b.dspark_candidates(
        &mut out.t_mut(),
        &mut scratch.t_mut(),
        &logits,
        logits_stride,
        &hidden,
        2 * u64::from(EMBD),
        &requests,
        &mut previous.t_mut(),
        s,
        s,
        s,
        VOCAB,
        rank,
        EMBD,
    )
    .unwrap();
    out.check(b, "dspark candidates");
    scratch.check(b, "dspark candidate scratch");
    assert!(
        b.dspark_candidates(
            &mut out.t_mut(),
            &mut scratch.t_mut(),
            &logits,
            logits_stride,
            &hidden,
            2 * u64::from(EMBD),
            &[VOCAB as i32],
            &mut previous.t_mut(),
            s,
            s,
            s,
            VOCAB,
            rank,
            EMBD
        )
        .is_err(),
        "previous tokens outside the vocabulary are refused"
    );
    let mut index = Out::new(b, 8 * 4);
    let mut key = Out::new(b, 8 * 8);
    let row = input_f32(b, u64::from(VOCAB));
    b.dspark_markov_argmax(
        &mut index.t_mut(),
        &mut key.t_mut(),
        &row,
        s,
        s,
        VOCAB,
        rank,
        VOCAB - 1,
    )
    .unwrap();
    b.dspark_markov_argmax_batch(
        &mut index.t_mut(),
        &mut key.t_mut(),
        &logits,
        &[1, 2],
        &mut previous.t_mut(),
        s,
        s,
        VOCAB,
        rank,
        logits_stride,
    )
    .unwrap();
    index.check(b, "dspark markov index");
    key.check(b, "dspark markov key");
}

#[allow(dead_code)]
fn unused(_: &Tensor) {}
