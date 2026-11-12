//! The FFN mixer's inject partials do not depend on the pass route
//! (`CheckFusedInjectPartials` of the C++ `hc_mix_ops_test.cpp`): a wide
//! prefill pass gets them from the attention combine
//! (`hc_combine_f16_deferred` with `inject_w`), a short pass over the same
//! rows computes them in the mixer epilogue (`hc_mix_epilogue_f16_inject`,
//! and `hc_mix_f16_gemm_inject` from 96 rows). A 1-ulp difference changes the
//! stream weights of the next combine and with them the greedy tokens:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next-kernels --features gpu --test hc_inject

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{DeviceBuffer, F16, GgmlType, Launch, QuantSlice, Stream};
use gufo_qwen38_flash_next_kernels as k;

const HIDDEN: usize = 2560;
const STREAMS: usize = 4;
const RANK: usize = 320;
const EPS: f32 = 1e-6;

/// A deterministic value in `[-scale, scale)`.
fn uniform(seed: &mut u32, scale: f32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
}

fn values(
    stream: &Stream,
    count: usize,
    mut seed: u32,
    scale: f32,
    offset: f32,
) -> DeviceBuffer<f32> {
    let host: Vec<f32> = (0..count)
        .map(|_| offset + uniform(&mut seed, scale))
        .collect();
    let mut buffer = stream.alloc::<f32>(count).unwrap();
    stream.upload(&host, buffer.as_mut_slice()).unwrap();
    buffer
}

fn bits(stream: &Stream, buffer: &DeviceBuffer<f32>) -> Vec<u32> {
    let mut host = vec![0f32; buffer.len()];
    stream.download(buffer.as_slice(), &mut host).unwrap();
    host.iter().map(|v| v.to_bits()).collect()
}

fn check_fused_inject_partials(stream: &Stream, tokens: usize) {
    let dim = STREAMS * HIDDEN;
    let parts = k::inject_parts_vec4(HIDDEN as u32) as usize;
    let rows = tokens * dim;
    let inject_count = tokens * STREAMS * parts;
    let mut res = values(stream, rows, 0x51CE_0001 + tokens as u32, 0.8, 0.0);
    let block = values(stream, tokens * HIDDEN, 0x51CE_0002, 0.3, 0.0);
    // The inject logits the combine reads; it leaves the fused partials in
    // their place.
    let mut fused = values(stream, inject_count, 0x51CE_0003, 2.0, 0.0);
    let gamma = values(stream, dim, 0x51CE_0004, 0.5, 1.0);
    let inject_w = values(stream, STREAMS * dim, 0x51CE_0005, 0.05, 0.0);
    let gate = values(stream, rows, 0x51CE_0006, 2.0, 0.0);
    let mut xn = stream.alloc::<F16>(rows).unwrap();
    let mut defer = stream.alloc::<f32>(tokens * STREAMS).unwrap();
    let mut mixed = stream.alloc::<f32>(tokens * HIDDEN).unwrap();
    let mut epilogue = stream.alloc::<f32>(inject_count).unwrap();
    let deferring = k::hc_combine_f16_deferred(
        stream,
        res.as_mut_slice(),
        block.as_slice(),
        fused.as_mut_slice(),
        parts as u32,
        gamma.as_slice(),
        xn.as_mut_slice(),
        None,
        tokens as u32,
        HIDDEN as u32,
        STREAMS as u32,
        EPS,
        defer.as_mut_slice(),
        Some(inject_w.as_slice()),
        parts as u32,
    )
    .unwrap();
    assert_eq!(deferring, Launch::Launched, "deferring HC combine rejected");
    k::hc_mix_epilogue_f16_inject(
        stream,
        xn.as_slice(),
        gate.as_slice(),
        inject_w.as_slice(),
        mixed.as_mut_slice(),
        None,
        None,
        epilogue.as_mut_slice(),
        tokens as u32,
        HIDDEN as u32,
    )
    .unwrap();
    let fused = bits(stream, &fused);
    let differ = fused
        .iter()
        .zip(bits(stream, &epilogue))
        .filter(|(a, b)| **a != *b)
        .count();
    assert_eq!(
        differ, 0,
        "{tokens} tokens: {differ} of {inject_count} inject partials differ between the fused combine and the mixer epilogue"
    );
    if tokens >= 96 {
        // The fused projection's partials over the same norm.
        let up_bytes = dim * (RANK / 32) * 34;
        let mut up = stream.alloc::<u8>(up_bytes).unwrap();
        stream.fill_bytes(up.as_mut_slice(), 0).unwrap();
        let mut low = stream.alloc::<F16>(tokens * RANK).unwrap();
        stream.fill_bytes(low.as_mut_slice(), 0).unwrap();
        let mut gemm = stream.alloc::<f32>(inject_count).unwrap();
        let launched = k::hc_mix_f16_gemm_inject(
            stream,
            QuantSlice::new(up.as_slice(), GgmlType::Q8_0),
            low.as_slice(),
            xn.as_slice(),
            inject_w.as_slice(),
            Some(mixed.as_mut_slice()),
            None,
            None,
            gemm.as_mut_slice(),
            tokens as u32,
            HIDDEN as u32,
            RANK as u32,
        )
        .unwrap();
        assert_eq!(launched, Launch::Launched, "HC fused projection rejected");
        assert!(
            bits(stream, &gemm) == fused,
            "{tokens} tokens: the fused projection's inject partials differ"
        );
    }
}

#[test]
fn inject_partials_do_not_depend_on_the_pass_route() {
    let stream = Stream::new().unwrap();
    for tokens in [7, 96, 1619] {
        check_fused_inject_partials(&stream, tokens);
    }
}
