//! The long-context prefill paths keep the bits:
//! the fused MoE combine without the last layer's unused norm writes the same
//! residual as with it and as the separate epilogue plus combine, and block
//! selection over bounded score scratch (compact rows, ragged chunks) gives
//! the masks of one full-stride launch:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next-kernels --features gpu --test prefill_tail

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{DeviceBuffer, DeviceCopy, F16, IndexSpec, Launch, Stream};
use gufo_qwen38_flash_next_kernels as k;

/// A deterministic value in `[-scale, scale)`.
fn uniform(seed: &mut u32, scale: f32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
}

fn random(seed: &mut u32, count: usize, scale: f32) -> Vec<f32> {
    (0..count).map(|_| uniform(seed, scale)).collect()
}

fn upload(stream: &Stream, host: &[f32]) -> DeviceBuffer<f32> {
    let mut buffer = stream.alloc::<f32>(host.len()).unwrap();
    stream.upload(host, buffer.as_mut_slice()).unwrap();
    buffer
}

fn download<T: DeviceCopy + Default>(stream: &Stream, buffer: &DeviceBuffer<T>) -> Vec<T> {
    let mut host = vec![T::default(); buffer.len()];
    stream.download(buffer.as_slice(), &mut host).unwrap();
    host
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn residual_only_combine_keeps_the_residual() {
    const TOKENS: usize = 37;
    const HIDDEN: usize = 2560;
    const STREAMS: usize = 4;
    const USED: usize = 10;
    const GATE_STRIDE: usize = 513;
    let stream = Stream::new().unwrap();
    let mut seed = 0x13579;
    let hc = STREAMS * HIDDEN;
    let parts = k::inject_parts_vec4(HIDDEN as u32) as usize;
    let residual = random(&mut seed, TOKENS * hc, 1.0);
    let experts = upload(&stream, &random(&mut seed, TOKENS * USED * HIDDEN, 0.5));
    let mut experts_half = stream.alloc::<F16>(TOKENS * USED * HIDDEN).unwrap();
    k::narrow_f16(
        &stream,
        experts.as_slice(),
        experts_half.as_mut_slice(),
        TOKENS * USED * HIDDEN,
    )
    .unwrap();
    let weights = upload(&stream, &random(&mut seed, TOKENS * USED, 1.0));
    let shared = upload(&stream, &random(&mut seed, TOKENS * HIDDEN, 0.5));
    let gate = upload(&stream, &random(&mut seed, TOKENS * GATE_STRIDE, 2.0));
    let inject = upload(&stream, &random(&mut seed, TOKENS * STREAMS * parts, 0.5));
    let gamma: Vec<f32> = (0..hc).map(|_| 1.0 + uniform(&mut seed, 0.5)).collect();
    let gamma = upload(&stream, &gamma);

    let fused = |normalize: bool| {
        let mut res = upload(&stream, &residual);
        let mut xn = stream.alloc::<F16>(TOKENS * hc).unwrap();
        let mut xn_q8 = stream
            .alloc::<u8>(TOKENS.div_ceil(128) * 8 * (hc / 32) * 576)
            .unwrap();
        let launched = k::hc_combine_moe_f16(
            &stream,
            res.as_mut_slice(),
            experts_half.as_slice(),
            weights.as_slice(),
            shared.as_slice(),
            gate.as_slice(),
            GATE_STRIDE as u32,
            USED as u32,
            inject.as_slice(),
            parts as u32,
            normalize.then(|| gamma.as_slice()),
            normalize.then(|| xn.as_mut_slice()),
            normalize.then(|| xn_q8.as_mut_slice()),
            TOKENS as u32,
            HIDDEN as u32,
            STREAMS as u32,
            1e-6,
        )
        .unwrap();
        assert_eq!(launched, Launch::Launched, "normalize {normalize}");
        bits(&download(&stream, &res))
    };
    let with_norm = fused(true);
    let residual_only = fused(false);
    assert_eq!(
        with_norm, residual_only,
        "skipping the unused norm changed the residual"
    );

    let mut block_out = stream.alloc::<f32>(TOKENS * HIDDEN).unwrap();
    k::moe_epilogue_vec4_f16(
        &stream,
        experts_half.as_slice(),
        weights.as_slice(),
        shared.as_slice(),
        gate.as_slice(),
        GATE_STRIDE as u32,
        block_out.as_mut_slice(),
        TOKENS as u32,
        USED as u32,
        HIDDEN as u32,
    )
    .unwrap();
    let mut separate = upload(&stream, &residual);
    let mut unused = stream.alloc::<f32>(TOKENS * hc).unwrap();
    k::hc_combine(
        &stream,
        separate.as_mut_slice(),
        block_out.as_slice(),
        inject.as_slice(),
        parts as u32,
        None,
        unused.as_mut_slice(),
        TOKENS as u32,
        HIDDEN as u32,
        STREAMS as u32,
        1e-6,
        false,
    )
    .unwrap();
    assert_eq!(
        bits(&download(&stream, &separate)),
        residual_only,
        "fusing the residual-only combine changed it"
    );
}

#[test]
fn chunked_selection_over_compact_rows_keeps_the_masks() {
    const HEADS: usize = 4;
    const DIM: usize = 128;
    const RATIO: u32 = 4;
    const BUDGET: u32 = 8;
    const MAX_CONTEXT: u32 = 4096;
    const CHUNK: u32 = 31;
    let stream = Stream::new().unwrap();
    let mut seed = 0x2468;
    let max_blocks_full = (MAX_CONTEXT / RATIO).div_ceil(32) * 32;
    let mask_words = max_blocks_full / 32;
    let blocks_f32 = upload(
        &stream,
        &random(&mut seed, (MAX_CONTEXT / RATIO) as usize * DIM, 1.0),
    );
    let mut blocks = stream
        .alloc::<F16>((MAX_CONTEXT / RATIO) as usize * DIM)
        .unwrap();
    k::narrow_f16(
        &stream,
        blocks_f32.as_slice(),
        blocks.as_mut_slice(),
        (MAX_CONTEXT / RATIO) as usize * DIM,
    )
    .unwrap();
    for (start_pos, n) in [(1000u32, 100u32), (37, 300), (3000, 64)] {
        let q = upload(&stream, &random(&mut seed, n as usize * HEADS * DIM, 1.0));
        let mut position = stream
            .alloc_indices_with(1, IndexSpec::below(MAX_CONTEXT - n + 1))
            .unwrap();
        stream.upload_indices(&[start_pos], &mut position).unwrap();
        let select = |chunk: u32, max_blocks: u32| {
            let mut mask = stream
                .alloc::<u32>(n as usize * mask_words as usize)
                .unwrap();
            let mut scores = stream
                .alloc::<f32>(chunk.min(n) as usize * max_blocks as usize)
                .unwrap();
            let mut first = 0;
            while first < n {
                let count = chunk.min(n - first);
                k::select_blocks(
                    &stream,
                    q.slice(first as usize * HEADS * DIM..n as usize * HEADS * DIM),
                    blocks.as_slice(),
                    mask.slice_mut(first as usize * mask_words as usize..mask.len()),
                    scores.as_mut_slice(),
                    count,
                    position.view(),
                    first,
                    HEADS as u32,
                    DIM as u32,
                    RATIO,
                    BUDGET,
                    mask_words,
                    max_blocks,
                    (start_pos + first + count) / RATIO,
                    MAX_CONTEXT,
                )
                .unwrap();
                first += count;
            }
            download(&stream, &mask)
        };
        let full = select(n, max_blocks_full);
        let compact = (start_pos + n).div_ceil(RATIO).div_ceil(32) * 32;
        assert_eq!(
            select(CHUNK, compact),
            full,
            "start {start_pos} n {n}: chunked selection changed the mask"
        );
        assert!(full.iter().any(|&w| w != 0 && w != u32::MAX));
    }
}
