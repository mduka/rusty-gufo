//! `mmq_q8_moe_gated_vec` over the Q8_1 rows of `mmq_quantize_q8_1` equals
//! `mmq_moe_gated_vec`, which quantizes its input itself, bit for bit, for
//! every weight format and token count of the vector route:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next-kernels --features gpu --test moe_gated

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{GgmlType, IndexSpec, Launch, QuantSlice, Stream};
use gufo_qwen38_flash_next_kernels as k;

const M: usize = 640;
const K: usize = 2560;
const EXPERTS: usize = 16;
const USED: usize = 8;
/// The weight tail the launch contract requires.
const TAIL: usize = 4096;
const SCRATCH: usize = 4 << 20;
const GUARD: u32 = 0xA5A5_A5A5;

/// A deterministic value in `[-scale, scale)`.
fn uniform(seed: &mut u32, scale: f32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
}

/// Random blocks of `format` with finite F16 scales (`scales` bytes at the
/// start of each block), followed by the zero tail.
fn weights(format: GgmlType, mut seed: u32) -> Vec<u8> {
    let (block_bytes, block_values, scales) = match format {
        GgmlType::Q4K => (144, 256, 2),
        GgmlType::Q5K => (176, 256, 2),
        GgmlType::Q8_0 => (34, 32, 1),
        _ => unreachable!(),
    };
    let body = EXPERTS * M * K / block_values * block_bytes;
    let mut bytes = vec![0u8; body + TAIL];
    for block in bytes[..body].chunks_mut(block_bytes) {
        for byte in block.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (seed >> 24) as u8;
        }
        for scale in 0..scales {
            // 0.0078 to 0.0156: F16 exponent 8, a random mantissa.
            let bits = (8u16 << 10) | (u16::from(block[2 * scale]) << 2);
            block[2 * scale..2 * scale + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    bytes
}

fn check(format: GgmlType, tokens: &[usize]) {
    let stream = Stream::new().unwrap();
    let host_w = weights(format, 7);
    let mut gate = stream.alloc::<u8>(host_w.len()).unwrap();
    stream.upload(&host_w, gate.as_mut_slice()).unwrap();
    let host_w = weights(format, 8);
    let mut up = stream.alloc::<u8>(host_w.len()).unwrap();
    stream.upload(&host_w, up.as_mut_slice()).unwrap();
    let max = *tokens.iter().max().unwrap();
    let mut seed = 29;
    let x: Vec<f32> = (0..max * K).map(|_| uniform(&mut seed, 3.0)).collect();
    let mut dx = stream.alloc::<f32>(x.len()).unwrap();
    stream.upload(&x, dx.as_mut_slice()).unwrap();
    // Distinct experts per token, repeated across tokens so groups form.
    let host_ids: Vec<u32> = (0..max * USED)
        .map(|i| ((i / USED) * 3 + (i % USED) * 5) as u32 % EXPERTS as u32)
        .collect();
    let mut ids = stream
        .alloc_indices_with(host_ids.len(), IndexSpec::below(EXPERTS as u32).signed())
        .unwrap();
    stream.upload_indices(&host_ids, &mut ids).unwrap();
    let mut scratch = stream.alloc::<u8>(SCRATCH).unwrap();
    let mut q = stream.alloc::<u8>(k::q8_1_bytes(max, K)).unwrap();
    for &n in tokens {
        let count = n * USED * M;
        let mut library = stream.alloc::<f32>(count).unwrap();
        let launched = k::mmq_moe_gated_vec(
            &stream,
            QuantSlice::new(gate.as_slice(), format),
            QuantSlice::new(up.as_slice(), format),
            dx.slice(0..n * K),
            ids.slice(0..n * USED),
            library.as_mut_slice(),
            M as u32,
            K as u32,
            n as u32,
            EXPERTS as u32,
            USED as u32,
            scratch.as_mut_slice(),
            SCRATCH,
        )
        .unwrap();
        assert_eq!(launched, Launch::Launched, "{format:?} n {n}");
        let bytes = k::q8_1_bytes(n, K);
        let launched = k::mmq_quantize_q8_1(
            &stream,
            dx.slice(0..n * K),
            q.slice_mut(0..bytes),
            n as u32,
            K as u32,
        )
        .unwrap();
        assert_eq!(launched, Launch::Launched);
        let mut own = stream.alloc::<f32>(count + 4).unwrap();
        stream.fill_bytes(own.as_mut_slice(), 0xA5).unwrap();
        let launched = k::mmq_q8_moe_gated_vec(
            &stream,
            QuantSlice::new(gate.as_slice(), format),
            QuantSlice::new(up.as_slice(), format),
            q.slice(0..bytes),
            ids.slice(0..n * USED),
            own.slice_mut(0..count),
            M as u32,
            K as u32,
            n as u32,
            EXPERTS as u32,
            USED as u32,
            scratch.as_mut_slice(),
            SCRATCH,
        )
        .unwrap();
        assert_eq!(launched, Launch::Launched, "{format:?} n {n}");
        let mut expected = vec![0f32; count];
        stream.download(library.as_slice(), &mut expected).unwrap();
        let mut host = vec![0f32; count + 4];
        stream.download(own.as_slice(), &mut host).unwrap();
        assert!(
            expected.iter().all(|v| v.is_finite()) && expected.iter().any(|&v| v != 0.0),
            "{format:?} n {n}"
        );
        let first = host[..count]
            .iter()
            .zip(&expected)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert_eq!(first, None, "{format:?} n {n}: the outputs differ");
        assert!(
            host[count..].iter().all(|v| v.to_bits() == GUARD),
            "{format:?} n {n}: the projection overwrote its guard"
        );
    }
}

#[test]
fn prequantized_rows_reproduce_the_quantizing_projection() {
    let vector: Vec<usize> = (1..=8).collect();
    let mut q4k = vector.clone();
    q4k.extend([9, 16, 40, 64]);
    check(GgmlType::Q4K, &q4k);
    check(GgmlType::Q5K, &vector);
    check(GgmlType::Q8_0, &vector);
}
