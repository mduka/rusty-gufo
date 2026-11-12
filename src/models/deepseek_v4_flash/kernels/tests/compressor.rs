//! Compressor spans (`compressor_test` of the C++ tests): positions stored
//! in spans up to the next pool boundary leave the state of single-position
//! stores after every span (values, scores with their positional bias,
//! untouched slots and the ratio-4 shift) and bit-identical compressed
//! rows; a span across a pool boundary is refused:
//! cargo test -p gufo-deepseek-v4-flash-kernels --features gpu --test compressor

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::io::Write;

use gufo_deepseek_v4_flash_kernels::{
    Backend, CompressorWeights, DeviceTensor, Input, Rope, Weight,
};
use gufo_hip::MappedFile;

/// F16 positional-bias values, then the F32 norm weights.
const APE_VALUES: usize = 128 * 1024;
const NORM_OFFSET: u64 = APE_VALUES as u64 * 2;

/// The synthetic positional bias: multiples of 1/1024, exact in F16.
fn ape(i: usize) -> f32 {
    ((i % 257) as i32 - 128) as f32 / 1024.0
}

/// The F16 bits of `value`, a multiple of 1/1024 of magnitude at most 1/8.
fn f16_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exponent) && bits & 0x1fff == 0);
    (((bits >> 16) & 0x8000) | ((exponent as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16
}

fn floats(b: &Backend, t: &impl Input, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    b.read(t, 0, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

fn tensor(b: &Backend, values: &[f32]) -> DeviceTensor {
    let mut t = b.alloc(values.len() as u64 * 4).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    b.write(&mut t, 0, &bytes).unwrap();
    t
}

fn run(b: &Backend, dim: u32, ratio: u32, phase: u32) {
    let w = CompressorWeights {
        ape: Weight::support(0),
        norm: Weight::support(NORM_OFFSET),
    };
    let rope = Rope {
        n_rot: 16,
        pos0: 0,
        n_ctx_orig: 0,
        freq_base: 10000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let coff = if ratio == 4 { 2 } else { 1 };
    let width = (coff * dim) as usize;
    let (rows, start) = (2 * ratio + 3, ratio * 3 + phase);
    let outputs = (phase + rows) / ratio;
    let state_size = coff as usize * ratio as usize * width;
    let kv: Vec<f32> = (0..rows as usize * width)
        .map(|i| (((i * 19) % 521) as i32 - 260) as f32 / 256.0)
        .collect();
    let scores: Vec<f32> = (0..kv.len())
        .map(|i| (((i * 13) % 257) as i32 - 128) as f32 / 32.0)
        .collect();
    let (whole_kv, whole_sc) = (tensor(b, &kv), tensor(b, &scores));
    let mut sequential = Vec::new();
    for grouped in [false, true] {
        let mut expected_kv = vec![0.0f32; state_size];
        let mut expected_score = vec![-100.0f32; state_size];
        let mut state = tensor(b, &expected_kv);
        let mut ss = tensor(b, &expected_score);
        let mut output = b.alloc(u64::from(outputs * dim) * 4).unwrap();
        assert!(
            b.compressor_store_batch(
                &whole_kv,
                &whole_sc,
                &mut state,
                &mut ss,
                w.ape,
                dim,
                ratio,
                start,
                ratio - phase + 1,
            )
            .is_err(),
            "a store spanning a pool boundary is rejected"
        );
        let mut produced = 0;
        let mut t = 0;
        while t < rows {
            let count = if grouped {
                (rows - t).min(ratio - (start + t) % ratio)
            } else {
                1
            };
            // Views fall on 16-byte offsets: each span gets its own tensors.
            let span = t as usize * width..(t + count) as usize * width;
            let (k, s) = (tensor(b, &kv[span.clone()]), tensor(b, &scores[span]));
            let position = start + t + count - 1;
            if grouped {
                b.compressor_store_batch(
                    &k,
                    &s,
                    &mut state,
                    &mut ss,
                    w.ape,
                    dim,
                    ratio,
                    start + t,
                    count,
                )
                .unwrap();
            }
            b.compressor_update(
                &k,
                &s,
                &mut state,
                &mut ss,
                &mut output,
                w,
                dim,
                ratio,
                position,
                produced,
                rope,
                1e-6,
                grouped,
            )
            .unwrap();
            for row in t..t + count {
                let slot = ((start + row) % ratio) as usize;
                let dst = if ratio == 4 {
                    ratio as usize + slot
                } else {
                    slot
                };
                for d in 0..width {
                    expected_kv[dst * width + d] = kv[row as usize * width + d];
                    expected_score[dst * width + d] =
                        scores[row as usize * width + d] + ape(slot * width + d);
                }
            }
            if (position + 1).is_multiple_of(ratio) {
                produced += 1;
                if ratio == 4 {
                    let half = ratio as usize * width;
                    expected_kv.copy_within(half..2 * half, 0);
                    expected_score.copy_within(half..2 * half, 0);
                }
            }
            assert!(
                floats(b, &state, state_size) == expected_kv,
                "state values match the independent row mapping (dim {dim} ratio {ratio} phase {phase})"
            );
            assert!(
                floats(b, &ss, state_size) == expected_score,
                "scores, untouched slots and ratio-four shift match (dim {dim} ratio {ratio} phase {phase})"
            );
            t += count;
        }
        assert_eq!(produced, outputs, "all compression boundaries emitted");
        let result: Vec<u32> = floats(b, &output, (outputs * dim) as usize)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        if grouped {
            assert_eq!(
                result, sequential,
                "grouped and sequential compressed rows are bit-identical"
            );
        } else {
            sequential = result;
        }
    }
}

#[test]
fn spans_up_to_a_pool_boundary_keep_single_position_states() {
    let mut b = Backend::init().unwrap();
    let mut model: Vec<u8> = (0..APE_VALUES)
        .flat_map(|i| f16_bits(ape(i)).to_le_bytes())
        .collect();
    model.extend((0..512).flat_map(|_| 1.0f32.to_le_bytes()));
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&model).unwrap();
    file.flush().unwrap();
    let map = MappedFile::open(file.path()).unwrap();
    let len = model.len() as u64;
    b.set_support(
        map,
        len.next_multiple_of(256),
        &[(0, len, "compressor fixture")],
    )
    .unwrap();
    for ratio in [4u32, 128] {
        for dim in [17u32, 128, 512] {
            for phase in [0, 1, ratio - 1] {
                run(&b, dim, ratio, phase);
            }
        }
    }
}
