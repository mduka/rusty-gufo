//! The attention kernel against a direct f64 evaluation for both head
//! widths, packed sequences, and windowed and whole-sequence masks:
//! cargo test -p gufo-embedding-gemma2-kernels --features gpu --test attention

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_embedding_gemma2_kernels as k;
use gufo_hip::{Bf16, DeviceBuffer, DeviceCopy, DeviceIndices, Launch, Stream};

const HEADS: u32 = 4;

/// xorshift64 with Box-Muller normals.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unit(), self.unit());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }
}

fn device<T: DeviceCopy>(stream: &Stream, host: &[T]) -> DeviceBuffer<T> {
    let mut buffer = stream.alloc::<T>(host.len().max(1)).unwrap();
    if !host.is_empty() {
        stream
            .upload(host, buffer.slice_mut(0..host.len()))
            .unwrap();
    }
    buffer
}

fn indices(stream: &Stream, host: &[u32], bound: u32) -> DeviceIndices {
    let mut indices = stream.alloc_indices(host.len().max(1), bound).unwrap();
    if !host.is_empty() {
        stream.upload_indices(host, &mut indices).unwrap();
    }
    indices
}

fn run_case(head_dim: u32, kv_heads: u32, window: u32, lengths: &[u32]) {
    let stream = Stream::new().unwrap();
    let tokens: usize = lengths.iter().map(|&l| l as usize).sum();
    let hd = head_dim as usize;
    let mut random = Rng(0x9e37_79b9_7f4a_7c15 ^ u64::from(head_dim + window));
    let mut fill = |heads: u32| -> Vec<Bf16> {
        (0..tokens * heads as usize * hd)
            .map(|_| Bf16::from_f32(random.normal()))
            .collect()
    };
    let query = fill(HEADS);
    let key = fill(kv_heads);
    let value = fill(kv_heads);
    let mut begin = Vec::new();
    let mut end = Vec::new();
    let mut offset = 0u32;
    for &length in lengths {
        for _ in 0..length {
            begin.push(offset);
            end.push(offset + length);
        }
        offset += length;
    }

    let d_query = device(&stream, &query);
    let d_key = device(&stream, &key);
    let d_value = device(&stream, &value);
    let rows = tokens as u32;
    let d_begin = indices(&stream, &begin, rows);
    let d_end = indices(&stream, &end, rows + 1);
    let mut d_context = stream.alloc::<Bf16>(tokens * HEADS as usize * hd).unwrap();
    let launch = k::attention(
        &stream,
        d_query.as_slice(),
        d_key.as_slice(),
        d_value.as_slice(),
        d_context.as_mut_slice(),
        d_begin.view(),
        d_end.view(),
        tokens,
        HEADS,
        kv_heads,
        head_dim,
        window,
        tokens,
    )
    .unwrap();
    assert_eq!(launch, Launch::Launched, "supported geometry");
    let mut context = vec![Bf16::default(); d_context.len()];
    stream.download(d_context.as_slice(), &mut context).unwrap();

    let mut worst = 0.0f64;
    for t in 0..tokens {
        let (low, high) = if window == 0 {
            (i64::from(begin[t]), i64::from(end[t]) - 1)
        } else {
            (
                i64::from(begin[t]).max(t as i64 - i64::from(window)),
                (i64::from(end[t]) - 1).min(t as i64 + i64::from(window)),
            )
        };
        for h in 0..HEADS as usize {
            let kvh = h / (HEADS / kv_heads) as usize;
            let kv = kv_heads as usize;
            let scores: Vec<f64> = (low..=high)
                .map(|s| {
                    let s = s as usize;
                    (0..hd)
                        .map(|i| {
                            f64::from(query[(t * HEADS as usize + h) * hd + i].to_f32())
                                * f64::from(key[(s * kv + kvh) * hd + i].to_f32())
                        })
                        .sum()
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
            let total: f64 = weights.iter().sum();
            for i in 0..hd {
                let expected: f64 = (low..=high)
                    .zip(&weights)
                    .map(|(s, w)| {
                        w / total * f64::from(value[(s as usize * kv + kvh) * hd + i].to_f32())
                    })
                    .sum();
                let actual = f64::from(context[(t * HEADS as usize + h) * hd + i].to_f32());
                worst = worst.max((actual - expected).abs());
            }
        }
    }
    println!("head {head_dim} window {window} tokens {tokens}: worst error {worst}");
    // Outputs are rounded to BF16, whose relative step is 2^-8.
    assert!(
        worst < 0.02,
        "attention differs from the f64 evaluation: {worst}"
    );
}

#[test]
fn attention_matches_a_direct_evaluation() {
    run_case(256, 2, 5, &[40, 17, 1, 130]);
    run_case(256, 2, 0, &[40, 17, 1, 130]);
    run_case(512, 1, 0, &[64, 33, 7]);
    run_case(512, 1, 9, &[64, 33, 7]);
    run_case(256, 2, 100, &[700]);
    run_case(256, 2, 0, &[31, 32, 33, 1]);
    run_case(512, 1, 0, &[32, 16, 15]);
}

#[test]
fn unsupported_geometries_are_declined() {
    let stream = Stream::new().unwrap();
    let decline = |tokens: usize, heads: u32, kv_heads: u32, head_dim: u32| {
        let rows = tokens.max(1);
        let width = rows * heads as usize * head_dim as usize;
        let query = stream.alloc::<Bf16>(width).unwrap();
        let key = stream
            .alloc::<Bf16>(rows * kv_heads as usize * head_dim as usize)
            .unwrap();
        let mut context = stream.alloc::<Bf16>(width).unwrap();
        let begin = stream.alloc_indices(rows, rows as u32).unwrap();
        let end = stream.alloc_indices(rows, rows as u32 + 1).unwrap();
        k::attention(
            &stream,
            query.as_slice(),
            key.as_slice(),
            key.as_slice(),
            context.as_mut_slice(),
            begin.view(),
            end.view(),
            tokens,
            heads,
            kv_heads,
            head_dim,
            0,
            rows,
        )
        .unwrap()
    };
    assert_eq!(
        decline(1, 8, 2, 128),
        Launch::Unsupported,
        "an unsupported head dimension is rejected"
    );
    assert_eq!(
        decline(0, 8, 2, 256),
        Launch::Unsupported,
        "an empty batch is rejected"
    );
}
