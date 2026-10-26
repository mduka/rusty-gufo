//! Device tests: cargo test -p gufo-qwen3-asr-kernels --features gpu
//!
//! References reproduce the documented BF16 rounding sequence of each kernel.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{Bf16, DeviceBuffer, DeviceCopy, Launch, LaunchError, Stream};
use gufo_qwen3_asr_kernels as k;

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| scale * self.unit()).collect()
    }
}

fn bf16(x: f32) -> f32 {
    Bf16::from_f32(x).to_f32()
}

fn device<T: DeviceCopy>(stream: &Stream, host: &[T]) -> DeviceBuffer<T> {
    let mut buffer = stream.alloc::<T>(host.len()).unwrap();
    stream.upload(host, buffer.as_mut_slice()).unwrap();
    buffer
}

fn host<T: DeviceCopy + Default>(stream: &Stream, buffer: &DeviceBuffer<T>) -> Vec<T> {
    let mut values = vec![T::default(); buffer.len()];
    stream.download(buffer.as_slice(), &mut values).unwrap();
    values
}

fn rms_norm_reference(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<Bf16> {
    input
        .chunks(weight.len())
        .flat_map(|row| {
            let mean_square = row
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                / row.len() as f64;
            let inverse = 1.0 / ((mean_square as f32) + epsilon).sqrt();
            row.iter()
                .zip(weight)
                .map(move |(&v, &w)| Bf16::from_f32(bf16(v * inverse) * w))
        })
        .collect()
}

#[test]
fn rms_norm_is_bit_identical_to_the_bf16_reference() {
    let stream = Stream::new().unwrap();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (batch, dim) = (4, 2048);
    let input: Vec<f32> = rng.vec(batch * dim, 1.0).into_iter().map(bf16).collect();
    let weight: Vec<f32> = rng
        .vec(dim, 0.5)
        .into_iter()
        .map(|w| bf16(1.0 + w))
        .collect();
    let (d_input, d_weight) = (device(&stream, &input), device(&stream, &weight));
    let mut d_output = stream.alloc::<Bf16>(batch * dim).unwrap();
    k::rms_norm(
        &stream,
        d_input.as_slice(),
        d_weight.as_slice(),
        d_output.as_mut_slice(),
        batch,
        dim,
        1e-6,
    )
    .unwrap();
    assert_eq!(
        host(&stream, &d_output),
        rms_norm_reference(&input, &weight, 1e-6)
    );
}

#[test]
fn frozen_weights_feed_launches_on_any_stream() {
    let (loader, compute) = (Stream::new().unwrap(), Stream::new().unwrap());
    let mut rng = Rng(5);
    let (batch, dim) = (2, 1024);
    let input: Vec<f32> = rng.vec(batch * dim, 1.0).into_iter().map(bf16).collect();
    let weight: Vec<f32> = rng
        .vec(dim, 0.5)
        .into_iter()
        .map(|w| bf16(1.0 + w))
        .collect();
    let d_weight = device(&loader, &weight).freeze(&loader).unwrap();
    let d_input = device(&compute, &input);
    let mut d_output = compute.alloc::<Bf16>(batch * dim).unwrap();
    k::rms_norm(
        &compute,
        d_input.as_slice(),
        d_weight.view(),
        d_output.as_mut_slice(),
        batch,
        dim,
        1e-6,
    )
    .unwrap();
    assert_eq!(
        host(&compute, &d_output),
        rms_norm_reference(&input, &weight, 1e-6)
    );
}

#[test]
fn residual_add_updates_hidden_and_normalizes_the_sum() {
    let stream = Stream::new().unwrap();
    let mut rng = Rng(7);
    let (batch, dim) = (3, 1024);
    let hidden: Vec<f32> = rng.vec(batch * dim, 1.0).into_iter().map(bf16).collect();
    let update = rng.vec(batch * dim, 1.0);
    let weight: Vec<f32> = rng
        .vec(dim, 0.5)
        .into_iter()
        .map(|w| bf16(1.0 + w))
        .collect();
    let mut d_hidden = device(&stream, &hidden);
    let (d_update, d_weight) = (device(&stream, &update), device(&stream, &weight));
    let mut d_output = stream.alloc::<Bf16>(batch * dim).unwrap();
    k::residual_add_rms_norm(
        &stream,
        d_hidden.as_mut_slice(),
        d_update.as_slice(),
        d_weight.as_slice(),
        d_output.as_mut_slice(),
        batch,
        dim,
        1e-6,
    )
    .unwrap();
    let summed: Vec<f32> = hidden
        .iter()
        .zip(&update)
        .map(|(&h, &u)| bf16(h + bf16(u)))
        .collect();
    assert_eq!(host(&stream, &d_hidden), summed);
    assert_eq!(
        host(&stream, &d_output),
        rms_norm_reference(&summed, &weight, 1e-6)
    );
}

#[test]
fn swiglu_matches_the_bf16_reference() {
    let stream = Stream::new().unwrap();
    let mut rng = Rng(11);
    let count = 1 << 16;
    let (gate, up) = (rng.vec(count, 4.0), rng.vec(count, 1.0));
    let (d_gate, d_up) = (device(&stream, &gate), device(&stream, &up));
    let mut d_output = stream.alloc::<Bf16>(count).unwrap();
    k::swiglu(
        &stream,
        d_gate.as_slice(),
        d_up.as_slice(),
        d_output.as_mut_slice(),
        count,
    )
    .unwrap();
    let worst_ulps = host(&stream, &d_output)
        .iter()
        .zip(gate.iter().zip(&up))
        .map(|(got, (&g, &u))| {
            let g = bf16(g);
            let expected = Bf16::from_f32(bf16(g / (1.0 + (-g).exp())) * bf16(u));
            (i32::from(got.0 as i16) - i32::from(expected.0 as i16)).unsigned_abs()
        })
        .max()
        .unwrap();
    // Device and host `expf` may differ in the last f32 bit.
    assert!(worst_ulps <= 1, "{worst_ulps} BF16 ulp");
}

#[test]
fn bf16_round_trip_is_exact() {
    let stream = Stream::new().unwrap();
    let values = Rng(3).vec(4096, 100.0);
    let mut d_values = device(&stream, &values);
    k::round_bf16(&stream, d_values.as_mut_slice(), values.len()).unwrap();
    let rounded: Vec<f32> = values.iter().map(|&v| bf16(v)).collect();
    assert_eq!(host(&stream, &d_values), rounded);

    let packed: Vec<Bf16> = values.iter().map(|&v| Bf16::from_f32(v)).collect();
    let d_packed = device(&stream, &packed);
    let mut d_widened = stream.alloc::<f32>(values.len()).unwrap();
    k::bf16_to_f32(
        &stream,
        d_packed.as_slice(),
        d_widened.as_mut_slice(),
        values.len(),
    )
    .unwrap();
    assert_eq!(host(&stream, &d_widened), rounded);
}

struct Attention {
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    key_cache: DeviceBuffer<f32>,
    value_cache: DeviceBuffer<f32>,
}

const HEADS: u32 = 16;
const KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const MAX_CONTEXT: u32 = 64;
const LAYERS: u32 = 2;
const TOKENS: usize = 3;

impl Attention {
    fn new(stream: &Stream) -> Self {
        let zeros = |len: usize| device(stream, &vec![0.0f32; len]);
        let cache = (LAYERS * KV_HEADS * MAX_CONTEXT * HEAD_DIM) as usize;
        Self {
            query: zeros(TOKENS * (HEADS * HEAD_DIM) as usize),
            key: zeros(TOKENS * (KV_HEADS * HEAD_DIM) as usize),
            value: zeros(TOKENS * (KV_HEADS * HEAD_DIM) as usize),
            norm: device(stream, &vec![1.0f32; HEAD_DIM as usize]),
            key_cache: zeros(cache),
            value_cache: zeros(cache),
        }
    }

    fn launch(
        &mut self,
        stream: &Stream,
        layer: u32,
        head_dim: u32,
        cache_len: usize,
    ) -> Result<Launch, LaunchError> {
        k::qk_norm_rope(
            stream,
            self.query.as_mut_slice(),
            self.key.as_mut_slice(),
            self.value.as_mut_slice(),
            self.norm.as_slice(),
            self.norm.as_slice(),
            TOKENS,
            HEADS,
            KV_HEADS,
            head_dim,
            0,
            1_000_000.0,
            1e-6,
            Some(self.key_cache.slice_mut(0..cache_len)),
            Some(self.value_cache.slice_mut(0..cache_len)),
            layer,
            MAX_CONTEXT,
        )
    }
}

#[test]
fn kv_cache_extents_are_checked_before_launch() {
    let stream = Stream::new().unwrap();
    let mut attention = Attention::new(&stream);
    let full = attention.key_cache.len();
    assert_eq!(
        attention.launch(&stream, 1, HEAD_DIM, full),
        Ok(Launch::Launched)
    );
    assert_eq!(
        attention.launch(&stream, 2, HEAD_DIM, full),
        Err(LaunchError::ExtentTooShort {
            param: "key_cache",
            required: full / 2 * 3,
            actual: full
        })
    );
    assert_eq!(
        attention.launch(&stream, 1, HEAD_DIM, full - 1),
        Err(LaunchError::ExtentTooShort {
            param: "key_cache",
            required: full,
            actual: full - 1
        })
    );
    stream.synchronize().unwrap();
}

#[test]
fn unsupported_geometry_is_reported_not_launched() {
    let stream = Stream::new().unwrap();
    let mut attention = Attention::new(&stream);
    let full = attention.key_cache.len();
    assert_eq!(
        attention.launch(&stream, 0, HEAD_DIM - 1, full),
        Ok(Launch::Unsupported)
    );
}

#[test]
fn buffers_of_another_stream_are_refused() {
    let (stream, other) = (Stream::new().unwrap(), Stream::new().unwrap());
    let gate = device(&stream, &[1.0f32; 8]);
    let mut output = other.alloc::<Bf16>(8).unwrap();
    let error = k::swiglu(
        &other,
        gate.as_slice(),
        gate.as_slice(),
        output.as_mut_slice(),
        8,
    )
    .unwrap_err();
    assert_eq!(error, LaunchError::ForeignStream { param: "gate" });
}

#[test]
fn embedding_rows_are_gathered_by_checked_indices_only() {
    let stream = Stream::new().unwrap();
    let (rows, hidden) = (6usize, 40usize);
    let table: Vec<Bf16> = (0..rows * hidden)
        .map(|i| Bf16::from_f32(i as f32))
        .collect();
    let table = device(&stream, &table);
    let mut ids = stream.alloc_indices(3, rows as u32).unwrap();
    stream.upload_indices(&[5, 0, 2], &mut ids).unwrap();
    let mut output = stream.alloc::<f32>(3 * hidden).unwrap();
    k::embedding_lookup(
        &stream,
        table.as_slice(),
        ids.view(),
        output.as_mut_slice(),
        3,
        hidden,
        rows,
    )
    .unwrap();
    let values = host(&stream, &output);
    for (b, row) in [5usize, 0, 2].into_iter().enumerate() {
        let expected: Vec<f32> = (0..hidden).map(|c| (row * hidden + c) as f32).collect();
        assert_eq!(values[b * hidden..(b + 1) * hidden], expected[..]);
    }

    // Indices checked against a larger table cannot index a smaller one.
    let wide = stream.alloc_indices(3, rows as u32 + 1).unwrap();
    let error = k::embedding_lookup(
        &stream,
        table.as_slice(),
        wide.view(),
        output.as_mut_slice(),
        3,
        hidden,
        rows,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            LaunchError::IndexBound {
                param: "token_ids",
                bound: 7,
                limit: 6
            }
        ),
        "{error}"
    );
    // Nor can the contract claim a table larger than the buffer.
    let error = k::embedding_lookup(
        &stream,
        table.as_slice(),
        ids.view(),
        output.as_mut_slice(),
        3,
        hidden,
        rows + 1,
    )
    .unwrap_err();
    assert!(
        matches!(error, LaunchError::ExtentTooShort { param: "table", .. }),
        "{error}"
    );
}

#[test]
fn requirements_refuse_positions_beyond_the_cache() {
    let stream = Stream::new().unwrap();
    let (heads, kv, dim, context) = (2u32, 1u32, 64u32, 8u32);
    let query = stream.alloc::<f32>(4 * (heads * dim) as usize).unwrap();
    let cache_len = (kv * context * dim) as usize;
    let keys = stream.alloc::<f32>(cache_len).unwrap();
    let values = stream.alloc::<f32>(cache_len).unwrap();
    let mut out = stream.alloc::<f32>(4 * (heads * dim) as usize).unwrap();
    let error = k::batched_attention(
        &stream,
        query.as_slice(),
        keys.as_slice(),
        values.as_slice(),
        out.as_mut_slice(),
        None,
        0,
        5,
        4,
        context,
        heads,
        kv,
        dim,
    )
    .unwrap_err();
    assert_eq!(
        error,
        LaunchError::Precondition {
            launcher: "LaunchTextBatchedAttention",
            requirement: "start_position + tokens <= max_context",
        }
    );
    let error = k::batched_attention(
        &stream,
        query.as_slice(),
        keys.as_slice(),
        values.as_slice(),
        out.as_mut_slice(),
        None,
        0,
        0,
        4,
        context,
        3,
        2,
        dim,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            LaunchError::Precondition {
                requirement: "num_heads % num_key_value_heads == 0",
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn argmax_takes_the_lowest_index_and_fails_closed_on_nan() {
    let stream = Stream::new().unwrap();
    let count = 151_936;
    let mut logits = vec![-1.0f32; count];
    logits[70_000] = 3.5;
    logits[90_000] = 3.5;
    logits[5] = 3.25;
    let device_logits = device(&stream, &logits);
    let mut scratch = stream.alloc::<f32>(2048).unwrap();
    let mut token = stream.alloc::<u32>(1).unwrap();
    for with_scratch in [true, false] {
        k::argmax(
            &stream,
            device_logits.as_slice(),
            token.as_mut_slice(),
            count,
            with_scratch.then(|| scratch.as_mut_slice()),
        )
        .unwrap();
        assert_eq!(host(&stream, &token), [70_000]);
    }
    logits[123] = f32::NAN;
    let device_logits = device(&stream, &logits);
    k::argmax(
        &stream,
        device_logits.as_slice(),
        token.as_mut_slice(),
        count,
        Some(scratch.as_mut_slice()),
    )
    .unwrap();
    assert_eq!(host(&stream, &token), [u32::MAX]);
}
