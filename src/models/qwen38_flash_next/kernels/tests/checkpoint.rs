//! A prefill checkpoint inside `gated_delta_net` (`gdn_ops_test.cpp`): the full pass is unchanged bit for bit, the captured
//! recurrent state and convolution history equal those of a pass ending at
//! the checkpoint, and nothing is written past either buffer. Both
//! recurrences (row split and per-row) and the `history_prefix` launcher:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next-kernels --features gpu --test checkpoint

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{DeviceBuffer, Launch, Stream};
use gufo_qwen38_flash_next_kernels as k;

const K_HEADS: usize = 16;
const V_HEADS: usize = 48;
const D: usize = 128;
const KERNEL: usize = 4;
const GUARD: usize = 16;
const SENTINEL: f32 = 12345.0;

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

fn download(stream: &Stream, buffer: &DeviceBuffer<f32>) -> Vec<u32> {
    let mut host = vec![0.0f32; buffer.len()];
    stream.download(buffer.as_slice(), &mut host).unwrap();
    host.iter().map(|v| v.to_bits()).collect()
}

/// One layer's inputs and the state a pass starts from.
struct Layer {
    qkvz: DeviceBuffer<f32>,
    alpha_beta: DeviceBuffer<f32>,
    conv_w: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    dt: DeviceBuffer<f32>,
    norm_w: DeviceBuffer<f32>,
    conv_state: Vec<f32>,
    state: Vec<f32>,
}

/// What a pass over the first `n` rows leaves: state, history and output.
struct Pass {
    state: Vec<u32>,
    history: Vec<u32>,
    out: Vec<u32>,
}

fn channels() -> usize {
    2 * K_HEADS * D + V_HEADS * D
}

fn run(
    stream: &Stream,
    layer: &Layer,
    n: usize,
    row_split: bool,
    checkpoint: Option<(&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>, u32)>,
) -> Pass {
    let channels = channels();
    let value = V_HEADS * D;
    let stride = channels + value;
    let mut conv = upload(stream, &layer.conv_state);
    let mut state = upload(stream, &layer.state);
    let mut scratch = stream.alloc::<f32>((n + KERNEL - 1) * channels).unwrap();
    let mut qn = stream.alloc::<f32>(n * K_HEADS * D).unwrap();
    let mut kn = stream.alloc::<f32>(n * K_HEADS * D).unwrap();
    let mut raw = stream.alloc::<f32>(n * value).unwrap();
    let mut out = stream.alloc::<f32>(n * value).unwrap();
    let (capture_state, capture_history, tokens) = match checkpoint {
        Some((s, h, t)) => (
            Some(s.slice_mut(0..V_HEADS * D * D)),
            Some(h.slice_mut(0..(KERNEL - 1) * channels)),
            t,
        ),
        None => (None, None, 0),
    };
    let launched = k::gated_delta_net(
        stream,
        layer.qkvz.slice(0..(n - 1) * stride + channels),
        stride as u32,
        layer
            .qkvz
            .slice(channels..(n - 1) * stride + channels + value),
        stride as u32,
        layer.alpha_beta.slice(0..n * 2 * V_HEADS),
        layer.conv_w.as_slice(),
        layer.a.as_slice(),
        layer.dt.as_slice(),
        layer.norm_w.as_slice(),
        conv.as_mut_slice(),
        scratch.as_mut_slice(),
        qn.as_mut_slice(),
        kn.as_mut_slice(),
        raw.as_mut_slice(),
        state.as_mut_slice(),
        out.as_mut_slice(),
        None,
        None,
        &mut [],
        &mut [],
        0,
        n as u32,
        K_HEADS as u32,
        V_HEADS as u32,
        KERNEL as u32,
        row_split,
        false,
        1e-6,
        capture_state,
        capture_history,
        tokens,
    )
    .unwrap();
    assert_eq!(launched, Launch::Launched);
    Pass {
        state: download(stream, &state),
        history: download(stream, &conv),
        out: download(stream, &out),
    }
}

#[test]
fn gated_delta_net_checkpoint_equals_a_pass_ending_there() {
    let stream = Stream::new().unwrap();
    let channels = channels();
    let stride = channels + V_HEADS * D;
    let mut seed = 71;
    for n in [3usize, 11, 130] {
        let layer = Layer {
            qkvz: upload(&stream, &random(&mut seed, n * stride, 1.0)),
            alpha_beta: upload(&stream, &random(&mut seed, n * 2 * V_HEADS, 2.0)),
            conv_w: upload(&stream, &random(&mut seed, channels * KERNEL, 0.5)),
            a: upload(&stream, &random(&mut seed, V_HEADS, 1.0)),
            dt: upload(&stream, &random(&mut seed, V_HEADS, 1.0)),
            norm_w: upload(
                &stream,
                &(0..D)
                    .map(|_| 1.0 + uniform(&mut seed, 0.3))
                    .collect::<Vec<_>>(),
            ),
            conv_state: random(&mut seed, (KERNEL - 1) * channels, 1.0),
            state: random(&mut seed, V_HEADS * D * D, 0.1),
        };
        for row_split in [true, false] {
            let full = run(&stream, &layer, n, row_split, None);
            for prefix in [1, (n / 2).max(1), n] {
                let what = format!("n {n} prefix {prefix} row split {row_split}");
                let state_floats = V_HEADS * D * D;
                let history_floats = (KERNEL - 1) * channels;
                let mut state = upload(&stream, &vec![SENTINEL; state_floats + GUARD]);
                let mut history = upload(&stream, &vec![SENTINEL; history_floats + GUARD]);
                let captured = run(
                    &stream,
                    &layer,
                    n,
                    row_split,
                    Some((&mut state, &mut history, prefix as u32)),
                );
                assert!(
                    captured.state == full.state
                        && captured.history == full.history
                        && captured.out == full.out,
                    "{what}: the checkpoint changed the full pass"
                );
                let ending = run(&stream, &layer, prefix, row_split, None);
                let state = download(&stream, &state);
                let history = download(&stream, &history);
                assert!(
                    state[..state_floats] == ending.state[..],
                    "{what}: captured state differs from a pass ending there"
                );
                assert!(
                    history[..history_floats] == ending.history[..],
                    "{what}: captured history differs from a pass ending there"
                );
                assert!(
                    state[state_floats..]
                        .iter()
                        .chain(&history[history_floats..])
                        .all(|&v| v == SENTINEL.to_bits()),
                    "{what}: the checkpoint wrote past its buffers"
                );
            }
        }
    }
}

/// `history_prefix` leaves the rolling history a pass over its first rows
/// would, without advancing the live one.
#[test]
fn history_prefix_is_the_shifted_window() {
    let stream = Stream::new().unwrap();
    let mut seed = 5;
    for (tokens, stride, channels, rows) in [(5usize, 2600usize, 2560usize, 9usize), (2, 40, 33, 3)]
    {
        let input = random(&mut seed, (tokens - 1) * stride + channels, 1.0);
        let history = random(&mut seed, rows * channels, 1.0);
        let input_d = upload(&stream, &input);
        let history_d = upload(&stream, &history);
        let mut destination = upload(&stream, &vec![SENTINEL; rows * channels]);
        k::history_prefix(
            &stream,
            input_d.as_slice(),
            stride as u32,
            history_d.as_slice(),
            destination.as_mut_slice(),
            tokens as u32,
            channels as u32,
            rows as u32,
        )
        .unwrap();
        let got = download(&stream, &destination);
        for j in 0..rows {
            let source = tokens + j;
            for c in 0..channels {
                let expected = if source < rows {
                    history[source * channels + c]
                } else {
                    input[(source - rows) * stride + c]
                };
                assert_eq!(
                    got[j * channels + c],
                    expected.to_bits(),
                    "row {j} channel {c}"
                );
            }
        }
        assert_eq!(
            download(&stream, &history_d),
            history.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
