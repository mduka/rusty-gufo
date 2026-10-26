//! The kernels that write the Q8_1 rows of the next decode GEMV themselves
//! equal the kernel they extend followed by `mmq_quantize_q8_1`, bit for bit:
//! `silu_scale_q8_1`, `sigmoid_mul_q8_1`, `hc_combine_q8_1`, the
//! `hc_mix_epilogue*_q8_1` pair and `gated_delta_net_q8_1` against the plain
//! launcher (floats and Q8_1 bytes). Only `silu_scale_q8_1` takes padded
//! rows, and leaves their padding blocks as they are; the others refuse
//! them:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next-kernels --features gpu --test q8_rows

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{DeviceBuffer, DeviceCopy, Launch, Stream};
use gufo_qwen38_flash_next_kernels as k;

/// A deterministic value in `[-scale, scale)`.
fn uniform(seed: &mut u32, scale: f32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
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

/// Rows the quantizer pads: the guard pattern must not survive in them.
fn quantized(stream: &Stream, x: &DeviceBuffer<f32>, n: usize, k: usize) -> Vec<u8> {
    let mut rows = stream.alloc::<u8>(k::q8_1_bytes(n, k)).unwrap();
    stream.fill_bytes(rows.as_mut_slice(), 0xA5).unwrap();
    let launched = k::mmq_quantize_q8_1(
        stream,
        x.slice(0..n * k),
        rows.as_mut_slice(),
        n as u32,
        k as u32,
    )
    .unwrap();
    assert_eq!(launched, Launch::Launched);
    download(stream, &rows)
}

fn assert_same<T: PartialEq + std::fmt::Debug>(own: &[T], reference: &[T], what: &str) {
    assert_eq!(own.len(), reference.len(), "{what}: lengths differ");
    let first = own.iter().zip(reference).position(|(a, b)| a != b);
    assert_eq!(first, None, "{what}: first difference");
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The blocks of `n` Q8_1 rows of `k` values that hold values, without the
/// padding blocks after each row.
fn unpadded(rows: &[u8], n: usize, k: usize) -> Vec<u8> {
    let (row, used) = (k::q8_1_bytes(1, k), k / 32 * 36);
    (0..n)
        .flat_map(|r| rows[r * row..r * row + used].to_vec())
        .collect()
}

#[test]
fn silu_scale_rows_equal_the_quantizer() {
    let stream = Stream::new().unwrap();
    let shapes = [(1, 320), (8, 320), (3, 1024), (8, 10240), (5, 544)];
    for (round, (n, kk)) in (0..8).flat_map(|r| shapes.map(|s| (r, s))) {
        let mut seed = 17 + kk as u32 + 1000 * round;
        let mut host: Vec<f32> = (0..n * kk).map(|_| uniform(&mut seed, 6.0)).collect();
        // A zero block (scale 0) and a block of one repeated value.
        host[32..64].fill(0.0);
        host[64..96].fill(1.5);
        let mut reference = upload(&stream, &host);
        k::silu_scale(&stream, reference.as_mut_slice(), 0.25, n * kk).unwrap();
        let expected_rows = quantized(&stream, &reference, n, kk);

        let mut own = upload(&stream, &host);
        let mut rows = stream.alloc::<u8>(k::q8_1_bytes(n, kk)).unwrap();
        stream.fill_bytes(rows.as_mut_slice(), 0x5A).unwrap();
        let launched = k::silu_scale_q8_1(
            &stream,
            own.as_mut_slice(),
            0.25,
            rows.as_mut_slice(),
            n as u32,
            kk as u32,
        )
        .unwrap();
        assert_eq!(launched, Launch::Launched);
        let what = format!("silu_scale round {round} n {n} k {kk}");
        assert_same(
            &bits(&download(&stream, &own)),
            &bits(&download(&stream, &reference)),
            &format!("{what} values"),
        );
        assert_same(
            &unpadded(&download(&stream, &rows), n, kk),
            &unpadded(&expected_rows, n, kk),
            &format!("{what} rows"),
        );
    }
}

#[test]
fn hc_combine_rows_equal_the_quantizer() {
    const HIDDEN: usize = 2560;
    const STREAMS: usize = 4;
    const PARTS: usize = 3;
    let hc = HIDDEN * STREAMS;
    let stream = Stream::new().unwrap();
    let mut seed = 5;
    let gamma: Vec<f32> = (0..hc).map(|_| 1.0 + uniform(&mut seed, 0.5)).collect();
    let gamma = upload(&stream, &gamma);
    // Sums of 32 values with heavy cancellation need many blocks to show a
    // contraction: ~110,000 blocks.
    for (round, n) in (0..8).flat_map(|r| (1..=8).chain([15]).map(move |n| (r, n))) {
        let res: Vec<f32> = (0..n * hc).map(|_| uniform(&mut seed, 4.0)).collect();
        let block_out: Vec<f32> = (0..n * HIDDEN).map(|_| uniform(&mut seed, 2.0)).collect();
        let inject: Vec<f32> = (0..n * STREAMS * PARTS)
            .map(|_| uniform(&mut seed, 3.0))
            .collect();
        let (block_out, inject) = (upload(&stream, &block_out), upload(&stream, &inject));

        let mut reference_res = upload(&stream, &res);
        let mut reference_xn = stream.alloc::<f32>(n * hc).unwrap();
        k::hc_combine(
            &stream,
            reference_res.as_mut_slice(),
            block_out.as_slice(),
            inject.as_slice(),
            PARTS as u32,
            Some(gamma.as_slice()),
            reference_xn.as_mut_slice(),
            n as u32,
            HIDDEN as u32,
            STREAMS as u32,
            1e-6,
            false,
        )
        .unwrap();
        let expected_rows = quantized(&stream, &reference_xn, n, hc);

        let mut own_res = upload(&stream, &res);
        let mut own_xn = stream.alloc::<f32>(n * hc).unwrap();
        let mut rows = stream.alloc::<u8>(k::q8_1_bytes(n, hc)).unwrap();
        stream.fill_bytes(rows.as_mut_slice(), 0x5A).unwrap();
        let launched = k::hc_combine_q8_1(
            &stream,
            own_res.as_mut_slice(),
            block_out.as_slice(),
            inject.as_slice(),
            PARTS as u32,
            gamma.as_slice(),
            own_xn.as_mut_slice(),
            rows.as_mut_slice(),
            n as u32,
            HIDDEN as u32,
            STREAMS as u32,
            1e-6,
        )
        .unwrap();
        assert_eq!(launched, Launch::Launched, "n {n}");
        let what = format!("hc_combine round {round} n {n}");
        assert_same(
            &bits(&download(&stream, &own_res)),
            &bits(&download(&stream, &reference_res)),
            &format!("{what} residual"),
        );
        assert_same(
            &bits(&download(&stream, &own_xn)),
            &bits(&download(&stream, &reference_xn)),
            &format!("{what} norm"),
        );
        assert_same(
            &download(&stream, &rows),
            &expected_rows,
            &format!("{what} rows"),
        );
    }
}

#[test]
fn hc_combine_rows_refuse_padding() {
    let stream = Stream::new().unwrap();
    let call = |n: usize, hidden: usize, streams: usize| {
        let hc = hidden * streams;
        let mut res = stream.alloc::<f32>(n * hc).unwrap();
        let block_out = stream.alloc::<f32>(n * hidden).unwrap();
        let inject = stream.alloc::<f32>(n * streams).unwrap();
        let gamma = stream.alloc::<f32>(hc).unwrap();
        let mut xn = stream.alloc::<f32>(n * hc).unwrap();
        let mut rows = stream.alloc::<u8>(k::q8_1_bytes(n, hc)).unwrap();
        k::hc_combine_q8_1(
            &stream,
            res.as_mut_slice(),
            block_out.as_slice(),
            inject.as_slice(),
            1,
            gamma.as_slice(),
            xn.as_mut_slice(),
            rows.as_mut_slice(),
            n as u32,
            hidden as u32,
            streams as u32,
            1e-6,
        )
        .unwrap()
    };
    // 3 x 96 values are padded to 512.
    assert_eq!(call(16, 2560, 4), Launch::Launched);
    assert_eq!(call(2, 96, 3), Launch::Unsupported);
    assert_eq!(call(2, 128, 4), Launch::Launched);
}

fn random(seed: &mut u32, count: usize, scale: f32) -> Vec<f32> {
    (0..count).map(|_| uniform(seed, scale)).collect()
}

#[test]
fn sigmoid_mul_rows_equal_the_quantizer() {
    let stream = Stream::new().unwrap();
    let shapes = [(1, 4096), (8, 4096), (3, 544), (5, 320)];
    for (round, (n, kk)) in (0..8).flat_map(|r| shapes.map(|s| (r, s))) {
        let mut seed = 31 + kk as u32 + 1000 * round;
        let x = random(&mut seed, n * kk, 4.0);
        let g = upload(&stream, &random(&mut seed, n * kk, 6.0));
        let mut reference = upload(&stream, &x);
        k::sigmoid_mul(&stream, reference.as_mut_slice(), g.as_slice(), n * kk).unwrap();
        let expected_rows = quantized(&stream, &reference, n, kk);

        let mut own = upload(&stream, &x);
        let mut rows = stream.alloc::<u8>(k::q8_1_bytes(n, kk)).unwrap();
        stream.fill_bytes(rows.as_mut_slice(), 0x5A).unwrap();
        let launched = k::sigmoid_mul_q8_1(
            &stream,
            own.as_mut_slice(),
            g.as_slice(),
            rows.as_mut_slice(),
            n as u32,
            kk as u32,
        )
        .unwrap();
        if kk % 512 != 0 {
            // Padded rows: the plain gate and the quantizer instead.
            assert_eq!(launched, Launch::Unsupported, "n {n} k {kk}");
            continue;
        }
        assert_eq!(launched, Launch::Launched);
        let what = format!("sigmoid_mul round {round} n {n} k {kk}");
        assert_same(
            &bits(&download(&stream, &own)),
            &bits(&download(&stream, &reference)),
            &format!("{what} values"),
        );
        assert_same(
            &download(&stream, &rows),
            &expected_rows,
            &format!("{what} rows"),
        );
    }
}

#[test]
fn hc_mix_epilogue_rows_equal_the_quantizer() {
    let stream = Stream::new().unwrap();
    let mut seed = 41;
    for (round, (hidden, streams)) in
        (0..6).flat_map(|r| [(2560usize, 4usize), (512, 8), (1024, 1)].map(|s| (r, s)))
    {
        let hc = hidden * streams;
        let parts = hidden.div_ceil(256);
        let inject_w = upload(&stream, &random(&mut seed, streams * hc, 0.05));
        for n in [1usize, 2, 5, 8] {
            let xn = upload(&stream, &random(&mut seed, n * hc, 3.0));
            let gate = upload(&stream, &random(&mut seed, n * hc, 4.0));
            for with_inject in [false, true] {
                let mut reference = stream.alloc::<f32>(n * hidden).unwrap();
                let mut reference_inject = stream.alloc::<f32>(n * streams * parts).unwrap();
                let mut own = stream.alloc::<f32>(n * hidden).unwrap();
                let mut own_inject = stream.alloc::<f32>(n * streams * parts).unwrap();
                let mut rows = stream.alloc::<u8>(k::q8_1_bytes(n, hidden)).unwrap();
                stream.fill_bytes(rows.as_mut_slice(), 0x5A).unwrap();
                let (h, s, t) = (hidden as u32, streams as u32, n as u32);
                let launched = if with_inject {
                    k::hc_mix_epilogue_inject(
                        &stream,
                        xn.as_slice(),
                        gate.as_slice(),
                        inject_w.as_slice(),
                        reference.as_mut_slice(),
                        reference_inject.as_mut_slice(),
                        t,
                        h,
                        s,
                    )
                    .unwrap();
                    k::hc_mix_epilogue_inject_q8_1(
                        &stream,
                        xn.as_slice(),
                        gate.as_slice(),
                        inject_w.as_slice(),
                        own.as_mut_slice(),
                        own_inject.as_mut_slice(),
                        rows.as_mut_slice(),
                        t,
                        h,
                        s,
                    )
                    .unwrap()
                } else {
                    k::hc_mix_epilogue(
                        &stream,
                        xn.as_slice(),
                        gate.as_slice(),
                        reference.as_mut_slice(),
                        t,
                        h,
                        s,
                    )
                    .unwrap();
                    k::hc_mix_epilogue_q8_1(
                        &stream,
                        xn.as_slice(),
                        gate.as_slice(),
                        own.as_mut_slice(),
                        rows.as_mut_slice(),
                        t,
                        h,
                        s,
                    )
                    .unwrap()
                };
                assert_eq!(launched, Launch::Launched);
                let what = format!(
                    "hc_mix round {round} h {hidden} s {streams} n {n} inject {with_inject}"
                );
                assert_same(
                    &bits(&download(&stream, &own)),
                    &bits(&download(&stream, &reference)),
                    &format!("{what} mixed"),
                );
                if with_inject {
                    assert_same(
                        &bits(&download(&stream, &own_inject)),
                        &bits(&download(&stream, &reference_inject)),
                        &format!("{what} inject"),
                    );
                }
                let expected_rows = quantized(&stream, &reference, n, hidden);
                assert_same(
                    &download(&stream, &rows),
                    &expected_rows,
                    &format!("{what} rows"),
                );
            }
        }
    }
}

#[test]
fn gated_delta_net_rows_equal_the_quantizer() {
    const K_HEADS: usize = 16;
    const V_HEADS: usize = 48;
    const D: usize = 128;
    const KERNEL: usize = 4;
    let channels = 2 * K_HEADS * D + V_HEADS * D;
    let value = V_HEADS * D;
    let stride = channels + value;
    let stream = Stream::new().unwrap();
    let mut seed = 53;
    let conv_w = upload(&stream, &random(&mut seed, channels * KERNEL, 0.5));
    let a = upload(&stream, &random(&mut seed, V_HEADS, 1.0));
    let dt = upload(&stream, &random(&mut seed, V_HEADS, 1.0));
    let norm_w: Vec<f32> = (0..D).map(|_| 1.0 + uniform(&mut seed, 0.3)).collect();
    let norm_w = upload(&stream, &norm_w);
    for (round, n) in (0..4).flat_map(|r| [1usize, 2, 3, 5].map(move |n| (r, n))) {
        let qkvz = upload(&stream, &random(&mut seed, n * stride, 1.0));
        let alpha_beta = upload(&stream, &random(&mut seed, n * 2 * V_HEADS, 2.0));
        let conv_state = random(&mut seed, (KERNEL - 1) * channels, 1.0);
        let state = random(&mut seed, V_HEADS * D * D, 0.1);
        let run = |rows: Option<&mut DeviceBuffer<u8>>| {
            let mut conv = upload(&stream, &conv_state);
            let mut st = upload(&stream, &state);
            let mut scratch = stream.alloc::<f32>((n + KERNEL - 1) * channels).unwrap();
            let mut qn = stream.alloc::<f32>(n * K_HEADS * D).unwrap();
            let mut kn = stream.alloc::<f32>(n * K_HEADS * D).unwrap();
            let mut raw = stream.alloc::<f32>(n * value).unwrap();
            let mut out = stream.alloc::<f32>(n * value).unwrap();
            let (t, kh, vh) = (n as u32, K_HEADS as u32, V_HEADS as u32);
            let launched = match rows {
                Some(rows) => k::gated_delta_net_q8_1(
                    &stream,
                    qkvz.as_slice(),
                    stride as u32,
                    qkvz.slice(channels..qkvz.len()),
                    stride as u32,
                    alpha_beta.as_slice(),
                    conv_w.as_slice(),
                    a.as_slice(),
                    dt.as_slice(),
                    norm_w.as_slice(),
                    conv.as_mut_slice(),
                    scratch.as_mut_slice(),
                    qn.as_mut_slice(),
                    kn.as_mut_slice(),
                    raw.as_mut_slice(),
                    st.as_mut_slice(),
                    out.as_mut_slice(),
                    rows.as_mut_slice(),
                    None,
                    None,
                    &mut [],
                    &mut [],
                    0,
                    t,
                    kh,
                    vh,
                    KERNEL as u32,
                    false,
                    false,
                    1e-6,
                    None,
                    None,
                    0,
                ),
                None => k::gated_delta_net(
                    &stream,
                    qkvz.as_slice(),
                    stride as u32,
                    qkvz.slice(channels..qkvz.len()),
                    stride as u32,
                    alpha_beta.as_slice(),
                    conv_w.as_slice(),
                    a.as_slice(),
                    dt.as_slice(),
                    norm_w.as_slice(),
                    conv.as_mut_slice(),
                    scratch.as_mut_slice(),
                    qn.as_mut_slice(),
                    kn.as_mut_slice(),
                    raw.as_mut_slice(),
                    st.as_mut_slice(),
                    out.as_mut_slice(),
                    None,
                    None,
                    &mut [],
                    &mut [],
                    0,
                    t,
                    kh,
                    vh,
                    KERNEL as u32,
                    false,
                    false,
                    1e-6,
                    None,
                    None,
                    0,
                ),
            }
            .unwrap();
            assert_eq!(launched, Launch::Launched);
            let floats = |b: &DeviceBuffer<f32>| bits(&download(&stream, b));
            (floats(&out), floats(&st), floats(&conv), out)
        };
        let (out, state_after, conv_after, reference_out) = run(None);
        let values: Vec<f32> = out.iter().map(|&b| f32::from_bits(b)).collect();
        assert!(
            values.iter().all(|v| v.is_finite()) && values.iter().any(|&v| v != 0.0),
            "gated_delta_net round {round} n {n}: degenerate output"
        );
        let mut rows = stream.alloc::<u8>(k::q8_1_bytes(n, value)).unwrap();
        stream.fill_bytes(rows.as_mut_slice(), 0x5A).unwrap();
        let (own_out, own_state, own_conv, _) = run(Some(&mut rows));
        let what = format!("gated_delta_net round {round} n {n}");
        assert_same(&own_out, &out, &format!("{what} out"));
        assert_same(&own_state, &state_after, &format!("{what} state"));
        assert_same(&own_conv, &conv_after, &format!("{what} conv state"));
        let expected_rows = quantized(&stream, &reference_out, n, value);
        assert_same(
            &download(&stream, &rows),
            &expected_rows,
            &format!("{what} rows"),
        );
    }
}
