//! The matrix-core products and the weight tiles against the host
//! (`tests/products_test.cpp`): random GGML Q8_0 blocks and bf16 activations
//! at VoxCPM2's shapes and odd ones, every row count up to 32, products
//! compared with a double-precision sum of the same operands; and
//! [`Bf16Gemm::linear`] on tiles, which chunks up to 128 rows and expands
//! the tiles for hipBLASLt beyond.
//!
//! cargo test --profile gpu-test -p gufo-voxcpm --features gpu --test products

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{Bf16, DeviceBuffer, DeviceCopy, Launch, Stream};
use gufo_voxcpm::gemm::{Bf16Gemm, plan_wmma};
use gufo_voxcpm::matrix::{MatrixView, WeightFormat, bf16_tiles_bytes, q8_tiles_bytes};
use gufo_voxcpm_kernels as k;

/// splitmix64; the operands only need to be varied and reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[low, high]`.
    fn int(&mut self, low: i64, high: i64) -> i64 {
        low + (self.next() % (high - low + 1) as u64) as i64
    }

    /// Uniform in `(0, 1]`.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Normal with mean 0 (Box-Muller).
    fn normal(&mut self, sigma: f64) -> f32 {
        let (u, v) = (self.unit(), self.unit());
        (sigma * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Round to nearest even, as the test's C++ `FloatToBf16`.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn half_to_f32(bits: u16) -> f64 {
    let sign = (bits >> 15) & 1;
    let exponent = i32::from((bits >> 10) & 0x1F);
    let mantissa = f64::from(bits & 0x3FF);
    let value = if exponent == 0 {
        mantissa * 2f64.powi(-24)
    } else {
        (mantissa + 1024.0) * 2f64.powi(exponent - 25)
    };
    if sign != 0 { -value } else { value }
}

/// Random Q8_0 weights `[out][in]` as GGML blocks, and their values.
struct Q8Weights {
    blocks: Vec<u8>,
    values: Vec<f64>,
}

fn make_weights(out: usize, input: usize, rng: &mut Rng) -> Q8Weights {
    let blocks = out * input / 32;
    let mut weights = Q8Weights {
        blocks: vec![0; blocks * 34],
        values: vec![0.0; out * input],
    };
    for block in 0..blocks {
        let target = &mut weights.blocks[block * 34..(block + 1) * 34];
        // Scales around 1e-3, as trained weights have them (float16 normals).
        let bits = rng.int(0x1000, 0x1C00) as u16;
        target[..2].copy_from_slice(&bits.to_le_bytes());
        let scale = half_to_f32(bits);
        for i in 0..32 {
            let q = rng.int(-128, 127);
            target[2 + i] = q as i8 as u8;
            weights.values[block * 32 + i] = scale * q as f64;
        }
    }
    weights
}

fn make_activations(rows: usize, input: usize, rng: &mut Rng) -> Vec<Bf16> {
    (0..rows * input)
        .map(|_| Bf16(f32_to_bf16(rng.normal(2.0))))
        .collect()
}

fn upload<T: DeviceCopy>(stream: &Stream, host: &[T]) -> DeviceBuffer<T> {
    let mut device = stream.alloc(host.len().max(1)).unwrap();
    stream
        .upload(host, device.slice_mut(0..host.len()))
        .unwrap();
    device
}

fn download<T: DeviceCopy + Default>(
    stream: &Stream,
    device: &DeviceBuffer<T>,
    count: usize,
) -> Vec<T> {
    let mut host = vec![T::default(); count];
    stream.download(device.slice(0..count), &mut host).unwrap();
    host
}

/// Compares `y` with the exact product; the limit scales with
/// `sum |x||w|` of each output (float32 sums of exact products).
#[allow(clippy::too_many_arguments)]
fn compare(
    name: &str,
    y: &[f32],
    x: &[Bf16],
    w: &[f64],
    before: Option<&[f32]>,
    rows: usize,
    input: usize,
    out: usize,
    relative: f64,
) -> bool {
    let mut worst = 0.0f64;
    for m in 0..rows {
        for n in 0..out {
            let mut exact = before.map_or(0.0, |b| f64::from(b[m * out + n]));
            let mut magnitude = 0.0f64;
            for kk in 0..input {
                let product = f64::from(bf16_to_f32(x[m * input + kk].0)) * w[n * input + kk];
                exact += product;
                magnitude += product.abs();
            }
            let error = (f64::from(y[m * out + n]) - exact).abs() / magnitude.max(1e-30);
            worst = worst.max(error);
        }
    }
    let ok = worst <= relative;
    println!(
        "{} {name} rows {rows} in {input} out {out}: worst {worst:.3e} (limit {relative:.3e})",
        if ok { "ok  " } else { "FAIL" }
    );
    ok
}

/// The weights as Q8_0 tiles, dequantized bf16 rows and bf16 tiles.
struct Tiles {
    q8: DeviceBuffer<u8>,
    dequantized: DeviceBuffer<Bf16>,
    bf16: DeviceBuffer<Bf16>,
}

fn tiles(stream: &Stream, weights: &Q8Weights, out: usize, input: usize) -> Tiles {
    let blocks = upload(stream, &weights.blocks);
    let mut q8 = stream.alloc::<u8>(q8_tiles_bytes(out, input)).unwrap();
    stream.fill_bytes(q8.as_mut_slice(), 0).unwrap();
    k::q8_0_to_tiles(
        stream,
        blocks.slice(0..weights.blocks.len()),
        out,
        input,
        q8.as_mut_slice(),
    )
    .unwrap();
    let mut dequantized = stream.alloc::<Bf16>(out * input).unwrap();
    k::q8_tiles_to_bf16(
        stream,
        q8.as_slice(),
        out,
        input,
        dequantized.as_mut_slice(),
    )
    .unwrap();
    let mut bf16 = stream
        .alloc::<Bf16>(bf16_tiles_bytes(out, input) / 2)
        .unwrap();
    stream.fill_bytes(bf16.as_mut_slice(), 0).unwrap();
    k::bf16_to_tiles(
        stream,
        dequantized.as_slice(),
        out,
        input,
        bf16.as_mut_slice(),
    )
    .unwrap();
    Tiles {
        q8,
        dequantized,
        bf16,
    }
}

fn test_shape(
    stream: &Stream,
    input: usize,
    out: usize,
    row_counts: &[usize],
    rng: &mut Rng,
) -> usize {
    let mut failures = 0;
    let weights = make_weights(out, input, rng);
    let t = tiles(stream, &weights, out, input);
    let mut unpacked = stream.alloc::<Bf16>(out * input).unwrap();
    k::bf16_tiles_to_rows(
        stream,
        t.bf16.as_slice(),
        out,
        input,
        unpacked.as_mut_slice(),
    )
    .unwrap();
    let host_bf16 = download(stream, &t.dequantized, out * input);
    let host_unpacked = download(stream, &unpacked, out * input);
    let dequant_ok = host_bf16
        .iter()
        .zip(&weights.values)
        .all(|(&value, &exact)| value.0 == f32_to_bf16(exact as f32));
    println!(
        "{} dequantization in {input} out {out}",
        if dequant_ok { "ok  " } else { "FAIL" }
    );
    failures += usize::from(!dequant_ok);
    let tiles_ok = host_unpacked == host_bf16;
    println!(
        "{} bf16 tiles round trip in {input} out {out}",
        if tiles_ok { "ok  " } else { "FAIL" }
    );
    failures += usize::from(!tiles_ok);
    let bf16_values: Vec<f64> = host_bf16
        .iter()
        .map(|v| f64::from(bf16_to_f32(v.0)))
        .collect();

    let mut partials = stream.alloc::<f32>(32 * 64 * out).unwrap();
    let mut counters = stream.alloc::<u32>(out.div_ceil(16)).unwrap();
    stream.fill_bytes(counters.as_mut_slice(), 0).unwrap();
    for &rows in row_counts {
        let x = make_activations(rows, input, rng);
        let x_device = upload(stream, &x);
        let before: Vec<f32> = (0..rows * out).map(|_| rng.normal(1.0)).collect();
        let mut y = stream.alloc::<f32>(rows * out).unwrap();
        for accumulate in [false, true] {
            for waves in [1, 0, 4096] {
                let plan = plan_wmma(rows, input, out, waves);
                for q8 in [true, false] {
                    let name = format!(
                        "{} wmma ({} splits){}",
                        if q8 { "q8" } else { "bf16" },
                        plan.splits,
                        if accumulate { " (accumulate)" } else { "" }
                    );
                    stream.upload(&before, y.as_mut_slice()).unwrap();
                    let (partial_floats, counter_count) = (partials.len(), counters.len());
                    let launch = if q8 {
                        k::wmma_q8(
                            stream,
                            x_device.slice(0..rows * input),
                            t.q8.as_slice(),
                            y.as_mut_slice(),
                            rows,
                            input,
                            out,
                            accumulate,
                            plan.splits,
                            plan.blocks_per_split,
                            Some(partials.as_mut_slice()),
                            partial_floats,
                            Some(counters.as_mut_slice()),
                            counter_count,
                        )
                    } else {
                        k::wmma_bf16(
                            stream,
                            x_device.slice(0..rows * input),
                            t.bf16.as_slice(),
                            y.as_mut_slice(),
                            rows,
                            input,
                            out,
                            accumulate,
                            plan.splits,
                            plan.blocks_per_split,
                            Some(partials.as_mut_slice()),
                            partial_floats,
                            Some(counters.as_mut_slice()),
                            counter_count,
                        )
                    }
                    .unwrap();
                    if launch != Launch::Launched {
                        println!("FAIL {name} rows {rows} in {input} out {out}: not launched");
                        failures += 1;
                        continue;
                    }
                    stream.synchronize().unwrap();
                    let result = download(stream, &y, rows * out);
                    let w = if q8 { &weights.values } else { &bf16_values };
                    let ok = compare(
                        &name,
                        &result,
                        &x,
                        w,
                        accumulate.then_some(&before[..]),
                        rows,
                        input,
                        out,
                        1e-5,
                    );
                    failures += usize::from(!ok);
                }
            }
        }
    }
    if download(stream, &counters, counters.len())
        .iter()
        .any(|&value| value != 0)
    {
        println!("FAIL split counters not reset");
        failures += 1;
    }
    failures
}

/// [`Bf16Gemm::linear`] on Q8_0 and bf16 tiles: its split scratch, the
/// chunks of up to 128 rows and the expansion for hipBLASLt beyond.
fn test_linear(stream: &Stream, input: usize, out: usize, rng: &mut Rng) -> usize {
    let mut failures = 0;
    let weights = make_weights(out, input, rng);
    let t = tiles(stream, &weights, out, input);
    let host_bf16 = download(stream, &t.dequantized, out * input);
    let bf16_values: Vec<f64> = host_bf16
        .iter()
        .map(|v| f64::from(bf16_to_f32(v.0)))
        .collect();
    let mut gemm = Bf16Gemm::new(stream).unwrap();
    for rows in [1, 22, 40, 130] {
        let x = make_activations(rows, input, rng);
        let x_device = upload(stream, &x);
        let before: Vec<f32> = (0..rows * out).map(|_| rng.normal(1.0)).collect();
        let mut y = stream.alloc::<f32>(rows * out).unwrap();
        for accumulate in [false, true] {
            for q8 in [true, false] {
                let matrix = if q8 {
                    MatrixView {
                        format: WeightFormat::Q8Tiles,
                        data: t.q8.as_slice(),
                    }
                } else {
                    MatrixView {
                        format: WeightFormat::Bf16Tiles,
                        data: t.bf16.as_slice().cast::<u8>().unwrap(),
                    }
                };
                stream.upload(&before, y.as_mut_slice()).unwrap();
                gemm.linear(
                    stream,
                    x_device.slice(0..rows * input),
                    matrix,
                    y.as_mut_slice(),
                    rows,
                    input,
                    out,
                    accumulate,
                )
                .unwrap();
                stream.synchronize().unwrap();
                // Beyond 128 rows Q8_0 is expanded to bf16(scale * q) for
                // hipBLASLt, whose reduction order differs.
                let expanded = rows > 128;
                let name = format!(
                    "linear {} tiles{}",
                    if q8 { "q8" } else { "bf16" },
                    if accumulate { " (accumulate)" } else { "" }
                );
                let ok = compare(
                    &name,
                    &download(stream, &y, rows * out),
                    &x,
                    if q8 && !expanded {
                        &weights.values
                    } else {
                        &bf16_values
                    },
                    accumulate.then_some(&before[..]),
                    rows,
                    input,
                    out,
                    if expanded { 1e-4 } else { 1e-5 },
                );
                failures += usize::from(!ok);
            }
        }
    }
    failures
}

#[test]
fn products_match_a_double_precision_sum() {
    let stream = Stream::new().unwrap();
    let mut rng = Rng(1234);
    let all = [1, 2, 4, 5, 8, 11, 16, 22, 32];
    let mut failures = 0;
    // VoxCPM2's shapes (feature and FSQ inputs, local stacks, LM, stop
    // head), odd row counts.
    failures += test_shape(&stream, 64, 1024, &[4, 8], &mut rng);
    failures += test_shape(&stream, 512, 1024, &[1, 4], &mut rng);
    failures += test_shape(&stream, 1024, 2560, &all, &mut rng);
    failures += test_shape(&stream, 2048, 1024, &[1, 5, 22], &mut rng);
    failures += test_shape(&stream, 4096, 1024, &[1, 5, 22], &mut rng);
    failures += test_shape(&stream, 2048, 2048, &[1, 5], &mut rng);
    failures += test_shape(&stream, 6144, 2048, &[1, 3], &mut rng);
    failures += test_shape(&stream, 8192, 512, &[1, 22], &mut rng);
    failures += test_shape(&stream, 1024, 72, &[3, 22], &mut rng); // rows not a multiple of 16
    failures += test_shape(&stream, 2048, 2, &[1], &mut rng);
    failures += test_linear(&stream, 1024, 2560, &mut rng);
    failures += test_linear(&stream, 4096, 1024, &mut rng);
    assert_eq!(failures, 0, "{failures} failures");
}
