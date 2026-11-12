//! Helpers of the kernel ops tests: device buffers from host values, a
//! SplitMix64 generator (normal and uniform values like the C++ tests'
//! `std::mt19937` distributions; the values differ, the checks do not),
//! binary16 rounding, and the GGML block formats the kernels read: random
//! blocks with bounded scales and their CPU dequantization (ggml-quants.c's
//! `dequantize_row_*`).

#![allow(dead_code)]

use gufo_hip::{
    DeviceBuffer, DeviceCopy, DeviceIndices, F16, GgmlType, IndexSpec, Launch, LaunchError,
    QuantSlice, Stream,
};

/// SplitMix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit() as f32
    }

    pub fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    pub fn normals(&mut self, n: usize, sigma: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * sigma).collect()
    }

    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            values.swap(i, j);
        }
    }
}

pub fn stream() -> Stream {
    Stream::new().expect("stream")
}

/// A device copy of `host` (at least one element).
pub fn up<T: DeviceCopy>(s: &Stream, host: &[T]) -> DeviceBuffer<T> {
    let mut buffer = s.alloc::<T>(host.len().max(1)).expect("alloc");
    if !host.is_empty() {
        s.upload(host, buffer.slice_mut(0..host.len()))
            .expect("upload");
    }
    buffer
}

/// `n` zeroed elements.
pub fn zeros<T: DeviceCopy>(s: &Stream, n: usize) -> DeviceBuffer<T> {
    filled(s, n, 0)
}

/// `n` elements of the byte `byte`.
pub fn filled<T: DeviceCopy>(s: &Stream, n: usize, byte: u8) -> DeviceBuffer<T> {
    let mut buffer = s.alloc::<T>(n.max(1)).expect("alloc");
    s.fill_bytes(buffer.as_mut_slice(), byte).expect("fill");
    buffer
}

pub fn down<T: DeviceCopy + Default + Clone>(
    s: &Stream,
    buffer: &DeviceBuffer<T>,
    n: usize,
) -> Vec<T> {
    s.synchronize().expect("sync");
    let mut host = vec![T::default(); n];
    s.download_blocking(buffer.slice(0..n), &mut host)
        .expect("download");
    host
}

/// Device indices holding `values` under `spec`.
pub fn indices_with(s: &Stream, values: &[u32], spec: IndexSpec) -> DeviceIndices {
    let mut buffer = s
        .alloc_indices_with(values.len().max(1), spec)
        .expect("indices");
    let mut host = values.to_vec();
    if host.is_empty() {
        host.push(0);
    }
    s.upload_indices(&host, &mut buffer)
        .expect("upload indices");
    buffer
}

/// Signed (`int32`) device indices below `bound`.
pub fn signed_indices(s: &Stream, values: &[i32], bound: u32) -> DeviceIndices {
    let host: Vec<u32> = values.iter().map(|&v| v as u32).collect();
    indices_with(s, &host, IndexSpec::below(bound).signed())
}

/// `n` device indices for a kernel to write under `spec`.
pub fn index_out(s: &Stream, n: usize, spec: IndexSpec) -> DeviceIndices {
    s.alloc_indices_with(n.max(1), spec).expect("indices")
}

pub fn down_indices(s: &Stream, buffer: &DeviceIndices, n: usize) -> Vec<u32> {
    s.synchronize().expect("sync");
    let mut host = vec![0u32; n];
    s.download_blocking(buffer.view().values().range(0..n), &mut host)
        .expect("download indices");
    host
}

/// Quantized weights of `format` on the device.
pub fn quant<'a>(buffer: &'a DeviceBuffer<u8>, format: GgmlType) -> QuantSlice<'a> {
    QuantSlice::new(buffer.as_slice(), format)
}

/// The result of a `void` or a `supported` launcher.
pub trait Accepted {
    fn accepted(self, what: &str);
}

impl Accepted for Result<(), LaunchError> {
    fn accepted(self, what: &str) {
        if let Err(error) = self {
            panic!("{what}: {error:?}");
        }
    }
}

impl Accepted for Result<Launch, LaunchError> {
    fn accepted(self, what: &str) {
        match self {
            Ok(Launch::Launched) => {}
            other => panic!("{what}: {other:?}"),
        }
    }
}

/// A launch the kernel accepted.
pub fn ok(result: impl Accepted, what: &str) {
    result.accepted(what);
}

pub fn half(x: f32) -> F16 {
    F16::from_f32(x)
}

pub fn from_half(h: F16) -> f32 {
    h.to_f32()
}

/// Binary16 bits of `x` (round to nearest even).
pub fn half_bits(x: f32) -> u16 {
    F16::from_f32(x).0
}

pub fn half_from_bits(bits: u16) -> f32 {
    F16(bits).to_f32()
}

/// Bitwise equality of two float slices.
pub fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

pub fn same_halves(a: &[F16], b: &[F16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.0 == y.0)
}

pub fn gelu_tanh(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044715 * x * x * x)).tanh())
}

// --- GGML block formats -------------------------------------------------------

/// A block format: values and bytes per block and its binary16 scale fields.
#[derive(Clone, Copy)]
pub struct Format {
    pub ty: GgmlType,
    pub block: usize,
    pub bytes: usize,
    pub half_offsets: &'static [usize],
    pub name: &'static str,
}

pub const Q4_K: Format = Format {
    ty: GgmlType::Q4K,
    block: 256,
    bytes: 144,
    half_offsets: &[0, 2],
    name: "Q4_K",
};
pub const Q5_K: Format = Format {
    ty: GgmlType::Q5K,
    block: 256,
    bytes: 176,
    half_offsets: &[0, 2],
    name: "Q5_K",
};
pub const Q6_K: Format = Format {
    ty: GgmlType::Q6K,
    block: 256,
    bytes: 210,
    half_offsets: &[208],
    name: "Q6_K",
};
pub const Q8_0: Format = Format {
    ty: GgmlType::Q8_0,
    block: 32,
    bytes: 34,
    half_offsets: &[0],
    name: "Q8_0",
};
pub const Q5_1: Format = Format {
    ty: GgmlType::Q5_1,
    block: 32,
    bytes: 24,
    half_offsets: &[0, 2],
    name: "Q5_1",
};
pub const Q4_0: Format = Format {
    ty: GgmlType::Q4_0,
    block: 32,
    bytes: 18,
    half_offsets: &[0],
    name: "Q4_0",
};
pub const Q5_0: Format = Format {
    ty: GgmlType::Q5_0,
    block: 32,
    bytes: 22,
    half_offsets: &[0],
    name: "Q5_0",
};
/// Binary16 weights, two bytes per value.
pub const F16_FORMAT: Format = Format {
    ty: GgmlType::F16,
    block: 1,
    bytes: 2,
    half_offsets: &[],
    name: "F16",
};

impl Format {
    pub fn row_bytes(&self, cols: usize) -> usize {
        cols / self.block * self.bytes
    }
}

/// Random blocks with bounded, finite binary16 scales (binary16 weights:
/// trained-weight magnitudes). 4096 spare bytes follow, as the C++ tests
/// allocate them.
pub fn random_matrix(f: &Format, rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let blocks = rows * cols / f.block;
    let mut data = vec![0u8; blocks * f.bytes + 4096];
    if f.ty == GgmlType::F16 {
        for i in 0..rows * cols {
            let h = half_bits(rng.normal() * 0.02);
            data[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
        }
        return data;
    }
    for byte in &mut data[..blocks * f.bytes] {
        *byte = rng.below(256) as u8;
    }
    for b in 0..blocks {
        for &off in f.half_offsets {
            let h = half_bits(rng.uniform(0.0005, 0.004));
            data[b * f.bytes + off..b * f.bytes + off + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
    data
}

fn f16_at(bytes: &[u8], at: usize) -> f32 {
    half_from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `out.len()` values of the row at `src` (whole blocks of `f`).
pub fn dequantize_row(f: &Format, src: &[u8], out: &mut [f32]) {
    let blocks = out.len() / f.block;
    for b in 0..blocks {
        let x = &src[b * f.bytes..(b + 1) * f.bytes];
        let y = &mut out[b * f.block..(b + 1) * f.block];
        match f.ty {
            GgmlType::F16 => y[0] = f16_at(x, 0),
            GgmlType::Q8_0 => {
                let d = f16_at(x, 0);
                for (i, v) in y.iter_mut().enumerate() {
                    *v = d * f32::from(x[2 + i] as i8);
                }
            }
            GgmlType::Q4_0 => {
                let d = f16_at(x, 0);
                for j in 0..16 {
                    y[j] = (i32::from(x[2 + j] & 0xF) - 8) as f32 * d;
                    y[j + 16] = (i32::from(x[2 + j] >> 4) - 8) as f32 * d;
                }
            }
            GgmlType::Q5_0 | GgmlType::Q5_1 => {
                let d = f16_at(x, 0);
                let (m, qh_at, qs_at) = if f.ty == GgmlType::Q5_1 {
                    (f16_at(x, 2), 4, 8)
                } else {
                    (0.0, 2, 6)
                };
                let qh = u32::from_le_bytes([x[qh_at], x[qh_at + 1], x[qh_at + 2], x[qh_at + 3]]);
                let bias = if f.ty == GgmlType::Q5_1 { 0 } else { 16 };
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    let x0 = (u32::from(x[qs_at + j] & 0x0F) | xh0) as i32 - bias;
                    let x1 = (u32::from(x[qs_at + j] >> 4) | xh1) as i32 - bias;
                    y[j] = x0 as f32 * d + m;
                    y[j + 16] = x1 as f32 * d + m;
                }
            }
            GgmlType::Q4K => {
                let d = f16_at(x, 0);
                let min = f16_at(x, 2);
                let scales = &x[4..16];
                let qs = &x[16..144];
                for (n, chunk) in y.chunks_mut(64).enumerate() {
                    let (sc1, m1) = scale_min_k4(2 * n, scales);
                    let (sc2, m2) = scale_min_k4(2 * n + 1, scales);
                    let (d1, m1) = (d * f32::from(sc1), min * f32::from(m1));
                    let (d2, m2) = (d * f32::from(sc2), min * f32::from(m2));
                    let q = &qs[32 * n..32 * n + 32];
                    for l in 0..32 {
                        chunk[l] = d1 * f32::from(q[l] & 0xF) - m1;
                        chunk[32 + l] = d2 * f32::from(q[l] >> 4) - m2;
                    }
                }
            }
            GgmlType::Q5K => {
                let d = f16_at(x, 0);
                let min = f16_at(x, 2);
                let scales = &x[4..16];
                let qh = &x[16..48];
                let qs = &x[48..176];
                for (n, chunk) in y.chunks_mut(64).enumerate() {
                    let (sc1, m1) = scale_min_k4(2 * n, scales);
                    let (sc2, m2) = scale_min_k4(2 * n + 1, scales);
                    let (d1, m1) = (d * f32::from(sc1), min * f32::from(m1));
                    let (d2, m2) = (d * f32::from(sc2), min * f32::from(m2));
                    let (u1, u2) = (1u8 << (2 * n), 2u8 << (2 * n));
                    let q = &qs[32 * n..32 * n + 32];
                    for l in 0..32 {
                        let hi1 = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                        let hi2 = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                        chunk[l] = d1 * (f32::from(q[l] & 0xF) + hi1) - m1;
                        chunk[32 + l] = d2 * (f32::from(q[l] >> 4) + hi2) - m2;
                    }
                }
            }
            GgmlType::Q6K => {
                let d = f16_at(x, 208);
                for (n, chunk) in y.chunks_mut(128).enumerate() {
                    let ql = &x[64 * n..64 * n + 64];
                    let qh = &x[128 + 32 * n..128 + 32 * n + 32];
                    let sc = &x[192 + 8 * n..192 + 8 * n + 8];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                        let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        let s = |i: usize| f32::from(sc[i] as i8);
                        chunk[l] = d * s(is) * q1 as f32;
                        chunk[l + 32] = d * s(is + 2) * q2 as f32;
                        chunk[l + 64] = d * s(is + 4) * q3 as f32;
                        chunk[l + 96] = d * s(is + 6) * q4 as f32;
                    }
                }
            }
            other => panic!("no CPU dequantization for {other:?}"),
        }
    }
}

/// max |got - want| / sum |w x| over sampled outputs of `y = W x` against
/// FP64 dots of the dequantized rows (`x` rows of `k`).
#[allow(clippy::too_many_arguments)]
pub fn worst_dot_error(
    f: &Format,
    w: &[u8],
    k: usize,
    x_row: impl Fn(usize) -> Vec<f32>,
    got: impl Fn(usize, usize) -> f32,
    slots: impl Iterator<Item = (usize, usize)>,
    outputs: impl Iterator<Item = usize> + Clone,
) -> f64 {
    let row_bytes = f.row_bytes(k);
    let mut row = vec![0f32; k];
    let mut worst = 0f64;
    for (slot, weight_row0) in slots {
        let xs = x_row(slot);
        for o in outputs.clone() {
            let at = (weight_row0 + o) * row_bytes;
            dequantize_row(f, &w[at..at + row_bytes], &mut row);
            let (mut want, mut magnitude) = (0f64, 0f64);
            for c in 0..k {
                let p = f64::from(row[c]) * f64::from(xs[c]);
                want += p;
                magnitude += p.abs();
            }
            worst = worst.max((f64::from(got(slot, o)) - want).abs() / magnitude.max(1e-300));
        }
    }
    worst
}
