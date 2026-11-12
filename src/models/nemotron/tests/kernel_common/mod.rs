//! Helpers of the kernel ops tests: device buffers from host values, a
//! SplitMix64 generator (the C++ tests' `Rng`), error measures and FP16/BF16
//! rounding.

#![allow(dead_code)]

use gufo_hip::{Bf16, DeviceBuffer, DeviceCopy, DeviceIndices, F16, Launch, LaunchError, Stream};

/// SplitMix64, as the C++ vector product tests seed it.
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

    pub fn random(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }

    pub fn normals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * scale).collect()
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
    let mut buffer = s.alloc::<T>(n.max(1)).expect("alloc");
    s.fill_bytes(buffer.as_mut_slice(), 0).expect("fill");
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

/// Device indices below `bound` holding `values`.
pub fn indices(s: &Stream, values: &[i32], bound: u32) -> DeviceIndices {
    let mut buffer = s
        .alloc_indices(values.len().max(1), bound)
        .expect("indices");
    let host: Vec<u32> = values.iter().map(|&v| v as u32).collect();
    s.upload_indices(&host, &mut buffer)
        .expect("upload indices");
    buffer
}

/// `n` device indices below `bound` for a kernel to write.
pub fn index_out(s: &Stream, n: usize, bound: u32) -> DeviceIndices {
    s.alloc_indices(n.max(1), bound).expect("indices")
}

pub fn down_indices(s: &Stream, buffer: &DeviceIndices, n: usize) -> Vec<i32> {
    s.synchronize().expect("sync");
    let mut host = vec![0u32; n];
    s.download_blocking(buffer.view().values().range(0..n), &mut host)
        .expect("download indices");
    host.into_iter().map(|v| v as i32).collect()
}

/// A launch the kernel accepted.
pub fn ok(result: Result<Launch, LaunchError>, what: &str) {
    match result {
        Ok(Launch::Launched) => {}
        other => panic!("{what}: {other:?}"),
    }
}

/// Max |a - b| relative to max |b|.
pub fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let mut err = 0.0f64;
    let mut scale = 1e-12f64;
    for (x, y) in a.iter().zip(b) {
        err = err.max((f64::from(*x) - f64::from(*y)).abs());
        scale = scale.max(f64::from(*y).abs());
    }
    err / scale
}

pub fn half(x: f32) -> F16 {
    F16::from_f32(x)
}

pub fn half_round(x: f32) -> f32 {
    F16::from_f32(x).to_f32()
}

/// BF16 of `x`, round to nearest even (finite values).
pub fn bf16(x: f32) -> Bf16 {
    Bf16::from_f32(x)
}

pub fn bf16_round(x: f32) -> f32 {
    Bf16::from_f32(x).to_f32()
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}
