//! Scalar CPU evaluations of the trunk's operators for the reference model
//! (`cpu_ops.cpp`).

use gufo_formats::GgmlType;
use gufo_formats::dequant::{dequantize, f16_to_f32};

use crate::iq2xxs::IQ2XXS_GRID;
use crate::nvfp4::dequantize_nvfp4;
use crate::weights::{GgufShard, TensorRef};

/// llama.cpp's `ksigns_iq2xs`: seven sign bits and their parity as the
/// eighth.
fn ksign(index: u32) -> u8 {
    let bits = index as u8 & 0x7F;
    bits | (((bits.count_ones() & 1) as u8) << 7)
}

/// `quant::DequantizeIQ2_XXS`: per 256 values an F16 scale, then per 32
/// values four grid indices and seven-bit sign groups with a four-bit scale
/// in the top bits.
fn dequantize_iq2_xxs(src: &[u8], dst: &mut [f32]) {
    for (block, y) in src
        .as_chunks::<66>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for ib32 in 0..8 {
            let q = &block[2 + 8 * ib32..2 + 8 * ib32 + 8];
            let aux1 = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[usize::from(q[l])].to_le_bytes();
                let signs = ksign((aux1 >> (7 * l)) & 127);
                let out = &mut y[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
                for (j, value) in out.iter_mut().enumerate() {
                    let sign = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    *value = db * f32::from(grid[j]) * sign;
                }
            }
        }
    }
}

/// Decodes one row of `t` (expert `e`) into `out` (`t.cols` floats). The
/// per-tensor scale is not applied (`DequantizeRow`).
pub fn dequantize_row(t: &TensorRef, shards: &[GgufShard<'_>], e: u64, row: u64, out: &mut [f32]) {
    let row_bytes = t.row_bytes();
    let src = &t.expert(shards, e)[row as usize * row_bytes..][..row_bytes];
    let k = t.cols as usize;
    let out = &mut out[..k];
    match t.ty {
        GgmlType::Nvfp4 => dequantize_nvfp4(src, out),
        GgmlType::F32 => {
            for (value, raw) in out.iter_mut().zip(src.as_chunks::<4>().0) {
                *value = f32::from_le_bytes(*raw);
            }
        }
        GgmlType::F16 => {
            for (value, raw) in out.iter_mut().zip(src.as_chunks::<2>().0) {
                *value = f16_to_f32(u16::from_le_bytes(*raw));
            }
        }
        GgmlType::Bf16 => {
            for (value, raw) in out.iter_mut().zip(src.as_chunks::<2>().0) {
                *value = f32::from_bits(u32::from(u16::from_le_bytes(*raw)) << 16);
            }
        }
        GgmlType::Q8_0 => {
            if dequantize(GgmlType::Q8_0, src, out).is_none() {
                out.fill(0.0);
            }
        }
        GgmlType::Iq2Xxs => dequantize_iq2_xxs(src, out),
        _ => out.fill(0.0),
    }
}

/// `y[r] = scale * sum_c W[e][r][c] * x[c]` for every row of the matrix
/// (`MatVec`): products and sums in double, rows split over threads.
pub fn mat_vec(t: &TensorRef, shards: &[GgufShard<'_>], e: u64, x: &[f32], y: &mut [f32]) {
    let rows = t.rows as usize;
    let cols = t.cols as usize;
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, rows.max(1));
    let chunk = rows.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        for (index, part) in y[..rows].chunks_mut(chunk).enumerate() {
            scope.spawn(move || {
                let mut w = vec![0.0f32; cols];
                for (offset, value) in part.iter_mut().enumerate() {
                    let r = (index * chunk + offset) as u64;
                    dequantize_row(t, shards, e, r, &mut w);
                    let mut acc = 0.0f64;
                    for (&wi, &xi) in w.iter().zip(x) {
                        acc += f64::from(wi) * f64::from(xi);
                    }
                    *value = acc as f32 * t.scale;
                }
            });
        }
    });
}

/// RMSNorm over `x` in place with gamma `w` (`RmsNorm`).
pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
    let mut ss = 0.0f64;
    for &v in x.iter() {
        ss += f64::from(v) * f64::from(v);
    }
    let scale = 1.0f32 / ((ss / x.len() as f64) as f32 + eps).sqrt();
    for (value, &gamma) in x.iter_mut().zip(w) {
        *value = *value * scale * gamma;
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// ggml's softplus: identity above 20.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// max(x, 0)^2, the ReLU^2 activation of every Nemotron-H MLP.
pub fn relu2(x: f32) -> f32 {
    let r = if x > 0.0 { x } else { 0.0 };
    r * r
}

#[cfg(test)]
mod tests {
    use super::ksign;

    #[test]
    fn ksigns_match_llama_cpp() {
        let table = [
            0u8, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
        ];
        for (i, &value) in table.iter().enumerate() {
            assert_eq!(ksign(i as u32), value);
        }
        assert_eq!(ksign(127), 255);
        assert_eq!(ksign(126), 126);
    }
}
