//! Host dequantization of every weight format Gemma 4 binds
//! (`quant::DequantizeRow` of Gufo's `src/core/quant/ggml_dequant.cpp`),
//! for the CPU reference and the kernel tests. Q8_0, Q4_K and Q6_K come from
//! `gufo_formats::dequant`; the other formats follow the C++ code with its
//! `f32` operation order (no fused multiply-add).

use gufo_formats::GgmlType;
use gufo_formats::dequant::{dequantize as dequantize_shared, f16_to_f32};

fn half(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// BF16 bits to single precision.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// `DequantizeQ4_0`: d, then 16 bytes holding values i (low nibble) and
/// i + 16 (high nibble), offset by 8.
fn q4_0_block(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    for (j, &q) in block[2..18].iter().enumerate() {
        out[j] = (i32::from(q & 0x0F) - 8) as f32 * d;
        out[j + 16] = (i32::from(q >> 4) - 8) as f32 * d;
    }
}

/// `DequantizeQ5_0`: d, 32 high bits, then 16 bytes of low nibbles, offset
/// by 16.
fn q5_0_block(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    for (j, &q) in block[6..22].iter().enumerate() {
        let xh0 = ((qh >> j) << 4) & 0x10;
        let xh1 = (qh >> (j + 12)) & 0x10;
        out[j] = ((i32::from(q & 0x0F) | xh0 as i32) - 16) as f32 * d;
        out[j + 16] = ((i32::from(q >> 4) | xh1 as i32) - 16) as f32 * d;
    }
}

/// `DequantizeQ5_1`: d, m, 32 high bits, then 16 bytes of low nibbles.
fn q5_1_block(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let m = half(block, 2);
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for (j, &q) in block[8..24].iter().enumerate() {
        let xh0 = ((qh >> j) & 1) << 4;
        let xh1 = ((qh >> (j + 16)) & 1) << 4;
        out[j] = (u32::from(q & 0x0F) | xh0) as f32 * d + m;
        out[j + 16] = (u32::from(q >> 4) | xh1) as f32 * d + m;
    }
}

/// Scale and minimum of sub-block `index` of a K-quant block
/// (`GetQ4ScaleMin`).
fn q4_scale_min(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 0x3F, packed[index + 4] & 0x3F)
    } else {
        (
            (packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

/// One Q5_K block (176 bytes: d, dmin, 12 scale bytes, 32 high-bit bytes,
/// 128 quant bytes; `Q5Value`).
fn q5_k_block(block: &[u8], out: &mut [f32]) {
    let (d, dmin) = (half(block, 0), half(block, 2));
    let (scales, qh, qs) = (&block[4..16], &block[16..48], &block[48..176]);
    for (index, value) in out.iter_mut().enumerate().take(256) {
        let gg = index / 64;
        let wv = index % 64;
        let lane = wv % 32;
        let low_half = wv < 32;
        let qb = qs[gg * 32 + lane];
        let quant4 = if low_half { qb & 0x0F } else { qb >> 4 };
        let bit = 2 * gg + usize::from(!low_half);
        let quant = quant4 + if (qh[lane] >> bit) & 1 != 0 { 16 } else { 0 };
        let (sc, m) = q4_scale_min(2 * gg + usize::from(!low_half), scales);
        *value = d * f32::from(sc) * f32::from(quant) - dmin * f32::from(m);
    }
}

/// Dequantizes `out.len()` values of one row in `ty` from `bytes`
/// (`DequantizeRow`). `None` for other formats or a partial block.
pub fn dequantize_row(ty: GgmlType, bytes: &[u8], out: &mut [f32]) -> Option<()> {
    let k = out.len();
    match ty {
        GgmlType::F32 => {
            for (value, b) in out.iter_mut().zip(bytes.get(..4 * k)?.as_chunks::<4>().0) {
                *value = f32::from_le_bytes(*b);
            }
            return Some(());
        }
        GgmlType::F16 | GgmlType::Bf16 => {
            let convert = if ty == GgmlType::F16 {
                f16_to_f32
            } else {
                bf16_to_f32
            };
            for (value, b) in out.iter_mut().zip(bytes.get(..2 * k)?.as_chunks::<2>().0) {
                *value = convert(u16::from_le_bytes(*b));
            }
            return Some(());
        }
        GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K => {
            return dequantize_shared(ty, bytes, out);
        }
        _ => {}
    }
    let decode: fn(&[u8], &mut [f32]) = match ty {
        GgmlType::Q4_0 => q4_0_block,
        GgmlType::Q5_0 => q5_0_block,
        GgmlType::Q5_1 => q5_1_block,
        GgmlType::Q5K => q5_k_block,
        _ => return None,
    };
    let (elements, block_bytes) = ty.block();
    if !k.is_multiple_of(elements) || bytes.len() < k / elements * block_bytes {
        return None;
    }
    for (block, values) in bytes
        .chunks_exact(block_bytes)
        .zip(out.chunks_exact_mut(elements))
    {
        decode(block, values);
    }
    Some(())
}
