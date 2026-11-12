//! Host dequantisation of the formats the DFlash2 and dense MTP loaders
//! repack and the Flash-Next n-gram table stores, a port of
//! `src/core/quant/ggml_dequant.cpp` with its `f32` operation order (no fused
//! multiply-add, as the C++ host build). The results match the C++ loader bit
//! for bit.

use crate::GgmlType;

/// IEEE half to single precision (`Fp16ToFloat`).
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half & 0x8000) << 16;
    let exponent = u32::from((half >> 10) & 0x1F);
    let mantissa = u32::from(half & 0x3FF);
    match exponent {
        0 if mantissa == 0 => f32::from_bits(sign),
        0 => {
            // Subnormal: (mantissa / 1024) * 2^-14, exact in f32.
            let value = (mantissa as f32 / 1024.0) * f32::from_bits(0x3880_0000);
            if half & 0x8000 != 0 { -value } else { value }
        }
        31 => f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

fn half(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// Scale and minimum of sub-block `index` of a Q4_K block (`GetQ4ScaleMin`).
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

/// One Q4_K block (144 bytes: d, dmin, 12 scale bytes, 128 quant bytes).
fn q4_k_block(block: &[u8], out: &mut [f32]) {
    let (d, dmin) = (half(block, 0), half(block, 2));
    let (scales, quants) = (&block[4..16], &block[16..144]);
    for (index, value) in out.iter_mut().enumerate().take(256) {
        let group = index / 32;
        let packed = quants[(group / 2) * 32 + index % 32];
        let quant = if group & 1 != 0 {
            packed >> 4
        } else {
            packed & 0x0F
        };
        let (scale, minimum) = q4_scale_min(group, scales);
        *value = d * f32::from(scale) * f32::from(quant) - dmin * f32::from(minimum);
    }
}

/// One Q6_K block (210 bytes: 128 low, 64 high, 16 scales, d).
fn q6_k_block(block: &[u8], out: &mut [f32]) {
    let (low_bits, high_bits, scales) = (&block[..128], &block[128..192], &block[192..208]);
    let d = half(block, 208);
    for (index, value) in out.iter_mut().enumerate().take(256) {
        let (half_index, within) = (index / 128, index % 128);
        let (segment, lane) = (within / 32, within % 32);
        let base = half_index * 64;
        let qh = high_bits[half_index * 32 + lane];
        let (low, high) = match segment {
            0 => (low_bits[base + lane] & 0x0F, qh & 0x03),
            1 => (low_bits[base + 32 + lane] & 0x0F, (qh >> 2) & 0x03),
            2 => (low_bits[base + lane] >> 4, (qh >> 4) & 0x03),
            _ => (low_bits[base + 32 + lane] >> 4, (qh >> 6) & 0x03),
        };
        let scale = scales[half_index * 8 + lane / 16 + segment * 2] as i8;
        let quant = i32::from((high << 4) | low) - 32;
        *value = d * f32::from(scale) * quant as f32;
    }
}

/// One Q3_K block (110 bytes: 32 high-bit mask bytes, 64 quant bytes, 12
/// packed 6-bit scales, d), with `UnpackQ3Scales` and `Q3Value`.
fn q3_k_block(block: &[u8], out: &mut [f32]) {
    let (hmask, quants, packed) = (&block[..32], &block[32..96], &block[96..108]);
    let d = half(block, 108);
    let scales: [i8; 16] = std::array::from_fn(|index| {
        let low = if index < 8 {
            packed[index] & 0x0F
        } else {
            (packed[index - 8] >> 4) & 0x0F
        };
        let high = (packed[8 + index % 4] >> (2 * (index / 4))) & 0x03;
        (i32::from(low | (high << 4)) - 32) as i8
    });
    for (index, value) in out.iter_mut().enumerate().take(256) {
        let (half_index, within) = (index / 128, index % 128);
        let (pair, lane) = (within / 32, within % 32);
        let low = (quants[half_index * 32 + lane] >> (2 * pair)) & 0x03;
        let high = hmask[lane] & (1 << (half_index * 4 + pair)) != 0;
        let quant = low as i8 - if high { 0 } else { 4 };
        let scale = scales[half_index * 8 + pair * 2 + lane / 16];
        *value = d * f32::from(scale) * f32::from(quant);
    }
}

/// One Q8_0 block (34 bytes: d, 32 signed quants).
/// Non-linear 4-bit codebook of IQ4_NL (`kValuesIq4Nl`).
const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// IQ4_NL: one scale, then 16 bytes holding values i (low nibble) and
/// i + 16 (high nibble) (`Iq4NlValue`).
fn iq4_nl_block(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    for (lane, &packed) in block[2..18].iter().enumerate() {
        out[lane] = d * f32::from(IQ4_NL_VALUES[usize::from(packed & 0x0F)]);
        out[lane + 16] = d * f32::from(IQ4_NL_VALUES[usize::from(packed >> 4)]);
    }
}

fn q8_0_block(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    for (value, &quant) in out.iter_mut().zip(&block[2..34]) {
        *value = d * f32::from(quant as i8);
    }
}

/// Dequantises `out.len()` values from `bytes` (whole blocks of `format`).
/// Returns `None` for other formats or sizes.
pub fn dequantize(format: GgmlType, bytes: &[u8], out: &mut [f32]) -> Option<()> {
    let (elements, block_bytes) = format.block();
    let decode: fn(&[u8], &mut [f32]) = match format {
        GgmlType::Q3K => q3_k_block,
        GgmlType::Q4K => q4_k_block,
        GgmlType::Q6K => q6_k_block,
        GgmlType::Q8_0 => q8_0_block,
        GgmlType::Iq4Nl => iq4_nl_block,
        _ => return None,
    };
    if !out.len().is_multiple_of(elements) || bytes.len() < out.len() / elements * block_bytes {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_convert_like_the_cpp_helper() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    }

    #[test]
    fn iq4_nl_splits_each_byte_into_two_halves_of_the_block() {
        let mut block = vec![0u8; 18];
        block[..2].copy_from_slice(&0x3800u16.to_le_bytes());
        block[2] = 0xF0;
        block[17] = 0x08;
        let mut out = vec![0f32; 32];
        dequantize(GgmlType::Iq4Nl, &block, &mut out).unwrap();
        assert_eq!((out[0], out[16]), (-63.5, 56.5));
        assert_eq!((out[15], out[31]), (0.5, -63.5));
        assert_eq!(out[1], -63.5);
    }

    #[test]
    fn blocks_decode_scales_minima_and_signs() {
        // Q8_0: d = 0.5, quants -2 and 3.
        let mut q8 = vec![0u8; 34];
        q8[..2].copy_from_slice(&0x3800u16.to_le_bytes());
        q8[2] = (-2i8) as u8;
        q8[3] = 3;
        let mut out = vec![0f32; 32];
        dequantize(GgmlType::Q8_0, &q8, &mut out).unwrap();
        assert_eq!(&out[..3], &[-1.0, 1.5, 0.0]);
        // Q4_K: d = 1, dmin = 1, sub-block 0 scale 2 min 1, first nibble 7.
        let mut q4 = vec![0u8; 144];
        q4[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        q4[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());
        q4[4] = 2;
        q4[8] = 1;
        q4[16] = 0x57;
        let mut out = vec![0f32; 256];
        dequantize(GgmlType::Q4K, &q4, &mut out).unwrap();
        assert_eq!(out[0], 2.0 * 7.0 - 1.0);
        // Sub-block 1 reads the high nibble with scale/min of index 1 (0/0).
        assert_eq!(out[32], 0.0);
        // Q6_K: d = 1, scale 3 for the first 16 lanes, quant 0 + 1 = 1 - 32.
        let mut q6 = vec![0u8; 210];
        q6[0] = 1;
        q6[192] = 3;
        q6[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
        let mut out = vec![0f32; 256];
        dequantize(GgmlType::Q6K, &q6, &mut out).unwrap();
        assert_eq!(out[0], 3.0 * -31.0);
        assert_eq!(out[1], 3.0 * -32.0);
        assert!(dequantize(GgmlType::Q5K, &q6, &mut out).is_none());
        // Q3_K against upstream's C++ `DequantizeQ3_K` over two
        // patterned blocks with d = 0.0625 and -1.5 (FNV-1a 64 of the bytes).
        let mut q3: Vec<u8> = (0..220).map(|i| ((i * 37 + 11) % 251) as u8).collect();
        q3[108..110].copy_from_slice(&0x2C00u16.to_le_bytes());
        q3[218..220].copy_from_slice(&0xBE00u16.to_le_bytes());
        let mut out = vec![0f32; 512];
        dequantize(GgmlType::Q3K, &q3, &mut out).unwrap();
        assert_eq!(
            (out[0], out[1], out[255], out[511]),
            (0.1875, -0.25, 0.0, 39.0)
        );
        let digest = out
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            });
        assert_eq!(digest, 0xf761_4da0_af72_d8aa);
        assert!(dequantize(GgmlType::Q6K, &q6[..100], &mut out).is_none());
    }
}
