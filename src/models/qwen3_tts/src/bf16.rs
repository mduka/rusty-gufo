//! BF16 rounding as the C++ host code does it (`RoundBfloat16`): round to
//! nearest even on the bit pattern, infinities and NaNs truncated.

pub fn round_bf16(value: f32) -> f32 {
    let mut bits = value.to_bits();
    if bits & 0x7F80_0000 != 0x7F80_0000 {
        bits = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    }
    f32::from_bits(bits & 0xFFFF_0000)
}

/// The f32 value of BF16 bits.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_to_nearest_even() {
        assert_eq!(round_bf16(1.0), 1.0);
        // Halfway between 1.0 and the next BF16 value rounds to even (1.0).
        assert_eq!(round_bf16(f32::from_bits(0x3F80_8000)), 1.0);
        assert_eq!(
            round_bf16(f32::from_bits(0x3F81_8000)).to_bits(),
            0x3F82_0000
        );
        assert!(round_bf16(f32::NAN).is_nan());
        assert_eq!(round_bf16(f32::INFINITY), f32::INFINITY);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
    }
}
