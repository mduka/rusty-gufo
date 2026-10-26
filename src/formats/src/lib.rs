//! GGML tensor storage types of the supported GGUF artifacts and their byte
//! geometry (`src/core/quant/ggml_dequant.cpp`). Storage support does not
//! imply that a model has a compute kernel for the type.
//!
//! Kernel contracts size quantized buffers from this table, so it belongs to
//! the trusted base (docs/RUST.md): a wrong block size lets a kernel
//! overrun a buffer that passed its extent check.

#![forbid(unsafe_code)]

pub mod dequant;

/// A GGML storage type, by its GGUF type ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq4Nl,
    Iq3S,
    Iq4Xs,
    I32,
    Bf16,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            23 => Self::Iq4Xs,
            26 => Self::I32,
            30 => Self::Bf16,
            _ => return None,
        })
    }

    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::Iq2Xxs => 16,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq4Xs => 23,
            Self::I32 => 26,
            Self::Bf16 => 30,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::I32 => "I32",
            Self::Bf16 => "BF16",
        }
    }

    /// The type named like [`GgmlType::name`] (`"Q4_K"`, `"BF16"`, ...).
    pub fn from_name(name: &str) -> Option<Self> {
        (0..64).filter_map(Self::from_id).find(|t| t.name() == name)
    }

    /// Elements per block, and bytes per block.
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::F32 | Self::I32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Iq4Nl => (32, 18),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::Iq2Xxs => (256, 66),
            Self::Iq3S => (256, 110),
            Self::Iq4Xs => (256, 136),
        }
    }

    /// Bytes of `elements` values; `None` for a partial block or overflow.
    pub fn encoded_bytes(self, elements: u64) -> Option<u64> {
        let (per_block, bytes) = self.block();
        if !elements.is_multiple_of(per_block as u64) {
            return None;
        }
        (elements / per_block as u64).checked_mul(bytes as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_geometry_matches_the_cpp_table() {
        for id in 0..64 {
            if let Some(ty) = GgmlType::from_id(id) {
                assert_eq!(ty.id(), id, "{}", ty.name());
            }
        }
        assert_eq!(GgmlType::from_id(4), None);
        assert_eq!(GgmlType::from_name("IQ4_XS"), Some(GgmlType::Iq4Xs));
        assert_eq!(GgmlType::from_name("Q4"), None);
        assert_eq!(GgmlType::Q4K.encoded_bytes(5120), Some(20 * 144));
        assert_eq!(GgmlType::Q8_0.encoded_bytes(64), Some(68));
        assert_eq!(GgmlType::Q4K.encoded_bytes(100), None);
        assert_eq!(GgmlType::F32.encoded_bytes(3), Some(12));
        assert_eq!(
            GgmlType::Q8K.encoded_bytes(u64::MAX / 256 * 256),
            None,
            "overflow"
        );
    }
}
