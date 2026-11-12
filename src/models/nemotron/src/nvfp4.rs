//! NVFP4 weights (`nvfp4.cpp`): GGML_TYPE_NVFP4 in llama.cpp. Gufo's GGUF
//! reader has no name for the type and leaves these tensors unsized; this
//! crate sizes and decodes them.

/// The GGUF type ID of NVFP4 (`kNvfp4`).
pub const NVFP4_TYPE_ID: u32 = 40;

/// Values per NVFP4 block (`kNvfp4BlockElements`).
pub const NVFP4_BLOCK_ELEMENTS: usize = 64;

/// Bytes per NVFP4 block (`sizeof(BlockNvfp4)`).
pub const NVFP4_BLOCK_BYTES: usize = 36;

/// NVFP4 block (`BlockNvfp4`): 64 E2M1 values as four 16-element
/// sub-blocks, each with an unsigned E4M3 scale. Within a sub-block, low
/// nibbles hold elements 0..7 and high nibbles elements 8..15. A per-tensor
/// F32 scale is applied by the caller.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockNvfp4 {
    pub d: [u8; 4],
    pub qs: [u8; 32],
}

impl BlockNvfp4 {
    /// The block's storage bytes: `d`, then `qs`.
    pub fn to_bytes(&self) -> [u8; NVFP4_BLOCK_BYTES] {
        let mut bytes = [0u8; NVFP4_BLOCK_BYTES];
        bytes[..4].copy_from_slice(&self.d);
        bytes[4..].copy_from_slice(&self.qs);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; NVFP4_BLOCK_BYTES]) -> Self {
        let mut block = Self::default();
        block.d.copy_from_slice(&bytes[..4]);
        block.qs.copy_from_slice(&bytes[4..]);
        block
    }
}

/// E2M1 values doubled, as in ggml's `kvalues_mxfp4`; the halving is folded
/// into [`ue4m3_to_float`].
const VALUES_E2M1_X2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Unsigned E4M3 (bias 7) scale times 0.5, mirroring `ggml_ue4m3_to_fp32`.
fn ue4m3_to_float(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exponent = i32::from((x >> 3) & 0x0F);
    let mantissa = f32::from(x & 0x07);
    // ldexp by a power of two in range is exact, as is the product.
    let raw = if exponent == 0 {
        mantissa * (1.0 / 512.0)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    raw * 0.5
}

/// Dequantizes `dst.len()` values (a multiple of 64) from the blocks in
/// `src`, without the per-tensor scale (`DequantizeNvfp4`).
///
/// # Panics
/// When `src` holds fewer blocks than `dst` needs.
pub fn dequantize_nvfp4(src: &[u8], dst: &mut [f32]) {
    let blocks = dst.len() / NVFP4_BLOCK_ELEMENTS;
    for b in 0..blocks {
        let block = &src[b * NVFP4_BLOCK_BYTES..(b + 1) * NVFP4_BLOCK_BYTES];
        let (d, qs) = block.split_at(4);
        for s in 0..4 {
            let scale = ue4m3_to_float(d[s]);
            let y = &mut dst[b * NVFP4_BLOCK_ELEMENTS + s * 16..][..16];
            for j in 0..8 {
                let packed = qs[s * 8 + j];
                y[j] = f32::from(VALUES_E2M1_X2[usize::from(packed & 0x0F)]) * scale;
                y[j + 8] = f32::from(VALUES_E2M1_X2[usize::from(packed >> 4)]) * scale;
            }
        }
    }
}
