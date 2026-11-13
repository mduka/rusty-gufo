//! How a weight matrix is stored on the device (`matrix.hpp`).

use gufo_hip::DeviceSlice;

/// Rows of a weight tile of the matrix-core products (`kTileRows`).
pub const TILE_ROWS: usize = 16;
/// Columns of a weight tile: one Q8_0 block of each row (`kTileColumns`).
pub const TILE_COLUMNS: usize = 32;
/// Bytes of a Q8_0 tile: 512 int8 values `[16][32]` and the 16 rows'
/// float16 scales (`kQ8TileBytes`).
pub const Q8_TILE_BYTES: usize = 16 * 32 + 16 * 2;
/// Bytes of a bfloat16 tile `[16][32]` (`kBf16TileBytes`).
pub const BF16_TILE_BYTES: usize = 16 * 32 * 2;

/// Bytes of a `[rows][in]` matrix in Q8_0 tiles (`Q8TilesBytes`).
pub const fn q8_tiles_bytes(rows: usize, input: usize) -> usize {
    rows.div_ceil(TILE_ROWS) * (input / TILE_COLUMNS) * Q8_TILE_BYTES
}

/// Bytes of a `[rows][in]` matrix in bfloat16 tiles (`Bf16TilesBytes`).
pub const fn bf16_tiles_bytes(rows: usize, input: usize) -> usize {
    rows.div_ceil(TILE_ROWS) * (input / TILE_COLUMNS) * BF16_TILE_BYTES
}

/// How a weight matrix is stored on the device (`WeightFormat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// bfloat16 `[out][in]`, row-major.
    Bf16,
    /// bfloat16 tiles for the matrix cores.
    Bf16Tiles,
    /// GGML Q8_0 tiles for the matrix cores.
    Q8Tiles,
}

impl WeightFormat {
    /// Bytes of an `[out][in]` matrix in this format.
    pub const fn bytes(self, out: usize, input: usize) -> usize {
        match self {
            Self::Q8Tiles => q8_tiles_bytes(out, input),
            Self::Bf16Tiles => bf16_tiles_bytes(out, input),
            Self::Bf16 => out * input * 2,
        }
    }
}

/// A weight matrix `[out][in]` in the device weight storage
/// ([`crate::weights::DeviceWeights`]): its format and byte range (`Matrix`,
/// whose pointer becomes an offset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix {
    pub format: WeightFormat,
    pub offset: usize,
    pub bytes: usize,
}

impl Default for Matrix {
    /// Empty bfloat16 rows (the C++ default, a null pointer).
    fn default() -> Self {
        Self {
            format: WeightFormat::Bf16,
            offset: 0,
            bytes: 0,
        }
    }
}

/// A weight matrix's device bytes, as the products read them.
#[derive(Clone, Copy)]
pub struct MatrixView<'a> {
    pub format: WeightFormat,
    pub data: DeviceSlice<'a, u8>,
}
