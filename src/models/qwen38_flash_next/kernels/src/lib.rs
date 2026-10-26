//! Qwen3.8-Flash-Next kernels. The launchers are generated from
//! `launch.toml`; the generated FFI calls are the crate's only `unsafe`.
//! The functions below size the buffers those launch contracts describe,
//! as the C++ host helpers of `kernels.hpp` and the quantized tier do.

#![deny(unsafe_code)]

#[allow(unsafe_code)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/launch.rs"));
}

pub use generated::*;

/// Bytes of tiled Q8 activations of `batch` rows of `k` values: whole
/// 128-token macro tiles of 16-token, 32-value tiles of 576 bytes
/// (`Q8TiledBytes`).
pub const fn q8_tiled_bytes(batch: usize, k: usize) -> usize {
    batch.div_ceil(128) * 8 * (k / 32) * 576
}

/// Bytes of Q8_1 activations of the quantized tier: `n` rows padded to 512
/// values, 36 bytes per 32 (`qfn_mmq_q8_1_bytes`).
pub const fn q8_1_bytes(n: usize, k: usize) -> usize {
    n * k.div_ceil(512) * 512 / 32 * 36
}

/// Partial sums per inject logit of the scalar mixer epilogue
/// (`HcInjectParts`).
pub const fn inject_parts(hidden: u32) -> u32 {
    hidden.div_ceil(256)
}

/// Partial sums per inject logit of the four-lane mixer epilogues
/// (`HcInjectPartsVec4`).
pub const fn inject_parts_vec4(hidden: u32) -> u32 {
    hidden.div_ceil(4).div_ceil(256)
}

/// Compact routed rows of `slots` assignments with every expert's bucket
/// padded to 16 rows (`RoutedCompactRows`).
pub const fn routed_compact_rows(slots: usize, experts: usize) -> usize {
    slots + experts * 15
}

/// IDs in each of the two MTP candidate buffers
/// (`MtpCandidateWorkspaceSize`).
pub const fn mtp_candidate_workspace(vocab: u32) -> u32 {
    let tiles = vocab.div_ceil(1024);
    (if tiles > 2 { tiles } else { 2 }) * 64
}

/// Column-tile width for the tiled routed projections fitted to the expert
/// bucket sizes: the width among 16..=80 with the least padded work
/// (`qfn_mmq_routed_tile_cols_for_counts`); 0 without experts.
pub fn routed_tile_cols_for_counts(counts: &[u32]) -> u32 {
    const PANEL: i64 = 16;
    if counts.is_empty() {
        return 0;
    }
    let mut best = (0u32, 0i64);
    for width in (16u32..=80).step_by(16) {
        let cost: i64 = counts
            .iter()
            .filter(|&&count| count != 0)
            // The C++ bucket count is unsigned 32-bit arithmetic.
            .map(|&count| {
                i64::from(count.wrapping_add(width - 1) / width) * (i64::from(width) + PANEL)
            })
            .sum();
        if best.0 == 0 || cost < best.1 {
            best = (width, cost);
        }
    }
    best.0
}

const fn round_256(bytes: usize) -> usize {
    bytes.div_ceil(256) * 256
}

/// Scratch of `mmq_moe_vec` and `mmq_moe_vec_pair`: the Q8_1 rows.
pub const fn moe_vec_scratch(n_tokens: usize, k: usize) -> usize {
    round_256(q8_1_bytes(n_tokens, k))
}

/// Scratch of `mmq_moe_gated_vec`: the Q8_1 rows and the larger of both
/// expert group layouts.
pub const fn moe_gated_vec_scratch(n_tokens: usize, k: usize, n_used: usize) -> usize {
    let groups = 4 + n_tokens * n_used * 72 + n_tokens * n_used * (n_tokens + 1) * 4;
    round_256(q8_1_bytes(n_tokens, k) + groups)
}

/// Scratch of `mmq_moe_tiled` and `mmq_q4k_moe_pair_tiled`: two row maps,
/// the expert bounds and the gathered Q8_1 rows (144 bytes per 128 values)
/// with the widest column tile's padding.
pub const fn moe_tiled_scratch(
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    n_used: usize,
) -> usize {
    let rows = n_tokens * n_used;
    round_256(rows * 4) * 2
        + round_256((n_experts + 1) * 4)
        + round_256(rows * (k.div_ceil(512) * 512) * 144 / 128 + 128 * 144)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_the_cpp_helpers() {
        assert_eq!(q8_tiled_bytes(1, 2560), 8 * 80 * 576);
        assert_eq!(q8_tiled_bytes(129, 32), 16 * 576);
        assert_eq!(q8_1_bytes(8, 10240), 8 * 320 * 36);
        assert_eq!(q8_1_bytes(1, 640), 32 * 36);
        assert_eq!(inject_parts(2560), 10);
        assert_eq!(inject_parts_vec4(2560), 3);
        assert_eq!(inject_parts_vec4(10), 1);
        assert_eq!(routed_compact_rows(20480, 512), 20480 + 7680);
        assert_eq!(mtp_candidate_workspace(248_320), 243 * 64);
        assert_eq!(mtp_candidate_workspace(50), 128);
    }

    #[test]
    fn tile_width_minimises_padded_work() {
        assert_eq!(routed_tile_cols_for_counts(&[]), 0);
        // One row per expert: the narrowest tile.
        assert_eq!(routed_tile_cols_for_counts(&[1; 512]), 16);
        // 80-row buckets fit the widest tile exactly.
        assert_eq!(routed_tile_cols_for_counts(&[80, 0, 80]), 80);
        // 40 rows: 48 costs 64 per bucket, 16 costs 3 * 32 = 96.
        assert_eq!(routed_tile_cols_for_counts(&[40; 4]), 48);
    }
}
