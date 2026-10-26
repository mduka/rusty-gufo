//! Paths of the local artifacts, from the environment variables of the C++
//! tests.

#![allow(dead_code)]

use std::path::PathBuf;

/// The IQ2XXS target GGUF (`GUFO_DEEPSEEK_V4_FLASH_MODEL`).
pub fn model_path() -> PathBuf {
    artifact("GUFO_DEEPSEEK_V4_FLASH_MODEL")
}

/// The DSpark support GGUF (`GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL`).
pub fn dspark_path() -> PathBuf {
    artifact("GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL")
}

/// The artifact `variable` names. Unset, the path is the variable itself, so
/// a missing-artifact message says what to set.
fn artifact(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .filter(|path| !path.is_empty())
        .map_or_else(|| PathBuf::from(format!("${variable}")), PathBuf::from)
}

/// FNV-1a 64 as lowercase hex, the digest of the fixtures.
pub fn fnv(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The little-endian bytes of token IDs.
pub fn token_bytes(tokens: &[i32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}
