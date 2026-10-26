//! Artifacts and sampling cases shared by the Flash-Next model tests.

#![allow(dead_code)]

use std::path::PathBuf;

use gufo_core::sampling::SamplingConfig;

/// The context of the C++ QUALITY checks.
pub const CONTEXT: u32 = 6145;

/// The artifact `variable` names (docs/TESTING.md). Unset, the path is the
/// variable itself, so a missing-artifact message says what to set.
pub fn artifact(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .filter(|path| !path.is_empty())
        .map_or_else(|| PathBuf::from(format!("${variable}")), PathBuf::from)
}

/// The first shard of the UD-Q4_K_XL trunk (`GUFO_FLASH_NEXT_MODEL`).
pub fn trunk_path() -> PathBuf {
    artifact("GUFO_FLASH_NEXT_MODEL")
}

/// The shared Q8_0 MTP sidecar (`GUFO_FLASH_NEXT_MTP_MODEL`).
pub fn mtp_path() -> PathBuf {
    artifact("GUFO_FLASH_NEXT_MTP_MODEL")
}

/// The vision sidecar (`GUFO_FLASH_NEXT_MMPROJ`, else the pinned one in the
/// Hugging Face cache).
pub fn vision_sidecar() -> PathBuf {
    std::env::var_os("GUFO_FLASH_NEXT_MMPROJ").map_or_else(
        || {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(
                ".cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/\
                 38bb39ee97821de2c9009abb7e93950eec396e66/mmproj-BF16.gguf",
            )
        },
        PathBuf::from,
    )
}

/// `QwenSamplingCases()` (`tests/models/qwen27b/sampling_cases.hpp`).
pub fn sampling_cases() -> Vec<(&'static str, SamplingConfig)> {
    let t = |temperature| SamplingConfig {
        temperature,
        ..SamplingConfig::default()
    };
    let hot = t(0.8);
    let mut cases = vec![
        SamplingConfig::default(),
        t(0.05),
        t(1.0),
        t(2.0),
        SamplingConfig {
            top_k: 1,
            ..hot.clone()
        },
        SamplingConfig {
            top_k: 3,
            ..hot.clone()
        },
        SamplingConfig {
            top_p: 0.7,
            ..hot.clone()
        },
        SamplingConfig {
            top_p: 0.0,
            ..hot.clone()
        },
        SamplingConfig {
            top_p: 0.0,
            min_keep: 3,
            ..hot.clone()
        },
        SamplingConfig {
            min_p: 0.3,
            ..hot.clone()
        },
        SamplingConfig {
            min_p: 1.0,
            ..hot.clone()
        },
        SamplingConfig {
            top_k: 1,
            min_keep: 3,
            ..hot.clone()
        },
        SamplingConfig {
            top_p: 0.01,
            min_keep: 3,
            ..hot.clone()
        },
        SamplingConfig {
            min_p: 1.0,
            min_keep: 3,
            ..hot.clone()
        },
        SamplingConfig {
            repeat_penalty: 1.5,
            ..hot.clone()
        },
        SamplingConfig {
            repeat_penalty: 0.7,
            ..hot.clone()
        },
        SamplingConfig {
            frequency_penalty: 0.4,
            ..hot.clone()
        },
        SamplingConfig {
            frequency_penalty: -0.4,
            ..hot.clone()
        },
        SamplingConfig {
            presence_penalty: 0.4,
            ..hot.clone()
        },
        SamplingConfig {
            presence_penalty: -0.4,
            ..hot.clone()
        },
        SamplingConfig {
            repeat_penalty: 1.5,
            repeat_last_n: 1,
            ..hot.clone()
        },
        SamplingConfig {
            repeat_penalty: 1.5,
            repeat_last_n: 0,
            frequency_penalty: 0.4,
            presence_penalty: 0.4,
            ..hot.clone()
        },
        SamplingConfig {
            repeat_penalty: 1.5,
            repeat_last_n: 3,
            frequency_penalty: 0.4,
            presence_penalty: -0.4,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            top_k: 4,
            top_p: 0.7,
            min_p: 0.3,
            min_keep: 2,
            repeat_penalty: 1.2,
            repeat_last_n: 3,
            frequency_penalty: 0.2,
            presence_penalty: -0.1,
            ..hot.clone()
        },
        SamplingConfig {
            top_k: 1,
            top_p: 0.01,
            min_p: 1.0,
            min_keep: 3,
            ..hot.clone()
        },
    ];
    for case in &mut cases {
        case.seed = 73;
    }
    cases[2].seed = 0;
    cases[3].seed = 808;
    NAMES.into_iter().zip(cases).collect()
}

const NAMES: [&str; 25] = [
    "greedy",
    "temperature-cold",
    "temperature",
    "temperature-hot",
    "top-k-one",
    "top-k",
    "top-p",
    "top-p-zero",
    "top-p-zero-floor",
    "min-p",
    "min-p-one",
    "top-k-floor",
    "top-p-floor",
    "min-p-floor",
    "repeat-penalty",
    "repeat-reward",
    "frequency-penalty",
    "frequency-reward",
    "presence-penalty",
    "presence-reward",
    "history-one",
    "history-disabled",
    "greedy-penalties",
    "all-filters",
    "all-floors",
];
