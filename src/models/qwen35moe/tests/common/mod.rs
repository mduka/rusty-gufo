//! What the model tests share: the model files, the fixtures of the C++
//! reference (`tests/fixtures`, recorded from the extension's C++ build on
//! ROCm 10.1's libraries; the local reference set named by
//! `GUFO_QWEN35MOE_FIXTURES` holds the photos and full logit rows) and their
//! prompt conventions.

#![allow(dead_code, clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Mutex;

use gufo_qwen35moe::Engine;
use serde_json::Value;

/// One model on the device at a time.
pub static SERIAL: Mutex<()> = Mutex::new(());

/// A quantization of Qwen3.6 35B-A3B: its fixtures' set name and the
/// environment variable naming its GGUF.
#[derive(Clone, Copy)]
pub struct Quant {
    pub set: &'static str,
    pub variable: &'static str,
}

/// `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (unsloth).
pub const UD_Q4_K_M: Quant = Quant {
    set: "ud-q4_k_m",
    variable: "GUFO_QWEN35MOE_MODEL",
};

/// `Qwen3.6-35B-A3B-Q8_0.gguf` (unsloth).
pub const Q8_0: Quant = Quant {
    set: "q8_0",
    variable: "GUFO_QWEN35MOE_Q8_MODEL",
};

impl Quant {
    /// The model file. Unset, the variable's name stands in for the path,
    /// so the failing load says what to set.
    pub fn model(&self) -> PathBuf {
        std::env::var_os(self.variable)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(self.variable))
    }

    /// The fixture of `mode` (`tests/fixtures/qwen3.6-35b-a3b-<set>-<mode>.json`).
    pub fn fixture(&self, mode: &str) -> Value {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("qwen3.6-35b-a3b-{}-{mode}.json", self.set));
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let fixture: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(fixture["repeat_identical"], Value::Bool(true));
        assert_eq!(fixture["kv_cache"], "f16");
        // hipBLAS and rocBLAS of another ROCm round the prefill's F32 GEMMs
        // differently.
        assert_eq!(fixture["rocm_libraries"], "/opt/rocm/core-10.1/lib");
        fixture
    }
}

/// FNV-1a 64 over the little-endian bytes of the logits row.
pub fn fnv1a64(values: &[f32]) -> String {
    let hash = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
    format!("{hash:016x}")
}

pub fn u32s(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

/// The chat prompt around `body` synthetic tokens: `head + body + tail`.
pub fn synthetic(prefill: &Value, body: usize) -> Vec<u32> {
    let body = (0..body as u32).map(|i| 1000 + (i * 7919 + 13) % 50_000);
    u32s(&prefill["head"])
        .into_iter()
        .chain(body)
        .chain(u32s(&prefill["tail"]))
        .collect()
}

/// A case's prompt: its own ids, the synthetic `body`, or the haiku prompt.
pub fn case_prompt(prefill: &Value, case: &Value) -> Vec<u32> {
    if let Some(prompt) = case.get("prompt").filter(|p| p.is_array()) {
        return u32s(prompt);
    }
    if let Some(ids) = case.get("prompt_ids").filter(|p| p.is_array()) {
        return u32s(ids);
    }
    match case["body"].as_u64() {
        Some(body) => synthetic(prefill, body as usize),
        None => u32s(&prefill["haiku"]),
    }
}

/// The first divergence of a fixture trace (`ids`, `logits_fnv1a64`) from
/// the engine's: entry 0 is `next` and the logits the engine holds, entry
/// `k` follows `forward_token(ids[k - 1], start + k - 1)`. With
/// `skip_first_logits` entry 0's logits are not compared (a restored
/// snapshot holds none).
pub fn trace(
    engine: &mut Engine,
    entry: &Value,
    mut next: Option<u32>,
    start: usize,
    skip_first_logits: bool,
) -> Option<String> {
    let ids = u32s(&entry["ids"]);
    let digests = entry["logits_fnv1a64"].as_array().unwrap();
    for (step, pos) in (0..ids.len()).zip(start as u32..) {
        let token = next.unwrap();
        if token != ids[step] {
            return Some(format!("step {step}: token {token}, C++ {}", ids[step]));
        }
        if step > 0 || !skip_first_logits {
            let digest = fnv1a64(&engine.logits().unwrap());
            if digest != digests[step].as_str().unwrap() {
                return Some(format!(
                    "step {step}: logits {digest}, C++ {}",
                    digests[step]
                ));
            }
        }
        if step + 1 < ids.len() {
            next = engine.forward_token(token, pos, true).unwrap();
        }
    }
    None
}
