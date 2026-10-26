//! The fixed DeepSeek V4 Flash shape (`model_data_internal.h`); the loader
//! refuses artifacts that differ.

pub const N_LAYER: u32 = 43;
pub const N_EMBD: u32 = 4096;
pub const N_VOCAB: u32 = 129280;
pub const N_HEAD: u32 = 64;
pub const N_HEAD_KV: u32 = 1;
pub const N_HEAD_DIM: u32 = 512;
pub const N_VALUE_DIM: u32 = 512;
pub const N_ROT: u32 = 64;
pub const N_OUT_GROUP: u32 = 8;
pub const N_LORA_Q: u32 = 1024;
pub const N_LORA_O: u32 = 1024;
pub const N_EXPERT: u32 = 256;
pub const N_EXPERT_USED: u32 = 6;
pub const N_EXPERT_SHARED: u32 = 1;
pub const N_FF_EXP: u32 = 2048;
pub const N_HASH_LAYER: u32 = 3;
pub const N_SWA: u32 = 128;
pub const N_INDEXER_HEAD: u32 = 64;
pub const N_INDEXER_HEAD_DIM: u32 = 128;
pub const N_INDEXER_TOP_K: u32 = 512;
pub const N_HC: u32 = 4;
pub const N_HC_SINKHORN_ITER: u32 = 20;
pub const MAX_DIMS: usize = 8;

pub const MODEL_SHAPE_NAME: &str = "DeepSeek V4 Flash";
pub const RMS_EPS: f32 = 1.0e-6;
pub const HC_EPS: f32 = 1.0e-6;
pub const EXPERT_WEIGHT_SCALE: f32 = 1.5;
pub const SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const ROPE_FREQ_BASE: f32 = 10000.0;
pub const ROPE_SCALE_FACTOR: f32 = 16.0;
pub const ROPE_YARN_BETA_FAST: f32 = 32.0;
pub const ROPE_YARN_BETA_SLOW: f32 = 1.0;
pub const COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
pub const ROPE_ORIG_CTX: u64 = 65536;

/// GGML tensor type IDs the model uses (`tensor_types.h`).
pub mod tensor_type {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q8_0: u32 = 8;
    pub const Q2_K: u32 = 10;
    pub const Q4_K: u32 = 12;
    pub const IQ2_XXS: u32 = 16;
    pub const I32: u32 = 26;
}

/// The attention compression ratio the model layout fixes for `layer`:
/// none in the first two layers, then 4 and 128 alternating
/// (`ds4_expected_layer_compress_ratio`).
pub const fn expected_compress_ratio(layer: u32) -> u32 {
    if layer < 2 {
        0
    } else if layer & 1 == 0 {
        4
    } else {
        128
    }
}
