#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_ROCM_DSPARK_CANDIDATES_H_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_ROCM_DSPARK_CANDIDATES_H_

#include <cstdint>

// The DSpark candidates kernel's output for one drafted position (68 bytes),
// read by `Candidates::from_bytes` in
// src/models/deepseek_v4_flash/src/dspark_sampler.rs.
inline constexpr uint32_t DS4_DSPARK_CANDIDATES = 8;

struct ds4_dspark_candidates {
  int32_t ids[DS4_DSPARK_CANDIDATES];
  float logits[DS4_DSPARK_CANDIDATES];
  float confidence;
};

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_ROCM_DSPARK_CANDIDATES_H_
