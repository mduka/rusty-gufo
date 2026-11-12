#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

/* The routed experts of DeepSeek V4 Flash and the largest compression ratio
 * of its compressors. */
enum {
  DS4_ROCM_N_EXPERT = 256u,
  DS4_ROCM_N_EXPERT_USED = 6u,
  DS4_ROCM_COMPRESSOR_MAX_RATIO = 128u
};
#define DS4_ROCM_EXPERT_WEIGHT_SCALE 1.5f

/* Prompt width from which the mixed-attention, fused norm/rope, padded
 * output-B and inverse-rope routes run: enabling them at 64 exceeded the
 * conversational maximum-error envelope. The 32/64/96 batch-versus-sequential
 * logit gate in ds4.target covers these boundaries. */
#define DS4_ROCM_ATTENTION_WIDE_ROWS 96u

/* Tile geometry for dequant_q8_0_to_f16_transpose_tiled_kernel; see its
 * definition in ds4_rocm_q8.hip.hpp for why the transpose is tiled. */
#define DS4_Q8_T_TILE_I 32u
#define DS4_Q8_T_TILE_ROW 64u
#define DS4_Q8_T_LDS_PITCH (DS4_Q8_T_TILE_ROW + 8u)

__global__ static void dequant_q8_0_to_f16_kernel(__half *out,
                                                  const unsigned char *w,
                                                  uint64_t in_dim,
                                                  uint64_t out_dim,
                                                  uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(float *out,
                                                  const unsigned char *w,
                                                  uint64_t in_dim,
                                                  uint64_t out_dim,
                                                  uint64_t blocks);
