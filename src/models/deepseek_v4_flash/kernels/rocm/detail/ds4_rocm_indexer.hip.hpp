#include "ds4_dspark_sampling.hip.hpp"
#include "ds4_rocm_indexer_score.hip.hpp"
#include "ds4_rocm_indexer_topk.hip.hpp"

template <typename Kernel>
static hipError_t ds4_hip_set_dynamic_shared_memory(Kernel kernel,
                                                     int bytes) {
    return hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                               hipFuncAttributeMaxDynamicSharedMemorySize,
                               bytes);
}

__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}

__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) dot += qh[d] * kh[d];
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}

/* DSpark Markov correction: select argmax(logits + W2 * W1[prev]) without
 * moving the vocabulary row back to the host. W1 and W2 are Q8_0. */
__global__ static void dspark_markov_argmax_kernel(
        unsigned long long *out_key,
        const float *logits,
        const unsigned char *w1_row,
        const unsigned char *w2,
        uint32_t vocab,
        uint32_t rank_blocks) {
    __shared__ float state[256];
    const uint32_t tid = threadIdx.x;
    if (tid < rank_blocks * 32u) {
        const uint32_t block = tid >> 5u;
        const uint32_t lane = tid & 31u;
        const unsigned char *qblock = w1_row + (uint64_t)block * 34u;
        const float scale = __half2float(*(const __half *)qblock);
        state[tid] =
            scale * (float)((const int8_t *)(qblock + 2u))[lane];
    }
    __syncthreads();

    float best_value = -INFINITY;
    uint32_t best_index = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < vocab;
         i += gridDim.x * blockDim.x) {
        const unsigned char *row =
            w2 + (uint64_t)i * rank_blocks * 34u;
        float acc = 0.0f;
        for (uint32_t block = 0; block < rank_blocks; block++) {
            const unsigned char *qblock = row + (uint64_t)block * 34u;
            const float scale = __half2float(*(const __half *)qblock);
            const int8_t *quants = (const int8_t *)(qblock + 2u);
            float sum = 0.0f;
#pragma unroll
            for (uint32_t lane = 0; lane < 32u; lane++) {
                sum += (float)quants[lane] * state[block * 32u + lane];
            }
            acc += scale * sum;
        }
        const float value = logits[i] + acc;
        if (topk_score_better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    __shared__ float values[256];
    __shared__ uint32_t indices[256];
    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride &&
            topk_score_better(values[tid + stride], indices[tid + stride],
                              values[tid], indices[tid])) {
            values[tid] = values[tid + stride];
            indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        const unsigned int bits = __float_as_uint(values[0]);
        const unsigned int value_key =
            (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        const unsigned long long key =
            ((unsigned long long)value_key << 32) |
            (unsigned int)(~indices[0]);
        atomicMax(out_key, key);
    }
}

/*
 * Resolve one DSpark position for several requests while streaming Markov W2
 * once. Each request keeps the scalar kernel's block/lane accumulation and
 * reduction order; only the independent request dimension is fused.
 */
template <uint32_t N_ROWS>
__global__ static void dspark_markov_argmax_batch_kernel(
        unsigned long long *out_keys,
        const float *logits,
        uint64_t logits_row_stride,
        const int32_t *previous_tokens,
        const unsigned char *w1,
        const unsigned char *w2,
        uint32_t vocab,
        uint32_t rank_blocks) {
    __shared__ float states[N_ROWS][256];
    __shared__ float values[N_ROWS][256];
    __shared__ uint32_t indices[N_ROWS][256];
    const uint32_t tid = threadIdx.x;
    const uint64_t row_bytes = (uint64_t)rank_blocks * 34u;

#pragma unroll
    for (uint32_t request = 0; request < N_ROWS; ++request) {
        if (tid < rank_blocks * 32u) {
            const uint32_t block = tid >> 5u;
            const uint32_t lane = tid & 31u;
            const unsigned char *qblock =
                w1 + (uint64_t)(uint32_t)previous_tokens[request] * row_bytes +
                (uint64_t)block * 34u;
            const float scale = __half2float(*(const __half *)qblock);
            states[request][tid] =
                scale * (float)((const int8_t *)(qblock + 2u))[lane];
        }
    }
    __syncthreads();

    float best_values[N_ROWS];
    uint32_t best_indices[N_ROWS];
#pragma unroll
    for (uint32_t request = 0; request < N_ROWS; ++request) {
        best_values[request] = -INFINITY;
        best_indices[request] = 0u;
    }

    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < vocab;
         i += gridDim.x * blockDim.x) {
        const unsigned char *row = w2 + (uint64_t)i * row_bytes;
        float acc[N_ROWS] = {};
        for (uint32_t block = 0; block < rank_blocks; ++block) {
            const unsigned char *qblock = row + (uint64_t)block * 34u;
            const float scale = __half2float(*(const __half *)qblock);
            const int8_t *quants = (const int8_t *)(qblock + 2u);
            float sums[N_ROWS] = {};
#pragma unroll
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                const float quant = (float)quants[lane];
#pragma unroll
                for (uint32_t request = 0; request < N_ROWS; ++request) {
                    sums[request] +=
                        quant * states[request][block * 32u + lane];
                }
            }
#pragma unroll
            for (uint32_t request = 0; request < N_ROWS; ++request) {
                acc[request] += scale * sums[request];
            }
        }
#pragma unroll
        for (uint32_t request = 0; request < N_ROWS; ++request) {
            const float value =
                logits[(uint64_t)request * logits_row_stride + i] +
                acc[request];
            if (topk_score_better(value, i, best_values[request],
                                  best_indices[request])) {
                best_values[request] = value;
                best_indices[request] = i;
            }
        }
    }

#pragma unroll
    for (uint32_t request = 0; request < N_ROWS; ++request) {
        values[request][tid] = best_values[request];
        indices[request][tid] = best_indices[request];
    }
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
#pragma unroll
            for (uint32_t request = 0; request < N_ROWS; ++request) {
                if (topk_score_better(
                        values[request][tid + stride],
                        indices[request][tid + stride],
                        values[request][tid],
                        indices[request][tid])) {
                    values[request][tid] =
                        values[request][tid + stride];
                    indices[request][tid] =
                        indices[request][tid + stride];
                }
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
#pragma unroll
        for (uint32_t request = 0; request < N_ROWS; ++request) {
            const unsigned int bits =
                __float_as_uint(values[request][0]);
            const unsigned int value_key =
                (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
            const unsigned long long key =
                ((unsigned long long)value_key << 32) |
                (unsigned int)(~indices[request][0]);
            atomicMax(out_keys + request, key);
        }
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

/* Decode the packed (value, index) key produced by the Markov argmax reduce. */
__global__ static void dspark_markov_key_decode_kernel(
        int32_t *out_index,
        const unsigned long long *key) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    out_index[0] = (int32_t)(~(unsigned int)(key[0] & 0xffffffffull));
}

__global__ static void dspark_markov_key_decode_batch_kernel(
        int32_t *out_indices,
        const unsigned long long *keys,
        uint32_t n_rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    out_indices[row] =
        (int32_t)(~(unsigned int)(keys[row] & 0xffffffffull));
}

