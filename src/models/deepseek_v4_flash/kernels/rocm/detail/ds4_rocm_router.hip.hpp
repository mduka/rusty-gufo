#if __clang_major__ >= 24
extern "C" __device__ __attribute__((pure)) double __ocml_exp_f64(double);
extern "C" __device__ __attribute__((pure)) double __ocml_log1p_f64(double);
extern "C" __device__ __attribute__((const)) double __ocml_sqrt_f64(double);

__device__ static float router_probability(float input) {
    // Round only the final score. Intermediate F32 transcendental rounding
    // varies across ROCm libraries and can change routing/quantization later.
    const double x = input;
    const double softplus = x > 20.0 ? x :
        x < -10.0 ? __ocml_exp_f64(x) :
                    __ocml_log1p_f64(__ocml_exp_f64(x));
    return static_cast<float>(__ocml_sqrt_f64(softplus));
}
#else
__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -10.0f) return ds4_precise_expf(x);
    return ds4_precise_log1pf(ds4_precise_expf(x));
}
#endif

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
#if __clang_major__ >= 24
#pragma clang fp reassociate(off)
#endif
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * DS4_ROCM_N_EXPERT;
    float *prob = probs + (uint64_t)t * DS4_ROCM_N_EXPERT;
    int32_t *sel = selected + (uint64_t)t * DS4_ROCM_N_EXPERT_USED;
    float *w = weights + (uint64_t)t * DS4_ROCM_N_EXPERT_USED;
    __shared__ float sprob[4][DS4_ROCM_N_EXPERT];
    float local_prob[DS4_ROCM_N_EXPERT / 32u];
    float local_score[DS4_ROCM_N_EXPERT / 32u];

    #pragma unroll
    for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT / 32u; j++) {
        const uint32_t e = lane + j * 32u;
#if __clang_major__ >= 24
        const float p = router_probability(log[e]);
#else
        const float p = ds4_precise_sqrtf(softplus_dev(log[e]));
#endif
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t] : token_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * DS4_ROCM_N_EXPERT_USED;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT_USED; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < DS4_ROCM_N_EXPERT) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT_USED; j++) w[j] = w[j] / sum * DS4_ROCM_EXPERT_WEIGHT_SCALE;
        }
        return;
    }

    float out_prob[DS4_ROCM_N_EXPERT_USED] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[DS4_ROCM_N_EXPERT_USED] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < DS4_ROCM_N_EXPERT_USED; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT / 32u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(FULL_WARP_MASK, best_score, mask);
            const float other_prob = __shfl_xor_sync(FULL_WARP_MASK, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(FULL_WARP_MASK, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT / 32u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT_USED; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT_USED; j++) w[j] = w[j] / sum * DS4_ROCM_EXPERT_WEIGHT_SCALE;
    }
}

