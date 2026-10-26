#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <type_traits>
#include <rocwmma/rocwmma.hpp>

/* Blocked F16 GEMM for the attention output B projection.
 *
 * The engine sends this shape (`m=4096 n=chunk k=8192`, `opA=N`, F16 operands
 * and an F32 result) to hipBLASLt, which picks a `MT96x96x32` Tensile kernel.
 * That tile re-reads each panel `m/96` and `n/96` times, about 5.7 GB for a
 * 4,096-token chunk, and a kernel trace prices the call at 25.58 ms, which is
 * roughly DRAM peak for that traffic. A 256x128 macro tile moves 3.22 GB
 * instead. `rocm/tools/f16_gemm_bench.cpp` scored every geometry with three
 * rotating operand copies so that the 32 MB MALL could not hide the re-reads:
 * 15.82 ms for the production candidate against 11.80 ms here.
 *
 * This is the only projection shape worth taking off hipBLASLt. Tensile wins
 * every `opA=T` shape in the same harness, by 1.4x to 2.6x.
 *
 * Layout: A is the transposed weight cache `Wt[k][m]` row-major, so the LDS
 * tile is `[k][m]` with a `col_major` fragment and both the global read and the
 * LDS write stay contiguous in `m`. B is `low[n][k]` row-major, staged `[n][k]`
 * with a `col_major` matrix_b. C is `out[n][m]` row-major, which is column-major
 * `[m][n]` with leading dimension `m`.
 *
 * Accumulators live in registers for the whole K loop, so each output element
 * is a single ascending walk over K: deterministic, though not hipBLASLt's
 * order, so this needs a likelihood gate rather than an exact logit hash. */
template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WMF, uint32_t WNF,
          uint32_t SWZ, bool A_ROWMAJOR, typename OutT>
__global__ __launch_bounds__((BM / 16u / WMF) * (BN / 16u / WNF) * 32u) void
ds4_gemm_f16_wmma_kernel(OutT *__restrict__ out,
                         const __half *__restrict__ weight,
                         const __half *__restrict__ act,
                         uint32_t m,
                         uint32_t n,
                         uint32_t k,
                         uint32_t ldc,
                         uint64_t stride_a,
                         uint64_t stride_b,
                         uint64_t stride_c) {
    /* `blockIdx.z` walks independent problems for the strided-batched form; a
     * plain GEMM passes a z extent of one and zero strides. `ldc` lets the
     * result land in a wider matrix, which the attention output A GEMM needs
     * because each group writes its own `rank` block of a `low_dim` row. */
    const __half *__restrict__ weight_t = weight + (uint64_t)blockIdx.z * stride_a;
    const __half *__restrict__ low_h = act + (uint64_t)blockIdx.z * stride_b;
    out += (uint64_t)blockIdx.z * stride_c;
    constexpr uint32_t WAVES_M = BM / 16u / WMF;
    constexpr uint32_t WAVES_N = BN / 16u / WNF;
    constexpr uint32_t THREADS = WAVES_M * WAVES_N * 32u;
    /* A's LDS tile follows the source's contiguous axis: [m][k] for a row-major
     * W[m][k] (opA=T) and [k][m] for a row-major Wt[k][m] (opA=N), so both the
     * global read and the LDS write stay contiguous either way. */
    constexpr uint32_t A_PITCH = A_ROWMAJOR ? BK : BM;
    constexpr uint32_t A_LINES = A_ROWMAJOR ? BM : BK;
    __shared__ __half shA[A_LINES * A_PITCH];
    __shared__ __half shB[BN * BK];
    /* Narrowing epilogue scratch, one 16x16 F32 tile per wave. Only the F16
     * result path needs it; the F32 path stores straight to global. */
    __shared__ float shEpi[std::is_same<OutT, __half>::value
                                   ? WAVES_M * WAVES_N * 256u
                                   : 1u];

    uint32_t bm, bn;
    if (SWZ <= 1u) {
        bm = blockIdx.x;
        bn = blockIdx.y;
    } else {
        const uint32_t tiles_m = gridDim.x;
        const uint32_t group = SWZ * tiles_m;
        const uint32_t linear = blockIdx.y * tiles_m + blockIdx.x;
        const uint32_t gid = linear / group;
        const uint32_t within = linear - gid * group;
        const uint32_t cols = min(SWZ, gridDim.y - gid * SWZ);
        bm = within / cols;
        bn = gid * SWZ + (within - bm * cols);
    }
    const uint32_t m0 = bm * BM;
    const uint32_t n0 = bn * BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t wm = wave % WAVES_M;
    const uint32_t wn = wave / WAVES_M;

    using frag_a = rocwmma::fragment<
            rocwmma::matrix_a, 16, 16, 16, __half,
            typename std::conditional<A_ROWMAJOR, rocwmma::row_major,
                                      rocwmma::col_major>::type>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half,
                                     rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    frag_c acc[WMF][WNF];
    for (uint32_t i = 0; i < WMF; i++)
        for (uint32_t j = 0; j < WNF; j++) rocwmma::fill_fragment(acc[i][j], 0.0f);

    constexpr uint32_t A_GROUPS = BM * BK / 8u;
    constexpr uint32_t B_GROUPS = BN * BK / 8u;
    constexpr uint32_t A_RUN = A_PITCH / 8u;
    constexpr uint32_t B_RUN = BK / 8u;

    for (uint32_t k0 = 0; k0 < k; k0 += BK) {
        for (uint32_t g = tid; g < A_GROUPS; g += THREADS) {
            const uint32_t run = g % A_RUN;
            const uint32_t line = g / A_RUN;
            const uint64_t src = A_ROWMAJOR
                    ? (uint64_t)(m0 + line) * k + k0 + run * 8u
                    : (uint64_t)(k0 + line) * m + m0 + run * 8u;
            *reinterpret_cast<uint4 *>(&shA[line * A_PITCH + run * 8u]) =
                    *reinterpret_cast<const uint4 *>(&weight_t[src]);
        }
        for (uint32_t g = tid; g < B_GROUPS; g += THREADS) {
            const uint32_t run = g % B_RUN;
            const uint32_t line = g / B_RUN;
            *reinterpret_cast<uint4 *>(&shB[line * BK + run * 8u]) =
                    *reinterpret_cast<const uint4 *>(
                            &low_h[(uint64_t)(n0 + line) * k + k0 + run * 8u]);
        }
        __syncthreads();
        for (uint32_t kk = 0; kk < BK; kk += 16u) {
            frag_a fa[WMF];
            frag_b fb[WNF];
            for (uint32_t i = 0; i < WMF; i++) {
                const uint32_t mm = (wm * WMF + i) * 16u;
                if (A_ROWMAJOR) {
                    rocwmma::load_matrix_sync(fa[i], &shA[mm * A_PITCH + kk],
                                              A_PITCH);
                } else {
                    rocwmma::load_matrix_sync(fa[i], &shA[kk * A_PITCH + mm],
                                              A_PITCH);
                }
            }
            for (uint32_t j = 0; j < WNF; j++) {
                rocwmma::load_matrix_sync(
                        fb[j], &shB[(wn * WNF + j) * 16u * BK + kk], BK);
            }
            for (uint32_t i = 0; i < WMF; i++)
                for (uint32_t j = 0; j < WNF; j++)
                    rocwmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }

    if (std::is_same<OutT, float>::value) {
        for (uint32_t i = 0; i < WMF; i++) {
            for (uint32_t j = 0; j < WNF; j++) {
                const uint32_t mo = m0 + (wm * WMF + i) * 16u;
                const uint32_t no = n0 + (wn * WNF + j) * 16u;
                rocwmma::store_matrix_sync(
                        (float *)out + mo + (uint64_t)no * ldc, acc[i][j], ldc,
                        rocwmma::mem_col_major);
            }
        }
        return;
    }
    /* An F16 result needs a narrowing epilogue that rocWMMA's F32 accumulator
     * cannot store directly, so page each wave's tiles back through LDS. */
    float *tile = shEpi + (uint64_t)wave * 256u;
    const uint32_t lane = tid & 31u;
    for (uint32_t i = 0; i < WMF; i++) {
        for (uint32_t j = 0; j < WNF; j++) {
            rocwmma::store_matrix_sync(tile, acc[i][j], 16u,
                                       rocwmma::mem_col_major);
            const uint32_t mo = m0 + (wm * WMF + i) * 16u;
            const uint32_t no = n0 + (wn * WNF + j) * 16u;
            for (uint32_t e = lane; e < 256u; e += 32u) {
                const uint32_t c = e >> 4u;
                const uint32_t r = e & 15u;
                ((__half *)out)[(uint64_t)(no + c) * ldc + mo + r] =
                        __float2half(tile[c * 16u + r]);
            }
        }
    }
}

/* 256x128 with four m-fragments and two n-fragments per wave, 512 threads, and
 * a two-column swizzle scored best across every shape in
 * `rocm/tools/f16_gemm_bench.cpp`. Bigger macro tiles spill: 256x256 needs 128
 * accumulator VGPRs and measured 1.8x slower. */
#define DS4_WMMA_GEMM_BM 256u
#define DS4_WMMA_GEMM_BN 128u
#define DS4_WMMA_GEMM_BK 32u

/* A 256x128 tile only pays when the grid can fill the 40 CUs several times
 * over. Routing every `opA=T` projection through it cost +456 ms per two-chunk
 * trace, because `m=256` and `m=512` leave 32 and 64 workgroups for 40 CUs while
 * Tensile's MT96x96 and MT32x32 tiles keep hundreds in flight. Six workgroups
 * per CU is the floor that kept only the shapes this kernel wins. */
#define DS4_WMMA_GEMM_MIN_BLOCKS 240u

/* Narrow-n companion tile, and the padding rule that lets an arbitrary chunk
 * width reach either of them.
 *
 * `DS4_WMMA_GEMM_MIN_BLOCKS` was calibrated to keep this kernel off the small-m
 * `opA=T` projections, where 32 to 64 workgroups lose to Tensile. It also
 * excluded the one shape the kernel was written for whenever the prompt chunk
 * was under 2,048 rows -- which is every conversational turn -- and there the
 * fallback is not Tensile's good `opA=T` tile but rocBLAS on `opA=N`.
 * `tools/bench/dsv4_attn_out_gemm_bench.hip` at `m=4096 k=8192`:
 *
 *   n      rocBLAS  256x128 W4x2  128x128 W2x2 SWZ8
 *    128  1.533 ms      0.934 ms           0.653 ms
 *    256  2.879 ms      1.018 ms           0.949 ms
 *    512  5.202 ms      2.094 ms           1.887 ms
 *   1024 10.521 ms      3.505 ms           4.042 ms
 *
 * rocBLAS holds 5.6-6.6 TFLOP/s at every width while both tiles reach 16-18, so
 * the block floor was giving away a factor of three; 128x128 wins below n=1024
 * and 256x128 above it. Every WMMA geometry in that harness agrees bit for bit:
 * the macro tile only partitions the output, and each element still walks K once
 * in ascending order, so this is a scheduling choice with no numerical
 * consequence. Only leaving rocBLAS is a numerical change, and at these widths
 * the two disagree by 3e-4.
 *
 * A width that is not a multiple of the tile is rounded up. The caller owns
 * both the activation staging buffer and the result, so it sizes them for the
 * padded width and zeroes the activation pad; the extra result columns are
 * computed from zeros and never read. */
#define DS4_WMMA_GEMM_NARROW_BM 128u
#define DS4_WMMA_GEMM_NARROW_BN 128u
#define DS4_WMMA_GEMM_WIDE_MIN_N 1024u

/* Strided-batched form for the attention output A GEMM.
 *
 * `hipblasGemmStridedBatchedEx` sends `rank x n_tokens x group_dim` once per
 * head group to rocBLAS, which picks `Cijk_Alik_Bljk_HHS_MT128x128x32` and
 * reaches 2.7 TFLOP/s: 12.88 ms per call for 34.4 GFLOP. Each group's A panel
 * is only `rank * group_dim * 2` bytes, so it stays in the MALL across the whole
 * n sweep and the call should be bound by B, which is read once.
 *
 * A 128-row macro tile matches `rank` exactly, so `m` needs no padding, and the
 * output leading dimension is `low_dim` because group `g` owns the `rank` block
 * at offset `g * rank` of every token's row.
 *
 * `m` must equal the tile: relaxing it to any multiple, which lets `rank=1024`
 * through, measured 855.99 ms per 4,096-token chunk against rocBLAS's 563.72 ms
 * for the same call. Each group's A panel then no longer stays MALL-resident
 * across the n sweep, which is the whole premise of the tile. */
#define DS4_WMMA_BATCH_BM 128u
#define DS4_WMMA_BATCH_BN 128u

#endif
__global__ static void matmul_f16_tiny_batch_wave_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (row >= out_dim || tok >= n_tok) return;

    const __half *wr = w + (uint64_t)row * in_dim;
    const float *xr = x + (uint64_t)tok * in_dim;
    float sum = 0.0f;
    for (uint32_t i = lane; i < in_dim; i += 32u) {
        const float xv = __half2float(__float2half(xr[i]));
        sum += __half2float(wr[i]) * xv;
    }
    sum = warp_sum_f32(sum);
    if (lane == 0u) {
        out[(uint64_t)tok * out_dim + row] = sum;
    }
}

