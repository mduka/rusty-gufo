// C ABI over the hipBLAS / hipBLASLt / rocBLAS subset gufo uses
// (docs/RUST.md).
//
// With shim/device.cpp one of the two hand-written C++ host files besides
// kernel launchers. It exists so that the compiler checks every call against
// the installed ROCm headers and so that the C++-only hipblaslt_ext functions
// are reachable from Rust.
// Every function is noexcept, returns a hipblasStatus_t (or -1 for a caught
// C++ exception), and takes handles and enums as plain C types.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <hipblaslt/hipblaslt-ext.hpp>

namespace {

// Rust keeps an algorithm as opaque bytes of this size.
constexpr std::size_t kAlgoBytes = 32;
static_assert(sizeof(hipblasLtMatmulAlgo_t) <= kAlgoBytes);
static_assert(sizeof(hipblasOperation_t) == sizeof(std::int32_t));
static_assert(sizeof(hipDataType) == sizeof(std::int32_t));
static_assert(sizeof(hipblasComputeType_t) == sizeof(std::int32_t));
static_assert(sizeof(hipblasStatus_t) == sizeof(std::int32_t));
static_assert(sizeof(rocblas_operation) == sizeof(std::int32_t));
static_assert(sizeof(rocblas_status) == sizeof(std::int32_t));

constexpr std::int32_t kException = -1;

hipblasLtMatmulAlgo_t ToAlgo(const std::uint8_t* bytes) {
  hipblasLtMatmulAlgo_t algo{};
  std::memcpy(&algo, bytes, sizeof(algo));
  return algo;
}

}  // namespace

extern "C" {

struct GufoBlasLtHeuristic {
  std::uint8_t algo[kAlgoBytes];
  std::uint64_t workspace_bytes;
  std::int32_t state;
  std::int32_t index;
  float waves;
};

// Atomics stay disabled: hipBLAS may otherwise pick kernels whose
// accumulation order varies from run to run.
std::int32_t gufo_blas_create(void** handle, void* stream) noexcept {
  auto status = hipblasCreate(reinterpret_cast<hipblasHandle_t*>(handle));
  if (status == HIPBLAS_STATUS_SUCCESS) {
    status = hipblasSetStream(static_cast<hipblasHandle_t>(*handle),
                              static_cast<hipStream_t>(stream));
    if (status == HIPBLAS_STATUS_SUCCESS) {
      status = hipblasSetAtomicsMode(static_cast<hipblasHandle_t>(*handle),
                                     HIPBLAS_ATOMICS_NOT_ALLOWED);
    }
    if (status != HIPBLAS_STATUS_SUCCESS) {
      (void)hipblasDestroy(static_cast<hipblasHandle_t>(*handle));
      *handle = nullptr;
    }
  }
  return status;
}

// A handle with hipBLAS's default atomics mode, for engines whose C++ code
// leaves it unset (Qwen-Image-2.1).
std::int32_t gufo_blas_create_default(void** handle, void* stream) noexcept {
  auto status = hipblasCreate(reinterpret_cast<hipblasHandle_t*>(handle));
  if (status == HIPBLAS_STATUS_SUCCESS) {
    status = hipblasSetStream(static_cast<hipblasHandle_t>(*handle),
                              static_cast<hipStream_t>(stream));
    if (status != HIPBLAS_STATUS_SUCCESS) {
      (void)hipblasDestroy(static_cast<hipblasHandle_t>(*handle));
      *handle = nullptr;
    }
  }
  return status;
}

// rocBLAS f32 GEMM exactly as the C++ Qwen3-TTS speech decoder and encoders
// call it (`rocblas_gemm_ex`, f32 compute, standard algorithm, D = C), on a
// handle bound to one stream with atomics disabled.
std::int32_t gufo_rocblas_create(void** handle, void* stream) noexcept {
  auto status =
      rocblas_create_handle(reinterpret_cast<rocblas_handle*>(handle));
  if (status == rocblas_status_success) {
    status = rocblas_set_atomics_mode(static_cast<rocblas_handle>(*handle),
                                      rocblas_atomics_not_allowed);
    if (status == rocblas_status_success) {
      status = rocblas_set_stream(static_cast<rocblas_handle>(*handle),
                                  static_cast<hipStream_t>(stream));
    }
    if (status != rocblas_status_success) {
      (void)rocblas_destroy_handle(static_cast<rocblas_handle>(*handle));
      *handle = nullptr;
    }
  }
  return status;
}

std::int32_t gufo_rocblas_destroy(void* handle) noexcept {
  return rocblas_destroy_handle(static_cast<rocblas_handle>(handle));
}

std::int32_t gufo_rocblas_gemm_f32(void* handle, std::int32_t trans_a,
                                   std::int32_t trans_b, std::int32_t m,
                                   std::int32_t n, std::int32_t k,
                                   const float* a, std::int32_t lda,
                                   const float* b, std::int32_t ldb, float* c,
                                   std::int32_t ldc) noexcept {
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  return rocblas_gemm_ex(
      static_cast<rocblas_handle>(handle),
      static_cast<rocblas_operation>(trans_a),
      static_cast<rocblas_operation>(trans_b), m, n, k, &alpha, a,
      rocblas_datatype_f32_r, lda, b, rocblas_datatype_f32_r, ldb, &beta, c,
      rocblas_datatype_f32_r, ldc, c, rocblas_datatype_f32_r, ldc,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
}

std::int32_t gufo_blas_destroy(void* handle) noexcept {
  return hipblasDestroy(static_cast<hipblasHandle_t>(handle));
}

std::int32_t gufo_blas_gemm_ex(
    void* handle, std::int32_t trans_a, std::int32_t trans_b, std::int32_t m,
    std::int32_t n, std::int32_t k, const float* alpha, const void* a,
    std::int32_t a_type, std::int32_t lda, const void* b, std::int32_t b_type,
    std::int32_t ldb, const float* beta, void* c, std::int32_t c_type,
    std::int32_t ldc, std::int32_t compute, std::int32_t algo) noexcept {
  return hipblasGemmEx(static_cast<hipblasHandle_t>(handle),
                       static_cast<hipblasOperation_t>(trans_a),
                       static_cast<hipblasOperation_t>(trans_b), m, n, k, alpha,
                       a, static_cast<hipDataType>(a_type), lda, b,
                       static_cast<hipDataType>(b_type), ldb, beta, c,
                       static_cast<hipDataType>(c_type), ldc,
                       static_cast<hipblasComputeType_t>(compute),
                       static_cast<hipblasGemmAlgo_t>(algo));
}

std::int32_t gufo_blas_gemm_strided_batched_ex(
    void* handle, std::int32_t trans_a, std::int32_t trans_b, std::int32_t m,
    std::int32_t n, std::int32_t k, const float* alpha, const void* a,
    std::int32_t a_type, std::int32_t lda, std::int64_t stride_a, const void* b,
    std::int32_t b_type, std::int32_t ldb, std::int64_t stride_b,
    const float* beta, void* c, std::int32_t c_type, std::int32_t ldc,
    std::int64_t stride_c, std::int32_t batch_count, std::int32_t compute,
    std::int32_t algo) noexcept {
  return hipblasGemmStridedBatchedEx(
      static_cast<hipblasHandle_t>(handle),
      static_cast<hipblasOperation_t>(trans_a),
      static_cast<hipblasOperation_t>(trans_b), m, n, k, alpha, a,
      static_cast<hipDataType>(a_type), lda, stride_a, b,
      static_cast<hipDataType>(b_type), ldb, stride_b, beta, c,
      static_cast<hipDataType>(c_type), ldc, stride_c, batch_count,
      static_cast<hipblasComputeType_t>(compute),
      static_cast<hipblasGemmAlgo_t>(algo));
}

std::int32_t gufo_blas_sgemm(void* handle, std::int32_t trans_a,
                             std::int32_t trans_b, std::int32_t m,
                             std::int32_t n, std::int32_t k, const float* alpha,
                             const float* a, std::int32_t lda, const float* b,
                             std::int32_t ldb, const float* beta, float* c,
                             std::int32_t ldc) noexcept {
  return hipblasSgemm(static_cast<hipblasHandle_t>(handle),
                      static_cast<hipblasOperation_t>(trans_a),
                      static_cast<hipblasOperation_t>(trans_b), m, n, k, alpha,
                      a, lda, b, ldb, beta, c, ldc);
}

std::int32_t gufo_blas_sgemm_strided_batched(
    void* handle, std::int32_t trans_a, std::int32_t trans_b, std::int32_t m,
    std::int32_t n, std::int32_t k, const float* alpha, const float* a,
    std::int32_t lda, std::int64_t stride_a, const float* b, std::int32_t ldb,
    std::int64_t stride_b, const float* beta, float* c, std::int32_t ldc,
    std::int64_t stride_c, std::int32_t batch_count) noexcept {
  return hipblasSgemmStridedBatched(
      static_cast<hipblasHandle_t>(handle),
      static_cast<hipblasOperation_t>(trans_a),
      static_cast<hipblasOperation_t>(trans_b), m, n, k, alpha, a, lda,
      stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count);
}

std::int32_t gufo_blaslt_create(void** handle) noexcept {
  return hipblasLtCreate(reinterpret_cast<hipblasLtHandle_t*>(handle));
}

std::int32_t gufo_blaslt_destroy(void* handle) noexcept {
  return hipblasLtDestroy(static_cast<hipblasLtHandle_t>(handle));
}

std::int32_t gufo_blaslt_desc_create(void** desc, std::int32_t compute,
                                     std::int32_t scale_type) noexcept {
  return hipblasLtMatmulDescCreate(
      reinterpret_cast<hipblasLtMatmulDesc_t*>(desc),
      static_cast<hipblasComputeType_t>(compute),
      static_cast<hipDataType>(scale_type));
}

// `operand` 0 sets TRANSA, 1 sets TRANSB. Only attributes a caller sets are
// touched: setting the default explicitly changes hipBLASLt's heuristics.
std::int32_t gufo_blaslt_desc_set_transpose(void* desc, std::int32_t operand,
                                            std::int32_t op) noexcept {
  const auto value = static_cast<hipblasOperation_t>(op);
  const auto attribute = operand == 0 ? HIPBLASLT_MATMUL_DESC_TRANSA
                                      : HIPBLASLT_MATMUL_DESC_TRANSB;
  return hipblasLtMatmulDescSetAttribute(
      static_cast<hipblasLtMatmulDesc_t>(desc), attribute, &value,
      sizeof(value));
}

std::int32_t gufo_blaslt_desc_destroy(void* desc) noexcept {
  return hipblasLtMatmulDescDestroy(static_cast<hipblasLtMatmulDesc_t>(desc));
}

std::int32_t gufo_blaslt_layout_create(void** layout, std::int32_t type,
                                       std::uint64_t rows, std::uint64_t cols,
                                       std::int64_t ld) noexcept {
  return hipblasLtMatrixLayoutCreate(
      reinterpret_cast<hipblasLtMatrixLayout_t*>(layout),
      static_cast<hipDataType>(type), rows, cols, ld);
}

std::int32_t gufo_blaslt_layout_set_batch(void* layout, std::int32_t count,
                                          std::int64_t stride) noexcept {
  auto matrix = static_cast<hipblasLtMatrixLayout_t>(layout);
  auto status = hipblasLtMatrixLayoutSetAttribute(
      matrix, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count));
  if (status == HIPBLAS_STATUS_SUCCESS)
    status = hipblasLtMatrixLayoutSetAttribute(
        matrix, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
        sizeof(stride));
  return status;
}

std::int32_t gufo_blaslt_layout_destroy(void* layout) noexcept {
  return hipblasLtMatrixLayoutDestroy(
      static_cast<hipblasLtMatrixLayout_t>(layout));
}

// Ranked heuristic results for D = C layout; also reports each algorithm's
// solution index (hipblaslt_ext::getIndexFromAlgo).
std::int32_t gufo_blaslt_heuristics(void* handle, void* desc, void* a, void* b,
                                    void* c, std::uint64_t max_workspace,
                                    std::int32_t requested,
                                    GufoBlasLtHeuristic* results,
                                    std::int32_t* found) noexcept {
  try {
    *found = 0;
    hipblasLtMatmulPreference_t preference = nullptr;
    auto status = hipblasLtMatmulPreferenceCreate(&preference);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      return status;
    }
    status = hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace,
        sizeof(max_workspace));
    constexpr int kMaxRequested = 64;
    hipblasLtMatmulHeuristicResult_t raw[kMaxRequested]{};
    const int count = requested < kMaxRequested ? requested : kMaxRequested;
    int returned = 0;
    if (status == HIPBLAS_STATUS_SUCCESS) {
      status = hipblasLtMatmulAlgoGetHeuristic(
          static_cast<hipblasLtHandle_t>(handle),
          static_cast<hipblasLtMatmulDesc_t>(desc),
          static_cast<hipblasLtMatrixLayout_t>(a),
          static_cast<hipblasLtMatrixLayout_t>(b),
          static_cast<hipblasLtMatrixLayout_t>(c),
          static_cast<hipblasLtMatrixLayout_t>(c), preference, count, raw,
          &returned);
    }
    (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      return status;
    }
    for (int index = 0; index < returned; ++index) {
      GufoBlasLtHeuristic& out = results[index];
      std::memset(&out, 0, sizeof(out));
      std::memcpy(out.algo, &raw[index].algo, sizeof(raw[index].algo));
      out.workspace_bytes = raw[index].workspaceSize;
      out.state = raw[index].state;
      out.waves = raw[index].wavesCount;
      out.index = hipblaslt_ext::getIndexFromAlgo(raw[index].algo);
    }
    *found = returned;
    return HIPBLAS_STATUS_SUCCESS;
  } catch (...) {
    return kException;
  }
}

// Whether hipBLASLt can run `algo_bytes` for the matmul (alpha 1, beta 0,
// C as D); `workspace` receives the bytes it would need
// (`hipblaslt_ext::matmulIsAlgoSupported`).
std::int32_t gufo_blaslt_is_supported(void* handle, void* desc, void* a,
                                      void* b, void* c,
                                      const std::uint8_t* algo_bytes,
                                      std::uint64_t* workspace) noexcept {
  try {
    auto algo = ToAlgo(algo_bytes);
    const float one = 1.0F;
    const float zero = 0.0F;
    std::size_t bytes = 0;
    const auto status = hipblaslt_ext::matmulIsAlgoSupported(
        static_cast<hipblasLtHandle_t>(handle),
        static_cast<hipblasLtMatmulDesc_t>(desc), &one,
        static_cast<hipblasLtMatrixLayout_t>(a),
        static_cast<hipblasLtMatrixLayout_t>(b), &zero,
        static_cast<hipblasLtMatrixLayout_t>(c),
        static_cast<hipblasLtMatrixLayout_t>(c), algo, bytes);
    *workspace = bytes;
    return status;
  } catch (...) {
    return kException;
  }
}

// Resolves the solution and kernel names of an installed algorithm. The
// lookups initialise hipBLASLt's solution library lazily, which changes the
// heuristic lists of later shapes, so the C++ engines' plan selection relies
// on them (see src/models/qwen3_asr/hip/blas.cpp).
std::int32_t gufo_blaslt_resolve_names(
    void* handle, const std::uint8_t* algo_bytes) noexcept {
  try {
    auto algo = ToAlgo(algo_bytes);
    const auto lt = static_cast<hipblasLtHandle_t>(handle);
    (void)hipblaslt_ext::getSolutionNameFromAlgo(lt, algo);
    (void)hipblaslt_ext::getKernelNameFromAlgo(lt, algo);
    return HIPBLAS_STATUS_SUCCESS;
  } catch (...) {
    return kException;
  }
}

// The kernel name of an installed algorithm
// (`hipblaslt_ext::getKernelNameFromAlgo`), truncated to `capacity` bytes;
// `length` receives its full length. Like gufo_blaslt_resolve_names, the
// lookup initialises hipBLASLt's solution library.
std::int32_t gufo_blaslt_kernel_name(void* handle,
                                     const std::uint8_t* algo_bytes, char* name,
                                     std::uint64_t capacity,
                                     std::uint64_t* length) noexcept {
  try {
    auto algo = ToAlgo(algo_bytes);
    const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(
        static_cast<hipblasLtHandle_t>(handle), algo);
    *length = kernel.size();
    std::memcpy(name, kernel.data(),
                std::min<std::uint64_t>(capacity, kernel.size()));
    return HIPBLAS_STATUS_SUCCESS;
  } catch (...) {
    return kException;
  }
}

std::int32_t gufo_blaslt_matmul(void* handle, void* desc, const float* alpha,
                                const void* a, void* a_layout, const void* b,
                                void* b_layout, const float* beta, void* c,
                                void* c_layout, const std::uint8_t* algo_bytes,
                                void* workspace, std::uint64_t workspace_bytes,
                                void* stream) noexcept {
  const auto algo = ToAlgo(algo_bytes);
  return hipblasLtMatmul(static_cast<hipblasLtHandle_t>(handle),
                         static_cast<hipblasLtMatmulDesc_t>(desc), alpha, a,
                         static_cast<hipblasLtMatrixLayout_t>(a_layout), b,
                         static_cast<hipblasLtMatrixLayout_t>(b_layout), beta,
                         c, static_cast<hipblasLtMatrixLayout_t>(c_layout), c,
                         static_cast<hipblasLtMatrixLayout_t>(c_layout), &algo,
                         workspace, workspace_bytes,
                         static_cast<hipStream_t>(stream));
}

}  // extern "C"
