#include "src/cli/kernels/hip/allocation.hpp"

namespace gufo::cli::diagnostics::hip {
namespace {

constexpr unsigned int kThreads = 256;

__global__ void FillPatternKernel(std::uint8_t* data, std::size_t size,
                                  std::uint8_t seed) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index = thread; index < size; index += stride) {
    data[index] = static_cast<std::uint8_t>(index + seed);
  }
}

// Wave32 reduction over 256 threads, then one atomic add per block.
__global__ void ChecksumKernel(const std::uint8_t* data, std::size_t size,
                               unsigned long long* checksum) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (std::size_t index = thread; index < size; index += stride) {
    local += data[index];
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    local += __shfl_down(local, offset, 32);
  }
  __shared__ unsigned long long wave_sums[kThreads / 32];
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int wave = threadIdx.x >> 5U;
  if (lane == 0U) {
    wave_sums[wave] = local;
  }
  __syncthreads();
  if (wave == 0U) {
    local = lane < kThreads / 32 ? wave_sums[lane] : 0ULL;
    for (int offset = 16; offset > 0; offset /= 2) {
      local += __shfl_down(local, offset, 32);
    }
    if (lane == 0U) {
      atomicAdd(checksum, local);
    }
  }
}

}  // namespace

void LaunchFillPattern(std::uint8_t* data, std::size_t size, std::uint32_t seed,
                       std::uint32_t blocks, hipStream_t stream) {
  hipLaunchKernelGGL(FillPatternKernel, dim3(blocks), dim3(kThreads), 0, stream,
                     data, size, static_cast<std::uint8_t>(seed));
}

void LaunchChecksum(const std::uint8_t* data, std::uint64_t* sum,
                    std::size_t size, std::uint32_t blocks,
                    hipStream_t stream) {
  static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long));
  hipLaunchKernelGGL(ChecksumKernel, dim3(blocks), dim3(kThreads), 0, stream,
                     data, size, reinterpret_cast<unsigned long long*>(sum));
}

}  // namespace gufo::cli::diagnostics::hip
