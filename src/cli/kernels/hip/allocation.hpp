// Launchers of the HIP allocation diagnostic (`gufo diagnose --benchmark
// allocation`), the kernels of `src/core/hip/allocation_benchmark.hip`.

#ifndef GUFO_CLI_KERNELS_HIP_ALLOCATION_HPP_
#define GUFO_CLI_KERNELS_HIP_ALLOCATION_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::cli::diagnostics::hip {

// Writes `(index + seed) mod 256` to every byte of `data` with `blocks`
// blocks of 256 threads.
void LaunchFillPattern(std::uint8_t* data, std::size_t size, std::uint32_t seed,
                       std::uint32_t blocks, hipStream_t stream);

// Adds every byte of `data` into `*sum` with `blocks` blocks of 256 threads;
// the caller clears `*sum` first.
void LaunchChecksum(const std::uint8_t* data, std::uint64_t* sum,
                    std::size_t size, std::uint32_t blocks, hipStream_t stream);

}  // namespace gufo::cli::diagnostics::hip

#endif  // GUFO_CLI_KERNELS_HIP_ALLOCATION_HPP_
