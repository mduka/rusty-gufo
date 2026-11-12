// Expert routing tables in launches of their own (adapters.hpp): the group
// table and the per-expert counts the routed GEMMs index by, derived from the
// expert IDs a routing launch wrote, so a consumer's contract can require
// that its table belongs to exactly those IDs.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/gemma4/kernels/rocm/adapters.hpp"
#include "src/models/gemma4/kernels/rocm/moe.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

constexpr int kThreads = 256;

// --- Verbatim from moe.hip.cpp (BuildGroups, GroupsKernel) -------------------

/// The group table of `slots` expert ids, built by one 256-thread block:
/// one thread per expert collects its slots in order; a block scan over the
/// non-empty experts places the groups.
__device__ inline void BuildGroups(const std::int32_t* __restrict__ ids,
                                   std::int32_t* __restrict__ groups,
                                   std::uint32_t slots, std::uint32_t experts) {
  __shared__ std::int32_t s_ids[kMaxGroupSlots * 8];
  __shared__ int scan[kThreads];
  for (std::uint32_t i = threadIdx.x; i < slots; i += kThreads) {
    s_ids[i] = ids[i];
  }
  __syncthreads();
  const auto e = static_cast<std::int32_t>(threadIdx.x);
  int mine[kMaxGroupSlots];
  int n = 0;
  if (static_cast<std::uint32_t>(e) < experts) {
    for (std::uint32_t s = 0; s < slots; ++s) {
      if (s_ids[s] == e && n < static_cast<int>(kMaxGroupSlots)) {
        mine[n++] = static_cast<int>(s);
      }
    }
  }
  scan[threadIdx.x] = n > 0 ? 1 : 0;
  __syncthreads();
  for (int offset = 1; offset < kThreads; offset <<= 1) {
    const int add = threadIdx.x >= static_cast<unsigned>(offset)
                        ? scan[threadIdx.x - offset]
                        : 0;
    __syncthreads();
    scan[threadIdx.x] += add;
    __syncthreads();
  }
  if (n > 0) {
    std::int32_t* g = groups + 1 + (scan[threadIdx.x] - 1) * kGroupInts;
    g[0] = e;
    g[1] = n;
    for (int i = 0; i < n; ++i) {
      g[2 + i] = mine[i];
    }
  }
  if (threadIdx.x == kThreads - 1) {
    groups[0] = scan[kThreads - 1];
  }
}

__global__ void __launch_bounds__(kThreads)
    GroupsKernel(const std::int32_t* __restrict__ ids,
                 std::int32_t* __restrict__ groups, std::uint32_t slots,
                 std::uint32_t experts) {
  BuildGroups(ids, groups, slots, experts);
}

// --- Adapter kernels ---------------------------------------------------------

/// Most experts ExpertCountsKernel counts (the routing kernels' limit).
constexpr std::uint32_t kMaxCountedExperts = 256;

/// counts[e] = slots s with ids[s] == e: the counts TopKKernel accumulates
/// beside the IDs it writes, in one block.
__global__ void __launch_bounds__(kThreads)
    ExpertCountsKernel(const std::int32_t* __restrict__ ids,
                       std::uint32_t* __restrict__ counts, std::uint32_t slots,
                       std::uint32_t experts) {
  __shared__ std::uint32_t bins[kMaxCountedExperts];
  for (std::uint32_t e = threadIdx.x; e < experts; e += kThreads) {
    bins[e] = 0;
  }
  __syncthreads();
  for (std::uint32_t s = threadIdx.x; s < slots; s += kThreads) {
    const std::int32_t id = ids[s];
    if (id >= 0 && static_cast<std::uint32_t>(id) < experts) {
      atomicAdd(&bins[id], 1U);
    }
  }
  __syncthreads();
  for (std::uint32_t e = threadIdx.x; e < experts; e += kThreads) {
    counts[e] = bins[e];
  }
}

}  // namespace

namespace flat {

bool ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t slots, std::uint32_t experts,
                  hipStream_t stream) {
  if (experts == 0 || experts > kMaxCountedExperts) {
    return false;
  }
  ExpertCountsKernel<<<1, kThreads, 0, stream>>>(ids, counts, slots, experts);
  return true;
}

bool ExpertGroups(const std::int32_t* ids, std::int32_t* groups,
                  std::uint32_t slots, std::uint32_t experts,
                  hipStream_t stream) {
  // BuildGroups stages at most kMaxGroupSlots * 8 IDs and takes one thread
  // per expert.
  if (slots == 0 || slots > kMaxGroupSlots * 8 || experts == 0 ||
      experts > static_cast<std::uint32_t>(kThreads)) {
    return false;
  }
  GroupsKernel<<<1, kThreads, 0, stream>>>(ids, groups, slots, experts);
  return true;
}

}  // namespace flat
}  // namespace gufo::models::gemma4::rocm
