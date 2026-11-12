#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <climits>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/nemotron/kernels/rocm/adapters.hpp"
#include "src/models/nemotron/kernels/rocm/gemm.hpp"
#include "src/models/nemotron/kernels/rocm/gemv.hpp"
#include "src/models/nemotron/kernels/rocm/kernels.hpp"
#include "src/models/nemotron/kernels/rocm/mmq/nb_mmq.h"
#include "src/models/nemotron/kernels/rocm/sample.hpp"
#include "src/models/nemotron/kernels/rocm/vision.hpp"

namespace gufo::models::nemotron::rocm::flat {

// The Rust side's `sizes` constants.
static_assert(kGemvMaxTokens == 8, "sizes::GEMV_MAX_TOKENS");
static_assert(kRouteShortSlots == 128, "sizes::ROUTE_SHORT_SLOTS");

namespace {

/// The launchers' `int` counts.
bool Fits(std::size_t value) {
  return value <= static_cast<std::size_t>(INT_MAX);
}

template<typename... T>
bool Fit(T... values) {
  return (Fits(values) && ...);
}

int I(std::size_t value) {
  return static_cast<int>(value);
}

/// The launcher's result: rejected arguments queue nothing and read as
/// unsupported; other errors (a failed launch) throw.
bool Done(hipError_t status) {
  if (status == hipSuccess)
    return true;
  if (status == hipErrorInvalidValue)
    return false;
  throw std::runtime_error(std::string("Nemotron-H kernel launch failed: ") +
                           hipGetErrorString(status));
}

const __half* H(const void* p) {
  return static_cast<const __half*>(p);
}

__half* H(void* p) {
  return static_cast<__half*>(p);
}

bool Operand(std::uint32_t op, VisionOperand* out) {
  if (op > 1)
    return false;
  *out = op == 0 ? VisionOperand::kBf16 : VisionOperand::kF16;
  return true;
}

/// Copies host lists of device pointers, one after another, into
/// `device_tables`. A pageable copy: the host lists may be released once
/// this returns.
void CopyTables(void* device_tables,
                std::initializer_list<const void* const*> lists,
                std::size_t rows, hipStream_t stream) {
  std::vector<const void*> host;
  host.reserve(lists.size() * rows);
  for (const void* const* list : lists)
    host.insert(host.end(), list, list + rows);
  const hipError_t copied =
      hipMemcpyAsync(device_tables, host.data(), host.size() * sizeof(void*),
                     hipMemcpyHostToDevice, stream);
  if (copied != hipSuccess)
    throw std::runtime_error(
        std::string("cannot copy Nemotron-H row pointers: ") +
        hipGetErrorString(copied));
}

/// Initialises the quantized tier once per process (thread-safe static).
bool MmqReady() {
  static const bool ready = nb_mmq_init(0) == 0;
  return ready;
}

bool MmqDense(int type, const void* w, const float* x, float* out,
              std::size_t m, std::size_t n, std::size_t k, hipStream_t stream) {
  return MmqReady() && Fit(m, n, k) &&
         nb_mmq_dense(type, w, x, out, I(m), I(n), I(k), stream) == 0;
}

bool MmqDenseQ8_1(int type, const void* w, const void* xq, float* out,
                  std::size_t m, std::size_t n, std::size_t k,
                  hipStream_t stream) {
  return MmqReady() && Fit(m, n, k) &&
         nb_mmq_dense_q8_1(type, w, xq, 1.0F, nullptr, out, I(m), I(n), I(k),
                           stream) == 0;
}

bool MmqResidual(int type, const void* w, const void* xq, float scale,
                 float* out, std::size_t m, std::size_t k, hipStream_t stream) {
  return MmqReady() && Fit(m, k) &&
         nb_mmq_dense_q8_1(type, w, xq, scale, out, out, I(m), 1, I(k),
                           stream) == 0;
}

}  // namespace

// --- kernels.hpp -------------------------------------------------------------

bool EmbedQ8_0(const void* table, const std::int32_t* tokens, float* x,
               std::size_t n_tokens, std::size_t hidden, hipStream_t stream) {
  return Fit(n_tokens, hidden) &&
         Done(
             rocm::EmbedQ8_0(table, tokens, x, I(n_tokens), I(hidden), stream));
}

bool RmsNorm(const float* x, const float* w, float* out, std::size_t rows,
             std::size_t width, float eps, hipStream_t stream) {
  return Fit(rows, width) &&
         Done(rocm::RmsNorm(x, w, out, I(rows), I(width), eps, stream));
}

bool RmsNormF16(const float* x, const float* w, void* out, std::size_t rows,
                std::size_t width, float eps, hipStream_t stream) {
  return Fit(rows, width) &&
         Done(rocm::RmsNorm(x, w, H(out), I(rows), I(width), eps, stream));
}

bool RmsNormQ8(const float* x, const float* w, float* out, void* q8,
               std::size_t rows, std::size_t width, float eps,
               hipStream_t stream) {
  return Fit(rows, width) &&
         Done(rocm::RmsNormQ8(x, w, out, q8, I(rows), I(width), eps, stream));
}

bool QuantizeQ8_1(const float* x, void* q8, std::size_t rows, std::size_t width,
                  hipStream_t stream) {
  return Fit(rows, width) &&
         Done(rocm::QuantizeQ8_1(x, q8, I(rows), I(width), stream));
}

bool AddScaled(float* x, const float* a, float scale, const float* b,
               std::size_t n, hipStream_t stream) {
  return Done(rocm::AddScaled(x, a, scale, b, n, stream));
}

bool Relu2(float* x, std::size_t n, hipStream_t stream) {
  return Done(rocm::Relu2(x, n, stream));
}

bool MambaConv(const float* proj, std::size_t stride, std::size_t xbc_offset,
               float in_scale, const float* weight, const float* bias,
               const float* state, float* out, std::size_t n_tokens,
               std::size_t channels, std::size_t kernel, hipStream_t stream) {
  return Fit(stride, xbc_offset, n_tokens, channels, kernel) &&
         Done(rocm::MambaConv(proj, I(stride), I(xbc_offset), in_scale, weight,
                              bias, state, out, I(n_tokens), I(channels),
                              I(kernel), stream));
}

bool MambaConvState(const float* proj, std::size_t stride,
                    std::size_t xbc_offset, float in_scale, float* state,
                    std::size_t n_tokens, std::size_t channels,
                    std::size_t kernel, hipStream_t stream) {
  return Fit(stride, xbc_offset, n_tokens, channels, kernel) &&
         Done(rocm::MambaConvState(proj, I(stride), I(xbc_offset), in_scale,
                                   state, I(n_tokens), I(channels), I(kernel),
                                   stream));
}

bool MambaScan(const float* xbc, std::size_t channels, const float* proj,
               std::size_t stride, std::size_t dt_offset, float in_scale,
               const float* dt_bias, const float* a, const float* d,
               float* state, float* y, std::size_t n_tokens, std::size_t heads,
               std::size_t head_dim, std::size_t groups, bool write_state,
               hipStream_t stream) {
  if (!Fit(channels, stride, dt_offset, n_tokens, heads, head_dim, groups))
    return false;
  MambaScanArgs args{};
  args.xbc = xbc;
  args.channels = I(channels);
  args.proj = proj;
  args.stride = I(stride);
  args.dt_offset = I(dt_offset);
  args.in_scale = in_scale;
  args.dt_bias = dt_bias;
  args.a = a;
  args.d = d;
  args.state = state;
  args.y = y;
  args.n_tokens = I(n_tokens);
  args.heads = I(heads);
  args.head_dim = I(head_dim);
  args.groups = I(groups);
  args.write_state = write_state;
  args.row_states = nullptr;
  return Done(rocm::MambaScan(args, stream));
}

bool MambaGateNormQ8(const float* y, const float* proj, std::size_t stride,
                     float in_scale, const float* weight, void* q8,
                     std::size_t n_tokens, std::size_t inner,
                     std::size_t groups, float eps, hipStream_t stream) {
  return Fit(stride, n_tokens, inner, groups) &&
         Done(rocm::MambaGateNormQ8(y, proj, I(stride), in_scale, weight, q8,
                                    I(n_tokens), I(inner), I(groups), eps,
                                    stream));
}

bool MambaGateNorm(float* y, const float* proj, std::size_t stride,
                   float in_scale, const float* weight, std::size_t n_tokens,
                   std::size_t inner, std::size_t groups, float eps,
                   hipStream_t stream) {
  return Fit(stride, n_tokens, inner, groups) &&
         Done(rocm::MambaGateNorm(y, proj, I(stride), in_scale, weight,
                                  I(n_tokens), I(inner), I(groups), eps,
                                  stream));
}

bool MambaGateNormF16(float* y, const float* proj, std::size_t stride,
                      float in_scale, const float* weight, void* out,
                      std::size_t n_tokens, std::size_t inner,
                      std::size_t groups, float eps, hipStream_t stream) {
  return Fit(stride, n_tokens, inner, groups) &&
         Done(rocm::MambaGateNorm(y, proj, I(stride), in_scale, weight, H(out),
                                  I(n_tokens), I(inner), I(groups), eps,
                                  stream));
}

bool KvAppend(const float* k, const float* v, void* k_cache, void* v_cache,
              const std::int32_t* pos, std::size_t n_tokens, std::size_t kv_dim,
              hipStream_t stream) {
  return Fit(n_tokens, kv_dim) &&
         Done(rocm::KvAppend(k, v, H(k_cache), H(v_cache), pos, I(n_tokens),
                             I(kv_dim), stream));
}

bool Attention(const float* q, const void* k_cache, const void* v_cache,
               float* out, const std::int32_t* pos, std::size_t n_tokens,
               std::size_t heads, std::size_t kv_heads, float scale,
               hipStream_t stream) {
  return Fit(n_tokens, heads, kv_heads) &&
         Done(rocm::Attention(q, H(k_cache), H(v_cache), out, pos, I(n_tokens),
                              I(heads), I(kv_heads), scale, stream));
}

bool AttentionShort(const float* q, const void* k_cache, const void* v_cache,
                    float* out, const std::int32_t* pos, std::size_t n_tokens,
                    std::size_t heads, std::size_t kv_heads, float scale,
                    float* partial, hipStream_t stream) {
  return Fit(n_tokens, heads, kv_heads) &&
         Done(rocm::AttentionShort(q, H(k_cache), H(v_cache), out, pos,
                                   I(n_tokens), I(heads), I(kv_heads), scale,
                                   partial, stream));
}

bool MambaConvRows(const float* proj, std::size_t stride,
                   std::size_t xbc_offset, float in_scale, const float* weight,
                   const float* bias, float* const* states, void* device_tables,
                   float* out, std::size_t rows, std::size_t channels,
                   std::size_t kernel, hipStream_t stream) {
  if (!Fit(stride, xbc_offset, rows, channels, kernel))
    return false;
  CopyTables(device_tables, {reinterpret_cast<const void* const*>(states)},
             rows, stream);
  return Done(
      rocm::MambaConvRows(proj, I(stride), I(xbc_offset), in_scale, weight,
                          bias, static_cast<float* const*>(device_tables), out,
                          I(rows), I(channels), I(kernel), stream));
}

bool MambaScanRows(const float* xbc, std::size_t channels, const float* proj,
                   std::size_t stride, std::size_t dt_offset, float in_scale,
                   const float* dt_bias, const float* a, const float* d,
                   float* const* states, void* device_tables, float* y,
                   std::size_t rows, std::size_t heads, std::size_t head_dim,
                   std::size_t groups, hipStream_t stream) {
  if (!Fit(channels, stride, dt_offset, rows, heads, head_dim, groups))
    return false;
  CopyTables(device_tables, {reinterpret_cast<const void* const*>(states)},
             rows, stream);
  MambaScanArgs args{};
  args.xbc = xbc;
  args.channels = I(channels);
  args.proj = proj;
  args.stride = I(stride);
  args.dt_offset = I(dt_offset);
  args.in_scale = in_scale;
  args.dt_bias = dt_bias;
  args.a = a;
  args.d = d;
  args.state = nullptr;
  args.y = y;
  args.n_tokens = I(rows);
  args.heads = I(heads);
  args.head_dim = I(head_dim);
  args.groups = I(groups);
  args.write_state = true;
  args.row_states = static_cast<float* const*>(device_tables);
  return Done(rocm::MambaScan(args, stream));
}

bool KvAppendRows(const float* k, const float* v, void* const* k_caches,
                  void* const* v_caches, void* device_tables,
                  const std::int32_t* row_positions, std::size_t rows,
                  std::size_t kv_dim, hipStream_t stream) {
  if (!Fit(rows, kv_dim))
    return false;
  CopyTables(device_tables,
             {reinterpret_cast<const void* const*>(k_caches),
              reinterpret_cast<const void* const*>(v_caches)},
             rows, stream);
  auto* const* tables = static_cast<__half* const*>(device_tables);
  return Done(rocm::KvAppendRows(k, v, tables, tables + rows, row_positions,
                                 I(rows), I(kv_dim), stream));
}

bool AttentionShortRows(const float* q, const void* const* k_caches,
                        const void* const* v_caches, void* device_tables,
                        float* out, const std::int32_t* row_positions,
                        std::size_t rows, std::size_t heads,
                        std::size_t kv_heads, float scale, float* partial,
                        hipStream_t stream) {
  if (!Fit(rows, heads, kv_heads))
    return false;
  CopyTables(device_tables, {k_caches, v_caches}, rows, stream);
  const auto* const* tables = static_cast<const __half* const*>(device_tables);
  return Done(rocm::AttentionShortRows(q, tables, tables + rows, out,
                                       row_positions, I(rows), I(heads),
                                       I(kv_heads), scale, partial, stream));
}

bool RouterLogits(const float* w, const float* x, float* out, std::size_t rows,
                  std::size_t cols, std::size_t n_tokens, hipStream_t stream) {
  return Fit(rows, cols, n_tokens) &&
         Done(rocm::RouterLogits(w, x, out, I(rows), I(cols), I(n_tokens),
                                 stream));
}

bool RouterTopK(const float* logits, const float* bias, std::size_t n_experts,
                std::size_t n_tokens, std::size_t k, bool normalize,
                float scale, std::int32_t* ids, float* weights,
                hipStream_t stream) {
  return Fit(n_experts, n_tokens, k) &&
         Done(rocm::RouterTopK(logits, bias, I(n_experts), I(n_tokens), I(k),
                               normalize, scale, ids, weights, stream));
}

bool WeightedSlotSum(const float* slots, const float* weights, float* out,
                     std::size_t n_tokens, std::size_t k, std::size_t width,
                     hipStream_t stream) {
  return Fit(n_tokens, k, width) &&
         Done(rocm::WeightedSlotSum(slots, weights, out, I(n_tokens), I(k),
                                    I(width), stream));
}

bool WeightedSlotSumQ8(const float* slots, const float* weights, void* q8,
                       std::size_t n_tokens, std::size_t k, std::size_t width,
                       hipStream_t stream) {
  return Fit(n_tokens, k, width) &&
         Done(rocm::WeightedSlotSumQ8(slots, weights, q8, I(n_tokens), I(k),
                                      I(width), stream));
}

// --- gemm.hpp ----------------------------------------------------------------

bool Nvfp4Gemm(const void* weights, const void* x, float* out, std::size_t rows,
               std::size_t cols, std::size_t n_tokens, hipStream_t stream) {
  return Fit(rows, cols, n_tokens) &&
         Done(rocm::Nvfp4Gemm(weights, H(x), out, I(rows), I(cols), I(n_tokens),
                              stream));
}

namespace {

Iq2xxsRoute Route(std::int32_t* cursor, const std::int32_t* order,
                  const std::int32_t* tiles, const std::int32_t* num_tiles,
                  std::size_t slots, std::size_t experts) {
  Iq2xxsRoute route;
  route.cursor = cursor;
  route.order = const_cast<std::int32_t*>(order);
  route.tiles = reinterpret_cast<int4*>(const_cast<std::int32_t*>(tiles));
  route.num_tiles = const_cast<std::int32_t*>(num_tiles);
  route.max_tiles = Iq2xxsRouteMaxTiles(I(slots), I(experts));
  return route;
}

}  // namespace

bool Iq2xxsRouteSlots(const std::int32_t* ids, std::int32_t* cursor,
                      std::int32_t* order, std::int32_t* tiles,
                      std::int32_t* num_tiles, std::size_t slots,
                      std::size_t experts, hipStream_t stream) {
  if (!Fit(slots, experts) ||
      slots / kIq2xxsRoutedTokens + experts > static_cast<std::size_t>(INT_MAX))
    return false;
  const Iq2xxsRoute route =
      Route(cursor, order, tiles, num_tiles, slots, experts);
  return Done(rocm::Iq2xxsRouteSlots(ids, I(slots), I(experts), route, stream));
}

bool Iq2xxsRoutedGemm(const std::int32_t* /*ids*/, const void* weights,
                      const std::uint32_t* grid, const float* x, float* out,
                      const std::int32_t* order, const std::int32_t* tiles,
                      const std::int32_t* num_tiles, std::size_t rows,
                      std::size_t cols, bool relu2, std::size_t slots,
                      std::size_t experts, std::size_t used, bool x_per_slot,
                      hipStream_t stream) {
  if (!Fit(rows, cols, slots, experts, used) ||
      slots / kIq2xxsRoutedTokens + experts > static_cast<std::size_t>(INT_MAX))
    return false;
  // The routed kernel only reads the route; cursor is the routing's scratch.
  const Iq2xxsRoute route =
      Route(nullptr, order, tiles, num_tiles, slots, experts);
  Iq2xxsGemmArgs args{};
  args.weights = weights;
  args.grid = grid;
  args.x = x;
  args.x_per_slot = x_per_slot;
  args.out = out;
  args.rows = I(rows);
  args.cols = I(cols);
  args.relu2 = relu2;
  args.route = &route;
  args.used = I(used);
  return Done(rocm::Iq2xxsRoutedGemm(args, stream));
}

// --- gemv.hpp ----------------------------------------------------------------

bool Iq2xxsGemv(const void* weights, const std::uint32_t* grid, const float* x,
                float* out, std::size_t rows, std::size_t cols,
                std::size_t n_tokens, bool relu2, bool accumulate,
                hipStream_t stream) {
  if (!Fit(rows, cols, n_tokens))
    return false;
  Iq2xxsGemvArgs args{};
  args.weights = weights;
  args.grid = grid;
  args.x = x;
  args.out = out;
  args.rows = I(rows);
  args.cols = I(cols);
  args.n_tokens = I(n_tokens);
  args.relu2 = relu2;
  args.accumulate = accumulate;
  return Done(rocm::Iq2xxsGemv(args, stream));
}

bool Iq2xxsGemvRouted(const void* weights, const std::uint32_t* grid,
                      const float* x, const std::int32_t* ids, float* out,
                      std::size_t rows, std::size_t cols, std::size_t n_tokens,
                      std::size_t used, bool x_per_slot, bool relu2,
                      bool accumulate, hipStream_t stream) {
  if (!Fit(rows, cols, n_tokens, used))
    return false;
  Iq2xxsGemvArgs args{};
  args.weights = weights;
  args.grid = grid;
  args.x = x;
  args.ids = ids;
  args.used = I(used);
  args.x_per_slot = x_per_slot;
  args.out = out;
  args.rows = I(rows);
  args.cols = I(cols);
  args.n_tokens = I(n_tokens);
  args.relu2 = relu2;
  args.accumulate = accumulate;
  return Done(rocm::Iq2xxsGemv(args, stream));
}

bool ActivationsToBf16(const float* x, void* out, std::size_t n,
                       hipStream_t stream) {
  return Done(rocm::ToBf16(x, out, n, stream));
}

bool Iq2xxsGemvBf16(const void* weights, const std::uint32_t* grid,
                    const void* x_bf16, float* out, std::size_t rows,
                    std::size_t cols, std::size_t n_tokens, bool relu2,
                    bool accumulate, hipStream_t stream) {
  if (!Fit(rows, cols, n_tokens))
    return false;
  Iq2xxsGemvArgs args{};
  args.weights = weights;
  args.grid = grid;
  args.x_bf16 = x_bf16;
  args.out = out;
  args.rows = I(rows);
  args.cols = I(cols);
  args.n_tokens = I(n_tokens);
  args.relu2 = relu2;
  args.accumulate = accumulate;
  return Done(rocm::Iq2xxsGemv(args, stream));
}

bool Iq2xxsGemvBf16StoreBf16(const void* weights, const std::uint32_t* grid,
                             const void* x_bf16, void* out_bf16,
                             std::size_t rows, std::size_t cols,
                             std::size_t n_tokens, bool relu2,
                             hipStream_t stream) {
  if (!Fit(rows, cols, n_tokens))
    return false;
  Iq2xxsGemvArgs args{};
  args.weights = weights;
  args.grid = grid;
  args.x_bf16 = x_bf16;
  args.out_bf16 = out_bf16;
  args.rows = I(rows);
  args.cols = I(cols);
  args.n_tokens = I(n_tokens);
  args.relu2 = relu2;
  return Done(rocm::Iq2xxsGemv(args, stream));
}

namespace {

Iq2xxsRoute ShortRoute(const std::int32_t* order, const std::int32_t* tiles,
                       const std::int32_t* num_tiles, std::size_t slots) {
  Iq2xxsRoute route;
  route.order = const_cast<std::int32_t*>(order);
  route.tiles = reinterpret_cast<int4*>(const_cast<std::int32_t*>(tiles));
  route.num_tiles = const_cast<std::int32_t*>(num_tiles);
  route.max_tiles = I(slots);
  return route;
}

/// Iq2xxsGemv over a short route; `out` or `out_bf16` receives the rows.
bool Tiles(const std::int32_t* ids, const void* weights,
           const std::uint32_t* grid, const void* x_bf16, float* out,
           void* out_bf16, const std::int32_t* order,
           const std::int32_t* tiles, const std::int32_t* num_tiles,
           std::size_t rows, std::size_t cols, std::size_t slots,
           std::size_t used, bool x_per_slot, bool relu2, bool accumulate,
           hipStream_t stream) {
  if (!Fit(rows, cols, slots, used) || used == 0 || slots % used != 0)
    return false;
  const Iq2xxsRoute route = ShortRoute(order, tiles, num_tiles, slots);
  Iq2xxsGemvArgs args{};
  args.weights = weights;
  args.grid = grid;
  args.ids = ids;
  args.used = I(used);
  args.x_per_slot = x_per_slot;
  args.x_bf16 = x_bf16;
  args.route = &route;
  args.out = out;
  args.out_bf16 = out_bf16;
  args.rows = I(rows);
  args.cols = I(cols);
  args.n_tokens = I(slots / used);
  args.relu2 = relu2;
  args.accumulate = accumulate;
  return Done(rocm::Iq2xxsGemv(args, stream));
}

}  // namespace

bool Iq2xxsRouteShort(const std::int32_t* ids, std::int32_t* order,
                      std::int32_t* tiles, std::int32_t* num_tiles,
                      std::size_t slots, hipStream_t stream) {
  if (!Fit(slots))
    return false;
  return Done(rocm::Iq2xxsRouteShort(
      ids, I(slots), ShortRoute(order, tiles, num_tiles, slots), stream));
}

bool Iq2xxsGemvTiles(const std::int32_t* ids, const void* weights,
                     const std::uint32_t* grid, const void* x_bf16, float* out,
                     const std::int32_t* order, const std::int32_t* tiles,
                     const std::int32_t* num_tiles, std::size_t rows,
                     std::size_t cols, std::size_t slots, std::size_t used,
                     bool x_per_slot, bool relu2, bool accumulate,
                     hipStream_t stream) {
  return Tiles(ids, weights, grid, x_bf16, out, nullptr, order, tiles,
               num_tiles, rows, cols, slots, used, x_per_slot, relu2,
               accumulate, stream);
}

bool Iq2xxsGemvTilesStoreBf16(const std::int32_t* ids, const void* weights,
                              const std::uint32_t* grid, const void* x_bf16,
                              void* out_bf16, const std::int32_t* order,
                              const std::int32_t* tiles,
                              const std::int32_t* num_tiles, std::size_t rows,
                              std::size_t cols, std::size_t slots,
                              std::size_t used, bool x_per_slot, bool relu2,
                              hipStream_t stream) {
  return Tiles(ids, weights, grid, x_bf16, nullptr, out_bf16, order, tiles,
               num_tiles, rows, cols, slots, used, x_per_slot, relu2, false,
               stream);
}

// --- sample.hpp --------------------------------------------------------------

bool LogitHeads(const float* logits, std::size_t rows, std::size_t vocab,
                float temperature, std::size_t keep, float top_p,
                std::size_t minimum, void* scratch, void* records,
                void* candidates, hipStream_t stream) {
  if (!Fit(rows, vocab, keep, minimum))
    return false;
  const LogitHeadParams params{temperature, I(keep), top_p, I(minimum)};
  return Done(rocm::LogitHeads(logits, I(rows), I(vocab), params, scratch,
                               static_cast<LogitHeadRecord*>(records),
                               static_cast<LogitHeadCandidate*>(candidates),
                               stream));
}

// --- vision.hpp --------------------------------------------------------------

namespace {

std::uint16_t* U16(void* p) {
  return static_cast<std::uint16_t*>(p);
}

const std::uint16_t* U16(const void* p) {
  return static_cast<const std::uint16_t*>(p);
}

}  // namespace

bool VisionToOperand(const float* in, void* out, std::size_t n,
                     std::uint32_t op, hipStream_t stream) {
  VisionOperand o{};
  return Operand(op, &o) &&
         Done(rocm::VisionToOperand(in, U16(out), n, o, stream));
}

bool VisionBf16ToOperand(const void* in, void* out, std::size_t n,
                         std::uint32_t op, hipStream_t stream) {
  VisionOperand o{};
  return Operand(op, &o) &&
         Done(rocm::VisionBf16ToOperand(U16(in), U16(out), n, o, stream));
}

bool VisionEmbed(float* x, const float* prefix, const float* positions,
                 std::size_t prefix_tokens, std::size_t grid_rows,
                 std::size_t grid_cols, std::size_t side, std::size_t width,
                 hipStream_t stream) {
  return Fit(prefix_tokens, grid_rows, grid_cols, side, width) &&
         Fit(prefix_tokens + grid_rows * grid_cols) &&
         Done(rocm::VisionEmbed(x, prefix, positions, I(prefix_tokens),
                                I(grid_rows), I(grid_cols), I(side), I(width),
                                stream));
}

bool VisionLayerNorm(const float* x, const float* w, const float* b, void* out,
                     std::size_t rows, std::size_t width, float eps,
                     std::uint32_t op, hipStream_t stream) {
  VisionOperand o{};
  return Operand(op, &o) && Fit(rows, width) &&
         Done(rocm::VisionLayerNorm(x, w, b, U16(out), I(rows), I(width), eps,
                                    o, stream));
}

bool VisionBias(const float* y, const float* bias, void* out, std::size_t rows,
                std::size_t cols, std::uint32_t op, hipStream_t stream) {
  VisionOperand o{};
  return Operand(op, &o) && Fit(rows, cols) &&
         Done(rocm::VisionBias(y, bias, U16(out), I(rows), I(cols), o, stream));
}

bool VisionBiasGelu(const float* y, const float* bias, void* out,
                    std::size_t rows, std::size_t cols, std::uint32_t op,
                    hipStream_t stream) {
  VisionOperand o{};
  return Operand(op, &o) && Fit(rows, cols) &&
         Done(rocm::VisionBiasGelu(y, bias, U16(out), I(rows), I(cols), o,
                                   stream));
}

bool VisionResidual(float* x, const float* y, const float* bias,
                    std::size_t rows, std::size_t cols, hipStream_t stream) {
  return Fit(rows, cols) &&
         Done(rocm::VisionResidual(x, y, bias, I(rows), I(cols), stream));
}

bool VisionAttention(const void* qkv, void* out, std::size_t tokens,
                     std::size_t heads, float scale, std::uint32_t op,
                     hipStream_t stream) {
  VisionOperand o{};
  return Operand(op, &o) && Fit(tokens, heads) &&
         Done(rocm::VisionAttention(U16(qkv), U16(out), I(tokens), I(heads),
                                    scale, o, stream));
}

bool VisionMergeNorm(const float* x, const float* w, void* out,
                     std::size_t prefix_tokens, std::size_t grid_rows,
                     std::size_t grid_cols, std::size_t width, float eps,
                     std::uint32_t op, hipStream_t stream) {
  VisionOperand o{};
  return Operand(op, &o) && Fit(prefix_tokens, grid_rows, grid_cols, width) &&
         Done(rocm::VisionMergeNorm(x, w, U16(out), I(prefix_tokens),
                                    I(grid_rows), I(grid_cols), I(width), eps,
                                    o, stream));
}

bool VisionRelu2(const float* y, void* out, std::size_t n, std::uint32_t op,
                 hipStream_t stream) {
  VisionOperand o{};
  return Operand(op, &o) && Done(rocm::VisionRelu2(y, U16(out), n, o, stream));
}

// --- mmq/nb_mmq.h ------------------------------------------------------------

bool MmqBindStream(hipStream_t stream) {
  return MmqReady() && nb_mmq_bind_stream(stream) == 0;
}

void MmqReleaseStream(hipStream_t stream) {
  // The arena may be freed only once the stream is idle.
  if (hipStreamSynchronize(stream) == hipSuccess)
    nb_mmq_release_stream(stream);
}

bool MmqDenseQ8_0(const void* w, const float* x, float* out, std::size_t m,
                  std::size_t n, std::size_t k, hipStream_t stream) {
  return MmqDense(NB_MMQ_Q8_0, w, x, out, m, n, k, stream);
}

bool MmqDenseIq2Xxs(const void* w, const float* x, float* out, std::size_t m,
                    std::size_t n, std::size_t k, hipStream_t stream) {
  return MmqDense(NB_MMQ_IQ2_XXS, w, x, out, m, n, k, stream);
}

bool MmqDenseNvfp4(const void* w, const float* x, float* out, std::size_t m,
                   std::size_t n, std::size_t k, hipStream_t stream) {
  return MmqDense(NB_MMQ_NVFP4, w, x, out, m, n, k, stream);
}

bool MmqDenseQ8_1Q8_0(const void* w, const void* xq, float* out, std::size_t m,
                      std::size_t n, std::size_t k, hipStream_t stream) {
  return MmqDenseQ8_1(NB_MMQ_Q8_0, w, xq, out, m, n, k, stream);
}

bool MmqDenseQ8_1Iq2Xxs(const void* w, const void* xq, float* out,
                        std::size_t m, std::size_t n, std::size_t k,
                        hipStream_t stream) {
  return MmqDenseQ8_1(NB_MMQ_IQ2_XXS, w, xq, out, m, n, k, stream);
}

bool MmqDenseQ8_1Nvfp4(const void* w, const void* xq, float* out, std::size_t m,
                       std::size_t n, std::size_t k, hipStream_t stream) {
  return MmqDenseQ8_1(NB_MMQ_NVFP4, w, xq, out, m, n, k, stream);
}

bool MmqResidualQ8_0(const void* w, const void* xq, float scale, float* out,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  return MmqResidual(NB_MMQ_Q8_0, w, xq, scale, out, m, k, stream);
}

bool MmqResidualIq2Xxs(const void* w, const void* xq, float scale, float* out,
                       std::size_t m, std::size_t k, hipStream_t stream) {
  return MmqResidual(NB_MMQ_IQ2_XXS, w, xq, scale, out, m, k, stream);
}

bool MmqResidualNvfp4(const void* w, const void* xq, float scale, float* out,
                      std::size_t m, std::size_t k, hipStream_t stream) {
  return MmqResidual(NB_MMQ_NVFP4, w, xq, scale, out, m, k, stream);
}

}  // namespace gufo::models::nemotron::rocm::flat
