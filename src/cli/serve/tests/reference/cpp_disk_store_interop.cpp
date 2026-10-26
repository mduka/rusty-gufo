// Writes or reads a continuation disk cache with the C++ store, for the
// cross-check against the Rust port (`examples/disk_store_interop.rs`).
//
//   cpp_disk_store_interop write <dir>   stores the three fixed entries
//   cpp_disk_store_interop read <dir>    restores the four fixed queries
//
// Build against the legacy tree (sources and libgufo_core.a) of
// $UPSTREAM, an upstream C++ checkout; see docs/RUST.md.

#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/cli/serve/continuation_disk_store.hpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextExecutionPlan;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPersistenceDescriptor;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerSnapshot;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;

struct InteropSnapshot final : TextRunnerSnapshot {
  std::vector<std::uint8_t> bytes;
  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return bytes.size();
  }
};

struct InteropState final : TextRunnerState {
  std::vector<std::uint8_t> payload;
  void Invalidate() noexcept override { payload.clear(); }
};

class Runner final : public TextModelRunner {
public:
  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "interop",
        .state_abi = "interop-state",
        .max_context = 64,
        .capabilities = TextRunnerCapabilities{.snapshot = true, .fork = true},
        .persistence =
            TextRunnerPersistenceDescriptor{
                .compatibility_identity = {'i', 'n', 't', 'e', 'r', 'o', 'p',
                                           '-', 'v', '1'},
                .payload_version = 7,
            },
    };
  }
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {};
  }
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {TextExecutionPlan{}};
  }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view) const override {
    return {};
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken>) const override {
    return {};
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<InteropState>();
  }
  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState&,
                                        std::span<const TextRunnerToken>,
                                        std::size_t,
                                        std::size_t) const override {
    return {};
  }
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState&, gufo::sampling::SamplerState&) const override {
    return {};
  }
  void Advance(TextRunnerState&, TextRunnerToken) const override {}
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState&) const override {
    return 0;
  }
  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    return snapshot.PayloadBytes();
  }
  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto& bytes = dynamic_cast<const InteropSnapshot&>(snapshot).bytes;
    std::memcpy(destination.data(), bytes.data(), bytes.size());
    return bytes.size();
  }
  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    dynamic_cast<InteropState&>(state).payload.assign(payload.begin(),
                                                      payload.end());
  }
};

struct Entry {
  std::vector<TextRunnerToken> tokens;
  std::string payload;
  std::vector<std::uint8_t> identity;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s write|read <dir>\n", argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  Runner runner;
  gufo::server::ContinuationDiskStore store({
      .directory = argv[2],
      .capacity_bytes = 1 << 20,
      .staging_capacity_bytes = 1 << 20,
  });
  if (mode == "write") {
    const std::vector<Entry> entries = {
        {{1, 2, 3}, "alpha", {}},
        {{1, 2, 3, 4, 5}, "bravo-bravo", {}},
        {{9, 8}, "image", {0xAA, 0xBB}},
    };
    for (const auto& entry : entries) {
      InteropSnapshot snapshot;
      snapshot.bytes.assign(entry.payload.begin(), entry.payload.end());
      const auto saved =
          store.Save(runner, entry.tokens, snapshot, entry.identity);
      std::printf("stored=%d file_bytes=%zu payload_bytes=%zu\n",
                  saved.stored ? 1 : 0, saved.file_bytes, saved.payload_bytes);
    }
    return 0;
  }
  if (mode == "read") {
    const std::vector<Entry> queries = {
        {{1, 2, 3, 4, 5, 6}, "", {}},
        {{1, 2, 3, 9}, "", {}},
        {{9, 8, 7}, "", {0xAA, 0xBB}},
        {{9, 8, 7}, "", {}},
    };
    for (const auto& query : queries) {
      InteropState state;
      const auto restored = store.RestoreLongestPrefix(
          runner, state, query.tokens, query.identity);
      std::printf(
          "restored=%d tokens=%zu file_bytes=%zu payload=%s\n",
          restored.restored ? 1 : 0, restored.token_count, restored.file_bytes,
          std::string(state.payload.begin(), state.payload.end()).c_str());
    }
    return 0;
  }
  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
