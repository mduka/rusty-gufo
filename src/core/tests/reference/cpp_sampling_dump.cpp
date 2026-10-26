// Records the C++ sampler's token sequences, RNG states and distribution
// digests for the Rust differential test (`tests/sampling.rs`, fixture
// `tests/fixtures/sampling_cpp.json`). Every scenario is mirrored exactly by
// `differential_scenarios` there.
//
// Regenerate (only `src/core/sampling.cpp` of an upstream checkout, $UPSTREAM,
// is needed):
//   g++ -std=c++20 -O2 -I$UPSTREAM
//   src/core/tests/reference/cpp_sampling_dump.cpp \
//     $UPSTREAM/src/core/sampling.cpp -o /tmp/cpp_sampling_dump
//   /tmp/cpp_sampling_dump > src/core/tests/fixtures/sampling_cpp.json

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"

namespace {

using gufo::sampling::SamplerState;
using gufo::sampling::SamplingConfig;
using gufo::sampling::SamplingDistribution;
using gufo::sampling::TokenId;

struct Case {
  const char* name;
  SamplingConfig config;
};

// `QwenSamplingCases()` plus filters that reach the other selection routes.
std::vector<Case> Cases() {
  std::vector<Case> cases{
      {"greedy", {}},
      {"temperature-cold", {.temperature = 0.05F}},
      {"temperature", {.temperature = 1.0F}},
      {"temperature-hot", {.temperature = 2.0F}},
      {"top-k-one", {.temperature = 0.8F, .top_k = 1}},
      {"top-k", {.temperature = 0.8F, .top_k = 3}},
      {"top-p", {.temperature = 0.8F, .top_p = 0.7F}},
      {"top-p-zero", {.temperature = 0.8F, .top_p = 0.0F}},
      {"top-p-zero-floor", {.temperature = 0.8F, .top_p = 0.0F, .min_keep = 3}},
      {"min-p", {.temperature = 0.8F, .min_p = 0.3F}},
      {"min-p-one", {.temperature = 0.8F, .min_p = 1.0F}},
      {"top-k-floor", {.temperature = 0.8F, .top_k = 1, .min_keep = 3}},
      {"top-p-floor", {.temperature = 0.8F, .top_p = 0.01F, .min_keep = 3}},
      {"min-p-floor", {.temperature = 0.8F, .min_p = 1.0F, .min_keep = 3}},
      {"repeat-penalty", {.temperature = 0.8F, .repeat_penalty = 1.5F}},
      {"repeat-reward", {.temperature = 0.8F, .repeat_penalty = 0.7F}},
      {"frequency-penalty", {.temperature = 0.8F, .frequency_penalty = 0.4F}},
      {"frequency-reward", {.temperature = 0.8F, .frequency_penalty = -0.4F}},
      {"presence-penalty", {.temperature = 0.8F, .presence_penalty = 0.4F}},
      {"presence-reward", {.temperature = 0.8F, .presence_penalty = -0.4F}},
      {"history-one",
       {.temperature = 0.8F, .repeat_penalty = 1.5F, .repeat_last_n = 1}},
      {"history-disabled",
       {.temperature = 0.8F,
        .repeat_penalty = 1.5F,
        .repeat_last_n = 0,
        .frequency_penalty = 0.4F,
        .presence_penalty = 0.4F}},
      {"greedy-penalties",
       {.repeat_penalty = 1.5F,
        .repeat_last_n = 3,
        .frequency_penalty = 0.4F,
        .presence_penalty = -0.4F}},
      {"all-filters",
       {.temperature = 0.8F,
        .top_k = 4,
        .top_p = 0.7F,
        .min_p = 0.3F,
        .min_keep = 2,
        .repeat_penalty = 1.2F,
        .repeat_last_n = 3,
        .frequency_penalty = 0.2F,
        .presence_penalty = -0.1F}},
      {"all-floors",
       {.temperature = 0.8F,
        .top_k = 1,
        .top_p = 0.01F,
        .min_p = 1.0F,
        .min_keep = 3}},
      {"top-p-wide", {.temperature = 1.3F, .top_p = 0.95F}},
      {"min-p-linear", {.temperature = 1.0F, .min_p = 0.05F}},
      {"min-p-keep", {.temperature = 0.9F, .min_p = 0.2F, .min_keep = 4}},
      {"top-k-wide", {.temperature = 0.7F, .top_k = 300, .top_p = 0.9F}},
      {"penalties-all",
       {.temperature = 1.0F,
        .repeat_penalty = 1.3F,
        .repeat_last_n = 8,
        .frequency_penalty = 0.3F,
        .presence_penalty = 0.2F}},
  };
  for (auto& test : cases)
    test.config.seed = 73;
  cases[2].config.seed = 0;
  cases[3].config.seed = 808;
  return cases;
}

// Logits in [-4, 4) times `scale` from a 64-bit LCG, all float arithmetic;
// rows longer than 12 carry NaN and both infinities.
std::vector<float> Row(std::uint64_t seed, std::size_t size, float scale) {
  std::vector<float> row(size);
  std::uint64_t state = seed * 2 + 1;
  for (auto& value : row) {
    state =
        state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    value =
        (static_cast<float>(state >> 40) / 16777216.0F * 8.0F - 4.0F) * scale;
  }
  if (size > 12) {
    row[3] = std::numeric_limits<float>::quiet_NaN();
    row[7] = std::numeric_limits<float>::infinity();
    row[11] = -std::numeric_limits<float>::infinity();
  }
  return row;
}

struct Digest {
  std::uint64_t value{UINT64_C(0xcbf29ce484222325)};
  void Byte(std::uint8_t byte) {
    value = (value ^ byte) * UINT64_C(0x100000001b3);
  }
  void Word(std::uint64_t word, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i)
      Byte(static_cast<std::uint8_t>(word >> (8 * i)));
  }
  void Distribution(const SamplingDistribution& distribution) {
    Word(distribution.entries().size(), 8);
    for (const auto& entry : distribution.entries()) {
      Word(entry.token, 4);
      std::uint64_t bits;
      static_assert(sizeof(bits) == sizeof(entry.value));
      __builtin_memcpy(&bits, &entry.value, sizeof(bits));
      Word(bits, 8);
    }
  }
};

std::string Hex(std::uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx",
                static_cast<unsigned long long>(value));
  return buffer;
}

}  // namespace

int main() {
  const std::vector<TokenId> history{1, 1, 2, 3, 9, 9};
  std::vector<std::string> lines;
  for (const std::size_t size : {5U, 2048U, 4099U}) {
    for (const float scale : {1.0F, 6.0F}) {
      for (const auto& test : Cases()) {
        for (const bool own_seed : {true, false}) {
          auto config = test.config;
          if (!own_seed)
            config.seed = 5;
          SamplerState sampler(config, history);
          Digest digest;
          std::string tokens;
          std::vector<TokenId> generated;
          for (std::uint64_t step = 0; step < 10; ++step) {
            const auto row =
                Row(size * 131 + static_cast<std::uint64_t>(scale * 7.0F) * 17 +
                        step,
                    size, scale);
            if (step % 3 == 0)
              digest.Distribution(sampler.Distribution(row));
            const auto token = sampler.Sample(row);
            tokens += (tokens.empty() ? "" : ",") + std::to_string(token);
            generated.push_back(token);
            sampler.Accept(token);
          }
          const auto first = Row(size * 131 + 3, size, scale);
          digest.Distribution(gufo::sampling::BuildDistribution(
              first, config, history, generated));
          const std::vector<TokenId> compact_ids{10, 1, 9, 2, 3, 12, 40, 41};
          const std::vector<float> compact(first.begin(), first.begin() + 8);
          if (size >= 8)
            digest.Distribution(sampler.Distribution(compact, compact_ids));
          std::string residual;
          if (config.temperature > 0.0F) {
            SamplerState draft(config);
            const std::vector<TokenId> ids{0, 1, 2, 4, 1};
            const std::vector<float> probabilities{0.5F, 0.125F, 0.125F, 0.125F,
                                                   0.125F};
            for (std::uint64_t step = 0; step < 5; ++step) {
              const auto row = Row(size + 1000 + step, size, scale);
              if (step == 2)
                draft.DeferSample(4);
              const auto token =
                  step == 2 ? draft.Sample(row)
                            : draft.SampleResidual(row, ids, probabilities);
              residual += (residual.empty() ? "" : ",") + std::to_string(token);
            }
            residual += " rng=" + Hex(draft.rng_state());
          }
          lines.push_back(std::string(test.name) + "/" + std::to_string(size) +
                          "/" + std::to_string(static_cast<int>(scale)) + "/" +
                          std::to_string(config.seed) + " tokens=" + tokens +
                          " rng=" + Hex(sampler.rng_state()) + " digest=" +
                          Hex(digest.value) + " residual=" + residual);
        }
      }
    }
  }
  std::printf(
      "{\n  \"generator\": "
      "\"src/core/tests/reference/cpp_sampling_dump.cpp\",\n"
      "  \"lines\": [\n");
  for (std::size_t i = 0; i < lines.size(); ++i)
    std::printf("    \"%s\"%s\n", lines[i].c_str(),
                i + 1 == lines.size() ? "" : ",");
  std::printf("  ]\n}\n");
  return 0;
}
