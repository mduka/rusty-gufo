// Records the C++ audio feature extractor (ProcessAudio) for bit-exact
// comparison with the Rust extractor (docs/development/TESTING.md): for every
// case of the official audio_features_golden.json fixture, the frame count,
// valid tokens, mask and the SHA-256 of the little-endian float32 features.
//
// Build against a Gufo v0.28.1 checkout ($UPSTREAM, built as in
// docs/development/PARITY.md), linking its
// compiled extractor so the arithmetic is the release build's:
//   O=$UPSTREAM B=$O/build/release
//   g++ -std=c++20 -O2 -I$O cpp_audio_features.cpp -o cpp_audio_features \
//     $B/src/models/embedding_gemma2/libgufo_embedding_gemma2.a \
//     $B/libgufo_core.a $B/src/core/platform/libgufo_platform.a -lcrypto
//   F=$O/tests/fixtures/embedding_gemma2/audio_features_golden.json
//   ./cpp_audio_features $F > audio_features_cpp.json
#include <openssl/sha.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/embedding_gemma2/audio_processor.hpp"

namespace {
// The waveforms of tests/models/embedding_gemma2/audio_processor_test.cpp.
std::vector<float> Waveform(std::size_t count, const std::string& kind) {
  std::vector<float> samples(count, 0);
  for (std::size_t i = 0; i < count; ++i) {
    if (kind == "triangle") {
      const auto phase = static_cast<int>(i % 80);
      samples[i] = static_cast<float>(std::min(phase, 80 - phase) - 20) / 64;
    } else if (kind == "noise") {
      samples[i] = static_cast<float>(
                       static_cast<int>((i * 73 + i * i * 19) % 1024) - 512) /
                   1024;
    }
  }
  return samples;
}

std::string Sha256(const std::vector<float>& values) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(values.data()),
         values.size() * sizeof(float), digest);
  std::string hex;
  char byte[3];
  for (const unsigned char value : digest) {
    std::snprintf(byte, sizeof(byte), "%02x", value);
    hex += byte;
  }
  return hex;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: cpp_audio_features audio_features_golden.json\n";
    return 2;
  }
  std::ifstream input(argv[1]);
  std::ostringstream contents;
  contents << input.rdbuf();
  const auto fixture = gufo::json::parse(contents.str());
  std::cout << "{\"cases\":[";
  bool first = true;
  for (const auto& probe : fixture.find("cases")->items()) {
    const auto features = gufo::models::embedding_gemma2::ProcessAudio(
        Waveform(probe.member_size("samples"), probe.member_str("kind")),
        probe.member_size("padding_samples"));
    std::cout << (first ? "" : ",") << "\n{\"name\":\""
              << probe.member_str("name")
              << "\",\"samples\":" << probe.member_size("samples")
              << ",\"kind\":\"" << probe.member_str("kind")
              << "\",\"padding_samples\":"
              << probe.member_size("padding_samples")
              << ",\"frames\":" << features.mask.size()
              << ",\"soft_tokens\":" << features.soft_tokens() << ",\"mask\":[";
    for (std::size_t i = 0; i < features.mask.size(); ++i)
      std::cout << (i ? "," : "") << int{features.mask[i]};
    std::cout << "],\"features_sha256\":\"" << Sha256(features.values) << "\"}";
    first = false;
  }
  std::cout << "\n]}\n";
  return 0;
}
