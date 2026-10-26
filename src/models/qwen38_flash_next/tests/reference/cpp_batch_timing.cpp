// Times the C++ executor's decode forwards for the Rust port's speed checks
// (docs/RUST.md): four sessions prefilled
// with the first len - 7 i prompt tokens, then ROUNDS batched decode
// forwards of 2, 2, 1 and 1 tokens (ForwardBatch plus SelectBatchLogits of
// every chain), and ROUNDS single-token forwards of session 0 (captured
// decode graphs). Prints the upload time and the mean time of each after
// ROUNDS / 4 warm-up rounds. Build like cpp_forward_dump.cpp:
//   ./cpp_batch_timing MODEL PROMPT ROUNDS
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace fn = gufo::models::qwen38_flash_next;

namespace {

void Check(bool ok, const std::string& what, const std::string& error) {
  if (!ok) {
    std::cerr << what << ": " << error << "\n";
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " MODEL PROMPT ROUNDS\n";
    return 2;
  }
  std::ifstream in(argv[2], std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  std::vector<std::int32_t> prompt(bytes.size() / 4);
  std::copy_n(bytes.data(), prompt.size() * 4,
              reinterpret_cast<char*>(prompt.data()));
  const int rounds = std::atoi(argv[3]);
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  Check(reader != nullptr, "open", error);
  auto weights = fn::ModelWeights::Bind(*reader, &error);
  Check(weights.has_value(), "bind", error);
  const auto& c = weights->config;
  const auto& t = weights->ple_table;
  auto ngram = fn::NgramTable::Open(
      reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
      t.rows, c.ple_head_dim, t.type, &error);
  Check(ngram != nullptr, "n-gram table", error);
  const auto upload_start = std::chrono::steady_clock::now();
  auto device = fn::rocm::DeviceModel::Upload(*weights, *reader, nullptr,
                                              nullptr, &error);
  Check(device != nullptr, "upload", error);
  std::cout << "upload "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             upload_start)
                   .count()
            << " s\n"
            << std::flush;
  fn::rocm::Executor::Options options;
  options.max_batch = 2048;
  options.max_logit_rows = 8;
  options.max_speculative = 8;
  auto exec = fn::rocm::Executor::Create(*device, ngram.get(), options, &error);
  Check(exec != nullptr, "executor", error);
  std::vector<std::unique_ptr<fn::rocm::Session>> sessions;
  std::vector<float> logits(8 * c.vocab_size);
  for (std::size_t i = 0; i < 4; ++i) {
    auto session = exec->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                       4096, &error);
    Check(session != nullptr, "session", error);
    Check(exec->Forward(
              *session, std::span(prompt).first(prompt.size() - 7 * i), 1,
              logits.data(), fn::rocm::Executor::ForwardMode::kPrefill, &error),
          "prefill", error);
    sessions.push_back(std::move(session));
  }
  const std::uint32_t widths[] = {2, 2, 1, 1};
  const std::int32_t tokens[] = {1000, 1001};
  const auto warmup = rounds / 4;
  double batch_ms = 0;
  for (int round = 0; round < rounds + warmup; ++round) {
    std::vector<fn::rocm::Executor::BatchItem> items;
    for (std::size_t i = 0; i < 4; ++i)
      items.push_back({sessions[i].get(), std::span(tokens, widths[i]), false});
    const auto start = std::chrono::steady_clock::now();
    Check(exec->ForwardBatch(items, &error), "batch", error);
    std::uint32_t offset = 0;
    for (const auto width : widths) {
      Check(exec->SelectBatchLogits(offset, width, logits.data(), &error),
            "select", error);
      offset += width;
    }
    if (round >= warmup)
      batch_ms += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  }
  double single_ms = 0;
  for (int round = 0; round < rounds + warmup; ++round) {
    const auto start = std::chrono::steady_clock::now();
    Check(exec->Forward(*sessions[0], std::span(tokens, 1), 1, logits.data(),
                        fn::rocm::Executor::ForwardMode::kDecode, &error),
          "decode", error);
    if (round >= warmup)
      single_ms += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  }
  std::cout << "batch forward " << batch_ms / rounds << " ms, single forward "
            << single_ms / rounds << " ms\n";
  return 0;
}
