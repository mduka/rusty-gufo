// Reference token IDs and logits of the C++ Flash-Next executor for the Rust
// port (docs/RUST.md). Raw little-endian
// files, the names the Rust tests read:
//   prefill: the prompt in max_batch chunks through Forward(kPrefill), then
//            N greedy steps through Forward(kDecode); ids.bin (u32: the
//            first token after the prompt, then one per step) and
//            logits_<step>.bin (f32, vocabulary) for step 0..=N.
//   decode:  every prompt token through Forward(kDecode), then N greedy
//            steps; the same files.
// PROMPT is a file of u32 token IDs. An autoregressive session without the
// MTP block; max_logit_rows = max_speculative = 1, as the C++ engine sets
// them without MTP.
//
// Build against the C++ baseline (upstream clone, built as documented):
//   R=/opt/rocm/core-10.0 O=$UPSTREAM B=$O/build/release
//   F=$B/src/models/qwen38_flash_next
//   $R/llvm/bin/clang++ -std=c++20 -O2 -DENGINE_ENABLE_HIP=1 \
//     -D__HIP_PLATFORM_AMD__ -I$O -isystem $R/include \
//     --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15 cpp_forward_dump.cpp \
//     -o cpp_forward_dump $F/libgufo_qwen38_flash_next.a \
//     $F/libgufo_qwen38_flash_next_mmq.a $B/libgufo_core.a -L$R/lib \
//     -Wl,-rpath,$R/lib -lhipblaslt -lhipblas -lrocblas -lamdhip64 -licuuc \
//     -licui18n -lcrypto -lpng -lz -ljpeg -lcurl -lgomp -lpthread -ldl
//   ./cpp_forward_dump MODEL PROMPT OUTDIR prefill|decode N [MAX_CONTEXT]
//   [MAX_BATCH]
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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

template<typename T>
void Write(const std::filesystem::path& path, std::span<const T> values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size_bytes()));
  if (!out)
    throw std::runtime_error("cannot write " + path.string());
}

std::vector<std::int32_t> ReadPrompt(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  std::vector<std::int32_t> ids(bytes.size() / 4);
  std::copy_n(bytes.data(), ids.size() * 4,
              reinterpret_cast<char*>(ids.data()));
  return ids;
}

std::int32_t Argmax(std::span<const float> logits) {
  return static_cast<std::int32_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

void Check(bool ok, const std::string& what, const std::string& error) {
  if (!ok) {
    std::cerr << what << ": " << error << "\n";
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: " << argv[0]
              << " MODEL PROMPT OUTDIR prefill|decode N [MAX_CONTEXT] "
                 "[MAX_BATCH]\n";
    return 2;
  }
  const std::string model_path = argv[1];
  const auto prompt = ReadPrompt(argv[2]);
  const std::filesystem::path out = argv[3];
  const std::string mode = argv[4];
  const int steps = std::atoi(argv[5]);
  const std::uint32_t max_context =
      argc > 6 ? static_cast<std::uint32_t>(std::atoi(argv[6])) : 4096;
  const std::uint32_t max_batch =
      argc > 7 ? static_cast<std::uint32_t>(std::atoi(argv[7]))
               : std::min<std::uint32_t>(2048, max_context);
  std::filesystem::create_directories(out);

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  Check(reader != nullptr, "open", error);
  auto weights = fn::ModelWeights::Bind(*reader, &error);
  Check(weights.has_value(), "bind", error);
  const auto& c = weights->config;
  std::unique_ptr<fn::NgramTable> ngram;
  if (c.ple_layer >= 0) {
    const auto& t = weights->ple_table;
    ngram = fn::NgramTable::Open(
        reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, &error);
    Check(ngram != nullptr, "n-gram table", error);
  }
  auto device = fn::rocm::DeviceModel::Upload(*weights, *reader, nullptr,
                                              nullptr, &error);
  Check(device != nullptr, "upload", error);
  fn::rocm::Executor::Options options;
  options.max_batch = max_batch;
  options.max_logit_rows = 1;
  options.max_speculative = 1;
  auto exec = fn::rocm::Executor::Create(*device, ngram.get(), options, &error);
  Check(exec != nullptr, "executor", error);
  auto session = exec->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     max_context, &error);
  Check(session != nullptr, "session", error);

  const std::size_t vocab = c.vocab_size;
  std::vector<float> logits(vocab);
  using Mode = fn::rocm::Executor::ForwardMode;
  if (mode == "prefill") {
    for (std::size_t off = 0; off < prompt.size(); off += max_batch) {
      const auto n = std::min<std::size_t>(max_batch, prompt.size() - off);
      Check(exec->Forward(*session, std::span(prompt).subspan(off, n), 1,
                          logits.data(), Mode::kPrefill, &error),
            "prefill", error);
    }
  } else if (mode == "decode") {
    for (const auto token : prompt) {
      Check(exec->Forward(*session, std::span(&token, 1), 1, logits.data(),
                          Mode::kDecode, &error),
            "decode", error);
    }
  } else {
    std::cerr << "unknown mode " << mode << "\n";
    return 2;
  }
  std::vector<std::uint32_t> ids;
  for (int step = 0;; ++step) {
    Write(out / ("logits_" + std::to_string(step) + ".bin"),
          std::span<const float>(logits));
    const auto next = Argmax(logits);
    ids.push_back(static_cast<std::uint32_t>(next));
    if (step == steps)
      break;
    Check(exec->Forward(*session, std::span(&next, 1), 1, logits.data(),
                        Mode::kDecode, &error),
          "step", error);
  }
  Write(out / "ids.bin", std::span<const std::uint32_t>(ids));
  std::cout << "wrote " << ids.size() << " tokens to " << out << "\n";
  return 0;
}
