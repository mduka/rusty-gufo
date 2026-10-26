// Reference tokens, logits and snapshots of the C++ Flash-Next engine for the
// Rust port (docs/RUST.md). Raw little-endian
// files, the names the Rust tests read. With the MTP sidecar and K sessions:
// session i syncs to the first len - 7 * i prompt tokens (K = 1: all of
// them) and writes sync_<i>.bin (its snapshot after the prompt); then STEPS
// rounds of DecodeStep (K = 1) or DecodeBatch (K > 1) with a budget of
// eight tokens and stop_at_eos off, after which it writes tokens_<i>.bin
// (u32, every generated token), steps_<i>.bin (u32 tokens per round),
// logits_<i>.bin (f32, the last token's logits), snapshot_<i>.bin and
// stats_<i>.bin (u64 cycles, drafted, accepted). TEMP 0 decodes greedily;
// otherwise the sampler has that temperature, top_k 40 and seed SEED + i.
// It also prints the sync and decode times.
//
// Build against the C++ baseline like cpp_forward_dump.cpp, compiling this
// file instead:
//   ./cpp_session_dump MODEL MTP PROMPT OUTDIR K STEPS TEMP SEED [MAX_CONTEXT]
#include <chrono>
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

#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

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

void Check(bool ok, const std::string& what, const std::string& error) {
  if (!ok) {
    std::cerr << what << ": " << error << "\n";
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 9) {
    std::cerr << "usage: " << argv[0]
              << " MODEL MTP PROMPT OUTDIR K STEPS TEMP SEED [MAX_CONTEXT]\n";
    return 2;
  }
  const auto prompt = ReadPrompt(argv[3]);
  const std::filesystem::path out = argv[4];
  const auto k = static_cast<std::size_t>(std::atoi(argv[5]));
  const int steps = std::atoi(argv[6]);
  const float temperature = std::strtof(argv[7], nullptr);
  const std::int64_t seed = std::atoll(argv[8]);
  fn::ModelOptions options;
  options.max_context =
      argc > 9 ? static_cast<std::uint32_t>(std::atoi(argv[9])) : 4096;
  options.mtp_model_path = argv[2];
  options.decode_concurrency = static_cast<std::uint32_t>(k);
  std::filesystem::create_directories(out);

  std::string error;
  auto model = fn::Model::Load(argv[1], options, &error);
  Check(model != nullptr, "load", error);
  std::vector<std::unique_ptr<fn::Session>> sessions;
  std::vector<gufo::sampling::SamplerState> samplers;
  for (std::size_t i = 0; i < k; ++i) {
    auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                        options.max_context, &error);
    Check(session != nullptr, "session", error);
    const auto length = prompt.size() - 7 * i;
    const auto sync_start = std::chrono::steady_clock::now();
    Check(session->Sync(std::span(prompt).first(length), &error), "sync",
          error);
    std::cout << "session " << i << ": sync of " << length << " tokens in "
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - sync_start)
                     .count()
              << " ms\n";
    auto snapshot = session->SaveSnapshot(&error);
    Check(snapshot != nullptr, "snapshot", error);
    Write(out / ("sync_" + std::to_string(i) + ".bin"), snapshot->bytes());
    gufo::sampling::SamplingConfig config;
    if (temperature > 0) {
      config.temperature = temperature;
      config.top_k = 40;
      config.seed = seed + static_cast<std::int64_t>(i);
    }
    samplers.emplace_back(config);
    sessions.push_back(std::move(session));
  }
  std::vector<std::vector<std::uint32_t>> tokens(k), counts(k);
  std::vector<fn::Session::DecodeResult> results(k);
  std::vector<fn::Session::BatchOutcome> outcomes(k);
  const auto decode_start = std::chrono::steady_clock::now();
  auto warm_start = decode_start;
  std::size_t warm_tokens = 0;
  for (int step = 0; step < steps; ++step) {
    if (step == steps / 2)
      warm_start = std::chrono::steady_clock::now();
    if (k == 1) {
      Check(sessions[0]->DecodeStep(8, samplers[0], &results[0], &error, false),
            "decode", error);
    } else {
      std::vector<fn::Session::DecodeRequest> requests;
      for (std::size_t i = 0; i < k; ++i) {
        requests.push_back({sessions[i].get(), 8, &samplers[i], &results[i],
                            false, &outcomes[i]});
      }
      Check(fn::Session::DecodeBatch(requests, &error), "decode batch", error);
    }
    for (std::size_t i = 0; i < k; ++i) {
      if (step >= steps / 2)
        warm_tokens += results[i].tokens.size();
      counts[i].push_back(static_cast<std::uint32_t>(results[i].tokens.size()));
      for (const auto token : results[i].tokens)
        tokens[i].push_back(static_cast<std::uint32_t>(token));
    }
  }
  const double decode_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - decode_start)
                               .count();
  std::size_t generated = 0;
  for (const auto& t : tokens)
    generated += t.size();
  std::cout << "decode: " << generated << " tokens in " << decode_ms << " ms\n";
  for (std::size_t i = 0; i < k; ++i) {
    const auto name = [&](const char* what) {
      return out / (std::string(what) + "_" + std::to_string(i) + ".bin");
    };
    Write(name("tokens"), std::span<const std::uint32_t>(tokens[i]));
    Write(name("steps"), std::span<const std::uint32_t>(counts[i]));
    Write(name("logits"), sessions[i]->Logits());
    auto snapshot = sessions[i]->SaveSnapshot(&error);
    Check(snapshot != nullptr, "snapshot", error);
    Write(name("snapshot"), snapshot->bytes());
    const auto& stats = sessions[i]->Statistics();
    const std::uint64_t values[] = {stats.cycles, stats.drafted,
                                    stats.accepted};
    Write(name("stats"), std::span<const std::uint64_t>(values));
    std::cout << "session " << i << ": " << tokens[i].size() << " tokens, "
              << stats.accepted << "/" << stats.drafted << " drafts accepted\n";
  }
  return 0;
}
