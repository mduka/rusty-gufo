// The C++ side of gufo-gemma4's comparison with the extension
// franzmoca/gemma4-gufo-extension (commit a31ce42, on Gufo 0e2de43e;
// docs/development/models/gemma-4/CHECKS.md): evaluates a prompt, dumps
// float32 logits of every prompt row (raw [n][vocab]), optionally continues
// with DecodeStep (MTP when --mtp is given) and reports rates and drafting
// statistics. examples/gemma4_runtime_probe.rs is the Rust side with the same
// arguments and output. This project's code (MIT OR Apache-2.0), written
// against the extension's headers.
//
// Build in the extension's tree ($EXTENSION at a31ce42): copy this file to
// $EXTENSION/tests/runtime_probe.cpp, apply runtime-probe-hook.patch (a
// gemma4_runtime_probe test target in extension.cmake and extension.json),
// then configure Gufo 0e2de43e ($UPSTREAM) with the cmake preset gpu-test on
// ROCm 10.1 ($ROCM_PATH) and build the target, from this directory:
//   cp runtime_probe.cpp $EXTENSION/tests/
//   git -C $EXTENSION apply $PWD/runtime-probe-hook.patch
//   cd $UPSTREAM && export HIP_PATH=$ROCM_PATH R=$ROCM_PATH
//   G=$(dirname "$(cc -print-libgcc-file-name)")
//   cmake --preset gpu-test -B build/gemma4-gpu -DROCM_PATH=$R \
//     -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R \
//     "-DCMAKE_HIP_FLAGS=--rocm-path=$R -isystem $R/include \
//       --gcc-install-dir=$G" -DGUFO_EXTENSION_DIRS=$EXTENSION
//   cmake --build build/gemma4-gpu --target gemma4_runtime_probe
// The binary, build/gemma4-gpu/extensions/build/gemma4/build/
// gemma4_runtime_probe, has no RUNPATH; run it on the ROCm it was built with:
//   env LD_LIBRARY_PATH=$ROCM_PATH/lib gemma4_runtime_probe --model GGUF
//       [--mtp GGUF] (--tokens 1,2 | --text T | --text-file P | --chat T)
//       [--dump L.bin] [--generate N] [--context N] [--step]
//       [--draft-tokens N] [--draft-policy calibrated|confidence|fixed]
//       [--temperature T] [--seed S] [--top-k K] [--top-p P]
// and the Rust side with the same arguments (cargo run --release -p
// gufo-gemma4 --example gemma4_runtime_probe -- ...); compare the --dump
// files byte for byte and the printed ids, drafting statistics and text.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "chat_template.hpp"
#include "check.hpp"
#include "engine.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

int main(int argc, char** argv) {
  std::string model, mtp, tokens_arg, text, chat, dump, policy = "calibrated";
  std::size_t generate = 0;
  std::uint32_t context = 8192, draft_tokens = g4::kDefaultDraftTokens;
  bool step = false;
  gufo::sampling::SamplingConfig sampling;
  sampling.temperature = 0.0F;
  sampling.seed = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        std::exit(2);
      return argv[++i];
    };
    if (arg == "--model")
      model = value();
    else if (arg == "--mtp")
      mtp = value();
    else if (arg == "--tokens")
      tokens_arg = value();
    else if (arg == "--text")
      text = value();
    else if (arg == "--text-file") {
      std::ifstream in(value(), std::ios::binary);
      std::stringstream s;
      s << in.rdbuf();
      text = s.str();
    } else if (arg == "--chat")
      chat = value();
    else if (arg == "--dump")
      dump = value();
    else if (arg == "--generate")
      generate = std::stoul(value());
    else if (arg == "--context")
      context = std::stoul(value());
    else if (arg == "--step")
      step = true;
    else if (arg == "--draft-tokens")
      draft_tokens = std::stoul(value());
    else if (arg == "--draft-policy")
      policy = value();
    else if (arg == "--temperature")
      sampling.temperature = std::stof(value());
    else if (arg == "--seed")
      sampling.seed = std::stoull(value());
    else if (arg == "--top-k")
      sampling.top_k = std::stoi(value());
    else if (arg == "--top-p")
      sampling.top_p = std::stof(value());
    else {
      std::cerr << "unknown argument " << arg << '\n';
      return 2;
    }
  }
  return gemma4_test::Run([&] {
    std::string error;
    g4::ModelOptions options;
    options.max_context = context;
    options.mtp_model_path = mtp;
    options.draft_tokens = draft_tokens;
    options.draft_policy = policy == "fixed" ? g4::DraftPolicy::kFixed
                           : policy == "confidence"
                               ? g4::DraftPolicy::kConfidence
                               : g4::DraftPolicy::kCalibrated;
    options.draft_calibration = g4::DraftCalibrationScope::kRequest;
    const auto load_start = std::chrono::steady_clock::now();
    auto m = g4::Model::Load(model, options, &error);
    Require(m != nullptr, error);
    std::printf("loaded %.2f GB in %.1f s\n", m->ResidentBytes() / 1e9,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              load_start)
                    .count());
    std::vector<g4::TokenId> tokens;
    if (!chat.empty()) {
      gufo::ext::ChatMessage message(gufo::ext::ChatRole::kUser, chat);
      const auto rendered =
          g4::ChatTemplate::Render(std::span(&message, 1), {}, {}, &error);
      Require(rendered.has_value(), error);
      tokens = m->Tokenize(rendered->text);
    } else if (!text.empty()) {
      tokens = m->Tokenize(text);
    } else {
      for (std::stringstream list(tokens_arg); list.good();) {
        std::string item;
        std::getline(list, item, ',');
        if (!item.empty())
          tokens.push_back(std::stoi(item));
      }
    }
    Require(tokens.size() > 1, "no prompt");
    const std::size_t vocab = m->VocabSize();
    auto session = m->CreateSession(0, &error);
    Require(session != nullptr, error);
    std::vector<float> logits;
    const auto prefill_start = std::chrono::steady_clock::now();
    if (step) {
      Require(session->Sync(std::span(tokens).first(1), &error), error);
      auto row = session->Logits();
      logits.insert(logits.end(), row.begin(), row.end());
      for (std::size_t i = 1; i < tokens.size(); ++i) {
        Require(session->Evaluate(tokens[i], &error), error);
        row = session->Logits();
        logits.insert(logits.end(), row.begin(), row.end());
      }
    } else if (!dump.empty()) {
      Require(session->EvaluateAll(tokens, &logits, &error), error);
    } else {
      Require(session->Sync(tokens, &error), error);
    }
    const double prefill = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - prefill_start)
                               .count();
    std::printf("prefill %zu tokens in %.3f s (%.1f tokens/s)\n", tokens.size(),
                prefill, tokens.size() / prefill);
    if (!dump.empty()) {
      Require(logits.size() == tokens.size() * vocab, "logit rows");
      std::ofstream out(dump, std::ios::binary);
      out.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * 4));
      Require(out.good(), "cannot write " + dump);
    }
    if (generate > 0) {
      gufo::sampling::SamplerState sampler(sampling, {});
      std::vector<g4::TokenId> generated;
      const auto start = std::chrono::steady_clock::now();
      while (generated.size() < generate) {
        g4::Session::DecodeResult result;
        Require(session->DecodeStep(generate - generated.size(), sampler,
                                    &result, &error),
                error);
        generated.insert(generated.end(), result.tokens.begin(),
                         result.tokens.end());
        if (result.stop || result.tokens.empty())
          break;
      }
      const double decode = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - start)
                                .count();
      const auto& s = session->Statistics();
      if (s.cycles != 0) {
        std::printf(
            "cycles %llu, drafted %llu, accepted %llu, copied %llu/%llu, "
            "siblings %llu/%llu\n",
            (unsigned long long)s.cycles, (unsigned long long)s.drafted,
            (unsigned long long)s.accepted,
            (unsigned long long)s.copied_accepted, (unsigned long long)s.copied,
            (unsigned long long)s.siblings_accepted,
            (unsigned long long)s.siblings);
      }
      std::printf("generated %zu tokens in %.3f s (%.2f tokens/s)\n",
                  generated.size(), decode, generated.size() / decode);
      std::printf("ids:");
      for (const auto t : generated)
        std::printf(" %d", t);
      std::printf("\ntext: %s\n", m->Decode(generated).c_str());
    }
  });
}
