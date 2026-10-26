// Reference token IDs and logits of the C++ Qwen3.8 engine for the Rust port
// (docs/RUST.md). Raw little-endian
// files, the same names as the Rust tests read:
//   decode:   every prompt token through ForwardToken, then N greedy steps;
//             ids.bin (u32: first token after the prompt, then one per step)
//             and logits_<step>.bin (f32, vocabulary) for step 0..=N.
//   prefill:  the prompt through ForwardPromptBatch, then N greedy steps;
//             the same files.
//   generate: Generate(prompt, max_new_tokens = N) -> ids.bin.
//   dflash:   DFlash2 speculative decoding as `gufo bench --speculative
//             dflash2` runs it (7 draft tokens, batched verification, EOS
//             ignored): Prime(prompt), then VerifyStep until N tokens; ids.bin
//             and steps.txt ("draft_count accepted_count emitted..." per step).
//             Needs DFLASH_MODEL as the last argument.
//   session:  Generate(prompt, N) -> ids.bin, then GenerateFromPrefix over
//             prompt + those tokens + SUFFIX (u32 file) with the first two
//             parts retained -> ids2.bin.
//   mtp-head: the prompt through ForwardPromptBatch with the final hidden
//             rows captured (hidden.bin, f32 [token][hidden]), the MTP head
//             primed with them as QwenMtpGpuDraftBackend does, then N + 1
//             greedy head steps from the target's first token: ids.bin (the
//             target's token, then one per head step) and logits_<step>.bin.
//             Needs MTP_MODEL as the last argument.
//   mtp:      MTP speculative decoding as `gufo prompt --speculative mtp`
//             runs it (7 draft tokens, adaptive length, batched
//             verification, EOS ignored); ids.bin and steps.txt as dflash.
//             Needs MTP_MODEL as the last argument.
// PROMPT is a file of u32 token IDs.
//
// Build against the C++ baseline (upstream clone, built as documented):
//   R=/opt/rocm/core-10.0 O=$UPSTREAM B=$O/build/release
//   $R/llvm/bin/clang++ -std=c++20 -O2 -DENGINE_ENABLE_HIP=1 \
//     -D__HIP_PLATFORM_AMD__ -I$O -isystem $R/include \
//     --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15 cpp_decode_dump.cpp \
//     -o cpp_decode_dump $B/libgufo_core.a -L$R/lib -Wl,-rpath,$R/lib \
//     -lhipblaslt -lhipblas -lrocblas -lamdhip64 -licuuc -licui18n -lcrypto \
//     -lpng -lz -ljpeg -lcurl -lgomp -lpthread -ldl
//   ./cpp_decode_dump MODEL PROMPT OUTDIR decode|prefill|generate N
//   [MAX_CONTEXT]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/qwen/dflash_policy.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"

namespace {

template<typename T>
void Write(const std::filesystem::path& path, std::span<const T> values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size_bytes()));
  if (!out)
    throw std::runtime_error("cannot write " + path.string());
}

std::vector<std::uint32_t> ReadIds(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
  if (bytes.empty() || bytes.size() % 4 != 0)
    throw std::runtime_error("bad prompt file " + path.string());
  std::vector<std::uint32_t> ids(bytes.size() / 4);
  std::memcpy(ids.data(), bytes.data(), bytes.size());
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: " << argv[0]
              << " MODEL PROMPT OUTDIR "
                 "decode|prefill|generate|session|dflash|mtp-head|mtp N "
                 "[MAX_CONTEXT] [SUFFIX|DFLASH_MODEL|MTP_MODEL]\n";
    return 2;
  }
  try {
    const std::filesystem::path out_dir = argv[3];
    const std::string mode = argv[4];
    const auto steps = static_cast<std::size_t>(std::stoul(argv[5]));
    const auto max_context =
        argc > 6 ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 4096U;
    const auto prompt = ReadIds(argv[2]);
    std::filesystem::create_directories(out_dir);

    std::string error;
    std::shared_ptr<const gufo::core::GgufReader> reader =
        gufo::core::GgufReader::OpenFile(argv[1], &error);
    if (!reader)
      throw std::runtime_error(error);
    auto executor =
        gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &error, max_context);
    if (!executor)
      throw std::runtime_error(error);

    std::vector<std::uint32_t> ids;
    if (mode == "mtp-head") {
      if (argc < 8)
        throw std::runtime_error("mtp-head needs an MTP_MODEL");
      std::shared_ptr<const gufo::core::GgufReader> mtp_reader =
          gufo::core::GgufReader::OpenFile(argv[7], &error);
      if (!mtp_reader)
        throw std::runtime_error(error);
      auto model = gufo::hip::QwenMtpGpuModel::Create(
          mtp_reader, executor->GetSharedModel(), &error);
      if (!model)
        throw std::runtime_error("MTP: " + error);
      auto head =
          gufo::hip::QwenMtpGpuExecutor::Create(model, max_context, &error);
      if (!head)
        throw std::runtime_error("MTP: " + error);
      executor->Reset();
      executor->SetPromptHiddenCapture(true, {});
      auto token = executor->ForwardPromptBatch(prompt, 0);
      const auto hidden = executor->GetPromptHiddenStates();
      Write(out_dir / "hidden.bin", hidden);
      const std::size_t width = hidden.size() / prompt.size();
      for (std::size_t row = 0; row + 1 < prompt.size(); ++row) {
        (void)head->ForwardTargetHidden(prompt[row + 1],
                                        hidden.subspan(row * width, width),
                                        static_cast<std::uint32_t>(row), false);
      }
      auto pos = static_cast<std::uint32_t>(prompt.size() - 1);
      ids.push_back(token);
      for (std::size_t step = 0; step <= steps; ++step) {
        token = step == 0 ? head->ForwardTargetHidden(
                                token, hidden.subspan(pos * width, width), pos)
                          : head->ForwardFeedback(token, pos);
        ++pos;
        ids.push_back(token);
        Write(out_dir / ("logits_" + std::to_string(step) + ".bin"),
              head->CopyLastLogits());
      }
    } else if (mode == "mtp") {
      if (argc < 8)
        throw std::runtime_error("mtp needs an MTP_MODEL");
      gufo::hip::QwenMtpGpuDraftConfig draft_config{
          .max_context = max_context,
          .max_draft_tokens = 7,
          .vision_input = &executor->VisionInput(),
      };
      auto draft = gufo::hip::QwenMtpGpuDraftBackend::CreateFromGguf(
          argv[7], executor->GetSharedModel(), draft_config, &error);
      if (!draft)
        throw std::runtime_error("MTP: " + error);
      gufo::speculative::SpeculativeOptions options;
      options.max_draft_tokens = 7;
      options.min_draft_tokens = 1;
      options.initial_draft_tokens = 7;
      options.use_batched_verification = true;
      gufo::speculative::SpeculativeVerifier verifier(
          *executor, std::move(draft), options);
      std::vector<std::uint32_t> sequence = prompt;
      auto current = verifier.Prime(sequence);
      auto pos = static_cast<std::uint32_t>(sequence.size());
      sequence.push_back(current);
      std::ofstream log(out_dir / "steps.txt");
      while (ids.size() < steps) {
        const auto step =
            verifier.VerifyStep(sequence, pos, current, 999999,
                                static_cast<std::uint32_t>(steps - ids.size()));
        log << step.draft_count << ' ' << step.accepted_count;
        for (const auto token : step.emitted_tokens) {
          log << ' ' << token;
          if (ids.size() < steps) {
            ids.push_back(token);
            sequence.push_back(token);
            ++pos;
          }
        }
        log << '\n';
        current = step.next_token;
      }
    } else if (mode == "dflash") {
      if (argc < 8)
        throw std::runtime_error("dflash needs a DFLASH_MODEL");
      gufo::hip::QwenDFlashGpuDraftConfig draft_config{
          .max_context = max_context,
          .max_draft_tokens = 7,
          .policy = gufo::speculative::ParseDFlashDraftPolicy(""),
      };
      auto draft = gufo::hip::QwenDFlashGpuDraftBackend::CreateFromGguf(
          argv[7], executor->GetSharedModel(), draft_config, &error);
      if (!draft)
        throw std::runtime_error("DFlash2: " + error);
      gufo::speculative::SpeculativeOptions options;
      options.max_draft_tokens = 7;
      options.min_draft_tokens = 1;
      options.initial_draft_tokens = 7;
      options.enable_adaptive_draft_length = false;
      options.use_batched_verification = true;
      gufo::speculative::SpeculativeVerifier verifier(
          *executor, std::move(draft), options);
      std::vector<std::uint32_t> sequence = prompt;
      auto current = verifier.Prime(sequence);
      auto pos = static_cast<std::uint32_t>(sequence.size());
      sequence.push_back(current);
      std::ofstream log(out_dir / "steps.txt");
      while (ids.size() < steps) {
        const auto step =
            verifier.VerifyStep(sequence, pos, current, 999999,
                                static_cast<std::uint32_t>(steps - ids.size()));
        log << step.draft_count << ' ' << step.accepted_count;
        for (const auto token : step.emitted_tokens) {
          log << ' ' << token;
          if (ids.size() < steps) {
            ids.push_back(token);
            sequence.push_back(token);
            ++pos;
          }
        }
        log << '\n';
        current = step.next_token;
      }
    } else if (mode == "generate" || mode == "session") {
      gufo::models::GenerationOptions options;
      options.max_new_tokens = steps;
      ids = executor->Generate(prompt, options);
      if (mode == "session") {
        if (argc < 8)
          throw std::runtime_error("session needs a SUFFIX file");
        auto next_prompt = prompt;
        next_prompt.insert(next_prompt.end(), ids.begin(), ids.end());
        const auto retained = next_prompt.size();
        const auto suffix = ReadIds(argv[7]);
        next_prompt.insert(next_prompt.end(), suffix.begin(), suffix.end());
        const auto second =
            executor->GenerateFromPrefix(next_prompt, retained, options);
        Write(out_dir / "ids2.bin", std::span<const std::uint32_t>(second));
        for (const auto id : second)
          std::cout << id << ' ';
        std::cout << "| ";
      }
    } else if (mode == "decode" || mode == "prefill") {
      executor->Reset();
      std::uint32_t pos = 0;
      std::uint32_t next = 0;
      if (mode == "prefill") {
        next = executor->ForwardPromptBatch(prompt, 0);
        pos = static_cast<std::uint32_t>(prompt.size());
      } else {
        for (const auto token : prompt) {
          const bool last = pos + 1 == prompt.size();
          next = executor->ForwardToken(token, pos++, last);
        }
      }
      for (std::size_t step = 0;; ++step) {
        ids.push_back(next);
        const auto logits = executor->CopyLastLogits();
        Write(out_dir / ("logits_" + std::to_string(step) + ".bin"), logits);
        if (step == steps)
          break;
        next = executor->ForwardToken(next, pos++);
      }
    } else {
      throw std::runtime_error(
          "mode must be decode, prefill, generate, session, dflash, mtp-head "
          "or mtp");
    }
    Write(out_dir / "ids.bin", std::span<const std::uint32_t>(ids));
    for (const auto id : ids)
      std::cout << id << ' ';
    std::cout << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
