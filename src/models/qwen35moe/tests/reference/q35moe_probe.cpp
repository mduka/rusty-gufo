// C++ reference dumps of the Qwen3.6 35B-A3B (qwen35moe) runner of the
// extension slimsami/strycast-qwen35moe-gufo-extension (commit 4882b7a, on
// Gufo 0fb50ce4) for the tests of gufo-qwen35moe
// (docs/development/models/qwen3.6-35b-a3b/CHECKS.md). Mirrors
// src/models/qwen35/tests/reference/cpp_decode_dump.cpp (the dense Qwen 27B
// dump tool) on the extension's API and writes into OUTDIR one compact JSON
// file per mode, meta.json, timings.json, libraries.json (the HIP-stack
// libraries the process mapped) and a few full logit rows under rows/;
// record_fixtures.py formats them into the fixtures. This project's code
// (MIT OR Apache-2.0), written against the extension's headers.
//
// Build in the extension's tree (README.md beside this file): copy this file
// and q35moe_probe.cmake into $EXTENSION/tests/, apply q35moe-probe-hook.patch
// (and vision-gate.patch for mode vision), configure Gufo 0fb50ce4
// ($UPSTREAM) with the cmake preset gpu-test and
// -DGUFO_EXTENSION_DIRS=$EXTENSION, then
//   cmake --build build/q35moe-gpu --target q35moe_probe
// The binary has no RUNPATH; run it on the ROCm it was built with:
//   env LD_LIBRARY_PATH=$ROCM_PATH/lib q35moe_probe MODEL OUTDIR
//       [--max-context N] [--kv-cache f16|q8_0] [--mtp-model PATH]
//       [--dflash-model PATH] [--modes a,b,...] [--mmproj PATH]
//       [--images a.png,b.png]
//
// Modes (entry points in brackets, all strycast::hip unless noted):
//   prefill   [QwenGpuExecutor::ForwardPromptBatch, then ForwardToken] cold
//             prefill of several lengths (crossing the 8-row small-chunk,
//             128/256/512/1024-row route and 4096-row chunk boundaries), plus
//             incremental prefills over pieces (ForwardPromptBatch at
//             offsets, as the serve runner's Prefill with a token budget),
//             then greedy steps; per step token, FNV-1a 64 of the f32 logits
//             bytes (CopyLastLogits) and the top 5; a few full rows as .f32.
//             Also Generate(prompt, max_new) greedy ids.
//   decode    [ForwardToken] every prompt token through ForwardToken (logits
//             on the last only), then N greedy steps.
//   session   [Generate, GenerateFromPrefix] as cpp_decode_dump's session.
//   snapshot  [SaveSnapshot / RestoreSnapshot / SerializeCompact /
//             RestoreCompactSnapshot, ForwardPromptBatch at an offset].
//   batched   [ForwardTokenBatch] 2, 3 and 4 sessions (own executors on the
//             shared QwenGpuModel) at different positions, against the same
//             sessions decoded alone.
//   sampled   [Generate with a seeded SamplingConfig].
//   mtp-head  [QwenMtpGpuExecutor: PrimeTargetHiddenBatch, ForwardTargetHidden,
//             ForwardFeedback, full head] over the target's prompt hidden rows.
//   mtp       [SpeculativeVerifier (Gufo core) over QwenMtpGpuDraftBackend:
//             Prime, VerifyStep] as `gufo prompt --speculative mtp`
//             (drivers.cpp CreateQwenVerifier), EOS ignored.
//   dflash    the same over QwenDFlashGpuDraftBackend as `gufo prompt
//             --speculative dflash2`; "serve" cases as the serve runner
//             configures it (lookup copy of up to 64 rows, verifier width 63).
//   vision    [vision::Encoder::Open, vision::Prepare, Encoder::Encode,
//             QwenGpuExecutor::ConfigureVision, ForwardPromptBatch,
//             ForwardToken, Generate; MTP/DFlash verifiers] image prompts as
//             `gufo prompt --image --think off` builds them (drivers.cpp
//             LoadQwenVision, AttachImages, PrepareVision), with
//             --mmproj and --images. Needs vision-gate.patch for
//             the 2048-wide Qwen3.6-35B-A3B.
//   mtp-batch, dflash-batch  [SpeculativeVerifier::VerifyBatch] two sessions
//             as the serve runner's DecodeBatch (serve configuration,
//             ConcurrentDraftCap applied with SetStepDraftCap).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "gufo/ext/crypto.hpp"
#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/image.hpp"
#include "gufo/ext/sampling.hpp"
#include "gufo/ext/speculative_verifier.hpp"
#include "q35/family/text/chat_template.hpp"
#include "q35/family/vision/encoder.hpp"
#include "q35/family/vision/prompt.hpp"
#include "q35/model/dflash_policy.hpp"
#include "q35/model/generator.hpp"
#include "q35/model/hip/dflash.hpp"
#include "q35/model/hip/executor.hpp"
#include "q35/model/hip/mtp.hpp"
#include "q35/model/hip/qwen_speculative_verifier.hpp"

namespace {

namespace qh = strycast::hip;
namespace spec = strycast::speculative;
namespace qv = strycast::models::qwen::vision;
using Ids = std::vector<std::uint32_t>;

constexpr std::uint32_t kIgnoredEos = 999999;
constexpr std::size_t kTopK = 5;

// --- prompts (as rusty-gufo's qwen3.8-27b fixtures) -------------------------

const Ids kHead = {248045, 846, 198};
const Ids kTail = {248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
// "<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n<|im_start|>"
// "assistant\n<think>\n\n</think>\n\n"
const Ids kHaiku = {248045, 846, 198,    7734, 264,    6185, 36974,
                    883,    279, 9117,   13,   248046, 198,  248045,
                    74455,  198, 248068, 271,  248069, 271};
// The 27B session fixture's follow-up turn.
const Ids kSuffix = {248046, 198,    248045, 846,    198, 6820,   799,
                     883,    22767,  13,     248046, 198, 248045, 74455,
                     198,    248068, 271,    248069, 271};

Ids Synthetic(std::size_t body) {
  Ids ids = kHead;
  for (std::size_t i = 0; i < body; ++i)
    ids.push_back(static_cast<std::uint32_t>(1000 + (7919 * i + 13) % 50000));
  ids.insert(ids.end(), kTail.begin(), kTail.end());
  return ids;
}

Ids HaikuPrefix(std::size_t length) {
  return {kHaiku.begin(), kHaiku.begin() + static_cast<std::ptrdiff_t>(length)};
}

// --- hashing and JSON -------------------------------------------------------

std::string Fnv1a64(std::span<const float> values) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  for (std::size_t i = 0; i < values.size_bytes(); ++i) {
    hash ^= bytes[i];
    hash *= 0x100000001b3ULL;
  }
  char text[17];
  std::snprintf(text, sizeof(text), "%016llx",
                static_cast<unsigned long long>(hash));
  return text;
}

std::string Float(float value) {
  if (std::isnan(value))
    return "\"nan\"";
  if (std::isinf(value))
    return value > 0 ? "\"inf\"" : "\"-inf\"";
  char text[32];
  std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(value));
  return text;
}

std::string Quote(std::string_view text) {
  std::string out = "\"";
  std::size_t i = 0;
  while (i < text.size()) {
    const auto c = static_cast<unsigned char>(text[i]);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
      ++i;
    } else if (c < 0x20) {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
      out += escaped;
      ++i;
    } else if (c < 0x80) {
      out += static_cast<char>(c);
      ++i;
    } else {
      // Pass valid UTF-8 through, replace anything else.
      std::size_t length = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
      bool valid = length != 0 && i + length <= text.size() && c < 0xF5;
      for (std::size_t k = 1; valid && k < length; ++k)
        valid = (static_cast<unsigned char>(text[i + k]) & 0xC0) == 0x80;
      if (valid) {
        out.append(text.substr(i, length));
        i += length;
      } else {
        out += "\\ufffd";
        ++i;
      }
    }
  }
  return out + "\"";
}

template<typename T>
std::string Array(const std::vector<T>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      out += ", ";
    if constexpr (std::is_same_v<T, std::string>)
      out += values[i];
    else
      out += std::to_string(values[i]);
  }
  return out + "]";
}

std::vector<std::string> Quoted(const std::vector<std::string>& values) {
  std::vector<std::string> out;
  for (const auto& v : values)
    out.push_back(Quote(v));
  return out;
}

/// The top `k` logits, descending, lowest id first on ties: [[id, value]...].
std::string Top(std::span<const float> logits, std::size_t k = kTopK) {
  std::vector<std::uint32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0U);
  k = std::min(k, order.size());
  std::partial_sort(
      order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k),
      order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
      });
  std::string out = "[";
  for (std::size_t i = 0; i < k; ++i) {
    if (i)
      out += ", ";
    out +=
        "[" + std::to_string(order[i]) + ", " + Float(logits[order[i]]) + "]";
  }
  return out + "]";
}

/// A JSON object assembled member by member.
class Obj {
public:
  Obj& Add(std::string_view key, const std::string& raw) {
    if (!body_.empty())
      body_ += ", ";
    body_ += Quote(key) + ": " + raw;
    return *this;
  }
  Obj& Str(std::string_view key, std::string_view value) {
    return Add(key, Quote(value));
  }
  Obj& Num(std::string_view key, std::uint64_t value) {
    return Add(key, std::to_string(value));
  }
  Obj& Bool(std::string_view key, bool value) {
    return Add(key, value ? "true" : "false");
  }
  [[nodiscard]] std::string Json() const { return "{" + body_ + "}"; }

private:
  std::string body_;
};

std::string Join(const std::vector<std::string>& items) {
  std::string out = "[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i)
      out += ", ";
    out += items[i];
  }
  return out + "]";
}

void WriteText(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out(path);
  out << text << '\n';
  if (!out)
    throw std::runtime_error("cannot write " + path.string());
}

void WriteRow(const std::filesystem::path& path, std::span<const float> row) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(row.data()),
            static_cast<std::streamsize>(row.size_bytes()));
  if (!out)
    throw std::runtime_error("cannot write " + path.string());
}

/// Greedy tokens with their logits digests and top candidates.
struct Trace {
  Ids ids;
  std::vector<std::string> fnv;
  std::vector<std::string> top;

  void Record(std::uint32_t token, std::span<const float> logits,
              bool with_top = true) {
    ids.push_back(token);
    fnv.push_back(Fnv1a64(logits));
    if (with_top)
      top.push_back(Top(logits));
  }
  void AddTo(Obj& obj) const {
    obj.Add("ids", Array(ids)).Add("logits_fnv1a64", Array(Quoted(fnv)));
    if (!top.empty())
      obj.Add("top5", Join(top));
  }
  [[nodiscard]] bool operator==(const Trace& other) const {
    return ids == other.ids && fnv == other.fnv;
  }
};

/// `steps` greedy ForwardToken steps after a frontier whose logits are the
/// executor's last; records the frontier and every step (steps + 1 entries).
Trace Continue(qh::QwenGpuExecutor& executor, std::uint32_t next,
               std::uint32_t pos, std::size_t steps, bool with_top = true) {
  Trace trace;
  for (std::size_t step = 0;; ++step) {
    trace.Record(next, executor.CopyLastLogits(), with_top);
    if (step == steps)
      break;
    next = executor.ForwardToken(next, pos++);
  }
  return trace;
}

struct Context {
  std::filesystem::path out_dir;
  std::shared_ptr<const strycast::core::GgufReader> reader;
  std::unique_ptr<qh::QwenGpuExecutor> executor;
  std::vector<std::unique_ptr<qh::QwenGpuExecutor>> extra;  // batched sessions
  std::uint32_t max_context{6144};
  qh::QwenExecutionPolicy policy;
  std::string model_path;
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string mmproj_path;
  std::vector<std::string> image_paths;
  std::shared_ptr<const qh::QwenMtpGpuModel> mtp_model;
  std::shared_ptr<const qh::QwenDFlashGpuModel> dflash_model;

  qh::QwenGpuExecutor& Session(std::size_t index) {
    if (index == 0)
      return *executor;
    while (extra.size() < index) {
      std::string error;
      auto session = qh::QwenGpuExecutor::Create(executor->GetSharedModel(),
                                                 &error, max_context, policy);
      if (!session)
        throw std::runtime_error("session executor: " + error);
      extra.push_back(std::move(session));
    }
    return *extra[index - 1];
  }

  const std::shared_ptr<const qh::QwenMtpGpuModel>& Mtp() {
    if (!mtp_model) {
      if (mtp_model_path.empty())
        throw std::runtime_error("MTP modes need --mtp-model");
      std::string error;
      std::shared_ptr<const strycast::core::GgufReader> mtp_reader =
          strycast::core::GgufReader::OpenFile(mtp_model_path, &error);
      if (!mtp_reader)
        throw std::runtime_error("MTP GGUF: " + error);
      mtp_model = qh::QwenMtpGpuModel::Create(
          std::move(mtp_reader), executor->GetSharedModel(), &error);
      if (!mtp_model)
        throw std::runtime_error("MTP model: " + error);
    }
    return mtp_model;
  }

  const std::shared_ptr<const qh::QwenDFlashGpuModel>& DFlash() {
    if (!dflash_model) {
      if (dflash_model_path.empty())
        throw std::runtime_error("DFlash modes need --dflash-model");
      std::string error;
      std::shared_ptr<const strycast::core::GgufReader> dflash_reader =
          strycast::core::GgufReader::OpenFile(dflash_model_path, &error);
      if (!dflash_reader)
        throw std::runtime_error("DFlash GGUF: " + error);
      dflash_model = qh::QwenDFlashGpuModel::Create(
          std::move(dflash_reader), executor->GetSharedModel(), &error);
      if (!dflash_model)
        throw std::runtime_error("DFlash model: " + error);
    }
    return dflash_model;
  }

  void Write(std::string_view mode, const Obj& obj) const {
    WriteText(out_dir / (std::string(mode) + ".json"), obj.Json());
  }
};

// --- prefill ----------------------------------------------------------------

struct PrefillCase {
  std::string name;
  std::optional<std::size_t> body;  // synthetic body, else a haiku prefix
  std::size_t haiku_length{0};
  std::vector<std::size_t> pieces;  // incremental prefill pieces
  bool dump_row{false};

  [[nodiscard]] Ids Prompt() const {
    return body ? Synthetic(*body) : HaikuPrefix(haiku_length);
  }
};

void RunPrefill(Context& ctx) {
  constexpr std::size_t kSteps = 8;
  std::vector<PrefillCase> cases;
  for (const std::size_t length : {1, 5, 8, 9, 20}) {
    cases.push_back({.name = "haiku" + std::to_string(length),
                     .haiku_length = length,
                     .dump_row = length == 1 || length == 20});
  }
  for (const std::size_t body :
       {7, 100, 116, 244, 245, 341, 500, 1012, 1129, 4084, 4088, 5077}) {
    cases.push_back({.name = "body" + std::to_string(body),
                     .body = body,
                     .dump_row = body == 5077});
  }
  cases.push_back(
      {.name = "haiku20-pieces-4-16", .haiku_length = 20, .pieces = {4, 16}});
  cases.push_back({.name = "body1129-pieces-512-512-117",
                   .body = 1129,
                   .pieces = {512, 512, 117}});

  auto& executor = *ctx.executor;
  executor.SetPromptHiddenCapture(false);
  std::vector<std::string> entries;
  for (const auto& c : cases) {
    const auto prompt = c.Prompt();
    executor.Reset();
    std::uint32_t next = 0;
    if (c.pieces.empty()) {
      next = executor.ForwardPromptBatch(prompt, 0);
    } else {
      std::size_t offset = 0;
      for (const auto piece : c.pieces) {
        const bool last = offset + piece == prompt.size();
        next = executor.ForwardPromptBatch(
            std::span(prompt).subspan(offset, piece),
            static_cast<std::uint32_t>(offset), last);
        offset += piece;
      }
      if (offset != prompt.size())
        throw std::logic_error("prefill pieces do not cover " + c.name);
    }
    if (c.dump_row) {
      WriteRow(ctx.out_dir / "rows" / ("prefill-" + c.name + ".f32"),
               executor.CopyLastLogits());
    }
    const auto trace = Continue(
        executor, next, static_cast<std::uint32_t>(prompt.size()), kSteps);
    Obj entry;
    entry.Str("name", c.name);
    if (c.body) {
      entry.Num("body", *c.body);
    } else {
      entry.Add("body", "null").Add("prompt", Array(prompt));
    }
    entry.Num("prompt_len", prompt.size());
    if (!c.pieces.empty())
      entry.Add("pieces", Array(c.pieces));
    if (c.dump_row)
      entry.Str("row", "rows/prefill-" + c.name + ".f32");
    trace.AddTo(entry);
    entries.push_back(entry.Json());
    std::cerr << "prefill " << c.name << " (" << prompt.size()
              << "): " << Array(trace.ids) << '\n';
  }

  // Greedy Generate (stops at a stop token), as the 27B prefill fixture.
  std::vector<std::string> generated;
  struct GenerateCase {
    std::optional<std::size_t> body;
    std::size_t max_new;
  };
  for (const auto& g :
       std::vector<GenerateCase>{{7, 24}, {1129, 24}, {{}, 64}}) {
    const auto prompt = g.body ? Synthetic(*g.body) : kHaiku;
    strycast::models::GenerationOptions options;
    options.max_new_tokens = g.max_new;
    const auto ids = executor.Generate(prompt, options);
    Obj entry;
    if (g.body)
      entry.Num("body", *g.body);
    else
      entry.Add("body", "null").Str("prompt", "haiku");
    entry.Num("max_new_tokens", g.max_new).Add("ids", Array(ids));
    generated.push_back(entry.Json());
    std::cerr << "generate: " << Array(ids) << '\n';
  }

  Obj obj;
  obj.Num("steps", kSteps)
      .Str("prompt",
           "head + body + tail; body[i] = 1000 + (7919 i + 13) % 50000, "
           "or the first prompt_len tokens of the haiku prompt")
      .Add("head", Array(kHead))
      .Add("tail", Array(kTail))
      .Add("haiku", Array(kHaiku))
      .Str("haiku_text", executor.GetTokenizer().Decode(kHaiku))
      .Num("max_prompt_batch", executor.GetMaxPromptBatch())
      .Add("prefill", Join(entries))
      .Add("generate", Join(generated));
  ctx.Write("prefill", obj);
}

// --- decode -----------------------------------------------------------------

void RunDecode(Context& ctx) {
  constexpr std::size_t kSteps = 140;
  auto& executor = *ctx.executor;
  executor.SetPromptHiddenCapture(false);
  executor.Reset();
  std::uint32_t pos = 0;
  std::uint32_t next = 0;
  for (const auto token : kHaiku) {
    const bool last = pos + 1 == kHaiku.size();
    next = executor.ForwardToken(token, pos++, last);
  }
  const auto trace = Continue(executor, next, pos, kSteps, false);
  Obj decode;
  decode.Num("steps", kSteps);
  trace.AddTo(decode);
  Obj obj;
  obj.Str("prompt_text", executor.GetTokenizer().Decode(kHaiku))
      .Add("prompt", Array(kHaiku))
      .Add("decode", decode.Json());
  ctx.Write("decode", obj);
  std::cerr << "decode: " << Array(trace.ids) << '\n';
}

// --- session (Generate, then GenerateFromPrefix) ----------------------------

void RunSession(Context& ctx) {
  auto& executor = *ctx.executor;
  executor.SetPromptHiddenCapture(false);
  struct SessionCase {
    std::optional<std::size_t> body;
    std::size_t max_new;
  };
  std::vector<std::string> cases;
  for (const auto& s : std::vector<SessionCase>{{7, 24}, {{}, 64}}) {
    const auto prompt = s.body ? Synthetic(*s.body) : kHaiku;
    strycast::models::GenerationOptions options;
    options.max_new_tokens = s.max_new;
    const auto first = executor.Generate(prompt, options);
    auto next_prompt = prompt;
    next_prompt.insert(next_prompt.end(), first.begin(), first.end());
    const auto retained = next_prompt.size();
    next_prompt.insert(next_prompt.end(), kSuffix.begin(), kSuffix.end());
    const auto second =
        executor.GenerateFromPrefix(next_prompt, retained, options);
    Obj entry;
    entry.Str("prompt", s.body ? "synthetic body" : "haiku");
    if (s.body)
      entry.Num("body", *s.body);
    else
      entry.Add("body", "null");
    entry.Num("max_new_tokens", s.max_new)
        .Add("first", Array(first))
        .Add("second", Array(second));
    cases.push_back(entry.Json());
    std::cerr << "session: " << Array(first) << " | " << Array(second) << '\n';
  }
  Obj obj;
  obj.Add("suffix", Array(kSuffix)).Add("cases", Join(cases));
  ctx.Write("session", obj);
}

// --- snapshot / restore -----------------------------------------------------

void RunSnapshot(Context& ctx) {
  constexpr std::size_t kSteps = 16;
  constexpr std::size_t kSuffixSteps = 8;
  auto& executor = *ctx.executor;
  executor.SetPromptHiddenCapture(false);
  executor.Reset();
  const auto prompt = kHaiku;
  const auto length = static_cast<std::uint32_t>(prompt.size());
  const auto frontier = executor.ForwardPromptBatch(prompt, 0);
  const auto frontier_view = executor.CopyLastLogits();
  const std::vector<float> frontier_logits(frontier_view.begin(),
                                           frontier_view.end());
  const auto snapshot = executor.SaveSnapshot(length);
  const auto fresh = Continue(executor, frontier, length, kSteps);

  executor.RestoreSnapshot(*snapshot);
  // The frontier logits are not part of the snapshot: decode from the token.
  Trace restored;
  {
    std::uint32_t next = frontier;
    std::uint32_t pos = length;
    restored.Record(next, frontier_logits);
    for (std::size_t step = 0; step < kSteps; ++step) {
      next = executor.ForwardToken(next, pos++);
      restored.Record(next, executor.CopyLastLogits());
    }
  }

  // Compact (disk cache) payload of the same snapshot.
  std::vector<std::uint8_t> payload(snapshot->CompactPayloadBytes());
  payload.resize(snapshot->SerializeCompact(payload));
  executor.Reset();
  executor.RestoreCompactSnapshot(payload, length);
  Trace compact;
  {
    std::uint32_t next = frontier;
    std::uint32_t pos = length;
    compact.Record(next, frontier_logits);
    for (std::size_t step = 0; step < kSteps; ++step) {
      next = executor.ForwardToken(next, pos++);
      compact.Record(next, executor.CopyLastLogits());
    }
  }

  // Continuation: restore the prompt, prefill a suffix at its offset.
  executor.RestoreSnapshot(*snapshot);
  const auto suffix_frontier = executor.ForwardPromptBatch(kSuffix, length);
  const auto continued = Continue(
      executor, suffix_frontier,
      length + static_cast<std::uint32_t>(kSuffix.size()), kSuffixSteps);

  // The same tokens as one cold prefill.
  auto whole = prompt;
  whole.insert(whole.end(), kSuffix.begin(), kSuffix.end());
  executor.Reset();
  const auto cold_frontier = executor.ForwardPromptBatch(whole, 0);
  const auto cold =
      Continue(executor, cold_frontier,
               static_cast<std::uint32_t>(whole.size()), kSuffixSteps);

  Obj fresh_obj, restored_obj, compact_obj, continued_obj, cold_obj;
  fresh.AddTo(fresh_obj);
  restored.AddTo(restored_obj);
  compact.AddTo(compact_obj);
  continued.AddTo(continued_obj);
  cold.AddTo(cold_obj);
  Obj obj;
  obj.Str("prompt", "haiku")
      .Add("prompt_ids", Array(prompt))
      .Add("suffix", Array(kSuffix))
      .Num("steps", kSteps)
      .Num("suffix_steps", kSuffixSteps)
      .Num("compact_payload_bytes", payload.size())
      .Add("fresh", fresh_obj.Json())
      .Add("restored", restored_obj.Json())
      .Bool("restored_matches_fresh", restored == fresh)
      .Add("compact_restored", compact_obj.Json())
      .Bool("compact_matches_fresh", compact == fresh)
      .Add("continued", continued_obj.Json())
      .Add("cold", cold_obj.Json())
      .Bool("continued_matches_cold", continued == cold);
  ctx.Write("snapshot", obj);
  std::cerr << "snapshot: restored==fresh " << (restored == fresh)
            << " compact==fresh " << (compact == fresh) << " continued==cold "
            << (continued == cold) << '\n';
}

// --- batched decode ---------------------------------------------------------

void RunBatched(Context& ctx) {
  constexpr std::size_t kSteps = 12;
  struct Session {
    std::string name;
    Ids prompt;
  };
  const std::vector<Session> sessions = {{"haiku", kHaiku},
                                         {"body7", Synthetic(7)},
                                         {"body100", Synthetic(100)},
                                         {"body341", Synthetic(341)}};
  for (std::size_t i = 0; i < sessions.size(); ++i)
    ctx.Session(i).SetPromptHiddenCapture(false);

  std::vector<Trace> serial;
  for (const auto& s : sessions) {
    auto& executor = *ctx.executor;
    executor.Reset();
    const auto next = executor.ForwardPromptBatch(s.prompt, 0);
    serial.push_back(Continue(
        executor, next, static_cast<std::uint32_t>(s.prompt.size()), kSteps));
  }

  std::vector<std::string> widths;
  for (const std::size_t width : {2, 3, 4}) {
    std::vector<Trace> traces(width);
    std::vector<std::uint32_t> next(width);
    std::vector<std::uint32_t> pos(width);
    for (std::size_t i = 0; i < width; ++i) {
      auto& executor = ctx.Session(i);
      executor.Reset();
      next[i] = executor.ForwardPromptBatch(sessions[i].prompt, 0);
      pos[i] = static_cast<std::uint32_t>(sessions[i].prompt.size());
      traces[i].Record(next[i], executor.CopyLastLogits());
    }
    for (std::size_t step = 0; step < kSteps; ++step) {
      std::vector<qh::QwenGpuBatchItem> items;
      for (std::size_t i = 0; i < width; ++i)
        items.push_back({.executor = &ctx.Session(i),
                         .token_id = next[i],
                         .position = pos[i]});
      const auto out = qh::QwenGpuExecutor::ForwardTokenBatch(items);
      for (std::size_t i = 0; i < width; ++i) {
        next[i] = out.at(i);
        ++pos[i];
        traces[i].Record(next[i], ctx.Session(i).CopyLastLogits());
      }
    }
    std::vector<std::string> rows;
    bool matches = true;
    for (std::size_t i = 0; i < width; ++i) {
      Obj row;
      row.Str("session", sessions[i].name);
      traces[i].AddTo(row);
      rows.push_back(row.Json());
      matches = matches && traces[i] == serial[i];
      std::cerr << "batched width " << width << " " << sessions[i].name << ": "
                << Array(traces[i].ids) << '\n';
    }
    Obj entry;
    entry.Num("width", width)
        .Add("sessions", Join(rows))
        .Bool("matches_serial", matches);
    widths.push_back(entry.Json());
  }

  std::vector<std::string> session_info, serial_rows;
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Obj info;
    info.Str("name", sessions[i].name)
        .Num("prompt_len", sessions[i].prompt.size());
    session_info.push_back(info.Json());
    Obj row;
    row.Str("session", sessions[i].name);
    serial[i].AddTo(row);
    serial_rows.push_back(row.Json());
  }
  Obj obj;
  obj.Str("prompt",
          "sessions: haiku, then head + body + tail as the prefill fixture")
      .Num("steps", kSteps)
      .Add("sessions", Join(session_info))
      .Add("serial", Join(serial_rows))
      .Add("batched", Join(widths));
  ctx.Write("batched", obj);
}

// --- seeded sampling --------------------------------------------------------

void RunSampled(Context& ctx) {
  constexpr std::size_t kMaxNew = 64;
  auto& executor = *ctx.executor;
  executor.SetPromptHiddenCapture(false);
  struct SampledCase {
    std::string prompt;
    Ids ids;
    strycast::sampling::SamplingConfig config;
  };
  std::vector<SampledCase> cases;
  {
    strycast::sampling::SamplingConfig a;
    a.temperature = 0.8F;
    a.top_k = 40;
    a.top_p = 0.95F;
    a.seed = 1234;
    cases.push_back({"haiku", kHaiku, a});
    cases.push_back({"body100", Synthetic(100), a});
    strycast::sampling::SamplingConfig b;
    b.temperature = 1.0F;
    b.min_p = 0.05F;
    b.seed = 7;
    cases.push_back({"haiku", kHaiku, b});
  }
  std::vector<std::string> entries;
  for (const auto& c : cases) {
    strycast::models::GenerationOptions options;
    options.max_new_tokens = kMaxNew;
    options.sampling = c.config;
    const auto ids = executor.Generate(c.ids, options);
    Obj sampling;
    sampling.Add("temperature", Float(c.config.temperature))
        .Add("top_k", std::to_string(c.config.top_k))
        .Add("top_p", Float(c.config.top_p))
        .Add("min_p", Float(c.config.min_p))
        .Add("seed", std::to_string(c.config.seed));
    Obj entry;
    entry.Str("prompt", c.prompt)
        .Add("prompt_ids", Array(c.ids))
        .Add("sampling", sampling.Json())
        .Num("max_new_tokens", kMaxNew)
        .Add("ids", Array(ids));
    entries.push_back(entry.Json());
    std::cerr << "sampled " << c.prompt << ": " << Array(ids) << '\n';
  }
  Obj obj;
  obj.Str("entry",
          "QwenGpuExecutor::Generate with SamplingConfig (SamplerState over "
          "the prompt, GPU sampling via SampleLastLogits); stops at a stop "
          "token")
      .Add("cases", Join(entries));
  ctx.Write("sampled", obj);
}

// --- MTP head ---------------------------------------------------------------

void RunMtpHead(Context& ctx) {
  constexpr std::size_t kSteps = 8;
  auto& executor = *ctx.executor;
  const auto& model = ctx.Mtp();
  std::vector<std::string> entries;
  for (const auto& [name, prompt] : std::vector<std::pair<std::string, Ids>>{
           {"haiku", kHaiku}, {"body100", Synthetic(100)}}) {
    std::string error;
    auto head = qh::QwenMtpGpuExecutor::Create(
        model, ctx.max_context, qh::kMtpDefaultPrimeChunk, &error);
    if (!head)
      throw std::runtime_error("MTP executor: " + error);
    executor.Reset();
    executor.SetPromptHiddenCapture(true, {});
    const auto first = executor.ForwardPromptBatch(prompt, 0);
    const auto hidden_view = executor.GetPromptHiddenStates();
    const std::vector<float> hidden(hidden_view.begin(), hidden_view.end());
    executor.SetPromptHiddenCapture(false);
    const std::size_t width = hidden.size() / prompt.size();
    const std::size_t n = prompt.size();
    // As QwenMtpGpuDraftBackend::PrimeTargetContext with prime_chunk != 0.
    head->PrimeTargetHiddenBatch(0, std::span(prompt).subspan(1),
                                 std::span(hidden).first((n - 1) * width));
    head->Synchronize();
    Trace trace;
    auto pos = static_cast<std::uint32_t>(n - 1);
    std::uint32_t token = first;
    for (std::size_t step = 0; step < kSteps; ++step) {
      token = step == 0 ? head->ForwardTargetHidden(
                              token,
                              std::span(hidden).subspan((n - 1) * width, width),
                              pos, true, false)
                        : head->ForwardFeedback(token, pos, true, false);
      ++pos;
      trace.Record(token, head->CopyLastLogits());
    }
    Obj entry;
    entry.Str("prompt", name)
        .Num("prompt_len", n)
        .Num("hidden_width", width)
        .Str("hidden_fnv1a64", Fnv1a64(hidden))
        .Num("target_first", first);
    trace.AddTo(entry);
    entries.push_back(entry.Json());
    std::cerr << "mtp-head " << name << ": " << first << " " << Array(trace.ids)
              << '\n';
  }
  Obj obj;
  obj.Num("steps", kSteps)
      .Bool("has_draft_vocab", model->HasDraftVocab())
      .Add("cases", Join(entries));
  ctx.Write("mtp-head", obj);
}

// --- speculative ------------------------------------------------------------

struct SpecResult {
  std::uint32_t first{0};
  Ids ids;
  std::vector<std::string> steps;
  spec::SpeculativeStats stats;
};

std::string StepRow(const spec::SpeculativeVerifier::StepResult& step,
                    std::optional<std::size_t> width = {}) {
  std::vector<std::uint64_t> row;
  if (width)
    row.push_back(*width);
  row.push_back(step.draft_count);
  row.push_back(step.accepted_count);
  for (const auto token : step.emitted_tokens)
    row.push_back(token);
  return Array(row);
}

/// cpp_decode_dump's dflash/mtp loop: Prime, then VerifyStep with EOS ignored
/// until `tokens` tokens after the first.
SpecResult RunVerifier(spec::SpeculativeVerifier& verifier, const Ids& prompt,
                       std::size_t tokens) {
  SpecResult result;
  std::vector<std::uint32_t> sequence = prompt;
  auto current = verifier.Prime(sequence);
  result.first = current;
  auto pos = static_cast<std::uint32_t>(sequence.size());
  sequence.push_back(current);
  while (result.ids.size() < tokens) {
    const auto step = verifier.VerifyStep(
        sequence, pos, current, kIgnoredEos,
        static_cast<std::uint32_t>(tokens - result.ids.size()));
    result.steps.push_back(StepRow(step));
    for (const auto token : step.emitted_tokens) {
      if (result.ids.size() < tokens) {
        result.ids.push_back(token);
        sequence.push_back(token);
        ++pos;
      }
    }
    current = step.next_token;
  }
  result.stats = verifier.GetStats();
  return result;
}

std::string SpecJson(const std::string& name, const Ids& prompt,
                     const SpecResult& r) {
  Obj stats;
  stats.Num("drafted", r.stats.total_draft_tokens)
      .Num("accepted", r.stats.total_accepted_tokens)
      .Num("verification_steps", r.stats.total_verification_steps)
      .Num("emitted", r.stats.total_emitted_tokens);
  Obj entry;
  entry.Str("prompt", name).Num("prompt_len", prompt.size());
  if (name.starts_with("body"))
    entry.Num("body", prompt.size() - kHead.size() - kTail.size());
  else
    entry.Add("body", "null").Add("prompt_ids", Array(prompt));
  entry.Num("first", r.first)
      .Add("steps", Join(r.steps))
      .Add("ids", Array(r.ids))
      .Add("stats", stats.Json());
  return entry.Json();
}

// A chat turn asking to repeat a passage: the DFlash backend's prompt lookup
// copies it (wide verification of up to 64 rows in the serve configuration).
constexpr std::string_view kCopyRequest =
    "Repeat the following paragraph exactly, word for word, with nothing "
    "else:\n\nThe lighthouse keeper climbed the spiral stairs every evening "
    "at dusk. He trimmed the wick, polished the great lens, and wrote the "
    "weather in a worn leather logbook. Ships passed far out on the dark "
    "water, their lights small and steady, and none of the sailors ever knew "
    "his name. In the morning he walked along the shore, collecting driftwood "
    "and glass floats, and counted the gulls that rested on the rocks.";

Ids CopyPrompt(const qh::QwenGpuExecutor& executor) {
  Ids ids = kHead;
  const auto text = executor.GetTokenizer().Encode(kCopyRequest);
  ids.insert(ids.end(), text.begin(), text.end());
  ids.insert(ids.end(), kTail.begin(), kTail.end());
  return ids;
}

std::vector<std::pair<std::string, Ids>> SpecPrompts(
    const qh::QwenGpuExecutor& executor) {
  return {{"haiku", kHaiku},
          {"body100", Synthetic(100)},
          {"body1129", Synthetic(1129)},
          {"copy", CopyPrompt(executor)}};
}

constexpr std::size_t kSpecTokens = 64;

spec::SpeculativeOptions CliOptions(bool dflash) {
  // drivers.cpp CreateQwenVerifier with the host defaults (7, 1, no policy).
  spec::SpeculativeOptions options;
  options.max_draft_tokens = 7;
  options.min_draft_tokens = 1;
  options.initial_draft_tokens = 7;
  options.enable_adaptive_draft_length = !dflash;
  options.use_batched_verification = true;
  return options;
}

spec::SpeculativeOptions ServeOptions(bool dflash, std::uint32_t copy_rows) {
  // runner.cpp CreateQwenTextRunner and BindComponentState.
  spec::SpeculativeOptions options;
  options.max_draft_tokens = 7;
  options.min_draft_tokens = 1;
  options.initial_draft_tokens = 7;
  options.use_batched_verification = true;
  options.retain_frontier_logits = true;
  options.enable_adaptive_draft_length = !dflash;
  if (dflash && copy_rows > options.max_draft_tokens + 1) {
    options.max_draft_tokens = copy_rows - 1;
    options.initial_draft_tokens = copy_rows - 1;
  }
  return options;
}

constexpr std::uint32_t kServeCopyRows =
    static_cast<std::uint32_t>(qh::QwenGpuExecutor::kMaxWideVerificationRows);

std::unique_ptr<qh::QwenMtpGpuDraftBackend> MtpBackend(
    Context& ctx, qh::QwenGpuExecutor& target) {
  std::string error;
  auto backend = qh::QwenMtpGpuDraftBackend::Create(
      ctx.Mtp(),
      {.max_context = ctx.max_context,
       .max_draft_tokens = 7,
       .prime_chunk = qh::kMtpDefaultPrimeChunk,
       .vision_input = &target.VisionInput()},
      &error);
  if (!backend)
    throw std::runtime_error("MTP backend: " + error);
  return backend;
}

std::unique_ptr<qh::QwenDFlashGpuDraftBackend> DFlashBackend(Context& ctx,
                                                             bool serve) {
  std::string error;
  auto backend = qh::QwenDFlashGpuDraftBackend::Create(
      ctx.DFlash(),
      {.max_context = ctx.max_context,
       .max_draft_tokens = 7,
       .policy = spec::ParseDFlashDraftPolicy(""),
       .max_copy_rows = serve ? kServeCopyRows : 0U},
      &error);
  if (!backend)
    throw std::runtime_error("DFlash backend: " + error);
  return backend;
}

void RunMtp(Context& ctx) {
  auto& executor = *ctx.executor;
  std::vector<std::string> cases;
  for (const auto& [name, prompt] : SpecPrompts(executor)) {
    auto verifier = qh::CreateQwenGpuSpeculativeVerifier(
        executor, MtpBackend(ctx, executor), CliOptions(false));
    const auto r = RunVerifier(*verifier, prompt, kSpecTokens);
    cases.push_back(SpecJson(name, prompt, r));
    std::cerr << "mtp " << name << ": " << Join(r.steps) << '\n';
  }
  executor.SetPromptHiddenCapture(false);
  Obj obj;
  obj.Str("config",
          "drivers.cpp CreateQwenVerifier (gufo prompt --speculative mtp "
          "--mtp-model <target GGUF>): 7 draft tokens, min 1, adaptive "
          "length, batched verification, prime_chunk 1024; EOS ignored")
      .Num("tokens", kSpecTokens)
      .Str("steps_format", "drafted accepted emitted...")
      .Str("copy_user_text", kCopyRequest)
      .Add("cases", Join(cases));
  ctx.Write("mtp", obj);
}

void RunDFlash(Context& ctx) {
  auto& executor = *ctx.executor;
  std::vector<std::string> cli, serve;
  for (const auto& [name, prompt] : SpecPrompts(executor)) {
    {
      auto verifier = qh::CreateQwenGpuSpeculativeVerifier(
          executor, DFlashBackend(ctx, false), CliOptions(true));
      const auto r = RunVerifier(*verifier, prompt, kSpecTokens);
      cli.push_back(SpecJson(name, prompt, r));
      std::cerr << "dflash " << name << ": " << Join(r.steps) << '\n';
    }
    {
      auto verifier = qh::CreateQwenGpuSpeculativeVerifier(
          executor, DFlashBackend(ctx, true),
          ServeOptions(true, kServeCopyRows));
      const auto r = RunVerifier(*verifier, prompt, kSpecTokens);
      serve.push_back(SpecJson(name, prompt, r));
      std::cerr << "dflash serve " << name << ": " << Join(r.steps) << '\n';
    }
  }
  executor.SetPromptHiddenCapture(false);
  Obj obj;
  obj.Str("config",
          "cases: drivers.cpp CreateQwenVerifier (gufo prompt --speculative "
          "dflash2): 7 draft tokens, min 1, adaptive policy, fixed length, "
          "batched verification, no lookup copy. serve_cases: runner.cpp "
          "(serve): draft 7 with max_copy_rows 64, verifier max/initial "
          "draft 63, retain_frontier_logits. EOS ignored")
      .Num("tokens", kSpecTokens)
      .Str("steps_format", "drafted accepted emitted...")
      .Str("copy_user_text", kCopyRequest)
      .Add("cases", Join(cli))
      .Add("serve_cases", Join(serve));
  ctx.Write("dflash", obj);
}

/// runner.cpp ConcurrentDraftCap.
std::uint32_t ConcurrentDraftCap(std::size_t active, bool mtp) {
  if (active <= 1)
    return std::numeric_limits<std::uint32_t>::max();
  if (active == 2)
    return mtp ? 1 : 2;
  return mtp ? 0 : 1;
}

void RunSpecBatch(Context& ctx, bool mtp) {
  constexpr std::size_t kTokens = 48;
  const std::vector<std::pair<std::string, Ids>> prompts = {
      {"haiku", kHaiku}, {"body100", Synthetic(100)}};
  struct Lane {
    std::unique_ptr<spec::SpeculativeVerifier> verifier;
    qh::QwenMtpGpuDraftBackend* mtp{nullptr};
    qh::QwenDFlashGpuDraftBackend* dflash{nullptr};
    Ids sequence;
    std::uint32_t pos{0};
    std::uint32_t current{0};
    std::uint32_t first{0};
    Ids ids;
    std::vector<std::string> steps;
    strycast::sampling::SamplerState sampler;

    void SetCap(std::uint32_t cap) {
      if (mtp)
        mtp->SetStepDraftCap(cap);
      if (dflash)
        dflash->SetStepDraftCap(cap);
    }
    void Take(const spec::SpeculativeVerifier::StepResult& step,
              std::size_t width, std::size_t tokens) {
      steps.push_back(StepRow(step, width));
      for (const auto token : step.emitted_tokens) {
        if (ids.size() < tokens) {
          ids.push_back(token);
          sequence.push_back(token);
          ++pos;
        }
      }
      current = step.next_token;
    }
  };
  std::vector<Lane> lanes(prompts.size());
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    auto& executor = ctx.Session(i);
    auto& lane = lanes[i];
    if (mtp) {
      auto backend = MtpBackend(ctx, executor);
      lane.mtp = backend.get();
      lane.verifier = qh::CreateQwenGpuSpeculativeVerifier(
          executor, std::move(backend), ServeOptions(false, 0));
    } else {
      auto backend = DFlashBackend(ctx, true);
      lane.dflash = backend.get();
      lane.verifier = qh::CreateQwenGpuSpeculativeVerifier(
          executor, std::move(backend), ServeOptions(true, kServeCopyRows));
    }
    lane.sequence = prompts[i].second;
    lane.sampler = strycast::sampling::SamplerState({}, prompts[i].second);
    lane.current = lane.verifier->Prime(lane.sequence);
    lane.first = lane.current;
    lane.pos = static_cast<std::uint32_t>(lane.sequence.size());
    lane.sequence.push_back(lane.current);
  }
  for (;;) {
    std::vector<Lane*> active;
    for (auto& lane : lanes)
      if (lane.ids.size() < kTokens)
        active.push_back(&lane);
    if (active.empty())
      break;
    if (active.size() == 1) {
      // runner.cpp DecodeStep: one session drafts without the cap.
      auto& lane = *active.front();
      lane.SetCap(ConcurrentDraftCap(1, mtp));
      const auto step = lane.verifier->VerifyStep(
          lane.sequence, lane.pos, lane.current, kIgnoredEos,
          static_cast<std::uint32_t>(kTokens - lane.ids.size()), lane.sampler);
      lane.Take(step, 1, kTokens);
      continue;
    }
    std::vector<spec::SpeculativeVerifier::StepRequest> requests;
    for (auto* lane : active) {
      lane->SetCap(ConcurrentDraftCap(active.size(), mtp));
      requests.push_back(
          {*lane->verifier, lane->sequence, lane->pos, lane->current,
           kIgnoredEos, static_cast<std::uint32_t>(kTokens - lane->ids.size()),
           lane->sampler});
    }
    const auto results = spec::SpeculativeVerifier::VerifyBatch(requests);
    for (std::size_t i = 0; i < active.size(); ++i)
      active[i]->Take(results.at(i), active.size(), kTokens);
  }
  std::vector<std::string> entries;
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    Obj entry;
    entry.Str("prompt", prompts[i].first)
        .Num("prompt_len", prompts[i].second.size())
        .Num("first", lanes[i].first)
        .Add("steps", Join(lanes[i].steps))
        .Add("ids", Array(lanes[i].ids));
    entries.push_back(entry.Json());
    std::cerr << (mtp ? "mtp" : "dflash") << "-batch " << prompts[i].first
              << ": " << Join(lanes[i].steps) << '\n';
  }
  for (std::size_t i = 0; i < lanes.size(); ++i)
    ctx.Session(i).SetPromptHiddenCapture(false);
  lanes.clear();
  Obj obj;
  obj.Str("config",
          std::string(
              "runner.cpp DecodeBatch: two sessions (own executors and "
              "draft backends, serve configuration), greedy SamplerState "
              "over the prompt, SpeculativeVerifier::VerifyBatch while both "
              "decode with SetStepDraftCap(ConcurrentDraftCap(2)) = ") +
              (mtp ? "1" : "2") +
              ", then VerifyStep alone without the cap; EOS ignored")
      .Num("tokens", kTokens)
      .Str("steps_format", "width drafted accepted emitted...")
      .Add("sessions", Join(entries));
  ctx.Write(mtp ? "mtp-batch" : "dflash-batch", obj);
}

// --- vision -----------------------------------------------------------------

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
  if (bytes.empty())
    throw std::runtime_error("cannot read " + path);
  return bytes;
}

std::string Sha256(std::span<const std::uint8_t> bytes) {
  return strycast::crypto::Sha256Hex(bytes);
}

std::string Floats(std::span<const float> values) {
  std::vector<std::string> out;
  for (const auto v : values)
    out.push_back(Float(v));
  return Join(out);
}

std::string ImageName(const std::string& path) {
  return std::filesystem::path(path).stem().string();
}

/// Encoder::Encode with an observer: digests of every stage the encoder
/// reports ("patch", "layer0".."layer26", "embedding") and the embedding.
struct Encoded {
  std::vector<std::pair<std::string, std::string>> stages;
  std::vector<float> embedding;
  std::uint32_t rows{0};
  std::uint32_t width{0};
};

Encoded EncodeObserved(qv::Encoder& encoder,
                       const strycast::core::Image& image) {
  Encoded out;
  const auto embedding = encoder.Encode(
      image, [&](std::string_view stage, std::span<const float> values) {
        out.stages.emplace_back(std::string(stage), Fnv1a64(values));
        if (stage == "embedding")
          out.embedding.assign(values.begin(), values.end());
      });
  out.rows = embedding->rows();
  out.width = embedding->width();
  return out;
}

std::vector<float> EncodePlain(qv::Encoder& encoder,
                               const strycast::core::Image& image) {
  const auto embedding = encoder.Encode(image);
  std::vector<float> host(std::size_t{embedding->rows()} * embedding->width());
  if (hipMemcpy(host.data(), embedding->data(), host.size() * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess)
    throw std::runtime_error("cannot copy the vision embedding");
  return host;
}

void RunVision(Context& ctx) {
  constexpr std::size_t kSteps = 32;
  constexpr std::size_t kSpecTokens = 32;
  if (ctx.mmproj_path.empty() || ctx.image_paths.empty())
    throw std::runtime_error("vision needs --mmproj and --images");
  auto& executor = *ctx.executor;
  const auto& tokenizer = executor.GetTokenizer();
  // drivers.cpp LoadQwenVision: the trunk width (2048 with the gate patch).
  const std::uint32_t width = executor.GetConfig().hidden_size;
  auto encoder = qv::Encoder::Open(ctx.model_path, ctx.mmproj_path, width);
  if (!encoder)
    throw std::runtime_error("no vision encoder");
  // drivers.cpp PromptChatOptions / PrepareVision for --think off.
  strycast::ReasoningOptions reasoning;
  reasoning.enabled = false;
  const auto chat =
      tokenizer.UsesOrnithChatTemplate()
          ? strycast::tokenization::ResolveOrnithChatOptions(reasoning, false)
          : strycast::tokenization::ResolveQwenChatOptions(reasoning, false);
  constexpr std::uint32_t kDefaultContext = 4096;  // drivers.cpp PrepareVision

  struct Case {
    std::string name;
    std::vector<std::string> images;
    std::string text;
  };
  std::vector<Case> cases;
  for (const auto& path : ctx.image_paths)
    cases.push_back(
        {ImageName(path), {path}, "Describe the image in one sentence."});
  if (ctx.image_paths.size() >= 2)
    cases.push_back({"pair-" + ImageName(ctx.image_paths[0]) + "-" +
                         ImageName(ctx.image_paths[1]),
                     {ctx.image_paths[0], ctx.image_paths[1]},
                     "Describe each image in one sentence."});

  std::vector<std::string> image_entries, case_entries;
  std::vector<std::string> encoded_names;
  for (const auto& c : cases) {
    std::vector<strycast::tokenization::ChatMessage> messages;
    messages.push_back(
        {strycast::tokenization::ChatRole::kUser, c.text, "", ""});
    std::vector<std::shared_ptr<const std::vector<std::uint8_t>>> files;
    for (const auto& path : c.images) {
      auto bytes = std::make_shared<std::vector<std::uint8_t>>(ReadBytes(path));
      files.push_back(bytes);
      messages.back().images.push_back(
          {0, bytes, strycast::tokenization::ChatMessage::ImageDetail::kAuto});
    }
    std::string error;
    const auto rendered = strycast::tokenization::QwenChatTemplate::Render(
        messages, {}, chat, &error);
    if (!rendered)
      throw std::runtime_error("template: " + error);
    auto prompt = std::make_shared<const qv::Prompt>(qv::Prepare(
        tokenizer, messages, {}, chat, encoder->identity(), kDefaultContext));

    // Per image: decode, resize, grid, encoder stages (once per image file).
    std::vector<std::string> grids;
    for (std::size_t i = 0; i < prompt->images.size(); ++i) {
      const auto& image = prompt->images[i];
      Obj grid;
      grid.Str("image", ImageName(c.images[i]))
          .Num("offset", image.grid.offset)
          .Num("height", image.grid.height)
          .Num("width", image.grid.width)
          .Num("tokens", std::size_t{image.grid.height} * image.grid.width);
      grids.push_back(grid.Json());
      const auto name = ImageName(c.images[i]);
      if (std::ranges::find(encoded_names, name) != encoded_names.end())
        continue;
      encoded_names.push_back(name);
      const auto decoded = strycast::core::DecodeImage(*files[i]);
      const auto& resized = *image.pixels;
      const auto observed = EncodeObserved(*encoder, resized);
      const auto plain = EncodePlain(*encoder, resized);
      const std::size_t w = observed.width;
      std::vector<std::string> row_fnv;
      for (std::uint32_t r = 0; r < observed.rows; ++r)
        row_fnv.push_back(
            Fnv1a64(std::span(observed.embedding).subspan(r * w, w)));
      std::vector<std::string> stages;
      for (const auto& [stage, digest] : observed.stages)
        stages.push_back("[" + Quote(stage) + ", " + Quote(digest) + "]");
      Obj entry;
      entry.Str("image", name)
          .Str("file_sha256", Sha256(*files[i]))
          .Add("decoded", Obj()
                              .Num("width", decoded.width)
                              .Num("height", decoded.height)
                              .Str("rgb8_sha256", Sha256(decoded.pixels))
                              .Json())
          .Add("resized",
               Obj()
                   .Num("width", resized.width)
                   .Num("height", resized.height)
                   .Bool("resampled", resized.width != decoded.width ||
                                          resized.height != decoded.height)
                   .Str("rgb8_sha256", Sha256(resized.pixels))
                   .Json())
          .Num("patches",
               std::size_t{resized.width / 16} * (resized.height / 16))
          .Num("grid_height", resized.height / 32)
          .Num("grid_width", resized.width / 32)
          .Num("embedding_rows", observed.rows)
          .Num("embedding_width", observed.width)
          .Str("embedding_fnv1a64", Fnv1a64(observed.embedding))
          .Bool("encode_repeat_identical", plain == observed.embedding)
          .Add("embedding_row0_head",
               Floats(std::span(observed.embedding).first(8)))
          .Add("embedding_last_row_head",
               Floats(std::span(observed.embedding)
                          .subspan((observed.rows - 1) * w, 8)))
          .Add("embedding_rows_fnv1a64", Array(Quoted(row_fnv)))
          .Add("stages_fnv1a64", Join(stages));
      if (observed.embedding.size() * sizeof(float) <= (1U << 20)) {
        const auto file = "rows/vision-embedding-" + name + ".f32";
        WriteRow(ctx.out_dir / file, observed.embedding);
        entry.Str("embedding_file", file);
      }
      image_entries.push_back(entry.Json());
      std::cerr << "vision encode " << name << ": " << observed.rows << " x "
                << observed.width << " " << Fnv1a64(observed.embedding)
                << " repeat " << (plain == observed.embedding) << '\n';
    }

    // Prefill and greedy steps on the image prompt.
    executor.SetPromptHiddenCapture(false);
    executor.Reset();
    executor.ConfigureVision(prompt, encoder);
    const auto first = executor.ForwardPromptBatch(prompt->tokens, 0);
    if (c.name == cases.front().name)
      WriteRow(ctx.out_dir / "rows" / ("vision-prefill-" + c.name + ".f32"),
               executor.CopyLastLogits());
    const auto trace =
        Continue(executor, first,
                 static_cast<std::uint32_t>(prompt->tokens.size()), kSteps);
    strycast::models::GenerationOptions options;
    options.max_new_tokens = kSteps;
    const auto generated = executor.Generate(prompt->tokens, options);

    // MTP (vision-aware draft) and DFlash2 (no vision input) as gufo prompt.
    const auto speculative = [&](bool mtp) -> std::string {
      try {
        auto verifier =
            mtp ? qh::CreateQwenGpuSpeculativeVerifier(
                      executor, MtpBackend(ctx, executor), CliOptions(false))
                : qh::CreateQwenGpuSpeculativeVerifier(
                      executor, DFlashBackend(ctx, false), CliOptions(true));
        const auto r = RunVerifier(*verifier, prompt->tokens, kSpecTokens);
        Obj stats;
        stats.Num("drafted", r.stats.total_draft_tokens)
            .Num("accepted", r.stats.total_accepted_tokens)
            .Num("verification_steps", r.stats.total_verification_steps);
        auto seq = r.ids;
        seq.insert(seq.begin(), r.first);
        const bool matches =
            std::equal(generated.begin(), generated.end(), seq.begin(),
                       seq.begin() + static_cast<std::ptrdiff_t>(std::min(
                                         seq.size(), generated.size())));
        std::cerr << (mtp ? "vision mtp " : "vision dflash ") << c.name << ": "
                  << Join(r.steps) << '\n';
        return Obj()
            .Num("first", r.first)
            .Add("steps", Join(r.steps))
            .Add("ids", Array(r.ids))
            .Add("stats", stats.Json())
            .Bool("matches_generate", matches)
            .Json();
      } catch (const std::exception& e) {
        std::cerr << "vision speculative " << c.name << ": " << e.what()
                  << '\n';
        return Obj().Str("error", e.what()).Json();
      }
    };
    const auto mtp_json = speculative(true);
    const auto dflash_json = speculative(false);
    executor.SetPromptHiddenCapture(false);

    std::vector<std::string> image_names;
    for (const auto& path : c.images)
      image_names.push_back(Quote(ImageName(path)));
    Obj entry;
    entry.Str("name", c.name)
        .Add("images", Join(image_names))
        .Str("user_text", c.text)
        .Str("rendered", *rendered)
        .Num("prompt_len", prompt->tokens.size())
        .Add("prompt", Array(prompt->tokens))
        .Add("image_grids", Join(grids))
        .Add("rope_delta", std::to_string(prompt->rope.Delta()))
        .Num("stable_prefix_tokens", prompt->stable_prefix_tokens);
    if (c.name == cases.front().name)
      entry.Str("row", "rows/vision-prefill-" + c.name + ".f32");
    trace.AddTo(entry);
    entry
        .Add("generate", Obj()
                             .Num("max_new_tokens", kSteps)
                             .Add("ids", Array(generated))
                             .Str("text", tokenizer.Decode(generated))
                             .Json())
        .Add("mtp", mtp_json)
        .Add("dflash", dflash_json);
    case_entries.push_back(entry.Json());
    std::cerr << "vision " << c.name << " (" << prompt->tokens.size()
              << "): " << tokenizer.Decode(generated) << '\n';
  }
  executor.ConfigureVision(nullptr, nullptr);
  executor.Reset();

  Obj obj;
  obj.Str("mmproj", std::filesystem::path(ctx.mmproj_path).filename().string())
      .Str("encoder_identity", encoder->identity())
      .Num("image_token", qv::kImageToken)
      .Num("steps", kSteps)
      .Num("speculative_tokens", kSpecTokens)
      .Str("speculative_steps_format", "drafted accepted emitted...")
      .Add("images", Join(image_entries))
      .Add("cases", Join(case_entries));
  ctx.Write("vision", obj);
}

/// Shared objects of the HIP stack this process has mapped, from
/// /proc/self/maps (after all modes, so libraries opened lazily are in).
std::vector<std::string> MappedRuntimeLibraries() {
  std::ifstream maps("/proc/self/maps");
  std::vector<std::string> paths;
  std::string line;
  while (std::getline(maps, line)) {
    const auto slash = line.find('/');
    if (slash == std::string::npos)
      continue;
    const auto path = line.substr(slash);
    const auto name = std::filesystem::path(path).filename().string();
    if (name.find(".so") == std::string::npos)
      continue;
    if (path.find("/rocm") == std::string::npos &&
        name.find("hip") == std::string::npos &&
        name.find("blas") == std::string::npos &&
        name.find("hsa") == std::string::npos &&
        name.find("comgr") == std::string::npos &&
        name.find("roc") == std::string::npos &&
        name.find("drm") == std::string::npos)
      continue;
    if (std::ranges::find(paths, path) == paths.end())
      paths.push_back(path);
  }
  std::ranges::sort(paths);
  return paths;
}

std::vector<std::string> Split(const std::string& text) {
  std::vector<std::string> out;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ','))
    if (!item.empty())
      out.push_back(item);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " MODEL OUTDIR [--max-context N] [--kv-cache f16|q8_0] "
                 "[--mtp-model PATH] [--dflash-model PATH] [--modes a,b,...] "
                 "[--mmproj PATH] [--images a.png,b.png]\n";
    return 2;
  }
  try {
    Context ctx;
    ctx.out_dir = argv[2];
    std::string modes =
        "prefill,decode,session,snapshot,batched,sampled,mtp-head,mtp,dflash,"
        "mtp-batch,dflash-batch";
    std::string kv_cache = "f16";
    for (int i = 3; i < argc; ++i) {
      const std::string arg = argv[i];
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value for " + arg);
      const std::string value = argv[++i];
      if (arg == "--max-context")
        ctx.max_context = static_cast<std::uint32_t>(std::stoul(value));
      else if (arg == "--kv-cache")
        kv_cache = value;
      else if (arg == "--mtp-model")
        ctx.mtp_model_path = value;
      else if (arg == "--dflash-model")
        ctx.dflash_model_path = value;
      else if (arg == "--modes")
        modes = value;
      else if (arg == "--mmproj")
        ctx.mmproj_path = value;
      else if (arg == "--images")
        ctx.image_paths = Split(value);
      else
        throw std::invalid_argument("unknown option " + arg);
    }
    if (kv_cache == "q8_0")
      ctx.policy.kv_cache_storage = qh::QwenKvCacheStorage::kQ8;
    else if (kv_cache != "f16")
      throw std::invalid_argument("--kv-cache must be f16 or q8_0");
    std::filesystem::create_directories(ctx.out_dir);
    ctx.model_path = argv[1];

    const auto load_start = std::chrono::steady_clock::now();
    std::string error;
    ctx.reader = strycast::core::GgufReader::OpenFile(argv[1], &error);
    if (!ctx.reader)
      throw std::runtime_error(error);
    ctx.executor = qh::QwenGpuExecutor::CreateFromGguf(
        ctx.reader, &error, ctx.max_context, ctx.policy);
    if (!ctx.executor)
      throw std::runtime_error(error);
    const double load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      load_start)
            .count();
    const auto& config = ctx.executor->GetConfig();
    Obj meta;
    meta.Str("model", std::filesystem::path(argv[1]).filename().string())
        .Str("architecture", config.architecture)
        .Num("vocab_size", config.vocab_size)
        .Num("hidden_size", config.hidden_size)
        .Num("num_layers", config.num_layers)
        .Num("expert_count", config.expert_count)
        .Num("expert_used_count", config.expert_used_count)
        .Num("eos", ctx.executor->GetTokenizer().GetEosTokenId())
        .Num("max_context", ctx.max_context)
        .Num("max_prompt_batch", ctx.executor->GetMaxPromptBatch())
        .Str("kv_cache", kv_cache)
        .Str("mtp_model",
             std::filesystem::path(ctx.mtp_model_path).filename().string())
        .Str("dflash_model",
             std::filesystem::path(ctx.dflash_model_path).filename().string());
    ctx.Write("meta", meta);

    Obj timings;
    timings.Add("load_seconds", std::to_string(load_seconds));
    for (const auto& mode : Split(modes)) {
      const auto start = std::chrono::steady_clock::now();
      std::cerr << "== " << mode << '\n';
      if (mode == "prefill")
        RunPrefill(ctx);
      else if (mode == "decode")
        RunDecode(ctx);
      else if (mode == "session")
        RunSession(ctx);
      else if (mode == "snapshot")
        RunSnapshot(ctx);
      else if (mode == "batched")
        RunBatched(ctx);
      else if (mode == "sampled")
        RunSampled(ctx);
      else if (mode == "mtp-head")
        RunMtpHead(ctx);
      else if (mode == "mtp")
        RunMtp(ctx);
      else if (mode == "dflash")
        RunDFlash(ctx);
      else if (mode == "mtp-batch")
        RunSpecBatch(ctx, true);
      else if (mode == "dflash-batch")
        RunSpecBatch(ctx, false);
      else if (mode == "vision")
        RunVision(ctx);
      else
        throw std::invalid_argument("unknown mode " + mode);
      timings.Add(mode,
                  std::to_string(std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count()));
    }
    WriteText(ctx.out_dir / "timings.json", timings.Json());
    WriteText(ctx.out_dir / "libraries.json",
              Obj()
                  .Add("mapped", Array(Quoted(MappedRuntimeLibraries())))
                  .Str("LD_LIBRARY_PATH", std::getenv("LD_LIBRARY_PATH")
                                              ? std::getenv("LD_LIBRARY_PATH")
                                              : "")
                  .Json());
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
