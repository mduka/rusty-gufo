// Reference token IDs and logits of the C++ DeepSeek V4 Flash engine for the
// Rust port (docs/RUST.md). Raw little-endian
// files the Rust tests and recorders read:
//   tokens:  CORPUS is a sequence of (u32 length, bytes) texts. Writes
//            text.bin and rendered.bin, each a sequence of (u32 count, u32
//            ids) per text (Model::Tokenize without and with special-token
//            parsing), chat.bin with EncodeChat("", text) per text, and
//            token_text.bin with (u32 length, bytes) of DecodeToken for every
//            vocabulary entry.
//   prefill: Sync(PROMPT), then N greedy steps (SelectNext(0), Evaluate);
//            ids.bin (u32: the token after the prompt, then one per step) and
//            logits_<step>.bin (f32, vocabulary) for step 0..=N.
//   chat:    as prefill, with PROMPT a text file encoded as EncodeChat("",
//            text); the tokens go to prompt.bin.
//   extend:  Sync(PROMPT[..K]), then Sync(PROMPT) (extending the cached
//            prefix), then N greedy steps; the files of prefill.
//   snapshot: Sync(EncodeChat("", TEXT)) in a session of CONTEXT, two
//            greedy steps, SaveSnapshot to payload.bin; then RestoreSnapshot
//            into a session of RESTORE_CONTEXT and N greedy steps, the files
//            of prefill.
//   dspark:  the model loaded with the support model DSPARK; one
//            speculative session per text of CORPUS (EncodeChat("", text)),
//            then N greedy cycles (DsparkStep alone, DsparkStepBatch for
//            several, 32 tokens and MAX_DRAFT drafts at most); per session i
//            and cycle c emitted_<i>.bin ((u32 count, ids) per cycle),
//            logits_<i>_<c>.bin, stats_<i>.bin (nine u64 and the u32
//            support context) and payload_<i>.bin (SaveSnapshot). A single
//            session is then restored into a fresh one and cycled R more
//            times: restored_emitted.bin, restored_logits_<c>.bin. SAMPLING
//            gives each session "greedy" or "T,P,K,SEED": a request sampler
//            lent to every cycle through DsparkSamplerBridge, its RNG copied
//            back and its emitted tokens accepted afterwards, as the serving
//            runner does.
//   render:  per case of the JSON array CASES (messages, tools, options)
//            rendered_<i>.txt (RenderChat) and tokens_<i>.bin (EncodeChat);
//            see tests/reference/record_chat_template.py for the case format.
//   batch:   one session per text of CORPUS (EncodeChat("", text), synced
//            in order), then N greedy steps of all sessions together
//            (Model::EvaluateBatch); per session i prompt_<i>.bin, ids_<i>.bin
//            and logits_<i>_<step>.bin.
// PROMPT is a file of u32 token IDs.
//
// Build against the C++ baseline (upstream clone, built as documented):
//   R=/opt/rocm/core-10.1 O=$UPSTREAM B=$O/build/release
//   D=$B/src/models/deepseek_v4_flash
//   $R/llvm/bin/clang++ -std=c++20 -O2 -DENGINE_ENABLE_HIP=1 \
//     -D__HIP_PLATFORM_AMD__ -I$O -isystem $R/include \
//     --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15 ds4_ref_dump.cpp \
//     -o ds4_ref_dump -Wl,--start-group $D/libgufo_deepseek_v4_flash.a \
//     $D/libgufo_deepseek_v4_flash_mmq.a $B/libgufo_core.a \
//     $B/libgufo_hip_host.a $B/libgufo_logging.a \
//     $B/src/core/platform/libgufo_platform.a -Wl,--end-group -L$R/lib \
//     -Wl,-rpath,$R/lib -lhipblaslt -lhipblas -lrocblas -lamdhip64 -licuuc \
//     -licui18n -lcrypto -lpng -lz -ljpeg -lwebp -lcurl -lgomp -lpthread -ldl
//   ./ds4_ref_dump MODEL OUTDIR tokens CORPUS
//   ./ds4_ref_dump MODEL OUTDIR prefill PROMPT N [MAX_CONTEXT]
//   ./ds4_ref_dump MODEL OUTDIR chat TEXT N [MAX_CONTEXT]
//   ./ds4_ref_dump MODEL OUTDIR extend PROMPT N K [MAX_CONTEXT]
//   ./ds4_ref_dump MODEL OUTDIR batch CORPUS N [MAX_CONTEXT]
//   ./ds4_ref_dump MODEL OUTDIR render CASES
//   ./ds4_ref_dump MODEL OUTDIR snapshot TEXT N CONTEXT RESTORE_CONTEXT
//   ./ds4_ref_dump MODEL OUTDIR dspark DSPARK CORPUS N MAX_DRAFT R [CONTEXT
//     [SAMPLING;SAMPLING...]]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/models/deepseek_v4_flash/chat_template.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"

namespace ds4 = gufo::models::deepseek_v4_flash;

namespace {

std::vector<char> ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(2);
  }
  return {std::istreambuf_iterator<char>(input), {}};
}

void WriteFile(const std::filesystem::path& path, const void* data,
               std::size_t bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(bytes));
  if (!output) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(2);
  }
}

void AppendU32(std::vector<std::uint8_t>* out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i)
    out->push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

void AppendTokens(std::vector<std::uint8_t>* out,
                  const std::vector<int>& tokens) {
  AppendU32(out, static_cast<std::uint32_t>(tokens.size()));
  for (int token : tokens)
    AppendU32(out, static_cast<std::uint32_t>(token));
}

int Tokens(const ds4::Model& model, const std::filesystem::path& out,
           const std::string& corpus_path) {
  const std::vector<char> corpus = ReadFile(corpus_path);
  std::vector<std::uint8_t> text, rendered, chat, token_text;
  std::size_t pos = 0;
  while (pos + 4 <= corpus.size()) {
    std::uint32_t length = 0;
    std::memcpy(&length, corpus.data() + pos, 4);
    pos += 4;
    if (pos + length > corpus.size()) {
      std::fprintf(stderr, "truncated corpus\n");
      return 2;
    }
    const std::string_view value(corpus.data() + pos, length);
    pos += length;
    AppendTokens(&text, model.Tokenize(value, false));
    AppendTokens(&rendered, model.Tokenize(value, true));
    AppendTokens(&chat, model.EncodeChat("", value));
  }
  for (int token = 0; token < model.VocabSize(); ++token) {
    const std::string bytes = model.DecodeToken(token);
    AppendU32(&token_text, static_cast<std::uint32_t>(bytes.size()));
    token_text.insert(token_text.end(), bytes.begin(), bytes.end());
  }
  WriteFile(out / "text.bin", text.data(), text.size());
  WriteFile(out / "rendered.bin", rendered.data(), rendered.size());
  WriteFile(out / "chat.bin", chat.data(), chat.size());
  WriteFile(out / "token_text.bin", token_text.data(), token_text.size());
  return 0;
}

int Prefill(ds4::Model& model, const std::filesystem::path& out,
            std::vector<int> prompt, int steps, std::uint32_t max_context,
            std::size_t prefix) {
  std::string error;
  auto session = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     max_context, &error);
  if (!session) {
    std::fprintf(stderr, "session: %s\n", error.c_str());
    return 1;
  }
  if (prefix != 0 &&
      !session->Sync(std::span<const int>(prompt.data(), prefix), &error)) {
    std::fprintf(stderr, "sync prefix: %s\n", error.c_str());
    return 1;
  }
  if (!session->Sync(prompt, &error)) {
    std::fprintf(stderr, "sync: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::uint32_t> ids;
  for (int step = 0; step <= steps; ++step) {
    const std::vector<float> logits = session->CopyLogits(&error);
    if (logits.empty()) {
      std::fprintf(stderr, "logits: %s\n", error.c_str());
      return 1;
    }
    WriteFile(out / ("logits_" + std::to_string(step) + ".bin"), logits.data(),
              logits.size() * sizeof(float));
    const int next = session->SelectNext(0.0F, nullptr);
    ids.push_back(static_cast<std::uint32_t>(next));
    if (step < steps && !session->Evaluate(next, &error)) {
      std::fprintf(stderr, "evaluate: %s\n", error.c_str());
      return 1;
    }
  }
  WriteFile(out / "ids.bin", ids.data(), ids.size() * 4);
  return 0;
}

int Batch(ds4::Model& model, const std::filesystem::path& out,
          const std::string& corpus_path, int steps,
          std::uint32_t max_context) {
  const std::vector<char> corpus = ReadFile(corpus_path);
  std::vector<std::unique_ptr<ds4::Session>> sessions;
  std::string error;
  std::size_t pos = 0;
  while (pos + 4 <= corpus.size()) {
    std::uint32_t length = 0;
    std::memcpy(&length, corpus.data() + pos, 4);
    pos += 4;
    if (pos + length > corpus.size()) {
      std::fprintf(stderr, "truncated corpus\n");
      return 2;
    }
    const std::vector<int> prompt =
        model.EncodeChat("", std::string_view(corpus.data() + pos, length));
    pos += length;
    WriteFile(out / ("prompt_" + std::to_string(sessions.size()) + ".bin"),
              prompt.data(), prompt.size() * 4);
    auto session = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                       max_context, &error);
    if (!session || !session->Sync(prompt, &error)) {
      std::fprintf(stderr, "session: %s\n", error.c_str());
      return 1;
    }
    sessions.push_back(std::move(session));
  }
  std::vector<std::vector<std::uint32_t>> ids(sessions.size());
  for (int step = 0; step <= steps; ++step) {
    std::vector<ds4::SessionBatchItem> items;
    for (std::size_t i = 0; i < sessions.size(); ++i) {
      const std::vector<float> logits = sessions[i]->CopyLogits(&error);
      if (logits.empty()) {
        std::fprintf(stderr, "logits: %s\n", error.c_str());
        return 1;
      }
      WriteFile(out / ("logits_" + std::to_string(i) + "_" +
                       std::to_string(step) + ".bin"),
                logits.data(), logits.size() * sizeof(float));
      const int next = sessions[i]->SelectNext(0.0F, nullptr);
      ids[i].push_back(static_cast<std::uint32_t>(next));
      items.push_back({sessions[i].get(), next});
    }
    if (step < steps && !model.EvaluateBatch(items, &error)) {
      std::fprintf(stderr, "evaluate batch: %s\n", error.c_str());
      return 1;
    }
  }
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    WriteFile(out / ("ids_" + std::to_string(i) + ".bin"), ids[i].data(),
              ids[i].size() * 4);
  }
  return 0;
}

int Snapshot(ds4::Model& model, const std::filesystem::path& out,
             const std::string& text_path, int steps, std::uint32_t context,
             std::uint32_t restore_context) {
  const std::vector<char> text = ReadFile(text_path);
  const std::vector<int> prompt =
      model.EncodeChat("", std::string_view(text.data(), text.size()));
  WriteFile(out / "prompt.bin", prompt.data(), prompt.size() * 4);
  std::string error;
  auto saved = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                   context, &error);
  if (!saved || !saved->Sync(prompt, &error)) {
    std::fprintf(stderr, "session: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::uint32_t> ids;
  for (int step = 0; step < 2; ++step) {
    const int next = saved->SelectNext(0.0F, nullptr);
    ids.push_back(static_cast<std::uint32_t>(next));
    if (!saved->Evaluate(next, &error)) {
      std::fprintf(stderr, "evaluate: %s\n", error.c_str());
      return 1;
    }
  }
  const auto snapshot = saved->SaveSnapshot(&error);
  if (!snapshot || snapshot->SizeBytes() != saved->PayloadBytes()) {
    std::fprintf(stderr, "snapshot: %s\n", error.c_str());
    return 1;
  }
  WriteFile(out / "payload.bin", snapshot->bytes().data(),
            snapshot->SizeBytes());
  saved.reset();
  auto restored = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      restore_context, &error);
  if (!restored || !restored->RestoreSnapshot(snapshot->bytes(), &error)) {
    std::fprintf(stderr, "restore: %s\n", error.c_str());
    return 1;
  }
  for (int step = 0; step <= steps; ++step) {
    const std::vector<float> logits = restored->CopyLogits(&error);
    if (logits.empty()) {
      std::fprintf(stderr, "logits: %s\n", error.c_str());
      return 1;
    }
    WriteFile(out / ("logits_" + std::to_string(step) + ".bin"), logits.data(),
              logits.size() * sizeof(float));
    const int next = restored->SelectNext(0.0F, nullptr);
    ids.push_back(static_cast<std::uint32_t>(next));
    if (step < steps && !restored->Evaluate(next, &error)) {
      std::fprintf(stderr, "evaluate: %s\n", error.c_str());
      return 1;
    }
  }
  WriteFile(out / "ids.bin", ids.data(), ids.size() * 4);
  return 0;
}

void AppendEmitted(std::vector<std::uint8_t>* out,
                   const std::vector<int>& emitted) {
  const auto count = static_cast<std::uint32_t>(emitted.size());
  const auto* p = reinterpret_cast<const std::uint8_t*>(&count);
  out->insert(out->end(), p, p + 4);
  for (const int token : emitted) {
    const auto id = static_cast<std::uint32_t>(token);
    p = reinterpret_cast<const std::uint8_t*>(&id);
    out->insert(out->end(), p, p + 4);
  }
}

bool WriteLogits(ds4::Session& session, const std::filesystem::path& path,
                 std::string* error) {
  const std::vector<float> logits = session.CopyLogits(error);
  if (logits.empty())
    return false;
  WriteFile(path, logits.data(), logits.size() * sizeof(float));
  return true;
}

std::vector<std::optional<gufo::sampling::SamplingConfig>> ParseSampling(
    const std::string& spec, std::size_t sessions) {
  std::vector<std::optional<gufo::sampling::SamplingConfig>> configs(sessions);
  std::size_t start = 0;
  for (std::size_t index = 0; index < sessions && start <= spec.size();
       ++index) {
    const std::size_t end = std::min(spec.find(';', start), spec.size());
    const std::string item = spec.substr(start, end - start);
    if (!item.empty() && item != "greedy") {
      gufo::sampling::SamplingConfig config;
      long long seed = -1;
      std::sscanf(item.c_str(), "%f,%f,%d,%lld", &config.temperature,
                  &config.top_p, &config.top_k, &seed);
      config.seed = seed;
      configs[index] = config;
    }
    start = end + 1;
  }
  return configs;
}

int Dspark(ds4::Model& model, const std::filesystem::path& out,
           const std::string& corpus_path, int cycles, std::uint32_t max_draft,
           int restore_cycles, std::uint32_t context,
           const std::string& sampling) {
  const std::vector<char> corpus = ReadFile(corpus_path);
  std::vector<std::unique_ptr<ds4::Session>> sessions;
  std::vector<std::vector<int>> prompts;
  std::string error;
  std::size_t pos = 0;
  while (pos + 4 <= corpus.size()) {
    std::uint32_t length = 0;
    std::memcpy(&length, corpus.data() + pos, 4);
    pos += 4;
    if (pos + length > corpus.size()) {
      std::fprintf(stderr, "truncated corpus\n");
      return 2;
    }
    const std::vector<int> prompt =
        model.EncodeChat("", std::string_view(corpus.data() + pos, length));
    pos += length;
    WriteFile(out / ("prompt_" + std::to_string(sessions.size()) + ".bin"),
              prompt.data(), prompt.size() * 4);
    auto session = model.CreateSession(gufo::core::SessionMode::kSpeculative,
                                       context, &error);
    if (!session || !session->Sync(prompt, &error)) {
      std::fprintf(stderr, "session: %s\n", error.c_str());
      return 1;
    }
    sessions.push_back(std::move(session));
    prompts.push_back(prompt);
  }
  const auto configs = ParseSampling(sampling, sessions.size());
  std::vector<std::optional<gufo::sampling::SamplerState>> samplers(
      sessions.size());
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    if (configs[i]) {
      const std::vector<gufo::sampling::TokenId> history(prompts[i].begin(),
                                                         prompts[i].end());
      samplers[i].emplace(*configs[i], history);
    }
  }
  std::vector<std::vector<std::uint8_t>> emitted_log(sessions.size());
  for (int cycle = 0; cycle < cycles; ++cycle) {
    std::vector<std::vector<int>> emitted(sessions.size());
    std::vector<std::optional<ds4::DsparkSamplerBridge>> bridges(
        sessions.size());
    std::vector<ds4::SessionDsparkBatchItem> items;
    for (std::size_t i = 0; i < sessions.size(); ++i) {
      if (samplers[i])
        bridges[i].emplace(*samplers[i]);
      items.push_back({.session = sessions[i].get(),
                       .max_tokens = 32,
                       .max_draft_tokens = max_draft,
                       .emitted = &emitted[i],
                       .sampler = bridges[i] ? bridges[i]->hook() : nullptr});
    }
    if (!model.DsparkStepBatch(items, &error)) {
      std::fprintf(stderr, "dspark step: %s\n", error.c_str());
      return 1;
    }
    for (std::size_t i = 0; i < sessions.size(); ++i) {
      if (samplers[i]) {
        samplers[i]->SetRngState(bridges[i]->rng_state());
        for (const int token : emitted[i])
          samplers[i]->Accept(static_cast<gufo::sampling::TokenId>(token));
      }
      AppendEmitted(&emitted_log[i], emitted[i]);
      if (!WriteLogits(*sessions[i],
                       out / ("logits_" + std::to_string(i) + "_" +
                              std::to_string(cycle) + ".bin"),
                       &error)) {
        std::fprintf(stderr, "logits: %s\n", error.c_str());
        return 1;
      }
    }
  }
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    WriteFile(out / ("emitted_" + std::to_string(i) + ".bin"),
              emitted_log[i].data(), emitted_log[i].size());
    const auto stats = sessions[i]->DsparkStatistics();
    const std::uint64_t values[9] = {stats.verifier_rows,
                                     stats.verifier_accepted,
                                     stats.support_drafted,
                                     stats.support_accepted,
                                     stats.positional_accepted,
                                     stats.anchors,
                                     stats.full_blocks,
                                     stats.steps,
                                     stats.skipped};
    std::vector<std::uint8_t> bytes(9 * 8 + 4);
    std::memcpy(bytes.data(), values, 9 * 8);
    std::memcpy(bytes.data() + 9 * 8, &stats.context_tokens, 4);
    WriteFile(out / ("stats_" + std::to_string(i) + ".bin"), bytes.data(),
              bytes.size());
    const auto snapshot = sessions[i]->SaveSnapshot(&error);
    if (!snapshot) {
      std::fprintf(stderr, "snapshot: %s\n", error.c_str());
      return 1;
    }
    WriteFile(out / ("payload_" + std::to_string(i) + ".bin"),
              snapshot->bytes().data(), snapshot->SizeBytes());
  }
  if (sessions.size() != 1 || restore_cycles == 0)
    return 0;
  const auto snapshot = sessions[0]->SaveSnapshot(&error);
  sessions.clear();
  auto restored = model.CreateSession(gufo::core::SessionMode::kSpeculative,
                                      context, &error);
  if (!snapshot || !restored ||
      !restored->RestoreSnapshot(snapshot->bytes(), &error)) {
    std::fprintf(stderr, "restore: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::uint8_t> restored_log;
  for (int cycle = 0; cycle < restore_cycles; ++cycle) {
    std::vector<int> emitted;
    if (!restored->DsparkStep(32, max_draft, &emitted, &error)) {
      std::fprintf(stderr, "restored step: %s\n", error.c_str());
      return 1;
    }
    AppendEmitted(&restored_log, emitted);
    if (!WriteLogits(
            *restored,
            out / ("restored_logits_" + std::to_string(cycle) + ".bin"),
            &error)) {
      std::fprintf(stderr, "logits: %s\n", error.c_str());
      return 1;
    }
  }
  WriteFile(out / "restored_emitted.bin", restored_log.data(),
            restored_log.size());
  return 0;
}

gufo::ReasoningEffort ParseEffort(const std::string& name) {
  if (name == "minimal")
    return gufo::ReasoningEffort::kMinimal;
  if (name == "medium")
    return gufo::ReasoningEffort::kMedium;
  if (name == "high")
    return gufo::ReasoningEffort::kHigh;
  if (name == "xhigh")
    return gufo::ReasoningEffort::kXHigh;
  if (name == "max")
    return gufo::ReasoningEffort::kMax;
  return gufo::ReasoningEffort::kLow;
}

std::string Member(const gufo::json::Value& value, const char* key,
                   const std::string& def = "") {
  const auto* member = value.find(key);
  return member != nullptr ? member->get_str(def) : def;
}

int Render(ds4::Model& model, const std::filesystem::path& out,
           const std::string& cases_path) {
  const std::vector<char> raw = ReadFile(cases_path);
  const gufo::json::Value cases =
      gufo::json::parse(std::string_view(raw.data(), raw.size()));
  std::size_t index = 0;
  for (const auto& item : cases.items()) {
    std::vector<ds4::ChatMessage> messages;
    if (const auto* list = item.find("messages")) {
      for (const auto& m : list->items()) {
        ds4::ChatMessage message{
            .role = Member(m, "role"),
            .content = Member(m, "content"),
            .reasoning_content = Member(m, "reasoning"),
            .tool_calls = {},
            .tool_call_id = Member(m, "tool_call_id"),
        };
        if (const auto* calls = m.find("tool_calls")) {
          for (const auto& c : calls->items()) {
            ds4::ChatMessage::ToolCall call{.name = Member(c, "name"),
                                            .arguments = {},
                                            .id = Member(c, "id")};
            if (const auto* arguments = c.find("arguments")) {
              for (const auto& a : arguments->items()) {
                const auto* is_string = a.find("string");
                call.arguments.push_back({
                    .name = Member(a, "name"),
                    .value = Member(a, "value"),
                    .is_string =
                        is_string == nullptr || is_string->as_bool(true),
                });
              }
            }
            message.tool_calls.push_back(std::move(call));
          }
        }
        messages.push_back(std::move(message));
      }
    }
    std::vector<ds4::ChatTool> tools;
    if (const auto* list = item.find("tools")) {
      for (const auto& t : list->items()) {
        tools.push_back({
            .name = Member(t, "name"),
            .description = Member(t, "description"),
            .parameters_json = Member(t, "parameters", "{}"),
            .definition_json = Member(t, "definition"),
        });
      }
    }
    ds4::ChatTemplateOptions options;
    if (const auto* o = item.find("options")) {
      const auto flag = [&](const char* key) {
        const auto* value = o->find(key);
        return value != nullptr && value->as_bool(false);
      };
      options.enable_thinking = flag("thinking");
      options.reasoning_effort = ParseEffort(Member(*o, "effort", "low"));
      options.preserve_thinking = flag("preserve");
      options.tools_present = flag("tools_present");
    }
    const std::string rendered = ds4::RenderChat(messages, tools, options);
    WriteFile(out / ("rendered_" + std::to_string(index) + ".txt"),
              rendered.data(), rendered.size());
    const std::vector<int> tokens = model.EncodeChat(messages, tools, options);
    WriteFile(out / ("tokens_" + std::to_string(index) + ".bin"), tokens.data(),
              tokens.size() * 4);
    ++index;
  }
  return 0;
}

std::vector<int> ReadTokens(const std::string& path) {
  const std::vector<char> raw = ReadFile(path);
  std::vector<int> tokens(raw.size() / 4);
  std::memcpy(tokens.data(), raw.data(), tokens.size() * 4);
  return tokens;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(
        stderr,
        "usage: %s MODEL OUTDIR tokens CORPUS | prefill PROMPT N "
        "[MAX_CONTEXT] | chat TEXT N [MAX_CONTEXT] | extend PROMPT N "
        "K [MAX_CONTEXT] | batch CORPUS N [MAX_CONTEXT] | snapshot TEXT N "
        "CONTEXT RESTORE_CONTEXT\n",
        argv[0]);
    return 2;
  }
  const std::string mode = argv[3];
  const int context_arg = mode == "extend" ? 7 : (mode == "dspark" ? 9 : 6);
  // A snapshot restores into the larger of its two contexts.
  const int restore_arg = mode == "snapshot" ? 7 : 0;
  const std::uint32_t max_context =
      mode != "tokens" && argc > context_arg
          ? static_cast<std::uint32_t>(
                std::strtoul(argv[context_arg], nullptr, 10))
          : 4096;
  ds4::ModelOptions options;
  options.max_context = max_context;
  if (mode == "dspark")
    options.dspark_model_path = argv[4];
  if (restore_arg != 0 && argc > restore_arg)
    options.max_context = std::max<std::uint32_t>(
        max_context, static_cast<std::uint32_t>(
                         std::strtoul(argv[restore_arg], nullptr, 10)));
  std::string error;
  auto model = ds4::Model::Load(argv[1], options, &error);
  if (!model) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  const std::filesystem::path out = argv[2];
  std::filesystem::create_directories(out);
  if (mode == "tokens")
    return Tokens(*model, out, argv[4]);
  if (mode == "render")
    return Render(*model, out, argv[4]);
  if (argc < 6) {
    std::fprintf(stderr, "missing step count\n");
    return 2;
  }
  const int steps = std::atoi(argv[5]);
  if (mode == "prefill")
    return Prefill(*model, out, ReadTokens(argv[4]), steps, max_context, 0);
  if (mode == "chat") {
    const std::vector<char> text = ReadFile(argv[4]);
    const std::vector<int> prompt =
        model->EncodeChat("", std::string_view(text.data(), text.size()));
    WriteFile(out / "prompt.bin", prompt.data(), prompt.size() * 4);
    return Prefill(*model, out, prompt, steps, max_context, 0);
  }
  if (mode == "snapshot" && argc > 7)
    return Snapshot(
        *model, out, argv[4], steps,
        static_cast<std::uint32_t>(std::strtoul(argv[6], nullptr, 10)),
        static_cast<std::uint32_t>(std::strtoul(argv[7], nullptr, 10)));
  if (mode == "dspark" && argc > 8)
    return Dspark(
        *model, out, argv[5], std::atoi(argv[6]),
        static_cast<std::uint32_t>(std::strtoul(argv[7], nullptr, 10)),
        std::atoi(argv[8]), max_context, argc > 10 ? argv[10] : "");
  if (mode == "batch")
    return Batch(*model, out, argv[4], steps, max_context);
  if (mode == "extend" && argc > 6)
    return Prefill(*model, out, ReadTokens(argv[4]), steps, max_context,
                   std::strtoul(argv[6], nullptr, 10));
  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
