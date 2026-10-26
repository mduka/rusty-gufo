// Dumps every Qwen3-ASR stage of the C++ engine for bit-exact comparison with
// the Rust port (docs/RUST.md): the same
// file names as `cargo run --example stage_dump`, raw little-endian values.
// With DUMP_IDS_ONLY set, it only transcribes three times in a fresh process.
//
// Build against the C++ baseline (upstream clone, built as documented):
//   R=/opt/rocm/core-10.0 O=$UPSTREAM B=$O/build/release
//   $R/llvm/bin/clang++ -std=c++20 -O2 -DENGINE_ENABLE_HIP=1 \
//     -D__HIP_PLATFORM_AMD__ -I$O -isystem $R/include \
//     --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15 cpp_stage_dump.cpp \
//     -o cpp_stage_dump $B/src/models/qwen3_asr/libgufo_qwen3_asr.a \
//     $B/libgufo_core.a -L$R/lib -Wl,-rpath,$R/lib -lhipblaslt -lhipblas \
//     -lamdhip64 -licuuc -lpthread
//   ./cpp_stage_dump MODEL WAV OUTDIR
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "src/models/qwen3_asr/audio.hpp"
#include "src/models/qwen3_asr/hip/audio_encoder_runtime.hpp"
#include "src/models/qwen3_asr/hip/text_decoder_runtime.hpp"
#include "src/models/qwen3_asr/hip/transcription_runtime.hpp"
#include "src/models/qwen3_asr/prompt.hpp"
#include "src/models/qwen3_asr/tokenizer.hpp"

namespace asr = gufo::models::qwen3_asr;
namespace asr_hip = gufo::models::qwen3_asr::hip;

template<typename T>
void Write(const std::filesystem::path& path, const std::vector<T>& values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  std::cerr << path.filename().string() << ": " << values.size() << "\n";
}

[[noreturn]] void Fail(const std::string& m) {
  std::cerr << "FAIL " << m << "\n";
  std::exit(1);
}

int main(int argc, char** argv) {
  if (argc != 4)
    Fail("usage: dump MODEL WAV OUTDIR");
  const std::string model = argv[1];
  const std::filesystem::path out = argv[3];
  std::filesystem::create_directories(out);
  std::ifstream input(argv[2], std::ios::binary);
  std::vector<char> chars((std::istreambuf_iterator<char>(input)), {});
  std::vector<std::byte> wav(chars.size());
  for (std::size_t i = 0; i < chars.size(); ++i)
    wav[i] = static_cast<std::byte>(static_cast<unsigned char>(chars[i]));

  std::string error;
  if (std::getenv("DUMP_IDS_ONLY") != nullptr) {
    auto runtime =
        asr_hip::TranscriptionHipRuntime::Create(model, 1024, &error);
    if (!runtime)
      Fail(error);
    asr::TranscriptionResult result;
    for (int run = 0; run < 3; ++run) {
      if (!runtime->Transcribe({.wav = wav, .max_new_tokens = 256}, {}, &result,
                               &error))
        Fail(error);
      Write(out / ("generated_run" + std::to_string(run) + ".u32"),
            result.generated_ids);
    }
    std::cout << result.text << "\n";
    return 0;
  }
  asr::AudioBuffer audio;
  if (!asr::DecodeWav(wav, &audio, &error))
    Fail(error);
  Write(out / "decoded.f32", audio.samples);
  const auto waveform = asr::ResampleMono16k(audio);
  Write(out / "waveform.f32", waveform);
  const auto features = asr::ComputeLogMelFeatures(waveform);
  Write(out / "features.f32", features.values);

  asr::Tokenizer tokenizer;
  if (!asr::Tokenizer::Load(model, &tokenizer, &error))
    Fail(error);
  const auto tokens = asr::AudioEmbeddingTokenCount(features.frames);
  const auto prompt =
      asr::BuildPrompt(tokenizer, {.context = {},
                                   .language = std::nullopt,
                                   .audio_embedding_tokens = tokens});
  Write(out / "prompt.u32", prompt);
  const auto context_ids =
      tokenizer.Encode("Hello, world! Grüße <|im_start|>x<asr_text>  y\n");
  Write(out / "encode_sample.u32", context_ids);

  {
    auto encoder = asr_hip::AudioEncoderHipRuntime::Create(model, &error);
    if (!encoder)
      Fail(error);
    asr_hip::AudioEncoderTrace trace;
    if (!encoder->Encode(features.values, features.frames, &trace, &error))
      Fail(error);
    Write(out / "encoder_frontend.f32", trace.frontend.values);
    Write(out / "encoder_layer0.f32", trace.layer0.values);
    Write(out / "encoder_final.f32", trace.final.values);

    auto decoder = asr_hip::TextDecoderHipRuntime::Create(model, 1024, &error);
    if (!decoder)
      Fail(error);
    asr_hip::TextDecoderTrace text;
    if (!decoder->Prefill(prompt, trace.final.values, trace.final.tokens, &text,
                          &error))
      Fail(error);
    Write(out / "decoder_layer0.f32", text.layer0);
    Write(out / "logits.f32", text.logits);
  }

  auto runtime = asr_hip::TranscriptionHipRuntime::Create(model, 1024, &error);
  if (!runtime)
    Fail(error);
  asr::TranscriptionResult result;
  if (!runtime->Transcribe({.wav = wav, .max_new_tokens = 256}, {}, &result,
                           &error))
    Fail(error);
  Write(out / "generated.u32", result.generated_ids);
  std::cout << result.language << "\n" << result.text << "\n";
  return 0;
}
