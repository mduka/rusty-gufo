// Dumps the C++ Qwen3.8 vision path for bit-exact comparison with the Rust
// port (docs/RUST.md): per image the decoded
// RGB8 pixels, the resized grid and the encoder's stages, raw little-endian.
//
// Build against the C++ baseline (upstream clone, built as documented):
//   R=/opt/rocm/core-10.0 O=$UPSTREAM B=$O/build/release
//   $R/llvm/bin/clang++ -std=c++20 -O2 -DENGINE_ENABLE_HIP=1 \
//     -D__HIP_PLATFORM_AMD__ -I$O -isystem $R/include \
//     --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15 cpp_vision_dump.cpp \
//     -o cpp_vision_dump $B/libgufo_core.a -L$R/lib -Wl,-rpath,$R/lib \
//     -lhipblaslt -lhipblas -lamdhip64 -licuuc -lpng -ljpeg -lcurl -lcrypto \
//     -lpthread
//   ./cpp_vision_dump MMPROJ WIDTH OUTDIR IMAGE...
// WIDTH is the projector width (5120 for the 27B, 2560 for Flash-Next). For
// image i it writes i.rgb and i.resized ("W H\n" + RGB8), i.<stage>.f32 for
// every encoder stage (patch, layer0..layer26, embedding), identity.txt and
// sidecar.txt (the sidecar path and width), and images.txt (the inputs).
// Without MMPROJ ("-") it only decodes and resizes; i.error holds the
// message of an image that fails either step.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/prompt.hpp"

namespace {
void WriteImage(const std::filesystem::path& path,
                const gufo::core::Image& image) {
  std::ofstream out(path, std::ios::binary);
  out << image.width << ' ' << image.height << '\n';
  out.write(reinterpret_cast<const char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()));
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: cpp_vision_dump MMPROJ WIDTH OUTDIR IMAGE...\n";
    return 2;
  }
  const std::string mmproj = argv[1];
  const auto width = static_cast<std::uint32_t>(std::stoul(argv[2]));
  const std::filesystem::path out = argv[3];
  std::filesystem::create_directories(out);
  std::unique_ptr<gufo::models::qwen::vision::Encoder> encoder;
  if (mmproj != "-") {
    encoder =
        std::make_unique<gufo::models::qwen::vision::Encoder>(mmproj, width);
    std::ofstream(out / "identity.txt") << encoder->identity() << '\n';
    std::ofstream(out / "sidecar.txt") << mmproj << '\n' << width << '\n';
  }
  std::ofstream list(out / "images.txt");
  for (int i = 4; i < argc; ++i)
    list << std::filesystem::absolute(argv[i]).string() << '\n';
  list.close();
  for (int i = 4; i < argc; ++i) {
    std::ifstream in(argv[i], std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          {});
    const auto index = std::to_string(i - 4);
    gufo::core::Image decoded;
    try {
      decoded = gufo::core::DecodeImage(bytes);
    } catch (const std::exception& error) {
      std::ofstream(out / (index + ".error")) << error.what() << '\n';
      continue;
    }
    WriteImage(out / (index + ".rgb"), decoded);
    gufo::core::Image resized;
    try {
      resized = gufo::models::qwen::vision::ResizeImage(decoded);
    } catch (const std::exception& error) {
      std::ofstream(out / (index + ".error")) << error.what() << '\n';
      continue;
    }
    WriteImage(out / (index + ".resized"), resized);
    if (!encoder)
      continue;
    (void)encoder->Encode(
        resized, [&](std::string_view stage, std::span<const float> values) {
          std::ofstream file(out / (index + "." + std::string(stage) + ".f32"),
                             std::ios::binary);
          file.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size_bytes()));
        });
  }
  return 0;
}
