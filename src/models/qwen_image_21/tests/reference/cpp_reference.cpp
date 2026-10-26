// Dumps the C++ Qwen-Image-2.1 engine's outputs for the parity cases of
// tests/fixtures/cases.json (docs/RUST.md): token IDs of the tokenizer strings,
// and per generation the observer tensors as raw little-endian FP32
// (`<name>.f32`, rows x cols in `tensors.txt`; a repeated name gets
// `~<occurrence>`) and the decoded RGBA
// (`image.rgba`). The Rust parity test
// records its fixture from this directory.
//
// Build against an upstream v0.23.0 checkout configured with
// `cmake --preset gpu-test` (BUILD_TESTING=ON) and built:
//   R=/opt/rocm/core-10.0 S=<sources> B=<build>
//   $R/llvm/bin/clang++ -std=c++20 -O2 -DENGINE_ENABLE_HIP=1 \
//     -D__HIP_PLATFORM_AMD__ -I$S -isystem $R/include \
//     --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15 cpp_reference.cpp \
//     -o cpp_reference $B/src/models/qwen_image_21/libgufo_qwen_image_21.a \
//     $B/libgufo_core.a -L$R/lib -Wl,-rpath,$R/lib -lhipblaslt -lhipblas \
//     -lamdhip64 -licuuc -licui18n -lpng -ljpeg -lcurl -lpthread
//   ./cpp_reference MODEL CASES OUTDIR
//   ./cpp_reference decode OUTDIR FILE...   (tests/decode.rs)
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace image = gufo::models::qwen_image_21;
namespace json = gufo::json;

namespace {

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

// The synthetic reference image of cases.json.
image::Image Pattern(int width, int height, int pattern) {
  image::Image result{width, height, {}};
  result.rgba.resize(static_cast<std::size_t>(width) * height * 4);
  const auto p = static_cast<std::uint32_t>(pattern);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const auto ux = static_cast<std::uint32_t>(x);
      const auto uy = static_cast<std::uint32_t>(y);
      auto* pixel = &result.rgba[(static_cast<std::size_t>(y) * width + x) * 4];
      pixel[0] = static_cast<std::uint8_t>((ux * (3 + p) + 5 * uy) & 255);
      pixel[1] = static_cast<std::uint8_t>((ux * uy + 17 * p) & 255);
      pixel[2] = static_cast<std::uint8_t>(((ux ^ uy) * 7 + 31 * p) & 255);
      pixel[3] =
          static_cast<std::uint8_t>(p % 2 ? ((ux + 2 * uy) * 3) & 255 : 255);
    }
  return result;
}

bool Coarse(std::string_view name) {
  return name == "noise" || name == "sigmas" || name == "text.embedding" ||
         name == "text.output" || name == "vision.embedding" ||
         name == "vision.output" || name == "vae.encoded" ||
         name == "vae.decoder.mid" || name == "vae.decoded" ||
         name.starts_with("latent.") || name.ends_with(".output") ||
         name.ends_with(".modulation");
}

}  // namespace

// `cpp_reference decode OUTDIR FILE...`: DecodeImage of each file, as
// `<basename>.rgba` and a line `<basename> <width> <height>` (or
// `<basename> error <message>`) in `decoded.txt`.
int Decode(int argc, char** argv) {
  const std::filesystem::path output = argv[2];
  std::filesystem::create_directories(output);
  std::ofstream listing(output / "decoded.txt");
  for (int i = 3; i < argc; ++i) {
    const std::filesystem::path path = argv[i];
    const auto name = path.filename().string();
    const auto bytes = ReadFile(path);
    try {
      const auto decoded = image::DecodeImage(std::span(
          reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
      std::ofstream rgba(output / (name + ".rgba"), std::ios::binary);
      rgba.write(reinterpret_cast<const char*>(decoded.rgba.data()),
                 static_cast<std::streamsize>(decoded.rgba.size()));
      listing << name << ' ' << decoded.width << ' ' << decoded.height << '\n';
    } catch (const std::exception& error) {
      listing << name << " error " << error.what() << '\n';
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 3 && std::string_view(argv[1]) == "decode")
    return Decode(argc, argv);
  if (argc != 4) {
    std::cerr << "usage: cpp_reference MODEL CASES OUTDIR\n"
                 "       cpp_reference decode OUTDIR FILE...\n";
    return 2;
  }
  try {
    const std::filesystem::path output = argv[3];
    std::filesystem::create_directories(output);
    const auto cases = json::parse(ReadFile(argv[2]));
    image::Model model(argv[1]);
    {
      std::ofstream tokens(output / "tokens.txt");
      for (const auto& text : cases.find("tokenizer")->items()) {
        for (const auto id : model.Tokenize(text.str()))
          tokens << id << ' ';
        tokens << '\n';
      }
    }
    for (const auto& item : cases.find("generations")->items()) {
      const auto name = item.member_str("name");
      const auto directory = output / name;
      std::filesystem::create_directories(directory);
      image::Request request;
      request.prompt = item.member_str("prompt");
      request.width = static_cast<int>(item.member_double("width"));
      request.height = static_cast<int>(item.member_double("height"));
      request.steps = static_cast<int>(item.member_double("steps"));
      request.seed = static_cast<std::uint64_t>(item.member_double("seed"));
      for (const auto& reference : item.find("images")->items())
        request.images.push_back(
            Pattern(static_cast<int>(reference.member_double("width")),
                    static_cast<int>(reference.member_double("height")),
                    static_cast<int>(reference.member_double("pattern"))));
      const bool all = item.member_str("observe") == "all";
      std::ofstream shapes(directory / "tensors.txt");
      // Tensors observed again (one vision tower per reference image) get
      // their occurrence as a suffix: `vision.output~2`.
      std::map<std::string, int, std::less<>> seen;
      image::Observer observer;
      observer.include = [all](std::string_view tensor) {
        return all || Coarse(tensor);
      };
      observer.write = [&](std::string_view tensor,
                           std::span<const float> values, int rows, int cols) {
        const int occurrence = ++seen[std::string(tensor)];
        const auto name = occurrence == 1 ? std::string(tensor)
                                          : std::string(tensor) + "~" +
                                                std::to_string(occurrence);
        std::ofstream out(directory / (name + ".f32"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size_bytes()));
        shapes << name << ' ' << rows << ' ' << cols << '\n';
      };
      const auto result = model.Generate(request, {}, observer);
      std::ofstream rgba(directory / "image.rgba", std::ios::binary);
      rgba.write(reinterpret_cast<const char*>(result.image.rgba.data()),
                 static_cast<std::streamsize>(result.image.rgba.size()));
      std::cerr << name << ": " << result.image.width << 'x'
                << result.image.height
                << " total_ms=" << result.metrics.total_ms << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
