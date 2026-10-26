// ICU 78 and the C++ Qwen tokenizer as references for gufo-core
// (docs/RUST.md): `unicode` prints FNV-1a digests of the
// pre-tokenizer's Unicode predicates over every code point (pinned in
// src/core/src/unicode.rs); `encode GGUF TEXTS` prints the token IDs of each
// NUL-separated text in TEXTS.
//
// Build against an upstream C++ checkout ($UPSTREAM, built as documented):
//   O=$UPSTREAM B=$O/build/release
//   c++ -std=c++20 -O2 -I$O icu_tokenizer_check.cpp -o icu_tokenizer_check \
//     $B/libgufo_core.a -licuuc -licudata -lpthread
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

struct Fnv {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  void Byte(std::uint8_t b) { hash = (hash ^ b) * 0x100000001b3ULL; }
  void U32(std::uint32_t v) {
    for (int i = 0; i < 4; ++i)
      Byte(static_cast<std::uint8_t>(v >> (8 * i)));
  }
  void Bytes(const std::string& s) {
    U32(static_cast<std::uint32_t>(s.size()));
    for (const char c : s)
      Byte(static_cast<std::uint8_t>(c));
  }
};

std::string Utf8(const icu::UnicodeString& s) {
  std::string out;
  s.toUTF8String(out);
  return out;
}

int Unicode() {
  UErrorCode status = U_ZERO_ERROR;
  const auto* nfc = icu::Normalizer2::getNFCInstance(status);
  const auto* nfd = icu::Normalizer2::getNFDInstance(status);
  if (U_FAILURE(status))
    return 1;
  Fnv flags, folds, singles, recomposed;
  std::uint32_t fold_count = 0, single_count = 0, recomposed_count = 0;
  for (UChar32 cp = 0; cp <= 0x10FFFF; ++cp) {
    if (cp >= 0xD800 && cp <= 0xDFFF)
      continue;
    const auto mask = U_GET_GC_MASK(cp);
    const bool letter = (mask & U_GC_L_MASK) != 0;
    const bool mark = (mask & U_GC_M_MASK) != 0;
    const auto type = static_cast<UCharCategory>(u_charType(cp));
    const bool number = type == U_DECIMAL_DIGIT_NUMBER ||
                        type == U_LETTER_NUMBER || type == U_OTHER_NUMBER;
    const bool space = u_isUWhiteSpace(cp) != 0;
    flags.Byte(static_cast<std::uint8_t>(letter | (letter || mark) << 1 |
                                         number << 2 | space << 3));
    const UChar32 folded = u_foldCase(cp, U_FOLD_CASE_DEFAULT);
    for (const char letter_char : std::string("strevmld")) {
      if (folded == letter_char) {
        folds.U32(static_cast<std::uint32_t>(cp));
        folds.Byte(static_cast<std::uint8_t>(letter_char));
        ++fold_count;
      }
    }
    const icu::UnicodeString one(cp);
    const icu::UnicodeString normalized = nfc->normalize(one, status);
    if (normalized != one) {
      singles.U32(static_cast<std::uint32_t>(cp));
      singles.Bytes(Utf8(normalized));
      ++single_count;
    }
    const icu::UnicodeString decomposed = nfd->normalize(one, status);
    if (decomposed != one) {
      recomposed.U32(static_cast<std::uint32_t>(cp));
      recomposed.Bytes(Utf8(nfc->normalize(decomposed, status)));
      ++recomposed_count;
    }
  }
  if (U_FAILURE(status))
    return 1;
  std::printf("icu %s unicode %s\n", U_ICU_VERSION, U_UNICODE_VERSION);
  std::printf("flags 0x%016llx\n", static_cast<unsigned long long>(flags.hash));
  std::printf("folds %u 0x%016llx\n", fold_count,
              static_cast<unsigned long long>(folds.hash));
  std::printf("nfc_singles %u 0x%016llx\n", single_count,
              static_cast<unsigned long long>(singles.hash));
  std::printf("nfc_recomposed %u 0x%016llx\n", recomposed_count,
              static_cast<unsigned long long>(recomposed.hash));
  return 0;
}

int Encode(const char* gguf, const char* texts_path) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(gguf, &error);
  if (!reader) {
    std::cerr << error << "\n";
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::cerr << error << "\n";
    return 1;
  }
  std::ifstream input(texts_path, std::ios::binary);
  const std::string all((std::istreambuf_iterator<char>(input)), {});
  std::size_t start = 0;
  while (start <= all.size()) {
    const std::size_t end = std::min(all.find('\0', start), all.size());
    if (end == start && end == all.size())
      break;
    for (const auto id : tokenizer->Encode(all.substr(start, end - start)))
      std::printf("%u ", id);
    std::printf("\n");
    start = end + 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "unicode")
    return Unicode();
  if (argc == 4 && std::string(argv[1]) == "encode")
    return Encode(argv[2], argv[3]);
  std::cerr << "usage: icu_tokenizer_check unicode | encode GGUF TEXTS\n";
  return 2;
}
