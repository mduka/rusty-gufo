// WebP references for the Rust decoder (`tests/image.rs`): a synthetic
// 40x28 image (a gradient with sharp edges, not a multiple of the 16-pixel
// macroblock) encoded lossy and lossless, with and without alpha, and
// libwebp's own RGB decode of each (WebPDecodeRGBInto, as the C++ server
// decodes). Writes NAME.webp and NAME.rgb into OUTDIR.
//
//   cc -O2 src/core/tests/reference/libwebp_reference.c -lwebp -o /tmp/libwebp_reference
//   /tmp/libwebp_reference src/core/tests/fixtures/webp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <webp/decode.h>
#include <webp/encode.h>

enum { kWidth = 40, kHeight = 28 };

static void Write(const char* dir, const char* name, const char* suffix,
                  const uint8_t* data, size_t size) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s.%s", dir, name, suffix);
  FILE* file = fopen(path, "wb");
  if (file == NULL || fwrite(data, 1, size, file) != size) {
    fprintf(stderr, "cannot write %s\n", path);
    exit(1);
  }
  fclose(file);
}

static void Record(const char* dir, const char* name, uint8_t* encoded,
                   size_t size) {
  if (size == 0) {
    fprintf(stderr, "encoding %s failed\n", name);
    exit(1);
  }
  uint8_t rgb[kWidth * kHeight * 3];
  if (WebPDecodeRGBInto(encoded, size, rgb, sizeof(rgb), kWidth * 3) == NULL) {
    fprintf(stderr, "decoding %s failed\n", name);
    exit(1);
  }
  Write(dir, name, "webp", encoded, size);
  Write(dir, name, "rgb", rgb, sizeof(rgb));
  WebPFree(encoded);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: libwebp_reference OUTDIR\n");
    return 2;
  }
  uint8_t rgb[kWidth * kHeight * 3];
  uint8_t rgba[kWidth * kHeight * 4];
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const int i = y * kWidth + x;
      const uint8_t pixel[4] = {
          (uint8_t)(x * 6), (uint8_t)(y * 9),
          (uint8_t)(((x / 5 + y / 7) % 2) ? 230 : 20),
          (uint8_t)(x < kWidth / 2 ? 255 : 60 + y * 4)};
      memcpy(rgb + 3 * i, pixel, 3);
      memcpy(rgba + 4 * i, pixel, 4);
    }
  }
  uint8_t* encoded = NULL;
  size_t size = WebPEncodeRGB(rgb, kWidth, kHeight, kWidth * 3, 80, &encoded);
  Record(argv[1], "lossy", encoded, size);
  size = WebPEncodeRGBA(rgba, kWidth, kHeight, kWidth * 4, 80, &encoded);
  Record(argv[1], "lossy_alpha", encoded, size);
  size = WebPEncodeLosslessRGB(rgb, kWidth, kHeight, kWidth * 3, &encoded);
  Record(argv[1], "lossless", encoded, size);
  size = WebPEncodeLosslessRGBA(rgba, kWidth, kHeight, kWidth * 4, &encoded);
  Record(argv[1], "lossless_alpha", encoded, size);
  printf("libwebp %d.%d.%d\n", WebPGetDecoderVersion() >> 16,
         (WebPGetDecoderVersion() >> 8) & 0xff, WebPGetDecoderVersion() & 0xff);
  return 0;
}
