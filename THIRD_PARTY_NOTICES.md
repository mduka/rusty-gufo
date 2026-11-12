# Third-party notices

rusty-gufo's original source is licensed under `MIT OR Apache-2.0`; code adapted
from Gufo and the projects below keeps its MIT license. This does not relicense dependencies or
model weights. This inventory covers direct dependencies, adapted code and
acknowledged optimization references.
Transitive system/toolchain dependencies retain their upstream licenses.
`Cargo.lock` pins every Rust crate the binary compiles in; the ROCm
libraries come from the installed ROCm (10.1.0 is qualified).
`tools/requirements.txt` pins the Python tools, and
`tools/bench/build-reference.sh` pins the commits of the benchmark reference
runtimes. The container image of the [Containerfile](Containerfile) also ships
Ubuntu's FFmpeg libraries and AMD's ROCm runtime packages, under their own
licenses (their `copyright` files are in the image). Distributions must retain
the notices of the versions they actually ship.

## Runtime and compiled code

| Component Name | Relationship | License (SPDX) | Pinned Revision / Version | Upstream Source / Location |
| --- | --- | --- | --- | --- |
| ROCm / HIP | Linked runtime | `MIT` | ROCm 10.1.0; installed ROCm | [ROCm/clr](https://github.com/ROCm/clr) |
| hipBLAS | Linked | `MIT` | ROCm 10.1.0; installed ROCm | [ROCm/hipBLAS](https://github.com/ROCm/hipBLAS) |
| hipBLASLt | Linked | `MIT` | ROCm 10.1.0; installed ROCm | [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt) |
| rocBLAS | Linked | `MIT` | ROCm 10.1.0; installed ROCm | [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) |
| hipCUB | Headers compiled into kernels | `BSD-3-Clause` | ROCm 10.1.0; installed ROCm | [ROCm/hipCUB](https://github.com/ROCm/hipCUB) |
| rocPRIM | Headers compiled into kernels | `MIT` | ROCm 10.1.0; installed ROCm | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) |
| rocWMMA | Headers compiled into kernels | `MIT` | ROCm 10.1.0; installed ROCm | [ROCm/rocWMMA](https://github.com/ROCm/rocWMMA) |
| Composable Kernel | Adapted short-attention arithmetic; no library/header dependency | `MIT` | ROCm 10.1.0; installed ROCm | [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) |
| ureq / rustls / ring | Compiled in; HTTPS for evaluation and `image_url`, with webpki-roots and Mozilla's root certificates | `MIT OR Apache-2.0` (ureq); `Apache-2.0 OR ISC OR MIT` (rustls); `Apache-2.0 AND ISC` (ring); `ISC` (rustls-webpki); `CDLA-Permissive-2.0` (webpki-roots, text in `licenses/webpki-roots.txt`) | ureq 3.4.2, rustls 0.23.46, ring 0.17.14, webpki-roots 1.0.11; Cargo.lock | [ureq](https://github.com/algesten/ureq), [rustls](https://github.com/rustls/rustls), [ring](https://github.com/briansmith/ring), [webpki-roots](https://github.com/rustls/webpki-roots) |
| png / zune-jpeg / image-webp | Compiled in; image decoding and PNG encoding | `MIT OR Apache-2.0` (png, image-webp); `MIT OR Apache-2.0 OR Zlib` (zune-jpeg) | png 0.18.1, zune-jpeg 0.5.15, image-webp 0.2.4; Cargo.lock | [image-png](https://github.com/image-rs/image-png), [zune-image](https://github.com/etemesi254/zune-image), [image-webp](https://github.com/image-rs/image-webp) |
| unicode-normalization / unicode-properties | Compiled in; tokenizer normalization | `MIT OR Apache-2.0` | 0.1.25 / 0.1.4; Cargo.lock | [unicode-rs](https://github.com/unicode-rs) |
| hound / claxon / lewton / nanomp3 / rubato | Compiled in; WAV, FLAC, Ogg Vorbis and MP3 decoding and resampling of audio embedding input | `Apache-2.0` (hound, claxon); `MIT OR Apache-2.0` (lewton, nanomp3, rubato) | hound 3.5.1, claxon 0.4.3, lewton 0.10.2, nanomp3 0.2.0, rubato 5.0.1; Cargo.lock | [hound](https://github.com/ruuda/hound), [claxon](https://github.com/ruuda/claxon), [lewton](https://github.com/RustAudio/lewton), [nanomp3](https://crates.io/crates/nanomp3), [rubato](https://github.com/HEnquist/rubato) |
| sha2 / sha1_smol | Compiled in; artifact identities and the WebSocket handshake | `MIT OR Apache-2.0` (sha2); `BSD-3-Clause` (sha1_smol) | 0.10.9 / 1.0.1; Cargo.lock | [RustCrypto](https://github.com/RustCrypto/hashes), [sha1-smol](https://github.com/mitsuhiko/sha1-smol) |
| Other Rust crates | Compiled in: serde, serde_json, chrono, libc, rustix, signal-hook, toml, jobserver | `MIT OR Apache-2.0` (rustix also `Apache-2.0 WITH LLVM-exception`) | Cargo.lock | [crates.io](https://crates.io) |
| FFmpeg | Loaded on the first video embedding (libavformat, libavcodec, libavutil, libswscale); headers compiled into `gufo-ffmpeg`'s field shim; not linked | `LGPL-2.1-or-later` (a build with GPL components: `GPL-2.0-or-later` or `GPL-3.0-or-later`) | installed FFmpeg, the major version of the build's headers (8.x qualified) | [FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| GNU C/C++/OpenMP runtimes | System runtime libraries; no source import | `LGPL-2.1-or-later AND (GPL-3.0-or-later WITH GCC-exception-3.1)` | glibc/GCC of the build host | [GNU](https://www.gnu.org/software/) |
| llama.cpp / ggml | Adapted quantization, attention and model-private HIP kernels | `MIT` | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`; attention `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| mikealanni/gufo | Adapted Flash-Next GDN row reuse and SSM stride optimization | `MIT` | `f450218743bcaf77e3e1f8fab78f4cd4e4aafae3` | [gufo fork](https://github.com/mikealanni/gufo/tree/f450218743bcaf77e3e1f8fab78f4cd4e4aafae3) |
| Qwen chat templates | Reference Jinja and adapted renderer behavior | `Apache-2.0` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (27B), `de4b8e4d43b917e7706784d8bb445c9af86a3540` (Flash-Next) | [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) |
| DS4 | Adapted loader, tokenizer, sessions and HIP kernels | `MIT` | `84cc882352757baf628a1776badf7cc54d584e28` | [antirez/ds4](https://github.com/antirez/ds4) |
| DS4 GB10/GX10 fork | Adapted paired MoE launch code, now HIP | `MIT` | `910501e` | [xangel82/DS4](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA) |
| OmniVoice extension for Gufo | Ported to Rust (`src/models/omnivoice`); its HIP kernels compiled in | `MIT` | `4bf376c` | [gufo-extensions/omnivoice](https://github.com/gufo-extensions/omnivoice) |
| OmniVoice | Prompt preprocessing, reference and output audio processing, the unmasking schedule and position choice, generation with chunking, and the language table, through the OmniVoice extension | `Apache-2.0` | `08be0b4` | [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) |
| Hugging Face transformers | The OmniVoice audio tokenizer's encoder and decoder (HiggsAudioV2Tokenizer, DAC, HuBERT) and the generator's Qwen3 layers, through the OmniVoice extension | `Apache-2.0` | `5.22.2` | [huggingface/transformers](https://github.com/huggingface/transformers) |
| pydub | Silence detection and removal reproduced in OmniVoice's audio processing | `MIT` | `0.25.1` | [jiaaro/pydub](https://github.com/jiaaro/pydub) |
| torchaudio | Sinc resampler reproduced in OmniVoice's audio processing | `BSD-2-Clause` | `2.11.1` | [pytorch/audio](https://github.com/pytorch/audio) |
| Gemma 4 extension for Gufo (Francesco Moca) | Ported to Rust (`src/models/gemma4`); its HIP kernels compiled in, among them its copies of Gufo's GGML and binary16 kernels | `MIT` | `a31ce42` | [franzmoca/gemma4-gufo-extension](https://github.com/franzmoca/gemma4-gufo-extension) |
| Qwen3.6 35B-A3B extension for Gufo (Strycast) | Ported to Rust (`src/models/qwen35moe`); its HIP kernels compiled in, among them its copies of Gufo's Qwen kernels and, through Flash-Next's copies, llama.cpp's quantized matrix kernels (`src/models/qwen35moe/kernels/vendor`) | `MIT` | `4882b7a` (Strycast `8c946372`) | [slimsami/strycast-qwen35moe-gufo-extension](https://github.com/slimsami/strycast-qwen35moe-gufo-extension) |
| Nemotron extension for Gufo | Ported to Rust (`src/models/nemotron`); its HIP kernels compiled in, among them llama.cpp's quantized matrix kernels through Gufo's and DS4's copies (`src/models/nemotron/kernels/rocm/mmq/VENDOR.md`) | `MIT` | `62f85dc` | [gufo-extensions/nemotron](https://github.com/gufo-extensions/nemotron) |
| NVIDIA Nemotron-H Omni model code | The image processor's patch grid and spans, the C-RADIO encoder and projector, through the Nemotron extension | `Apache-2.0` | BF16 checkpoint `e09a4cedb1787db1760375ee5539f222cae0c672`; transformers 5.22.2 | [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16) |
| PyTorch | Antialiased bicubic image resize and bilinear resize of the position table in the Nemotron vision encoder, through the Nemotron extension | `BSD-3-Clause` | 2.15.0 | [pytorch/pytorch](https://github.com/pytorch/pytorch) |

The OmniVoice rows' license texts are `licenses/omnivoice-extension.txt`,
`licenses/OmniVoice.txt`, `licenses/transformers.txt`, `licenses/pydub.txt`
and `licenses/torchaudio.txt`.
The Gemma 4 row's license text is `licenses/gemma4-extension.txt`; the parts
it copied from Gufo keep Gufo's MIT license (`LICENSE-MIT`).
The Qwen3.6 35B-A3B row's license text is `licenses/qwen35moe-extension.txt`;
its llama.cpp kernels are under `licenses/llama.cpp.txt`.
The Nemotron rows' license texts are `licenses/nemotron-extension.txt`,
`licenses/nemotron-omni.txt` and `licenses/pytorch.txt`; its quantized matrix
kernels are under `licenses/llama.cpp.txt` and
[src/models/deepseek_v4_flash/LICENSE.ds4](src/models/deepseek_v4_flash/LICENSE.ds4).

Upstream license texts are retained in `licenses/` and
[src/models/deepseek_v4_flash/LICENSE.ds4](src/models/deepseek_v4_flash/LICENSE.ds4).
Distribute these with `LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE` and this inventory alongside the
binary. Preserve embedded copyright notices when modifying adapted code. Model-private changes and import boundaries are recorded in the
[DS4 provenance](src/models/deepseek_v4_flash/UPSTREAM.md),
[DS4 HIP import](src/models/deepseek_v4_flash/kernels/rocm/mmq/VENDOR.md),
and [Flash-Next HIP import](src/models/qwen38_flash_next/kernels/rocm/mmq/VENDOR.md).

The mikealanni/gufo fork retains the same Copyright (c) 2026 gufo contributors
notice and MIT text as Gufo's root `LICENSE`.

rusty-gufo loads the installed FFmpeg libraries on demand for video embeddings; a
system FFmpeg build can carry GPL terms through its enabled components.
Redistributors bundling FFmpeg must also provide its applicable license texts
and corresponding source/build materials, not just rusty-gufo's licenses.

External model files are never part of the binary package.

## Optimization inspiration

The following projects informed Gufo's Strix Halo optimization work:

- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp) — [MIT license](https://github.com/LaurentZuijdwijk/llama.cpp/blob/11bfe8a633fa02bac251db6cf21bd5ddab282a64/LICENSE).
- [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp) — [MIT license](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/ce15ecca66e5e5a9aefa1ebc82357cb901330c86/LICENSE).
- [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) — [MIT license](https://github.com/gaetan-puleo/llama-cpp-strix-halo/blob/860c828363988d3e4b3d5c2b701dcba3d7b9f26c/LICENSE).

License links pin the notices reviewed for these inspiration credits.
The two llama.cpp forks share the
ggml authors' notice in [licenses/llama.cpp.txt](licenses/llama.cpp.txt);
Nathan Wilson's notice is retained in
[licenses/strix-halo-llamacpp.txt](licenses/strix-halo-llamacpp.txt).

## Build and development only

| Component Name | Relationship | License (SPDX) | Pinned Revision / Version | Upstream Source / Location |
| --- | --- | --- | --- | --- |
| LLVM/Clang | HIP build toolchain | `Apache-2.0 WITH LLVM-exception` | ROCm 10.1.0 LLVM; installed ROCm | [ROCm/llvm-project](https://github.com/ROCm/llvm-project) |
| ROCprofiler SDK / ROCTx | Development profiling and benchmark markers only | `MIT` | ROCm 10.1.0; installed ROCm | [ROCm/rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) |
| PyTorch ROCm | Independent evaluation; not shipped | `BSD-3-Clause` | 2.12.0 (ROCm 7.2 wheel); tools/requirements.txt | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| Torchvision | Evaluation; not shipped | `BSD-3-Clause` | 0.27.0 (ROCm 7.2 wheel); tools/requirements.txt | [pytorch/vision](https://github.com/pytorch/vision) |
| Transformers / Accelerate / Safetensors / Hugging Face Hub | Evaluation, artifact inspection and model acquisition | `Apache-2.0` | tools/requirements.txt | [Hugging Face](https://github.com/huggingface) |
| NumPy / SciPy | Evaluation arrays and signal analysis | `BSD-3-Clause` | tools/requirements.txt | [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy) |
| Requests | Evaluation HTTP client | `Apache-2.0` | tools/requirements.txt | [Requests](https://github.com/psf/requests) |
| OpenAI Python SDK | Local API compatibility checks; not shipped | `Apache-2.0` | 2.41.1; tools/requirements.txt | [openai-python](https://github.com/openai/openai-python) |
| SoX | Development audio utility | `GPL-2.0-or-later` | System package | [SoX](https://sourceforge.net/projects/sox/) |
| antirez/ds4 | Optional benchmark package; not shipped | `MIT` | `0aaea5a238fb41a35106a551e73c8409dfb751ac` | [Build script](tools/bench/build-reference.sh) |
| llama.cpp reference builds | Optional benchmark packages; not shipped | `MIT` | Release `68d9053a`; Flash-Next MTP `6fcaa16f` | [Build script](tools/bench/build-reference.sh) |

Python/PyTorch and reference scripts are not installed with the binary. Their
own transitive dependencies, including Triton and MIOpen, remain
development-only. Kernel microbenchmarks build separately
(`tools/bench/build.sh`); ROCprofiler comes with ROCm. Ordinary build tools
(Cargo, rustc, hipcc and the host compiler) retain their own licenses. The C++
reference programs under `tests/reference/` link libraries such as ICU,
libpng, libjpeg-turbo and libcurl of the build host; they are development
tools and not part of the binary.

## Test data

The test inputs in the repository are synthetic or written for the tests,
except for the data below. None of it is part of the binary.

| Data | Files | Source | License |
| --- | --- | --- | --- |
| English speech recording (10 s) and its transcript | `tests/functional/fixtures/audio/clear-english-voice.*` | Gufo's `tests/functional/fixtures/audio`, supplied and released for publication by Gufo's maintainer | `MIT` (Gufo) |
| 75 benchmark questions: 25 GPQA Diamond (one modified, marked in the file), 25 SuperGPQA, 25 AIME 2025 | `tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json`; one GPQA question also in `src/models/deepseek_v4_flash/tests/fixtures/forward.json` | The DS4 capability subset, [antirez/ds4](https://github.com/antirez/ds4) `ds4_eval.c` at `84cc882352757baf628a1776badf7cc54d584e28`, from [GPQA](https://huggingface.co/datasets/Wanfq/gpqa) (Rein et al., 2023), [SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) (M-A-P, 2025) and [AIME 2025](https://huggingface.co/datasets/test-time-compute/aime_2025) | `CC-BY-4.0` (GPQA), `ODC-By-1.0` (SuperGPQA), `MIT` as the AIME 2025 dataset declares (the problems are the Mathematical Association of America's) |
| 105 DeepSeek prompts and official 0731 continuations | `tests/models/deepseek_v4_flash/fixtures/official-0731.json` | [antirez/ds4](https://github.com/antirez/ds4) at `6289c516273979173abbc062209a81dd3706b804`; the fixture retains source hashes and its upstream notice | `MIT` |
| Transcript of Qwen's English ASR example | `src/models/qwen3_asr/tests/fixtures/qwen3-asr-en.json` | The `asr_en.wav` example of Qwen3-ASR; the recording itself stays out of the repository ([local fixtures](docs/development/TESTING.md#local-fixtures)) | Qwen's example |
| Opening of *Alice's Adventures in Wonderland* (Lewis Carroll, 1865) | `tests/models/qwen3_tts/quality/long-form.txt` and the Qwen3-TTS reference sentence | [Project Gutenberg eBook #11](https://www.gutenberg.org/ebooks/11) | Public domain |

## External Model Artifacts (Non-Distributed)

Model checkpoints are not part of the rusty-gufo source or binary
distribution. Operators obtain them directly from their publisher and remain
responsible for the terms governing their location and use.

### Qwen3.8 27B and Qwen3.5

- **Artifact**: Unsloth's GGUFs of Qwen's Qwen3.8 27B
  ([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  at `4ca720788d1e01f1bff70c033e0d0028fd02e502`, with its BF16 vision
  sidecar), z-lab's DFlash2 drafter
  ([z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)
  at `2d9571f8ce46e151f61c6499c99dee6079e1d610`) and the dense Qwen3.5 0.8B
  to 9B (`unsloth/Qwen3.5-*-GGUF`, revisions in the model page).
- **License**: Apache-2.0, as the GGUF metadata and model cards declare.
  rusty-gufo neither ships nor automatically downloads these weights.

### Qwen3.8 Flash-Next

- **Artifact**: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
  at `38bb39ee97821de2c9009abb7e93950eec396e66`: the UD-Q4_K_XL shards, the
  shared Q8_0 MTP sidecar and the BF16 vision sidecar.
- **License**: Qwen's Qwen Community License 1.0, as the
  [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) declares.
  rusty-gufo neither ships nor automatically downloads these weights.

### DeepSeek V4 Flash

- **Artifact**: antirez's [deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)
  Flash 0731 IQ2XXS GGUF at `1cd7b564460821938add0475a60b942c409295e0` and the
  DSpark support GGUF at `e7f04037032990db0346398d249baf9fb9df1ccc`.
- **License**: MIT, as the GGUF metadata and DeepSeek's model card declare.
  rusty-gufo neither ships nor automatically downloads these weights.

### Qwen3-ASR and Qwen3-TTS

- **Artifact**: Qwen's [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
  at `7278e1e70fe206f11671096ffdd38061171dd6e5` and the Qwen3-TTS 12Hz 1.7B
  CustomVoice, VoiceDesign and Base checkpoints (revisions in the model page),
  BF16 safetensors.
- **License**: Apache-2.0, as the model cards declare. rusty-gufo neither
  ships nor automatically downloads these weights.

### EmbeddingGemma 2

- **Artifact**: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, with BF16 safetensors, a tokenizer, and a model configuration.  rusty-gufo reads the safetensors directly.
- **License**: Apache-2.0 as declared in the model card metadata.  rusty-gufo neither ships nor automatically downloads these weights.
- **Reference implementation**: Transformers and sentence-transformers, both Apache-2.0.  The Python reference dependencies are development-only.

### Gemma 4

- **Artifact**: Unsloth's GGUFs of Google's Gemma 4 26B-A4B, 31B and 31B QAT
  instruction-tuned models, with their `gemma4-assistant` MTP drafters and
  BF16 vision sidecars.
- **License**: Apache-2.0, Google's [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license),
  as the model cards declare. rusty-gufo neither ships nor automatically downloads
  these weights.

### Qwen3.6 35B-A3B

- **Artifact**: Unsloth's GGUFs of Qwen's Qwen3.6 35B-A3B (`unsloth/Qwen3.6-35B-A3B-GGUF`),
  with its MTP layer and BF16 vision sidecar, and the DFlash2 draft
  `aminya/Qwen3.6-35B-A3B-DFlash2-GGUF`, converted from `incoai/Qwen3.6-35B-A3B-DFlash2`.
- **License**: Apache-2.0, as the model cards declare. rusty-gufo neither ships nor
  automatically downloads these weights.

### Nemotron 3.5 Bolt

- **Artifact**: NVIDIA's `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` and
  vision projector
  ([nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF)
  at `e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246`) and the optional LiLiCorr
  drafter (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF` at
  `182d2987939e12f458298b5d5f6ef98912f4ad6c`).
- **License**: NVIDIA's model license as published with the files.
  rusty-gufo neither ships nor automatically downloads these weights.

### OmniVoice

- **Artifact**: [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
  at `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`, the Hugging Face directory with float32 safetensors, a tokenizer, the
  configuration and the Higgs Audio v2 audio tokenizer (`audio_tokenizer/`).
- **License**: the weights under CC-BY-NC-4.0 and the audio tokenizer under
  the Boson Higgs Audio 2 Community License, as their model cards declare.
  rusty-gufo neither ships nor automatically downloads these weights.
- **Reference implementation**: k2-fsa/OmniVoice's Python package
  (Apache-2.0) with transformers (Apache-2.0); development-only.

### Qwen-Image-2.1

- **Artifact**: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
  revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`; separate BF16
  safetensors, tokenizer and model configuration.
- **License**: [Qwen Research License Agreement, September 20, 2026](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE).
  The granted use is non-commercial; commercial use requires a separate license.
  rusty-gufo neither ships nor automatically downloads these weights.
- **Reference implementation**: Diffusers and Transformers, Apache-2.0.
  Native inference follows the [pinned operator contract](src/models/qwen_image_21/UPSTREAM.md);
  Python reference dependencies are development-only.
- **Publisher notice**: “Qwen is licensed under the Qwen RESEARCH LICENSE
  AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd.
  All Rights Reserved.”
