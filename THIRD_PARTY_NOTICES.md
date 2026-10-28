# Third-party notices

Gufo's original source is MIT licensed; this does not relicense dependencies or
model weights. This inventory covers direct dependencies, adapted code and
acknowledged optimization references.
Transitive system/toolchain dependencies retain their upstream licenses.
`Cargo.lock` pins every Rust crate the binary compiles in; the ROCm
libraries come from the installed ROCm (10.1.0 is qualified). `flake.lock`
pins Nixpkgs revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` for the
development shell and the reference packages. Distributions must retain the
notices of the versions they actually ship.

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
| ureq / rustls / ring | Compiled in; HTTPS for evaluation and `image_url` | `MIT OR Apache-2.0` (ureq); `Apache-2.0 OR ISC OR MIT` (rustls); `Apache-2.0 AND ISC` (ring); `ISC` (rustls-webpki) | ureq 3.4.2, rustls 0.23.46, ring 0.17.14; Cargo.lock | [ureq](https://github.com/algesten/ureq), [rustls](https://github.com/rustls/rustls), [ring](https://github.com/briansmith/ring) |
| png / zune-jpeg / image-webp | Compiled in; image decoding and PNG encoding | `MIT OR Apache-2.0` (png, image-webp); `MIT OR Apache-2.0 OR Zlib` (zune-jpeg) | png 0.18.1, zune-jpeg 0.5.15, image-webp 0.2.4; Cargo.lock | [image-png](https://github.com/image-rs/image-png), [zune-image](https://github.com/etemesi254/zune-image), [image-webp](https://github.com/image-rs/image-webp) |
| unicode-normalization / unicode-properties | Compiled in; tokenizer normalization | `MIT OR Apache-2.0` | 0.1.25 / 0.1.4; Cargo.lock | [unicode-rs](https://github.com/unicode-rs) |
| sha2 / sha1_smol | Compiled in; artifact identities and the WebSocket handshake | `MIT OR Apache-2.0` (sha2); `BSD-3-Clause` (sha1_smol) | 0.10.9 / 1.0.1; Cargo.lock | [RustCrypto](https://github.com/RustCrypto/hashes), [sha1-smol](https://github.com/mitsuhiko/sha1-smol) |
| Other Rust crates | Compiled in: serde, serde_json, chrono, libc, rustix, signal-hook, toml, jobserver | `MIT OR Apache-2.0` (rustix also `Apache-2.0 WITH LLVM-exception`) | Cargo.lock | [crates.io](https://crates.io) |
| GNU C/C++/OpenMP runtimes | System runtime libraries; no Gufo source import | `LGPL-2.1-or-later AND (GPL-3.0-or-later WITH GCC-exception-3.1)` | glibc/GCC packages in flake.lock | [GNU](https://www.gnu.org/software/) |
| llama.cpp / ggml | Adapted quantization, attention and model-private HIP kernels | `MIT` | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`; attention `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| mikealanni/gufo | Adapted Flash-Next GDN row reuse and SSM stride optimization | `MIT` | `f450218743bcaf77e3e1f8fab78f4cd4e4aafae3` | [gufo fork](https://github.com/mikealanni/gufo/tree/f450218743bcaf77e3e1f8fab78f4cd4e4aafae3) |
| Qwen chat templates | Reference Jinja and adapted renderer behavior | `Apache-2.0` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (27B), `de4b8e4d43b917e7706784d8bb445c9af86a3540` (Flash-Next) | [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) |
| DS4 | Adapted loader, tokenizer, sessions and HIP kernels | `MIT` | `84cc882352757baf628a1776badf7cc54d584e28` | [antirez/ds4](https://github.com/antirez/ds4) |
| DS4 GB10/GX10 fork | Adapted paired MoE launch code, now HIP | `MIT` | `910501e` | [xangel82/DS4](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA) |

Upstream license texts are retained in `licenses/` and
[src/models/deepseek_v4_flash/LICENSE.ds4](src/models/deepseek_v4_flash/LICENSE.ds4).
Distribute these with `LICENSE`, `NOTICE` and this inventory alongside the
binary. Preserve embedded copyright notices when modifying adapted code. Model-private changes and import boundaries are recorded in the
[DS4 provenance](src/models/deepseek_v4_flash/UPSTREAM.md),
[DS4 HIP import](src/models/deepseek_v4_flash/kernels/rocm/mmq/VENDOR.md),
and [Flash-Next HIP import](src/models/qwen38_flash_next/kernels/rocm/mmq/VENDOR.md).

The mikealanni/gufo fork retains the same Copyright (c) 2026 gufo contributors
notice and MIT text as Gufo's root `LICENSE`.

The official 0731 DeepSeek continuations in
`tests/models/deepseek_v4_flash/fixtures/official-0731.json` come from DS4 revision
`6289c516273979173abbc062209a81dd3706b804`; the fixture retains source hashes and
its upstream MIT notice. External model files are never part of the binary package.

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
| ROCprofiler SDK / ROCTx | Development profiling and benchmark markers only | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) |
| PyTorch ROCm | Independent evaluation; not shipped | `BSD-3-Clause` | 2.12.0; flake.lock | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| Torchvision | Evaluation; not shipped | `BSD-3-Clause` | 0.27.0; flake.lock | [pytorch/vision](https://github.com/pytorch/vision) |
| Transformers / Accelerate / Safetensors / Hugging Face Hub | Evaluation, artifact inspection and model acquisition | `Apache-2.0` | flake.lock | [Hugging Face](https://github.com/huggingface) |
| NumPy / SciPy | Evaluation arrays and signal analysis | `BSD-3-Clause` | flake.lock | [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy) |
| Requests | Evaluation HTTP client | `Apache-2.0` | flake.lock | [Requests](https://github.com/psf/requests) |
| OpenAI Python SDK | Local API compatibility checks; not shipped | `Apache-2.0` | 2.41.1; flake.lock | [openai-python](https://github.com/openai/openai-python) |
| SoX | Development audio utility | `GPL-2.0-or-later` | flake.lock | [SoX](https://sourceforge.net/projects/sox/) |
| antirez/ds4 | Optional benchmark package; not shipped with Gufo | `MIT` | `0aaea5a238fb41a35106a551e73c8409dfb751ac` | [Nix recipe](.devops/nix/ds4-reference.nix) |
| llama.cpp reference builds | Optional benchmark packages; not shipped with Gufo | `MIT` | Release `68d9053a`; Flash-Next MTP `6fcaa16f` | [Nix recipes](.devops/nix/llama-cpp-reference.nix) |

Python/PyTorch and reference scripts are not installed with Gufo. Their own
transitive dependencies, including Triton and MIOpen, remain development-only. Kernel
microbenchmarks build separately (`tools/bench/build.sh`); ROCprofiler is supplied
by the development shell. Ordinary build tools (Cargo, rustc, hipcc and the host
compiler) retain their own licenses and are not Gufo code.

## External Model Artifacts (Non-Distributed)

Model checkpoints are not part of the gufo source or binary
distribution. Operators obtain them directly from their publisher and remain
responsible for the terms governing their location and use.

### EmbeddingGemma 2

- **Artifact**: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, with BF16 safetensors, a tokenizer, and a model configuration.  Gufo reads the text backbone only.
- **License**: Apache-2.0 as declared in the model card metadata.  Gufo neither ships nor automatically downloads these weights.
- **Reference implementation**: Transformers and sentence-transformers, both Apache-2.0.  The Python reference dependencies are development-only.

### Qwen-Image-2.1

- **Artifact**: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
  revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`; separate BF16
  safetensors, tokenizer and model configuration.
- **License**: [Qwen Research License Agreement, September 20, 2026](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE).
  The granted use is non-commercial; commercial use requires a separate license.
  Gufo neither ships nor automatically downloads these weights.
- **Reference implementation**: Diffusers and Transformers, Apache-2.0.
  Native inference follows the [pinned operator contract](src/models/qwen_image_21/UPSTREAM.md);
  Python reference dependencies are development-only.
- **Publisher notice**: “Qwen is licensed under the Qwen RESEARCH LICENSE
  AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd.
  All Rights Reserved.”
