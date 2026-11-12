# qwen35moe HIP kernels

Copied from the strycast-qwen35moe-gufo-extension (MIT, `LICENSE`) at
`4882b7a`: its `q35/` tree as far as the kernel targets of its
`extension.cmake` compile it (`strycast_q35_kernels`, the two `.hip` sources
of `strycast_q35_hip`). The extension's own copies are Strycast's Qwen
family kernels and Flash-Next's routed-expert kernels with llama.cpp's
quantized GEMM tier (`fnext/kernels/rocm/mmq`, see its `VENDOR.md`).

The paths mirror `q35/`. The copy changes only:

- include paths: `"q35/..."` to `"src/models/qwen35moe/kernels/vendor/..."`,
  the SDK headers `<gufo/ext/...>` to rusty-gufo's core shims
  (`src/core/gguf_reader.hpp`, `src/core/sampling.hpp`,
  `src/core/hip/hip_utils.hpp`, `src/core/hip/managed_alloc.hpp`,
  `src/core/hip/detail/{dispatch_telemetry,gemv_dispatcher}.hpp`,
  `src/core/quant/ggml_dequant.hpp`), and `gemv.hip`'s `"qfn_mmq.h"`
  (an include directory of the C++ build) to its full path;
- the namespace `strycast` to `gufo::models::qwen35moe` (`ns.hpp` keeps the
  extension's mirror of the gufo namespaces, without its host
  `model_config.hpp`); the MMQ tier keeps `q35_mmq` and its `q35_` C names;
- host code only, each change marked `rusty-gufo`:
  - `family/hip_common/gemm_route.hpp`: the GGML types rusty-gufo's
    `core::GgmlType` does not name (IQ2_XS, IQ3_XXS, IQ1_S, IQ2_S, IQ1_M)
    by value;
  - `fnext/mtp_sampling.hpp`: the host sampler (Gufo's `SamplerState`)
    removed, the kernels' constants kept;
  - `family/kernels/sample.hip`, `ops/token.hpp`: resident penalties and
    draft candidates (null host pointers skip the upload), the dense Qwen
    port's `LaunchGPUSampling*Resident` and `GpuSamplingSortStorageBytes`;
  - `fnext/kernels/rocm/mmq/qfn_ggml_stubs.hip.cpp`, `qfn_mmq.h`: caller
    scratch for the tier's allocations (`q35_mmq_set_scratch`), as in the
    Flash-Next crate.

Line numbers of the kernel sources are unchanged, and the device code is
the C++ build's (Gufo 0fb50ce4, ROCm 10.1): every function of every code
object matches after the namespace rename, except llama.cpp's
`no_device_code` abort stub, whose message length is its source path's.
`hipblas_gemm.hip` and `hipblaslt_gemm.hip` hold no device code and are not
copied; gufo-hip's `Blas` and `BlasLt` replace them (`src/lib.rs`, `blas`).

The format check skips this directory (a `vendor` path component); the
hand-written adapters next to it are formatted.
