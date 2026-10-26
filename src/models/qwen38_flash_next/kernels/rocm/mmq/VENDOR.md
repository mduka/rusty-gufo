# Quantized HIP kernels

Adapted from `ggml-org/llama.cpp` at
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT; see `LICENSE`).
The upstream source directory is `ggml/src/ggml-cuda`; Gufo builds these
sources with HIP for AMD Strix Halo (`gfx1151`).

`qfn_mmq.h` is the public adapter. It supplies Q8_0 dense projections and
Q4_K, Q5_K, Q5_1 and Q8_0 routed expert projections, including the paired
Q4_K gate/up path. `qfn_ggml_stubs.*` supplies the minimal context and scratch
pool without the ggml graph runtime. Gufo owns scheduling and HIP graphs.

The kernels use native HIP runtime, hipBLAS and HIP numeric types. The
upstream runtime aliases, NVIDIA capability dispatch and PTX branches have
been removed; retained matrix fragments use AMD WMMA/MFMA. Headers use the
`.hpp` suffix. Gufo owns scheduling and HIP graphs.

When importing an upstream fix, preserve the Gufo namespace, adapter and
HIP-only dispatch, update the pin, and run the kernel crate's tests
(`cargo test --profile gpu-test -p gufo-qwen38-flash-next-kernels --features
gufo-hip/gpu,gufo-qwen38-flash-next-kernels/gpu`) plus the model's forward,
session and MTP parity tests. The C++ operator
tests that compared every routed format against a floating-point reference
went with the C++ engine; an upstream v0.23.0 checkout keeps them.
