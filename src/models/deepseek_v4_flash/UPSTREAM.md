# DeepSeek V4 Flash Upstream

The model-private DeepSeek V4 Flash engine is adapted from DS4 commit
`84cc882352757baf628a1776badf7cc54d584e28`.

Only the model graph, GGUF/tokenizer support, request-session state, and ROCm
kernel dependency closure are retained. DS4 command-line, HTTP server, agent,
evaluation, and disk-cache frontends are intentionally excluded so Gufo owns
those product surfaces.

The imported backend was converted to native ROCm/HIP source names and APIs.
It does not call Qwen kernels or share Qwen tensor layouts, dispatch policies,
or mutable state.

## Integration Boundary

The host side is the Rust crate `gufo-deepseek-v4-flash` (`src/`): GGUF and
model data, tokenizer, chat template, weights, the ROCm graph executor,
request sessions, sampling, DSpark, snapshots and the serving runner, ported
from DS4's C++ runtime bit for bit (docs/RUST.md). The C++ runtime and its
`engine.hpp` API are gone; the parity fixtures were recorded against them
(`tests/reference/ds4_ref_dump.cpp`, built against an upstream v0.23.0
checkout).

- `kernels/rocm` owns every DeepSeek numerical kernel and ROCm dispatch rule,
  behind the C ABI of `resident_api.h`, which the kernel crate
  `gufo-deepseek-v4-flash-kernels` binds.
- `src` owns the host code and calls the kernels only through that ABI.

The model package intentionally duplicates numerical code instead of calling
Qwen kernels. A DeepSeek kernel change must not alter another model's output or
performance.

The standalone distributed, tensor-parallel, SSD weight streaming, multi-GPU
placement, CPU reference graph, MTP, steering, and DS4 frontend
implementations are not built or shipped. The production runtime has one
single-device resident ROCm route and one authoritative state layout.

## Update Policy

When updating the imported engine:

1. Record the new immutable upstream commit.
2. Import only code exercised by the single-device ROCm route for the
   supported artifact.
3. Preserve the ROCm-only source and naming contract.
4. Reapply model-private changes inside this directory, never in Qwen code;
   host changes go into the Rust crate.
5. Run the DeepSeek model tests (`tools/ds4/check.py`), the fast hardware
   suite, and the canonical gate before recording new performance.
