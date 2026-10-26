# HIP quantized matrix kernels

These model-private kernels derive from llama.cpp commit
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT, The ggml authors),
originally under `ggml/src/ggml-cuda/`. The DS4 adapter also carries changes
from `xangel82/DS4-GB10-GX10-DSpark-CUDA`, commit `910501e`.

Gufo builds this directory with HIP for AMD `gfx1151`. Device headers use
`.hip.hpp`; translation units use `.hip.cpp`. The adapter calls HIP/hipBLAS
directly and uses `ds4_ggml_hip_*` symbols to remain separate from Qwen's
matrix backend. There is no CUDA build or CUDA-to-HIP API alias layer.

`ds4_mmq.h` exposes the raw-pointer C ABI. `ds4_ggml_stubs.h` supplies the
small ggml type and allocation interface needed by the templates. Quantization
layouts, tables and active AMD arithmetic remain from the imported kernels.
Unused NVIDIA/MUSA preprocessing paths, ggml graph support and full ggml
tensor entry points are removed; DS4 owns its HIP graph execution separately.

When importing upstream changes, retain the HIP implementation and rerun the
focused DS4 projection/attention checks
(`tests/models/deepseek_v4_flash/run_operator_checks.sh`). The backend and all six
matrix translation units were also compared before and after preprocessing
for `gfx1151`; only private symbol names and unsupported-kernel diagnostics
changed. Current measurements belong in the model benchmark README.

Gufo changes after the import: the persistent dense and pair-map scratches
in `ds4_mmq.hip.cpp` serve every stream, not only the legacy default stream,
since DS4 orders all its MMQ calls on the one stream of its runtime
(`ds4_gpu_set_stream`).
