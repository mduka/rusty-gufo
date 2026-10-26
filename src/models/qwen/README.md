# Qwen language models: shared parts

What the Qwen language models share across their architectures: the chat
template, the vision encoder with its image prompts, and the kernel crate over
upstream's HIP sources in this directory. The architectures live in their own
crates: [`qwen35`](../qwen35/README.md) (Qwen3.8 27B, dense Qwen3.5) and
[Flash-Next](../qwen38_flash_next/) (`qwen4exp`). Audio and image models named
Qwen are model-private.

[27B usage](../../../docs/models/qwen3.8-27b/README.md) ·
[Flash-Next](../../../docs/models/qwen3.8-flash-next/README.md)

## Source ownership

| Path | Responsibility |
|---|---|
| [`src/chat_template.rs`](src/chat_template.rs) | The Qwen chat template: the pinned Qwen3.8 reasoning template, the ChatML fallback, rendering and tokenizing. |
| [`src/vision/`](src/vision/) | The vision encoder, image prompts, the interleaved mRoPE layout, resizing and the per-session image input. |
| [`kernels/`](kernels/) | The kernel crate: [`launch.toml`](kernels/launch.toml) declares every launcher over [`hip/`](hip/), [`vision/`](vision/) and the adapters. |
| [`hip/`](hip/), [`gemm_route.hpp`](gemm_route.hpp), [`vision/`](vision/) | Upstream's HIP kernels and launchers for sampling, vision and the `qwen35` layers, and the GEMM route resolver they compile with. |
| [`tests/`](tests/) | Chat template tests; [`tests/reference/`](tests/reference/) holds the C++ vision dump program. |
| [`reference/`](reference/) | The pinned upstream chat template and its source record. |

Pinned upstream source/license records remain beside imported kernels and under
`reference/`.
