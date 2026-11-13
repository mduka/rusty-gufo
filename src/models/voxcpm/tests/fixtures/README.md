# VoxCPM2 test fixtures

Copied from the C++ VoxCPM extension
([gufo-extensions/voxcpm](https://github.com/gufo-extensions/voxcpm), `0941cd1`,
`tests/fixtures`); the scripts that recorded them are in
[`tools/reference`](../../tools/reference).

| File | Recorded by | Contents |
| --- | --- | --- |
| `tokenizer_golden.json` | `tokenizer_golden.py` | Token IDs of 21 texts from openbmb/VoxCPM2's `tokenizer.json` (the `tokenizers` library, transformers 4 behaviour) split by VoxCPM's `mask_multichar_chinese_tokens` |
| `sample.pth`, `view.pth` | `checkpoint_sample.py` | Small `torch.save` checkpoints: float32, bfloat16 and int32 tensors, views at offsets into a shared storage, a zero-dimensional tensor; `view.pth` a transposed (non-contiguous) tensor |
| `generator_golden.safetensors` | `generator_golden.py` | Tensors of OpenBMB's VoxCPM2Model (revision `300e426`) on the CPU in float32 with seeded diffusion noise: zero-shot and reference-mode prompts, one DiT evaluation, AudioVAE decoding and encoding |

The texts are the extension author's; the IDs and tensors derive from the
openbmb/VoxCPM2 checkpoint (Apache-2.0).
