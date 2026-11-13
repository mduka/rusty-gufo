# VoxCPM2: checks

Tests (`GUFO_VOXCPM_MODEL` names the Hugging Face directory,
`GUFO_VOXCPM_GGUF` audio.cpp's `voxcpm2-bf16.gguf` and `GUFO_VOXCPM_Q8_GGUF`
its `voxcpm2-q8_0.gguf`; [testing](../../TESTING.md)):

```sh
cargo test -p gufo-voxcpm                                   # CPU: config, probe, tokenizer, weight readers
cargo test --profile gpu-test -p gufo-voxcpm --features gpu # the matrix-core products and weight tiles
GUFO_VOXCPM_MODEL=models/VoxCPM2 \
GUFO_VOXCPM_GGUF=models/VoxCPM2-GGUF/voxcpm2-bf16.gguf \
GUFO_VOXCPM_Q8_GGUF=models/VoxCPM2-GGUF/voxcpm2-q8_0.gguf \
  cargo test --profile gpu-test -p gufo-voxcpm --features gpu,models
```

The tests are those of the extension: the CPU tests check the
configuration, the probe, the tokenizer (21 reference cases) and the
safetensors, `torch.save` and GGUF readers; the products test compares the
bfloat16 and Q8_0 tiles with a double-precision sum of the same operands;
the model tests compare the generator and AudioVAE with tensors that the
extension's `tools/reference/generator_golden.py` recorded from OpenBMB's
implementation, within the extension's limits.

## The reference

The reference is the C++ extension
[gufo-extensions/voxcpm](https://github.com/gufo-extensions/voxcpm) at
commit `0941cd1`, built into Gufo 0.34.0 (`2845939b`) with ROCm 10.1 and run
on ROCm 10.1's runtime libraries.

`examples/synthesize.rs` takes the arguments of the extension's
`voxcpm_synthesize` and writes the same 16-bit WAV; with `--raw FILE` it
also writes the float32 samples, which a local patch makes the C++ tool
write as well (`GUFO_REF_F32=FILE`). The GPU tests give the extension's
numbers exactly (its quality report): for example the latents of the first
three patches 2.06e-3, 1.23e-2 and 8.64e-3 from the reference, AudioVAE
decoding 1.09e-6, and with `voxcpm2-q8_0.gguf` the first latent 1.67e-2.
Against the C++ build:

| Case | Files | Result |
| --- | --- | --- |
| Automatic voice (seed 42), buffered and streamed | Hugging Face directory, `voxcpm2-q8_0.gguf` | Identical float32 samples; buffered equals streamed in both builds |
| Automatic voice | `voxcpm2-bf16.gguf` | Identical |
| Voice design, German | Hugging Face directory | Identical |
| Cloning with and without the transcript | Hugging Face directory | One of the C++ build's two outputs per process (below) |
| Cloning | `voxcpm2-bf16.gguf`, `voxcpm2-q8_0.gguf` | Identical |
| A prompt of more than 128 rows | `voxcpm2-q8_0.gguf` | Identical |
| `gufo serve tts`: automatic voice, WAV and PCM stream | Hugging Face directory | The C++ tool's samples within one 16-bit step (the WAV writers round differently); stream equals WAV; a repeated request gives the same audio |

Prompts of more than 32 rows with bfloat16 weights run on hipBLASLt, whose
algorithm both builds choose by timing candidates, once per process. Such
prompts then give one of two outputs per process in the C++ build itself:
in repeated runs, cloning without the transcript gave the same two outputs
in both builds, and cloning with it gave rusty-gufo's output in one of
three C++ runs. Speed (`machine-lock perf`, three alternating rounds of
four streamed requests per process): first audio 82.2 against 82.1 ms and
2.154 against 2.153 s in all on the Hugging Face weights, 42.1 ms and
1.330 s in both builds on `voxcpm2-q8_0.gguf`.

## Not covered

`voxcpm2-orig.gguf` is not checked. The serving comparison covers the
automatic voice; the other modes ran over HTTP without a C++ comparison,
and Gufo's functional TTS suite (which expects 24 kHz) did not run. The C++
runner rejects `speaker_embedding_only`, the only form Gufo's API gives
cloning without a transcript; rusty-gufo serves it as that cloning.

The extension's quality report, its benchmarks against audio.cpp and its
reference scripts are in
[gufo-extensions/voxcpm](https://github.com/gufo-extensions/voxcpm/tree/0941cd1/docs/models/voxcpm2).
