# VoxCPM2

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) is OpenBMB's tokenizer-free
multilingual text-to-speech model (2B parameters, 30 languages, 48 kHz
output). It generates speech as continuous latents, one patch of four
latent frames (160 ms of audio) at a time: a 28-layer MiniCPM4 language
model reads the text and the patches so far, an 8-layer residual language
model adds acoustic detail, a 12-layer local diffusion transformer turns
both into the next patch in 10 Euler steps with classifier-free guidance,
and a 12-layer local encoder feeds the patch back. AudioVAE V2 decodes the
latents to 48 kHz audio and encodes reference recordings at 16 kHz.

rusty-gufo runs it with the HIP kernels of the
[VoxCPM extension for Gufo](https://github.com/gufo-extensions/voxcpm)
(C++, commit `0941cd1`) and its own host code
([verification](#verification)).

Speed: [performance](../../PERFORMANCE.md#voxcpm2) · Quality: [the
extension's quality report](https://github.com/gufo-extensions/voxcpm/blob/0941cd1/docs/models/voxcpm2/QUALITY.md).

## Model

| Model ID | Files | Conditioning |
| --- | --- | --- |
| `voxcpm2` | The Hugging Face directory [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2/tree/7936080b41163874f38610f46f438e2c0757f9fd), or one of audio.cpp's VoxCPM2 GGUFs | Automatic voice, voice design and style (`instructions`), cloning (reference audio with or without its transcript, or a `--voice` preset) |

| File | Repository | Size | Checked |
| --- | --- | --- | --- |
| `config.json`, `model.safetensors` (bfloat16), `audiovae.pth` (float32), `tokenizer.json` | [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2/tree/7936080b41163874f38610f46f438e2c0757f9fd) | 4.62 GiB | Yes |
| `voxcpm2-bf16.gguf` | [`audio-cpp/audio.cpp-gguf`](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/73ce4162f6cce370dd533c6782b772135c1b2a8e/VoxCPM2-GGUF) | 4.44 GiB | Yes |
| `voxcpm2-q8_0.gguf` | the same | 2.75 GiB | Yes |
| `voxcpm2-orig.gguf` | the same | 4.62 GiB | No |

The directory loads `audiovae.pth` without PyTorch. A GGUF embeds the
configuration and the tokenizer; its Q8_0 matrices stay 8-bit on the GPU
(the products convert the values exactly and keep the activations in
bfloat16), so the quantized file saves memory and generation time. Float16
and float32 tensors, and Q8_0 vectors and embedding tables, are converted to
bfloat16 on load. Other community GGUF conversions of VoxCPM2 use their own
layouts and are not supported. The weights aren't part of this repository:
OpenBMB publishes them under the Apache License 2.0; read the terms of
audio.cpp's repository before you download its conversions.

## Use

```sh
hf download openbmb/VoxCPM2 \
  --revision 7936080b41163874f38610f46f438e2c0757f9fd \
  config.json model.safetensors audiovae.pth tokenizer.json \
  --local-dir models/VoxCPM2
./target/release/gufo serve tts --model models/VoxCPM2 \
  --voice anna=/voices/anna.wav --voice-text anna=/voices/anna.txt

# or one of audio.cpp's GGUFs
hf download audio-cpp/audio.cpp-gguf \
  --revision 73ce4162f6cce370dd533c6782b772135c1b2a8e \
  VoxCPM2-GGUF/voxcpm2-q8_0.gguf --local-dir models
./target/release/gufo serve tts --model models/VoxCPM2-GGUF/voxcpm2-q8_0.gguf
```

`gufo serve tts` recognizes the directory by its `config.json` and a GGUF
file by its header; the same command serves Qwen3-TTS and OmniVoice.

- `GET /v1/audio/voices` lists `auto` (the model picks a voice),
  `voice-design` and every preset. Requests without `voice` use `auto`.
- **Voice design and style:** `instructions` describes the voice or the
  delivery in natural language ("A young woman, gentle and sweet voice",
  "slightly faster, cheerful tone"); it is put in parentheses before the
  text, as the reference implementation does.
- **Cloning:** a preset (`--voice NAME=WAV`) or `reference_audio` (base64
  WAV). With its transcript (`--voice-text`, a `.txt` beside the WAV, or
  `reference_text`) the recording conditions the voice and generation
  continues it, which the reference calls ultimate cloning. Without one (a
  preset without transcript, or `voice_clone_mode: speaker_embedding_only`)
  the recording conditions the voice in isolation. Any sample rate and
  channel count works: the recording is mixed down and resampled to 16 kHz.
  A reference is encoded once and cached (eight references).
- `seed` selects the diffusion noise; the same seed reproduces the audio
  exactly within one server process. `max_new_tokens` caps the patches (6.25 per second). Like the
  reference, generation also stops at 6 patches per text token plus 10, and
  a buffered request that reaches that cap is generated again with the next
  seed, up to three attempts. `language`, `greedy` and the sampling fields
  don't apply to VoxCPM2.
- Streams (`response_format: pcm`, SSE) send every 160 ms patch as soon as
  it is decoded.
- Output is 48 kHz mono.

The server checks requests by what the model declares it supports (an
unconditioned voice, optional reference audio and presets, optional
instructions) ([audio API](../../SERVER.md#other-model-services)).

## Known limits

- One request runs at a time; the server queues the others.
- The decoding parameters are the reference's defaults (10 Euler steps,
  guidance 2.0); the API has no fields for them.
- `gufo serve tts` asks the HIP runtime for one hardware queue unless
  `GPU_MAX_HW_QUEUES` is set
  ([hardware compute queues](../../SERVER.md#hardware-compute-queues)).
  With `GPU_MAX_HW_QUEUES=2`, AudioVAE decodes a patch alongside the
  generation of the next one; the extension measured that 3–4 % faster on
  Gufo, and it is not measured on rusty-gufo yet.
- Prompts of more than 128 rows (text tokens and reference patches) run on
  hipBLASLt, which reads bfloat16: with a Q8_0 file, each matrix of their
  prefill is first expanded to bfloat16, so their prefill computes with
  slightly different weights than generation.
- Not implemented: the reference's optional text normalization and
  denoiser, LoRA adapters, VoxCPM 0.5B and 1.5.
- The tokenizer follows `tokenizer.json` as transformers 4 applies it;
  transformers 5 tokenizes text that starts with a space differently.
- The extension's runner rejects `voice_clone_mode: speaker_embedding_only`;
  rusty-gufo clones from the recording alone instead, as the extension's
  development driver does without a transcript.

## Responsible use

Voice cloning produces realistic speech in another person's voice. Clone
only voices you are allowed to use, and don't use VoxCPM2 for
impersonation, fraud or disinformation; OpenBMB forbids those uses of VoxCPM
and recommends marking generated audio as such
([risks and limitations](https://github.com/OpenBMB/VoxCPM/blob/300e4260b509066151d623dc9c0aa922cf82232b/README.md#%EF%B8%8F-risks-and-limitations)).
rusty-gufo adds no watermark or other marking to the audio; laws on
synthetic media, such as labeling duties, may apply where you publish it.

## Verification

The audio samples equal those of the extension's C++ build bit for bit,
buffered and streamed, for the automatic voice, voice design, other
languages and cloning, on the Hugging Face weights and both checked GGUFs,
at the same speed. Prompts of more than 32 rows with bfloat16 weights, such
as most cloning prompts, run on hipBLASLt, whose algorithm both builds pick
by timing once per process: each build then gives one of two outputs per
process, and rusty-gufo's are among the C++ build's. Within a process, a
seed reproduces the audio exactly.

The checks against the extension's C++ build are listed in
[VoxCPM2 checks](../../development/models/voxcpm2/CHECKS.md).
