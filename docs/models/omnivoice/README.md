# OmniVoice

[OmniVoice](https://github.com/k2-fsa/OmniVoice) is k2-fsa's massively
multilingual zero-shot text-to-speech model. It fills in the audio codes of
a whole utterance over 32 unmasking steps with classifier-free guidance. Its
backbone has Qwen3-0.6B's shape but attends in both directions over text and
audio positions. The Higgs Audio v2 tokenizer turns eight codebooks at 25
frames per second into 24 kHz audio and encodes reference voices for
cloning.

rusty-gufo runs it with the HIP kernels of the
[OmniVoice extension for Gufo](https://github.com/gufo-extensions/omnivoice)
(C++, commit `4bf376c`) and its own host code; its audio codes equal the
extension's ([verification](#verification)).

Speed: [performance](../../PERFORMANCE.md#omnivoice) · Quality:
[the extension's quality report](https://github.com/gufo-extensions/omnivoice/blob/4bf376c/docs/models/omnivoice/QUALITY.md).

## Model

| Model ID | Files | Conditioning |
| --- | --- | --- |
| `omnivoice` | The Hugging Face directory [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice): `config.json`, `model.safetensors`, `tokenizer.json` and `audio_tokenizer/` | Automatic voice, voice design (`instructions`), cloning (reference audio and transcript, or a `--voice` preset) |

The weights load in float32 and are converted to float16 for the
generator's GEMM operands; the residual stream, norms and the audio
tokenizer stay in float32. GGUF packages of OmniVoice (audio.cpp's Q8_0 file
and other community conversions) are not supported. The weights aren't
part of this repository: OmniVoice's weights are published under
CC-BY-NC-4.0 and its audio tokenizer under the Boson Higgs Audio 2 Community
License; read both before you download them.

## Use

```sh
hf download k2-fsa/OmniVoice \
  --revision c5fdb5ccb189668d56333f77ba2629f4cd7535f4 \
  --local-dir models/OmniVoice
./target/release/gufo serve tts --model models/OmniVoice \
  --voice anna=/voices/anna.wav --voice-text anna=/voices/anna.txt --voice-lang anna=de
```

`gufo serve tts` recognizes the directory by its `config.json`; the same
command serves Qwen3-TTS checkpoints.

- `GET /v1/audio/voices` lists `auto` (the model picks a voice),
  `voice-design` and every preset. Requests without `voice` use `auto`.
- **Voice design:** `instructions` takes OmniVoice's attribute items,
  separated by commas: gender (`male`, `female`), age (`child`, `teenager`,
  `young adult`, `middle-aged`, `elderly`), pitch (`very low pitch` …
  `very high pitch`), `whisper`, an English accent (`british accent`,
  `indian accent`, …) or, in Chinese, a dialect (`四川话`, …). Other text is
  rejected, as the reference implementation rejects it.
- **Cloning:** a preset (`--voice NAME=WAV` with its transcript) or
  `reference_audio` (base64 WAV) with `reference_text`. OmniVoice needs the
  transcript, so a preset needs `--voice-text` (or a `.txt` beside its WAV)
  and `voice_clone_mode: speaker_embedding_only` is rejected. A reference is
  encoded once and cached (eight references). End a reference with a short
  silence after its last word: the generated audio follows on from the
  reference, so a reference cut off in the middle of a sound starts every
  generation with the rest of that sound, a click before each sentence of a
  segmented stream.
- `language` takes a name or code (`German`, `de`, … 646 languages). The
  server sends `english` when a request names none, so give presets their
  language with `--voice-lang`.
- `seed` selects the unmasking noise; the same seed reproduces the audio
  exactly within one server process. `greedy: true` turns the noise off.
  `max_new_tokens` caps the frames (25 per second). Other sampling fields
  don't apply to OmniVoice.
- Texts estimated longer than 30 s are generated in chunks of about 15 s at
  punctuation and cross-faded, as the reference does.
- Streams (`response_format: pcm`, SSE) send the utterance's 0.1 s of leading
  silence at once and the rest in 0.5 s pieces when the utterance is
  generated.
- For text that arrives in pieces, such as an LLM's answer, the WebSocket
  `/v1/audio/speech/stream` with `split_granularity: sentence`
  ([protocol](../qwen3-tts/README.md#sampling-and-streaming)) generates and
  sends each sentence on its own, so speech starts after the first sentence
  ([measurements](../../development/models/omnivoice/EXPERIMENTS.md#sentence-streaming)). Every sentence is a
  generation of its own: `auto` picks a new voice for each one, so stream
  with a preset or voice design. Each sentence's level is set on its own.

The server checks requests by what the model declares it supports (an
unconditioned voice, optional reference audio and presets, optional
instructions) ([audio API](../../SERVER.md#other-model-services)).

## Known limits

- Only the Hugging Face directory loads; GGUF packages don't.
- One request runs at a time; the server queues the others.
- The decoding parameters are OmniVoice's defaults (32 steps, guidance 2.0,
  position temperature 5.0, greedy classes); the API has no fields for them,
  the speaking rate or a fixed duration.
- Free-text voice descriptions aren't understood, so the functional TTS
  suite fails its voice-design variant.
- No automatic transcription of references and no text normalization of
  numbers (both optional in the reference).
- Sampled output can't match the reference implementation's for the same
  seed: the noise comes from another generator.
- Invalid voice-design items reach the runtime and come back as HTTP 500, and
  `/v1/models` names the model `owned_by: operator-supplied-qwen`, as in Gufo.
- The server needs about 2.5 GiB of GPU memory.


## Verification

rusty-gufo produces the same audio codes as the C++ extension it comes from
(gufo-extensions/omnivoice `4bf376c` on Gufo `0e2de43e`) for seven requests
covering every mode; the 16-bit audio differs by one step in at most 117 of
1.6 million samples. The checks are listed in
[OmniVoice checks](../../development/models/omnivoice/CHECKS.md).
