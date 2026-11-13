# Nemotron 3.5 ASR Streaming 0.6B

[Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
is NVIDIA's multilingual streaming speech recognition model (40 languages,
punctuation and capitalization). The encoder turns 16 kHz audio into
128-band log-mel features, subsamples them eight times to one frame per
80 ms and runs 24 cache-aware FastConformer layers whose attention sees 56
frames back and a fixed lookahead of 0, 3, 6 or 13 frames ahead. The audio
is encoded in chunks of lookahead + 1 frames; each chunk reuses the cached
attention keys and convolution states of the ones before it, so a chunk is
never computed twice. A one-hot language prompt (a locale or automatic
detection) is projected into the encoder output, and an RNN-T prediction
network (two LSTM layers) and joint network emit the text greedily, up to
ten tokens per frame.

rusty-gufo runs it with the HIP kernels of the
[Nemotron ASR extension for Gufo](https://github.com/gufo-extensions/nemotron_asr)
(C++, commit `c256b69`, version 0.1.2) and its own host code
([verification](#verification)). Every matrix product, the attention's
included, runs on the matrix cores with float16 operands and float32 sums;
the subsampling's and the prompt projector's activations are float16 too
(saturated at 65504), while the encoder's activations, norms, caches and
the LSTM state stay float32 (1.5 GB of GPU memory in all).

Speed: [performance](../../PERFORMANCE.md#nemotron-35-asr) · Quality: [the extension's
quality report](https://github.com/gufo-extensions/nemotron_asr/blob/c256b69/docs/models/nemotron-3.5-asr/QUALITY.md).

## Model

| Model ID | Files | Inputs |
| --- | --- | --- |
| `nemotron-3.5-asr-streaming-0.6b` | The Hugging Face directory [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/ea30d66debe3740a08b573244286791d423d6b3e) | WAV uploads, PCM from the realtime WebSocket |

| File | Repository | Size | Checked |
| --- | --- | --- | --- |
| `config.json`, `processor_config.json`, `tokenizer.json`, `model.safetensors` (float32) | [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/ea30d66debe3740a08b573244286791d423d6b3e) | 2.38 GiB | Yes |
| `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`, the `.nemo` archive | the same | | Not supported |

The weights aren't part of this repository. NVIDIA publishes them under the
[OpenMDW-1.1](https://openmdw.ai/license/1-1/) license.

## Use

```sh
hf download nvidia/nemotron-3.5-asr-streaming-0.6b \
  --revision ea30d66debe3740a08b573244286791d423d6b3e \
  config.json processor_config.json tokenizer.json model.safetensors \
  --local-dir models/nemotron-3.5-asr-streaming-0.6b
./target/release/gufo serve asr --model models/nemotron-3.5-asr-streaming-0.6b
curl -s localhost:8080/v1/audio/transcriptions -F file=@talk.wav -F language=de
```

`gufo serve asr` recognizes the directory by its `config.json` and
`processor_config.json`; the same command serves Qwen3-ASR. `gufo
transcribe` runs Qwen3-ASR only.

- `language` takes a code (`de`), a locale (`de-DE`), an English name
  (`German`) or nothing for automatic detection; other values are rejected.
  `verbose_json` reports the detected language.
- Uploads are WAV files with 16-, 24- or 32-bit integer PCM or 32-bit float
  samples, at any sample rate and with up to eight channels; the server
  mixes them to mono and resamples them to 16 kHz as Gufo does. They are
  transcribed at the widest lookahead (1.12 s), the most accurate, in passes
  of up to 16 chunks (18 s of audio). A request may hold up to four hours of
  audio, but the server accepts request bodies up to 8 MiB by default, about
  4.4 minutes of 16 kHz mono 16-bit WAV; raise `--max-request-bytes` for
  longer recordings.
- `stream=true` sends the text pass by pass, in passes of about 4.5 s of
  audio.
- `prompt` is ignored: the model has no text context. `max_tokens` has no
  effect, and `temperature` must be 0 (decoding is greedy).
- The end of an utterance is transcribed as the reference's offline
  transcription ends it: the last chunk holds only the frames that exist, so
  the last sentence keeps its punctuation.
- The realtime WebSocket (`GET /v1/realtime?intent=transcription`,
  [Qwen3-ASR's streaming examples](../qwen3-asr/README.md#streaming)) works
  as for Qwen3-ASR: each committed utterance is transcribed after the
  commit, its text streamed pass by pass.

## Known limits

- **Formats:** FLAC, MP3 and Ogg uploads are refused, as are 8-bit,
  A-law, µ-law and ADPCM WAV files. The extension reads them through Gufo
  0.34.0's audio decoder, which needs the symphonia crate; rusty-gufo reads
  WAV through Gufo's strict WAV reader, which also clamps float samples to
  [-1, 1].
- **No live transcription:** rusty-gufo transcribes a realtime session's
  utterance after its commit, at lookahead 13. Gufo 0.34.0 transcribes
  English and automatic-detection sessions live while audio arrives, at
  lookahead 6 (text every 0.56 s). Realtime audio arrives at 24 kHz and is
  resampled by the server's windowed-sinc resampler (Qwen3-ASR's front end),
  not Gufo's; each committed buffer holds at most 120 seconds.
- Words have no timestamps.
- Encoder passes of concurrent requests take turns on the GPU in arrival
  order.
- Tested on Strix Halo (`gfx1151`) only.

## Verification

The token IDs, the log-mel features and every checked activation equal
those of the extension's C++ build bit for bit: the features of both test
clips, the encoder's chunks, the prediction and joint networks, and 16
offline transcriptions (two clips, the clip's language and automatic
detection, four lookaheads). Over HTTP, rusty-gufo and Gufo 0.34.0 with the
extension answer WAV uploads identically, buffered and streamed, also for a
48 kHz stereo recording. Against transformers in float32, the tests keep the
extension's limits and its 16/16 identical transcriptions. The speed matches
the C++ build's. The checks are listed in
[Nemotron 3.5 ASR checks](../../development/models/nemotron-3.5-asr/CHECKS.md).
