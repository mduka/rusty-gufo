# Nemotron 3.5 ASR: checks

Tests (`GUFO_NEMOTRON_ASR_MODEL` names the Hugging Face directory;
[testing](../../TESTING.md)):

```sh
cargo test -p gufo-nemotron-asr                                   # CPU: configuration, languages, transcripts, features
cargo test --profile gpu-test -p gufo-nemotron-asr-kernels --features gpu   # launcher extents
GUFO_NEMOTRON_ASR_MODEL=models/nemotron-3.5-asr-streaming-0.6b \
  cargo test --profile gpu-test -p gufo-nemotron-asr --features gpu,models
```

The tests are those of the extension, plus a bit-for-bit comparison with
its C++ build:

- `host`: the configuration parser, the language prompts and the transcript
  cleaning; with `models`, the released checkpoint's configuration.
- `features`: the log-mel front end against transformers' feature extractor
  (`golden.safetensors`), every frame of both clips, the audio pushed in
  pieces of 1 and 997 samples and whole, within 1e-4 of the range (5.9e-4
  of 16.6, as in the extension's report); every split gives the same bits.
- `tokenizer` (`models`): the 16 reference runs' token IDs to raw and clean
  transcripts.
- `model` (`gpu`, `models`): the encoder outputs of the first three chunks
  (subsampled and encoder output within 2e-3, projection within 1e-2: the
  subsampling's activations and every matrix are float16), the
  prediction network's start state (5e-4) and the joint's first logits
  (2e-3) against transformers in float32; the token IDs, transcripts, chunk
  and frame counts of all 16 reference runs (both clips, the clip's language
  and automatic detection, lookaheads 0, 3, 6 and 13) with the audio pushed
  in random pieces; and the same encoder outputs, logits and start state,
  and every run's token IDs and the prediction network's final output,
  against the C++ build bit for bit (each clip pushed at once).
- The kernel crate's extents test probes every launcher of `launch.toml`.

## The reference

The reference is the C++ extension
[gufo-extensions/nemotron_asr](https://github.com/gufo-extensions/nemotron_asr)
at commit `c256b69` (version 0.1.2: float16 subsampling and projections,
matrix-core attention over groups of chunks, position reads kept inside the
table), built into Gufo 0.34.0 (`2845939b`) with ROCm 10.1
(the `gpu-test` preset) and run on ROCm 10.1's runtime libraries. Two
development tools on top of it write its outputs as raw float32: one for the
decoded samples, the mel filters and the features of both clips, one for
the model's outputs, which
`src/models/nemotron_asr/tools/reference/pack_cxx_reference.py` packs into
`tests/fixtures/cxx_reference.safetensors`. Two runs of the C++ tool in
separate processes gave the same bytes.

| Case | Result |
| --- | --- |
| Decoded samples, mel filters, features of both clips (pushed in pieces of 997 samples, then ended) | Identical bytes |
| Encoder: subsampled, encoder and projected outputs of three chunks, both clips, lookahead 13 | Identical |
| Prediction network's start state, joint logits of the first frame | Identical |
| 16 transcriptions (both clips, two language settings, four lookaheads): token IDs and the prediction network's final output | Identical |
| `gufo serve asr` against Gufo 0.34.0's: `english.wav` (JSON), `german.wav` with `language=de` (`verbose_json`) and `language=German` (text), `english.wav` streamed, a 48 kHz stereo copy of `english.wav` (mixing and resampling), an unknown language | Identical responses, streamed deltas and error |

Speed: the extension's development driver (`nemotron_asr_transcribe`)
against `examples/transcribe.rs` built with `--release`, the same model,
alternating rounds under `machine-lock perf` (Balanced profile), medians of
the repeated runs:

| Audio | C++ | Rust |
| --- | ---: | ---: |
| `english.wav`, 10.08 s | 22.5 ms | 22.1 ms |
| The same at lookahead 0 (127 chunks of one frame) | 22.2 ms | 21.8 ms |
| 308.7 s recording (the two clips with 1 s pauses, repeated) | 0.519 s | 0.510 s |
| The same, pushed in pieces of 560 ms | 3.03 s | 3.02 s |

The encoder takes the same time on both; the decoder is 2 to 6 % faster in
Rust and the features 5 to 8 % faster at the same position in a round
(the CPU's clock falls during a round in both builds, which blurs the
medians of the features). The C++ side is the `gpu-test` build
(RelWithDebInfo); its kernels carry the extension's `-O3`, as the Rust
build's do. Over HTTP (version 0.1.1), with both servers up and the
requests alternating, the 10 s clip took 24.2 ms in C++ and 23.4 ms in
Rust and the 308.7 s recording the same (median ratio of the pairs 1.002);
separate rounds of ten requests each drift by up to 15 % as the machine
warms.

Version 0.1.1 was first published as `0823ee4`, which this port followed
at first, then amended to `d3c4e4c`: attention blocks take as many whole
chunks as fit in 16 queries (lookaheads 0, 3 and 6), chunks may exceed 16
frames, float16 activations saturate, and the prompt projector's second
product writes float16 directly. At lookaheads 0 and 3 its groups of 16
queries read up to two rows before and after the position table (sized
for chunks of 14 frames); the C++ model tool faulted on it while
`model_test` passed. Version 0.1.2 (`c256b69`) clamps those reads, whose
terms are never used. Every output compared here is the same at `0823ee4`
and `c256b69`, and the 10 s clip and the recording took the same time.
The extents test probes the grouped chunks and chunks of 28 frames.

Before version 0.1.1 (`d54c7da`: float32 subsampling and projections through
`hipblasSgemm`, attention on the vector units) the same comparison gave
25.0 against 24.6 ms and 0.630 against 0.625 s, bit-identical as well.

```sh
cargo run --release -p gufo-nemotron-asr --example transcribe -- \
  MODEL AUDIO [--language L] [--lookahead N] [--repeat N] [--push MS]
```

## Not covered

Live transcription is not ported (rusty-gufo has no live transcription API;
realtime sessions are transcribed after the commit), so the extension's
live runner checks and its streaming resampler are not compared. FLAC, MP3
and Ogg uploads are refused rather than decoded. Concurrent requests were
not compared beyond the FIFO gate between their encoder passes. Gufo's
functional ASR suite did not run against rusty-gufo.

The extension's quality report (word error rates, the float32 comparison),
its benchmarks against audio.cpp and its reference script are in
[gufo-extensions/nemotron_asr](https://github.com/gufo-extensions/nemotron_asr/tree/c256b69/docs/models/nemotron-3.5-asr).
