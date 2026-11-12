# Qwen3-TTS 12Hz 1.7B: checks

`tests/parity.rs` holds the codec codes, samples and streamed chunks of
every variant (CustomVoice, VoiceDesign, Base) to the fixture
`tests/fixtures/qwen3-tts-parity.json`, which `record_fixture` wrote from
this engine on ROCm 10.1. The CustomVoice and VoiceDesign audio equals that
of Gufo v0.28.1's C++ build on the same ROCm in the
[A/B comparison](../../PARITY.md#performance-against-gufo); Base was not
part of it. The official
reference tooling stays in [`tests/models/qwen3_tts`](../../../../tests/models/qwen3_tts)
with the [intelligibility tool](../../../../tests/models/qwen3_tts/quality/README.md).
Base requires a reference WAV and its complete matching transcript.

```sh
cargo test --profile gpu-test -p gufo-qwen3-tts --features gpu,models
QWEN3_TTS_REFERENCE_ROOT=/path/to/Qwen3-TTS \
QWEN3_TTS_MODEL_ROOT=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  tools/audio/run_ref_tts.sh greedy
```

Use fixed-code waveform comparisons to isolate decoder error and natural-EOS
sampled speech for intelligibility. Run affected checks; do not relax thresholds.
