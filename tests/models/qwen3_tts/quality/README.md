# Qwen3-TTS long-form quality input

The canonical exactness contract uses the short sentence in
`reference/generate_artifacts.py`. Long-form listening tests read
[`long-form.txt`](long-form.txt): the first five paragraphs of Lewis
Carroll's *Alice's Adventures in Wonderland* (1865, public domain), taken
from Project Gutenberg eBook #11 (`pg11.txt` of June 26, 2025, SHA-256
`01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`) with
the wrapped lines joined and the italics underscores removed.

Run from the repository root with the official Qwen3-TTS checkout and the
CustomVoice checkpoint:

```sh
QWEN3_TTS_REFERENCE_ROOT=/path/to/Qwen3-TTS \
QWEN3_TTS_MODEL_ROOT=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  tools/audio/run_ref_tts_quality.sh
```

The discrete codec tokens and waveform are written below
`artifacts/qwen3_tts/quality/`, which is gitignored. They are a
listening/continuity fixture, not the exact PR contract.
The quality route keeps upstream sampling enabled; fully greedy generation is
reserved for the short bounded exactness artifact.

For sampled implementations with a different RNG stream, waveform identity is
not a meaningful quality gate. Compare intelligibility using the same native
Qwen3-ASR-1.7B runtime for the upstream and native waveforms:

```sh
python3 \
  tests/models/qwen3_tts/quality/compare_intelligibility.py \
  --reference-npy artifacts/qwen3_tts/rocm/waveform.npy \
  --candidate-wav /tmp/qwen3-tts-native.wav \
  --asr-model models/Qwen3-ASR-1.7B \
  --gufo target/release/gufo
```

For native optimization A/B tests, pass the retained baseline directly with
`--reference-wav` instead of `--reference-npy`.

The report checks word error rate against the requested text, transcript
agreement, duration, finite samples, peak, and RMS. It complements—not
replaces—the exact parity gate of `gufo-qwen3-tts` (`tests/parity.rs`), which
requires the C++ engine's codec codes and samples.
