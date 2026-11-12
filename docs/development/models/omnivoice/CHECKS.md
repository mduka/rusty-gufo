# OmniVoice: checks

Tests (`GUFO_OMNIVOICE_MODEL` names the model directory; [testing](../../TESTING.md)):

```sh
cargo test -p gufo-omnivoice                                   # CPU: config, tokenizer, text, prompt, audio, decoding
GUFO_OMNIVOICE_MODEL=models/OmniVoice \
  cargo test --profile gpu-test -p gufo-omnivoice --features gpu,models
```

The CPU tests compare the configuration, tokenizer, prompt building,
waveform helpers and decoding schedule with fixtures from the reference
implementation (`tools/reference/`), as the extension's tests do. The GPU
tests compare the generator and the audio tokenizer with the reference model
and cover synthesis end to end; they give the extension's numbers exactly:

| Stage | Check | Result |
| --- | --- | --- |
| Generator, one CFG pass | Argmax of the guided target logits, 2 x 8 x 40 frame-codebooks | conditional 311/320, unconditional 316/320 |
| Generator, scoring | Guided predictions of 320 frame-codebooks | 314/320; confidences within 0.02 |
| Generator, greedy generation | First step where three generations leave the float32 reference | 11, 20 and 22 of 32 |
| Audio tokenizer, decode | 65 frames to 62,400 samples | samples within 1.0e-6 |
| Audio tokenizer, encode | 3 s synthetic signal to 8 x 75 codes | 600/600 identical |

Against the C++ extension, `examples/synthesize.rs` writes what the
extension's `omnivoice_synthesize` writes: a 16-bit WAV and, with
`GUFO_REF_CODES=FILE`, the codes. Seven requests (automatic voice with two
seeds, greedy, German, voice design, cloning and a 66 s story in five chunks,
13,096 codes) give identical codes in both builds, and each build repeats its
own output exactly. The 16-bit audio differs by one step in a few samples (at
most 117 of 1.6 million, [experiments](EXPERIMENTS.md)). Speed is not
compared yet: these were single cold runs, whose first request includes the
GEMM tuning (0.1–0.6 s); a clean A/B measurement is open. The extension's quality report, its
benchmarks against audio.cpp and its reference scripts are in
[gufo-extensions/omnivoice](https://github.com/gufo-extensions/omnivoice/tree/4bf376c/docs/models/omnivoice).
