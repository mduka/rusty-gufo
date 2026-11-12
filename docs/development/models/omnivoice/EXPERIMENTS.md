# OmniVoice experiments

What has been tried on this engine for OmniVoice. The extension's own
development notes (attention kernels, GEMM choices, audio.cpp comparisons)
are in its [quality report](https://github.com/gufo-extensions/omnivoice/blob/4bf376c/docs/models/omnivoice/QUALITY.md)
and [benchmarks](https://github.com/gufo-extensions/omnivoice/blob/4bf376c/docs/models/omnivoice/BENCHMARKS.md).

## Tried

- **Kernels at `-O2`.** The first build compiled the extension's kernels at
  the kernel crates' default `-O2`; the extension's CMake target compiles them
  at `-O3`, and so do they now. Against the C++ build (`examples/synthesize.rs`
  and the extension's `omnivoice_synthesize`, seven requests) the codes are
  identical at `-O3` in every case. A few 16-bit samples of the decoded audio
  (at most 117 of 1.6 million) still differ by one step, at `-O2` and `-O3`
  alike, while the decoder gives the C++ test's result on the reference codes:
  not located yet.

## Sentence streaming

The WebSocket `/v1/audio/speech/stream` with `split_granularity: sentence`,
fed delta by delta from Gemma 4 26B-A4B (MTP) beside OmniVoice under the GPU
share arbiter, in one run on an otherwise idle machine (German, seed 1):

| Case | Whole text, then speech | Sentence by sentence |
| --- | --- | --- |
| 8-sentence LLM answer (36–44 s of audio), first speech after the LLM request | 11.0 s (`auto`), 11.8 s (preset) | 1.4 s, 1.8 s |
| 6-sentence fixed text (16–22 s of audio), first speech | 2.6–5.0 s | 0.75–1.07 s |

- A sentence is generated about four times faster than it plays, so after
  the first one the next was always ready, except for one 0.33 s gap after a
  0.8 s first sentence. All sentences together take about 1.6 times as long
  as the whole text at once, as every generation runs its 32 steps.
- The arbiter held the LLM at a quarter of the GPU until the first sentence
  arrived and gave it all back once the playback buffer filled; the LLM's
  answer took 4.3–5.3 s instead of 2.3 s, still far ahead of the speech.
- `auto` changed the voice from sentence to sentence (median pitch 104 to
  242 Hz); a preset and voice design kept theirs. Each sentence is
  normalized on its own, so loudness varies by up to 3 dB between sentences.
- The preset's reference (`tests/functional/fixtures/audio/clear-english-voice.wav`)
  ends inside the final "t" of "Extract.", and every generation began with
  a 60–80 ms noise burst before its speech, heard as a click between
  sentences. The same reference with 0.4 s of its own room tone appended
  started every sentence cleanly.
- 159 deltas sent one message each stayed below the server's 128 queued
  messages per WebSocket, beyond which it closes the connection (1009); a
  client feeding a faster LLM should send words or sentences rather than
  single tokens.

## Not tried yet

- Fewer, larger GEMMs for short sentences: with about 150 rows per pass the
  generator's linear layers run hipBLASLt at a fraction of the GPU's
  throughput (the extension's benchmarks).
