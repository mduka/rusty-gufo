# Nemotron 3.5 ASR experiments

The extension's comparison with audio.cpp (loading, uploads, concurrent
uploads, streamed text) is in its
[benchmarks](https://github.com/gufo-extensions/nemotron_asr/blob/c256b69/docs/models/nemotron-3.5-asr/BENCHMARKS.md);
its kernels' design notes are in the comments of `src/models/nemotron_asr/hip`.

## The FFT loops of the feature extractor

The first port of `features.cpp` kept its indexed loops: the window
multiplication sample by sample with a range check, and the radix-2
butterflies over `fft[block + k]` and `fft[block + k + size / 2]`. In a
release build its features took 12 % longer than C++ on the 308.7 s
recording (98.9 against 87.8 ms) and 4.5 % on the 10 s clip; the encoder
and decoder were at C++'s speed or faster.

Rewritten over slices (a frame inside the signal multiplies one slice by the
window; the butterflies walk `chunks_exact_mut` blocks split at the half,
with the twiddles stepped by the stride), the loops lose their bounds
checks and compute the same products and sums in the same order: the
frames stay bit-identical to C++'s. The features then took 66 ms alone and
10 to 12 % less than C++ at the same position in an alternating round.

## Not tried yet

- Live transcription and its lookahead-6 passes, once the server has a live
  transcription interface.
