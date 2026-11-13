# Performance

Current speed of rusty-gufo on an AMD Ryzen AI MAX+ 395 with the Radeon
8060S (`gfx1151`), 128 GiB of memory, ROCm 10.1 and the Balanced power
profile, measured on an otherwise idle machine: release 0.1.0, and for the
dense Qwen3.5 and VoxCPM2 the main branch after it (Qwen3.5's faster
prefill attention and VoxCPM2 are in the [changelog](../CHANGELOG.md)). How the numbers are taken is in
[benchmarking](development/BENCHMARKING.md); how they compare with Gufo's
C++ build in [relation to Gufo](development/PARITY.md#performance-against-gufo).

Text models give llama-bench's points from `gufo bench`, in tokens per
second, greedy: `pp2048` and `pp32768` process a prompt of 2,048 or 32,768
tokens from an empty context, and `tg128 @ 2K` and `tg128 @ 32K` generate
128 tokens after 2,048 or 32,768 tokens of context, as after an
instruction or a long document. The same points come from llama.cpp's
`llama-bench -p 2048,32768 -n 0` and `llama-bench -p 0 -n 128 -d 2048,32768`.
`gufo bench` feeds a synthetic token sequence, as llama-bench does. Drafts
depend on the text, so speculative decoding is measured over HTTP with a
real prose prompt (402 to 471 tokens, depending on the tokenizer; 128
generated, greedy): the server's decode rate of a request alone, the output
tokens per second of whole requests, prefill included, alone (C1) and four
at once (C4, together), and the time to the first token. Speech, embedding
and image timings come from requests to their servers.

## Qwen3.8 27B

| File | pp2048 | pp32768 | tg128 @ 2K | tg128 @ 32K |
| --- | --- | --- | --- | --- |
| UD-Q4_K_XL | 579 | 498 | 11.9 | 10.8 |
| UD-Q8_K_XL | 413 | 371 | 7.2 | 6.8 |

With the DFlash2 draft, over HTTP, UD-Q4_K_XL:

| Draft | Decode | C1 | C4 | First token |
| --- | --- | --- | --- | --- |
| DFlash2 | 36.4 | 26.4 | 30.3 | 1,319 ms |

## Qwen3.5 (dense)

| Model | pp2048 | pp32768 | tg128 @ 2K | tg128 @ 32K |
| --- | --- | --- | --- | --- |
| Qwen3.5 0.8B Q8_0 | 14,090 | 9,092 | 204.2 | 127.2 |
| Qwen3.5 2B Q4_K_M | 6,854 | 5,348 | 123.6 | 90.1 |
| Qwen3.5 4B Q4_K_M | 2,882 | 2,222 | 59.7 | 45.6 |
| Qwen3.5 9B Q4_K_M | 1,583 | 1,350 | 33.0 | 28.1 |

## Qwen3.6 35B-A3B

| File | pp2048 | pp32768 | tg128 @ 2K | tg128 @ 32K |
| --- | --- | --- | --- | --- |
| UD-Q4_K_M | 3,447 | 2,524 | 62.8 | 53.5 |
| Q8_0 | 3,295 | 2,512 | 54.9 | 47.6 |

With drafts, over HTTP, UD-Q4_K_M:

| Draft | Decode | C1 | C4 | First token |
| --- | --- | --- | --- | --- |
| MTP | 101.3 | 85.9 | 112.6 | 224 ms |
| DFlash2 | 122.5 | 99.9 | 132.1 | 233 ms |

Against the extension's C++ build, prefill is equal within the spread of the
runs (pp2048 3,478 against 3,483 tokens/s, alternating) and MTP decoding is
1.4% (UD-Q4_K_M) to 3.0% (Q8_0) faster.

## Qwen3.8 Flash-Next

| File | pp2048 | pp32768 | tg128 @ 2K | tg128 @ 32K |
| --- | --- | --- | --- | --- |
| UD-Q4_K_XL | 1,673 | 1,621 | 28.7 | 28.8 |

With the MTP draft, over HTTP:

| Draft | Decode | C1 | C4 | First token |
| --- | --- | --- | --- | --- |
| MTP | 55.0 | 46.4 | 81.8 | 427 ms |

## DeepSeek V4 Flash

| File | pp2048 | pp32768 | tg128 @ 2K | tg128 @ 32K |
| --- | --- | --- | --- | --- |
| Flash 0731, mixed IQ2/Q2/Q8 | 467 | 469 | 16.2 | 15.2 |

With the DSpark draft, over HTTP:

| Draft | Decode | C1 | C4 | First token |
| --- | --- | --- | --- | --- |
| DSpark | 27.0 | 19.6 | 24.6 | 1,783 ms |

## Gemma 4

| File | pp2048 | pp32768 | tg128 @ 2K | tg128 @ 32K |
| --- | --- | --- | --- | --- |
| 26B-A4B UD-Q4_K_M | 3,704 | 2,618 | 53.1 | 48.2 |

With the MTP draft, over HTTP:

| Draft | Decode | C1 | C4 | First token |
| --- | --- | --- | --- | --- |
| MTP | 86.7 | 76.8 | 129.6 | 190 ms |

## Nemotron 3.5 Bolt 74B-A9B

| File | pp2048 | pp32768 | tg128 @ 2K | tg128 @ 32K |
| --- | --- | --- | --- | --- |
| 24 GB GGUF | 1,037 | 958 | 36.3 | 34.6 |

With the LiLiCorr draft, over HTTP:

| Draft | Decode | C1 | C4 | First token |
| --- | --- | --- | --- | --- |
| LiLiCorr | 77.4 | 55.9 | 60.8 | 691 ms |

## Qwen3-ASR

Qwen3-ASR 1.7B transcribes a 15.1-second English recording (49 tokens of
text) in 0.94 s, 16 times faster than real time: 39 ms in the audio
encoder and 902 ms in the text decoder.

## Qwen3-TTS

The 1.7B models, streamed, seed 42, `max_new_tokens` 64 (5.12 s of audio) of
"When the first light reaches the harbor, speak gently, with quiet
confidence and a warm, resonant tone.":

| Model | First audio | Whole utterance | Per second of audio |
| --- | --- | --- | --- |
| CustomVoice (voice `vivian`) | 199 ms | 2.02 s | 0.39 s |
| VoiceDesign ("A calm, warm female voice.") | 200 ms | 2.02 s | 0.39 s |

## OmniVoice

The same sentence, streamed, seed 42, `max_new_tokens` 64 (2.76 s of audio): the
whole utterance in 0.79 s, 0.29 s per second of audio. The audio arrives
when it is complete; the 0.1 s of silence that opens it is sent at once.

## VoxCPM2

The same sentence, streamed, seed 42, automatic voice (7.36 s of audio at
48 kHz):

| Files | First audio | Whole utterance | Per second of audio |
| --- | --- | --- | --- |
| Hugging Face directory (bfloat16) | 84 ms | 2.25 s | 0.31 s |
| `voxcpm2-q8_0.gguf` | 45 ms | 1.42 s | 0.19 s |

The extension's C++ build is as fast
([checks](development/models/voxcpm2/CHECKS.md)).

## EmbeddingGemma 2

A batch of 32 documents of about 512 tokens each takes 0.52 s (about
31,000 tokens per second); a single short query 7.9 ms.

## Qwen-Image-2.1

A 1024×1024 image in 40 steps takes 109.7 s: 80 ms for the prompt, 108.7 s
of denoising (2.72 s per step) and 0.86 s for the VAE decoder.
