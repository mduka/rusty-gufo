# Performance

Current speed of rusty-gufo 0.1.0 on an AMD Ryzen AI MAX+ 395 with the
Radeon 8060S (`gfx1151`), 128 GiB of memory, ROCm 10.1 and the Balanced power
profile, measured on an otherwise idle machine. How the numbers are taken is
in [benchmarking](development/BENCHMARKING.md); how they compare with Gufo's
C++ build in [relation to Gufo](development/PARITY.md#performance-against-gufo).

Text models give llama-bench's points from `gufo bench`, in tokens per
second, greedy: `pp512` and `pp32768` process a prompt of 512 or 32,768
tokens from an empty context, `tg128` generates 128 tokens, and
`tg128 @ 32K` generates them after 32,768 tokens of context. The same
points come from llama.cpp's `llama-bench -p 512,32768 -n 128` and
`llama-bench -p 0 -n 128 -d 32768`. `gufo bench` feeds a synthetic token
sequence, as llama-bench does. Drafts depend on the text, so speculative
decoding is measured over HTTP with a real prose prompt (402 tokens, 128
generated, greedy): the server's decode rate of a request alone, and the
output tokens per second of whole requests, prefill included, alone (C1) and
four at once (C4, together). Speech, embedding and image timings come from
requests to their servers.

## Qwen3.8 27B

Not measured for 0.1.0 yet; the numbers follow.

## Qwen3.5 (dense)

| Model | pp512 | pp32768 | tg128 | tg128 @ 32K |
| --- | --- | --- | --- | --- |
| Qwen3.5 0.8B Q8_0 | 10,967 | 654 | 208.9 | 126.6 |
| Qwen3.5 2B Q4_K_M | 5,502 | 619 | 126.4 | 88.7 |
| Qwen3.5 4B Q4_K_M | 2,419 | 140 | 60.8 | 45.1 |
| Qwen3.5 9B Q4_K_M | 1,412 | 135 | 33.2 | 28.0 |

Prompts past 16,128 tokens take a slower attention path on these models
([known limits](models/qwen3.5/README.md#known-limits)).

## Qwen3.6 35B-A3B

| File | pp512 | pp32768 | tg128 | tg128 @ 32K |
| --- | --- | --- | --- | --- |
| UD-Q4_K_M | 2,421 | 2,524 | 63.1 | 53.5 |
| Q8_0 | 2,143 | 2,512 | 54.4 | 47.6 |

With drafts, over HTTP, UD-Q4_K_M:

| Draft | Decode | C1 | C4 | First token |
| --- | --- | --- | --- | --- |
| MTP | 101.3 | 85.9 | 112.6 | 224 ms |
| DFlash2 | 122.5 | 99.9 | 132.1 | 233 ms |

Against the extension's C++ build, prefill is equal within the spread of the
runs (pp512 2,492 against 2,497 tokens/s, alternating) and MTP decoding is
1.4% (UD-Q4_K_M) to 3.0% (Q8_0) faster.

## Qwen3.8 Flash-Next

Not measured for 0.1.0 yet; the numbers follow.

## DeepSeek V4 Flash

Not measured for 0.1.0 yet; the numbers follow.

## Gemma 4

Not measured for 0.1.0 yet; the numbers follow.

## Nemotron 3.5 Bolt 74B-A9B

Not measured for 0.1.0 yet; the numbers follow.

## Qwen3-ASR

Not measured for 0.1.0 yet; the numbers follow.

## Qwen3-TTS

Not measured for 0.1.0 yet; the numbers follow.

## OmniVoice

Not measured for 0.1.0 yet; the numbers follow.

## EmbeddingGemma 2

Not measured for 0.1.0 yet; the numbers follow.

## Qwen-Image-2.1

Not measured for 0.1.0 yet; the numbers follow.
