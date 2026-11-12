# Qwen3.8 27B experiments

What has been tried on this engine for the dense Qwen models (the 27B and the
Qwen3.5 checkpoints from 0.8B to 9B, which the same engine `src/models/qwen35`
runs), so that later optimization rounds do not try it again. Above all it
keeps what did not help or made things worse, with the measurement and the
reason; what the code already does gets a line. Gufo's
[experiments](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-27b/EXPERIMENTS.md)
keep the same for its C++ engine, whose kernels this engine runs: read both
before an optimization round. Ideas not tried yet come last. The core and what
spans models are in [optimizations](../../OPTIMIZATIONS.md).

## Tried without gain

Nothing on this engine yet beyond the core's no-wait host copies, measured on
the 27B with DFlash2 ([optimizations](../../OPTIMIZATIONS.md#tried-without-gain)).

## In the code

- **Bounded prefill submission for cancellation** (Gufo's, ported): the time
  to the first token is unchanged (B/A 1.0009 [0.9991, 1.0034] at 2.5K
  tokens, Q4 target, C1), and a short request after a client abandons a
  7.7K-token prefill finishes in 1.5–1.7 s instead of 13.7–14.4 s.

## Not tried yet

Measured in October 2026 on an earlier build of this code (at Gufo v0.23.0): trace one again on the
current tree before working on it.

| ID | Optimization | Scope | Bits | Status | Expected gain |
| --- | --- | --- | --- | --- | --- |
| G1 | Graphs for decode past the split-K threshold | dense Qwen | bit-exact | open | the 27B decode beyond position 127 runs eagerly today |
| Q1 | WMMA BF16 prefill instead of the exact BF16 GEMM | dense Qwen | changes bits | open | 0.8B BF16 pp 0.88 ms/token against 0.11 for Q8_0 |
| Q2 | WMMA or tiled prefill attention for 8/2 and 16/4 heads | Qwen3.5 0.8B to 9B | changes bits | open | long prompts; speeds up prompts past 16,128 positions |
| Q3 | The 27B fast paths for the other geometries | Qwen3.5 0.8B to 9B | changes bits | open | FP16 prefill from 1024 rows, packed projections |
| Q4 | Q8_0 LM head (and q/k/v) for the 27B UD-Q8_K_XL | dense Qwen | changes bits | open | ~5 ms of 139 ms per token for the head alone |
| Q5 | Smaller LM heads for the small models | Qwen3.5 0.8B to 4B | changes bits | open | the head is 24 to 27% of decode kernel time |
| Q6 | Quantize from FP32 where a norm feeds BF16 and Q8 readers | dense Qwen | changes bits | open | one pass per norm in mixed layers |
| Q7 | DFlash2 draft scratch sized per use | dense Qwen | bit-exact | open | 70 MB (4.63 to 4.70 GiB, 5.86 to 5.93 GiB) |
| Q8 | Residual decode gap to C++ on the small models | dense Qwen | bit-exact | open | 0.2 to 0.4% of decode below the 9B |
| Q9 | Prefill of the 2B and 4B Q4_K_M | dense Qwen | bit-exact | open | 1 to 4% of prefill |
| Q10 | Graph capture outside the first timed `bench` repetition | dense Qwen | bit-exact | open | measurement only |
| Q11 | Short-prompt prefill latency of the 27B | dense Qwen | changes bits | open | TTFT |
| Q12 | Load time of the small models | dense Qwen | bit-exact | open | ~9 ms of ~200 ms on the 0.8B |

**G1. Graphs for decode past the split-K threshold** (dense Qwen,
`src/models/qwen35/src/engine.rs`, `decode.rs`; bit-exact). Split-K decode
attention takes the position as a host scalar, so from position 127 on the
27B decode step runs eagerly on both engines; real chats are past that point after the first
turn. Options: a device-side position for the split-K launcher, keys per
split count, or typed kernel-node parameter setters for graph replay (the
planned `gufo-hip` extension, never built).

**Q1. WMMA BF16 prefill** (`prefill.rs` `project`; changes bits). BF16
projections take the exact small-batch GEMM over FP32 rows
(`kHipPrefillBf16Fp32`), with the decode GEMV's accumulation order: the
0.8B BF16 prefills at 0.88 ms per token, the Q8_0 checkpoint of the same
model at 0.11 ms. A BF16 WMMA GEMM (activations rounded to BF16 or kept in
FP32 tiles) would help every BF16 checkpoint and the BF16 q/k/v of the 27B
UD-Q8_K_XL.

**Q2. Prefill attention for 8/2 and 16/4 heads** (`prefill.rs`,
`attention_wmma.hip`; changes bits). The WMMA and tiled prefill attention
take 24/4 heads of 256 only; the Qwen3.5 0.8B to 9B fall back to
`LaunchBatchedAttention`, one block per query row and head with the row's
scores in LDS. That is slow for long prompts. Beyond 16,128 visible positions a row's
scores exceed the 64 KiB of LDS of a workgroup: the Rust engine then takes
`batched_attention_long` with the decode's bounded online arithmetic (about
760 tokens/s at 32K on the 0.8B), while the C++ launcher does not check the
bound. The models' context is 262,144. A WMMA attention for these head
groups fixes both.

**Q3. The 27B fast paths for the other geometries** (`routes.rs`
`use_fp16_prefill`, `verify.rs`, `batch.rs`; changes bits). The FP16
prefill from 1024 rows, the packed gate/up projection and the packed
alpha/beta projection of verification and batched decode are gated on the
27B's shapes, as in C++. The small models would need their own measurements
of the FP16 prefill's quality; whether the packed projections give the same
bits as separate ones has not been checked.

**Q4. Q8_0 LM head for the 27B UD-Q8_K_XL** (`weights.rs`; changes bits).
The BF16 head (248,320 × 5120) reads 2.54 GB per decode token, about
11.6 ms of the 139 ms step at ~220 GB/s; a Q8_0 copy reads 1.35 GB. With
DFlash2 the copy already exists for the draft (`QuantizeDraftHead`,
1.26 GiB). The BF16 q/k/v of the 16 attention layers read another 2.35 GB
per token (1.25 GB as Q8_0). Both trade the artifact's chosen precision for
speed, so they need the quality checks of the 27B.

**Q5. Smaller LM heads for the small models** (changes bits). The tied
Q6_K head is 27% of the decode kernel time of the 0.8B IQ4_XS (952 of
3,555 µs per token) and 24% of the 2B Q4_K_M (1,805 of 7,537 µs); it runs
at ~220 to 230 GB/s, the memory bandwidth, so only fewer bytes help: a
lower-bit head, or speculative decoding (DFlash2 drafts exist only for the
27B).

**Q6. Quantize from FP32 in mixed layers** (`prefill.rs` `unfused_norm`;
changes bits). Where some readers of a norm are BF16 and others read Q8_1,
the norm writes FP32 rows and a BF16 copy, and the Q8_1 activation is
quantized from that copy, as in C++: two roundings and one pass more than
quantizing the FP32 rows directly (which the fused norm and the attention
output already do). No published checkpoint mixes them in one norm group
yet.

**Q7. DFlash2 draft scratch per use** (`dflash/draft.rs`; bit-exact). The
draft scratch is sized for the 64 block rows of batched serving also in
single-session `bench` and CLI runs: peak memory 4.63 to 4.70 GiB on the
27B UD-Q4_K_XL, 5.86 to 5.93 GiB on the UD-Q8_K_XL.

**Q8. Residual decode gap on the small models** (bit-exact, open). With
validation after the launch and token and position in one copy, the
Qwen3.5 0.8B IQ4_XS decodes 0.4% slower than C++ ([1.0007, 1.0076]), the
2B 0.26% and the 4B 0.21%; from the 9B on decode is level. Kernel traces
are identical. After a pp512 prefill Rust showed twice as many
host-side stalls over 100 µs as C++ (77 against 38; p99 891 against
510 µs). Not explained yet; `perf` needs `perf_event_paranoid` lowered.

**Q9. Prefill of the 2B and 4B Q4_K_M** (bit-exact, open). The A/B of
2026-10-01 had Rust 4% (2B, 0.26 / 0.27 ms) and 1.6% (4B, 0.61 / 0.62 ms)
slower in pp512 with the same kernels; on 2026-10-02 the 2B was level and
the 4B again 0.61 / 0.62 ms. Possibly a route difference for the K-quants,
or noise; not investigated.

**Q10. Graph capture in the first timed repetition** (`src/cli/src/bench.rs`;
bit-exact). `GraphCache` captures on the second eligible step, so `gufo
bench` captures inside its first timed tg repetition, while C++ captures in
the untimed warm-up. Visible in `bench` numbers only.

**Q11. Short-prompt prefill of the 27B** (changes bits). The time to the
first token for short prompts seemed high, on both engines; prefill kernel
work.

**Q12. Load time of the small models** (bit-exact, open). The Qwen3.5
0.8B loads in about 201 ms against 190 ms (C++) in every A/B since the
port, the 4B in 285 against 275 ms; from the 9B on the difference is within
the spread. Load time is secondary to prefill and decode; not investigated.
