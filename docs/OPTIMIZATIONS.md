# Optimizations

Every port to Rust stayed bit-identical to the C++ engine ([Gufo in
Rust](RUST.md)); further optimizations noticed along the way are collected
here. Which entry comes next is the maintainers' call, and entries that
change bits need their go.

Each entry names the component and code, the change, what was measured, and
whether it keeps the bits. "Bit-exact" entries keep the outputs of the
current engine, checked against the kernel or path they replace. "Changes
bits" entries need a quality check of their own (perplexity, token
agreement, image metrics). Add every new finding here when it is noticed,
with its measurement. Techniques that carry over to other models are
collected under "Transferable techniques".

The kernels in this tree are upstream's. Entries marked "measured" were
implemented and measured on the Rust engine, but their kernel changes are
kept apart, to land as changes of their own that the C++ engine can take as
well.

## Index

| ID | Optimization | Component | Bits | Expected gain |
| --- | --- | --- | --- | --- |
| G1 | Graphs for decode past the split-K threshold | dense Qwen | bit-exact | the 27B decode beyond position 127 runs eagerly today |
| G2 | Graphs for batched decode and MTP | Flash-Next | bit-exact | rejected: eager decode is as fast as graph replay (2026-10-05) |
| G3 | Device-side token feedback between decode steps | dense Qwen, Flash-Next | bit-exact | ~55 µs per token (1.3% on the 0.8B) |
| G4 | GPU-side wait for the PLE rows | Flash-Next | bit-exact | host wait before the suffix launch |
| G5 | Replay through PM4 packets instead of HIP graphs | gufo-hip | bit-exact | short decode steps |
| B1 | hipBLASLt plan policies with workspace and retuning | ASR, Flash-Next, vision, Qwen-Image | changes bits | unmeasured |
| B2 | `Placement::Map` as the default weight placement | all models | bit-exact | load time, one copy of the weights less |
| B3 | Kernels at `-O3`, host with `target-cpu=native`, FMA, LTO | build | changes bits (LTO alone: bit-exact) | unmeasured |
| B4 | Kernels without clang 23 register spills | DeepSeek MMQ, dense Qwen, Flash-Next | bit-exact | 230 kernels use scratch; the Q2_K MMQ tiles spill up to 683 VGPRs |
| Q1 | WMMA BF16 prefill instead of the exact BF16 GEMM | dense Qwen | changes bits | 0.8B BF16 pp 0.88 ms/token against 0.11 for Q8_0 |
| Q2 | WMMA or tiled prefill attention for 8/2 and 16/4 heads | Qwen3.5 0.8B to 9B | changes bits | long prompts; lifts the 16,128-position limit |
| Q3 | The 27B fast paths for the other geometries | Qwen3.5 0.8B to 9B | changes bits | FP16 prefill from 1024 rows, packed projections |
| Q4 | Q8_0 LM head (and q/k/v) for the 27B UD-Q8_K_XL | dense Qwen | changes bits | ~5 ms of 139 ms per token for the head alone |
| Q5 | Smaller LM heads for the small models | Qwen3.5 0.8B to 4B | changes bits | the head is 24 to 27% of decode kernel time |
| Q6 | Quantize from FP32 where a norm feeds BF16 and Q8 readers | dense Qwen | changes bits | one pass per norm in mixed layers |
| Q7 | DFlash2 draft scratch sized per use | dense Qwen | bit-exact | 70 MB (4.63 to 4.70 GiB, 5.86 to 5.93 GiB) |
| Q8 | Residual decode gap to C++ on the small models | dense Qwen | bit-exact | 0.2 to 0.4% of decode below the 9B |
| Q9 | Prefill of the 2B and 4B Q4_K_M | dense Qwen | bit-exact | 1 to 4% of prefill |
| Q10 | Graph capture outside the first timed `bench` repetition | dense Qwen | bit-exact | measurement only |
| Q11 | Short-prompt prefill latency of the 27B | dense Qwen | changes bits | TTFT |
| Q12 | Load time of the small models | dense Qwen | bit-exact | ~9 ms of ~200 ms on the 0.8B |
| S1 | Grammar masks on the device | serving | changes bits unless exact | 27B DFlash2: 26.7 against 28.8 tok/s with a JSON schema |
| S2 | Asynchronous warm-continuation snapshot | serving | bit-exact | scheduler stall with several requests |
| S3 | Decode during a snapshot capture | serving | bit-exact | c2: 2 × 37.8 ms instead of one ~48 ms step |
| F1 | Logits written to host-visible memory by a kernel | Flash-Next (measured), dense Qwen | bit-exact | Flash-Next c2 −0.5%, c1 and MTP level |
| F2 | MTP host time per cycle | Flash-Next | bit-exact | ~0.1 to 0.3% |
| F3 | Preallocated rollback rows and control slots | Flash-Next | bit-exact | 4.6 to 8.2 ms on first growth, 20 to 290 µs per new width |
| F4 | One copy less for the PLE rows | Flash-Next | bit-exact | one host copy per token |
| F5 | Host allocations in the batched GatedDeltaNet path | Flash-Next | bit-exact | ~64 `Vec`s per linear layer and step |
| F6 | Prefill kernels | Flash-Next | changes bits | pp512 1026 against 1043 t/s (C++) |
| F7 | Device stall after the PLE convolution | Flash-Next | bit-exact | none: a profiler artifact (2026-10-05) |
| F8 | Fewer decode kernels: shared and fused activation quantization | Flash-Next (measured) | bit-exact | AR −1.7%, MTP −1.6%; 1,623 → 1,247 kernels per step |
| F9 | Latency-bound decode kernels | Flash-Next (HC combine, HC mix epilogue, small GEMM measured) | bit-exact | AR −5.5%, MTP −3.1% |
| F10 | Quantizations after a GEMV, in the consuming GEMV | Flash-Next | bit-exact | ~94 kernels per step, ~0.5% |
| F11 | Remaining short decode chains and small grids | Flash-Next | bit-exact | ≤ 0.5 ms per token |
| F12 | Serving measurement of F1 to F9 | Flash-Next | measurement only | c1, c2, c4 against the build before F1 |
| A1 | Lazy encoder upload | Qwen3-ASR | bit-exact | load 415 against 346 ms (serve) |
| I1 | Device memory below the C++ pool order | Qwen-Image | bit-exact | peaks of 32.42 and 40.55 GiB |
| I2 | Faster hashing for the tokenizer | Qwen-Image | bit-exact | `/ready` ~309 against ~261 ms |
| P1 | Drop the bit-parity emulations | TTS, Qwen-Image, host | changes bits | simpler, faster host code |

## Transferable techniques

The Flash-Next decode work (F1, F8, F9, measured; tg128 37.86 → 35.16 ms AR
and 23.07 → 22.02 ms MTP, outputs identical) found techniques that apply to the
other engines. Each needs its own trace and A/B there; the last column
names where to look, not a measured gain.

| Technique | In Flash-Next | Precondition | Where to look next |
| --- | --- | --- | --- |
| Logits written to host-visible memory by a compute-queue kernel (F1) | per-token download | downloads queue behind copy-engine work (snapshots) | dense Qwen (F1) |
| One quantized input for every projection of the same rows (F8 step 1) | shared and routed experts | two launchers quantize the same rows | DeepSeek V4 Flash's MoE (own `ds4_mmq` quantizer) |
| The producing kernel writes the next GEMV's Q8_1 rows (`StoreQ8_1`, F8 step 2) | five producers, 376 kernels fewer per step | a wave holds 32 consecutive values; padded rows get zero blocks; write only when the next projection reads Q8_1 | dense Qwen already fuses norm and SwiGLU quantization (`batched_*_quantize_q8_1`): check its decode trace for standalone `quantize_q8_1`; DeepSeek V4 Flash |
| Load-first kernels for small grids (F9) | HC combine, HC mix epilogue, small F16 GEMM: 2 to 3.7 times faster | far below bandwidth with few blocks per token; element mapping and sum order stay | decode kernels with a short grid and a loop over K or streams, in every engine |

Bit-exact pitfalls, all met here (details under F8, F9 and in
`.agents/skills/optimize-kernel/SKILL.md`):

- Under `-ffast-math` a fused quantizer must sum the rounded value: hide it
  from the optimizer (`asm volatile("" : "+v"(v))`), or the producer's
  multiply folds into the first addition as one FMA.
- A reduction loop's compiled form is not its source: the scalar small GEMM
  folds each element into the accumulator as one FMA. Read the old kernel's
  ISA (`hipcc --cuda-device-only -S` with the source's flags) and write that
  order explicitly.
- Compare a rewritten kernel with the old one bit for bit: through a
  temporary reference launcher, or permanently by keeping the old kernel as
  the fallback of a dual path
  (`src/models/qwen38_flash_next/kernels/tests/small_gemm.rs`).
- Kernel counts and kernel times from `rocprofv3 --kernel-trace` hold; its
  gaps and host times do not (F7).
- Per removed kernel F8 gained ~1.4 to 2 µs of a decode step; the large
  GEMVs already stream at ~190 to 225 GB/s, so the remaining bit-exact
  headroom is small kernels and launch count.

## Graphs and submission

**G1. Graphs for decode past the split-K threshold** (dense Qwen,
`src/models/qwen35/src/engine.rs`, `decode.rs`; bit-exact). Split-K decode
attention takes the position as a host scalar, so from position 127 on the
27B decode step runs eagerly on both engines; real chats are past that point after the first
turn. Options: a device-side position for the split-K launcher, keys per
split count, or typed kernel-node parameter setters for graph replay (the
planned `gufo-hip` extension, never built).

**G2. Graphs for batched decode and MTP** (Flash-Next,
`src/models/qwen38_flash_next/src/batch.rs`, `mtp.rs`; bit-exact; rejected
2026-10-05). Width-2 decode and the MTP draft and verify forwards run
eagerly in both engines: an MTP tg run issued 17,331 eager launches against
198 graph launches (C++ 14,643 eager launches). Graphs have nothing to win
here: with graphs disabled, AR decode (tg128, same build) ran 26.89 and
26.90 tok/s against 26.82 and 26.85 with them, because eager launches stay
ahead of the GPU; a decode step's `hipGraphLaunch` takes ~11 µs of host
time. The GPU idles about 9% of a decode token (37.4 ms period, ~34 ms of
kernels), almost all of it in the ~2 µs between the ~1,650 dependent
kernels of a step and ~0.23 ms at the token boundary; graphs shorten
neither, fewer and fused kernels would. Profiles taken with `rocprofv3
--kernel-trace` overstate host time: it intercepts every packet inside the
launch (a traced `hipGraphLaunch` took 7 to 8 ms) and starves the GPU at
places where an untraced run has no gap (F7).

**G3. Device-side token feedback** (dense Qwen and Flash-Next; bit-exact).
Greedy decode downloads each token and synchronises before the next step
uploads it again; between the argmax and the next embedding lookup the
device idles about 55 µs per token on the Qwen3.5 2B (C++: 5 µs to the
token's download, 33 µs to the next upload, 17 µs to the first kernel). A graph that copies the argmax
result into the next step's input on the device, with the host enqueuing
step n + 1 before reading token n (stop tokens then cost one discarded
step), removes the round trip. Needs lookahead in the runner's stop
handling.

**G4. GPU-side wait for the PLE rows** (Flash-Next; bit-exact). The host
waits for the n-gram embedding rows before it launches the suffix graph;
the suffix could be launched first and wait on the device for the rows'
upload.

**G5. Replay through PM4 packets** (gufo-hip; bit-exact). hipfire's
"Redline" submits recorded command packets directly instead of
`hipGraphLaunch`. A graph launch takes 15 to 20 µs here (the 7 ms measured
once was a profiler artefact), so the gain is on short decode steps only.

## BLAS, placement and build

**B1. hipBLASLt plan policies** (`gufo_hip::blas`, per model `AlgoPolicy`;
changes bits). Every model pins the C++ choice, because the plan decides
the kernel and so the accumulation order: ASR rank 6 without workspace,
Flash-Next the first workspace-free supported result, the vision encoder 16
heuristic results without workspace at rank 4 or 5. Allowing a workspace,
other ranks or per-shape tuning was never measured. The F32 and Q8_K
projection routes of the dense Qwen engine (hipBLAS/hipBLASLt) are not
ported at all; no published checkpoint needs them.

**B2. `Placement::Map` as the default** (`src/hip/src/weights.rs`;
bit-exact). Mapping the GGUF instead of copying it loaded the ASR
checkpoint in 0.002 s against 0.21 s, and a streaming kernel read mapped
weights at the same 77 GB/s; it avoids a second copy of the weights in RAM
(4.38 GiB for ASR). Whether weight-reusing GEMMs keep their speed on mapped
memory was never measured; every model follows the C++ placement.

**B3. Build flags** (changes bits, except LTO). Kernels compile with
`-O2 -g` as the CMake preset did, the host for baseline x86-64 without FMA
contraction, so that host arithmetic matches C++. `-O3` for kernels,
`target-cpu=native` and FMA for the host change results; LTO and
`codegen-units = 1` in the release profile do not and were simply never
tried.

**B4. Clang 23 register spills** (`tools/ci/kernel-resources.json`;
bit-exact where only scheduling changes). ROCm 10's AMD clang 23 kept 252
of 2,425 kernels in scratch memory, most of them spills that upstream's
clang 22 build does not have. Upstream fixed the dense Qwen prefill GEMMs
(Q8 pp2048 291 to 464 tok/s on clang 23) and the DeepSeek 8-row IQ2 gate/up
tiles; both are ported, and 230 kernels are left.
The Q4_K 8-row tile (Q4_K experts, 48 B) still keeps a per-pair array in
scratch. The largest are the ggml MMQ tiles DeepSeek runs:
`mul_mat_q<Q2_K, 80>` with 683 spilled VGPRs (2,728 B), the 64- and 48-row
Q2_K tiles with 390 and 233, the Q8_0 tiles with up to 66 and the MXFP4 and
IQ2_XXS tiles with up to 49 and 17; the upstream C++ build with the same
compiler spills the same. Flash-Next's own MMQ copy
(`qfn_mmq`) spills up to 59 VGPRs in its Q8_0 and Q5_1 tiles. The DeepSeek
compressor pools (1,040 B) and the indexer top-k (528 B) use scratch by
design (private arrays). Each fix needs the kernel's bitwise test and an ISA
check; `--compare` against the previous binary shows the effect.

## Dense Qwen (`src/models/qwen35`)

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
scores in LDS. That is slow for long prompts and cannot run beyond 16,128
visible positions, where a row's scores exceed the 64 KiB of LDS of a
workgroup: the Rust contract refuses the launch, the C++ launcher does not
check it. The models' context is 262,144. A WMMA attention for these head
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

## Serving (`src/cli/serve`, `gufo-core`)

**S1. Grammar masks on the device** (changes bits unless host sampling is
reproduced exactly). Constrained requests download the logits, mask and
sample on the host, as in C++: the 27B with DFlash2 ran 26.7 tok/s with a
JSON schema against 28.8 without (outputs differed, so rough); Flash-Next
showed no difference.

**S2. Asynchronous warm-continuation snapshot** (bit-exact). The snapshot
is copied synchronously before the prefill, as upstream does, which stalls
the scheduler thread while several requests are in flight; the existing
snapshot queue could take it.

**S3. Decode during a snapshot capture** (bit-exact). A request whose state
is being captured skips decode, so at c2 its peer decodes alone at width 1
(2 × 37.8 ms instead of one ~48 ms width-2 step); the same in C++.

## Qwen3.8 Flash-Next (`src/models/qwen38_flash_next`)

**F1. Logits into host-visible memory** (bit-exact; measured on Flash-Next
2026-10-05). A kernel on the compute queue (`copy_f32`) writes the logit
rows into managed memory that the host reads after the stream, instead of
a copy-engine download that holds the process's single SDMA queue behind
each forward and makes a peer's snapshot copies wait; the workaround that
synchronised before downloads during captures is gone. `ab.py serve` (four
sessions, 8192 context, two runs of 4 ABBA rounds, outputs identical): c2
−0.59% [0.9915, 0.9987] and −0.53% [0.9873, 1.0012], c1, decode, TTFT and c4
level; `ab.py text` AR and MTP tg128 level. A device copy into the same
memory (`hipMemcpyAsync` device to device) still ran on the copy engine.
Dense Qwen could do the same.

**F2. MTP host time per cycle** (bit-exact; ~0.1 to 0.3%). The n-gram hash
and read start (~40 µs), separate uploads of tokens and control words, two
waits after verification, and the 1 MB logits copy into `s.logits` (~50 µs;
lazy logits touch the runner, snapshots and the batch path).

**F3. Preallocated rollback rows and control slots** (bit-exact). The first
growth of the rollback pool to a new depth takes 4.6 to 8.2 ms (~75 MB per
row; C++ allocates per session), and each new draft width gets its own
pinned control slot (20 to 290 µs). Preallocation must respect the serve
memory gate and the rule not to reserve RAM.

**F4. PLE rows without the extra pinned copy** (`ngram.rs`; bit-exact).

**F5. Host allocations in batched GatedDeltaNet** (`batch.rs`; bit-exact).
About 40 `Vec`s per linear layer in the batch path and 24 pointer-array
`Vec`s in the generated wrapper, per step.

**F6. Prefill kernels** (changes bits). pp512 at depth 0 is 1026 against
1043 t/s (C++), deep prefill level: the kernels bound it.

**F7. Device stall after the PLE convolution** (closed 2026-10-05: a
profiler artifact). Kernel traces showed the GPU idle ~0.3 to 0.4 ms after
`PleConvKernel` in every decode step and ~0.7 ms per MTP cycle. The HSA
trace put the host inside the queue doorbell store of `hipGraphLaunch` for
0.8 to 0.9 ms meanwhile: `rocprofv3 --kernel-trace` processes the
intercepted packets there. GPU clock probes (`wall_clock64` into a ring,
without a profiler) time the PLE convolution block at a median 14.1 µs in
decode and 21.3 µs in MTP, its kernels and two probes included: there is
no stall.

**F8. Fewer decode kernels** (bit-exact; measured 2026-10-07). A decode
step launched ~1,620 kernels and the GPU idles ~2 µs between dependent
ones. 462 of them were `quantize_q8_1` (~1.2 µs each), one before every
quantized GEMV.

- Step 1 (done): the routed experts' gated projection reads the shared
  expert's Q8_1 rows of the same input (`mmq_q8_moe_gated_vec`) instead of
  quantizing it again, 46 kernels fewer per step; AR tg128 −0.23% [0.9974,
  0.9989], MTP −0.17% [0.9977, 0.9988] (`ab.py`, ABBA).
- Step 2, SiLU scale and HC combine (done): `silu_scale_q8_1` and
  `hc_combine_q8_1` write the Q8_1 rows of the following decode GEMV
  (`StoreQ8_1`), 188 kernels fewer per step (1,577 → 1,389); AR −0.73%
  [0.9857, 0.9932], MTP −0.73% [0.9918, 0.9943] against step 1, about
  1.4 µs per removed kernel after the producers' extra work. The HC
  combine grows ~8.1 → ~9.9 µs (traced): a decode token runs it as four
  blocks, so the quantization adds latency, not bandwidth.
- Step 2, rest (done): the HC mix epilogue writes the Q8_1 rows of `mixed`
  for its first projection (attention or GatedDeltaNet input, shared
  expert, output head; `route.mixed_q8`, cleared by every other write of
  `mixed`), the GatedDeltaNet epilogue and the sigmoid gate those of their
  output projections, 142 kernels fewer per step (1,389 → 1,247); AR
  −0.77% [0.9919, 0.9930], MTP −0.69% [0.9921, 0.9937] against F9.
- Left: the ~94 quantizations after a GEMV (shared expert down
  projection, routed down projection) need values from several waves.

Bit-exact as long as each 32-value block keeps the quantizer's butterfly
reduction of its sum and sums the rounded value: `StoreQ8_1` hides the
value from the optimizer, since fast math otherwise folds the producer's
multiply into the first addition (one FMA rounding), which changed the sum
of a block with heavy cancellation
(`src/models/qwen38_flash_next/kernels/tests/q8_rows.rs`). The producers
write the rows only when the next projection takes the Q8_0 decode GEMV.

**F9. Latency-bound decode kernels** (bit-exact; HC combine, HC mix
epilogue and small GEMM measured 2026-10-07). A decode token runs the HC combine as four blocks (one per
stream), and each thread waited for its loads once per element chunk in
both passes, reloading the residual in the second. `HcCombineRegKernel`
keeps the element order and reductions, issues a thread's loads before its
arithmetic and holds the updated residual in registers: 9.9 → 5.0 µs per
call (traced, with the Q8_1 rows), equal to the scalar kernel bit for bit
over 16 million values; AR tg128 −1.26% [0.9874, 0.9878], MTP −0.67%
[0.9927, 0.9939] (`ab.py`, ABBA).

The HC mix epilogue (ten blocks per token) waited on one load chain per
stream and per inject logit, and could not load the inject weights before
its stores to `mixed`. `HcMixEpilogue4Kernel` (four streams) loads values,
gates and weights first with the scalar kernel's arithmetic: 7.4 → 3.4 µs
per call (traced), equal bit for bit over 2.8 million values, inject
partial sums included; AR −1.0% [0.9884, 0.9905], MTP −0.57% [0.9937,
0.9946].

`SmallGemmKernel` (one wave per row, a K loop of four-element steps with
element-wise F16 loads, waiting once per step) took 27.8 µs for the
513-row router (47 per token), 14.7 µs for the 96-row alpha/beta
projection (35) and 16 µs for a 128-row attention projection (12), 2 ms
of a 32 ms step. `SmallGemmAheadKernel` (k % 4 == 0, aligned rows) loads
up to ten steps of weights and token values ahead with 8- and 16-byte
loads: 14.0, 3.95 and 5.2 µs (traced; the router streams ~190 GB/s); AR
−3.3% [0.9666, 0.9677], MTP −1.9% [0.9800, 0.9824], prefill unchanged.
The scalar kernel's compiled sum is no per-step dot: fast math folds each
element into the accumulator as one FMA, in element order, and the new
kernel writes exactly that chain (a per-step dot changed BF16 rows).
`src/models/qwen38_flash_next/kernels/tests/small_gemm.rs` keeps both kernels equal bit for bit (an
unaligned input copy takes the scalar one).

**F10. Quantizations after a GEMV** (bit-exact; open). 94 of the 96
`quantize_q8_1` left per decode step follow a GEMV: the shared expert's
gated projection before its down projection, and the routed gate/up
projection before the routed down projection (inside `mmq_moe_vec`). A
Q8_1 block's 32 values come from several waves there, so the producer
cannot write the rows. The consuming GEMV could quantize its input itself,
each block redundantly with `StoreQ8_1`'s arithmetic (k = 640, cheap); at
F8's rate ~0.2 ms per token (~0.5%).

**F11. Remaining short decode chains and small grids** (bit-exact; open).
Per decode step: `SsmConv4` → `GdnPrep` (35 each, ~2.9 and ~1.4 µs) could
be one kernel; so could the small attention steps (`PrepareAttention`,
`StoreRows`, 12 each). `RouterTopK` (47 × 6.1 µs) runs one block per token
after the router GEMM, and the register HC combine still runs four blocks
per token (93 × 5.2 µs). Together at most ~0.5 ms per token.

**F12. Serving measurement of F1 to F9** (measurement only; open). The
series was measured with `ab.py text` (tg128, AR and MTP); an `ab.py serve`
run (c1, c2, c4) against the build before F1 shows how much reaches the
server.

## Qwen3-ASR and Qwen-Image

**A1. Lazy encoder upload** (Qwen3-ASR; bit-exact). C++ uploads the 24
encoder layers during the first request, Rust at load (serve load 415
against 346 ms); load plus first request take the same time.

**I1. Device memory below the C++ pool order** (Qwen-Image; bit-exact).
Temporaries die where the C++ full expressions end, so that peak memory
matches C++ (32.42 GiB for a 1024² generation, 40.55 GiB for an edit).
Freeing activations at their last use, or reusing them in place, could
lower both peaks.

**I2. Tokenizer hashing** (Qwen-Image; bit-exact). The ~300,000-entry
vocabulary map is the suspected cause of `/ready` at ~309 against ~261 ms;
C++ showed the same spread in some rounds.

## Bit-parity emulations

**P1.** These exist only to match C++ bit for bit and can be simplified or
replaced once parity is no longer required (changes bits): the x87
`long double` filterbank sum (`X87Sum`) of the TTS speaker features; the
reproduced libstdc++ `mt19937_64`, `generate_canonical`,
`discrete_distribution` and `normal_distribution`; host arithmetic without
FMA; GCC's right-to-left argument order in the Qwen-Image VAE attention; and
the BF16 GEMV load width tied to a 16-byte-shifted weight region.
