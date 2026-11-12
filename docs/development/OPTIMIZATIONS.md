# Optimizations

Optimizations of the core and of what spans models: graphs and submission,
`gufo-hip`, BLAS and build, serving, the bit-parity emulations, and the
techniques that carry over from one model to others. Like the models'
experiments (`docs/models/<model>/EXPERIMENTS.md`, [models](models/README.md)),
it keeps above all what was tried and did not help or made things worse, so
that later optimization rounds do not try it again; what the code already
does gets a line, and ideas not tried yet follow with their expected gain.

Each entry names the component and code, the change, what was measured, and
its status. "Bit-exact" entries keep the current engine's outputs, checked
against the kernel or path they replace. "Changes bits" entries need a
quality check of their own (perplexity, token agreement, image metrics) and
must keep what [relation to Gufo](PARITY.md) fixes: the same
output tokens, audio codes, image latents and embedding vectors as Gufo
v0.28.1. Add every finding when it is noticed, with its measurement: to the
model's experiments when it concerns one model, here when it concerns the
core or several.

## Tried without gain

**Scoped no-wait host copies** (`gufo-hip`; built and removed in November
2026). A `Stream::with_host_copies` scope proved at compile time that host
memory outlives its asynchronous copies, so the 27B's DFlash2 inject and the
sampled verification could skip their stream synchronisations. Clean direct
A/B: prefill +0.08% [−0.06, +0.20], decode −0.06% [−0.26, +0.11]. A
synchronisation on an idle stream costs nothing measurable here.

## In the code

- **Long-lived memory outside the stream pool:** on gfx1151 the pool behind
  `Stream::alloc` maps 128 MiB chunks that it unmapped at every
  synchronisation (5 to 8 ms to map one again), and kernels on pool memory ran
  at lower clocks (gfx ~2190 against ~2380 MHz, same ISA); Flash-Next pp512
  was +2.2% against C++ (October 2026). Memory that lives as long as a model or
  a server slot comes from `alloc_persistent`, and `retain_pool_memory` keeps
  up to 2 GiB of freed pool memory mapped, set after the pool's first
  allocation (HIP 7.15 ignores it before).

## Transferable techniques

The Flash-Next decode work ([F1, F8, F9](models/qwen3.8-flash-next/EXPERIMENTS.md), measured; tg128 37.86 → 35.16 ms AR
and 23.07 → 22.02 ms MTP, outputs identical) found techniques that apply to the
other engines. Each needs its own trace and A/B there; the last column
names where to look, not a measured gain.

| Technique | In Flash-Next | Precondition | Where to look next |
| --- | --- | --- | --- |
| Logits written to host-visible memory by a compute-queue kernel (F1) | per-token download | downloads queue behind copy-engine work (snapshots); on ROCm 10.1 they run as copy kernels already | dense Qwen (F1), only where a trace shows the download on the copy engine |
| One quantized input for every projection of the same rows (F8 step 1) | shared and routed experts | two launchers quantize the same rows | DeepSeek V4 Flash's MoE (own `ds4_mmq` quantizer) |
| The producing kernel writes the next GEMV's Q8_1 rows (`StoreQ8_1`, F8 step 2) | five producers, 376 kernels fewer per step | a wave holds 32 consecutive values; padded rows get zero blocks; write only when the next projection reads Q8_1 | dense Qwen already fuses norm and SwiGLU quantization (`batched_*_quantize_q8_1`): check its decode trace for standalone `quantize_q8_1`; DeepSeek V4 Flash |
| Load-first kernels for small grids (F9) | HC combine, HC mix epilogue, small F16 GEMM: 2 to 3.7 times faster | far below bandwidth with few blocks per token; element mapping and sum order stay | decode kernels with a short grid and a loop over K or streams, in every engine |

Bit-exact pitfalls, all met here (details in
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

## Not tried yet

Most were measured in October 2026 on an earlier build of this code (at Gufo v0.23.0); the engine
now runs Gufo v0.28.1's kernels unchanged. Trace an entry again on the
current tree before working on it.

| ID | Optimization | Component | Bits | Status | Expected gain |
| --- | --- | --- | --- | --- | --- |
| G3 | Device-side token feedback between decode steps | dense Qwen, Flash-Next | bit-exact | open | ~55 µs per token (1.3% on the 0.8B) |
| G5 | Replay through PM4 packets instead of HIP graphs | gufo-hip | bit-exact | open | short decode steps |
| B1 | hipBLASLt plan policies with workspace and retuning | ASR, Flash-Next, vision, Qwen-Image | changes bits | open | unmeasured |
| B2 | `Placement::Map` as the default weight placement | all models | bit-exact | open | load time, one copy of the weights less |
| B3 | Kernels at `-O3`; host with `x86-64-v3`, FMA contraction, LTO | build | changes bits (LTO and the Rust CPU target: tokens to check) | open | unmeasured; bounded by the host's share of the time |
| B4 | Kernels without clang 23 register spills | DeepSeek MMQ, dense Qwen, Flash-Next | bit-exact | open | 230 kernels use scratch; the Q2_K MMQ tiles spill up to 683 VGPRs |
| S1 | Grammar masks on the device | serving | changes bits unless exact | open | 27B DFlash2: 26.7 against 28.8 tok/s with a JSON schema |
| S2 | Asynchronous warm-continuation snapshot | serving | bit-exact | open | scheduler stall with several requests |
| S3 | Decode during a snapshot capture | serving | bit-exact | open | c2: 2 × 37.8 ms instead of one ~48 ms step |
| P1 | Drop the bit-parity emulations | TTS, Qwen-Image, host | changes bits | open | simpler, faster host code |

### Graphs and submission

**G3. Device-side token feedback** (dense Qwen and Flash-Next; bit-exact).
Greedy decode downloads each token and synchronises before the next step
uploads it again; between the argmax and the next embedding lookup the
device idles about 55 µs per token on the Qwen3.5 2B (C++: 5 µs to the
token's download, 33 µs to the next upload, 17 µs to the first kernel). A graph that copies the argmax
result into the next step's input on the device, with the host enqueuing
step n + 1 before reading token n (stop tokens then cost one discarded
step), removes the round trip. Needs lookahead in the runner's stop
handling.

**G5. Replay through PM4 packets** (gufo-hip; bit-exact). hipfire's
"Redline" submits recorded command packets directly instead of
`hipGraphLaunch`. A graph launch takes 15 to 20 µs here (the 7 ms measured
once was a profiler artefact), so the gain is on short decode steps only.

### BLAS, placement and build

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

**B3. Build flags** (changes bits, except LTO and the Rust CPU target).
Kernels compile with `-O2 -g` as the CMake preset did, the host for baseline
x86-64 (no `target-cpu` for Rust, no `-march` for the C++ host code of the
kernel crates, which Rust flags do not reach) without FMA contraction, so that
host arithmetic matches C++; Gufo's C++ build has no `-march` either. `-O3`
for kernels and FMA contraction in the C++ host code change results. For the
Rust code, `-C target-cpu=x86-64-v3` only widens vectorisation: Rust neither
contracts nor reorders floating-point operations, but a crate may choose
another path through `cfg(target_feature)`, so the token comparison decides.
Every Strix and Gorgon Halo has x86-64-v3, as do the hosted CI runners;
x86-64-v4 or `znver5` would fail on runners without AVX-512. The CPU target,
LTO and `codegen-units = 1` in the release profile were never measured; the
host's share of the time (grammar masks, the host sampler of sampled MTP, the
tokenizer, image and audio preprocessing) bounds the gain. Measure them by
A/B with a token comparison before setting them
(`[target.x86_64-unknown-linux-gnu] rustflags` in `.cargo/config.toml`).

**B4. Clang 23 register spills** (`tools/ci/kernel-resources.json`;
bit-exact where only scheduling changes). ROCm 10's AMD clang 23 kept 252
of 2,425 kernels in scratch memory, most of them spills that Gufo's
clang 22 build does not have. Gufo fixed the dense Qwen prefill GEMMs
(Q8 pp2048 291 to 464 tok/s on clang 23) and the DeepSeek 8-row IQ2 gate/up
tiles; both are ported, and 230 kernels are left.
The Q4_K 8-row tile (Q4_K experts, 48 B) still keeps a per-pair array in
scratch. The largest are the ggml MMQ tiles DeepSeek runs:
`mul_mat_q<Q2_K, 80>` with 683 spilled VGPRs (2,728 B), the 64- and 48-row
Q2_K tiles with 390 and 233, the Q8_0 tiles with up to 66 and the MXFP4 and
IQ2_XXS tiles with up to 49 and 17; Gufo's C++ build with the same
compiler spills the same. Flash-Next's own MMQ copy
(`qfn_mmq`) spills up to 59 VGPRs in its Q8_0 and Q5_1 tiles. The DeepSeek
compressor pools (1,040 B) and the indexer top-k (528 B) use scratch by
design (private arrays). Each fix needs the kernel's bitwise test and an ISA
check; `--compare` against the previous binary shows the effect.

### Serving (`src/cli/serve`, `gufo-core`)

**S1. Grammar masks on the device** (changes bits unless host sampling is
reproduced exactly). Constrained requests download the logits, mask and
sample on the host, as in C++: the 27B with DFlash2 ran 26.7 tok/s with a
JSON schema against 28.8 without (outputs differed, so rough); Flash-Next
showed no difference.

**S2. Asynchronous warm-continuation snapshot** (bit-exact). The snapshot
is copied synchronously before the prefill, as Gufo does, which stalls
the scheduler thread while several requests are in flight; the existing
snapshot queue could take it.

**S3. Decode during a snapshot capture** (bit-exact). A request whose state
is being captured skips decode, so at c2 its peer decodes alone at width 1
(2 × 37.8 ms instead of one ~48 ms width-2 step); the same in C++.

### Bit-parity emulations

**P1.** These exist only to match C++ bit for bit and can be simplified or
replaced once parity is no longer required (changes bits): the x87
`long double` filterbank sum (`X87Sum`) of the TTS speaker features; the
reproduced libstdc++ `mt19937_64`, `generate_canonical`,
`discrete_distribution` and `normal_distribution`; host arithmetic without
FMA; GCC's right-to-left argument order in the Qwen-Image VAE attention; and
the BF16 GEMV load width tied to a 16-byte-shifted weight region.
