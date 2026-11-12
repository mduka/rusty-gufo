# Qwen3.8 Flash-Next experiments

What has been tried on this engine for Qwen3.8 Flash-Next, so that later
optimization rounds do not try it again. Above all it keeps what did not help
or made things worse, with the measurement and the reason; what the code
already does gets a line. Gufo's
[experiments](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-flash-next/EXPERIMENTS.md)
keep the same for its C++ engine, whose kernels this engine runs: read both
before an optimization round. Ideas not tried yet come last. The core and what
spans models are in [optimizations](../../OPTIMIZATIONS.md). The techniques
the decode work below found for other engines are there as well.

## Tried without gain

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

**F7. Device stall after the PLE convolution** (closed 2026-10-05: a
profiler artifact). Kernel traces showed the GPU idle ~0.3 to 0.4 ms after
`PleConvKernel` in every decode step and ~0.7 ms per MTP cycle. The HSA
trace put the host inside the queue doorbell store of `hipGraphLaunch` for
0.8 to 0.9 ms meanwhile: `rocprofv3 --kernel-trace` processes the
intercepted packets there. GPU clock probes (`wall_clock64` into a ring,
without a profiler) time the PLE convolution block at a median 14.1 µs in
decode and 21.3 µs in MTP, its kernels and two probes included: there is
no stall.

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
Dense Qwen could do the same. On ROCm 10.1 there is nothing left to gain: a
trace of Gufo's `main` (eddcfea6, 2026-11-01) shows its logits download
(`hipMemcpyAsync` into pinned memory) as a copy kernel,
`__amd_rocclr_copyBuffer`, on the queue of the forward's kernels (10.5 µs),
not on the copy engine.

**Snapshot settle and restore** (serving with MTP; 2026-11-01). A repeated
prompt restores its snapshot's frozen state (about 118 MB) from pageable host
memory in 3.5 to 4.6 ms, C++ from pinned memory in 2.2 to 2.8 ms
([results](../../PARITY.md#results)). Tried and dropped: settling the snapshot
into host memory at prefill start (the staged copy during the prefill added 6
to 10 ms of GPU wait); a pinned buffer allocated per settle (a 236 MB
`hipHostMalloc` per request, 35 to 55 ms more to the first token); batching
the restore's copies into one transfer (no gain: the pageable transfer
dominates, not its 87 synchronisations). Matching C++ needs pinned memory held
from load, which this engine does not reserve.

**Snapshot settle and restore** (serving with MTP; 2026-11-01). A repeated
prompt restores its snapshot's frozen state (about 118 MB) from pageable host
memory in 3.5 to 4.6 ms, C++ from pinned memory in 2.2 to 2.8 ms
([results](../../PARITY.md#results)). Tried and dropped: settling the snapshot
into host memory at prefill start (the staged copy during the prefill added 6
to 10 ms of GPU wait); a pinned buffer allocated per settle (a 236 MB
`hipHostMalloc` per request, 35 to 55 ms more to the first token); batching
the restore's copies into one transfer (no gain: the pageable transfer
dominates, not its 87 synchronisations). Matching C++ needs pinned memory held
from load, which this engine does not reserve.

## In the code

- **F8 and F9** (fewer and latency-bound decode kernels): Gufo's #518, in
  this tree since v0.23.0, lets the producers write the Q8_1 rows of the next
  decode GEMV, `qfn_mmq_moe_gated_vec_preq` reuses the shared expert's rows,
  and the HC combine, HC mix epilogue and `SmallGemm` issue their loads first.
  This engine's own versions had gained AR −1.7% and MTP −1.6% (F8), AR −5.5%
  and MTP −3.1% (F9) in October 2026; the bit-exact pitfalls they met are
  under transferable techniques in [optimizations](../../OPTIMIZATIONS.md).
  Gufo's #894 (the HC mixer as one kernel) is not in this tree (F11). F12,
  their serving measurement, is moot.
- **Settled snapshots:** after 10 generated tokens a request's frozen
  snapshot state moves into pageable huge-page host memory on a thread, in
  16 MiB pieces (`SETTLE_AFTER_TOKENS` in `src/runner.rs`), and a serving
  restore keeps the MTP rollback rows and graphs (`restore_reusing_mtp`).

## Not tried yet

Measured in October 2026 on an earlier build of this code (at Gufo v0.23.0), F10 and F11 on top of
its own F8 and F9 kernels: trace them again on the current tree before
working on them.

| ID | Optimization | Bits | Status | Expected gain |
| --- | --- | --- | --- | --- |
| G4 | GPU-side wait for the PLE rows | bit-exact | open | host wait before the suffix launch |
| F2 | MTP host time per cycle | bit-exact | open | ~0.1 to 0.3% |
| F3 | Preallocated rollback rows and control slots | bit-exact | open | 4.6 to 8.2 ms on first growth, 20 to 290 µs per new width |
| F4 | One copy less for the PLE rows | bit-exact | open | one host copy per token |
| F5 | Host allocations in the batched GatedDeltaNet path | bit-exact | open | ~64 `Vec`s per linear layer and step |
| F6 | Prefill kernels | changes bits | open | pp512 1026 against 1043 t/s (C++) |
| F10 | Quantizations after a GEMV, in the consuming GEMV | bit-exact | open | ~94 kernels per step, ~0.5% |
| F11 | Remaining short decode chains and small grids | bit-exact | open | ≤ 0.5 ms per token |

**G4. GPU-side wait for the PLE rows** (Flash-Next; bit-exact). The host
waits for the n-gram embedding rows before it launches the suffix graph;
the suffix could be launched first and wait on the device for the rows'
upload.

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
per token (93 × 5.2 µs; Gufo's #894 runs it inside one mixer kernel).
Together at most ~0.5 ms per token.
