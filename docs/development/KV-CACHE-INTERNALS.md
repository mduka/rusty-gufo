# KV cache internals

Where the continuation cache lives in the code, the invariants it keeps and
how to verify a change to it. What the cache does for users is in
[KV cache](../KV-CACHE.md).

## Code map

| Part | Where |
| --- | --- |
| Records, lookup, removal priorities | `ContinuationCache`, `src/cli/serve/src/continuation_cache.rs` |
| Prefill checkpoints, captures, RAM budget | `TextRunnerPool`, `PoolRequest`, `src/cli/serve/src/pool/` |
| Disk tier | `ContinuationDiskStore`, `src/cli/serve/src/disk_store/` |
| Waiting for a peer's prefix | `wait_for_shared_prefix`, `src/cli/serve/src/scheduler/admission.rs` |
| Waiting for a disk write | `wait_for_publication`, `src/cli/serve/src/scheduler/admission.rs`; `PublicationWaitEstimate`, `src/cli/serve/src/publication_wait.rs` |
| Host snapshot budget | `host_snapshot_budget_bytes`, `host_snapshot_ceiling_bytes`, `src/core/src/text/runner.rs` |
| Flags | `src/cli/src/serve_cli.rs` |

## Invariants

These constraints are why several design choices are not preferences. Change
them deliberately or not at all.

**A checkpoint must contain the state at its exact token boundary.** Most
runners stop prefill at that boundary before capturing it. Flash-Next can
capture a text prompt's stable boundary within the final prefill pass: its
recurrent kernels retain the boundary state, convolution and PLE retain their
history, and a one-row head computes the boundary logits. Attention queries
on either side keep the grouping of separate passes. The final batch can
include up to 128 tokens beyond the normal 2,048-token chunk, so a prompt that
ends just past a chunk needs no separate tail pass; the stable boundary must be
within eight tokens of that batch's end.
The cache reserves the full payload before enabling this capture. Disk,
image, intermediate and learned-prefix checkpoints still stop prefill at
their boundary.

**The reused frontier must be frozen before prefill.** Prefill mutates the
leased state in place, so a frontier another request might branch from has to
be captured first.

Flash-Next copies its mutable recurrent, convolution and PLE state into
private device storage at capture; the small indexer-ring, draft-residual and
kept-row regions go into a second device block. Committed K/V and pooled
rows remain in the live session while it appends, without a K/V copy.
A same-session rewind allocates backing blocks for only the rows it would
overwrite and copies them once for checkpoints sharing those rows. Appending
allocates no K/V backing blocks. Restore keeps the still-live prefix without
a K/V upload.
Reset leaves append-only rows intact; the first write that would overwrite
them preserves pending checkpoints. Session destruction preserves remaining
rows before releasing device buffers. Byte readers assemble a complete
payload before export. Forks copy the live prefix directly between device
buffers, including detached rows in the shared backing blocks.
A private HIP pool shared by the executor's sessions allocates and releases
snapshot storage without synchronizing peer inference streams. Each fresh
allocation commits device pages, about 4.5 ms per 111 MB on gfx1151, so a
restore into a slot whose rows other checkpoints still borrow pays that cost
for the rows it protects. A retained
Flash-Next snapshot prefers its source execution slot when
that slot is available, avoiding a full-prefix copy for a rewritten turn.
A busy source never blocks a branch into another slot; other choices use LRU.
RAM admission continues to charge the complete payload size, and the disk
byte format is unchanged. Captures intended for disk complete all device
copies before prefill resumes, so disk serialization does not introduce a
background K/V transfer during inference.

**Admission is advisory, never fatal.** A refused reservation or a failed
capture is a skipped optimisation. The request must still complete.

**Disk eviction cannot skip a victim.** If unlinking the least recently used
file fails, eviction stops rather than trying the next candidate, and the new
checkpoint is dropped.

## Verifying changes

- `tests/functional/` covers rotation through independent conversations, a long
  history interrupted by tiny side requests, edits/rewind, advancing boundaries,
  cancellation and disk restart. Uncached controls check the answers; CPU tests
  exercise byte and record pressure without loading models.
- `tests/functional/continuation.py` covers cancellation, reasoning replay,
  images and restart persistence against a live server. After a restart,
  greedy and exact restores must reproduce their output; sampled restores may
  re-prefill less than one disk step.
- `tests/functional/cache_disk_spacing.py`, part of the `cache` suite, checks
  disk checkpoint spacing for a growing conversation, the restore after
  restart, and that a learned branch boundary is still written.
- `tests/functional/cache_concurrency.py` (`cache-concurrency`) sends
  identical prompts, shared-system fan-out with short and long tasks, short or
  no shared prefixes, a retained conversation beside a newcomer and a
  cancelled leader at once, checking waits, prefill work and uncached answers.

The cache tests use fake runners and need no GPU or model:

```sh
cargo test -p gufo-serve --test continuation_cache --test pool \
  --test scheduler --test disk_store
```

- `continuation_cache` covers retention, admission, byte and record pressure,
  removal priorities, edited tails, branch points and image identity.
- `pool` covers prompt and stable boundaries on cold and warm requests,
  intermediate checkpoints and history edits, learned boundaries in RAM and
  on disk, coincident boundaries, worker captures, cancellation, the startup
  line and the `entry_capacity` log.
- `scheduler` sends identical prompts, shared-system fan-out and short shared
  prefixes at once and checks waits, prefill work and outputs, as well as a
  leader cancelled before its follower is admitted.
- `disk_store` covers restart restore, compatibility identity, corruption,
  byte and staging limits, LRU by file bytes, atomic private publication,
  shared-prefix boundaries, images across restarts and checkpoint spacing.
