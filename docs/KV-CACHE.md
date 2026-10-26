# KV Cache

## What "KV cache" means here

Two different things share the name. Keeping them apart avoids most of the
confusion.

**Within one request.** The standard transformer mechanism. Prefill computes
key and value tensors for every prompt token once; decode then appends one K/V
pair per generated token and attends over the stored set. Without it, each new
token would re-run attention across the whole sequence. This is intrinsic to
inference and is not configurable.

**Across requests.** Keeping that state alive after a response finishes, so
the next turn of the same conversation does not recompute the prefix it shares
with the previous one. This is what the rest of this document is about: the
`ContinuationCache`, and the optional on-disk tier behind `--cache-disk`.

The second is where the important subtlety lives. What Gufo retains is **not**
literally K/V tensors. It is an opaque snapshot owned by the model runner
(`TextModelRunner::snapshot`). For Qwen3.8-27B it holds the attention K/V rows
together with the DeltaNet convolution and recurrent state, the frontier
logits and any speculative verifier state. For Qwen3.8-Flash-Next it contains
KV, GDN recurrent state, the indexer, position and sampling state. The cache
accounts for the payload bytes but cannot inspect or interpret them.

That choice buys one retention, admission, eviction and persistence mechanism
that is correct for every model, including hybrid-recurrent ones. It costs the
ability to trim a checkpoint: a saved snapshot can only be reused **whole**,
and only when its tokens are an exact prefix of the new prompt.

Upstream v0.23.0 serves Qwen3.8-27B (AR and DFlash2) and Qwen3.8-Flash-Next
through the component cache of its
[cache redesign](https://github.com/gufo-org/gufo/blob/v0.23.0/docs/cache-redesign/RFC.md)
(`src/cache`) and keeps this snapshot cache for DeepSeek. The Rust host code
keeps the snapshot cache for every text model instead, with upstream's
policies: the deepest checkpoint across RAM and disk, metadata-only startup,
oversized checkpoints on disk only and the wait for checkpoints being
published ([implemented differently](RUST.md#implemented-differently)).

## Why it exists

Prefill cost scales with prompt length. Decode cost scales with tokens
generated. Agent conversations have a large prompt and a small output, and
turn *N*'s prompt is nearly all of turn *N−1*'s prompt. Without cross-request
reuse, every turn pays for the whole conversation again and total cost grows
quadratically with conversation length.

Figures marked **(measured)** were taken with the C++ engine before the Rust
port, whose cache policy this implementation keeps. They show the scale and
have not been repeated since.

What reuse is worth, same conversation, second turn against a cold start
**(measured)**:

| Model | Prompt | Cold | Reused |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | 101,545 | 280.4 s | 2.4 s |
| Qwen3.8-Flash-Next | 203,047 | 184.4 s | 0.7 s |

A recorded 28-turn `pi` coding session reused 95.9% of its prompt tokens. The
same session replayed with reuse broken reprocessed 9.3x the tokens and took
2.5x the wall time, at only 15k tokens **(measured)**.

## The shape of the system

```mermaid
flowchart LR
  REQ[Request] --> LOOK{Deepest checkpoint<br/>in RAM or on disk}
  LOOK -->|RAM, or equal depth| USE[Reuse live state<br/>or restore a snapshot]
  LOOK -->|disk deeper| DISK{Disk restore}
  DISK -->|restored| SUF
  DISK -->|rejected| USE
  LOOK -->|none| COLD[No reuse]
  USE --> SUF[Prefill only the new suffix]
  COLD --> ALL[Prefill the whole prompt]
```

The in-memory tier is always on. The disk tier is opt-in, adds restart safety,
and can additionally *learn* boundaries between conversations that merely share
a prefix. A lookup takes the deepest usable checkpoint across both tiers, the
RAM one on equal lengths, as upstream's tiered `PrefixIndex` does: the disk
probe reads only the store's metadata, and a store busy with a metadata update
keeps the RAM choice. A deeper disk file that fails its checksum or restore
gives way to a retained RAM snapshot beneath it.

| Part | Where |
| --- | --- |
| Records, lookup, removal priorities | `ContinuationCache`, `src/cli/serve/src/continuation_cache.rs` |
| Prefill checkpoints, captures, RAM budget | `TextRunnerPool`, `PoolRequest`, `src/cli/serve/src/pool/` |
| Disk tier | `ContinuationDiskStore`, `src/cli/serve/src/disk_store/` |
| Waiting for a peer's prefix | `wait_for_shared_prefix`, `src/cli/serve/src/scheduler/admission.rs` |
| Waiting for a disk write | `wait_for_publication`, `src/cli/serve/src/scheduler/admission.rs`; `PublicationWaitEstimate`, `src/cli/serve/src/publication_wait.rs` |
| Host snapshot budget | `host_snapshot_budget_bytes`, `host_snapshot_ceiling_bytes`, `src/core/src/text/runner.rs` |
| Flags | `src/cli/src/serve_cli.rs` |

## Finding reuse

A request arrives with a token sequence. `TextRunnerPool::acquire_with` asks
the cache for the longest retained prefix of it, in two different forms.

```mermaid
flowchart TD
  A[New prompt] --> B{Is a live frontier<br/>a prefix of it?}
  B -->|yes| C[Reuse that session in place<br/>no restore, no copy]
  B -->|no| D{Is a retained snapshot<br/>a prefix of it?}
  D -->|yes| E[Restore the snapshot<br/>into a free session]
  D -->|no| F[Miss<br/>report reason and common prefix]
  C --> G[Prefill the suffix only]
  E --> G
  F --> H[Prefill everything]
```

A **live frontier** is where a session's state actually ended up after its last
request, including the tokens it generated. It is preferred even when a
snapshot matches the same length, because the state is already sitting there:
no restore and no device copy. This is why continuing the previous turn of a
conversation is nearly free, and why reuse can cover generated tokens and not
just prompt tokens. A request that continues no live frontier takes the least
recently used idle session, so it does not replace the frontier of a
conversation that has just replied.

A **snapshot** is an immutable retained checkpoint. Reusing one means restoring
it into a session, which costs a copy.

Both require an **exact prefix match**. The cache still computes the longest
common prefix on a miss and reports it, which is why a miss can say "we agreed
on 7,094 of your 7,103 tokens" and still re-prefill everything: it had no
checkpoint at that position to resume from.

Chat prompts mark a stable boundary before the assistant framing. A
checkpoint past that boundary, including an exact retry of the complete
prompt, is reused only while a checkpoint at or before the boundary is
retained too, so a client that rewrites the assistant turn can still branch.
When a prompt ends with a text-only user turn that follows a tool result or
another user message, the boundary moves before that turn: agent clients send
per-turn runtime context there and replace it with the next request. A user
turn after an assistant reply is ordinary chat and keeps the boundary before
the assistant framing.

### Worked example

A conversation with a 5,000-token system prompt, thinking off, client replaying
the assistant turn verbatim, each reply 100 tokens long:

| Turn | Prompt | Reused | Prefilled | Why |
| --- | ---: | ---: | ---: | --- |
| 1 | 5,020 | 0 | 5,020 | nothing retained yet |
| 2 | 5,320 | 5,120 | 200 | live frontier of turn 1, with its reply |
| 3 | 5,620 | 5,420 | 200 | live frontier of turn 2 |

Now change one thing: edit the *last user message* of turn 3 near its end and
resend. The first 5,599 tokens are identical. The longest usable retained
checkpoint wins, which here is the turn-2 frontier at 5,420 that turn 3 froze
before its prefill. If that has been evicted, the intermediate checkpoint at
4,096 that turn 1's cold prefill kept can be restored and only the suffix is
prefilled. If earlier checkpoints have been evicted or refused by the byte
budget, and all remaining checkpoints fall after the edit, reuse is
impossible:

| Turn | Prompt | Reused | Prefilled | Common prefix |
| --- | ---: | ---: | ---: | ---: |
| 3', intermediate checkpoint retained | 5,620 | 4,096 | 1,524 | 5,599 |
| 3', no usable earlier checkpoint | 5,620 | 0 | 5,620 | 5,599 |

That gap between "common prefix" and "reused" is the signature of the
exact-prefix limitation. See #331.

## Concurrent requests sharing a prefix

Checkpoints help a request only once they exist. When several requests that
share a prefix arrive together, such as subagents started from one system
prompt or a retry sent before the first attempt returns, none of them finds a
checkpoint, and each would prefill the shared tokens again. Prefill runs one
request at a time, so the copies cost the sum of all prefills, and all
requests finish together at the end.

Instead, a cold request that shares a long prefix with a request already
prefilling it waits for that one:

```mermaid
sequenceDiagram
  participant L as Leader (resident)
  participant S as Scheduler
  participant F as Follower (arrives later)
  F->>S: arrives, shares N tokens with L
  S->>L: publish a checkpoint at N
  Note over F: parked, keeps its slot but no state
  L->>L: prefill to N, capture
  S->>F: admit
  F->>F: restore N, prefill its own tail
```

1. On admission, the scheduler compares the new prompt with every resident
   request still prefilling and takes the longest exact token prefix with
   matching input identity.
2. It waits only when the shared prefix is at least 512 tokens longer than
   what the request could already restore, the resident request has not yet
   prefilled past that point, and the runner supports snapshots. A
   conversation continuing from its own retained turn therefore never waits
   for a newcomer that shares only its system prompt.
3. The resident request stops its prefill at the shared position and
   publishes a RAM checkpoint there (`PoolRequest::share_checkpoint`), as for
   intermediate checkpoints. A checkpoint it already plans within 64 tokens
   is used instead. Identical prompts share up to the leader's last prompt
   token or its stable boundary.
4. The waiting request holds no execution state, but keeps the admission slot
   it reserved, so newer arrivals stay queued and `/slots` shows it
   processing. Once the checkpoint is retained, or the leader passes the
   position, finishes or is cancelled, it resumes ahead of newer requests and
   acquires its cache as usual.

Because requests are waiting for it, that checkpoint is retained with the
priority of a conversation's stable boundary: on a full cache it replaces
other checkpoints, lowest priority first, rather than being refused as an
optional copy. Waiting is still advisory, like admission: if the checkpoint
does not fit at all or the leader is cancelled, the request prefills by
itself. A request waits at most once, and at most `--sessions - 1` requests
wait at a time. Requests with `cache_prompt: false` never wait. With
`--sessions 1` nothing waits, because a second request is never admitted
while the first prefills.

Measured on Flash-Next with `--sessions 4` and a 3.8k-token shared system
prompt, four concurrent requests took 13.4 s with each prefilling 3.9k tokens.
Prefilling the shared prefix once and restoring it took 4.6 s, each request
then prefilling 115 tokens **(measured)**. With a 242-token shared prefix the
gain was about 1 s of 9 s, which is why short prefixes do not wait.

## What gets retained

The unit of retention is a **checkpoint**, not a conversation. A single request
can retain up to four intermediate checkpoints in addition to its branching
fallback and complete prompt. Prefill stops at those positions and captures the
whole model state before advancing. Positions lie on a 2,048-token grid spread
across the prompt; the final grid point is within 2,048 tokens of its end.
Warm continuations skip grid positions less than 2,048 tokens beyond the reused
frontier, and every request skips grid positions within 128 tokens of its end:
such a point would split the final prefill pass for a checkpoint next to the
complete prompt. Capture runs asynchronously while that session is frozen, so other
requests can continue; a single execution slot skips the extra worker. A
request whose first token is already published decodes as soon as its capture
finishes, before the next bounded prefill chunk of a peer.
Coincident RAM and disk boundaries share one copy. Admission
remains subject to the existing byte budget, and intermediate copies preserve
the original branching fallback. These intermediate checkpoints live in RAM;
the disk tier continues to retain prompt and learned shared-prefix boundaries.

### Learned divergence points

Grid checkpoints sit at fixed positions, not where conversations actually
diverge. Agents start many conversations with the same system prompt and tool
definitions, then a different task:

```text
Conversation 1:  [ shared 4,500 tokens ][ task A, 10,000 tokens ]
grid checkpoints:       ^2,048    ^6,144    ^10,240    ^14,336
```

Only the 2,048 checkpoint lies inside the shared part, so a second
conversation would restore 2,048 tokens and prefill the other 2,452 shared
tokens again. The longer the tasks, the further the grid spreads.

The point that matters, token 4,500, is unknown until a second prompt shows
where the two differ. When a request arrives,
`ContinuationCache::common_prefix_tokens` compares it with the tokens of every
retained checkpoint and live frontier of the same input identity. If the
longest common prefix adds at least 512 tokens to what the request can
already restore, the request stops its prefill there and retains an extra
checkpoint. Other conversations depend on it, so it is retained like a stable
boundary, not as an optional copy, and the branch-point rule below keeps it
while it is shared. From the third conversation on, every new one restores
the whole shared prefix and prefills only its own task. A divergence within
the request's last 64 tokens, such as an edited final message, is covered by
its own stable checkpoint and takes no extra copy. The same position from the
same prefix family is captured once; records whose images differ from the
prompt are not used to find it. Without disk, this needs no configuration.

Execution and retention have separate limits. `--sessions N` allocates N
mutable execution states and controls active request concurrency. A separate
pool holds **128 immutable checkpoint records**
(`TextRunnerRamCacheOptions::MAX_ENTRIES`), regardless of session count.
These records allocate no extra execution states: with `--sessions 1`,
several inactive conversations can still be remembered.

This is a checkpoint limit, not a guaranteed conversation count. One history
can occupy several records, and the byte budget may bind before all 128 are
used. The server identifies related checkpoints by exact token prefixes and
matching input identities, not a client-supplied conversation ID.

When space is needed, RAM retention prefers removing:

1. A full-prompt retry copy with a stable fallback still retained.
2. An intermediate copy with a related stable continuation retained.
3. An older stable boundary covered by a newer one. A checkpoint that two
   retained conversations extend and then diverge after is the prefix they
   share, not an older turn, and never counts as covered. A learned
   divergence point stays protected after the older branch is gone, since a
   client that rewrote the prompt there (a chat bridge replaying its last
   user message) diverges there again; a deeper learned point of the same
   family supersedes it, and republishing exactly that prefix keeps it.
4. Remaining checkpoints, oldest-used first within each priority.

A continuation being admitted counts as the newer copy for the checkpoints it
extends, so a new conversation replaces its own earlier copies first.

Under entry pressure, an edited branch can first replace its incompatible
tail, preserving earlier shared checkpoints. Optional history/retry copies
are skipped rather than removing another prefix family's last useful copy.
Retry copies cannot displace earlier history or stable boundaries either.
An unchanged retry then restores the stable boundary and prefills only the
assistant opening, such as `<|im_start|>assistant\n<think>\n`, again. If
the prompt ends with a text-only user turn after a tool result or another user
turn, Qwen templates place the stable boundary before that turn, so the retry
prefills it too. The server logs `event=snapshot action=skipped` with that
prompt's length.
A new stable boundary can replace its own older boundary when that avoids
removing another family's last copy. Sources are rechecked before replacement
because another request may have changed the record.

The original resume point stays protected during intermediate/retry admission.
Earlier checkpoints remain best-effort: under pressure, edits and rewind may
need more prefill. Rotation beyond the actual memory or record capacity can
still lose reuse; this is not unlimited retention.

The actual selected limits are reported at startup:

```text
event=snapshot_cache_configured sessions=1 snapshot_entries=128 capacity_bytes=8589934592 automatic_bytes=8589934592 max_bytes=12884901888
```

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

## Limits

Independent limits can each bind first. Payload memory is allocated only when
captured; RAM-retained snapshots and their captures in progress share the RAM budget.

| Limit | Default | Set by |
| --- | --- | --- |
| Retained snapshot bytes, RAM | smaller of 32 GiB and half the available host RAM; an explicit value up to the available host RAM minus 4 GiB; 27B also checks HIP free memory | `--cache-ram-bytes` |
| RAM checkpoint records | 128, independent of `--sessions` | internal safety limit |
| Disk bytes | 8 GiB | `--cache-disk-bytes` |
| Disk staging bytes | smaller of an eighth of the available host RAM and the disk budget | `--cache-disk-staging-bytes` |

`--cache-ram-bytes 0` selects automatic sizing. Zero does not disable reuse.
A positive value replaces the automatic budget and may exceed it: it trades the
free half of host RAM for retention, but always leaves 4 GiB to the OS and other
processes; larger values are clamped to that ceiling. For an 8 GiB cap, use
`--cache-ram-bytes 8589934592`. The startup line reports the selected
`capacity_bytes`, the `automatic_bytes` budget and the `max_bytes` an explicit
value may claim. With a DFlash2 draft, Qwen3.8-27B captures prompt features
into one pinned host buffer per model, which grows to the longest prompt and
serves every session; it is not charged to `capacity_bytes`, and its growth
after startup is not subtracted from the budgets
([implemented differently](RUST.md#implemented-differently)).

Both are sampled after weights and execution states are allocated, then fixed
for the server run. DeepSeek V4 Flash, Flash-Next and Qwen3.8-27B use half the
available host RAM, respecting container/cgroup limits. Available RAM is
`MemAvailable` minus `CmaFree`: free CMA pages, such as the kernel's KHO
scratch area on Ubuntu 26.04, only hold movable pages, not GPU allocations.
Inactive page cache, such as recently read model files, counts as available
because the kernel reclaims it before failing an allocation. Dirty and
writeback pages do not. 27B also clamps the budget and the ceiling to HIP's
free device memory. CPU and GPU allocations compete for the same physical RAM
on Strix Halo, so HIP's free-memory estimate alone is not enough.
For example, 44 GiB available at load gives a 22 GiB automatic RAM cache and
allows an explicit limit of up to 40 GiB, even when HIP reports more free
memory.
Memory that another process claims after startup is not subtracted. When
other GPU servers share the machine, such as an image or speech server, set an
explicit `--cache-ram-bytes` that leaves room for their peak use; starting
them first only accounts for what they hold at that moment.
The RAM payload budget excludes weights, execution states, token metadata,
disk staging and Qwen's separate CPU image-preprocessing cache (up to 128 MiB
of retained encoded/RGB vectors plus bounded metadata). It also excludes temporary
host buffers used to save checkpoints to disk; this is not a limit on total
server memory.

The 32 GiB automatic cap limits default growth on a lightly loaded machine.
It leaves room for several 27B histories: two 3.7 GB checkpoints per history
would consume about 30 GB for four conversations, before optional copies.
The 128-record safety limit bounds metadata and lookup work without requiring
more execution sessions. Neither guarantees a conversation count: roughly,
divide the byte budget by the retained checkpoint bytes per conversation.
For example, 8 GiB holds at most eight sets of two 512 MiB checkpoints, or
two sets of two 2 GiB checkpoints, before extra copies and in-flight captures.

Increasing context or sessions reduces the memory left after loading, and
with it both budgets, exactly when snapshots grow. On Flash-Next with
`--sessions 2` at 262144 context, 14.8 GB remained after loading: a 7.76 GB
automatic budget against 5.70 GB snapshots at 200k tokens, so only one deep
conversation stays retained **(measured)**. A newer conversation's stable
boundary then replaces the older one's checkpoint; only optional intermediate
and retry copies are refused. This is a hardware ratio, not a cache policy:
weights, execution states and the OS leave no more memory. To keep more deep
conversations:

- Raise `--cache-ram-bytes` towards the reported `max_bytes`: about 10.5 GB
  here, enough for a 200k-token checkpoint plus a 50k one instead of only the
  first.
- Run fewer `--sessions`: each Flash-Next session at 262144 context holds about
  6.3 GB of state that the cache can use instead.
- Add `--cache-disk` with `--cache-disk-staging-bytes` above the snapshot size,
  so checkpoints that leave RAM can still be restored from disk.

See #343.

Snapshot size scales with retained tokens and differs sharply between models:
roughly 0.5 GB at 5k tokens on Flash-Next, and 3.7 GB at 24k tokens on
Qwen3.8-27B **(measured)**. Automatic disk staging therefore has no fixed cap:
it follows available RAM and the disk budget, so `--cache-disk` keeps 27B
checkpoints at moderate depth without an explicit limit. See #259.

RAM eviction follows the checkpoint priorities above. Disk eviction remains
global least-recently-used, without conversation or rebuild-cost awareness.

## The disk tier

`--cache-disk DIR` adds a second tier that survives restarts. A checkpoint
written before a restart restored a 24,866-token prompt in 1.8 s against about
50 s for a cold prefill **(measured)**.

Entries are keyed by the model's compatibility identity, the input identity
and the exact tokens; files are checksummed, published atomically and
owner-only, and interrupted or invalid files are removed at startup. Startup
reads each file's header, identity and tokens only, so its cost no longer
grows with the stored bytes: the payload checksum is verified when an entry
is restored, which removes a corrupt file and tries the next candidate. Each
checkpoint is one file, written in 1 MiB pieces; it is serialised and
restored as a whole payload, so a disk transfer stages the whole checkpoint
in host memory.

**A request waits for a checkpoint still being written** when its prompt
extends it, the checkpoint saves more than 4096 tokens over what a lookup
reuses now, and recent prefill steps say how long prefilling them would take.
It waits at most that long, and at most 120 s, and resumes as soon as the
write finished or the prefix is available otherwise; `event=publication_wait`
logs `stage=start` and `stage=end` with the `release` reason (`not_pending`,
`available`, `cost_bound` or `cap`), or `stage=skip release=no_estimate`
before any prefill was measured. Agent loops whose next turn arrives while the
previous checkpoint is written to disk restore it instead of prefilling the
conversation again.

It also **learns shared-prefix boundaries** that survive restarts. On
admission, the store looks up where the prompt leaves the token paths of
stored entries without an entry there. The four longest such positions of at
least 128 tokens that lie beyond what the request restored become extra
prefill stops whose snapshots are written to disk
(`cache_shared_prefix_snapshots`). RAM learns a
divergence point from the second conversation and restores it from the third
(see above); the disk index needs more conversations, and in one run its
boundary became usable from the fifth **(measured)**. When both tiers choose the
same position, one capture feeds both. See #267.

**Disk checkpoints are spaced at least 2048 tokens apart.** A conversation
advances a few hundred tokens per turn, so writing every turn serialises, fsyncs
and retains a nearly identical snapshot. A checkpoint less than 2048 tokens past
the longest stored entry that is still a prefix of it is skipped and logged as
`min_step`; the RAM tier still retains it. When RAM does not retain it either,
the device state is not copied at all, so a skipped checkpoint costs no capture
time on the request path. Learned shared-prefix boundaries are
exempt, because they sit close to a deeper entry by construction. On a
Qwen3.8-27B functional run this cut disk writes from 31 to 5 and written bytes
from 6.97 GiB to 1.37 GiB **(measured)**.

The trade-off is what a disk-only restore resumes from. After a restart, or
once RAM evicted the conversation, a request restores the nearest stored entry
and re-prefills up to 2048 tokens plus its own turn, instead of resuming
exactly at the previous frontier. Greedy output is unchanged. Sampled output can
differ from the pre-restart response, because the re-prefilled gap follows
different chunk shapes, as with any partial hit. Plain continuation from RAM is
unaffected.

Disk entries are evicted by global LRU on last access, so one active run whose
checkpoints are all recent will displace every other conversation in age order
**(measured)**. See #275.

## What invalidates reuse

Anything that changes the token prefix. In practice:

- **Tool definitions** — adding, removing or reordering changes the prompt head
  and invalidates everything after it.
- **System prompt** — including anything volatile inside it, such as a
  timestamp or a working directory.
- **Response schema** — `response_format` is rendered into the prompt.
- **Images** — changing, removing or moving an earlier image invalidates
  checkpoints after it. Appending a new one does not.
- **Reasoning the client cannot replay** — thinking is on by default for Qwen
  and `reasoning_content` is not part of the OpenAI schema, so an ordinary
  client omits it when sending the conversation back. Gufo retains each turn's
  boundary before the assistant opening, so the next turn can reuse the prior
  history and process the rewritten assistant plus new input. That boundary
  must be saved on warm hits too; retaining only the old resume point caused
  cache reuse to stop advancing for the rest of the session. See #335.

These do **not** invalidate reuse **(measured)**: changing `temperature`,
`top_p`, `seed`, `max_tokens`, penalties or `stop` between turns; streaming
versus not, which reuse identically and interoperate within one conversation.

`cache_prompt: false` on `/v1/chat/completions`, `/v1/responses`,
`/v1/completions` or `/completion` bypasses lookup for a single request. The
result can still populate the cache. `null` keeps the default, `true`.

## Observing it

| Log line | Meaning |
| --- | --- |
| `event=snapshot_cache_configured` | retained capacity at startup, with the automatic budget and the most an explicit `--cache-ram-bytes` may claim |
| `event=disk_cache_configured` | disk and staging capacity at startup |
| `event=snapshot action=removed reason=entry_capacity` | a retained prefix was evicted because every entry was taken |
| `event=snapshot action=skipped reason=entry_capacity` | no checkpoint record could be replaced safely for this capture |
| `event=snapshot action=skipped reason=byte_capacity` | a checkpoint did not fit the RAM budget |
| `event=disk_cache action=removed reason=lru` | a disk entry was evicted to stay inside `--cache-disk-bytes` |
| `event=disk_cache action=skipped reason=staging_capacity` | a checkpoint did not fit `--cache-disk-staging-bytes` beside queued writes and was never written |
| `event=disk_cache action=skipped reason=min_step` | a checkpoint was less than 2048 tokens past a stored prefix; RAM still retains it |

`event=snapshot` lines also carry the checkpoint's `bytes` and `tokens` and the
`retained_bytes`, `reserved_bytes` and `capacity_bytes` of the RAM budget;
disk lines carry `file_bytes`, `payload_bytes`, `tokens`, `retained_bytes`,
`capacity_bytes`, `staging_capacity_bytes` and `staging_used_bytes`. Routine
replacement and byte-pressure removal in RAM are not logged.

Per-request outcomes appear in the completion log as `cache` (`memory`,
`disk` or `miss`), `cached_tokens`, `cache_restore_ms`,
`shared_prefix_wait_ms` and, on a miss, `cache_miss_reason`,
`common_prefix_tokens` and `nearest_checkpoint_tokens`. The Chat Completions
`usage.gufo` object reports `cache_hit`, `cache_disk_hit`,
`cache_miss_reason`, `cache_common_prefix_tokens` and
`cache_checkpoint_tokens` (the last three on a miss), `cache_restore_bytes`,
`cache_restore_ms`, `cache_snapshot_bytes`, `cache_disk_queued_bytes`,
`cache_shared_prefix_snapshots` and `shared_prefix_wait_ms` (time spent
waiting for a concurrent request, part of `queue_ms`). The llama.cpp-compatible
`timings` object has `prompt_n` (newly processed) and `cache_n` (reused). Miss
reasons are `no_checkpoint`, `prefix_changed`, `input_changed` and
`disabled`.

Reading them:

- `cache_n` growing turn over turn while `prompt_n` stays flat is healthy
  reuse.
- Concurrent requests with a shared prefix: one has `shared_prefix_wait_ms`
  zero and prefills the prefix; the others wait, then report the prefix in
  `cache_n` and only their own tail in `prompt_n`.
- `cache_n` **pinned** at the same value while `prompt_n` grows every turn is a
  checkpoint that stopped advancing. See #335.
- A large `cache_common_prefix_tokens` on a miss means a long prefix agreed and
  could not be resumed from: a mid-history edit, or an evicted checkpoint.

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

## Known gaps

Tracked under #336, with the upstream reports #331, #335, #259, #267, #275,
#300 and #318.
