# Gemma 4 experiments

What has been tried on this engine for Gemma 4. The extension's own
development notes are in its
[repository](https://github.com/franzmoca/gemma4-gufo-extension/tree/a31ce42/docs/models).

## How the port differs from the extension

- **One stream per model.** Every session's work runs on the model's stream
  under its lock; gufo-hip lets a kernel use only buffers of the stream it
  runs on. The experts run on a second stream beside the dense MLP, forked
  and joined as in the extension, so the arithmetic is unchanged.
- **Routing in three launches.** The extension routes a decode row in one
  kernel that also builds the expert groups. gufo-hip's launch contract
  cannot yet express an index buffer derived from another index output of the
  same launch, so routing, the expert counts and the groups run as three
  launches. A proposed fix: an index derivation in gufo-hip and a
  `derived_from` on an output of the same launch in the kernel contract.
- **Copies.** The extension's `CopyRuns` kernel copies many runs through raw
  pointers, also into registered host memory; the contract has no kind for
  that. Snapshots and restores use gufo-hip copies (about a thousand
  asynchronous copies per snapshot), and the draft head's row order and table
  rows are gathered with a row-gather kernel.
- **Logit candidates.** The adapter clears the candidate workspace before
  every launch (about 0.5 MB for 16 rows); the extension clears it once.
- **Sliding caches** always hold the full ring, also for a context shorter
  than the ring.
- **Vision upload** stages the weights through two pinned 16 MiB buffers; the
  encoder runs under a lock, so it can be shared between requests.

## Measured

Single runs on 26B-A4B UD-Q4_K_M against the extension's probe, not clean A/B
measurements:

| Change | Before | After |
| --- | --- | --- |
| Experts on a second stream | decode 53.1 against 54.7 tokens/s | 57.0 against 57.6 tokens/s |

## Not tried yet

- The first prompt of a process is about 20 ms slower than the extension's
  (per-width index tables, synchronous uploads, the first hipBLAS call are
  suspects).
- Graph capture of the decode and verification forwards.
- In-pass checkpoints (`prefill_through`) for the serving cache.
- The prompt lookahead hook of the serving pool (Gufo's `SetPromptLookahead`
  in the extension's base): evaluates the generation prompt with the
  prompt's last forward.
- Loading the vision sidecar and its first encode take about 60 ms more than
  in the extension; the first encode alone about 40 ms (293 against 257 ms for
  768×768, 303 against 264 ms for 1056×576 on the released sidecar).
