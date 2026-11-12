# Nemotron 3.5 Bolt experiments

What has been tried on this engine for Nemotron 3.5 Bolt. The extension's own
development notes are in its
[repository](https://github.com/gufo-extensions/nemotron/tree/62f85dc/docs/models/nemotron-3.5-bolt).

## How the port differs from the extension

- **One stream per model.** The extension gives each session its own stream.
  gufo-hip lets a kernel use only buffers of the stream it runs on, and a
  batched decode writes the state of several sessions, so here the model owns
  one stream and hipBLAS handle that every session's work runs on under a
  lock. Device work is serialized by the serving runtime anyway.
- **Direct launches.** The extension captures the one-token decode and each
  width of the batched decode in a HIP graph; this port launches the same
  kernels directly. The batched-row kernels read the sessions' buffers
  through a device table that each launch fills from a checked host list, so
  they could not be captured as they are.
- **Drafter checks.** The drafter downloads its top-8 candidate IDs and checks
  them before the candidate embedding reads them (one extra host round trip
  per draft): a row with fewer than eight finite logits pads with `INT32_MAX`,
  which the C++ would use as an index.
- **Projector load.** Each BF16 matrix of the vision projector is staged and
  converted on its own; loading and preparing took 12.3 s and 5.5 s against
  about 8 s for the extension. Not measured further yet.

- **Not ported from 0.2.3.** The extension splits a prompt pass beside
  decoding sessions into layer groups and pauses the vision encoder about
  every 250 ms, through the scheduler's layer-group yields (Gufo #1039, after
  v0.28.1).

## Tried

- **The shared expert on a side stream** (gufo-hip's fork, as the extension
  does): with the shared expert first on the main stream, verifying eight
  tokens took 52.4 ms against the extension's 50.4 to 51.0 ms, and the gap
  grew with the block length, since the BF16 products of 0.2.2 shorten the
  routed experts; on the side stream 51.0 ms against 51.0 ms, and a
  one-token step 27.9 against 28.0 ms (was 28.4). Clean runs in the order
  C++, Rust, Rust, C++, November 11, 2026.
- **Verified rows kept across rounds**, as the extension's thread-local
  buffers keep them, instead of a cleared 4 MB buffer per round.

## Not tried yet

- Graph capture of the one-token decode and of the batched decode (a
  device pointer table that gufo-hip can check would let the row kernels
  replay).
- Index buffers that the drafter's top-8 kernel writes under a checked
  contract, which would remove the drafter's extra round trip.
- Reusing the prefill routing buffers across chunks instead of allocating them
  per chunk.
- A disk cache (persistent snapshots) for this model.
