# Qwen3.6 35B-A3B: experiments

Changes against the extension's C++ runner that keep its output bits, and
ideas not taken yet. Measurements on UD-Q4_K_M and Q8_0 with `machine-lock
perf`, alternating the two builds.

## Taken

| Change | Effect |
| --- | --- |
| MTP feedback replay (`AcceptFeedback`) writes only the K/V cache row of an accepted token and skips the rest of the step, which nothing reads | MTP `gufo bench` tg128 +1.4% (UD-Q4_K_M) and +3.6% (Q8_0) over the extension; same cache bits and tokens |
| Verified rows stay on the device for the MTP draft instead of a host round trip | part of the MTP gain above |
| The MTP draft allocates no FP16 cache plane, which it never reads | 2 × KV size × context × 2 bytes less per session (0.5 GB at 262,144 tokens) |
| Batched decode grows its logits buffer to the batch width instead of a fixed 64 rows | 4 MB instead of 63.6 MB at width 4 |

## Not taken yet

| Idea | Expected effect |
| --- | --- |
| Let the MTP head read the target's expert weights instead of copying the MTP block's experts | 0.49 GB (UD-Q4_K_M) or 0.86 GB (Q8_0) less memory |
| Chain the MTP draft steps on the device (a kernel writing a bounded token index) instead of reading each draft token back | about +0.3% with MTP |
| Merge the two ports of the draft vocabulary (`mtp/vocab.rs`, `dflash/draft_vocab.rs`) | less code |
| Pass the session to the draft backend's `propose`, `accept` and `restore` | MTP proposals after a disk restore use the session's image positions at once instead of from the next prime or append (tokens are unaffected) |
| The routes for other expert formats (per-slot GEMV, grouped BF16) | UD-Q6_K_XL and UD-Q8_K_XL files |
| The 8-bit KV cache (`--kv-cache q8_0`) | half the attention KV memory |
