# Qwen3.8 27B: checks

Run only the affected [model suite](../../../../tools/qwen27b/check.py):

```sh
python3 tools/qwen27b/check.py fast
python3 tools/qwen27b/check.py kernels
python3 tools/qwen27b/check.py serving --model "$MODEL" --dflash-model "$DRAFT"
python3 tools/qwen27b/check.py model --model "$MODEL" --mtp-model "$MTP" \
  --dflash-model "$DRAFT" --mmproj "$MMPROJ"
```

The comparison against a BF16 reference GGUF (Gufo's
[quantization comparison](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-27b/QUALITY.md#quantization-comparison)) ran in the C++ test
`qwen27b_target_test`; a Gufo v0.28.1 checkout keeps it. The Rust engine
reproduces the C++ logits bit for bit, so the recorded result stands.

[Draft oracle](../../../../tools/qwen27b/dflash_reference.py),
[vision oracle](../../../../tools/qwen27b/vision_reference.py) and
[vision serving checks](../../../../tools/qwen27b/vision_check.py) retain source
pins and compare operators, full logits, RNG, image/state isolation and replay.
Full logits stay outside Git. A missing-weight skip is not a pass.

## Draft lookup, penalties and verification schedules

Draft probabilities change where the request has numeric penalties or the
draft copies its prompt history; the target distribution, p/q acceptance and
residual correction do not. The Rust checks:

- Prompt lookup (`src/models/qwen35/tests/dflash.rs`): greedy copies open
  after 12 matching tokens only once adaptive neural acceptance has dropped,
  after 24 under the fixed policy, and replay after memory and persistent
  restore. Sampled copies need 24 generated tokens, report point mass and
  consume no draft RNG; a continuation crossing `<|im_end|>` selects an
  earlier source, while real thinking and tool tokens stay copyable. A new
  request cannot resume the previous answer's copy, a v3 payload restores
  without lookup state, and lookup positions beyond the restored history are
  refused. The shared index's source filter and resume requirement have
  their own tests (`gufo_core::prompt_lookup`); Flash-Next's default source
  selection is unchanged.
- Penalized cohorts: two to eight sessions at distinct positions,
  temperatures and fixed/adaptive policies, with repeat, frequency and
  presence penalties, keep each session's tokens, selector probabilities,
  draws and persistent state byte for byte against drafting alone.
- Draft q (`src/models/qwen35/tests/sampling.rs`): eight penalty policies,
  including repeat-window expiry and negative penalties, match an
  independent CPU distribution conditional on the earlier drafts, and every
  draft lies inside its reported CDF. Bounded top-k candidates equal the
  full GPU sort and an independent stable CPU order for vocabularies of
  1,023 to 248,320 tokens, top counts 1 to 64, ties, signed zeros and
  nonfinite logits.
- Verification projections (`src/models/qwen/kernels/tests/small_batch.rs`):
  five-token IQ4_XS and three/four/eight-token Q4_K rows on the 27B gate/up
  and down shapes equal the decode GEMV byte for byte, with no write past
  their rows.
- Final prefill tile (`src/models/qwen35/tests/prefill.rs`, kernel test
  `attention.rs`): logits and continued cache state equal a forced
  full-layer capture for 2,048 rows at depth 0 and 1,025 rows behind 8,192
  cached tokens, with and without DFlash2 taps; the WMMA attention's
  retained rows equal the full call's bytes across packed head groups, and
  skipped rows stay untouched. The DFlash2 step fixtures hold; the Q4 haiku
  fixture records the C++ engine's copied verse at the same output.

Snapshots keep only the images their context consumed, so a request that
appends an image restores the earlier checkpoint
(`src/models/qwen/src/vision/prompt.rs`). A prefill cancellation check that
never fires keeps the logits bit for bit; one that fires inside the layer
stack stops the prefill, and the reset session prefills exactly again
(`src/models/qwen35/tests/prefill.rs`).
