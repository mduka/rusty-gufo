# Testing

Run the smallest check that covers the change. Correctness and model-owned
quality limits pass before performance results count. Production targets
Linux gfx1151; GPU tests need the device and run outside `nix develop`.

## Commands

```sh
# The gate before every commit: scoped to the packages the change reaches
tools/ci/check-rust.sh
tools/ci/check-rust.sh --full          # every test

# Focused tests (replace the package and test)
cargo test -p gufo-core                                    # host-only tests
cargo test --profile gpu-test -p gufo-qwen35 --features gpu,models --test decode

# Model QUALITY checks are ignored by default; run them explicitly
cargo test --profile gpu-test -p gufo-deepseek-v4-flash \
  --features gpu,models --test quality -- --ignored
tools/ds4/check.py model --model <gguf> --dspark-model <support gguf>

# The DeepSeek kernel operator checks (standalone HIP programs)
tests/models/deepseek_v4_flash/run_operator_checks.sh

# Formatting of the hand-written C++ and HIP, with CI's pinned formatter
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting.

# Documentation changes
python3 tools/ci/check-docs.py --root .
```

Repository mechanics are in [development](DEVELOPMENT.md); the gate and how
it selects packages are in [Gufo in Rust](RUST.md#tests-and-the-gate).
GPU tests sit behind each package's `gpu` feature, tests that read model
files behind `models` or `#[ignore]`. A test whose model or device is absent fails with an
explicit message; that is not a quality pass.

Model tests find their files in the Hugging Face cache
(`~/.cache/huggingface/hub`, at pinned revisions) unless an environment
variable names them; the names are those of the C++ tests where they exist.
Files without a pinned repository revision have no default:
`GUFO_FLASH_NEXT_MODEL` (the first shard of the Flash-Next UD-Q4_K_XL
trunk), `GUFO_FLASH_NEXT_MTP_MODEL` (its shared Q8_0 MTP sidecar),
`GUFO_QWEN27B_MMPROJ` (the 27B `mmproj-BF16.gguf`),
`GUFO_DEEPSEEK_V4_FLASH_MODEL` and `GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL` (the
IQ2XXS target and its DSpark support GGUF). Unset, the variable's name
stands in for the path, so the failure says what to set.

Fixtures that come from upstream's C++ engine or another reference name
their source: a program under the package's `tests/reference/`. Those
programs build against a checkout of the upstream reference
(`$UPSTREAM`); each file's header gives the command. Their Rust
counterparts are the package's `examples/`.

Two test groups also read local artifacts that the repository does not
hold. The dense Qwen MTP test reads the test head that
`src/models/qwen35/tests/reference/mtp_test_head.py` derives from the Qwen3.6 27B MTP
GGUF (`GUFO_QWEN27B_MTP`, default `target/qwen-mtp/qwen36-mtp-head.gguf`).
The Flash-Next forward and session tests read fixtures in the layout of the
C++ dumps of `src/models/qwen38_flash_next/tests/reference/cpp_*.cpp`
(`GUFO_FLASH_NEXT_FIXTURES`, default `target/flash-next`, 2.4 GB), recorded
by a C++ build of the reference release with the same ROCm as the Rust
build: the kernels are upstream's, but another compiler orders their
arithmetic differently. A fresh checkout points these variables at an
existing copy.

## Local checks

Hosted runners have neither a gfx1151 nor the ROCm the binary links, so
hosted CI ([ci.yml](../.github/workflows/ci.yml)) runs only what needs
neither: rustfmt, the unsafe policy, the tool tests, the functional planner,
runner and cache-workload tests, the documentation and dependency inventory
and clang-format. The gate runs on the development host and covers formatting,
clippy, the GPU and model tests of the packages a change reaches (all of
them with `--full`), the kernel extent probes, the DeepSeek operator checks
when their kernels change, the unsafe policy, the tool tests (A/B harness,
gate scope, kernel resources, documentation and dependency checks,
profiler, serving benchmark, speculative corpus), the DeepSeek
dataset provenance, the CLI parser (`tests/cli/serve_test.py`), the
documentation and the dependency inventory. Model QUALITY checks,
long-context sweeps and A/B runs against another binary
(`tools/bench/ab.py`, whose defaults are the gates of
[Gufo in Rust](RUST.md#performance-against-upstream)) are explicit.

The text API functional suites ([tests/functional](../tests/functional/README.md))
start `target/release/gufo serve llm` with local weights and check real
responses through the OpenAI SDK; `impact.py` selects the suites a change
reaches. They are explicit, like the QUALITY checks.

## Match validation to the change

| Change | Relevant checks |
| --- | --- |
| Documentation | local links, anchors, JSON examples |
| Loader/container/tokenizer | independent fixtures, malformed/truncated input, exact token IDs |
| Kernel or quantizer | analytic oracle, affected layer/model boundaries, declared quality limits |
| Attention or state layout | long-context boundaries, snapshot/restore, fresh-versus-reused state |
| Sampling/speculation | distributions, proposal/rejection/residual draws, seeded replay, EOS commit |
| Scheduler/server | direct-versus-HTTP behavior, independent batched requests, multi-turn reuse, cancellation, failure and shutdown |
| Performance | affected quality gates followed by one comparable release measurement |

Broaden testing when a change crosses shared ownership boundaries or a failure
reveals an unresolved risk. Full model sweeps, media generation, and repeated
timing runs are not the default development loop.

Model-owned procedures and outstanding qualification gaps live with the model:

- [DeepSeek V4 Flash](models/deepseek-v4-flash/QUALITY.md)
- [Qwen3.8 27B](models/qwen3.8-27b/QUALITY.md)
- [Qwen3.8 Flash-Next](models/qwen3.8-flash-next/QUALITY.md)
- [Qwen3-TTS](models/qwen3-tts/QUALITY.md)
- [Qwen3-ASR](models/qwen3-asr/QUALITY.md)
- [EmbeddingGemma 2](models/embeddinggemma-2/QUALITY.md)
- [Qwen-Image-2.1](models/qwen-image-2.1/QUALITY.md)

## Independent references and artifacts

Prefer hand-checkable analytic fixtures or the source checkpoint in a pinned
official runtime. A promoted Gufo build is a regression reference; agreement
with it alone does not establish upstream correctness. Record checkpoint
storage type separately from compute and accumulation precision.

Unchanged-arithmetic refactors retain exact boundaries. Reassociated kernels
need declared numerical limits and model-level checks. Quantization is
evaluated against a full-quality teacher with distribution, perplexity, and
capability gates. Do not loosen tolerances or silently replace goldens to
admit a change.

Golden artifacts record immutable model and oracle revisions, file hashes,
tokenizer/template identity, prompt/token-stream hashes, dtypes, parameters,
and schema version. Large tensor/logit dumps stay outside Git. Keep small
independent fixtures and concise results needed to reproduce qualification;
see [benchmark artifact retention](BENCHMARKS.md).

## Matched-token and layer comparisons

Tokenize once and feed the same predetermined history to teacher and
candidate. Compare full next-token distributions at each position before
appending the next evaluation token. Free-running text diverges after one
different choice and cannot isolate numerical or quantization error.

Relevant metrics include non-finite counts, normalized-logit error, KL,
teacher-token NLL/perplexity, top-1 agreement, and top-k overlap. See
[benchmark quality methodology](BENCHMARKS.md#quality-method-matched-token-per-position).
For a discrepancy, capture the earliest changed embedding, normalization,
attention/recurrent state, routed expert, layer output, or LM-head boundary.
Preserve the failing input and add a focused reproducer before fixing it.

Pin sampling controls, seeds, cache history, and batching when testing replay.
Run longer multi-turn cases when session reuse or EOS handling changes.
Do not average away numerical mismatches or classify device failures as passes.

## Performance evidence

Compare identical model artifacts, prompt/output counts, context depths,
batching, sampling, and cache state. Measure request latency and aggregate
throughput separately. Run baseline and candidate close together; add
repetitions only when noise or a suspected regression requires them.
Record hardware/software identities and the exact timed scope. Use
[profiling tools](PERFORMANCE.md) to explain a gain before retaining it.
