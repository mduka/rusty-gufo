# Testing

Correctness and the model-owned quality limits pass before performance
results count. Run the smallest check that covers a change; the GPU and model
tests need a gfx1151 with ROCm 10.1, and most model tests need the model's
weights.

## Test tiers

| Tier | Command | Needs |
| --- | --- | --- |
| Host | `cargo test -p <package>` | Nothing beyond the build prerequisites |
| GPU | `cargo test --profile gpu-test -p <package> --features gpu` | A gfx1151 with ROCm 10.1 |
| Model | `cargo test --profile gpu-test -p <package> --features gpu,models --test <name>` | The model's weights ([model files](#model-files)), some also [local fixtures](#local-fixtures) |
| Quality | the same with `-- --ignored`, e.g. `--test quality -- --ignored` | Weights; run on demand |
| Functional | [tests/functional](../../tests/functional/README.md) against `target/release/gufo` | Weights; run on demand |
| A/B and benchmarks | `tools/bench/ab.py`, `gufo bench` ([benchmarking](BENCHMARKING.md)) | An otherwise idle machine |

`gpu-test` is optimised like release, with debug assertions and overflow
checks. GPU tests sit behind each package's `gpu` feature, tests that read
model files behind `models` or `#[ignore]`. A test whose model or device is
absent fails with an explicit message; that is not a quality pass. Model
QUALITY checks are `#[ignore]` tests run explicitly:

```sh
cargo test --profile gpu-test -p gufo-deepseek-v4-flash \
  --features gpu,models --test quality -- --ignored
tools/ds4/check.py model --model <gguf> --dspark-model <support gguf>
tests/models/deepseek_v4_flash/run_operator_checks.sh   # DeepSeek kernel operator checks
```

## The gate

`tools/ci/check-rust.sh` runs, in order:

- `cargo fmt --all --check` and `cargo gpu-clippy` (clippy with every GPU
  feature, warnings as errors; `.cargo/config.toml` defines the aliases);
- the GPU and model tests (`cargo gpu-test`) of the packages a change reaches
  and their dependents: `tools/ci/gate_scope.py` maps each changed file to
  packages through Cargo's and `hipcc`'s dependency records; the kernel
  crates' tests include the extent probes of every launcher;
- the DeepSeek V4 Flash operator checks when their kernels change;
- clang-format of the hand-written C++ and HIP
  (`tools/ci/check-format.py`, with the formatter pinned in
  `tools/ci/requirements.txt`; `--fix` applies it);
- the unsafe policy (`tools/ci/check-unsafe.py`);
- the tool tests under `tests/tools/` (A/B harness, gate scope, runner and
  durations, kernel resources, documentation and dependency checks,
  profiler, serving benchmark, speculative corpus, DeepSeek dataset
  provenance) and the CLI parser test (`tests/cli/serve_test.py`);
- kernel scratch memory against `tools/ci/kernel-resources.json`;
- the documentation (`tools/ci/check-docs.py --root .`: required documents,
  local links and anchors, fenced JSON) and the dependency inventory
  (`tools/ci/check-dependencies.py` against
  [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)).

```sh
tools/ci/check-rust.sh                 # scoped to the packages the change reaches
tools/ci/check-rust.sh --long          # with the long test binaries
tools/ci/check-rust.sh --base <rev>    # scoped to what differs from <rev>
tools/ci/check-rust.sh --full          # every test of every package
tools/ci/gate-at.sh <rev> --full       # the same on a fixed revision, in a worktree
```

Some test binaries take minutes (the large-model tests above all). The
scoped gate skips binaries that took longer than `$GUFO_GATE_LONG_SECONDS`
(60 by default) in earlier runs on the same machine: every run records each
binary's time in a local file outside Git (`target/gate-durations.tsv`, or
`$GUFO_GATE_DURATIONS`, through `tools/ci/gate_durations.py`), and a checkout
without a record runs everything. `--long` runs them as well, and `--full`
runs every test regardless of what changed. The gate ends with the time of
each step and the binaries over the limit.

`--base <rev>` scopes the gate to the difference from a revision that passed,
which saves a rerun of everything after a rebase or after squashing commits.
`tools/ci/gate-at.sh <rev> [options]` checks out `<rev>` into a worktree of its
own (`target/gate-worktree`, with its own target directory) and runs the
gate there, so a long run checks exactly that revision while the checkout
moves on. There each test binary that passes is recorded for the revision
(`target/gate-passed/<commit>.tsv` in the worktree: the binary's SHA-256,
package, name and arguments), and a later run at the same revision keeps the
recorded binaries whose contents are unchanged: a run stopped at the end of
a window resumes with the binaries it had not finished, and its summary
counts those it kept. Delete the record for a run from the start. The
working tree's own gate keeps no record, since it cannot tell uncommitted
inputs apart.

### Hosted CI

Hosted runners have neither a gfx1151 nor the ROCm the binary links, so
hosted CI (a GitHub workflow, published after 0.1.0) runs what needs neither:
rustfmt, the unsafe policy, the tool tests, the functional planner, runner
and cache-workload tests, the documentation and dependency inventory,
clang-format, the dependency policy of [`deny.toml`](../../deny.toml)
(`cargo deny`: advisories, licences, duplicate versions, sources), and clippy
and the tests of the crates that link no ROCm library (`gufo-core`,
`gufo-eval`, `gufo-formats`, `gufo-kernel-build`, `gufo-serve`). Everything
else runs in the local gate on a gfx1151.

## Choosing checks

| Level | Checks | When |
| --- | --- | --- |
| Documentation and tools | `check-docs.py`, the changed tool's tests | Every documentation or tool change |
| Focused | `tools/ci/check-rust.sh`, or one test binary (`--test <name>`, the smallest model that covers the change) | Before every commit with code |
| Long | `check-rust.sh --long` or `--full`, the affected functional suites | Before a merge; `--full` for shared code |
| Measurement | A/B runs, benchmarks | On an otherwise idle machine |

- Shared code (`gufo-core`, `gufo-hip`, the scheduler and the server), the
  toolchain, ROCm and dependency updates take the focused level right away
  and `--full` before they are merged.
- A pull request states which checks ran. Contributors without every model's
  weights run the tiers they can; maintainers run `--full`, with all
  weights, before a merge.
- After a failure, rerun the failing test, not the whole gate.

## Model files

Model tests find their files in the Hugging Face cache
(`~/.cache/huggingface/hub`, at pinned revisions) unless an environment
variable names them; the names are those of the C++ tests where they exist.
Files without a pinned repository revision have no default:
`GUFO_FLASH_NEXT_MODEL` (the first shard of the Flash-Next UD-Q4_K_XL
trunk), `GUFO_FLASH_NEXT_MTP_MODEL` (its shared Q8_0 MTP sidecar),
`GUFO_QWEN27B_MMPROJ` (the 27B `mmproj-BF16.gguf`),
`GUFO_DEEPSEEK_V4_FLASH_MODEL` and `GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL` (the
IQ2XXS target and its DSpark support GGUF), `GUFO_NEMOTRON_BOLT_GGUF` (the
Nemotron 3.5 Bolt trunk; its drafter and projector are found beside it),
`GUFO_GEMMA4_MODEL`, `GUFO_GEMMA4_MTP_MODEL` and `GUFO_GEMMA4_MMPROJ` (a Gemma
4 target, its MTP drafter and its vision sidecar), `GUFO_QWEN35MOE_MODEL`,
`GUFO_QWEN35MOE_Q8_MODEL` and `GUFO_QWEN35MOE_MMPROJ` (unsloth's
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` and `Qwen3.6-35B-A3B-Q8_0.gguf`, which the
`gufo-qwen35moe` fixtures were recorded with, and the 35B's
`mmproj-BF16.gguf`), `GUFO_OMNIVOICE_MODEL` (the OmniVoice Hugging Face
directory), and `GUFO_VOXCPM_MODEL`, `GUFO_VOXCPM_GGUF` and
`GUFO_VOXCPM_Q8_GGUF` (the VoxCPM2 Hugging Face directory and audio.cpp's
`voxcpm2-bf16.gguf` and `voxcpm2-q8_0.gguf`). Unset, the
variable's name stands in for the path, so the failure says what to set.
`GUFO_QWEN35MOE_FIXTURES` names the local Qwen3.6 35B-A3B reference set
instead (default `target/qwen35moe`). Each model's
[page](../models/README.md) gives its files and how to download them; the
largest models take 80 to 100 GB each.

## Local fixtures

Most fixtures are in the repository. Three sets are too large and are
recorded locally; the tests that read them fail with the path they expect:

| Set | Variable (default) | Read by | Recorded by |
| --- | --- | --- | --- |
| Flash-Next C++ dumps (2.4 GB) | `GUFO_FLASH_NEXT_FIXTURES` (`target/flash-next`) | The Flash-Next forward and session tests | `src/models/qwen38_flash_next/tests/reference/cpp_*.cpp`, built against Gufo v0.28.1 with the same ROCm as the Rust build (each file's header gives the command): the kernels are Gufo's, but another compiler orders their arithmetic differently |
| Dense Qwen MTP test head | `GUFO_QWEN27B_MTP` (`target/qwen-mtp/qwen36-mtp-head.gguf`) | `gufo-qwen35`'s MTP test | `src/models/qwen35/tests/reference/mtp_test_head.py` from the Qwen3.6 27B MTP GGUF |
| Qwen-Image C++ dumps and PNG corpus | `QWEN_IMAGE_CPP_DUMPS`, `QWEN_IMAGE_PNG_CORPUS` (`target/image-ref/cpp`, `/png`) | Only the ignored recording and comparison tests of `gufo-qwen-image-21` | `src/models/qwen_image_21/tests/reference/cpp_reference.cpp` and the ignored `write_png_corpus` test |

The Qwen3-ASR tests, the Qwen3-TTS Base voice-clone tests and the ASR
benchmark also read Qwen's English example, which is Qwen's recording and
stays out of the repository: `asr_en.wav` of the Qwen3-ASR README
(<https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav>,
48 kHz mono, 15 s, SHA-256
`f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f`). Save it
as `~/.cache/gufo/fixtures/qwen3-asr-en.wav`, or name it with
`QWEN3_ASR_TEST_WAV` (ASR) and `QWEN3_TTS_REFERENCE_WAV` (TTS).

## Reference programs

Fixtures that come from a C++ reference or another implementation name their
source. The programs that record them live with their package:
`tests/reference/` (C++ dumps built against a checkout of the reference,
`$UPSTREAM`, and Python recorders), `reference/` (an official Python runtime
or a chat template), `tools/` (golden recorders of the Gemma 4, Nemotron and
OmniVoice packages) and, for tools shared by several tests, the repository's
`tests/models/<model>/` and `tools/models/<model>/`. Each file's header gives
the command. Their Rust counterparts are the package's `examples/`, which
write the same dumps for a comparison of the two builds.

Each model's checks against its reference, and the open gaps, are in its
`CHECKS.md` under [development/models](models/README.md).

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

## Independent references and artifacts

Prefer hand-checkable analytic fixtures or the source checkpoint in a pinned
official runtime. A C++ reference build is a regression reference; agreement
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
see [benchmark artifact retention](BENCHMARKING.md#reporting-and-artifact-retention).

## Matched-token and layer comparisons

Tokenize once and feed the same predetermined history to teacher and
candidate. Compare full next-token distributions at each position before
appending the next evaluation token. Free-running text diverges after one
different choice and cannot isolate numerical or quantization error.

Relevant metrics include non-finite counts, normalized-logit error, KL,
teacher-token NLL/perplexity, top-1 agreement, and top-k overlap. See
[benchmark quality methodology](BENCHMARKING.md#quality-method-matched-token-per-position).
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
