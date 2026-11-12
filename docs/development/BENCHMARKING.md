# Benchmark methodology

Build with `cargo build --release -p gufo` and measure `target/release/gufo`.
Current numbers of rusty-gufo are on the user [performance](../PERFORMANCE.md)
page, its speed against Gufo's C++ engine in
[relation to Gufo](PARITY.md#performance-against-gufo). Per-model results of the C++
engine, with their settings and qualification gaps, are Gufo's records:

- [DeepSeek V4 Flash](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/deepseek-v4-flash/BENCHMARKS.md)
- [Qwen3.8 27B](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-27b/BENCHMARKS.md)
- [Qwen3.8 Flash-Next](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-flash-next/BENCHMARKS.md)
- [Qwen3-TTS](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3-tts/BENCHMARKS.md)
- [Qwen3-ASR](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3-asr/BENCHMARKS.md)
- [EmbeddingGemma 2](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/embeddinggemma-2/BENCHMARKS.md)
- [Qwen-Image-2.1](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen-image-2.1/BENCHMARKS.md)

## Model benchmark driver

`tools/bench/model-bench.py` measures benchmark tables over HTTP for
rusty-gufo and for the reference project of a model's category (llama.cpp
for Qwen; antirez/ds4 for DeepSeek), writes per-table artifacts, and renders
the tables and SVG charts into a Markdown page between `<!-- bench:<id> -->`
markers. Workloads and table layouts are declared in a model's
`artifacts/bench.json` (none is checked in yet); the `benchmark-model` skill
in `.agents/skills` describes the procedure.

Reference runtimes build from pinned commits against the installed ROCm with
`tools/bench/build-reference.sh`, into `target/references/<name>/bin`; they
are never part of rusty-gufo or the hosted checks:

| Reference | Build | Executable |
| --- | --- | --- |
| Qwen AR / DFlash2 | `llama-cpp` | `llama-server` |
| Flash-Next MTP | `llama-cpp-mtp` | `llama-server-mtp` |
| DeepSeek AR / DSpark | `ds4` | `ds4-server`, `ds4-bench` |
| Qwen3-ASR / Qwen3-TTS | `audio-cpp` | `audiocpp_server` |

For example, build the needed baselines and put them on `PATH`:

```sh
tools/bench/build-reference.sh llama-cpp
tools/bench/build-reference.sh llama-cpp-mtp
PATH="$PWD/target/references/llama-cpp/bin:$PWD/target/references/llama-cpp-mtp/bin:$PATH" \
python3 tools/bench/model-bench.py --model qwen3.8-flash-next --gguf "$MODEL" \
  --mtp "$MTP" run --target reference --table single-mtp
```

DeepSeek's HTTP payload omits pp/tg durations. The driver reads the pinned
server's existing stage timers from its log and checks them against HTTP token
counts. Native `ds4-bench` also supports single-session AR/DSpark, but not
concurrency. The pinned ROCm server disables DSpark when batching, so C>1
DSpark comparisons are **N/A**. See the
[DS4 method](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/deepseek-v4-flash/QUALITY.md#benchmark-method).

New or refreshed text-model cards use the same pp2048 prose/copying prompts
for single-user d0 and concurrency, with tg128. The driver shares their prompt
generator and records prompt hashes. C1 should cross-check d0; investigate
input, cache, controller and timing differences before attributing a gap to
batching. Short repeated-word corpora remain useful quality probes, but their
rates cannot replace these matched workloads.

For decode comparisons, set `prefill_first: true` in the concurrency table.
Prepare every session with the full prompt and one output token, wait for all
preparations, then time identical tg128 requests with prefix reuse. The driver
pins llama.cpp slots and rejects replay exceeding four prompt tokens. This keeps
peers' long prefills outside llama.cpp's elapsed generation timer; rusty-gufo reports
active decode time. Preserve a separate fresh AR completion-hash control.

## Direct and serving measurements

`gufo bench` measures the model path. Keep model artifact, prompt length,
generation length, context depth, speculative mode, and sampling controls
fixed when comparing runs. `pp2048` means 2,048 new prompt tokens; context
depth is separate from the amount of new prefill work.

`tools/serving/gufo-serving-bench.py` measures HTTP serving with synchronized
requests. It distinguishes server-stage prefill/decode throughput, scheduler
and client TTFT, token ITL, whole-request throughput, and aggregate concurrency
throughput. C1 serving and a direct single-user benchmark are comparable only
when prompt, cache state, sampling, speculation, and timed scope match.

Model tables report the **sum of individual request decode rates** in each
concurrent group, averaged across measured groups. Prefill and queue time
remain in the latency diagnostics. Pending measurements are **TODO**;
comparisons prevented by missing reference features are **N/A**, with a reason.
Whole-request throughput is not a substitute for decode timings.

`--endpoint-profile openai` supports other OpenAI-compatible servers. Missing
server-stage metrics remain null. `--reference-report` compares completion
hashes against an existing C1 reference; category filters select a focused
subset. Qualify acceptance with a fresh server so prompt-cache hits do not
hide work. See [serving benchmark commands](PERFORMANCE.md) for invocation.

`tools/bench/speculative-corpus.py` compares direct autoregressive and
speculative token traces. Request correctness, acceptance, and end-to-end
speed are separate requirements.

## Quality method (matched-token, per position)

Use the independent references and model-owned limits described in
[testing](TESTING.md). Tokenize once and feed teacher and candidate the same
predetermined history. Compare full next-token distributions before appending
the next evaluation token. Free-running text cannot isolate numerical error
after the first differing choice.

```text
p_t = softmax(teacher_logits)
p_c = softmax(candidate_logits)
KL  = sum_v p_t[v] * (log p_t[v] - log p_c[v])
NLL = -log(p_c[target_token])
```

Record non-finite counts, normalized-logit error, KL, teacher-token
NLL/perplexity, top-1 agreement, and top-k overlap as applicable. Aggregates
are over scored positions. A changed tokenizer, template, suite, or history
changes the comparison identity.

Capability evaluation is separate: [gufo eval](../CLI.md#benchmarks) grades free-running
answers through the OpenAI-compatible server.

## Focused kernel measurements

Standalone microbenchmarks under `tools/bench` (`tools/bench/build.sh`) and
model-specific tool folders exercise HIP kernels without loading a model and
report component correctness alongside timing. They are not part of the
binary, and their numbers do not replace end-to-end model measurements.
Collect profiler traces separately and use
[the performance guide](PERFORMANCE.md) to identify the limiting operation.

## Reporting and artifact retention

Record model/engine revisions, artifact hashes, hardware/software fingerprint,
power profile and CPU/GPU temperatures, input identity, context, output count,
sampling, concurrency, cache state, oracle identity, and the exact timed scope.
A numerical mismatch or device failure is not a passing result. Add repetitions
when timing noise or a suspected regression requires them.

Checked-in JSON retains fixtures, model provenance, controller calibration,
and bounded result summaries. Serving summaries preserve aggregate metrics,
per-case completion hashes, and the original report digest. Raw samples,
profiler traces, generated tuning databases, and full-logit captures stay in
ignored `artifacts/`, except for independent reference data needed by a
maintained correctness test.

Serving reports exclude endpoint hosts, prompts, generated text, model paths,
raw token IDs, and timestamps. The canonical machine fingerprint excludes
hostnames, usernames, secrets, timestamps, and private paths.
`gufo diagnose --validate-artifact` validates its supported diagnostic schema;
each model/tool owns the validation of its own result schema.

Downloaded tokenizer files are local artifacts.
