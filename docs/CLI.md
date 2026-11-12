# Command-Line Interface

The `gufo` executable provides terminal interfaces for:

- Interactive conversations.
- One-shot prompt testing.
- Piped and scripted generation.
- Benchmarking a model and evaluating a running OpenAI-compatible server.
- Checking the system the engine runs on.

Terminal generation and HTTP serving share model runtimes, tokenizers, and
sampling. The HTTP scheduler also manages concurrent requests and persistent
prompt caching.

## Options by purpose

The options below are grouped by what they do. Each is a long option of the
commands it applies to (`gufo serve llm`, `prompt`, `chat`, `bench`);
`gufo <command> --help` lists exactly which command takes which, with its
default.

The supported model types are LLM, image, audio (TTS, ASR) and embeddings.
`gufo serve embedding` provides [EmbeddingGemma 2 embeddings](SERVER.md#embeddings)
of text, images, video and audio.
`gufo serve image` provides [Qwen-Image-2.1 generation and editing](models/qwen-image-2.1/README.md).

### LLM or text

**Model identity**

- `--served-model-name` — the name the server reports to clients (for example in
  `/v1/models`). In practice: what your client's "model" field should say.
  Purely cosmetic, it does not change which weights are loaded.

**Size and memory**

- `--context` — maximum context capacity; memory use depends on the model and
  the number of resident sessions.
- `--enable-yarn` — Qwen3.8-Flash-Next only: allow a `--context` above the native
  262144 tokens through static YaRN, which applies to every request. Required
  above 262144 and rejected at or below it.
- `--prefill-chunk` — prompt tokens processed between active decode steps in
  HTTP serving. Model runtimes choose their own internal prefill batch sizes.
- `--max-tokens` — maximum number of tokens a request may generate. In practice:
  stops a model that rambles forever.

**Sampling (how the next token is picked)**

- `--temperature` — randomness of the choice. Low (for example
  0.2) gives predictable, repetitive answers; high (1.0+) gives varied,
  creative ones. Greedy decoding is the low extreme. Unset sampling options
  follow the model's recommended preset for the thinking mode (Qwen3.8:
  1.0 / top-p 0.95 / top-k 20 with thinking, 0.7 / 0.8 / 20 with presence
  penalty 1.5 without); `bench` stays greedy.
- `--top-p` — only consider the smallest set of tokens whose probabilities add
  up to P (nucleus sampling). In practice: a "keep only plausible tokens"
  filter; lower means safer output.
- `--top-k` — only consider the K most likely tokens. The same idea as `--top-p` but
  with a fixed count instead of a probability mass.
- `--min-p` — discard tokens whose probability is below a fraction of the best
  token's probability. A modern alternative to `--top-p`/`--top-k` that stays
  consistent across models.
- `--min-keep` — always keep at least N candidates after those filters. In
  practice: prevents the filters from ever leaving just one choice, so the
  sampler keeps working as intended.
- `--seed` — fixed random seed. Same seed plus same prompt gives the same output,
  which makes runs reproducible.
- `--repeat-penalty` — makes tokens that already appeared less likely. In
  practice: the anti-loop knob, stops "the the the the".
- `--repeat-last-n` — how far back the repeat penalty looks. Larger windows catch
  loops that build up over long outputs.
- `--frequency-penalty` — a penalty that grows the more often a token has already
  been used. Reduces word-for-word repetition.
- `--presence-penalty` — a flat penalty for any token used at least once. Pushes
  the model toward new words and topics even if it used each only once.

**Reasoning models**

- `--think` (`"on"` / `"off"` / `"auto"`) — whether the model produces its
  visible thinking before answering. In practice: off for quick chats, on
  for hard problems. Auto uses the model default: enabled for Qwen3.8 and
  DeepSeek V4 Flash.
- `--reasoning-effort` (`"auto"`, `"minimal"`, `"low"`, `"medium"`, `"high"`,
  `"xhigh"`, `"max"`) — model instructions controlling reasoning depth,
  not a token limit. Auto selects Qwen `xhigh` or DeepSeek `high`; more effort
  can increase response time and generated tokens.
- `--preserve-thinking` (`"on"` / `"off"` / `"auto"`) — whether earlier turns'
  thinking is kept in the conversation history. In practice: keeping it can
  help follow-up questions that build on the previous reasoning, at the cost
  of eating context.

**Disk cache**

- `--cache-disk` — enable an on-disk cache of computed prompt state, reused
  across requests and restarts. In practice: repeated or shared prompts get
  much faster; costs disk space.
- `--cache-disk-bytes` — the maximum size of that cache on disk.
- `--cache-disk-staging-bytes` — host RAM that bounds queued captures, disk writes
  and each disk read; a checkpoint is staged whole, so it must fit. See
  [the server guide](SERVER.md) for the defaults.

**Speculative decoding**

A draft model proposes tokens for the target model to verify. Greedy decoding
must reproduce the target's tokens; sampled decoding must preserve the target's
distribution, but need not match an autoregressive run's sequence for the same
seed. Speed depends on acceptance and verification cost.

- `--speculative` — `dflash2` for Qwen3.8 27B and for Nemotron 3.5 Bolt with
  its LiLiCorr drafter, `dspark` for DeepSeek V4 Flash, `mtp` for Flash-Next
  and Gemma 4, or `off`. Every mode works in `prompt`, `chat`, `bench` and
  HTTP serving.
- `--dflash-model` — path to the DFlash2 companion draft.
- `--dspark-model` — path to the DSpark support GGUF.
- `--mtp-model` — path to the MTP draft model.
- `--draft-tokens` — maximum tokens drafted per step. Wider drafts go faster
  when the guesses are accepted and waste work when they are rejected.
- `--draft-policy` — DFlash2's block length, `adaptive` (default) or
  `fixed`; for Gemma 4's MTP, how many drafts a cycle verifies: `calibrated`
  (default), `confidence` or `fixed`.
- `--min-draft-tokens` — minimum draft length where supported. DFlash2, DSpark,
  and Flash-Next MTP require the default of one; Gemma 4's MTP takes a floor.

**Server limits (protecting the machine from clients)**

- `--max-pending` — maximum requests waiting in the queue. Over the limit,
  clients wait or are refused rather than exhausting memory.
- `--max-pending-per-client` — the same cap but per client IP. It defaults to
  `--max-pending`; set it lower so one client cannot hog the whole queue.
- `--request-timeout-ms` — requests are killed after this many milliseconds. In
  practice: prevents stuck requests from holding GPU sessions forever.
- `--stall-timeout-ms` — the server exits as `device_lost` when generation makes
  no progress for this many milliseconds, so the service manager restarts
  it. Off unless set; at least 60000.
- `--max-output-bytes` — maximum response size per request.
- `--max-buffered-output-bytes` — maximum generated-but-not-yet-delivered output
  buffered for one request; protects against slow clients.
- `--max-buffered-output-total` — the same buffer budget across all requests.

### Audio (TTS and ASR)

Use `gufo serve tts --model DIR --context 4096` for synthesis or
`gufo serve asr --model DIR --context 1024` for transcription. Each process
loads one checkpoint; llama-swap can route both services under one API URL.

`--context` reserves the model's token capacity. ASR applies that capacity
per audio chunk, so long files do not require one enormous context. The
standalone `gufo transcribe` command also uses `--model` and `--context`;
`--prompt` supplies optional transcription hints.

**Voices**

- `--voice`, `--voice-text`, `--voice-lang` — named voice presets built from reference WAVs. Each entry is
  either a bare reference WAV (its transcript is read from a `.txt` sidecar
  file beside it) or the `--voice` / `--voice-text` / `--voice-lang` triple:
  the reference WAV, its transcript (inline text or a path to a file holding
  it), and an optional default language for that voice. In practice: clients
  pick a voice by name in the request, for example `{"voice":
"narrator_ita"}`, without shipping audio; the optional language means the
  caller does not need to repeat it. Voices require a TTS checkpoint.

```sh
gufo serve tts \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator_eng=/audio/clear-english-voice.wav \
  --voice-text narrator_eng=/audio/clear-english-voice.txt \
  --voice narrator_ita=/audio/clear-italian-voice.wav \
  --voice-text narrator_ita="Questo racconto e' cresciuto..." \
  --voice-lang narrator_ita=italian
```

### Embeddings

Use `gufo serve embedding --model DIR` to serve the EmbeddingGemma 2 embedding model, `google/embeddinggemma-2`, at `POST /v1/embeddings`.  The model has 768 dimensions, an 8192-token context, and BF16 weights.  The process loads one checkpoint and refuses to start when `DIR` is not an EmbeddingGemma 2 checkpoint.

- `--model` (`-m`) is the EmbeddingGemma 2 model directory
- `--served-model-name` is the model ID that clients send and `/v1/models` reports, `embeddinggemma-2` by default
- `--default-prompt-name` names a prompt from the checkpoint's `config_sentence_transformers.json` that prefixes text inputs of requests without `prompt_name`; image-only and audio-only inputs stay unprefixed; none by default

The command takes the shared server options but not `--sessions` or `--context`, because concurrent text requests share device batches on one encoder and the model fixes the context window.  The [server guide](SERVER.md#embeddings) lists the request fields, the limits, and the task prefixes the model expects.

```sh
gufo serve --port 8080 embedding --model ./models/embeddinggemma-2
```

### Server options (all modalities)

These apply to every modality, before or after the subcommand:

- `--host` / `port` — where to listen (`-i` / `-p`). In practice: set
  `--host 0.0.0.0` if you serve from inside a container, or the published
  port will not answer. Use `--host ::` to listen on IPv6 and IPv4 together.
- `--sessions` (`-j`) — preallocated GPU request sessions. In practice: more
  sessions means more requests can compute at once, at the cost of memory
  per session. This option applies to LLM serving; image and speech servers use
  their own bounded queues. The embedding server packs waiting text requests
  into shared device batches and runs each request with images in its own
  batch. `--max-connections` bounds the waiting requests.
- `--max-connections` — maximum simultaneous HTTP connections. In practice:
  clients beyond the limit queue or are refused instead of piling up. A
  connection stops counting once its response is fully written, so a client
  that reads its reply and reconnects immediately is not refused while the
  server is still closing the old connection. Connections that are only
  closing can briefly add to the count of open sockets.
- `--max-request-bytes` — maximum request body size. In practice: matters mostly
  for ASR, since requests carry whole audio files.
- `--api-key` — if set, requests require `Authorization: Bearer <key>`, including
  health checks. Browser CORS preflight (`OPTIONS`) remains unauthenticated.
- `--sse-ping-interval` (`--sse-ping-interval <N>`, default 15) — seconds an SSE
  stream may stay silent before a `: ping` comment, as in llama-server; `-1`
  disables pings. In practice: keeps proxies and clients such as undici from
  dropping a stream during a long prefill or queue wait.
- `--log-level` (`--log-level <error|warn|info|debug>`, default `info`) — how much
  the server logs. In practice: `debug` is the level that explains a stalled
  client, because it also shows health polls, admission refusals with the limit
  that rejected them, and cache candidate decisions. `verbose` (`-v`) is
  shorthand for `--log-level=debug`; passing both is a usage error. The tier is
  a threshold all the way down, and the `event=listening` startup confirmation
  is INFO-tier, so `warn` and `error` boot and stop silently. Prompt text,
  message bodies and the API key are never logged at any level. Debug admission
  lines do name the client by the peer IP address of its connection. See
  [server logs](SERVER.md#troubleshooting-logs).
- `--arbiter` (`--arbiter <HOST:PORT>`) — report to a `gufo arbiter` and follow
  its GPU grants. In practice: start the TTS and the LLM of a voice assistant
  with the same `--arbiter`, and text generation yields the GPU while streamed
  speech falls behind. Without it a server runs alone, as before. See
  [sharing the GPU](ARBITER.md).

## Commands

| Command | Purpose |
| --- | --- |
| `gufo serve` | The OpenAI-compatible server, one model per process: `llm`, `tts`, `asr`, `embedding` or `image` ([server](SERVER.md)) |
| `gufo arbiter` | Shares the GPU between servers started with `--arbiter` ([sharing the GPU](ARBITER.md)) |
| `gufo chat` | An interactive conversation in the terminal |
| `gufo prompt` | One request, then exit |
| `gufo bench` | Prompt-processing and generation throughput, in llama-bench's table |
| `gufo eval` | The pinned DS4 capability questions against a running OpenAI-compatible server |
| `gufo transcribe` | Speech recognition of a WAV file with Qwen3-ASR 1.7B |
| `gufo diagnose` | System, driver, memory and GPU checks |

Every command takes `--help`; `gufo --version` prints the version. Unknown
options and invalid combinations return an error instead of being ignored.

### Server mode

`gufo serve` takes the server options (host, port, limits, API key, logging),
then the model type and its options:

```sh
gufo serve --port 8080 llm --model models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
gufo serve --port 8081 tts --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator=voices/narrator.wav --voice-text narrator=voices/narrator.txt
gufo serve --port 8082 embedding --model models/embeddinggemma-2
```

`gufo serve --help` lists the server options and `gufo serve llm --help` (or
`tts`, `asr`, `embedding`, `image`) those of a model type. The model pages
under [docs/models](models/README.md) give each model's command.

### Benchmarks

`gufo bench` loads a model and prints llama-bench's table of prompt
processing (`pp`) and generation (`tg`) throughput, greedy by default:

```sh
gufo bench --model models/qwen3.5/Qwen3.5-4B-Q4_K_M.gguf -p 512,2048 -n 128 -r 3
```

`-p`, `-n` and `-d` set prompt lengths, generated tokens and the context
depth prepared before the timed region; `-r` repeats each point. The
speculative options of `prompt` apply, so `--speculative` measures a draft
mode. The [performance](PERFORMANCE.md) page lists current results.

### Evaluation

`gufo eval` sends the pinned DS4 capability questions to a running
OpenAI-compatible server and grades the answers:

```sh
gufo eval --base-url http://127.0.0.1:8080/v1 --questions 75 --greedy \
  --output eval.json
```

The result file is sanitized: the API key, the server address and home
directory paths are redacted.

### Diagnose

`gufo diagnose` checks that the system can run the engine: the GPU and its
driver, the ROCm runtime the binary loads, access to `/dev/kfd` and
`/dev/dri`, memory, and the toolchain the build pinned. `--json` writes the
result in a fixed schema, `--section` selects a part, and `--benchmark
bandwidth` or `allocation` measures memory bandwidth or HIP allocation.
