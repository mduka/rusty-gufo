# Sharing the GPU (`gufo arbiter`)

`gufo arbiter` lets several `gufo serve` processes share one GPU by priority.
Servers started with `--arbiter HOST:PORT` report to it. While a streamed
text-to-speech server falls behind real time, the arbiter grants text
generation only a share of the GPU, and the text scheduler pauses between
its steps to keep to it. Otherwise text generation runs at full speed.
Without `--arbiter` none of this runs.

## Why

A voice assistant runs a language model, speech synthesis and speech
recognition side by side. Each is a `gufo serve` process, and all of them
share the GPU and its memory bandwidth. A language model writes text far
faster than anyone speaks, so fluent speech should be possible. On one
device, however, text generation takes the GPU from the speech server, and
the streamed audio breaks up.

CPU priorities (`nice`) do not reach GPU work. Priorities among GPU queues
order dispatch, but cannot divide memory bandwidth between kernels that
already run. So the language model has to yield itself, and only as long as
speech needs it. The arbiter coordinates that across processes.

Measured on a Radeon 8060S (gfx1151): Qwen3.8 Flash-Next with MTP decodes
continuously while Qwen3-TTS CustomVoice streams 50.5 s of speech. Unthrottled,
the TTS fell to 0.79× real time, and a player starting with the first chunk
heard 39 gaps with 14.5 s of silence in total. Text generation lost only 7% of
its speed. With the arbiter, speech stays above real time and text
generation keeps about 88% of its speed (see [measurements](#measurements)).

## How it works

```text
gufo serve tts --arbiter A ──status (playback buffer)──▶ gufo arbiter (A)
gufo serve llm --arbiter A ◀──share (GPU grant, lease)── │  rules: --config
gufo serve asr --arbiter A ──status (presence)─────────▶ │  HTTP: monitoring, control
```

- **Roles follow the modality.** `tts` is *realtime*: its reports raise
  pressure. `llm` is *background*: it yields under pressure. `asr`,
  `embedding` and `image` are *observed*: they appear in the arbiter's view.
- **Streamed TTS reports its playback buffer.** A client plays streamed audio
  as it arrives. Every chunk fills its buffer by the chunk's duration, and
  playback drains it at one second per second. The server follows that
  buffer, assuming playback starts with the first chunk, and reports it with
  whether it synthesizes. While it waits for text it reports no activity, so
  it never slows text generation for nothing. Buffered (whole-file) speech
  reports nothing.
- **The arbiter decides.** Pressure from a synthesizing TTS is
  `1 − buffer / buffer_target_s`, from 0 to 1. Clients can add pressure of
  their own (`POST /v1/pressure`). The highest pressure sets the share of
  every background server: `1 − pressure × (1 − min_share)`. The arbiter sends
  a share with a lease and renews it while it applies.
- **Text generation keeps to its share.** After each step (a decode batch or
  a prefill chunk), the scheduler pauses for `step × (1 / share − 1)`, in
  slices of at most 5 ms. It continues as soon as the share rises or the
  server stops. While a share applies, prefill runs in bounded chunks, so a
  long prompt does not hold the device for seconds.
- **Failures lift the grant.** A lease lapses unless the arbiter renews it.
  When the connection drops, the server returns to full speed at once and
  reconnects with backoff. A server never goes below a 5% share, whatever an
  arbiter sends. Two servers with the same identity cannot both connect.
- **Off costs nothing.** Without `--arbiter` there is no thread and no
  socket, and the scheduler checks an empty option once per step.

## Usage

```sh
gufo arbiter                                   # 127.0.0.1:7071, default rules
gufo serve tts --model <TTS> --arbiter 127.0.0.1:7071
gufo serve llm --model <GGUF> --arbiter 127.0.0.1:7071
```

The order does not matter: a server that starts before the arbiter connects
when the arbiter comes up. `gufo arbiter` takes `--host`, `--port`,
`--config FILE` and the logging options of `gufo serve`. A server's identity
is `modality:model@host:port`, with the model ID it serves and the address
it listens on.

## Rules

The rules live in the arbiter. `--config FILE` reads them from TOML, and every
key is optional:

```toml
[background]           # text generation
min_share = 0.25       # the least GPU share under full pressure (0.05 to 1)
lease_ms = 500         # a grant lapses unless renewed within this (100 to 10000)

[tts]                  # streamed speech
buffer_target_s = 2.5  # pressure while the playback buffer holds less

[[instances]]          # rules for matching servers; the first match applies
modality = "llm"
model = "qwen3.8-flash-next"
min_share = 0.4
```

Unknown keys and values out of range are errors at startup, so a misspelt
rule never silently does nothing. `GET /v1/config` shows the rules in force.

The defaults come from the measurements below. A 25% floor keeps Flash-Next at
about 8.5 tokens per second, twice the rate of speech, so the TTS never runs
out of text. A 2.5 s target leaves about 2 s of audio in the buffer during
synthesis, at about 2% of text throughput more than a 1.5 s target.

## Interfaces

The arbiter is an HTTP server.

| Route | Use |
| --- | --- |
| `GET /v1/connect` | WebSocket for `gufo serve --arbiter` |
| `GET /v1/events` | WebSocket of state snapshots (on connect and on every change, at most four a second) |
| `GET /v1/instances` | Servers with identity, role, last status and share; client pressure |
| `GET /v1/config` | The rules in force |
| `GET /metrics` | Prometheus text (below) |
| `POST /v1/pressure` | `{"source": "assistant", "pressure": 0.7, "ttl_ms": 500}` |
| `POST /v1/instances/{id}/share` | `{"share": 0.5, "ttl_ms": 10000}`, or `{"share": null}` to end it |
| `GET /health` | Liveness |

An assistant that knows its real playback buffer can report it as pressure,
which is more precise than the server's model. A manual share overrides the
rules for one background server until it expires. Ids in paths may be
percent-encoded, since model IDs can contain spaces.

### Protocol

One JSON object with a `type` per WebSocket text frame. Version 1:

| From | `type` | Fields |
| --- | --- | --- |
| server | `hello` | `protocol`, `modality`, `model`, `address`, `pid` |
| server | `status` | text: `share`, `yielded_s`; TTS: `active`, `buffer_s`, `audio_s` |
| arbiter | `welcome` | `protocol`, `id` |
| arbiter | `share` | `share`, `ttl_ms` (a share of 1 or a zero lifetime ends the grant) |
| arbiter | `error` | `message`; the arbiter then closes the connection |

```text
{"type": "hello", "protocol": 1, "modality": "llm", "model": "qwen3.8-flash-next", "address": "127.0.0.1:8080", "pid": 4242}
{"type": "status", "share": 0.86, "yielded_s": 12.4}
{"type": "share", "share": 0.86, "ttl_ms": 500}
```

A busy TTS reports every 50 ms and on every change of activity, other servers
every 250 ms. The arbiter decides every 50 ms. A status older than a second
raises no pressure.

### Metrics

| Metric | Labels | Meaning |
| --- | --- | --- |
| `gufo_arbiter_pressure` | | Pressure in force, 0 to 1 |
| `gufo_arbiter_instances` | `role` | Connected servers |
| `gufo_arbiter_share` | `instance`, `modality`, `model` | Share granted to a background server |
| `gufo_arbiter_yielded_seconds_total` | same | Time a background server paused for grants |
| `gufo_arbiter_tts_buffer_seconds` | same | The modelled playback buffer of a streamed TTS |
| `gufo_arbiter_tts_audio_seconds_total` | same | Audio a streamed TTS sent |

## Measurements

Radeon 8060S (gfx1151), 128 GiB, ROCm 10.0, 2026-10-26.

- **Text generation:** Qwen3.8 Flash-Next UD-Q4_K_XL with its MTP sidecar,
  one session and a context of 8,192 tokens. It decodes continuously: raw
  completions of 256 tokens with `ignore_eos`.
- **Speech:** Qwen3-TTS 1.7B CustomVoice streams PCM for a 150-word English
  text, 50.5 s of audio, three requests per row.
- **Gaps:** a simulated player starts with the first chunk, or after 0.5 s
  of pre-buffering.
- **Fixed shares:** taken from an experimental build that held one share for
  the whole run.

| Text generation beside the TTS | TTS speed | Gaps, no pre-buffer | Gaps, 0.5 s pre-buffer | Text tokens/s |
| --- | --- | --- | --- | --- |
| none (TTS alone) | 2.39× real time | 1 (0.19 s) | 0 | – |
| unthrottled | 0.79× | 39 (14.5 s) | 39 (14.0 s) | 34.6 |
| fixed share 0.5 | 1.60× | 1 (0.44 s) | 0 | 17.4 |
| fixed share 0.25 | 2.01× | 1 (0.27 s) | 0 | 8.5 |
| fixed share 0.1 | 2.24× | 1 (0.22 s) | 0 | 3.4 |
| arbiter, target 1.5 s | 1.01 to 1.03× | 1 to 2 (0.32 to 0.75 s) | 0 to 2 (at most 0.25 s) | 30.4 |
| arbiter, target 2.5 s | 1.01 to 1.05× | 1 (0.32 to 0.76 s) | 0 to 1 (at most 0.26 s) | 29.8 |

- **What the arbiter does:** under it, the share settles at about 0.9 with
  a 1.5 s target and 0.86 with 2.5 s. The buffer during synthesis holds 1.3 s
  and 2.0 s (medians).
- **No gaps mid-stream.** The remaining gap comes at the start of a request.
  In the first request of each run it is as small as with the TTS alone
  (0.32 s, none with pre-buffering). The later requests started right after
  the previous one, so the server's buffer model still held the previous
  request's audio, while the simulated player started empty. A client that
  plays sentences back to back keeps that audio. For separate utterances the
  buffer has drained and the first case applies.
- **Text speed per step is unchanged.** Each request decoded at about 36
  tokens per second while it ran. With the arbiter connected and no pressure,
  text generation ran at 43 tokens per second. Rates between separate runs
  vary by about 10%, so compare within a row.
- **Off, there is no cost.** The A/B runs of
  [Gufo in Rust](RUST.md#performance-against-upstream) use servers without
  `--arbiter`.

## Limits

- **Only text generation yields.** Image and embedding servers report but do
  not pause; ASR reports its presence but no pressure yet.
- **The arbiter has no authentication.** It listens on loopback by default.
  A process that can reach it can throttle text generation, or end a grant.
- **One device:** the arbiter does not tell GPUs apart.
- **Possible extensions:** pressure from realtime recognition, background
  pauses for image generation, and serialized model loading. Two large models
  loading at once can exhaust the shared memory of an APU.
