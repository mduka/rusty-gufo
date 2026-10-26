#!/usr/bin/env python3
"""Differential serve session for the Rust migration (docs/RUST.md): one scripted HTTP session against the C++ and the Rust server with
the same model and options.

Each server runs alone (one model in memory at a time), first the baseline,
then the candidate. The session covers every text route, streaming and
non-streaming, reasoning, tools, error paths, retained turns, overload (429),
client disconnects while queued and while decoding, and a restart on the disk
cache. Responses must be equal after masking IDs, timestamps and timings, and
so must the non-timing fields of each server's request log (tokens, cache
hits, cached tokens, draft counts, finish reason). With --interchange, each
server then starts on the other's disk cache and must restore the retained
requests with the same responses. With --images (a server with --mmproj),
the session adds image inputs: single, streaming, several with and without
vision IDs, image-only turns, retained image turns, cache identity across
different images, JPEG (status and token grid only: decoders may differ)
and invalid inputs.

    tools/serving/serve-diff.py --baseline <upstream checkout>/build/release/gufo \\
        --candidate target/release/gufo --output diff.json -- \\
        --model <gguf> --context 4096 --sessions 1

SERVE_ARGS go to `gufo serve llm` on both sides; the tool adds --port,
--served-model-name, the queue limits the overload steps need, and
--cache-disk with a fresh directory per side. Greedy sampling throughout, so
equal outputs mean equal token sequences.

With --asr, SERVE_ARGS go to `gufo serve asr` (`--model <Qwen3-ASR dir>`)
and the session covers transcription instead: every response format,
streaming, a forced language and context, long audio split into chunks,
resampled stereo input, error paths, concurrent requests, a client that
leaves mid-stream, and a realtime WebSocket session with its error events.
Its audio is --audio, the Qwen3-ASR reference recording by default.

The Anthropic and Responses adapters of the Rust server split reasoning from
the answer as llama.cpp does and accept its controls, where the C++ server
returns the raw text and rejects them (docs/RUST.md): the C++
answers are split the same way before the comparison, and for the new
controls and returned reasoning the C++ server answers the equivalent chat
completion in the adapter's shape. The C++ server returns Responses
reasoning as `summary_text` items and `reasoning_summary_*` events, which
are renamed to llama.cpp's
`reasoning_text` ones the Rust server sends with
`--responses-reasoning-target reasoning-text` (its default is upstream's
summary).

Against the C++ server: its Messages route returns thinking blocks itself, its Responses reasoning keeps the newline before
`</think>` (the Rust server trims it like its chat completions), adapter
steps compare reasoning token counts only where the C++ chat completion
determines them, range errors print bounds without `std::to_string`'s
decimals, and a stream cancelled before any output is compared by method,
path and finish only (the Rust server commits an empty response, which
fails once the client has gone). The tool steps cover native calls with
nested, union and parallel arguments, schemas native tags cannot enforce
exactly (a pattern, a strict union, beside an ordinary tool), strict and
named choices, reasoning, disabled tools and Responses tools; a native model
keeps its call syntax for those schemas too (a baseline that answers with a
JSON envelope instead is compared in that form).
"""

import argparse
import base64
import http.client
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
import zlib

MASKED_KEYS = {"id", "item_id", "call_id", "created", "created_at", "completed_at",
               "system_fingerprint", "timings", "server_timing"}
MASKED_SUFFIXES = ("_ms", "_per_second")
# Request-log fields that depend on wall time or host state.
TIMING_LOG_FIELDS = {"request", "duration_ms", "queue_ms", "ttft_ms", "prefill_tps",
                     "decode_tps", "cache_restore_ms", "rss_mib", "host_available_mib",
                     "resident_at_admission", "queue_depth", "cache_snapshot_bytes",
                     "cache_disk_queued_bytes", "gpu_device_used_mib",
                     # ASR: timings and the realtime item (which holds the request ID).
                     "feature_ms", "encoder_ms", "decoder_ms", "item"}
LOG_LINE = re.compile(r"\[(?:INFO|WARN|ERROR)\] \[(?:http|asr)\] (request=.*)$")
SERVED_NAME = "diff"
ASR = False
AUDIO = None
# The side a session runs on: "baseline" (C++) or "candidate" (Rust).
SIDE = None
# Adapter routes share the chat log fields; their paths are compared as one.
ADAPTER_PATHS = {"/v1/messages", "/v1/responses"}
# Realtime events carry the request ID in their event, item and session IDs.
REALTIME_MASKED_KEYS = {"event_id", "item_id", "previous_item_id", "id"}


def mask(value):
    if isinstance(value, dict):
        return {key: ("<masked>" if key in MASKED_KEYS or key.endswith(MASKED_SUFFIXES)
                      else mask(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [mask(item) for item in value]
    return value


def normalize_body(content_type, body):
    if "application/json" in content_type:
        try:
            return mask(json.loads(body))
        except ValueError:
            return body.decode("utf-8", "replace")
    if "text/event-stream" in content_type:
        events = []
        for line in body.decode("utf-8", "replace").splitlines():
            if not line.startswith("data: "):
                if line.startswith("event: "):
                    events.append(line)
                continue
            payload = line[6:]
            try:
                events.append(mask(json.loads(payload)))
            except ValueError:
                events.append(payload)
        return events
    text = body.decode("utf-8", "replace")
    if content_type.startswith("text/plain"):
        # /metrics: speeds and processing times are timings, token totals are
        # compared.
        text = re.sub(r"^(llamacpp:(?:(?:prompt|predicted)_tokens_seconds|prompt_seconds_total"
                      r"|tokens_predicted_seconds_total)) .*$", r"\1 <masked>",
                      text, flags=re.M)
    return text


class Server:
    def __init__(self, binary, args, port, cache_dir, log_path):
        self.binary = binary
        self.args = args
        self.port = port
        self.cache_dir = cache_dir
        self.log_path = log_path
        self.process = None
        self.log = None

    def start(self):
        self.log = open(self.log_path, "ab")
        env = {k: v for k, v in os.environ.items()
               if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
        env["NO_COLOR"] = "1"
        if ASR:
            command = [self.binary, "serve", "asr", "--port", str(self.port),
                       "--served-model-name", SERVED_NAME, *self.args]
        else:
            command = [self.binary, "serve", "llm", "--port", str(self.port),
                       "--served-model-name", SERVED_NAME,
                       "--max-pending", "2", "--max-pending-per-client", "1",
                       "--cache-disk", str(self.cache_dir), *self.args]
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                        env=env)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.binary} exited with {self.process.returncode}; "
                                   f"see {self.log_path}")
            try:
                status, _, _ = request(self.port, "GET", "/health", timeout=2)
                if status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"{self.binary} did not become healthy")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def request_log(self):
        entries = []
        for line in Path(self.log_path).read_text(errors="replace").splitlines():
            match = LOG_LINE.search(line)
            if not match:
                continue
            fields = dict(item.split("=", 1) for item in match.group(1).split()
                          if "=" in item)
            if fields.get("finish") == "cancelled" and fields.get("generated_tokens") == "0":
                # A stream cancelled before its headers: the Rust server still
                # commits an empty response, which fails once the client has
                # gone ("disconnected"); the C++ one writes nothing
                # ("completed").
                fields = {k: fields.get(k) for k in ("method", "path", "finish")}
            elif fields.get("outcome") == "disconnected":
                # How far a disconnected request got depends on timing.
                fields = {k: fields.get(k) for k in ("method", "path", "outcome")}
            if fields.get("path") in ADAPTER_PATHS:
                fields["path"] = "/v1/chat/completions"
            if fields.get("path") == "/v1/chat/completions":
                # The C++ side answers some adapter steps as chat completions.
                fields.pop("body_bytes", None)
            entries.append({k: v for k, v in fields.items() if k not in TIMING_LOG_FIELDS})
        # Completion order of concurrent requests varies; compare as a multiset.
        return sorted(entries, key=lambda entry: json.dumps(entry, sort_keys=True))


def connection(port, timeout=600, source="127.0.0.1"):
    return http.client.HTTPConnection("127.0.0.1", port, timeout=timeout,
                                      source_address=(source, 0))


def request(port, method, path, body=None, headers=None, timeout=600, source="127.0.0.1"):
    conn = connection(port, timeout, source)
    data = None
    all_headers = dict(headers or {})
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        all_headers.setdefault("Content-Type", "application/json")
    conn.request(method, path, data, all_headers)
    response = conn.getresponse()
    payload = response.read()
    content_type = response.getheader("Content-Type", "")
    conn.close()
    return response.status, content_type, payload


def chat(messages, max_tokens=24, stream=False, think=False, **extra):
    body = {"model": SERVED_NAME, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": think}}
    body.update(extra)
    return body


def user(text):
    return [{"role": "user", "content": text}]


def split_reasoning(raw):
    """The reasoning and the answer of a Qwen3.8 generation that starts
    inside a reasoning block, split as the Rust server does."""
    content = raw[len("<think>"):] if raw.startswith("<think>") else raw
    end = content.find("</think>")
    if end < 0:
        return content.strip(" \t\n\r\f\v"), ""
    return content[:end].strip(" \t\n\r\f\v"), content[end + len("</think>"):].lstrip("\n\r")


def anthropic_content(reasoning, text):
    content = []
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if text:
        content.append({"type": "text", "text": text})
    return content


def responses_output(reasoning, text, limited):
    """The Rust server's Responses items: a length limit cuts the last one
    short."""
    output = []
    if reasoning:
        output.append({"id": "<masked>", "type": "reasoning", "summary": [],
                       "content": [{"type": "reasoning_text", "text": reasoning}],
                       "encrypted_content": "", "status": "completed"})
    if text:
        output.append({"id": "<masked>", "type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "annotations": [],
                                    "logprobs": [], "text": text}],
                       "status": "completed"})
    if output and limited:
        output[-1]["status"] = "incomplete"
    return output


def responses_reasoning(value):
    """C++ Responses carry reasoning as
    `summary_text` parts and `reasoning_summary_*` events; the Rust server,
    like llama.cpp, as `reasoning_text` content (docs/RUST.md).
    Converts a C++ body or stream event to the Rust names, and adds the
    `completed_at` the Rust server sends for a completed response."""
    if isinstance(value, list):
        return [responses_reasoning(item) for item in value]
    if isinstance(value, str):
        for kind in ("part.added", "part.done", "text.delta", "text.done"):
            source = "event: response.reasoning_summary_" + kind
            if value == source:
                return ("event: response.content_" + kind if kind.startswith("part")
                        else "event: response.reasoning_" + kind)
        return value
    if not isinstance(value, dict):
        return value
    value = {key: responses_reasoning(item) for key, item in value.items()}
    if value.get("type") == "reasoning" and "content" not in value:
        # The Rust server trims the reasoning as its chat completions do
        # (llama.cpp); the C++ one keeps the newline before `</think>`.
        parts = value.get("summary", [])
        value = {**value, "summary": [], "encrypted_content": "",
                 "content": [{**part, "type": "reasoning_text",
                              "text": part.get("text", "").strip(" \t\n\r\f\v")}
                             for part in parts]}
    if value.get("type") == "summary_text":
        value = {**value, "type": "reasoning_text"}
    kind = value.get("type", "")
    if isinstance(kind, str) and kind.startswith("response.reasoning_summary_"):
        rest = kind[len("response.reasoning_summary_"):]
        value = {**value, "type": ("response.content_" + rest if rest.startswith("part")
                                   else "response.reasoning_" + rest)}
        value.pop("summary_index", None)
        value["content_index"] = 0
    if value.get("object") == "response" and value.get("status") == "completed":
        value.setdefault("completed_at", "<masked>")
    return value


def split_adapter(path, body):
    """A C++ adapter answer in the Rust server's shape: Anthropic's raw
    text split, Responses' reasoning items renamed (an older raw answer is
    split as well)."""
    if path == "/v1/messages":
        if any(block.get("type") == "thinking" for block in body["content"]):
            return body  # the C++ server splits it too
        raw = "".join(block.get("text", "") for block in body["content"])
        return {**body, "content": anthropic_content(*split_reasoning(raw))}
    if isinstance(body, list):
        return responses_reasoning(body)
    output = body["output"]
    if not any(item.get("type") == "reasoning" for item in output):
        raw = "".join(part.get("text", "") for item in output
                      for part in item.get("content", []))
        if "</think>" in raw:
            limited = body.get("status") == "incomplete"
            body = {**body, "output": responses_output(*split_reasoning(raw), limited)}
    return responses_reasoning(body)


def mask_split_reasoning_tokens(body):
    """A Responses body whose output has both reasoning and an answer: a C++
    chat completion has no token count to convert for its reasoning
    (`chat_as_adapter`), so neither side's count is compared."""
    if not isinstance(body, dict) or body.get("object") != "response":
        return body
    kinds = {item.get("type") for item in body.get("output", [])}
    if {"reasoning", "message"} <= kinds:
        details = body.get("usage", {}).get("output_tokens_details", {})
        if "reasoning_tokens" in details:
            details["reasoning_tokens"] = "<masked>"
    return body


def chat_as_adapter(path, body):
    """A chat completion in the shape of the adapter at `path`."""
    choice = body["choices"][0]
    message = choice["message"]
    reasoning, text = message.get("reasoning_content") or "", message.get("content") or ""
    usage = body["usage"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    limited = choice["finish_reason"] == "length"
    # Tokens before `</think>`: none without reasoning, all when the answer
    # never started; a split answer has no token count to take them from.
    if not reasoning:
        reasoning_tokens = 0
    elif not text:
        reasoning_tokens = usage["completion_tokens"]
    else:
        reasoning_tokens = None
    if path == "/v1/messages":
        return {"id": "<masked>", "type": "message", "role": "assistant",
                "model": body["model"], "content": anthropic_content(reasoning, text),
                "stop_reason": "max_tokens" if limited else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": usage["prompt_tokens"],
                          "output_tokens": usage["completion_tokens"],
                          "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": cached},
                "timings": "<masked>"}
    response = {"id": "<masked>", "object": "response", "created_at": "<masked>",
                "model": body["model"],
                "status": "incomplete" if limited else "completed", "error": None,
                "incomplete_details": {"reason": "max_output_tokens"} if limited else None,
                "usage": {"input_tokens": usage["prompt_tokens"],
                          "output_tokens": usage["completion_tokens"],
                          "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                          "input_tokens_details": {
                              "cached_tokens": cached,
                              "cache_write_tokens": usage["gufo"]["prefill_tokens"]},
                          "output_tokens_details": {"reasoning_tokens": reasoning_tokens}},
                "output": responses_output(reasoning, text, limited),
                "store": False, "parallel_tool_calls": False, "tool_choice": "none",
                "tools": [], "timings": "<masked>"}
    if not limited:
        response["completed_at"] = "<masked>"
    return response


# Vision steps (--images): synthetic PNGs, and one JPEG when ffmpeg exists.
IMAGES = False


def png(width, height, pixel):
    """An RGB PNG whose pixel(x, y) returns (r, g, b)."""
    rows = b"".join(b"\0" + bytes(c for x in range(width) for c in pixel(x, y))
                    for y in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def flags(x, y):
    """Red left half, blue right half, a green square in the middle."""
    if 96 <= x < 160 and 64 <= y < 128:
        return (0, 200, 0)
    return (220, 20, 20) if x < 128 else (20, 40, 220)


def gradient(x, y):
    return (x * 255 // 319, y * 255 // 239, 128)


def stripes(x, y):
    return (255, 255, 255) if (x // 16) % 2 == 0 else (0, 0, 0)


def data_url(kind, data):
    return f"data:image/{kind};base64," + base64.b64encode(data).decode()


def jpeg(png_bytes):
    """The PNG as JPEG through ffmpeg, or None without ffmpeg."""
    if shutil.which("ffmpeg") is None:
        return None
    result = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "png_pipe", "-i", "-", "-f", "mjpeg",
         "-q:v", "3", "-"], input=png_bytes, capture_output=True, check=True)
    return result.stdout


def image_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


def image_steps(port, record, records):
    """Image inputs: single, streaming, several, image-only, retained turns,
    cache identity, JPEG and error paths. Returns the retained conversation
    for the restart."""
    first = data_url("png", png(256, 192, flags))
    second = data_url("png", png(320, 240, gradient))
    third = data_url("png", png(64, 64, stripes))
    question = [image_part(first), {"type": "text", "text": "Which colors do you see?"}]
    record("image", "POST", "/v1/chat/completions",
           chat([{"role": "user", "content": question}], 32))
    record("image again", "POST", "/v1/chat/completions",
           chat([{"role": "user", "content": question}], 32))
    record("image stream", "POST", "/v1/chat/completions",
           chat([{"role": "user", "content": question}], 32, stream=True))
    other = [image_part(second), {"type": "text", "text": "Which colors do you see?"}]
    record("other image same text", "POST", "/v1/chat/completions",
           chat([{"role": "user", "content": other}], 32))
    several = [{"type": "text", "text": "Compare these pictures."},
               image_part(first), image_part(second), image_part(third)]
    record("three images", "POST", "/v1/chat/completions",
           chat([{"role": "user", "content": several}], 48))
    record("three images vision id", "POST", "/v1/chat/completions",
           chat([{"role": "user", "content": several}], 48,
                chat_template_kwargs={"enable_thinking": False, "add_vision_id": True}))
    record("image only", "POST", "/v1/chat/completions",
           chat([{"role": "user", "content": [image_part(third)]},
                 {"role": "assistant", "content": "I see stripes."},
                 {"role": "user", "content": "How many stripes?"}], 24))
    encoded = jpeg(png(256, 192, flags))
    if encoded is not None:
        # JPEG decoders may differ by a few levels (the Rust server does not
        # use libjpeg-turbo), so only the status and the image's token grid
        # must agree.
        status, _, payload = request(port, "POST", "/v1/chat/completions", chat(
            [{"role": "user", "content": [image_part(data_url("jpeg", encoded)),
                                          {"type": "text", "text": "Describe it."}]}], 24))
        usage = json.loads(payload).get("usage", {}) if status == 200 else {}
        records.append(("jpeg", {"status": status,
                                 "prompt_tokens": usage.get("prompt_tokens")}))
    for name, url in (("bad base64", "data:image/png;base64,@@@@"),
                      ("gif", "data:image/gif;base64,R0lGODlhAQABAAAAACw="),
                      ("http url", "http://example.com/a.png"),
                      ("private https", "https://127.0.0.1/a.png"),
                      ("not an image", data_url("png", b"not a png"))):
        record(f"image error {name}", "POST", "/v1/chat/completions",
               chat([{"role": "user", "content": [image_part(url)]}], 8))
    record("too many images", "POST", "/v1/chat/completions",
           chat([{"role": "user", "content": [image_part(third)] * 17}], 8))

    status, _, payload = request(port, "POST", "/v1/chat/completions",
                                 chat([{"role": "user", "content": question}], 32))
    reply = json.loads(payload)["choices"][0]["message"].get("content", "")
    records.append(("image turn 1", {"status": status, "body": mask(json.loads(payload))}))
    conversation = [{"role": "user", "content": question},
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "Where is the green square?"}]
    record("image turn 2", "POST", "/v1/chat/completions", chat(conversation, 32))
    record("image turn 2 again", "POST", "/v1/chat/completions", chat(conversation, 32))
    return conversation


WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
STRICT_WEATHER_TOOL = {"type": "function", "function": {
    **WEATHER_TOOL["function"], "strict": True,
    "parameters": {**WEATHER_TOOL["function"]["parameters"],
                   "additionalProperties": False}}}
# Nested objects and arrays in native parameter tags.
EDIT_TOOL = {"type": "function", "function": {
    "name": "edit_file", "description": "Replace exact text in a file",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "oldText": {"type": "string"}, "newText": {"type": "string"}},
            "required": ["oldText", "newText"]}}},
        "required": ["path", "edits"]}}}
# Union arguments: typed alternatives before text.
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search", "description": "Search the web",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"},
        "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "site": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["query"]}}}
# A pattern native parameter tags cannot carry: raw text in native tags.
CODE_TOOL = {"type": "function", "function": {
    "name": "set_airport", "description": "Select an airport by IATA code",
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string", "pattern": "^[A-Z]{3}$"}},
        "required": ["code"], "additionalProperties": False}}}
# A strict union: best-effort native arguments.
TIMEOUT_TOOL = {"type": "function", "function": {
    "name": "set_timeout", "description": "Set the request timeout", "strict": True,
    "parameters": {"type": "object", "properties": {
        "seconds": {"anyOf": [{"type": "integer"}, {"type": "string"}]}},
        "required": ["seconds"], "additionalProperties": False}}}


def session(port, restart):
    """The scripted session; returns [(step, record)] and the retained
    conversation for the cache interchange."""
    records = []

    def record(name, method, path, body=None, headers=None, source="127.0.0.1",
               split=False):
        status, content_type, payload = request(port, method, path, body, headers,
                                                source=source)
        normalized = normalize_body(content_type, payload)
        if split and SIDE == "baseline" and status == 200:
            normalized = split_adapter(path, normalized)
        records.append((name, {"status": status,
                               "content_type": content_type.split(";")[0],
                               "body": normalized}))

    def adapter(name, path, body, chat_body):
        """An adapter request only the Rust server accepts; the C++ server
        answers the equivalent chat completion in the adapter's shape."""
        if SIDE == "candidate":
            record(name, "POST", path, body)
            records[-1][1]["body"] = mask_split_reasoning_tokens(records[-1][1]["body"])
            return
        status, content_type, payload = request(port, "POST", "/v1/chat/completions",
                                                chat_body)
        converted = normalize_body(content_type, payload)
        if status == 200:
            converted = mask(chat_as_adapter(path, json.loads(payload)))
        records.append((name, {"status": status, "content_type": content_type.split(";")[0],
                               "body": mask_split_reasoning_tokens(converted)}))

    for path in ("/health", "/v1/health", "/ready", "/v1/models", "/props",
                 f"/props?model={SERVED_NAME}", "/slots"):
        record(f"get {path}", "GET", path)
    record("unknown route", "GET", "/v1/unknown")
    record("options", "OPTIONS", "/v1/chat/completions")

    record("chat", "POST", "/v1/chat/completions", chat(user("Name three colors.")))
    record("chat stream", "POST", "/v1/chat/completions",
           chat(user("Name three colors."), stream=True,
                stream_options={"include_usage": True}))
    record("chat think stream", "POST", "/v1/chat/completions",
           chat(user("What is 17 * 23?"), max_tokens=64, stream=True, think=True))
    record("chat think", "POST", "/v1/chat/completions",
           chat(user("What is 17 * 23?"), max_tokens=64, think=True))
    record("chat length 1", "POST", "/v1/chat/completions",
           chat(user("Count to ten."), max_tokens=1))
    record("chat system", "POST", "/v1/chat/completions",
           chat([{"role": "system", "content": "Answer in French."},
                 {"role": "user", "content": "Say hello."}]))
    record("tools auto", "POST", "/v1/chat/completions",
           chat(user("What is the weather in Paris? Use the tool."), max_tokens=64,
                tools=[WEATHER_TOOL]))
    record("tools required stream", "POST", "/v1/chat/completions",
           chat(user("Weather in Rome?"), max_tokens=64, stream=True,
                tools=[WEATHER_TOOL], tool_choice="required"))
    record("tool result", "POST", "/v1/chat/completions", chat(
        [{"role": "user", "content": "Weather in Oslo?"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call_1", "type": "function",
              "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]},
         {"role": "tool", "tool_call_id": "call_1", "content": "{\"celsius\": -3}"}],
        max_tokens=32, tools=[WEATHER_TOOL]))
    # Historical integers render in decimal, not as 1e+05.
    record("tool result integers", "POST", "/v1/chat/completions", chat(
        [{"role": "user", "content": "Weather in Oslo, hourly for the next 100000 seconds?"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call_1", "type": "function",
              "function": {"name": "get_weather",
                           "arguments": "{\"city\": \"Oslo\", \"seconds\": 100000}"}}]},
         {"role": "tool", "tool_call_id": "call_1", "content": "{\"celsius\": -3}"}],
        max_tokens=32, tools=[WEATHER_TOOL]))
    # Constrained calls in each native shape, schemas native tags cannot
    # enforce exactly, strict and named choices, reasoning, disabled tools and
    # Responses.
    edit = user("In notes.txt replace 'teh' with 'the'. Use the tool.")
    record("tools nested", "POST", "/v1/chat/completions",
           chat(edit, max_tokens=96, tools=[EDIT_TOOL]))
    record("tools nested stream", "POST", "/v1/chat/completions",
           chat(edit, max_tokens=96, stream=True, tools=[EDIT_TOOL]))
    record("tools union", "POST", "/v1/chat/completions",
           chat(user("Search the web for Rust HIP bindings, at most 3 results."),
                max_tokens=96, tools=[SEARCH_TOOL]))
    record("tools parallel", "POST", "/v1/chat/completions",
           chat(user("What is the weather in Paris and in Rome? Call the tool once per city."),
                max_tokens=128, tools=[WEATHER_TOOL]))
    record("tools pattern", "POST", "/v1/chat/completions",
           chat(user("Select the airport OSL."), max_tokens=64, tools=[CODE_TOOL],
                tool_choice="required"))
    record("tools strict union", "POST", "/v1/chat/completions",
           chat(user("Set the timeout to 30 seconds."), max_tokens=64, tools=[TIMEOUT_TOOL]))
    record("tools mixed stream", "POST", "/v1/chat/completions",
           chat(user("Weather in Lima? Use the tool."), max_tokens=64, stream=True,
                tools=[WEATHER_TOOL, CODE_TOOL]))
    record("tools strict", "POST", "/v1/chat/completions",
           chat(user("Weather in Madrid?"), max_tokens=64, tools=[STRICT_WEATHER_TOOL],
                parallel_tool_calls=False))
    record("tools named stream", "POST", "/v1/chat/completions",
           chat(user("Weather in Berlin?"), max_tokens=64, stream=True,
                tools=[WEATHER_TOOL, SEARCH_TOOL],
                tool_choice={"type": "function", "function": {"name": "get_weather"}}))
    record("tools think", "POST", "/v1/chat/completions",
           chat(user("Weather in Vienna?"), max_tokens=256, think=True,
                tools=[WEATHER_TOOL]))
    record("tools think stream", "POST", "/v1/chat/completions",
           chat(user("Weather in Prague?"), max_tokens=256, stream=True, think=True,
                tools=[WEATHER_TOOL]))
    record("tools none", "POST", "/v1/chat/completions",
           chat(user("Show briefly how a <tool_call> block looks."), max_tokens=48,
                tools=[WEATHER_TOOL], tool_choice="none"))
    # Explicitly non-strict: without `strict` the C++ Responses route
    # normalizes the schema to strict, the Rust one keeps llama.cpp's
    # non-strict default (docs/RUST.md).
    lisbon = {"model": SERVED_NAME, "input": "Weather in Lisbon? Use the tool.",
              "max_output_tokens": 64, "temperature": 0,
              "tools": [{"type": "function", **WEATHER_TOOL["function"], "strict": False}]}
    record("responses tools", "POST", "/v1/responses", lisbon, split=True)
    # Without reasoning: the C++ stream sends the newline before `</think>`
    # as a delta of its own, which the Rust server trims.
    record("responses tools stream", "POST", "/v1/responses",
           {**lisbon, "stream": True, "reasoning": {"effort": "none"}}, split=True)

    record("completions", "POST", "/v1/completions",
           {"model": SERVED_NAME, "prompt": "The capital of France is", "max_tokens": 12,
            "temperature": 0})
    record("completions stream", "POST", "/v1/completions",
           {"model": SERVED_NAME, "prompt": "Once upon a time", "max_tokens": 12,
            "temperature": 0, "stream": True})
    record("llama completion", "POST", "/completion",
           {"prompt": "1, 2, 3,", "n_predict": 12, "temperature": 0})
    record("responses", "POST", "/v1/responses",
           {"model": SERVED_NAME, "input": "Say hi.", "max_output_tokens": 16,
            "temperature": 0}, split=True)
    record("responses messages", "POST", "/v1/responses",
           {"model": SERVED_NAME, "max_output_tokens": 16, "temperature": 0,
            "instructions": "Answer briefly.",
            "input": [{"role": "user", "content": "Say hi."}]}, split=True)
    record("responses answer", "POST", "/v1/responses",
           {"model": SERVED_NAME, "input": "What is 17 * 23? Answer briefly.",
            "max_output_tokens": 400, "temperature": 0}, split=True)
    record("responses stream", "POST", "/v1/responses",
           {"model": SERVED_NAME, "input": "Say hi.", "max_output_tokens": 16,
            "temperature": 0, "stream": True}, split=True)
    record("messages", "POST", "/v1/messages",
           {"model": SERVED_NAME, "max_tokens": 16, "temperature": 0,
            "messages": [{"role": "user", "content": "Say hi."}]}, split=True)
    record("messages system", "POST", "/v1/messages",
           {"model": SERVED_NAME, "max_tokens": 16, "temperature": 0,
            "system": "Answer in German.",
            "messages": [{"role": "user", "content": "Say hi."}]}, split=True)
    record("messages answer", "POST", "/v1/messages",
           {"model": SERVED_NAME, "max_tokens": 400, "temperature": 0,
            "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}]},
           split=True)
    # Reasoning controls and returned reasoning (Rust only; C++ answers the
    # equivalent chat completion).
    question = user("What is 17 * 23? Answer briefly.")
    history = [question[0], {"role": "assistant", "content": "391",
                             "reasoning_content": "17 * 23 = 391."},
               {"role": "user", "content": "And 17 * 24?"}]
    adapter("responses effort low", "/v1/responses",
            {"model": SERVED_NAME, "input": question[0]["content"],
             "max_output_tokens": 64, "temperature": 0,
             "reasoning": {"effort": "low", "summary": "auto"}},
            chat(question, 64, think=True, reasoning_effort="low"))
    adapter("responses effort none", "/v1/responses",
            {"model": SERVED_NAME, "input": question[0]["content"],
             "max_output_tokens": 64, "temperature": 0, "reasoning": {"effort": "none"}},
            chat(question, 64, think=False))
    adapter("responses history", "/v1/responses",
            {"model": SERVED_NAME, "max_output_tokens": 64, "temperature": 0,
             "input": [question[0],
                       {"type": "reasoning", "id": "rs_1", "summary": [],
                        "content": [{"type": "reasoning_text", "text": "17 * 23 = 391."}]},
                       {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "391"}]},
                       history[2]]},
            chat(history, 64, think=True))
    adapter("messages thinking disabled", "/v1/messages",
            {"model": SERVED_NAME, "max_tokens": 64, "temperature": 0,
             "thinking": {"type": "disabled"}, "messages": question},
            chat(question, 64, think=False))
    adapter("messages thinking enabled", "/v1/messages",
            {"model": SERVED_NAME, "max_tokens": 64, "temperature": 0,
             "thinking": {"type": "enabled", "budget_tokens": 1024}, "messages": question},
            chat(question, 64, think=True))
    adapter("messages history", "/v1/messages",
            {"model": SERVED_NAME, "max_tokens": 64, "temperature": 0,
             "messages": [question[0],
                          {"role": "assistant", "content": [
                              {"type": "thinking", "thinking": "17 * 23 = 391.",
                               "signature": ""},
                              {"type": "text", "text": "391"}]},
                          history[2]]},
            chat(history, 64, think=True))
    record("messages stream", "POST", "/v1/messages",
           {"model": SERVED_NAME, "max_tokens": 16, "temperature": 0, "stream": True,
            "messages": [{"role": "user", "content": "Say hi."}]})

    record("bad json", "POST", "/v1/chat/completions", b"{not json")
    record("missing messages", "POST", "/v1/chat/completions", {"model": SERVED_NAME})
    record("bad role", "POST", "/v1/chat/completions",
           {"model": SERVED_NAME, "messages": [{"role": "robot", "content": "x"}]})
    record("bad max tokens", "POST", "/v1/chat/completions",
           {"model": SERVED_NAME, "messages": user("x"), "max_tokens": -1})
    record("bad temperature", "POST", "/v1/chat/completions",
           {"model": SERVED_NAME, "messages": user("x"), "temperature": 9})
    # The C++ range message prints its bounds with `std::to_string`.
    error = records[-1][1]["body"].get("error", {}) if isinstance(records[-1][1]["body"], dict) else {}
    if "message" in error:
        error["message"] = re.sub(r"\b(\d+)\.0+\b", r"\1", error["message"])
    record("context overflow", "POST", "/v1/chat/completions",
           chat(user("word " * 20000), max_tokens=8))
    record("transfer encoding", "POST", "/v1/chat/completions", b"{}",
           {"Transfer-Encoding": "chunked"})

    # Retained turns: the second turn extends the first exactly.
    first = user("Tell me a short fact about the moon.")
    status, _, payload = request(port, "POST", "/v1/chat/completions", chat(first, 32))
    reply = json.loads(payload)["choices"][0]["message"].get("content", "")
    records.append(("turn 1", {"status": status, "body": mask(json.loads(payload))}))
    second = first + [{"role": "assistant", "content": reply},
                      {"role": "user", "content": "And one about Mars."}]
    record("turn 2", "POST", "/v1/chat/completions", chat(second, 32))
    record("turn 2 again", "POST", "/v1/chat/completions", chat(second, 32))
    record("turn 2 no cache", "POST", "/v1/chat/completions",
           chat(second, 32, cache_prompt=False))

    image_second = image_steps(port, record, records) if IMAGES else None

    # Before the disconnects: their token counts depend on timing.
    record("metrics", "GET", "/metrics")
    overload_and_disconnects(port, records)
    record("health after disconnects", "GET", "/health")
    record("chat after disconnects", "POST", "/v1/chat/completions",
           chat(user("Name three colors.")))

    if restart is not None:
        time.sleep(2.0)  # let the disk writer finish
        restart()
        records.extend(restored(port, (second, image_second)))
    return records, (second, image_second)


def restored(port, retained):
    """Requests whose prompts the previous server instance persisted."""
    second, image_second = retained
    records = []
    requests = [("restart turn 2", chat(second, 32)),
                ("restart chat", chat(user("Name three colors.")))]
    if image_second is not None:
        requests.append(("restart image turn 2", chat(image_second, 32)))
    for name, body in requests:
        status, content_type, payload = request(port, "POST", "/v1/chat/completions", body)
        records.append((name, {"status": status,
                               "content_type": content_type.split(";")[0],
                               "body": normalize_body(content_type, payload)}))
    return records


def open_stream(port, body, source="127.0.0.1"):
    """Starts a streaming chat and returns (connection, response) after the
    first generated token. The role event comes before prefill; waiting for a
    token makes what follows independent of how fast each server admits
    requests during a prefill."""
    conn = connection(port, source=source)
    conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                 {"Content-Type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        return conn, response
    for line in response:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        delta = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {})
        if delta.get("content") or delta.get("reasoning_content"):
            break
    return conn, response


def drop(conn):
    try:
        if conn.sock is not None:
            conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    conn.close()


def send_only(port, body, source):
    """Sends a chat request without reading the response."""
    conn = connection(port, source=source)
    conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                 {"Content-Type": "application/json"})
    return conn


def overload_and_disconnects(port, records):
    """With one session, --max-pending 2 and --max-pending-per-client 1:
    client 1 decodes, clients 2 and 3 queue, a second request of client 2
    and a request of client 4 are rejected, then every client disconnects
    (queued ones and the decoding one). With more sessions the queued
    requests run instead, and the later ones are served."""
    long_body = chat(user("Write the numbers from 1 to 3000, one per line, without any "
                          "other text."), max_tokens=3000, stream=True)
    decoding, response = open_stream(port, long_body, source="127.0.0.1")
    records.append(("decoding stream", {"status": response.status}))
    queued = [send_only(port, chat(user("Queued request"), max_tokens=8), "127.0.0.2")]
    time.sleep(0.3)
    status, content_type, payload = request(port, "POST", "/v1/chat/completions",
                                            chat(user("Hi"), max_tokens=4),
                                            source="127.0.0.2")
    records.append(("client queue full", {"status": status,
                                          "body": normalize_body(content_type, payload)}))
    queued.append(send_only(port, chat(user("Queued too"), max_tokens=8), "127.0.0.3"))
    time.sleep(0.3)
    status, content_type, payload = request(port, "POST", "/v1/chat/completions",
                                            chat(user("Hi"), max_tokens=4),
                                            source="127.0.0.4")
    records.append(("queue full", {"status": status,
                                   "body": normalize_body(content_type, payload)}))
    for conn in queued:
        drop(conn)
    time.sleep(0.3)
    drop(decoding)
    time.sleep(1.0)
    # A long prompt whose client leaves during prefill.
    prefill = send_only(port, chat(user("lorem ipsum " * 1500), max_tokens=8, stream=True),
                        "127.0.0.5")
    time.sleep(0.05)
    drop(prefill)
    time.sleep(2.0)


def wav_bytes(channels, width, rate, frames):
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return out.getvalue()


def audio_inputs(path):
    """The recording (mono PCM24 at 48 kHz), 16-bit stereo at 24 kHz
    (mixing and resampling), four copies at 16-bit mono 24 kHz (several
    chunks at --context 1024, within the 8 MiB body limit), and the 16-bit
    mono samples for the realtime socket."""
    with wave.open(str(path)) as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    if (params.nchannels, params.sampwidth, params.framerate) != (1, 3, 48000):
        raise SystemExit(f"{path}: expected mono PCM24 at 48 kHz")
    # Every second 24-bit sample, truncated to its upper 16 bits.
    pcm16 = b"".join(frames[i + 1:i + 3] for i in range(0, len(frames), 6))
    stereo = b"".join(pcm16[i:i + 2] * 2 for i in range(0, len(pcm16), 2))
    return {"wav": path.read_bytes(),
            "long": wav_bytes(1, 2, 24000, pcm16 * 4),
            "stereo": wav_bytes(2, 2, 24000, stereo),
            "pcm24k": pcm16}


def multipart(fields, boundary="gufo-serve-diff"):
    body = b""
    for name, value, filename in fields:
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'.encode()
        if filename:
            body += f'; filename="{filename}"\r\nContent-Type: audio/wav'.encode()
        body += b"\r\n\r\n" + (value if isinstance(value, bytes) else value.encode())
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def transcription(audio, **fields):
    return multipart([("file", audio, "audio.wav"),
                      *((name, str(value), None) for name, value in fields.items())])


def realtime_mask(value):
    if isinstance(value, dict):
        return {key: ("<masked>" if key in REALTIME_MASKED_KEYS else realtime_mask(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [realtime_mask(item) for item in value]
    return value


class WebSocket:
    """A minimal client: text frames out (masked), frames in (unmasked)."""

    def __init__(self, port, path):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=600)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                           "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                           "\r\n").encode())
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self.sock.recv(65536)
            if not chunk:
                break
            data += chunk
        head, _, self.buffer = data.partition(b"\r\n\r\n")
        self.status = int(head.split()[1]) if head else 0
        if self.status != 101:
            # The rejection body follows the headers until the server closes.
            while chunk := self.sock.recv(65536):
                self.buffer += chunk
            self.body = self.buffer.decode("utf-8", "replace")

    def send(self, event):
        payload = json.dumps(event).encode()
        header = bytes([0x81])
        if len(payload) < 126:
            header += bytes([0x80 | len(payload)])
        elif len(payload) < 65536:
            header += bytes([0x80 | 126]) + struct.pack(">H", len(payload))
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", len(payload))
        mask_key = os.urandom(4)
        masked = bytes(byte ^ mask_key[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(header + mask_key + masked)

    def read(self, count):
        while len(self.buffer) < count:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("WebSocket closed")
            self.buffer += chunk
        data, self.buffer = self.buffer[:count], self.buffer[count:]
        return data

    def receive(self):
        """The next text event as masked JSON, or ("close", code)."""
        first, second = self.read(2)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack(">H", self.read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self.read(8))[0]
        payload = self.read(length)
        opcode = first & 0x0F
        if opcode == 8:
            return ["close", struct.unpack(">H", payload[:2])[0] if payload else None]
        return realtime_mask(json.loads(payload))

    def until(self, kinds):
        events = []
        while True:
            event = self.receive()
            events.append(event)
            if isinstance(event, list) or event.get("type") in kinds:
                return events

    def close(self):
        self.sock.close()


def asr_session(port):
    """The scripted transcription session; returns [(step, record)]."""
    records = []
    audio = audio_inputs(AUDIO)

    def call(method, path, body=None, headers=None):
        conn = connection(port)
        conn.request(method, path, body, headers or {})
        response = conn.getresponse()
        complete = True
        try:
            payload = response.read()
        except http.client.IncompleteRead as error:
            # A stream that fails omits its terminating chunk.
            payload, complete = error.partial, False
        content_type = response.getheader("Content-Type", "")
        result = {"status": response.status, "complete": complete,
                  "content_type": content_type.split(";")[0],
                  "schema": response.getheader("X-Gufo-Schema"),
                  "backend": response.getheader("X-Gufo-ASR-Backend"),
                  "timing": [part.split(";")[0] for part in
                             response.getheader("Server-Timing", "").split(",") if part],
                  "body": normalize_body(content_type, payload)}
        conn.close()
        return result

    def record(name, method, path, body=None, headers=None):
        records.append((name, call(method, path, body, headers)))

    def post(name, form):
        record(name, "POST", "/v1/audio/transcriptions", *form)

    for path in ("/health", "/v1/health", "/ready", "/v1/models", "/metrics"):
        record(f"get {path}", "GET", path)
    record("unknown route", "GET", "/v1/unknown")
    record("options", "OPTIONS", "/v1/audio/transcriptions")
    record("get transcriptions", "GET", "/v1/audio/transcriptions")

    post("json", transcription(audio["wav"]))
    post("text", transcription(audio["wav"], response_format="text"))
    post("verbose json", transcription(audio["wav"], response_format="verbose_json",
                                       model=SERVED_NAME, temperature="0"))
    post("stream", transcription(audio["wav"], stream="true"))
    post("forced language and context",
         transcription(audio["wav"], language="en", prompt="A podcast about music."))
    post("language by name", transcription(audio["wav"], language="English"))
    post("max tokens 8", transcription(audio["wav"], max_tokens="8"))
    post("max tokens 8 stream", transcription(audio["wav"], max_tokens="8", stream="true"))
    post("long audio", transcription(audio["long"], response_format="verbose_json"))
    post("long audio stream", transcription(audio["long"], stream="true"))
    post("stereo 24 kHz", transcription(audio["stereo"]))

    post("unsupported language", transcription(audio["wav"], language="xx"))
    post("unsupported language stream",
         transcription(audio["wav"], language="xx", stream="true"))
    post("invalid wav", transcription(b"RIFF\x00\x00\x00\x00WAVEjunk"))
    post("missing file", multipart([("model", SERVED_NAME, None)]))
    post("unknown field", transcription(audio["wav"], speed="1"))
    post("invalid model", transcription(audio["wav"], model="whisper-1"))
    post("temperature", transcription(audio["wav"], temperature="0.5"))
    post("timestamps", transcription(audio["wav"], **{"timestamp_granularities[]": "word"}))
    post("stream text", transcription(audio["wav"], stream="true", response_format="text"))
    post("stream invalid", transcription(audio["wav"], stream="yes"))
    post("max tokens 0", transcription(audio["wav"], max_tokens="0"))
    post("max tokens 513", transcription(audio["wav"], max_tokens="513"))
    post("invalid format", transcription(audio["wav"], response_format="srt"))
    record("not multipart", "POST", "/v1/audio/transcriptions", b"{}",
           {"Content-Type": "application/json"})

    # Three at once: two share the frontend, the third waits; the long
    # recording yields the device between its chunks.
    concurrent = {}

    def run(name, form):
        concurrent[name] = call("POST", "/v1/audio/transcriptions", *form)

    threads = [threading.Thread(target=run, args=(name, form)) for name, form in (
        ("concurrent long", transcription(audio["long"])),
        ("concurrent short", transcription(audio["wav"], stream="true")),
        ("concurrent stereo", transcription(audio["stereo"], response_format="text")))]
    for thread in threads:
        thread.start()
        time.sleep(0.2)
    for thread in threads:
        thread.join()
    records.extend(sorted(concurrent.items()))

    # A streaming client that leaves after its first delta, then a request
    # that must be served normally.
    body, headers = transcription(audio["long"], stream="true")
    conn = connection(port)
    conn.request("POST", "/v1/audio/transcriptions", body, headers)
    # The response keeps the socket open after the connection closes.
    sock = conn.sock
    response = conn.getresponse()
    for line in response:
        if line.startswith(b"data: "):
            break
    sock.shutdown(socket.SHUT_RDWR)
    response.close()
    conn.close()
    time.sleep(1.0)
    post("after disconnect", transcription(audio["wav"], response_format="text"))

    # Realtime transcription.
    for name, path in (("realtime invalid model", "/v1/realtime?model=whisper-1"),
                       ("realtime invalid intent", "/v1/realtime?intent=conversation")):
        socket_ = WebSocket(port, path)
        records.append((name, {"status": socket_.status, "body": socket_.body}))
        socket_.close()
    ws = WebSocket(port, f"/v1/realtime?intent=transcription&model={SERVED_NAME}")
    events = [ws.receive()]

    def exchange(name, *sent, until=("error",)):
        for event in sent:
            ws.send(event)
        records.append((f"realtime {name}", ws.until(until)))

    def append(pcm, step=48000):
        return [{"type": "input_audio_buffer.append",
                 "audio": base64.b64encode(pcm[i:i + step]).decode()}
                for i in range(0, len(pcm), step)]

    records.append(("realtime created", events))
    exchange("commit empty", {"type": "input_audio_buffer.commit", "event_id": "c1"})
    exchange("unsupported event", {"type": "response.create"})
    exchange("invalid audio", {"type": "input_audio_buffer.append", "audio": "@@@"})
    exchange("bad rate", {"type": "session.update", "session": {"audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 16000}}}}})
    exchange("turn detection", {"type": "session.update", "session": {"audio": {"input": {
        "turn_detection": {"type": "server_vad"}}}}})
    exchange("update", {"type": "session.update", "session": {
        "type": "transcription", "audio": {"input": {"transcription": {
            "model": SERVED_NAME, "language": "en", "prompt": "A podcast."}}}}},
        until=("session.updated", "error"))
    exchange("transcribe", *append(audio["pcm24k"]), {"type": "input_audio_buffer.commit"},
             until=("conversation.item.input_audio_transcription.completed",
                    "conversation.item.input_audio_transcription.failed", "error"))
    exchange("update with audio", *append(audio["pcm24k"][:96000]),
             {"type": "session.update", "session": {}})
    exchange("clear", {"type": "input_audio_buffer.clear"},
             until=("input_audio_buffer.cleared", "error"))
    exchange("auto language", {"type": "session.update", "session": {"audio": {"input": {
        "transcription": {"language": ""}}}}}, until=("session.updated", "error"))
    exchange("transcribe again", *append(audio["pcm24k"][:240000]),
             {"type": "input_audio_buffer.commit"},
             until=("conversation.item.input_audio_transcription.completed",
                    "conversation.item.input_audio_transcription.failed", "error"))
    ws.close()
    time.sleep(0.5)
    record("metrics after", "GET", "/metrics")
    return records


def compare(baseline, candidate):
    differences = []
    names = [name for name, _ in baseline]
    if names != [name for name, _ in candidate]:
        differences.append({"step": "<sequence>", "baseline": names,
                            "candidate": [name for name, _ in candidate]})
        return differences
    for (name, left), (_, right) in zip(baseline, candidate):
        if left != right:
            differences.append({"step": name, "baseline": left, "candidate": right})
    return differences


def side_args(label, args):
    """The serve arguments of a side: the Rust server sends llama.cpp's
    Responses reasoning, which the C++ answers are converted to
    (`responses_reasoning`)."""
    if label.startswith("candidate") and not ASR:
        return [*args, "--responses-reasoning-target", "reasoning-text"]
    return args


def run_side(binary, args, port, work, label):
    cache_dir = work / f"{label}-cache"
    cache_dir.mkdir()
    log_path = work / f"{label}.log"
    server = Server(binary, side_args(label, args), port, cache_dir, log_path)
    server.start()

    def restart():
        server.stop()
        server.start()

    try:
        if ASR:
            records, second = asr_session(port), []
        else:
            records, second = session(port, restart)
    finally:
        server.stop()
    time.sleep(2.0)
    return records, server.request_log(), second


def interchange(binary, args, port, cache_dir, work, label, second):
    """Starts `binary` on another server's disk cache and replays the
    restored requests."""
    server = Server(binary, side_args(label, args), port, cache_dir, work / f"{label}.log")
    server.start()
    try:
        return restored(port, second)
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="C++ gufo binary")
    parser.add_argument("--candidate", required=True, help="Rust gufo binary")
    parser.add_argument("--port", type=int, default=18431)
    parser.add_argument("--output", type=Path, help="JSON report")
    parser.add_argument("--keep", action="store_true", help="keep logs and caches")
    parser.add_argument("--work", type=Path, help="parent directory for logs and caches")
    parser.add_argument("--images", action="store_true",
                        help="add image inputs (the server needs a vision model)")
    parser.add_argument("--interchange", action="store_true",
                        help="also start each server on the other's disk cache")
    parser.add_argument("--asr", action="store_true",
                        help="serve Qwen3-ASR and run the transcription session")
    parser.add_argument("--audio", type=Path,
                        default=Path.home() / ".cache/gufo/fixtures/qwen3-asr-en.wav",
                        help="mono PCM24 WAV at 48 kHz for --asr")
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    global IMAGES, ASR, AUDIO, SIDE
    IMAGES = args.images
    ASR, AUDIO = args.asr, args.audio
    if ASR and (args.images or args.interchange):
        parser.error("--asr has no image inputs or disk cache")
    serve_args = args.serve_args[1:] if args.serve_args[:1] == ["--"] else args.serve_args

    work = Path(tempfile.mkdtemp(prefix="gufo-serve-diff-", dir=args.work))
    try:
        results = {}
        for label, binary in (("baseline", args.baseline), ("candidate", args.candidate)):
            SIDE = label
            started = time.monotonic()
            records, log, second = run_side(binary, serve_args, args.port, work, label)
            results[label] = {"records": records, "log": log, "second": second,
                              "seconds": round(time.monotonic() - started, 1)}
            print(f"{label}: {len(records)} steps, {len(log)} logged requests "
                  f"in {results[label]['seconds']} s", file=sys.stderr)
        differences = compare(results["baseline"]["records"],
                              results["candidate"]["records"])
        if args.interchange:
            expected = [item for item in results["baseline"]["records"]
                        if item[0].startswith("restart ")]
            for label, binary, cache in (
                    ("candidate on baseline cache", args.candidate, work / "baseline-cache"),
                    ("baseline on candidate cache", args.baseline, work / "candidate-cache")):
                replayed = interchange(binary, serve_args, args.port, cache, work,
                                       label.replace(" ", "-"),
                                       results["baseline"]["second"])
                for difference in compare(expected, replayed):
                    difference["step"] = f"{label}: {difference['step']}"
                    differences.append(difference)
                print(f"{label}: {len(replayed)} restored requests", file=sys.stderr)
        base_log, cand_log = results["baseline"]["log"], results["candidate"]["log"]
        if base_log != cand_log:
            for index in range(max(len(base_log), len(cand_log))):
                left = base_log[index] if index < len(base_log) else None
                right = cand_log[index] if index < len(cand_log) else None
                if left != right:
                    differences.append({"step": f"log[{index}]", "baseline": left,
                                        "candidate": right})
        report = {"baseline": args.baseline, "candidate": args.candidate,
                  "serve_args": serve_args, "steps": len(results["baseline"]["records"]),
                  "logged_requests": len(base_log), "differences": differences,
                  "work": str(work) if args.keep else None}
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        for difference in differences[:20]:
            print(json.dumps(difference, indent=1)[:4000], file=sys.stderr)
        verdict = "PASS" if not differences else f"FAIL ({len(differences)} differences)"
        print(f"serve-diff: {verdict}: {report['steps']} steps, "
              f"{report['logged_requests']} logged requests", file=sys.stderr)
        return 0 if not differences else 1
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept {work}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
