#!/usr/bin/env python3
"""
ab.py — A/B comparison of two gufo binaries on one machine.

Used for the Rust migration gates (docs/RUST.md): the baseline (A)
is the C++ build from the separate upstream clone, the candidate (B) the Rust
build. There are two procedures with the same gates.

Interleaved (`ab.py --baseline A --candidate B <workload>`): both run the
same workload alternately in ABBA order, so drift in clocks, temperature or
background load affects both sides equally. The first invocation of a cold
device runs at clocks no later one reaches, so `--warmup-rounds` ABBA rounds
run first and are not judged. Performance is judged per round: the ratio of
B's to A's median request time within one round, bootstrapped over rounds.

Recorded (`ab.py record`, then `ab.py judge`): `record` runs one binary
alone, its warmup rounds and then its rounds, and stores the run. A baseline
is two runs of A recorded before the candidate's two (AABB), so each side has
a first run and a second that may profit from the page cache. It serves later
candidates, which record only BB, for as long as the same A binary runs the
same workload (the runs' `key`). `judge` judges each candidate run on its own
against the pooled baseline runs, unpaired: the ratio of B's median over its
invocations to A's, bootstrapped by resampling each side's invocations
independently. Two runs of one side are consistent when, for every gated
metric, their medians m1 and m2 satisfy |m1/m2 - 1| <= its tolerance. Two
inconsistent baseline runs, or two candidate runs that are inconsistent or of
which only one passes, call for a third (`next_run` in the report, exit
status 3); of three candidate runs, two must pass.

The candidate passes a gate when the upper bound of the 95% interval is at
most 1 + tolerance, i.e. it is plausibly no more than `tolerance` slower.
Load time and peak memory are judged like the request time, as are the
stages a workload gates (for `gufo serve` the client TTFT, the decode time
per token and the aggregate time per generated token at every concurrency
level).

Every invocation must produce identical outputs (transcript and token counts,
and the generated token IDs where both sides report them; for `gufo bench` the
SHA-256 traces of every tg run's token IDs and the DFlash2 acceptance; for
`gufo serve` the completions at the lowest concurrency), within a run and
across runs, so a candidate's outputs must equal the stored baseline's; the
first mismatch aborts. With `--allow-output-mismatch` the sides may differ,
but each must repeat its own. When a model's reference fixture matches the
audio (by SHA-256) and token limit, every reported ID sequence must equal it
as well.

Peak memory is what a run takes from the rest of the machine: host memory
the kernel cannot reclaim (`/proc/meminfo` MemTotal minus MemAvailable,
which includes the GTT) plus VRAM in use (sysfs `mem_info_vram_used`, outside
MemTotal), sampled while the process runs, above the level before it
started. The page cache counts as available, including the model files both
sides map from it. The counters are system-wide, so other processes add
noise. Peak device memory (VRAM plus GTT) is reported beside it but not
judged: on the APU both are the same RAM, and a buffer one side keeps in host
memory and the other on the device changes only the device counter.

Absolute numbers depend on firmware power limits and governors; the report
records them, but only the ratio is a verdict.

Both sides run with `oom_score_adj` 1000: device memory in GTT does not count
toward a process's OOM badness, so without it an overcommitted machine kills
unrelated processes (a shell, an editor session) instead of the model.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import platform
import os
import random
import re
import shlex
import signal
import statistics
import subprocess
import struct
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "gufo.ab.v2"
# One recorded run of one binary, and the judgement of recorded runs.
RUN_SCHEMA = "gufo.ab.run.v1"
JUDGE_SCHEMA = "gufo.ab.judge.v1"
REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))
from gufo import serving_bench  # noqa: E402
MEMORY_SAMPLE_SECONDS = 0.005


@dataclass
class Invocation:
    """One process run: `repeat` measured requests after warmup."""

    side: str
    round: int
    request_ms: list[float]
    load_ms: float
    stages: dict[str, float]
    identity: dict[str, Any]
    generated_ids: list[int] | None = None
    peak_memory_bytes: int | None = None
    peak_device_memory_bytes: int | None = None


def kill_first_on_oom() -> None:
    """Makes the child the OOM killer's first victim (raising needs no privilege)."""
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")


@dataclass
class Workload:
    name: str
    command: list[str]
    samples: str
    stage_fields: list[str] = field(default_factory=list)
    identity_fields: list[str] = field(default_factory=list)
    # Stages judged like the request time, with the same tolerance.
    gated_stages: list[str] = field(default_factory=list)

    def invoke(self, binary: Path, side: str, round_index: int, timeout: float) -> Invocation:
        completed = subprocess.run(
            [str(binary), *self.command], capture_output=True, text=True, timeout=timeout,
            preexec_fn=kill_first_on_oom,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"{side} ({binary}) failed with exit code {completed.returncode}: {detail}")
        return self.parse(side, round_index, completed.stdout, completed.stderr)

    def parse(self, side: str, round_index: int, stdout: str, stderr: str = "") -> Invocation:
        report = json.loads(stdout)
        runs = report["runs"]
        if not runs:
            raise ValueError(f"{side}: no measured runs")
        return Invocation(
            side=side,
            round=round_index,
            request_ms=[float(run[self.samples]) for run in runs],
            load_ms=float(report["timings"]["load_ms"]),
            stages={name: statistics.median(float(run[name]) for run in runs) for name in self.stage_fields},
            identity={name: report[name] for name in self.identity_fields},
            generated_ids=report.get("generated_ids"),
        )


def asr_workload(args: argparse.Namespace) -> Workload:
    return Workload(
        name="qwen3-asr-transcribe",
        command=[
            "transcribe",
            "--model", str(args.model),
            "--audio", str(args.audio),
            "--max-tokens", str(args.max_tokens),
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
            "--format", "json",
        ],
        samples="total_ms",
        stage_fields=["audio_encoder_ms", "text_decoder_ms"],
        identity_fields=["language", "text", "audio_samples", "audio_tokens", "prompt_tokens", "generated_tokens"],
    )


# The Qwen3.8 27B table has a params column, the Flash-Next and DeepSeek
# tables have not.
BENCH_ROW = re.compile(r"^\|(?:[^|]*\|){3,4}\s*(?P<test>[^|]+?)\s*\|\s*(?P<mean>[0-9.]+) ± (?P<deviation>[0-9.]+)\s*\|$")
BENCH_TRACE = re.compile(
    r"^(?:\[QwenBenchTrace\]: (?P<trace>.+)"
    r"|Qwen3\.8-Flash-Next tg (?P<flash>depth=\S+ cycles=\S+ drafted=\S+ accepted=\S+"
    r" lookup_proposed=\S+ lookup_accepted=\S+ output_sha256=\S+)"
    r"|DeepSeek tg (?P<deepseek>C=\S+ depth=\S+ request=\S+ accepted=\S+ drafted=\S+ steps=\S+"
    r" skipped=\S+ output_sha256=\S+))"
)
BENCH_ACCEPTANCE = re.compile(r"^(?P<test>\S+) acceptance=(?P<value>[0-9.]+)")


@dataclass
class TextWorkload(Workload):
    """`gufo bench`: the request time is milliseconds per generated token (the
    tg row); pp is a gated stage. The traces of every tg run's token IDs (and
    the speculative acceptance) must be identical."""

    def parse(self, side: str, round_index: int, stdout: str, stderr: str = "") -> Invocation:
        if "graph mismatches" in stderr:
            raise ValueError(f"{side}: decode graph mismatches reported")
        rows = {}
        load = None
        for line in stdout.splitlines():
            if line.startswith("[Model Load]:"):
                load = float(line.split()[2]) * 1000.0
            match = BENCH_ROW.match(line.strip())
            if match and match["test"] != "test":
                rows[match["test"]] = float(match["mean"])
        tg = [value for test, value in rows.items() if test.startswith("tg")]
        pp = [value for test, value in rows.items() if test.startswith("pp")]
        if load is None or len(tg) != 1 or len(pp) > 1:
            raise ValueError(f"{side}: expected one tg row and a load time in the bench output")
        traces = [m["trace"] or m["flash"] or m["deepseek"]
                  for line in stderr.splitlines() if (m := BENCH_TRACE.match(line))]
        acceptance = [m["value"] for line in stderr.splitlines() if (m := BENCH_ACCEPTANCE.match(line))]
        if not traces:
            raise ValueError(f"{side}: no token-ID traces (run the bench with -v)")
        return Invocation(
            side=side,
            round=round_index,
            request_ms=[1000.0 / tg[0]],
            load_ms=load,
            stages={"pp_ms_per_token": 1000.0 / pp[0]} if pp else {},
            identity={"tests": sorted(rows), "traces": traces, "acceptance": acceptance},
        )


def gguf_architecture(path: Path) -> str:
    """`general.architecture` of a GGUF file (version 2 or 3), read from its
    header; empty for an unreadable file or when the key is not among the
    leading string entries."""
    try:
        with open(path, "rb") as file:
            header = file.read(1 << 20)
    except OSError:
        return ""
    if header[:4] != b"GGUF":
        return ""
    offset = 4 + 4 + 8
    (count,) = struct.unpack_from("<Q", header, offset)
    offset += 8
    for _ in range(count):
        (length,) = struct.unpack_from("<Q", header, offset)
        key = header[offset + 8:offset + 8 + length].decode("utf-8", "replace")
        offset += 8 + length
        (kind,) = struct.unpack_from("<I", header, offset)
        offset += 4
        if kind != 8:  # only string values precede the architecture
            return ""
        (length,) = struct.unpack_from("<Q", header, offset)
        value = header[offset + 8:offset + 8 + length].decode("utf-8", "replace")
        offset += 8 + length
        if key == "general.architecture":
            return value
    return ""


def text_workload(args: argparse.Namespace) -> Workload:
    command = ["bench", "--model", str(args.model)]
    if args.prompt_tokens:
        command += ["-p", str(args.prompt_tokens)]
    command += ["-n", str(args.gen_tokens), "-r", str(args.repeat), "-v"]
    if args.depth:
        command += ["-d", str(args.depth)]
    suffix = ""
    if args.dflash_model is not None:
        command += ["--speculative", "dflash2", "--dflash-model", str(args.dflash_model)]
        suffix = "-dflash2"
    if args.mtp_model is not None:
        command += ["--speculative", "mtp", "--mtp-model", str(args.mtp_model)]
        suffix = "-mtp"
    if args.dspark_model is not None:
        command += ["--speculative", "dspark", "--dspark-model", str(args.dspark_model)]
        suffix = "-dspark"
    return TextWorkload(
        name=("deepseek-v4-flash-bench" if gguf_architecture(args.model) == "deepseek4"
              else "qwen3.8-bench") + suffix,
        command=command,
        samples="",
        stage_fields=["pp_ms_per_token"] if args.prompt_tokens else [],
        # Prompt processing is judged like generation.
        gated_stages=["pp_ms_per_token"] if args.prompt_tokens else [],
    )


@dataclass
class ServeWorkload(Workload):
    """`gufo serve` under gufo-serving-bench load. Each invocation starts the
    server, waits for /health (the load time), runs the warmup and measured
    rounds at every concurrency level and stops it. Both sides get the same
    prompts (a fixed nonce), distinct per round and request, so no request
    hits a prompt cache. The request time is the whole-request wall time at
    the lowest level. Its client TTFT and server decode time per token, and
    the aggregate time per generated token at every level, are gated too.
    Completions at the lowest level must be identical; with several requests
    in flight, batch composition depends on timing."""

    port: int = 18531
    prompt: str = ""
    max_tokens: int = 128
    concurrency: list[int] = field(default_factory=list)
    warmup: int = 1
    repetitions: int = 3

    def invoke(self, binary: Path, side: str, round_index: int, timeout: float) -> Invocation:
        env = {k: v for k, v in os.environ.items() if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
        env["NO_COLOR"] = "1"
        base_url = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + timeout
        with tempfile.TemporaryFile() as log:
            started = time.perf_counter()
            process = subprocess.Popen(
                [str(binary), *self.command, "--port", str(self.port)],
                stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=kill_first_on_oom,
            )
            try:
                while not healthy(base_url):
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        tail = log.read().decode(errors="replace").strip().splitlines()[-20:]
                        raise RuntimeError(f"{side} ({binary}) did not become healthy: " + "\n".join(tail))
                    time.sleep(0.05)
                load_ms = (time.perf_counter() - started) * 1000.0
                results, _ = serving_bench.measure_levels(
                    base_url=base_url,
                    model=serving_bench.discover_model(base_url, 30.0),
                    prompt=self.prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    concurrency_levels=self.concurrency,
                    warmup_rounds=self.warmup,
                    repetitions=self.repetitions,
                    timeout_seconds=max(1.0, deadline - time.monotonic()),
                    cache_prompt=False,
                    run_nonce="ab",
                )
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        return self.summarize(side, round_index, load_ms, results)

    def summarize(self, side: str, round_index: int, load_ms: float, results: dict[str, Any]) -> Invocation:
        low = results[f"c{min(self.concurrency)}"]
        samples = low["samples"]
        if any(sample["cache_hit"] for level in results.values() for sample in level["samples"]):
            raise ValueError(f"{side}: a benchmark request hit a prompt cache")
        stages = {
            "client_ttft_ms": statistics.median(sample["client_ttft_ms"] for sample in samples),
            "decode_ms_per_token": statistics.median(
                sample["decode_ms"] / sample["completion_tokens"] for sample in samples
            ),
        }
        for level in self.concurrency:
            rate = results[f"c{level}"]["aggregate"]["output_tokens_per_second"]["overall"]
            stages[f"c{level}_ms_per_output_token"] = 1000.0 / rate
        ordered = sorted(samples, key=lambda sample: (sample["repetition"], sample["request_index"]))
        return Invocation(
            side=side,
            round=round_index,
            request_ms=[sample["wall_ms"] for sample in samples],
            load_ms=load_ms,
            stages=stages,
            identity={
                "completions": [sample["completion_sha256"] for sample in ordered],
                "completion_tokens": [sample["completion_tokens"] for sample in ordered],
                "prompt_tokens": [sample["prompt_tokens"] for sample in ordered],
            },
        )


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/ready", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


@dataclass
class TtsWorkload(Workload):
    """`gufo serve tts`. Each invocation starts the server, waits for /ready
    (the load time), sends `warmup` buffered requests, then `repeat` buffered
    WAV requests (the request time) and as many streamed PCM requests, whose
    time to the first audio bytes and to the end are gated stages. Every
    request is seeded, so the WAV and the PCM stream must be identical
    across repetitions, rounds and sides."""

    port: int = 18541
    body: dict[str, Any] = field(default_factory=dict)
    warmup: int = 1
    repetitions: int = 3

    def post(self, base_url: str, body: dict[str, Any], timeout: float) -> tuple[bytes, float, float]:
        request = urllib.request.Request(
            base_url + "/v1/audio/speech", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        first = None
        chunks = []
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                if first is None:
                    first = time.perf_counter()
                chunks.append(chunk)
        finished = time.perf_counter()
        return b"".join(chunks), ((first or finished) - started) * 1000.0, (finished - started) * 1000.0

    def invoke(self, binary: Path, side: str, round_index: int, timeout: float) -> Invocation:
        env = {k: v for k, v in os.environ.items() if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
        env["NO_COLOR"] = "1"
        base_url = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + timeout
        with tempfile.TemporaryFile() as log:
            started = time.perf_counter()
            process = subprocess.Popen(
                [str(binary), *self.command, "--port", str(self.port)],
                stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=kill_first_on_oom,
            )
            try:
                while not ready(base_url):
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        tail = log.read().decode(errors="replace").strip().splitlines()[-20:]
                        raise RuntimeError(f"{side} ({binary}) did not become ready: " + "\n".join(tail))
                    time.sleep(0.05)
                load_ms = (time.perf_counter() - started) * 1000.0
                with urllib.request.urlopen(base_url + "/v1/models", timeout=30) as response:
                    model = json.load(response)["data"][0]["id"]
                buffered = {**self.body, "model": model, "response_format": "wav"}
                streamed = {**buffered, "response_format": "pcm"}
                remaining = lambda: max(1.0, deadline - time.monotonic())  # noqa: E731
                for _ in range(self.warmup):
                    self.post(base_url, buffered, remaining())
                request_ms, first_audio, streamed_ms, wavs, pcms = [], [], [], set(), set()
                for _ in range(self.repetitions):
                    wav, _, total = self.post(base_url, buffered, remaining())
                    request_ms.append(total)
                    wavs.add(hashlib.sha256(wav).hexdigest())
                    pcm, first, total = self.post(base_url, streamed, remaining())
                    first_audio.append(first)
                    streamed_ms.append(total)
                    pcms.add(hashlib.sha256(pcm).hexdigest())
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if len(wavs) != 1 or len(pcms) != 1:
            raise ValueError(f"{side}: seeded requests produced different audio")
        return Invocation(
            side=side,
            round=round_index,
            request_ms=request_ms,
            load_ms=load_ms,
            stages={
                "first_audio_ms": statistics.median(first_audio),
                "streamed_ms": statistics.median(streamed_ms),
            },
            identity={"wav_sha256": wavs.pop(), "pcm_sha256": pcms.pop()},
        )


@dataclass
class EmbeddingWorkload(Workload):
    """`gufo serve embedding`. Each invocation starts the server, waits for
    /ready (the load time), sends `warmup` batches, then `repeat` times one
    batch of `batch` inputs (the request time) and one single short input,
    whose latency is a gated stage. Vectors return as base64, so the bytes
    must be identical across repetitions, rounds and sides."""

    port: int = 18543
    inputs: list[str] = field(default_factory=list)
    query: str = ""
    warmup: int = 1
    repetitions: int = 3

    def post(self, base_url: str, body: dict[str, Any], timeout: float) -> tuple[bytes, float]:
        request = urllib.request.Request(
            base_url + "/v1/embeddings", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        elapsed = (time.perf_counter() - started) * 1000.0
        vectors = b"".join(base64.b64decode(item["embedding"]) for item in payload["data"])
        return vectors, elapsed

    def invoke(self, binary: Path, side: str, round_index: int, timeout: float) -> Invocation:
        env = {k: v for k, v in os.environ.items() if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
        env["NO_COLOR"] = "1"
        base_url = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + timeout
        with tempfile.TemporaryFile() as log:
            started = time.perf_counter()
            process = subprocess.Popen(
                [str(binary), *self.command, "--port", str(self.port)],
                stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=kill_first_on_oom,
            )
            try:
                while not ready(base_url):
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        tail = log.read().decode(errors="replace").strip().splitlines()[-20:]
                        raise RuntimeError(f"{side} ({binary}) did not become ready: " + "\n".join(tail))
                    time.sleep(0.05)
                load_ms = (time.perf_counter() - started) * 1000.0
                with urllib.request.urlopen(base_url + "/v1/models", timeout=30) as response:
                    model = json.load(response)["data"][0]["id"]
                batch = {"model": model, "input": self.inputs, "encoding_format": "base64"}
                single = {"model": model, "input": [self.query], "encoding_format": "base64"}
                remaining = lambda: max(1.0, deadline - time.monotonic())  # noqa: E731
                for _ in range(self.warmup):
                    self.post(base_url, batch, remaining())
                    self.post(base_url, single, remaining())
                request_ms, single_ms, batches, singles = [], [], set(), set()
                for _ in range(self.repetitions):
                    vectors, elapsed = self.post(base_url, batch, remaining())
                    request_ms.append(elapsed)
                    batches.add(hashlib.sha256(vectors).hexdigest())
                    vectors, elapsed = self.post(base_url, single, remaining())
                    single_ms.append(elapsed)
                    singles.add(hashlib.sha256(vectors).hexdigest())
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if len(batches) != 1 or len(singles) != 1:
            raise ValueError(f"{side}: identical requests produced different vectors")
        return Invocation(
            side=side,
            round=round_index,
            request_ms=request_ms,
            load_ms=load_ms,
            stages={"single_ms": statistics.median(single_ms)},
            identity={"batch_sha256": batches.pop(), "single_sha256": singles.pop()},
        )


def png_rgba(png: bytes) -> bytes:
    """The pixels of a non-interlaced 8-bit RGBA PNG (both servers write
    that, with different compressed bytes)."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    offset, width, height, data = 8, 0, 0, b""
    while offset < len(png):
        length = int.from_bytes(png[offset:offset + 4], "big")
        kind = png[offset + 4:offset + 8]
        chunk = png[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            width, height = int.from_bytes(chunk[:4], "big"), int.from_bytes(chunk[4:8], "big")
            if chunk[8:13] != bytes([8, 6, 0, 0, 0]):
                raise ValueError("PNG is not 8-bit RGBA without interlacing")
        elif kind == b"IDAT":
            data += chunk
        offset += 12 + length
    raw = zlib.decompress(data)
    stride = width * 4
    rows: list[bytearray] = []
    previous = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        kind, row = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        for i in range(stride):
            left = row[i - 4] if i >= 4 else 0
            up = previous[i]
            corner = previous[i - 4] if i >= 4 else 0
            if kind == 1:
                row[i] = (row[i] + left) & 255
            elif kind == 2:
                row[i] = (row[i] + up) & 255
            elif kind == 3:
                row[i] = (row[i] + (left + up) // 2) & 255
            elif kind == 4:
                p = left + up - corner
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                row[i] = (row[i] + (left if pa <= pb and pa <= pc else up if pb <= pc else corner)) & 255
        rows.append(row)
        previous = row
    return b"".join(rows)


def multipart(fields: dict[str, Any], files: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = "gufo-ab-" + hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data) in files.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


IMAGE_METRICS = re.compile(r"event=generated .*? prompt_ms=(\S+) denoise_ms=(\S+) vae_ms=(\S+) total_ms=(\S+)")


@dataclass
class ImageWorkload(Workload):
    """`gufo serve image`. Each invocation starts the server, waits for
    /ready (the load time), sends `warmup` requests (the first uploads the
    weights), then `repeat` seeded requests (the request time); generations
    post JSON, edits multipart with the reference image. The server's
    prompt, denoise and VAE times are gated stages. The decoded pixels must
    be identical across repetitions, rounds and sides."""

    port: int = 18542
    fields: dict[str, Any] = field(default_factory=dict)
    edit: Path | None = None
    warmup: int = 1
    repetitions: int = 3

    def post(self, base_url: str, model: str, timeout: float) -> tuple[bytes, float]:
        fields = {**self.fields, "model": model}
        if self.edit is None:
            body, kind = json.dumps(fields).encode(), "application/json"
            path = "/v1/images/generations"
        else:
            body, kind = multipart(fields, {"image": (self.edit.name, self.edit.read_bytes())})
            path = "/v1/images/edits"
        request = urllib.request.Request(base_url + path, data=body, headers={"Content-Type": kind})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        finished = time.perf_counter()
        return base64.b64decode(payload["data"][0]["b64_json"]), (finished - started) * 1000.0

    def invoke(self, binary: Path, side: str, round_index: int, timeout: float) -> Invocation:
        env = {k: v for k, v in os.environ.items() if k not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}
        env["NO_COLOR"] = "1"
        base_url = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + timeout
        with tempfile.TemporaryFile() as log:
            started = time.perf_counter()
            process = subprocess.Popen(
                [str(binary), *self.command, "--port", str(self.port)],
                stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=kill_first_on_oom,
            )
            try:
                while not ready(base_url):
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        tail = log.read().decode(errors="replace").strip().splitlines()[-20:]
                        raise RuntimeError(f"{side} ({binary}) did not become ready: " + "\n".join(tail))
                    time.sleep(0.05)
                load_ms = (time.perf_counter() - started) * 1000.0
                with urllib.request.urlopen(base_url + "/v1/models", timeout=30) as response:
                    model = json.load(response)["data"][0]["id"]
                remaining = lambda: max(1.0, deadline - time.monotonic())  # noqa: E731
                for _ in range(self.warmup):
                    self.post(base_url, model, remaining())
                request_ms, pixels = [], set()
                for _ in range(self.repetitions):
                    png, total = self.post(base_url, model, remaining())
                    request_ms.append(total)
                    pixels.add(hashlib.sha256(png_rgba(png)).hexdigest())
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            log.seek(0)
            metrics = IMAGE_METRICS.findall(log.read().decode(errors="replace"))
        if len(pixels) != 1:
            raise ValueError(f"{side}: seeded requests produced different images")
        measured = [[float(value) for value in line] for line in metrics[self.warmup:]]
        if len(measured) != self.repetitions:
            raise ValueError(f"{side}: expected {self.repetitions} generation log lines, found {len(measured)}")
        return Invocation(
            side=side,
            round=round_index,
            request_ms=request_ms,
            load_ms=load_ms,
            stages={
                "prompt_ms": statistics.median(line[0] for line in measured),
                "denoise_ms": statistics.median(line[1] for line in measured),
                "vae_ms": statistics.median(line[2] for line in measured),
            },
            identity={"rgba_sha256": pixels.pop()},
        )


def image_workload(args: argparse.Namespace) -> Workload:
    fields: dict[str, Any] = {
        "prompt": args.prompt, "size": args.size, "steps": args.steps, "seed": args.seed,
        "response_format": "b64_json",
    }
    return ImageWorkload(
        name="gufo-serve-image" + ("-" + args.name if args.name else ""),
        command=["serve", "image", "--model", str(args.model), *shlex.split(args.server_args)],
        samples="",
        stage_fields=["prompt_ms", "denoise_ms", "vae_ms"],
        gated_stages=["prompt_ms", "denoise_ms", "vae_ms"],
        port=args.port,
        fields=fields,
        edit=args.edit,
        warmup=args.warmup,
        repetitions=args.repeat,
    )


def tts_workload(args: argparse.Namespace) -> Workload:
    body: dict[str, Any] = {
        "input": args.text, "seed": args.seed, "max_new_tokens": args.max_tokens,
        **({"voice": args.voice} if args.voice else {}), **json.loads(args.request),
    }
    return TtsWorkload(
        name="gufo-serve-tts" + ("-" + args.name if args.name else ""),
        command=["serve", "tts", "--model", str(args.model), *shlex.split(args.server_args)],
        samples="",
        stage_fields=["first_audio_ms", "streamed_ms"],
        gated_stages=["first_audio_ms", "streamed_ms"],
        port=args.port,
        body=body,
        warmup=args.warmup,
        repetitions=args.repeat,
    )


def embedding_workload(args: argparse.Namespace) -> Workload:
    # Distinct documents of about `--tokens` tokens each, as an index build
    # embeds them, and one short query.
    sentence = ("The harbor master logged {index} ships at dawn, each carrying grain, timber "
                "and letters for the northern towns, while the tide turned slowly. ")
    words = max(1, args.tokens // 24)
    inputs = [f"title: none | text: {sentence.format(index=i) * words}" for i in range(args.batch)]
    return EmbeddingWorkload(
        name=f"gufo-serve-embedding-b{args.batch}-t{args.tokens}",
        command=["serve", "embedding", "--model", str(args.model), *shlex.split(args.server_args)],
        samples="",
        stage_fields=["single_ms"],
        gated_stages=["single_ms"],
        port=args.port,
        inputs=inputs,
        query="task: search result | query: What do the ships carry to the northern towns?",
        warmup=args.warmup,
        repetitions=args.repeat,
    )


def serve_workload(args: argparse.Namespace) -> Workload:
    levels = serving_bench.parse_concurrency_levels(args.concurrency)
    command = [
        "serve", "llm",
        "--model", str(args.model),
        "--sessions", str(max(levels)),
        "--context", str(args.context),
        *shlex.split(args.server_args),
    ]
    prompt = "\n\n".join(serving_bench.DEFAULT_PROMPT for _ in range(args.prompt_repeat))
    return ServeWorkload(
        name="gufo-serve" + ("-" + args.name if args.name else ""),
        command=command,
        samples="",
        stage_fields=["client_ttft_ms", "decode_ms_per_token", *(f"c{c}_ms_per_output_token" for c in levels)],
        gated_stages=["client_ttft_ms", "decode_ms_per_token", *(f"c{c}_ms_per_output_token" for c in levels)],
        port=args.port,
        prompt=prompt,
        max_tokens=args.max_tokens,
        concurrency=levels,
        warmup=args.warmup,
        repetitions=args.repeat,
    )


def abba_order(round_index: int) -> tuple[str, str]:
    """A B, B A, A B, ...: each side leads equally often."""
    return ("A", "B") if round_index % 2 == 0 else ("B", "A")


def paired_ratios(invocations: list[Invocation], metric) -> list[float]:
    """Per round: metric(B) / metric(A), both measured back to back."""
    by_round: dict[int, dict[str, Invocation]] = {}
    for invocation in invocations:
        by_round.setdefault(invocation.round, {})[invocation.side] = invocation
    return [metric(sides["B"]) / metric(sides["A"]) for _, sides in sorted(by_round.items()) if len(sides) == 2]


def bootstrap_interval(values: list[float], samples: int = 10_000, seed: int = 20260926) -> tuple[float, float]:
    """95% percentile interval of the median, resampling rounds."""
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)
    )
    return medians[int(0.025 * samples)], medians[int(0.975 * samples) - 1]


def verdict(ratios: list[float], tolerance: float) -> dict[str, Any]:
    low, high = bootstrap_interval(ratios)
    return {
        "median_ratio": statistics.median(ratios),
        "ci95": [low, high],
        "rounds": len(ratios),
        "tolerance": tolerance,
        "pass": high <= 1.0 + tolerance,
    }


def unpaired_interval(candidate: list[float], baseline: list[float], samples: int = 10_000,
                      seed: int = 20260926) -> tuple[float, float]:
    """95% percentile interval of median(candidate) / median(baseline),
    resampling each side's invocations independently."""
    rng = random.Random(seed)
    ratios = sorted(
        statistics.median(rng.choices(candidate, k=len(candidate)))
        / statistics.median(rng.choices(baseline, k=len(baseline)))
        for _ in range(samples)
    )
    return ratios[int(0.025 * samples)], ratios[int(0.975 * samples) - 1]


def unpaired_verdict(candidate: list[float], baseline: list[float], tolerance: float) -> dict[str, Any]:
    """B's median against A's for runs that were not interleaved, one value
    per invocation."""
    low, high = unpaired_interval(candidate, baseline)
    return {
        "median_ratio": statistics.median(candidate) / statistics.median(baseline),
        "ci95": [low, high],
        "rounds": len(candidate),
        "baseline_rounds": len(baseline),
        "tolerance": tolerance,
        "pass": high <= 1.0 + tolerance,
    }


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=10) if len(ordered) >= 2 else [ordered[0]] * 9
    return {"median": statistics.median(ordered), "p10": q[0], "p90": q[-1], "count": len(ordered)}


def first_identity_mismatch(invocations: list[Invocation], reference: dict[str, Any] | None = None) -> str | None:
    first = invocations[0]
    for invocation in invocations[1:]:
        for key, expected in first.identity.items():
            got = invocation.identity.get(key)
            if got != expected:
                return (
                    f"{key} differs: {first.side} round {first.round} gave {expected!r}, "
                    f"{invocation.side} round {invocation.round} gave {got!r}"
                )
    reporting = [inv for inv in invocations if inv.generated_ids is not None]
    for invocation in reporting[1:]:
        if invocation.generated_ids != reporting[0].generated_ids:
            return (
                f"generated_ids differ: {reporting[0].side} round {reporting[0].round} and "
                f"{invocation.side} round {invocation.round}"
            )
    if reference is not None:
        for invocation in invocations:
            if "text" in invocation.identity and invocation.identity["text"] != reference["text"]:
                return f"text of {invocation.side} round {invocation.round} differs from the reference fixture"
            if invocation.generated_ids is not None and invocation.generated_ids != reference["generated_ids"]:
                return f"generated_ids of {invocation.side} round {invocation.round} differ from the reference fixture"
    return None


def find_reference(audio: Path, max_tokens: int) -> tuple[Path, dict[str, Any]] | None:
    """The model fixture recorded for this audio file and token limit, if any."""
    try:
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    except OSError:
        return None
    for path in sorted(REPOSITORY.glob("src/models/*/tests/fixtures/*.json")):
        try:
            fixture = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(fixture, dict):
            continue
        if fixture.get("audio_sha256") == digest and fixture.get("max_new_tokens") == max_tokens:
            return path, fixture
    return None


@dataclass(frozen=True)
class MemorySample:
    """Bytes in use: `device` is VRAM plus GTT, `total` the host memory the
    kernel cannot reclaim plus VRAM (None without /proc/meminfo)."""

    device: int
    total: int | None

    def below(self, other: "MemorySample") -> bool:
        return self.device < other.device or (
            self.total is not None and other.total is not None and self.total < other.total
        )


class MemoryCounters:
    """The amdgpu memory counters of every device and the host's meminfo."""

    def __init__(self, sysfs: Path, meminfo: Path) -> None:
        self.vram = sorted(sysfs.glob("class/drm/card*/device/mem_info_vram_used"))
        self.gtt = sorted(sysfs.glob("class/drm/card*/device/mem_info_gtt_used"))
        self.meminfo = meminfo

    @property
    def device_measured(self) -> bool:
        return bool(self.vram or self.gtt)

    def sample(self) -> MemorySample:
        vram = sum(int(read_first(str(path)) or 0) for path in self.vram)
        gtt = sum(int(read_first(str(path)) or 0) for path in self.gtt)
        host = host_unavailable(self.meminfo)
        return MemorySample(device=vram + gtt, total=None if host is None else host + vram)


def host_unavailable(meminfo: Path) -> int | None:
    """MemTotal minus MemAvailable in bytes."""
    fields: dict[str, int] = {}
    try:
        with meminfo.open(encoding="utf-8") as f:
            for line in f:
                name, _, value = line.partition(":")
                if name in ("MemTotal", "MemAvailable"):
                    fields[name] = int(value.split()[0]) * 1024
                    if len(fields) == 2:
                        return fields["MemTotal"] - fields["MemAvailable"]
    except (OSError, ValueError, IndexError):
        pass
    return None


def settled_memory(counters: MemoryCounters, interval: float = 0.1, limit: float = 30.0) -> MemorySample:
    """Memory in use once it stops falling: the driver and the kernel release
    a finished process's memory asynchronously, and the previous server's
    would otherwise count as the baseline."""
    deadline = time.monotonic() + limit
    previous = counters.sample()
    while time.monotonic() < deadline:
        time.sleep(interval)
        current = counters.sample()
        if not current.below(previous):
            return current
        previous = current
    return previous


class MemorySampler:
    """Polls memory while a process runs and keeps the peak of each counter."""

    def __init__(self, counters: MemoryCounters) -> None:
        self.counters = counters
        self.device = 0
        self.total: int | None = None
        self.update()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def update(self) -> None:
        current = self.counters.sample()
        self.device = max(self.device, current.device)
        if current.total is not None:
            self.total = current.total if self.total is None else max(self.total, current.total)

    def poll(self) -> None:
        while not self.stopped.wait(MEMORY_SAMPLE_SECONDS):
            self.update()

    def __enter__(self) -> "MemorySampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stopped.set()
        self.thread.join()
        self.update()


def read_first(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def hip_runtime(binary: Path) -> dict[str, Any]:
    """The libamdhip64 the dynamic loader resolves for `binary`, and its HIP version."""
    try:
        ldd = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"library": None, "hip_version": None}
    match = re.search(r"libamdhip64\.so\S*\s+=>\s+(\S+)", ldd)
    library = Path(match.group(1)).resolve() if match else None
    version = None
    # <prefix>/lib/libamdhip64.so or /usr/lib/<triple>/libamdhip64.so
    for prefix in list(library.parents)[1:3] if library else []:
        text = read_first(str(prefix / "include/hip/hip_version.h"))
        fields = dict(re.findall(r"#define HIP_VERSION_(MAJOR|MINOR|PATCH) (\d+)", text or ""))
        if len(fields) == 3:
            version = f"{fields['MAJOR']}.{fields['MINOR']}.{fields['PATCH']}"
            break
    return {"library": str(library) if library else None, "hip_version": version}


def git_revision(binary: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(binary.resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def describe_binary(binary: Path) -> dict[str, Any]:
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    return {
        "path": str(binary.resolve()),
        "sha256": digest,
        "git_revision": git_revision(binary),
        **hip_runtime(binary),
    }


def environment() -> dict[str, Any]:
    dpm = sorted({level for p in Path("/sys/class/drm").glob("card*/device/power_dpm_force_performance_level")
                  if (level := read_first(str(p)))})
    cpu = "/sys/devices/system/cpu/cpu0/cpufreq"
    model = None
    for line in (read_first("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    return {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "kernel": platform.release(),
        "cpu": model,
        "gpu_dpm_performance_level": dpm,
        "cpu_governor": read_first(f"{cpu}/scaling_governor"),
        "cpu_energy_performance_preference": read_first(f"{cpu}/energy_performance_preference"),
    }


def gpu_temperature(sysfs: Path) -> float | None:
    """The GPU's edge temperature in °C (amdgpu hwmon `temp1_input`), if readable."""
    for path in sorted(sysfs.glob("class/drm/card*/device/hwmon/hwmon*/temp1_input")):
        value = read_first(str(path))
        if value is not None and value.isdigit():
            return int(value) / 1000.0
    return None


def cool_down(sysfs: Path, limit: float | None, timeout: float) -> dict[str, Any]:
    """Waits until the GPU is at most `limit` °C, for at most `timeout`
    seconds. The APU runs at its power limit, so a hotter GPU clocks lower:
    a run that starts after another one is slower than one that starts cool."""
    start = time.monotonic()
    temperature = gpu_temperature(sysfs)
    while (limit is not None and temperature is not None and temperature > limit
           and time.monotonic() - start < timeout):
        time.sleep(5)
        temperature = gpu_temperature(sysfs)
    return {"limit_c": limit, "waited_s": round(time.monotonic() - start, 1),
            "start_temperature_c": temperature}


def run_invocation(
    binary: Path, workload: Workload, side: str, round_index: int, timeout: float, counters: MemoryCounters
) -> Invocation:
    before = settled_memory(counters)
    with MemorySampler(counters) as memory:
        invocation = workload.invoke(binary, side, round_index, timeout)
    if before.total is not None and memory.total is not None:
        invocation.peak_memory_bytes = memory.total - before.total
    if counters.device_measured:
        invocation.peak_device_memory_bytes = memory.device - before.device
    return invocation


def measure(
    binaries: dict[str, Path], order, workload: Workload, args: argparse.Namespace,
    reference: dict[str, Any] | None, within_side: bool,
) -> list[Invocation]:
    """`--warmup-rounds` unjudged rounds, then `--rounds` measured ones, each
    running the sides in `order(round)`. The first output mismatch aborts:
    between any two invocations, or `within_side` only."""
    counters = MemoryCounters(args.sysfs_root, args.meminfo)
    for round_index in range(args.warmup_rounds):
        for side in order(round_index):
            run_invocation(binaries[side], workload, side, round_index, args.timeout, counters)
            print(f"warmup round {round_index + 1}/{args.warmup_rounds} {side}", file=sys.stderr)
    invocations: list[Invocation] = []
    for round_index in range(args.rounds):
        for side in order(round_index):
            invocation = run_invocation(binaries[side], workload, side, round_index, args.timeout, counters)
            invocations.append(invocation)
            # A speed comparison against a build with intended output
            # changes still requires each side to repeat its own outputs.
            checked = [inv for inv in invocations if inv.side == side] if within_side else invocations
            mismatch = first_identity_mismatch(checked, reference)
            if mismatch:
                raise RuntimeError(f"outputs differ, aborting: {mismatch}")
            memory, device = invocation.peak_memory_bytes, invocation.peak_device_memory_bytes
            print(
                f"round {round_index + 1}/{args.rounds} {side}: "
                f"median {statistics.median(invocation.request_ms):.1f} ms, load {invocation.load_ms:.1f} ms"
                + (f", peak memory {memory / 2**30:.3f} GiB" if memory is not None else "")
                + (f", device {device / 2**30:.3f} GiB" if device is not None else ""),
                file=sys.stderr,
            )
    return invocations


def reference_summary(found: tuple[Path, dict[str, Any]] | None, invocations: list[Invocation]) -> dict[str, Any]:
    return {
        "path": str(found[0].relative_to(REPOSITORY)) if found else None,
        "checked_invocations": sum(inv.generated_ids is not None for inv in invocations) if found else 0,
    }


def gated_metrics(gated_stages: list[str], args: argparse.Namespace) -> dict[str, tuple[Any, float]]:
    """The gated metrics by name: an invocation's value and the tolerance."""
    metrics = {
        "request": (lambda inv: statistics.median(inv.request_ms), args.tolerance),
        "load": (lambda inv: inv.load_ms, args.load_tolerance),
        "memory": (lambda inv: inv.peak_memory_bytes, args.memory_tolerance),
    }
    for name in gated_stages:
        metrics[name] = (lambda inv, name=name: inv.stages[name], args.tolerance)
    return metrics


def judged_sides(
    sides: dict[str, list[Invocation]], stage_fields: list[str], gated_stages: list[str],
    args: argparse.Namespace, gate,
) -> dict[str, Any]:
    """Each side's summaries and B's verdicts against A. `gate(pick,
    tolerance)` judges one metric, `pick` giving an invocation's value: per
    round when interleaved, against the pooled baseline when recorded."""

    def summaries(pick) -> dict[str, dict[str, float]]:
        return {side: summarize([value for inv in invocations for value in pick(inv)])
                for side, invocations in sides.items()}

    def memory_gate(pick, tolerance: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if all(pick(inv) for invocations in sides.values() for inv in invocations):
            return gate(pick, tolerance), summaries(lambda inv: [float(pick(inv))])
        # Without counters (or without any allocation) there is no verdict,
        # and the gate cannot pass.
        return {"median_ratio": None, "ci95": [None, None], "rounds": 0,
                "tolerance": tolerance, "pass": False}, None

    metrics = gated_metrics(gated_stages, args)
    request = gate(*metrics["request"])
    load = gate(*metrics["load"])
    gated = {name: gate(*metrics[name]) for name in gated_stages}
    memory, memory_bytes = memory_gate(*metrics["memory"])
    device_memory, device_memory_bytes = memory_gate(lambda inv: inv.peak_device_memory_bytes, args.memory_tolerance)
    return {
        "request_ms": summaries(lambda inv: inv.request_ms),
        "load_ms": summaries(lambda inv: [inv.load_ms]),
        "stages_ms": {name: summaries(lambda inv, name=name: [inv.stages[name]]) for name in stage_fields},
        "peak_memory_bytes": memory_bytes,
        "peak_device_memory_bytes": device_memory_bytes,
        # Reported, not judged.
        "device_memory": {key: device_memory[key] for key in ("median_ratio", "ci95", "rounds")},
        "verdict": {
            "request": request,
            "load": load,
            "memory": memory,
            "stages": gated,
            "pass": request["pass"] and load["pass"] and memory["pass"]
            and all(v["pass"] for v in gated.values()),
        },
    }


def compare(args: argparse.Namespace, workload: Workload) -> dict[str, Any]:
    binaries = {"A": args.baseline, "B": args.candidate}
    described = {side: describe_binary(path) for side, path in binaries.items()}
    runtimes = {described[side]["library"] for side in binaries}
    if len(runtimes) != 1 and not args.allow_mixed_runtime:
        raise RuntimeError(
            "baseline and candidate load different HIP runtimes "
            f"({described['A']['library']} vs {described['B']['library']}); "
            "build both against the same ROCm prefix or pass --allow-mixed-runtime"
        )

    found = find_reference(args.audio, args.max_tokens) if args.workload == "asr" else None
    reference = found[1] if found else None
    invocations = measure(binaries, abba_order, workload, args, reference, args.allow_output_mismatch)
    sides = {side: [inv for inv in invocations if inv.side == side] for side in binaries}
    measured = judged_sides(sides, workload.stage_fields, workload.gated_stages, args,
                            lambda pick, tolerance: verdict(paired_ratios(invocations, pick), tolerance))
    return {
        "schema": SCHEMA,
        "workload": {"name": workload.name, "arguments": workload.command},
        "procedure": {"rounds": args.rounds, "warmup_rounds": args.warmup_rounds, "order": "ABBA",
                      "warmup": args.warmup, "repeat": args.repeat},
        "environment": environment(),
        "baseline": described["A"],
        "candidate": described["B"],
        "identity": invocations[0].identity,
        "output_mismatch": first_identity_mismatch(invocations, reference),
        "generated_ids": next((inv.generated_ids for inv in invocations if inv.generated_ids is not None), None),
        "reference": reference_summary(found, invocations),
        **measured,
        "invocations": [inv.__dict__ for inv in invocations],
    }


def workload_digest(workload: Workload) -> str:
    """SHA-256 of what a workload runs and sends, for the servers more than
    the command line (prompts, request bodies, repetitions). The port does
    not change what is measured."""
    fields = {name: value for name, value in asdict(workload).items() if name != "port"}
    text = json.dumps([type(workload).__name__, fields], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def run_key(binary_sha256: str, workload: Workload) -> str:
    """What a run measured: the binary and the workload. Stored baseline runs
    serve a candidate only under one key, so another A binary or other
    workload arguments never reuse them."""
    return hashlib.sha256(f"{binary_sha256} {workload_digest(workload)}".encode()).hexdigest()


def run_report(
    side: str, workload: Workload, binary: dict[str, Any], procedure: dict[str, Any],
    invocations: list[Invocation], found: tuple[Path, dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "side": side,
        "key": run_key(binary["sha256"], workload),
        "workload": {"name": workload.name, "arguments": workload.command, "digest": workload_digest(workload),
                     "stage_fields": workload.stage_fields, "gated_stages": workload.gated_stages},
        "procedure": procedure,
        "environment": environment(),
        "binary": binary,
        "identity": invocations[0].identity,
        "generated_ids": next((inv.generated_ids for inv in invocations if inv.generated_ids is not None), None),
        "reference": reference_summary(found, invocations),
        "invocations": [inv.__dict__ for inv in invocations],
    }


def record(args: argparse.Namespace, workload: Workload) -> dict[str, Any]:
    """One run of one binary: its warmup rounds, then its rounds."""
    side, binary = ("A", args.baseline) if args.baseline is not None else ("B", args.candidate)
    described = describe_binary(binary)
    found = find_reference(args.audio, args.max_tokens) if args.workload == "asr" else None
    cooling = cool_down(args.sysfs_root, args.cool_to, args.cool_timeout)
    invocations = measure({side: binary}, lambda _: (side,), workload, args, found[1] if found else None, True)
    procedure = {"rounds": args.rounds, "warmup_rounds": args.warmup_rounds,
                 "warmup": args.warmup, "repeat": args.repeat}
    report = run_report(side, workload, described, procedure, invocations, found)
    report["cooling"] = cooling
    return report


@dataclass
class Run:
    """A recorded run, its invocations labelled with the run (A1, B2, ...)."""

    path: Path
    label: str
    report: dict[str, Any]
    invocations: list[Invocation]


def load_run(path: Path, side: str, number: int) -> Run:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{path} is not a recorded run ({RUN_SCHEMA})")
    if report["side"] != side:
        raise ValueError(f"{path} is a run of side {report['side']}, given as side {side}")
    label = f"{side}{number}"
    return Run(path, label, report, [Invocation(**{**inv, "side": label}) for inv in report["invocations"]])


def consistency(first: Run, second: Run, metrics: dict[str, tuple[Any, float]]) -> dict[str, Any]:
    """Whether two runs of one side agree: for every gated metric, the
    medians m1, m2 of their invocations satisfy |m1/m2 - 1| <= its
    tolerance. A metric a run did not measure is left out."""
    spreads = {}
    for name, (pick, tolerance) in metrics.items():
        values = [[pick(inv) for inv in run.invocations] for run in (first, second)]
        if all(all(side) for side in values):
            m1, m2 = (statistics.median(side) for side in values)
            spreads[name] = {"spread": abs(m1 / m2 - 1.0), "tolerance": tolerance}
    differ = [name for name, spread in spreads.items() if spread["spread"] > spread["tolerance"]]
    return {"runs": [first.label, second.label], "metrics": spreads, "differ": differ, "consistent": not differ}


def judge(args: argparse.Namespace) -> dict[str, Any]:
    baseline = [load_run(path, "A", number) for number, path in enumerate(args.baseline_run, 1)]
    candidate = [load_run(path, "B", number) for number, path in enumerate(args.candidate_run, 1)]
    workload = baseline[0].report["workload"]
    for run in baseline[1:] + candidate:
        other = run.report["workload"]
        if other != workload:
            raise ValueError(
                f"{run.path} ran another workload than {baseline[0].path}: {shlex.join(other['arguments'])} "
                f"({other['digest'][:12]}) vs {shlex.join(workload['arguments'])} ({workload['digest'][:12]})"
            )
    for name, runs in (("baseline", baseline), ("candidate", candidate)):
        if len({run.report["binary"]["sha256"] for run in runs}) > 1:
            raise ValueError(f"the {name} runs come from different binaries: "
                             + ", ".join(f"{run.path} {run.report['binary']['sha256'][:12]}" for run in runs))
    runtimes = {run.report["binary"]["library"] for run in baseline + candidate}
    if len(runtimes) != 1 and not args.allow_mixed_runtime:
        raise RuntimeError(
            f"the runs load different HIP runtimes ({', '.join(sorted(map(str, runtimes)))}); "
            "build both against the same ROCm prefix or pass --allow-mixed-runtime"
        )

    pooled = [inv for run in baseline for inv in run.invocations]
    candidates = [inv for run in candidate for inv in run.invocations]
    everything = pooled + candidates
    # With --allow-output-mismatch each side must still repeat its own
    # outputs, within and across its runs.
    for group in [pooled, candidates] if args.allow_output_mismatch else [everything]:
        mismatch = first_identity_mismatch(group) if group else None
        if mismatch:
            raise RuntimeError(f"outputs differ: {mismatch}")

    metrics = gated_metrics(workload["gated_stages"], args)
    baseline_consistency = consistency(baseline[0], baseline[1], metrics)
    next_run = "A" if len(baseline) == 2 and not baseline_consistency["consistent"] else None
    judged = []
    for run in candidate:
        def gate(pick, tolerance: float, run: Run = run) -> dict[str, Any]:
            return unpaired_verdict([pick(inv) for inv in run.invocations], [pick(inv) for inv in pooled], tolerance)

        judged.append(judged_sides({"A": pooled, "B": run.invocations}, workload["stage_fields"],
                                   workload["gated_stages"], args, gate))
    candidate_consistency = consistency(candidate[0], candidate[1], metrics) if candidate else None
    if next_run is None and len(candidate) == 2 and (
        not candidate_consistency["consistent"] or judged[0]["verdict"]["pass"] != judged[1]["verdict"]["pass"]
    ):
        next_run = "B"
    passed = sum(run["verdict"]["pass"] for run in judged)

    def describe(run: Run) -> dict[str, Any]:
        return {"label": run.label, "path": str(run.path),
                **{key: run.report[key] for key in ("procedure", "environment", "reference")}}

    return {
        "schema": JUDGE_SCHEMA,
        "workload": workload,
        "baseline_key": baseline[0].report["key"],
        "baseline": {
            "binary": baseline[0].report["binary"],
            "runs": [describe(run) for run in baseline],
            "consistency": baseline_consistency,
        },
        "candidate": {
            "binary": candidate[0].report["binary"] if candidate else None,
            "runs": [{**describe(run), **measured} for run, measured in zip(candidate, judged)],
            "consistency": candidate_consistency,
        },
        "identity": everything[0].identity,
        "output_mismatch": first_identity_mismatch(everything),
        "generated_ids": next((inv.generated_ids for inv in everything if inv.generated_ids is not None), None),
        # Two runs that agree decide alone, of three the majority.
        "verdict": None if not candidate else {
            "passed": passed, "runs": len(judged), "pass": None if next_run else 2 * passed > len(judged),
        },
        "next_run": next_run,
    }


def settings(env: dict[str, Any]) -> str:
    return (
        f"GPU DPM {','.join(env['gpu_dpm_performance_level']) or '?'}, governor "
        f"{env['cpu_governor']}, EPP {env['cpu_energy_performance_preference']}"
    )


def print_metrics(measured: dict[str, Any]) -> None:
    """The request, load, memory and stage lines of B against A."""

    def line(label: str, key: str, gate: str, unit: str = "ms", scale: float = 1.0) -> None:
        v = measured["verdict"][gate]
        if measured[key] is None:
            print(f"{label:<9} not measured  FAIL")
            return
        a, b = measured[key]["A"]["median"] / scale, measured[key]["B"]["median"] / scale
        low, high = v["ci95"]
        status = "PASS" if v["pass"] else "FAIL"
        print(
            f"{label:<9} A {a:9.2f} {unit}  B {b:9.2f} {unit}  B/A {v['median_ratio']:.4f} "
            f"[{low:.4f}, {high:.4f}]  ≤ {1 + v['tolerance']:.2f}  {status}"
        )

    line("request", "request_ms", "request")
    line("load", "load_ms", "load")
    line("memory", "peak_memory_bytes", "memory", "GiB", 2**30)
    device = measured["device_memory"]
    if measured["peak_device_memory_bytes"] is not None:
        a, b = (measured["peak_device_memory_bytes"][side]["median"] / 2**30 for side in "AB")
        low, high = device["ci95"]
        print(f"{'device':<9} A {a:9.2f} GiB  B {b:9.2f} GiB  B/A {device['median_ratio']:.4f} "
              f"[{low:.4f}, {high:.4f}]  not gated")
    for name, sides in measured["stages_ms"].items():
        text = f"{name:<18} A {sides['A']['median']:9.2f} ms  B {sides['B']['median']:9.2f} ms"
        v = measured["verdict"]["stages"].get(name)
        if v is not None:
            low, high = v["ci95"]
            text += (
                f"  B/A {v['median_ratio']:.4f} [{low:.4f}, {high:.4f}]  ≤ {1 + v['tolerance']:.2f}  "
                + ("PASS" if v["pass"] else "FAIL")
            )
        print(text)


def print_summary(report: dict[str, Any]) -> None:
    print(f"workload  {report['workload']['name']}, {report['procedure']['rounds']} rounds ABBA")
    print(f"runtime   A {report['baseline']['library']}  B {report['candidate']['library']}")
    print(f"settings  {settings(report['environment'])}")
    print_metrics(report)
    if report["output_mismatch"]:
        print(f"outputs   differ between A and B (allowed): {report['output_mismatch']}")
    else:
        print(f"outputs   identical in all {len(report['invocations'])} invocations")
    reference = report["reference"]
    if reference["path"]:
        print(f"reference {reference['path']}: token IDs of {reference['checked_invocations']} invocations match")
    print("verdict   " + ("PASS" if report["verdict"]["pass"] else "FAIL"))


def print_run(report: dict[str, Any], path: Path) -> None:
    invocations, binary = report["invocations"], report["binary"]
    print(f"recorded  {path}: {report['workload']['name']}, side {report['side']}, "
          f"{len(invocations)} rounds, key {report['key'][:12]}")
    print(f"binary    {binary['path']} ({binary['sha256'][:12]}), runtime {binary['library']}")
    if (cooling := report.get("cooling")) and cooling["start_temperature_c"] is not None:
        print(f"start     GPU at {cooling['start_temperature_c']:.1f} °C after {cooling['waited_s']:.0f} s")
    request = statistics.median(statistics.median(inv["request_ms"]) for inv in invocations)
    load = statistics.median(inv["load_ms"] for inv in invocations)
    print(f"median    request {request:.2f} ms, load {load:.2f} ms")


def print_consistency(name: str, side: dict[str, Any]) -> None:
    check, runs = side["consistency"], side["runs"]
    first, second = check["runs"]
    text = f"{first} and {second} " + ("consistent" if check["consistent"] else "differ: " + ", ".join(
        f"{metric} {check['metrics'][metric]['spread']:.2%} > {check['metrics'][metric]['tolerance']:.2%}"
        for metric in check["differ"]
    ))
    verdicts = [run["verdict"]["pass"] for run in runs[:2] if "verdict" in run]
    if len(verdicts) == 2 and verdicts[0] != verdicts[1]:
        text += f"; only {first if verdicts[0] else second} passes"
    if len(runs) == 3:
        text += f"; third run {runs[2]['label']}"
    print(f"{name:<9} {text}")


def print_judgement(report: dict[str, Any]) -> None:
    baseline, candidate = report["baseline"], report["candidate"]
    runs = baseline["runs"] + candidate["runs"]
    print(f"workload  {report['workload']['name']}, "
          + ", ".join(f"{run['label']} {run['procedure']['rounds']} rounds" for run in runs))
    print(f"runtime   A {baseline['binary']['library']}"
          + (f"  B {candidate['binary']['library']}" if candidate["binary"] else ""))
    texts = {run["label"]: settings(run["environment"]) for run in runs}
    if len(set(texts.values())) == 1:
        print(f"settings  {texts[runs[0]['label']]}")
    else:
        for label, text in texts.items():
            print(f"settings  {label}: {text}")
    print_consistency("baseline", baseline)
    pooled = "+".join(run["label"] for run in baseline["runs"])
    for run in candidate["runs"]:
        print(f"{run['label']:<9} against {pooled}")
        print_metrics(run)
        print(f"{run['label']:<9} " + ("PASS" if run["verdict"]["pass"] else "FAIL"))
    if candidate["runs"]:
        print_consistency("candidate", candidate)
    if report["output_mismatch"]:
        print(f"outputs   differ between A and B (allowed): {report['output_mismatch']}")
    else:
        print(f"outputs   identical in all {sum(run['procedure']['rounds'] for run in runs)} invocations")
    references = sorted({run["reference"]["path"] for run in runs if run["reference"]["path"]})
    if references:
        checked = sum(run["reference"]["checked_invocations"] for run in runs)
        print(f"reference {', '.join(references)}: token IDs of {checked} invocations match")
    verdict, next_run = report["verdict"], report["next_run"]
    if next_run:
        count = len((baseline if next_run == "A" else candidate)["runs"])
        print(f"verdict   none yet: record a third run, {next_run}{count + 1}")
    elif verdict is None:
        print(f"baseline  complete with {len(baseline['runs'])} runs")
    else:
        print(f"verdict   {'PASS' if verdict['pass'] else 'FAIL'} "
              f"({verdict['passed']} of {verdict['runs']} runs pass)")


def add_procedure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup-rounds", type=int, default=1,
                        help="rounds (ABBA when interleaved) run before the measured ones and discarded")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"), help=argparse.SUPPRESS)
    parser.add_argument("--meminfo", type=Path, default=Path("/proc/meminfo"), help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per invocation")


def add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    # The gates of docs/RUST.md: time within 0.5%, load within 5%, memory within 1%.
    parser.add_argument("--tolerance", type=float, default=0.005, help="allowed request slowdown")
    parser.add_argument("--load-tolerance", type=float, default=0.05, help="allowed load-time slowdown")
    parser.add_argument("--memory-tolerance", type=float, default=0.01, help="allowed peak-memory increase")
    parser.add_argument("--allow-mixed-runtime", action="store_true")
    parser.add_argument(
        "--allow-output-mismatch",
        action="store_true",
        help="compare speed only: A and B may differ in outputs (each side must repeat its own)",
    )


def check_procedure(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.rounds < 2 or args.repeat < 1 or args.warmup < 0:
        parser.error("need --rounds >= 2, --repeat >= 1 and --warmup >= 0")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        epilog="Recorded runs: `ab.py record --help`, `ab.py judge --help`.",
    )
    parser.add_argument("--baseline", type=Path, required=True, help="A: the C++ gufo binary")
    parser.add_argument("--candidate", type=Path, required=True, help="B: the binary under test")
    add_procedure_arguments(parser)
    add_gate_arguments(parser)
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    add_workloads(parser)
    return check_procedure(parser, parser.parse_args(argv))


def parse_record_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ab.py record", description="record one run of one binary for `ab.py judge`")
    binary = parser.add_mutually_exclusive_group(required=True)
    binary.add_argument("--baseline", type=Path, help="record side A: the C++ gufo binary")
    binary.add_argument("--candidate", type=Path, help="record side B: the binary under test")
    parser.add_argument("--output", type=Path, required=True, help="write the run here")
    parser.add_argument("--cool-to", type=float, help="first wait until the GPU is at most this many °C")
    parser.add_argument("--cool-timeout", type=float, default=300.0, help="seconds to wait at most")
    add_procedure_arguments(parser)
    add_workloads(parser)
    return check_procedure(parser, parser.parse_args(argv))


def parse_judge_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ab.py judge",
        description="judge recorded runs: each candidate run against the pooled baseline runs; "
        "exit status 3 asks for a third run (`next_run`)",
    )
    parser.add_argument("--baseline-run", type=Path, action="append", required=True,
                        help="a run of A: two, or three when the first two differ")
    parser.add_argument("--candidate-run", type=Path, action="append", default=[],
                        help="a run of B: two, or three when the first two differ or only one passes "
                        "(none: check the baseline alone)")
    add_gate_arguments(parser)
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)
    if len(args.baseline_run) not in (2, 3) or len(args.candidate_run) not in (0, 2, 3):
        parser.error("need two or three --baseline-run and no, two or three --candidate-run")
    return args


def add_workloads(parser: argparse.ArgumentParser) -> None:
    workloads = parser.add_subparsers(dest="workload", required=True)
    asr = workloads.add_parser("asr", help="gufo transcribe (Qwen3-ASR)")
    asr.add_argument("--model", type=Path, required=True)
    asr.add_argument("--audio", type=Path, required=True)
    asr.add_argument("--max-tokens", type=int, default=256)
    text = workloads.add_parser(
        "text", help="gufo bench (Qwen3.8 27B, Flash-Next or DeepSeek V4 Flash, optionally DFlash2, MTP or DSpark)")
    text.add_argument("--model", type=Path, required=True)
    drafts = text.add_mutually_exclusive_group()
    drafts.add_argument("--dflash-model", type=Path, help="run --speculative dflash2 with this draft")
    drafts.add_argument("--mtp-model", type=Path, help="run --speculative mtp with this MTP head (Flash-Next sidecar or dense Qwen)")
    drafts.add_argument("--dspark-model", type=Path, help="run --speculative dspark with this DeepSeek support model")
    text.add_argument("--prompt-tokens", type=int, default=512, help="pp row (0: tg only)")
    text.add_argument("--gen-tokens", type=int, default=128)
    text.add_argument("--depth", type=int, default=0)
    serve = workloads.add_parser("serve", help="gufo serve llm under gufo-serving-bench load")
    serve.add_argument("--model", type=Path, required=True)
    serve.add_argument("--name", default="", help="suffix for the workload name, e.g. dflash2")
    serve.add_argument("--server-args", default="", help="further `gufo serve llm` arguments (shell words)")
    serve.add_argument("--concurrency", default="1,2,4", help="levels; the largest sets --sessions")
    serve.add_argument("--context", type=int, default=8192)
    serve.add_argument("--max-tokens", type=int, default=128)
    serve.add_argument("--prompt-repeat", type=int, default=8)
    serve.add_argument("--port", type=int, default=18531)
    tts = workloads.add_parser("tts", help="gufo serve tts: buffered WAV and streamed PCM requests")
    tts.add_argument("--model", type=Path, required=True)
    tts.add_argument("--name", default="", help="suffix for the workload name, e.g. base")
    tts.add_argument("--server-args", default="", help="further `gufo serve tts` arguments (shell words)")
    tts.add_argument("--text", default="When the first light reaches the harbor, speak gently, with quiet "
                     "confidence and a warm, resonant tone.")
    tts.add_argument("--voice", default="vivian", help="empty: no voice field (VoiceDesign)")
    tts.add_argument("--seed", type=int, default=42)
    tts.add_argument("--max-tokens", type=int, default=64, help="max_new_tokens (codec frames)")
    tts.add_argument("--request", default="{}", help="further request fields as a JSON object")
    tts.add_argument("--port", type=int, default=18541)
    image = workloads.add_parser("image", help="gufo serve image: seeded generations or edits")
    image.add_argument("--model", type=Path, required=True)
    image.add_argument("--name", default="", help="suffix for the workload name, e.g. edit")
    image.add_argument("--server-args", default="", help="further `gufo serve image` arguments (shell words)")
    image.add_argument("--prompt", default="A red cube on a white table, studio light.")
    image.add_argument("--size", default="1024x1024")
    image.add_argument("--steps", type=int, default=8)
    image.add_argument("--seed", type=int, default=42)
    image.add_argument("--edit", type=Path, help="reference image: post to /v1/images/edits")
    image.add_argument("--port", type=int, default=18542)
    embedding = workloads.add_parser("embedding", help="gufo serve embedding: batched and single inputs")
    embedding.add_argument("--model", type=Path, required=True)
    embedding.add_argument("--server-args", default="", help="further `gufo serve embedding` arguments")
    embedding.add_argument("--batch", type=int, default=32, help="inputs per batch request")
    embedding.add_argument("--tokens", type=int, default=512, help="approximate tokens per input")
    embedding.add_argument("--port", type=int, default=18543)


def write_report(path: Path, report: dict[str, Any]) -> None:
    # Whole or not at all: a driver takes an existing run file as recorded.
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Exit status 0 pass (a run recorded, a baseline complete), 1 fail, 2
    error, 3 (`judge`) a third run needed."""
    argv = sys.argv[1:] if argv is None else argv
    mode = argv[0] if argv[:1] in (["record"], ["judge"]) else "compare"
    if mode == "judge":
        args = parse_judge_args(argv[1:])
        try:
            report = judge(args)
        except (RuntimeError, ValueError, OSError) as error:
            print(f"[ab] {error}", file=sys.stderr)
            return 2
        print_judgement(report)
        if args.output is not None:
            write_report(args.output, report)
        if report["next_run"]:
            return 3
        return 0 if report["verdict"] is None or report["verdict"]["pass"] else 1
    args = parse_record_args(argv[1:]) if mode == "record" else parse_args(argv)
    workload = {
        "asr": asr_workload, "text": text_workload, "serve": serve_workload, "tts": tts_workload,
        "image": image_workload,
        "embedding": embedding_workload,
    }[args.workload](args)
    try:
        report = record(args, workload) if mode == "record" else compare(args, workload)
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"[ab] {error}", file=sys.stderr)
        return 2
    if mode == "record":
        write_report(args.output, report)
        print_run(report, args.output)
        return 0
    print_summary(report)
    if args.output is not None:
        write_report(args.output, report)
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
