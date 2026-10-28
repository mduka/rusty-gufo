#!/usr/bin/env python3
"""Tests for tools/bench/ab.py with fake gufo binaries (no GPU needed)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools" / "bench"))
import ab  # noqa: E402

# Shared by the fakes: device memory in use, which on the APU is host
# memory as well, plus FAKE_HOST_MEMORY bytes in host memory only.
HOLD = """
def hold(device):
    with open(os.environ["FAKE_GTT"], "w") as f:
        f.write(str(device))
    host = device + int(os.environ.get("FAKE_HOST_MEMORY", "0")) if device else 0
    meminfo = os.environ["FAKE_MEMINFO"]
    with open(meminfo + ".tmp", "w") as f:
        f.write(f"MemTotal: 100000000 kB\\nMemFree: 0 kB\\nMemAvailable: {100000000 - host // 1024} kB\\n")
    os.replace(meminfo + ".tmp", meminfo)
"""

FAKE_GUFO = """#!/usr/bin/env python3
import json, os, sys, time
""" + HOLD + """log = os.environ["FAKE_LOG"]
with open(log, "a") as f:
    f.write(os.environ["FAKE_SIDE"] + "\\n")
scale = float(os.environ.get("FAKE_SCALE", "1.0"))
# Invocations after the first FAKE_SLOW_AFTER of this binary are 5% slower.
with open(log) as f:
    if f.read().split().count(os.environ["FAKE_SIDE"]) > int(os.environ.get("FAKE_SLOW_AFTER", "1000000")):
        scale *= 1.05
repeat = int(sys.argv[sys.argv.index("--repeat") + 1])
hold(int(os.environ.get("FAKE_MEMORY", "1000000000")))
time.sleep(0.05)
hold(0)
runs = [{"total_ms": 1000.0 * scale + i, "audio_encoder_ms": 30.0, "text_decoder_ms": 900.0 * scale}
        for i in range(repeat)]
report = {
    "language": "English", "text": os.environ.get("FAKE_TEXT", "hello"), "audio_samples": 16000,
    "audio_tokens": 196, "prompt_tokens": 211, "generated_tokens": 3,
    "timings": {"load_ms": 300.0}, "runs": runs,
}
if "FAKE_IDS" in os.environ:
    report["generated_ids"] = json.loads(os.environ["FAKE_IDS"])
print(json.dumps(report))
"""


FAKE_BENCH = """#!/usr/bin/env python3
import os, sys, time
""" + HOLD + """log = os.environ["FAKE_LOG"]
with open(log + ".oom", "a") as f, open("/proc/self/oom_score_adj") as adj:
    f.write(adj.read())
with open(log, "a") as f:
    f.write(os.environ["FAKE_SIDE"] + "\\n")
scale = float(os.environ.get("FAKE_SCALE", "1.0"))
hold(int(os.environ.get("FAKE_MEMORY", "1000000000")))
time.sleep(0.05)
hold(0)
print("[Model Load]: 0.65 s")
flash = "--mtp-model" in sys.argv
deepseek = "--dspark-model" in sys.argv
if deepseek:
    print("| model | size | backend | test | t/s |")
    print("| --- | --- | --- | --- | --- |")
    print(f"| DeepSeek V4 Flash | 80.80 GiB | ROCm (HIP) |           pp512 |  {900.0 / scale:.2f} ± 1.00 |")
    print(f"| DeepSeek V4 Flash | 80.80 GiB | ROCm (HIP) |           tg128 |  {25.0 / scale:.2f} ± 0.01 |")
elif flash:
    print("| model | size | backend | test | t/s |")
    print("| --- | --- | --- | --- | --- |")
    print(f"| Qwen3.8 Flash Next | 103.69 GiB | ROCm (HIP) |           pp512 |  {1000.0 / scale:.2f} ± 1.00 |")
    print(f"| Qwen3.8 Flash Next | 103.69 GiB | ROCm (HIP) |           tg128 |  {30.0 / scale:.2f} ± 0.01 |")
else:
    print("| model | size | params | backend | test | t/s |")
    print("| --- | --- | --- | --- | --- | --- |")
    print(f"| Qwen | 16.35 GiB | 27.32 B | ROCm (HIP) |           pp512 |  {300.0 / scale:.2f} ± 1.00 |")
    print(f"| Qwen | 16.35 GiB | 27.32 B | ROCm (HIP) |           tg128 |  {11.0 / scale:.2f} ± 0.01 |")
for _ in range(int(sys.argv[sys.argv.index("-r") + 1])):
    if deepseek:
        print("DeepSeek tg C=1 depth=0 request=0 accepted=60 drafted=90 steps=40 skipped=0 output_sha256="
              + os.environ.get("FAKE_TRACE", "ab12"), file=sys.stderr)
    elif flash:
        print("Qwen3.8-Flash-Next tg depth=0 cycles=40 drafted=120 accepted=90"
              " lookup_proposed=6 lookup_accepted=4 output_sha256="
              + os.environ.get("FAKE_TRACE", "ab12") + " text=The quick", file=sys.stderr)
    else:
        print("[QwenBenchTrace]: depth=0 count=128 sha256=" + os.environ.get("FAKE_TRACE", "ab12"), file=sys.stderr)
if "FAKE_MISMATCHES" in os.environ:
    print("graph mismatches: 1", file=sys.stderr)
"""


FAKE_SERVER = """#!/usr/bin/env python3
import hashlib, json, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
""" + HOLD + """log = os.environ["FAKE_LOG"]
with open(log, "a") as f:
    f.write(os.environ["FAKE_SIDE"] + "\\n")
scale = float(os.environ.get("FAKE_SCALE", "1.0"))
port = int(sys.argv[sys.argv.index("--port") + 1])
time.sleep(0.05)
hold(int(os.environ.get("FAKE_MEMORY", "1000000000")))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def reply(self, body, kind="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path in ("/health", "/ready"):
            self.reply(b"{}")
        else:
            self.reply(json.dumps({"data": [{"id": "fake"}]}).encode())
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path in ("/v1/images/generations", "/v1/images/edits"):
            time.sleep(0.02 * scale)
            import base64, struct, zlib
            def chunk(kind, data):
                return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
            # A 2x1 RGBA image whose pixels depend on FAKE_TEXT; the filter
            # bytes vary per side, so only the pixels agree.
            pixel = hashlib.sha256(os.environ.get("FAKE_TEXT", "").encode()).digest()[:8]
            filter_kind = 0 if os.environ["FAKE_SIDE"] == "A" else 1
            row = pixel if filter_kind == 0 else pixel[:4] + bytes((pixel[4 + i] - pixel[i]) & 255 for i in range(4))
            png = (b"\\x89PNG\\r\\n\\x1a\\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 6, 0, 0, 0))
                   + chunk(b"IDAT", zlib.compress(bytes([filter_kind]) + row)) + chunk(b"IEND", b""))
            print(f"[image] request=x event=generated mode=generate seed=1 size=2x1 references=0 steps=2 "
                  f"prompt_ms={10 * scale} denoise_ms={50 * scale} vae_ms={5 * scale} total_ms={65 * scale} "
                  "resident_weight_mib=1", flush=True)
            self.reply(json.dumps({"created": 1, "data": [{"b64_json": base64.b64encode(png).decode()}]}).encode())
            return
        request = json.loads(raw)
        if self.path == "/v1/audio/speech":
            time.sleep(0.02 * scale)
            audio = (os.environ.get("FAKE_TEXT", "") + request["input"] + request["response_format"]).encode()
            self.reply(audio, "audio/" + request["response_format"])
            return
        prompt = request["messages"][0]["content"]
        time.sleep(0.02 * scale)
        text = os.environ.get("FAKE_TEXT", "") + hashlib.sha256(prompt.encode()).hexdigest()[:8]
        usage = {"prompt_tokens": 40, "completion_tokens": 8, "prompt_tokens_details": {"cached_tokens": 0},
                 "gufo": {"prefill_tokens": 40, "queue_ms": 0.1, "prefill_ms": 5.0 * scale,
                          "decode_ms": 15.0 * scale, "ttft_ms": 5.0 * scale, "mean_inter_token_ms": 2.0,
                          "max_inter_token_ms": 3.0, "requested_logical_concurrency": 1,
                          "physical_execution_width": 1, "execution_plan": "single"}}
        events = [{"choices": [{"delta": {"content": text}}]}, {"choices": [], "usage": usage}]
        body = "".join(f"data: {json.dumps(e)}\\n\\n" for e in events) + "data: [DONE]\\n\\n"
        self.reply(body.encode(), "text/event-stream")

server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
def stop(*_):
    hold(0)
    os._exit(0)
signal.signal(signal.SIGINT, stop)
server.serve_forever()
"""

# The fake servers' requests are real sleeps of milliseconds and their load
# time is Python's start, so on a busy machine (a hosted runner) both vary by
# tens of percent. Runs of equal sides check measuring, comparing and
# recording, with tolerances no such noise reaches; a slower candidate sleeps
# three times as long.
EQUAL_SIDES = ("--tolerance", "10", "--load-tolerance", "10")
# Millisecond requests: scheduling noise exceeds the default 0.5 %.
SLOWER_SIDES = ("--tolerance", "0.2")
SLOWER = "3"


class AbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.log = self.dir / "order.log"
        device = self.dir / "sys/class/drm/card0/device"
        device.mkdir(parents=True)
        (device / "mem_info_vram_used").write_text("0")
        self.gtt = device / "mem_info_gtt_used"
        self.gtt.write_text("0")
        self.meminfo = self.dir / "meminfo"
        self.meminfo.write_text("MemTotal: 100000000 kB\nMemFree: 0 kB\nMemAvailable: 100000000 kB\n")
        self.counters = ["--sysfs-root", str(self.dir / "sys"), "--meminfo", str(self.meminfo)]
        self.audio = self.dir / "a.wav"
        self.audio.write_bytes(b"RIFF fake")

    def fake(self, name: str, **env: str) -> Path:
        variables = {"FAKE_SIDE": name, "FAKE_LOG": str(self.log), "FAKE_GTT": str(self.gtt),
                     "FAKE_MEMINFO": str(self.meminfo), **env}
        exports = "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in variables.items())
        path = self.dir / name
        path.write_text(FAKE_GUFO.replace("import json, os, sys, time\n", "import json, os, sys, time\n" + exports))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def run_ab(self, baseline: Path, candidate: Path, *extra: str) -> tuple[int, dict | None]:
        output = self.dir / "report.json"
        output.unlink(missing_ok=True)
        argv = ["--baseline", str(baseline), "--candidate", str(candidate), "--rounds", "4", "--repeat", "3",
                "--output", str(output), *self.counters, *extra,
                "asr", "--model", "m", "--audio", str(self.audio)]
        code = ab.main(argv)
        return code, json.loads(output.read_text()) if output.exists() else None

    def fake_bench(self, name: str, **env: str) -> Path:
        variables = {"FAKE_SIDE": name, "FAKE_LOG": str(self.log), "FAKE_GTT": str(self.gtt),
                     "FAKE_MEMINFO": str(self.meminfo), **env}
        exports = "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in variables.items())
        path = self.dir / f"bench-{name}"
        path.write_text(FAKE_BENCH.replace("import os, sys, time\n", "import os, sys, time\n" + exports))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def run_text(self, baseline: Path, candidate: Path, *extra: str,
                 model: str = "m.gguf") -> tuple[int, dict | None]:
        output = self.dir / "text.json"
        output.unlink(missing_ok=True)
        argv = ["--baseline", str(baseline), "--candidate", str(candidate), "--rounds", "4", "--repeat", "2",
                "--output", str(output), *self.counters,
                "text", "--model", model, "--prompt-tokens", "512", "--gen-tokens", "128", *extra]
        code = ab.main(argv)
        return code, json.loads(output.read_text()) if output.exists() else None

    def fake_server(self, name: str, **env: str) -> Path:
        variables = {"FAKE_SIDE": name, "FAKE_LOG": str(self.log), "FAKE_GTT": str(self.gtt),
                     "FAKE_MEMINFO": str(self.meminfo), **env}
        exports = "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in variables.items())
        path = self.dir / f"server-{name}"
        path.write_text(FAKE_SERVER.replace("import hashlib, json, os, signal, sys, time\n",
                                            "import hashlib, json, os, signal, sys, time\n" + exports))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def run_serve(self, baseline: Path, candidate: Path,
                  timing: tuple[str, ...] = SLOWER_SIDES) -> tuple[int, dict | None]:
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        output = self.dir / "serve.json"
        output.unlink(missing_ok=True)
        argv = ["--baseline", str(baseline), "--candidate", str(candidate), "--rounds", "4", "--repeat", "2",
                "--output", str(output), *self.counters, *timing,
                "serve", "--model", "m.gguf", "--concurrency", "1,2", "--port", str(port),
                "--server-args", "--speculative dflash2"]
        code = ab.main(argv)
        return code, json.loads(output.read_text()) if output.exists() else None

    def test_serve_judges_latency_and_throughput_per_level(self) -> None:
        code, report = self.run_serve(self.fake_server("A"), self.fake_server("B"), EQUAL_SIDES)
        self.assertEqual(code, 0)
        self.assertEqual(self.log.read_text().split(), ["A", "B"] + ["A", "B", "B", "A", "A", "B", "B", "A"])
        self.assertEqual(report["workload"]["arguments"][:4], ["serve", "llm", "--model", "m.gguf"])
        self.assertIn("--speculative", report["workload"]["arguments"])
        self.assertEqual(sorted(report["verdict"]["stages"]),
                         ["c1_ms_per_output_token", "c2_ms_per_output_token", "client_ttft_ms",
                          "decode_ms_per_token"])
        self.assertEqual(len(report["identity"]["completions"]), 2)
        self.assertAlmostEqual(report["peak_memory_bytes"]["A"]["median"], 1e9, delta=1024)
        code, report = self.run_serve(self.fake_server("A"), self.fake_server("B", FAKE_SCALE=SLOWER))
        self.assertEqual(code, 1)
        self.assertFalse(report["verdict"]["stages"]["c2_ms_per_output_token"]["pass"])

    def run_tts(self, baseline: Path, candidate: Path,
                timing: tuple[str, ...] = SLOWER_SIDES) -> tuple[int, dict | None]:
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        output = self.dir / "tts.json"
        output.unlink(missing_ok=True)
        argv = ["--baseline", str(baseline), "--candidate", str(candidate), "--rounds", "4", "--repeat", "2",
                "--output", str(output), *self.counters, *timing,
                "tts", "--model", "voice", "--port", str(port), "--voice", "",
                "--request", '{"instructions": "calm"}']
        code = ab.main(argv)
        return code, json.loads(output.read_text()) if output.exists() else None

    def test_tts_judges_requests_and_first_audio(self) -> None:
        code, report = self.run_tts(self.fake_server("A"), self.fake_server("B"), EQUAL_SIDES)
        self.assertEqual(code, 0)
        self.assertEqual(report["workload"]["arguments"][:4], ["serve", "tts", "--model", "voice"])
        self.assertEqual(sorted(report["verdict"]["stages"]), ["first_audio_ms", "streamed_ms"])
        self.assertEqual(sorted(report["identity"]), ["pcm_sha256", "wav_sha256"])
        code, report = self.run_tts(self.fake_server("A"), self.fake_server("B", FAKE_SCALE=SLOWER))
        self.assertEqual(code, 1)
        self.assertFalse(report["verdict"]["request"]["pass"])

    def test_tts_aborts_on_other_audio(self) -> None:
        code, report = self.run_tts(self.fake_server("A"), self.fake_server("B", FAKE_TEXT="x"))
        self.assertEqual((code, report), (2, None))

    def run_image(self, baseline: Path, candidate: Path, *extra: str,
                  timing: tuple[str, ...] = SLOWER_SIDES) -> tuple[int, dict | None]:
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        output = self.dir / "image.json"
        output.unlink(missing_ok=True)
        argv = ["--baseline", str(baseline), "--candidate", str(candidate), "--rounds", "4", "--repeat", "2",
                "--output", str(output), *self.counters, *timing,
                "image", "--model", "qwen-image", "--port", str(port), "--size", "256x256", *extra]
        code = ab.main(argv)
        return code, json.loads(output.read_text()) if output.exists() else None

    def test_image_judges_requests_and_server_stages(self) -> None:
        code, report = self.run_image(self.fake_server("A"), self.fake_server("B"), timing=EQUAL_SIDES)
        self.assertEqual(code, 0)
        self.assertEqual(report["workload"]["arguments"][:4], ["serve", "image", "--model", "qwen-image"])
        self.assertEqual(sorted(report["verdict"]["stages"]), ["denoise_ms", "prompt_ms", "vae_ms"])
        self.assertEqual(sorted(report["identity"]), ["rgba_sha256"])
        code, report = self.run_image(self.fake_server("A"), self.fake_server("B", FAKE_SCALE=SLOWER))
        self.assertEqual(code, 1)
        self.assertFalse(report["verdict"]["stages"]["denoise_ms"]["pass"])

    def test_image_edits_post_multipart_and_abort_on_other_pixels(self) -> None:
        reference = self.dir / "reference.png"
        reference.write_bytes(b"fake")
        code, _ = self.run_image(self.fake_server("A"), self.fake_server("B"), "--edit", str(reference),
                                 timing=EQUAL_SIDES)
        self.assertEqual(code, 0)
        code, report = self.run_image(self.fake_server("A"), self.fake_server("B", FAKE_TEXT="x"))
        self.assertEqual((code, report), (2, None))

    def test_png_pixels_survive_every_filter(self) -> None:
        import struct
        import zlib

        rows = [bytes(range(i, i + 8)) for i in (3, 90, 200)]

        def encode(kinds: list[int]) -> bytes:
            raw = b""
            previous = bytes(8)
            for kind, row in zip(kinds, rows):
                out = bytearray()
                for i, value in enumerate(row):
                    left = row[i - 4] if i >= 4 else 0
                    up, corner = previous[i], previous[i - 4] if i >= 4 else 0
                    p = left + up - corner
                    pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                    predictor = [0, left, up, (left + up) // 2,
                                 left if pa <= pb and pa <= pc else up if pb <= pc else corner][kind]
                    out.append((value - predictor) & 255)
                raw += bytes([kind]) + bytes(out)
                previous = row
            def chunk(name: bytes, data: bytes) -> bytes:
                return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
            return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 3, 8, 6, 0, 0, 0))
                    + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

        for kinds in ([0, 1, 2], [3, 4, 4], [4, 3, 1]):
            self.assertEqual(ab.png_rgba(encode(kinds)), b"".join(rows), kinds)

    def test_serve_aborts_on_other_completions(self) -> None:
        code, report = self.run_serve(self.fake_server("A"), self.fake_server("B", FAKE_TEXT="x"))
        self.assertEqual((code, report), (2, None))

    def test_text_bench_judges_generation_speed_and_traces(self) -> None:
        code, report = self.run_text(self.fake_bench("A"), self.fake_bench("B"))
        self.assertEqual(code, 0)
        self.assertAlmostEqual(report["request_ms"]["A"]["median"], 1000.0 / 11.0, places=3)
        self.assertEqual(report["identity"]["traces"], ["depth=0 count=128 sha256=ab12"] * 2)
        self.assertEqual(report["workload"]["arguments"][:5], ["bench", "--model", "m.gguf", "-p", "512"])
        self.assertIn("pp_ms_per_token", report["stages_ms"])
        self.assertEqual(Path(str(self.log) + ".oom").read_text().split(), ["1000"] * 10)  # with the warmup round
        code, report = self.run_text(self.fake_bench("A"), self.fake_bench("B", FAKE_SCALE="1.05"))
        self.assertEqual(code, 1)
        self.assertFalse(report["verdict"]["request"]["pass"])

    def test_text_bench_reads_the_flash_next_table_and_traces(self) -> None:
        extra = ("--mtp-model", "mtp.gguf")
        code, report = self.run_text(self.fake_bench("A"), self.fake_bench("B"), *extra)
        self.assertEqual(code, 0)
        self.assertEqual(report["workload"]["name"], "qwen3.8-bench-mtp")
        self.assertAlmostEqual(report["request_ms"]["A"]["median"], 1000.0 / 30.0, places=3)
        self.assertEqual(report["identity"]["traces"],
                         ["depth=0 cycles=40 drafted=120 accepted=90 lookup_proposed=6 lookup_accepted=4"
                          " output_sha256=ab12"] * 2)
        self.assertIn("--speculative", report["workload"]["arguments"])
        code, report = self.run_text(self.fake_bench("A"), self.fake_bench("B", FAKE_TRACE="cd34"), *extra)
        self.assertEqual((code, report), (2, None))

    def test_text_bench_reads_the_deepseek_table_and_traces(self) -> None:
        model = self.dir / "ds4.gguf"
        key, value = b"general.architecture", b"deepseek4"
        model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", len(key)) + key
                          + struct.pack("<IQ", 8, len(value)) + value)
        extra = ("--dspark-model", "support.gguf")
        code, report = self.run_text(self.fake_bench("A"), self.fake_bench("B"), *extra, model=str(model))
        self.assertEqual(code, 0)
        self.assertEqual(report["workload"]["name"], "deepseek-v4-flash-bench-dspark")
        self.assertAlmostEqual(report["request_ms"]["A"]["median"], 1000.0 / 25.0, places=3)
        self.assertEqual(report["identity"]["traces"],
                         ["C=1 depth=0 request=0 accepted=60 drafted=90 steps=40 skipped=0 output_sha256=ab12"] * 2)
        code, report = self.run_text(self.fake_bench("A"), self.fake_bench("B", FAKE_TRACE="cd34"), *extra,
                                     model=str(model))
        self.assertEqual((code, report), (2, None))

    def test_text_bench_aborts_on_other_tokens_or_graph_mismatches(self) -> None:
        code, report = self.run_text(self.fake_bench("A"), self.fake_bench("B", FAKE_TRACE="cd34"))
        self.assertEqual((code, report), (2, None))
        code, report = self.run_text(self.fake_bench("A"), self.fake_bench("B", FAKE_MISMATCHES="1"))
        self.assertEqual((code, report), (2, None))

    def test_speed_only_comparison_allows_other_outputs_between_sides(self) -> None:
        output = self.dir / "text.json"
        argv = ["--baseline", str(self.fake_bench("A")), "--candidate",
                str(self.fake_bench("B", FAKE_TRACE="cd34")), "--allow-output-mismatch",
                "--rounds", "4", "--repeat", "2", "--output", str(output),
                *self.counters,
                "text", "--model", "m.gguf", "--prompt-tokens", "512", "--gen-tokens", "128"]
        self.assertEqual(ab.main(argv), 0)
        report = json.loads(output.read_text())
        self.assertIn("traces differs", report["output_mismatch"])
        self.assertTrue(report["verdict"]["pass"])

    def test_abba_order_alternates_the_leading_side(self) -> None:
        self.assertEqual([ab.abba_order(i) for i in range(4)], [("A", "B"), ("B", "A"), ("A", "B"), ("B", "A")])

    def test_identical_builds_pass_and_run_in_abba_order(self) -> None:
        code, report = self.run_ab(self.fake("A"), self.fake("B"))
        self.assertEqual(code, 0)
        # The warmup round runs first and is not judged.
        self.assertEqual(self.log.read_text().split(), ["A", "B"] + ["A", "B", "B", "A", "A", "B", "B", "A"])
        self.assertEqual(len(report["invocations"]), 8)
        self.assertEqual(report["procedure"]["warmup_rounds"], 1)
        self.assertAlmostEqual(report["verdict"]["request"]["median_ratio"], 1.0)
        self.assertTrue(report["verdict"]["pass"])
        self.assertEqual(report["identity"]["generated_tokens"], 3)
        # The fake meminfo counts whole KiB.
        self.assertAlmostEqual(report["peak_memory_bytes"]["A"]["median"], 1e9, delta=1024)
        self.assertEqual(report["peak_device_memory_bytes"]["A"]["median"], 1e9)
        self.assertAlmostEqual(report["verdict"]["memory"]["median_ratio"], 1.0)

    def test_a_slower_candidate_fails(self) -> None:
        code, report = self.run_ab(self.fake("A"), self.fake("B", FAKE_SCALE="1.05"))
        self.assertEqual(code, 1)
        self.assertFalse(report["verdict"]["request"]["pass"])
        self.assertGreater(report["verdict"]["request"]["ci95"][0], 1.04)

    def test_a_faster_candidate_passes(self) -> None:
        code, report = self.run_ab(self.fake("A"), self.fake("B", FAKE_SCALE="0.9"))
        self.assertEqual(code, 0)
        self.assertLess(report["verdict"]["request"]["median_ratio"], 0.91)

    def test_different_outputs_abort_before_any_verdict(self) -> None:
        code, report = self.run_ab(self.fake("A"), self.fake("B", FAKE_TEXT="hullo"))
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        self.assertEqual(self.log.read_text().split(), ["A", "B", "A", "B"])

    def test_more_peak_memory_fails(self) -> None:
        code, report = self.run_ab(self.fake("A"), self.fake("B", FAKE_MEMORY="1100000000"))
        self.assertEqual(code, 1)
        self.assertTrue(report["verdict"]["request"]["pass"])
        self.assertFalse(report["verdict"]["memory"]["pass"])
        self.assertAlmostEqual(report["verdict"]["memory"]["median_ratio"], 1.1, places=5)

    def test_more_host_memory_fails(self) -> None:
        code, report = self.run_ab(self.fake("A"), self.fake("B", FAKE_HOST_MEMORY="100000000"))
        self.assertEqual(code, 1)
        self.assertFalse(report["verdict"]["memory"]["pass"])
        self.assertAlmostEqual(report["verdict"]["memory"]["median_ratio"], 1.1, places=5)
        self.assertAlmostEqual(report["device_memory"]["median_ratio"], 1.0)

    def test_memory_moved_to_the_device_is_judged_by_the_total(self) -> None:
        # A keeps 0.2 GB in host memory that B keeps on the device, and B
        # needs 0.1 GB less in all.
        code, report = self.run_ab(self.fake("A", FAKE_HOST_MEMORY="200000000"),
                                   self.fake("B", FAKE_MEMORY="1100000000"))
        self.assertEqual(code, 0)
        self.assertTrue(report["verdict"]["pass"])
        self.assertAlmostEqual(report["verdict"]["memory"]["median_ratio"], 1.1 / 1.2, places=5)
        self.assertAlmostEqual(report["device_memory"]["median_ratio"], 1.1)
        self.assertNotIn("device_memory", report["verdict"])

    def test_unmeasured_memory_cannot_pass(self) -> None:
        code, report = self.run_ab(self.fake("A", FAKE_MEMORY="0"), self.fake("B", FAKE_MEMORY="0"))
        self.assertEqual(code, 1)
        self.assertIsNone(report["peak_memory_bytes"])
        self.assertIsNone(report["peak_device_memory_bytes"])

    def test_generated_ids_are_compared_where_reported(self) -> None:
        code, report = self.run_ab(self.fake("A"), self.fake("B", FAKE_IDS="[1, 2, 3]"))
        self.assertEqual(code, 0)
        self.assertEqual(report["generated_ids"], [1, 2, 3])
        self.log.unlink()
        code, report = self.run_ab(self.fake("A", FAKE_IDS="[1, 2, 3]"), self.fake("B", FAKE_IDS="[1, 2, 4]"))
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_ids_must_match_a_fixture_for_the_same_audio(self) -> None:
        fixtures = self.dir / "repo/src/models/demo/tests/fixtures"
        fixtures.mkdir(parents=True)
        digest = hashlib.sha256(self.audio.read_bytes()).hexdigest()
        fixture = {"audio_sha256": digest, "max_new_tokens": 256, "text": "hello", "generated_ids": [1, 2, 3]}
        (fixtures / "demo.json").write_text(json.dumps(fixture))
        original = ab.REPOSITORY
        ab.REPOSITORY = self.dir / "repo"
        try:
            code, report = self.run_ab(self.fake("A"), self.fake("B", FAKE_IDS="[1, 2, 3]"))
            self.assertEqual(code, 0)
            self.assertEqual(report["reference"]["checked_invocations"], 4)
            code, _ = self.run_ab(self.fake("A"), self.fake("B", FAKE_IDS="[1, 2, 9]"))
            self.assertEqual(code, 2)
            code, _ = self.run_ab(self.fake("A", FAKE_TEXT="hullo"), self.fake("B", FAKE_TEXT="hullo"))
            self.assertEqual(code, 2)
        finally:
            ab.REPOSITORY = original

    def test_mixed_runtimes_are_refused(self) -> None:
        original = ab.hip_runtime
        ab.hip_runtime = lambda binary: {"library": f"/{binary.name}/libamdhip64.so", "hip_version": None}
        try:
            code, _ = self.run_ab(self.fake("A"), self.fake("B"))
            self.assertEqual(code, 2)
            self.assertFalse(self.log.exists())
            code, _ = self.run_ab(self.fake("A"), self.fake("B"), "--allow-mixed-runtime")
            self.assertEqual(code, 0)
        finally:
            ab.hip_runtime = original

    def test_bootstrap_interval_brackets_the_median(self) -> None:
        low, high = ab.bootstrap_interval([0.99, 1.0, 1.01, 1.0, 0.995, 1.005])
        self.assertLessEqual(low, 1.0)
        self.assertGreaterEqual(high, 1.0)
        self.assertLess(high - low, 0.02)

    def asr(self, **changes: object) -> ab.Workload:
        return ab.asr_workload(argparse.Namespace(**{
            "model": Path("m"), "audio": self.audio, "max_tokens": 256, "warmup": 1, "repeat": 3, **changes}))

    def record(self, binary: Path, side: str, name: str) -> Path:
        output = self.dir / f"{name}.json"
        argv = ["record", "--baseline" if side == "A" else "--candidate", str(binary), "--rounds", "4",
                "--repeat", "3", "--output", str(output), *self.counters,
                "asr", "--model", "m", "--audio", str(self.audio)]
        self.assertEqual(ab.main(argv), 0)
        return output

    def stored_run(self, name: str, side: str, request: float = 1000.0, *, load: float = 300.0,
                   text: str = "hello", sha256: str = "", library: str = "/rocm/lib/libamdhip64.so",
                   **workload: object) -> Path:
        """A recorded run of eight rounds, request times within 0.05% of `request`."""
        invocations = [
            ab.Invocation(side=side, round=i, request_ms=[request * (1 + 0.00025 * (i % 5 - 2))], load_ms=load,
                          stages={"audio_encoder_ms": 30.0, "text_decoder_ms": 900.0}, identity={"text": text},
                          peak_memory_bytes=10**9, peak_device_memory_bytes=10**9)
            for i in range(8)
        ]
        binary = {"path": f"/{side}", "sha256": sha256 or side * 64, "git_revision": None,
                  "library": library, "hip_version": None}
        procedure = {"rounds": 8, "warmup_rounds": 1, "warmup": 1, "repeat": 3}
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(ab.run_report(side, self.asr(**workload), binary, procedure, invocations, None)))
        return path

    def judge(self, baseline: list[Path], candidate: list[Path], *extra: str) -> tuple[int, dict | None]:
        output = self.dir / "judge.json"
        output.unlink(missing_ok=True)
        argv = ["judge", *(f"--baseline-run={path}" for path in baseline),
                *(f"--candidate-run={path}" for path in candidate), "--output", str(output), *extra]
        code = ab.main(argv)
        return code, json.loads(output.read_text()) if output.exists() else None

    def test_recorded_runs_are_judged_against_the_pooled_baseline(self) -> None:
        a, b = self.fake("A"), self.fake("B")
        runs = [self.record(a, "A", "A1"), self.record(a, "A", "A2"),
                self.record(b, "B", "B1"), self.record(b, "B", "B2")]
        # AABB, each run after its own warmup round.
        self.assertEqual(self.log.read_text().split(), ["A"] * 10 + ["B"] * 10)
        a1, a2, b1 = (json.loads(path.read_text()) for path in runs[:3])
        self.assertEqual((a1["schema"], a1["side"], len(a1["invocations"])), (ab.RUN_SCHEMA, "A", 4))
        self.assertEqual(a1["key"], ab.run_key(hashlib.sha256(a.read_bytes()).hexdigest(), self.asr()))
        self.assertEqual(a1["key"], a2["key"])
        self.assertNotEqual(a1["key"], b1["key"])
        code, report = self.judge(runs[:2], runs[2:])
        self.assertEqual(code, 0)
        self.assertEqual((report["next_run"], report["verdict"]), (None, {"passed": 2, "runs": 2, "pass": True}))
        self.assertTrue(report["baseline"]["consistency"]["consistent"])
        self.assertTrue(report["candidate"]["consistency"]["consistent"])
        first = report["candidate"]["runs"][0]
        self.assertEqual(first["label"], "B1")
        request = first["verdict"]["request"]
        self.assertEqual((request["median_ratio"], request["rounds"], request["baseline_rounds"]), (1.0, 4, 8))
        self.assertAlmostEqual(first["verdict"]["memory"]["median_ratio"], 1.0, places=5)

    def test_unpaired_verdict_resamples_each_side(self) -> None:
        baseline = [100.0, 100.2, 99.8, 100.1, 99.9, 100.0, 100.3, 99.7]
        same = ab.unpaired_verdict(baseline[:4], baseline, 0.005)
        low, high = same["ci95"]
        self.assertLessEqual(low, same["median_ratio"])
        self.assertGreaterEqual(high, same["median_ratio"])
        self.assertTrue(same["pass"])
        self.assertEqual((same["rounds"], same["baseline_rounds"]), (4, 8))
        slower = ab.unpaired_verdict([value * 1.02 for value in baseline[:4]], baseline, 0.005)
        self.assertAlmostEqual(slower["median_ratio"], 1.02 * 100.05 / 100.0)
        self.assertGreater(slower["ci95"][0], 1.0)
        self.assertFalse(slower["pass"])
        # Seeded, and a side without spread adds none.
        self.assertEqual(ab.unpaired_interval(baseline, baseline), ab.unpaired_interval(baseline, baseline))
        self.assertEqual(ab.unpaired_interval([2.0] * 3, [1.0] * 5), (2.0, 2.0))

    def test_consistent_candidate_runs_decide_alone(self) -> None:
        baseline = [self.stored_run("A1", "A"), self.stored_run("A2", "A")]
        code, report = self.judge(baseline, [self.stored_run("B1", "B"), self.stored_run("B2", "B")])
        self.assertEqual((code, report["verdict"]["pass"], report["next_run"]), (0, True, None))
        slow = [self.stored_run("B1", "B", 1050.0), self.stored_run("B2", "B", 1050.0)]
        code, report = self.judge(baseline, slow)
        self.assertEqual(code, 1)
        self.assertEqual((report["verdict"], report["next_run"]), ({"passed": 0, "runs": 2, "pass": False}, None))
        self.assertTrue(report["candidate"]["consistency"]["consistent"])

    def test_differing_candidate_runs_call_for_a_third_and_the_majority_decides(self) -> None:
        baseline = [self.stored_run("A1", "A"), self.stored_run("A2", "A")]
        runs = [self.stored_run("B1", "B"), self.stored_run("B2", "B", 1050.0)]
        code, report = self.judge(baseline, runs)
        self.assertEqual((code, report["next_run"], report["verdict"]["pass"]), (3, "B", None))
        self.assertEqual(report["candidate"]["consistency"]["differ"], ["request"])
        code, report = self.judge(baseline, [*runs, self.stored_run("B3", "B")])
        self.assertEqual((code, report["next_run"]), (0, None))
        self.assertEqual(report["verdict"], {"passed": 2, "runs": 3, "pass": True})
        code, report = self.judge(baseline, [*runs, self.stored_run("B3", "B", 1050.0)])
        self.assertEqual((code, report["verdict"]), (1, {"passed": 1, "runs": 3, "pass": False}))

    def test_one_failing_run_of_two_consistent_ones_calls_for_a_third(self) -> None:
        # 0.4% apart, within the tolerance of 0.5%, but only B1 is within 0.5% of A.
        baseline = [self.stored_run("A1", "A"), self.stored_run("A2", "A")]
        code, report = self.judge(baseline, [self.stored_run("B1", "B", 1003.0), self.stored_run("B2", "B", 1007.0)])
        self.assertEqual((code, report["next_run"]), (3, "B"))
        self.assertTrue(report["candidate"]["consistency"]["consistent"])
        self.assertEqual([run["verdict"]["pass"] for run in report["candidate"]["runs"]], [True, False])

    def test_differing_baseline_runs_call_for_a_third(self) -> None:
        first, second, third = (self.stored_run("A1", "A"), self.stored_run("A2", "A", load=330.0),
                                self.stored_run("A3", "A"))
        candidate = [self.stored_run("B1", "B"), self.stored_run("B2", "B")]
        code, report = self.judge([first, second], [])
        self.assertEqual((code, report["next_run"], report["verdict"]), (3, "A", None))
        self.assertEqual(report["baseline"]["consistency"]["differ"], ["load"])
        # No candidate verdict before the third baseline run.
        code, report = self.judge([first, second], candidate)
        self.assertEqual((code, report["next_run"], report["verdict"]["pass"]), (3, "A", None))
        code, report = self.judge([first, second, third], [])
        self.assertEqual((code, report["next_run"], report["verdict"]), (0, None, None))
        code, report = self.judge([first, second, third], candidate)
        self.assertEqual(code, 0)
        self.assertEqual(report["candidate"]["runs"][0]["verdict"]["load"]["baseline_rounds"], 24)

    def test_judge_refuses_other_workloads_binaries_sides_and_runtimes(self) -> None:
        baseline = [self.stored_run("A1", "A"), self.stored_run("A2", "A")]
        candidate = [self.stored_run("B1", "B"), self.stored_run("B2", "B")]
        refused = {
            "workload": (baseline, [candidate[0], self.stored_run("other", "B", max_tokens=128)]),
            "binary": ([baseline[0], self.stored_run("stale", "A", sha256="c" * 64)], candidate),
            "side": (baseline, [candidate[0], baseline[1]]),
        }
        for name, (a, b) in refused.items():
            self.assertEqual(self.judge(a, b), (2, None), name)
        mixed = [self.stored_run(f"mixed{i}", "B", library="/other/lib/libamdhip64.so") for i in (1, 2)]
        self.assertEqual(self.judge(baseline, mixed), (2, None))
        self.assertEqual(self.judge(baseline, mixed, "--allow-mixed-runtime")[0], 0)
        # A single candidate run decides nothing.
        with self.assertRaises(SystemExit):
            self.judge(baseline, candidate[:1])

    def test_run_key_separates_binaries_and_workloads(self) -> None:
        key = ab.run_key("a" * 64, self.asr())
        self.assertEqual(key, ab.run_key("a" * 64, self.asr()))
        self.assertNotEqual(key, ab.run_key("b" * 64, self.asr()))
        self.assertNotEqual(key, ab.run_key("a" * 64, self.asr(max_tokens=128)))

        def tts(**changes: object) -> ab.Workload:
            return ab.tts_workload(argparse.Namespace(**{
                "model": Path("voice"), "name": "", "server_args": "", "text": "Hello.", "voice": "vivian",
                "seed": 42, "max_tokens": 64, "request": "{}", "port": 18541, "warmup": 1, "repeat": 3,
                **changes}))

        # What a server is sent counts beyond its command line; its port does not.
        key = ab.run_key("a" * 64, tts())
        self.assertEqual(tts().command, tts(repeat=5, text="Goodbye.").command)
        self.assertEqual(key, ab.run_key("a" * 64, tts(port=1)))
        self.assertNotEqual(key, ab.run_key("a" * 64, tts(repeat=5)))
        self.assertNotEqual(key, ab.run_key("a" * 64, tts(text="Goodbye.")))

    def test_candidate_outputs_must_equal_the_stored_baseline(self) -> None:
        baseline = [self.stored_run("A1", "A"), self.stored_run("A2", "A")]
        other = [self.stored_run("B1", "B", text="hullo"), self.stored_run("B2", "B", text="hullo")]
        self.assertEqual(self.judge(baseline, other), (2, None))
        code, report = self.judge(baseline, other, "--allow-output-mismatch")
        self.assertEqual(code, 0)
        self.assertEqual(report["output_mismatch"], "text differs: A1 round 0 gave 'hello', B1 round 0 gave 'hullo'")
        # Each side must still repeat its own outputs across its runs.
        self.assertEqual(self.judge(baseline, [other[0], self.stored_run("B2h", "B")], "--allow-output-mismatch"),
                         (2, None))
        self.assertEqual(self.judge([baseline[0], self.stored_run("A2h", "A", text="hullo")], [],
                                    "--allow-output-mismatch"), (2, None))

    def test_upstream_script_stores_the_baseline_and_adds_a_third_run(self) -> None:
        tree = self.dir / "tree/tools/bench"
        tree.mkdir(parents=True)
        shutil.copy(REPOSITORY / "tools/bench/ab-upstream.sh", tree)
        # ab.py itself, recording with the fake memory counters.
        (tree / "ab.py").write_text(
            f"import sys\nsys.path.insert(0, {str(REPOSITORY / 'tools/bench')!r})\nimport ab\n"
            f"argv = sys.argv[1:]\nif argv[:1] == ['record']:\n    argv[1:1] = {self.counters!r}\n"
            "sys.exit(ab.main(argv))\n"
        )
        out = self.dir / "out"

        def upstream(a: Path, b: Path, expected: int) -> None:
            env = {**os.environ, "A": str(a), "B": str(b), "OUT": str(out),
                   "GUFO_QWEN3_ASR_MODEL": "m", "GUFO_QWEN3_ASR_AUDIO": str(self.audio)}
            completed = subprocess.run(["bash", str(tree / "ab-upstream.sh"), "asr"], env=env,
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)

        a = self.fake("A")
        upstream(a, self.fake("B"), 0)
        # AABB: eight rounds per run after a warmup round.
        self.assertEqual(self.log.read_text().split(), ["A"] * 18 + ["B"] * 18)
        (baseline,) = (out / f"baseline-{hashlib.sha256(a.read_bytes()).hexdigest()[:12]}").iterdir()
        self.assertTrue(baseline.name.startswith("asr-"))
        self.assertEqual(sorted(path.name for path in baseline.glob("A*.json")), ["A1.json", "A2.json"])
        (first,) = out.glob("candidate-*")
        self.assertTrue(json.loads((first / "asr/judge-2.json").read_text())["verdict"]["pass"])
        # The next candidate reuses the stored baseline. Its second run is 5%
        # slower than its first, so a third follows, slow as well.
        self.log.unlink()
        upstream(a, self.fake("C", FAKE_SLOW_AFTER="9"), 1)
        self.assertEqual(self.log.read_text().split(), ["C"] * 27)
        (second,) = set(out.glob("candidate-*")) - {first}
        self.assertEqual(sorted(path.name for path in (second / "asr").glob("*.json")),
                         ["B1.json", "B2.json", "B3.json", "judge-2.json", "judge-3.json"])
        report = json.loads((second / "asr/judge-3.json").read_text())
        self.assertEqual(report["verdict"], {"passed": 1, "runs": 3, "pass": False})
        self.assertEqual(len(report["baseline"]["runs"]), 2)
        self.assertTrue((first / "asr/B2.json").exists())


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    unittest.main()
