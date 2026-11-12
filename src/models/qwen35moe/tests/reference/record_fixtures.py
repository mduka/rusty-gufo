#!/usr/bin/env python3
"""Records the Qwen3.6 35B-A3B (qwen35moe) C++ reference fixtures of
gufo-qwen35moe with the probe of this directory (q35moe_probe.cpp, built as
README.md describes).

Runs the probe twice per target GGUF, the second time with the modes in
reverse order, compares the two runs (parsed JSON, row files byte for byte),
refuses a run that mapped a HIP-stack library outside $ROCM_PATH/lib, and
writes the first run's output, formatted, to OUT:
qwen3.6-35b-a3b-<set>-<mode>.json, qwen3.6-35b-a3b-<set>-meta.json and
rows/<set>/*.f32, with <set> the quantization plus "-kvq8" for
--kv-cache q8_0.

Paths come from the tests' environment variables unless given as arguments:
GUFO_QWEN35MOE_MODEL (UD-Q4_K_M), GUFO_QWEN35MOE_Q8_MODEL (Q8_0),
GUFO_QWEN35MOE_DFLASH_MODEL, GUFO_QWEN35MOE_MMPROJ (the BF16 sidecar) and
GUFO_QWEN35MOE_FIXTURES (OUT; its images/ holds the input PNGs).

--wrap is a command prefix for every probe process, for example
"systemd-run --user --scope --quiet -p OOMPolicy=stop -- choom -n 1000 --";
a wrapper that exits with 75 ("busy") is retried every 60 s.

Usage: record_fixtures.py --probe PATH --work DIR [--rocm-path DIR]
       [--out DIR] [--quant ud-q4_k_m|q8_0 ...] [--modes a,b] [--runs 2]
       [--kv-cache f16|q8_0] [--dflash-model PATH] [--mmproj PATH]
       [--images DIR] [--wrap CMD]
"""
import argparse, filecmp, hashlib, json, os, shlex, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
# Quantization -> the environment variable naming its GGUF (tests/common).
MODELS = {"ud-q4_k_m": "GUFO_QWEN35MOE_MODEL", "q8_0": "GUFO_QWEN35MOE_Q8_MODEL"}
IMAGES = ("synthetic-200x150", "bison-1024x668", "aldrin-960x966")
MODES = ["prefill", "decode", "session", "snapshot", "batched", "sampled",
         "mtp-head", "mtp", "dflash", "mtp-batch", "dflash-batch"]
DFLASH_MODES = {"dflash", "dflash-batch", "vision"}
MAX_CONTEXT = 6144
# The probe has no RUNPATH: without LD_LIBRARY_PATH it may load another
# ROCm's libraries (a distribution's) instead of the one it was built with.
HIP_STACK = ("libamdhip64", "librocblas", "libhipblas", "libhipblaslt", "libhsa-runtime64",
             "libamd_comgr", "libhsakmt", "librocsolver", "libroctx64", "librocprofiler")
SOURCE = ("C++ reference: slimsami/strycast-qwen35moe-gufo-extension 4882b7a with the probe "
          "src/models/qwen35moe/tests/reference/q35moe_probe.cpp, built into Gufo 0fb50ce4 "
          "(cmake preset gpu-test, ROCm 10.1) and run on ROCm 10.1's libraries "
          "(LD_LIBRARY_PATH={rocm_lib}); see src/models/qwen35moe/tests/reference/README.md")


def run_probe(args, quant, out, modes):
    # The MTP head is the target GGUF's own; meta.json names every model given.
    cmd = shlex.split(args.wrap) + [
        "env", f"LD_LIBRARY_PATH={args.rocm_lib}",
        args.probe, args.model[quant], out, "--max-context", str(MAX_CONTEXT),
        "--mtp-model", args.model[quant], "--kv-cache", args.kv_cache]
    if args.dflash_model:
        cmd += ["--dflash-model", args.dflash_model]
    if args.mmproj and args.images:
        cmd += ["--mmproj", args.mmproj, "--images",
                ",".join(os.path.join(args.images, f"{name}.png") for name in IMAGES)]
    cmd += ["--modes", ",".join(modes)]
    log = out + ".log"
    for attempt in range(60):
        os.makedirs(out, exist_ok=True)
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 75:
            break
        print(f"  GPU busy, retry in 60 s ({attempt + 1})", flush=True)
        time.sleep(60)
    if rc != 0:
        sys.exit(f"probe failed ({rc}), see {log}")
    check_libraries(args.rocm_lib, out)


def check_libraries(rocm_lib, out):
    """Every HIP-stack library the probe mapped must come from rocm_lib."""
    mapped = json.load(open(os.path.join(out, "libraries.json")))["mapped"]
    stack = [p for p in mapped if os.path.basename(p).startswith(HIP_STACK)]
    foreign = [p for p in stack if not p.startswith(rocm_lib + "/")]
    names = {os.path.basename(p).split(".so")[0] for p in stack}
    if foreign or not {"libamdhip64", "librocblas", "libhipblas", "libhipblaslt"} <= names:
        sys.exit(f"probe in {out} mapped foreign or missing ROCm libraries: {foreign or stack}")


def fmt(value, indent=0):
    """JSON with objects indented and short or scalar lists on one line."""
    pad, inner = " " * indent, " " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{inner}{json.dumps(k)}: {fmt(v, indent + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        compact = json.dumps(value, ensure_ascii=False)
        if all(not isinstance(v, (list, dict)) for v in value) or (
                len(compact) <= 160 and not any(isinstance(v, dict) for v in value)):
            return compact
        return "[\n" + ",\n".join(inner + fmt(v, indent + 1) for v in value) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False)


def diff(a, b, path=""):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        out = [f"{path}.{k}" for k in a.keys() ^ b.keys()]
        for k in a.keys() & b.keys():
            out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def parse_args():
    env = os.environ.get
    fixtures = env("GUFO_QWEN35MOE_FIXTURES")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--probe", required=True, help="the built q35moe_probe")
    parser.add_argument("--work", required=True, help="directory for the raw runs and logs")
    parser.add_argument("--rocm-path", default=env("ROCM_PATH"),
                        help="the ROCm 10.1 prefix the probe was built with ($ROCM_PATH)")
    parser.add_argument("--out", default=fixtures, help="fixture directory "
                        "($GUFO_QWEN35MOE_FIXTURES)")
    parser.add_argument("--quant", action="append", choices=MODELS)
    parser.add_argument("--modes", default=",".join(MODES),
                        help=f"comma-separated, of {','.join(MODES)},vision")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--kv-cache", default="f16", choices=["f16", "q8_0"])
    parser.add_argument("--dflash-model", default=env("GUFO_QWEN35MOE_DFLASH_MODEL"))
    parser.add_argument("--mmproj", default=env("GUFO_QWEN35MOE_MMPROJ"))
    parser.add_argument("--images", default=fixtures and os.path.join(fixtures, "images"),
                        help=f"directory with {', '.join(n + '.png' for n in IMAGES)}")
    parser.add_argument("--wrap", default="", help="command prefix for every probe run")
    args = parser.parse_args()
    args.quant = args.quant or list(MODELS)
    args.modes = args.modes.split(",")
    missing = [name for name, value in (("--rocm-path or ROCM_PATH", args.rocm_path),
                                        ("--out or GUFO_QWEN35MOE_FIXTURES", args.out))
               if not value]
    args.model = {}
    for quant in args.quant:
        args.model[quant] = env(MODELS[quant])
        if not args.model[quant]:
            missing.append(MODELS[quant])
    if DFLASH_MODES & set(args.modes) and not args.dflash_model:
        missing.append("--dflash-model or GUFO_QWEN35MOE_DFLASH_MODEL")
    if "vision" in args.modes:
        if not args.mmproj:
            missing.append("--mmproj or GUFO_QWEN35MOE_MMPROJ")
        if not args.images:
            missing.append("--images or GUFO_QWEN35MOE_FIXTURES")
    if missing:
        parser.error("missing " + ", ".join(missing))
    # /proc/self/maps names resolved paths.
    args.rocm_lib = os.path.join(os.path.realpath(args.rocm_path), "lib")
    args.probe = os.path.abspath(args.probe)
    return args


def main():
    args = parse_args()
    probe_sha = hashlib.sha256(
        open(os.path.join(HERE, "q35moe_probe.cpp"), "rb").read()).hexdigest()[:16]
    source = SOURCE.format(rocm_lib=args.rocm_lib)
    summary = {}
    kv_tag = "" if args.kv_cache == "f16" else "-kvq8"
    for quant in args.quant:
        runs = []
        for run in range(args.runs):
            out = os.path.join(args.work, quant + kv_tag, f"run{run + 1}")
            shutil.rmtree(out, ignore_errors=True)
            order = args.modes if run % 2 == 0 else args.modes[::-1]
            print(f"{quant} run {run + 1}: {','.join(order)}", flush=True)
            start = time.time()
            run_probe(args, quant, out, order)
            print(f"  done in {time.time() - start:.0f} s", flush=True)
            runs.append(out)
        meta = json.load(open(os.path.join(runs[0], "meta.json")))
        result = {}
        for mode in args.modes:
            docs = [json.load(open(os.path.join(r, f"{mode}.json"))) for r in runs]
            diffs = sorted({d for other in docs[1:] for d in diff(docs[0], other)})
            result[mode] = diffs
            fixture = {
                "source": source,
                "probe_sha256_16": probe_sha,
                "model": meta["model"],
                "max_context": meta["max_context"],
                "kv_cache": meta["kv_cache"],
                "mode": mode,
                "rocm_libraries": args.rocm_lib,
                "repeat_runs": len(runs),
                "repeat_identical": not diffs,
            }
            if diffs:
                fixture["repeat_differences"] = diffs[:50]
            fixture.update(docs[0])
            os.makedirs(args.out, exist_ok=True)
            name = f"qwen3.6-35b-a3b-{quant}{kv_tag}-{mode}.json"
            with open(os.path.join(args.out, name), "w") as f:
                f.write(fmt(fixture) + "\n")
        rows_dir = os.path.join(runs[0], "rows")
        if os.path.isdir(rows_dir):
            dest = os.path.join(args.out, "rows", quant + kv_tag)
            os.makedirs(dest, exist_ok=True)
            for row in sorted(os.listdir(rows_dir)):
                same = all(filecmp.cmp(os.path.join(rows_dir, row), os.path.join(r, "rows", row),
                                       shallow=False) for r in runs[1:]
                           if os.path.exists(os.path.join(r, "rows", row)))
                result[f"rows/{row}"] = [] if same else ["row bytes differ between runs"]
                shutil.copyfile(os.path.join(rows_dir, row), os.path.join(dest, row))
        meta_path = os.path.join(args.out, f"qwen3.6-35b-a3b-{quant}{kv_tag}-meta.json")
        meta_out = dict(meta)
        meta_out["libraries_run1"] = json.load(open(os.path.join(runs[0], "libraries.json")))
        # A run of some modes keeps the timings recorded for the others.
        timings = json.load(open(meta_path)).get("timings_run1", {}) if os.path.exists(meta_path) else {}
        timings.update(json.load(open(os.path.join(runs[0], "timings.json"))))
        meta_out["timings_run1"] = timings
        with open(meta_path, "w") as f:
            f.write(fmt(meta_out) + "\n")
        summary[quant] = result
        for mode, diffs in result.items():
            print(f"  {mode}: {'identical' if not diffs else f'{len(diffs)} differences'}")
            for d in diffs[:10]:
                print(f"    {d}")
    with open(os.path.join(args.work, f"summary{kv_tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
