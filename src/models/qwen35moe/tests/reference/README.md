# Qwen3.6 35B-A3B: C++ reference probe

The fixtures of `gufo-qwen35moe`
([checks](../../../../../docs/development/models/qwen3.6-35b-a3b/CHECKS.md))
come from the C++ extension
[slimsami/strycast-qwen35moe-gufo-extension](https://github.com/slimsami/strycast-qwen35moe-gufo-extension)
at commit `4882b7a6954fe472218d0d120ee3665076fe71fa`, built into Gufo
`0fb50ce436044a9687c78635e7afbc2bb65fc6f4` with ROCm 10.1, through the probe
in this directory. Cargo builds none of it.

| File | Contents |
| --- | --- |
| `q35moe_probe.cpp` | The probe: per mode, the extension's entry points with the CLI's and the serve runner's arguments (its header lists the modes) |
| `q35moe_probe.cmake` | Its test target in the extension's build, excluded from `all` |
| `q35moe-probe-hook.patch` | Hooks the target into the extension: an optional `include` of `tests/q35moe_probe.cmake` in `extension.cmake` and a `q35moe_probe` test target in `extension.json` |
| `vision-gate.patch` | Admits the projection width 2048 in the extension's three vision gates (`drivers.cpp` `LoadQwenVision`, `runner.cpp` `CreateQwenTextRunner`, `vision/encoder.hip`); without it the extension refuses images for this model. Gates only: no arithmetic or weight-format change, and the text modes give the same bits with and without it |
| `record_fixtures.py` | Runs the probe twice per target, compares the runs and writes the fixtures |

The probe, its CMake snippet and the script are this project's code (MIT OR
Apache-2.0), written against the extension's headers. The two patches are
small diffs of the extension's files, which stay under its
[MIT license](../../../../../licenses/qwen35moe-extension.txt).

## Build

`$UPSTREAM` is a Gufo checkout, `$EXTENSION` a checkout of the extension and
`$ROCM_PATH` a ROCm 10.1 prefix. From this directory:

```sh
git -C $UPSTREAM checkout 0fb50ce436044a9687c78635e7afbc2bb65fc6f4
git -C $EXTENSION checkout 4882b7a6954fe472218d0d120ee3665076fe71fa
mkdir -p $EXTENSION/tests
cp q35moe_probe.cpp q35moe_probe.cmake $EXTENSION/tests/
git -C $EXTENSION apply $PWD/q35moe-probe-hook.patch $PWD/vision-gate.patch
R=$ROCM_PATH G=$(dirname "$(cc -print-libgcc-file-name)")
cd $UPSTREAM
HIP_PATH=$R cmake --preset gpu-test -B build/q35moe-gpu -DROCM_PATH=$R \
  -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$R -isystem $R/include --gcc-install-dir=$G" \
  -DGUFO_EXTENSION_DIRS=$EXTENSION
HIP_PATH=$R cmake --build build/q35moe-gpu --parallel 16 --target q35moe_probe gufo
```

The probe lands in
`build/q35moe-gpu/extensions/build/strycast_qwen35moe/build/q35moe_probe`;
`build/q35moe-gpu/gufo` is the extension's CLI from the same build, which the
fixtures were cross-checked against. The `gpu-test` preset is RelWithDebInfo:
kernels without per-source options compile at `-O2`, not at the release
preset's `-O3`, so compare kernel ISA when a Rust build uses other flags.

## Run and record

The probe and `gufo` have no RUNPATH. Run them with
`env LD_LIBRARY_PATH=$ROCM_PATH/lib`: otherwise the loader may take another
ROCm's libraries, such as a distribution's in `/usr/lib`, whose hipBLAS,
rocBLAS and hipBLASLt round the prefill GEMMs and the vision encoder
differently.

```sh
env LD_LIBRARY_PATH=$ROCM_PATH/lib $PROBE MODEL OUTDIR --max-context 6144 \
  --mtp-model MODEL --modes prefill,decode
```

`record_fixtures.py` runs the probe that way, twice per target with the modes
in reverse order the second time; it refuses a run that mapped a HIP library
from outside `$ROCM_PATH/lib`, compares the two runs and writes the first,
formatted. It reads the files from the tests' variables (or arguments,
`--help`); the [user page](../../../../../docs/models/qwen3.6-35b-a3b/README.md#files)
lists where they come from:

```sh
export GUFO_QWEN35MOE_MODEL=$MODELS/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
export GUFO_QWEN35MOE_Q8_MODEL=$MODELS/Qwen3.6-35B-A3B-Q8_0.gguf
export GUFO_QWEN35MOE_DFLASH_MODEL=$MODELS/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf
export GUFO_QWEN35MOE_MMPROJ=$MODELS/mmproj-BF16.gguf
export GUFO_QWEN35MOE_FIXTURES=$FIXTURES   # the output; images/ holds the inputs
PROBE=$UPSTREAM/build/q35moe-gpu/extensions/build/strycast_qwen35moe/build/q35moe_probe
python3 record_fixtures.py --probe $PROBE --work $WORK --modes \
  prefill,decode,session,snapshot,batched,sampled,mtp-head,mtp,dflash,mtp-batch,dflash-batch,vision
python3 record_fixtures.py --probe $PROBE --work $WORK --quant ud-q4_k_m \
  --kv-cache q8_0
```

Each run loads a 21 or 35 GiB model. `--wrap` puts a command in front of every
probe process, for example
`systemd-run --user --scope --quiet -p OOMPolicy=stop -- choom -n 1000 --` so
that an out-of-memory kill takes the probe first and only its scope; a
wrapper that exits with 75 is retried every minute.

The images in `$FIXTURES/images` are decoded by Gufo's PNG decoder (8-bit
RGB): `synthetic-200x150.png` (sha256 `caf4230c…`, drawn by
[`make_synthetic.py`](make_synthetic.py)), `bison-1024x668.png` (`e8c2eb55…`, the public-domain
[American bison k5680-1](https://commons.wikimedia.org/wiki/File:American_bison_k5680-1.jpg),
its 1280-pixel thumbnail scaled with `ffmpeg -vf scale=1024:668:flags=lanczos
-pix_fmt rgb24 -map_metadata -1`) and `aldrin-960x966.png` (`17d2efd2…`, the
public-domain
[Aldrin Apollo 11 original](https://commons.wikimedia.org/wiki/File:Aldrin_Apollo_11_original.jpg),
its 960-pixel thumbnail converted with `-pix_fmt rgb24 -map_metadata -1`).

## The fixtures

`record_fixtures.py` writes into `$GUFO_QWEN35MOE_FIXTURES`:

- `qwen3.6-35b-a3b-<set>-<mode>.json`, `<set>` one of `ud-q4_k_m`, `q8_0`
  and `ud-q4_k_m-kvq8` (`--kv-cache q8_0`), `<mode>` one of `prefill`,
  `decode`, `session`, `snapshot`, `batched`, `sampled`, `mtp-head`, `mtp`,
  `dflash`, `mtp-batch`, `dflash-batch` and, for the first two sets,
  `vision`. Each starts with `source`, `probe_sha256_16` (of
  `q35moe_probe.cpp`), `model`, `max_context` (6144), `kv_cache`, `mode`,
  `rocm_libraries`, `repeat_runs` (2) and `repeat_identical`, followed by the
  probe's output: token ids, FNV-1a 64 digests of the exact f32 logit rows,
  the top 5 per step and, for the speculative modes, the drafted, accepted
  and emitted tokens per verification step.
- `qwen3.6-35b-a3b-<set>-meta.json`: the model's configuration (EOS 248046,
  `max_prompt_batch` 4096), the libraries run 1 mapped, and its load and mode
  timings, which no comparison uses.
- `rows/<set>/prefill-{haiku1,haiku20,body5077}.f32`: the step-0 logits of
  three prefill cases (248,320 f32 each); for `vision`,
  `rows/<set>/vision-embedding-synthetic-200x150.f32` (70 × 2048 f32) and
  `vision-prefill-synthetic-200x150.f32`.

The `ud-q4_k_m` and `q8_0` mode files are checked in as
`src/models/qwen35moe/tests/fixtures`; the rows, the images, the meta files
and the `-kvq8` set (the 8-bit KV cache is not ported) stay in the local
reference set. The tests require `repeat_identical`, `kv_cache` `f16` and
`rocm_libraries` `/opt/rocm/core-10.1/lib` (the script resolves symbolic
links). The checked-in fixtures name the probe that recorded them,
`399bd42e09d0edc0`; the copy here differs from it only in its header comment
and clang-format layout, so a new recording names another hash.
