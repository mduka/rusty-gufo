# CLI (`src/cli`)

The `gufo` binary (package `gufo`), its HTTP server (`serve`, package
`gufo-serve`) and the CLI's own HIP kernels (`kernels`, package
`gufo-cli-kernels`).

## Philosophy

1. **One Binary**: All user-facing capabilities (`serve`, `arbiter`, `prompt`, `chat`, `bench`, `eval`, `transcribe`, `diagnose`, `probe`) live under the single `gufo` executable.
2. **Subcommands Over Flag Soup**: Model-specific or modality-specific options belong under explicit subcommands (e.g. `gufo serve ... llm ...`, `gufo serve ... tts ...`), keeping global options clean and self-documenting.
3. **Declarative Parsing**: Flags and arguments are declared and bound to typed fields with `ArgParser` (`src/arg_parser.rs`). Never parse raw arguments manually.
4. **CLI Layer vs Engine Core**: The CLI handles argument parsing, terminal output and the server lifecycle. Model computation and kernel dispatch belong in `src/models/` and `src/core/`.

## Organization

```
src/cli/
├── src/main.rs           # Root dispatcher & global flags (-h, -v, help)
├── src/arg_parser.rs     # Declarative argument parser
├── src/arbiter.rs        # `gufo arbiter` (GPU sharing between servers)
├── src/bench.rs, bench/  # `gufo bench` (model throughput)
├── src/diagnose/         # `gufo diagnose` & `gufo probe` (hardware inventory & toolchain probe)
├── src/gpu_queues.rs     # The device-wide hardware compute-queue budget
├── src/eval.rs           # `gufo eval` (capability evaluation through a server)
├── src/prompt.rs, prompt/ # `gufo prompt` & `gufo chat` (text generation CLI & REPL)
├── src/serve.rs          # `gufo serve` startup per modality (`llm`, `tts`, `asr`, `image`)
├── src/serve_cli.rs      # `gufo serve` options and help
├── src/transcribe.rs     # `gufo transcribe` (Qwen3-ASR)
├── serve/                # gufo-serve: HTTP server, APIs, scheduler, caches
└── kernels/              # gufo-cli-kernels: the CLI's own HIP kernels
```

## Adding a Command or Flag

- **Adding a Flag**:
  - Add a field to the command's options struct.
  - Bind it in the command's `ArgParser` with `add_option`, `add_flag` or `add_custom_option`.
  - **Group Organization**: Place the flag in a logical group (e.g. `Model`, `Sampling`, `Speculative`, `Diagnostics`). If no existing group fits, create a new meaningful group.
  - **Descriptions**: Keep descriptions concise yet explanatory (include defaults and expected value formats).
  - **Partial Implementation Notation**: If a flag is not yet supported for all models, annotate the description with `(TODO: <model>)`.
- **Adding a Command**:
  1. Create its module under `src/` with a `run(args: &[String]) -> ExitCode`.
  2. Register the routing and help line in `src/main.rs`.
  3. Write unit tests beside the code and process-level tests in `tests/`.
