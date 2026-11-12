# llama.cpp parity tools

Standalone programs that dump llama.cpp's results in the layouts the
package's probes (`examples/`) read. They build against a llama.cpp
checkout, not rusty-gufo. The probes run with
`cargo run --release -p gufo-nemotron --features gpu --example <probe> -- …`;
below they are named by their example.

## Model logits

`llama_logits.cpp` writes llama.cpp float32 logits for every position of a
token sequence, or prints llama.cpp's tokenization of a text. Per-model
reference probes (for example `reference_probe`) write the
same layout, one vocabulary-wide row per position, and `compare_logits.py`
reports top-1 agreement, KL(reference || candidate) and the largest logit
difference per position.

```sh
L=/path/to/llama.cpp   # a checkout with its build in build/bin
g++ -std=c++17 -O2 -I$L/include -I$L/ggml/include \
  tools/llama_parity/llama_logits.cpp -L$L/build/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,$L/build/bin -o "$OUT/llama_logits"
TOKENS=$("$OUT/llama_logits" MODEL.gguf --text "The capital of France is")
"$OUT/llama_logits" MODEL.gguf --tokens "$TOKENS" --dump "$OUT/llama.bin" --f32-kv
# --cpu runs llama.cpp's CPU backend only (no layers or ops offloaded); compare
# both backends to see llama.cpp's own spread before judging a difference.
python3 tools/llama_parity/compare_logits.py "$OUT/llama.bin" "$OUT/rocm.bin" \
  --vocab VOCAB
```

For this model, [prompts/short.txt](prompts/short.txt) (a one-line question)
and [prompts/long.txt](prompts/long.txt) (a system message, a code answer and
a follow-up) are the chat-formatted prompts of the quality comparisons. Read
them with `--text-file`, which keeps the trailing newline after `<think>`, and
tokenize with `--special`. The ROCm runtime dumps the same layout, from one
prefill or, with `--step`, one token at a time through the decode graph:

```sh
TOKENS=$("$OUT/llama_logits" MODEL.gguf --special \
  --text-file tools/llama_parity/prompts/short.txt)
runtime_probe --model MODEL.gguf --tokens "$TOKENS" \
  --dump "$OUT/rocm.bin" [--step]
python3 tools/llama_parity/compare_logits.py "$OUT/llama.bin" "$OUT/rocm.bin" \
  --vocab 131072
```

## DFlash drafters

`llama_dflash.cpp` decodes a token sequence with a target model, injects the
extracted target-layer inputs into a DFlash drafter as llama.cpp's
`common/speculative.cpp` does, and drafts one block at each anchor position
from the features before it. It writes the target features, the drafter's
projected context rows and each anchor's block logits and LiLiCorr lattice
rows, and prints each anchor's greedy lattice walk (`*` marks a token that
differs from the sequence). Per-model probes (for example
`drafter_probe`) read the dump, inject the same features and
compare. It needs llama.cpp's internal `llama-ext.h` and `libllama-common`.

```sh
L=/path/to/llama.cpp   # a checkout with its build in build/bin
g++ -std=c++17 -O2 -I$L/common -I$L/include -I$L/ggml/include -I$L/vendor \
  -I$L/src tools/llama_parity/llama_dflash.cpp -L$L/build/bin \
  -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin \
  -o "$OUT/llama_dflash"
"$OUT/llama_dflash" TARGET.gguf DRAFT.gguf --tokens "$TOKENS" \
  --anchors 40,52,64 --dump "$OUT/llama_dflash.bin" --f32-kv
```

For Nemotron 3.5 Bolt the sequence is [prompts/fibonacci.txt](prompts/fibonacci.txt)
followed by the trunk's greedy answer (`runtime_probe --generate 240`
prints its IDs), with anchors every 12th position from 40 to 268:

```sh
drafter_probe --model TARGET.gguf --drafter DRAFT.gguf \
  --dump "$OUT/llama_dflash.bin"
```
