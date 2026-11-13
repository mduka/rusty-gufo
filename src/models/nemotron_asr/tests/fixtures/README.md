# Nemotron 3.5 ASR test fixtures

The clips and transformers' outputs are copied from the C++ Nemotron ASR
extension ([gufo-extensions/nemotron_asr](https://github.com/gufo-extensions/nemotron_asr),
`d54c7da`, `tests/fixtures`, unchanged at `c256b69`); the C++ outputs come
from a build of the extension at `c256b69` (0.1.2), the same bytes as at
0.1.1 (`0823ee4`, later amended to `d3c4e4c`). The scripts that recorded
them are in [`tools/reference`](../../tools/reference).

| File | Recorded by | Contents |
| --- | --- | --- |
| `english.wav`, `german.wav` | The extension's author, with VoxCPM2 voice design (no real person's voice) | Two sentences each, 16 kHz mono PCM16: English 10.08 s, German 6.08 s |
| `golden.safetensors`, `golden.json` | `golden.py` (transformers 5.22.2, float32 on the CPU, model revision `ea30d66d`) | Features of both clips; subsampled, encoder and projected outputs of the first three chunks at lookahead 13; the prediction network's start state and the joint's first logits; token IDs, transcripts, chunk and frame counts of 16 offline transcriptions (both clips, the clip's language and `auto`, lookaheads 0, 3, 6 and 13) |
| `cxx_reference.safetensors` | `pack_cxx_reference.py`, from the outputs of the extension's C++ build (Gufo 0.34.0 `2845939b`, ROCm 10.1) | The same encoder outputs, start state and logits from the golden features, and every golden run's token IDs (int32) and the prediction network's final output, each clip pushed at once |

None of them contains weights; the tensors derive from NVIDIA's
checkpoint (OpenMDW-1.1).
