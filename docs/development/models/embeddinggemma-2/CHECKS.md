# EmbeddingGemma 2: checks

The [reference model](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) is pinned to the qualified revision; the tests read it from `EMBEDDING_GEMMA2_MODEL_ROOT`, else the Hugging Face cache snapshot of that revision, and fail when it is missing.  The probes and their reference vectors are in [`src/models/embedding_gemma2/tests/fixtures`](../../../../src/models/embedding_gemma2/tests/fixtures).  The fixture stores each reference value with five significant digits, which moves a cosine by less than 1e-6.  [`tools/models/embedding_gemma2/reference_embeddings.py`](../../../../tools/models/embedding_gemma2/reference_embeddings.py) regenerates them with transformers that include `embedding_gemma2` and sentence-transformers 6.1, and [`tokenizer_golden.py`](../../../../tools/models/embedding_gemma2/tokenizer_golden.py) regenerates the tokenizer cases.

```sh
cargo test --profile gpu-test -p gufo-embedding-gemma2-kernels --features gpu
cargo test --profile gpu-test -p gufo-embedding-gemma2 --features gpu,models
cargo test --profile gpu-test -p gufo-serve --test embeddings_api --test embeddings_http
```

The tests are in [`src/models/embedding_gemma2/tests`](../../../../src/models/embedding_gemma2/tests): `config`, `loader`, `tokenizer`, `encoder` (against the float32 CPU reference in `tests/common/reference.rs`, packing and device turns), `batch_scheduler`, `embedding_golden`, `image` (every resized patch byte), `audio` (decoder probes, and features bit for bit against C++ v0.27.0) and `media` (image and audio embeddings on the GPU); the kernels crate checks the attention kernel against a float64 evaluation and every launcher's declared extents.

The image, boundary and video fixtures come from the official processor and
model in float32 at transformers commit
`50e35cc0ca18ec19b80653690de553e6decf74f5`, independently of Gufo; each
records its patch hashes, token IDs, reference vectors and library versions.
With that checkout on `PYTHONPATH`:

```sh
python3 tools/models/embedding_gemma2/reference_images.py \
  --model models/embeddinggemma-2 \
  --transformers-revision 50e35cc0ca18ec19b80653690de553e6decf74f5 \
  --output src/models/embedding_gemma2/tests/fixtures/image_golden.json
python3 tools/models/embedding_gemma2/reference_image_boundaries.py \
  --model models/embeddinggemma-2 \
  --transformers-revision 50e35cc0ca18ec19b80653690de553e6decf74f5 \
  --output src/models/embedding_gemma2/tests/fixtures/image_boundaries.json
python3 tools/models/embedding_gemma2/reference_video.py \
  --model models/embeddinggemma-2 \
  --output src/models/embedding_gemma2/tests/fixtures/video_golden.json
```

That transformers source needs Hugging Face Hub 1.31.0 or later, newer than
`tools/requirements.txt` pins. The video oracle reuses existing encoded
fixtures and records their SHA256 values; `--case NAME` generates one probe
for a focused investigation, which is not a full qualification.

Audio is decoded and resampled by Rust crates, which differ from the C++
engine's decoders; features agree bit for bit from the PCM samples on
([relation to Gufo](../../PARITY.md#implemented-differently)).

This engine has not been benchmarked yet. Gufo's
[benchmarks](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/embeddinggemma-2/BENCHMARKS.md) measured the C++ engine, whose
kernels, launch shapes, row-chunked GEMMs and hipBLASLt timing this engine
keeps. Measure HTTP latency and the encoder GEMMs with:

```sh
cargo build --release -p gufo
./target/release/gufo serve embedding --model models/embeddinggemma-2 &
python3 tools/models/embedding_gemma2/http_bench.py \
  --base-url http://127.0.0.1:8080/v1 --model embeddinggemma-2
cargo test --profile gpu-test -p gufo-embedding-gemma2 --features gpu \
  --test gemm_bench -- --ignored --nocapture
```
