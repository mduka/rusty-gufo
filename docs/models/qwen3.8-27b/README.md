# Qwen3.8 27B

Dense text/image model on gfx1151. Production target GGUFs are **UD-Q4_K_XL**
(16.35 GiB weights) and **UD-Q8_K_XL** (29.30 GiB); request state and an optional
draft/projector need additional memory. BF16 targets are reference-only.

Speed: [performance](../../PERFORMANCE.md#qwen38-27b) · Quality:
[Gufo's quality report](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-27b/QUALITY.md),
which applies because the outputs are Gufo's ([verification](#verification)).

## Load and run

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf \
  --local-dir models/qwen3.8-27b
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  --revision 2d9571f8ce46e151f61c6499c99dee6079e1d610 \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models/qwen3.8-27b
MODEL=models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFT=models/qwen3.8-27b/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
./target/release/gufo chat --model "$MODEL"
./target/release/gufo serve llm --model "$MODEL" --speculative dflash2 \
  --dflash-model "$DRAFT" --sessions 2 --context 32768
```

Substitute `Qwen3.8-27B-UD-Q8_K_XL.gguf` for Q8. Targets and drafts need not
have matching precision. Use the Q4_K_M DFlash2 draft with either target.

| Mode | Selection | Behavior |
| --- | --- | --- |
| AR | No speculative option | Target-only generation. |
| DFlash2 | `--speculative dflash2 --dflash-model PATH` | Adaptive default; `--draft-tokens` caps proposals, `--draft-policy fixed` is a comparison mode. |

Greedy speculation must match AR. Sampled modes preserve the target policy,
but AR and speculation need not share the same-seed sequence. See the
[quality contract](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-27b/QUALITY.md). Thinking/template controls and HTTP
sampling defaults are described in [the server guide](../../SERVER.md#reasoning-controls).

## Images

Use the matching `mmproj-BF16.gguf` beside the target or pass `--mmproj PATH`.
`prompt` and `chat` upload vision weights with the first image; `serve`
prepares them before it reports ready. Repeat `--image` in `prompt` or `chat`; chat
attaches them to the first user turn. Images count toward context capacity.

```sh
./target/release/gufo prompt --model "$MODEL" --image photo.jpg \
  --prompt "Describe this image." --temperature 0
```

Send an image to the server:

```python
import base64, json, urllib.request
image = base64.b64encode(open("photo.jpg", "rb").read()).decode()
body = {"model": "vision-test", "temperature": 0, "max_tokens": 64,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
            {"type": "text", "text": "Describe this image."}]}]}
request = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request).read().decode())
```

Start that server with `--served-model-name vision-test`. Ordered text/image
parts, multiple images and later turns work with AR/DFlash2 and caches.
PNG/JPEG data URLs and public HTTPS are supported; requests
share a 20 MiB encoded-byte, 16-image and 15-second download budget. Each image
is capped at 32 megapixels; private/loopback/link-local destinations are rejected.
Official dynamic resizing allows 64–16384 merged image tokens. `detail: low`
scales the image down to fit within 512 × 512 pixels and keeps its aspect ratio,
so it uses at most 256 merged image tokens and never more than `auto`.
`detail: high`, `original`, and `auto` use the full range, and `auto` is the
default.

Image numbering is off by default; `--add-vision-id` or
`chat_template_kwargs: {"add_vision_id": true}` adds `Picture N:` prefixes.
Cache identity includes processed pixels, placement, preprocessing version and
projector contents. See [vision checks and limits](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-27b/QUALITY.md#vision).


## Verification

Prefill and greedy generation reproduce the logits of Gufo v0.28.1's C++
build bit for bit, and every workload of the
[A/B comparison](../../development/PARITY.md#performance-against-gufo)
(autoregressive, DFlash2, serving, sampled) gave the same outputs on both
builds. The checks are listed in
[Qwen3.8 27B checks](../../development/models/qwen3.8-27b/CHECKS.md).
