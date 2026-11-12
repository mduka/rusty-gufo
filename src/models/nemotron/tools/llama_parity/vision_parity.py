#!/usr/bin/env python3
"""Compare image answers of two OpenAI-compatible servers.

Asks a server about each image (a single image before the question, greedy,
thinking off) and records the generated tokens with their top log
probabilities; `compare` then reports, per case, whether both servers chose
the same tokens and how far their log probabilities of the first token's
candidates lie apart.

    python3 vision_parity.py ask --port 8080 --out gufo.json IMAGE...
    python3 vision_parity.py ask --port 8081 --out llama.json IMAGE...
    python3 vision_parity.py compare gufo.json llama.json

Images whose sides are multiples of 32 pixels within 1024 to 13312
patches (512x512, 1024x1024) need no resizing in either implementation, so
both encoders see the same pixels and positions.
"""

import argparse
import base64
import json
import math
import mimetypes
import sys
import urllib.request
from pathlib import Path

STOP = {"<|im_end|>", ""}
QUESTIONS = [
    "Describe this image in one sentence.",
    "What is the most prominent color in this image? Answer with one word.",
]


def ask(port, image, question, tokens):
    mime = mimetypes.guess_type(image)[0] or "image/png"
    data = base64.b64encode(Path(image).read_bytes()).decode()
    body = {
        "model": "nemotron-3.5-bolt",
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{data}"}},
            {"type": "text", "text": question}]}],
        "max_tokens": tokens, "temperature": 0, "top_k": 1,
        "logprobs": True, "top_logprobs": 10,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request, timeout=900))
    choice = reply["choices"][0]
    steps = [{"token": c["token"], "logprob": c["logprob"],
              "top": {t["token"]: t["logprob"] for t in c["top_logprobs"]}}
             for c in (choice.get("logprobs") or {}).get("content") or []]
    return {"text": choice["message"].get("content"),
            "prompt_tokens": reply["usage"]["prompt_tokens"], "steps": steps}


def compare(a_path, b_path):
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    for key in a:
        x, y = a[key], b.get(key)
        if y is None:
            continue
        same = 0
        chosen = []
        for sx, sy in zip(x["steps"], y["steps"]):
            # Servers spell the end of turn differently ("<|im_end|>" or "").
            if sx["token"] != sy["token"] and {sx["token"], sy["token"]} != STOP:
                break
            same += 1
            chosen.append(abs(sx["logprob"] - sy["logprob"]))
        first_x, first_y = x["steps"][0], y["steps"][0]
        shared = set(first_x["top"]) & set(first_y["top"])
        worst = max((abs(first_x["top"][t] - first_y["top"][t]) for t in shared),
                    default=math.nan)
        print(f"{key}\n  prompt tokens {x['prompt_tokens']} / {y['prompt_tokens']}, "
              f"identical tokens {same} of {min(len(x['steps']), len(y['steps']))}, "
              f"first token {first_x['token']!r} / {first_y['token']!r}, "
              f"top-10 overlap {len(shared)}, max |dlogprob| {worst:.4f}, "
              f"chosen tokens' |dlogprob| mean {sum(chosen) / len(chosen):.4f} "
              f"max {max(chosen):.4f}\n"
              f"  A: {x['text']!r}\n  B: {y['text']!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("ask")
    run.add_argument("--port", type=int, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--tokens", type=int, default=48)
    run.add_argument("images", nargs="+")
    cmp = sub.add_parser("compare")
    cmp.add_argument("a")
    cmp.add_argument("b")
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.a, args.b)
        return
    results = {}
    for image in args.images:
        for question in QUESTIONS:
            key = f"{Path(image).name}: {question}"
            results[key] = ask(args.port, image, question, args.tokens)
            print(key, "->", repr(results[key]["text"]), file=sys.stderr)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
