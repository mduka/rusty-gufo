"""Records a prefill fixture of gufo-qwen35 (`tests/prefill.rs`) from the C++
engine: per prefill case `cpp_decode_dump prefill` (the prompt through
ForwardPromptBatch, then the case's greedy steps), per generate case
`cpp_decode_dump generate`. The fixture's prompt head, tail and context are
kept; SOURCE names the C++ build in the fixture.

Usage: record_prefill.py CPP_DECODE_DUMP MODEL WORKDIR FIXTURE SOURCE"""
import json, os, re, struct, subprocess, sys

tool, model, work, fixture_path, source = sys.argv[1:6]
fixture = json.load(open(fixture_path))


def fnv(data):
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def prompt(body):
    # As `prompt` in tests/prefill.rs.
    return fixture["head"] + [1000 + (i * 7919 + 13) % 50000 for i in range(body)] + fixture["tail"]


def dump(name, body, mode, n):
    out = os.path.join(work, name)
    os.makedirs(out, exist_ok=True)
    tokens = prompt(body)
    path = os.path.join(out, "prompt.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(tokens)}I", *tokens))
    subprocess.run(["choom", "-n", "1000", "--", tool, model, path, out, mode, str(n), str(fixture["max_context"])],
                   check=True, stdout=subprocess.DEVNULL)
    data = open(os.path.join(out, "ids.bin"), "rb").read()
    return len(tokens), out, list(struct.unpack(f"<{len(data) // 4}I", data))


prefill = []
for case in fixture["prefill"]:
    steps = len(case["ids"]) - 1
    length, out, ids = dump(f"prefill{case['body']}", case["body"], "prefill", steps)
    digests = [fnv(open(os.path.join(out, f"logits_{step}.bin"), "rb").read()) for step in range(steps + 1)]
    prefill.append({"body": case["body"], "prompt_len": length, "ids": ids, "logits_fnv1a64": digests})
    print(f"prefill {length}: {ids}")
generate = []
for case in fixture["generate"]:
    _, _, ids = dump(f"generate{case['body']}", case["body"], "generate", case["max_new_tokens"])
    generate.append({"body": case["body"], "max_new_tokens": case["max_new_tokens"], "ids": ids})
    print(f"generate after {case['body']}: {ids}")
fixture.update(source=source, prefill=prefill, generate=generate)
# Indented like the checked-in fixtures, with lists of numbers or strings on
# one line.
text = json.dumps(fixture, indent=1)
text = re.sub(r"\[\n\s*([^\[\]{}]*?)\n\s*\]", lambda m: "[" + re.sub(r",\n\s*", ", ", m.group(1)) + "]", text)
with open(fixture_path, "w") as f:
    f.write(text + "\n")
