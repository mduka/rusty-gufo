"""Records the forward fixture of gufo-deepseek-v4-flash from the C++
engine (`ds4_ref_dump chat`, `extend` and `batch`): per case the prompt
(text or a cached prefix length), the context, the greedy token IDs and the
FNV-1a 64 digests of every logit row (prefill, then one per decode step);
per batch the same for each session, the steps decoded together; per
snapshot the payload's size and digest after two greedy steps, and the
steps after restoring it.

Per DSpark case (the model loaded with the support model) the emitted
tokens and logits digest of every greedy cycle of each speculative session,
its statistics and snapshot, and for a single session the cycles after
restoring that snapshot.

Usage: record_forward.py DS4_REF_DUMP MODEL WORKDIR FIXTURE [SECTIONS [DSPARK]]
SECTIONS (default cases,batches,snapshots) selects what to record; the
others are kept from an existing FIXTURE. The dspark section needs the
support model DSPARK, and so does dspark_sampled (request samplers lent to
the cycles)."""
import json, os, struct, subprocess, sys

def fnv(data):
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"

tool, model, work, fixture = sys.argv[1:5]
sections = (sys.argv[5] if len(sys.argv) > 5 else "cases,batches,snapshots").split(",")
dspark_model = sys.argv[6] if len(sys.argv) > 6 else None
here = os.path.dirname(os.path.abspath(__file__))
suite = json.load(open(os.path.join(here, "../../../../../tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json")))
corpus = "\n\n".join(c["question"] + "\n" + "\n".join(map(str, c.get("choices") or [])) for c in suite["cases"])
cases = [
    ("short", "What is the capital of France? Answer in one word.", 16, 4096, 0),
    ("p300", corpus[:1200], 8, 4096, 0),
    ("p1500", corpus[:6000], 8, 4096, 0),
    ("p2600", corpus[:10500], 8, 4096, 0),
    ("p5000", corpus[:20000], 8, 8192, 0),
    ("extend3", corpus[:6000], 4, 4096, 3),
    ("extend40", corpus[:6000], 4, 4096, 40),
]
batches = [
    ("batch3", [cases[0][1], corpus[:1200], corpus[:6000]], 6, 4096),
    ("batch8", [corpus[k * 2000:k * 2000 + 300 + 250 * k] for k in range(8)], 4, 4096),
]
snapshots = [
    ("snap_same", corpus[:6000], 4, 4096, 4096),
    ("snap_grow", corpus[:10500], 4, 4096, 8192),
]
dsparks = [
    ("dspark1", [corpus[:1200]], 12, 5, 4, 4096),
    ("dspark1_long", [corpus[:10500]], 8, 5, 0, 4096),
    ("dspark2", [cases[0][1], corpus[:1200]], 8, 5, 0, 4096),
    ("dspark3", [cases[0][1], corpus[:1200], corpus[2000:2600]], 4, 5, 0, 4096),
    ("dspark4", [corpus[k * 2000:k * 2000 + 400 + 200 * k] for k in range(4)], 6, 5, 0, 4096),
]
# Sampled cohorts: "T,P,K,SEED" per session (greedy for none).
sampled = [
    ("sampled1", [corpus[:1200]], 10, 5, "0.8,0.95,0,1234", 4096),
    ("sampled2", [cases[0][1], corpus[:1200]], 8, 5, "0.8,0.95,0,1;0.6,0.9,40,2", 4096),
    ("mixed2", [cases[0][1], corpus[:1200]], 8, 5, "greedy;0.8,0.95,0,7", 4096),
    ("pointmass2", [cases[0][1], corpus[:1200]], 8, 5, "0.7,1,0,3;0.9,1,0,4", 4096),
    ("sampled4", [corpus[k * 2000:k * 2000 + 400 + 200 * k] for k in range(4)], 6, 5,
     "0.8,0.95,0,11;0.8,0.95,0,12;0.8,0.95,0,13;0.8,0.95,0,14", 4096),
]
previous = json.load(open(fixture)) if os.path.exists(fixture) else {}
out = {
    "source": "src/models/deepseek_v4_flash/tests/reference/ds4_ref_dump.cpp against the C++ engine",
    "model": os.path.basename(model),
    "cases": [],
}
out["cases"] = previous.get("cases", []) if "cases" not in sections else []
for name, text, steps, context, extend in cases if "cases" in sections else []:
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    text_path = os.path.join(d, "text.txt")
    open(text_path, "w", encoding="utf-8").write(text)
    if extend == 0:
        args = [tool, model, d, "chat", text_path, str(steps), str(context)]
    else:
        prompt = open(os.path.join(work, "p1500", "prompt.bin"), "rb").read()
        open(os.path.join(d, "prompt.bin"), "wb").write(prompt)
        k = len(prompt) // 4 - extend
        args = [tool, model, d, "extend", os.path.join(d, "prompt.bin"), str(steps), str(k), str(context)]
    subprocess.run(["choom", "-n", "1000", "--"] + args, check=True)
    prompt = open(os.path.join(d, "prompt.bin"), "rb").read()
    ids = list(struct.unpack(f"<{steps + 1}I", open(os.path.join(d, "ids.bin"), "rb").read()))
    logits = [fnv(open(os.path.join(d, f"logits_{s}.bin"), "rb").read()) for s in range(steps + 1)]
    case = {"name": name, "context": context, "steps": steps, "prompt_tokens": len(prompt) // 4,
            "prompt": fnv(prompt), "ids": ids, "logits": logits}
    if extend == 0:
        case["text"] = text
    else:
        case["extend_from"] = "p1500"
        case["cached"] = len(prompt) // 4 - extend
    out["cases"].append(case)
    print(name, len(prompt) // 4, "tokens", ids[:4], flush=True)
out["batches"] = previous.get("batches", []) if "batches" not in sections else []
for name, texts, steps, context in batches if "batches" in sections else []:
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    corpus_path = os.path.join(d, "texts.bin")
    with open(corpus_path, "wb") as f:
        for text in texts:
            data = text.encode("utf-8")
            f.write(struct.pack("<I", len(data)) + data)
    subprocess.run(["choom", "-n", "1000", "--", tool, model, d, "batch", corpus_path, str(steps), str(context)], check=True)
    sessions = []
    for i in range(len(texts)):
        prompt = open(os.path.join(d, f"prompt_{i}.bin"), "rb").read()
        ids = list(struct.unpack(f"<{steps + 1}I", open(os.path.join(d, f"ids_{i}.bin"), "rb").read()))
        logits = [fnv(open(os.path.join(d, f"logits_{i}_{s}.bin"), "rb").read()) for s in range(steps + 1)]
        sessions.append({"prompt_tokens": len(prompt) // 4, "prompt": fnv(prompt), "ids": ids, "logits": logits})
    out["batches"].append({"name": name, "context": context, "steps": steps, "texts": texts, "sessions": sessions})
    print(name, [s["prompt_tokens"] for s in sessions], flush=True)
out["snapshots"] = previous.get("snapshots", []) if "snapshots" not in sections else []
for name, text, steps, context, restore in snapshots if "snapshots" in sections else []:
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    text_path = os.path.join(d, "text.txt")
    open(text_path, "w", encoding="utf-8").write(text)
    subprocess.run(["choom", "-n", "1000", "--", tool, model, d, "snapshot", text_path, str(steps), str(context), str(restore)], check=True)
    prompt = open(os.path.join(d, "prompt.bin"), "rb").read()
    payload = open(os.path.join(d, "payload.bin"), "rb").read()
    ids = list(struct.unpack(f"<{steps + 3}I", open(os.path.join(d, "ids.bin"), "rb").read()))
    logits = [fnv(open(os.path.join(d, f"logits_{s}.bin"), "rb").read()) for s in range(steps + 1)]
    out["snapshots"].append({"name": name, "text": text, "context": context, "restore_context": restore, "steps": steps,
                             "prompt_tokens": len(prompt) // 4, "prompt": fnv(prompt), "payload_bytes": len(payload),
                             "payload": fnv(payload), "ids": ids, "logits": logits})
    print(name, len(prompt) // 4, "tokens", len(payload), "payload bytes", flush=True)
def read_emitted(path):
    data = open(path, "rb").read()
    cycles, pos = [], 0
    while pos < len(data):
        (count,) = struct.unpack_from("<I", data, pos)
        cycles.append(list(struct.unpack_from(f"<{count}I", data, pos + 4)))
        pos += 4 + 4 * count
    return cycles

def record_dspark(name, texts, cycles, max_draft, restore, context, sampling=""):
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    corpus_path = os.path.join(d, "texts.bin")
    with open(corpus_path, "wb") as f:
        for text in texts:
            data = text.encode("utf-8")
            f.write(struct.pack("<I", len(data)) + data)
    subprocess.run(["choom", "-n", "1000", "--", tool, model, d, "dspark", dspark_model, corpus_path, str(cycles),
                    str(max_draft), str(restore), str(context)] + ([sampling] if sampling else []), check=True)
    sessions = []
    for i in range(len(texts)):
        prompt = open(os.path.join(d, f"prompt_{i}.bin"), "rb").read()
        payload = open(os.path.join(d, f"payload_{i}.bin"), "rb").read()
        stats = open(os.path.join(d, f"stats_{i}.bin"), "rb").read()
        sessions.append({
            "prompt_tokens": len(prompt) // 4, "prompt": fnv(prompt),
            "emitted": read_emitted(os.path.join(d, f"emitted_{i}.bin")),
            "logits": [fnv(open(os.path.join(d, f"logits_{i}_{c}.bin"), "rb").read()) for c in range(cycles)],
            "stats": list(struct.unpack("<9Q", stats[:72])) + list(struct.unpack("<I", stats[72:76])),
            "payload_bytes": len(payload), "payload": fnv(payload)})
    case = {"name": name, "context": context, "cycles": cycles, "max_draft": max_draft, "texts": texts,
            "sessions": sessions}
    if sampling:
        case["sampling"] = sampling
    if restore:
        case["restored"] = {
            "emitted": read_emitted(os.path.join(d, "restored_emitted.bin")),
            "logits": [fnv(open(os.path.join(d, f"restored_logits_{c}.bin"), "rb").read()) for c in range(restore)]}
    print(name, [s["prompt_tokens"] for s in sessions], [sum(len(c) for c in s["emitted"]) for s in sessions],
          "emitted", flush=True)
    return case

out["dspark"] = previous.get("dspark", []) if "dspark" not in sections else []
for name, texts, cycles, max_draft, restore, context in dsparks if "dspark" in sections else []:
    out["dspark"].append(record_dspark(name, texts, cycles, max_draft, restore, context))
out["dspark_sampled"] = previous.get("dspark_sampled", []) if "dspark_sampled" not in sections else []
for name, texts, cycles, max_draft, sampling, context in sampled if "dspark_sampled" in sections else []:
    out["dspark_sampled"].append(record_dspark(name, texts, cycles, max_draft, 0, context, sampling))
json.dump(out, open(fixture, "w"), ensure_ascii=False, indent=1)
