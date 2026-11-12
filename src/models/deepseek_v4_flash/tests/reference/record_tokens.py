"""Turns `ds4_ref_dump tokens` output into the tokenizer fixture of
gufo-deepseek-v4-flash: per text the FNV-1a 64 of its token IDs (u32 LE)
for Tokenize, Tokenize with special tokens and EncodeChat("", text), and
one digest of DecodeToken over the vocabulary.

Usage: record_tokens.py OUTDIR CORPUS FIXTURE (see ds4_ref_dump.cpp)."""
import json, struct, sys

def fnv(data):
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"

def streams(path):
    data = open(path, 'rb').read()
    pos, out = 0, []
    while pos < len(data):
        (n,) = struct.unpack_from('<I', data, pos)
        pos += 4
        out.append((n, data[pos:pos + 4 * n]))
        pos += 4 * n
    return out

src = sys.argv[1]
corpus = open(sys.argv[2], 'rb').read()
texts, pos = [], 0
while pos < len(corpus):
    (n,) = struct.unpack_from('<I', corpus, pos)
    pos += 4
    texts.append(corpus[pos:pos + n])
    pos += n
text, rendered, chat = (streams(f"{src}/{m}.bin") for m in ("text", "rendered", "chat"))
assert len(text) == len(texts) == len(rendered) == len(chat)
cases = []
for t, a, b, c in zip(texts, text, rendered, chat):
    case = {}
    try:
        case["text"] = t.decode('utf-8')
    except UnicodeDecodeError:
        case["hex"] = t.hex()
    case.update({"tokens": [a[0], fnv(a[1])], "rendered": [b[0], fnv(b[1])], "chat": [c[0], fnv(c[1])]})
    cases.append(case)
fixture = {
    "source": "src/models/deepseek_v4_flash/tests/reference/ds4_ref_dump.cpp tokens against the C++ engine",
    "model": "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
    "token_text": fnv(open(f"{src}/token_text.bin", 'rb').read()),
    "cases": cases,
}
json.dump(fixture, open(sys.argv[3], 'w'), ensure_ascii=False, indent=0)
print(len(cases), "cases")
