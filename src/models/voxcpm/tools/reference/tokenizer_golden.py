"""Token IDs of VoxCPM2's text tokenizer for tests/fixtures/tokenizer_golden.json.

usage: tokenizer_golden.py VOXCPM_SRC MODEL_DIR OUT_JSON

VOXCPM_SRC is the `src` directory of an OpenBMB/VoxCPM checkout, MODEL_DIR the
openbmb/VoxCPM2 directory. VoxCPM2Model tokenizes with LlamaTokenizerFast
wrapped by `mask_multichar_chinese_tokens`, which splits multi-character
Chinese tokens into single characters. The IDs here apply that wrapper's split
to the tokens of tokenizer.json (the `tokenizers` library), which is what
LlamaTokenizerFast produces under transformers 4. transformers 5 replaces the
file's normalizer with a Metaspace pre-tokenizer that adds no extra "▁"
before text that already starts with a space; the script checks that this is
the only difference.
"""

import json
import sys

sys.path.insert(0, sys.argv[1])

from tokenizers import Tokenizer  # noqa: E402
from transformers import LlamaTokenizerFast  # noqa: E402

from voxcpm.model.utils import mask_multichar_chinese_tokens  # noqa: E402

TEXTS = [
    "",
    "Hello, world!",
    "VoxCPM2 brings multilingual support, creative voice design, and "
    "controllable voice cloning.",
    "(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!",
    "Grüße aus München: Straße, Äpfel, Öl und Übermut.",
    "  leading and trailing spaces  ",
    "two  spaces and\ttab\nnew line\n\nblank line",
    "你好世界，今天天气很好。",
    "我们一起去北京大学吧！",
    "日本語のテキストも読めます。",
    "한국어 문장입니다.",
    "Привет, как дела?",
    "مرحبا بالعالم",
    "สวัสดีครับ",
    "Emoji 🙂🚀 and symbols ∑∫√ ©®",
    "Numbers 3.14159, 1,000,000 and 2026-11-11.",
    "Mixed 中文 and English 混合 text.",
    "<|im_end|> inside text <s> and </s>",
    " non-breaking thin　ideographic spaces",
    "a" * 40,
    "𠀀𪚥 rare CJK extension B",
]


def main():
    src, model_dir, out = sys.argv[1:4]
    del src
    base = LlamaTokenizerFast.from_pretrained(model_dir)
    wrapped = mask_multichar_chinese_tokens(base)
    raw = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    added = [a.content for a in raw.get_added_tokens_decoder().values()]
    cases = []
    for text in TEXTS:
        encoding = raw.encode(text, add_special_tokens=False)
        ids = []
        for token, token_id in zip(encoding.tokens, encoding.ids):
            clean = token.replace("\u2581", "")
            if clean in wrapped.multichar_tokens:
                ids.extend(raw.token_to_id(c) for c in clean)
            else:
                ids.append(token_id)
        installed = list(wrapped(text))
        # transformers 5 also normalizes the text between added tokens
        # differently; the file's behavior is the one kept here.
        special = any(t in text for t in added)
        if ids != installed and not text[:1].isspace() and not special:
            raise SystemExit(f"the installed tokenizer disagrees on {text!r}: "
                             f"{ids} vs {installed}")
        cases.append({"text": text, "ids": ids})
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": "openbmb/VoxCPM2 tokenizer.json split by "
                             "voxcpm.model.utils.mask_multichar_chinese_tokens",
                   "cases": cases}, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
