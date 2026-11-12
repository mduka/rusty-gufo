#!/usr/bin/env python3
"""Writes token IDs from OmniVoice's Hugging Face tokenizer as test fixtures.

Usage: tokenizer_golden.py MODEL_DIR OUTPUT_JSON

Each case is encoded the way OmniVoice encodes prompt pieces:
tokenizer(text, add_special_tokens=False).
"""
import json
import sys

from transformers import AutoTokenizer

CASES = [
    "Hello, world!",
    "The little brown dog ran across the garden.",
    "I'm sure they'll say it's fine, aren't they? I'D SAY SO.",
    "Größenmaßstäbe für Übergrößen: äöü ÄÖÜ ß.",
    "Selbstverständlich, Sir. Ich prüfe das sofort.",
    "Numbers 1234567 and 3.14159, 2026-11-09, 1,000,000.",
    "Spaces   between\twords  and  trailing   ",
    "Line one\nline two\r\n\n  indented line\n\n\n",
    "   leading spaces",
    "Unit separators\x1c\x1d\x1e\x1fend",
    "你好，世界！今天天气很好。",
    "こんにちは、世界。カタカナとひらがな。",
    "안녕하세요 세계",
    "नमस्ते दुनिया, यह एक परीक्षण है।",
    "আমি বাংলায় কথা বলি।",
    "مرحبا بالعالم",
    "Привет, мир! Ёжик.",
    "Ελληνικά γράμματα",
    "é decomposed versus é composed",
    "Emoji 😀👍🏽 and symbols ©®™ €£¥",
    "Café naïve résumé coöperate",
    "Tabs\tand no-break em space",
    "<|text_start|>Hello there.<|text_end|>",
    "<|denoise|><|lang_start|>en<|lang_end|><|instruct_start|>None<|instruct_end|>",
    "<|lang_start|>None<|lang_end|><|instruct_start|>female, low pitch, british accent<|instruct_end|>",
    "[laughter] That's funny [sigh]",
    "Mixed 中文 and English 文字 together",
    "Ending with punctuation...!?",
    "'s at the start and 'll later",
    "a" * 40,
    "",
]


def main():
    model_dir, output = sys.argv[1], sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    cases = []
    for text in CASES:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        cases.append({"text": text, "ids": ids})
    with open(output, "w", encoding="utf-8") as stream:
        json.dump({"source": "k2-fsa/OmniVoice tokenizer.json via transformers",
                   "cases": cases}, stream, ensure_ascii=False, indent=1)
        stream.write("\n")
    print(f"{len(cases)} cases -> {output}")


if __name__ == "__main__":
    main()
