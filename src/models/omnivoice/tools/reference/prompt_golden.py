#!/usr/bin/env python3
"""Writes OmniVoice's prompt preprocessing results as test fixtures.

Usage: prompt_golden.py MODEL_DIR OUTPUT_JSON

Runs the reference package's own helpers: language and voice-design
resolution, text combination, punctuation, chunking, the duration estimator
and the token layout of the conditional prompt.
"""
import json
import sys
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from omnivoice.models import omnivoice as ov
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.text import add_punctuation, chunk_text_punctuation

LANGUAGES = ["en", "de", "English", "english", "ENGLISH", "German", "Deutsch",
             "zh", "Chinese", "Mandarin Chinese", "none", "None", "auto", "xx",
             "Ömie", "Fe'Fe'", "yue", "Japanese", "cmn"]

INSTRUCTS = [
    ("female, low pitch, british accent", False),
    ("Male,  Elderly ,whisper", False),
    ("male，young adult", False),
    ("female, low pitch", True),
    ("男，河南话", False),
    ("女，中年，高音调", False),
    ("male, 四川话", False),
    (",female,", False),
    ("female, male", False),
    ("female, british accent, 河南话", False),
    ("femal", False),
    ("very loud", False),
    ("   ", False),
]

TEXTS = [
    "The little brown dog ran across the garden.",
    "Selbstverständlich, Sir. Ich prüfe das sofort und kümmere mich darum.",
    "你好，世界！今天天气很好。",
    "こんにちは、世界。",
    "नमस्ते दुनिया",
    "مَرْحَبًا بِالْعَالَم",
    "Hello 🌍! This is fun 🎉 2026",
    "Chào thế giới",
    "Hi.",
    "  Line one\nline two\r\n with  double   spaces\tand tab  ",
    "混合 中文 and English 文字",
    "（括号）text",
    "[laughter] That's funny [sigh] really [question-ah]",
    "Mr. Smith met Dr. Jones at 5 p.m. on Main St. It was late! Wasn't it? Yes; indeed: quite.",
]

CHUNK_TEXTS = [
    ("First sentence. Second one! Third? Fourth; fifth: sixth, seventh.", 20),
    ("Mr. Smith met Dr. Jones. They talked for hours about nothing at all. Then they left.", 30),
    ("A. B. C. Long sentence follows here with many words in it.", 10),
    ("你好。今天天气很好！我们去公园吧？好的，走吧。", 6),
    ("No punctuation at all in this rather long text that goes on and on", 10),
    ("Quote: \"Hello.\" Then (aside.) more text here.", 12),
    ("..., leading punctuation. Then text.", 8),
]

PUNCTUATION = ["Hello", "Hello.", "Hello …", "你好", "你好。", "  spaced  ", "",
               "Ends with quote\"", "Ends with 、"]


def main():
    model_dir, output = sys.argv[1], sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    estimator = RuleDurationEstimator()
    fake = SimpleNamespace(
        text_tokenizer=tokenizer,
        config=SimpleNamespace(num_audio_codebook=8, audio_mask_id=1024),
        device=torch.device("cpu"))

    languages = [{"input": lang, "id": ov._resolve_language(lang)}
                 for lang in LANGUAGES]

    instructs = []
    for text, use_zh in INSTRUCTS:
        try:
            result = ov._resolve_instruct(text, use_zh=use_zh)
            instructs.append({"input": text, "use_zh": use_zh, "output": result})
        except ValueError:
            instructs.append({"input": text, "use_zh": use_zh, "error": True})

    weights = [{"text": text, "weight": estimator.calculate_total_weight(text)}
               for text in TEXTS]

    durations = []
    for text in TEXTS:
        for ref_text, ref_frames, speed in [(None, None, 1.0),
                                            ("Nice to meet you.", 25, 1.0),
                                            ("Hello, world. This is a reference.", 81, 1.0),
                                            ("Hello, world. This is a reference.", 81, 1.3),
                                            ("你好，这是参考。", 60, 0.8)]:
            frames = ov.OmniVoice._estimate_target_tokens(
                SimpleNamespace(duration_estimator=estimator), text, ref_text,
                ref_frames, speed)
            durations.append({"text": text, "ref_text": ref_text,
                              "ref_frames": ref_frames, "speed": speed,
                              "frames": frames})

    combined = []
    for text in TEXTS:
        for ref_text in [None, "  Reference words.\n", "参考文本。"]:
            combined.append({"text": text, "ref_text": ref_text,
                             "combined": ov._combine_text(text, ref_text=ref_text)})

    chunks = [{"text": text, "chunk_len": length,
               "chunks": chunk_text_punctuation(text, chunk_len=length, min_chunk_len=3)}
              for text, length in CHUNK_TEXTS]

    punctuation = [{"input": text, "output": add_punctuation(text)}
                   for text in PUNCTUATION]

    prompts = []
    cases = [
        ("Hello there.", None, None, None, False, 0),
        ("Hello there.", None, "en", "female, low pitch", False, 0),
        ("Selbstverständlich, Sir.", "Ich prüfe das.", "de", None, True, 37),
        ("Selbstverständlich, Sir.", "Ich prüfe das.", "de", None, False, 37),
        ("[laughter] That's funny.", None, "en", None, True, 0),
        ("你好，世界！", "参考文本。", "zh", "女，中年", True, 12),
    ]
    for text, ref_text, lang, instruct, denoise, ref_frames in cases:
        ref_tokens = (torch.arange(8 * ref_frames).view(8, ref_frames) % 1024
                      if ref_frames else None)
        inputs = ov.OmniVoice._prepare_inference_inputs(
            fake, text, 5, ref_text=ref_text, ref_audio_tokens=ref_tokens,
            lang=lang, instruct=instruct, denoise=denoise)
        row = inputs["input_ids"][0, 0].tolist()
        audio_start = int(inputs["audio_mask"][0].int().argmax())
        prompts.append({"text": text, "ref_text": ref_text, "lang": lang,
                        "instruct": instruct, "denoise": denoise,
                        "ref_frames": ref_frames, "target_frames": 5,
                        "text_ids": row[:audio_start], "audio_start": audio_start,
                        "length": len(row)})

    with open(output, "w", encoding="utf-8") as stream:
        json.dump({"source": "k2-fsa/OmniVoice reference helpers",
                   "languages": languages, "instructs": instructs,
                   "weights": weights, "durations": durations,
                   "combined": combined, "chunks": chunks,
                   "punctuation": punctuation, "prompts": prompts},
                  stream, ensure_ascii=False, indent=1)
        stream.write("\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
