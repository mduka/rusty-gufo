"""Qwen control tokens for the Python tools and tests.

The Rust tokenizer (``src/core/src/qwen_tokenizer.rs``) and chat template
(``src/models/qwen/src/chat_template.rs``) spell these strings; keep the values
identical. Validate changes against pinned model metadata and independent
reference fixtures; centralizing these strings does not change tokenization or
EOS handling.
"""

from __future__ import annotations

# The shared Qwen vocabulary, reused by Qwen3.8, Qwen-Image, Qwen3-TTS and
# Qwen3-ASR.
kEndOfText = "<|endoftext|>"
kImStart = "<|im_start|>"
kImEnd = "<|im_end|>"
kObjectRefStart = "<|object_ref_start|>"
kObjectRefEnd = "<|object_ref_end|>"
kBoxStart = "<|box_start|>"
kBoxEnd = "<|box_end|>"
kQuadStart = "<|quad_start|>"
kQuadEnd = "<|quad_end|>"
kVisionStart = "<|vision_start|>"
kVisionEnd = "<|vision_end|>"
kVisionPad = "<|vision_pad|>"
kImagePad = "<|image_pad|>"
kVideoPad = "<|video_pad|>"
