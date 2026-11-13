"""Records the reference outputs the extension's tests compare against.

Runs Hugging Face transformers' Nemotron 3.5 ASR implementation (float32, CPU)
on the test clips in tests/fixtures and writes
tests/fixtures/golden.safetensors (features, encoder activations, decoder and
joint outputs) and tests/fixtures/golden.json (token IDs and transcripts).

Features are the processor's for the whole clip: floor(samples / 160)
frames, the signal pre-emphasized and then padded with zeros. Token IDs come
from offline transcription (the whole clip at once, its last chunk partial),
the semantics the extension implements for uploads and the end of live
audio; encoder activations are recorded for the first chunks fed in
streaming, which never see the end.

    python -I tools/reference/golden.py /models/nemotron-3.5-asr-streaming-0.6b
"""

import json
import pathlib
import sys

import soundfile as sf
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, Nemotron3_5AsrForRNNT
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import Nemotron3_5AsrRNNTDecoderCache

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
CLIPS = {"english": "en-US", "german": "de-DE"}
LOOKAHEADS = (13, 6, 3, 0)  # the checkpoint's supported_num_lookahead_tokens
# Activations recorded per clip: the first chunks at lookahead 13 (the
# widest, which uploads use; the checkpoint's default is 3).
RECORDED_CHUNKS = 3


def offline_features(processor, audio):
    """The processor's log-mel features of the clip: [frames, 128], valid frames only."""
    features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    frames = int(features["attention_mask"][0].sum())
    assert frames == audio.shape[0] // 160
    return features["input_features"][0][:frames]


def first_chunks(features, lookahead, count):
    """The first `count` streaming chunks: 1 + 8 * lookahead frames, then 8 * (lookahead + 1)."""
    first = 1 + 8 * lookahead
    size = 8 * (lookahead + 1)
    out = [features[:first]]
    for start in range(first, first + (count - 1) * size, size):
        out.append(features[start:start + size])
    assert all(piece.shape[0] == (first if i == 0 else size) for i, piece in enumerate(out))
    return out


def clean(text):
    """The transcript without language tags: whitespace runs become one space."""
    return " ".join(text.split())


def main():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    model_dir = sys.argv[1]
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Nemotron3_5AsrForRNNT.from_pretrained(model_dir, dtype=torch.float32).eval()
    blank = model.config.blank_token_id
    prompts = processor.prompt_dictionary

    tensors = {}
    report = {"model_revision": pathlib.Path(model_dir, "REVISION").read_text().strip()
              if pathlib.Path(model_dir, "REVISION").exists() else None,
              "transformers": __import__("transformers").__version__, "clips": {}}

    captured = []
    model.encoder.subsampling.register_forward_hook(lambda m, i, o: captured.append(o.detach().clone()))

    for clip, language in CLIPS.items():
        audio, rate = sf.read(FIXTURES / f"{clip}.wav", dtype="float32")
        assert rate == 16000 and audio.ndim == 1
        features = offline_features(processor, audio)
        tensors[f"{clip}.features"] = features.contiguous()
        entry = {"samples": int(audio.shape[0]), "language": language, "runs": []}

        # Encoder activations chunk by chunk at lookahead 13.
        lookahead = 13
        prompt = torch.tensor([prompts[language]])
        pieces = first_chunks(features, lookahead, RECORDED_CHUNKS)
        captured.clear()
        past = padding = None
        with torch.no_grad():
            for index, piece in enumerate(pieces):
                out = model.get_audio_features(
                    input_features=piece[None], past_key_values=past, padding_cache=padding,
                    use_cache=True, num_lookahead_tokens=lookahead, prompt_ids=prompt,
                    output_attention_mask=False)
                past, padding = out.past_key_values, out.padding_cache
                tensors[f"{clip}.subsampled.{index}"] = captured[-1][0].contiguous()
                tensors[f"{clip}.encoder.{index}"] = out.last_hidden_state[0].contiguous()
                tensors[f"{clip}.projected.{index}"] = out.pooler_output[0].contiguous()

            # Decoder from the blank start state, one token, and the joint.
            cache = Nemotron3_5AsrRNNTDecoderCache(model.config)
            start = model.decoder(torch.tensor([[blank]]), cache=cache)[0, -1].clone()
            tensors[f"{clip}.decoder.start"] = start.contiguous()
            frame0 = tensors[f"{clip}.projected.0"][0]
            logits = model.joint(start[None, None], frame0[None, None])[0, 0]
            tensors[f"{clip}.joint.start"] = logits.contiguous()

        # Token IDs of offline transcription for every lookahead, with the
        # clip's language and auto.
        encoder_frames = int(model.encoder._get_subsampling_output_length(
            torch.tensor([features.shape[0]]))[0])
        for lookahead in LOOKAHEADS:
            for lang in (language, "auto"):
                inputs = processor(audio, language=lang, sampling_rate=16000)
                with torch.no_grad():
                    out = model.generate(
                        input_features=inputs["input_features"], attention_mask=inputs["attention_mask"],
                        prompt_ids=inputs["prompt_ids"], num_lookahead_tokens=lookahead)
                sequence = out.sequences[0].tolist()
                durations = out.durations[0].tolist()
                assert len(durations) == len(sequence)
                frame = 0
                ids, frames = [], []
                for token, advance in zip(sequence[1:], durations[1:]):
                    if token != blank:
                        ids.append(token)
                        frames.append(frame)
                    frame += advance
                raw = processor.decode(ids, skip_special_tokens=False)
                text = processor.decode(ids, skip_special_tokens=True)
                entry["runs"].append({
                    "lookahead": lookahead, "language": lang, "prompt": prompts[lang],
                    "encoder_frames": encoder_frames,
                    "chunks": -(-encoder_frames // (lookahead + 1)),
                    "ids": ids, "frames": frames, "raw": raw, "text": clean(text)})
                print(clip, lang, lookahead, clean(text))
        report["clips"][clip] = entry

    save_file(tensors, str(FIXTURES / "golden.safetensors"))
    (FIXTURES / "golden.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
