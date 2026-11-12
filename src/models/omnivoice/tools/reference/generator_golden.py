#!/usr/bin/env python3
"""Writes OmniVoice reference results for the GPU runtime's tests.

Usage: generator_golden.py MODEL_DIR OUTPUT_JSON

Runs the reference model in float32 on the CPU:
  * forward: one CFG batch (conditional + unconditional) whose target is
    partly unmasked; per target frame and codebook the logits' argmax, max
    and log-sum-exp, plus the scoring step's predictions and confidences;
  * generate: deterministic generations (position and class temperature 0)
    with the step at which each code was unmasked;
  * decode: the audio tokenizer's waveform for the first generation;
  * encode: the audio tokenizer's codes for a synthetic 24 kHz signal.
"""
import json
import math
import sys

import numpy as np
import torch

from omnivoice.models.omnivoice import (OmniVoice, OmniVoiceGenerationConfig,
                                        _get_time_steps)

STRIDE = 97
C = 8
MASK = 1024


def lcg_codes(frames, seed):
    state = seed
    codes = torch.empty(C, frames, dtype=torch.long)
    for f in range(frames):
        for c in range(C):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            codes[c, f] = (state >> 8) % 1024
    return codes


def batch_inputs(cond_ids, cond_audio_mask, target):
    """_generate_iterative's CFG batch for one item."""
    c_len = cond_ids.shape[-1]
    ids = torch.full((2, C, c_len), MASK, dtype=torch.long)
    mask = torch.zeros((2, c_len), dtype=torch.bool)
    attention = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool)
    ids[0] = cond_ids
    mask[0] = cond_audio_mask
    attention[0, :, :c_len, :c_len] = True
    ids[1, :, :target] = cond_ids[:, -target:]
    mask[1, :target] = cond_audio_mask[-target:]
    attention[1, :, :target, :target] = True
    pad = torch.arange(target, c_len)
    attention[1, :, pad, pad] = True
    return ids, mask, attention


def frame_stats(logits):
    """[C, T, V] -> per (c, t): argmax, max, logsumexp."""
    return {"argmax": logits.argmax(-1).tolist(),
            "max": logits.max(-1).values.tolist(),
            "logsumexp": torch.logsumexp(logits, -1).tolist()}


def forward_case(model, config):
    reference = lcg_codes(30, 5)
    target = 40
    inputs = model._prepare_inference_inputs(
        "Ich prüfe das sofort.", target, ref_text="Selbstverständlich, Sir.",
        ref_audio_tokens=reference, lang="de", instruct=None, denoise=True)
    cond_ids = inputs["input_ids"][0]
    cond_mask = inputs["audio_mask"][0]
    fill = lcg_codes(target, 9)
    for c in range(C):
        for f in range(target):
            if (7 * f + 3 * c) % 5 == 0:
                cond_ids[c, -target + f] = fill[c, f]
    ids, mask, attention = batch_inputs(cond_ids, cond_mask, target)
    with torch.no_grad():
        logits = model(input_ids=ids, audio_mask=mask,
                       attention_mask=attention).logits.float()
    c_len = cond_ids.shape[-1]
    c_logits = logits[0, :, c_len - target:c_len]
    u_logits = logits[1, :, :target]
    pred, score = model._predict_tokens_with_scoring(
        c_logits[None], u_logits[None], config)
    return {"input_ids": cond_ids.tolist(),
            "audio_start": int(cond_mask.int().argmax()),
            "target": target,
            "cond": frame_stats(c_logits), "uncond": frame_stats(u_logits),
            "pred": pred[0].tolist(), "confidence": score[0].tolist()}


def generate_case(model, config, text, ref_text, reference, lang, instruct):
    """_generate_iterative for one item, recording each code's unmask step."""
    target = model._estimate_target_tokens(
        text, ref_text, reference.shape[-1] if reference is not None else None)
    inputs = model._prepare_inference_inputs(
        text, target, ref_text=ref_text, ref_audio_tokens=reference, lang=lang,
        instruct=instruct, denoise=config.denoise)
    cond_ids = inputs["input_ids"][0]
    cond_mask = inputs["audio_mask"][0]
    ids, mask, attention = batch_inputs(cond_ids.clone(), cond_mask, target)
    c_len = cond_ids.shape[-1]
    tokens = torch.full((C, target), MASK, dtype=torch.long)
    unmasked_at = torch.full((C, target), -1, dtype=torch.long)
    steps = _get_time_steps(0.0, 1.0, config.num_step, config.t_shift).tolist()
    total = target * C
    remaining = total
    schedule = []
    for step in range(config.num_step):
        count = remaining if step == config.num_step - 1 else min(
            math.ceil(total * (steps[step + 1] - steps[step])), remaining)
        schedule.append(int(count))
        remaining -= int(count)
    layers = torch.arange(C).view(1, -1, 1)
    for step in range(config.num_step):
        k = schedule[step]
        if k <= 0:
            continue
        with torch.no_grad():
            logits = model(input_ids=ids, audio_mask=mask,
                           attention_mask=attention).logits.float()
        pred, scores = model._predict_tokens_with_scoring(
            logits[0:1, :, c_len - target:c_len], logits[1:2, :, :target], config)
        scores = scores - layers * config.layer_penalty_factor
        scores.masked_fill_(tokens[None] != MASK, -float("inf"))
        _, chosen = torch.topk(scores.flatten(), k)
        flat = tokens.flatten()
        flat[chosen] = pred.flatten()[chosen]
        steps_flat = unmasked_at.flatten()
        steps_flat[chosen] = step
        tokens = flat.view(C, target)
        unmasked_at = steps_flat.view(C, target)
        ids[0, :, c_len - target:c_len] = tokens
        ids[1, :, :target] = tokens
    return {"text": text, "ref_text": ref_text, "lang": lang, "instruct": instruct,
            "reference_codes": reference.tolist() if reference is not None else None,
            "input_ids": cond_ids.tolist(), "audio_start": int(cond_mask.int().argmax()),
            "target": target, "schedule": schedule, "tokens": tokens.tolist(),
            "unmasked_at": unmasked_at.tolist()}


def signal(seconds, seed):
    state = seed
    out = np.zeros(int(seconds * 24000), dtype=np.float32)
    for n in range(out.shape[0]):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 3 * n / 24000)
        tone = math.sin(2 * math.pi * 180 * n / 24000) + 0.4 * math.sin(2 * math.pi * 720 * n / 24000)
        noise = ((state >> 8) % 2001 - 1000) / 1000.0
        out[n] = np.float32(0.2 * envelope * tone + 0.02 * noise)
    return out


def stage(tensor):
    """Summary of an intermediate [channels, frames] or [frames, channels]."""
    flat = tensor.reshape(-1).double()
    return {"shape": list(tensor.shape), "rms": float(flat.pow(2).mean().sqrt()),
            "samples": [float(x) for x in flat[::STRIDE]]}


def encode_case(tokenizer, wave):
    """HiggsAudioV2TokenizerModel.encode with its intermediate stages."""
    stages = {}
    with torch.no_grad():
        values = torch.from_numpy(wave)[None, None]
        features = tokenizer._extract_semantic_features(values)
        stages["semantic_features"] = stage(features[0])
        semantic = tokenizer.encoder_semantic(features.transpose(1, 2))
        stages["semantic"] = stage(semantic[0])
        acoustic = tokenizer.acoustic_encoder(values)
        stages["acoustic"] = stage(acoustic[0])
        embeddings = torch.cat([acoustic, semantic], dim=1)
        embeddings = tokenizer.fc(embeddings.transpose(1, 2)).transpose(1, 2)
        stages["embeddings"] = stage(embeddings[0])
        codes = tokenizer.encode(values).audio_codes[0]
    return {"signal": {"seconds": 3.0, "seed": 13}, "codes": codes.tolist(),
            "stages": stages}


def main():
    model_dir, output = sys.argv[1], sys.argv[2]
    torch.manual_seed(0)
    model = OmniVoice.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    config = OmniVoiceGenerationConfig(position_temperature=0.0, class_temperature=0.0)
    result = {"source": "k2-fsa/OmniVoice reference model, float32 on CPU",
              "stride": STRIDE, "config": {
                  "num_step": config.num_step, "guidance_scale": config.guidance_scale,
                  "t_shift": config.t_shift, "layer_penalty_factor": config.layer_penalty_factor}}
    result["forward"] = forward_case(model, config)
    print("forward done", flush=True)
    result["generate"] = [
        generate_case(model, config, "The little brown dog ran across the garden.",
                      None, None, "en", None),
        generate_case(model, config, "Nice to meet you, how are you today?",
                      "Hello, this is my reference voice.", lcg_codes(40, 3), "en", None),
        generate_case(model, config, "Guten Morgen, wie geht es Ihnen?",
                      None, None, "de", "female, low pitch"),
    ]
    print("generate done", flush=True)
    tokens = torch.tensor(result["generate"][0]["tokens"])
    with torch.no_grad():
        audio = model.audio_tokenizer.decode(tokens[None]).audio_values.reshape(-1).float().numpy()
    result["decode"] = {"tokens_from": 0, "length": int(audio.shape[0]),
                        "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
                        "samples": [float(x) for x in audio[::STRIDE]]}
    wave = signal(3.0, 13)
    result["encode"] = encode_case(model.audio_tokenizer, wave)
    print("codec done", flush=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream)
        stream.write("\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
