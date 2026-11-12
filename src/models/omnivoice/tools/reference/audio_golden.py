#!/usr/bin/env python3
"""Writes OmniVoice's waveform helpers' results as test fixtures.

Usage: audio_golden.py OUTPUT_JSON

Inputs come from an integer generator that tests/audio_test.cpp reproduces
exactly; the fixture holds lengths, checksums and strided samples of the
reference helpers' outputs (pydub silence removal, torchaudio resampling,
fades, cross-fades, reference preparation and post-processing).
"""
import json
import sys

import numpy as np
import torch
import torchaudio

from omnivoice.utils.audio import (cross_fade_chunks, fade_and_pad_audio,
                                   remove_silence)

STRIDE = 97


def generate(spec, rate, seed):
    """Segments of (milliseconds, kind, amplitude): 'noise' or 'tone'."""
    state = seed
    out = []
    for ms, kind, amp in spec:
        for n in range(ms * rate // 1000):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            noise = (state >> 8) % (2 * min(amp, 64) + 1) - min(amp, 64)
            if kind == "tone":
                period = 55
                phase = n % period
                value = (4 * amp * abs(2 * phase - period)) // (2 * period) - amp + noise
            else:
                value = (state >> 8) % (2 * amp + 1) - amp
            out.append(max(-32768, min(32767, value)))
    return np.array(out, dtype=np.int16).astype(np.float32) / 32768.0


def summary(audio):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return {"length": int(audio.shape[0]),
            "samples": [float(x) for x in audio[::STRIDE]]}


def pcm_summary(audio):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = np.round(audio * 32768.0).astype(np.int64)
    return {"length": int(audio.shape[0]), "sum": int(pcm.sum()),
            "sum_squares": int((pcm * pcm).sum()),
            "samples": [int(x) for x in pcm[::STRIDE]]}


SPEECH = [[300, "noise", 20], [800, "tone", 8000], [700, "noise", 20],
          [450, "tone", 3000], [600, "noise", 50], [300, "tone", 12000],
          [250, "noise", 20]]
QUIET = [[150, "noise", 5], [900, "tone", 1500], [550, "noise", 5],
         [400, "tone", 900], [120, "noise", 5]]


def main():
    output = sys.argv[1]
    cases = {"silence": [], "fade": [], "crossfade": [], "resample": [],
             "reference": [], "postprocess": []}
    signals = [
        {"spec": SPEECH, "rate": 24000, "seed": 1},
        {"spec": QUIET, "rate": 24000, "seed": 2},
        {"spec": [[5, "tone", 4000]], "rate": 24000, "seed": 3},
        {"spec": [[1200, "noise", 10]], "rate": 24000, "seed": 4},
        # 1000.5 ms: len() rounds half to even.
        {"spec": [[400, "noise", 10], [600, "tone", 6000]], "rate": 24000,
         "seed": 5, "extra": 12},
    ]
    for signal in signals:
        audio = generate(signal["spec"], signal["rate"], signal["seed"])
        if signal.get("extra"):
            audio = np.concatenate([audio, np.full(signal["extra"], 0.25, np.float32)])
        for mid, lead, trail in [(200, 100, 200), (500, 100, 100), (0, 100, 300)]:
            result = remove_silence(audio[np.newaxis, :], 24000, mid, lead, trail)
            cases["silence"].append({"signal": signal, "params": [mid, lead, trail],
                                     **pcm_summary(result)})
        cases["fade"].append({"signal": signal,
                              **summary(fade_and_pad_audio(audio[np.newaxis, :]))})
        for rms in [None, 0.05, 0.2]:
            processed = remove_silence(audio[np.newaxis, :], 24000, 500, 100, 100)
            if rms is not None and rms < 0.1:
                processed = processed * rms / 0.1
            elif rms is None and processed.size > 0:
                peak = np.abs(processed).max()
                if peak > 1e-6:
                    processed = processed / peak * 0.5
            processed = fade_and_pad_audio(processed, sample_rate=24000)
            cases["postprocess"].append({"signal": signal, "rms": rms,
                                         **summary(processed)})

    chunks = [generate(SPEECH[:3], 24000, 7), generate([[40, "tone", 500]], 24000, 8),
              generate(QUIET, 24000, 9)]
    merged = cross_fade_chunks([c[np.newaxis, :] for c in chunks], 24000)
    cases["crossfade"].append({"signals": [
        {"spec": SPEECH[:3], "rate": 24000, "seed": 7},
        {"spec": [[40, "tone", 500]], "rate": 24000, "seed": 8},
        {"spec": QUIET, "rate": 24000, "seed": 9}], **summary(merged)})

    for source, target in [(44100, 24000), (48000, 24000), (16000, 24000),
                           (22050, 24000), (8000, 24000), (24000, 16000)]:
        signal = {"spec": [[300, "noise", 200], [700, "tone", 9000]],
                  "rate": source, "seed": 11}
        audio = torch.from_numpy(generate(signal["spec"], source, signal["seed"]))
        result = torchaudio.functional.resample(audio[None, :], source, target)
        cases["resample"].append({"signal": signal, "target": target,
                                  **summary(result.numpy())})

    for channels, rate, spec in [(2, 44100, QUIET), (1, 24000, SPEECH), (2, 48000, SPEECH)]:
        left = generate(spec, rate, 21)
        right = generate(spec, rate, 22)
        waveform = np.stack([left, right])[:channels]
        if waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0, keepdims=True)
        if rate != 24000:
            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform), orig_freq=rate, new_freq=24000).numpy()
        ref_rms = float(np.sqrt(np.mean(waveform ** 2)))
        if 0 < ref_rms < 0.1:
            waveform = waveform * 0.1 / ref_rms
        waveform = remove_silence(waveform, 24000, mid_sil=200, lead_sil=100, trail_sil=200)
        clip = int(waveform.shape[-1] % 960)
        waveform = waveform[:, :-clip] if clip > 0 else waveform
        cases["reference"].append({"channels": channels, "rate": rate, "spec": spec,
                                   "seeds": [21, 22], "rms": ref_rms,
                                   **pcm_summary(waveform)})

    with open(output, "w", encoding="utf-8") as stream:
        json.dump({"source": "k2-fsa/OmniVoice omnivoice.utils.audio, pydub, torchaudio",
                   "stride": STRIDE, **cases}, stream)
        stream.write("\n")
    print(f"wrote {output}: " + ", ".join(f"{k} {len(v)}" for k, v in cases.items()))


if __name__ == "__main__":
    main()
