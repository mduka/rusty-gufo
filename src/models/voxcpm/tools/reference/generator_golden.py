"""Reference tensors of VoxCPM2's generator and AudioVAE for the GPU tests.

usage: generator_golden.py VOXCPM_SRC MODEL_DIR OUT_SAFETENSORS

Runs OpenBMB's VoxCPM2Model on the CPU in float32 (the bfloat16 weights
widened) and records the intermediate tensors of
  - "zs": zero-shot generation of three patches for a short text,
  - "ref": a reference-mode prompt (random reference latents) and its first
    patch,
  - one call of the local DiT with its CFG batch,
  - the AudioVAE decoding of the zero-shot latents and the encoding of a
    synthetic 16 kHz signal.
The diffusion noise comes from a seeded generator and is recorded, so a test
can feed the same noise. The sampler is the reference's: UnifiedCFM.forward
is reproduced here only to take the noise from that generator.
"""

import sys
import types

import torch

sys.path.insert(0, sys.argv[1])

from safetensors.torch import save_file  # noqa: E402

from voxcpm.model.voxcpm2 import VoxCPM2Model  # noqa: E402

PATCHES = 3
TIMESTEPS = 10
CFG = 2.0


def load(model_dir):
    model = VoxCPM2Model.from_local(model_dir, optimize=False, device="cpu")
    model = model.to(torch.float32)
    model.config.dtype = "float32"
    for lm in (model.base_lm, model.residual_lm):
        lm.setup_cache(1, model.config.max_length, "cpu", torch.float32)
    return model.eval()


class Recorder:
    """Collects tensors under names; repeated names get a step suffix."""

    def __init__(self):
        self.tensors = {}

    def put(self, name, tensor):
        self.tensors[name] = tensor.detach().to(torch.float32).clone().contiguous()

    def wrap(self, module, attribute, name, capture):
        original = getattr(module, attribute)
        calls = []

        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            capture(self, f"{name}.{len(calls)}", args, kwargs, result)
            calls.append(None)
            return result

        setattr(module, attribute, wrapper)
        return lambda: setattr(module, attribute, original)


def seeded_cfm(model, generator, recorder, prefix):
    cfm = model.feat_decoder
    calls = []

    def forward(self, mu, n_timesteps, patch_size, cond, temperature=1.0,
                cfg_value=1.0, sway_sampling_coef=1.0, use_cfg_zero_star=True):
        b, _ = mu.shape
        z = torch.randn((b, self.in_channels, patch_size), generator=generator,
                        dtype=mu.dtype) * temperature
        step = len(calls)
        calls.append(None)
        recorder.put(f"{prefix}.noise.{step}", z[0])
        recorder.put(f"{prefix}.mu.{step}", mu[0])
        recorder.put(f"{prefix}.cond.{step}", cond[0])
        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device,
                                dtype=mu.dtype)
        t_span = t_span + sway_sampling_coef * (
            torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        result = self.solve_euler(x=z, t_span=t_span, mu=mu, cond=cond,
                                  cfg_value=cfg_value,
                                  use_cfg_zero_star=use_cfg_zero_star)
        recorder.put(f"{prefix}.feat.{step}", result[0])
        return result

    original = cfm.forward
    cfm.forward = types.MethodType(forward, cfm)
    return lambda: setattr(cfm, "forward", original)


def run(model, recorder, prefix, text_token, text_mask, feat, feat_mask,
        patches, seed):
    generator = torch.Generator().manual_seed(seed)
    restores = [seeded_cfm(model, generator, recorder, prefix)]

    def first_output(rec, name, args, kwargs, result):
        rec.put(name, result[0][0] if isinstance(result, tuple) else result[0])

    restores.append(recorder.wrap(model.base_lm, "forward", f"{prefix}.lm",
                                  first_output))
    restores.append(recorder.wrap(model.residual_lm, "forward",
                                  f"{prefix}.residual", first_output))
    restores.append(recorder.wrap(model.base_lm, "forward_step",
                                  f"{prefix}.lm_step", first_output))
    restores.append(recorder.wrap(model.residual_lm, "forward_step",
                                  f"{prefix}.residual_step", first_output))
    restores.append(recorder.wrap(model.stop_head, "forward",
                                  f"{prefix}.stop", first_output))
    try:
        feat_pred, generated, _ = next(model._inference(
            text_token, text_mask, feat, feat_mask, min_len=patches + 1,
            max_len=patches, inference_timesteps=TIMESTEPS, cfg_value=CFG,
            streaming=False))
    finally:
        for restore in reversed(restores):
            restore()
    recorder.put(f"{prefix}.text_token", text_token[0].to(torch.float32))
    recorder.put(f"{prefix}.text_mask", text_mask[0].to(torch.float32))
    recorder.put(f"{prefix}.feat_mask", feat_mask[0].to(torch.float32))
    recorder.put(f"{prefix}.feat_in", feat[0])
    recorder.put(f"{prefix}.latents", feat_pred[0])
    return feat_pred


def zero_shot_inputs(model, text):
    tokens = torch.LongTensor(model.text_tokenizer(text))
    tokens = torch.cat([tokens, torch.tensor([model.audio_start_token])])
    length = tokens.shape[0]
    feat = torch.zeros((length, model.patch_size, model.audio_vae.latent_dim))
    return (tokens.unsqueeze(0), torch.ones(1, length, dtype=torch.int32),
            feat.unsqueeze(0), torch.zeros(1, length, dtype=torch.int32))


def reference_inputs(model, text, ref_feat):
    tokens = torch.LongTensor(model.text_tokenizer(text))
    tokens = torch.cat([tokens, torch.tensor([model.audio_start_token])])
    length = tokens.shape[0]
    ref_tokens, ref_feats, ref_t, ref_a = model._make_ref_prefix(ref_feat,
                                                                 "cpu")
    pad = torch.zeros((length, model.patch_size, model.audio_vae.latent_dim))
    return (torch.cat([ref_tokens, tokens]).unsqueeze(0),
            torch.cat([ref_t, torch.ones(length, dtype=torch.int32)])
            .unsqueeze(0),
            torch.cat([ref_feats, pad]).unsqueeze(0),
            torch.cat([ref_a, torch.zeros(length, dtype=torch.int32)])
            .unsqueeze(0))


@torch.inference_mode()
def main():
    src, model_dir, out = sys.argv[1:4]
    del src
    torch.manual_seed(0)
    model = load(model_dir)
    recorder = Recorder()

    latents = run(model, recorder, "zs",
                  *zero_shot_inputs(model, "Hello, world."), PATCHES, seed=7)

    ref_feat = torch.randn((3, model.patch_size, model.audio_vae.latent_dim),
                           generator=torch.Generator().manual_seed(11))
    run(model, recorder, "ref",
        *reference_inputs(model, "Good morning.", ref_feat), 1, seed=13)

    # One estimator call with a CFG batch, as solve_euler builds it.
    generator = torch.Generator().manual_seed(17)
    dit = model.feat_decoder.estimator
    x = torch.randn((2, 64, model.patch_size), generator=generator)
    x[1] = x[0]
    mu = torch.randn((2, 2 * 1024), generator=generator)
    mu[1] = 0
    cond = torch.randn((2, 64, model.patch_size), generator=generator)
    cond[1] = cond[0]
    t = torch.tensor([0.63, 0.63])
    dt = torch.zeros(2)
    recorder.put("dit.x", x)
    recorder.put("dit.mu", mu)
    recorder.put("dit.cond", cond)
    recorder.put("dit.t", t)
    recorder.put("dit.out", dit(x, mu, t, cond, dt))

    vae = model.audio_vae
    recorder.put("vae.decoded", vae.decode(latents.to(torch.float32))[0, 0])
    with vae.streaming_decode() as decoder:
        pieces = [decoder.decode_chunk(latents[..., i:i + model.patch_size])
                  for i in range(0, latents.shape[-1], model.patch_size)]
    recorder.put("vae.streamed", torch.cat(pieces, dim=-1)[0, 0])
    time = torch.arange(2 * model.patch_size * vae.chunk_size) / 16000.0
    signal = (0.3 * torch.sin(2 * torch.pi * 220.0 * time)
              + 0.1 * torch.sin(2 * torch.pi * 1375.0 * time)
              + 0.02 * torch.randn(time.shape,
                                   generator=torch.Generator().manual_seed(19)))
    recorder.put("vae.signal", signal)
    recorder.put("vae.encoded", vae.encode(signal.view(1, 1, -1), 16000)[0])

    save_file(recorder.tensors, out, metadata={
        "source": "OpenBMB VoxCPM2Model, float32 on the CPU",
        "timesteps": str(TIMESTEPS), "cfg": str(CFG)})
    for name, tensor in sorted(recorder.tensors.items()):
        print(name, tuple(tensor.shape))


if __name__ == "__main__":
    main()
