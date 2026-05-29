"""
DRAFT per-style LoRA trainer for the DiT attention projections. Not
verified end to end.

The current loss is plain MSE against sampled noise (no teacher forward,
no distillation objective is actually implemented despite the file name).
`peft` is imported at runtime and is not declared in requirements.txt.
The save path here (save_pretrained(out/'lora')) also does not match the
load path in src/inference/t2v.py. See the top level README for status.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

try:
    from peft import LoraConfig, get_peft_model, PeftModel  # noqa
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

from src.data import WebVidSubset, collate_video_batch
from src.model import IntegratedT2VPipe
from src.train.train_motion import encode_videos_to_latents, cosine_lr, load_config


TARGET_MODULES_DEFAULT = ["to_q", "to_k", "to_v", "to_out.0"]


def inject_lora(pipe, r: int, alpha: int, target_modules):
    if not _HAS_PEFT:
        raise RuntimeError("peft not installed; add peft to requirements to train LoRA")
    lc = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    pipe.pipe.transformer = get_peft_model(pipe.pipe.transformer, lc)
    return [p for p in pipe.pipe.transformer.parameters() if p.requires_grad]


def train_lora(cfg_path: str, style_cfg_path: str, motion_ckpt: str, out_dir: str):
    base = load_config(cfg_path)
    style = load_config(style_cfg_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = IntegratedT2VPipe(
        backbone_id=base["backbone"],
        motion_channels=base["motion"]["channels"],
        motion_layers=base["motion"]["temporal_layers"],
        heads=base["motion"]["heads"],
        head_dim=base["motion"]["head_dim"],
        device=device,
    )
    state = torch.load(motion_ckpt, map_location=device)
    pipe.motion.load_state_dict(state["motion"])
    # freeze motion too - LoRA carries style, not motion
    for p in pipe.motion.parameters():
        p.requires_grad_(False)

    trainable = inject_lora(
        pipe, r=style["lora"]["r"], alpha=style["lora"]["alpha"],
        target_modules=style["lora"].get("target_modules", TARGET_MODULES_DEFAULT),
    )
    n_train = sum(p.numel() for p in trainable)
    print(f"lora trainable params: {n_train/1e6:.2f}M")

    ds = WebVidSubset(
        manifest_csv=style["data"]["manifest"],
        num_frames=base["num_frames"],
        fps=base["fps"],
        resolution=base["resolution"],
    )
    loader = DataLoader(ds, batch_size=style["train"]["batch_size"], shuffle=True, num_workers=2, collate_fn=collate_video_batch, drop_last=True)

    opt = torch.optim.AdamW(trainable, lr=style["train"]["lr"], weight_decay=1e-4)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    scheduler = pipe.pipe.scheduler
    max_steps = style["train"]["max_steps"]
    warmup = style["train"]["warmup_steps"]
    base_lr = style["train"]["lr"]

    step = 0
    while step < max_steps:
        for batch in loader:
            latents = encode_videos_to_latents(pipe, batch["video"])
            b, f, lc, lh, lw = latents.shape
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=latents.device)
            noisy = scheduler.add_noise(
                latents.reshape(b, f * lc, lh, lw), noise.reshape(b, f * lc, lh, lw), timesteps
            ).reshape(b, f, lc, lh, lw)

            pipe.enable_motion(num_frames=f, motion_scale=1.0)
            try:
                model_in = noisy.reshape(b * f, lc, lh, lw)
                prompt_embeds = pipe.pipe.encode_prompt(batch["caption"])[0].repeat_interleave(f, dim=0)
                pred = pipe.pipe.transformer(model_in, timesteps.repeat_interleave(f),
                                             encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            finally:
                pipe.disable_motion()

            target = noise.reshape(b * f, lc, lh, lw)
            loss = F.mse_loss(pred.float(), target.float())
            loss.backward()

            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, warmup, max_steps, base_lr)
            opt.step(); opt.zero_grad(set_to_none=True)

            if step % 25 == 0:
                print(f"[lora] step {step} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e}")

            step += 1
            if step >= max_steps: break

    # save LoRA adapter
    pipe.pipe.transformer.save_pretrained(out / "lora")
    print(f"lora saved -> {out/'lora'}")


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="base yaml")
    p.add_argument("--style", required=True, help="style yaml (anime/cinematic/...)")
    p.add_argument("--motion-ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    train_lora(args.config, args.style, args.motion_ckpt, args.out)


if __name__ == "__main__":
    _cli()
