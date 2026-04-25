"""
Train the motion module on top of a frozen SD3-Turbo backbone.

Standard v-prediction / flow-matching loss over the latent video sequence.
The backbone stays frozen; only the motion adapter parameters receive gradients.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

try:
    import wandb  # noqa: F401
    _WANDB = True
except Exception:
    _WANDB = False

from src.data import WebVidSubset, collate_video_batch, ColorJitterVideo
from src.model import IntegratedT2VPipe


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def encode_videos_to_latents(pipe, videos: torch.Tensor) -> torch.Tensor:
    """videos: (B, F, C, H, W) in [-1,1]. Returns (B, F, C', H', W') latents."""
    b, f, c, h, w = videos.shape
    frames = videos.reshape(b * f, c, h, w).to(pipe.device, pipe.dtype)
    with torch.no_grad():
        lat = pipe.pipe.vae.encode(frames).latent_dist.sample() * pipe.pipe.vae.config.scaling_factor
    lc = lat.shape[1]
    lh = lat.shape[2]
    lw = lat.shape[3]
    return lat.reshape(b, f, lc, lh, lw)


def train(cfg_path: str, out_dir: str, resume: str | None = None):
    cfg = load_config(cfg_path)
    torch.manual_seed(cfg.get("seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = IntegratedT2VPipe(
        backbone_id=cfg["backbone"],
        motion_channels=cfg["motion"]["channels"],
        motion_layers=cfg["motion"]["temporal_layers"],
        heads=cfg["motion"]["heads"],
        head_dim=cfg["motion"]["head_dim"],
        device=device,
    )

    if resume:
        state = torch.load(resume, map_location=device)
        pipe.motion.load_state_dict(state["motion"])
        start_step = int(state.get("step", 0))
    else:
        start_step = 0

    ds = WebVidSubset(
        manifest_csv=os.path.join(cfg["data"]["root"], "manifest.csv"),
        num_frames=cfg["num_frames"],
        fps=cfg["fps"],
        resolution=cfg["resolution"],
        min_clip_seconds=cfg["data"]["min_clip_seconds"],
        max_clip_seconds=cfg["data"]["max_clip_seconds"],
        transform=ColorJitterVideo(),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_video_batch,
        pin_memory=True,
        drop_last=True,
    )

    # split params: weight decay on 2D+ tensors only (matches nanoGPT convention)
    decay, no_decay = [], []
    for p in pipe.motion.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["train"]["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["train"]["lr"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["train"]["mixed_precision"] == "fp16"))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if _WANDB and cfg["logging"].get("project"):
        import wandb
        wandb.init(project=cfg["logging"]["project"], entity=cfg["logging"].get("entity"), config=cfg)

    step = start_step
    grad_accum = cfg["train"]["grad_accum"]
    max_steps = cfg["train"]["max_steps"]
    save_every = cfg["train"]["save_every"]
    warmup = cfg["train"]["warmup_steps"]
    base_lr = cfg["train"]["lr"]

    scheduler = pipe.pipe.scheduler
    t_start = time.time()

    while step < max_steps:
        for batch in loader:
            videos = batch["video"]  # (B, F, C, H, W)
            captions = batch["caption"]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=cfg["train"]["mixed_precision"] == "bf16"):
                latents = encode_videos_to_latents(pipe, videos)
            b, f, lc, lh, lw = latents.shape
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (b,), device=latents.device
            )
            # broadcast per-batch timestep across frames
            noisy = scheduler.add_noise(
                latents.reshape(b, f * lc, lh, lw), noise.reshape(b, f * lc, lh, lw), timesteps
            ).reshape(b, f, lc, lh, lw)

            pipe.enable_motion(num_frames=f, motion_scale=1.0)
            try:
                # flatten B*F into batch dim for the transformer
                model_in = noisy.reshape(b * f, lc, lh, lw)
                prompt_embeds = pipe.pipe.encode_prompt(captions)[0]
                # repeat prompt embeds per frame
                prompt_embeds = prompt_embeds.repeat_interleave(f, dim=0)
                pred = pipe.pipe.transformer(
                    model_in,
                    timesteps.repeat_interleave(f),
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            finally:
                pipe.disable_motion()

            target = noise.reshape(b * f, lc, lh, lw)
            loss = F.mse_loss(pred.float(), target.float()) / grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    pipe.motion.parameters(), cfg["train"]["clip_grad_norm"]
                )
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, warmup, max_steps, base_lr)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if step % 25 == 0:
                elapsed = time.time() - t_start
                print(f"step {step} loss {loss.item()*grad_accum:.4f} lr {opt.param_groups[0]['lr']:.2e} elapsed {elapsed:.0f}s")
                if _WANDB and cfg["logging"].get("project"):
                    import wandb
                    wandb.log(
                        {"loss": loss.item() * grad_accum, "lr": opt.param_groups[0]["lr"], "step": step}
                    )

            if step > 0 and step % save_every == 0:
                ckpt_path = out_path / f"motion_step{step:07d}.pt"
                torch.save(
                    {"motion": pipe.motion.state_dict(), "step": step, "cfg": cfg},
                    ckpt_path,
                )
                # keep only last 3 to save disk
                ckpts = sorted(out_path.glob("motion_step*.pt"))
                for old in ckpts[:-3]:
                    old.unlink(missing_ok=True)

            step += 1
            if step >= max_steps:
                break

    torch.save({"motion": pipe.motion.state_dict(), "step": step}, out_path / "motion_final.pt")


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args.config, args.out, args.resume)


if __name__ == "__main__":
    _cli()
