"""
CLI: text -> mp4.

Loads the integrated pipe, optionally applies a style LoRA, and writes an mp4.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml

from src.model import IntegratedT2VPipe, GenerationConfig


STYLE_LORAS = {
    "anime": "runs/lora/anime/lora.safetensors",
    "cinematic": "runs/lora/cinematic/lora.safetensors",
}


def _load_cfg(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def _build_pipe(base_cfg: dict, motion_ckpt: str) -> IntegratedT2VPipe:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = IntegratedT2VPipe(
        backbone_id=base_cfg["backbone"],
        motion_channels=base_cfg["motion"]["channels"],
        motion_layers=base_cfg["motion"]["temporal_layers"],
        heads=base_cfg["motion"]["heads"],
        head_dim=base_cfg["motion"]["head_dim"],
        device=device,
    )
    if motion_ckpt and os.path.exists(motion_ckpt):
        state = torch.load(motion_ckpt, map_location=device)
        pipe.motion.load_state_dict(state["motion"])
    return pipe


def write_mp4(frames: np.ndarray, out_path: str, fps: int = 12) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".gif":
        imageio.mimsave(out_path, list(frames), duration=1.0 / fps)
        return
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)


def run(
    prompt: str,
    out_path: str,
    config: str = "configs/base.yaml",
    motion_ckpt: str = "runs/motion/motion_final.pt",
    duration: float = 4.0,
    fps: int = 12,
    cfg_scale: float = 7.5,
    motion_scale: float = 1.0,
    style: str | None = None,
    seed: int | None = None,
):
    base = _load_cfg(config)
    pipe = _build_pipe(base, motion_ckpt)
    if style:
        lora_path = STYLE_LORAS.get(style)
        if lora_path and os.path.exists(lora_path):
            pipe.load_style_lora(lora_path)
        else:
            print(f"warn: style '{style}' not found on disk, ignoring")

    num_frames = int(round(duration * fps))
    gcfg = GenerationConfig(
        num_frames=num_frames,
        fps=fps,
        height=base["resolution"],
        width=base["resolution"],
        cfg_scale=cfg_scale,
        motion_scale=motion_scale,
        num_inference_steps=base["inference"]["num_inference_steps"],
        seed=seed,
        style=style,
    )
    frames = pipe.generate(prompt, gcfg)
    write_mp4(frames, out_path, fps=fps)
    print(f"wrote {out_path} ({num_frames} frames @ {fps} fps)")


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--motion-ckpt", type=str, default="runs/motion/motion_final.pt")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--cfg-scale", type=float, default=7.5)
    p.add_argument("--motion-scale", type=float, default=1.0)
    p.add_argument("--style", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    run(
        prompt=args.prompt,
        out_path=args.out,
        config=args.config,
        motion_ckpt=args.motion_ckpt,
        duration=args.duration,
        fps=args.fps,
        cfg_scale=args.cfg_scale,
        motion_scale=args.motion_scale,
        style=args.style,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
