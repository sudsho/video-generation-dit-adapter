"""
Batch generate mp4s from a prompts TSV.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from src.inference.t2v import run as run_single


def batch(
    prompts_tsv: str,
    out_dir: str,
    config: str,
    motion_ckpt: str,
    duration: float,
    fps: int,
    cfg_scale: float,
    motion_scale: float,
    default_style: str | None,
    seed: int | None,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(prompts_tsv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if not row:
                continue
            prompt = row[0]
            style = row[1] if len(row) > 1 and row[1] else default_style
            out = os.path.join(out_dir, f"clip_{i:04d}.mp4")
            run_single(
                prompt=prompt,
                out_path=out,
                config=config,
                motion_ckpt=motion_ckpt,
                duration=duration,
                fps=fps,
                cfg_scale=cfg_scale,
                motion_scale=motion_scale,
                style=style,
                seed=seed,
            )


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--motion-ckpt", type=str, default="runs/motion/motion_final.pt")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--cfg-scale", type=float, default=7.5)
    p.add_argument("--motion-scale", type=float, default=1.0)
    p.add_argument("--style", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    batch(
        args.prompts,
        args.out_dir,
        args.config,
        args.motion_ckpt,
        args.duration,
        args.fps,
        args.cfg_scale,
        args.motion_scale,
        args.style,
        args.seed,
    )


if __name__ == "__main__":
    _cli()
