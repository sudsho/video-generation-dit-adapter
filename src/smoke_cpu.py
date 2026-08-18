"""
Offline tiny-CPU plumbing smoke for the motion-module + hook path.

Runs with no GPU, no diffusers, no model downloads, no WebVid. It:

  1. Builds a tiny frozen stand-in backbone (`TinyDiTBackbone`, NOT SD3) and a
     tiny trainable `MotionModule`, wired through the real hook installer.
  2. Trains the motion module for a few steps on synthetic, temporally-smooth
     video latents corrupted with per-frame independent noise. The temporal
     attention learns to average correlated frames -> the temporal denoising
     loss decreases.
  3. Runs a generate step that produces a small frame sequence of the right
     shape and checks the temporal-consistency plumbing actually participates
     (motion on vs motion off produce different frames; the hooks fire).

This proves the plumbing. It is NOT a text-to-video result. Real T2V needs a
GPU and a real SD3-Turbo backbone (see README).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from src.model.tiny_backbone import TinyT2VPipe


def make_synthetic_video_tokens(
    batch: int, frames: int, seq_len: int, channels: int, sigma: float, seed: int = 0
):
    """Temporally-smooth 'clean' latents + a noisy observation of them.

    clean = shared per-clip base + a slow random walk along the frame axis, so
    adjacent frames are strongly correlated. noisy = clean + per-frame iid noise.
    Temporal averaging (what the motion module does) is the natural denoiser.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(batch, 1, seq_len, channels, generator=g)
    steps = 0.15 * torch.randn(batch, frames, seq_len, channels, generator=g)
    walk = torch.cumsum(steps, dim=1)
    walk = walk - walk.mean(dim=1, keepdim=True)
    clean = base + walk
    noise = sigma * torch.randn(batch, frames, seq_len, channels, generator=g)
    noisy = clean + noise
    return clean, noisy


@dataclass
class SmokeResult:
    frozen_params: int
    motion_params: int
    num_hooks: int
    losses: List[float]
    first_loss: float
    last_loss: float
    frames_shape: tuple
    motion_on_off_l1: float


def run_smoke(
    steps: int = 60,
    batch: int = 4,
    frames: int = 6,
    channels: int = 32,
    grid: int = 4,
    sigma: float = 0.8,
    lr: float = 2e-3,
    seed: int = 0,
    verbose: bool = True,
) -> SmokeResult:
    torch.manual_seed(seed)
    pipe = TinyT2VPipe(
        channels=channels,
        num_blocks=4,
        heads=2,
        grid=grid,
        motion_layers=2,
        motion_heads=2,
        motion_head_dim=8,
        max_frames=max(32, frames),
        every_n_blocks=2,
        seed=seed,
    )
    seq_len = grid * grid

    # count hooks once (also validates the plumbing wires up)
    num_hooks = pipe.enable_motion(num_frames=frames, motion_scale=1.0)
    pipe.disable_motion()

    clean, noisy = make_synthetic_video_tokens(
        batch, frames, seq_len, channels, sigma=sigma, seed=seed + 1
    )
    clean_flat = clean.reshape(batch * frames, seq_len, channels)
    noisy_flat = noisy.reshape(batch * frames, seq_len, channels)

    opt = torch.optim.Adam(pipe.motion.parameters(), lr=lr)

    losses: List[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = pipe.forward_tokens(noisy_flat, num_frames=frames, motion_scale=1.0)
        loss = F.mse_loss(pred, clean_flat)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if verbose and (step % 10 == 0 or step == steps - 1):
            print(f"step {step:3d}  temporal_denoise_loss {loss.item():.4f}")

    # generate: produce a tiny frame sequence, and confirm the temporal module
    # actually participates (motion on vs off must differ)
    frames_on = pipe.generate(num_frames=frames, seed=123, motion_scale=1.0)
    frames_off = pipe.generate(num_frames=frames, seed=123, motion_scale=0.0)
    on_off_l1 = float((frames_on.float() - frames_off.float()).abs().mean().item())

    result = SmokeResult(
        frozen_params=pipe.frozen_backbone_params(),
        motion_params=pipe.trainable_motion_params(),
        num_hooks=num_hooks,
        losses=losses,
        first_loss=losses[0],
        last_loss=losses[-1],
        frames_shape=tuple(frames_on.shape),
        motion_on_off_l1=on_off_l1,
    )

    if verbose:
        print()
        print(f"frozen backbone params : {result.frozen_params}")
        print(f"trainable motion params: {result.motion_params}")
        print(f"motion hooks installed : {result.num_hooks}")
        print(
            f"temporal denoise loss  : {result.first_loss:.4f} -> {result.last_loss:.4f} "
            f"({100.0 * (1 - result.last_loss / result.first_loss):.1f}% reduction)"
        )
        print(f"generated frames shape : {result.frames_shape}  (F, H, W, 3)")
        print(
            f"motion on vs off L1    : {result.motion_on_off_l1:.2f} "
            f"(non-zero => temporal-consistency plumbing ran)"
        )
    return result


def _cli():
    p = argparse.ArgumentParser(description="Tiny-CPU plumbing smoke (no GPU, no diffusers).")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    res = run_smoke(steps=args.steps, frames=args.frames, seed=args.seed)
    ok = res.last_loss < res.first_loss and res.motion_on_off_l1 > 0.0
    print()
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
