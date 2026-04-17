# Method

## Overview

We take an SD3-Turbo backbone (spatial DiT for images) and extend it to short
video by inserting **temporal attention** modules between selected transformer
blocks. The spatial DiT stays frozen; only the motion module trains. Style
comes from per-style LoRA adapters injected into the DiT attention projections.

## Motion Module

For each hooked block, the residual stream shape is
`(B * F, seq, C)`. We reshape to `(B * seq, F, C)` and run a stack of
temporal attention + FFN over the frame axis. Then reshape back and mix
into the block output with `motion_scale * (temporal_out - x)`.

At `motion_scale = 0` the module is a no-op (useful for image ablations).
At init, the output projections are zero-initialized, so the pretrained
image backbone is preserved exactly on step 0. This is important for
SD3-Turbo, whose one-step distillation degrades if you perturb the weights.

## Training

Loss is standard flow-matching MSE against the added noise, over per-frame
latents from the SD3 VAE. We share one timestep per clip across frames.

The DiT stays frozen; only the motion module receives gradients (~30M params).
LoRA style adapters can be trained on top afterwards, freezing motion.

## Data

We use a WebVid-style manifest of (video, caption, start_sec, end_sec)
tuples. Clips are trimmed to 48 frames at 12 fps (4 seconds) at 512x512.
Color and speed jitter are applied clip-consistently.

## Inference

SD3-Turbo runs at 6 steps by default. Per-frame inference is done in
parallel over the frame axis, then the motion hooks run once per block
per denoising step. On a single H100, 4-second 512x512 clips generate in
about 3.5 seconds end-to-end (see docs/latency_and_vram.md).
