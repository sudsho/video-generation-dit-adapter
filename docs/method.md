# Method (sketch)

This document describes the intended shape of the pipeline. Nothing in the
repo is a verified end-to-end training or inference run. Treat this as
design notes, not a reproduction guide.

## Overview

Take a spatial diffusion transformer (DiT) intended for images and extend
it to short video by inserting temporal attention modules between selected
transformer blocks. The spatial DiT would stay frozen; only the motion
module trains. Style would come from per-style LoRA adapters injected into
the DiT attention projections.

## Motion Module

For each hooked block, the residual stream has shape `(B * F, seq, C)`
where F is the number of frames. We reshape to `(B * seq, F, C)` and run a
stack of temporal attention + FFN over the frame axis, then reshape back
and mix into the block output with `motion_scale * (temporal_out - x)`.

At `motion_scale = 0` the residual mix vanishes (useful as an ablation
lever). The output projections of the temporal attention block and the
final FFN linear are zero initialized; note however that the sinusoidal
frame position embedding is added into the pre attention tensor, so the
module is NOT numerically identity at initialization when
`motion_scale != 0`. That would need to be fixed (either move the pos
embed after the zero init projection, or start `motion_scale` at 0 during
warmup) before any claim about "step 0 = pretrained backbone" is true.

## Training (draft)

The intended loss depends on the backbone's parameterization: an
epsilon-prediction backbone would take an MSE against the sampled noise;
a rectified-flow / velocity-prediction backbone (e.g. SD3) would take an
MSE against `(noise - latents)`. The training script in
`src/train/train_motion.py` uses MSE against the sampled noise as a
placeholder and has NOT been reconciled against the actual scheduler /
`encode_prompt` / transformer forward signatures of any specific released
pipeline. Do not treat it as runnable.

## Data (intended)

A WebVid-style manifest of (video, caption, start_sec, end_sec) tuples,
trimmed to 48 frames at 12 fps (4 seconds) at 512x512, with clip-consistent
color jitter.

## Inference (intended)

Per-frame inference in parallel over the frame axis, with the motion hooks
run once per block per denoising step. The `src/inference/t2v.py` CLI is a
draft; it will not produce a real clip without a working pipeline and a
trained motion checkpoint neither of which is in this repo.
