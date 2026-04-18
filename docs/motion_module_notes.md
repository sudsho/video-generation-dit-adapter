# Motion module notes

Design decisions and their tradeoffs.

## Why temporal-only attention (not full 3D)?

Full spatial-temporal attention would be O((F*H*W)^2). At 48 frames * 64 * 64
that's 12.6B pairs per head. Way too much. Temporal-only pays O(F^2) per pixel
which is manageable and matches what AnimateDiff and Stable Video Diffusion
converged on.

## Why zero-init the output projections?

SD3-Turbo is a step-distilled model. Any random perturbation of its residual
stream at step 0 wrecks the distillation and produces mush. Zero-init means
step 0 is bit-identical to the base image model, and the motion module
learns from a good starting point.

## Why hook every N blocks instead of all blocks?

Motion coherence saturates around 3-4 hook points in SD3-Turbo (24 blocks).
More hook points = more params to train and more VRAM, without matching FVD
improvement. `every_n_blocks=6` gives 4 hook points which is our default.

## Why sinusoidal positions instead of learned?

Learned frame positions locked in a fixed max frame count. Sinusoidal
generalizes to any frame count up to `max_frames`, so the same trained
motion module can produce 24-frame or 96-frame clips at inference.

## Params

Default: 4 temporal layers, 8 heads, head_dim=40 -> ~30M params. That's
about 1.5% of the SD3-Turbo transformer, and it's the only trainable part
of the network for motion training.
