# Benchmarks

Small internal benchmark set: 250 prompts drawn from the VBench category
distribution (single object, human motion, animals, landscapes, abstract).
Each config generates 5 seeds per prompt, we report mean.

## FVD vs baselines (I3D Kinetics features)

Lower is better.

| Config | Steps | Motion module | Style | FVD |
|---|---|---|---|---|
| SD3-Turbo image + last-frame repeat | 6 | none | - | 512.4 |
| SD3-Turbo + naive frame stitching | 6 | none | - | 408.7 |
| SVD-XT (baseline) | 25 | native | - | 214.3 |
| **ours (motion module only)** | 6 | trained | none | 191.6 |
| **ours + anime LoRA** | 6 | trained | anime | 218.9 |
| **ours + cinematic LoRA** | 6 | trained | cinematic | 205.2 |

## CLIP-T (per-frame text alignment, higher is better)

| Config | CLIP-T mean | CLIP-T min |
|---|---|---|
| SD3-Turbo image + repeat | 0.291 | 0.284 |
| ours | 0.288 | 0.269 |
| ours + anime | 0.276 | 0.251 |
| ours + cinematic | 0.283 | 0.263 |

Motion training costs a tiny bit of frame-level CLIP score (frames are
allowed to evolve from the initial spatial fit); LoRAs cost more because
they trade prompt fidelity for style consistency.

## Motion score (Farneback dense flow, higher = more motion)

| Config | mean | median | p90 |
|---|---|---|---|
| SD3-Turbo image + repeat | 0.02 | 0.01 | 0.05 |
| ours motion_scale=0 | 0.03 | 0.02 | 0.08 |
| ours motion_scale=0.5 | 3.14 | 2.61 | 6.44 |
| **ours motion_scale=1.0** | 4.87 | 4.11 | 9.72 |
| ours motion_scale=1.5 | 6.02 | 5.24 | 12.11 (jitter artifacts appear) |

motion_scale > 1.5 starts to produce jittery high-frequency motion that a
human viewer flags as unnatural. 1.0 is the sweet spot.

## Latency reminder

See `docs/latency_and_vram.md` for latency + VRAM tables.
