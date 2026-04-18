# Latency and VRAM

Measured on a single H100 80GB, bf16, SD3-Turbo backbone.

## Inference

| Resolution | Frames | Steps | Time (s) | Peak VRAM |
|---|---|---|---|---|
| 512 x 512 | 48 (4s @ 12fps) | 6 | 3.4 | 22.1 GB |
| 512 x 512 | 96 (8s @ 12fps) | 6 | 6.9 | 38.4 GB |
| 384 x 384 | 48 | 6 | 2.1 | 13.7 GB |
| 512 x 512 | 48 | 4 (fewer steps) | 2.3 | 22.1 GB |
| 512 x 512 | 24 (2s @ 12fps) | 6 | 1.7 | 12.8 GB |

Motion module overhead is ~14% of end-to-end latency; the rest is the
frozen SD3-Turbo forward. xformers memory-efficient attention drops
peak VRAM by ~18% on 96-frame clips.

## Training

| Config | Batch | Grad Accum | Effective Batch | Peak VRAM | Steps/s |
|---|---|---|---|---|---|
| base (48 frames, 512, bf16) | 2 | 4 | 8 | 61.2 GB | 0.44 |
| shorter clips (24 frames, 512, bf16) | 4 | 2 | 8 | 42.8 GB | 0.71 |
| bf16 + grad checkpointing | 2 | 4 | 8 | 44.1 GB | 0.31 |

Full motion adapter (20k steps at effective batch 8) trains in ~12h on H100.

## LoRA distillation

- 4k LoRA steps, batch 2 -> ~35 minutes on H100.
- LoRA memory footprint: ~4-15 MB per style depending on rank.
