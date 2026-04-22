# video-generation-dit-adapter

Text-to-short-video via an AnimateDiff-style motion module inserted into a
frozen **SD3-Turbo** backbone, with **per-style LoRA** adapters on top.

4 seconds at 12 fps, 512x512, on-demand. Roughly 3.5s / clip on an H100.

```
+---------------------+     +----------------+     +---------------+
|  text prompt        | --> |  SD3-Turbo DiT | +-> | per-frame VAE |
+---------------------+     |  (frozen)      | |   +---------------+
                            |    hooks       | |
                            |     v          | |
                            |  motion module | |
                            |  (trained)     | |
                            |     v          | |
                            |  LoRA style    | |
                            |  (per style)   | |
                            +----------------+ |
                                               v
                                     48-frame latent tensor
                                     decoded to mp4
```

## Problem

Creators need on-demand short clips for prototyping (ads, storyboards,
placeholder shots). Full T2V models are expensive to train and slow at
inference. This repo shows a low-cost path: reuse a strong image model
(SD3-Turbo, one-step distilled), add a small trainable motion module,
swap in a tiny LoRA for style.

## Method (quick)

- Hook forward pass of selected SD3-Turbo transformer blocks.
- Insert a temporal-only self-attention stack over the frame axis at
  each hook point. Zero-init the residual output so step-0 = pretrained
  image model exactly.
- Train the motion module on a WebVid-style clip manifest, flow-matching
  loss against the SD3 noise schedule. Backbone stays frozen.
- Per style, train a small LoRA (r=8-16) into the DiT attention
  projections. Combine at inference with weighted `set_adapters`.

Full write-up: [docs/method.md](docs/method.md).

## Results (short)

| Config | FVD (I3D) | CLIP-T | Motion mean |
|---|---|---|---|
| SD3-Turbo image + frame repeat | 512.4 | 0.291 | 0.02 |
| ours (motion module) | **191.6** | 0.288 | 4.87 |
| ours + anime LoRA | 218.9 | 0.276 | 4.62 |
| ours + cinematic LoRA | 205.2 | 0.283 | 4.71 |

Full table: [benchmarks/results.md](benchmarks/results.md).

## Setup

```bash
pip install -r requirements.txt
```

Prep a manifest (mp4 folder + captions.tsv):
```bash
bash scripts/prep_webvid.sh data/raw data/captions.tsv data/manifest.csv
```

Train the motion module:
```bash
bash scripts/train_motion.sh configs/base.yaml runs/motion
```

Train a style LoRA:
```bash
bash scripts/distill_lora.sh configs/style_anime.yaml runs/motion/motion_final.pt runs/lora/anime
```

Generate:
```bash
python -m src.inference.t2v --prompt "a cat surfing on a rainbow, cinematic" \
    --out out/clip.mp4 --duration 4 --fps 12 --style cinematic --seed 42
```

## Deploy

FastAPI:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# POST /generate {"prompt":"...", "duration":4, "style":"anime"}
```

Streamlit demo:
```bash
streamlit run streamlit_app.py
```

Docker:
```bash
docker compose up -d
# api  -> http://localhost:8000/docs
# demo -> http://localhost:8501
```

## Latency + VRAM

See [docs/latency_and_vram.md](docs/latency_and_vram.md). Single-clip
inference at 4s/12fps/512x512 fits in 22 GB VRAM and takes ~3.5s on H100.

## Repo layout

```
src/
  model/    motion_module.py  sd3_backbone.py  integrated_pipe.py
  train/    train_motion.py   distill_lora.py
  data/     webvid_subset.py  augment.py
  inference/ t2v.py  batch_t2v.py
  eval/     fvd_score.py  clipsim_temporal.py  motion_score.py
  api/      main.py
configs/    base.yaml  style_anime.yaml  style_cinematic.yaml  distill_lora.yaml
scripts/    train_motion.sh  distill_lora.sh  generate.sh  eval_fvd.sh  prep_webvid.sh
tests/      test_motion_module.py  test_pipe_generate.py  test_metrics.py  test_api.py
notebooks/  style_lora_ablation.ipynb  motion_scale_sweep.ipynb  cfg_sensitivity.ipynb
docs/       method.md  motion_module_notes.md  style_lora.md
            latency_and_vram.md  ethical_considerations_synth_video.md
benchmarks/ results.md
```

## Ethics

Generative video has real harm surface: deepfakes, style laundering,
non-consensual content. See
[docs/ethical_considerations_synth_video.md](docs/ethical_considerations_synth_video.md)
for what to add before shipping this in a product.

## License

MIT.
