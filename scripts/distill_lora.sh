#!/usr/bin/env bash
# per-style LoRA distillation
set -euo pipefail

STYLE_CFG="${1:-configs/style_anime.yaml}"
MOTION_CKPT="${2:-runs/motion/motion_final.pt}"
OUT="${3:-runs/lora/anime}"

python -m src.train.distill_lora \
    --config configs/base.yaml \
    --style "$STYLE_CFG" \
    --motion-ckpt "$MOTION_CKPT" \
    --out "$OUT"
