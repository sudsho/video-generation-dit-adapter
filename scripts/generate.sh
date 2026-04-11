#!/usr/bin/env bash
# quick t2v smoke gen
set -euo pipefail

PROMPT="${1:-a cat surfing on a rainbow, cinematic}"
OUT="${2:-out/sample.mp4}"

python -m src.inference.t2v \
    --prompt "$PROMPT" \
    --out "$OUT" \
    --duration 4 \
    --fps 12 \
    --cfg-scale 7.5 \
    --motion-scale 1.0 \
    --seed 42
