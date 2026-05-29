#!/usr/bin/env bash
# Draft launcher for the motion module trainer. Single-process only.
# The training script itself is not wired to a real pipeline (see the top
# level README); this shell script is here for future use.
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
OUT="${2:-runs/motion}"

mkdir -p "$OUT"
python -m src.train.train_motion \
    --config "$CONFIG" \
    --out "$OUT" \
    "${@:3}"
