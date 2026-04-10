#!/usr/bin/env bash
# Kick off motion module training on a single-node multi-GPU box.
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
OUT="${2:-runs/motion}"

mkdir -p "$OUT"
accelerate launch \
    --mixed_precision bf16 \
    --num_processes "${NGPU:-1}" \
    -m src.train.train_motion \
    --config "$CONFIG" \
    --out "$OUT" \
    "${@:3}"
