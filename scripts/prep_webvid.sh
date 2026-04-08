#!/usr/bin/env bash
# Build a small webvid-style manifest from a folder of mp4 clips + a captions.tsv.
# Usage: prep_webvid.sh <video-dir> <captions.tsv> <out-manifest.csv>
set -euo pipefail

VIDEO_DIR="${1:-data/raw}"
CAPTIONS="${2:-data/captions.tsv}"
OUT="${3:-data/manifest.csv}"

mkdir -p "$(dirname "$OUT")"
echo "video_path,caption,start_sec,end_sec" > "$OUT"

while IFS=$'\t' read -r fname caption; do
    path="$VIDEO_DIR/$fname"
    if [ ! -f "$path" ]; then
        echo "skip missing: $path" >&2
        continue
    fi
    # probe duration via ffprobe
    dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$path" || echo 0)
    printf '%s,"%s",0,%s\n' "$path" "$caption" "$dur" >> "$OUT"
done < "$CAPTIONS"

echo "wrote $(wc -l < "$OUT") rows to $OUT"
