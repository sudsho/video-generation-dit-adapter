#!/usr/bin/env bash
# quick fvd sanity harness. expects real/ and fake/ dirs full of mp4s.
set -euo pipefail

REAL="${1:-eval/real}"
FAKE="${2:-eval/fake}"

python -c "
from src.eval.fvd_score import compute_fvd, I3DFeatures
import imageio, glob, os, numpy as np

def load_dir(d, num_frames=48, size=112):
    import cv2
    out = []
    for p in sorted(glob.glob(os.path.join(d, '*.mp4'))):
        rd = imageio.get_reader(p)
        frames = []
        for i, fr in enumerate(rd):
            if i >= num_frames: break
            frames.append(cv2.resize(fr, (size, size)))
        rd.close()
        if len(frames) == num_frames:
            out.append(np.stack(frames))
    return np.stack(out)

r = load_dir('$REAL')
f = load_dir('$FAKE')
print('n_real', r.shape[0], 'n_fake', f.shape[0])
print('FVD:', compute_fvd(r, f, extractor=I3DFeatures(feat_dim=256)))
"
