"""
Optical-flow-based motion score. Uses Farneback dense flow between adjacent
frames and averages the magnitude. Higher = more motion.

A useful sanity signal: if motion_score is near zero on an animated prompt,
the motion module is doing nothing.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _to_grey(frame_uint8: np.ndarray) -> np.ndarray:
    # simple luma conversion
    r = frame_uint8[..., 0].astype(np.float32)
    g = frame_uint8[..., 1].astype(np.float32)
    b = frame_uint8[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def motion_score(frames_uint8: np.ndarray) -> Dict[str, float]:
    """
    frames_uint8: (F, H, W, 3). Returns mean/median magnitude of dense flow
    between consecutive frames.
    """
    import cv2  # local, expensive-ish import

    f, h, w, _ = frames_uint8.shape
    prev = _to_grey(frames_uint8[0])
    mags = []
    for i in range(1, f):
        cur = _to_grey(frames_uint8[i])
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.linalg.norm(flow, axis=-1)
        mags.append(mag)
        prev = cur
    stack = np.stack(mags)
    return {
        "mean": float(stack.mean()),
        "median": float(np.median(stack)),
        "p90": float(np.percentile(stack, 90)),
        "std_per_frame": float(stack.mean(axis=(1, 2)).std()),
    }
