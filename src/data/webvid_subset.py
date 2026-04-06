"""
WebVid-style loader. We consume an mp4 manifest CSV with columns
(video_path, caption, start_sec, end_sec) and trim clip-by-clip
to a fixed number of frames at the target fps.

Public WebVid mirrors are periodically restricted. This loader also
works with any (video_path, caption) parquet you assemble locally
from other permissively licensed sources.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import av  # PyAV
except Exception:  # pragma: no cover
    av = None


@dataclass
class ClipRecord:
    path: str
    caption: str
    start_sec: float
    end_sec: float


def read_manifest(manifest_csv: str) -> List[ClipRecord]:
    rows: List[ClipRecord] = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ClipRecord(
                    path=r["video_path"],
                    caption=r["caption"],
                    start_sec=float(r.get("start_sec", 0.0)),
                    end_sec=float(r.get("end_sec", 0.0)),
                )
            )
    return rows


def _read_frames_pyav(
    path: str, start_sec: float, num_frames: int, fps: int
) -> np.ndarray:
    """Trim a clip to num_frames at target fps starting at start_sec."""
    if av is None:
        raise RuntimeError("PyAV missing; install av==13.1.0")
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    time_base = float(stream.time_base)
    frame_step = 1.0 / fps
    target_times = [start_sec + i * frame_step for i in range(num_frames)]
    frames: List[np.ndarray] = []
    ti = 0
    for frame in container.decode(video=0):
        t = float(frame.pts) * time_base
        while ti < len(target_times) and t >= target_times[ti]:
            frames.append(frame.to_ndarray(format="rgb24"))
            ti += 1
        if ti >= len(target_times):
            break
    container.close()
    if len(frames) < num_frames:
        # pad by repeating last frame
        if not frames:
            raise RuntimeError(f"could not decode any frames from {path}")
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return np.stack(frames[:num_frames], axis=0)


class WebVidSubset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        num_frames: int = 48,
        fps: int = 12,
        resolution: int = 512,
        min_clip_seconds: float = 2.5,
        max_clip_seconds: float = 6.0,
        transform=None,
        text_transform=None,
    ):
        self.records = [
            r
            for r in read_manifest(manifest_csv)
            if (r.end_sec - r.start_sec) >= min_clip_seconds
        ]
        self.num_frames = num_frames
        self.fps = fps
        self.resolution = resolution
        self.max_clip_seconds = max_clip_seconds
        self.transform = transform
        self.text_transform = text_transform

    def __len__(self) -> int:
        return len(self.records)

    def _resize_and_center_crop(self, frames_hwc: np.ndarray) -> np.ndarray:
        import cv2  # local import so tests without cv2 don't error

        f, h, w, _ = frames_hwc.shape
        scale = self.resolution / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        out = np.empty((f, self.resolution, self.resolution, 3), dtype=frames_hwc.dtype)
        for i in range(f):
            resized = cv2.resize(frames_hwc[i], (nw, nh), interpolation=cv2.INTER_AREA)
            y0 = (nh - self.resolution) // 2
            x0 = (nw - self.resolution) // 2
            out[i] = resized[y0 : y0 + self.resolution, x0 : x0 + self.resolution]
        return out

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        need_seconds = self.num_frames / self.fps
        max_start = max(rec.start_sec, rec.end_sec - need_seconds)
        start = float(np.random.uniform(rec.start_sec, max_start)) if max_start > rec.start_sec else rec.start_sec
        frames = _read_frames_pyav(rec.path, start, self.num_frames, self.fps)
        frames = self._resize_and_center_crop(frames)
        # to (F, C, H, W), float in [-1, 1]
        video = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        caption = rec.caption
        if self.transform is not None:
            video = self.transform(video)
        if self.text_transform is not None:
            caption = self.text_transform(caption)
        return {"video": video, "caption": caption}


def collate_video_batch(batch):
    videos = torch.stack([b["video"] for b in batch], dim=0)  # (B, F, C, H, W)
    captions = [b["caption"] for b in batch]
    return {"video": videos, "caption": captions}
