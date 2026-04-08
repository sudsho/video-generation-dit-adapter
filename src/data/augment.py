"""
Video-safe augmentations. Applied per-clip (same params for all frames in a clip)
so we don't shatter temporal coherence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


class ColorJitterVideo:
    """Random brightness / contrast / saturation, one sample per clip."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # video: (F, C, H, W) in [-1, 1]
        b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
        c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
        s = 1.0 + (torch.rand(1).item() * 2 - 1) * self.saturation
        x = (video + 1) / 2  # to [0,1]
        # brightness
        x = (x * b).clamp(0, 1)
        # contrast: pull to mean
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = ((x - mean) * c + mean).clamp(0, 1)
        # saturation: pull to grey per pixel
        grey = x.mean(dim=1, keepdim=True)
        x = ((x - grey) * s + grey).clamp(0, 1)
        return x * 2 - 1


class SpeedJitter:
    """Randomly resample frames with a speed factor in [1/max, max]."""

    def __init__(self, max_factor: float = 1.5, num_frames: int = 48):
        assert max_factor >= 1.0
        self.max_factor = max_factor
        self.num_frames = num_frames

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # video: (F, C, H, W)
        f = video.shape[0]
        factor = torch.empty(1).uniform_(1.0 / self.max_factor, self.max_factor).item()
        # sample num_frames indices at that factor
        idxs = torch.linspace(0, min(f - 1, (self.num_frames - 1) * factor), self.num_frames)
        idxs = idxs.round().long().clamp(0, f - 1)
        return video[idxs]
