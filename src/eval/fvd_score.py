"""
Frechet Video Distance (FVD) using the I3D feature extractor.

We rely on an I3D checkpoint (Kinetics-pretrained). If the checkpoint is not
present locally we fall back to a random-projection surrogate that at least
gives comparable numbers under matched pipelines (dev-time convenience, not
a substitute for real FVD).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """Symmetric PSD matrix square root via eigendecomposition (stable)."""
    w, v = np.linalg.eigh((mat + mat.T) / 2)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.T


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean = _sqrtm_psd(sigma1 @ sigma2)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


class I3DFeatures:
    def __init__(self, ckpt: str = "checkpoints/i3d_kinetics.pt", feat_dim: int = 400):
        self.ckpt = ckpt
        self.feat_dim = feat_dim
        self._i3d = None
        if os.path.exists(ckpt):
            self._load_i3d()

    def _load_i3d(self):
        import torch  # lazy
        # user-supplied I3D backbone. we keep it optional so unit tests don't
        # need the full 300MB checkpoint.
        self._i3d = torch.jit.load(self.ckpt).eval()

    def features(self, videos: np.ndarray) -> np.ndarray:
        """videos: (N, F, H, W, 3) uint8 -> (N, D) float32 features."""
        if self._i3d is None:
            # deterministic surrogate: hash pixels into a fixed random projection
            rng = np.random.default_rng(0)
            proj = rng.standard_normal((videos.shape[1] * videos.shape[2] * videos.shape[3] * 3, self.feat_dim)).astype(np.float32)
            flat = videos.reshape(videos.shape[0], -1).astype(np.float32) / 255.0
            return flat @ proj
        import torch
        v = torch.from_numpy(videos).float().permute(0, 4, 1, 2, 3) / 255.0
        with torch.no_grad():
            return self._i3d(v).cpu().numpy()


def compute_fvd(real: np.ndarray, fake: np.ndarray, extractor: I3DFeatures | None = None) -> float:
    extractor = extractor or I3DFeatures()
    r = extractor.features(real)
    f = extractor.features(fake)
    mu_r, mu_f = r.mean(0), f.mean(0)
    sig_r = np.cov(r, rowvar=False) + 1e-6 * np.eye(r.shape[1])
    sig_f = np.cov(f, rowvar=False) + 1e-6 * np.eye(f.shape[1])
    return frechet_distance(mu_r, sig_r, mu_f, sig_f)
