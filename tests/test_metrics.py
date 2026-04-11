import numpy as np

from src.eval.fvd_score import compute_fvd, frechet_distance, I3DFeatures
from src.eval.motion_score import motion_score


def test_frechet_distance_identical_zero():
    mu = np.zeros(8)
    sig = np.eye(8)
    assert frechet_distance(mu, sig, mu, sig) < 1e-6


def test_fvd_surrogate_close_for_identical_batches():
    ext = I3DFeatures(ckpt="/nonexistent/path.pt", feat_dim=64)
    rng = np.random.default_rng(0)
    vids = rng.integers(0, 255, size=(4, 8, 16, 16, 3), dtype=np.uint8)
    d = compute_fvd(vids, vids.copy(), extractor=ext)
    assert d < 5.0  # sanity: not exactly zero because of cov jitter


def test_motion_score_zero_on_static_video():
    try:
        import cv2  # noqa: F401
    except Exception:
        import pytest
        pytest.skip("opencv not available")
    f = np.full((6, 32, 32, 3), 128, dtype=np.uint8)
    s = motion_score(f)
    assert s["mean"] < 1e-3
