"""API tests with the heavy generate() call mocked out."""
import os
from unittest.mock import patch

import numpy as np
import pytest


def _get_client():
    from fastapi.testclient import TestClient
    from src.api import main as api_main
    return TestClient(api_main.app), api_main


def test_health():
    client, _ = _get_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generate_validation_rejects_zero_duration():
    client, _ = _get_client()
    r = client.post("/generate", json={"prompt": "hi", "duration": 0.1})
    assert r.status_code == 422


def test_generate_validation_rejects_empty_prompt():
    client, _ = _get_client()
    r = client.post("/generate", json={"prompt": ""})
    assert r.status_code == 422


def test_generate_mocked(tmp_path):
    client, api_main = _get_client()

    class FakePipe:
        def generate(self, prompt, cfg):
            return np.zeros((cfg.num_frames, 32, 32, 3), dtype=np.uint8)
        def load_style_lora(self, *a, **kw): pass
        def clear_loras(self): pass

    os.environ["OUT_DIR"] = str(tmp_path)
    with patch.object(api_main, "_get_pipe", return_value=FakePipe()), \
         patch.object(api_main, "_load_cfg" if hasattr(api_main, "_load_cfg") else "os.path", create=True):
        # patch _load_cfg import path used inside the handler
        with patch("src.inference.t2v._load_cfg", return_value={
            "resolution": 32, "inference": {"num_inference_steps": 6}
        }), patch("src.inference.t2v.write_mp4"):
            r = client.post("/generate", json={"prompt": "cat", "duration": 1.0, "fps": 12})
    assert r.status_code == 200
    j = r.json()
    assert j["frames"] == 12
    assert j["fps"] == 12
