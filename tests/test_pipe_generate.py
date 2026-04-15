"""End-to-end pipe test using a fully mocked diffusers backbone."""
from unittest.mock import MagicMock, patch

import numpy as np


def test_generation_config_defaults():
    from src.model import GenerationConfig
    g = GenerationConfig()
    assert g.num_frames == 48
    assert g.fps == 12
    assert g.height == g.width == 512


def test_pipe_generate_mocked(monkeypatch):
    from src.model import integrated_pipe

    fake_pipe = MagicMock()
    fake_pipe.transformer.transformer_blocks = []
    fake_pipe.transformer.parameters.return_value = []
    fake_pipe.__call__ = MagicMock(return_value=MagicMock(images=np.zeros((6, 32, 32, 3))))

    with patch.object(integrated_pipe, "load_sd3_turbo", return_value=fake_pipe), \
         patch.object(integrated_pipe, "freeze_backbone", return_value=0), \
         patch.object(integrated_pipe, "install_motion_hooks", return_value=[]), \
         patch.object(integrated_pipe, "remove_hooks"):
        # avoid moving MotionModule to cuda
        with patch("torch.cuda.is_available", return_value=False):
            pipe = integrated_pipe.IntegratedT2VPipe(device="cpu", dtype=None)
    # smoke: enable/disable hooks and clear loras don't blow up
    pipe.enable_motion(num_frames=6)
    pipe.disable_motion()
    pipe.clear_loras()


def test_style_loras_map():
    from src.inference.t2v import STYLE_LORAS
    assert "anime" in STYLE_LORAS and "cinematic" in STYLE_LORAS
