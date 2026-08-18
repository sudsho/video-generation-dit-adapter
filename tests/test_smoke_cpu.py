"""Tiny-CPU plumbing smoke: motion module trains and generate runs offline.

No GPU, no diffusers, no downloads. Verifies the temporal-consistency plumbing
end to end on tiny synthetic tensors.
"""
import numpy as np
import torch

from src.model.tiny_backbone import TinyT2VPipe
from src.smoke_cpu import make_synthetic_video_tokens, run_smoke


def test_hooks_install_and_remove():
    pipe = TinyT2VPipe(channels=16, num_blocks=4, grid=4, seed=0)
    n = pipe.enable_motion(num_frames=4, motion_scale=1.0)
    assert n >= 1  # at least one motion hook fired into the frozen backbone
    pipe.disable_motion()
    assert pipe._hooks == []


def test_backbone_frozen_motion_trainable():
    pipe = TinyT2VPipe(channels=16, num_blocks=4, grid=4, seed=0)
    assert all(not p.requires_grad for p in pipe.transformer.parameters())
    assert pipe.trainable_motion_params() > 0


def test_forward_tokens_shape_preserved():
    pipe = TinyT2VPipe(channels=16, num_blocks=4, grid=4, seed=0)
    frames = 5
    tokens = torch.randn(2 * frames, pipe.seq_len, pipe.channels)
    out = pipe.forward_tokens(tokens, num_frames=frames, motion_scale=1.0)
    assert out.shape == tokens.shape


def test_synthetic_tokens_are_temporally_smooth():
    clean, noisy = make_synthetic_video_tokens(2, 6, 16, 8, sigma=0.8, seed=1)
    # adjacent clean frames are more similar than random pairs -> temporal structure
    adjacent = (clean[:, 1:] - clean[:, :-1]).pow(2).mean()
    shuffled = (clean[:, 1:] - clean[:, :-1].flip(0)).pow(2).mean()
    assert adjacent < shuffled


def test_generate_shape_and_dtype():
    pipe = TinyT2VPipe(channels=16, num_blocks=4, grid=4, seed=0)
    frames = pipe.generate(num_frames=6, seed=123, motion_scale=1.0)
    assert frames.shape == (6, 4, 4, 3)
    assert frames.dtype == torch.uint8


def test_motion_changes_output():
    pipe = TinyT2VPipe(channels=16, num_blocks=4, grid=4, seed=0)
    on = pipe.generate(num_frames=6, seed=123, motion_scale=1.0)
    off = pipe.generate(num_frames=6, seed=123, motion_scale=0.0)
    # temporal module must actually participate in the output
    assert not torch.equal(on, off)


def test_temporal_loss_decreases():
    res = run_smoke(steps=40, seed=0, verbose=False)
    assert res.num_hooks >= 1
    assert res.last_loss < 0.6 * res.first_loss  # clear reduction from temporal denoising
    assert res.frames_shape == (6, 4, 4, 3)
    assert res.motion_on_off_l1 > 0.0
