import torch

from src.model.motion_module import (
    MotionModule,
    TemporalAttentionBlock,
    sinusoidal_pos_embed,
)


def test_temporal_block_shape():
    x = torch.randn(3, 8, 64)
    blk = TemporalAttentionBlock(channels=64, heads=4, head_dim=16)
    y = blk(x)
    assert y.shape == x.shape


def test_pos_embed_shape():
    e = sinusoidal_pos_embed(32, 128)
    assert e.shape == (32, 128)


def test_motion_module_forward_shape():
    b, f, c, h, w = 2, 6, 32, 8, 8
    m = MotionModule(channels=c, num_layers=2, heads=4, head_dim=8, max_frames=16)
    x = torch.randn(b * f, c, h, w)
    y = m(x, num_frames=f)
    assert y.shape == x.shape


def test_motion_module_identity_at_init():
    """Zero-init FFN + attention residual means output should be x + small delta.
    We check the residual math: at init, ffn contributes zero exactly."""
    b, f, c, h, w = 1, 4, 16, 4, 4
    m = MotionModule(channels=c, num_layers=1, heads=2, head_dim=8, max_frames=8)
    # zero the attention output too so we can assert bitwise-identity
    for blk in m.blocks:
        for p in blk.to_out[0].parameters():
            torch.nn.init.zeros_(p)
    x = torch.randn(b * f, c, h, w)
    y = m(x, num_frames=f, motion_scale=1.0)
    assert torch.allclose(y, x, atol=1e-6)


def test_motion_scale_zero_is_identity():
    b, f, c, h, w = 1, 4, 16, 4, 4
    m = MotionModule(channels=c, num_layers=2, heads=2, head_dim=8, max_frames=8)
    x = torch.randn(b * f, c, h, w)
    y = m(x, num_frames=f, motion_scale=0.0)
    assert torch.allclose(y, x)
