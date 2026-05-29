"""
Temporal / motion module inserted between spatial diffusion blocks.

Roughly AnimateDiff-style: given a latent tensor shaped
(batch * num_frames, channels, height, width), reshape to
(batch * height * width, num_frames, channels) and apply causal-free
self attention along the temporal axis. Optional sinusoidal position
embeddings along the frame dimension.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def sinusoidal_pos_embed(num_positions: int, dim: int, device=None) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )
    t = torch.arange(num_positions, device=device).float().unsqueeze(1)
    args = t * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class TemporalAttentionBlock(nn.Module):
    """Multi-head self attention along the temporal axis only."""

    def __init__(self, channels: int, heads: int = 8, head_dim: int = 40, dropout: float = 0.0):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(channels)
        self.to_qkv = nn.Linear(channels, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, channels), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, C)
        residual = x
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "n t (h d) -> n h t d", h=self.heads) for t in qkv)
        # prefer sdpa when available (calls into xformers/flash under the hood)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v
        out = rearrange(out, "n h t d -> n t (h d)")
        return self.to_out(out) + residual


class MotionModule(nn.Module):
    """
    A stack of temporal attention blocks + FFN, applied to (B*F, C, H, W) latents.

    Output projections of the attention blocks and of the final FFN linear
    are zero initialized. Note: the sinusoidal frame position embedding is
    added before the attention stack, so the module is not strictly
    identity at init when motion_scale != 0; use motion_scale=0 for a true
    passthrough.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 4,
        heads: int = 8,
        head_dim: int = 40,
        max_frames: int = 64,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.max_frames = max_frames
        self.register_buffer(
            "pos_emb", sinusoidal_pos_embed(max_frames, channels), persistent=False
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TemporalAttentionBlock(channels, heads=heads, head_dim=head_dim))

        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * ffn_mult),
            nn.GELU(),
            nn.Linear(channels * ffn_mult, channels),
        )
        # zero-init the last projection (identity at start of training)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        # zero-init attention output projection too, matches animatediff init
        for blk in self.blocks:
            nn.init.zeros_(blk.to_out[0].weight)
            nn.init.zeros_(blk.to_out[0].bias)

    def forward(self, x: torch.Tensor, num_frames: int, motion_scale: float = 1.0) -> torch.Tensor:
        # x is expected as (B*F, C, H, W)
        bf, c, h, w = x.shape
        if bf % num_frames != 0:
            # nothing sensible to do; passthrough
            return x
        b = bf // num_frames
        if num_frames > self.pos_emb.shape[0]:
            # generalize sinusoidal embed on the fly (still deterministic)
            self.pos_emb = sinusoidal_pos_embed(num_frames, self.channels, device=x.device).to(x.dtype)

        # reshape to (B*H*W, F, C)
        z = rearrange(x, "(b f) c h w -> (b h w) f c", b=b, f=num_frames)

        # add temporal position embedding
        z = z + self.pos_emb[:num_frames].unsqueeze(0)

        for blk in self.blocks:
            z = blk(z)

        # Pre-LN style FFN with residual; ffn is zero-init so identity at start.
        z = z + self.ffn(z)

        z = rearrange(z, "(b h w) f c -> (b f) c h w", b=b, h=h, w=w)
        # motion_scale = 0 disables the module at inference, 1 = trained default
        return x + motion_scale * (z - x)
