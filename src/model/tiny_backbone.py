"""
Tiny CPU stand-in backbone for the offline plumbing smoke.

This is NOT SD3-Turbo. It is a small token-stream transformer that stands in
for a frozen image diffusion transformer, so the AnimateDiff-style motion
module and its hook plumbing can be exercised end to end on CPU with tiny
synthetic tensors: no diffusers, no model downloads, no GPU, no WebVid.

Design notes:
- The backbone consumes token streams shaped (B*F, S, C), the same layout a
  real DiT block emits, so the temporal hook in `sd3_backbone.install_motion_hooks`
  runs unchanged (it reshapes S -> h*w, applies temporal attention over the
  frame axis, reshapes back).
- The spatial blocks are real multi-head-attention + FFN blocks, but their
  output projections are zero initialized so the frozen backbone starts as an
  identity map. That mirrors the realistic setting where the backbone is a
  frozen, already-trained reconstructor and the trainable temporal adapter
  does all the temporal work. It also makes any loss reduction attributable to
  the motion module rather than to the (frozen) backbone.

Real text-to-video needs a GPU and a real SD3 backbone. See README.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .motion_module import MotionModule
from .sd3_backbone import install_motion_hooks, remove_hooks, HookHandle


class _SpatialBlock(nn.Module):
    """Tiny spatial self-attention + FFN block over the token (S) axis.

    Operates on (B*F, S, C) so the block output matches the token-stream layout
    the motion hook expects. Output projections are zero initialized -> identity
    at init.
    """

    def __init__(self, channels: int, heads: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        # zero-init output paths so the frozen backbone is identity at init
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class TinyDiTBackbone(nn.Module):
    """Stand-in frozen backbone.

    Exposes `.transformer_blocks` so the real `install_motion_hooks` code path
    (from `sd3_backbone`) discovers and hooks the blocks with no changes.
    """

    def __init__(self, channels: int = 32, num_blocks: int = 4, heads: int = 2, grid: int = 4):
        super().__init__()
        self.channels = channels
        self.grid = grid  # tokens per side; sequence length S = grid * grid
        self.transformer_blocks = nn.ModuleList(
            [_SpatialBlock(channels, heads=heads) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*F, S, C)
        for blk in self.transformer_blocks:
            x = blk(x)
        return x


class TinyT2VPipe:
    """CPU-only text-to-video plumbing stand-in.

    Wires a frozen `TinyDiTBackbone` and a trainable `MotionModule` through the
    same hook mechanism used for the real SD3 path. Provides a differentiable
    forward for training the motion module and a `generate` for producing a
    tiny frame sequence.
    """

    def __init__(
        self,
        channels: int = 32,
        num_blocks: int = 4,
        heads: int = 2,
        grid: int = 4,
        motion_layers: int = 2,
        motion_heads: int = 2,
        motion_head_dim: int = 8,
        max_frames: int = 32,
        every_n_blocks: int = 2,
        seed: Optional[int] = 0,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.device = "cpu"
        self.dtype = torch.float32
        self.channels = channels
        self.grid = grid
        self.seq_len = grid * grid
        self.every_n_blocks = every_n_blocks

        # `.transformer` mirrors the diffusers pipe attribute the hook installer reads
        self.transformer = TinyDiTBackbone(
            channels=channels, num_blocks=num_blocks, heads=heads, grid=grid
        )
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.motion = MotionModule(
            channels=channels,
            num_layers=motion_layers,
            heads=motion_heads,
            head_dim=motion_head_dim,
            max_frames=max_frames,
        )
        self._hooks: List[HookHandle] = []

    def frozen_backbone_params(self) -> int:
        return sum(p.numel() for p in self.transformer.parameters())

    def trainable_motion_params(self) -> int:
        return sum(p.numel() for p in self.motion.parameters() if p.requires_grad)

    def enable_motion(self, num_frames: int, motion_scale: float = 1.0):
        self.disable_motion()
        self._hooks = install_motion_hooks(
            self,
            self.motion,
            every_n_blocks=self.every_n_blocks,
            num_frames=num_frames,
            motion_scale=motion_scale,
        )
        return len(self._hooks)

    def disable_motion(self):
        remove_hooks(self._hooks)

    def forward_tokens(
        self, tokens: torch.Tensor, num_frames: int, motion_scale: float = 1.0
    ) -> torch.Tensor:
        """Run frozen backbone + motion hooks. tokens: (B*F, S, C) -> (B*F, S, C)."""
        self.enable_motion(num_frames=num_frames, motion_scale=motion_scale)
        try:
            out = self.transformer(tokens)
        finally:
            self.disable_motion()
        return out

    def _tokens_to_frames(self, tokens: torch.Tensor) -> torch.Tensor:
        """(F, S, C) -> (F, grid, grid, 3) uint8 for a tiny visual frame sequence."""
        f, s, c = tokens.shape
        g = self.grid
        img = tokens.reshape(f, g, g, c)[..., :3]
        lo = img.amin()
        hi = img.amax()
        img = (img - lo) / (hi - lo + 1e-6)
        return (img * 255.0).clamp(0, 255).to(torch.uint8)

    @torch.no_grad()
    def generate(
        self, num_frames: int, seed: Optional[int] = None, motion_scale: float = 1.0
    ) -> torch.Tensor:
        """Produce a tiny frame sequence (num_frames, grid, grid, 3) uint8."""
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        latents = torch.randn(num_frames, self.seq_len, self.channels, generator=gen)
        out = self.forward_tokens(latents, num_frames=num_frames, motion_scale=motion_scale)
        return self._tokens_to_frames(out)
