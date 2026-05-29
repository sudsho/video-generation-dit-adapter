"""
DRAFT backbone loader and hook installer.

Loads a diffusers StableDiffusion3Pipeline by id (caller must supply a
real, downloadable model id; `stabilityai/stable-diffusion-3-turbo` used
as a default here is a placeholder that does NOT resolve on Hugging Face)
and registers forward hooks on selected transformer blocks.

Caveat on the hook: diffusers' SD3 JointTransformerBlock.forward returns
`(encoder_hidden_states, hidden_states)` - i.e. `output[0]` is the TEXT
stream, not the image residual stream. The hook below currently treats
`output[0]` as the image stream, which is wrong for that specific block
layout and needs to be reconciled against the actual pipeline before use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn

try:
    from diffusers import StableDiffusion3Pipeline
except Exception:  # pragma: no cover
    StableDiffusion3Pipeline = None


@dataclass
class HookHandle:
    layer_name: str
    handle: torch.utils.hooks.RemovableHandle


def load_sd3_turbo(
    model_id: str = "",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    if StableDiffusion3Pipeline is None:
        raise RuntimeError("diffusers not available; install requirements.txt")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def freeze_backbone(pipe) -> int:
    n = 0
    for p in pipe.transformer.parameters():
        p.requires_grad_(False)
        n += p.numel()
    return n


def install_motion_hooks(
    pipe,
    motion_module: nn.Module,
    every_n_blocks: int = 2,
    num_frames: int = 48,
    motion_scale: float = 1.0,
) -> List[HookHandle]:
    """
    Register forward hooks on selected transformer blocks. The hook runs the
    motion module on the block output and returns the residual mix.

    Assumes the diffusers SD3 transformer exposes .transformer_blocks. Falls
    back to a name-based scan if not.
    """
    handles: List[HookHandle] = []

    blocks = getattr(pipe.transformer, "transformer_blocks", None)
    if blocks is None:
        blocks = [m for n, m in pipe.transformer.named_modules() if "block" in n.lower()]

    def make_hook(block_name: str) -> Callable:
        def hook(_module, _inputs, output):
            # output can be a Tensor or a tuple (hidden, ...)
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            # hidden here is a token stream (B*F, seq_len, C). We need to
            # reshape to (B*F, C, H, W) if the block gives us image-like
            # features; otherwise apply the temporal pass in the (B*F, seq, C)
            # layout by treating each token independently.
            if hidden.dim() == 3:
                # (B*F, S, C) -> transpose to (B*F, C, S) then treat S as 1x?
                bf, s, c = hidden.shape
                h = w = int(round(s ** 0.5))
                if h * w != s:
                    # not a square patch grid, fall back to (S, 1)
                    h, w = s, 1
                z = hidden.transpose(1, 2).reshape(bf, c, h, w)
                z = motion_module(z, num_frames=num_frames, motion_scale=motion_scale)
                hidden = z.reshape(bf, c, h * w).transpose(1, 2)
            elif hidden.dim() == 4:
                hidden = motion_module(hidden, num_frames=num_frames, motion_scale=motion_scale)
            if rest:
                return (hidden,) + rest
            return hidden

        return hook

    for i, blk in enumerate(blocks):
        if i % every_n_blocks != 0:
            continue
        handle = blk.register_forward_hook(make_hook(f"block_{i}"))
        handles.append(HookHandle(layer_name=f"block_{i}", handle=handle))

    return handles


def remove_hooks(handles: List[HookHandle]) -> None:
    for h in handles:
        h.handle.remove()
    handles.clear()
