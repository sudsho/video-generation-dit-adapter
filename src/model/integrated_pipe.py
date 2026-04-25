"""
End-to-end text-to-video pipeline. Wraps SD3-Turbo + motion module + optional
LoRA style adapter behind a single .generate() call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import numpy as np

from .motion_module import MotionModule
from .sd3_backbone import (
    load_sd3_turbo,
    freeze_backbone,
    install_motion_hooks,
    remove_hooks,
    HookHandle,
)


@dataclass
class GenerationConfig:
    num_frames: int = 48
    fps: int = 12
    height: int = 512
    width: int = 512
    cfg_scale: float = 7.5
    motion_scale: float = 1.0
    num_inference_steps: int = 6
    seed: Optional[int] = None
    style: Optional[str] = None


class IntegratedT2VPipe:
    def __init__(
        self,
        backbone_id: str = "stabilityai/stable-diffusion-3-turbo",
        motion_channels: int = 320,
        motion_layers: int = 4,
        heads: int = 8,
        head_dim: int = 40,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.device = device
        self.dtype = dtype
        self.pipe = load_sd3_turbo(backbone_id, dtype=dtype, device=device)
        freeze_backbone(self.pipe)
        self.motion = MotionModule(
            channels=motion_channels,
            num_layers=motion_layers,
            heads=heads,
            head_dim=head_dim,
        ).to(device=device, dtype=dtype)
        self._hooks: List[HookHandle] = []
        self._loras: List[str] = []

    def enable_motion(self, num_frames: int, motion_scale: float = 1.0, every_n: int = 2):
        self.disable_motion()
        self._hooks = install_motion_hooks(
            self.pipe,
            self.motion,
            every_n_blocks=every_n,
            num_frames=num_frames,
            motion_scale=motion_scale,
        )

    def disable_motion(self):
        remove_hooks(self._hooks)

    def load_style_lora(self, path: str, adapter_name: str = "style", scale: float = 1.0):
        """Delegates to diffusers' PEFT-backed LoRA loader on the transformer.
        Safe to call multiple times with different adapter names.
        """
        if adapter_name in self._loras:
            # already loaded, just adjust weight
            self.pipe.set_adapters([adapter_name], adapter_weights=[scale])
            return
        self.pipe.load_lora_weights(path, adapter_name=adapter_name)
        self.pipe.set_adapters([adapter_name], adapter_weights=[scale])
        self._loras.append(adapter_name)

    def combine_loras(self, weights: dict[str, float]) -> None:
        """Set weighted blend of multiple loaded LoRAs, e.g. {'anime':0.6,'cinematic':0.4}."""
        names = list(weights.keys())
        w = [weights[n] for n in names]
        self.pipe.set_adapters(names, adapter_weights=w)

    def clear_loras(self):
        if self._loras and hasattr(self.pipe, "unload_lora_weights"):
            self.pipe.unload_lora_weights()
        self._loras.clear()

    @torch.no_grad()
    def generate(self, prompt: Union[str, List[str]], cfg: GenerationConfig) -> np.ndarray:
        if isinstance(prompt, str):
            prompts = [prompt] * cfg.num_frames
        else:
            assert len(prompt) == cfg.num_frames, "prompt list must match num_frames"
            prompts = prompt

        gen = None
        if cfg.seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(cfg.seed)

        self.enable_motion(num_frames=cfg.num_frames, motion_scale=cfg.motion_scale)
        try:
            out = self.pipe(
                prompt=prompts,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.cfg_scale,
                generator=gen,
                output_type="np",
            )
        finally:
            self.disable_motion()

        # out.images: (num_frames, H, W, 3) float in [0,1]
        frames = out.images
        return (frames * 255).clip(0, 255).astype(np.uint8)
