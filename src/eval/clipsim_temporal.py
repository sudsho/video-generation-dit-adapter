"""
Per-frame CLIP alignment: for each frame, compute cosine sim(text, image),
then aggregate mean + min. A drop between mean and min flags a frame that
drifted off-prompt.
"""
from __future__ import annotations

from typing import List

import numpy as np

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover
    torch = None
    CLIPModel = None
    CLIPProcessor = None


class CLIPTemporal:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str | None = None):
        if CLIPModel is None:
            raise RuntimeError("transformers not installed")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(self, prompt: str, frames_uint8: np.ndarray) -> dict:
        # frames_uint8: (F, H, W, 3)
        images = [f for f in frames_uint8]
        batch = self.proc(text=[prompt], images=images, return_tensors="pt", padding=True).to(self.device)
        text_emb = self.model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        img_emb = self.model.get_image_features(pixel_values=batch["pixel_values"])
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
        img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
        sims = (img_emb @ text_emb.T).squeeze(-1).cpu().numpy()
        return {
            "mean": float(sims.mean()),
            "min": float(sims.min()),
            "max": float(sims.max()),
            "std": float(sims.std()),
            "per_frame": sims.tolist(),
        }
