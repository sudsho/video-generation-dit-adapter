"""
FastAPI serving: POST /generate returns a small mp4 encoded as a stream.

Backend loads the pipe once on startup; requests share the pipe (single GPU,
one request in flight at a time; use a queue in front for multi-user).
"""
from __future__ import annotations

import io
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import imageio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# lazy-import to keep the API testable without torch loaded
_PIPE = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512)
    duration: float = Field(4.0, ge=1.0, le=8.0)
    fps: int = Field(12, ge=6, le=24)
    cfg_scale: float = Field(7.5, ge=0.0, le=20.0)
    motion_scale: float = Field(1.0, ge=0.0, le=2.0)
    style: Optional[str] = Field(None, description="anime|cinematic|null")
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    id: str
    frames: int
    duration: float
    fps: int
    style: Optional[str]
    url: str


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        # heavy import happens here
        import torch
        from src.inference.t2v import _build_pipe, _load_cfg
        cfg = _load_cfg(os.environ.get("CONFIG_PATH", "configs/base.yaml"))
        _PIPE = _build_pipe(cfg, os.environ.get("MOTION_CKPT", "runs/motion/motion_final.pt"))
    return _PIPE


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm the pipe if requested
    if os.environ.get("WARMUP") == "1":
        _get_pipe()
    yield


app = FastAPI(title="video-generation-dit-adapter", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    from src.model import GenerationConfig
    from src.inference.t2v import write_mp4, STYLE_LORAS, _load_cfg
    pipe = _get_pipe()
    if req.style:
        lora = STYLE_LORAS.get(req.style)
        if lora and os.path.exists(lora):
            pipe.load_style_lora(lora)
        else:
            pipe.clear_loras()

    cfg = _load_cfg(os.environ.get("CONFIG_PATH", "configs/base.yaml"))
    num_frames = int(round(req.duration * req.fps))
    gcfg = GenerationConfig(
        num_frames=num_frames, fps=req.fps,
        height=cfg["resolution"], width=cfg["resolution"],
        cfg_scale=req.cfg_scale, motion_scale=req.motion_scale,
        num_inference_steps=cfg["inference"]["num_inference_steps"],
        seed=req.seed, style=req.style,
    )
    frames = pipe.generate(req.prompt, gcfg)
    vid_id = uuid.uuid4().hex[:12]
    out_dir = os.environ.get("OUT_DIR", "out")
    out_path = os.path.join(out_dir, f"{vid_id}.mp4")
    write_mp4(frames, out_path, fps=req.fps)
    return GenerateResponse(
        id=vid_id, frames=num_frames, duration=req.duration, fps=req.fps,
        style=req.style, url=f"/videos/{vid_id}.mp4",
    )


@app.get("/videos/{name}")
def get_video(name: str):
    path = os.path.join(os.environ.get("OUT_DIR", "out"), name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="video/mp4")
