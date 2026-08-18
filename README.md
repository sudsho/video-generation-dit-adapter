# video-generation-dit-adapter

Architectural sketch for an AnimateDiff-style temporal motion module that
would sit on top of a frozen image diffusion transformer (DiT) backbone,
with per-style LoRA adapters as a hypothetical extension.

Status: research scaffold. The motion module class and its shape / dtype /
generalization tests run standalone. The training loop, inference CLI, FastAPI
server, Streamlit demo, LoRA distiller and FVD harness are early drafts and
have not been wired to any real diffusion pipeline end to end. No
checkpoints, benchmark artifacts, or measured metrics are shipped in this
repo.

There is a small offline plumbing smoke that runs the motion module end to
end on CPU against a tiny stand-in backbone, so the temporal hook path is
actually executed and verified without a GPU or any downloads. See "Quick
start (runs offline)" below. This smoke is plumbing only; it is not a
text-to-video result.

## Quick start (runs offline)

No GPU, no diffusers, no model downloads, no WebVid. The smoke builds a tiny
frozen stand-in backbone (a small attention net, NOT SD3), wires a tiny
trainable motion module through the same hook path used for the real SD3
backbone, trains a few steps on synthetic temporally-smooth video latents so
the temporal denoising loss drops, then runs a generate step that produces a
small frame sequence and confirms the temporal module participates.

```bash
pip install torch einops numpy pyyaml   # already present in the shared env
python -m src.smoke_cpu
```

Real output on CPU (torch 2.5.1, Python 3.11):

```
step   0  temporal_denoise_loss 2.6521
step  10  temporal_denoise_loss 0.7954
step  20  temporal_denoise_loss 0.6990
step  30  temporal_denoise_loss 0.5390
step  40  temporal_denoise_loss 0.4466
step  50  temporal_denoise_loss 0.3928
step  59  temporal_denoise_loss 0.3526

frozen backbone params : 34176
trainable motion params: 12704
motion hooks installed : 2
temporal denoise loss  : 2.6521 -> 0.3526 (86.7% reduction)
generated frames shape : (6, 4, 4, 3)  (F, H, W, 3)
motion on vs off L1    : 32.28 (non-zero => temporal-consistency plumbing ran)

SMOKE PASS
```

Tests (all offline, one skip when opencv is absent):

```bash
pytest -q
# 22 passed, 1 skipped
```

What the smoke proves: the motion module reshapes token streams to the
`(B*H*W, F, C)` temporal layout, the sinusoidal frame positions and temporal
attention run, the hook installer registers and removes forward hooks on the
backbone blocks, gradients flow only into the motion module while the backbone
stays frozen, temporal averaging measurably reduces a denoising loss, and the
generate path emits a frame sequence of the correct shape whose output changes
when the temporal module is enabled.

What it does NOT prove: this is a tiny synthetic stand-in, not SD3-Turbo. It
says nothing about real text-to-video quality, FVD, or motion fidelity. The
headline result (text prompt to a coherent video clip) needs a GPU and a real
SD3-Turbo backbone plus the reconciliation work listed in the Roadmap. The
diffusers / SD3 / xformers / WebVid code paths are guarded so importing and
running the smoke never touches them.

## What is actually here

- `src/model/motion_module.py`: standalone temporal self attention block +
  MotionModule wrapper that reshapes `(B*F, C, H, W)` latents to
  `(B*H*W, F, C)`, adds sinusoidal frame positions, runs a small stack of
  temporal attention + FFN, and mixes back with a `motion_scale` residual.
- `src/model/sd3_backbone.py`, `src/model/integrated_pipe.py`: draft
  scaffolding that would hook a frozen diffusion transformer. The
  `install_motion_hooks` code path has NOT been validated against the
  hidden state / encoder state tuple layout of any specific released
  diffusion transformer, so it should not be assumed to touch the video
  residual stream correctly.
- `src/train/train_motion.py`, `src/train/distill_lora.py`: sketches that
  call diffusers pipeline APIs. They have not been reconciled against the
  actual method signatures of any specific pipeline / scheduler version,
  and no training run has been performed with them.
- `src/inference/t2v.py`, `src/api/main.py`, `streamlit_app.py`,
  `Dockerfile`, `docker-compose.yml`: reference wiring only. They will not
  run without a working pipeline + trained checkpoint neither of which is
  provided.
- `src/model/tiny_backbone.py`: a tiny CPU stand-in backbone (NOT SD3) plus a
  `TinyT2VPipe` that wires the motion module through the real hook installer.
  This exists only to exercise the plumbing offline.
- `src/smoke_cpu.py`: the offline smoke entrypoint (`python -m src.smoke_cpu`).
- `tests/test_motion_module.py`: unit tests for the motion module (shape,
  dtype, frame generalization, `motion_scale=0` passthrough).
- `tests/test_smoke_cpu.py`: end-to-end offline plumbing tests (hooks install
  and remove, backbone frozen while motion trains, temporal loss decreases,
  generate shape and dtype, motion actually changes the output).

## Method (sketch)

- Hook the forward pass of selected transformer blocks in a spatial DiT.
- Insert a temporal self attention stack over the frame axis at each hook
  point, mixed back into the block output as a residual.
- Would train the motion module on a WebVid-style clip manifest with a
  frozen backbone, using a diffusion-style regression loss appropriate to
  the backbone's parameterization (that reconciliation is not done here).
- Per style, would train a small LoRA (r=8-16) into the attention
  projections and load one adapter at a time at inference.

## Setup

```bash
pip install -r requirements.txt   # full stack (diffusers, SD3, etc.)
pytest -q
```

The offline smoke and its tests only need `torch`, `einops`, `numpy`, and
`pyyaml`; they do not import diffusers. See "Quick start (runs offline)".

## Repo layout

```
src/
  model/     motion_module.py  sd3_backbone.py  integrated_pipe.py  tiny_backbone.py
  smoke_cpu.py                                  (offline plumbing smoke)
  train/     train_motion.py   distill_lora.py    (drafts)
  data/      webvid_subset.py  augment.py
  inference/ t2v.py  batch_t2v.py                 (drafts)
  eval/      fvd_score.py  clipsim_temporal.py  motion_score.py  (drafts)
  api/       main.py                              (draft)
configs/     base.yaml  style_anime.yaml  style_cinematic.yaml
scripts/     train_motion.sh  distill_lora.sh  generate.sh  eval_fvd.sh  prep_webvid.sh
tests/       test_motion_module.py  test_smoke_cpu.py  test_pipe_generate.py
             test_metrics.py  test_api.py
docs/        method.md  motion_module_notes.md  style_lora.md
             ethical_considerations_synth_video.md
```

## Ethics

Generative video has real harm surface: deepfakes, style laundering,
non-consensual content. See
[docs/ethical_considerations_synth_video.md](docs/ethical_considerations_synth_video.md)
for what would need to be added before shipping anything in this direction
as a product.

## Roadmap

- Reconcile the hook path against a specific released DiT pipeline (verify
  the residual-stream tuple layout, dimension width, and scheduler /
  encode_prompt / transformer signatures) before treating training as
  runnable.
- Add an actual I3D checkpoint plumb, or drop the FVD harness entirely.
- Write an end-to-end run that produces a real video output before
  publishing any benchmark numbers.

## License

MIT.
