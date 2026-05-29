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
- `tests/test_motion_module.py`: unit tests for the motion module (shape,
  dtype, frame generalization, `motion_scale=0` passthrough).

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
pip install -r requirements.txt
pytest -q tests/test_motion_module.py
```

## Repo layout

```
src/
  model/     motion_module.py  sd3_backbone.py  integrated_pipe.py
  train/     train_motion.py   distill_lora.py    (drafts)
  data/      webvid_subset.py  augment.py
  inference/ t2v.py  batch_t2v.py                 (drafts)
  eval/      fvd_score.py  clipsim_temporal.py  motion_score.py  (drafts)
  api/       main.py                              (draft)
configs/     base.yaml  style_anime.yaml  style_cinematic.yaml
scripts/     train_motion.sh  distill_lora.sh  generate.sh  eval_fvd.sh  prep_webvid.sh
tests/       test_motion_module.py
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
