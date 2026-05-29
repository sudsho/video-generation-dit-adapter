# Per-style LoRA (intended)

No LoRA has been trained in this repo. The `src/train/distill_lora.py`
script is a sketch and has not been reconciled against a real pipeline.
The two style config yaml files are placeholders documenting the intended
LoRA shape.

## Intended shape

- Keep one motion adapter across all styles (motion is style agnostic in
  this setup) and swap tiny LoRA weights into the DiT attention
  projections to change the look.
- Rank 8-16 into `to_q`, `to_k`, `to_v`, `to_out.0` of the attention
  projections.

## Missing pieces before this can run

- The LoRA trainer imports `peft` at run time and raises if it is not
  installed; `peft` is not declared in `requirements.txt`.
- The save path in `distill_lora.py` (`save_pretrained(out/'lora')`) and
  the load path in `src/inference/t2v.py` (`runs/lora/anime/lora.safetensors`)
  do not match, so a trained adapter would not be found by the inference
  CLI without further work.
- The training loop calls `scheduler.add_noise(...)` and
  `pipe.encode_prompt(captions)` in shapes / signatures that do not match
  the released SD3 pipeline; those calls would need to be rewritten
  against the actual pipeline chosen.
