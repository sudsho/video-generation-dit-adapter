# artifacts

Small sample gifs, one per style, for README embedding.

- `sample_default.gif` - base motion module, no style LoRA
- `sample_anime.gif` - anime LoRA
- `sample_cinematic.gif` - cinematic LoRA

These are compressed placeholders; regenerate at full quality by running:

```
python -m src.inference.t2v --prompt "a cat surfing on a rainbow" --out artifacts/sample_default.gif --style ""
```

(swap `--style anime` etc for the other two)
