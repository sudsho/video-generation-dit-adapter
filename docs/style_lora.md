# Per-style LoRA

## Rationale

We keep one motion adapter across all styles (motion is style-agnostic in our
setup) and swap tiny LoRA weights into the DiT attention projections to change
the look. That gives one 30M motion checkpoint plus 4-15MB LoRA files per style.

## Rank choice

- `anime`: r=16 alpha=32. Anime look is a strong OOD shift from SD3-Turbo's
  natural photo prior. Higher rank helps.
- `cinematic`: r=8 alpha=16. Cinematic is closer to base, r=8 is enough.
- We tried r=4 on both. Anime got flat and washed. Cinematic looked fine.

## Training set size

50-500 clips per style, all with style-consistent captions
("anime portrait, cel-shaded, ...", "cinematic shot, teal-and-orange grade, ..."),
is enough to learn a style. More clips = better generalization but diminishing
returns after ~500.

## Combining LoRAs

`pipe.set_adapters([...])` supports weighted combinations. E.g. `anime@0.6 +
cinematic@0.4` produces animated shots with cinematic color grading. Weights
that sum to >1 tend to oversaturate.
