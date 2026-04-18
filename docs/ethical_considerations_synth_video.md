# Ethical considerations for synthetic short video

Short video generation is fun and useful for content prototyping, but it also
sits at the intersection of several real harms. Some notes on how we think
about it.

## Deepfake risk

The stack here is not trained for identity swaps and does not include a face
LoRA workflow. But short-video generation of a plausible human speaking is a
misuse vector. If you fork this, do not add face-conditioning without also
adding provenance signals (C2PA metadata is cheap to embed on the output mp4)
and a clear no-deepfake statement in your README.

## Copyright and style laundering

Per-style LoRAs are easy to train on a small clip set. It is also easy to
train them on someone's copyrighted footage and produce derivative work that
skirts credit. Style LoRAs shipped with this repo are trained on
permissively-licensed clip sets only. Style transfer of a living artist's
work without permission is out of scope and not encouraged.

## Bias and stereotype amplification

SD3-Turbo inherits biases from its pretraining data (gender / occupation
skews, western-centric aesthetics). Motion training does not fix these and
can compound them for action-verb prompts. Evaluate for bias on any downstream
use before shipping in a product.

## Environmental cost

A 20k-step motion training run is ~12 H100-hours (~$25-40 depending on your
provider). Full retrains per iteration compound. Reuse the pretrained motion
adapter across styles; only fine-tune LoRA per style.

## What we recommend

- Add a watermark or C2PA provenance to generated mp4s.
- Publish training data provenance for any style LoRA you release.
- Add an allowlist of prompts your product will accept (no named individuals,
  no sexual content, no self-harm) in front of the API.
- Log requests with a takedown / abuse contact.
