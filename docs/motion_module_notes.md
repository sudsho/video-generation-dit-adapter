# Motion module notes

Design notes and their intended tradeoffs. This is scaffold-level
documentation; nothing here has been validated against a real backbone
training run.

## Why temporal-only attention (not full 3D)?

Full spatial-temporal attention is O((F*H*W)^2). For a 48-frame, 64x64
latent grid that is on the order of billions of pairs per head. Temporal
only pays O(F^2) per pixel which is manageable and matches what
AnimateDiff and similar architectures use.

## Zero initialization

The output projections of the temporal attention block and the final FFN
linear are zero initialized. Note the caveat in `docs/method.md`: because
the sinusoidal frame position embedding is added into the pre attention
tensor, the module is not strictly identity at init when
`motion_scale != 0`. That's a bug to fix (either delay the pos embed, or
warmup `motion_scale`) if strict identity-at-init is required.

## Hook density (intended)

Hooking every N transformer blocks instead of every block would be a knob
to trade quality against parameter count and VRAM. No ablation has been
run in this repo.

## Sinusoidal positions

Sinusoidal frame positions generalize to any frame count up to the
buffered `max_frames`; the module also regenerates the buffer on demand
for longer clips (note this reassignment mutates the module and is not
side effect free for subsequent smaller-frame calls).

## Parameter count (measured, default config)

With `channels=320, num_layers=4, heads=8, head_dim=40`, `MotionModule`
instantiates roughly 2.46M parameters. Any larger number in earlier drafts
of these notes was aspirational and did not correspond to the shipped
default.
