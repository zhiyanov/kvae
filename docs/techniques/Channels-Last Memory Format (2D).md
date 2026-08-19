---
tags: [technique]
aliases: [channels_last, NHWC]
---

# Channels-Last Memory Format (2D)

**Verdict: REJECTED — measured 0.43–0.79× in eager mode** ([[Optimization Results]], round 3).
The original reasoning below assumed conv kernels dominate; in practice the model's
replicate/reflect pads and GroupNorm lack fast NHWC kernels and eager NHWC is up to 2.3×
*slower* (peak memory does drop 20–37%). It also hurts under compile (1.28× vs 1.63× without) —
inductor picks its own layouts internally, so pre-converting is counterproductive. Kept below
for the record; revisit only if the model ever moves to zero padding + channel-only norms.

**Original analysis** — the video repo kept everything in `channels_last_3d` (NDHWC); the 2D
equivalent is plain `torch.channels_last` (NHWC), which is *better* supported than the 3D
variant — cuDNN's fastest bf16 tensor-core conv kernels are NHWC-native.

## Port

```python
vae = vae.to(memory_format=torch.channels_last)         # converts Conv2d weights
images = images.to(memory_format=torch.channels_last)   # per batch, in the inference loop
```

(For `nn.Conv2d`, `module.to(memory_format=...)` converts weights correctly — the dedicated
`convert_conv3d_weight_memory_format` helper the video repo needed is a 3D-only workaround.)

## Why it matters here

- Every conv in [[Encoder2D]]/[[Decoder2D]] is a 3×3 or 1×1 `nn.Conv2d` in bf16 — the exact
  case where NHWC + tensor cores wins and NCHW forces either slower kernels or hidden
  transposes.
- [[GroupNorm]] and the `DecoderSpatialNorm2D` 1×1 convs reduce/operate along channels — in
  NHWC the channel axis is contiguous per pixel.
- Mixing layouts is the failure mode: one NCHW-only op mid-network inserts two implicit
  transposes — full extra read+writes of the activation, which on a memory-bound model erases
  the gain. All ops in the 2D forward (`Conv2d`, `GroupNorm`, `PixelShuffle/Unshuffle`,
  `interpolate`, elementwise) propagate `channels_last`, so an end-to-end pass is achievable;
  verify with `t.is_contiguous(memory_format=torch.channels_last)` at the seams.

## Differences from the video port

- **No input-channel padding needed by default.** The video repo padded the 3-channel input to
  4 (and the `conv_in` weights to match) for NDHWC alignment. For 2D NHWC, cuDNN handles
  C=3 inputs reasonably; the first conv is a tiny fraction of total time. Keep the trick in the
  back pocket (it is lossless if the padded weight slice is zero and the padded input channel is
  zero), but measure before bothering.
- `PixelShuffle`/`PixelUnshuffle` in [[PXS Sampling Blocks]] are layout-polymorphic but were a
  known source of layout round-trips in older PyTorch; the [[PXS Simplifications]] rewrites
  (avg-pool / nearest-interpolate) are unambiguously layout-clean.

Compose with [[torch.compile Tuning (2D)]]: declared memory format flows into generated kernels.
