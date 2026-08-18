---
tags: [technique, model]
aliases: [GroupNorm, RMS norm 2D, exact tiling precondition]
---

# Normalization and Exact Tiling

The 2D fate of the video vault's *SplittedRMSNorm*. **Verdict: applies, but it is a model change
(retrain), and in 2D it must cover the decoder's conditioning norm as well.**

## Where normalization lives in `KVAEImage`

- **Encoder:** [[GroupNorm]] (32 groups, fp32 statistics) twice per `ResnetBlock2D`
  (`resnet.py:56,68`) plus `norm_out` (`kvae_2d.py:158`) — statistics reduce over
  `(C/32, H, W)`.
- **Decoder:** every norm is a `DecoderSpatialNorm2D` (`norm.py:14`), which **wraps a GroupNorm**
  and modulates it with two 1×1 convs of the nearest-upsampled latent:
  `GroupNorm(f) · conv_y(zq) + conv_b(zq)`.

## Two independent reasons to change it (same as video)

1. **Exact tiling.** GroupNorm statistics depend on the whole H×W extent — a tile computes
   different statistics than the full image, and no [[Spatial Tiling (2D)|halo]] can fix a
   global reduction. A channel-only RMS norm is per-pixel, hence tile-invariant: with it,
   convolutions become the only tile-coupling ops and halos make tiling numerically exact. This
   was the precondition that forced the video 4×16×16 retrain.
2. **Speed.** Per-pixel RMS over a contiguous channel axis
   ([[Channels-Last Memory Format (2D)]]) fuses with scale+bias+SiLU into one kernel — one read,
   one write — vs GroupNorm's grouped fp32 statistics pass. Under [[torch.compile Tuning (2D)]]
   GroupNorm compiles reasonably well, so the *speed* delta in 2D is modest; the *tiling
   exactness* is the real prize.

## The decoder wrinkle

The modulation structure of `DecoderSpatialNorm2D` is already tile-friendly — `conv_y`/`conv_b`
are 1×1 and the nearest-interpolation of `zq` is per-pixel-local given latent-aligned tiles
([[Decoder2D]]). Only the inner GroupNorm breaks exactness. A retrained model would swap it for
the RMS variant inside the same modulation:

```python
new_f = rms_norm_over_channels(f) * conv_y(zq) + conv_b(zq)
```

## If retraining is off the table

- **Approximate tiling** with per-tile GroupNorm statistics: at image scale only 2–4 large tiles
  are ever needed ([[Spatial Tiling (2D)]]), so per-tile statistics are computed over ≥¼ of the
  image and deviate far less than in the video 16-tile regime. Quality cost should be measured
  (PSNR/LPIPS parity vs untiled, as the video repo's `metrics_v3.py` did), not assumed.
- **Two-pass exact GroupNorm** (compute global statistics by aggregating per-tile moments, then
  normalize per tile) is mathematically possible but requires a synchronization at every one of
  the ~50 norms per forward — a barrier-per-layer structure that defeats the purpose of tiling
  on a single GPU. Not recommended; noted for completeness.
