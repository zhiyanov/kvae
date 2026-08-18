---
tags: [model]
aliases: [PXSDownsample, PXSUpsample, pixel-shuffle sampling]
---

# PXS Sampling Blocks

The image model's resolution changes are done by `PXSDownsample` / `PXSUpsample`
(`kvae/layers/sampling.py:120` / `:15`) — the 2D analogue of the video model's
`EffDownsample`/`CachedPXSUpsample`: a **learned strided/interp conv** plus a
**parameter-free pixel-shuffle residual**, followed by a 1×1 linear mix. The residual guarantees
the mean signal survives downsampling even with untrained conv weights.

## PXSDownsample (`sampling.py:120`)

```python
pxs_interm = PixelUnshuffle(2)(x)                       # (B, 4C, H/2, W/2)  ← 4C temporary
pxs_out    = pxs_interm.view(B, C, 4, H/2, W/2).mean(2) # space-to-channel mean
conv_out   = Conv2d(3×3, stride 2, reflect)(x)
return Conv2d(1×1)(conv_out + pxs_out)
```

- Channels are **preserved** (no doubling — unlike the video `EffDownsampleV2`); channel growth
  happens in the next level's first ResNet block instead. See [[Encoder2D]].
- The unshuffle→view→mean chain is **exactly `F.avg_pool2d(x, 2)`** (verified numerically) —
  see [[PXS Simplifications]] for why the current form is wasteful.

## PXSUpsample (`sampling.py:15`)

```python
repeated       = x.repeat_interleave(4, dim=1)          # (B, 4C, H, W)  ← 4C temporary
pxs_interm     = PixelShuffle(2)(repeated)              # (B, C, 2H, 2W)
image_like_ups = F.interpolate(x, 2, mode="nearest")    # (B, C, 2H, 2W)
conv_out       = Conv2d(3×3, reflect)(image_like_ups)
return Conv2d(1×1)(conv_out + pxs_interm)
```

- `pxs_interm` is **exactly equal to `image_like_ups`** (verified numerically: repeating each
  channel 4× and pixel-shuffling by 2 places identical values in every 2×2 output block, i.e.
  nearest-neighbor upsampling). The forward computes the same tensor twice by different routes,
  and the `repeat_interleave` route materializes a 4C-channel temporary to do it — the single
  largest tensor in the whole model at the last decoder upsample ([[Decoder2D]]).
  See [[PXS Simplifications]].

## Tiling properties

Both blocks are **local** operators: the conv has a 3×3 receptive field, the residual paths are
2×2-block-local. They are therefore tile-safe under [[Spatial Tiling (2D)]] with halos, provided
tile boundaries are aligned to even pixel coordinates (the shuffle/pool granularity) — in
practice tiles aligned to multiples of 8 input pixels (the full compression factor) keep every
stage aligned. The `reflect` padding applies only at true image borders, exactly like the video
model's case (*video vault: Zero Spatial Padding — Interaction with tiling*).
