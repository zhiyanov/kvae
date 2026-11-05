---
tags: [technique, finding]
aliases: [sampling block rewrites, free wins]
---

# PXS Simplifications

**Verdict: new finding, applies as-is, numerically exact.**
*Status: the upsample rewrite (§1) is landed and validated — bitwise parity, decode peak
−24…−27%; see [[Optimization Results]]. The downsample rewrite (§2) is the T1 candidate.*

Two algebraic identities in the
[[PXS Sampling Blocks]] let both residual paths be computed without their 4C-channel
temporaries. Verified numerically against PyTorch's PixelShuffle/PixelUnshuffle semantics
(bit-exact for the upsample identity; the downsample identity is exact up to float summation
order).

## 1. PXSUpsample computes the same tensor twice

`PixelShuffle(2)(x.repeat_interleave(4, dim=1))` places the four identical copies of each
channel into the 2×2 output block → it **is** nearest-neighbor 2× upsampling, which the forward
already computes one line later as `image_like_ups` (`kvae/layers/sampling.py:32-35`):

```python
# current
repeated = x.repeat_interleave(self.factor**2, dim=1)   # (B, 4C, H, W) temporary
pxs_interm = self.shuffle(repeated)
image_like_ups = F.interpolate(x, scale_factor=2, mode="nearest")
conv_out = self.spatial_conv(image_like_ups)
out = conv_out + pxs_interm

# equivalent
up = F.interpolate(x, scale_factor=2, mode="nearest")
out = self.spatial_conv(up) + up
```

Impact: removes one shuffle kernel and a `(B, 4C, H, W)` temporary per upsample. At the last
decoder upsample (256 ch at H/2) that temporary is `256·HW` elements — **the largest tensor in
the entire model** (8 GiB bf16 at 4096²), and it currently sets the decoder's memory peak
([[Decoder2D]]). This is simultaneously a bandwidth win and a capacity win (raises the
resolution at which [[Spatial Tiling (2D)]] becomes necessary).

## 2. PXSDownsample's residual is an avg-pool

`PixelUnshuffle(2)` groups each channel's 2×2 block into 4 consecutive channels, so
`unshuffle → view(B, C, 4, h, w) → mean(dim=2)` (`kvae/layers/sampling.py:137-141`) **is**
2×2 average pooling:

```python
# current
pxs_interm = self.unshuffle(x)                      # (B, 4C, H/2, W/2) temporary
pxs_out = pxs_interm.view(b, c//4, 4, h, w).mean(2)

# equivalent
pxs_out = F.avg_pool2d(x, 2)
```

Impact: removes a `(B, 4C, H/2, W/2)` ≡ `(C·HW)`-element temporary per downsample — at encoder
level 0 that is another `128·HW` tensor, equal to the level-0 activation itself.
*Measured* ([[Optimization Results]]): encode time +1–2%, encode **peak unchanged** — the
encoder's peak is set by the replicate-pad copies inside the ResNet blocks (~3× the level-0
activation live at once), which exceed the downsample-moment working set (~2.3×). Bandwidth
win only; did not clear the acceptance gate on its own.

## Notes

- Both rewrites are **inference-only refactors**: no weights involved, outputs identical, so
  they are safe for the released checkpoints. (Float non-associativity in the pooling mean may
  flip last-ulp bits in bf16 — irrelevant in practice, but don't expect bitwise-identical
  regression hashes for the downsample path.)
- `torch.compile` can fuse the current forms' arithmetic but **cannot** shrink the declared
  intermediate sizes; the rewrite helps compiled runs too.
- The same identities hold for the video model's `CachedPXSUpsample.spatial_upsample` /
  `CachedPXSDownsample.spatial_downsample` (`kvae/layers/sampling.py:74-101, 208-223`) with the
  extra time dimension — a candidate back-port for the 3D path.
