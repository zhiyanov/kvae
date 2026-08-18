---
tags: [model]
aliases: [Decoder2D, DecoderSpatialNorm2D]
---

# Decoder2D

`Decoder2D` (`kvae/models/kvae_2d.py:192`) is the upsampling mirror of [[Encoder2D]] — with two
structural differences that make it the **more expensive half** of the image VAE.

```
conv_in  (3×3, replicate, z_ch → 1024)                              kvae_2d.py:217
mid:     ResnetBlock2D ×2 (1024, spatial-norm conditioned)
level 3: ResnetBlock2D ×3 (1024)      ──► PXSUpsample (→ H/4)
level 2: ResnetBlock2D ×3 (1024→512)  ──► PXSUpsample (→ H/2)
level 1: ResnetBlock2D ×3 (512→256)   ──► PXSUpsample (→ H)
level 0: ResnetBlock2D ×3 (256→128)
DecoderSpatialNorm2D + SiLU + conv_out (3×3, replicate) ──► RGB     kvae_2d.py:263-271
```

## Difference 1: more blocks, all latent-conditioned

- Each level has `num_res_blocks + 1 = 3` blocks (`kvae_2d.py:246`) vs the encoder's 2 —
  14 ResNet blocks total vs the encoder's 10.
- Every normalization in the decoder is a `DecoderSpatialNorm2D` (`kvae/layers/norm.py:14`),
  a MoVQ/CogVideoX-style conditioning norm:

  ```python
  zq = F.interpolate(zq, size=f.shape[2:], mode="nearest")   # latent → feature resolution
  out = GroupNorm(f) * conv_y(zq) + conv_b(zq)               # two 1×1 convs per call
  ```

  There are ~29 such calls per forward (2 per block × 14 blocks + `norm_out`), each one
  re-interpolating the latent to the current feature size and running two 1×1 convs. At level 0
  that means several `z_ch→128` 1×1 convs **at full resolution** plus a nearest-upsampled copy
  of the latent per call. Under eager execution none of this fuses; it is prime
  [[torch.compile Tuning (2D)]] material.

## Difference 2: the peak tensor lives here

The last [[PXS Sampling Blocks|PXSUpsample]] (level 1 → level 0, 256 channels at H/2) currently
materializes `repeat_interleave(x, 4, dim=1)` — a **1024-channel tensor at H/2×W/2 = 256·HW
elements**, twice the encoder's peak (512 MiB at 1024², 8 GiB at 4K, in bf16). This temporary is
redundant — the residual path it feeds is exactly nearest-neighbor upsampling, which the block
already computes; see [[PXS Simplifications]].

## Why the decoder can't be ignored this time

In the video repo the decoder was deliberately left unoptimized (*video vault: Decoder
Streaming*): temporal streaming bounded its memory and the encoder dominated. For images:

- there is **no time axis to stream over** — [[Non-Transferable Techniques]];
- the decoder does **more work than the encoder** (more blocks, the conditioning norms, and
  FLOPs again flat per level but with a 3/2 block ratio);
- its memory peak is 2× the encoder's, so at high resolution it OOMs *first*.

Any [[Spatial Tiling (2D)]] plan for high-resolution images must therefore cover the decoder,
where the tiling story has an extra wrinkle: `zq` must be cropped per tile consistently with the
feature tile. Since compression is exactly 8×8 and the interpolation is `nearest` with integer
scale, latent-aligned tiles (multiples of 8 input pixels) make the `zq` crop exact. The
[[GroupNorm]] inside every `DecoderSpatialNorm2D` remains the exactness blocker, same as in the
encoder ([[Normalization and Exact Tiling]]).
