---
tags: [concept]
aliases: [group normalization]
---

# GroupNorm

`nn.GroupNorm(num_groups=32, eps=1e-6)` — the normalization used throughout `KVAEImage`:
directly in the encoder's `ResnetBlock2D` and `norm_out`, and *inside* every decoder
`DecoderSpatialNorm2D` ([[Decoder2D]]).

For each sample and each of the 32 channel groups it computes mean and variance over
`(C/32, H, W)` — i.e. **over the entire spatial extent** — then normalizes and applies per-channel
affine parameters. Statistics are computed in fp32 even under bf16 inference.

Two properties matter for acceleration:

1. **It is a global spatial reduction.** A tile of the image produces different statistics than
   the full image, so any [[Spatial Tiling (2D)]] under GroupNorm is approximate regardless of
   halo width. This is the blocker for exact tiling, identically to the video model — resolved
   there by retraining with a channel-only RMS norm ([[Normalization and Exact Tiling]]).
2. **It is a bandwidth tax in eager mode.** Each call is a full read (statistics pass) plus a
   read+write (normalize pass) of the activation, separate from the neighboring SiLU and conv
   kernels. [[torch.compile Tuning (2D)]] fuses the normalize pass with its elementwise
   neighbors, but the statistics reduction remains a separate pass by construction.
