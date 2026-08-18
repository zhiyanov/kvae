---
tags: [model]
aliases: [KVAEImage, KVAE-2D, image tokenizer]
---

# KVAE Image

`KVAEImage` (`kvae/models/kvae_2d.py:20`) is a **2D convolutional variational autoencoder for
images**. It maps an RGB image to a compact latent for downstream diffusion models and back. Compared to the previous-generation video VAE it is the same architectural family — a downsampling ResNet pyramid — with **all temporal machinery removed**: no causal convolutions, no caches, no streaming, no `SafeConv3d`. Plain `nn.Conv2d` everywhere.

## Input/output contract

```
image  (B, 3, H, W)
   │   Encoder2D                  → (B, 2·z_ch, H/8, W/8)   (mean ‖ logvar, double_z)
   ▼
latent (B, z_ch, H/8, W/8)        via DiagonalGaussianDistribution .mode() / .sample()
   │   Decoder2D  (conditioned on the latent via DecoderSpatialNorm2D)
   ▼
image  (B, 3, H, W)
```

- Spatial compression is fixed **8×8** (three [[PXS Sampling Blocks|PXSDownsample]] stages, `ch_mult=(1,2,4,8)`).
- Unlike the video model, **batching is supported and trivial** — there is no mutable cache
  state; `encode`/`decode` are pure functions of their input.
- Inference dtype in the shipped script is bfloat16.

## Variants

Both released checkpoints are served by the same class, configured via the HF Hub config
(`from_pretrained`); code defaults are `channels=128, ch_mult=(1,2,4,8), z_channels=16`.

| | KVAE-2D-1.0 | KVAE-2D-2.0 |
|---|---|---|
| spatial compression | 8 × 8 | 8 × 8 |
| latent channels | 16 | 32 |
| HF repo | `kandinskylab/KVAE-2D-1.0` | `kandinskylab/KVAE-2D-2.0` |

Both use [[GroupNorm]]-family normalization ([[Normalization and Exact Tiling]]) and
`replicate`/`reflect` spatial padding ([[Padding Modes]]) — i.e. neither of the two *retrained*
optimizations of the video 4×16×16 model (RMS norm, zero spatial padding) has been applied to an image checkpoint yet.

## The cost profile — how it differs from video

The video encoder's defining problem was **memory capacity**: a 241-frame 1024² clip produces a
64 GB level-0 activation. An image is a single frame, so the same level-0 activation is

```
128 · H · W elements  →  256 MiB (1024²) · 4 GiB (4096²) · 16 GiB (8192²)   in bf16
```

Consequences, which drive the whole [[Technique Applicability]] analysis:

1. **Fitting is not the default problem.** Up to ~2K everything fits comfortably; capacity
   techniques ([[Spatial Tiling (2D)]]) only matter at ≥4K or with large batches.
2. **Bandwidth and launch overhead are the problem.** FLOPs are *flat* across pyramid levels
   (channel doubling exactly cancels the 4× pixel reduction: `C²·HW` is constant), but memory
   traffic halves per level — so wall time concentrates in the full-resolution level exactly as
   in the video model, and the eager-mode costs (pad copies, unfused norm→SiLU→conv chains, no autotuning) dominate. See [[Encoder2D]] for the table.
3. **The decoder is the bigger half.** It has more blocks per level (3 vs 2), runs
   `DecoderSpatialNorm2D` conditioning at every block, and its peak temporary
   (inside the last [[PXS Sampling Blocks|PXSUpsample]]) is `256·H·W` elements — **2× the encoder's peak**. And unlike video, there is no time axis to stream over, so the decoder cannot keep the old repo's "leave the decoder alone, it streams" strategy — see [[Decoder2D]] and [[Non-Transferable Techniques]].

## Where acceleration would go

The natural home is an `encode`/`decode` fast path or a wrapper module, mirroring the video
repo's `efficient_vae_v2c.py`. Ordered by cost/benefit in [[Technique Applicability]].
