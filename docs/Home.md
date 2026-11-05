---
tags: [index]
aliases: [Index, Start Here]
---

# KVAE Image — Inference Acceleration Knowledge Base

This vault documents the **image (2D) part** of the KVAE repository — the `KVAEImage` model
(`kvae/models/kvae_2d.py`) — and analyzes which of the encoder-acceleration techniques developed for the previous-generation **video VAE** (documented in the `y2e4TRQz` repo's vault) apply here.

## Suggested reading order

1. **The model** — what we would be accelerating
   - [[KVAE Image]] — the model, its variants, and its cost profile
   - [[Encoder2D]] — the downsampling pyramid in detail
   - [[Decoder2D]] — the upsampling mirror, and why it is the *bigger* half for images
   - [[PXS Sampling Blocks]] — the down/upsample modules, including two exact algebraic simplifications found during this analysis
1. **The status quo**
   - [[Current Inference Path]] — what `scripts/inference_2d_kvae.py` runs today (eager, unoptimized)
3. **The analysis** — the deliverable
   - [[Technique Applicability]] — every video-VAE technique, with a verdict for the 2D model
   - [[Speed-Up Testing Specification]] — parity tiers, benchmark protocol, and acceptance gates for every optimization
   - [[Optimization Results]] — running log of landed optimizations and their measured effect
1. **The techniques** — details per technique, adapted to 2D
   - [[torch.compile Tuning (2D)]] · [[Channels-Last Memory Format (2D)]] · [[PXS Simplifications]]
   - [[Spatial Tiling (2D)]] · [[Normalization and Exact Tiling]] · [[Padding Modes]]
   - [[Non-Transferable Techniques]] — the temporal family, and what its absence implies

## The one-paragraph summary

`KVAEImage` is a plain (non-causal, non-cached) 2D convolutional VAE — essentially the spatial
skeleton of the old video encoder with the entire temporal machinery removed. Its inference path
today is fully eager: no `torch.compile`, no channels-last layout, batch size 1 by default, and
every ResNet/sampling conv pays a hidden `F.pad` copy for its `replicate`/`reflect` padding.
Because images are ~100–1000× smaller workloads than video clips, the *memory-capacity* half of the old toolkit ([[Spatial Tiling (2D)]] with halos and tile-count selection) only matters at
≥4K resolution or large batches; the *memory-bandwidth* half ([[torch.compile Tuning (2D)]], [[Channels-Last Memory Format (2D)]], padding placement, norm fusion) transfers directly and is
the first-order win. Two techniques remain **model changes** requiring retraining, exactly as in
the video repo: zero spatial padding and channel-only (RMS) normalization — the latter is the
precondition for *exact* tiling and is blocked today by [[GroupNorm]] in the encoder and inside
every `DecoderSpatialNorm2D` in the decoder. Two new, free, numerically-exact simplifications of the [[PXS Sampling Blocks]] were verified during this analysis ([[PXS Simplifications]]).

## Repository map (image part)

| file | role |
| --- | --- |
| `kvae/models/kvae_2d.py` | [[KVAE Image]] — `KVAEImage`, [[Encoder2D]], [[Decoder2D]] |
| `kvae/layers/resnet.py` (`ResnetBlock2D`) | the ResNet block, see [[Encoder2D]] |
| `kvae/layers/sampling.py` (`PXSDownsample`, `PXSUpsample`) | [[PXS Sampling Blocks]] |
| `kvae/layers/norm.py` (`DecoderSpatialNorm2D`) | decoder conditioning norm, see [[Decoder2D]] |
| `scripts/inference_2d_kvae.py` | [[Current Inference Path]] |

The previous-generation video VAE and its acceleration work live in the sibling repo `y2e4TRQz/`
(vault: `y2e4TRQz/docs/`); its technique names are referenced throughout as *(video vault:
Spatial Tiling)* etc.
