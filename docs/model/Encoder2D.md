---
tags: [model]
aliases: [encoder pyramid 2D, Encoder2D]
---

# Encoder2D

`Encoder2D` (`kvae/models/kvae_2d.py:93`) is a classic downsampling pyramid — the spatial
skeleton of the old video `KandinskyEncoder3D` with the temporal axis removed.

```
conv_in  (3×3 conv, zeros padding, 3 → 128 ch)                      kvae_2d.py:116
level 0: ResnetBlock2D ×2 (128)  ──► PXSDownsample (½ res)
level 1: ResnetBlock2D ×2 (128→256) ──► PXSDownsample (¼ res)
level 2: ResnetBlock2D ×2 (256→512) ──► PXSDownsample (⅛ res)
level 3: ResnetBlock2D ×2 (512→1024)
mid:     ResnetBlock2D ×2 (1024)
GroupNorm(32) + SiLU + conv_out (3×3, zeros) ──► 2·z_channels       kvae_2d.py:158-167
```

## ResNet block

`ResnetBlock2D` (`kvae/layers/resnet.py:39`):

```
x ──► GroupNorm ─ SiLU ─ Conv2d(3×3, replicate) ─ GroupNorm ─ SiLU ─ Conv2d(3×3, replicate) ──►(+)── y
 └──────────────────────────── 1×1 conv if c_in ≠ c_out ────────────────────────────────────┘
```

- Norm is [[GroupNorm]] (32 groups, eps 1e-6) — a full-H×W reduction, the blocker for exact
  tiling ([[Normalization and Exact Tiling]]).
- Both 3×3 convs use `padding_mode="replicate"` — in eager PyTorch each one runs a separate `F.pad` kernel producing a padded copy of its input ([[Padding Modes]]).
- Channel growth happens **inside the first block of each level** (via `nin_shortcut`), not in
  the downsample — unlike the video 4×16×16 model's channel-doubling `EffDownsampleV2`. So this model carries 128 channels at full resolution, like the old 4×8×8 variant.

## Downsampling

Each non-final level ends with a [[PXS Sampling Blocks|PXSDownsample]]: strided 3×3 conv
(`reflect` padding) + a parameter-free space-to-channel-mean residual (≡ 2×2 avg-pool, see
[[PXS Simplifications]]) + 1×1 linear mix. Channels are preserved.

## Memory and compute profile

Per-activation sizes (bf16, batch 1, `HW` = input pixels). FLOP share uses `C²·pixels`, which the
`ch_mult=(1,2,4,8)` schedule keeps **constant across levels**:

| stage | resolution | channels | elements | 1024² | 4096² | FLOP share |
|---|---|---|---|---|---|---|
| conv_in + level 0 | H, W | 128 | 128·HW | 256 MiB | 4 GiB | ~equal |
| level 1 | H/2, W/2 | 256 | 64·HW | 128 MiB | 2 GiB | ~equal |
| level 2 | H/4, W/4 | 512 | 32·HW | 64 MiB | 1 GiB | ~equal |
| level 3 + mid | H/8, W/8 | 1024 | 16·HW | 32 MiB | 512 MiB | ~equal |

Two consequences:

1. **FLOPs are spread evenly, memory traffic is front-loaded.** Since the network is
   memory-bound in eager bf16, wall time still concentrates at level 0 — the same shape as the
   video encoder, just ~2 orders of magnitude smaller in absolute size.
2. **Capacity only binds at extreme sizes.** The whole encoder forward fits on one GPU up to
   roughly 4K even with eager pad copies; [[Spatial Tiling (2D)]] becomes relevant at ~8K or
   with large batches. Compare: the video model needed tiling already at 1024².
