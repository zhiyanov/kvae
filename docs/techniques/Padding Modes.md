---
tags: [technique]
aliases: [replicate padding cost, zero spatial padding 2D, zsp 2D]
---

# Padding Modes

The 2D descendant of two video-vault techniques: *Pad-Free Causal Convolution* (its spatial
half — the temporal half is [[Non-Transferable Techniques|n/a]]) and *Zero Spatial Padding*.

## Where the model pads today

| module | padding mode |
|---|---|
| `ResnetBlock2D` conv1/conv2 (`resnet.py:60,72`) | `replicate` |
| `PXSDownsample`/`PXSUpsample` spatial convs (`sampling.py:20,125`) | `reflect` |
| decoder `conv_in`/`conv_out` (`kvae_2d.py:217,265`) | `replicate` |
| encoder `conv_in`/`conv_out` (`kvae_2d.py:116,162`) | zeros (default) |

## The hidden cost (applies as-is)

`nn.Conv2d` with `padding_mode="replicate"`/`"reflect"` cannot push the padding into the cuDNN
kernel: eager PyTorch runs an explicit `F.pad`, materializing a **padded copy of the activation
before every conv** — one extra read+write of the largest tensors, ~40 times per encode+decode.
This is exactly the traffic the video repo's pad-free conv eliminated. In 2D the fix splits into
a free part and a retrain part:

### Free part: let the compiler eat the pads

[[torch.compile Tuning (2D)]]: inductor generally fuses `F.pad` into the consuming convolution's
index arithmetic (or emits a fused pad+conv), removing the standalone copy without touching
semantics or weights. This is the 2D analogue of v2c moving spatial padding "inside" the conv —
here the compiler does the moving. Verify in the inductor logs that no standalone
`constant_pad_nd`/`replication_pad2d` kernels survive at level 0/1 sizes.

### Retrain part: zero spatial padding (zsp)

Zero padding is implemented natively by cuDNN/CUTLASS kernels — literally free, simpler graphs.
But it changes what edge pixels see, so **weights trained with replicate/reflect are not
compatible** (video vault shipped dedicated `*_zsp` checkpoints). Verdict for images: same as
video —

- a pure inference-time swap to `padding_mode="zeros"` would perturb a border ring of the output
  (the perturbation width is the network's receptive field at each scale — visually the outer
  few dozen pixels); not acceptable for a quality-focused tokenizer;
- fold `padding_mode="zeros"` into the **next training run**, as the video team did. Interaction
  with [[Spatial Tiling (2D)]] is clean: padding only ever applies at true image borders.

## Priority note

For images this is worth *less* than it was for video v2c pre-compile: once compilation fuses
the pads, the remaining zsp gain is kernel simplicity, not eliminated traffic. Measure the
compiled model before scheduling a retrain for zsp alone; bundle it with the
[[Normalization and Exact Tiling]] norm change if a retrain happens anyway.
