---
tags: [analysis]
aliases: [applicability matrix, verdicts]
---

# Technique Applicability

Every acceleration technique from the video-VAE vault (`y2e4TRQz/docs/techniques/`), with a
verdict for `KVAEImage`. Legend: **✔ as-is** = pure code change, checkpoint-compatible ·
**◐ adapted** = applies with 2D-specific changes · **⚠ retrain** = requires new weights ·
**✘ n/a** = does not transfer.

| video-vault technique | verdict | 2D story |
|---|---|---|
| torch.compile Tuning | **✔ as-is** | [[torch.compile Tuning (2D)]] — biggest free win; the model is eager today |
| Channels-Last Memory Format | **✔ as-is** | [[Channels-Last Memory Format (2D)]] — `channels_last` instead of `channels_last_3d` |
| cudnn.benchmark / expandable_segments | **✔ as-is** | already written but commented out (`common_utils.py:41-44`) |
| Pre-allocated buffers, no `torch.cat` | **◐ adapted** | model code is already cat-free; the real equivalent is [[PXS Simplifications]] |
| Spatial Tiling | **◐ adapted** | [[Spatial Tiling (2D)]] — needed only ≥4K or big batches; must include the decoder |
| Halo Overlap | **◐ adapted** | same mechanism, receptive fields are smaller (2D 3×3 convs); exactness blocked by GroupNorm |
| Piece Decomposition | **◐ adapted** | degenerates to "tile levels 0–1, run the tail whole"; simpler than video (no temporal pieces) |
| Tile Count Selection | **◐ adapted** | same two-peak memory model minus all temporal terms; decoder peak (`256·HW`) sets the budget |
| Pad-Free Causal Convolution | **◐ adapted (spatial half only)** | temporal half is n/a; the spatial insight — replicate/reflect pads cost a full extra copy per conv — applies to every ResNet/sampling conv here. See [[Padding Modes]] |
| Zero Spatial Padding (zsp) | **⚠ retrain** | identical story: zeros are free inside conv kernels, but edge behavior changes → new checkpoints. [[Padding Modes]] |
| SplittedRMSNorm | **⚠ retrain** | identical story *plus* the decoder: `DecoderSpatialNorm2D` wraps GroupNorm too. Precondition for exact tiling. [[Normalization and Exact Tiling]] |
| EffDownsampleV2 (channel doubling) | **⚠ retrain** | architecture change; would cut level-0/1 bandwidth, but image workloads rarely justify a retrain for this alone |
| Temporal Streaming | **✘ n/a** | no time axis. [[Non-Transferable Techniques]] |
| Causal conv caching (`CachedCausalConv3d`) | **✘ n/a** | nothing is causal or cached in 2D |
| Decoder Streaming ("leave decoder alone") | **✘ n/a — and the conclusion inverts** | with no streaming escape, the decoder becomes the *primary* optimization target. [[Decoder2D]] |
| SafeConv3d chunking | **✘ n/a** | its job (fit oversized conv inputs) is done properly by tiling in 2D |

## New techniques not in the video vault

| technique | verdict | note |
|---|---|---|
| [[PXS Simplifications]] | **✔ as-is, exact** | residual paths of both sampling blocks are algebraically redundant/reducible; removes the model's largest temporary |
| Batching | **✔ as-is** | images have no cache state; the video model's B=1 restriction does not apply |

## Recommended order of attack

1. **Session/runtime flags + batching** — `cudnn.benchmark = True`,
   `PYTORCH_ALLOC_CONF=expandable_segments:True`, batch inputs of equal size. No code risk.
2. **[[PXS Simplifications]]** — three-line exact rewrites; helps both eager and compiled runs.
3. **[[torch.compile Tuning (2D)]] + [[Channels-Last Memory Format (2D)]]** — port the v2c
   compile recipe (`dynamic=True`, `max_autotune`, `combo_kernels`, `memory_planning`,
   channels-last weights). Expected to recover most of the 2–3× the video vault attributes to
   fusion/autotuning on this class of memory-bound conv stacks, since the 2D model has the same
   norm→SiLU→conv→residual chains — plus the ~29 unfused `DecoderSpatialNorm2D` interpolate+1×1
   chains in the decoder.
4. **[[Spatial Tiling (2D)]]** (approximate, or exact-after-retrain) — only if ≥4K images or
   large batches are an actual requirement.
5. **Model changes** ([[Padding Modes]] zeros, [[Normalization and Exact Tiling]] RMS,
   channel-doubling downsample) — bundle into the next training run, as the video team did for
   the 4×16×16 model, rather than retraining for inference alone.
