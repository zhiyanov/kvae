---
tags: [analysis]
aliases: [inference_2d_kvae, status quo]
---

# Current Inference Path

What actually runs today for images (`scripts/inference_2d_kvae.py`), and the gap against the
video repo's accelerated encoders.

## What the script does

```python
vae = KVAEImage.from_pretrained(...).eval().to(device).to(torch.bfloat16)
with torch.no_grad():
    latent = vae.encode(images).latent_dist.mode()
    rec    = vae.decode(latent).clip(-1, 1)
```

- **bfloat16** end-to-end — good, matches the video setup.
- **Batch size 1 by default**; batching requires `--img_size` (images must share a shape).
- **Fully eager**: no `torch.compile`, no inductor flags.
- **Default memory format** (NCHW), weights and activations.
- `set_seed_and_optimal_cuda_env` (`scripts/common_utils.py:32`) has `cudnn.benchmark`,
  TF32, and deterministic-algorithm switches **commented out** (`common_utils.py:41-44`) —
  despite the "optimal cuda env" name, it currently only seeds RNGs.
- No tiling / no memory management of any kind — a sufficiently large image or batch simply
  OOMs, with the [[Decoder2D]] peak (`256·HW` elements) hit first.

## Gap table vs the video repo's v2c encoder

| optimization (video vault name) | present in video v2c | present here |
|---|---|---|
| torch.compile + inductor tuning | ✔ | ✘ |
| channels-last memory format | ✔ | ✘ |
| cudnn.benchmark / allocator env | ✔ | ✘ (commented out) |
| pad-free convolution / zero-padding checkpoints | ✔ | ✘ (`replicate`/`reflect` everywhere) |
| channel-only RMS norm | ✔ (retrained) | ✘ (GroupNorm) |
| spatial tiling + halo + tile-count model | ✔ | ✘ |
| pre-allocated buffers, no `torch.cat` | ✔ | mostly n/a (code is clean, but see [[PXS Simplifications]]) |
| batching | n/a (video is B=1) | **available but off by default** |

## Cheapest levers first

1. **Batching** — unique to the image model: no caches, pure functions, so throughput scales
   with batch size until memory/compute saturates. Zero code risk.
2. [[torch.compile Tuning (2D)]] + [[Channels-Last Memory Format (2D)]] + uncommenting the
   cudnn/allocator flags — pure code changes, checkpoint-compatible.
3. [[PXS Simplifications]] — exact, free, removes the model's largest temporaries.
4. [[Spatial Tiling (2D)]] — only needed for ≥4K/8K inputs or very large batches.
5. [[Padding Modes]] zero-padding and [[Normalization and Exact Tiling]] RMS norm — model
   changes requiring retraining, exactly as they were for the video 4×16×16 model.

See [[Technique Applicability]] for the full technique-by-technique verdict.
