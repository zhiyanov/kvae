---
tags: [technique]
aliases: [inductor flags 2D, compile 2D]
---

# torch.compile Tuning (2D)

**Verdict: applies as-is — the highest-leverage pure code change.** The image model today is
exactly where the video baseline was: an interpreter-driven sequence of eager kernels over
memory-bound norm→SiLU→conv→residual chains. The video vault measured that leaving this class of
model uncompiled "leaves 2–3× on the table before any algorithmic change"; the same reasoning
holds for [[Encoder2D]]/[[Decoder2D]], which are the same chains minus the temporal axis.

## Port of the v2c recipe

```python
import torch._inductor.config as cfg
cfg.max_autotune = True
cfg.max_autotune_gemm_backends = "ATEN,TRITON,CUTLASS"
cfg.combo_kernels = True          # fuse independent small elementwise kernels
cfg.memory_planning = True
cfg.force_same_precision = True   # keep bf16, no silent fp32 upcasts

encoder = torch.compile(vae.encoder, dynamic=True, fullgraph=False)
decoder = torch.compile(vae.decoder, dynamic=True, fullgraph=False)
```

- `dynamic=True` matters here for a *different* reason than in video: the inference script
  supports arbitrary per-image sizes (batch 1, no resize). One dynamic graph avoids a
  recompilation per unique (H, W). With `--img_size` batching, shapes are static and even
  `dynamic=False` autotuning works.
- With tiling ([[Spatial Tiling (2D)]]), compile per piece as in the video repo, under
  `torch._dynamo.utils.disable_cache_limit()`.
- No graph-break landmines: unlike the video model there are no data-dependent branches, caches,
  or `SafeConv3d` emergency paths in the 2D forward. `fullgraph=True` is likely achievable.

## What fusion buys specifically here

- **ResNet chains**: GroupNorm (fp32 reduction) + SiLU + conv fold into fewer
  activation round-trips — ~24 such chains per encode+decode.
- **`DecoderSpatialNorm2D`** (~29 calls per decode): `interpolate(nearest) → GroupNorm →
  mul(conv_y(zq)) → add(conv_b(zq))` is currently 5+ kernels each, several at full resolution.
  Inductor fuses the elementwise tail and can often fold the nearest-interpolate into the
  consumer.
- **Replicate/reflect pads**: inductor typically fuses `F.pad` into the consuming conv's index
  arithmetic, recovering much of [[Padding Modes]] *without* retraining.
- **[[PXS Sampling Blocks]]**: the residual arithmetic fuses — but do the exact rewrites in
  [[PXS Simplifications]] anyway; they shrink the *tensor sizes*, which fusion cannot.

## Runtime environment (same as video)

```python
torch.backends.cudnn.benchmark = True     # currently commented out in common_utils.py:43
```
```bash
export PYTORCH_ALLOC_CONF='expandable_segments:True'
```

Plus [[Channels-Last Memory Format (2D)]] for weights and activations — the two compose:
inductor preserves and exploits the declared memory format.
