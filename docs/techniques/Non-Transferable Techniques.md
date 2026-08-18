---
tags: [technique, analysis]
aliases: [temporal techniques, n/a techniques]
---

# Non-Transferable Techniques

The video vault's temporal family has no 2D counterpart — but its *absence* changes two
conclusions, so the reasoning is recorded here.

## Not applicable, and why

| video technique | why it doesn't transfer |
|---|---|
| **Temporal Streaming** | streaming slices the time axis; images have none. Its 2D "equivalent" — slicing the batch axis — is just choosing a batch size. |
| **Causal Convolution / `CachedCausalConv3d`** | nothing in `KVAEImage` is causal or stateful; all convs are plain `nn.Conv2d`. The elaborate cache/offset arithmetic has no target. |
| **Pad-Free Causal Convolution (temporal half)** | the prefix-conv trick removed the cost of *temporal* causal padding; only its spatial insight survives → [[Padding Modes]]. |
| **`SafeConv3d` emergency chunking** | existed to push >2 GB conv inputs through cuDNN by slicing time. 2D conv inputs of that size only occur at ≥8K, where [[Spatial Tiling (2D)]] is the correct tool. |
| **First-frame special-casing** (frame-0 bypass in temporal downsamples, `16k+1` frame contract) | purely temporal-causality bookkeeping. |

## Two conclusions that flip for images

1. **"Leave the decoder alone" inverts.** The video decoder was never optimized because temporal
   streaming already bounded its memory and its thin segments were an acceptable fit for an
   output-heavy stage. With no time axis to stream over, the image decoder — which does *more*
   work than the encoder and holds the model's largest tensor ([[Decoder2D]]) — becomes the
   **primary** optimization target. Every technique doc in this vault is written for both halves.
2. **"Batch size 1 only" inverts.** The video model's mutable conv caches made concurrent
   samples unsafe, pinning inference to B=1. `KVAEImage` is stateless, so batching is the
   cheapest throughput lever available — it produces exactly the large, dense, static-shaped
   workloads that the video repo had to *construct* via [[Spatial Tiling (2D)|tiling]].
