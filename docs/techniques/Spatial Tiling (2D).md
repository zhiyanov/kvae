---
tags: [technique]
aliases: [tiled image inference, halo 2D, piece decomposition 2D]
---

# Spatial Tiling (2D)

**Verdict: applies with adaptations — but is second-order for images.** This one file covers the
2D fate of four video-vault techniques that formed one system there: *Spatial Tiling*,
*Halo Overlap*, *Piece Decomposition*, and *Tile Count Selection*.

## When it is needed at all

The video encoder needed tiling at 1024² because 241 frames multiplied every activation. A single
image fits until roughly:

| input | encoder peak (128·HW) | decoder peak today (256·HW, see [[Decoder2D]]) |
|---|---|---|
| 2048² | 1 GiB | 2 GiB |
| 4096² | 4 GiB | 8 GiB |
| 8192² | 16 GiB | 32 GiB |

(bf16, batch 1, per-tensor; live-set peak is a small multiple.) With the [[PXS Simplifications]]
applied, the decoder peak halves. So tiling is a **≥4K-or-large-batch feature**, not a
prerequisite as it was for video. Note batching multiplies all rows by B — a tiled path also
unlocks bigger batches.

## What transfers directly

- **The tile→stitch→tail structure.** Run levels 0–1 per tile, stitch into a pre-allocated
  buffer at ¼ resolution, run levels 2–3 + mid + head once. Piece boundaries are chosen by the
  same logic (activation-size cliffs), but the 2D pyramid is simpler: no temporal sub-pieces, no
  `piece_1a`. The video trick of deferring each downsample's 1×1 `linear` conv to the next piece
  transfers verbatim (the [[PXS Sampling Blocks]] have the same trailing 1×1).
- **Halo Overlap.** Same mechanism, smaller halos: each tiled piece is a stack of 3×3 convs, so
  the halo is `(#convs before stitch) × 1` pixel at each scale, doubled per downsample when
  mapped back to input scale. All layers in the tiled region are local ([[PXS Sampling Blocks]]
  residuals included) **except the norms** — see the blocker below.
- **Tile Count Selection.** The v2c two-peak memory model (per-tile peak incl. halo vs stitch
  buffer) simplifies: drop all temporal terms. New wrinkle: the budget must be set by the
  **decoder's** peak, and the same machinery must dispatch decoder tiles.
- **Both tiled and stitched buffers pre-allocated**, written by slice — as in v2c.

## What changes for images

1. **The decoder must be tiled too.** Video left the decoder on temporal streaming; images have
   no time axis ([[Non-Transferable Techniques]]), and the decoder OOMs first. Decoder tiling
   tiles the *latent*: crop `z` per tile (latent-aligned tiles — multiples of 8 input pixels —
   make the `DecoderSpatialNorm2D` nearest-interpolation of `zq` per-tile exact, see
   [[Decoder2D]]), run mid+up levels per tile with halos, stitch at full resolution.
2. **Exactness is blocked by GroupNorm on both sides.** Encoder ResNet blocks use [[GroupNorm]];
   every decoder norm wraps GroupNorm inside `DecoderSpatialNorm2D`. Full-H×W statistics differ
   per tile and no halo fixes that — the identical conclusion the video vault reached, which
   forced the retrained RMS-norm 4×16×16 model. Options, same menu as then:
   - **approximate tiling** (hard edges or halo-with-per-tile-stats): cheap, small quality loss
     that shrinks as tiles grow — with only 2–4 tiles at 4K–8K, per-tile GroupNorm statistics
     deviate far less than in the video 16-tile regime;
   - **exact tiling**: requires channel-only normalization → retraining
     ([[Normalization and Exact Tiling]]).
3. **Boundary padding semantics** are unchanged: `replicate`/`reflect` apply only at true image
   borders; interior tile edges receive real neighbor pixels via halos. So tiling composes
   cleanly with the current checkpoints and with a future zero-padding retrain ([[Padding Modes]]).
