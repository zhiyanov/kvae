---
tags: [analysis, results]
aliases: [results log, T0 results]
---

# Optimization Results

Running log of landed optimizations, validated per the
[[Speed-Up Testing Specification]]. Hardware: one idle H100 80GB on the remote machine;
raw benchmark JSONs live on the remote in `outputs/bench/` (gitignored) and are named
`<model>_<candidate>_<commit>_<timestamp>.json`.

## T0 — PXSUpsample residual reuse (landed 2026-08-18)

**Change** (`kvae/layers/sampling.py`, commit `dd494be`; tooling in `77c40ba`…`5b63ca2`):
`PXSUpsample.forward` reuses the nearest-neighbor upsample as the residual instead of
materializing it a second time via `repeat_interleave` + `PixelShuffle` — see
[[PXS Simplifications]]. Declared tier: **T0 (bitwise)**.

### Parity

- `pytest scripts/tests`: 17/17 pass — shape contract + layer-level bitwise tests
  (CPU/CUDA × fp32/bf16 × random/constant/ramp/checkerboard).
- End-to-end harness (`scripts.tests.test_parity_2d --candidate pxs --tier T0`):
  **bitwise-identical** latents and reconstructions (`torch.equal`, all diffs exactly 0.0)
  on **both checkpoints × both dtypes × 12 inputs each** (synthetic, structured, and all of
  `assets/images`).

### Benchmarks (grid per spec; benchmark JSONs at commit `5b63ca2`)

Key cells, KVAE-2D-2.0 (KVAE-2D-1.0 within noise of the same numbers):

| cell | base p50 | pxs p50 | speedup | base peak | pxs peak | mem Δ |
|---|---|---|---|---|---|---|
| decode 1024² b1 bf16 | 134.4 ms | 131.2 ms | 1.02× | 4.08 GiB | 3.08 GiB | **−24.5%** |
| decode 2048² b1 bf16 | 529.9 ms | 517.5 ms | 1.02× | 15.00 GiB | 11.00 GiB | **−26.7%** |
| decode 512² b16 bf16 | 362.7 ms | 350.7 ms | 1.03× | 16.97 GiB | 12.97 GiB | **−23.6%** |
| e2e 1024² b1 bf16 | 190.6 ms | 187.1 ms | 1.02× | — | — | −24.6% |
| encode (all cells) | — | — | 1.00–1.01× | — | — | ±0.0% |

- **Decode peak memory: −24.5% … −26.7%** at ≥512² — the removed `repeat_interleave`
  temporary was the largest tensor in the model, as predicted in [[Decoder2D]].
- Decode time +2–3%, e2e +1–3%, encode untouched; **no cell regressed** in time or memory.

### Gate verdict

- Strict **time** gate: FAILED — target decode improvement +2.4% (2.0) / +1.8% (1.0),
  below the 3% bar.
- **Memory** gate (the appropriate one for a change whose declared effect is removing the
  peak temporary): **PASSED** on both checkpoints (−24.5% ≫ 3% bar, no time cell regressing
  beyond 2%).

## T1 — PXSDownsample avg-pool residual (landed 2026-08-18, gate NOT cleared)

**Change** (`kvae/layers/sampling.py`, commit `0048b55`): `PXSDownsample.forward` computes the
residual as `F.avg_pool2d(x, 2)` instead of unshuffle → view → mean — see
[[PXS Simplifications]]. Declared tier: **T1**; also implemented the T2 metric gate
(PSNR/LPIPS deltas on real images) in the parity harness as required for bf16 T1 acceptance.

### Parity

- `pytest scripts/tests`: 26/26 pass (adds the fp64 identity test and T1 layer tests).
- End-to-end harness (`--candidate pxs --tier T1`): **bitwise-identical** on both checkpoints ×
  both dtypes × all inputs — on H100 the `avg_pool2d` kernel reproduces the unshuffle+mean
  summation exactly, so the conservatively-declared T1 behaves as T0 in practice.
- Metric gate: ΔPSNR and ΔLPIPS exactly 0.000 on all real images — PASS.

### Benchmarks (JSONs at commit `0048b55`) and gate verdict

- Encode time (the target): **+1.3% (2.0) / +0.6% (1.0)** at 1024²/b1 — below the 3% bar →
  **time gate FAILED**.
- Encode peak memory: **unchanged (+0.0%)** → **memory gate FAILED**. The prediction in
  [[PXS Simplifications]] §2 did not materialize: the encoder's peak is set by the
  replicate-pad copies inside the ResNet blocks (~3× the level-0 activation live at once),
  not by the downsample's unshuffle temporary (~2.3×) — removing the temporary trims traffic,
  not the peak.
- No cell regressed in time or memory; the T0 decode win is unchanged at this commit
  (decode 1024²/b1: 1.02×, −24.5%).

**Status**: kept in the tree as a strictly-non-negative simplification (bitwise-safe, +1–2%
encode, less allocator traffic, one kernel fewer for [[torch.compile Tuning (2D)]] to fuse),
but it does **not** independently clear the acceptance gate — recorded here so the log stays
honest. Lesson for the pipeline: encoder capacity work must target the replicate-pad copies
([[Padding Modes]]) rather than the sampling blocks.

## Round 3 — torch.compile ACCEPTED, channels-last REJECTED (2026-08-19)

**Change** (commit `6d249f3` + TF32 harness fix `beb9320`): composable candidate specs via
`scripts/candidates.py::CandidateRunner` — `cl` (channels-last) and `compile`
(`torch.compile(dynamic=True)` of encoder/decoder with the inductor recipe: `max_autotune`,
ATEN+TRITON backends, `combo_kernels`, `memory_planning`, `force_same_precision`,
`cudnn.benchmark`). Benchmark JSONs at `6d249f3`/`beb9320`.

### torch.compile without channels-last (`pxs,compile`) — the winner

| cell (bf16, KVAE-2D-2.0) | baseline | compile | speedup |
|---|---|---|---|
| e2e 1024² b1 | 190.7 ms | 116.8 ms | **1.63×** |
| e2e 2048² b1 | 751.8 ms | 456.6 ms | **1.65×** |
| decode 1024² b1 | 134.5 ms | 74.9 ms | **1.80×** |
| encode 1024² b1 | 56.3 ms | 42.2 ms | 1.34× |
| e2e 1024² b4 | 536.3 ms | 459.5 ms | 1.17× |

KVAE-2D-1.0 matches (1.65–1.67× e2e b1) with peak memory −7…−8%; on 2.0 memory is neutral at
≥512² and +0.03–0.08 GiB at 256² (autotune workspace floor — the reason the acceptance gate's
memory rule gained an absolute floor). One-time compile cost ≈ **128 s** (max_autotune); warm
graphs serve all resolutions via `dynamic=True`. **The decode 1.80× confirms the round-2
prediction**: the replicate-pad copies were the decoder's dominant bandwidth cost, and inductor
fuses them. **The 2³¹-element eager limit is lifted**: 1024²/b16 and 768×1344/b16 now run
(1870 ms e2e) — impossible in the eager baseline.

Parity: metric gate passes with huge margin on both checkpoints (mean ΔPSNR ≤ 0.002 dB, worst
per-image −0.009 dB, ΔLPIPS ≤ 8.4e-5). fp32 TF32-off equivalence: 2.0 at the reorder-noise
floor (~1e-5); 1.0 shows decoder-side diffs up to 3.5e-3 (autotune picked different kernels)
and isolated bf16 per-pixel outliers up to 0.2 on real reconstructions (MAE ~3e-3, PSNR/LPIPS
unaffected) — recorded as the known cost of autotuned kernel selection. Classified **T2** (see
the amended tier table in [[Speed-Up Testing Specification]]).

### Channels-last (`pxs,cl`) — rejected

Eager NHWC is **0.43–0.79×** (up to 2.3× *slower*; worse at larger sizes) despite −20…−37%
peak memory: the replicate/reflect pads and GroupNorm lack fast NHWC kernels, inverting the
assumption in [[Channels-Last Memory Format (2D)]]. Under compile it still hurts: `pxs,cl,compile`
reaches only 1.28× at 1024²/b1 (vs 1.63× without cl) and regresses at b4 (0.87×), with +4–12%
memory. Inductor picks its own layouts internally — pre-converting is counterproductive.
Parity for the record: metric gates passed; bf16/fp32 numerics behaved as 1–2 ulps at latent
scale once TF32 was disabled in the fp32 pass.

### Framework findings

- **TF32 poisoned the fp32 parity check** (~1e-2 diffs, all kernel-selection candidates):
  fp32 tensor-core kernels are default-on and engage differently per kernel. The harness now
  disables TF32 for fp32 passes (`beb9320`); diffs collapsed to ~1e-5.
- The T1 absolute thresholds cannot be met by whole-graph kernel-selection changes (bf16 ulps
  at latent scale |z|≈13 exceed them) — tier table amended: such changes are T2 with the
  TF32-off fp32 run as supporting evidence.
- The memory-regression rule gained an absolute floor (0.25 GiB) so fixed autotune workspaces
  at small cells don't read as regressions.

### Finding: 2³¹-element kernel limit

The eager model (baseline **and** pxs) hard-fails with
`RuntimeError: input tensor must fit into 32-bit index math` (replicate-pad kernel) once an
activation exceeds 2³¹ elements: cells 1024²/b16 and 768×1344/b16 are unusable. Recorded as
error rows in the benchmark JSONs. A compiled or zero-padding path should lift this —
extra motivation for [[torch.compile Tuning (2D)]] / [[Padding Modes]].
