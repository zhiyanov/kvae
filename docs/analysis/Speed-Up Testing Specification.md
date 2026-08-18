---
tags: [analysis, validation, specification]
aliases: [testing spec, parity tests, benchmarking]
---

# Speed-Up Testing Specification

How every optimization of [[KVAE Image]] is validated. Two axes, mirroring the video repo's
`metrics_v3.py` / `times_v3.py` split: **parity** (the speed-up does not change inference) and
**speed** (it is actually faster). No optimization is accepted on one axis alone.

## Ground rules

- **Everything runs on the remote machine** (`agd`, per `CLAUDE.md`): single pinned GPU
  (`CUDA_VISIBLE_DEVICES`), `nvidia-smi` confirmed otherwise idle, long runs under `nohup`.
- **The baseline is the current eager implementation at a pinned commit.** Every result records `git rev-parse HEAD`; runs on a dirty tree are invalid.
- **Baseline and candidate run in the same process** (same checkpoint object, same inputs, same cuDNN autotune state). No stored "golden" tensors — goldens go stale across
  torch/driver/checkpoint changes and can't be committed anyway (size).
- Both checkpoints are covered: `KVAE-2D-1.0` and `KVAE-2D-2.0`.
- Fixed seed; `latent_dist.mode()` only (no sampling) so forwards are deterministic given the same kernels.

## Shape contract

Input H and W must be **divisible by 8** — each of the three `PixelUnshuffle(2)` stages needs an
even extent (verified: 66² and 68² raise `RuntimeError`, 72×80 works: latent `(9,10)`, exact
round-trip shape). The test grid uses only conforming sizes; a dedicated contract test asserts
that conforming sizes round-trip `(H, W) → (H/8, W/8) → (H, W)` and non-conforming sizes fail loudly rather than silently mis-shape.

## Part 1 — Parity tests (inference does not change)

### Equivalence tiers

Each optimization declares its tier up front; the tier determines the pass criterion.

| tier                            | claim                                   | check                                                                                                                                            | applies to                                                                                                                             |
| ------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| **T0 — bitwise**                | mathematically identical, same op order | `torch.equal` on latent and reconstruction                                                                                                       | [[PXS Simplifications]] upsample rewrite (values are copies)                                                                           |
| **T1 — numerically equivalent** | same math, different op/kernel order    | fp32: `max_abs ≤ 1e-5` on latent and reconstruction. bf16: `max_abs ≤ 5e-2`, `MAE ≤ 1e-3` on reconstruction in [-1,1] **and** the T2 metric gate | PXS downsample rewrite (sum reorder), [[Channels-Last Memory Format (2D)]], [[torch.compile Tuning (2D)]], exact tiling (post-retrain) |
| **T2 — metric parity**          | output may differ, quality must not     | on the eval set: `ΔPSNR ≤ 0.05 dB`, `ΔLPIPS ≤ 0.001` (means), and per-image worst case `ΔPSNR ≥ -0.2 dB`                                         | approximate [[Spatial Tiling (2D)]]; fallback gate for every T1 change in bf16                                                         |

Rationale for the split: parity is checked **in fp32 with tight tolerances** to isolate
*algorithmic* differences from bf16 noise (bf16's ~0.4% relative eps × reduction reordering can
legitimately produce ~1e-2 output diffs that say nothing about correctness), and **in bf16 with
the metric gate** to validate the actual production path.

### Test inputs

- **Synthetic**: fixed-seed random tensors + structured patterns (constant, ramp, checkerboard – edge/padding bugs show up on structured inputs, not noise).
- **Real**: every image in `assets/image_test` (the shipped eval set).
- **Size grid**: 256², 512², 1024², and non-square 768×1344 (aspect-ratio bugs), batch {1, 4}.
- **Boundary attention**: parity failures concentrate at image borders ([[Padding Modes]]) and —
  for tiling — at tile seams. The harness reports max-abs-diff **separately for the border
  region (16 px) and the interior**, and for tiling additionally along seam lines.

### Deliverables

- `scripts/tests/test_layers_2d.py` — module-level unit tests (pytest): old vs new
  `PXSDownsample`/`PXSUpsample`/ any rewritten layer on random + structured tensors, CPU and GPU, fp32 and bf16, asserting the declared tier.
- `scripts/tests/test_parity_2d.py` — end-to-end A/B harness: loads one checkpoint, runs
  baseline and candidate paths in-process over the test inputs, prints a per-tier report
  (max-abs / MAE / border-vs-interior / ΔPSNR / ΔLPIPS) and exits nonzero on gate failure.
  Candidate selection via flags mirroring the video repo (`--candidate compile|cl|pxs|tile`,
  composable) so each technique is testable in isolation and in combination.
- `scripts/tests/test_contract_2d.py` — shape contract (round-trip shapes, divisibility
  failures, dtype preservation, batch independence: `f(cat(a,b)) == cat(f(a), f(b))` within T1).

## Part 2 — Benchmarks (it is actually faster)

### Measurement protocol

- **Timing**: CUDA events around `encode`, `decode`, and end-to-end separately;
  `torch.cuda.synchronize()` before start/after stop; host-side `perf_counter` recorded as a
  cross-check. **Warmup ≥ 10 iterations** (excluded), then **median and p90 over ≥ 50
  iterations**. For compiled candidates, cold-start (first-call) time is reported as its own
  number, never mixed into steady-state.
- **Memory**: `torch.cuda.reset_peak_memory_stats()` per phase; report `max_memory_allocated` for encode and decode separately (the decode peak is the
  binding one — see [[Decoder2D]]).
- **Throughput normalization**: report ms and megapixels/s, so batch and resolution rows are
  comparable.

### Grid

| dimension | values |
|---|---|
| resolution | 256², 512², 1024², 2048², 768×1344; 4096² for tiling candidates |
| batch | 1, 4, 16 (at ≤1024²) |
| dtype | bf16 (production); fp32 single row at 512² as sanity |
| checkpoint | KVAE-2D-1.0, KVAE-2D-2.0 |
| phase | encode, decode, end-to-end |

The full grid shards across the 8 GPUs as one `nohup` job per GPU (per `CLAUDE.md`), each GPU
sweeping a disjoint slice; a run manifest maps slice → GPU → log file.

### Result format

One JSON per run in `outputs/bench/` (gitignored):
`{meta: {commit, candidate flags, torch/cuda versions, GPU name, driver, timestamp},
results: [{checkpoint, phase, H, W, batch, dtype, ms_p50, ms_p90, ms_cold, mpix_per_s,
peak_mem_bytes, n_iters}]}`. Raw JSONs are the source of truth; tables are derived.

### Deliverables

- `scripts/benchmark_2d.py` — grid runner → JSON (flags: `--candidate`, `--grid full|smoke`,
  `--gpu-slice i/n`).
- `scripts/compare_bench_2d.py` — takes two JSONs (baseline, candidate), prints a table of
  speedup factors and memory deltas per grid cell, and applies the acceptance gate below.

## Part 3 — Acceptance gate per optimization

An optimization is accepted when, at its declared parity tier:

1. all parity tests pass (Part 1);
2. median end-to-end time improves by **≥ 3%** on at least the 1024²/batch-1 cell it targets,
   with **no cell regressing by > 2%** (p50) and **no peak-memory regression > 5%** — unless the
   optimization explicitly trades one for the other (e.g. tiling trades time for fitting, gate
   is then memory at the target resolution);
3. the benchmark JSONs (baseline + candidate, same GPU, same day) are archived with the PR/commit message referencing their paths and the commit hashes they were produced from.

Expected declarations for the currently planned work ([[Technique Applicability]]):

| optimization | parity tier | primary gate |
|---|---|---|
| PXS upsample rewrite | T0 | decode time + decode peak memory |
| PXS downsample rewrite | T1 (fp32 tight) | encode time + encode peak memory |
| channels-last | T1 | encode/decode time |
| torch.compile | T1 | encode/decode time; cold-start reported |
| batching defaults | T0 (per-image outputs identical within T1 batch-independence) | mpix/s |
| approximate tiling | T2 | peak memory at 4096²; ΔPSNR/ΔLPIPS within gate |

## Part 4 — Order of implementation

1. `test_contract_2d.py` + `benchmark_2d.py` + baseline JSONs on both checkpoints — the
   baseline must exist before any optimization lands.
2. `test_layers_2d.py` with the PXS identities as its first cases (they are also the first
   planned optimization).
3. `test_parity_2d.py` A/B harness, extended candidate-by-candidate as optimizations arrive.
