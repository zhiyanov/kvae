"""Benchmark grid runner for KVAEImage (Speed-Up Testing Specification, Part 2).

Measures encode / decode / end-to-end wall time (CUDA events, warmup excluded,
p50/p90 over repeated iterations, cold-start reported separately) and per-phase
peak memory over the resolution x batch x dtype grid. Writes one JSON per run
to outputs/bench/.

Run on the remote machine from the repo root (long-running — use nohup):

    .venv/bin/python -m scripts.benchmark_2d \
        --model KVAE_2.0 --candidate baseline --grid full --device 0

Shard the grid across GPUs with --shard i/n (disjoint cell slices).
"""

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from kvae.models import KVAEImage
from scripts.candidates import CandidateRunner

MODEL_PATHS = {
    "KVAE_1.0": "kandinskylab/KVAE-2D-1.0",
    "KVAE_2.0": "kandinskylab/KVAE-2D-2.0",
}

FULL_RES = [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (768, 1344)]
SMOKE_RES = [(256, 256), (512, 512)]
MAX_BATCH_PIXELS = 1024 * 1024  # batch 4/16 rows only at or below this size
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def build_cells(grid: str, include_4k: bool):
    cells = []
    for h, w in FULL_RES if grid == "full" else SMOKE_RES:
        cells.append((h, w, 1, "bf16"))
        if grid == "full" and h * w <= MAX_BATCH_PIXELS:
            cells += [(h, w, 4, "bf16"), (h, w, 16, "bf16")]
    if grid == "full":
        cells.append((512, 512, 1, "fp32"))  # sanity row
    if include_4k:
        cells.append((4096, 4096, 1, "bf16"))
    # Group by dtype so the model converts at most once per dtype.
    return sorted(cells, key=lambda c: c[3])


def git_meta():
    def run(*a):
        return subprocess.run(a, capture_output=True, text=True).stdout.strip()

    return run("git", "rev-parse", "HEAD"), run("git", "status", "--porcelain", "-uno") != ""


def bench_phase(fn, arg, warmup: int, iters: int):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(arg)
    torch.cuda.synchronize()
    cold = (time.perf_counter() - t0) * 1e3

    for _ in range(max(warmup - 1, 0)):
        fn(arg)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    host0 = time.perf_counter()
    for _ in range(iters):
        start.record()
        fn(arg)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    host_mean = (time.perf_counter() - host0) * 1e3 / iters
    times.sort()
    return {
        "ms_cold": cold,
        "ms_p50": statistics.median(times),
        "ms_p90": times[min(len(times) - 1, round(0.9 * (len(times) - 1)))],
        "ms_host_mean": host_mean,
        "peak_mem_bytes": torch.cuda.max_memory_allocated(),
        "n_iters": iters,
    }, out


def main():
    p = argparse.ArgumentParser(description="KVAE 2D benchmark grid runner")
    p.add_argument("--model", choices=sorted(MODEL_PATHS), default="KVAE_2.0")
    p.add_argument("--candidate", required=True,
                   help="comma-separated composable spec, e.g. 'baseline' or 'pxs,cl,compile'")
    p.add_argument("--grid", choices=["full", "smoke"], default="full")
    p.add_argument("--include-4k", action="store_true", help="add the 4096x4096 tiling cell")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--shard", default="0/1", help="i/n: run the i-th of n disjoint cell slices")
    p.add_argument("--out", default=None, help="output JSON path (default: outputs/bench/...)")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()

    commit, dirty = git_meta()
    if dirty and not args.allow_dirty:
        raise SystemExit("refusing to benchmark a dirty tree (spec: runs on a dirty tree are invalid); use --allow-dirty to override")

    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(111)

    vae = KVAEImage.from_pretrained(MODEL_PATHS[args.model]).eval().to(device)
    runner = CandidateRunner(vae, args.candidate)

    shard_i, shard_n = (int(s) for s in args.shard.split("/"))
    cells = build_cells(args.grid, args.include_4k)[shard_i::shard_n]

    results, current_dtype = [], None
    g = torch.Generator().manual_seed(111)
    for h, w, batch, dtype_name in cells:
        if dtype_name != current_dtype:
            vae.to(DTYPES[dtype_name])
            runner.activate()  # forwards, memory format, compiled/eager modules
            current_dtype = dtype_name
        label = f"{args.model} {h}x{w} b{batch} {dtype_name}"
        try:
            with torch.no_grad():
                x = runner.prep(
                    (torch.rand((batch, 3, h, w), generator=g) * 2 - 1).to(
                        device, DTYPES[dtype_name]
                    )
                )
                phases = {}
                phases["encode"], lat = bench_phase(
                    lambda t: vae.encode(t).latent_dist.mode(), x, args.warmup, args.iters
                )
                phases["decode"], _ = bench_phase(vae.decode, lat, args.warmup, args.iters)
                phases["e2e"], _ = bench_phase(
                    lambda t: vae.decode(vae.encode(t).latent_dist.mode()),
                    x,
                    args.warmup,
                    args.iters,
                )
                del x, lat
            for phase, r in phases.items():
                results.append(
                    {"checkpoint": args.model, "phase": phase, "H": h, "W": w,
                     "batch": batch, "dtype": dtype_name,
                     "mpix_per_s": batch * h * w / 1e6 / (r["ms_p50"] / 1e3), **r}
                )
            print(f"{label}: e2e p50 {phases['e2e']['ms_p50']:.2f} ms, "
                  f"decode peak {phases['decode']['peak_mem_bytes'] / 2**30:.2f} GiB", flush=True)
        except RuntimeError as e:  # OOM, 32-bit-index limits of eager kernels, ...
            torch.cuda.empty_cache()
            err = "OOM" if isinstance(e, torch.OutOfMemoryError) else str(e).splitlines()[-1][:120]
            results.append({"checkpoint": args.model, "phase": "all", "H": h, "W": w,
                            "batch": batch, "dtype": dtype_name, "error": err})
            print(f"{label}: ERROR {err}", flush=True)
        torch.cuda.empty_cache()

    out_path = Path(args.out) if args.out else Path("outputs/bench") / (
        f"{args.model}_{runner.spec.replace(',', '+')}_{commit[:7]}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        + (f"_shard{shard_i}of{shard_n}" if shard_n > 1 else "") + ".json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "commit": commit, "dirty": dirty, "candidate": runner.spec,
        "model": args.model, "grid": args.grid, "shard": args.shard,
        "warmup": args.warmup, "iters": args.iters,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
