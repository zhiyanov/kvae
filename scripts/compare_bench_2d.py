"""Compare two benchmark JSONs and apply the acceptance gate (Testing Spec, Part 3).

    .venv/bin/python -m scripts.compare_bench_2d baseline.json candidate.json

Gate (time mode, the default): the target cell (1024x1024, batch 1, bf16, e2e)
must improve by >= 3% (p50); no cell may regress by > 2% (p50) or > 5% peak
memory. With --gate memory the primary criterion is instead peak memory at the
target cell. Exits nonzero if the gate fails.
"""

import argparse
import json
import sys

CELL_KEY = ("checkpoint", "phase", "H", "W", "batch", "dtype")


def load(path):
    data = json.loads(open(path).read())
    cells = {}
    for r in data["results"]:
        if "error" in r:
            continue
        cells[tuple(r[k] for k in CELL_KEY)] = r
    return data["meta"], cells


def main():
    p = argparse.ArgumentParser(description="Compare KVAE 2D benchmark runs")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--gate", choices=["time", "memory"], default="time")
    p.add_argument("--target", default="1024x1024x1", help="target cell HxWxB")
    p.add_argument("--target-phase", default="e2e")
    p.add_argument("--min-improvement", type=float, default=0.03)
    p.add_argument("--max-time-regression", type=float, default=0.02)
    p.add_argument("--max-mem-regression", type=float, default=0.05)
    args = p.parse_args()

    meta_b, base = load(args.baseline)
    meta_c, cand = load(args.candidate)
    th, tw, tb = (int(v) for v in args.target.split("x"))

    print(f"baseline : {meta_b['candidate']} @ {meta_b['commit'][:9]} ({meta_b['gpu']})")
    print(f"candidate: {meta_c['candidate']} @ {meta_c['commit'][:9]} ({meta_c['gpu']})")
    if meta_b["gpu"] != meta_c["gpu"]:
        print("WARNING: different GPUs — comparison invalid")

    header = (f"{'cell':<38}{'base p50':>10}{'cand p50':>10}{'speedup':>9}"
              f"{'base GiB':>10}{'cand GiB':>10}{'mem Δ':>8}")
    print(header)
    print("-" * len(header))

    common = sorted(set(base) & set(cand))
    if not common:
        raise SystemExit("no common cells between the two runs")
    target_speedup, failures = None, []
    for key in common:
        rb, rc = base[key], cand[key]
        speedup = rb["ms_p50"] / rc["ms_p50"]
        dmem = (rc["peak_mem_bytes"] - rb["peak_mem_bytes"]) / rb["peak_mem_bytes"]
        ckpt, phase, h, w, batch, dtype = key
        cell = f"{ckpt} {phase} {h}x{w} b{batch} {dtype}"
        print(f"{cell:<38}{rb['ms_p50']:>10.2f}{rc['ms_p50']:>10.2f}{speedup:>8.2f}x"
              f"{rb['peak_mem_bytes'] / 2**30:>10.2f}{rc['peak_mem_bytes'] / 2**30:>10.2f}"
              f"{dmem:>+8.1%}")

        is_target = (phase, h, w, batch, dtype) == (args.target_phase, th, tw, tb, "bf16")
        if is_target:
            target_speedup = speedup
            target_dmem = dmem
        if args.gate == "time" and speedup < 1 - args.max_time_regression:
            failures.append(f"time regression at {cell}: {speedup:.3f}x")
        if dmem > args.max_mem_regression and not (args.gate == "memory" and not is_target):
            failures.append(f"memory regression at {cell}: {dmem:+.1%}")

    print("-" * len(header))
    if target_speedup is None:
        failures.append(f"target cell {args.target} {args.target_phase} missing from both runs")
    elif args.gate == "time" and target_speedup < 1 + args.min_improvement:
        failures.append(
            f"target improvement {target_speedup - 1:+.1%} below required {args.min_improvement:+.0%}"
        )
    elif args.gate == "memory" and target_dmem > -args.min_improvement:
        failures.append(
            f"target memory delta {target_dmem:+.1%} does not improve by {args.min_improvement:.0%}"
        )

    if failures:
        print("GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"GATE PASSED (target {args.target_phase} @ {args.target}: {target_speedup:.3f}x)")


if __name__ == "__main__":
    main()
