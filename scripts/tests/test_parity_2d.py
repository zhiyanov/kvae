"""End-to-end A/B parity harness (Speed-Up Testing Specification, Part 1).

Runs the baseline (reference) and candidate implementations in the same
process, over synthetic and real inputs, and checks the declared parity tier:

  T0: bitwise — torch.equal on latent and reconstruction, every input.
  T1: fp32 max_abs <= 1e-5; bf16 max_abs <= 5e-2 and MAE <= 1e-3, plus the T2
      metric gate on the real images: |dPSNR| <= 0.05 dB and |dLPIPS| <= 0.001
      (means), per-image worst case dPSNR >= -0.2 dB.

Not a pytest module — run directly on the remote machine from the repo root:

    .venv/bin/python -m scripts.tests.test_parity_2d \
        --model KVAE_2.0 --candidate pxs --tier T0 --device 0

Exits nonzero if any input fails the gate.
"""

import argparse
import sys
from pathlib import Path

import torch

from data.image_dataset import ImageDataset
from kvae.models import KVAEImage
from scripts.candidates import CandidateRunner, parse_candidate

MODEL_PATHS = {
    "KVAE_1.0": "kandinskylab/KVAE-2D-1.0",
    "KVAE_2.0": "kandinskylab/KVAE-2D-2.0",
}

SYNTH_SIZES = [(256, 256, 1), (512, 512, 1), (1024, 1024, 1), (768, 1344, 1), (512, 512, 4)]
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
T1_GATES = {"fp32": {"max_abs": 1e-5}, "bf16": {"max_abs": 5e-2, "mae": 1e-3}}
METRIC_GATES = {"d_psnr_mean": 0.05, "d_lpips_mean": 1e-3, "d_psnr_worst": -0.2}
BORDER_PX = 16


def synth_inputs():
    g = torch.Generator().manual_seed(111)
    for h, w, b in SYNTH_SIZES:
        yield f"random_{h}x{w}_b{b}", torch.rand((b, 3, h, w), generator=g) * 2 - 1
    h = w = 512
    yield "constant_512", torch.full((1, 3, h, w), 0.5)
    ramp = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(1, 3, h, w).contiguous()
    yield "ramp_512", ramp
    checker = (
        ((torch.arange(h).view(h, 1) // 8 + torch.arange(w).view(1, w) // 8) % 2) * 2 - 1
    ).float().expand(1, 3, h, w).contiguous()
    yield "checker_512", checker


def real_inputs(data_dir):
    ds = ImageDataset(root_path=data_dir, regex="*.png")
    for i in range(len(ds)):
        item = ds[i]
        x = item["frames"]
        _, h, w = x.shape
        # Center-crop to the shape contract (H, W divisible by 8).
        h8, w8 = h // 8 * 8, w // 8 * 8
        top, left = (h - h8) // 2, (w - w8) // 2
        yield Path(item["paths"]).name, x[:, top : top + h8, left : left + w8].unsqueeze(0)


def run_pass(runner, x):
    with torch.no_grad():
        lat = runner.encode(x)
        rec = runner.decode(lat)
    return lat, rec


def border_interior_max(diff, px=BORDER_PX):
    interior = diff[..., px:-px, px:-px]
    mask = torch.zeros_like(diff, dtype=torch.bool)
    mask[..., px:-px, px:-px] = True
    border = diff[~mask]
    return (
        border.max().item() if border.numel() else 0.0,
        interior.max().item() if interior.numel() else 0.0,
    )


def main():
    p = argparse.ArgumentParser(description="KVAE 2D speed-up parity harness")
    p.add_argument("--model", choices=sorted(MODEL_PATHS), default="KVAE_2.0")
    p.add_argument("--candidate", required=True,
                   help="comma-separated composable spec, e.g. 'pxs' or 'pxs,cl,compile'")
    p.add_argument("--tier", choices=["T0", "T1"], required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dtypes", default="bf16,fp32")
    p.add_argument("--data-dir", default="./assets/images")
    args = p.parse_args()

    if "baseline" in parse_candidate(args.candidate):
        raise SystemExit("--candidate must not be 'baseline'")

    device = torch.device(f"cuda:{args.device}")
    vae = KVAEImage.from_pretrained(MODEL_PATHS[args.model]).eval().to(device)
    runner_base = CandidateRunner(vae, "baseline")
    runner_cand = CandidateRunner(vae, args.candidate)

    header = f"{'input':<28}{'dtype':>6}{'bitwise':>9}{'lat max':>11}{'rec max':>11}{'rec MAE':>11}{'border':>11}{'interior':>11}  verdict"
    print(f"parity: {args.model}  candidate={args.candidate}  tier={args.tier}")
    print(header)
    print("-" * len(header))

    n_fail = 0
    for dtype_name in args.dtypes.split(","):
        dtype = DTYPES[dtype_name]
        vae.to(dtype)
        inputs = [(n, x, False) for n, x in synth_inputs()] + [
            (n, x, True) for n, x in real_inputs(args.data_dir)
        ]

        # The T2 metric gate is part of T1 acceptance in bf16 (testing spec).
        need_metrics = args.tier == "T1" and dtype_name == "bf16"
        if need_metrics:
            from torchmetrics.functional.image import peak_signal_noise_ratio
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

            lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
            metric_rows = []

        for name, x, is_real in inputs:
            x = x.to(device=device, dtype=dtype)
            lat_b, rec_b = run_pass(runner_base, x)
            lat_c, rec_c = run_pass(runner_cand, x)

            bitwise = torch.equal(lat_b, lat_c) and torch.equal(rec_b, rec_c)
            dlat = (lat_b.float() - lat_c.float()).abs()
            drec = (rec_b.float() - rec_c.float()).abs()
            lat_max, rec_max, mae = dlat.max().item(), drec.max().item(), drec.mean().item()
            border, interior = border_interior_max(drec)

            if args.tier == "T0":
                ok = bitwise
            else:
                gate = T1_GATES[dtype_name]
                ok = max(lat_max, rec_max) <= gate["max_abs"] and mae <= gate.get("mae", float("inf"))
            n_fail += 0 if ok else 1
            print(
                f"{name:<28}{dtype_name:>6}{str(bitwise):>9}{lat_max:>11.3e}{rec_max:>11.3e}"
                f"{mae:>11.3e}{border:>11.3e}{interior:>11.3e}  {'PASS' if ok else 'FAIL'}"
            )

            if need_metrics and is_real:
                xf = x.float()
                rb, rc = rec_b.float().clip(-1, 1), rec_c.float().clip(-1, 1)
                metric_rows.append((
                    name,
                    peak_signal_noise_ratio(rb, xf, data_range=(-1.0, 1.0)).item(),
                    peak_signal_noise_ratio(rc, xf, data_range=(-1.0, 1.0)).item(),
                    lpips_metric(rb, xf).item(),
                    lpips_metric(rc, xf).item(),
                ))
            del lat_b, rec_b, lat_c, rec_c, dlat, drec

        if need_metrics:
            d_psnr = [c - b for _, b, c, _, _ in metric_rows]
            d_lpips_mean = sum(c - b for _, _, _, b, c in metric_rows) / len(metric_rows)
            d_psnr_mean = sum(d_psnr) / len(d_psnr)
            print(f"metric gate (bf16, {len(metric_rows)} real images):")
            for (name, pb, pc, lb, lc), dp in zip(metric_rows, d_psnr):
                print(f"  {name:<26} PSNR {pb:7.3f} -> {pc:7.3f} ({dp:+.3f} dB)"
                      f"   LPIPS {lb:.4f} -> {lc:.4f} ({lc - lb:+.5f})")
            metrics_ok = (
                abs(d_psnr_mean) <= METRIC_GATES["d_psnr_mean"]
                and abs(d_lpips_mean) <= METRIC_GATES["d_lpips_mean"]
                and min(d_psnr) >= METRIC_GATES["d_psnr_worst"]
            )
            n_fail += 0 if metrics_ok else 1
            print(f"  mean dPSNR {d_psnr_mean:+.4f} dB, mean dLPIPS {d_lpips_mean:+.6f}, "
                  f"worst dPSNR {min(d_psnr):+.4f} dB  {'PASS' if metrics_ok else 'FAIL'}")

    print("-" * len(header))
    print(f"{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'} (tier {args.tier})")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
