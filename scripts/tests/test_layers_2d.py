"""Module-level parity tests for rewritten layers (Testing Spec, Part 1).

Cases: the T0 (bitwise) PXSUpsample rewrite and the T1 (numerically
equivalent) PXSDownsample rewrite — reference vs library forward on random and
structured inputs, CPU and GPU, fp32 and bf16.

Run from the repo root: `.venv/bin/python -m pytest scripts/tests -q`
"""

import pytest
import torch
import torch.nn.functional as F

from kvae.layers.sampling import PXSDownsample, PXSUpsample
from scripts.candidates import (
    _pxs_downsample_forward_reference,
    _pxs_upsample_forward_reference,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.bfloat16]


def _patterns(b, c, h, w):
    g = torch.Generator().manual_seed(0)
    ramp = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(b, c, h, w).contiguous()
    checker = (
        ((torch.arange(h).view(h, 1) // 8 + torch.arange(w).view(1, w) // 8) % 2) * 2 - 1
    ).float().expand(b, c, h, w).contiguous()
    return {
        "random": torch.rand((b, c, h, w), generator=g) * 2 - 1,
        "constant": torch.full((b, c, h, w), 0.5),
        "ramp": ramp,
        "checkerboard": checker,
    }


def test_shuffle_of_repeat_is_nearest_upsample():
    # The algebraic identity behind the rewrite, in isolation.
    x = torch.randn(2, 8, 6, 10)
    shuffled = F.pixel_shuffle(x.repeat_interleave(4, dim=1), 2)
    nearest = F.interpolate(x, scale_factor=2, mode="nearest")
    assert torch.equal(shuffled, nearest)


def test_unshuffle_mean_is_avg_pool():
    # The algebraic identity behind the downsample rewrite, in fp64.
    x = torch.randn(2, 8, 12, 20, dtype=torch.float64)
    u = F.pixel_unshuffle(x, 2)
    b, c, h, w = u.shape
    m = u.view(b, c // 4, 4, h, w).mean(dim=2)
    assert torch.allclose(m, F.avg_pool2d(x, 2), atol=1e-12, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 8, 16, 16), (2, 64, 24, 40)])
def test_pxs_downsample_rewrite_is_t1(device, dtype, shape):
    # Same math, possibly different summation order in the pooling kernel ->
    # T1 tolerances: tight in fp32, loose in bf16 (see the testing spec).
    atol = 1e-6 if dtype == torch.float32 else 5e-2
    torch.manual_seed(0)
    b, c, h, w = shape
    mod = PXSDownsample(in_channels=c).to(device=device, dtype=dtype).eval()
    for name, x in _patterns(b, c, h, w).items():
        x = x.to(device=device, dtype=dtype)
        with torch.no_grad():
            y_ref = _pxs_downsample_forward_reference(mod, x)
            y_new = PXSDownsample.forward(mod, x)
        max_abs = (y_ref.float() - y_new.float()).abs().max().item()
        assert max_abs <= atol, f"{name} @ {device}/{dtype}: max abs diff {max_abs:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 8, 16, 16), (2, 64, 24, 40)])
def test_pxs_upsample_rewrite_is_bitwise(device, dtype, shape):
    torch.manual_seed(0)
    b, c, h, w = shape
    mod = PXSUpsample(in_channels=c).to(device=device, dtype=dtype).eval()
    for name, x in _patterns(b, c, h, w).items():
        x = x.to(device=device, dtype=dtype)
        with torch.no_grad():
            y_ref = _pxs_upsample_forward_reference(mod, x)
            y_new = PXSUpsample.forward(mod, x)
        assert torch.equal(y_ref, y_new), (
            f"{name} @ {device}/{dtype}: "
            f"max abs diff {(y_ref.float() - y_new.float()).abs().max().item():.3e}"
        )
