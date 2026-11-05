"""Shape-contract tests for KVAEImage (Speed-Up Testing Specification, Part 1).

Run from the repo root: `.venv/bin/python -m pytest scripts/tests -q`
"""

import pytest
import torch

from kvae.models import KVAEImage


@pytest.fixture(scope="module")
def model():
    # Reduced-width random model: the contract is architectural, weights don't matter.
    # channels=32 keeps every GroupNorm(num_groups=32) valid.
    torch.manual_seed(0)
    return KVAEImage(channels=32, num_enc_blocks=1, num_dec_blocks=1, z_channels=4).eval()


@pytest.mark.parametrize("hw", [(64, 64), (72, 80), (256, 256)])
def test_round_trip_shapes(model, hw):
    h, w = hw
    x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        lat = model.encode(x).latent_dist.mode()
        rec = model.decode(lat)
    assert lat.shape[-2:] == (h // 8, w // 8)
    assert rec.shape == x.shape


@pytest.mark.parametrize("hw", [(66, 66), (68, 68), (64, 100)])
def test_non_divisible_by_8_fails_loudly(model, hw):
    # H and W must be divisible by 8 (three PixelUnshuffle(2) stages); anything
    # else must raise rather than silently mis-shape.
    x = torch.randn(1, 3, *hw)
    with pytest.raises(RuntimeError), torch.no_grad():
        model.decode(model.encode(x).latent_dist.mode())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 is the CUDA production path")
def test_dtype_preserved_bf16():
    torch.manual_seed(0)
    m = (
        KVAEImage(channels=32, num_enc_blocks=1, num_dec_blocks=1, z_channels=4)
        .eval()
        .to("cuda", torch.bfloat16)
    )
    x = torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        lat = m.encode(x).latent_dist.mode()
        rec = m.decode(lat)
    assert lat.dtype == torch.bfloat16
    assert rec.dtype == torch.bfloat16


def test_batch_independence(model):
    # f(cat(a, b)) == cat(f(a), f(b)) within T1 fp32 tolerance (GroupNorm is
    # per-sample, so batching must not couple samples).
    torch.manual_seed(1)
    a, b = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ra, rb = model(a), model(b)
        rab = model(torch.cat([a, b]))
    assert torch.allclose(rab, torch.cat([ra, rb]), atol=1e-5, rtol=1e-4)
