"""A/B candidate switching for parity tests and benchmarks.

Optimized implementations live in the library code; the *reference*
(pre-speed-up, pinned-baseline) implementations are preserved here verbatim so
that baseline and candidate can run in the same process — the requirement of
docs/analysis/Speed-Up Testing Specification.md ("no stored golden tensors").

A candidate is a comma-separated spec of composable names:

  baseline  — the pinned reference forwards, eager, NCHW (exclusive)
  pxs       — the PXS sampling-block rewrites (the current library forwards)
  cl        — channels-last memory format for weights and inputs
  compile   — torch.compile(dynamic=True) encoder/decoder with the inductor
              recipe from docs/techniques/torch.compile Tuning (2D).md

Use `CandidateRunner(vae, spec)`: `.activate()` swaps the forwards, memory
format, and compiled/eager modules; `.encode/.decode` activate lazily per call
(for interleaved A/B); `.prep(t)` converts an input to the spec's memory
format. Activation is global (class-level forwards, in-place weight layout),
so two runners over the same model must not run concurrently.
"""

import torch
import torch.nn.functional as F

from kvae.layers.sampling import PXSDownsample, PXSUpsample

VALID_NAMES = {"baseline", "pxs", "cl", "compile"}


def _pxs_upsample_forward_reference(self, x: torch.Tensor) -> torch.Tensor:
    """PXSUpsample.forward as of the pinned baseline commit.

    (`self.shuffle` was an `nn.PixelShuffle(self.factor)`; the functional call
    below is the same operator.)
    """
    repeated = x.repeat_interleave(self.factor**2, dim=1)
    pxs_interm = F.pixel_shuffle(repeated, self.factor)

    image_like_ups = F.interpolate(x, scale_factor=2, mode="nearest")
    conv_out = self.spatial_conv(image_like_ups)

    out = conv_out + pxs_interm
    return self.linear(out)


def _pxs_downsample_forward_reference(self, x: torch.Tensor) -> torch.Tensor:
    """PXSDownsample.forward as of the pinned baseline commit.

    (`self.unshuffle` was an `nn.PixelUnshuffle(self.factor)`.)
    """
    pxs_interm = F.pixel_unshuffle(x, self.factor)

    b, c, h, w = pxs_interm.shape
    pxs_interm_view = pxs_interm.view(b, c // self.factor**2, self.factor**2, h, w)
    pxs_out = torch.mean(pxs_interm_view, dim=2)

    conv_out = self.spatial_conv(x)

    out = conv_out + pxs_out
    return self.linear(out)


# Captured at import time: whatever the library currently ships.
_OPTIMIZED_FORWARDS = {
    PXSUpsample: PXSUpsample.forward,
    PXSDownsample: PXSDownsample.forward,
}
_REFERENCE_FORWARDS = {
    PXSUpsample: _pxs_upsample_forward_reference,
    PXSDownsample: _pxs_downsample_forward_reference,
}


def parse_candidate(spec: str) -> frozenset:
    names = frozenset(n.strip() for n in spec.split(",") if n.strip())
    unknown = names - VALID_NAMES
    if unknown:
        raise ValueError(f"unknown candidate names {sorted(unknown)}; valid: {sorted(VALID_NAMES)}")
    if not names:
        raise ValueError("empty candidate spec")
    if "baseline" in names and len(names) > 1:
        raise ValueError("'baseline' cannot be combined with other names")
    return names


def apply_candidate(name: str) -> None:
    """Swap the class-level forwards: 'baseline' -> reference, else optimized."""
    forwards = _REFERENCE_FORWARDS if name == "baseline" else _OPTIMIZED_FORWARDS
    for cls, fwd in forwards.items():
        cls.forward = fwd


def _configure_inductor() -> None:
    # The recipe from docs/techniques/torch.compile Tuning (2D).md.
    import torch._inductor.config as cfg

    cfg.max_autotune = True
    cfg.max_autotune_gemm_backends = "ATEN,TRITON"
    cfg.combo_kernels = True
    cfg.memory_planning = True
    cfg.force_same_precision = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 128


class CandidateRunner:
    def __init__(self, vae, spec: str):
        self.vae = vae
        self.names = parse_candidate(spec)
        self.spec = ",".join(sorted(self.names))
        self._orig_encoder = vae.encoder
        self._orig_decoder = vae.decoder
        self._compiled = None

    @property
    def memory_format(self):
        return torch.channels_last if "cl" in self.names else torch.contiguous_format

    def activate(self) -> "CandidateRunner":
        apply_candidate("baseline" if "baseline" in self.names else "pxs")
        self.vae.to(memory_format=self.memory_format)
        if "compile" in self.names:
            if self._compiled is None:
                _configure_inductor()
                self._compiled = (
                    torch.compile(self._orig_encoder, dynamic=True),
                    torch.compile(self._orig_decoder, dynamic=True),
                )
            self.vae.encoder, self.vae.decoder = self._compiled
        else:
            self.vae.encoder, self.vae.decoder = self._orig_encoder, self._orig_decoder
        return self

    def prep(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(memory_format=self.memory_format)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.activate()
        return self.vae.encode(self.prep(x)).latent_dist.mode()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.activate()
        return self.vae.decode(self.prep(z))
