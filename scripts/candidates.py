"""A/B candidate switching for parity tests and benchmarks.

Optimized implementations live in the library code; the *reference*
(pre-speed-up, pinned-baseline) implementations are preserved here verbatim so
that baseline and candidate can run in the same process — the requirement of
docs/analysis/Speed-Up Testing Specification.md ("no stored golden tensors").

`apply_candidate("baseline")` swaps the reference forwards in;
`apply_candidate("<name>")` restores the optimized library forwards. The swap
is global (class-level), so callers must apply the desired candidate before
each measured/compared pass.
"""

import torch
import torch.nn.functional as F

from kvae.layers.sampling import PXSUpsample


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


# Captured at import time: whatever the library currently ships.
_OPTIMIZED_FORWARDS = {PXSUpsample: PXSUpsample.forward}
_REFERENCE_FORWARDS = {PXSUpsample: _pxs_upsample_forward_reference}

# "baseline" = reference implementations; every other name is an optimized
# candidate (currently they all map to the library code — candidates that need
# extra state, e.g. torch.compile, will extend this mechanism).
CANDIDATES = ("baseline", "pxs")


def apply_candidate(name: str) -> None:
    if name == "baseline":
        for cls, fwd in _REFERENCE_FORWARDS.items():
            cls.forward = fwd
    elif name in CANDIDATES:
        for cls, fwd in _OPTIMIZED_FORWARDS.items():
            cls.forward = fwd
    else:
        raise ValueError(f"unknown candidate {name!r}; expected one of {CANDIDATES}")
