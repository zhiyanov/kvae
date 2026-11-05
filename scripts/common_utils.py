import os
import random
from typing import Tuple

import numpy as np
import torch


def configure_inductor():
    """
    The inductor/runtime recipe from docs/techniques/torch.compile Tuning (2D).md.
    """
    import torch._inductor.config as cfg

    cfg.max_autotune = True
    cfg.max_autotune_gemm_backends = "ATEN,TRITON"
    cfg.combo_kernels = True
    cfg.memory_planning = True
    cfg.force_same_precision = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 128


def compile_vae(vae: torch.nn.Module) -> torch.nn.Module:
    """
    Compile the encoder and decoder of a KVAE model for inference.

    Measured on KVAEImage (see docs/analysis/Optimization Results.md): 1.6-1.7x
    end-to-end at >=1024^2, at the cost of a one-time ~2 min compilation on the
    first forward. `dynamic=True` lets the compiled graphs serve all input
    resolutions without recompiling.
    """
    configure_inductor()
    vae.encoder = torch.compile(vae.encoder, dynamic=True)
    vae.decoder = torch.compile(vae.decoder, dynamic=True)
    return vae


def parse_int_tuple(s: str) -> Tuple[int, int]:
    """
    The function takes a string with two comma-separated
    integer values and returns a tuple of those integers.

    Parameters:
    -----------
    s
        input string with two comma-separated integer values

    Returns:
    -------
        A tuple of those integers
    """
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"Exactly 2 comma-separated values are expected. Received: {len(parts)}"
        )
    tuple_vals = tuple(int(p.strip()) for p in parts)
    return tuple_vals


def set_seed_and_optimal_cuda_env(seed: int = 42):
    """
    Captures randomness at all levels
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True