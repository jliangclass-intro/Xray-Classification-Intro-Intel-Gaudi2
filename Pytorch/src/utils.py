"""Hardware detection, random seeding, and filesystem utility functions."""
import os
import random

import numpy as np
import torch

# Register torch.hpu and HPU kernels on Intel Gaudi 2 when SynapseAI is installed,
# even if PT_HPU_AUTOLOAD=1 is not set in the environment.
try:
    import habana_frameworks.torch.core as htcore  # pyrefly: ignore[missing-import]  # Optional Intel Gaudi SynapseAI bridge
except ImportError:
    htcore = None


def hpu_mark_step() -> None:
    """Trigger graph compilation and execution step on Intel Gaudi 2 (HPU) in Lazy Mode.

    In SynapseAI Lazy Mode (PT_HPU_LAZY_MODE=1), operations are accumulated in a
    graph and only executed when mark_step() is called. On NVIDIA CUDA or CPU,
    this function is a safe no-op.
    """
    if htcore is not None and getattr(torch, "hpu", None) is not None and torch.hpu.is_available():
        htcore.mark_step()


def seed_everything(seed: int) -> None:
    """Set random seeds across Python, NumPy, PyTorch CPU, NVIDIA CUDA, and Intel Gaudi HPU.

    Args:
        seed: Integer seed value for reproducible data shuffling and weight initialization.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch, "hpu", None) is not None and getattr(torch.hpu, "is_available", lambda: False)():
        getattr(torch.hpu, "manual_seed_all", lambda _: None)(seed)
        if htcore is not None:
            try:
                import habana_frameworks.torch.hpu.random as htrandom  # pyrefly: ignore[missing-import]  # Optional Intel Gaudi RNG module
                htrandom.manual_seed_all(seed)
            except ImportError:
                pass


def get_best_accelerator() -> str:
    """Detect and return the best available hardware accelerator type.

    Returns:
        Device type string: ``'cuda'`` (NVIDIA GPU), ``'hpu'`` (Intel Gaudi 2), or ``'cpu'``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "hpu", None) is not None and getattr(torch.hpu, "is_available", lambda: False)():
        return "hpu"
    return "cpu"


def get_device() -> torch.device:
    """Return the active ``torch.device`` corresponding to ``get_best_accelerator()``.

    Returns:
        A ``torch.device`` object for ``cuda``, ``hpu``, or ``cpu``.
    """
    return torch.device(get_best_accelerator())


def ensure_dir(path: str) -> None:
    """Create a directory and any missing parent directories if they do not exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


