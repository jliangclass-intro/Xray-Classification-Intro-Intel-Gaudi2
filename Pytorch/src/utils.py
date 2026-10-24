import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch, "hpu", None) is not None and getattr(torch.hpu, "is_available", lambda: False)():
        getattr(torch.hpu, "manual_seed_all", lambda _: None)(seed)


def get_best_accelerator() -> str:
    """Return best available device type: 'cuda', 'hpu', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "hpu", None) is not None and getattr(torch.hpu, "is_available", lambda: False)():
        return "hpu"
    return "cpu"


def get_device() -> torch.device:
    return torch.device(get_best_accelerator())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
