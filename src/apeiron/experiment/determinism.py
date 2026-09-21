"""Making a run a function of its config.

Two runs of one config should do the same thing, or comparing their signatures
means nothing. That needs the seed actually applied, and it needs every random
stream captured and restored together when a run is resumed.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every generator the run draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    """Capture every random stream the run draws from."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Put every random stream back where it was."""
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
