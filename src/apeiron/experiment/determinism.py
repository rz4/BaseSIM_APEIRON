"""Making a run a function of its config.

Two runs of one config should do the same thing, or comparing their signatures
means nothing. That takes three things: the seed actually applied, every random
stream captured and restored together when a run is resumed, and -- for a run
resumed in the middle of training -- shuffles that do not depend on how much
randomness was drawn before them.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Iterator

import numpy as np
import torch


def stable_seed(*parts: object) -> int:
    """A seed derived by hashing, not by arithmetic.

    ``seed + window`` collides: window 2 of seed 40 shares a stream with
    window 1 of seed 41. Hashing the parts does not, and unlike ``hash()``
    the result is the same in every process.
    """
    text = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


def seed_everything(seed: int) -> None:
    """Seed every generator the run draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def window_generator(seed: int, window: int, role: str = "") -> torch.Generator:
    """A generator for one window and role that does not depend on history.

    Sharing the global generator makes a window's shuffle depend on how much
    randomness everything before it happened to consume, which a resumed run
    cannot cheaply reproduce.
    """
    generator = torch.Generator()
    generator.manual_seed(stable_seed(seed, window, role))
    return generator


class EpochSeededSampler(torch.utils.data.Sampler[int]):
    """A shuffling sampler whose every epoch is a pure function of its index.

    A normal shuffling loader draws a fresh seed from the global generator
    each time its iterator is made, so epoch three's order depends on
    everything that drew randomness before it. Resuming then means replaying
    that history exactly.

    Here epoch k's permutation is ``f(seed, window, role, k)``, so carrying a
    resume across it needs one integer -- and a loader that wraps around
    mid-round reproduces the same wrap on replay.
    """

    def __init__(self, length: int, seed: int, window: int, role: str):
        self.length = length
        self.seed = seed
        self.window = window
        self.role = role
        self.epochs_started = 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(
            stable_seed(self.seed, self.window, self.role, "epoch", self.epochs_started)
        )
        self.epochs_started += 1
        return iter(torch.randperm(self.length, generator=generator).tolist())


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
