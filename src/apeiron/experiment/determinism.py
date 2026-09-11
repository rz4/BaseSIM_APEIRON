"""Determinism helpers: one seed in, reproducible runs out.

Two layers:

- :func:`seed_everything` seeds the global RNGs (torch, numpy, random) once
  at startup from ``cfg.seed``, making a whole run reproducible as long as
  the operation order is identical.
- :func:`window_generator` derives an *order-independent* ``torch.Generator``
  for a specific (seed, window, role) triple, so per-window DataLoader
  shuffles do not depend on how much global RNG state was consumed before
  the window started. Harnesses should use it for every shuffling loader.

Seeds are derived by hashing, never by arithmetic, so distinct roles and
windows cannot collide (``seed+1`` schemes do: window 2 of seed 40 would
share a stream with window 1 of seed 41).
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from a sequence of parts (ints, strings, ...).

    Stable across processes and platforms (unlike ``hash()``, which is
    salted per process).
    """
    text = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def seed_everything(seed: int) -> None:
    """Seed torch (CPU + CUDA), numpy, and the stdlib RNG."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def window_generator(seed: int, window: int, role: str = "") -> torch.Generator:
    """Order-independent generator for one stream window and role.

    :param seed: the run's ``cfg.seed``.
    :param window: window / task counter.
    :param role: distinguishes concurrent uses within one window
        (e.g. ``"train"`` vs ``"stream"``).
    """
    g = torch.Generator()
    g.manual_seed(stable_seed(seed, window, role))
    return g
