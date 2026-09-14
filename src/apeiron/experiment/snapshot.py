"""Resilience snapshots: full-state bundles for exact restart.

A snapshot captures everything an interrupted run needs to continue as if
the interruption never happened (see docs/experiment.md "Resilience"):
model and optimizer state, RNG states, monitor counters including the
partial metric buffer, the phase marker (monitoring vs mid-CL), pickled
detector internals, updater memory, and task-registry references. What is
recomputable from the config -- data, shuffle orders, loader positions --
is recounted on restore, never stored.

Snapshots live under ``<run>/checkpoints/resilience/`` next to the
post-CL analysis checkpoints (``analysis/``). Writes are atomic (temp
file + rename) and the previous snapshot is kept until the new one is on
disk, so a kill during the write can never leave zero usable snapshots.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1


def rng_states() -> Dict[str, Any]:
    """Capture torch/CUDA/numpy/python RNG states."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_states(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


class SnapshotManager:
    """Atomic writer/reader for resilience snapshots, keep-last-N."""

    def __init__(self, directory: str | Path, keep: int = 2):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    def save(self, state: Dict[str, Any], step: int) -> Path:
        """Write ``snapshot_<step>.pt`` atomically; prune beyond keep-last-N."""
        state = {**state, "schema": SCHEMA_VERSION, "step": step}
        final = self.directory / f"snapshot_{step:09d}.pt"
        tmp = self.directory / f".tmp-{final.name}"
        torch.save(state, tmp)
        tmp.rename(final)
        (self.directory / "latest").write_text(final.name)
        self._prune()
        return final

    def _prune(self) -> None:
        snaps = sorted(self.directory.glob("snapshot_*.pt"))
        while len(snaps) > self.keep:
            snaps.pop(0).unlink()

    def latest_path(self) -> Optional[Path]:
        pointer = self.directory / "latest"
        if not pointer.exists():
            return None
        path = self.directory / pointer.read_text().strip()
        return path if path.exists() else None

    def load_latest(self, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
        path = self.latest_path()
        if path is None:
            return None
        state = torch.load(path, map_location=map_location, weights_only=False)
        assert state.get("schema") == SCHEMA_VERSION, (
            f"snapshot schema {state.get('schema')} != {SCHEMA_VERSION}; "
            "refusing to resume across incompatible versions"
        )
        return state
