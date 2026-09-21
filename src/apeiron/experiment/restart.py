"""Saving and reloading enough state to continue a killed run.

A run that hits a walltime limit, or is killed, leaves a file in the run's
``restart/`` directory holding everything the monitoring loop needs to pick up
where it left off: weights, optimizer, all random number generator state, the
drift detector, the updater's memory, and the loop's counters.

Saves happen at batch boundaries in the monitoring loop, never inside a
training round. A run killed during training replays that round from its
start, which costs at most one round and keeps the trainer out of the resume
path entirely.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1

# How many restart files to keep. Two, so a save interrupted by a second kill
# still leaves a complete older one.
KEEP = 2

_STATE_RE = re.compile(r"^state_(\d+)\.pt$")


class RunInterrupted(Exception):
    """Raised to unwind the monitoring loop after a save-and-exit signal."""


class RestartStore:
    """The ``restart/`` directory: newest-wins, keep a couple, write atomically."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def save(self, state: dict[str, Any]) -> Path:
        """Write one state file. Atomic: a crash mid-write leaves the old one."""
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"state_{int(state['batch_count']):06d}.pt"
        # Same directory as the target, so replace() is a rename and not a copy.
        tmp = self.directory / f".{target.name}.{os.getpid()}"
        torch.save(state, tmp)
        tmp.replace(target)
        self._prune()
        return target

    def files(self) -> list[Path]:
        """Existing state files, oldest first."""
        found = []
        if self.directory.is_dir():
            for p in self.directory.iterdir():
                if m := _STATE_RE.match(p.name):
                    found.append((int(m.group(1)), p))
        return [p for _, p in sorted(found)]

    def latest(self) -> Path | None:
        files = self.files()
        return files[-1] if files else None

    def load_latest(self) -> dict[str, Any] | None:
        """Load the newest state file, or None if there is none.

        Uses a full unpickle: the saved drift detector is an arbitrary object.
        Only load restart files your own runs produced.
        """
        path = self.latest()
        if path is None:
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    def _prune(self) -> None:
        for path in self.files()[:-KEEP]:
            path.unlink(missing_ok=True)
