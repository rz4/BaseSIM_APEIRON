"""Experiment workspace: the view across an experiment's runs.

An experiment directory holds many runs over the same stream (a detector
run and its control arms, an updater sweep) plus their shared artifact
store. This module reads that state -- it never mutates runs:

- :meth:`Experiment.runs` — one summary row per run, from each journal.
- :meth:`Experiment.store_stats` — artifact store accounting.
- :meth:`Experiment.gc_pins` — the one mutation: drop pins left behind by
  runs that already finished or whose directory is gone (a crashed run
  never reaches ``Run.finish()``, so its pins would otherwise block
  eviction forever).
- ``python -m apeiron.experiment <path>`` prints the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from apeiron.experiment.artifacts import ArtifactStore
from apeiron.experiment.journal import Journal


@dataclass(frozen=True)
class RunInfo:
    """Summary of one run, derived from its journal."""

    name: str
    status: str  # "finished" | "running-or-crashed"
    continued: int  # times the run was continued
    windows: int
    drift_events: int
    checkpoints: int
    last_fwt: Optional[float]
    last_bwt: Optional[float]
    signature: str  # behavior hash (short)


class Experiment:
    """Read-only view over one experiment workspace."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.runs_root = self.path / "runs"
        assert self.runs_root.is_dir(), f"Not an experiment workspace: {self.path}"

    # ----- runs -----

    def run_names(self) -> List[str]:
        return sorted(p.name for p in self.runs_root.iterdir() if p.is_dir())

    def runs(self) -> List[RunInfo]:
        infos = []
        for name in self.run_names():
            journal_path = self.runs_root / name / "journal.sqlite"
            if not journal_path.exists():
                continue
            j = Journal(journal_path)
            try:
                cl = j.events(kind="cl_finished")
                last = cl[-1] if cl else {}
                infos.append(
                    RunInfo(
                        name=name,
                        status=(
                            "finished"
                            if j.events(kind="run_finished")
                            else "running-or-crashed"
                        ),
                        continued=len(j.events(kind="run_continued")),
                        windows=len(j.events(kind="window_started")),
                        drift_events=len(j.events(kind="drift_detected")),
                        checkpoints=len(j.events(kind="checkpoint_saved")),
                        last_fwt=last.get("fwt"),
                        last_bwt=last.get("bwt"),
                        signature=j.signature()[:12],
                    )
                )
            finally:
                j.close()
        return infos

    # ----- artifact store -----

    def store_stats(self) -> Optional[dict]:
        index = self.path / "artifacts" / "index.sqlite"
        if not index.exists():
            return None
        store = ArtifactStore(index.parent)
        try:
            return {
                "artifacts": len(store.evictable_lru()) + len(store.pinned_uris()),
                "total_bytes": store.total_bytes(),
                "pinned": len(store.pinned_uris()),
            }
        finally:
            store.close()

    def gc_pins(self) -> int:
        """Release pins of finished or vanished runs; returns pins dropped."""
        index = self.path / "artifacts" / "index.sqlite"
        if not index.exists():
            return 0
        finished = {r.name for r in self.runs() if r.status == "finished"}
        existing = set(self.run_names())
        store = ArtifactStore(index.parent)
        try:
            dropped = 0
            owners = {
                owner
                for (owner,) in store._conn.execute(
                    "SELECT DISTINCT owner FROM pins"
                ).fetchall()
            }
            for owner in owners:
                if owner in finished or owner not in existing:
                    dropped += store.release_owner(owner)
            return dropped
        finally:
            store.close()

    # ----- report -----

    def report(self) -> str:
        lines = [f"Experiment: {self.path}"]
        stats = self.store_stats()
        if stats is not None:
            lines.append(
                f"Artifact store: {stats['artifacts']} artifacts, "
                f"{stats['total_bytes'] / 1e9:.2f} GB, {stats['pinned']} pinned"
            )
        header = (
            f"{'run':<16} {'status':<19} {'wins':>4} {'drifts':>6} "
            f"{'ckpts':>5} {'cont':>4} {'last fwt':>10} {'last bwt':>10} {'signature':<12}"
        )
        lines += [header, "-" * len(header)]
        for r in self.runs():
            fwt = f"{r.last_fwt:.4f}" if r.last_fwt is not None else "-"
            bwt = f"{r.last_bwt:.4f}" if r.last_bwt is not None else "-"
            lines.append(
                f"{r.name:<16} {r.status:<19} {r.windows:>4} {r.drift_events:>6} "
                f"{r.checkpoints:>5} {r.continued:>4} {fwt:>10} {bwt:>10} {r.signature:<12}"
            )
        return "\n".join(lines)
