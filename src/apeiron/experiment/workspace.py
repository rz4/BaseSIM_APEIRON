"""A read-only look at an experiment's runs.

Once an experiment has more than a handful of runs, the question is never
"what is in this one file" but "which of these finished, which drifted, and
do any two of them agree". Everything here is derived from what the runs
already wrote; nothing is cached and nothing is written back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from apeiron.experiment.journal import Journal

_RUN_GLOB = "run_*"


@dataclass(frozen=True)
class RunInfo:
    """What one run's directory says about it."""

    name: str
    status: str
    windows: int
    drifts: int
    checkpoints: int
    continues: int
    fwt: Optional[float]
    bwt: Optional[float]
    signature: str

    @property
    def short_signature(self) -> str:
        return self.signature[:8] if self.signature else "-"


def _status(journal: Journal, run_dir: Path) -> str:
    """How a run ended, from its own last word.

    A run with no closing event never got to write one, so it is either still
    going or it died without warning -- from the outside those look the same,
    and saying so is more honest than guessing.
    """
    finished = journal.last("run_finished")
    if finished is not None:
        return str(finished.payload.get("status", "finished"))
    if (run_dir / "restart").is_dir():
        return "running or crashed (resumable)"
    return "running or crashed"


def read_run(run_dir: Path) -> RunInfo:
    """Summarise one run directory."""
    journal = Journal(run_dir / "journal.sqlite")
    try:
        last_cl = journal.last("cl_finished")
        signature_file = run_dir / "signature.txt"
        return RunInfo(
            name=run_dir.name,
            status=_status(journal, run_dir),
            windows=journal.count("window"),
            drifts=journal.count("drift"),
            checkpoints=journal.count("checkpoint"),
            continues=journal.count("run_continued"),
            fwt=None if last_cl is None else last_cl.payload.get("fwt"),
            bwt=None if last_cl is None else last_cl.payload.get("bwt"),
            # A run that did not finish never wrote one; compute it so an
            # interrupted run can still be compared.
            signature=(
                signature_file.read_text().strip()
                if signature_file.is_file()
                else journal.signature()
            ),
        )
    finally:
        journal.close()


def _directory_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


class Experiment:
    """The runs under one experiment directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def run_dirs(self) -> list[Path]:
        if not self.path.is_dir():
            return []
        return sorted(
            p for p in self.path.glob(_RUN_GLOB) if (p / "journal.sqlite").is_file()
        )

    def runs(self) -> list[RunInfo]:
        return [read_run(p) for p in self.run_dirs()]

    def datasets_bytes(self) -> int:
        return _directory_bytes(self.path / "datasets")

    def children(self) -> list[tuple[str, int]]:
        """Sub-experiments, for a directory holding several of them."""
        if not self.path.is_dir():
            return []
        found = []
        for child in sorted(self.path.iterdir()):
            if child.is_dir():
                count = len(Experiment(child).run_dirs())
                if count:
                    found.append((child.name, count))
        return found

    def report(self) -> str:
        runs = self.runs()
        if not runs:
            children = self.children()
            if children:
                lines = [f"{self.path} holds {len(children)} experiment(s):", ""]
                lines += [f"  {name}  ({count} run(s))" for name, count in children]
                return "\n".join(lines)
            return f"no runs under {self.path}"

        header = f"{'run':<20} {'status':<30} {'win':>4} {'drift':>5} {'ckpt':>4} {'cont':>4} {'fwt':>10} {'bwt':>10}  signature"
        lines = [f"{self.path}", "", header, "-" * len(header)]
        for run in runs:
            fwt = "-" if run.fwt is None else f"{run.fwt:.4g}"
            bwt = "-" if run.bwt is None else f"{run.bwt:.4g}"
            lines.append(
                f"{run.name:<20} {run.status:<30} {run.windows:>4} {run.drifts:>5} "
                f"{run.checkpoints:>4} {run.continues:>4} {fwt:>10} {bwt:>10}  "
                f"{run.short_signature}"
            )

        agreeing: dict[str, list[str]] = {}
        for run in runs:
            agreeing.setdefault(run.signature, []).append(run.name)
        repeated = [names for names in agreeing.values() if len(names) > 1]
        if repeated:
            lines.append("")
            for names in repeated:
                lines.append(f"same behaviour: {', '.join(names)}")

        data = self.datasets_bytes()
        if data:
            lines += ["", f"datasets: {_human(data)}"]
        return "\n".join(lines)
