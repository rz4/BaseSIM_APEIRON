"""The run directory: one self-contained directory per run.

Everything a run reads that is not data, and everything a run writes, lives
under ``<experiment path>/runs/run_NNNN``::

    run_0001/
      config.toml           copy of the config file that was passed
      config.resolved.json  the config that actually ran, after overrides
      model.json            what model was used: class, shapes, hyperparameters
      journal.sqlite        event log
      metrics.csv           existing apeiron output, pointed here
      log.txt               console output
      signature.txt         hash of the journal, written when the run ends
      checkpoints/          existing save_ckpt output, pointed here

Existing apeiron code is not told about any of this: :meth:`Run.bind` rewrites
the output paths in the config, so the logger and the harness write into the
run directory without knowing it exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import time
from dataclasses import asdict, replace
from pathlib import Path

from typing import TYPE_CHECKING, Any

from apeiron.config.configuration import Config
from apeiron.experiment.journal import Journal
from apeiron.experiment.model_info import describe_model, summarize

if TYPE_CHECKING:
    from apeiron.model.torch_model_harness import BaseModelHarness

_RUN_DIR_RE = re.compile(r"^run_(\d+)")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Bound on the allocation retry loop. Only reached if many processes start at
# the same instant, which is exactly what an array job does.
MAX_ALLOC_ATTEMPTS = 100


def _safe_name(name: str) -> str:
    return _SAFE_NAME_RE.sub("-", name).strip("-")


def _next_index(runs_root: Path) -> int:
    used = [
        int(m.group(1))
        for p in runs_root.iterdir()
        if p.is_dir() and (m := _RUN_DIR_RE.match(p.name))
    ]
    return max(used, default=0) + 1


def behavior_hash(cfg: Config) -> str:
    """Hash of everything in the config except the ``[experiment]`` section.

    Two runs with the same behavior hash should do the same work; they differ
    only in where their output lands.
    """
    body = {k: v for k, v in asdict(cfg).items() if k != "experiment"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()
    ).hexdigest()


class Run:
    """A single run's directory and its event log."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.journal = Journal(self.run_dir / "journal.sqlite")
        self._started = time.monotonic()

    # -- allocation ---------------------------------------------------------
    @classmethod
    def create(cls, cfg: Config, config_path: str | Path | None = None) -> Run:
        """Allocate the next free run directory under the experiment path."""
        if cfg.experiment is None:
            raise ValueError("Run.create requires an [experiment] section")

        runs_root = Path(cfg.experiment.path).expanduser() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        suffix = _safe_name(cfg.experiment.run_name)
        suffix = f"_{suffix}" if suffix else ""

        for _ in range(MAX_ALLOC_ATTEMPTS):
            candidate = runs_root / f"run_{_next_index(runs_root):04d}{suffix}"
            try:
                # Exclusive create: two processes racing for the same number
                # cannot both win, and the loser simply takes the next one.
                candidate.mkdir()
            except FileExistsError:
                continue
            run = cls(candidate)
            break
        else:
            raise RuntimeError(f"could not allocate a run directory under {runs_root}")

        if config_path is not None:
            shutil.copyfile(config_path, run.run_dir / "config.toml")

        run.record(
            "run_started",
            run_dir=str(run.run_dir),
            config_sha256=behavior_hash(cfg),
            hostname=socket.gethostname(),
            pid=os.getpid(),
        )
        return run

    # -- layout -------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "log.txt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv"

    @property
    def signature_path(self) -> Path:
        return self.run_dir / "signature.txt"

    # -- config -------------------------------------------------------------
    def bind(self, cfg: Config) -> Config:
        """Return a copy of the config whose outputs point into this run.

        Also writes ``config.resolved.json``: the config that actually ran,
        which is what a later resume reads instead of the original command line.
        """
        model = replace(cfg.model, ckpts_path=str(self.checkpoints_dir))
        logging_cfg = (
            None
            if cfg.logging is None
            else replace(cfg.logging, metrics_output_path=str(self.metrics_path))
        )
        experiment = (
            None
            if cfg.experiment is None
            else replace(cfg.experiment, run_name=self.name)
        )
        bound = replace(cfg, model=model, logging=logging_cfg, experiment=experiment)

        (self.run_dir / "config.resolved.json").write_text(
            json.dumps(asdict(bound), indent=2)
        )
        return bound

    # -- events -------------------------------------------------------------
    def record(self, kind: str, **payload: object) -> int:
        return self.journal.record(kind, **payload)

    def record_model(self, harness: BaseModelHarness) -> dict[str, Any]:
        """Write ``model.json`` and log what model this run is using.

        The event carries everything except the per-tensor table, so the
        architecture takes part in the signature: a harness change that alters
        the network shows up as a different run even though the config is
        unchanged.
        """
        description = describe_model(harness)
        (self.run_dir / "model.json").write_text(json.dumps(description, indent=2))
        self.record("model", **summarize(description))
        return description

    def finish(self, status: str = "finished") -> str:
        """Close the run: record how it ended and write ``signature.txt``."""
        self.record(
            "run_finished", status=status, elapsed_s=time.monotonic() - self._started
        )
        signature = self.journal.signature()
        self.signature_path.write_text(signature + "\n")
        self.journal.close()
        return signature
