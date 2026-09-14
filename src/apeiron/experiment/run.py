"""Run directories: one bounded directory per run, holding every output.

Layout under the experiment workspace (``[experiment] path``)::

    <path>/
        runs/
            run_0001/
                config/
                    original/<name>.toml   # raw config text as given
                    resolved.json          # post-derivation frozen config
                journal.sqlite             # event journal (apeiron.experiment.Journal)
                metrics.csv                # logger CSV, redirected here
                checkpoints/               # model.ckpts_path, redirected here

Creating a :class:`Run` allocates the directory; :meth:`Run.bind` returns a
new frozen ``Config`` whose output paths all point inside it, so the rest of
the pipeline needs no knowledge of experiment mode. Without an
``[experiment]`` config section nothing here is used and apeiron behaves
exactly as before.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from apeiron.config.configuration import Config, LoggingCfg
from apeiron.experiment.journal import Journal


class Run:
    """A single run's bounded directory plus its journal."""

    def __init__(self, run_dir: Path, journal: Journal):
        self.run_dir = run_dir
        self.journal = journal

    # ----- construction -----

    @classmethod
    def create(cls, cfg: Config, original_config: Optional[str | Path] = None) -> "Run":
        """Allocate the next run directory and record run_started.

        :param cfg: resolved config; ``cfg.experiment`` must be set.
        :param original_config: path of the TOML the run was launched with;
            its raw text (comments included) is preserved under ``config/original/``.
        """
        assert cfg.experiment is not None, "Run.create requires [experiment] config"
        runs_root = Path(cfg.experiment.path) / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        name = cfg.experiment.run_name or cls._next_run_name(runs_root)
        run_dir = runs_root / name
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config" / "original").mkdir(parents=True)
        (run_dir / "checkpoints").mkdir()

        resolved = json.dumps(asdict(cfg), indent=2)
        (run_dir / "config" / "resolved.json").write_text(resolved)
        if original_config is not None:
            src = Path(original_config)
            (run_dir / "config" / "original" / src.name).write_text(src.read_text())

        journal = Journal(run_dir / "journal.sqlite")
        journal.record(
            "run_started",
            run_name=name,
            config_sha256=cls._behavior_config_hash(cfg),
            original_config=str(original_config) if original_config else None,
        )
        return cls(run_dir=run_dir, journal=journal)

    @staticmethod
    def _behavior_config_hash(cfg: Config) -> str:
        """Hash of the config fields that determine run behavior.

        The ``[experiment]`` section is allocation metadata (workspace path,
        run name) -- two reruns of the same experiment legitimately differ
        there, so it is excluded. This keeps journal signatures equal across
        reruns while still catching any behavioral config change.
        """
        d = asdict(cfg)
        d.pop("experiment", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    @classmethod
    def open(cls, run_dir: str | Path) -> "Run":
        """Reopen an existing run directory (for continuing a run).

        The journal is appended to; nothing is recorded here -- the caller
        records its own ``run_continued`` event with whatever state it
        restored.
        """
        run_dir = Path(run_dir)
        journal_path = run_dir / "journal.sqlite"
        assert journal_path.exists(), f"No journal at {journal_path}; not a run dir"
        return cls(run_dir=run_dir, journal=Journal(journal_path))

    def resolved_config(self) -> dict:
        """The run's resolved config, as written at creation time."""
        return json.loads((self.run_dir / "config" / "resolved.json").read_text())

    @property
    def latest_checkpoint(self) -> Optional[Path]:
        """Path of the newest analysis checkpoint, or None if none was saved."""
        analysis = self.run_dir / "checkpoints" / "analysis"
        pointer = analysis / "latest"
        if not pointer.exists():
            return None
        return analysis / pointer.read_text().strip()

    def latest_snapshot(self, map_location: str = "cpu") -> Optional[dict]:
        """The newest resilience snapshot's full state, or None."""
        from apeiron.experiment.snapshot import SnapshotManager

        resilience = self.run_dir / "checkpoints" / "resilience"
        if not resilience.is_dir():
            return None
        return SnapshotManager(resilience).load_latest(map_location=map_location)

    @staticmethod
    def _next_run_name(runs_root: Path) -> str:
        taken = {p.name for p in runs_root.iterdir() if p.is_dir()}
        i = 1
        while f"run_{i:04d}" in taken:
            i += 1
        return f"run_{i:04d}"

    # ----- config binding -----

    def bind(self, cfg: Config) -> Config:
        """Return a config whose output paths all point inside the run dir.

        - ``logging.metrics_output_path`` -> ``<run>/metrics.csv`` (a logging
          section is created if the config had none, preserving the default
          backend, so an experiment run always records its metrics).
        - ``model.ckpts_path`` -> ``<run>/checkpoints`` (enable by setting
          ``model.max_ckpts > 0`` as before).
        - ``experiment.run_name`` -> the allocated name, so downstream
          consumers (e.g. residency pins) know which run they act for.
        """
        logging_cfg = cfg.logging or LoggingCfg(backend="wandb")
        logging_cfg = dataclasses.replace(
            logging_cfg, metrics_output_path=str(self.run_dir / "metrics.csv")
        )
        model_cfg = dataclasses.replace(
            cfg.model, ckpts_path=str(self.run_dir / "checkpoints")
        )
        assert cfg.experiment is not None
        experiment_cfg = dataclasses.replace(cfg.experiment, run_name=self.run_dir.name)
        return dataclasses.replace(
            cfg, logging=logging_cfg, model=model_cfg, experiment=experiment_cfg
        )

    # ----- lifecycle -----

    def finish(self, exit_code: int = 0) -> None:
        """Record run_finished, release this run's artifact pins, close journal."""
        released = self._release_pins()
        self.journal.record("run_finished", exit_code=exit_code, pins_released=released)
        self.journal.close()

    def _release_pins(self) -> int:
        """Drop this run's pins in the experiment artifact store, if one exists."""
        index = self.run_dir.parent.parent / "artifacts" / "index.sqlite"
        if not index.exists():
            return 0
        from apeiron.experiment.artifacts import ArtifactStore

        store = ArtifactStore(index.parent)
        released = store.release_owner(self.run_dir.name)
        store.close()
        return released
