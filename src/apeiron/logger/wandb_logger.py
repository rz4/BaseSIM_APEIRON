"""Weights & Biases metrics backend.

The logger keeps its own global step and a per-stage step, prefixes metric
keys with the active stage (``eval/acc``), and -- when a ``csv_path`` is
configured -- keeps every recorded row so the run can be written out as a
long-format CSV with no service involved.

``WandBLogger(enabled=False)`` is a fully functional null object: it records
and writes CSVs but never opens a run, which is what ``backend = "none"``
is built on.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

try:  # the module-level name is patched by tests and by offline runs
    import wandb
except Exception:  # pragma: no cover - wandb is optional
    wandb = None  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore[assignment]

#: The stages a run is divided into.
StageType = Literal["eval", "drift", "cl"]

#: The only values :meth:`WandBLogger.stage` accepts.
VALID_STAGES: tuple[str, ...] = ("eval", "drift", "cl")


class WandBLogger:
    """Metrics backend writing to Weights & Biases and/or a CSV.

    Parameters
    ----------
    enabled : bool
        When false no run is opened and nothing is sent; recording and CSV
        output keep working.
    csv_path : str | Path | None
        Where :meth:`to_csv` writes. History is only retained when a path
        is configured.
    """

    def __init__(
        self,
        enabled: bool = True,
        csv_path: str | Path | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.csv_path: Path | None = Path(csv_path) if csv_path is not None else None
        self.run: Any = None

        self._step = 0
        self._current_stage: str | None = None
        self._stage_steps: dict[str, int] = {stage: 0 for stage in VALID_STAGES}
        self.metrics_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def step(self) -> int:
        """The global step, incremented by every committing :meth:`log`."""
        return self._step

    @property
    def current_stage(self) -> str | None:
        """The stage most recently named, or ``None``."""
        return self._current_stage

    @property
    def url(self) -> str | None:
        """The run's web URL, or ``None`` when no run is open."""
        if self.run is None:
            return None
        return getattr(self.run, "url", None)

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def stage(self, name: StageType) -> None:
        """Select the stage subsequent metrics belong to."""
        if name not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage {name!r}; expected one of {list(VALID_STAGES)}"
            )
        self._current_stage = name

    def get_stage_step(self, stage: StageType) -> int:
        """The step counter for ``stage``; ``0`` for one never logged to."""
        return self._stage_steps.get(stage, 0)

    # ------------------------------------------------------------------
    # run lifecycle
    # ------------------------------------------------------------------
    def init(
        self,
        cfg: Any,
        project: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        group: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open a run, unless disabled.

        The frozen config dataclass is passed through as a plain nested
        dict, and ``step`` is declared as the x-axis for every metric.
        """
        if not self.enabled or wandb is None:
            return None

        self.run = wandb.init(
            project=project,
            name=name,
            tags=tags,
            notes=notes,
            group=group,
            config=_as_dict(cfg),
            settings=wandb.Settings(silent=True),
            **kwargs,
        )
        # wandb.init can return None; define_metric must not be called on it
        if self.run is not None:
            self.run.define_metric("step")
            self.run.define_metric("*", step_metric="step")
        return self.run

    def finish(
        self, exit_code: int | None = None, save_csv: bool = True
    ) -> Path | None:
        """Close the run, optionally writing the CSV first.

        Returns
        -------
        Path | None
            The CSV that was written, if any.
        """
        written = self.to_csv() if save_csv else None
        if self.enabled and self.run is not None and wandb is not None:
            wandb.finish(exit_code=exit_code)
        self.run = None
        return written

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
        prefix: bool = True,
        increment: bool = True,
    ) -> None:
        """Record one row of metrics.

        The row carries two columns of the logger's own: the global ``step``
        and, when a stage is active, ``{stage}/step``.
        """
        if increment:
            self._step += 1
            if self._current_stage:
                self._stage_steps[self._current_stage] = (
                    self._stage_steps.get(self._current_stage, 0) + 1
                )

        current_step = self._step if step is None else step

        prefixed_metrics: dict[str, Any] = {}
        for key, value in metrics.items():
            if prefix and self._current_stage and "/" not in key:
                prefixed_metrics[f"{self._current_stage}/{key}"] = value
            else:
                prefixed_metrics[key] = value

        prefixed_metrics["step"] = current_step
        if self._current_stage:
            prefixed_metrics[f"{self._current_stage}/step"] = self._stage_steps.get(
                self._current_stage, 0
            )

        if self.enabled and self.run is not None and wandb is not None:
            wandb.log(prefixed_metrics, commit=commit)

        if self.csv_path is not None:
            self.metrics_history.append(prefixed_metrics)

    def save(
        self,
        path: str | Path,
        base_path: str | Path | None = None,
        policy: str = "now",
    ) -> None:
        """Upload a file as a run artifact; a no-op without a run."""
        if self.enabled and self.run is not None and wandb is not None:
            wandb.save(
                str(path),
                base_path=str(base_path) if base_path is not None else None,
                policy=policy,
            )

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------
    def to_dataframe(self) -> Any:
        """The recorded history in long format.

        Columns are ``step``, ``metric``, ``value`` -- one row per recorded
        value, with ``step`` lifted out of each entry rather than repeated
        as a metric of its own.
        """
        if not self.metrics_history or pd is None:
            return None
        rows = [
            {"step": entry.get("step"), "metric": key, "value": value}
            for entry in self.metrics_history
            for key, value in entry.items()
            if key != "step"
        ]
        if not rows:
            return None
        return pd.DataFrame(rows, columns=["step", "metric", "value"])

    def to_csv(self, csv_path: str | Path | None = None) -> Path | None:
        """Write the history to ``csv_path`` or the configured path."""
        target = csv_path if csv_path is not None else self.csv_path
        if target is None:
            return None
        frame = self.to_dataframe()
        if frame is None:
            return None
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        return target


def _as_dict(cfg: Any) -> Any:
    """A config dataclass as a plain nested dict; anything else unchanged."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    return cfg
